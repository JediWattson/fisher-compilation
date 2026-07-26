"""Fit a source-independent static graph across the complete toy transformer.

This is the model-level follow-up to :mod:`variable_full_span_experiment`.
That predecessor established a useful rank-14 Fisher output span, but its
one-stage conditional router could not replace the source computation.  This
experiment keeps the rank-14 span fixed and asks a narrower question:

``Can a small causal transformer graph reconstruct the demanded answer row
without executing any of the three source transformer blocks?``

The artifacted protocol is fail-closed:

* every semantic context used by the predecessor is excluded;
* ``graph_fit_a`` updates weights;
* ``graph_stop_a`` selects checkpoints within a declared seed;
* ``graph_select_a`` selects architecture and seed;
* ``calibration_c`` is not evaluated until that choice is frozen;
* official validation is evaluated once only if calibration C passes;
* test remains hash-only and untouched.

A recorded post-run audit also checks an earlier interactive prototype whose
contexts predate this protocol.  Because that probe overlaps the nominal
``calibration_c`` role in the saved run, the module marks the panel
exploratory and fails its independence gate.  It never silently promotes that
panel to confirmation.

The executor is query sparse at its output boundary: all causal prefix rows
are available as graph inputs, while only the answer row is decoded back into
the source residual stream.  The PyTorch reference implementation still uses
dense kernels.  Reported MAC savings are therefore ideal logical work, not a
wall-clock speed claim.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .adapters import (
    SequenceContext,
    SequenceInputOrigin,
    module_state_fingerprint,
)
from .full_span_accounting import native_transformer_span_accounting
from .modes import FisherModeBasis
from .model import ToyTransformer
from .static_transformer_span_executor import (
    StaticTransformerSpanExecutor,
    StaticTransformerSpanExecutorConfig,
)
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
from .variable_conditional_experiment import (
    _behavior_record,
    _file_sha256,
    _jsonable,
    _per_example_nll,
)
from .variable_full_span_experiment import (
    DEFAULT_OUTPUT as DEFAULT_PREDECESSOR,
    _answer_rows,
    _compact_behavior,
    _key_mask,
    _logical_positions,
    _projected_answer_logits,
    _query_mask,
    verify_variable_full_span_artifacts,
)


DEFAULT_OUTPUT = Path(
    ".local-runs/variable-associative/static-transformer-full-span.pt"
)
DEFAULT_INPUT_SITE = "layer.0.input"
DEFAULT_OUTPUT_SITE = "layer.2.output"
DEFAULT_RETAINED_RANK = 14
DEFAULT_ROLE_SIZES: dict[str, int] = {
    "graph_fit_a": 512,
    "graph_stop_a": 128,
    "graph_select_a": 128,
    "calibration_c": 128,
}
ROLE_NAMES = tuple(DEFAULT_ROLE_SIZES)
DEFAULT_PROTOCOL_SALT = (
    "fisher_graph.variable_static_full_span.roles.v1"
)
# An earlier interactive architecture probe used these sequential train
# context rows before this artifact protocol was implemented.  They are
# recorded explicitly so the resulting overlap cannot be mistaken for clean
# confirmation.  The saved run remains reproducible, but calibration C is
# demoted to exploratory evidence whenever this audit finds overlap.
AD_HOC_PROBE_ROLE_RANGES: dict[str, tuple[int, int]] = {
    "prototype_fit": (120, 632),
    "prototype_stop": (632, 760),
    "prototype_select": (760, 888),
}
DEFAULT_SEEDS = (92_101, 92_102, 92_103)
DEFAULT_BOOTSTRAP_SEED = 92_104
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_MAXIMUM_RELATIVE_WORK = 0.90
DEFAULT_MAXIMUM_RELATIVE_STORAGE = 0.90

_SCHEMA = "fisher_graph.variable_static_transformer_full_span"
_FORMAT_VERSION = 1
_ROLE_HASH_DOMAIN = b"fisher_graph.variable_static_full_span.role.v1\0"


@dataclass(frozen=True, slots=True)
class StaticGraphTrainingConfig:
    """Predeclared optimization and checkpoint-selection protocol."""

    learning_rate: float = 3e-3
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_steps: int = 2_000
    evaluation_interval: int = 200
    gradient_clip_norm: float = 1.0
    modal_mse_weight: float = 0.1
    cross_entropy_weight: float = 1.0
    teacher_kl_weight: float = 1.0
    label_smoothing: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            len(self.betas) != 2
            or any(not math.isfinite(value) or not 0 <= value < 1 for value in self.betas)
        ):
            raise ValueError("betas must contain two finite values in [0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        for name in ("batch_size", "max_steps", "evaluation_interval"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_steps % self.evaluation_interval:
            raise ValueError("max_steps must be divisible by evaluation_interval")
        for name in (
            "gradient_clip_norm",
            "modal_mse_weight",
            "cross_entropy_weight",
            "teacher_kl_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.gradient_clip_norm == 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class StaticGraphCandidate:
    """One declared mini-transformer capacity."""

    name: str
    hidden_width: int
    layer_count: int
    head_count: int
    feed_forward_width: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("candidate name must be nonempty")
        for name in (
            "hidden_width",
            "layer_count",
            "head_count",
            "feed_forward_width",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_width % self.head_count:
            raise ValueError("hidden_width must be divisible by head_count")

    def executor_config(
        self,
        *,
        residual_width: int,
        retained_rank: int,
    ) -> StaticTransformerSpanExecutorConfig:
        return StaticTransformerSpanExecutorConfig(
            residual_width=residual_width,
            hidden_width=self.hidden_width,
            layer_count=self.layer_count,
            head_count=self.head_count,
            feed_forward_width=self.feed_forward_width,
            retained_rank=retained_rank,
        )


# The one-layer entry is an explicit multi-hop ablation.  Two-layer width 16
# tests the smallest promising prototype; width 32 is the declared fallback.
DEFAULT_CANDIDATES = (
    StaticGraphCandidate("l1_h32", 32, 1, 4, 64),
    StaticGraphCandidate("l2_h16", 16, 2, 4, 32),
    StaticGraphCandidate("l2_h32", 32, 2, 4, 64),
)


@dataclass(slots=True)
class NativeBoundaryCorpus:
    """Frozen source boundaries for one complete-context role."""

    split: VariableAssociativeRecallSplit
    incoming: Tensor
    raw_delta_coordinates: Tensor
    teacher_logits: Tensor

    def __post_init__(self) -> None:
        samples = self.split.samples
        if self.incoming.shape[:2] != (
            samples,
            self.split.maximum_sequence_length,
        ):
            raise ValueError("incoming boundaries do not align with split")
        if self.raw_delta_coordinates.shape[0] != samples:
            raise ValueError("delta coordinates do not align with split")
        if self.teacher_logits.shape[0] != samples:
            raise ValueError("teacher logits do not align with split")
        for value in (
            self.incoming,
            self.raw_delta_coordinates,
            self.teacher_logits,
        ):
            if value.device.type != "cpu" or not torch.isfinite(value).all():
                raise ValueError("boundary corpus tensors must be finite CPU tensors")


@dataclass(slots=True)
class GraphBoundaryCorpus:
    """Source inputs plus standardized fixed-span targets."""

    split: VariableAssociativeRecallSplit
    incoming: Tensor
    standardized_coordinates: Tensor
    teacher_logits: Tensor


def _hash_rank_contexts(
    source: VariableAssociativeRecallSplit,
    *,
    excluded_hashes: set[str],
    salt: str,
) -> tuple[int, ...]:
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be nonempty")
    candidates = [
        (index, semantic_hash)
        for index, semantic_hash in enumerate(source.semantic_context_hashes)
        if semantic_hash not in excluded_hashes
    ]

    def rank(item: tuple[int, str]) -> tuple[str, str]:
        _, semantic_hash = item
        digest = hashlib.sha256()
        digest.update(_ROLE_HASH_DOMAIN)
        digest.update(salt.encode("utf-8"))
        digest.update(b"\0")
        digest.update(semantic_hash.encode("ascii"))
        return digest.hexdigest(), semantic_hash

    return tuple(index for index, _ in sorted(candidates, key=rank))


def _allocate_context_roles(
    source: VariableAssociativeRecallSplit,
    *,
    predecessor_role_hashes: Mapping[str, Sequence[str]],
    validation_hashes: Sequence[str],
    test_hashes: Sequence[str],
    role_sizes: Mapping[str, int] = DEFAULT_ROLE_SIZES,
    salt: str = DEFAULT_PROTOCOL_SALT,
) -> tuple[
    dict[str, VariableAssociativeRecallSplit],
    tuple[str, ...],
]:
    """Hash-rank whole contexts after excluding all earlier/evaluation roles."""

    if tuple(role_sizes) != ROLE_NAMES:
        raise ValueError(f"role_sizes must use the declared order {ROLE_NAMES}")
    if any(type(size) is not int or size <= 0 for size in role_sizes.values()):
        raise ValueError("role sizes must be positive integers")
    excluded = set(validation_hashes) | set(test_hashes)
    for hashes in predecessor_role_hashes.values():
        excluded.update(hashes)
    ranked = _hash_rank_contexts(
        source,
        excluded_hashes=excluded,
        salt=salt,
    )
    required = sum(role_sizes.values())
    if len(ranked) < required:
        raise ValueError("too few fresh semantic contexts for graph roles")

    roles: dict[str, VariableAssociativeRecallSplit] = {}
    cursor = 0
    for name, size in role_sizes.items():
        rows = torch.tensor(ranked[cursor : cursor + size], dtype=torch.int64)
        roles[name] = subset_variable_associative_recall_split(
            source,
            context_rows=rows,
            name=name,
        )
        cursor += size
    reserve_hashes = tuple(
        source.semantic_context_hashes[index] for index in ranked[cursor:]
    )

    all_role_hashes: list[set[str]] = [
        set(role.semantic_context_hashes) for role in roles.values()
    ]
    for left in range(len(all_role_hashes)):
        for right in range(left + 1, len(all_role_hashes)):
            if not all_role_hashes[left].isdisjoint(all_role_hashes[right]):
                raise RuntimeError("graph role semantic contexts overlap")
    used = set().union(*all_role_hashes)
    if used & excluded:
        raise RuntimeError("graph roles overlap predecessor or evaluation contexts")
    if set(reserve_hashes) & (used | excluded):
        raise RuntimeError("reserve contexts overlap a consumed role")
    return roles, reserve_hashes


def _ad_hoc_probe_role_hashes(
    source: VariableAssociativeRecallSplit,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for name, (start, stop) in AD_HOC_PROBE_ROLE_RANGES.items():
        if not 0 <= start < stop <= source.contexts:
            raise ValueError("source split is too small for the recorded probe")
        result[name] = source.semantic_context_hashes[start:stop]
    return result


def _development_overlap_audit(
    roles: Mapping[str, VariableAssociativeRecallSplit],
    reserve_hashes: Sequence[str],
    *,
    ad_hoc_probe_role_hashes: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    probe = set().union(
        *(set(hashes) for hashes in ad_hoc_probe_role_hashes.values())
    )
    role_overlaps = {
        name: tuple(
            semantic_hash
            for semantic_hash in split.semantic_context_hashes
            if semantic_hash in probe
        )
        for name, split in roles.items()
    }
    clean_reserve = tuple(
        semantic_hash
        for semantic_hash in reserve_hashes
        if semantic_hash not in probe
    )
    calibration_overlap = role_overlaps["calibration_c"]
    return {
        "recorded_before_protocol": True,
        "ad_hoc_probe_context_count": len(probe),
        "role_overlap_counts": {
            name: len(hashes) for name, hashes in role_overlaps.items()
        },
        "role_overlap_hashes": role_overlaps,
        "calibration_c_overlap_count": len(calibration_overlap),
        "calibration_c_confirmatory": len(calibration_overlap) == 0,
        "clean_unconsumed_reserve_context_count": len(clean_reserve),
        "clean_unconsumed_reserve_context_hashes": clean_reserve,
        "interpretation": (
            "calibration C is exploratory when overlap is nonzero; only clean "
            "unconsumed reserve hashes may support a later confirmation"
        ),
    }


def _sequence_context(
    split: VariableAssociativeRecallSplit,
    *,
    rows: Tensor | None = None,
    device: torch.device | str = "cpu",
) -> SequenceContext:
    query = _query_mask(split)
    keys = _key_mask(split)
    positions = _logical_positions(split)
    if rows is not None:
        selected = rows.detach().to(device="cpu", dtype=torch.int64)
        query = query.index_select(0, selected)
        keys = keys.index_select(0, selected)
        positions = positions.index_select(0, selected)
    target_device = torch.device(device)
    query = query.to(target_device)
    keys = keys.to(target_device)
    positions = positions.to(target_device)
    return SequenceContext(
        query_valid_mask=query,
        key_valid_mask=keys,
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=False,
            cache_positions_supplied=False,
        ),
        cache_state=None,
        adapter_payload=None,
    )


@torch.no_grad()
def _collect_native_boundaries(
    model: ToyTransformer,
    split: VariableAssociativeRecallSplit,
    *,
    basis_vectors: Tensor,
    input_site: str,
    output_site: str,
    batch_size: int = 512,
) -> NativeBoundaryCorpus:
    """Run the frozen source once and retain only compiler boundary tensors."""

    if (
        not isinstance(basis_vectors, Tensor)
        or basis_vectors.ndim != 2
        or basis_vectors.shape[0] != model.config.d_model
    ):
        raise ValueError("basis_vectors must have shape [residual width, rank]")
    incoming_chunks: list[Tensor] = []
    coordinate_chunks: list[Tensor] = []
    current_input: Tensor | None = None
    cursor = 0

    def capture_input(values: Tensor) -> Tensor:
        nonlocal current_input
        if current_input is not None:
            raise RuntimeError("input boundary captured twice before output")
        current_input = values
        incoming_chunks.append(values.detach().cpu().clone())
        return values

    def capture_output(values: Tensor) -> Tensor:
        nonlocal current_input, cursor
        if current_input is None:
            raise RuntimeError("output boundary captured before input")
        batch = values.shape[0]
        positions = split.supervised_positions[cursor : cursor + batch].to(
            values.device
        )
        rows = torch.arange(batch, device=values.device)
        incoming_answer = current_input[rows, positions]
        outgoing_answer = values[rows, positions]
        vectors = basis_vectors.to(
            device=values.device,
            dtype=values.dtype,
        )
        coordinate_chunks.append(
            ((outgoing_answer - incoming_answer) @ vectors)
            .detach()
            .cpu()
            .clone()
        )
        current_input = None
        cursor += batch
        return values

    teacher_logits = variable_associative_answer_logits(
        model,
        split,
        batch_size=batch_size,
        activation_interventions={
            input_site: capture_input,
            output_site: capture_output,
        },
    )
    if cursor != split.samples or current_input is not None:
        raise RuntimeError("source boundary collection did not consume split")
    return NativeBoundaryCorpus(
        split=split,
        incoming=torch.cat(incoming_chunks),
        raw_delta_coordinates=torch.cat(coordinate_chunks),
        teacher_logits=teacher_logits,
    )


def _coordinate_scale(corpus: NativeBoundaryCorpus) -> Tensor:
    scale = corpus.raw_delta_coordinates.to(torch.float32).std(
        dim=0,
        unbiased=False,
    )
    return scale.clamp_min(1e-3)


def _standardize_corpus(
    corpus: NativeBoundaryCorpus,
    scale: Tensor,
) -> GraphBoundaryCorpus:
    if (
        not isinstance(scale, Tensor)
        or scale.ndim != 1
        or scale.shape[0] != corpus.raw_delta_coordinates.shape[1]
        or not torch.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError("coordinate scale must be finite, positive, and rank-sized")
    return GraphBoundaryCorpus(
        split=corpus.split,
        incoming=corpus.incoming.to(torch.float32),
        standardized_coordinates=(
            corpus.raw_delta_coordinates.to(torch.float32) / scale
        ),
        teacher_logits=corpus.teacher_logits.to(torch.float32),
    )


def _strong_behavior_gates(record: Mapping[str, object]) -> dict[str, bool]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("behavior metrics must be a mapping")
    minimum_stratum = min(
        float(metrics["minimum_query_accuracy"]),
        float(metrics["minimum_pair_order_accuracy"]),
        float(metrics["minimum_layout_accuracy"]),
        float(metrics["minimum_length_accuracy"]),
    )
    return {
        "delta_nll_at_most_0.01": float(record["delta_nll"]) <= 0.01,
        "answer_accuracy_exact": float(metrics["answer_accuracy"]) == 1.0,
        "paired_context_accuracy_exact": (
            float(metrics["paired_context_accuracy"]) == 1.0
        ),
        "minimum_stratum_accuracy_exact": minimum_stratum == 1.0,
        "top1_agreement_exact": float(record["top1_agreement"]) == 1.0,
        "native_teacher_kl_at_most_0.01": (
            float(record["native_teacher_kl"]) <= 0.01
        ),
        "p90_absolute_delta_nll_at_most_0.025": (
            float(record["p90_absolute_delta_nll"]) <= 0.025
        ),
    }


def _behavior_summary(
    logits: Tensor,
    baseline_logits: Tensor,
    split: VariableAssociativeRecallSplit,
) -> dict[str, object]:
    record = _behavior_record(logits, baseline_logits, split)
    strong_gates = _strong_behavior_gates(record)
    return {
        **_compact_behavior(record),
        "minimum_viability_passed": record["passed"] is True,
        "strong_gates": strong_gates,
        "strong_passed": all(strong_gates.values()),
    }


def _student_outputs(
    executor: StaticTransformerSpanExecutor,
    model: ToyTransformer,
    corpus: GraphBoundaryCorpus,
    rows: Tensor,
) -> tuple[Tensor, Tensor]:
    selected = rows.detach().to(device="cpu", dtype=torch.int64)
    incoming = corpus.incoming.index_select(0, selected)
    sequence = _sequence_context(corpus.split, rows=selected)
    execution = executor.forward_components(incoming, sequence)
    coordinates = execution.demanded_coordinates
    expected_queries = int(sequence.query_valid_mask.sum().item())
    if coordinates.shape != (expected_queries, executor.config.retained_rank):
        raise RuntimeError("executor demanded coordinates have invalid shape")
    answer_hidden = execution.output[sequence.query_valid_mask]
    logits = model.lm_head(model.final_norm(answer_hidden))
    return coordinates, logits


@torch.no_grad()
def _evaluate_executor(
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
    modal_mse = F.mse_loss(
        predicted_coordinates,
        corpus.standardized_coordinates,
    )
    summary = _behavior_summary(
        logits,
        corpus.teacher_logits,
        corpus.split,
    )
    summary["standardized_modal_mse"] = float(modal_mse.item())
    return summary, logits


def _make_executor(
    candidate: StaticGraphCandidate,
    *,
    residual_width: int,
    retained_rank: int,
    decoder: Tensor,
    seed: int,
) -> StaticTransformerSpanExecutor:
    torch.manual_seed(seed)
    return StaticTransformerSpanExecutor(
        config=candidate.executor_config(
            residual_width=residual_width,
            retained_rank=retained_rank,
        ),
        decoder=decoder,
        dtype=torch.float32,
        device="cpu",
    )


def _checkpoint_score(
    behavior: Mapping[str, object],
    *,
    step: int,
) -> tuple[int, int, float, float, int]:
    return (
        0 if behavior["strong_passed"] is True else 1,
        0 if behavior["minimum_viability_passed"] is True else 1,
        float(behavior["hard_nll"]),
        float(behavior["standardized_modal_mse"]),
        step,
    )


def _train_candidate_seed(
    candidate: StaticGraphCandidate,
    *,
    seed: int,
    model: ToyTransformer,
    fit: GraphBoundaryCorpus,
    stop: GraphBoundaryCorpus,
    select: GraphBoundaryCorpus,
    decoder: Tensor,
    training: StaticGraphTrainingConfig,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    executor = _make_executor(
        candidate,
        residual_width=model.config.d_model,
        retained_rank=fit.standardized_coordinates.shape[1],
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
    best_score: tuple[int, int, float, float, int] | None = None
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
        target_coordinates = fit.standardized_coordinates.index_select(0, rows)
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

        if step % training.evaluation_interval:
            continue
        stop_behavior, _ = _evaluate_executor(
            executor,
            model,
            stop,
        )
        evaluation = {
            "step": step,
            "training_loss": float(loss.detach().item()),
            "training_modal_mse": float(modal_loss.detach().item()),
            "training_cross_entropy": float(cross_entropy.detach().item()),
            "training_teacher_kl": float(teacher_kl.detach().item()),
            "gradient_norm": float(gradient_norm.detach().item()),
            "graph_stop_a": stop_behavior,
        }
        history.append(evaluation)
        score = _checkpoint_score(stop_behavior, step=step)
        if best_score is None or score < best_score:
            best_score = score
            best_step = step
            executor.eval()
            best_artifact = executor.artifact_state_dict()
            executor.train()
        if progress is not None:
            progress(
                f"{candidate.name} seed={seed} step={step}: "
                f"stop_nll={float(stop_behavior['hard_nll']):.6f}, "
                f"delta={float(stop_behavior['delta_nll']):+.6f}, "
                f"accuracy={float(stop_behavior['answer_accuracy']):.6f}, "
                f"strong={stop_behavior['strong_passed']}"
            )

    if best_artifact is None or best_score is None:
        raise RuntimeError("candidate training produced no checkpoint")
    best_executor = StaticTransformerSpanExecutor.from_artifact_state_dict(
        best_artifact
    )
    select_behavior, _ = _evaluate_executor(
        best_executor,
        model,
        select,
    )
    record = {
        "candidate": asdict(candidate),
        "seed": seed,
        "best_step": best_step,
        "graph_stop_a": history[
            next(
                index
                for index, item in enumerate(history)
                if item["step"] == best_step
            )
        ]["graph_stop_a"],
        "graph_select_a": select_behavior,
        "history": history,
        "executor_fingerprint": best_executor.execution_fingerprint(),
        "runtime_stored_coefficient_count": (
            best_executor.total_runtime_coefficient_count
        ),
    }
    return record, best_artifact


def _logical_pair_count(split: VariableAssociativeRecallSplit) -> int:
    prefix_lengths = _key_mask(split).sum(dim=1).to(torch.int64)
    return int((prefix_lengths * (prefix_lengths + 1) // 2).sum().item())


def _compute_accounting(
    model: ToyTransformer,
    split: VariableAssociativeRecallSplit,
    executor: StaticTransformerSpanExecutor,
) -> dict[str, object]:
    """Compare complete answer-row work against the native source span."""

    query_rows = split.samples
    valid_rows = int(_key_mask(split).sum().item())
    logical_pairs = _logical_pair_count(split)
    width = model.config.d_model
    graph = executor.config
    graph_input = valid_rows * width * graph.hidden_width
    graph_blocks = graph.layer_count * (
        4 * valid_rows * graph.hidden_width * graph.hidden_width
        + 2 * logical_pairs * graph.hidden_width
        + 2
        * valid_rows
        * graph.hidden_width
        * graph.feed_forward_width
    )
    graph_modal_head = (
        query_rows * graph.hidden_width * graph.retained_rank
    )
    graph_decoder = query_rows * graph.retained_rank * width
    shared_answer_head = query_rows * width * model.config.vocab_size
    graph_complete = (
        graph_input
        + graph_blocks
        + graph_modal_head
        + graph_decoder
        + shared_answer_head
    )
    native = native_transformer_span_accounting(
        valid_rows=valid_rows,
        causal_pairs=logical_pairs,
        width=width,
        feed_forward_width=model.config.d_ff,
        layer_count=model.config.n_layers,
    )
    native_complete = native.logical_total_macs + shared_answer_head
    native_storage = native.total_parameter_count
    graph_storage = executor.total_runtime_coefficient_count
    reference = executor.logical_accounting(_sequence_context(split))
    if reference.logical_total_macs != graph_complete - shared_answer_head:
        raise RuntimeError("executor and experiment logical MAC accounting disagree")
    reference_complete = (
        reference.reference_dense_prefix_total_macs + shared_answer_head
    )
    return {
        "scope": "one demanded answer row with every causal prefix row available",
        "mac_definition": "matrix multiply-accumulates only",
        "reference_kernel_dense": True,
        "wall_clock_speed_claim": False,
        "valid_prefix_rows": valid_rows,
        "logical_causal_pairs": logical_pairs,
        "demanded_query_rows": query_rows,
        "native": {
            **asdict(native),
            "shared_answer_head_macs": shared_answer_head,
            "complete_macs": native_complete,
            "source_span_stored_coefficients": native_storage,
        },
        "static_graph": {
            "input_projection_macs": graph_input,
            "mini_transformer_macs": graph_blocks,
            "modal_head_macs": graph_modal_head,
            "fixed_decoder_macs": graph_decoder,
            "shared_answer_head_macs": shared_answer_head,
            "complete_macs": graph_complete,
            "runtime_stored_coefficients": graph_storage,
            "learned_parameters": executor.learned_parameter_count,
            "fixed_runtime_coefficients": (
                executor.fixed_runtime_coefficient_count
            ),
        },
        "static_graph_reference_dense_prefix": {
            **asdict(reference),
            "logical_graph_macs": reference.logical_total_macs,
            "dense_reference_to_logical_ratio": (
                reference.dense_reference_to_logical_ratio
            ),
            "shared_answer_head_macs": shared_answer_head,
            "complete_macs": reference_complete,
        },
        "graph_to_native_complete_mac_ratio": (
            graph_complete / native_complete
        ),
        "reference_dense_graph_to_native_complete_mac_ratio": (
            reference_complete / native_complete
        ),
        "graph_to_native_source_storage_ratio": (
            graph_storage / native_storage
        ),
    }


def _select_architecture_and_seed(
    records: Sequence[Mapping[str, object]],
    *,
    required_strong_seeds: int = 2,
) -> dict[str, object]:
    """Apply the frozen 2-of-3 rule, then select cheapest strong architecture."""

    if type(required_strong_seeds) is not int or required_strong_seeds <= 0:
        raise ValueError("required_strong_seeds must be positive")
    by_name: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        candidate = record.get("candidate")
        if not isinstance(candidate, Mapping):
            raise TypeError("candidate record is invalid")
        by_name.setdefault(str(candidate["name"]), []).append(record)
    architecture_records: list[dict[str, object]] = []
    runtime_lookup: dict[tuple[str, int], Mapping[str, object]] = {}
    for name, group in by_name.items():
        if len({int(record["seed"]) for record in group}) != len(group):
            raise ValueError("candidate seeds must be unique")
        strong = [
            record
            for record in group
            if isinstance(record.get("graph_select_a"), Mapping)
            and record["graph_select_a"]["strong_passed"] is True  # type: ignore[index]
        ]
        architecture_passed = len(strong) >= required_strong_seeds
        eligible = strong if architecture_passed else group
        ranked_seeds = sorted(
            eligible,
            key=lambda record: (
                float(record["graph_select_a"]["hard_nll"]),  # type: ignore[index]
                int(record["seed"]),
            ),
        )
        selected_seed_record = ranked_seeds[len(ranked_seeds) // 2]
        accounting = selected_seed_record.get("accounting")
        if not isinstance(accounting, Mapping):
            raise TypeError("candidate accounting is missing")
        summary = {
            "candidate": copy.deepcopy(dict(group[0]["candidate"])),  # type: ignore[arg-type]
            "seed_count": len(group),
            "strong_seed_count": len(strong),
            "required_strong_seeds": required_strong_seeds,
            "architecture_passed": architecture_passed,
            "selected_median_strong_seed": int(selected_seed_record["seed"]),
            "selected_seed_graph_select_a": copy.deepcopy(
                selected_seed_record["graph_select_a"]
            ),
            "accounting": copy.deepcopy(dict(accounting)),
        }
        architecture_records.append(summary)
        for record in group:
            runtime_lookup[(name, int(record["seed"]))] = record

    passing = [
        record
        for record in architecture_records
        if record["architecture_passed"] is True
    ]
    selected_architecture: dict[str, object] | None = None
    selected_key: tuple[str, int] | None = None
    if passing:
        selected_architecture = min(
            passing,
            key=lambda record: (
                int(record["accounting"]["static_graph"]["complete_macs"]),  # type: ignore[index]
                int(
                    record["accounting"]["static_graph"][  # type: ignore[index]
                        "runtime_stored_coefficients"
                    ]
                ),
                float(
                    record["selected_seed_graph_select_a"]["hard_nll"]  # type: ignore[index]
                ),
                str(record["candidate"]["name"]),  # type: ignore[index]
            ),
        )
        selected_key = (
            str(selected_architecture["candidate"]["name"]),  # type: ignore[index]
            int(selected_architecture["selected_median_strong_seed"]),
        )
    return {
        "architectures": architecture_records,
        "selected": copy.deepcopy(selected_architecture),
        "selected_runtime_key": selected_key,
    }


def _bootstrap_nll_degradation(
    candidate_logits: Tensor,
    baseline_logits: Tensor,
    split: VariableAssociativeRecallSplit,
    *,
    seed: int,
    samples: int,
) -> dict[str, object]:
    if type(samples) is not int or samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    degradation = (
        _per_example_nll(candidate_logits, split)
        - _per_example_nll(baseline_logits, split)
    ).detach().cpu()
    contexts = split.example_context_indices.to(torch.int64)
    unique_contexts = contexts.unique(sorted=True)
    context_means = torch.stack(
        [
            degradation[contexts == context].mean()
            for context in unique_contexts
        ]
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bootstrap_means = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(
            context_means.numel(),
            (context_means.numel(),),
            generator=generator,
        )
        bootstrap_means[index] = context_means.index_select(0, rows).mean()
    return {
        "resampling_unit": "semantic_context",
        "resampling_units": int(context_means.numel()),
        "samples": samples,
        "mean_nll_degradation": float(context_means.mean().item()),
        "lower_95_percent": float(
            torch.quantile(bootstrap_means, 0.025).item()
        ),
        "upper_95_percent": float(
            torch.quantile(bootstrap_means, 0.975).item()
        ),
    }


@torch.no_grad()
def _direct_replacement_answer_logits(
    model: ToyTransformer,
    executor: StaticTransformerSpanExecutor,
    split: VariableAssociativeRecallSplit,
    *,
    batch_size: int = 512,
) -> tuple[Tensor, tuple[int, ...]]:
    """Embed tokens, execute the graph, and bypass every source block."""

    calls = [0 for _ in model.layers]
    handles = []
    for layer_index, layer in enumerate(model.layers):
        def count_call(
            _module: object,
            _arguments: tuple[object, ...],
            *,
            index: int = layer_index,
        ) -> None:
            calls[index] += 1

        handles.append(layer.register_forward_pre_hook(count_call))

    logits: list[Tensor] = []
    try:
        for start in range(0, split.samples, batch_size):
            stop = min(start + batch_size, split.samples)
            rows = torch.arange(start, stop, dtype=torch.int64)
            input_ids = split.input_ids.index_select(0, rows)
            positions = torch.arange(
                split.maximum_sequence_length,
                dtype=torch.int64,
            ).unsqueeze(0).expand(stop - start, -1)
            incoming = model.embedding_dropout(
                model.token_embedding(input_ids)
                + model.position_embedding(positions)
            )
            sequence = _sequence_context(split, rows=rows)
            execution = executor.forward_components(incoming, sequence)
            answer_hidden = execution.output[sequence.query_valid_mask]
            logits.append(
                model.lm_head(model.final_norm(answer_hidden)).cpu()
            )
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(logits), tuple(calls)


@torch.no_grad()
def _no_op_answer_logits(
    model: ToyTransformer,
    corpus: NativeBoundaryCorpus,
) -> Tensor:
    answer = _answer_rows(corpus.incoming, corpus.split)
    return model.lm_head(model.final_norm(answer)).cpu()


@torch.no_grad()
def _future_invariance_audit(
    executor: StaticTransformerSpanExecutor,
    corpus: GraphBoundaryCorpus,
    *,
    examples: int = 64,
) -> dict[str, object]:
    count = min(examples, corpus.split.samples)
    rows = torch.arange(count, dtype=torch.int64)
    incoming = corpus.incoming.index_select(0, rows)
    sequence = _sequence_context(corpus.split, rows=rows)
    positions = sequence.logical_positions
    answer_positions = corpus.split.supervised_positions.index_select(
        0,
        rows,
    ).unsqueeze(1)
    future = positions > answer_positions
    changed = incoming.clone()
    deterministic_noise = torch.arange(
        changed.numel(),
        dtype=changed.dtype,
    ).reshape_as(changed)
    changed[future] = changed[future] + 10.0 + deterministic_noise[future] * 1e-4
    first = executor.forward_components(incoming, sequence).output[
        sequence.query_valid_mask
    ]
    second = executor.forward_components(changed, sequence).output[
        sequence.query_valid_mask
    ]
    maximum = float((first - second).abs().max().item())
    return {
        "examples": count,
        "future_tensor_slots_changed": int(future.sum().item()),
        "maximum_absolute_answer_difference": maximum,
        "passed": maximum == 0.0,
    }


def _span_membership_audit(
    executor: StaticTransformerSpanExecutor,
    corpus: GraphBoundaryCorpus,
    basis_vectors: Tensor,
    *,
    examples: int = 64,
) -> dict[str, object]:
    count = min(examples, corpus.split.samples)
    rows = torch.arange(count, dtype=torch.int64)
    incoming = corpus.incoming.index_select(0, rows)
    sequence = _sequence_context(corpus.split, rows=rows)
    with torch.no_grad():
        output = executor.forward_components(incoming, sequence).output
    delta = (output - incoming)[sequence.query_valid_mask]
    vectors = basis_vectors.to(dtype=delta.dtype)
    projected = (delta @ vectors) @ vectors.T
    maximum = float((delta - projected).abs().max().item())
    return {
        "examples": count,
        "maximum_absolute_orthogonal_residual": maximum,
        "tolerance": 2e-5,
        "passed": maximum <= 2e-5,
    }


def _required_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_artifact_contract(
    payload: Mapping[str, object],
    report: Mapping[str, object],
) -> StaticTransformerSpanExecutor | None:
    expected_payload = {
        "schema",
        "format_version",
        "source",
        "predecessor",
        "protocol",
        "basis",
        "coordinate_scale",
        "executor",
        "analysis",
        "scientific_status",
        "contains_source_model_weights",
        "contains_compiled_executor_weights",
    }
    expected_report = {
        "schema",
        "format_version",
        "source",
        "predecessor",
        "protocol",
        "analysis",
        "scientific_status",
        "contains_source_model_weights",
        "contains_compiled_executor_weights",
    }
    if set(payload) != expected_payload:
        raise ValueError("static full-span artifact fields are invalid")
    if set(report) != expected_report:
        raise ValueError("static full-span JSON report fields are invalid")
    if (
        payload["schema"] != _SCHEMA
        or payload["format_version"] != _FORMAT_VERSION
        or report["schema"] != _SCHEMA
        or report["format_version"] != _FORMAT_VERSION
    ):
        raise ValueError("unsupported static full-span artifact")
    if payload["contains_source_model_weights"] is not False:
        raise ValueError("artifact must not contain source model weights")
    if (
        _jsonable(
            {
                key: payload[key]
                for key in expected_report
            }
        )
        != _jsonable(dict(report))
    ):
        raise ValueError("artifact and JSON report disagree")

    basis = FisherModeBasis.from_state_dict(
        _required_mapping(payload["basis"], name="basis")
    )
    scale = payload["coordinate_scale"]
    if (
        not isinstance(scale, Tensor)
        or scale.device.type != "cpu"
        or scale.ndim != 1
        or not scale.is_floating_point()
        or not torch.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError("artifact coordinate scale is invalid")
    raw_executor = payload["executor"]
    executor: StaticTransformerSpanExecutor | None = None
    analysis = _required_mapping(payload["analysis"], name="analysis")
    selection = _required_mapping(
        analysis["selection_a"],
        name="selection_a",
    )
    selected = selection.get("selected")
    if raw_executor is None:
        if payload["contains_compiled_executor_weights"] is not False:
            raise ValueError("executor weight flag is inconsistent")
        if selected is not None:
            raise ValueError("selected executor is missing from artifact")
    else:
        if payload["contains_compiled_executor_weights"] is not True:
            raise ValueError("executor weight flag is inconsistent")
        selected_mapping = _required_mapping(
            selected,
            name="selected architecture",
        )
        selected_candidate = _required_mapping(
            selected_mapping["candidate"],
            name="selected candidate",
        )
        selected_seed = int(
            selected_mapping["selected_median_strong_seed"]
        )
        runs = selection.get("runs")
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
            raise ValueError("selection runs must be a sequence")
        matching_runs = [
            _required_mapping(run, name="selection run")
            for run in runs
            if isinstance(run, Mapping)
            and isinstance(run.get("candidate"), Mapping)
            and run["candidate"].get("name") == selected_candidate.get("name")
            and int(run.get("seed", -1)) == selected_seed
        ]
        if len(matching_runs) != 1:
            raise ValueError("selected executor run binding is ambiguous")
        selected_run = matching_runs[0]
        executor = StaticTransformerSpanExecutor.from_artifact_state_dict(
            _required_mapping(raw_executor, name="executor")
        )
        if executor.config.retained_rank != scale.numel():
            raise ValueError("executor rank and coordinate scale disagree")
        expected_config = {
            "residual_width": basis.width,
            "hidden_width": int(selected_candidate["hidden_width"]),
            "layer_count": int(selected_candidate["layer_count"]),
            "head_count": int(selected_candidate["head_count"]),
            "feed_forward_width": int(
                selected_candidate["feed_forward_width"]
            ),
            "retained_rank": int(scale.numel()),
        }
        if asdict(executor.config) != expected_config:
            raise ValueError("selected executor config binding is invalid")
        if (
            executor.execution_fingerprint()
            != selected_run.get("executor_fingerprint")
        ):
            raise ValueError("selected executor fingerprint binding is invalid")
        if (
            executor.total_runtime_coefficient_count
            != int(selected_run["runtime_stored_coefficient_count"])
        ):
            raise ValueError("selected executor storage binding is invalid")
        expected_decoder = (
            basis.vectors[:, : scale.numel()].to(torch.float32)
            * scale.to(torch.float32).unsqueeze(0)
        )
        torch.testing.assert_close(
            executor.decoder.cpu(),
            expected_decoder,
            rtol=0.0,
            atol=0.0,
        )
    return executor


def verify_variable_static_full_span_artifacts(
    artifact: Path,
    *,
    report: Path | None = None,
) -> dict[str, object]:
    """Strict-load a weights-only artifact and authenticate its JSON mirror."""

    artifact_path = Path(artifact)
    report_path = (
        artifact_path.with_suffix(".json")
        if report is None
        else Path(report)
    )
    raw_payload = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    payload = _required_mapping(raw_payload, name="artifact")
    raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    report_mapping = _required_mapping(raw_report, name="JSON report")
    _validate_artifact_contract(payload, report_mapping)
    return copy.deepcopy(dict(report_mapping))


def _save_result(
    *,
    output: Path,
    source: dict[str, object],
    predecessor: dict[str, object],
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
        "predecessor": predecessor,
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(
            _jsonable(report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    verify_variable_static_full_span_artifacts(destination)
    return report


def run_variable_static_full_span_experiment(
    *,
    checkpoint: Path = DEFAULT_VARIABLE_ASSOCIATIVE_CHECKPOINT,
    predecessor: Path = DEFAULT_PREDECESSOR,
    output: Path = DEFAULT_OUTPUT,
    retained_rank: int = DEFAULT_RETAINED_RANK,
    role_sizes: Mapping[str, int] = DEFAULT_ROLE_SIZES,
    protocol_salt: str = DEFAULT_PROTOCOL_SALT,
    candidates: Sequence[StaticGraphCandidate] = DEFAULT_CANDIDATES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    training: StaticGraphTrainingConfig | None = None,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    maximum_relative_work: float = DEFAULT_MAXIMUM_RELATIVE_WORK,
    maximum_relative_storage: float = DEFAULT_MAXIMUM_RELATIVE_STORAGE,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Fit/select the graph, then fail closed through calibration and validation."""

    training_config = training or StaticGraphTrainingConfig()
    if type(retained_rank) is not int or retained_rank <= 0:
        raise ValueError("retained_rank must be positive")
    if (
        not candidates
        or any(not isinstance(candidate, StaticGraphCandidate) for candidate in candidates)
    ):
        raise ValueError("candidates must contain StaticGraphCandidate values")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    if (
        len(seeds) != 3
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("exactly three unique integer seeds are required")
    for name, value in (
        ("maximum_relative_work", maximum_relative_work),
        ("maximum_relative_storage", maximum_relative_storage),
    ):
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    checkpoint_path = Path(checkpoint)
    predecessor_path = Path(predecessor)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not predecessor_path.is_file():
        raise FileNotFoundError(predecessor_path)

    predecessor_report = verify_variable_full_span_artifacts(predecessor_path)
    predecessor_payload = _required_mapping(
        torch.load(
            predecessor_path,
            map_location="cpu",
            weights_only=True,
        ),
        name="predecessor artifact",
    )
    predecessor_source = _required_mapping(
        predecessor_payload["source"],
        name="predecessor source",
    )
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if predecessor_source.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("predecessor and source checkpoint bindings disagree")
    predecessor_analysis = _required_mapping(
        predecessor_payload["analysis"],
        name="predecessor analysis",
    )
    predecessor_calibration = _required_mapping(
        predecessor_analysis["calibration_b"],
        name="predecessor calibration",
    )
    predecessor_static = _required_mapping(
        predecessor_calibration["smallest_passing_static"],
        name="predecessor static comparator",
    )
    if predecessor_static.get("rank") != retained_rank:
        raise ValueError(
            "retained_rank must equal the predecessor's locked static rank"
        )
    basis = FisherModeBasis.from_state_dict(
        _required_mapping(predecessor_payload["basis"], name="basis")
    )

    model, splits, checkpoint_metadata = load_variable_associative_checkpoint(
        checkpoint_path
    )
    model.eval()
    model.requires_grad_(False)
    if retained_rank > model.config.d_model or basis.width != model.config.d_model:
        raise ValueError("Fisher basis does not match source residual width")
    output_site = f"layer.{model.config.n_layers - 1}.output"
    if output_site != DEFAULT_OUTPUT_SITE:
        raise ValueError("this locked protocol expects a three-layer source")
    predecessor_protocol = _required_mapping(
        predecessor_payload["protocol"],
        name="predecessor protocol",
    )
    predecessor_role_hashes = _required_mapping(
        predecessor_protocol["role_context_hashes"],
        name="predecessor role hashes",
    )
    roles, reserve_hashes = _allocate_context_roles(
        splits.train,
        predecessor_role_hashes=predecessor_role_hashes,  # type: ignore[arg-type]
        validation_hashes=splits.validation.semantic_context_hashes,
        test_hashes=splits.test.semantic_context_hashes,
        role_sizes=role_sizes,
        salt=protocol_salt,
    )
    ad_hoc_probe_role_hashes = _ad_hoc_probe_role_hashes(splits.train)
    development_overlap_audit = _development_overlap_audit(
        roles,
        reserve_hashes,
        ad_hoc_probe_role_hashes=ad_hoc_probe_role_hashes,
    )

    basis_vectors = basis.vectors[:, :retained_rank].to(torch.float32)
    if progress is not None:
        progress(
            "collecting graph_fit_a source boundaries; calibration C remains unopened"
        )
    fit_native = _collect_native_boundaries(
        model,
        roles["graph_fit_a"],
        basis_vectors=basis_vectors,
        input_site=DEFAULT_INPUT_SITE,
        output_site=output_site,
    )
    coordinate_scale = _coordinate_scale(fit_native)
    fit = _standardize_corpus(fit_native, coordinate_scale)
    stop = _standardize_corpus(
        _collect_native_boundaries(
            model,
            roles["graph_stop_a"],
            basis_vectors=basis_vectors,
            input_site=DEFAULT_INPUT_SITE,
            output_site=output_site,
        ),
        coordinate_scale,
    )
    select = _standardize_corpus(
        _collect_native_boundaries(
            model,
            roles["graph_select_a"],
            basis_vectors=basis_vectors,
            input_site=DEFAULT_INPUT_SITE,
            output_site=output_site,
        ),
        coordinate_scale,
    )
    decoder = basis_vectors * coordinate_scale.unsqueeze(0)

    run_records: list[dict[str, object]] = []
    run_artifacts: dict[tuple[str, int], dict[str, object]] = {}
    for candidate in candidates:
        for seed in seeds:
            record, artifact = _train_candidate_seed(
                candidate,
                seed=seed,
                model=model,
                fit=fit,
                stop=stop,
                select=select,
                decoder=decoder,
                training=training_config,
                progress=progress,
            )
            executor = StaticTransformerSpanExecutor.from_artifact_state_dict(
                artifact
            )
            record["accounting"] = _compute_accounting(
                model,
                roles["graph_select_a"],
                executor,
            )
            run_records.append(record)
            run_artifacts[(candidate.name, seed)] = artifact

    selection = _select_architecture_and_seed(run_records)
    selected_key = selection.pop("selected_runtime_key")
    selected_artifact: dict[str, object] | None = None
    selected_executor: StaticTransformerSpanExecutor | None = None
    calibration: dict[str, object] | None = None
    validation: dict[str, object] | None = None

    if selected_key is not None:
        selected_artifact = run_artifacts[selected_key]  # type: ignore[index]
        selected_executor = (
            StaticTransformerSpanExecutor.from_artifact_state_dict(
                selected_artifact
            )
        )
        if progress is not None:
            progress(
                f"selection A froze {selected_key[0]} seed={selected_key[1]}; "
                "opening calibration C"
            )
        calibration_native = _collect_native_boundaries(
            model,
            roles["calibration_c"],
            basis_vectors=basis_vectors,
            input_site=DEFAULT_INPUT_SITE,
            output_site=output_site,
        )
        calibration_corpus = _standardize_corpus(
            calibration_native,
            coordinate_scale,
        )
        boundary_behavior, boundary_logits = _evaluate_executor(
            selected_executor,
            model,
            calibration_corpus,
        )
        direct_logits, source_layer_calls = _direct_replacement_answer_logits(
            model,
            selected_executor,
            roles["calibration_c"],
        )
        direct_boundary_maximum = float(
            (direct_logits - boundary_logits).abs().max().item()
        )
        direct_behavior = _behavior_summary(
            direct_logits,
            calibration_native.teacher_logits,
            roles["calibration_c"],
        )
        static_mask = torch.zeros(model.config.d_model, dtype=torch.bool)
        static_mask[:retained_rank] = True
        representation_oracle_logits = _projected_answer_logits(
            model,
            roles["calibration_c"],
            basis,
            input_site=DEFAULT_INPUT_SITE,
            output_site=output_site,
            static_mask=static_mask,
        )
        oracle_behavior = _behavior_summary(
            representation_oracle_logits,
            calibration_native.teacher_logits,
            roles["calibration_c"],
        )
        no_op_logits = _no_op_answer_logits(model, calibration_native)
        no_op_behavior = _behavior_summary(
            no_op_logits,
            calibration_native.teacher_logits,
            roles["calibration_c"],
        )
        bootstrap = _bootstrap_nll_degradation(
            direct_logits,
            calibration_native.teacher_logits,
            roles["calibration_c"],
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        )
        accounting = _compute_accounting(
            model,
            roles["calibration_c"],
            selected_executor,
        )
        reloaded = StaticTransformerSpanExecutor.from_artifact_state_dict(
            selected_artifact
        )
        reloaded_logits, reloaded_source_calls = (
            _direct_replacement_answer_logits(
                model,
                reloaded,
                roles["calibration_c"],
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
            "no_prior_development_context_overlap": (
                development_overlap_audit["calibration_c_confirmatory"]
                is True
            ),
            "strong_behavior": direct_behavior["strong_passed"] is True,
            "context_bootstrap_upper_at_most_0.01": (
                float(bootstrap["upper_95_percent"]) <= 0.01
            ),
            "zero_source_layer_calls": (
                source_layer_calls == (0,) * model.config.n_layers
            ),
            "source_free_executor_state": source_free_state,
            "boundary_and_direct_paths_match": direct_boundary_maximum == 0.0,
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
        }
        calibration = {
            "baseline": asdict(
                variable_associative_metrics_from_logits(
                    roles["calibration_c"],
                    calibration_native.teacher_logits,
                )
            ),
            "replacement": direct_behavior,
            "boundary_replay": boundary_behavior,
            "rank_14_native_output_representation_oracle": oracle_behavior,
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
                progress("calibration C passed; evaluating official validation once")
            validation_native_logits = variable_associative_answer_logits(
                model,
                splits.validation,
            )
            validation_logits, validation_source_calls = (
                _direct_replacement_answer_logits(
                    model,
                    selected_executor,
                    splits.validation,
                )
            )
            validation_behavior = _behavior_summary(
                validation_logits,
                validation_native_logits,
                splits.validation,
            )
            validation_gates = {
                "strong_behavior": (
                    validation_behavior["strong_passed"] is True
                ),
                "zero_source_layer_calls": (
                    validation_source_calls
                    == (0,) * model.config.n_layers
                ),
            }
            validation = {
                "baseline": asdict(
                    variable_associative_metrics_from_logits(
                        splits.validation,
                        validation_native_logits,
                    )
                ),
                "replacement": validation_behavior,
                "source_layer_call_counts": validation_source_calls,
                "gates": validation_gates,
                "passed": all(validation_gates.values()),
            }

    analysis = {
        "development_overlap_audit": development_overlap_audit,
        "selection_a": {
            "runs": run_records,
            **selection,
        },
        "calibration_c": calibration,
        "validation": validation,
    }
    calibration_passed = calibration is not None and calibration["passed"] is True
    validation_passed = validation is not None and validation["passed"] is True
    scientific_status = {
        "source_model_frozen": True,
        "full_transformer_span_replaced": selected_executor is not None,
        "fixed_rank_fisher_output_span": retained_rank,
        "source_independent_graph_fitted": selected_executor is not None,
        "calibration_c_evaluated": calibration is not None,
        "calibration_c_passed": calibration_passed,
        "calibration_c_confirmatory": (
            development_overlap_audit["calibration_c_confirmatory"] is True
        ),
        "calibration_c_exploratory_due_to_prior_probe_overlap": (
            development_overlap_audit["calibration_c_confirmatory"] is False
        ),
        "validation_evaluated": validation is not None,
        "validation_passed": validation_passed,
        "test_evaluated": False,
        "zero_source_layer_calls_in_replacement": (
            calibration is not None
            and calibration["gates"]["zero_source_layer_calls"] is True  # type: ignore[index]
        ),
        "ideal_compute_and_storage_reduction_achieved": (
            calibration is not None
            and calibration["gates"][  # type: ignore[index]
                "ideal_complete_macs_at_most_90_percent_native"
            ]
            is True
            and calibration["gates"][  # type: ignore[index]
                "stored_coefficients_at_most_90_percent_native_span"
            ]
            is True
        ),
        "reference_kernel_wall_clock_speed_claim": False,
        "scope_is_query_sparse_associative_recall_not_general_lm": True,
        "model_level_viable": validation_passed,
    }
    source = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_fingerprint": module_state_fingerprint(model),
        "dataset_sha256": checkpoint_metadata["dataset_sha256"],
        "model_config": asdict(model.config),
    }
    predecessor_record = {
        "artifact": str(predecessor_path),
        "artifact_sha256": _file_sha256(predecessor_path),
        "report_sha256": _file_sha256(predecessor_path.with_suffix(".json")),
        "schema": predecessor_payload["schema"],
        "locked_static_rank": retained_rank,
        "predecessor_model_level_eligible": predecessor_report[
            "scientific_status"
        ]["model_level_eligible"],
    }
    protocol = {
        "input_site": DEFAULT_INPUT_SITE,
        "output_site": output_site,
        "source_layers": model.config.n_layers,
        "retained_rank": retained_rank,
        "role_allocation": (
            "salted SHA-256 rank over whole contexts fresh relative to the "
            "artifacted predecessor; a separate audit records the earlier "
            "interactive-probe overlap"
        ),
        "role_salt": protocol_salt,
        "role_sizes": dict(role_sizes),
        "role_context_hashes": {
            name: split.semantic_context_hashes
            for name, split in roles.items()
        },
        "excluded_predecessor_role_context_hashes": {
            str(name): tuple(hashes)
            for name, hashes in predecessor_role_hashes.items()
        },
        "recorded_ad_hoc_probe_role_context_hashes": (
            ad_hoc_probe_role_hashes
        ),
        "reserve_context_hashes": reserve_hashes,
        "validation_context_hashes": splits.validation.semantic_context_hashes,
        "test_context_hashes": splits.test.semantic_context_hashes,
        "candidate_grid": tuple(asdict(candidate) for candidate in candidates),
        "seeds": tuple(seeds),
        "architecture_pass_rule": "at least two of three strong graph_select_a seeds",
        "checkpoint_rule": (
            "strong pass, minimum pass, hard NLL, modal MSE, then earlier step"
        ),
        "seed_rule": "median hard-NLL strong seed",
        "architecture_rule": (
            "lowest ideal complete MACs, then storage, NLL, and name"
        ),
        "training": asdict(training_config),
        "strong_behavior_thresholds": {
            "delta_nll": 0.01,
            "answer_accuracy": 1.0,
            "paired_context_accuracy": 1.0,
            "minimum_stratum_accuracy": 1.0,
            "top1_agreement": 1.0,
            "native_teacher_kl": 0.01,
            "p90_absolute_delta_nll": 0.025,
        },
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_upper_degradation_threshold": 0.01,
        "maximum_relative_work": maximum_relative_work,
        "maximum_relative_storage": maximum_relative_storage,
        "calibration_policy": (
            "unopened within the artifacted run until architecture, seed, "
            "checkpoint, rank, and decoder freeze; post-run audit demotes "
            "prior-probe overlap to exploratory evidence"
        ),
        "validation_policy": (
            "evaluate once only after calibration C independence and joint pass"
        ),
        "test_policy": "hash-only; never evaluate in this command",
    }
    return _save_result(
        output=Path(output),
        source=source,
        predecessor=predecessor_record,
        protocol=protocol,
        basis=basis,
        coordinate_scale=coordinate_scale,
        executor_artifact=selected_artifact,
        analysis=analysis,
        scientific_status=scientific_status,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a fixed-rank source-independent mini-transformer graph across "
            "the complete variable associative-recall source span."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_VARIABLE_ASSOCIATIVE_CHECKPOINT,
    )
    parser.add_argument(
        "--predecessor",
        type=Path,
        default=DEFAULT_PREDECESSOR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--evaluation-interval", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_variable_static_full_span_experiment(
        checkpoint=arguments.checkpoint,
        predecessor=arguments.predecessor,
        output=arguments.output,
        training=StaticGraphTrainingConfig(
            max_steps=arguments.max_steps,
            evaluation_interval=arguments.evaluation_interval,
            batch_size=arguments.batch_size,
        ),
        bootstrap_samples=arguments.bootstrap_samples,
        progress=None if arguments.quiet else print,
    )
    selection = report["analysis"]["selection_a"]  # type: ignore[index]
    print(
        json.dumps(
            {
                "artifact": str(arguments.output),
                "selected": selection["selected"],  # type: ignore[index]
                "calibration_c": report["analysis"]["calibration_c"],  # type: ignore[index]
                "validation": report["analysis"]["validation"],  # type: ignore[index]
                "scientific_status": report["scientific_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
