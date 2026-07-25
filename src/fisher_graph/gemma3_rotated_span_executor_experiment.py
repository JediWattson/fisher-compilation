"""Train and evaluate a true Gemma block replacement in a locked span.

This rung consumes a validated codimension-one rotation artifact.  Unlike the
earlier intervention experiments, the student path never executes the source
layers in the selected block.  It runs:

```
native prefix -> rotated-span causal executor -> native suffix -> LM head
```

The source model remains frozen.  Calibration A supplies a deterministic
ridge initialization and fixed-step downstream distillation.  Calibration B
is used once as a pass/fail lock; validation is tokenized only when the frozen
executor passes every calibration-B fidelity and resource gate.  Test prompts
are parsed and hashed but never tokenized by this command.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import (
    Gemma3CausalLMAdapter,
    LayerBlockBoundaryPlan,
    ModelAdapter,
    SequenceContext,
)
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .gated_executor import (
    GatedCausalModalExecutorConfig,
    ResidualGatedCausalModalExecutor,
)
from .gemma3_ablation_experiment import (
    _FrozenModelTensorGuard,
    _is_sha256,
    _update_payload_digest,
    _validate_model_metadata,
)
from .gemma3_codimension_rotation_experiment import (
    _file_sha256,
    _semantic_numeric_equal,
    load_gemma3_codimension_rotation_artifact,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_gated_executor_experiment import (
    _BoundaryBatch,
    _aggregate_direct_examples,
    _behavior_aggregate,
    _behavior_examples,
    _direct_example,
    _materialize_split,
    _percentile,
    _source_block_macs,
    _source_block_static,
)
from .gemma3_stability_experiment import (
    _library_versions,
    _ordered_prompt_hash_digest,
    _tokenizer_provenance,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .modal_ablation import _causal_lm_batch_scores, _example_ids
from .rotated_span_executor import (
    RotatedSpanBlockExecutor,
    deterministic_orthogonal_complement,
)


DEFAULT_PROMPT_SPLITS = Path(
    "examples/gemma3_rotated_span_executor_prompts.json"
)
DEFAULT_EXPERT_COUNT = 2
DEFAULT_EXPERT_RANK = 16
DEFAULT_ROUTER_WIDTH = 16
DEFAULT_MAX_POSITIVE_LAG: int | None = None
DEFAULT_MODAL_WARMUP_STEPS = 100
DEFAULT_MODAL_WARMUP_LEARNING_RATE = 1e-3
DEFAULT_TRAIN_STEPS = 64
DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE = 4
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_RIDGE_REGULARIZATION = 1e-3
DEFAULT_GROUND_TRUTH_WEIGHT = 1.0
DEFAULT_TEACHER_KL_WEIGHT = 1.0

DEFAULT_NLL_ATOL = 0.05
DEFAULT_TOP1_MIN = 0.95
DEFAULT_TEACHER_KL_MAX = 0.05
DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX = 0.10
DEFAULT_PER_PROMPT_P10_TOP1_MIN = 0.90
DEFAULT_MAX_STORED_COEFFICIENT_RATIO = 0.75
DEFAULT_MAX_ANALYTIC_MAC_RATIO = 0.75
DEFAULT_ORACLE_NLL_ATOL = 0.05
DEFAULT_ORACLE_TOP1_MIN = 0.95

_PROMPT_STATUS = (
    "rotated_span_executor_fresh_train_b_validation_test_hash_only"
)
_ARTIFACT_SCHEMA = "fisher_graph.gemma3_rotated_span_executor"
_ARTIFACT_FORMAT_VERSION = 1
_PAYLOAD_DOMAIN = b"fisher_graph.gemma3_rotated_span_executor_payload.v1\0"
_REPORT_DOMAIN = b"fisher_graph.gemma3_rotated_span_executor_report.v1\0"


def default_gemma3_rotated_span_executor_output(
    model_id: str = DEFAULT_MODEL_ID,
    start_layer: int = 4,
    end_layer: int = 6,
) -> Path:
    """Return an ignored model/block-specific replacement artifact path."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if (
        type(start_layer) is not int
        or type(end_layer) is not int
        or start_layer < 0
        or end_layer < start_layer
    ):
        raise ValueError("layer range must be nonnegative and ascending")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip("._-")
    return (
        Path(".local-runs")
        / (slug or "gemma3-model")
        / f"layers-{start_layer}-{end_layer}-rotated-span-executor.pt"
    )


def _scientific_payload_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, payload)
    return digest.hexdigest()


def _report_sha256(report: Mapping[str, object]) -> str:
    encoded = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_REPORT_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor, *, domain: bytes) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return result


def _prompt_hashes(metadata: Mapping[str, object]) -> set[str]:
    per_prompt = metadata.get("per_prompt_sha256")
    if not isinstance(per_prompt, Mapping):
        raise ValueError("prompt metadata is missing per-prompt hashes")
    values: set[str] = set()
    for hashes in per_prompt.values():
        if not isinstance(hashes, (list, tuple)):
            raise ValueError("per-prompt hashes must be sequences")
        for value in hashes:
            if not _is_sha256(value) or value in values:
                raise ValueError("prompt hashes are invalid or duplicated")
            values.add(value)
    return values


def _assert_fresh_prompt_disjointness(
    *,
    fresh: Mapping[str, object],
    rotation: Mapping[str, object],
) -> dict[str, object]:
    """Bind the new fixture against every recursively recorded predecessor."""

    fresh_hashes = _prompt_hashes(fresh)
    protocol = rotation["protocol"]
    source = rotation["source_projection"]
    if not isinstance(protocol, Mapping) or not isinstance(source, Mapping):
        raise ValueError("rotation prompt provenance is invalid")
    rotation_prompts = protocol.get("prompt_splits")
    source_disjointness = source.get("prompt_disjointness")
    if not isinstance(rotation_prompts, Mapping) or not isinstance(
        source_disjointness,
        Mapping,
    ):
        raise ValueError("rotation prompt lineage is missing")
    predecessor_groups = {
        "rotation": _prompt_hashes(rotation_prompts),
        "projection": set(
            source_disjointness.get("projection_prompt_sha256", ())
        ),
        "weighted": set(
            source_disjointness.get("weighted_prompt_sha256", ())
        ),
        "gated": set(
            source_disjointness.get("gated_prompt_sha256", ())
        ),
    }
    for label, values in predecessor_groups.items():
        if not values or any(not _is_sha256(value) for value in values):
            raise ValueError(f"{label} prompt lineage is invalid")
        overlap = fresh_hashes & values
        if overlap:
            raise ValueError(
                f"executor prompts overlap the {label} predecessor"
            )
    return {
        "fresh_prompt_sha256": tuple(sorted(fresh_hashes)),
        "predecessor_prompt_sha256": {
            label: tuple(sorted(values))
            for label, values in predecessor_groups.items()
        },
        "fresh_count": len(fresh_hashes),
        "predecessor_counts": {
            label: len(values)
            for label, values in predecessor_groups.items()
        },
        "overlap_counts": {
            label: 0 for label in predecessor_groups
        },
        "verified_before_model_load_or_tokenization": True,
    }


@dataclass(frozen=True, slots=True)
class _NativeStackResult:
    sequence: SequenceContext
    block_input: Tensor
    block_output: Tensor
    final_hidden: Tensor
    logits: Tensor | None
    selected_logits: Tensor | None


@dataclass(frozen=True, slots=True)
class _TrainingBatch:
    batch: CalibrationBatch
    block_input: Tensor
    block_output: Tensor
    selected_positions: Tensor
    teacher_logits: Tensor
    ground_truth_targets: Tensor
    example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        selected_count = int(self.selected_positions.sum().item())
        if (
            self.block_input.shape != self.block_output.shape
            or self.block_input.ndim != 3
            or self.selected_positions.shape
            != self.block_input.shape[:2]
            or self.selected_positions.dtype is not torch.bool
            or self.teacher_logits.ndim != 2
            or self.teacher_logits.shape[0] != selected_count
            or self.ground_truth_targets.shape != (selected_count,)
            or len(self.example_ids) != self.block_input.shape[0]
        ):
            raise ValueError("training batch tensors are inconsistent")


def _selected_training_positions(
    batch: CalibrationBatch,
    *,
    positions_per_sequence: int,
) -> Tensor:
    if type(positions_per_sequence) is not int or positions_per_sequence <= 0:
        raise ValueError("positions_per_sequence must be positive")
    supervised = batch.targets != CausalLanguageModelNLL().ignore_index
    selected = torch.zeros_like(supervised)
    for row in range(batch.batch_size):
        indexes = supervised[row].nonzero(as_tuple=False).flatten()
        count = int(indexes.numel())
        if count == 0:
            raise ValueError("training sequence has no supervised positions")
        if count <= positions_per_sequence:
            chosen = indexes
        else:
            offsets = torch.linspace(
                0,
                count - 1,
                positions_per_sequence,
                dtype=torch.float64,
                device=indexes.device,
            ).round().to(torch.long)
            chosen = indexes[offsets].unique(sorted=True)
        selected[row, chosen] = True
    return selected


def _project_selected_logits(
    adapter: ModelAdapter,
    hidden_states: Tensor,
    sequence: SequenceContext,
    selected_positions: Tensor,
) -> Tensor:
    """Project only selected rows when the live Gemma head is available."""

    if (
        selected_positions.shape != hidden_states.shape[:2]
        or selected_positions.dtype is not torch.bool
        or not selected_positions.any()
    ):
        raise ValueError("selected_positions must select hidden-state rows")
    module = adapter.module
    backbone = getattr(module, "model", None)
    norm = getattr(backbone, "norm", None)
    get_output_embeddings = getattr(module, "get_output_embeddings", None)
    output_embedding = (
        get_output_embeddings()
        if callable(get_output_embeddings)
        else None
    )
    if isinstance(norm, nn.Module) and isinstance(
        output_embedding,
        nn.Module,
    ):
        selected = hidden_states[selected_positions]
        logits = output_embedding(norm(selected))
        config = getattr(module, "config", None)
        softcap = getattr(config, "final_logit_softcapping", None)
        if softcap is not None:
            cap = float(softcap)
            logits = torch.tanh(logits / cap) * cap
        return logits
    return adapter.project_logits(
        hidden_states,
        sequence,
    )[selected_positions]


def _run_native_stack(
    adapter: ModelAdapter,
    batch: CalibrationBatch,
    *,
    plan: LayerBlockBoundaryPlan,
    selected_positions: Tensor | None = None,
    full_logits: bool,
) -> _NativeStackResult:
    """Run every native segment through public adapter boundaries."""

    sequence = adapter.prepare_sequence(batch.model_inputs)
    current = adapter.embed(batch.model_inputs, sequence).hidden_states
    block_input: Tensor | None = None
    block_output: Tensor | None = None
    for segment in adapter.segments:
        if segment.ordinal == plan.start_ordinal:
            block_input = current
        current = adapter.run_segment(
            segment,
            current,
            sequence,
        ).hidden_states
        if segment.ordinal == plan.end_ordinal:
            block_output = current
    if block_input is None or block_output is None:
        raise RuntimeError("native stack did not cross the selected block")
    logits = (
        adapter.project_logits(current, sequence)
        if full_logits
        else None
    )
    selected_logits = (
        None
        if selected_positions is None
        else _project_selected_logits(
            adapter,
            current,
            sequence,
            selected_positions,
        )
    )
    return _NativeStackResult(
        sequence=sequence,
        block_input=block_input,
        block_output=block_output,
        final_hidden=current,
        logits=logits,
        selected_logits=selected_logits,
    )


def _run_suffix_from_boundary(
    adapter: ModelAdapter,
    batch: CalibrationBatch,
    *,
    plan: LayerBlockBoundaryPlan,
    sequence: SequenceContext,
    boundary_output: Tensor,
    selected_positions: Tensor | None = None,
    full_logits: bool,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    current = boundary_output
    for segment in adapter.segments:
        if segment.ordinal <= plan.end_ordinal:
            continue
        current = adapter.run_segment(
            segment,
            current,
            sequence,
        ).hidden_states
    logits = (
        adapter.project_logits(current, sequence)
        if full_logits
        else None
    )
    selected_logits = (
        None
        if selected_positions is None
        else _project_selected_logits(
            adapter,
            current,
            sequence,
            selected_positions,
        )
    )
    return current, logits, selected_logits


def _run_replacement_stack(
    adapter: ModelAdapter,
    batch: CalibrationBatch,
    *,
    plan: LayerBlockBoundaryPlan,
    executor: RotatedSpanBlockExecutor,
    selected_positions: Tensor | None = None,
    full_logits: bool,
) -> _NativeStackResult:
    """Run prefix, one grouped executor, and suffix, skipping the block."""

    sequence = adapter.prepare_sequence(batch.model_inputs)
    current = adapter.embed(batch.model_inputs, sequence).hidden_states
    executed_prefix: list[str] = []
    for segment in adapter.segments:
        if segment.ordinal >= plan.start_ordinal:
            break
        current = adapter.run_segment(
            segment,
            current,
            sequence,
        ).hidden_states
        executed_prefix.extend(segment.layer_ids)
    block_input = current
    predicted_boundary = executor(block_input, sequence)
    block_output = predicted_boundary
    executed_suffix: list[str] = []
    for segment in adapter.segments:
        if segment.ordinal <= plan.end_ordinal:
            continue
        block_output_run = adapter.run_segment(
            segment,
            block_output,
            sequence,
        )
        block_output = block_output_run.hidden_states
        executed_suffix.extend(segment.layer_ids)
    final_hidden = block_output
    logits = (
        adapter.project_logits(final_hidden, sequence)
        if full_logits
        else None
    )
    selected_logits = (
        None
        if selected_positions is None
        else _project_selected_logits(
            adapter,
            final_hidden,
            sequence,
            selected_positions,
        )
    )
    skipped = set(plan.layer_ids)
    if skipped & set((*executed_prefix, *executed_suffix)):
        raise RuntimeError("replacement path executed a skipped source layer")
    return _NativeStackResult(
        sequence=sequence,
        block_input=block_input,
        block_output=predicted_boundary,
        final_hidden=final_hidden,
        logits=logits,
        selected_logits=selected_logits,
    )


def _collect_training_batches(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    plan: LayerBlockBoundaryPlan,
    positions_per_sequence: int,
) -> tuple[_TrainingBatch, ...]:
    """Capture frozen teacher boundaries and selected downstream targets."""

    module = adapter.module
    if module.training or any(
        parameter.requires_grad for parameter in module.parameters()
    ):
        raise ValueError("teacher collection requires a frozen eval model")
    values = []
    sequence_offset = 0
    with torch.no_grad():
        for batch in batches:
            selected = _selected_training_positions(
                batch,
                positions_per_sequence=positions_per_sequence,
            )
            native = _run_native_stack(
                adapter,
                batch,
                plan=plan,
                selected_positions=selected,
                full_logits=False,
            )
            if native.selected_logits is None:
                raise RuntimeError("teacher selected logits were not produced")
            ids = _example_ids(batch, sequence_offset=sequence_offset)
            sequence_offset += batch.batch_size
            values.append(
                _TrainingBatch(
                    batch=batch,
                    block_input=native.block_input.detach().clone(),
                    block_output=native.block_output.detach().clone(),
                    selected_positions=selected.detach().clone(),
                    teacher_logits=native.selected_logits.detach()
                    .to(device="cpu", dtype=torch.bfloat16)
                    .clone(),
                    ground_truth_targets=batch.targets[selected]
                    .detach()
                    .to(device="cpu", dtype=torch.long)
                    .clone(),
                    example_ids=ids,
                )
            )
    if not values:
        raise ValueError("training split cannot be empty")
    return tuple(values)


def _initialize_rotated_executor(
    training: Sequence[_TrainingBatch],
    *,
    normal: Tensor,
    expert_count: int,
    expert_rank: int,
    router_width: int,
    max_positive_lag: int | None,
    ridge_regularization: float,
    seed: int,
    device: torch.device,
) -> tuple[RotatedSpanBlockExecutor, dict[str, object]]:
    """Ridge-initialize the same-position path on projected native deltas."""

    if not training:
        raise ValueError("training batches cannot be empty")
    width = int(normal.numel())
    inputs = []
    targets = []
    basis = deterministic_orthogonal_complement(normal)
    for item in training:
        valid = item.batch.valid_positions
        source = item.block_input[valid].to(
            device="cpu",
            dtype=torch.float64,
        )
        target = item.block_output[valid].to(
            device="cpu",
            dtype=torch.float64,
        )
        inputs.append(source)
        targets.append((target - source) @ basis)
    design = torch.cat(inputs, dim=0)
    modal_target = torch.cat(targets, dim=0)
    input_mean = design.mean(dim=0)
    centered = design - input_mean
    input_scale = centered.square().mean(dim=0).sqrt().clamp_min(1e-4)
    normalized = centered / input_scale
    target_mean = modal_target.mean(dim=0)
    centered_target = modal_target - target_mean
    gram = normalized.T @ normalized
    scale = max(
        float(torch.diagonal(gram).mean().item()),
        torch.finfo(torch.float64).tiny,
    )
    regularized = gram + (
        ridge_regularization
        * scale
        * torch.eye(width, dtype=torch.float64)
    )
    weight = torch.linalg.solve(
        regularized,
        normalized.T @ centered_target,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        graph = ResidualGatedCausalModalExecutor(
            GatedCausalModalExecutorConfig(
                input_modes=width,
                output_modes=width - 1,
                expert_count=expert_count,
                expert_rank=expert_rank,
                router_width=router_width,
                same_position_skip=False,
                max_positive_lag=max_positive_lag,
                router_activation="tanh",
            ),
            dtype=torch.float32,
            device="cpu",
        )
    with torch.no_grad():
        graph.same_position_weight.copy_(
            weight.to(dtype=graph.dtype)
        )
        graph.same_position_bias.copy_(
            target_mean.to(dtype=graph.dtype)
        )
        # The exact ridge solution is the initialization.  Positive-lag
        # parameters enter smoothly: expert output learns first, then routing
        # and expert-input tensors receive gradients on later steps.
        graph.expert_output_weight.zero_()
    graph.to(device=device)
    executor = RotatedSpanBlockExecutor(
        normal=normal,
        input_mean=input_mean,
        input_scale=input_scale,
        graph=graph,
    ).to(device=device)

    with torch.no_grad():
        ridge_prediction = (
            normalized @ weight + target_mean
        )
        error = ridge_prediction - modal_target
        target_energy = float(modal_target.square().sum().item())
        return executor, {
            "method": "calibration_a_valid_position_ridge",
            "regularization": ridge_regularization,
            "regularization_scaled_by_mean_gram_diagonal": True,
            "valid_positions": int(design.shape[0]),
            "input_width": width,
            "output_modes": width - 1,
            "modal_mse": float(error.square().mean().item()),
            "modal_nrmse": math.sqrt(
                float(error.square().sum().item())
                / max(target_energy, torch.finfo(torch.float64).tiny)
            ),
            "input_scale_min": float(input_scale.min().item()),
            "input_scale_max": float(input_scale.max().item()),
            "input_scale_mean": float(input_scale.mean().item()),
        }


def _teacher_targets_sha256(
    training: Sequence[_TrainingBatch],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        b"fisher_graph.rotated_span_executor.teacher_top1.v1\0"
    )
    for item in training:
        ids = item.teacher_logits.float().argmax(dim=-1)
        digest.update(ids.contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fit_modal_warmup(
    adapter: ModelAdapter,
    executor: RotatedSpanBlockExecutor,
    training: Sequence[_TrainingBatch],
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> dict[str, object]:
    """Warm up the causal paths against the locked projected block delta."""

    if type(steps) is not int or steps < 0:
        raise ValueError("modal_warmup_steps must be nonnegative")
    if steps == 0:
        return {
            "steps": 0,
            "enabled": False,
            "objective": "projected_modal_block_delta_mse",
            "checkpoint_selection": "not_applicable",
        }
    parameters = tuple(executor.parameters())
    source_parameter_ids = {
        id(parameter) for parameter in adapter.module.parameters()
    }
    if any(id(parameter) in source_parameter_ids for parameter in parameters):
        raise RuntimeError("executor parameters alias source-model weights")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    executor.train()
    rows = []
    for step in range(steps):
        item = training[step % len(training)]
        sequence = adapter.prepare_sequence(item.batch.model_inputs)
        source = item.block_input.to(device=executor.graph.device)
        target = item.block_output.to(device=executor.graph.device)
        optimizer.zero_grad(set_to_none=True)
        components = executor.forward_components(source, sequence)
        target_modal = (
            (target.to(torch.float32) - source.to(torch.float32))
            @ executor.span_basis
        )
        mask = item.batch.valid_positions.unsqueeze(-1)
        squared = (
            components.modal_delta - target_modal
        ).square() * mask
        denominator = (
            int(item.batch.valid_positions.sum().item())
            * executor.retained_rank
        )
        loss = squared.sum() / denominator
        if not torch.isfinite(loss):
            raise RuntimeError("modal warmup loss is nonfinite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("modal warmup gradient is nonfinite")
        if any(
            parameter.grad is not None
            for parameter in adapter.module.parameters()
        ):
            raise RuntimeError("source-model parameter received a gradient")
        optimizer.step()
        rows.append(
            {
                "step": step + 1,
                "batch_index": step % len(training),
                "modal_mse": float(loss.detach().item()),
                "gradient_norm_before_clip": float(
                    gradient_norm.detach().item()
                ),
            }
        )
    executor.eval()
    losses = [float(row["modal_mse"]) for row in rows]
    return {
        "steps": steps,
        "enabled": True,
        "objective": "projected_modal_block_delta_mse",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "fixed_update_schedule": True,
        "early_stopping": False,
        "checkpoint_selection": "final_fixed_step_then_downstream_fit",
        "first_modal_mse": losses[0],
        "last_modal_mse": losses[-1],
        "minimum_observed_modal_mse": min(losses),
        "updates": rows,
    }


def _fit_downstream(
    adapter: ModelAdapter,
    executor: RotatedSpanBlockExecutor,
    training: Sequence[_TrainingBatch],
    *,
    plan: LayerBlockBoundaryPlan,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    ground_truth_weight: float,
    teacher_kl_weight: float,
) -> dict[str, object]:
    """Fit executor parameters through the frozen native suffix and LM head."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("train_steps must be positive")
    source_parameter_ids = {
        id(parameter) for parameter in adapter.module.parameters()
    }
    parameters = tuple(executor.parameters())
    if not parameters or any(
        id(parameter) in source_parameter_ids for parameter in parameters
    ):
        raise RuntimeError("executor parameters alias source-model weights")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    executor.train()
    rows = []
    for step in range(steps):
        item = training[step % len(training)]
        sequence = adapter.prepare_sequence(item.batch.model_inputs)
        optimizer.zero_grad(set_to_none=True)
        predicted_boundary = executor(
            item.block_input.to(device=executor.graph.device),
            sequence,
        )
        _, _, student_logits = _run_suffix_from_boundary(
            adapter,
            item.batch,
            plan=plan,
            sequence=sequence,
            boundary_output=predicted_boundary,
            selected_positions=item.selected_positions,
            full_logits=False,
        )
        if student_logits is None:
            raise RuntimeError("student selected logits were not produced")
        compute_student = student_logits.float()
        teacher_logits = item.teacher_logits.to(
            device=compute_student.device,
            dtype=torch.float32,
        )
        ground_truth = item.ground_truth_targets.to(
            device=compute_student.device,
        )
        ground_truth_ce = F.cross_entropy(
            compute_student,
            ground_truth,
            reduction="mean",
        )
        teacher_log_probabilities = F.log_softmax(
            teacher_logits,
            dim=-1,
        )
        student_log_probabilities = F.log_softmax(
            compute_student,
            dim=-1,
        )
        teacher_kl = F.kl_div(
            student_log_probabilities,
            teacher_log_probabilities,
            reduction="batchmean",
            log_target=True,
        )
        loss = (
            ground_truth_weight * ground_truth_ce
            + teacher_kl_weight * teacher_kl
        )
        if not torch.isfinite(loss):
            raise RuntimeError("downstream executor loss is nonfinite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("executor gradient norm is nonfinite")
        if any(parameter.grad is not None for parameter in adapter.module.parameters()):
            raise RuntimeError("source-model parameter received a gradient")
        optimizer.step()
        rows.append(
            {
                "step": step + 1,
                "batch_index": step % len(training),
                "selected_positions": int(
                    item.selected_positions.sum().item()
                ),
                "ground_truth_ce": float(
                    ground_truth_ce.detach().item()
                ),
                "teacher_kl": float(teacher_kl.detach().item()),
                "total_loss": float(loss.detach().item()),
                "gradient_norm_before_clip": float(
                    gradient_norm.detach().item()
                ),
            }
        )
    executor.eval()
    losses = [float(row["total_loss"]) for row in rows]
    return {
        "steps": steps,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "ground_truth_ce_weight": ground_truth_weight,
        "teacher_kl_weight": teacher_kl_weight,
        "teacher_temperature": 1.0,
        "selected_positions_per_sequence": (
            int(training[0].selected_positions.sum().item())
            // training[0].batch.batch_size
        ),
        "fixed_update_schedule": True,
        "early_stopping": False,
        "checkpoint_selection": "final_fixed_step",
        "first_total_loss": losses[0],
        "last_total_loss": losses[-1],
        "minimum_observed_total_loss": min(losses),
        "teacher_top1_ids_sha256": _teacher_targets_sha256(training),
        "source_parameter_gradients_observed": False,
        "updates": rows,
    }


def _behavior_examples_with_kl(
    *,
    batch: CalibrationBatch,
    example_ids: Sequence[str],
    baseline_logits: Tensor,
    predicted_logits: Tensor,
) -> list[dict[str, object]]:
    objective = CausalLanguageModelNLL()
    baseline = _causal_lm_batch_scores(
        baseline_logits,
        batch,
        objective=objective,
    )
    predicted = _causal_lm_batch_scores(
        predicted_logits,
        batch,
        objective=objective,
    )
    rows = _behavior_examples(
        batch=batch,
        example_ids=example_ids,
        baseline=baseline,
        predicted=predicted,
    )
    supervised = (batch.targets != objective.ignore_index).to(
        device=baseline_logits.device
    )
    teacher_log = F.log_softmax(baseline_logits.float(), dim=-1)
    student_log = F.log_softmax(predicted_logits.float(), dim=-1)
    token_kl = (
        teacher_log.exp() * (teacher_log - student_log)
    ).sum(dim=-1)
    for index, row in enumerate(rows):
        summed = float(
            token_kl[index, supervised[index]].sum().item()
        )
        row["teacher_kl_summed"] = summed
        row["teacher_kl_per_token"] = (
            summed / int(row["supervised_tokens"])
        )
    return rows


def _aggregate_behavior_with_kl(
    examples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result = _behavior_aggregate(examples)
    tokens = int(result["supervised_tokens"])
    kl_sum = sum(float(row["teacher_kl_summed"]) for row in examples)
    result["teacher_kl_summed"] = kl_sum
    result["teacher_kl_per_token"] = kl_sum / tokens
    result["per_example_teacher_kl_per_token"] = {
        "p50": _percentile(
            [
                float(row["teacher_kl_per_token"])
                for row in examples
            ],
            0.50,
        ),
        "p90": _percentile(
            [
                float(row["teacher_kl_per_token"])
                for row in examples
            ],
            0.90,
        ),
        "worst": max(
            float(row["teacher_kl_per_token"])
            for row in examples
        ),
    }
    return result


def _direct_rows(
    boundary: _BoundaryBatch,
    prediction: Tensor,
) -> list[dict[str, object]]:
    rows = []
    for index, example_id in enumerate(boundary.example_ids):
        valid = boundary.valid_positions[index]
        rows.append(
            _direct_example(
                example_id=example_id,
                valid_tokens=int(valid.sum().item()),
                source=boundary.input_hidden[index, valid].to(
                    torch.float64
                ),
                target=boundary.output_hidden[index, valid].to(
                    torch.float64
                ),
                prediction=prediction[index, valid].to(torch.float64),
            )
        )
    return rows


def _behavior_gates(
    behavior: Mapping[str, object],
    *,
    nll_atol: float,
    top1_min: float,
    teacher_kl_max: float,
    p90_abs_nll_max: float,
    p10_top1_min: float,
) -> dict[str, bool]:
    delta_distribution = behavior["per_example_delta_nll_per_token"]
    top1_distribution = behavior["per_example_top1_agreement"]
    if not isinstance(delta_distribution, Mapping) or not isinstance(
        top1_distribution,
        Mapping,
    ):
        raise ValueError("behavior distributions are invalid")
    return {
        "absolute_delta_nll_per_token": (
            abs(float(behavior["delta_nll_per_token"])) <= nll_atol
        ),
        "aggregate_top1_agreement": (
            float(behavior["top1_agreement_to_baseline"]) >= top1_min
        ),
        "teacher_kl_per_token": (
            float(behavior["teacher_kl_per_token"]) <= teacher_kl_max
        ),
        "per_prompt_p90_absolute_delta_nll": (
            float(delta_distribution["p90_absolute"])
            <= p90_abs_nll_max
        ),
        "per_prompt_p10_top1_agreement": (
            float(top1_distribution["p10"]) >= p10_top1_min
        ),
    }


def _run_replacement_with_call_audit(
    adapter: ModelAdapter,
    batch: CalibrationBatch,
    *,
    plan: LayerBlockBoundaryPlan,
    executor: RotatedSpanBlockExecutor,
    full_logits: bool,
) -> tuple[_NativeStackResult, dict[str, object]]:
    source_calls = {layer_id: 0 for layer_id in plan.layer_ids}
    executor_calls = 0
    handles = []
    for layer_id in plan.layer_ids:

        def count_source(
            _module: nn.Module,
            _args: tuple[object, ...],
            _output: object,
            *,
            selected_layer: str = layer_id,
        ) -> None:
            source_calls[selected_layer] += 1

        handles.append(
            adapter.source_module(layer_id).register_forward_hook(
                count_source
            )
        )

    def count_executor(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
    ) -> None:
        nonlocal executor_calls
        executor_calls += 1

    handles.append(executor.register_forward_hook(count_executor))
    try:
        result = _run_replacement_stack(
            adapter,
            batch,
            plan=plan,
            executor=executor,
            full_logits=full_logits,
        )
    finally:
        for handle in reversed(handles):
            handle.remove()
    if any(source_calls.values()) or executor_calls != 1:
        raise RuntimeError(
            "replacement call audit failed: "
            f"source_calls={source_calls}, executor_calls={executor_calls}"
        )
    return result, {
        "source_layer_calls": source_calls,
        "executor_calls": executor_calls,
        "source_block_calls_total": sum(source_calls.values()),
        "passed": True,
    }


def _split_evaluation(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    plan: LayerBlockBoundaryPlan,
    executor: RotatedSpanBlockExecutor,
    rotated_projector: object,
    include_controls: bool,
) -> dict[str, object]:
    """Evaluate a true replacement and optional target-informed controls."""

    behavior_rows: dict[str, list[dict[str, object]]] = {
        "executor": [],
    }
    direct_rows: dict[str, list[dict[str, object]]] = {
        "executor": [],
    }
    if include_controls:
        behavior_rows.update(
            {
                "rotated_span_oracle": [],
                "identity_block_skip": [],
            }
        )
        direct_rows.update(
            {
                "rotated_span_oracle": [],
                "identity_block_skip": [],
            }
        )
    sequence_offset = 0
    source_call_total = 0
    executor_call_total = 0
    prefix_errors = []
    out_of_span_energy = 0.0
    raw_delta_energy = 0.0
    boundaries = []

    with torch.no_grad():
        for batch in batches:
            ids = _example_ids(batch, sequence_offset=sequence_offset)
            sequence_offset += batch.batch_size
            native = _run_native_stack(
                adapter,
                batch,
                plan=plan,
                full_logits=True,
            )
            if native.logits is None:
                raise RuntimeError("native evaluation logits are missing")
            boundary = _BoundaryBatch(
                input_hidden=native.block_input.detach(),
                output_hidden=native.block_output.detach(),
                valid_positions=batch.valid_positions,
                logical_positions=native.sequence.logical_positions,
                example_ids=ids,
            )
            boundaries.append(boundary)
            replacement, audit = _run_replacement_with_call_audit(
                adapter,
                batch,
                plan=plan,
                executor=executor,
                full_logits=True,
            )
            if replacement.logits is None:
                raise RuntimeError("replacement evaluation logits are missing")
            source_call_total += int(audit["source_block_calls_total"])
            executor_call_total += int(audit["executor_calls"])
            prefix_errors.append(
                float(
                    (
                        replacement.block_input.to(torch.float64)
                        - native.block_input.to(torch.float64)
                    )
                    .abs()
                    .max()
                    .item()
                )
            )
            behavior_rows["executor"].extend(
                _behavior_examples_with_kl(
                    batch=batch,
                    example_ids=ids,
                    baseline_logits=native.logits,
                    predicted_logits=replacement.logits,
                )
            )
            direct_rows["executor"].extend(
                _direct_rows(boundary, replacement.block_output)
            )
            valid = batch.valid_positions
            delta = (
                replacement.block_output - replacement.block_input
            )[valid].to(torch.float64)
            normal = executor.normal.to(
                device=delta.device,
                dtype=torch.float64,
            )
            components = delta @ normal
            out_of_span_energy += float(components.square().sum().item())
            raw_delta_energy += float(delta.square().sum().item())

            if include_controls:
                projector = rotated_projector
                projected = projector.project_output(  # type: ignore[attr-defined]
                    native.block_input,
                    native.block_output,
                    valid_positions=batch.valid_positions,
                )
                _, oracle_logits, _ = _run_suffix_from_boundary(
                    adapter,
                    batch,
                    plan=plan,
                    sequence=native.sequence,
                    boundary_output=projected,
                    full_logits=True,
                )
                _, identity_logits, _ = _run_suffix_from_boundary(
                    adapter,
                    batch,
                    plan=plan,
                    sequence=native.sequence,
                    boundary_output=native.block_input,
                    full_logits=True,
                )
                if oracle_logits is None or identity_logits is None:
                    raise RuntimeError("control evaluation logits are missing")
                behavior_rows["rotated_span_oracle"].extend(
                    _behavior_examples_with_kl(
                        batch=batch,
                        example_ids=ids,
                        baseline_logits=native.logits,
                        predicted_logits=oracle_logits,
                    )
                )
                behavior_rows["identity_block_skip"].extend(
                    _behavior_examples_with_kl(
                        batch=batch,
                        example_ids=ids,
                        baseline_logits=native.logits,
                        predicted_logits=identity_logits,
                    )
                )
                direct_rows["rotated_span_oracle"].extend(
                    _direct_rows(boundary, projected)
                )
                direct_rows["identity_block_skip"].extend(
                    _direct_rows(boundary, native.block_input)
                )

    if source_call_total != 0 or executor_call_total != len(batches):
        raise RuntimeError("aggregate replacement call audit failed")
    return {
        "behavior": {
            name: _aggregate_behavior_with_kl(rows)
            for name, rows in behavior_rows.items()
        },
        "direct": {
            name: _aggregate_direct_examples(
                rows,
                width=executor.width,
            )
            for name, rows in direct_rows.items()
        },
        "execution_audit": {
            "batches": len(batches),
            "executor_calls": executor_call_total,
            "source_block_calls_total": source_call_total,
            "source_layer_calls": {
                layer_id: 0 for layer_id in plan.layer_ids
            },
            "maximum_prefix_boundary_replay_error": max(prefix_errors),
            "native_layers_skipped": plan.layer_ids,
            "student_path": (
                "native_prefix_then_one_grouped_executor_then_native_suffix"
            ),
            "passed": (
                source_call_total == 0
                and executor_call_total == len(batches)
                and max(prefix_errors) == 0.0
            ),
        },
        "span_audit": {
            "out_of_span_squared_energy": out_of_span_energy,
            "raw_delta_squared_energy": raw_delta_energy,
            "relative_out_of_span_energy": (
                out_of_span_energy
                / max(raw_delta_energy, torch.finfo(torch.float64).tiny)
            ),
            "maximum_allowed_relative_out_of_span_energy": 1e-6,
            "passed": (
                out_of_span_energy
                / max(raw_delta_energy, torch.finfo(torch.float64).tiny)
                <= 1e-6
            ),
        },
        "boundaries": tuple(boundaries),
    }


def _graph_structural_probes(
    adapter: ModelAdapter,
    executor: RotatedSpanBlockExecutor,
    training: Sequence[_TrainingBatch],
) -> dict[str, object]:
    """Probe tensor-slot causality and right-padding invariance."""

    first = training[0]
    sequence = adapter.prepare_sequence(first.batch.model_inputs)
    hidden = first.block_input
    with torch.no_grad():
        baseline = executor(hidden, sequence)
        changed = hidden.clone()
        causal_errors = []
        for row in range(hidden.shape[0]):
            valid_indexes = (
                first.batch.valid_positions[row]
                .nonzero(as_tuple=False)
                .flatten()
            )
            last = int(valid_indexes[-1].item())
            changed[row, last] += 7.0
            candidate = executor(changed, sequence)
            if last > 0:
                causal_errors.append(
                    float(
                        (
                            candidate[row, :last]
                            - baseline[row, :last]
                        )
                        .abs()
                        .max()
                        .item()
                    )
                )
            changed[row, last] = hidden[row, last]

        per_example_errors = []
        for index in range(first.batch.batch_size):
            sample = first.batch.sample(index)
            valid_indexes = (
                sample.valid_positions[0]
                .nonzero(as_tuple=False)
                .flatten()
            )
            trim_length = int(valid_indexes[-1].item()) + 1
            sequence_length = sample.valid_positions.shape[1]
            trimmed_inputs: dict[str, Tensor] = {}
            for name, value in sample.model_inputs.items():
                if (
                    name in sample.shared_input_names
                    and value.ndim >= 1
                    and value.shape[0] == sequence_length
                ):
                    trimmed_inputs[name] = value[:trim_length]
                elif (
                    name not in sample.shared_input_names
                    and value.ndim >= 2
                    and value.shape[1] == sequence_length
                ):
                    trimmed_inputs[name] = value[:, :trim_length]
                else:
                    trimmed_inputs[name] = value
            trimmed = CalibrationBatch(
                model_inputs=trimmed_inputs,
                targets=sample.targets[:, :trim_length],
                valid_positions=sample.valid_positions[:, :trim_length],
                shared_input_names=sample.shared_input_names,
                example_ids=sample.example_ids,
            )
            sample_sequence = adapter.prepare_sequence(
                trimmed.model_inputs
            )
            single = executor(
                hidden[index : index + 1, :trim_length],
                sample_sequence,
            )
            valid = trimmed.valid_positions[0]
            per_example_errors.append(
                float(
                    (
                        single[0, valid]
                        - baseline[index, :trim_length][valid]
                    )
                    .abs()
                    .max()
                    .item()
                )
            )
    causal_max = max(causal_errors, default=0.0)
    batching_max = max(per_example_errors, default=0.0)
    return {
        "future_slot_perturbation_max_earlier_error": causal_max,
        "future_slot_tolerance": 1e-5,
        "future_slot_causality_passed": causal_max <= 1e-5,
        "padded_batch_vs_single_max_valid_error": batching_max,
        "batching_probe_scope": (
            "batched_padded_vs_trimmed_single_right_padding"
        ),
        "batching_tolerance": 1e-5,
        "batching_equivalence_passed": batching_max <= 1e-5,
        "passed": causal_max <= 1e-5 and batching_max <= 1e-5,
    }


def _executor_accounting(
    executor: RotatedSpanBlockExecutor,
    boundaries: Sequence[_BoundaryBatch],
    *,
    source_static: Mapping[str, object],
    source_macs: Mapping[str, object],
) -> dict[str, object]:
    graph_macs = 0
    graph_breakdown: dict[str, int] = {}
    valid_positions = 0
    positive_lag_edges = 0
    for boundary in boundaries:
        accounting = executor.graph.execution_accounting(
            boundary.valid_positions.shape[1],
            batch_size=boundary.valid_positions.shape[0],
            query_valid_mask=boundary.valid_positions,
            key_valid_mask=boundary.valid_positions,
            logical_positions=boundary.logical_positions,
            key_logical_positions=boundary.logical_positions,
        )
        row = asdict(accounting)
        graph_macs += int(row["total_mac_count"])
        valid_positions += int(row["valid_query_tokens"])
        positive_lag_edges += int(row["positive_lag_edges"])
        for key, value in row.items():
            if key.endswith("_mac_count"):
                graph_breakdown[key] = (
                    graph_breakdown.get(key, 0) + int(value)
                )
    decode_macs = (
        valid_positions * executor.retained_rank * executor.width
    )
    total_macs = graph_macs + decode_macs
    learned = executor.learned_parameter_count
    fixed = executor.fixed_runtime_coefficient_count
    stored = learned + fixed
    source_parameters = int(source_static["parameter_count"])
    source_total_macs = int(source_macs["total_macs"])
    return {
        "source_block_parameter_count": source_parameters,
        "graph_trainable_parameter_count": learned,
        "fixed_runtime_coefficient_count": fixed,
        "runtime_stored_coefficient_count": stored,
        "stored_coefficient_ratio_to_source": stored / source_parameters,
        "graph_analytic_mac_count": graph_macs,
        "basis_decode_analytic_mac_count": decode_macs,
        "total_analytic_mac_count": total_macs,
        "source_block_analytic_mac_count": source_total_macs,
        "analytic_mac_ratio_to_source": total_macs / source_total_macs,
        "valid_positions": valid_positions,
        "positive_lag_edges": positive_lag_edges,
        "graph_mac_breakdown": graph_breakdown,
        "source_comparison": copy.deepcopy(dict(source_macs)),
        "normalization_elementwise_operations_excluded": True,
        "reference_kernel_speed_claim": False,
    }


def _resource_gates(
    accounting: Mapping[str, object],
    *,
    max_stored_coefficient_ratio: float,
    max_analytic_mac_ratio: float,
) -> dict[str, bool]:
    return {
        "stored_coefficient_ratio": (
            float(accounting["stored_coefficient_ratio_to_source"])
            <= max_stored_coefficient_ratio
        ),
        "analytic_mac_ratio": (
            float(accounting["analytic_mac_ratio_to_source"])
            <= max_analytic_mac_ratio
        ),
    }


def _build_report(
    payload: Mapping[str, object],
    *,
    output: Path,
    scientific_digest: str,
) -> dict[str, object]:
    training = payload["training"]
    selection = payload["selection"]
    validation = payload["validation"]
    assert isinstance(training, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(validation, Mapping)
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "scientific_status": copy.deepcopy(
            dict(payload["scientific_status"])  # type: ignore[arg-type]
        ),
        "model": copy.deepcopy(
            dict(payload["model"])  # type: ignore[arg-type]
        ),
        "source_rotation": copy.deepcopy(
            dict(payload["source_rotation"])  # type: ignore[arg-type]
        ),
        "protocol": copy.deepcopy(
            dict(payload["protocol"])  # type: ignore[arg-type]
        ),
        "analysis": {
            "training": {
                "ridge_initialization": copy.deepcopy(
                    dict(training["ridge_initialization"])  # type: ignore[arg-type]
                ),
                "modal_warmup": copy.deepcopy(
                    dict(training["modal_warmup"])  # type: ignore[arg-type]
                ),
                "downstream_fit": copy.deepcopy(
                    dict(training["downstream_fit"])  # type: ignore[arg-type]
                ),
                "structural_probes": copy.deepcopy(
                    dict(training["structural_probes"])  # type: ignore[arg-type]
                ),
                "tokenized_stream": copy.deepcopy(
                    dict(training["tokenized_stream"])  # type: ignore[arg-type]
                ),
            },
            "selection": copy.deepcopy(dict(selection)),
            "validation": copy.deepcopy(dict(validation)),
        },
        "artifact": {
            "tensor_output": output.name,
            "contains_source_model_weights": False,
            "contains_executor_weights": True,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "scientific_payload_sha256": scientific_digest,
        },
    }


def run_gemma3_rotated_span_executor(
    *,
    rotation_artifact_path: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    prompt_splits_path: Path | str = DEFAULT_PROMPT_SPLITS,
    max_length: int = 128,
    tokenization_batch_size: int = 2,
    expert_count: int = DEFAULT_EXPERT_COUNT,
    expert_rank: int = DEFAULT_EXPERT_RANK,
    router_width: int = DEFAULT_ROUTER_WIDTH,
    max_positive_lag: int | None = DEFAULT_MAX_POSITIVE_LAG,
    modal_warmup_steps: int = DEFAULT_MODAL_WARMUP_STEPS,
    modal_warmup_learning_rate: float = (
        DEFAULT_MODAL_WARMUP_LEARNING_RATE
    ),
    train_steps: int = DEFAULT_TRAIN_STEPS,
    train_positions_per_sequence: int = (
        DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE
    ),
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM,
    ridge_regularization: float = DEFAULT_RIDGE_REGULARIZATION,
    ground_truth_weight: float = DEFAULT_GROUND_TRUTH_WEIGHT,
    teacher_kl_weight: float = DEFAULT_TEACHER_KL_WEIGHT,
    selection_nll_atol: float = DEFAULT_NLL_ATOL,
    selection_top1_min: float = DEFAULT_TOP1_MIN,
    selection_teacher_kl_max: float = DEFAULT_TEACHER_KL_MAX,
    selection_p90_abs_nll_max: float = (
        DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX
    ),
    selection_p10_top1_min: float = (
        DEFAULT_PER_PROMPT_P10_TOP1_MIN
    ),
    max_stored_coefficient_ratio: float = (
        DEFAULT_MAX_STORED_COEFFICIENT_RATIO
    ),
    max_analytic_mac_ratio: float = DEFAULT_MAX_ANALYTIC_MAC_RATIO,
    seed: int = 7301,
    device_name: str = "cpu",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Train on A, gate on B, and validate a true grouped replacement."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    for label, value in (
        ("expert_count", expert_count),
        ("expert_rank", expert_rank),
        ("router_width", router_width),
        ("train_steps", train_steps),
        ("train_positions_per_sequence", train_positions_per_sequence),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be positive")
    if max_positive_lag is not None and (
        type(max_positive_lag) is not int or max_positive_lag <= 0
    ):
        raise ValueError("max_positive_lag must be positive or None")
    if type(modal_warmup_steps) is not int or modal_warmup_steps < 0:
        raise ValueError("modal_warmup_steps must be nonnegative")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    learning_rate = _finite(
        learning_rate,
        label="learning_rate",
        minimum=0.0,
    )
    modal_warmup_learning_rate = _finite(
        modal_warmup_learning_rate,
        label="modal_warmup_learning_rate",
        minimum=0.0,
    )
    weight_decay = _finite(
        weight_decay,
        label="weight_decay",
        minimum=0.0,
    )
    gradient_clip_norm = _finite(
        gradient_clip_norm,
        label="gradient_clip_norm",
        minimum=torch.finfo(torch.float64).tiny,
    )
    ridge_regularization = _finite(
        ridge_regularization,
        label="ridge_regularization",
        minimum=0.0,
    )
    ground_truth_weight = _finite(
        ground_truth_weight,
        label="ground_truth_weight",
        minimum=0.0,
    )
    teacher_kl_weight = _finite(
        teacher_kl_weight,
        label="teacher_kl_weight",
        minimum=0.0,
    )
    if ground_truth_weight == 0.0 and teacher_kl_weight == 0.0:
        raise ValueError("at least one downstream loss weight must be positive")
    thresholds = {
        "nll_atol": _finite(
            selection_nll_atol,
            label="selection_nll_atol",
            minimum=0.0,
        ),
        "top1_min": _finite(
            selection_top1_min,
            label="selection_top1_min",
            minimum=0.0,
            maximum=1.0,
        ),
        "teacher_kl_max": _finite(
            selection_teacher_kl_max,
            label="selection_teacher_kl_max",
            minimum=0.0,
        ),
        "p90_abs_nll_max": _finite(
            selection_p90_abs_nll_max,
            label="selection_p90_abs_nll_max",
            minimum=0.0,
        ),
        "p10_top1_min": _finite(
            selection_p10_top1_min,
            label="selection_p10_top1_min",
            minimum=0.0,
            maximum=1.0,
        ),
        "max_stored_coefficient_ratio": _finite(
            max_stored_coefficient_ratio,
            label="max_stored_coefficient_ratio",
            minimum=0.0,
        ),
        "max_analytic_mac_ratio": _finite(
            max_analytic_mac_ratio,
            label="max_analytic_mac_ratio",
            minimum=0.0,
        ),
    }

    source_path = Path(rotation_artifact_path)
    rotation = load_gemma3_codimension_rotation_artifact(source_path)
    rotation_model = rotation["model"]
    rotation_metadata = rotation["metadata"]
    rotation_status = rotation["report"]["scientific_status"]  # type: ignore[index]
    locked_candidate = rotation["locked_candidate"]
    rotated_projector = rotation["rotated_projector"]
    assert isinstance(rotation_model, Mapping)
    assert isinstance(rotation_metadata, Mapping)
    assert isinstance(rotation_status, Mapping)
    assert isinstance(locked_candidate, Mapping)
    protocol = rotation_metadata["protocol"]
    source_rotation = rotation_metadata["source_projection"]
    assert isinstance(protocol, Mapping)
    assert isinstance(source_rotation, Mapping)
    width = protocol.get("residual_width")
    start_layer = protocol.get("start_layer")
    end_layer = protocol.get("end_layer_inclusive")
    layer_ids = protocol.get("layer_ids")
    boundaries = protocol.get("canonical_boundaries")
    if (
        rotation_status.get("basis_ordering_supported") is not True
        or rotation_status.get("rank_639_fidelity_viable") is not True
        or rotation_status.get("selection_failed") is not False
        or rotation_status.get("test_evaluated") is not False
        or locked_candidate.get("normal_source")
        != "calibration_a_balanced_tail_rotation"
        or type(width) is not int
        or width < 2
        or locked_candidate.get("retained_rank") != width - 1
        or type(start_layer) is not int
        or type(end_layer) is not int
        or not isinstance(layer_ids, tuple)
        or not isinstance(boundaries, tuple)
    ):
        raise ValueError(
            "rotated-span executor requires the validated rotated rank-639 "
            "predecessor"
        )
    if rotation_model.get("model_id") != model_id:
        raise ValueError("requested model_id does not match rotation source")
    if revision is not None and revision not in {
        rotation_model.get("requested_revision"),
        rotation_model.get("resolved_commit"),
    }:
        raise ValueError("explicit revision does not match rotation source")

    prompts = load_gemma3_prompt_splits(prompt_splits_path)
    prompt_metadata = prompts.metadata()
    if (
        prompts.scientific_status != _PROMPT_STATUS
        or prompt_metadata["counts"]
        != {
            "calibration_a": 64,
            "calibration_b": 16,
            "validation": 16,
            "test": 16,
        }
    ):
        raise ValueError("rotated-span executor prompt fixture is noncanonical")
    prompt_disjointness = _assert_fresh_prompt_disjointness(
        fresh=prompt_metadata,
        rotation=rotation_metadata,
    )
    normal = rotated_projector.normal
    basis = deterministic_orthogonal_complement(normal)
    basis_projector_error = float(
        (
            basis @ basis.T
            - (
                torch.eye(width, dtype=torch.float64)
                - torch.outer(normal, normal)
            )
        )
        .abs()
        .max()
        .item()
    )
    basis_gram_error = float(
        (
            basis.T @ basis
            - torch.eye(width - 1, dtype=torch.float64)
        )
        .abs()
        .max()
        .item()
    )
    if basis_projector_error > 1e-10 or basis_gram_error > 1e-10:
        raise RuntimeError("locked rotated basis audit failed")

    resolved_output = (
        default_gemma3_rotated_span_executor_output(
            model_id,
            start_layer,
            end_layer,
        )
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    if resolved_output.exists() or resolved_output.with_suffix(
        ".json"
    ).exists():
        raise FileExistsError(
            "refusing to overwrite an existing rotated-span executor "
            "artifact; use a new output and a new held-out fixture for a "
            "new scientific run"
        )

    device = resolve_torch_device(device_name)
    if device.type == "mps":
        raise ValueError(
            "the rotated-span executor requires CPU or CUDA because its "
            "strict basis and diagnostic audits use float64"
        )
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    requested_revision = (
        revision
        if revision is not None
        else (
            rotation_model.get("resolved_commit")
            or rotation_model.get("requested_revision")
        )
    )
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=requested_revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    model.eval()
    model.requires_grad_(False)
    guard = _FrozenModelTensorGuard(model)
    adapter = Gemma3CausalLMAdapter(model)
    plan = adapter.plan_layer_block(start_layer, end_layer)
    if (
        plan.layer_ids != layer_ids
        or plan.activation_sites != boundaries
        or plan.widths != (width,) * len(boundaries)
    ):
        raise ValueError("live adapter block does not match rotation source")
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=requested_revision,
    )
    for field in ("model_id", "config_sha256", "hidden_size"):
        if rotation_model.get(field) != model_metadata.get(field):
            raise ValueError(f"live model {field} does not match rotation")
    if (
        rotation_model.get("resolved_commit") is not None
        and model_metadata.get("resolved_commit") is not None
        and rotation_model["resolved_commit"]
        != model_metadata["resolved_commit"]
    ):
        raise ValueError("live model commit does not match rotation")

    train_batches, train_stream = _materialize_split(
        tokenizer,
        prompts.calibration_a,
        split_name="calibration_a",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    training = _collect_training_batches(
        adapter,
        train_batches,
        plan=plan,
        positions_per_sequence=train_positions_per_sequence,
    )
    executor, ridge = _initialize_rotated_executor(
        training,
        normal=normal,
        expert_count=expert_count,
        expert_rank=expert_rank,
        router_width=router_width,
        max_positive_lag=max_positive_lag,
        ridge_regularization=ridge_regularization,
        seed=seed,
        device=device,
    )
    if {
        id(parameter) for parameter in executor.parameters()
    } & {
        id(parameter) for parameter in model.parameters()
    }:
        raise RuntimeError("executor aliases source-model parameters")
    modal_warmup = _fit_modal_warmup(
        adapter,
        executor,
        training,
        steps=modal_warmup_steps,
        learning_rate=modal_warmup_learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
    )
    downstream_fit = _fit_downstream(
        adapter,
        executor,
        training,
        plan=plan,
        steps=train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        ground_truth_weight=ground_truth_weight,
        teacher_kl_weight=teacher_kl_weight,
    )
    structural_probes = _graph_structural_probes(
        adapter,
        executor,
        training,
    )
    if structural_probes["passed"] is not True:
        raise RuntimeError("trained executor failed structural probes")
    guard.assert_unchanged()

    selection_batches, selection_stream = _materialize_split(
        tokenizer,
        prompts.calibration_b,
        split_name="calibration_b",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    selection_result = _split_evaluation(
        adapter,
        selection_batches,
        plan=plan,
        executor=executor,
        rotated_projector=rotated_projector,
        include_controls=True,
    )
    selection_boundaries = selection_result.pop("boundaries")
    selection_behavior = selection_result["behavior"]
    selection_direct = selection_result["direct"]
    assert isinstance(selection_behavior, Mapping)
    assert isinstance(selection_direct, Mapping)
    source_static = _source_block_static(adapter, plan)
    source_macs = _source_block_macs(
        adapter,
        plan,
        selection_boundaries,  # type: ignore[arg-type]
        static=source_static,
    )
    accounting = _executor_accounting(
        executor,
        selection_boundaries,  # type: ignore[arg-type]
        source_static=source_static,
        source_macs=source_macs,
    )
    behavior_gates = _behavior_gates(
        selection_behavior["executor"],  # type: ignore[arg-type]
        nll_atol=thresholds["nll_atol"],
        top1_min=thresholds["top1_min"],
        teacher_kl_max=thresholds["teacher_kl_max"],
        p90_abs_nll_max=thresholds["p90_abs_nll_max"],
        p10_top1_min=thresholds["p10_top1_min"],
    )
    oracle_gates = _behavior_gates(
        selection_behavior["rotated_span_oracle"],  # type: ignore[arg-type]
        nll_atol=DEFAULT_ORACLE_NLL_ATOL,
        top1_min=DEFAULT_ORACLE_TOP1_MIN,
        teacher_kl_max=thresholds["teacher_kl_max"],
        p90_abs_nll_max=thresholds["p90_abs_nll_max"],
        p10_top1_min=thresholds["p10_top1_min"],
    )
    resource_gates = _resource_gates(
        accounting,
        max_stored_coefficient_ratio=thresholds[
            "max_stored_coefficient_ratio"
        ],
        max_analytic_mac_ratio=thresholds[
            "max_analytic_mac_ratio"
        ],
    )
    execution_passed = (
        selection_result["execution_audit"]["passed"] is True  # type: ignore[index]
        and selection_result["span_audit"]["passed"] is True  # type: ignore[index]
    )
    selection_passed = (
        all(behavior_gates.values())
        and all(oracle_gates.values())
        and all(resource_gates.values())
        and execution_passed
    )
    guard.assert_unchanged()

    validation_evaluated = selection_passed
    validation_stream: dict[str, object] | None = None
    validation_result: dict[str, object] | None = None
    validation_gates: dict[str, bool] | None = None
    validation_accounting: dict[str, object] | None = None
    validation_passed = False
    if validation_evaluated:
        validation_batches, validation_stream = _materialize_split(
            tokenizer,
            prompts.validation,
            split_name="validation",
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        validation_result = _split_evaluation(
            adapter,
            validation_batches,
            plan=plan,
            executor=executor,
            rotated_projector=rotated_projector,
            include_controls=False,
        )
        validation_boundaries = validation_result.pop("boundaries")
        validation_behavior = validation_result["behavior"]["executor"]  # type: ignore[index]
        validation_gates = _behavior_gates(
            validation_behavior,
            nll_atol=thresholds["nll_atol"],
            top1_min=thresholds["top1_min"],
            teacher_kl_max=thresholds["teacher_kl_max"],
            p90_abs_nll_max=thresholds["p90_abs_nll_max"],
            p10_top1_min=thresholds["p10_top1_min"],
        )
        validation_accounting = _executor_accounting(
            executor,
            validation_boundaries,  # type: ignore[arg-type]
            source_static=source_static,
            source_macs=_source_block_macs(
                adapter,
                plan,
                validation_boundaries,  # type: ignore[arg-type]
                static=source_static,
            ),
        )
        validation_resource_gates = _resource_gates(
            validation_accounting,
            max_stored_coefficient_ratio=thresholds[
                "max_stored_coefficient_ratio"
            ],
            max_analytic_mac_ratio=thresholds[
                "max_analytic_mac_ratio"
            ],
        )
        validation_passed = (
            all(validation_gates.values())
            and all(validation_resource_gates.values())
            and validation_result["execution_audit"]["passed"] is True
            and validation_result["span_audit"]["passed"] is True
        )
        validation_gates.update(
            {
                f"resource_{key}": value
                for key, value in validation_resource_gates.items()
            }
        )
    guard.assert_unchanged()

    source_binding = {
        "schema": rotation["report"]["schema"],  # type: ignore[index]
        "format_version": rotation["report"]["format_version"],  # type: ignore[index]
        "scientific_payload_sha256": rotation_metadata[
            "scientific_payload_sha256"
        ],
        "report_sha256": rotation_metadata["report_sha256"],
        "tensor_file_sha256": _file_sha256(source_path),
        "locked_candidate": copy.deepcopy(dict(locked_candidate)),
        "basis_ordering_supported": True,
        "rank_639_fidelity_viable": True,
        "model_binding": {
            field: rotation_model.get(field)
            for field in (
                "model_id",
                "config_sha256",
                "resolved_commit",
                "hidden_size",
                "num_hidden_layers",
            )
        },
        "block_geometry": {
            "start_layer": start_layer,
            "end_layer_inclusive": end_layer,
            "layer_ids": layer_ids,
            "canonical_boundaries": boundaries,
            "residual_width": width,
        },
        "prompt_disjointness": prompt_disjointness,
        "normal_sha256": _tensor_sha256(
            normal,
            domain=b"fisher_graph.rotated_span.normal.v1\0",
        ),
        "basis_sha256": _tensor_sha256(
            basis,
            domain=b"fisher_graph.rotated_span.basis.v1\0",
        ),
        "basis_gram_max_absolute_error": basis_gram_error,
        "basis_projector_max_absolute_error": basis_projector_error,
    }
    tokenized_splits = {
        "calibration_a": train_stream,
        "calibration_b": selection_stream,
    }
    if validation_stream is not None:
        tokenized_splits["validation"] = validation_stream
    protocol_payload = {
        "start_layer": start_layer,
        "end_layer_inclusive": end_layer,
        "layer_ids": layer_ids,
        "canonical_boundaries": boundaries,
        "residual_width": width,
        "executor_input_modes": width,
        "executor_output_modes": width - 1,
        "executor_architecture": asdict(executor.graph.config),
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "train_steps": train_steps,
        "modal_warmup_steps": modal_warmup_steps,
        "train_positions_per_sequence": train_positions_per_sequence,
        "training_split": "calibration_a_only",
        "training_loss": (
            "ground_truth_cross_entropy_plus_native_teacher_kl"
        ),
        "initialization_and_warmup": (
            "ridge_same_position_then_fixed_projected_modal_mse_warmup"
        ),
        "fit_policy": "fixed_final_step_no_early_stopping",
        "selection_policy": (
            "single_frozen_executor_must_pass_all_calibration_b_gates"
        ),
        "validation_policy": (
            "tokenize_and_evaluate_one_locked_executor_only_if_b_passes"
        ),
        "test_policy": "parse_validate_hash_only",
        "student_execution": (
            "native_prefix_grouped_executor_native_suffix"
        ),
        "native_block_output_available_to_executor": False,
        "native_block_executed_in_student_path": False,
        "thresholds": thresholds,
        "oracle_thresholds": {
            "nll_atol": DEFAULT_ORACLE_NLL_ATOL,
            "top1_min": DEFAULT_ORACLE_TOP1_MIN,
        },
        "prompt_fixture_file_sha256": _file_sha256(
            prompt_splits_path
        ),
        "prompt_splits": prompt_metadata,
        "tokenized_splits": tokenized_splits,
        "library_versions": _library_versions(),
        "tokenizer": _tokenizer_provenance(tokenizer),
        "model_state_guard": guard.metadata(),
        "parameter_reduction_claim_requires_validation": True,
        "analytic_mac_reduction_claim_requires_validation": True,
        "latency_or_kernel_speed_claim": False,
    }
    selection_payload = {
        "executor_behavior": selection_behavior["executor"],
        "executor_direct_diagnostic": selection_direct["executor"],
        "rotated_span_oracle_behavior": selection_behavior[
            "rotated_span_oracle"
        ],
        "rotated_span_oracle_direct_diagnostic": selection_direct[
            "rotated_span_oracle"
        ],
        "identity_block_skip_behavior": selection_behavior[
            "identity_block_skip"
        ],
        "identity_block_skip_direct_diagnostic": selection_direct[
            "identity_block_skip"
        ],
        "executor_behavior_gates": behavior_gates,
        "oracle_behavior_gates": oracle_gates,
        "resource_gates": resource_gates,
        "execution_audit": selection_result["execution_audit"],
        "span_audit": selection_result["span_audit"],
        "accounting": accounting,
        "passed": selection_passed,
        "locked": selection_passed,
        "reason": (
            "single_predeclared_executor_passed_all_gates"
            if selection_passed
            else "executor_or_control_failed_calibration_b_gate"
        ),
        "tokenized_stream": selection_stream,
    }
    validation_payload = {
        "evaluated": validation_evaluated,
        "reason": (
            "one_calibration_b_locked_executor_evaluated"
            if validation_evaluated
            else "calibration_b_failed_validation_not_tokenized"
        ),
        "behavior": (
            None
            if validation_result is None
            else validation_result["behavior"]["executor"]  # type: ignore[index]
        ),
        "direct_diagnostic": (
            None
            if validation_result is None
            else validation_result["direct"]["executor"]  # type: ignore[index]
        ),
        "behavior_and_resource_gates": validation_gates,
        "execution_audit": (
            None
            if validation_result is None
            else validation_result["execution_audit"]
        ),
        "span_audit": (
            None
            if validation_result is None
            else validation_result["span_audit"]
        ),
        "accounting": validation_accounting,
        "passed": validation_passed,
        "tokenized_stream": validation_stream,
    }
    viable_replacement = selection_passed and validation_passed
    parameter_reduction_supported = (
        validation_accounting is not None
        and float(accounting["stored_coefficient_ratio_to_source"]) < 1.0
        and float(
            validation_accounting[
                "stored_coefficient_ratio_to_source"
            ]
        )
        < 1.0
    )
    analytic_mac_reduction_supported = (
        validation_accounting is not None
        and float(accounting["analytic_mac_ratio_to_source"]) < 1.0
        and float(
            validation_accounting["analytic_mac_ratio_to_source"]
        )
        < 1.0
    )
    payload = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_executor_weights": True,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "scientific_status": {
            "scope": (
                "source_independent_grouped_rotated_span_executor_"
                "replacement"
            ),
            "calibration_a_executor_fitted": True,
            "calibration_b_evaluated": True,
            "calibration_b_passed": selection_passed,
            "validation_evaluated": validation_evaluated,
            "validation_passed": validation_passed,
            "test_evaluated": False,
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "executor_weights_in_artifact": True,
            "prompt_text_in_artifact": False,
            "native_source_block_calls_in_student_path": 0,
            "source_block_removed_from_student_path": True,
            "locked_rotated_span_enforced_by_construction": True,
            "source_independent_executor": True,
            "fidelity_viable_replacement": viable_replacement,
            "parameter_reduction_supported": (
                parameter_reduction_supported
            ),
            "analytic_mac_reduction_supported": (
                analytic_mac_reduction_supported
            ),
            "latency_or_kernel_speed_claim": False,
        },
        "model": model_metadata,
        "source_rotation": source_binding,
        "protocol": protocol_payload,
        "executor": executor.artifact_state_dict(),
        "training": {
            "ridge_initialization": ridge,
            "modal_warmup": modal_warmup,
            "downstream_fit": downstream_fit,
            "structural_probes": structural_probes,
            "tokenized_stream": train_stream,
        },
        "selection": selection_payload,
        "validation": validation_payload,
    }
    digest = _scientific_payload_sha256(payload)
    report = _build_report(
        payload,
        output=resolved_output,
        scientific_digest=digest,
    )
    report_digest = _report_sha256(report)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **payload,
            "scientific_payload_sha256": digest,
            "report_sha256": report_digest,
        },
        resolved_output,
    )
    resolved_output.with_suffix(".json").write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def load_gemma3_rotated_span_executor_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strictly load and authenticate a rotated-span executor artifact."""

    artifact_path = Path(path)
    raw = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    required = {
        "schema",
        "format_version",
        "contains_model_weights",
        "contains_executor_weights",
        "contains_prompt_text",
        "contains_tokenizer_state",
        "scientific_status",
        "model",
        "source_rotation",
        "protocol",
        "executor",
        "training",
        "selection",
        "validation",
        "scientific_payload_sha256",
        "report_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("rotated-span executor artifact fields are invalid")
    if (
        raw["schema"] != _ARTIFACT_SCHEMA
        or raw["format_version"] != _ARTIFACT_FORMAT_VERSION
        or raw["contains_model_weights"] is not False
        or raw["contains_executor_weights"] is not True
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
        or not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError("unsupported or unsafe rotated-span executor artifact")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    digest = _scientific_payload_sha256(payload)
    if digest != raw["scientific_payload_sha256"]:
        raise ValueError(
            "rotated-span executor scientific digest mismatch"
        )
    model = _validate_model_metadata(raw["model"])
    source = raw["source_rotation"]
    protocol = raw["protocol"]
    executor_state = raw["executor"]
    training = raw["training"]
    selection = raw["selection"]
    validation = raw["validation"]
    status = raw["scientific_status"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            source,
            protocol,
            executor_state,
            training,
            selection,
            validation,
            status,
        )
    ):
        raise ValueError("rotated-span executor payload mappings are invalid")
    for field in (
        "scientific_payload_sha256",
        "report_sha256",
        "tensor_file_sha256",
        "normal_sha256",
        "basis_sha256",
    ):
        if not _is_sha256(source.get(field)):
            raise ValueError("rotated-span source digest is invalid")
    width = protocol.get("residual_width")
    start = protocol.get("start_layer")
    end = protocol.get("end_layer_inclusive")
    layer_ids = protocol.get("layer_ids")
    boundaries = protocol.get("canonical_boundaries")
    if (
        type(width) is not int
        or width < 2
        or type(start) is not int
        or type(end) is not int
        or start < 0
        or end < start
        or not isinstance(layer_ids, tuple)
        or len(layer_ids) != end - start + 1
        or not isinstance(boundaries, tuple)
        or len(boundaries) != len(layer_ids) + 1
        or protocol.get("executor_input_modes") != width
        or protocol.get("executor_output_modes") != width - 1
        or protocol.get("native_block_executed_in_student_path") is not False
        or protocol.get("test_policy") != "parse_validate_hash_only"
        or not isinstance(protocol.get("tokenized_splits"), Mapping)
        or "test" in protocol["tokenized_splits"]
    ):
        raise ValueError("rotated-span executor protocol is invalid")

    validation_evaluated_status = status.get("validation_evaluated")
    prompt_metadata = protocol.get("prompt_splits")
    tokenized_splits = protocol.get("tokenized_splits")
    split_names = (
        "calibration_a",
        "calibration_b",
        "validation",
        "test",
    )
    if (
        type(validation_evaluated_status) is not bool
        or not isinstance(prompt_metadata, Mapping)
        or set(prompt_metadata)
        != {
            "scientific_status",
            "counts",
            "normalized_sha256",
            "per_prompt_sha256",
        }
        or prompt_metadata.get("scientific_status") != _PROMPT_STATUS
        or not isinstance(tokenized_splits, Mapping)
    ):
        raise ValueError("rotated-span prompt provenance is invalid")
    counts = prompt_metadata.get("counts")
    normalized = prompt_metadata.get("normalized_sha256")
    per_prompt = prompt_metadata.get("per_prompt_sha256")
    if (
        not isinstance(counts, Mapping)
        or tuple(counts) != split_names
        or dict(counts)
        != {
            "calibration_a": 64,
            "calibration_b": 16,
            "validation": 16,
            "test": 16,
        }
        or not isinstance(normalized, Mapping)
        or tuple(normalized) != split_names
        or not isinstance(per_prompt, Mapping)
        or tuple(per_prompt) != split_names
    ):
        raise ValueError("rotated-span prompt split metadata is invalid")
    all_prompt_hashes: list[str] = []
    for split_name in split_names:
        hashes = per_prompt[split_name]
        if (
            not isinstance(hashes, list)
            or len(hashes) != counts[split_name]
            or any(not _is_sha256(value) for value in hashes)
            or normalized[split_name]
            != _ordered_prompt_hash_digest(hashes)
        ):
            raise ValueError("rotated-span prompt hashes are invalid")
        all_prompt_hashes.extend(hashes)
    if len(set(all_prompt_hashes)) != len(all_prompt_hashes):
        raise ValueError(
            "rotated-span prompt hashes must be globally disjoint"
        )
    expected_stream_names = (
        ("calibration_a", "calibration_b", "validation")
        if validation_evaluated_status
        else ("calibration_a", "calibration_b")
    )
    if tuple(tokenized_splits) != expected_stream_names:
        raise ValueError("rotated-span tokenized split set is invalid")
    validated_streams: dict[str, Mapping[str, object]] = {}
    for split_name in expected_stream_names:
        stream, _ = _validated_tokenized_stream(
            tokenized_splits[split_name],
            split_name=split_name,
        )
        if (
            stream["sequences"] != counts[split_name]
            or stream["source_prompt_sha256"]
            != per_prompt[split_name]
        ):
            raise ValueError(
                "rotated-span tokenized stream does not bind its prompt "
                "split"
            )
        validated_streams[split_name] = stream
    streamed_hashes: set[str] = set()
    for stream in validated_streams.values():
        hashes = stream["source_prompt_sha256"]
        assert isinstance(hashes, list)
        streamed_hashes.update(hashes)
    hash_only_splits = {"test"}
    if not validation_evaluated_status:
        hash_only_splits.add("validation")
    if streamed_hashes & {
        digest
        for split_name in hash_only_splits
        for digest in per_prompt[split_name]
    }:
        raise ValueError(
            "rotated-span hash-only prompt entered a tokenized stream"
        )
    if (
        training.get("tokenized_stream")
        != validated_streams["calibration_a"]
        or selection.get("tokenized_stream")
        != validated_streams["calibration_b"]
        or (
            validation_evaluated_status
            and validation.get("tokenized_stream")
            != validated_streams["validation"]
        )
        or (
            not validation_evaluated_status
            and validation.get("tokenized_stream") is not None
        )
    ):
        raise ValueError(
            "rotated-span duplicated stream provenance is inconsistent"
        )

    executor = RotatedSpanBlockExecutor.from_artifact_state_dict(
        executor_state
    )
    if executor.width != width or executor.retained_rank != width - 1:
        raise ValueError("rotated-span executor geometry is invalid")
    architecture = protocol.get("executor_architecture")
    model_binding = source.get("model_binding")
    block_geometry = source.get("block_geometry")
    if (
        not isinstance(architecture, Mapping)
        or dict(architecture) != asdict(executor.graph.config)
        or not isinstance(model_binding, Mapping)
        or set(model_binding)
        != {
            "model_id",
            "config_sha256",
            "resolved_commit",
            "hidden_size",
            "num_hidden_layers",
        }
        or any(
            model_binding[field] != model.get(field)
            for field in model_binding
        )
        or not isinstance(block_geometry, Mapping)
        or dict(block_geometry)
        != {
            "start_layer": start,
            "end_layer_inclusive": end,
            "layer_ids": layer_ids,
            "canonical_boundaries": boundaries,
            "residual_width": width,
        }
        or (
            model.get("hidden_size") is not None
            and model["hidden_size"] != width
        )
    ):
        raise ValueError("rotated-span model or architecture binding is invalid")
    normal_sha = _tensor_sha256(
        executor.normal,
        domain=b"fisher_graph.rotated_span.normal.v1\0",
    )
    basis_sha = _tensor_sha256(
        deterministic_orthogonal_complement(executor.normal),
        domain=b"fisher_graph.rotated_span.basis.v1\0",
    )
    if (
        normal_sha != source["normal_sha256"]
        or basis_sha != source["basis_sha256"]
    ):
        raise ValueError("rotated-span basis binding is invalid")
    basis = deterministic_orthogonal_complement(executor.normal)
    basis_gram_error = float(
        (
            basis.T @ basis
            - torch.eye(width - 1, dtype=torch.float64)
        )
        .abs()
        .max()
        .item()
    )
    basis_projector_error = float(
        (
            basis @ basis.T
            - (
                torch.eye(width, dtype=torch.float64)
                - torch.outer(executor.normal, executor.normal)
            )
        )
        .abs()
        .max()
        .item()
    )
    if (
        not math.isclose(
            float(source.get("basis_gram_max_absolute_error", math.nan)),
            basis_gram_error,
            rel_tol=1e-9,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(
                source.get(
                    "basis_projector_max_absolute_error",
                    math.nan,
                )
            ),
            basis_projector_error,
            rel_tol=1e-9,
            abs_tol=1e-15,
        )
        or basis_gram_error > 1e-10
        or basis_projector_error > 1e-10
    ):
        raise ValueError("rotated-span basis audit is invalid")

    thresholds = protocol.get("thresholds")
    oracle_thresholds = protocol.get("oracle_thresholds")
    if (
        not isinstance(thresholds, Mapping)
        or set(thresholds)
        != {
            "nll_atol",
            "top1_min",
            "teacher_kl_max",
            "p90_abs_nll_max",
            "p10_top1_min",
            "max_stored_coefficient_ratio",
            "max_analytic_mac_ratio",
        }
        or not isinstance(oracle_thresholds, Mapping)
        or set(oracle_thresholds) != {"nll_atol", "top1_min"}
    ):
        raise ValueError("rotated-span threshold policy is invalid")

    def validate_behavior(
        value: object,
        *,
        label: str,
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} behavior is invalid")
        examples = value.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"{label} behavior examples are invalid")
        recomputed = _aggregate_behavior_with_kl(examples)
        if not _semantic_numeric_equal(value, recomputed):
            raise ValueError(f"{label} behavior aggregate is invalid")
        return value

    def validate_direct(
        value: object,
        *,
        label: str,
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} direct diagnostic is invalid")
        examples = value.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"{label} direct examples are invalid")
        recomputed = _aggregate_direct_examples(examples, width=width)
        if not _semantic_numeric_equal(value, recomputed):
            raise ValueError(f"{label} direct aggregate is invalid")
        return value

    def validate_execution_audit(
        value: object,
        *,
        label: str,
    ) -> bool:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} execution audit is invalid")
        layer_calls = value.get("source_layer_calls")
        passed = (
            type(value.get("batches")) is int
            and value["batches"] > 0
            and value.get("executor_calls") == value["batches"]
            and value.get("source_block_calls_total") == 0
            and isinstance(layer_calls, Mapping)
            and tuple(layer_calls) == layer_ids
            and all(layer_calls[layer_id] == 0 for layer_id in layer_ids)
            and tuple(value.get("native_layers_skipped", ())) == layer_ids
            and value.get("maximum_prefix_boundary_replay_error") == 0.0
            and value.get("passed") is True
        )
        if not passed:
            raise ValueError(f"{label} execution audit did not recompute")
        return True

    def validate_span_audit(
        value: object,
        *,
        label: str,
    ) -> bool:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} span audit is invalid")
        removed = value.get("out_of_span_squared_energy")
        total = value.get("raw_delta_squared_energy")
        limit = value.get(
            "maximum_allowed_relative_out_of_span_energy"
        )
        if any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in (removed, total, limit)
        ):
            raise ValueError(f"{label} span energies are invalid")
        ratio = float(removed) / max(
            float(total),
            torch.finfo(torch.float64).tiny,
        )
        expected_passed = ratio <= float(limit)
        if (
            not math.isclose(
                float(value.get("relative_out_of_span_energy", math.nan)),
                ratio,
                rel_tol=1e-12,
                abs_tol=1e-18,
            )
            or value.get("passed") is not expected_passed
        ):
            raise ValueError(f"{label} span audit did not recompute")
        return expected_passed

    def validate_accounting(
        value: object,
        *,
        label: str,
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} accounting is invalid")
        learned = value.get("graph_trainable_parameter_count")
        fixed = value.get("fixed_runtime_coefficient_count")
        stored = value.get("runtime_stored_coefficient_count")
        source_parameters = value.get("source_block_parameter_count")
        graph_macs = value.get("graph_analytic_mac_count")
        decode_macs = value.get("basis_decode_analytic_mac_count")
        total_macs = value.get("total_analytic_mac_count")
        source_macs = value.get("source_block_analytic_mac_count")
        valid_positions = value.get("valid_positions")
        breakdown = value.get("graph_mac_breakdown")
        if (
            learned != executor.learned_parameter_count
            or fixed != executor.fixed_runtime_coefficient_count
            or type(stored) is not int
            or stored != learned + fixed
            or type(source_parameters) is not int
            or source_parameters <= 0
            or type(graph_macs) is not int
            or graph_macs < 0
            or type(decode_macs) is not int
            or type(total_macs) is not int
            or total_macs != graph_macs + decode_macs
            or type(source_macs) is not int
            or source_macs <= 0
            or type(valid_positions) is not int
            or valid_positions <= 0
            or decode_macs
            != valid_positions * executor.retained_rank * executor.width
            or not isinstance(breakdown, Mapping)
            or breakdown.get("total_mac_count") != graph_macs
            or not math.isclose(
                float(
                    value.get(
                        "stored_coefficient_ratio_to_source",
                        math.nan,
                    )
                ),
                stored / source_parameters,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                float(
                    value.get(
                        "analytic_mac_ratio_to_source",
                        math.nan,
                    )
                ),
                total_macs / source_macs,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(f"{label} accounting did not recompute")
        return value

    structural = training.get("structural_probes")
    if not isinstance(structural, Mapping):
        raise ValueError("rotated-span structural probes are invalid")
    expected_causal = (
        float(
            structural.get(
                "future_slot_perturbation_max_earlier_error",
                math.inf,
            )
        )
        <= float(structural.get("future_slot_tolerance", -1.0))
    )
    expected_batching = (
        float(
            structural.get(
                "padded_batch_vs_single_max_valid_error",
                math.inf,
            )
        )
        <= float(structural.get("batching_tolerance", -1.0))
    )
    if (
        structural.get("future_slot_causality_passed")
        is not expected_causal
        or structural.get("batching_equivalence_passed")
        is not expected_batching
        or structural.get("passed")
        is not (expected_causal and expected_batching)
        or structural.get("passed") is not True
        or structural.get(
            "batching_probe_scope",
            "batched_vs_single_same_padding",
        )
        not in {
            "batched_vs_single_same_padding",
            "batched_padded_vs_trimmed_single_right_padding",
        }
    ):
        raise ValueError("rotated-span structural probes did not recompute")

    executor_behavior = validate_behavior(
        selection.get("executor_behavior"),
        label="selection executor",
    )
    oracle_behavior = validate_behavior(
        selection.get("rotated_span_oracle_behavior"),
        label="selection oracle",
    )
    validate_behavior(
        selection.get("identity_block_skip_behavior"),
        label="selection identity skip",
    )
    validate_direct(
        selection.get("executor_direct_diagnostic"),
        label="selection executor",
    )
    validate_direct(
        selection.get("rotated_span_oracle_direct_diagnostic"),
        label="selection oracle",
    )
    validate_direct(
        selection.get("identity_block_skip_direct_diagnostic"),
        label="selection identity skip",
    )
    expected_behavior_gates = _behavior_gates(
        executor_behavior,
        nll_atol=float(thresholds["nll_atol"]),
        top1_min=float(thresholds["top1_min"]),
        teacher_kl_max=float(thresholds["teacher_kl_max"]),
        p90_abs_nll_max=float(thresholds["p90_abs_nll_max"]),
        p10_top1_min=float(thresholds["p10_top1_min"]),
    )
    expected_oracle_gates = _behavior_gates(
        oracle_behavior,
        nll_atol=float(oracle_thresholds["nll_atol"]),
        top1_min=float(oracle_thresholds["top1_min"]),
        teacher_kl_max=float(thresholds["teacher_kl_max"]),
        p90_abs_nll_max=float(thresholds["p90_abs_nll_max"]),
        p10_top1_min=float(thresholds["p10_top1_min"]),
    )
    selection_accounting = validate_accounting(
        selection.get("accounting"),
        label="selection",
    )
    expected_resource_gates = _resource_gates(
        selection_accounting,
        max_stored_coefficient_ratio=float(
            thresholds["max_stored_coefficient_ratio"]
        ),
        max_analytic_mac_ratio=float(
            thresholds["max_analytic_mac_ratio"]
        ),
    )
    execution_passed = validate_execution_audit(
        selection.get("execution_audit"),
        label="selection",
    )
    span_passed = validate_span_audit(
        selection.get("span_audit"),
        label="selection",
    )
    expected_selection_passed = (
        all(expected_behavior_gates.values())
        and all(expected_oracle_gates.values())
        and all(expected_resource_gates.values())
        and execution_passed
        and span_passed
    )
    if (
        selection.get("executor_behavior_gates")
        != expected_behavior_gates
        or selection.get("oracle_behavior_gates")
        != expected_oracle_gates
        or selection.get("resource_gates") != expected_resource_gates
        or selection.get("passed") is not expected_selection_passed
        or selection.get("locked") is not expected_selection_passed
    ):
        raise ValueError("rotated-span selection gates did not recompute")

    selection_passed = selection.get("passed")
    validation_evaluated = validation.get("evaluated")
    validation_passed = validation.get("passed")
    validation_accounting: Mapping[str, object] | None = None
    if validation_evaluated:
        validation_behavior = validate_behavior(
            validation.get("behavior"),
            label="validation executor",
        )
        validate_direct(
            validation.get("direct_diagnostic"),
            label="validation executor",
        )
        validation_accounting = validate_accounting(
            validation.get("accounting"),
            label="validation",
        )
        expected_validation_gates = _behavior_gates(
            validation_behavior,
            nll_atol=float(thresholds["nll_atol"]),
            top1_min=float(thresholds["top1_min"]),
            teacher_kl_max=float(thresholds["teacher_kl_max"]),
            p90_abs_nll_max=float(thresholds["p90_abs_nll_max"]),
            p10_top1_min=float(thresholds["p10_top1_min"]),
        )
        expected_validation_gates.update(
            {
                f"resource_{key}": value
                for key, value in _resource_gates(
                    validation_accounting,
                    max_stored_coefficient_ratio=float(
                        thresholds["max_stored_coefficient_ratio"]
                    ),
                    max_analytic_mac_ratio=float(
                        thresholds["max_analytic_mac_ratio"]
                    ),
                ).items()
            }
        )
        validation_execution = validate_execution_audit(
            validation.get("execution_audit"),
            label="validation",
        )
        validation_span = validate_span_audit(
            validation.get("span_audit"),
            label="validation",
        )
        expected_validation_passed = (
            all(expected_validation_gates.values())
            and validation_execution
            and validation_span
        )
        if (
            validation.get("behavior_and_resource_gates")
            != expected_validation_gates
            or validation_passed is not expected_validation_passed
        ):
            raise ValueError(
                "rotated-span validation gates did not recompute"
            )
    elif any(
        validation.get(field) is not None
        for field in (
            "behavior",
            "direct_diagnostic",
            "behavior_and_resource_gates",
            "execution_audit",
            "span_audit",
            "accounting",
            "tokenized_stream",
        )
    ) or validation_passed is not False:
        raise ValueError("unevaluated validation contains results")
    expected_parameter_reduction = (
        validation_accounting is not None
        and float(
            selection_accounting[
                "stored_coefficient_ratio_to_source"
            ]
        )
        < 1.0
        and float(
            validation_accounting[
                "stored_coefficient_ratio_to_source"
            ]
        )
        < 1.0
    )
    expected_analytic_mac_reduction = (
        validation_accounting is not None
        and float(
            selection_accounting["analytic_mac_ratio_to_source"]
        )
        < 1.0
        and float(
            validation_accounting["analytic_mac_ratio_to_source"]
        )
        < 1.0
    )
    if (
        type(selection_passed) is not bool
        or type(validation_evaluated) is not bool
        or type(validation_passed) is not bool
        or validation_evaluated is not selection_passed
        or selection_passed is not expected_selection_passed
        or validation_passed and not validation_evaluated
        or status.get("calibration_b_passed") is not selection_passed
        or status.get("validation_evaluated") is not validation_evaluated
        or status.get("validation_passed") is not validation_passed
        or status.get("test_evaluated") is not False
        or status.get("model_weights_changed") is not False
        or status.get("source_block_removed_from_student_path") is not True
        or status.get("native_source_block_calls_in_student_path") != 0
        or status.get("source_independent_executor") is not True
        or status.get("fidelity_viable_replacement")
        is not (selection_passed and validation_passed)
        or status.get("parameter_reduction_supported")
        is not expected_parameter_reduction
        or status.get("analytic_mac_reduction_supported")
        is not expected_analytic_mac_reduction
        or status.get("latency_or_kernel_speed_claim") is not False
    ):
        raise ValueError("rotated-span scientific status is invalid")
    expected_report = _build_report(
        payload,
        output=artifact_path,
        scientific_digest=digest,
    )
    report = json.loads(
        artifact_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(report, Mapping)
        or _report_sha256(report) != raw["report_sha256"]
        or report
        != json.loads(
            json.dumps(
                expected_report,
                sort_keys=True,
                allow_nan=False,
            )
        )
    ):
        raise ValueError(
            "rotated-span executor JSON report does not match payload"
        )
    return {
        "model": model,
        "executor": executor,
        "source_rotation": copy.deepcopy(dict(source)),
        "protocol": copy.deepcopy(dict(protocol)),
        "training": copy.deepcopy(dict(training)),
        "selection": copy.deepcopy(dict(selection)),
        "validation": copy.deepcopy(dict(validation)),
        "scientific_status": copy.deepcopy(dict(status)),
        "metadata": {
            "scientific_payload_sha256": digest,
            "report_sha256": raw["report_sha256"],
        },
        "report": copy.deepcopy(dict(report)),
    }


def _parse_optional_positive_lag(value: str) -> int | None:
    if value.lower() in {"none", "unbounded"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "max positive lag must be positive or 'none'"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and validate a true Gemma grouped block replacement "
            "inside a locked codimension-one rotated span."
        )
    )
    parser.add_argument(
        "--rotation-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        default=DEFAULT_PROMPT_SPLITS,
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--expert-count",
        type=int,
        default=DEFAULT_EXPERT_COUNT,
    )
    parser.add_argument(
        "--expert-rank",
        type=int,
        default=DEFAULT_EXPERT_RANK,
    )
    parser.add_argument(
        "--router-width",
        type=int,
        default=DEFAULT_ROUTER_WIDTH,
    )
    parser.add_argument(
        "--max-positive-lag",
        type=_parse_optional_positive_lag,
        default=DEFAULT_MAX_POSITIVE_LAG,
    )
    parser.add_argument(
        "--modal-warmup-steps",
        type=int,
        default=DEFAULT_MODAL_WARMUP_STEPS,
    )
    parser.add_argument(
        "--modal-warmup-learning-rate",
        type=float,
        default=DEFAULT_MODAL_WARMUP_LEARNING_RATE,
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=DEFAULT_TRAIN_STEPS,
    )
    parser.add_argument(
        "--train-positions-per-sequence",
        type=int,
        default=DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_NORM,
    )
    parser.add_argument(
        "--ridge-regularization",
        type=float,
        default=DEFAULT_RIDGE_REGULARIZATION,
    )
    parser.add_argument(
        "--ground-truth-weight",
        type=float,
        default=DEFAULT_GROUND_TRUTH_WEIGHT,
    )
    parser.add_argument(
        "--teacher-kl-weight",
        type=float,
        default=DEFAULT_TEACHER_KL_WEIGHT,
    )
    parser.add_argument(
        "--selection-nll-atol",
        type=float,
        default=DEFAULT_NLL_ATOL,
    )
    parser.add_argument(
        "--selection-top1-min",
        type=float,
        default=DEFAULT_TOP1_MIN,
    )
    parser.add_argument(
        "--selection-teacher-kl-max",
        type=float,
        default=DEFAULT_TEACHER_KL_MAX,
    )
    parser.add_argument(
        "--selection-p90-abs-nll-max",
        type=float,
        default=DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX,
    )
    parser.add_argument(
        "--selection-p10-top1-min",
        type=float,
        default=DEFAULT_PER_PROMPT_P10_TOP1_MIN,
    )
    parser.add_argument(
        "--max-stored-coefficient-ratio",
        type=float,
        default=DEFAULT_MAX_STORED_COEFFICIENT_RATIO,
    )
    parser.add_argument(
        "--max-analytic-mac-ratio",
        type=float,
        default=DEFAULT_MAX_ANALYTIC_MAC_RATIO,
    )
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_rotated_span_executor(
        rotation_artifact_path=arguments.rotation_artifact,
        model_id=arguments.model_id,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        prompt_splits_path=arguments.prompt_splits,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        expert_count=arguments.expert_count,
        expert_rank=arguments.expert_rank,
        router_width=arguments.router_width,
        max_positive_lag=arguments.max_positive_lag,
        modal_warmup_steps=arguments.modal_warmup_steps,
        modal_warmup_learning_rate=(
            arguments.modal_warmup_learning_rate
        ),
        train_steps=arguments.train_steps,
        train_positions_per_sequence=(
            arguments.train_positions_per_sequence
        ),
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gradient_clip_norm=arguments.gradient_clip_norm,
        ridge_regularization=arguments.ridge_regularization,
        ground_truth_weight=arguments.ground_truth_weight,
        teacher_kl_weight=arguments.teacher_kl_weight,
        selection_nll_atol=arguments.selection_nll_atol,
        selection_top1_min=arguments.selection_top1_min,
        selection_teacher_kl_max=arguments.selection_teacher_kl_max,
        selection_p90_abs_nll_max=(
            arguments.selection_p90_abs_nll_max
        ),
        selection_p10_top1_min=(
            arguments.selection_p10_top1_min
        ),
        max_stored_coefficient_ratio=(
            arguments.max_stored_coefficient_ratio
        ),
        max_analytic_mac_ratio=arguments.max_analytic_mac_ratio,
        seed=arguments.seed,
        device_name=arguments.device,
        dtype=arguments.dtype,
        local_files_only=arguments.local_files_only,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PROMPT_SPLITS",
    "default_gemma3_rotated_span_executor_output",
    "load_gemma3_rotated_span_executor_artifact",
    "run_gemma3_rotated_span_executor",
]
