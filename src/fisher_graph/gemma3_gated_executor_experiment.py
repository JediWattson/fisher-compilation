"""Split-safe Gemma block-output experiment for a gated modal executor.

The source transformer is always frozen.  A previously validated
weighted-Jacobian artifact supplies only the activation codecs and their
provenance.  New, source-disjoint prompt splits have exactly four roles:

* calibration A fits executor parameters with a fixed update budget;
* calibration B selects one predeclared executor configuration;
* validation evaluates that locked executor once; and
* reserved test prompts are parsed and hashed, but never tokenized.

The executor predicts a *block residual*, not the full residual stream:

```
z_in = (h_in - mean_in) @ E_in[:, :r]
delta_z = gated_executor(z_in)
h_pred = h_in + delta_z @ D_out[:, :r].T
```

Keeping ``h_in`` as an exact raw-coordinate bypass avoids assuming that the
input and output generalized-Fisher gauges are aligned.  The native block is
still executed during this reference experiment so its output can be replaced
by ``h_pred`` at the final boundary.  Consequently, reported operation counts
are analytic graph counts, not wall-clock speed claims.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, LayerBlockBoundaryPlan, ModelAdapter
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
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    make_causal_lm_calibration_batches,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_stability_experiment import (
    _CalibrationStreamProvenance,
    _library_versions,
    _ordered_prompt_hash_digest,
    _tokenizer_provenance,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .gemma3_weighted_jacobian_experiment import (
    _codec_state_sha256,
    load_gemma3_weighted_jacobian_artifact,
)
from .linear_codec import LinearActivationCodec
from .modal_ablation import _causal_lm_batch_scores, _example_ids


DEFAULT_PROMPT_SPLITS = Path("examples/gemma3_gated_executor_prompts.json")
DEFAULT_RANKS = (320, 480)
DEFAULT_EXPERT_COUNTS = (1, 2)
DEFAULT_EXPERT_RANKS = (16,)
DEFAULT_ROUTER_WIDTHS = (16,)
DEFAULT_MAX_POSITIVE_LAGS: tuple[int | None, ...] = (None,)
DEFAULT_FIT_STEPS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRADIENT_CLIP_NORM = 1.0

# A candidate is scientifically viable only when every fidelity and resource
# gate passes.  A diagnostic best-on-B fallback is still locked and evaluated
# once when no candidate passes.
DEFAULT_MAX_RETAINED_FRACTION = 0.75
DEFAULT_MAX_STORED_COEFFICIENT_RATIO = 0.75
DEFAULT_MAX_ANALYTIC_MAC_RATIO = 0.75
DEFAULT_MAX_BLOCK_DELTA_NRMSE = 0.20
DEFAULT_MIN_BLOCK_DELTA_COSINE = 0.95
DEFAULT_NLL_ATOL = 0.05
DEFAULT_TOP1_MIN = 0.95
DEFAULT_IDENTITY_NLL_ATOL = 1e-5

_ARTIFACT_SCHEMA = "fisher_graph.gemma3_gated_executor"
_ARTIFACT_FORMAT_VERSION = 1
_PAYLOAD_DOMAIN = b"fisher_graph.gemma3_gated_executor_payload.v1\0"
_REPORT_DOMAIN = b"fisher_graph.gemma3_gated_executor_report.v1\0"


def default_gemma3_gated_executor_output(
    model_id: str = DEFAULT_MODEL_ID,
    start_layer: int = 4,
    end_layer: int = 6,
) -> Path:
    """Return an ignored model/block-specific executor artifact path."""

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
        / f"layers-{start_layer}-{end_layer}-gated-executor.pt"
    )


def _scientific_payload_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, payload)
    return digest.hexdigest()


def _report_sha256(report: Mapping[str, object]) -> str:
    serialized = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_REPORT_DOMAIN)
    digest.update(serialized)
    return digest.hexdigest()


def _finite(
    value: float,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
        or (minimum is not None and float(value) < minimum)
        or (maximum is not None and float(value) > maximum)
    ):
        raise ValueError(f"{label} is outside its valid range")
    return float(value)


def _positive_ints(
    values: Iterable[int],
    *,
    label: str,
    maximum: int | None = None,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be an iterable of integers")
    result = tuple(values)
    if not result:
        raise ValueError(f"{label} cannot be empty")
    if any(
        type(value) is not int
        or value <= 0
        or (maximum is not None and value > maximum)
        for value in result
    ):
        raise ValueError(f"{label} contains an invalid value")
    return tuple(sorted(set(result)))


def _lags(values: Iterable[int | None]) -> tuple[int | None, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("max_positive_lags must be an iterable")
    requested = tuple(values)
    if not requested:
        raise ValueError("max_positive_lags cannot be empty")
    if any(
        value is not None and (type(value) is not int or value <= 0)
        for value in requested
    ):
        raise ValueError("maximum positive lags must be positive or None")
    result: list[int | None] = []
    for value in requested:
        if value not in result:
            result.append(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GatedExecutorCandidate:
    """One predeclared modal executor hyperparameter choice."""

    retained_rank: int
    expert_count: int
    expert_rank: int
    router_width: int
    max_positive_lag: int | None

    @property
    def candidate_id(self) -> str:
        lag = "all" if self.max_positive_lag is None else str(
            self.max_positive_lag
        )
        return (
            f"rank_{self.retained_rank}.experts_{self.expert_count}."
            f"expert_rank_{self.expert_rank}.router_{self.router_width}."
            f"lag_{lag}"
        )

    def config(self) -> GatedCausalModalExecutorConfig:
        return GatedCausalModalExecutorConfig(
            input_modes=self.retained_rank,
            output_modes=self.retained_rank,
            expert_count=self.expert_count,
            expert_rank=self.expert_rank,
            router_width=self.router_width,
            same_position_skip=False,
            max_positive_lag=self.max_positive_lag,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            **asdict(self),
        }


def _candidate_schedule(
    *,
    width: int,
    ranks: Iterable[int],
    expert_counts: Iterable[int],
    expert_ranks: Iterable[int],
    router_widths: Iterable[int],
    max_positive_lags: Iterable[int | None],
) -> tuple[GatedExecutorCandidate, ...]:
    resolved_ranks = _positive_ints(ranks, label="ranks", maximum=width)
    counts = _positive_ints(expert_counts, label="expert_counts")
    factor_ranks = _positive_ints(
        expert_ranks,
        label="expert_ranks",
        maximum=width,
    )
    routers = _positive_ints(router_widths, label="router_widths")
    lags = _lags(max_positive_lags)
    return tuple(
        GatedExecutorCandidate(rank, count, factor_rank, router, lag)
        for rank in resolved_ranks
        for count in counts
        for factor_rank in factor_ranks
        for router in routers
        for lag in lags
    )


@dataclass(frozen=True, slots=True)
class _BoundaryBatch:
    input_hidden: Tensor
    output_hidden: Tensor
    valid_positions: Tensor
    logical_positions: Tensor
    example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.input_hidden.ndim != 3
            or self.output_hidden.shape != self.input_hidden.shape
            or self.valid_positions.shape != self.input_hidden.shape[:2]
            or self.valid_positions.dtype is not torch.bool
            or self.logical_positions.shape != self.valid_positions.shape
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
            or len(self.example_ids) != self.input_hidden.shape[0]
        ):
            raise ValueError("boundary batch tensors are inconsistent")


@dataclass(frozen=True, slots=True)
class _RuntimeCodec:
    input_mean: Tensor
    input_encoder: Tensor
    output_encoder: Tensor
    output_decoder: Tensor
    output_least_squares_encoder: Tensor


def _runtime_codec(
    input_codec: LinearActivationCodec,
    output_codec: LinearActivationCodec,
    *,
    rank: int,
    device: torch.device,
) -> _RuntimeCodec:
    dtype = torch.float32
    output_decoder = output_codec.decoder[:, :rank].to(
        device=device,
        dtype=dtype,
    )
    return _RuntimeCodec(
        input_mean=input_codec.mean.to(device=device, dtype=dtype),
        input_encoder=input_codec.encoder[:, :rank].to(
            device=device,
            dtype=dtype,
        ),
        output_encoder=output_codec.encoder[:, :rank].to(
            device=device,
            dtype=dtype,
        ),
        output_decoder=output_decoder,
        # For a non-orthogonal generalized codec, E_r D_r.T is a coordinate
        # truncation but not the raw-MSE-optimal point in span(D_r).  The
        # least-squares encoder supplies the actual subspace ceiling.
        output_least_squares_encoder=torch.linalg.pinv(
            output_decoder.to(dtype=torch.float64)
        ).T,
    )


def _predict_boundary(
    executor: ResidualGatedCausalModalExecutor,
    codec: _RuntimeCodec,
    boundary: _BoundaryBatch,
    *,
    components: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
    source = boundary.input_hidden.to(dtype=torch.float32)
    coordinates = (
        source - codec.input_mean
    ) @ codec.input_encoder
    if components:
        result = executor.forward_components(
            coordinates,
            query_valid_mask=boundary.valid_positions,
            key_valid_mask=boundary.valid_positions,
            logical_positions=boundary.logical_positions,
            key_logical_positions=boundary.logical_positions,
        )
        delta_modal = result.output
        same_modal = result.same_position_output
        cross_modal = result.positive_lag_output
        probabilities = result.router_probabilities
        edge_mask = result.positive_lag_mask
    else:
        delta_modal = executor(
            coordinates,
            query_valid_mask=boundary.valid_positions,
            key_valid_mask=boundary.valid_positions,
            logical_positions=boundary.logical_positions,
            key_logical_positions=boundary.logical_positions,
        )
        same_modal = torch.zeros_like(delta_modal)
        cross_modal = torch.zeros_like(delta_modal)
        probabilities = None
        edge_mask = None
    prediction = source + delta_modal @ codec.output_decoder.T
    same_raw = same_modal @ codec.output_decoder.T
    cross_raw = cross_modal @ codec.output_decoder.T
    return prediction, same_raw, cross_raw, probabilities, edge_mask


def _oracle_boundary(
    codec: _RuntimeCodec,
    boundary: _BoundaryBatch,
) -> Tensor:
    source = boundary.input_hidden.to(dtype=torch.float64)
    target = boundary.output_hidden.to(dtype=torch.float64)
    delta = target - source
    return source + (
        delta @ codec.output_least_squares_encoder
    ) @ codec.output_decoder.to(dtype=torch.float64).T


def _collect_boundaries(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    plan: LayerBlockBoundaryPlan,
) -> tuple[_BoundaryBatch, ...]:
    """Capture frozen native block inputs and outputs exactly once."""

    values = []
    sequence_offset = 0
    module = adapter.module
    if module.training or any(
        parameter.requires_grad for parameter in module.parameters()
    ):
        raise ValueError("boundary collection requires a frozen eval model")
    # These activations become inputs to a trainable executor.  `no_grad`
    # produces ordinary tensors, unlike inference tensors which autograd may
    # reject when saving values for an executor parameter gradient.
    with torch.no_grad():
        for batch in batches:
            run = adapter.forward(
                batch.model_inputs,
                capture_sites=(
                    plan.activation_sites[0],
                    plan.activation_sites[-1],
                ),
            )
            ids = _example_ids(batch, sequence_offset=sequence_offset)
            sequence_offset += batch.batch_size
            values.append(
                _BoundaryBatch(
                    input_hidden=run.activations[
                        plan.activation_sites[0]
                    ].detach().clone(),
                    output_hidden=run.activations[
                        plan.activation_sites[-1]
                    ].detach().clone(),
                    valid_positions=batch.valid_positions.clone(),
                    logical_positions=run.sequence.logical_positions.clone(),
                    example_ids=ids,
                )
            )
    if not values:
        raise ValueError("boundary split cannot be empty")
    return tuple(values)


def _initialize_executor(
    candidate: GatedExecutorCandidate,
    *,
    seed: int,
    device: torch.device,
) -> ResidualGatedCausalModalExecutor:
    candidate_seed = int.from_bytes(
        hashlib.sha256(candidate.candidate_id.encode("utf-8")).digest()[:8],
        "little",
    )
    # Construct on CPU so deterministic initialization does not mutate a
    # device-specific global generator.  Moving the initialized state is
    # deterministic.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((seed + candidate_seed) % (2**63 - 1))
        executor = ResidualGatedCausalModalExecutor(
            candidate.config(),
            dtype=torch.float32,
            device="cpu",
        )
    executor.to(device=device)
    # Exact raw skip at initialization.  The same-position matrix and expert
    # output acquire gradients immediately, while router/input-expert
    # gradients begin after the expert output leaves zero.
    with torch.no_grad():
        executor.same_position_weight.zero_()
        executor.same_position_bias.zero_()
        executor.expert_output_weight.zero_()
    return executor


def _fit_executor(
    executor: ResidualGatedCausalModalExecutor,
    codec: _RuntimeCodec,
    boundaries: Sequence[_BoundaryBatch],
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> dict[str, object]:
    """Fit only executor tensors with a fixed calibration-A update schedule."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("fit_steps must be positive")
    optimizer_parameters = tuple(executor.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    executor.train()
    initial_loss: float | None = None
    last_loss = 0.0
    for step in range(steps):
        boundary = boundaries[step % len(boundaries)]
        optimizer.zero_grad(set_to_none=True)
        prediction, _, _, _, _ = _predict_boundary(
            executor,
            codec,
            boundary,
            components=False,
        )
        target = boundary.output_hidden.to(dtype=torch.float32)
        mask = boundary.valid_positions.unsqueeze(-1)
        squared = (prediction - target).square() * mask
        denominator = int(boundary.valid_positions.sum().item()) * int(
            target.shape[-1]
        )
        loss = squared.sum() / denominator
        if not torch.isfinite(loss):
            raise RuntimeError("executor fit produced a nonfinite loss")
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            optimizer_parameters,
            max_norm=gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("executor fit produced nonfinite gradients")
        optimizer.step()
        last_loss = float(loss.detach().item())
    executor.eval()
    assert initial_loss is not None
    return {
        "steps": steps,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "fixed_update_schedule": True,
        "early_stopping": False,
        "checkpoint_selection": "final_fixed_step",
        "initial_batch_residual_space_mse": initial_loss,
        "last_batch_residual_space_mse": last_loss,
    }


def _safe_cosine(dot: float, left: float, right: float) -> float:
    denominator = math.sqrt(max(left, 0.0) * max(right, 0.0))
    if denominator == 0:
        return 1.0 if left == right else 0.0
    return max(-1.0, min(1.0, dot / denominator))


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile values cannot be empty")
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(math.ceil(fraction * len(ordered))) - 1,
        ),
    )
    return ordered[index]


def _direct_example(
    *,
    example_id: str,
    valid_tokens: int,
    source: Tensor,
    target: Tensor,
    prediction: Tensor,
) -> dict[str, object]:
    error = prediction - target
    delta = target - source
    predicted_delta = prediction - source
    error_energy = float(error.square().sum().item())
    delta_energy = float(delta.square().sum().item())
    output_energy = float(target.square().sum().item())
    prediction_energy = float(prediction.square().sum().item())
    delta_prediction_energy = float(predicted_delta.square().sum().item())
    return {
        "example_id": example_id,
        "valid_tokens": valid_tokens,
        "squared_error": error_energy,
        "block_delta_energy": delta_energy,
        "full_output_energy": output_energy,
        "prediction_energy": prediction_energy,
        "predicted_block_delta_energy": delta_prediction_energy,
        "block_delta_dot": float((delta * predicted_delta).sum().item()),
        "full_output_dot": float((target * prediction).sum().item()),
        "block_delta_nrmse": math.sqrt(
            error_energy / max(delta_energy, torch.finfo(torch.float64).tiny)
        ),
        "full_output_nrmse": math.sqrt(
            error_energy / max(output_energy, torch.finfo(torch.float64).tiny)
        ),
        "block_delta_cosine": _safe_cosine(
            float((delta * predicted_delta).sum().item()),
            delta_energy,
            delta_prediction_energy,
        ),
        "full_output_cosine": _safe_cosine(
            float((target * prediction).sum().item()),
            output_energy,
            prediction_energy,
        ),
    }


def _aggregate_direct_examples(
    examples: Sequence[Mapping[str, object]],
    *,
    width: int,
) -> dict[str, object]:
    if not examples:
        raise ValueError("direct examples cannot be empty")
    positions = sum(int(row["valid_tokens"]) for row in examples)
    error = sum(float(row["squared_error"]) for row in examples)
    delta = sum(float(row["block_delta_energy"]) for row in examples)
    output = sum(float(row["full_output_energy"]) for row in examples)
    predicted_delta = sum(
        float(row["predicted_block_delta_energy"]) for row in examples
    )
    prediction = sum(float(row["prediction_energy"]) for row in examples)
    delta_dot = sum(float(row["block_delta_dot"]) for row in examples)
    output_dot = sum(float(row["full_output_dot"]) for row in examples)
    nrmse_values = [
        float(row["block_delta_nrmse"]) for row in examples
    ]
    lengths: dict[int, list[Mapping[str, object]]] = {}
    for row in examples:
        lengths.setdefault(int(row["valid_tokens"]), []).append(row)
    return {
        "sequences": len(examples),
        "valid_positions": positions,
        "width": width,
        "mse": error / (positions * width),
        "rmse": math.sqrt(error / (positions * width)),
        "block_delta_nrmse": math.sqrt(
            error / max(delta, torch.finfo(torch.float64).tiny)
        ),
        "full_output_nrmse": math.sqrt(
            error / max(output, torch.finfo(torch.float64).tiny)
        ),
        "block_delta_cosine": _safe_cosine(
            delta_dot,
            delta,
            predicted_delta,
        ),
        "full_output_cosine": _safe_cosine(
            output_dot,
            output,
            prediction,
        ),
        "per_example_block_delta_nrmse": {
            "p50": _percentile(nrmse_values, 0.50),
            "p90": _percentile(nrmse_values, 0.90),
            "worst": max(nrmse_values),
        },
        "by_valid_length": {
            str(length): {
                "sequences": len(rows),
                "block_delta_nrmse_p50": _percentile(
                    [float(row["block_delta_nrmse"]) for row in rows],
                    0.50,
                ),
                "block_delta_nrmse_p90": _percentile(
                    [float(row["block_delta_nrmse"]) for row in rows],
                    0.90,
                ),
                "block_delta_nrmse_worst": max(
                    float(row["block_delta_nrmse"]) for row in rows
                ),
            }
            for length, rows in sorted(lengths.items())
        },
        "examples": [copy.deepcopy(dict(row)) for row in examples],
    }


def _evaluate_direct(
    executor: ResidualGatedCausalModalExecutor | None,
    codec: _RuntimeCodec,
    boundaries: Sequence[_BoundaryBatch],
    *,
    oracle: bool = False,
) -> dict[str, object]:
    examples = []
    same_energy = 0.0
    cross_energy = 0.0
    router_entropy_sum = 0.0
    router_edges = 0
    router_max_sum = 0.0
    for boundary in boundaries:
        if oracle:
            prediction = _oracle_boundary(codec, boundary)
            same = torch.zeros_like(prediction)
            cross = torch.zeros_like(prediction)
            probabilities = None
            edge_mask = None
        else:
            assert executor is not None
            prediction, same, cross, probabilities, edge_mask = (
                _predict_boundary(
                    executor,
                    codec,
                    boundary,
                    components=True,
                )
            )
        for index, example_id in enumerate(boundary.example_ids):
            valid = boundary.valid_positions[index]
            source = boundary.input_hidden[index, valid].to(torch.float64)
            target = boundary.output_hidden[index, valid].to(torch.float64)
            selected_prediction = prediction[index, valid].to(torch.float64)
            examples.append(
                _direct_example(
                    example_id=example_id,
                    valid_tokens=int(valid.sum().item()),
                    source=source,
                    target=target,
                    prediction=selected_prediction,
                )
            )
            same_energy += float(
                same[index, valid].to(torch.float64).square().sum().item()
            )
            cross_energy += float(
                cross[index, valid].to(torch.float64).square().sum().item()
            )
        if probabilities is not None and edge_mask is not None:
            selected = probabilities[edge_mask]
            if selected.numel():
                safe = selected.clamp_min(torch.finfo(selected.dtype).tiny)
                router_entropy_sum += float(
                    (-(safe * safe.log()).sum(dim=-1)).sum().item()
                )
                router_max_sum += float(selected.max(dim=-1).values.sum().item())
                router_edges += int(selected.shape[0])
    result = _aggregate_direct_examples(
        examples,
        width=boundaries[0].input_hidden.shape[-1],
    )
    result["path_energy"] = {
        "same_position": same_energy,
        "positive_lag": cross_energy,
        "positive_lag_fraction_of_path_energy": (
            cross_energy / max(same_energy + cross_energy, 1e-300)
        ),
    }
    result["router"] = {
        "positive_lag_edges": router_edges,
        "mean_entropy": (
            0.0 if router_edges == 0 else router_entropy_sum / router_edges
        ),
        "mean_max_probability": (
            0.0 if router_edges == 0 else router_max_sum / router_edges
        ),
        "collapsed_fraction_not_materialized": True,
    }
    return result


def _evaluate_full_width_delta_roundtrip(
    output_codec: LinearActivationCodec,
    boundaries: Sequence[_BoundaryBatch],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    """Measure the locked-family full-width delta reconstruction control."""

    encoder = output_codec.encoder.to(device=device, dtype=dtype)
    decoder = output_codec.decoder.to(device=device, dtype=dtype)
    examples = []
    for boundary in boundaries:
        for index, example_id in enumerate(boundary.example_ids):
            valid = boundary.valid_positions[index]
            source = boundary.input_hidden[index, valid].to(dtype)
            target = boundary.output_hidden[index, valid].to(dtype)
            delta = target - source
            prediction = source + (delta @ encoder) @ decoder.T
            examples.append(
                _direct_example(
                    example_id=example_id,
                    valid_tokens=int(valid.sum().item()),
                    source=source,
                    target=target,
                    prediction=prediction,
                )
            )
    result = _aggregate_direct_examples(
        examples,
        width=output_codec.width,
    )
    result["control"] = "locked_family_full_width_delta_roundtrip"
    result["compute_dtype"] = str(dtype).removeprefix("torch.")
    return result


def _behavior_aggregate(
    examples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not examples:
        raise ValueError("behavior examples cannot be empty")
    tokens = sum(int(row["supervised_tokens"]) for row in examples)
    baseline_sum = sum(
        float(row["baseline_summed_nll"]) for row in examples
    )
    predicted_sum = sum(
        float(row["predicted_summed_nll"]) for row in examples
    )
    matches = sum(int(row["top1_matches"]) for row in examples)
    delta_values = [
        float(row["delta_nll_per_token"]) for row in examples
    ]
    top1_values = [
        float(row["top1_agreement"]) for row in examples
    ]
    lengths: dict[int, list[Mapping[str, object]]] = {}
    for row in examples:
        lengths.setdefault(int(row["supervised_tokens"]), []).append(row)
    return {
        "sequences": len(examples),
        "supervised_tokens": tokens,
        "baseline_summed_nll": baseline_sum,
        "predicted_summed_nll": predicted_sum,
        "baseline_nll_per_token": baseline_sum / tokens,
        "predicted_nll_per_token": predicted_sum / tokens,
        "delta_nll_per_token": (predicted_sum - baseline_sum) / tokens,
        "top1_matches": matches,
        "top1_agreement_to_baseline": matches / tokens,
        "per_example_delta_nll_per_token": {
            "p50": _percentile(delta_values, 0.50),
            "p90_absolute": _percentile(
                [abs(value) for value in delta_values],
                0.90,
            ),
            "worst_absolute": max(abs(value) for value in delta_values),
        },
        "per_example_top1_agreement": {
            "p50": _percentile(top1_values, 0.50),
            "p10": -_percentile([-value for value in top1_values], 0.90),
            "worst": min(top1_values),
        },
        "by_supervised_length": {
            str(length): {
                "sequences": len(rows),
                "delta_nll_per_token_p50": _percentile(
                    [
                        float(row["delta_nll_per_token"])
                        for row in rows
                    ],
                    0.50,
                ),
                "absolute_delta_nll_per_token_p90": _percentile(
                    [
                        abs(float(row["delta_nll_per_token"]))
                        for row in rows
                    ],
                    0.90,
                ),
                "top1_agreement_worst": min(
                    float(row["top1_agreement"]) for row in rows
                ),
            }
            for length, rows in sorted(lengths.items())
        },
        "examples": [copy.deepcopy(dict(row)) for row in examples],
    }


def _behavior_examples(
    *,
    batch: CalibrationBatch,
    example_ids: Sequence[str],
    baseline: object,
    predicted: object,
) -> list[dict[str, object]]:
    # `_causal_lm_batch_scores` returns an internal dataclass whose stable
    # tensor attributes are intentionally reused here.
    if (
        not torch.equal(
            baseline.supervised_tokens,  # type: ignore[attr-defined]
            predicted.supervised_tokens,  # type: ignore[attr-defined]
        )
        or not torch.equal(
            baseline.supervised_mask,  # type: ignore[attr-defined]
            predicted.supervised_mask,  # type: ignore[attr-defined]
        )
    ):
        raise RuntimeError("executor changed supervised-token accounting")
    matches = (
        (
            baseline.predictions  # type: ignore[attr-defined]
            == predicted.predictions  # type: ignore[attr-defined]
        )
        & baseline.supervised_mask  # type: ignore[attr-defined]
    ).sum(dim=1)
    values = []
    for index, example_id in enumerate(example_ids):
        tokens = int(
            baseline.supervised_tokens[index].item()  # type: ignore[attr-defined]
        )
        baseline_sum = float(
            baseline.summed_nll[index].item()  # type: ignore[attr-defined]
        )
        predicted_sum = float(
            predicted.summed_nll[index].item()  # type: ignore[attr-defined]
        )
        matched = int(matches[index].item())
        values.append(
            {
                "example_id": example_id,
                "supervised_tokens": tokens,
                "baseline_summed_nll": baseline_sum,
                "predicted_summed_nll": predicted_sum,
                "delta_summed_nll": predicted_sum - baseline_sum,
                "delta_nll_per_token": (
                    predicted_sum - baseline_sum
                )
                / tokens,
                "top1_matches": matched,
                "top1_agreement": matched / tokens,
            }
        )
    return values


def _evaluate_behavior(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    plan: LayerBlockBoundaryPlan,
    candidates: Mapping[
        str,
        tuple[
            ResidualGatedCausalModalExecutor,
            _RuntimeCodec,
        ],
    ],
    include_identity: bool,
) -> tuple[dict[str, dict[str, object]], dict[str, object] | None]:
    """Evaluate replacement behavior with a plain native baseline."""

    objective = CausalLanguageModelNLL()
    per_candidate: dict[str, list[dict[str, object]]] = {
        name: [] for name in candidates
    }
    identity_examples: list[dict[str, object]] = []
    sequence_offset = 0
    module = adapter.module
    if module.training or any(
        parameter.requires_grad for parameter in module.parameters()
    ):
        raise ValueError("behavior evaluation requires a frozen eval model")

    with torch.inference_mode():
        for batch in batches:
            ids = _example_ids(batch, sequence_offset=sequence_offset)
            sequence_offset += batch.batch_size
            native = adapter.forward(batch.model_inputs)
            baseline = _causal_lm_batch_scores(
                native.logits,
                batch,
                objective=objective,
            )
            if include_identity:
                identity = adapter.forward(
                    batch.model_inputs,
                    interventions={
                        plan.activation_sites[-1]: lambda value: value,
                    },
                )
                identity_scores = _causal_lm_batch_scores(
                    identity.logits,
                    batch,
                    objective=objective,
                )
                identity_examples.extend(
                    _behavior_examples(
                        batch=batch,
                        example_ids=ids,
                        baseline=baseline,
                        predicted=identity_scores,
                    )
                )
            sequence = adapter.prepare_sequence(batch.model_inputs)
            for name, (executor, codec) in candidates.items():
                captured: dict[str, Tensor] = {}

                def capture_input(value: Tensor) -> Tensor:
                    captured["input"] = value
                    return value

                def replace_output(value: Tensor) -> Tensor:
                    source = captured.get("input")
                    if source is None:
                        raise RuntimeError(
                            "block output ran before input capture"
                        )
                    boundary = _BoundaryBatch(
                        input_hidden=source,
                        output_hidden=value,
                        valid_positions=batch.valid_positions,
                        logical_positions=sequence.logical_positions,
                        example_ids=tuple(ids),
                    )
                    prediction, _, _, _, _ = _predict_boundary(
                        executor,
                        codec,
                        boundary,
                        components=False,
                    )
                    prediction = prediction.to(dtype=value.dtype)
                    return torch.where(
                        batch.valid_positions.unsqueeze(-1),
                        prediction,
                        value,
                    )

                replaced = adapter.forward(
                    batch.model_inputs,
                    interventions={
                        plan.activation_sites[0]: capture_input,
                        plan.activation_sites[-1]: replace_output,
                    },
                )
                replaced_scores = _causal_lm_batch_scores(
                    replaced.logits,
                    batch,
                    objective=objective,
                )
                per_candidate[name].extend(
                    _behavior_examples(
                        batch=batch,
                        example_ids=ids,
                        baseline=baseline,
                        predicted=replaced_scores,
                    )
                )
    results = {
        name: _behavior_aggregate(examples)
        for name, examples in per_candidate.items()
    }
    identity = (
        None
        if not include_identity
        else _behavior_aggregate(identity_examples)
    )
    return results, identity


def _evaluate_projection_behavior(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    plan: LayerBlockBoundaryPlan,
    rank_codecs: Mapping[str, _RuntimeCodec],
    full_output_codec: LinearActivationCodec,
) -> dict[str, dict[str, object]]:
    """Evaluate least-squares rank ceilings and the full-width control."""

    objective = CausalLanguageModelNLL()
    names = (*rank_codecs, "full_width_codec_delta_roundtrip")
    per_condition: dict[str, list[dict[str, object]]] = {
        name: [] for name in names
    }
    # End-to-end full-width control deliberately uses the declared float32
    # executor-runtime precision.  A separate direct float64 result records
    # the mathematical dual identity.
    full_encoder = full_output_codec.encoder.to(
        device=next(iter(rank_codecs.values())).output_decoder.device,
        dtype=torch.float32,
    )
    full_decoder = full_output_codec.decoder.to(
        device=full_encoder.device,
        dtype=torch.float32,
    )
    sequence_offset = 0
    with torch.inference_mode():
        for batch in batches:
            ids = _example_ids(batch, sequence_offset=sequence_offset)
            sequence_offset += batch.batch_size
            native = adapter.forward(batch.model_inputs)
            baseline = _causal_lm_batch_scores(
                native.logits,
                batch,
                objective=objective,
            )
            for name in names:
                captured: dict[str, Tensor] = {}

                def capture_input(value: Tensor) -> Tensor:
                    captured["input"] = value
                    return value

                def replace_output(
                    value: Tensor,
                    *,
                    condition_name: str = name,
                ) -> Tensor:
                    source = captured["input"].to(torch.float32)
                    delta = value.to(torch.float32) - source
                    if condition_name == "full_width_codec_delta_roundtrip":
                        prediction = (
                            source + (delta @ full_encoder) @ full_decoder.T
                        )
                    else:
                        codec = rank_codecs[condition_name]
                        source64 = source.to(torch.float64)
                        delta64 = delta.to(torch.float64)
                        prediction = source64 + (
                            delta64 @ codec.output_least_squares_encoder
                        ) @ codec.output_decoder.to(torch.float64).T
                    return torch.where(
                        batch.valid_positions.unsqueeze(-1),
                        prediction.to(dtype=value.dtype),
                        value,
                    )

                projected = adapter.forward(
                    batch.model_inputs,
                    interventions={
                        plan.activation_sites[0]: capture_input,
                        plan.activation_sites[-1]: replace_output,
                    },
                )
                scores = _causal_lm_batch_scores(
                    projected.logits,
                    batch,
                    objective=objective,
                )
                per_condition[name].extend(
                    _behavior_examples(
                        batch=batch,
                        example_ids=ids,
                        baseline=baseline,
                        predicted=scores,
                    )
                )
    return {
        name: _behavior_aggregate(examples)
        for name, examples in per_condition.items()
    }


def _source_block_static(
    adapter: ModelAdapter,
    plan: LayerBlockBoundaryPlan,
) -> dict[str, object]:
    parameters: dict[int, Tensor] = {}
    linear_weight_counts: dict[str, int] = {}
    for layer_id in plan.layer_ids:
        module = adapter.source_module(layer_id)
        for parameter in module.parameters():
            parameters.setdefault(id(parameter), parameter)
        seen_linear: set[int] = set()
        linear_count = 0
        for child in module.modules():
            if isinstance(child, nn.Linear) and id(child) not in seen_linear:
                seen_linear.add(id(child))
                linear_count += child.weight.numel()
        linear_weight_counts[layer_id] = linear_count
    return {
        "parameter_count": sum(
            parameter.numel() for parameter in parameters.values()
        ),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters.values()
        ),
        "linear_weight_coefficients_by_layer": linear_weight_counts,
        "scope": "exact_selected_source_module_parameters",
    }


def _source_block_macs(
    adapter: ModelAdapter,
    plan: LayerBlockBoundaryPlan,
    boundaries: Sequence[_BoundaryBatch],
    *,
    static: Mapping[str, object],
) -> dict[str, object]:
    linear_by_layer = static["linear_weight_coefficients_by_layer"]
    assert isinstance(linear_by_layer, Mapping)
    linear_macs = 0
    attention_macs = 0
    attention_edges = 0
    valid_positions = 0
    per_layer = {}
    for layer_id in plan.layer_ids:
        layer = adapter.layer(layer_id)
        layer_linear = 0
        layer_attention = 0
        layer_edges = 0
        for boundary in boundaries:
            mask = boundary.valid_positions
            positions = boundary.logical_positions
            tokens = int(mask.sum().item())
            valid_positions += (
                tokens if layer_id == plan.layer_ids[0] else 0
            )
            layer_linear += tokens * int(linear_by_layer[layer_id])
            if layer.attention is not None:
                lag = positions.unsqueeze(2) - positions.unsqueeze(1)
                allowed = (
                    mask.unsqueeze(2)
                    & mask.unsqueeze(1)
                    & (lag >= 0)
                )
                if layer.attention.window_size is not None:
                    allowed = allowed & (
                        lag < layer.attention.window_size
                    )
                edges = int(allowed.sum().item())
                layer_edges += edges
                layer_attention += (
                    edges
                    * 2
                    * layer.attention.query_heads
                    * layer.attention.head_dimension
                )
        linear_macs += layer_linear
        attention_macs += layer_attention
        attention_edges += layer_edges
        per_layer[layer_id] = {
            "linear_projection_macs": layer_linear,
            "qk_and_av_attention_macs": layer_attention,
            "causal_attention_edges": layer_edges,
            "total_macs": layer_linear + layer_attention,
        }
    return {
        "valid_positions": valid_positions,
        "linear_projection_macs": linear_macs,
        "qk_and_av_attention_macs": attention_macs,
        "causal_attention_edges": attention_edges,
        "total_macs": linear_macs + attention_macs,
        "by_layer": per_layer,
        "semantics": (
            "linear_weight_MACs_plus_QK_and_AV_dot_products_on_same_valid_"
            "lengths"
        ),
        "excluded": (
            "normalization_elementwise_bias_activation_softmax_rope_"
            "masking_additions_and_memory_traffic"
        ),
    }


def _executor_accounting(
    executor: ResidualGatedCausalModalExecutor,
    *,
    width: int,
    rank: int,
    boundaries: Sequence[_BoundaryBatch],
    source_static: Mapping[str, object],
    source_macs: Mapping[str, object],
) -> dict[str, object]:
    graph_macs = 0
    positive_edges = 0
    core_breakdown: dict[str, int] = {}
    valid_positions = 0
    for boundary in boundaries:
        accounting = executor.execution_accounting(
            boundary.valid_positions.shape[1],
            batch_size=boundary.valid_positions.shape[0],
            query_valid_mask=boundary.valid_positions,
            key_valid_mask=boundary.valid_positions,
            logical_positions=boundary.logical_positions,
            key_logical_positions=boundary.logical_positions,
        )
        row = asdict(accounting)
        graph_macs += int(row["total_mac_count"])
        positive_edges += int(row["positive_lag_edges"])
        valid_positions += int(row["valid_query_tokens"])
        for key, value in row.items():
            if key.endswith("_mac_count"):
                core_breakdown[key] = core_breakdown.get(key, 0) + int(value)
    codec_coefficients = width + 2 * width * rank
    codec_macs = valid_positions * 2 * width * rank
    graph_parameters = executor.learned_parameter_count
    runtime_coefficients = codec_coefficients + graph_parameters
    total_macs = codec_macs + graph_macs
    source_parameters = int(source_static["parameter_count"])
    source_total_macs = int(source_macs["total_macs"])
    return {
        "retained_rank": rank,
        "retained_fraction": rank / width,
        "graph_trainable_parameter_count": graph_parameters,
        "fixed_runtime_codec_coefficient_count": codec_coefficients,
        "runtime_stored_coefficient_count": runtime_coefficients,
        "runtime_stored_bytes_float32": runtime_coefficients * 4,
        "source_block_parameter_count": source_parameters,
        "graph_parameter_ratio_to_source": (
            graph_parameters / source_parameters
        ),
        "stored_coefficient_ratio_to_source": (
            runtime_coefficients / source_parameters
        ),
        "valid_positions": valid_positions,
        "positive_lag_edges": positive_edges,
        "codec_encode_decode_mac_count": codec_macs,
        "graph_mac_count": graph_macs,
        "total_analytic_mac_count": total_macs,
        "source_block_analytic_mac_count": source_total_macs,
        "analytic_mac_ratio_to_source": (
            total_macs / source_total_macs
        ),
        "graph_mac_breakdown": core_breakdown,
        "codec_runtime_state": (
            "input_mean_input_encoder_slice_output_decoder_slice"
        ),
        "raw_residual_skip_parameter_count": 0,
        "raw_residual_skip_mac_count": 0,
        "soft_mixture_counts_all_experts": True,
        "reference_kernel_speed_claim": False,
        "source_comparison": copy.deepcopy(dict(source_macs)),
    }


def _gate_ledger(
    *,
    candidate: GatedExecutorCandidate,
    direct: Mapping[str, object],
    behavior: Mapping[str, object],
    accounting: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    gates = {
        "retained_fraction": (
            float(accounting["retained_fraction"])
            <= thresholds["max_retained_fraction"]
        ),
        "stored_coefficient_ratio": (
            float(accounting["stored_coefficient_ratio_to_source"])
            <= thresholds["max_stored_coefficient_ratio"]
        ),
        "analytic_mac_ratio": (
            float(accounting["analytic_mac_ratio_to_source"])
            <= thresholds["max_analytic_mac_ratio"]
        ),
        "block_delta_nrmse": (
            float(direct["block_delta_nrmse"])
            <= thresholds["max_block_delta_nrmse"]
        ),
        "block_delta_cosine": (
            float(direct["block_delta_cosine"])
            >= thresholds["min_block_delta_cosine"]
        ),
        "absolute_delta_nll": (
            abs(float(behavior["delta_nll_per_token"]))
            <= thresholds["nll_atol"]
        ),
        "top1_agreement": (
            float(behavior["top1_agreement_to_baseline"])
            >= thresholds["top1_min"]
        ),
    }
    return {
        "candidate": candidate.metadata(),
        "direct": copy.deepcopy(dict(direct)),
        "behavior": copy.deepcopy(dict(behavior)),
        "accounting": copy.deepcopy(dict(accounting)),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _lock_candidate(
    ledger: Sequence[Mapping[str, object]],
) -> tuple[str, dict[str, object]]:
    if not ledger:
        raise ValueError("selection ledger cannot be empty")
    passing = [row for row in ledger if row["passed"] is True]
    if passing:
        selected = min(
            passing,
            key=lambda row: (
                int(
                    row["accounting"]["runtime_stored_coefficient_count"]  # type: ignore[index]
                ),
                int(
                    row["accounting"]["total_analytic_mac_count"]  # type: ignore[index]
                ),
                str(row["candidate"]["candidate_id"]),  # type: ignore[index]
            ),
        )
        reason = "smallest_resource_candidate_passing_all_selection_gates"
        selection_viable = True
    else:
        selected = min(
            ledger,
            key=lambda row: (
                float(row["direct"]["block_delta_nrmse"]),  # type: ignore[index]
                abs(
                    float(
                        row["behavior"]["delta_nll_per_token"]  # type: ignore[index]
                    )
                ),
                -float(
                    row["behavior"]["top1_agreement_to_baseline"]  # type: ignore[index]
                ),
                str(row["candidate"]["candidate_id"]),  # type: ignore[index]
            ),
        )
        reason = "best_calibration_b_diagnostic_fallback_nonviable"
        selection_viable = False
    candidate_id = str(selected["candidate"]["candidate_id"])  # type: ignore[index]
    return candidate_id, {
        "ordering": (
            "passing_min_stored_coefficients_then_macs_then_id_else_"
            "delta_nrmse_abs_nll_top1_id"
        ),
        "locked_candidate_id": candidate_id,
        "reason": reason,
        "selection_viable": selection_viable,
        "calibration_b_only": True,
        "ledger": [copy.deepcopy(dict(row)) for row in ledger],
    }


def _assert_source_prompt_disjoint(
    source_protocol: Mapping[str, object],
    prompt_metadata: Mapping[str, object],
) -> dict[str, object]:
    source = source_protocol.get("prompt_splits")
    current = prompt_metadata.get("per_prompt_sha256")
    if not isinstance(source, Mapping) or not isinstance(current, Mapping):
        raise ValueError("prompt provenance is missing")
    source_hashes_raw = source.get("per_prompt_sha256")
    if not isinstance(source_hashes_raw, Mapping):
        raise ValueError("source prompt provenance is invalid")
    source_hashes = {
        item
        for values in source_hashes_raw.values()
        if isinstance(values, (list, tuple))
        for item in values
    }
    current_hashes = {
        item
        for values in current.values()
        if isinstance(values, (list, tuple))
        for item in values
    }
    if any(not _is_sha256(item) for item in source_hashes | current_hashes):
        raise ValueError("prompt provenance contains an invalid hash")
    overlap = source_hashes & current_hashes
    if overlap:
        raise ValueError(
            "gated-executor prompts overlap the source analysis artifact"
        )
    return {
        "source_prompt_hashes": len(source_hashes),
        "new_prompt_hashes": len(current_hashes),
        "overlap_count": 0,
        "verified_before_model_load_or_tokenization": True,
    }


def _identity_passed(
    behavior: Mapping[str, object],
    *,
    nll_atol: float,
) -> bool:
    return (
        abs(float(behavior["delta_nll_per_token"])) <= nll_atol
        and float(behavior["top1_agreement_to_baseline"]) == 1.0
    )


def _roundtrip_passed(
    direct: Mapping[str, object],
    behavior: Mapping[str, object],
    *,
    nll_atol: float,
) -> bool:
    return (
        float(direct["block_delta_nrmse"]) <= 1e-5
        and float(direct["block_delta_cosine"]) >= 1.0 - 1e-7
        and _identity_passed(behavior, nll_atol=nll_atol)
    )


def _materialize_split(
    tokenizer: object,
    prompts: Sequence[str],
    *,
    split_name: str,
    max_length: int,
    tokenization_batch_size: int,
    device: torch.device,
) -> tuple[tuple[CalibrationBatch, ...], dict[str, object]]:
    provenance = _CalibrationStreamProvenance(split_name, prompts)
    batches = tuple(
        provenance.wrap(
            make_causal_lm_calibration_batches(
                tokenizer,
                prompts,
                max_length=max_length,
                tokenization_batch_size=tokenization_batch_size,
                device=device,
            )
        )
    )
    return batches, provenance.metadata()


def _build_report(
    payload: Mapping[str, object],
    *,
    output: Path,
    scientific_digest: str,
) -> dict[str, object]:
    fit = payload["fit"]
    selection = payload["selection"]
    validation = payload["validation"]
    assert isinstance(fit, Mapping)
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
        "source_analysis": copy.deepcopy(
            dict(payload["source_analysis"])  # type: ignore[arg-type]
        ),
        "protocol": copy.deepcopy(
            dict(payload["protocol"])  # type: ignore[arg-type]
        ),
        "analysis": {
            "fit": {
                "candidate_training": copy.deepcopy(
                    fit["candidate_training"]
                ),
                "candidate_direct": copy.deepcopy(
                    fit["candidate_direct"]
                ),
                "rank_mse_lower_bound_direct": copy.deepcopy(
                    fit["rank_mse_lower_bound_direct"]
                ),
                "tokenized_stream": copy.deepcopy(
                    fit["tokenized_stream"]
                ),
            },
            "selection": copy.deepcopy(dict(selection)),
            "validation": copy.deepcopy(dict(validation)),
        },
        "artifact": {
            "tensor_output": output.name,
            "contains_model_state_dict": False,
            "contains_executor_state_dicts": True,
            "contains_codec_state": True,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "scientific_payload_sha256": scientific_digest,
        },
    }


def run_gemma3_gated_executor(
    *,
    weighted_artifact_path: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    prompt_splits_path: Path | str = DEFAULT_PROMPT_SPLITS,
    max_length: int = 128,
    tokenization_batch_size: int = 4,
    ranks: Iterable[int] = DEFAULT_RANKS,
    expert_counts: Iterable[int] = DEFAULT_EXPERT_COUNTS,
    expert_ranks: Iterable[int] = DEFAULT_EXPERT_RANKS,
    router_widths: Iterable[int] = DEFAULT_ROUTER_WIDTHS,
    max_positive_lags: Iterable[int | None] = DEFAULT_MAX_POSITIVE_LAGS,
    fit_steps: int = DEFAULT_FIT_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM,
    seed: int = 2029,
    max_retained_fraction: float = DEFAULT_MAX_RETAINED_FRACTION,
    max_stored_coefficient_ratio: float = (
        DEFAULT_MAX_STORED_COEFFICIENT_RATIO
    ),
    max_analytic_mac_ratio: float = DEFAULT_MAX_ANALYTIC_MAC_RATIO,
    max_block_delta_nrmse: float = DEFAULT_MAX_BLOCK_DELTA_NRMSE,
    min_block_delta_cosine: float = DEFAULT_MIN_BLOCK_DELTA_COSINE,
    selection_nll_atol: float = DEFAULT_NLL_ATOL,
    selection_top1_min: float = DEFAULT_TOP1_MIN,
    identity_nll_atol: float = DEFAULT_IDENTITY_NLL_ATOL,
    device_name: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Fit on A, lock on B, and evaluate one executor on validation."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    if type(fit_steps) is not int or fit_steps <= 0:
        raise ValueError("fit_steps must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")
    learning_rate = _finite(
        learning_rate,
        label="learning_rate",
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
    thresholds = {
        "max_retained_fraction": _finite(
            max_retained_fraction,
            label="max_retained_fraction",
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
        "max_block_delta_nrmse": _finite(
            max_block_delta_nrmse,
            label="max_block_delta_nrmse",
            minimum=0.0,
        ),
        "min_block_delta_cosine": _finite(
            min_block_delta_cosine,
            label="min_block_delta_cosine",
            minimum=-1.0,
            maximum=1.0,
        ),
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
    }
    identity_tolerance = _finite(
        identity_nll_atol,
        label="identity_nll_atol",
        minimum=0.0,
    )

    # Strictly validate the source analysis before loading a model.  It binds
    # the exact codec family, layer range, model revision, and old prompts.
    source = load_gemma3_weighted_jacobian_artifact(
        weighted_artifact_path
    )
    source_metadata = source["metadata"]
    source_selection = source["selection"]
    source_codecs = source["codecs"]
    assert isinstance(source_metadata, Mapping)
    assert isinstance(source_selection, Mapping)
    assert isinstance(source_codecs, Mapping)
    source_model = source_metadata["model"]
    source_protocol = source_metadata["protocol"]
    assert isinstance(source_model, Mapping)
    assert isinstance(source_protocol, Mapping)
    locked_source = source_selection["locked_candidate"]
    if not isinstance(locked_source, Mapping):
        raise ValueError("source artifact lacks a locked codec")
    if source_model.get("model_id") != model_id:
        raise ValueError(
            "requested model_id does not match the source artifact"
        )
    if revision is not None and revision not in {
        source_model.get("requested_revision"),
        source_model.get("resolved_commit"),
    }:
        raise ValueError(
            "explicit revision does not match the source artifact"
        )
    variant_id = locked_source["variant_id"]
    if not isinstance(variant_id, str) or variant_id not in source_codecs:
        raise ValueError("source locked codec variant is invalid")

    prompt_splits = load_gemma3_prompt_splits(prompt_splits_path)
    prompt_metadata = prompt_splits.metadata()
    disjointness = _assert_source_prompt_disjoint(
        source_protocol,
        prompt_metadata,
    )
    width = source_protocol["residual_width"]
    start_layer = source_protocol["start_layer"]
    end_layer = source_protocol["end_layer_inclusive"]
    boundaries = source_protocol["canonical_boundaries"]
    if (
        type(width) is not int
        or width <= 0
        or type(start_layer) is not int
        or type(end_layer) is not int
        or not isinstance(boundaries, tuple)
        or len(boundaries) < 2
    ):
        raise ValueError("source block geometry is invalid")
    schedule = _candidate_schedule(
        width=width,
        ranks=ranks,
        expert_counts=expert_counts,
        expert_ranks=expert_ranks,
        router_widths=router_widths,
        max_positive_lags=max_positive_lags,
    )
    scheduled_ranks = {candidate.retained_rank for candidate in schedule}
    if width == 640 and not {320, 480}.issubset(scheduled_ranks):
        raise ValueError("width-640 schedule must include ranks 320 and 480")
    resolved_output = (
        default_gemma3_gated_executor_output(
            model_id,
            start_layer,
            end_layer,
        )
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    requested_revision = (
        revision
        if revision is not None
        else (
            source_model.get("resolved_commit")
            or source_model.get("requested_revision")
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
        plan.activation_sites != boundaries
        or plan.widths != (width,) * len(boundaries)
    ):
        raise ValueError("live adapter block does not match source artifact")
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=requested_revision,
    )
    for field in ("model_id", "config_sha256", "hidden_size"):
        if source_model.get(field) != model_metadata.get(field):
            raise ValueError(f"live model {field} does not match source")
    if (
        source_model.get("resolved_commit") is not None
        and model_metadata.get("resolved_commit") is not None
        and source_model["resolved_commit"]
        != model_metadata["resolved_commit"]
    ):
        raise ValueError("live model commit does not match source artifact")

    by_site = source_codecs[variant_id]
    if not isinstance(by_site, Mapping):
        raise ValueError("source codec site mapping is invalid")
    input_codec = by_site[boundaries[0]]
    output_codec = by_site[boundaries[-1]]
    if not isinstance(input_codec, LinearActivationCodec) or not isinstance(
        output_codec,
        LinearActivationCodec,
    ):
        raise ValueError("source boundary codecs are invalid")
    runtime_codecs = {
        rank: _runtime_codec(
            input_codec,
            output_codec,
            rank=rank,
            device=device,
        )
        for rank in sorted(scheduled_ranks)
    }
    runtime_codec_audit = {
        f"rank_{rank}": {
            "input_encoder_frobenius_norm": float(
                torch.linalg.vector_norm(
                    codec.input_encoder.to(torch.float64)
                ).item()
            ),
            "input_encoder_max_absolute": float(
                codec.input_encoder.abs().max().item()
            ),
            "output_decoder_frobenius_norm": float(
                torch.linalg.vector_norm(
                    codec.output_decoder.to(torch.float64)
                ).item()
            ),
            "output_decoder_max_absolute": float(
                codec.output_decoder.abs().max().item()
            ),
            "output_least_squares_encoder_frobenius_norm": float(
                torch.linalg.vector_norm(
                    codec.output_least_squares_encoder
                ).item()
            ),
        }
        for rank, codec in runtime_codecs.items()
    }
    source_static = _source_block_static(adapter, plan)
    source_parameter_ids = {
        id(parameter) for parameter in model.parameters()
    }

    # Calibration A is the only executor-parameter fit split.
    fit_batches, fit_stream = _materialize_split(
        tokenizer,
        prompt_splits.calibration_a,
        split_name="calibration_a",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    fit_boundaries = _collect_boundaries(adapter, fit_batches, plan=plan)
    executors: dict[str, ResidualGatedCausalModalExecutor] = {}
    training = {}
    fit_direct = {}
    for candidate in schedule:
        executor = _initialize_executor(
            candidate,
            seed=seed,
            device=device,
        )
        if source_parameter_ids & {
            id(parameter) for parameter in executor.parameters()
        }:
            raise RuntimeError("executor aliases a source-model parameter")
        training[candidate.candidate_id] = _fit_executor(
            executor,
            runtime_codecs[candidate.retained_rank],
            fit_boundaries,
            steps=fit_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
        )
        fit_direct[candidate.candidate_id] = _evaluate_direct(
            executor,
            runtime_codecs[candidate.retained_rank],
            fit_boundaries,
        )
        executors[candidate.candidate_id] = executor
    fit_ceilings = {
        f"rank_{rank}": _evaluate_direct(
            None,
            runtime_codecs[rank],
            fit_boundaries,
            oracle=True,
        )
        for rank in sorted(scheduled_ranks)
    }
    guard.assert_unchanged()

    # Calibration B evaluates the frozen fitted schedule and performs the
    # only hyperparameter lock.
    selection_batches, selection_stream = _materialize_split(
        tokenizer,
        prompt_splits.calibration_b,
        split_name="calibration_b",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    selection_boundaries = _collect_boundaries(
        adapter,
        selection_batches,
        plan=plan,
    )
    selection_direct = {
        candidate.candidate_id: _evaluate_direct(
            executors[candidate.candidate_id],
            runtime_codecs[candidate.retained_rank],
            selection_boundaries,
        )
        for candidate in schedule
    }
    selection_behavior, selection_identity = _evaluate_behavior(
        adapter,
        selection_batches,
        plan=plan,
        candidates={
            candidate.candidate_id: (
                executors[candidate.candidate_id],
                runtime_codecs[candidate.retained_rank],
            )
            for candidate in schedule
        },
        include_identity=True,
    )
    assert selection_identity is not None
    rank_condition_names = {
        f"rank_{rank}": runtime_codecs[rank]
        for rank in sorted(scheduled_ranks)
    }
    selection_projection_behavior = _evaluate_projection_behavior(
        adapter,
        selection_batches,
        plan=plan,
        rank_codecs=rank_condition_names,
        full_output_codec=output_codec,
    )
    selection_ceilings = {
        f"rank_{rank}": _evaluate_direct(
            None,
            runtime_codecs[rank],
            selection_boundaries,
            oracle=True,
        )
        for rank in sorted(scheduled_ranks)
    }
    selection_roundtrip = _evaluate_full_width_delta_roundtrip(
        output_codec,
        selection_boundaries,
        device=device,
        dtype=torch.float32,
    )
    selection_roundtrip_math = _evaluate_full_width_delta_roundtrip(
        output_codec,
        selection_boundaries,
        device=device,
        dtype=torch.float64,
    )
    if not _identity_passed(
        selection_identity,
        nll_atol=identity_tolerance,
    ):
        raise RuntimeError("calibration-B no-op intervention identity failed")
    if not _roundtrip_passed(
        selection_roundtrip,
        selection_projection_behavior[
            "full_width_codec_delta_roundtrip"
        ],
        nll_atol=identity_tolerance,
    ):
        raise RuntimeError(
            "calibration-B full-width codec delta roundtrip failed"
        )
    selection_source_macs = _source_block_macs(
        adapter,
        plan,
        selection_boundaries,
        static=source_static,
    )
    selection_accounting = {
        candidate.candidate_id: _executor_accounting(
            executors[candidate.candidate_id],
            width=width,
            rank=candidate.retained_rank,
            boundaries=selection_boundaries,
            source_static=source_static,
            source_macs=selection_source_macs,
        )
        for candidate in schedule
    }
    ledger = [
        _gate_ledger(
            candidate=candidate,
            direct=selection_direct[candidate.candidate_id],
            behavior=selection_behavior[candidate.candidate_id],
            accounting=selection_accounting[candidate.candidate_id],
            thresholds=thresholds,
        )
        for candidate in schedule
    ]
    locked_id, selection_lock = _lock_candidate(ledger)
    locked_candidate = next(
        candidate
        for candidate in schedule
        if candidate.candidate_id == locked_id
    )
    guard.assert_unchanged()

    # Validation receives exactly one fitted executor configuration after the
    # lock.  The additional full-width and rank-ceiling interventions contain
    # no fitted executor parameters and are explicit controls.
    validation_batches, validation_stream = _materialize_split(
        tokenizer,
        prompt_splits.validation,
        split_name="validation",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    validation_boundaries = _collect_boundaries(
        adapter,
        validation_batches,
        plan=plan,
    )
    locked_executor = executors[locked_id]
    locked_codec = runtime_codecs[locked_candidate.retained_rank]
    validation_direct = _evaluate_direct(
        locked_executor,
        locked_codec,
        validation_boundaries,
    )
    validation_behavior_map, validation_identity = _evaluate_behavior(
        adapter,
        validation_batches,
        plan=plan,
        candidates={locked_id: (locked_executor, locked_codec)},
        include_identity=True,
    )
    assert validation_identity is not None
    validation_projection_behavior = _evaluate_projection_behavior(
        adapter,
        validation_batches,
        plan=plan,
        rank_codecs={
            f"rank_{locked_candidate.retained_rank}": locked_codec
        },
        full_output_codec=output_codec,
    )
    validation_ceiling = _evaluate_direct(
        None,
        locked_codec,
        validation_boundaries,
        oracle=True,
    )
    validation_roundtrip = _evaluate_full_width_delta_roundtrip(
        output_codec,
        validation_boundaries,
        device=device,
        dtype=torch.float32,
    )
    validation_roundtrip_math = _evaluate_full_width_delta_roundtrip(
        output_codec,
        validation_boundaries,
        device=device,
        dtype=torch.float64,
    )
    if not _identity_passed(
        validation_identity,
        nll_atol=identity_tolerance,
    ):
        raise RuntimeError("validation no-op intervention identity failed")
    if not _roundtrip_passed(
        validation_roundtrip,
        validation_projection_behavior[
            "full_width_codec_delta_roundtrip"
        ],
        nll_atol=identity_tolerance,
    ):
        raise RuntimeError(
            "validation full-width codec delta roundtrip failed"
        )
    validation_source_macs = _source_block_macs(
        adapter,
        plan,
        validation_boundaries,
        static=source_static,
    )
    validation_accounting = _executor_accounting(
        locked_executor,
        width=width,
        rank=locked_candidate.retained_rank,
        boundaries=validation_boundaries,
        source_static=source_static,
        source_macs=validation_source_macs,
    )
    validation_gate = _gate_ledger(
        candidate=locked_candidate,
        direct=validation_direct,
        behavior=validation_behavior_map[locked_id],
        accounting=validation_accounting,
        thresholds=thresholds,
    )
    overall_viable = (
        selection_lock["selection_viable"] is True
        and validation_gate["passed"] is True
    )
    guard.assert_unchanged()

    source_binding = {
        "schema": source_metadata["schema"],
        "format_version": source_metadata["format_version"],
        "scientific_payload_sha256": source_metadata[
            "scientific_payload_sha256"
        ],
        "report_sha256": source_metadata["report_sha256"],
        "locked_candidate": copy.deepcopy(dict(locked_source)),
        "codec_variant_id": variant_id,
        "input_codec_sha256": _codec_state_sha256(input_codec),
        "output_codec_sha256": _codec_state_sha256(output_codec),
        "prompt_disjointness": disjointness,
    }
    protocol = {
        "start_layer": start_layer,
        "end_layer_inclusive": end_layer,
        "layer_ids": plan.layer_ids,
        "canonical_boundaries": plan.activation_sites,
        "residual_width": width,
        "candidate_schedule": tuple(
            candidate.metadata() for candidate in schedule
        ),
        "fit_steps": fit_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "seed": seed,
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "fit_policy": "calibration_a_fixed_updates_only",
        "selection_policy": "calibration_b_conjunctive_gates",
        "validation_policy": (
            "one_locked_configuration_no_post_validation_selection"
        ),
        "test_policy": "parse_validate_hash_only",
        "raw_residual_bypass": True,
        "executor_target": "block_delta_in_output_decoder_subspace",
        "loss": "raw_residual_space_mse_after_output_decoder",
        "cross_path_attribution": (
            "one_expert_static_cross_vs_multi_expert_gated_cross_only_no_"
            "cross_disabled_fit_baseline"
        ),
        "fit_scope": (
            "one_seed_one_fixed_optimizer_schedule_failure_is_protocol_"
            "specific_not_capacity_impossibility"
        ),
        "thresholds": thresholds,
        "identity_nll_atol": identity_tolerance,
        "source_block_static": source_static,
        "source_codec_metadata": {
            "input": input_codec.metadata(),
            "output": output_codec.metadata(),
        },
        "runtime_codec_numeric_audit": runtime_codec_audit,
        "model_state_guard": guard.metadata(),
        "library_versions": _library_versions(),
        "tokenizer": _tokenizer_provenance(tokenizer),
        "tokenized_splits": {
            "calibration_a": fit_stream,
            "calibration_b": selection_stream,
            "validation": validation_stream,
        },
        "prompt_splits": prompt_metadata,
    }
    payload = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "scientific_status": {
            "scope": "residual_separated_gated_block_output_experiment",
            "fit_split": "calibration_a_only",
            "selection_split": "calibration_b_only",
            "validation_locked_before_evaluation": True,
            "locked_direct_boundary_evaluations_per_batch": 1,
            "locked_behavioral_interventions_per_batch": 1,
            "test_split_evaluated": False,
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "prompt_text_in_artifact": False,
            "selection_viable": selection_lock["selection_viable"],
            "validation_viable": validation_gate["passed"],
            "overall_viable": overall_viable,
            "runtime_speed_claim": False,
            "intervention_executes_native_block": True,
            "same_position_only_fit_baseline_evaluated": False,
            "causal_edge_benefit_claim": False,
            "capacity_impossibility_claim": False,
        },
        "model": model_metadata,
        "source_analysis": source_binding,
        "protocol": protocol,
        "codecs": {
            "input": input_codec.state_dict(),
            "output": output_codec.state_dict(),
        },
        "fit": {
            "executors": {
                candidate_id: executor.artifact_state_dict()
                for candidate_id, executor in executors.items()
            },
            "candidate_training": training,
            "candidate_direct": fit_direct,
            "rank_mse_lower_bound_direct": fit_ceilings,
            "tokenized_stream": fit_stream,
        },
        "selection": {
            "candidate_direct": selection_direct,
            "candidate_behavior": selection_behavior,
            "candidate_accounting": selection_accounting,
            "rank_mse_lower_bound_direct": selection_ceilings,
            "rank_projection_reference_behavior": {
                name: selection_projection_behavior[name]
                for name in rank_condition_names
            },
            "no_op_intervention_identity": selection_identity,
            "full_width_codec_delta_roundtrip": {
                "runtime_float32_direct": selection_roundtrip,
                "mathematical_float64_direct": (
                    selection_roundtrip_math
                ),
                "behavior": selection_projection_behavior[
                    "full_width_codec_delta_roundtrip"
                ],
                "passed": True,
            },
            "lock": selection_lock,
            "tokenized_stream": selection_stream,
        },
        "validation": {
            "locked_candidate": locked_candidate.metadata(),
            "direct": validation_direct,
            "behavior": validation_behavior_map[locked_id],
            "accounting": validation_accounting,
            "rank_mse_lower_bound_direct": validation_ceiling,
            "rank_projection_reference_behavior": (
                validation_projection_behavior[
                f"rank_{locked_candidate.retained_rank}"
                ]
            ),
            "no_op_intervention_identity": validation_identity,
            "full_width_codec_delta_roundtrip": {
                "runtime_float32_direct": validation_roundtrip,
                "mathematical_float64_direct": (
                    validation_roundtrip_math
                ),
                "behavior": validation_projection_behavior[
                    "full_width_codec_delta_roundtrip"
                ],
                "passed": True,
            },
            "viability_gate": validation_gate,
            "overall_viable": overall_viable,
            "tokenized_stream": validation_stream,
        },
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
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def load_gemma3_gated_executor_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strictly load, recompute, and cross-check a gated-executor artifact."""

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
        "contains_prompt_text",
        "contains_tokenizer_state",
        "scientific_status",
        "model",
        "source_analysis",
        "protocol",
        "codecs",
        "fit",
        "selection",
        "validation",
        "scientific_payload_sha256",
        "report_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("gated-executor artifact fields are invalid")
    if (
        raw["schema"] != _ARTIFACT_SCHEMA
        or raw["format_version"] != _ARTIFACT_FORMAT_VERSION
        or raw["contains_model_weights"] is not False
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
    ):
        raise ValueError("unsupported or unsafe gated-executor artifact")
    if (
        not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError("gated-executor digest fields are invalid")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    digest = _scientific_payload_sha256(payload)
    if digest != raw["scientific_payload_sha256"]:
        raise ValueError("gated-executor scientific payload digest mismatch")
    model = _validate_model_metadata(raw["model"])

    source = raw["source_analysis"]
    protocol = raw["protocol"]
    codecs_raw = raw["codecs"]
    fit = raw["fit"]
    selection = raw["selection"]
    validation = raw["validation"]
    status = raw["scientific_status"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            source,
            protocol,
            codecs_raw,
            fit,
            selection,
            validation,
            status,
        )
    ):
        raise ValueError("gated-executor payload mappings are invalid")
    protocol_fields = {
        "start_layer",
        "end_layer_inclusive",
        "layer_ids",
        "canonical_boundaries",
        "residual_width",
        "candidate_schedule",
        "fit_steps",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "seed",
        "maximum_tokenized_length",
        "tokenization_batch_size",
        "fit_policy",
        "selection_policy",
        "validation_policy",
        "test_policy",
        "raw_residual_bypass",
        "executor_target",
        "loss",
        "cross_path_attribution",
        "fit_scope",
        "thresholds",
        "identity_nll_atol",
        "source_block_static",
        "source_codec_metadata",
        "runtime_codec_numeric_audit",
        "model_state_guard",
        "library_versions",
        "tokenizer",
        "tokenized_splits",
        "prompt_splits",
    }
    if set(protocol) != protocol_fields:
        raise ValueError("gated-executor protocol fields are invalid")
    if (
        source.get("schema")
        != "fisher_graph.gemma3_weighted_jacobian"
        or not _is_sha256(source.get("scientific_payload_sha256"))
        or not _is_sha256(source.get("report_sha256"))
        or not _is_sha256(source.get("input_codec_sha256"))
        or not _is_sha256(source.get("output_codec_sha256"))
    ):
        raise ValueError("source-analysis binding is invalid")
    if set(codecs_raw) != {"input", "output"}:
        raise ValueError("runtime codec mapping is invalid")
    input_state = codecs_raw["input"]
    output_state = codecs_raw["output"]
    if not isinstance(input_state, Mapping) or not isinstance(
        output_state,
        Mapping,
    ):
        raise ValueError("runtime codec state is invalid")
    input_codec = LinearActivationCodec.from_state_dict(input_state)
    output_codec = LinearActivationCodec.from_state_dict(output_state)
    if (
        _codec_state_sha256(input_codec)
        != source["input_codec_sha256"]
        or _codec_state_sha256(output_codec)
        != source["output_codec_sha256"]
    ):
        raise ValueError("runtime codec does not match source binding")

    width = protocol.get("residual_width")
    schedule_raw = protocol.get("candidate_schedule")
    thresholds = protocol.get("thresholds")
    if (
        type(width) is not int
        or width <= 0
        or input_codec.width != width
        or output_codec.width != width
        or not isinstance(schedule_raw, tuple)
        or not schedule_raw
        or not isinstance(thresholds, Mapping)
    ):
        raise ValueError("gated-executor protocol geometry is invalid")
    candidates = []
    for row in schedule_raw:
        if not isinstance(row, Mapping) or set(row) != {
            "candidate_id",
            "retained_rank",
            "expert_count",
            "expert_rank",
            "router_width",
            "max_positive_lag",
        }:
            raise ValueError("candidate schedule entry is invalid")
        candidate = GatedExecutorCandidate(
            retained_rank=row["retained_rank"],  # type: ignore[arg-type]
            expert_count=row["expert_count"],  # type: ignore[arg-type]
            expert_rank=row["expert_rank"],  # type: ignore[arg-type]
            router_width=row["router_width"],  # type: ignore[arg-type]
            max_positive_lag=row["max_positive_lag"],  # type: ignore[arg-type]
        )
        if candidate.metadata() != row:
            raise ValueError("candidate schedule is noncanonical")
        candidates.append(candidate)
    if len({candidate.candidate_id for candidate in candidates}) != len(
        candidates
    ):
        raise ValueError("candidate schedule contains duplicates")

    if set(fit) != {
        "executors",
        "candidate_training",
        "candidate_direct",
        "rank_mse_lower_bound_direct",
        "tokenized_stream",
    }:
        raise ValueError("fit payload fields are invalid")
    executor_states = fit["executors"]
    candidate_ids = tuple(
        candidate.candidate_id for candidate in candidates
    )
    if (
        not isinstance(executor_states, Mapping)
        or tuple(executor_states) != candidate_ids
        or not isinstance(fit["candidate_training"], Mapping)
        or tuple(fit["candidate_training"]) != candidate_ids
        or not isinstance(fit["candidate_direct"], Mapping)
        or tuple(fit["candidate_direct"]) != candidate_ids
    ):
        raise ValueError("fit candidate bindings are invalid")
    executors = {}
    for candidate in candidates:
        state = executor_states[candidate.candidate_id]
        if not isinstance(state, Mapping):
            raise ValueError("executor artifact state is invalid")
        executor = (
            ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                state
            )
        )
        if executor.config != candidate.config():
            raise ValueError("executor config does not match schedule")
        executors[candidate.candidate_id] = executor

    split_names = ("calibration_a", "calibration_b", "validation")
    streams_raw = protocol.get("tokenized_splits")
    prompt_provenance = protocol.get("prompt_splits")
    if (
        not isinstance(streams_raw, Mapping)
        or tuple(streams_raw) != split_names
        or not isinstance(prompt_provenance, Mapping)
    ):
        raise ValueError("tokenized split provenance is invalid")
    validated_streams = {}
    for name in split_names:
        stream, _ = _validated_tokenized_stream(
            streams_raw[name],
            split_name=name,
        )
        validated_streams[name] = stream
    counts = prompt_provenance.get("counts")
    per_prompt = prompt_provenance.get("per_prompt_sha256")
    normalized = prompt_provenance.get("normalized_sha256")
    all_names = {*split_names, "test"}
    if (
        not isinstance(counts, Mapping)
        or set(counts) != all_names
        or not isinstance(per_prompt, Mapping)
        or set(per_prompt) != all_names
        or not isinstance(normalized, Mapping)
        or set(normalized) != all_names
    ):
        raise ValueError("prompt split provenance is invalid")
    all_hashes = []
    for name in all_names:
        hashes = per_prompt[name]
        if (
            type(counts[name]) is not int
            or counts[name] <= 0
            or not isinstance(hashes, list)
            or len(hashes) != counts[name]
            or any(not _is_sha256(item) for item in hashes)
            or normalized[name] != _ordered_prompt_hash_digest(hashes)
        ):
            raise ValueError("prompt split hashes are invalid")
        all_hashes.extend(hashes)
        if name in validated_streams and (
            validated_streams[name]["source_prompt_sha256"] != hashes
            or validated_streams[name]["sequences"] != counts[name]
        ):
            raise ValueError("tokenized stream does not bind prompt hashes")
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("prompt split hashes must be disjoint")
    streamed = {
        item
        for stream in validated_streams.values()
        for item in stream["source_prompt_sha256"]  # type: ignore[index]
    }
    if streamed & set(per_prompt["test"]):
        raise ValueError("reserved test hashes appear in tokenized streams")
    if (
        fit["tokenized_stream"] != validated_streams["calibration_a"]
        or selection.get("tokenized_stream")
        != validated_streams["calibration_b"]
        or validation.get("tokenized_stream")
        != validated_streams["validation"]
    ):
        raise ValueError("payload split stream aliases are invalid")

    selection_fields = {
        "candidate_direct",
        "candidate_behavior",
        "candidate_accounting",
        "rank_mse_lower_bound_direct",
        "rank_projection_reference_behavior",
        "no_op_intervention_identity",
        "full_width_codec_delta_roundtrip",
        "lock",
        "tokenized_stream",
    }
    validation_fields = {
        "locked_candidate",
        "direct",
        "behavior",
        "accounting",
        "rank_mse_lower_bound_direct",
        "rank_projection_reference_behavior",
        "no_op_intervention_identity",
        "full_width_codec_delta_roundtrip",
        "viability_gate",
        "overall_viable",
        "tokenized_stream",
    }
    if (
        set(selection) != selection_fields
        or set(validation) != validation_fields
    ):
        raise ValueError("selection/validation fields are invalid")
    selection_direct = selection["candidate_direct"]
    selection_behavior = selection["candidate_behavior"]
    selection_accounting = selection["candidate_accounting"]
    if (
        not isinstance(selection_direct, Mapping)
        or tuple(selection_direct) != candidate_ids
        or not isinstance(selection_behavior, Mapping)
        or tuple(selection_behavior) != candidate_ids
        or not isinstance(selection_accounting, Mapping)
        or tuple(selection_accounting) != candidate_ids
    ):
        raise ValueError("selection candidate metric bindings are invalid")
    expected_ledger = [
        _gate_ledger(
            candidate=candidate,
            direct=selection_direct[candidate.candidate_id],
            behavior=selection_behavior[candidate.candidate_id],
            accounting=selection_accounting[candidate.candidate_id],
            thresholds=thresholds,  # type: ignore[arg-type]
        )
        for candidate in candidates
    ]
    lock = selection.get("lock")
    if not isinstance(lock, Mapping) or not isinstance(
        lock.get("ledger"),
        list,
    ):
        raise ValueError("selection lock is invalid")
    if lock["ledger"] != expected_ledger:
        raise ValueError(
            "selection ledger does not match candidate metrics"
        )
    recomputed_id, recomputed_lock = _lock_candidate(expected_ledger)
    if recomputed_lock != lock:
        raise ValueError("selection lock does not recompute")
    locked_candidate = next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == recomputed_id
    )
    if validation.get("locked_candidate") != locked_candidate.metadata():
        raise ValueError("validation candidate does not match selection")
    direct = validation.get("direct")
    behavior = validation.get("behavior")
    accounting = validation.get("accounting")
    if not all(
        isinstance(value, Mapping)
        for value in (direct, behavior, accounting)
    ):
        raise ValueError("validation metrics are invalid")
    recomputed_gate = _gate_ledger(
        candidate=locked_candidate,
        direct=direct,
        behavior=behavior,
        accounting=accounting,
        thresholds=thresholds,  # type: ignore[arg-type]
    )
    if validation.get("viability_gate") != recomputed_gate:
        raise ValueError("validation viability gate does not recompute")
    expected_overall = (
        lock["selection_viable"] is True
        and recomputed_gate["passed"] is True
    )
    if (
        validation.get("overall_viable") is not expected_overall
        or status.get("overall_viable") is not expected_overall
        or status.get("selection_viable")
        is not lock["selection_viable"]
        or status.get("validation_viable")
        is not recomputed_gate["passed"]
    ):
        raise ValueError("overall viability binding is invalid")
    identity_tolerance = protocol["identity_nll_atol"]
    for section_name, section in (
        ("selection", selection),
        ("validation", validation),
    ):
        identity = section["no_op_intervention_identity"]
        roundtrip = section["full_width_codec_delta_roundtrip"]
        if (
            not isinstance(identity, Mapping)
            or not _identity_passed(
                identity,
                nll_atol=identity_tolerance,  # type: ignore[arg-type]
            )
            or not isinstance(roundtrip, Mapping)
            or set(roundtrip)
            != {
                "runtime_float32_direct",
                "mathematical_float64_direct",
                "behavior",
                "passed",
            }
            or roundtrip["passed"] is not True
            or not _roundtrip_passed(
                roundtrip["runtime_float32_direct"],  # type: ignore[arg-type]
                roundtrip["behavior"],  # type: ignore[arg-type]
                nll_atol=identity_tolerance,  # type: ignore[arg-type]
            )
        ):
            raise ValueError(
                f"{section_name} identity/roundtrip control is invalid"
            )
    if (
        status.get("test_split_evaluated") is not False
        or status.get("model_weights_changed") is not False
        or status.get("model_weights_in_artifact") is not False
        or status.get("prompt_text_in_artifact") is not False
        or status.get("runtime_speed_claim") is not False
    ):
        raise ValueError("scientific status claims are invalid")

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
        raise ValueError("gated-executor JSON report does not match payload")
    return {
        "model": model,
        "input_codec": input_codec,
        "output_codec": output_codec,
        "executors": executors,
        "locked_candidate": locked_candidate.metadata(),
        "locked_executor": executors[recomputed_id],
        "selection": copy.deepcopy(dict(selection)),
        "validation": copy.deepcopy(dict(validation)),
        "metadata": {
            "scientific_payload_sha256": digest,
            "report_sha256": raw["report_sha256"],
            "protocol": copy.deepcopy(dict(protocol)),
            "source_analysis": copy.deepcopy(dict(source)),
        },
        "report": copy.deepcopy(dict(report)),
    }


def _parse_lag(value: str) -> int | None:
    if value.lower() in {"none", "all", "unbounded"}:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "maximum lag must be a positive integer or 'none'"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "maximum lag must be a positive integer or 'none'"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit residual-separated gated modal executors on fresh Gemma "
            "splits and intervene at a locked block output."
        )
    )
    parser.add_argument("--weighted-artifact", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        default=DEFAULT_PROMPT_SPLITS,
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tokenization-batch-size", type=int, default=4)
    parser.add_argument(
        "--retained-ranks",
        dest="ranks",
        type=int,
        nargs="+",
        default=list(DEFAULT_RANKS),
    )
    parser.add_argument(
        "--expert-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXPERT_COUNTS),
    )
    parser.add_argument(
        "--expert-ranks",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXPERT_RANKS),
    )
    parser.add_argument(
        "--router-widths",
        type=int,
        nargs="+",
        default=list(DEFAULT_ROUTER_WIDTHS),
    )
    parser.add_argument(
        "--max-positive-lags",
        type=_parse_lag,
        nargs="+",
        default=list(DEFAULT_MAX_POSITIVE_LAGS),
    )
    parser.add_argument("--fit-steps", type=int, default=DEFAULT_FIT_STEPS)
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
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument(
        "--max-retained-fraction",
        type=float,
        default=DEFAULT_MAX_RETAINED_FRACTION,
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
    parser.add_argument(
        "--max-block-delta-nrmse",
        type=float,
        default=DEFAULT_MAX_BLOCK_DELTA_NRMSE,
    )
    parser.add_argument(
        "--min-block-delta-cosine",
        type=float,
        default=DEFAULT_MIN_BLOCK_DELTA_COSINE,
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
        "--identity-nll-atol",
        type=float,
        default=DEFAULT_IDENTITY_NLL_ATOL,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_gated_executor(
        weighted_artifact_path=arguments.weighted_artifact,
        model_id=arguments.model,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        prompt_splits_path=arguments.prompt_splits,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        ranks=arguments.ranks,
        expert_counts=arguments.expert_counts,
        expert_ranks=arguments.expert_ranks,
        router_widths=arguments.router_widths,
        max_positive_lags=arguments.max_positive_lags,
        fit_steps=arguments.fit_steps,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gradient_clip_norm=arguments.gradient_clip_norm,
        seed=arguments.seed,
        max_retained_fraction=arguments.max_retained_fraction,
        max_stored_coefficient_ratio=(
            arguments.max_stored_coefficient_ratio
        ),
        max_analytic_mac_ratio=arguments.max_analytic_mac_ratio,
        max_block_delta_nrmse=arguments.max_block_delta_nrmse,
        min_block_delta_cosine=arguments.min_block_delta_cosine,
        selection_nll_atol=arguments.selection_nll_atol,
        selection_top1_min=arguments.selection_top1_min,
        identity_nll_atol=arguments.identity_nll_atol,
        device_name=arguments.device,
        dtype=arguments.dtype,
        local_files_only=arguments.local_files_only,
        output=arguments.output,
    )
    validation = report["analysis"]["validation"]
    assert isinstance(validation, Mapping)
    print(
        json.dumps(
            {
                "output": report["artifact"]["tensor_output"],  # type: ignore[index]
                "locked_candidate": validation["locked_candidate"],
                "overall_viable": validation["overall_viable"],
                "validation_block_delta_nrmse": validation["direct"][
                    "block_delta_nrmse"
                ],
                "validation_delta_nll_per_token": validation["behavior"][
                    "delta_nll_per_token"
                ],
                "validation_top1_agreement": validation["behavior"][
                    "top1_agreement_to_baseline"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
