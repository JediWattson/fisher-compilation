"""Measure the current full-stack Gemma generator as a real model runtime.

This development runner strict-loads the frozen sequential-refit artifact,
prepares three complete Torch model scopes once, and then keeps validation,
hashing, device conversion, and MLP-stack switching outside timed forwards:

``native``
    The pinned Hugging Face Gemma 3 270M checkpoint.

``factorized_refit``
    The current composite catalog—base generators for layers 0-9 plus
    sequentially refit generators for layers 10-17—executed as two rank-640
    linear maps per MLP.

``fused_refit``
    The same generator maps materialized as one 640-by-640 affine map per MLP.

The fused scope is a runtime compaction probe, not a bit-exact rewrite.
Repeated floating-point rounding in the two-map source is different from one
materialized map, so the runner evaluates the fused scope on the complete
recorded development assessment before reporting its latency.

This is deliberately a Torch full-model measurement.  The repository's MLX
runtime currently covers local modal boundaries only; it does not yet contain
an MLX Gemma attention, RoPE, cache, tokenizer, or checkpoint-loading shell.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from typing import Any

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_modal_generator_dev_experiment import (
    DEFAULT_EVAL_EXPORT,
    DEFAULT_MAX_LENGTH,
    _safe_tokenized_stream_metadata,
    load_development_prompt_export,
)
from .modal_graph_rung_evaluation import (
    partition_development_export_for_interactions,
)
from .model_runtime_benchmark import (
    ModelRuntimeBenchmarkReport,
    benchmark_model_runtimes,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "evaluate_prepared_full_model_scopes",
    "load_gemma3_full_model_runtime_analysis",
    "main",
    "run_gemma3_full_model_runtime_analysis",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-full-model-runtime-analysis-dev-v1.json"
)
DEFAULT_CONTEXT_LENGTHS = (32, 128, 256)
DEFAULT_BATCH_SIZES = (1,)
DEFAULT_ROUNDS = 9
DEFAULT_WARMUP_CALLS = 3
DEFAULT_VOCABULARY_CHUNK_SIZE = 16384
_SYSTEMS = ("native", "factorized_refit", "fused_refit")
_SCHEMA = "fisher_graph.gemma3_full_model_runtime_analysis_development"
_FORMAT_VERSION = 1
_REPORT_DOMAIN = b"fisher-graph:gemma3-full-model-runtime-analysis:v1\0"
_PROVENANCE_SOURCE_FILES = (
    "src/fisher_graph/adapters/__init__.py",
    "src/fisher_graph/adapters/base.py",
    "src/fisher_graph/adapters/gemma3.py",
    "src/fisher_graph/gemma3_full_mlp_stack_executor.py",
    "src/fisher_graph/gemma3_full_mlp_stack_refit_runtime.py",
    "src/fisher_graph/gemma3_full_model_runtime_analysis.py",
    "src/fisher_graph/model_runtime_benchmark.py",
    "src/fisher_graph/prepared_gemma3_full_mlp_stack.py",
    "pyproject.toml",
)


def _progress(message: str) -> None:
    print(f"[gemma-full-model-runtime] {message}", file=sys.stderr, flush=True)


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdef"
            for character in value
        )
    )


def _source_code_provenance() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[2]
    files_sha256: dict[str, str] = {}
    for relative in _PROVENANCE_SOURCE_FILES:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"runtime provenance source is missing: {relative}"
            )
        files_sha256[relative] = _file_sha256(path)
    return {
        "binding": "sha256_of_listed_primary_runtime_sources",
        "files_sha256": files_sha256,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _report_sha256(value: object) -> str:
    return hashlib.sha256(
        _REPORT_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("percentile values must be nonempty")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _model_logits(output: object) -> Tensor:
    logits = (
        output.get("logits")
        if isinstance(output, Mapping)
        else getattr(output, "logits", None)
    )
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or not logits.is_floating_point()
    ):
        raise ValueError(
            "model output must expose floating [batch, sequence, vocab] logits"
        )
    return logits


def _selected_logits_and_targets(
    output: object,
    batch: CalibrationBatch,
) -> tuple[Tensor, Tensor]:
    logits = _model_logits(output)
    targets = batch.targets.to(device=logits.device)
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets and model logits positions differ")
    supervised = targets != -100
    valid = batch.valid_positions.to(device=logits.device)
    if valid.shape != supervised.shape or bool((supervised & ~valid).any()):
        raise ValueError("supervised targets must be valid positions")
    if batch.batch_size != 1 or not bool(supervised.any()):
        raise ValueError(
            "runtime fidelity replay requires nonempty batch-size-one inputs"
        )
    return logits[supervised].float(), targets[supervised].long()


def _condition_terms(
    logits: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    log_normalizer = torch.logsumexp(logits, dim=-1)
    rows = torch.arange(targets.shape[0], device=targets.device)
    nll = -(logits[rows, targets] - log_normalizer)
    top1 = logits.argmax(dim=-1)
    return nll, log_normalizer, top1


def _reference_to_candidate_kl_sum(
    reference: Tensor,
    candidate: Tensor,
    reference_lse: Tensor,
    candidate_lse: Tensor,
    *,
    vocabulary_chunk_size: int,
) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("pairwise logits shapes differ")
    total = 0.0
    for start in range(0, reference.shape[1], vocabulary_chunk_size):
        stop = min(start + vocabulary_chunk_size, reference.shape[1])
        reference_log_probability = (
            reference[:, start:stop] - reference_lse[:, None]
        )
        candidate_log_probability = (
            candidate[:, start:stop] - candidate_lse[:, None]
        )
        contribution = (
            reference_log_probability.exp()
            * (reference_log_probability - candidate_log_probability)
        )
        total += float(contribution.double().sum().item())
    return total


def _prompt_summary(
    *,
    nll_values: Sequence[float],
    native_nll_values: Sequence[float],
    top1_values: Sequence[float],
) -> dict[str, object]:
    if (
        len(nll_values) != len(native_nll_values)
        or len(nll_values) != len(top1_values)
        or not nll_values
    ):
        raise ValueError("prompt summaries require aligned nonempty values")
    deltas = tuple(
        candidate - native
        for candidate, native in zip(
            nll_values,
            native_nll_values,
            strict=True,
        )
    )
    return {
        "prompt_count": len(nll_values),
        "delta_nll_p50": _percentile(deltas, 0.50),
        "delta_nll_p90": _percentile(deltas, 0.90),
        "delta_nll_worst": max(deltas),
        "top1_agreement_p10": _percentile(top1_values, 0.10),
        "top1_agreement_worst": min(top1_values),
    }


def evaluate_prepared_full_model_scopes(
    switcher: PreparedGemma3FullMLPStackSwitcher,
    batches: Sequence[CalibrationBatch],
    *,
    vocabulary_chunk_size: int = DEFAULT_VOCABULARY_CHUNK_SIZE,
) -> dict[str, object]:
    """Replay native, factorized, and fused scopes on one frozen assessment."""

    materialized = tuple(batches)
    if (
        not materialized
        or any(not isinstance(batch, CalibrationBatch) for batch in materialized)
    ):
        raise ValueError("batches must contain CalibrationBatch values")
    if switcher.scopes != _SYSTEMS:
        raise ValueError("prepared switcher does not expose the exact systems")
    if (
        type(vocabulary_chunk_size) is not int
        or vocabulary_chunk_size <= 0
    ):
        raise ValueError("vocabulary_chunk_size must be positive")

    nll_sums = {name: 0.0 for name in _SYSTEMS}
    native_kl_sums = {name: 0.0 for name in _SYSTEMS[1:]}
    native_top1_matches = {name: 0 for name in _SYSTEMS[1:]}
    prompt_nlls = {name: [] for name in _SYSTEMS}
    prompt_native_top1 = {name: [] for name in _SYSTEMS[1:]}
    source_to_fused_kl_sum = 0.0
    source_to_fused_top1_matches = 0
    source_to_fused_diff_square_sum = 0.0
    source_logit_square_sum = 0.0
    source_to_fused_dot_sum = 0.0
    fused_logit_square_sum = 0.0
    source_to_fused_max_absolute = 0.0
    supervised_tokens = 0
    logical_valid_tokens = 0

    try:
        for batch in materialized:
            selected: dict[str, Tensor] = {}
            targets: Tensor | None = None
            with torch.inference_mode():
                for scope in _SYSTEMS:
                    switcher.switch(scope)
                    call_inputs: dict[str, object] = dict(batch.model_inputs)
                    call_inputs["use_cache"] = False
                    call_inputs["return_dict"] = True
                    output = switcher(**call_inputs)
                    scope_logits, scope_targets = (
                        _selected_logits_and_targets(output, batch)
                    )
                    if targets is None:
                        targets = scope_targets
                    elif not torch.equal(targets, scope_targets):
                        raise RuntimeError("runtime evaluation targets drifted")
                    selected[scope] = scope_logits
            assert targets is not None

            terms: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
            for scope in _SYSTEMS:
                terms[scope] = _condition_terms(selected[scope], targets)
                nll = terms[scope][0]
                nll_sum = float(nll.double().sum().item())
                nll_sums[scope] += nll_sum
                prompt_nlls[scope].append(nll_sum / targets.numel())

            native_lse = terms["native"][1]
            native_top1 = terms["native"][2]
            for scope in _SYSTEMS[1:]:
                native_kl_sums[scope] += (
                    _reference_to_candidate_kl_sum(
                        selected["native"],
                        selected[scope],
                        native_lse,
                        terms[scope][1],
                        vocabulary_chunk_size=vocabulary_chunk_size,
                    )
                )
                matches = int((terms[scope][2] == native_top1).sum().item())
                native_top1_matches[scope] += matches
                prompt_native_top1[scope].append(
                    matches / targets.numel()
                )

            factorized = selected["factorized_refit"]
            fused = selected["fused_refit"]
            source_to_fused_kl_sum += _reference_to_candidate_kl_sum(
                factorized,
                fused,
                terms["factorized_refit"][1],
                terms["fused_refit"][1],
                vocabulary_chunk_size=vocabulary_chunk_size,
            )
            source_to_fused_top1_matches += int(
                (
                    terms["factorized_refit"][2]
                    == terms["fused_refit"][2]
                ).sum().item()
            )
            for start in range(
                0,
                factorized.shape[1],
                vocabulary_chunk_size,
            ):
                stop = min(
                    start + vocabulary_chunk_size,
                    factorized.shape[1],
                )
                source_chunk = factorized[:, start:stop].double()
                fused_chunk = fused[:, start:stop].double()
                difference = fused_chunk - source_chunk
                source_to_fused_diff_square_sum += float(
                    (difference * difference).sum().item()
                )
                source_logit_square_sum += float(
                    (source_chunk * source_chunk).sum().item()
                )
                source_to_fused_dot_sum += float(
                    (source_chunk * fused_chunk).sum().item()
                )
                fused_logit_square_sum += float(
                    (fused_chunk * fused_chunk).sum().item()
                )
                source_to_fused_max_absolute = max(
                    source_to_fused_max_absolute,
                    float(difference.abs().max().item()),
                )

            supervised_tokens += targets.numel()
            logical_valid_tokens += int(batch.valid_positions.sum().item())
            del selected, terms, targets
            gc.collect()
    finally:
        switcher.switch("native")

    if supervised_tokens <= 0:
        raise RuntimeError("assessment replay has no supervised tokens")
    native_nll = nll_sums["native"] / supervised_tokens
    conditions: dict[str, object] = {
        "native": {
            "nll_per_token": native_nll,
            "delta_nll_per_token": 0.0,
            "native_to_candidate_kl_per_token": 0.0,
            "top1_agreement_to_native": 1.0,
        }
    }
    for scope in _SYSTEMS[1:]:
        nll = nll_sums[scope] / supervised_tokens
        conditions[scope] = {
            "nll_per_token": nll,
            "delta_nll_per_token": nll - native_nll,
            "native_to_candidate_kl_per_token": max(
                native_kl_sums[scope] / supervised_tokens,
                0.0,
            ),
            "top1_agreement_to_native": (
                native_top1_matches[scope] / supervised_tokens
            ),
            "prompt_tails": _prompt_summary(
                nll_values=prompt_nlls[scope],
                native_nll_values=prompt_nlls["native"],
                top1_values=prompt_native_top1[scope],
            ),
        }

    denominator = math.sqrt(
        source_logit_square_sum * fused_logit_square_sum
    )
    if source_logit_square_sum <= 0.0 or denominator <= 0.0:
        raise RuntimeError("factorized/fused logit norms are degenerate")
    factorized_nll = nll_sums["factorized_refit"] / supervised_tokens
    fused_nll = nll_sums["fused_refit"] / supervised_tokens
    return {
        "assessment_role": "open_development_assessment",
        "heldout_confirmation": False,
        "supervised_tokens": supervised_tokens,
        "logical_valid_tokens": logical_valid_tokens,
        "conditions": conditions,
        "fused_vs_factorized": {
            "delta_nll_per_token": fused_nll - factorized_nll,
            "factorized_to_fused_kl_per_token": max(
                source_to_fused_kl_sum / supervised_tokens,
                0.0,
            ),
            "top1_agreement_to_factorized": (
                source_to_fused_top1_matches / supervised_tokens
            ),
            "logit_nrmse": math.sqrt(
                source_to_fused_diff_square_sum
                / source_logit_square_sum
            ),
            "logit_cosine": (
                source_to_fused_dot_sum / denominator
            ),
            "logit_max_absolute_error": (
                source_to_fused_max_absolute
            ),
        },
    }


def _fixed_shape_inputs(
    tokenizer: object,
    prompts: Sequence[str],
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> dict[str, Tensor]:
    if (
        type(batch_size) is not int
        or batch_size <= 0
        or type(context_length) is not int
        or context_length < 2
    ):
        raise ValueError("benchmark batch and context sizes are invalid")
    if not prompts:
        raise ValueError("benchmark prompts must be nonempty")
    selected = tuple(prompts[index % len(prompts)] for index in range(batch_size))
    encoded = tokenizer(
        list(selected),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=context_length,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer must return a mapping")
    result = {
        name: value.to(device=device)
        for name, value in encoded.items()
        if isinstance(name, str) and isinstance(value, Tensor)
    }
    if (
        "input_ids" not in result
        or "attention_mask" not in result
        or result["input_ids"].shape != (batch_size, context_length)
        or result["attention_mask"].shape != (batch_size, context_length)
    ):
        raise ValueError("tokenizer did not create the fixed benchmark shape")
    return result


def _synchronize_for_device(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().to(device="cpu").contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _serialize_benchmark(
    report: ModelRuntimeBenchmarkReport,
    *,
    output_logits: Mapping[str, Tensor],
) -> dict[str, object]:
    if set(output_logits) != set(_SYSTEMS):
        raise RuntimeError("benchmark did not retain every system output")
    shapes = {tuple(output_logits[scope].shape) for scope in _SYSTEMS}
    dtypes = {output_logits[scope].dtype for scope in _SYSTEMS}
    if (
        len(shapes) != 1
        or len(dtypes) != 1
        or any(
            output_logits[scope].ndim != 3
            or output_logits[scope].shape[1] != 1
            or not bool(torch.isfinite(output_logits[scope]).all())
            for scope in _SYSTEMS
        )
    ):
        raise RuntimeError(
            "benchmark outputs must be finite same-shape last-logit tensors"
        )
    raw = asdict(report)
    timings = raw["timings"]
    if not isinstance(timings, dict):
        raise RuntimeError("benchmark timing serialization drifted")
    native_median = report.timings["native"].median_seconds
    speedups = {
        scope: native_median / report.timings[scope].median_seconds
        for scope in _SYSTEMS[1:]
    }
    speedups["fused_vs_factorized"] = (
        report.timings["factorized_refit"].median_seconds
        / report.timings["fused_refit"].median_seconds
    )
    return {
        **raw,
        "speedup_vs_native": speedups,
        "output_validation": {
            scope: {
                "shape": tuple(output_logits[scope].shape),
                "dtype": str(output_logits[scope].dtype),
                "sha256": _tensor_sha256(output_logits[scope]),
                "finite": True,
            }
            for scope in _SYSTEMS
        },
    }


def _benchmark_prefill(
    switcher: PreparedGemma3FullMLPStackSwitcher,
    model_inputs: Mapping[str, Tensor],
    *,
    rounds: int,
    warmup_calls: int,
    synchronize: Any,
) -> dict[str, object]:
    outputs: dict[str, Tensor] = {}

    def prepare(scope: str) -> None:
        outputs.pop(scope, None)
        switcher.switch(scope)

    def operation(scope: str) -> Any:
        def run() -> None:
            with torch.inference_mode():
                output = switcher(
                    **dict(model_inputs),
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
            outputs[scope] = _model_logits(output)

        return run

    report = benchmark_model_runtimes(
        {scope: operation(scope) for scope in _SYSTEMS},
        expected_systems=_SYSTEMS,
        processed_token_count=int(model_inputs["input_ids"].numel()),
        rounds=rounds,
        warmup_calls=warmup_calls,
        prepare_system=prepare,
        synchronize=synchronize,
    )
    switcher.switch("native")
    return _serialize_benchmark(report, output_logits=outputs)


def _benchmark_decode(
    switcher: PreparedGemma3FullMLPStackSwitcher,
    model_inputs: Mapping[str, Tensor],
    *,
    eos_token_id: int,
    rounds: int,
    warmup_calls: int,
    synchronize: Any,
) -> dict[str, object]:
    batch_size = int(model_inputs["input_ids"].shape[0])
    next_ids = torch.full(
        (batch_size, 1),
        eos_token_id,
        dtype=model_inputs["input_ids"].dtype,
        device=model_inputs["input_ids"].device,
    )
    extended_attention_mask = torch.cat(
        (
            model_inputs["attention_mask"],
            torch.ones(
                (batch_size, 1),
                dtype=model_inputs["attention_mask"].dtype,
                device=model_inputs["attention_mask"].device,
            ),
        ),
        dim=1,
    )
    states: dict[str, object] = {}
    outputs: dict[str, Tensor] = {}

    def prepare(scope: str) -> None:
        outputs.pop(scope, None)
        states.pop(scope, None)
        switcher.switch(scope)
        with torch.inference_mode():
            prefix = switcher(
                **dict(model_inputs),
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        past = getattr(prefix, "past_key_values", None)
        if past is None:
            raise RuntimeError("Gemma prefix did not return a KV cache")
        states[scope] = past

    def operation(scope: str) -> Any:
        def run() -> None:
            with torch.inference_mode():
                output = switcher(
                    input_ids=next_ids,
                    attention_mask=extended_attention_mask,
                    past_key_values=states[scope],
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
            outputs[scope] = _model_logits(output)

        return run

    report = benchmark_model_runtimes(
        {scope: operation(scope) for scope in _SYSTEMS},
        expected_systems=_SYSTEMS,
        processed_token_count=batch_size,
        rounds=rounds,
        warmup_calls=warmup_calls,
        prepare_system=prepare,
        synchronize=synchronize,
    )
    switcher.switch("native")
    return _serialize_benchmark(report, output_logits=outputs)


def _active_non_mlp_linear_macs(
    module: nn.Module,
) -> dict[str, int]:
    attention = 0
    head = 0
    other = 0
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear) or ".mlp." in name:
            continue
        count = child.weight.numel()
        if name == "lm_head" or name.endswith(".lm_head"):
            head += count
        elif ".self_attn." in name:
            attention += count
        else:
            other += count
    if attention <= 0 or head <= 0:
        raise RuntimeError("could not identify Gemma attention and LM head")
    return {
        "attention_projection_macs_per_token": attention,
        "other_linear_macs_per_token": other,
        "lm_head_macs_per_emitted_logit": head,
    }


def _resource_accounting(
    switcher: PreparedGemma3FullMLPStackSwitcher,
    *,
    source_whole_model_parameters: int,
) -> dict[str, object]:
    native_mlp = switcher.scope_accounting["native"].learned_parameter_count
    retained = source_whole_model_parameters - native_mlp
    if retained <= 0:
        raise RuntimeError("native Gemma parameter accounting is invalid")
    systems: dict[str, object] = {}
    for scope in _SYSTEMS:
        row = switcher.scope_accounting[scope]
        logical = (
            source_whole_model_parameters
            if scope == "native"
            else retained + row.learned_parameter_count
        )
        systems[scope] = {
            **asdict(row),
            "logical_whole_model_learned_parameters": logical,
            "logical_whole_model_parameter_savings": (
                source_whole_model_parameters - logical
            ),
            "logical_whole_model_parameter_reduction_fraction": (
                1.0 - logical / source_whole_model_parameters
            ),
            "native_mlp_linear_macs_saved_per_token": (
                switcher.scope_macs_per_token["native"]
                - switcher.scope_macs_per_token[scope]
            ),
            "native_mlp_linear_mac_reduction_fraction": (
                1.0
                - switcher.scope_macs_per_token[scope]
                / switcher.scope_macs_per_token["native"]
            ),
        }
    return {
        "source_whole_model_learned_parameters": (
            source_whole_model_parameters
        ),
        "retained_native_non_mlp_learned_parameters": retained,
        "systems": systems,
        "experimental_benchmark_resident_learned_parameters": (
            source_whole_model_parameters
            + switcher.scope_parameter_counts["factorized_refit"]
            + switcher.scope_parameter_counts["fused_refit"]
        ),
        "experimental_runtime_retains_all_scopes": True,
        "logical_deployment_retains_only_selected_scope": True,
    }


def _ideal_linear_speedups(
    *,
    kind: str,
    batch_size: int,
    context_length: int,
    scope_macs: Mapping[str, int],
    non_mlp_macs: Mapping[str, int],
) -> dict[str, float]:
    attention = non_mlp_macs["attention_projection_macs_per_token"]
    other = non_mlp_macs["other_linear_macs_per_token"]
    head = non_mlp_macs["lm_head_macs_per_emitted_logit"]
    token_positions = (
        batch_size * context_length if kind == "prefill" else batch_size
    )
    emitted_logits = batch_size
    totals = {
        scope: (
            token_positions * (attention + other + scope_macs[scope])
            + emitted_logits * head
        )
        for scope in _SYSTEMS
    }
    return {
        scope: totals["native"] / totals[scope]
        for scope in _SYSTEMS[1:]
    }


def _write_new_json(path: Path | str, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, destination)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite runtime analysis {destination}"
        ) from error
    finally:
        temp.unlink(missing_ok=True)


def _validate_runtime_analysis_payload(
    payload: Mapping[str, object],
) -> None:
    if (
        payload.get("schema") != _SCHEMA
        or payload.get("format_version") != _FORMAT_VERSION
    ):
        raise ValueError("runtime analysis schema or version is invalid")
    declared_digest = payload.get("report_sha256")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise ValueError("runtime analysis digest is invalid")
    without_digest = {
        key: value for key, value in payload.items() if key != "report_sha256"
    }
    if _report_sha256(without_digest) != declared_digest:
        raise ValueError("runtime analysis digest does not match its payload")

    protocol = payload.get("protocol")
    assessment = payload.get("assessment")
    resources = payload.get("resources")
    fidelity = payload.get("fidelity")
    benchmarks = payload.get("benchmarks")
    claim_scope = payload.get("claim_scope")
    source_code = payload.get("source_code")
    benchmark_environment = payload.get("benchmark_environment")
    if (
        not isinstance(protocol, Mapping)
        or tuple(protocol.get("systems", ())) != _SYSTEMS
        or not isinstance(assessment, Mapping)
        or not isinstance(resources, Mapping)
        or not isinstance(fidelity, Mapping)
        or isinstance(benchmarks, (str, bytes))
        or not isinstance(benchmarks, Sequence)
        or not benchmarks
        or not isinstance(claim_scope, Mapping)
        or not isinstance(source_code, Mapping)
        or not isinstance(benchmark_environment, Mapping)
        or not isinstance(benchmark_environment.get("processor"), str)
    ):
        raise ValueError("runtime analysis top-level structure is invalid")
    source_files = source_code.get("files_sha256")
    if (
        set(source_code) != {"binding", "files_sha256"}
        or source_code.get("binding")
        != "sha256_of_listed_primary_runtime_sources"
        or not isinstance(source_files, Mapping)
        or set(source_files) != set(_PROVENANCE_SOURCE_FILES)
        or any(not _is_sha256(value) for value in source_files.values())
    ):
        raise ValueError("runtime analysis source-code provenance is invalid")
    expected_claim_scope = {
        "logical_parameter_and_linear_mac_counts": True,
        "torch_full_model_latency_measured": True,
        "mlx_full_model_latency_measured": False,
        "fused_scope_is_a_rate_distortion_point": True,
        "fused_scope_is_bit_exact_to_factorized": False,
        "heldout_compression_claim": False,
        "downstream_accuracy_claim": False,
    }
    if dict(claim_scope) != expected_claim_scope:
        raise ValueError("runtime analysis claim scope is invalid")

    assessment_content = assessment.get("content_sha256")
    assessment_count = assessment.get("example_count")
    if (
        not _is_sha256(assessment.get("serialized_sha256"))
        or isinstance(assessment_content, (str, bytes))
        or not isinstance(assessment_content, Sequence)
        or type(assessment_count) is not int
        or assessment_count <= 0
        or len(assessment_content) != assessment_count
        or assessment.get("prompt_text_stored") is not False
        or assessment.get("token_ids_stored") is not False
        or assessment.get("family_disjoint_confirmation") is not False
    ):
        raise ValueError("runtime analysis assessment declaration is invalid")
    assessment_hashes = tuple(assessment_content)
    if (
        any(not _is_sha256(value) for value in assessment_hashes)
        or len(set(assessment_hashes)) != assessment_count
    ):
        raise ValueError("runtime analysis assessment declaration is invalid")

    protocol_rounds = protocol.get("rounds")
    protocol_warmup_calls = protocol.get("warmup_calls")
    protocol_batch_sizes = protocol.get("batch_sizes")
    protocol_context_lengths = protocol.get("context_lengths")
    if (
        type(protocol_rounds) is not int
        or protocol_rounds <= 0
        or type(protocol_warmup_calls) is not int
        or protocol_warmup_calls < 0
        or isinstance(protocol_batch_sizes, (str, bytes))
        or not isinstance(protocol_batch_sizes, Sequence)
        or not protocol_batch_sizes
        or any(
            type(value) is not int or value <= 0
            for value in protocol_batch_sizes
        )
        or len(set(protocol_batch_sizes)) != len(protocol_batch_sizes)
        or isinstance(protocol_context_lengths, (str, bytes))
        or not isinstance(protocol_context_lengths, Sequence)
        or not protocol_context_lengths
        or any(
            type(value) is not int or value < 2
            for value in protocol_context_lengths
        )
        or len(set(protocol_context_lengths))
        != len(protocol_context_lengths)
    ):
        raise ValueError("runtime analysis benchmark protocol is invalid")
    declared_batch_sizes = tuple(protocol_batch_sizes)
    declared_context_lengths = tuple(protocol_context_lengths)
    resource_systems = resources.get("systems")
    fidelity_conditions = fidelity.get("conditions")
    if (
        not isinstance(resource_systems, Mapping)
        or set(resource_systems) != set(_SYSTEMS)
        or not isinstance(fidelity_conditions, Mapping)
        or set(fidelity_conditions) != set(_SYSTEMS)
    ):
        raise ValueError("runtime analysis systems are incomplete")

    expected_timing_fields = {
        "cold_seconds",
        "raw_round_seconds",
        "median_seconds",
        "p10_seconds",
        "p90_seconds",
        "p95_seconds",
        "processed_token_count",
        "tokens_per_second",
    }
    expected_output_fields = {"shape", "dtype", "sha256", "finite"}
    expected_speedup_names = {
        "factorized_refit",
        "fused_refit",
        "fused_vs_factorized",
    }
    benchmark_kinds = {
        "prefill_last_logit",
        "cached_single_token_decode",
    }
    observed_matrix: set[tuple[str, int, int]] = set()
    observed_output_vocabulary_width: int | None = None
    observed_output_dtype: str | None = None
    for benchmark in benchmarks:
        if not isinstance(benchmark, Mapping):
            raise TypeError("runtime benchmark row must be a mapping")
        timings = benchmark.get("timings")
        speedups = benchmark.get("speedup_vs_native")
        rounds = benchmark.get("rounds")
        warmup_calls = benchmark.get("warmup_calls")
        kind = benchmark.get("kind")
        batch_size = benchmark.get("batch_size")
        context_length = benchmark.get("context_length")
        processed_token_count = benchmark.get("processed_token_count")
        if (
            not isinstance(kind, str)
            or kind not in benchmark_kinds
            or type(batch_size) is not int
            or batch_size <= 0
            or type(context_length) is not int
            or context_length < 2
        ):
            raise ValueError("runtime benchmark matrix coordinates are invalid")
        expected_processed_tokens = (
            batch_size * context_length
            if kind == "prefill_last_logit"
            else batch_size
        )
        if processed_token_count != expected_processed_tokens:
            raise ValueError(
                "runtime benchmark processed-token count is inconsistent"
            )
        matrix_key = (kind, batch_size, context_length)
        if matrix_key in observed_matrix:
            raise ValueError("runtime benchmark matrix contains a duplicate row")
        observed_matrix.add(matrix_key)

        round_orders = benchmark.get("round_orders")
        expected_round_orders = tuple(
            (
                _SYSTEMS[offset:] + _SYSTEMS[:offset]
            )
            for offset in (
                round_index % len(_SYSTEMS)
                for round_index in range(protocol_rounds)
            )
        )
        if (
            isinstance(round_orders, (str, bytes))
            or not isinstance(round_orders, Sequence)
            or len(round_orders) != protocol_rounds
            or any(
                isinstance(order, (str, bytes))
                or not isinstance(order, Sequence)
                for order in round_orders
            )
        ):
            raise ValueError("runtime benchmark round orders are invalid")
        normalized_round_orders = tuple(
            tuple(order) for order in round_orders
        )
        if (
            tuple(benchmark.get("system_names", ())) != _SYSTEMS
            or not isinstance(timings, Mapping)
            or set(timings) != set(_SYSTEMS)
            or not isinstance(speedups, Mapping)
            or set(speedups) != expected_speedup_names
            or type(rounds) is not int
            or rounds != protocol_rounds
            or type(warmup_calls) is not int
            or warmup_calls != protocol_warmup_calls
            or normalized_round_orders != expected_round_orders
        ):
            raise ValueError("runtime benchmark system declaration is invalid")

        output_validation = benchmark.get("output_validation")
        if (
            not isinstance(output_validation, Mapping)
            or set(output_validation) != set(_SYSTEMS)
        ):
            raise ValueError("runtime benchmark output validation is incomplete")
        output_shape: tuple[int, ...] | None = None
        output_dtype: str | None = None
        for scope in _SYSTEMS:
            output = output_validation[scope]
            if (
                not isinstance(output, Mapping)
                or set(output) != expected_output_fields
            ):
                raise ValueError(
                    "runtime benchmark output validation is invalid"
                )
            shape = output.get("shape")
            dtype = output.get("dtype")
            digest = output.get("sha256")
            if (
                isinstance(shape, (str, bytes))
                or not isinstance(shape, Sequence)
                or len(shape) != 3
                or any(type(value) is not int or value <= 0 for value in shape)
                or tuple(shape[:2]) != (batch_size, 1)
                or not isinstance(dtype, str)
                or not dtype
                or not _is_sha256(digest)
                or output.get("finite") is not True
            ):
                raise ValueError(
                    "runtime benchmark output validation is invalid"
                )
            if output_shape is None:
                output_shape = tuple(shape)
                output_dtype = dtype
            elif tuple(shape) != output_shape or dtype != output_dtype:
                raise ValueError(
                    "runtime benchmark output shapes or dtypes differ"
                )
        assert output_shape is not None
        assert output_dtype is not None
        if observed_output_vocabulary_width is None:
            observed_output_vocabulary_width = output_shape[2]
            observed_output_dtype = output_dtype
        elif (
            output_shape[2] != observed_output_vocabulary_width
            or output_dtype != observed_output_dtype
        ):
            raise ValueError(
                "runtime benchmark output vocabulary or dtype drifted"
            )

        medians: dict[str, float] = {}
        for scope in _SYSTEMS:
            timing = timings[scope]
            if (
                not isinstance(timing, Mapping)
                or set(timing) != expected_timing_fields
            ):
                raise TypeError("runtime timing row must be a mapping")
            raw = timing.get("raw_round_seconds")
            median = timing.get("median_seconds")
            cold = timing.get("cold_seconds")
            p10 = timing.get("p10_seconds")
            p90 = timing.get("p90_seconds")
            p95 = timing.get("p95_seconds")
            timing_processed_tokens = timing.get("processed_token_count")
            throughput = timing.get("tokens_per_second")
            if (
                isinstance(raw, (str, bytes))
                or not isinstance(raw, Sequence)
                or len(raw) != rounds
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in raw
                )
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in (cold, median, p10, p90, p95, throughput)
                )
                or timing_processed_tokens != processed_token_count
            ):
                raise ValueError("runtime timing samples are inconsistent")
            raw_values = tuple(float(value) for value in raw)
            expected_summaries = {
                "median_seconds": statistics.median(raw_values),
                "p10_seconds": _percentile(raw_values, 0.10),
                "p90_seconds": _percentile(raw_values, 0.90),
                "p95_seconds": _percentile(raw_values, 0.95),
            }
            for name, expected in expected_summaries.items():
                if not math.isclose(
                    float(timing[name]),
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "runtime timing summaries are inconsistent"
                    )
            median_value = float(median)
            if not math.isclose(
                float(throughput),
                processed_token_count / median_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("runtime timing throughput is inconsistent")
            medians[scope] = median_value
        expected_speedups = {
            "factorized_refit": (
                medians["native"] / medians["factorized_refit"]
            ),
            "fused_refit": medians["native"] / medians["fused_refit"],
            "fused_vs_factorized": (
                medians["factorized_refit"] / medians["fused_refit"]
            ),
        }
        for name, expected in expected_speedups.items():
            actual = speedups.get(name)
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("runtime benchmark speedup is inconsistent")

    expected_matrix = {
        (kind, batch_size, context_length)
        for kind in benchmark_kinds
        for batch_size in declared_batch_sizes
        for context_length in declared_context_lengths
    }
    if observed_matrix != expected_matrix:
        raise ValueError("runtime benchmark matrix is not complete")


def load_gemma3_full_model_runtime_analysis(
    path: Path | str,
) -> dict[str, object]:
    """Strict-load the report envelope and benchmark timing evidence."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("runtime analysis must be a JSON object")
    _validate_runtime_analysis_payload(raw)
    return raw


def run_gemma3_full_model_runtime_analysis(
    *,
    revision: str,
    eval_export_path: Path | str = DEFAULT_EVAL_EXPORT,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    output: Path | str = DEFAULT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    context_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    rounds: int = DEFAULT_ROUNDS,
    warmup_calls: int = DEFAULT_WARMUP_CALLS,
    vocabulary_chunk_size: int = DEFAULT_VOCABULARY_CHUNK_SIZE,
) -> dict[str, object]:
    """Run fidelity and prepared model-level latency on the current refit."""

    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("runtime analysis output must use .json")
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite runtime analysis {destination}"
        )
    context_lengths = tuple(context_lengths)
    batch_sizes = tuple(batch_sizes)
    if (
        not context_lengths
        or any(type(value) is not int or value < 2 for value in context_lengths)
        or len(set(context_lengths)) != len(context_lengths)
        or not batch_sizes
        or any(type(value) is not int or value <= 0 for value in batch_sizes)
        or len(set(batch_sizes)) != len(batch_sizes)
    ):
        raise ValueError("benchmark contexts and batches must be unique")

    base_path = Path(base_artifact_path)
    refit_path = Path(refit_artifact_path)
    _progress("artifacts: strict-load base plus sequential refit")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_path,
        refit_path,
    )
    replacements = catalog.replacements
    model_metadata = dict(catalog.model_metadata)
    analysis_split = dict(catalog.analysis_split)
    partition_metadata = dict(catalog.partition_metadata)
    frozen_refit_metrics = dict(catalog.frozen_refit_metrics)
    source_resources = dict(catalog.resource_accounting)
    source_model_sha256 = catalog.source_model_sha256
    artifact_metadata = {
        "base_tensor_file": str(base_path),
        "base_tensor_file_sha256": catalog.base_artifact_file_sha256,
        "base_scientific_payload_sha256": (
            catalog.base_scientific_payload_sha256
        ),
        "refit_tensor_file": str(refit_path),
        "refit_tensor_file_sha256": catalog.refit_artifact_file_sha256,
        "refit_scientific_payload_sha256": (
            catalog.refit_scientific_payload_sha256
        ),
    }
    del catalog
    gc.collect()

    if (
        model_metadata.get("model_id") != model_id
        or model_metadata.get("requested_revision") != revision
        or model_metadata.get("resolved_commit") != revision
        or model_metadata.get("local_files_only") is not True
    ):
        raise ValueError("requested model differs from the refit artifact")

    development_export = load_development_prompt_export(eval_export_path)
    selection_count = partition_metadata.get("selection_prompt_count")
    expected_prompt_count = partition_metadata.get("expected_prompt_count")
    if type(selection_count) is not int or type(expected_prompt_count) is not int:
        raise TypeError("artifact partition counts must be integers")
    partition = partition_development_export_for_interactions(
        development_export,
        selection_count=selection_count,
        expected_prompt_count=expected_prompt_count,
    )
    if partition.metadata() != partition_metadata:
        raise ValueError("live development partition differs from the artifact")

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
    if adapter.model_fingerprint() != source_model_sha256:
        raise ValueError("live Gemma fingerprint differs from the artifact")

    _progress("runtime: prepare native, factorized, and fused model scopes")
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {"factorized_refit": replacements},
        fused_variants={"fused_refit": "factorized_refit"},
    )
    del replacements
    gc.collect()

    try:
        _progress("fidelity: replay the complete recorded assessment20")
        assessment_batches, assessment_stream = _materialize_split(
            tokenizer,
            partition.assessment.prompts,
            split_name="full_mlp_stack_open_development_assessment",
            max_length=DEFAULT_MAX_LENGTH,
            tokenization_batch_size=1,
            device=device,
        )
        observed_split = _safe_tokenized_stream_metadata(assessment_stream)
        expected_content = analysis_split.get("content_sha256")
        if (
            observed_split.get("serialized_sha256")
            != analysis_split.get("serialized_sha256")
            or tuple(observed_split.get("content_sha256", ()))
            != tuple(expected_content or ())
            or observed_split.get("sequences")
            != analysis_split.get("example_count")
            or observed_split.get("valid_tokens", {}).get("total")
            != analysis_split.get("logical_valid_tokens")
            or observed_split.get("supervised_positions", {}).get("total")
            != analysis_split.get("supervised_tokens")
        ):
            raise ValueError(
                "runtime assessment tokenization differs from the artifact"
            )
        fidelity = evaluate_prepared_full_model_scopes(
            switcher,
            assessment_batches,
            vocabulary_chunk_size=vocabulary_chunk_size,
        )
        factorized_metrics = fidelity["conditions"][  # type: ignore[index]
            "factorized_refit"
        ]
        for name in (
            "nll_per_token",
            "delta_nll_per_token",
            "native_to_candidate_kl_per_token",
            "top1_agreement_to_native",
        ):
            if not math.isclose(
                float(factorized_metrics[name]),  # type: ignore[index]
                float(frozen_refit_metrics[name]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "prepared factorized endpoint differs from frozen metrics"
                )
        fidelity["control_validation"] = {
            "factorized_matches_authenticated_refit_metrics": True,
            "complete_assessment_membership_replayed": True,
            "assessment_used_for_runtime_selection": False,
            "fused_materialization_selected_before_replay": True,
        }
        del assessment_batches, assessment_stream
        gc.collect()

        source_whole_parameters = source_resources.get(
            "source_whole_model_learned_parameters"
        )
        if type(source_whole_parameters) is not int:
            raise TypeError("source whole-model parameter count is invalid")
        resources = _resource_accounting(
            switcher,
            source_whole_model_parameters=source_whole_parameters,
        )
        non_mlp_macs = _active_non_mlp_linear_macs(model)

        synchronize = (
            None
            if device.type == "cpu"
            else lambda: _synchronize_for_device(device)
        )
        benchmarks: list[dict[str, object]] = []
        for batch_size in batch_sizes:
            for context_length in context_lengths:
                inputs = _fixed_shape_inputs(
                    tokenizer,
                    partition.assessment.prompts,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                )
                _progress(
                    "benchmark: prefill "
                    f"batch={batch_size} context={context_length}"
                )
                prefill = _benchmark_prefill(
                    switcher,
                    inputs,
                    rounds=rounds,
                    warmup_calls=warmup_calls,
                    synchronize=synchronize,
                )
                prefill["kind"] = "prefill_last_logit"
                prefill["batch_size"] = batch_size
                prefill["context_length"] = context_length
                prefill["ideal_linear_only_speedup_vs_native"] = (
                    _ideal_linear_speedups(
                        kind="prefill",
                        batch_size=batch_size,
                        context_length=context_length,
                        scope_macs=switcher.scope_macs_per_token,
                        non_mlp_macs=non_mlp_macs,
                    )
                )
                benchmarks.append(prefill)

                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if type(eos_token_id) is not int:
                    raise TypeError("tokenizer eos_token_id must be an integer")
                _progress(
                    "benchmark: cached decode "
                    f"batch={batch_size} context={context_length}"
                )
                decode = _benchmark_decode(
                    switcher,
                    inputs,
                    eos_token_id=eos_token_id,
                    rounds=rounds,
                    warmup_calls=warmup_calls,
                    synchronize=synchronize,
                )
                decode["kind"] = "cached_single_token_decode"
                decode["batch_size"] = batch_size
                decode["context_length"] = context_length
                decode["ideal_linear_only_speedup_vs_native"] = (
                    _ideal_linear_speedups(
                        kind="decode",
                        batch_size=batch_size,
                        context_length=context_length,
                        scope_macs=switcher.scope_macs_per_token,
                        non_mlp_macs=non_mlp_macs,
                    )
                )
                benchmarks.append(decode)
                del inputs
                gc.collect()

        report_without_digest: dict[str, object] = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "scientific_status": (
                "self_attested_open_development_runtime_analysis"
            ),
            "model": {
                "model_id": model_id,
                "requested_revision": revision,
                "resolved_commit": revision,
                "adapter_model_fingerprint": source_model_sha256,
                "device": str(device),
                "dtype": dtype,
                "local_files_only": True,
            },
            "source_artifacts": artifact_metadata,
            "source_code": _source_code_provenance(),
            "protocol": {
                "systems": _SYSTEMS,
                "factorized_scope": (
                    "authenticated base L0-9 plus sequential-refit L10-17 "
                    "rank-640 generators"
                ),
                "fused_scope": (
                    "one float64-materialized then runtime-dtype affine "
                    "residual map per independent layer"
                ),
                "prepared_outside_timing": True,
                "scope_switching_outside_timing": True,
                "hashing_outside_timing": True,
                "device_transfer_outside_timing": True,
                "deterministic_rotating_system_order": True,
                "rounds": rounds,
                "warmup_calls": warmup_calls,
                "batch_sizes": batch_sizes,
                "context_lengths": context_lengths,
                "prefill_logits_to_keep": 1,
                "decode_uses_kv_cache": True,
                "assessment_role": "open_development_assessment",
                "heldout_confirmation": False,
                "downstream_task_accuracy_measured": False,
                "full_model_runtime_measured": True,
                "mlx_full_model_runtime_measured": False,
                "mlx_boundary_kernel_result_is_separate": True,
            },
            "assessment": {
                "serialized_sha256": analysis_split["serialized_sha256"],
                "content_sha256": tuple(analysis_split["content_sha256"]),
                "example_count": analysis_split["example_count"],
                "prompt_text_stored": False,
                "token_ids_stored": False,
                "family_disjoint_confirmation": False,
            },
            "fidelity": fidelity,
            "resources": {
                **resources,
                "active_non_mlp_linear_macs": non_mlp_macs,
                "linear_only_ideal_speedups_exclude": (
                    "attention score/value products, normalization, "
                    "activations, cache traffic, allocation, and framework "
                    "overhead"
                ),
            },
            "benchmark_environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "device": str(device),
                "dtype": dtype,
                "wall_clock": "time.perf_counter",
                "synchronization": (
                    "eager_cpu"
                    if device.type == "cpu"
                    else f"torch_{device.type}_synchronize"
                ),
                "run_started_unix_seconds": time.time(),
            },
            "benchmarks": benchmarks,
            "claim_scope": {
                "logical_parameter_and_linear_mac_counts": True,
                "torch_full_model_latency_measured": True,
                "mlx_full_model_latency_measured": False,
                "fused_scope_is_a_rate_distortion_point": True,
                "fused_scope_is_bit_exact_to_factorized": False,
                "heldout_compression_claim": False,
                "downstream_accuracy_claim": False,
            },
        }
        payload = {
            **report_without_digest,
            "report_sha256": _report_sha256(report_without_digest),
        }
        _write_new_json(destination, payload)
        restored = load_gemma3_full_model_runtime_analysis(destination)
        if _canonical_json_bytes(restored) != _canonical_json_bytes(payload):
            destination.unlink(missing_ok=True)
            raise RuntimeError("post-save runtime analysis replay differs")
    finally:
        switcher.close()

    _progress(f"wrote {destination}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the prepared native, factorized-refit, and fused-refit "
            "Gemma full-model runtimes."
        )
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--eval-export", type=Path, default=DEFAULT_EVAL_EXPORT)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_CONTEXT_LENGTHS,
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument(
        "--warmup-calls",
        type=int,
        default=DEFAULT_WARMUP_CALLS,
    )
    parser.add_argument(
        "--vocabulary-chunk-size",
        type=int,
        default=DEFAULT_VOCABULARY_CHUNK_SIZE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload = run_gemma3_full_model_runtime_analysis(
        revision=arguments.revision,
        eval_export_path=arguments.eval_export,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        output=arguments.output,
        model_id=arguments.model,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        context_lengths=arguments.context_lengths,
        batch_sizes=arguments.batch_sizes,
        rounds=arguments.rounds,
        warmup_calls=arguments.warmup_calls,
        vocabulary_chunk_size=arguments.vocabulary_chunk_size,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
