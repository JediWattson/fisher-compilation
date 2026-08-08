"""Authenticate an exact residual-carrier graph for every Gemma 3 block.

This development rung clones each of the pinned Gemma 3 270M transformer
blocks once, executes the clones through one dense residual-carrier wire, and
compares the resulting full transformer body with the native model.  The
embedding, final normalization, and tied language-model head remain native
boundary modules.

The cloned blocks retain checkpoint tensors and native width.  This is an
exact execution/ABI control, not source-free compilation, compression, or a
latency result.  Only cache-free eager prefill on CPU float32 is in scope.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, ModelAdapter
from .complete_block_generator_stack import (
    CompleteBlockGeneratorStackExecutor,
)
from .complete_block_residual_forms import (
    CompleteBlockResidualForm,
    ResidualForm,
)
from .gemma3_ablation_experiment import _FrozenModelTensorGuard
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_width_single_layer_experiment import (
    _assert_source_independence,
)
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
    DEFAULT_PANEL_PATH,
    DEFAULT_REVISION,
    _load_prompt_split,
    _tokenize_batches,
)
from .gemma3_one_block_residual_form_experiment import (
    _LogitMetricAccumulator,
    _TensorMetricAccumulator,
    _native_nll_sum,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)


FULL_BODY_RESIDUAL_CARRIER_EXPERIMENT_SCHEMA = (
    "fisher_graph.gemma3_full_body_residual_carrier.experiment.v1"
)
FULL_BODY_RESIDUAL_CARRIER_EXPERIMENT_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "full-body-residual-carrier-dev-v1.json"
)
DEFAULT_STAGE_ATOL = 2.0e-5
DEFAULT_LOGIT_ATOL = 2.0e-5
DEFAULT_NLL_ATOL = 2.0e-6
DEFAULT_KL_MAX = 2.0e-7
_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-full-body-residual-carrier-report:v1\0"
)
_TOKEN_HASH_DOMAIN = (
    b"fisher_graph:gemma3-full-body-residual-carrier-tokens:v1\0"
)
_COMPILE_SEED = 982_451_653
_EXPECTED_PINNED_LAYER_COUNT = 18
_REQUIRED_LEAF_SUFFIXES = (
    "attention_input_norm",
    "attention.q_proj",
    "attention.k_proj",
    "attention.v_proj",
    "attention.q_norm",
    "attention.k_norm",
    "attention.o_proj",
    "attention_output_norm",
    "feed_forward_input_norm",
    "feed_forward.gate_proj",
    "feed_forward.up_proj",
    "feed_forward.down_proj",
    "feed_forward_output_norm",
)


__all__ = [
    "DEFAULT_OUTPUT",
    "FULL_BODY_RESIDUAL_CARRIER_EXPERIMENT_FORMAT_VERSION",
    "FULL_BODY_RESIDUAL_CARRIER_EXPERIMENT_SCHEMA",
    "load_gemma3_full_body_residual_carrier_report",
    "run_gemma3_full_body_residual_carrier_experiment",
]


def _progress(message: str) -> None:
    print(f"[full-body-residual-carrier] {message}", flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _report_sha256(report_without_hash: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _REPORT_HASH_DOMAIN + _canonical_json_bytes(report_without_hash)
    ).hexdigest()


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Tensor):
        raise TypeError("reports cannot contain tensors")
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported report value: {type(value).__qualname__}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping_path(value: object, *keys: str) -> object:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _token_tensor_digest(
    batches: Sequence[Mapping[str, Tensor]],
    name: str,
) -> str:
    digest = hashlib.sha256(_TOKEN_HASH_DOMAIN + name.encode("ascii") + b"\0")
    for batch in batches:
        value = batch.get(name)
        if not isinstance(value, Tensor):
            raise ValueError(f"tokenized batch omitted {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tokenization_receipt(
    tokenizer: object,
    batches: Sequence[Mapping[str, Tensor]],
    *,
    requested_batch_size: int,
) -> dict[str, object]:
    if not batches or requested_batch_size <= 0:
        raise ValueError("tokenization receipt requires nonempty batches")
    shapes: list[list[int]] = []
    valid_lengths: list[int] = []
    padding = {"left": 0, "right": 0, "none": 0}
    padded_rows = 0
    examples = 0
    for batch in batches:
        input_ids = batch.get("input_ids")
        mask = batch.get("attention_mask")
        if not isinstance(input_ids, Tensor) or not isinstance(mask, Tensor):
            raise ValueError("tokenized batches require ids and masks")
        if input_ids.ndim != 2 or input_ids.shape != mask.shape:
            raise ValueError("tokenized ids and mask shapes disagree")
        boolean = mask.bool().detach().cpu()
        shapes.append(list(input_ids.shape))
        examples += input_ids.shape[0]
        for row in boolean:
            indices = row.nonzero(as_tuple=False).flatten()
            if not len(indices):
                raise ValueError("tokenized rows cannot be completely padded")
            start = int(indices[0])
            end = int(indices[-1]) + 1
            if not bool(row[start:end].all()):
                raise ValueError("packed or gapped token rows are unsupported")
            length = int(row.sum())
            valid_lengths.append(length)
            if length == row.numel():
                padding["none"] += 1
            elif start > 0 and end == row.numel():
                padding["left"] += 1
                padded_rows += 1
            elif start == 0 and end < row.numel():
                padding["right"] += 1
                padded_rows += 1
            else:
                raise ValueError("two-sided padding is unsupported")
    ids_hash = _token_tensor_digest(batches, "input_ids")
    mask_hash = _token_tensor_digest(batches, "attention_mask")
    combined = hashlib.sha256(
        _TOKEN_HASH_DOMAIN
        + bytes.fromhex(ids_hash)
        + bytes.fromhex(mask_hash)
    ).hexdigest()
    return {
        "schema": (
            "fisher_graph.gemma3_full_body_residual_carrier."
            "tokenization_receipt.v1"
        ),
        "tokenizer_class": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "requested_batch_size": requested_batch_size,
        "batch_count": len(batches),
        "example_count": examples,
        "batch_shapes": shapes,
        "minimum_sequence_width": min(shape[1] for shape in shapes),
        "maximum_sequence_width": max(shape[1] for shape in shapes),
        "minimum_valid_tokens": min(valid_lengths),
        "maximum_valid_tokens": max(valid_lengths),
        "valid_token_count": sum(valid_lengths),
        "supervised_pair_count": sum(length - 1 for length in valid_lengths),
        "padded_row_count": padded_rows,
        "padding_row_counts": padding,
        "position_ids_supplied": any("position_ids" in batch for batch in batches),
        "input_ids_sha256": ids_hash,
        "attention_mask_sha256": mask_hash,
        "combined_sha256": combined,
        "contains_prompt_text": False,
    }


def _stack_manifest(stack: CompleteBlockGeneratorStackExecutor) -> dict[str, object]:
    for name in ("architecture_manifest", "graph_manifest"):
        method = getattr(stack, name, None)
        if callable(method):
            value = _json_value(method())
            if isinstance(value, dict):
                return value
    raise TypeError("complete body stack must expose an architecture manifest")


def _compile_complete_body(
    adapter: ModelAdapter,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> CompleteBlockGeneratorStackExecutor:
    """Clone exactly one direct-state block for every adapter layer."""

    if not adapter.layers:
        raise ValueError("adapter must expose at least one transformer layer")
    blocks: list[CompleteBlockResidualForm] = []
    layer_ids = tuple(layer.id for layer in adapter.layers)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_COMPILE_SEED)
        for layer in adapter.layers:
            executor = StructuredTransformerLayerExecutor(
                StructuredTransformerLayerExecutorConfig.from_layer_spec(layer),
                dtype=dtype,
                device=device,
            )
            executor.transplant_gemma3_layer_weights_(
                adapter.source_module(layer.id)
            )
            executor.eval().requires_grad_(False)
            block = CompleteBlockResidualForm(
                executor=executor,
                form=ResidualForm.DIRECT_OUTPUT,
            )
            block.eval().requires_grad_(False)
            blocks.append(block)
        stack = CompleteBlockGeneratorStackExecutor(
            layer_ids=layer_ids,
            blocks=blocks,
        )
    stack.eval().requires_grad_(False)
    if len(stack.blocks) != len(adapter.layers):
        raise RuntimeError("complete body stack omitted one or more layers")
    if any(block.form is not ResidualForm.DIRECT_OUTPUT for block in stack.blocks):
        raise RuntimeError("complete body stack must use direct-state blocks")
    return stack


def _execution_layer_outputs(execution: object, count: int) -> tuple[Tensor, ...]:
    values = getattr(execution, "layer_outputs", None)
    if not isinstance(values, (tuple, list)) or len(values) != count:
        raise TypeError("stack execution omitted complete layer outputs")
    if any(not isinstance(value, Tensor) for value in values):
        raise TypeError("stack layer outputs must be tensors")
    return tuple(values)


def _execution_output(execution: object) -> Tensor:
    output = getattr(execution, "output", None)
    if not isinstance(output, Tensor):
        raise TypeError("stack execution omitted its output tensor")
    return output


def _required_leaf_modules(
    stack: CompleteBlockGeneratorStackExecutor,
) -> dict[str, nn.Module]:
    result: dict[str, nn.Module] = {}
    for layer_id, block in zip(stack.layer_ids, stack.blocks, strict=True):
        executor = block.executor
        modules = dict(
            zip(
                _REQUIRED_LEAF_SUFFIXES,
                (
                    executor.attention_input_norm,
                    executor.attention.q_proj,
                    executor.attention.k_proj,
                    executor.attention.v_proj,
                    executor.attention.q_norm,
                    executor.attention.k_norm,
                    executor.attention.o_proj,
                    executor.attention_output_norm,
                    executor.feed_forward_input_norm,
                    executor.feed_forward.gate_proj,
                    executor.feed_forward.up_proj,
                    executor.feed_forward.down_proj,
                    executor.feed_forward_output_norm,
                ),
                strict=True,
            )
        )
        for name, module in modules.items():
            result[f"{layer_id}.{name}"] = module
    return result


def _numeric_accounting(value: object) -> dict[str, int]:
    if is_dataclass(value) and not isinstance(value, type):
        raw: object = asdict(value)
    elif isinstance(value, Mapping):
        raw = value
    else:
        raw = {
            name: getattr(value, name)
            for name in dir(value)
            if not name.startswith("_")
            and isinstance(getattr(value, name, None), int)
        }
    if not isinstance(raw, Mapping):
        raise TypeError("stack accounting must be mapping-like")
    result = {
        str(name): int(item)
        for name, item in raw.items()
        if type(item) is int
    }
    total = getattr(value, "logical_total_macs", None)
    if type(total) is int:
        result["logical_total_macs"] = total
    if "logical_total_macs" not in result:
        candidates = (
            "attention_projection_macs",
            "attention_score_macs",
            "attention_value_macs",
            "feed_forward_macs",
        )
        if all(name in result for name in candidates):
            result["logical_total_macs"] = sum(result[name] for name in candidates)
    if "logical_total_macs" not in result:
        raise ValueError("stack accounting omitted logical_total_macs")
    return result


def _sum_accounting(total: dict[str, int], value: Mapping[str, int]) -> None:
    invariant_fields = {
        "source_parameter_count",
        "candidate_parameter_count",
        "removed_parameter_count",
    }
    for name, count in value.items():
        if name in invariant_fields:
            if name in total and total[name] != int(count):
                raise RuntimeError("stack parameter accounting changed by batch")
            total[name] = int(count)
        else:
            total[name] = total.get(name, 0) + int(count)


def _evaluate_complete_body(
    adapter: ModelAdapter,
    batches: Sequence[Mapping[str, Tensor]],
    stack: CompleteBlockGeneratorStackExecutor,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, int]]:
    if not batches:
        raise ValueError("full-body evaluation requires tokenized batches")
    layer_ids = tuple(layer.id for layer in adapter.layers)
    capture_sites = tuple(layer.output_site for layer in adapter.layers)
    native_calls = {layer_id: 0 for layer_id in layer_ids}
    compiled_calls = {layer_id: 0 for layer_id in layer_ids}
    phase = "idle"
    handles: list[Any] = []
    leaf_modules = _required_leaf_modules(stack)
    compiled_leaf_calls = {name: 0 for name in leaf_modules}

    def count_source(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
        *,
        layer_id: str,
    ) -> None:
        target = native_calls if phase == "native" else compiled_calls
        if phase not in ("native", "compiled"):
            raise RuntimeError("source block executed outside an audited phase")
        target[layer_id] += 1

    for layer in adapter.layers:
        handles.append(
            adapter.source_module(layer.id).register_forward_hook(
                lambda module, args, output, *, layer_id=layer.id: count_source(
                    module,
                    args,
                    output,
                    layer_id=layer_id,
                )
            )
        )
    for leaf_name, module in leaf_modules.items():

        def count_leaf(
            _module: nn.Module,
            _args: tuple[object, ...],
            _output: object,
            *,
            name: str = leaf_name,
        ) -> None:
            compiled_leaf_calls[name] += 1

        handles.append(module.register_forward_hook(count_leaf))

    logits = _LogitMetricAccumulator()
    stages = {layer_id: _TensorMetricAccumulator() for layer_id in layer_ids}
    native_nll_sum = 0.0
    native_tokens = 0
    stack_calls = 0
    carrier_versions: list[int] = []
    mutation_counts: list[int] = []
    accounting: dict[str, int] = {}
    try:
        for batch in batches:
            phase = "native"
            with torch.inference_mode():
                native = adapter.forward(batch, capture_sites=capture_sites)
            nll_sum, tokens = _native_nll_sum(native.logits, batch)
            native_nll_sum += nll_sum
            native_tokens += tokens

            phase = "compiled"
            with torch.inference_mode():
                sequence = adapter.prepare_sequence(batch)
                initial = adapter.embed(batch, sequence).hidden_states
                execution = stack.forward_components(
                    initial,
                    sequence,
                    capture_layer_outputs=True,
                )
                stack_calls += 1
                candidate_output = _execution_output(execution)
                layer_outputs = _execution_layer_outputs(
                    execution,
                    len(layer_ids),
                )
                candidate_logits = adapter.project_logits(
                    candidate_output,
                    sequence,
                )
                batch_accounting = _numeric_accounting(
                    stack.logical_accounting(sequence)
                )
            phase = "idle"
            if not torch.equal(candidate_output, layer_outputs[-1]):
                raise RuntimeError("stack output disagrees with its final layer")
            version = getattr(execution, "carrier_version", None)
            receipts = getattr(execution, "mutation_receipts", None)
            expected_mutations = 2 * len(layer_ids)
            if type(version) is not int or version != expected_mutations:
                raise RuntimeError("residual carrier did not complete its plan")
            if not isinstance(receipts, (tuple, list)) or len(receipts) != version:
                raise RuntimeError("residual carrier receipts are incomplete")
            carrier_versions.append(version)
            mutation_counts.append(len(receipts))
            _sum_accounting(accounting, batch_accounting)

            valid = sequence.query_valid_mask
            logits.update(native.logits, candidate_logits, batch)
            for layer, candidate in zip(
                adapter.layers,
                layer_outputs,
                strict=True,
            ):
                stages[layer.id].update(
                    native.activations[layer.output_site],
                    candidate,
                    valid,
                )
    finally:
        phase = "idle"
        for handle in reversed(handles):
            handle.remove()
    if native_tokens <= 0:
        raise RuntimeError("full-body evaluation produced no supervised tokens")
    expected_calls = len(batches)
    if any(count != expected_calls for count in native_calls.values()):
        raise RuntimeError("native reference did not execute every source block")
    if any(count != 0 for count in compiled_calls.values()):
        raise RuntimeError("compiled body called a native source block")
    if any(count != expected_calls for count in compiled_leaf_calls.values()):
        raise RuntimeError("compiled body did not execute every cloned leaf")
    native_nll = native_nll_sum / native_tokens
    metrics = {
        "native": {
            "supervised_tokens": native_tokens,
            "nll_per_token": native_nll,
        },
        "compiled_full_body": logits.result(native_nll_per_token=native_nll),
    }
    stage_metrics = {
        layer_id: accumulator.result()
        for layer_id, accumulator in stages.items()
    }
    audit = {
        "batch_count": len(batches),
        "native_source_block_calls": native_calls,
        "compiled_source_block_calls": compiled_calls,
        "expected_native_calls_per_block": expected_calls,
        "expected_compiled_calls_per_block": 0,
        "stack_execution_calls": stack_calls,
        "expected_stack_execution_calls": expected_calls,
        "compiled_required_leaf_calls": compiled_leaf_calls,
        "expected_compiled_leaf_calls": {
            name: expected_calls for name in compiled_leaf_calls
        },
        "carrier_versions": carrier_versions,
        "mutation_receipt_counts": mutation_counts,
        "expected_mutations_per_forward": 2 * len(layer_ids),
        "all_native_blocks_bypassed_by_compiled_path": True,
        "passed": True,
    }
    return metrics, stage_metrics, audit, accounting


def _exactness_gates(
    metrics: Mapping[str, object],
    stage_metrics: Mapping[str, object],
    *,
    stage_atol: float,
    logit_atol: float,
    nll_atol: float,
    kl_max: float,
) -> dict[str, object]:
    candidate = metrics.get("compiled_full_body")
    if not isinstance(candidate, Mapping) or not stage_metrics:
        raise ValueError("full-body exactness inputs are incomplete")
    layer_gates = {
        str(layer_id): (
            isinstance(value, Mapping)
            and float(value["maximum_absolute_error"]) <= stage_atol
        )
        for layer_id, value in stage_metrics.items()
    }
    checks = {
        "all_layer_boundaries_within_tolerance": all(layer_gates.values()),
        "maximum_logit_error_within_tolerance": (
            float(candidate["maximum_absolute_logit_error"]) <= logit_atol
        ),
        "nll_delta_within_tolerance": (
            abs(float(candidate["delta_nll_per_token"])) <= nll_atol
        ),
        "kl_within_tolerance": (
            float(candidate["native_to_candidate_kl_per_query"]) <= kl_max
        ),
        "top1_agreement_exact": (
            float(candidate["top1_agreement_to_native"]) == 1.0
        ),
    }
    return {
        "thresholds": {
            "stage_atol": stage_atol,
            "logit_atol": logit_atol,
            "nll_atol": nll_atol,
            "kl_max": kl_max,
        },
        "layer_boundary_gates": layer_gates,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _resource_report(
    *,
    source_model_parameters: int,
    source_model_stored_scalars: int,
    source_body_parameters: int,
    stack: CompleteBlockGeneratorStackExecutor,
    logical_accounting: Mapping[str, int],
    evaluated_valid_tokens: int,
    batch_count: int,
) -> dict[str, object]:
    candidate_body_parameters = sum(
        parameter.numel() for parameter in stack.parameters()
    )
    candidate_body_stored = sum(
        value.numel() for value in (*stack.parameters(), *stack.buffers())
    )
    if candidate_body_parameters != source_body_parameters:
        raise RuntimeError("exact compiled body changed the body parameter count")
    external = source_model_parameters - source_body_parameters
    candidate_deployment = external + candidate_body_parameters
    if external < 0 or candidate_deployment != source_model_parameters:
        raise RuntimeError("full-body parameter ledger did not close")
    layer_count = len(stack.blocks)
    return {
        "source_whole_model_parameter_count": source_model_parameters,
        "source_whole_model_parameter_and_buffer_scalars": (
            source_model_stored_scalars
        ),
        "source_transformer_body_parameter_count": source_body_parameters,
        "native_external_boundary_parameter_count": external,
        "transformer_body_fraction_of_whole_model": (
            source_body_parameters / source_model_parameters
        ),
        "maximum_whole_model_parameter_reduction_from_body_only": (
            source_body_parameters / source_model_parameters
        ),
        "compiled_body_learned_parameter_count": candidate_body_parameters,
        "compiled_body_stored_parameter_and_buffer_scalars": (
            candidate_body_stored
        ),
        "candidate_deployment_parameter_count": candidate_deployment,
        "physical_resident_experiment_scalars": (
            source_model_stored_scalars + candidate_body_stored
        ),
        "target_body_parameter_reduction_fraction": 0.0,
        "whole_model_parameter_reduction_fraction": 0.0,
        "target_body_logical_mac_reduction_fraction": 0.0,
        "logical_transformer_body_macs": int(
            logical_accounting["logical_total_macs"]
        ),
        "logical_residual_mutation_scalar_additions": (
            2 * layer_count * evaluated_valid_tokens * stack.width
        ),
        "carrier_mutations": 2 * layer_count * batch_count,
        "logical_accounting": dict(logical_accounting),
        "accounting_excludes": [
            "native_embedding_and_tied_lm_head_macs",
            "normalization",
            "activation",
            "masking",
            "softmax",
            "rope",
        ],
        "native_embedding_final_norm_and_head_retained": True,
        "contains_cloned_source_checkpoint_tensors": True,
        "compression_attempted": False,
        "source_free_artifact_established": False,
        "latency_measured": False,
        "kernel_speedup_claimed": False,
    }


def _library_versions() -> dict[str, str]:
    versions = {"torch": torch.__version__}
    try:
        versions["transformers"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        versions["transformers"] = "unavailable"
    return versions


def _validate_runner_arguments(
    *,
    revision: str,
    model_id: str,
    device_name: str,
    dtype: str,
    batch_size: int,
    output: Path | str,
) -> Path:
    if revision != DEFAULT_REVISION or model_id != DEFAULT_MODEL_ID:
        raise ValueError("full-body run requires the pinned Gemma checkpoint")
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("full-body run requires CPU float32")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("full-body output must use .json")
    if destination.exists():
        raise FileExistsError("refusing to overwrite a full-body report")
    return destination


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    if report.get("schema") != FULL_BODY_RESIDUAL_CARRIER_EXPERIMENT_SCHEMA:
        raise ValueError("unsupported full-body report schema")
    if report.get("format_version") != 1:
        raise ValueError("unsupported full-body report format")
    for field in (
        "contains_model_weights",
        "contains_executor_weights",
        "contains_prompt_text",
    ):
        if report.get(field) is not False:
            raise ValueError(f"full-body report {field} must be false")
    protocol = report.get("protocol")
    metrics = report.get("metrics")
    stages = report.get("stage_metrics")
    gates = report.get("exactness_gates")
    audit = report.get("execution_audit")
    resources = report.get("resources")
    restoration = report.get("source_restoration")
    independence = report.get("independence_audit")
    if not all(
        isinstance(value, Mapping)
        for value in (
            protocol,
            metrics,
            stages,
            gates,
            audit,
            resources,
            restoration,
            independence,
        )
    ):
        raise ValueError("full-body report sections are incomplete")
    assert isinstance(protocol, Mapping)
    assert isinstance(metrics, Mapping)
    assert isinstance(stages, Mapping)
    assert isinstance(gates, Mapping)
    assert isinstance(audit, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(restoration, Mapping)
    assert isinstance(independence, Mapping)
    thresholds = gates.get("thresholds")
    expected_thresholds = {
        "stage_atol": DEFAULT_STAGE_ATOL,
        "logit_atol": DEFAULT_LOGIT_ATOL,
        "nll_atol": DEFAULT_NLL_ATOL,
        "kl_max": DEFAULT_KL_MAX,
    }
    if not isinstance(thresholds, Mapping) or thresholds != expected_thresholds:
        raise ValueError("full-body thresholds are missing")
    recomputed = _exactness_gates(
        metrics,
        stages,
        stage_atol=float(thresholds["stage_atol"]),
        logit_atol=float(thresholds["logit_atol"]),
        nll_atol=float(thresholds["nll_atol"]),
        kl_max=float(thresholds["kl_max"]),
    )
    if gates != recomputed or gates.get("passed") is not True:
        raise ValueError("full-body exactness gates are invalid")
    batch_count = protocol.get("evaluation_batch_count")
    layer_count = protocol.get("target_layer_count")
    if type(batch_count) is not int or batch_count <= 0:
        raise ValueError("full-body evaluation batch count is invalid")
    if (
        type(layer_count) is not int
        or layer_count != _EXPECTED_PINNED_LAYER_COUNT
    ):
        raise ValueError("full-body layer count is invalid")
    expected_ids = [f"layer.{index}" for index in range(layer_count)]
    expected_leaf_calls = {
        f"{layer_id}.{suffix}": batch_count
        for layer_id in expected_ids
        for suffix in _REQUIRED_LEAF_SUFFIXES
    }
    if protocol.get("target_layer_ids") != expected_ids:
        raise ValueError("full-body layers are not complete and contiguous")
    native_calls = audit.get("native_source_block_calls")
    compiled_calls = audit.get("compiled_source_block_calls")
    if (
        not isinstance(native_calls, Mapping)
        or not isinstance(compiled_calls, Mapping)
        or dict(native_calls) != {name: batch_count for name in expected_ids}
        or dict(compiled_calls) != {name: 0 for name in expected_ids}
        or audit.get("stack_execution_calls") != batch_count
        or audit.get("compiled_required_leaf_calls") != expected_leaf_calls
        or audit.get("expected_compiled_leaf_calls") != expected_leaf_calls
        or audit.get("expected_mutations_per_forward") != 2 * layer_count
        or audit.get("carrier_versions") != [2 * layer_count] * batch_count
        or audit.get("mutation_receipt_counts")
        != [2 * layer_count] * batch_count
        or audit.get("passed") is not True
    ):
        raise ValueError("full-body execution audit is invalid")
    tokenization = protocol.get("tokenization")
    if (
        not isinstance(tokenization, Mapping)
        or tokenization.get("contains_prompt_text") is not False
        or tokenization.get("batch_count") != batch_count
        or not _is_sha256(tokenization.get("combined_sha256"))
    ):
        raise ValueError("full-body tokenization receipt is invalid")
    padding_counts = tokenization.get("padding_row_counts")
    example_count = tokenization.get("example_count")
    batch_shapes = tokenization.get("batch_shapes")
    requested_batch_size = tokenization.get("requested_batch_size")
    input_ids_hash = tokenization.get("input_ids_sha256")
    attention_mask_hash = tokenization.get("attention_mask_sha256")
    if (
        type(example_count) is not int
        or example_count <= 0
        or protocol.get("evaluation_prompt_count") != example_count
        or tokenization.get("schema")
        != (
            "fisher_graph.gemma3_full_body_residual_carrier."
            "tokenization_receipt.v1"
        )
        or tokenization.get("contains_prompt_text") is not False
        or tokenization.get("position_ids_supplied") is not False
        or tokenization.get("padding_side") != "left"
        or tokenization.get("pad_token_id") != 0
        or type(requested_batch_size) is not int
        or requested_batch_size <= 0
        or not isinstance(batch_shapes, list)
        or len(batch_shapes) != batch_count
        or any(
            not isinstance(shape, list)
            or len(shape) != 2
            or type(shape[0]) is not int
            or type(shape[1]) is not int
            or not 0 < shape[0] <= requested_batch_size
            or shape[1] <= 0
            for shape in batch_shapes
        )
        or sum(shape[0] for shape in batch_shapes) != example_count
        or tokenization.get("minimum_sequence_width")
        != min(shape[1] for shape in batch_shapes)
        or tokenization.get("maximum_sequence_width")
        != max(shape[1] for shape in batch_shapes)
        or not isinstance(padding_counts, Mapping)
        or set(padding_counts) != {"left", "right", "none"}
        or any(type(value) is not int or value < 0 for value in padding_counts.values())
        or sum(padding_counts.values()) != example_count
        or tokenization.get("padded_row_count")
        != padding_counts["left"] + padding_counts["right"]
        or type(tokenization.get("valid_token_count")) is not int
        or type(tokenization.get("supervised_pair_count")) is not int
        or not _is_sha256(input_ids_hash)
        or not _is_sha256(attention_mask_hash)
    ):
        raise ValueError("full-body token counts are invalid")
    assert isinstance(input_ids_hash, str)
    assert isinstance(attention_mask_hash, str)
    expected_combined_hash = hashlib.sha256(
        _TOKEN_HASH_DOMAIN
        + bytes.fromhex(input_ids_hash)
        + bytes.fromhex(attention_mask_hash)
    ).hexdigest()
    if tokenization.get("combined_sha256") != expected_combined_hash:
        raise ValueError("full-body combined token hash is invalid")
    native_metrics = metrics.get("native")
    candidate_metrics = metrics.get("compiled_full_body")
    if (
        not isinstance(native_metrics, Mapping)
        or not isinstance(candidate_metrics, Mapping)
        or native_metrics.get("supervised_tokens")
        != tokenization.get("supervised_pair_count")
        or candidate_metrics.get("supervised_tokens")
        != native_metrics.get("supervised_tokens")
        or candidate_metrics.get("compared_query_rows")
        != tokenization.get("valid_token_count")
        or set(stages) != set(expected_ids)
    ):
        raise ValueError("full-body metric token scope is invalid")
    if (
        restoration.get("passed") is not True
        or restoration.get("model_fingerprint_before")
        != restoration.get("model_fingerprint_after")
        or restoration.get("execution_fingerprint_before")
        != restoration.get("execution_fingerprint_after")
        or independence.get("passed") is not True
    ):
        raise ValueError("full-body source integrity audit is invalid")
    if (
        restoration.get("parameter_count_before")
        != restoration.get("parameter_count_after")
    ):
        raise ValueError("full-body source parameter restoration is invalid")
    if (
        resources.get("target_body_parameter_reduction_fraction") != 0.0
        or resources.get("whole_model_parameter_reduction_fraction") != 0.0
        or resources.get("target_body_logical_mac_reduction_fraction") != 0.0
        or resources.get("compression_attempted") is not False
        or resources.get("source_free_artifact_established") is not False
        or resources.get("latency_measured") is not False
        or resources.get("kernel_speedup_claimed") is not False
        or resources.get("native_embedding_final_norm_and_head_retained")
        is not True
    ):
        raise ValueError("full-body resource claims are invalid")
    source_parameters = resources.get("source_whole_model_parameter_count")
    source_stored = resources.get(
        "source_whole_model_parameter_and_buffer_scalars"
    )
    source_body = resources.get("source_transformer_body_parameter_count")
    external = resources.get("native_external_boundary_parameter_count")
    candidate_body = resources.get("compiled_body_learned_parameter_count")
    candidate_stored = resources.get(
        "compiled_body_stored_parameter_and_buffer_scalars"
    )
    model = report.get("model")
    compiled_body = report.get("compiled_body")
    claims = report.get("claims")
    status = report.get("scientific_status")
    prompt_split = protocol.get("prompt_split")
    runtime = protocol.get("runtime")
    manifest = (
        compiled_body.get("manifest")
        if isinstance(compiled_body, Mapping)
        else None
    )
    logical = resources.get("logical_accounting")
    independence_candidates = independence.get("candidates")
    manifest_blocks = (
        manifest.get("blocks") if isinstance(manifest, Mapping) else None
    )
    manifest_width = (
        _mapping_path(
            manifest_blocks[0],
            "leaf_executor",
            "config",
            "transformer",
            "attention_input_norm",
            "width",
        )
        if isinstance(manifest_blocks, list) and manifest_blocks
        else None
    )
    expected_status = {
        "scope": "pinned_gemma_full_transformer_body_exact_carrier",
        "development_only": True,
        "all_native_transformer_blocks_bypassed": True,
        "native_embedding_final_norm_and_tied_head_retained": True,
        "fitting_performed": False,
        "source_free_compilation_established": False,
        "compression_attempted": False,
        "latency_or_kernel_speedup_claimed": False,
        "decode_or_cache_supported": False,
        "invalid_padding_query_parity_claimed": False,
    }
    expected_claims = {
        "all_18_transformer_blocks_compiled": layer_count == 18,
        "full_transformer_body_not_literal_whole_model": True,
        "native_boundary_modules_retained": True,
        "residual_information_preserved_and_mutated_on_graph_wire": True,
        "native_block_runtime_calls_eliminated": True,
        "parameter_compression": False,
        "mac_compression": False,
        "source_free_artifact": False,
        "latency_or_speedup": False,
    }
    if (
        not isinstance(model, Mapping)
        or not isinstance(compiled_body, Mapping)
        or not isinstance(claims, Mapping)
        or not isinstance(status, Mapping)
        or not isinstance(prompt_split, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(manifest, Mapping)
        or not isinstance(logical, Mapping)
        or not isinstance(independence_candidates, Mapping)
        or dict(status) != expected_status
        or dict(claims) != expected_claims
        or not isinstance(manifest_blocks, list)
        or len(manifest_blocks) != layer_count
        or model.get("num_hidden_layers") != layer_count
        or model.get("model_id") != DEFAULT_MODEL_ID
        or model.get("requested_revision") != DEFAULT_REVISION
        or model.get("resolved_commit") != DEFAULT_REVISION
        or model.get("parameter_count") != source_parameters
        or model.get("hidden_size") != manifest_width
        or prompt_split.get("prompt_disjoint") is not True
        or prompt_split.get("family_disjoint") is not False
        or prompt_split.get("evaluation_example_count") != example_count
        or protocol.get("fit_prompt_count_in_panel")
        != prompt_split.get("fit_example_count")
        or protocol.get("family_disjoint_guard") is not False
        or protocol.get("fit_prompts_used_for_fitting") != 0
        or protocol.get("expected_carrier_mutations_per_forward")
        != 2 * layer_count
        or protocol.get("evidence_status")
        != "prompt_disjoint_development_only"
        or runtime
        != {
            "device": "cpu",
            "dtype": "float32",
            "phase": "prefill",
            "attention_implementation": "eager",
            "use_cache": False,
            "cache_positions_supported": False,
            "valid_query_rows_only": True,
            "invalid_padding_rows_may_differ": True,
        }
        or type(source_parameters) is not int
        or type(source_stored) is not int
        or type(source_body) is not int
        or type(external) is not int
        or type(candidate_body) is not int
        or type(candidate_stored) is not int
        or source_parameters <= 0
        or source_stored < source_parameters
        or source_body <= 0
        or external != source_parameters - source_body
        or candidate_body != source_body
        or resources.get("candidate_deployment_parameter_count")
        != source_parameters
        or resources.get("physical_resident_experiment_scalars")
        != source_stored + candidate_stored
        or resources.get("transformer_body_fraction_of_whole_model")
        != source_body / source_parameters
        or resources.get(
            "maximum_whole_model_parameter_reduction_from_body_only"
        )
        != source_body / source_parameters
        or resources.get("carrier_mutations")
        != 2 * layer_count * batch_count
        or resources.get("logical_residual_mutation_scalar_additions")
        != 2
        * layer_count
        * int(candidate_metrics["compared_query_rows"])
        * int(model["hidden_size"])
        or resources.get("logical_transformer_body_macs")
        != logical.get("logical_total_macs")
        or logical.get("valid_tokens")
        != candidate_metrics.get("compared_query_rows")
        or logical.get("source_parameter_count") != source_body
        or logical.get("candidate_parameter_count") != source_body
        or logical.get("removed_parameter_count") != 0
        or logical.get("source_logical_macs")
        != logical.get("logical_total_macs")
        or logical.get("candidate_logical_macs")
        != logical.get("logical_total_macs")
        or logical.get("removed_logical_macs") != 0
        or compiled_body.get("contains_cloned_source_checkpoint_tensors")
        is not True
        or compiled_body.get("owns_live_source_module_or_fallback") is not False
        or compiled_body.get("residual_wire_representation") != "dense_exact"
        or not _is_sha256(compiled_body.get("execution_fingerprint_before"))
        or compiled_body.get("execution_fingerprint_before")
        != compiled_body.get("execution_fingerprint_after")
        or compiled_body.get("parameter_count_before") != source_body
        or compiled_body.get("parameter_count_after") != source_body
        or manifest.get("kind") != "complete_block_generator_stack"
        or manifest.get("layer_ids") != expected_ids
        or manifest.get("residual_forms")
        != [ResidualForm.DIRECT_OUTPUT.value] * layer_count
        or manifest.get("ordered_mutation_count") != 2 * layer_count
        or manifest.get("contains_source_model_weights") is not True
        or manifest.get("executor_local_source_free") is not False
        or manifest.get("contains_source_fallback") is not False
        or manifest.get("compression_attempted") is not False
        or manifest.get("parameter_reduction") is not False
        or manifest.get("logical_mac_reduction") is not False
        or manifest.get("physical_kernel_fusion_measured") is not False
        or independence_candidates.get("compiled_full_body")
        != {
            "parameter_object_alias_count": 0,
            "module_object_alias_count": 0,
            "tensor_storage_alias_count": 0,
            "passed": True,
        }
    ):
        raise ValueError("full-body resource closure is invalid")
    if (
        report.get("diagnostic_outcome")
        != "exact_full_transformer_body_carrier_authenticated"
    ):
        raise ValueError("full-body scientific outcome is invalid")


def load_gemma3_full_body_residual_carrier_report(
    path: Path | str,
) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("full-body report must be a JSON object")
    supplied = value.get("report_sha256")
    if not _is_sha256(supplied):
        raise ValueError("full-body report hash is invalid")
    unhashed = dict(value)
    del unhashed["report_sha256"]
    if supplied != _report_sha256(unhashed):
        raise ValueError("full-body report hash mismatch")
    _validate_report_semantics(value)
    return value


def run_gemma3_full_body_residual_carrier_experiment(
    *,
    revision: str = DEFAULT_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 4,
    stage_atol: float = DEFAULT_STAGE_ATOL,
    logit_atol: float = DEFAULT_LOGIT_ATOL,
    nll_atol: float = DEFAULT_NLL_ATOL,
    kl_max: float = DEFAULT_KL_MAX,
) -> dict[str, object]:
    destination = _validate_runner_arguments(
        revision=revision,
        model_id=model_id,
        device_name=device_name,
        dtype=dtype,
        batch_size=batch_size,
        output=output,
    )
    supplied_thresholds = {
        "stage_atol": stage_atol,
        "logit_atol": logit_atol,
        "nll_atol": nll_atol,
        "kl_max": kl_max,
    }
    pinned_thresholds = {
        "stage_atol": DEFAULT_STAGE_ATOL,
        "logit_atol": DEFAULT_LOGIT_ATOL,
        "nll_atol": DEFAULT_NLL_ATOL,
        "kl_max": DEFAULT_KL_MAX,
    }
    if supplied_thresholds != pinned_thresholds or any(
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        for value in supplied_thresholds.values()
    ):
        raise ValueError("full-body exactness thresholds are pinned")
    fit_prompts, evaluation_prompts, split = _load_prompt_split(Path(panel_path))
    if not split.get("prompt_disjoint") or not evaluation_prompts:
        raise RuntimeError("A5e prompt-disjoint evaluation split is unavailable")
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("load pinned local Gemma")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval().requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    if len(adapter.layers) != _EXPECTED_PINNED_LAYER_COUNT:
        raise RuntimeError(
            "pinned Gemma checkpoint does not expose the expected 18 blocks"
        )
    source_parameters = sum(parameter.numel() for parameter in model.parameters())
    source_stored = source_parameters + sum(
        buffer.numel() for buffer in model.buffers()
    )
    source_body_parameters = sum(
        parameter.numel()
        for layer in adapter.layers
        for parameter in adapter.source_module(layer.id).parameters()
    )
    model_fingerprint_before = adapter.model_fingerprint()
    execution_fingerprint_before = adapter.execution_fingerprint()
    guard = _FrozenModelTensorGuard(model)
    batches = _tokenize_batches(
        tokenizer,
        evaluation_prompts,
        batch_size=batch_size,
        device=device,
    )
    tokenization = _tokenization_receipt(
        tokenizer,
        batches,
        requested_batch_size=batch_size,
    )

    _progress(f"clone all {len(adapter.layers)} blocks into one carrier stack")
    stack = _compile_complete_body(
        adapter,
        dtype=torch.float32,
        device=device,
    )
    independence = _assert_source_independence(
        model,
        {"compiled_full_body": stack},
    )
    stack_manifest = _stack_manifest(stack)
    stack_fingerprint_before = stack.execution_fingerprint()
    stack_parameters_before = sum(
        parameter.numel() for parameter in stack.parameters()
    )

    _progress("evaluate native and compiled full transformer body")
    metrics, stages, call_audit, accounting = _evaluate_complete_body(
        adapter,
        batches,
        stack,
    )
    gates = _exactness_gates(
        metrics,
        stages,
        **pinned_thresholds,
    )
    if not gates["passed"]:
        raise RuntimeError("compiled full body failed native parity gates")
    stack_fingerprint_after = stack.execution_fingerprint()
    if (
        stack_fingerprint_after != stack_fingerprint_before
        or sum(parameter.numel() for parameter in stack.parameters())
        != stack_parameters_before
    ):
        raise RuntimeError("compiled body changed during evaluation")

    guard.assert_unchanged()
    model_fingerprint_after = adapter.model_fingerprint()
    execution_fingerprint_after = adapter.execution_fingerprint()
    if (
        model_fingerprint_after != model_fingerprint_before
        or execution_fingerprint_after != execution_fingerprint_before
        or sum(parameter.numel() for parameter in model.parameters())
        != source_parameters
    ):
        raise RuntimeError("source model changed during full-body evaluation")
    candidate_metrics = metrics["compiled_full_body"]
    assert isinstance(candidate_metrics, Mapping)
    resources = _resource_report(
        source_model_parameters=source_parameters,
        source_model_stored_scalars=source_stored,
        source_body_parameters=source_body_parameters,
        stack=stack,
        logical_accounting=accounting,
        evaluated_valid_tokens=int(candidate_metrics["compared_query_rows"]),
        batch_count=len(batches),
    )
    layer_ids = [layer.id for layer in adapter.layers]
    report: dict[str, object] = {
        "schema": FULL_BODY_RESIDUAL_CARRIER_EXPERIMENT_SCHEMA,
        "format_version": FULL_BODY_RESIDUAL_CARRIER_EXPERIMENT_FORMAT_VERSION,
        "diagnostic_outcome": "exact_full_transformer_body_carrier_authenticated",
        "contains_model_weights": False,
        "contains_executor_weights": False,
        "contains_prompt_text": False,
        "scientific_status": {
            "scope": "pinned_gemma_full_transformer_body_exact_carrier",
            "development_only": True,
            "all_native_transformer_blocks_bypassed": True,
            "native_embedding_final_norm_and_tied_head_retained": True,
            "fitting_performed": False,
            "source_free_compilation_established": False,
            "compression_attempted": False,
            "latency_or_kernel_speedup_claimed": False,
            "decode_or_cache_supported": False,
            "invalid_padding_query_parity_claimed": False,
        },
        "model": {
            **_model_provenance(
                model,
                model_id=model_id,
                requested_revision=revision,
            ),
            "adapter_model_fingerprint": model_fingerprint_before,
            "adapter_execution_fingerprint": execution_fingerprint_before,
        },
        "protocol": {
            "name": "gemma3_full_body_residual_carrier_a5e_v1",
            "runtime": {
                "device": "cpu",
                "dtype": "float32",
                "phase": "prefill",
                "attention_implementation": "eager",
                "use_cache": False,
                "cache_positions_supported": False,
                "valid_query_rows_only": True,
                "invalid_padding_rows_may_differ": True,
            },
            "prompt_split": split,
            "evidence_status": "prompt_disjoint_development_only",
            "family_disjoint_guard": False,
            "fit_prompt_count_in_panel": len(fit_prompts),
            "fit_prompts_used_for_fitting": 0,
            "evaluation_prompt_count": len(evaluation_prompts),
            "evaluation_batch_count": len(batches),
            "target_layer_ids": layer_ids,
            "target_layer_count": len(layer_ids),
            "expected_carrier_mutations_per_forward": 2 * len(layer_ids),
            "tokenization": tokenization,
            "libraries": _library_versions(),
        },
        "compiled_body": {
            "manifest": stack_manifest,
            "execution_fingerprint_before": stack_fingerprint_before,
            "execution_fingerprint_after": stack_fingerprint_after,
            "parameter_count_before": stack_parameters_before,
            "parameter_count_after": sum(
                parameter.numel() for parameter in stack.parameters()
            ),
            "contains_cloned_source_checkpoint_tensors": True,
            "owns_live_source_module_or_fallback": False,
            "residual_wire_representation": "dense_exact",
        },
        "metrics": metrics,
        "stage_metrics": stages,
        "exactness_gates": gates,
        "execution_audit": call_audit,
        "independence_audit": independence,
        "source_restoration": {
            "tensor_guard": guard.metadata(),
            "model_fingerprint_before": model_fingerprint_before,
            "model_fingerprint_after": model_fingerprint_after,
            "execution_fingerprint_before": execution_fingerprint_before,
            "execution_fingerprint_after": execution_fingerprint_after,
            "parameter_count_before": source_parameters,
            "parameter_count_after": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "passed": True,
        },
        "resources": resources,
        "claims": {
            "all_18_transformer_blocks_compiled": len(layer_ids) == 18,
            "full_transformer_body_not_literal_whole_model": True,
            "native_boundary_modules_retained": True,
            "residual_information_preserved_and_mutated_on_graph_wire": True,
            "native_block_runtime_calls_eliminated": True,
            "parameter_compression": False,
            "mac_compression": False,
            "source_free_artifact": False,
            "latency_or_speedup": False,
        },
    }
    converted = _json_value(report)
    assert isinstance(converted, dict)
    _validate_report_semantics(converted)
    converted["report_sha256"] = _report_sha256(converted)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(converted, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError("refusing to overwrite a full-body report") from error
    loaded = load_gemma3_full_body_residual_carrier_report(destination)
    if loaded != converted:
        raise RuntimeError("full-body report changed across strict reload")
    return loaded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_gemma3_full_body_residual_carrier_experiment(
        revision=args.revision,
        output=args.output,
        panel_path=args.panel_path,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
