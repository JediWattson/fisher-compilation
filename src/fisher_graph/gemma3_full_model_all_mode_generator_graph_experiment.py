"""Full-model Gemma compilation into exact, affine, and chart mode graphs.

This development runner replaces every transformer MLP at once.  It preserves
all native mode nodes and compares an exact source-free graph with the global
affine and hidden-state-conditioned chart generators introduced by A5e.  No
arm deletes or merges a mode, and no held row may refit a graph.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
    CHART_COUNT,
    CHART_EDGE_RIDGE,
    CHART_RANK,
    CHART_TEMPERATURE,
    DEFAULT_PANEL_PATH,
    DEFAULT_REVISION,
    Gemma3AllModeGeneratorGraphMLP,
    _MetricAccumulator,
    _capture_fisher_rows,
    _fit_affine_all_mode_graph,
    _fit_chart_conditioned_all_mode_graph,
    _load_prompt_split,
    _materialize_all_modes,
    _native_nll,
    _relative_rmse,
    _temporary_mlp_overlay,
    _tokenize_batches,
)


FULL_MODEL_EXPERIMENT_SCHEMA = (
    "fisher_graph.gemma3_full_model_all_mode_generator_graph.experiment.v1"
)
FULL_MODEL_EXPERIMENT_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "full-model-all-mode-generator-graph-dev-v1.json"
)
_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-full-model-all-mode-generator-graph-report:v1\0"
)


__all__ = [
    "DEFAULT_OUTPUT",
    "FULL_MODEL_EXPERIMENT_FORMAT_VERSION",
    "FULL_MODEL_EXPERIMENT_SCHEMA",
    "run_gemma3_full_model_all_mode_generator_graph",
]


def _progress(message: str) -> None:
    print(f"[full-model-mode-graph] {message}", flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        _REPORT_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _full_model_layer_ordinals(layer_count: int) -> tuple[int, ...]:
    if type(layer_count) is not int or layer_count <= 0:
        raise ValueError("full-model layer count must be positive")
    return tuple(range(layer_count))


def _graph_resource_ledger(
    graphs: Mapping[int, Gemma3AllModeGeneratorGraphMLP],
    *,
    source_whole_model_parameters: int,
) -> dict[str, object]:
    if not graphs or source_whole_model_parameters <= 0:
        raise ValueError("full-model resource inputs are invalid")
    native_target_scalars = sum(
        3 * graph.hidden_width * graph.mode_count
        for graph in graphs.values()
    )
    stored_scalars = sum(
        graph.stored_coefficient_count for graph in graphs.values()
    )
    learned_parameters = sum(
        graph.learned_parameter_count for graph in graphs.values()
    )
    native_target_macs = native_target_scalars
    estimated_macs = sum(
        graph.estimated_total_macs_per_token for graph in graphs.values()
    )
    whole_model_stored = (
        source_whole_model_parameters
        - native_target_scalars
        + stored_scalars
    )
    return {
        "target_layer_count": len(graphs),
        "native_target_mlp_parameter_scalars": native_target_scalars,
        "compiled_learned_parameters": learned_parameters,
        "compiled_stored_coefficient_scalars": stored_scalars,
        "target_mlp_stored_scalar_reduction_fraction": (
            (native_target_scalars - stored_scalars)
            / native_target_scalars
        ),
        "candidate_whole_model_stored_scalar_estimate": whole_model_stored,
        "whole_model_stored_scalar_reduction_fraction": (
            (source_whole_model_parameters - whole_model_stored)
            / source_whole_model_parameters
        ),
        "native_target_mlp_matrix_macs_per_token": native_target_macs,
        "compiled_estimated_macs_per_token": estimated_macs,
        "target_mlp_estimated_mac_reduction_fraction": (
            (native_target_macs - estimated_macs) / native_target_macs
        ),
        "graph_metadata_by_layer": {
            str(ordinal): graph.graph_metadata()
            for ordinal, graph in graphs.items()
        },
    }


def _runtime_modules(
    graph: Gemma3AllModeGeneratorGraphMLP,
) -> dict[str, nn.Module]:
    result: dict[str, nn.Module] = {
        "graph_forward": graph,
        "mode_decoder": graph.mode_decoder,
    }
    if graph.graph_kind == "exact_swiglu":
        if graph.gate_generators is None or graph.up_generators is None:
            raise RuntimeError("exact graph runtime is incomplete")
        result["gate_generators"] = graph.gate_generators
        result["up_generators"] = graph.up_generators
    else:
        if graph.affine_generators is None:
            raise RuntimeError("approximate graph runtime is incomplete")
        result["affine_generators"] = graph.affine_generators
    return result


def _evaluate_native(
    model: nn.Module,
    batches: Sequence[Mapping[str, Tensor]],
) -> tuple[float, int]:
    nll_sum = 0.0
    tokens = 0
    with torch.no_grad():
        for batch in batches:
            output = model(**batch, use_cache=False, return_dict=True)
            batch_nll, batch_tokens = _native_nll(output.logits, batch)
            nll_sum += batch_nll
            tokens += batch_tokens
            del output
    if tokens <= 0:
        raise RuntimeError("full-model native evaluation has no tokens")
    return nll_sum / tokens, tokens


def _evaluate_graph_arm(
    *,
    model: nn.Module,
    adapter: Gemma3CausalLMAdapter,
    source_mlps: Mapping[int, nn.Module],
    graphs: Mapping[int, Gemma3AllModeGeneratorGraphMLP],
    batches: Sequence[Mapping[str, Tensor]],
    native_nll: float,
) -> tuple[dict[str, float | int], dict[str, object]]:
    if set(source_mlps) != set(graphs):
        raise ValueError("graph arm does not cover every source layer")
    accumulator = _MetricAccumulator()
    source_calls = {
        ordinal: {name: 0 for name in ("gate_proj", "up_proj", "down_proj")}
        for ordinal in graphs
    }
    runtime_calls = {
        ordinal: {name: 0 for name in _runtime_modules(graph)}
        for ordinal, graph in graphs.items()
    }
    runtime_handles = []
    for ordinal, graph in graphs.items():
        for operation, module in _runtime_modules(graph).items():
            def count_runtime(
                _module: nn.Module,
                _args: tuple[Tensor, ...],
                _output: object,
                *,
                layer_ordinal: int = ordinal,
                operation_name: str = operation,
            ) -> None:
                runtime_calls[layer_ordinal][operation_name] += 1

            runtime_handles.append(module.register_forward_hook(count_runtime))
    try:
        with torch.no_grad():
            for batch in batches:
                native_output = model(
                    **batch,
                    use_cache=False,
                    return_dict=True,
                )
                source_handles = []
                for ordinal, source_mlp in source_mlps.items():
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        def count_source(
                            _module: nn.Module,
                            _args: tuple[Tensor, ...],
                            _output: object,
                            *,
                            layer_ordinal: int = ordinal,
                            projection_name: str = projection,
                        ) -> None:
                            source_calls[layer_ordinal][projection_name] += 1

                        source_handles.append(
                            getattr(source_mlp, projection).register_forward_hook(
                                count_source
                            )
                        )
                try:
                    with _temporary_mlp_overlay(adapter, graphs):
                        candidate_output = model(
                            **batch,
                            use_cache=False,
                            return_dict=True,
                        )
                finally:
                    for handle in source_handles:
                        handle.remove()
                accumulator.update(
                    native_output.logits,
                    candidate_output.logits,
                    batch,
                )
                del native_output, candidate_output
    finally:
        for handle in runtime_handles:
            handle.remove()
    expected_calls = len(batches)
    if any(
        count != 0
        for layer in source_calls.values()
        for count in layer.values()
    ):
        raise RuntimeError("compiled full-model arm called a native projection")
    if any(
        count != expected_calls
        for layer in runtime_calls.values()
        for count in layer.values()
    ):
        raise RuntimeError("compiled full-model arm skipped a graph operation")
    return accumulator.result(native_nll=native_nll), {
        "native_source_projection_call_counts": {
            str(ordinal): counts for ordinal, counts in source_calls.items()
        },
        "compiled_runtime_call_counts": {
            str(ordinal): counts for ordinal, counts in runtime_calls.items()
        },
    }


def _assert_source_restored(
    *,
    adapter: Gemma3CausalLMAdapter,
    source_parameters: int,
    model_fingerprint: str,
    execution_fingerprint: str,
) -> None:
    if (
        adapter.model_fingerprint() != model_fingerprint
        or adapter.execution_fingerprint() != execution_fingerprint
        or sum(parameter.numel() for parameter in adapter.module.parameters())
        != source_parameters
    ):
        raise RuntimeError("full-model graph overlay did not restore the source")


def run_gemma3_full_model_all_mode_generator_graph(
    *,
    revision: str,
    output: Path | str = DEFAULT_OUTPUT,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 4,
) -> dict[str, object]:
    """Compile and evaluate every Gemma MLP as an all-mode graph."""

    if revision != DEFAULT_REVISION or model_id != DEFAULT_MODEL_ID:
        raise ValueError("full-model run requires the pinned Gemma checkpoint")
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("full-model run requires deterministic CPU float32")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite a full-model experiment")
    fit_prompts, evaluation_prompts, split = _load_prompt_split(Path(panel_path))
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
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    layer_ordinals = _full_model_layer_ordinals(len(adapter.layers))
    source_mlps = {
        ordinal: adapter.source_module(adapter.layers[ordinal].id).mlp
        for ordinal in layer_ordinals
    }
    source_parameters = sum(parameter.numel() for parameter in model.parameters())
    source_fingerprint = adapter.model_fingerprint()
    execution_fingerprint = adapter.execution_fingerprint()
    fit_batches = _tokenize_batches(
        tokenizer,
        fit_prompts,
        batch_size=batch_size,
        device=device,
    )
    evaluation_batches = _tokenize_batches(
        tokenizer,
        evaluation_prompts,
        batch_size=batch_size,
        device=device,
    )

    capture_started = time.monotonic()
    _progress(f"capture fit Fisher rows for all {len(layer_ordinals)} layers")
    fit_capture = _capture_fisher_rows(
        adapter,
        fit_batches,
        target_layers=layer_ordinals,
    )
    _progress("capture prompt-disjoint native evaluation rows")
    evaluation_capture = _capture_fisher_rows(
        adapter,
        evaluation_batches,
        target_layers=layer_ordinals,
    )
    capture_seconds = time.monotonic() - capture_started
    capture_receipt = {
        "fit_rows_by_layer": {
            str(ordinal): capture.inputs.shape[0]
            for ordinal, capture in fit_capture.items()
        },
        "evaluation_rows_by_layer": {
            str(ordinal): capture.inputs.shape[0]
            for ordinal, capture in evaluation_capture.items()
        },
        "fit_example_count": split["fit_example_count"],
        "evaluation_example_count": split["evaluation_example_count"],
        "capture_elapsed_seconds": capture_seconds,
    }

    _progress("evaluate native reference")
    evaluation_started = time.monotonic()
    native_nll, native_tokens = _evaluate_native(model, evaluation_batches)

    _progress("compile exact all-mode graph for every layer")
    exact_graphs = {
        ordinal: _materialize_all_modes(
            source_mlps[ordinal],
            model_fingerprint=source_fingerprint,
            layer_ordinal=ordinal,
        )
        for ordinal in layer_ordinals
    }
    exact_resources = _graph_resource_ledger(
        exact_graphs,
        source_whole_model_parameters=source_parameters,
    )
    exact_metrics, exact_calls = _evaluate_graph_arm(
        model=model,
        adapter=adapter,
        source_mlps=source_mlps,
        graphs=exact_graphs,
        batches=evaluation_batches,
        native_nll=native_nll,
    )
    if (
        exact_metrics["delta_nll_per_token"] != 0.0
        or exact_metrics["native_to_candidate_kl_per_token"] != 0.0
        or exact_metrics["logit_nrmse"] != 0.0
        or exact_metrics["top1_agreement_to_native"] != 1.0
    ):
        raise RuntimeError("exact full-model graph failed native equivalence")
    _assert_source_restored(
        adapter=adapter,
        source_parameters=source_parameters,
        model_fingerprint=source_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )
    del exact_graphs
    gc.collect()

    affine_graphs: dict[int, Gemma3AllModeGeneratorGraphMLP] = {}
    chart_graphs: dict[int, Gemma3AllModeGeneratorGraphMLP] = {}
    affine_local_metrics: dict[str, object] = {}
    chart_local_metrics: dict[str, object] = {}
    fit_started = time.monotonic()
    for index, ordinal in enumerate(layer_ordinals, start=1):
        _progress(
            f"fit affine and chart graphs for layer {ordinal} "
            f"({index}/{len(layer_ordinals)})"
        )
        affine, affine_metrics = _fit_affine_all_mode_graph(
            source_mlps[ordinal],
            fit_capture[ordinal],
            model_fingerprint=source_fingerprint,
            layer_ordinal=ordinal,
            fit_split_sha256=str(split["fit_example_ids_sha256"]),
        )
        affine_graphs[ordinal] = affine
        with torch.no_grad():
            affine_states = affine.affine_generators(
                evaluation_capture[ordinal].inputs
            )
            affine_output = affine(evaluation_capture[ordinal].inputs)
            native_output = source_mlps[ordinal](
                evaluation_capture[ordinal].inputs
            )
        affine_local_metrics[str(ordinal)] = {
            **affine_metrics,
            "evaluation_modal_state_nrmse": _relative_rmse(
                affine_states,
                evaluation_capture[ordinal].features,
            ),
            "evaluation_mlp_output_nrmse": _relative_rmse(
                affine_output,
                native_output,
            ),
        }
        chart, chart_metrics = _fit_chart_conditioned_all_mode_graph(
            source_mlps[ordinal],
            affine,
            fit_capture[ordinal],
            model_fingerprint=source_fingerprint,
            layer_ordinal=ordinal,
            fit_split_sha256=str(split["fit_example_ids_sha256"]),
        )
        chart_graphs[ordinal] = chart
        with torch.no_grad():
            chart_execution = chart.execute_graph(
                evaluation_capture[ordinal].inputs,
                capture_modal_states=True,
                capture_chart_states=True,
            )
        if (
            chart_execution.modal_states is None
            or chart_execution.chart_memberships is None
        ):
            raise RuntimeError("full-model chart instrumentation is incomplete")
        memberships = chart_execution.chart_memberships
        entropy = -(
            memberships * memberships.clamp_min(1.0e-12).log()
        ).sum(dim=1)
        chart_local_metrics[str(ordinal)] = {
            **chart_metrics,
            "evaluation_modal_state_nrmse": _relative_rmse(
                chart_execution.modal_states,
                evaluation_capture[ordinal].features,
            ),
            "evaluation_mlp_output_nrmse": _relative_rmse(
                chart_execution.output,
                native_output,
            ),
            "evaluation_mean_membership_entropy": float(entropy.mean()),
            "evaluation_mean_max_membership": float(
                memberships.max(dim=1).values.mean()
            ),
            "evaluation_hard_chart_occupancy": torch.bincount(
                memberships.argmax(dim=1),
                minlength=CHART_COUNT,
            ).tolist(),
        }
        del affine_states, affine_output, native_output, chart_execution
    fit_seconds = time.monotonic() - fit_started
    affine_resources = _graph_resource_ledger(
        affine_graphs,
        source_whole_model_parameters=source_parameters,
    )
    chart_resources = _graph_resource_ledger(
        chart_graphs,
        source_whole_model_parameters=source_parameters,
    )
    del fit_capture, evaluation_capture
    gc.collect()

    _progress("evaluate full-model global-affine graph")
    affine_metrics, affine_calls = _evaluate_graph_arm(
        model=model,
        adapter=adapter,
        source_mlps=source_mlps,
        graphs=affine_graphs,
        batches=evaluation_batches,
        native_nll=native_nll,
    )
    _assert_source_restored(
        adapter=adapter,
        source_parameters=source_parameters,
        model_fingerprint=source_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )
    del affine_graphs
    gc.collect()

    _progress("evaluate full-model chart-conditioned graph")
    chart_metrics, chart_calls = _evaluate_graph_arm(
        model=model,
        adapter=adapter,
        source_mlps=source_mlps,
        graphs=chart_graphs,
        batches=evaluation_batches,
        native_nll=native_nll,
    )
    _assert_source_restored(
        adapter=adapter,
        source_parameters=source_parameters,
        model_fingerprint=source_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )
    evaluation_seconds = time.monotonic() - evaluation_started

    report: dict[str, object] = {
        "schema": FULL_MODEL_EXPERIMENT_SCHEMA,
        "format_version": FULL_MODEL_EXPERIMENT_FORMAT_VERSION,
        "scientific_status": {
            "development_only": True,
            "prompt_disjoint_evaluation": True,
            "family_disjoint_evaluation": False,
            "heldout_confirmation": False,
            "all_transformer_mlp_layers_replaced": True,
            "deletion_or_merging_performed": False,
            "compression_success_claimed": False,
            "latency_or_kernel_speed_claimed": False,
        },
        "model": {
            "model_id": model_id,
            "revision": revision,
            "model_fingerprint": source_fingerprint,
            "device": device_name,
            "dtype": dtype,
            "source_whole_model_learned_parameters": source_parameters,
            "target_layer_ordinals": list(layer_ordinals),
            "target_layer_count": len(layer_ordinals),
        },
        "data": split,
        "compilation": {
            "all_modes_retained": True,
            "mode_deletion_count": 0,
            "mode_merge_count": 0,
            "chart_count_per_layer": CHART_COUNT,
            "chart_rank": CHART_RANK,
            "chart_temperature": CHART_TEMPERATURE,
            "chart_edge_ridge": CHART_EDGE_RIDGE,
            "routing_input": "normalized_hidden_state",
            "held_rows_may_refit_graphs": False,
        },
        "capture": capture_receipt,
        "local_evaluation": {
            "approximated_all_modes_graph": affine_local_metrics,
            "chart_conditioned_all_modes_graph": chart_local_metrics,
        },
        "full_model_evaluation": {
            "native": {
                "supervised_tokens": native_tokens,
                "nll_per_token": native_nll,
            },
            "conditions": {
                "compiled_all_modes_intact": exact_metrics,
                "approximated_all_modes_graph": affine_metrics,
                "chart_conditioned_all_modes_graph": chart_metrics,
            },
        },
        "execution_receipts": {
            "compiled_all_modes_intact": exact_calls,
            "approximated_all_modes_graph": affine_calls,
            "chart_conditioned_all_modes_graph": chart_calls,
            "source_model_restored_exactly": True,
        },
        "resources": {
            "compiled_all_modes_intact": exact_resources,
            "approximated_all_modes_graph": affine_resources,
            "chart_conditioned_all_modes_graph": chart_resources,
        },
        "timing": {
            "graph_fit_seconds": fit_seconds,
            "full_model_evaluation_seconds": evaluation_seconds,
        },
    }
    report["report_sha256"] = _json_sha256(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _progress(f"published: {destination}")
    return copy.deepcopy(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile every Gemma MLP into all-mode generator graphs."
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    run_gemma3_full_model_all_mode_generator_graph(
        revision=args.revision,
        output=args.output,
        panel_path=args.panel_path,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
