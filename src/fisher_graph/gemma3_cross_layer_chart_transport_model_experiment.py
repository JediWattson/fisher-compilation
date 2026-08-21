"""Runnable predictive screen for progressive chart-generator transport.

The worker is intentionally separate from the setup command.  Importing this
module performs no model or data access.  A run must authenticate the frozen
setup artifact, pinned local Gemma revision, and pinned development panel
before fitting independent, adjacent-transport, and shuffled-control
trajectories.  It is the first executable rung of the larger protocol, not the
full causal/guard ladder.

The current panel is prompt-disjoint but not family-disjoint.  Consequently,
this worker can produce development evidence only; it cannot publish a heldout
compression or dependency claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .gemma3_cross_layer_chart_transport_executor import (
    CrossLayerChartTransportEdge,
    Gemma3CrossLayerChartGraphExecutor,
    Gemma3FactorizedChartGeneratorLayer,
)
from .gemma3_cross_layer_chart_transport_experiment import (
    FrozenChartTransportStage,
    sequentially_freeze_cross_layer_chart_transport_stack,
)
from .gemma3_cross_layer_chart_transport_fit import (
    factor_chart_edge_bank,
    fit_fisher_weighted_affine_base,
    fit_fisher_weighted_dense_chart_edge_bank,
    fit_greedy_adjacent_transport_edges,
)
from .gemma3_cross_layer_chart_transport_protocol import (
    DEFAULT_PROTOCOL_OUTPUT,
    PINNED_DEVELOPMENT_PANEL_PATH,
    PINNED_DEVELOPMENT_PANEL_SHA256,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    build_cross_layer_chart_transport_protocol,
    require_aligned_transport_rows,
)
from .gemma3_experiment import (
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_model_all_mode_generator_graph_experiment import (
    _assert_source_restored,
    _evaluate_native,
)
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
    CHART_COUNT,
    CHART_EDGE_RIDGE,
    CHART_RANK,
    CHART_TEMPERATURE,
    _LayerCapture,
    _MetricAccumulator,
    _capture_fisher_rows,
    _chart_membership_coordinates,
    _deterministic_chart_geometry,
    _load_prompt_split,
    _relative_rmse,
    _tokenize_batches,
)


CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_SCHEMA = (
    "fisher_graph.gemma3_cross_layer_chart_transport.experiment.v1"
)
CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "cross-layer-chart-transport-dev-v1.json"
)
_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-cross-layer-chart-transport-report:v1\0"
)


@dataclass(frozen=True, slots=True)
class _KeyedBatch:
    model_inputs: Mapping[str, Tensor]
    row_keys: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _ProgressiveLayerCapture:
    layer: _LayerCapture
    row_keys: tuple[tuple[str, int], ...]
    source_emitted_latents: Tensor | None
    source_row_keys: tuple[tuple[str, int], ...] | None

    def __post_init__(self) -> None:
        rows = self.layer.inputs.shape[0]
        if len(self.row_keys) != rows:
            raise ValueError("progressive capture row identity drifted")
        if (
            self.source_emitted_latents is not None
            and self.source_emitted_latents.shape[0] != rows
        ):
            raise ValueError("progressive source packet rows drifted")
        if (self.source_emitted_latents is None) != (
            self.source_row_keys is None
        ):
            raise ValueError("progressive source packet identity is incomplete")
        if self.source_row_keys is not None:
            require_aligned_transport_rows(
                self.source_row_keys,
                self.row_keys,
            )


def _progress(message: str) -> None:
    print(f"[cross-layer-chart-transport] {message}", flush=True)


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


def _load_authenticated_protocol(path: Path) -> dict[str, object]:
    expected = build_cross_layer_chart_transport_protocol()
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != json.loads(json.dumps(expected)):
        raise ValueError("chart transport setup artifact is not canonical")
    return expected


def _ordered_example_ids(
    panel_path: Path,
    prompts: Sequence[str],
) -> tuple[str, ...]:
    payload = json.loads(panel_path.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("development panel examples are unavailable")
    prompt_to_id: dict[str, str] = {}
    for row in examples:
        if not isinstance(row, dict):
            raise TypeError("development panel row must be an object")
        prompt = row.get("prompt")
        example_id = row.get("example_id")
        if not isinstance(prompt, str) or not isinstance(example_id, str):
            raise ValueError("development panel identity is invalid")
        if prompt in prompt_to_id:
            raise ValueError("development panel prompts must be unique")
        prompt_to_id[prompt] = example_id
    try:
        return tuple(prompt_to_id[prompt] for prompt in prompts)
    except KeyError as error:
        raise ValueError("split prompt is absent from the pinned panel") from error


def _logical_row_keys(
    example_ids: Sequence[str],
    attention_mask: Tensor,
) -> tuple[tuple[str, int], ...]:
    mask = attention_mask.detach().to(device="cpu").bool()
    if mask.ndim != 2 or mask.shape[0] != len(example_ids):
        raise ValueError("token batch identity does not match attention mask")
    keys: list[tuple[str, int]] = []
    for row, example_id in enumerate(example_ids):
        logical_position = 0
        for valid in mask[row].tolist():
            if valid:
                keys.append((example_id, logical_position))
                logical_position += 1
    return tuple(keys)


def _tokenize_keyed_batches(
    tokenizer: object,
    prompts: Sequence[str],
    example_ids: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[_KeyedBatch, ...]:
    if len(prompts) != len(example_ids):
        raise ValueError("prompt and example identities do not align")
    batches = _tokenize_batches(
        tokenizer,
        prompts,
        batch_size=batch_size,
        device=device,
    )
    result: list[_KeyedBatch] = []
    offset = 0
    for batch in batches:
        count = int(batch["input_ids"].shape[0])
        identities = tuple(example_ids[offset : offset + count])
        offset += count
        result.append(
            _KeyedBatch(
                model_inputs=batch,
                row_keys=_logical_row_keys(
                    identities,
                    batch["attention_mask"],
                ),
            )
        )
    if offset != len(example_ids):
        raise RuntimeError("token batching omitted development examples")
    return tuple(result)


def _prefix_executor(
    adapter: Gemma3CausalLMAdapter,
    stages: Sequence[FrozenChartTransportStage],
) -> Gemma3CrossLayerChartGraphExecutor | None:
    if not stages:
        return None
    layers = {
        ordinal: stage.layer
        for ordinal, stage in enumerate(stages)
    }
    edges = tuple(
        edge
        for stage in stages
        for edge in stage.incoming_edges
    )
    if any(
        not isinstance(layer, Gemma3FactorizedChartGeneratorLayer)
        for layer in layers.values()
    ) or any(not isinstance(edge, CrossLayerChartTransportEdge) for edge in edges):
        raise TypeError("frozen prefix contains a non-executable artifact")
    return Gemma3CrossLayerChartGraphExecutor(
        adapter,
        layers=layers,
        edges=edges,
    )


def _capture_progressive_layer(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[_KeyedBatch],
    *,
    layer_ordinal: int,
    prefix: Sequence[FrozenChartTransportStage],
) -> _ProgressiveLayerCapture:
    executor = _prefix_executor(adapter, prefix)
    captures: list[_LayerCapture] = []
    packets: list[Tensor] = []
    packet_row_keys: list[tuple[str, int]] = []
    row_keys: list[tuple[str, int]] = []
    for keyed_batch in batches:
        def capture() -> _LayerCapture:
            return _capture_fisher_rows(
                adapter,
                (keyed_batch.model_inputs,),
                target_layers=(layer_ordinal,),
            )[layer_ordinal]

        if executor is None:
            observed = capture()
        else:
            observed, receipt = executor.run_with_overlay(
                keyed_batch.model_inputs,
                callback=capture,
                capture_emitted_layer_ordinals=(layer_ordinal - 1,),
            )
            packets.append(
                receipt.captured_emitted_generator_latents[
                    layer_ordinal - 1
                ]
            )
            if packets[-1].shape[0] != len(keyed_batch.row_keys):
                raise RuntimeError(
                    "captured source packet rows do not match token identity"
                )
            packet_row_keys.extend(keyed_batch.row_keys)
        if observed.inputs.shape[0] != len(keyed_batch.row_keys):
            raise RuntimeError("captured target rows do not match token identity")
        captures.append(observed)
        row_keys.extend(keyed_batch.row_keys)
    combined = _LayerCapture(
        inputs=torch.cat([value.inputs for value in captures]).contiguous(),
        features=torch.cat([value.features for value in captures]).contiguous(),
        score_gradients=torch.cat(
            [value.score_gradients for value in captures]
        ).contiguous(),
    )
    source_packets = torch.cat(packets).contiguous() if packets else None
    keys = tuple(row_keys)
    source_keys = tuple(packet_row_keys) if packets else None
    if source_keys is not None:
        require_aligned_transport_rows(source_keys, keys)
    return _ProgressiveLayerCapture(
        layer=combined,
        row_keys=keys,
        source_emitted_latents=source_packets,
        source_row_keys=source_keys,
    )


def _row_fisher_weights(capture: _LayerCapture) -> Tensor:
    weights = (
        capture.features * capture.score_gradients
    ).square().mean(dim=1)
    return (weights / weights.mean().clamp_min(1.0e-12)).clamp(
        min=0.1,
        max=10.0,
    )


def _cosine_similarity(actual: Tensor, expected: Tensor) -> float:
    actual_flat = actual.double().reshape(-1)
    expected_flat = expected.double().reshape(-1)
    return float(
        (actual_flat @ expected_flat)
        / (
            actual_flat.square().sum().sqrt()
            * expected_flat.square().sum().sqrt()
        ).clamp_min(1.0e-12)
    )


def _fisher_weighted_relative_rmse(
    actual: Tensor,
    expected: Tensor,
    row_weights: Tensor,
) -> float:
    weights = row_weights.double().unsqueeze(1)
    numerator = (
        weights * (actual.double() - expected.double()).square()
    ).sum().sqrt()
    denominator = (
        weights * expected.double().square()
    ).sum().sqrt().clamp_min(1.0e-12)
    return float(numerator / denominator)


def _shuffle_packets_by_example(
    packets: Tensor,
    row_keys: Sequence[tuple[str, int]],
) -> Tensor:
    if packets.shape[0] != len(row_keys):
        raise ValueError("packet shuffle row identity drifted")
    by_example: dict[str, list[tuple[int, int]]] = {}
    for index, (example_id, position) in enumerate(row_keys):
        by_example.setdefault(example_id, []).append((position, index))
    examples = tuple(sorted(by_example))
    if len(examples) < 2:
        raise ValueError("packet shuffle requires at least two examples")
    result = torch.empty_like(packets)
    for target_offset, target_id in enumerate(examples):
        donor_id = examples[(target_offset + 1) % len(examples)]
        target_rows = sorted(by_example[target_id])
        donor_rows = sorted(by_example[donor_id])
        for position, target_index in target_rows:
            donor_offset = min(position, len(donor_rows) - 1)
            result[target_index] = packets[donor_rows[donor_offset][1]]
    return result


def _fit_progressive_stack(
    *,
    adapter: Gemma3CausalLMAdapter,
    source_mlps: Mapping[int, nn.Module],
    fit_batches: Sequence[_KeyedBatch],
    model_fingerprint: str,
    generator_rank: int,
    maximum_incoming_charts_per_target: int,
    transport_ridge: float,
    transport_enabled: bool,
    shuffle_fit_packets: bool,
) -> tuple[
    tuple[FrozenChartTransportStage, ...],
    tuple[object, ...],
    dict[str, object],
]:
    layer_metrics: dict[str, object] = {}

    def capture_layer(
        ordinal: int,
        prefix: tuple[FrozenChartTransportStage, ...],
        prefix_catalog: str,
    ) -> _ProgressiveLayerCapture:
        del prefix_catalog
        _progress(f"capture layer {ordinal} on compiled prefix {len(prefix)}")
        return _capture_progressive_layer(
            adapter,
            fit_batches,
            layer_ordinal=ordinal,
            prefix=prefix,
        )

    def fit_stage(
        ordinal: int,
        observed: _ProgressiveLayerCapture,
        prefix: tuple[FrozenChartTransportStage, ...],
        prefix_catalog: str,
    ) -> tuple[
        Gemma3FactorizedChartGeneratorLayer,
        Sequence[CrossLayerChartTransportEdge],
    ]:
        source_mlp = source_mlps[ordinal]
        row_fisher = _row_fisher_weights(observed.layer)
        affine_weight, affine_bias, affine_metrics = (
            fit_fisher_weighted_affine_base(
                inputs=observed.layer.inputs,
                modal_states=observed.layer.features,
                row_fisher_weights=row_fisher,
                relative_ridge=1.0e-3,
            )
        )
        centers, bases, distance_scales, occupancy = (
            _deterministic_chart_geometry(
                observed.layer.inputs,
                row_fisher,
                chart_count=CHART_COUNT,
                chart_rank=CHART_RANK,
            )
        )
        memberships, coordinates = _chart_membership_coordinates(
            observed.layer.inputs,
            centers,
            bases,
            distance_scales,
            temperature=CHART_TEMPERATURE,
        )
        base_states = (
            observed.layer.inputs @ affine_weight.T + affine_bias
        )
        chart_edge_weight, dense_metrics = (
            fit_fisher_weighted_dense_chart_edge_bank(
                memberships=memberships,
                coordinates=coordinates,
                base_modal_residual=(
                    observed.layer.features - base_states
                ),
                row_fisher_weights=row_fisher,
                relative_ridge=CHART_EDGE_RIDGE,
            )
        )
        local_encoders, mode_decoders, factor_metrics = (
            factor_chart_edge_bank(
                chart_edge_weight,
                generator_rank,
            )
        )
        row_identity = require_aligned_transport_rows(
            (
                observed.source_row_keys
                if observed.source_row_keys is not None
                else observed.row_keys
            ),
            observed.row_keys,
        )
        layer = Gemma3FactorizedChartGeneratorLayer(
            affine_weight=affine_weight,
            affine_bias=affine_bias,
            mode_decoder_weight=source_mlp.down_proj.weight.detach(),
            chart_centers=centers,
            chart_bases=bases,
            chart_distance_scales=distance_scales,
            local_generator_encoders=local_encoders,
            chart_mode_decoders=mode_decoders,
            chart_temperature=CHART_TEMPERATURE,
            model_fingerprint=model_fingerprint,
            layer_ordinal=ordinal,
            fit_row_identity_sha256=row_identity,
            compiled_prefix_catalog_sha256=prefix_catalog,
        )
        with torch.no_grad():
            local_execution = layer.execute_with_incoming(
                observed.layer.inputs
            )
        residual = observed.layer.features - local_execution.modal_states
        edges: list[CrossLayerChartTransportEdge] = []
        edge_fit_source_packets: Tensor | None = None
        edge_metrics: dict[str, object] = {
            "edge_count": 0,
            "transport_enabled": transport_enabled,
        }
        if transport_enabled and ordinal:
            source_packets = observed.source_emitted_latents
            if source_packets is None:
                raise RuntimeError("transport fit omitted the source packet")
            if shuffle_fit_packets:
                source_packets = _shuffle_packets_by_example(
                    source_packets,
                    observed.row_keys,
                )
            edge_fit_source_packets = source_packets
            fitted_edges, edge_metrics = (
                fit_greedy_adjacent_transport_edges(
                    source_emitted_latents=source_packets,
                    target_memberships=local_execution.memberships,
                    target_mode_decoders=layer.chart_mode_decoders,
                    target_modal_residual=residual,
                    row_fisher_weights=row_fisher,
                    maximum_incoming_charts_per_target=(
                        maximum_incoming_charts_per_target
                    ),
                    relative_ridge=transport_ridge,
                )
            )
            source_layer = prefix[-1].layer
            if not isinstance(
                source_layer,
                Gemma3FactorizedChartGeneratorLayer,
            ):
                raise TypeError("transport source stage is not executable")
            for fitted in fitted_edges:
                edges.append(
                    CrossLayerChartTransportEdge(
                        source_layer_ordinal=ordinal - 1,
                        target_layer_ordinal=ordinal,
                        source_chart_index=fitted.source_chart_index,
                        target_chart_index=fitted.target_chart_index,
                        transport_weight=fitted.transport_weight,
                        model_fingerprint=model_fingerprint,
                        fit_row_identity_sha256=row_identity,
                        source_layer_artifact_sha256=(
                            source_layer.artifact_sha256
                        ),
                        target_layer_artifact_sha256=layer.artifact_sha256,
                    )
                )
        fitted_incoming = torch.zeros(
            observed.layer.inputs.shape[0],
            layer.chart_count,
            layer.generator_rank,
            dtype=observed.layer.inputs.dtype,
        )
        if edge_fit_source_packets is not None:
            for edge in edges:
                fitted_incoming[:, edge.target_chart_index] += edge(
                    edge_fit_source_packets[:, edge.source_chart_index]
                )
        with torch.no_grad():
            fitted_execution = layer.execute_with_incoming(
                observed.layer.inputs,
                fitted_incoming,
            )
            native_mlp_output = source_mlp(observed.layer.inputs)
        layer_metrics[str(ordinal)] = {
            "row_count": observed.layer.inputs.shape[0],
            "row_identity_sha256": row_identity,
            "affine": affine_metrics,
            "dense_chart": dense_metrics,
            "hard_chart_occupancy": occupancy.tolist(),
            "factorization": factor_metrics,
            "transport": edge_metrics,
            "factorized_local_modal_state_nrmse": _relative_rmse(
                local_execution.modal_states,
                observed.layer.features,
            ),
            "factorized_local_fisher_weighted_modal_nrmse": (
                _fisher_weighted_relative_rmse(
                    local_execution.modal_states,
                    observed.layer.features,
                    row_fisher,
                )
            ),
            "factorized_local_mlp_output_nrmse": _relative_rmse(
                local_execution.output,
                native_mlp_output,
            ),
            "factorized_local_mlp_output_cosine": _cosine_similarity(
                local_execution.output,
                native_mlp_output,
            ),
            "fitted_stage_modal_state_nrmse": _relative_rmse(
                fitted_execution.modal_states,
                observed.layer.features,
            ),
            "fitted_stage_fisher_weighted_modal_nrmse": (
                _fisher_weighted_relative_rmse(
                    fitted_execution.modal_states,
                    observed.layer.features,
                    row_fisher,
                )
            ),
            "fitted_stage_mlp_output_nrmse": _relative_rmse(
                fitted_execution.output,
                native_mlp_output,
            ),
            "fitted_stage_mlp_output_cosine": _cosine_similarity(
                fitted_execution.output,
                native_mlp_output,
            ),
        }
        return layer, tuple(edges)

    stages, receipts = sequentially_freeze_cross_layer_chart_transport_stack(
        layer_count=len(adapter.layers),
        capture_layer=capture_layer,
        fit_and_freeze_stage=fit_stage,
    )
    return stages, receipts, layer_metrics


def _executor_from_stages(
    adapter: Gemma3CausalLMAdapter,
    stages: Sequence[FrozenChartTransportStage],
) -> Gemma3CrossLayerChartGraphExecutor:
    executor = _prefix_executor(adapter, stages)
    if executor is None or len(stages) != len(adapter.layers):
        raise ValueError("full chart transport stack is incomplete")
    return executor


def _evaluate_executor(
    *,
    executor: Gemma3CrossLayerChartGraphExecutor,
    model: nn.Module,
    batches: Sequence[_KeyedBatch],
    native_nll: float,
    condition: str,
) -> tuple[dict[str, float | int], dict[str, object]]:
    accumulator = _MetricAccumulator()
    peak_packets = 0
    peak_packet_bytes = 0
    source_projection_calls = {
        ordinal: {name: 0 for name in ("gate_proj", "up_proj", "down_proj")}
        for ordinal in range(len(executor.chart_layers))
    }
    with torch.no_grad():
        for keyed in batches:
            native = model(
                **keyed.model_inputs,
                use_cache=False,
                return_dict=True,
            )
            handles = []
            for ordinal in range(len(executor.chart_layers)):
                source_mlp = model.model.layers[ordinal].mlp
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    def count_source(
                        _module: nn.Module,
                        _args: tuple[Tensor, ...],
                        _output: object,
                        *,
                        layer_ordinal: int = ordinal,
                        operation: str = projection,
                    ) -> None:
                        source_projection_calls[layer_ordinal][operation] += 1

                    handles.append(
                        getattr(source_mlp, projection).register_forward_hook(
                            count_source
                        )
                    )
            try:
                candidate = executor.run(
                    keyed.model_inputs,
                    condition=condition,
                )
            finally:
                for handle in handles:
                    handle.remove()
            output = candidate.model_output
            accumulator.update(
                native.logits,
                output.logits,
                keyed.model_inputs,
            )
            peak_packets = max(
                peak_packets,
                candidate.peak_live_chart_packets,
            )
            peak_packet_bytes = max(
                peak_packet_bytes,
                candidate.peak_live_chart_packet_bytes,
            )
    if any(
        count
        for layer_counts in source_projection_calls.values()
        for count in layer_counts.values()
    ):
        raise RuntimeError("compiled chart stack called a native projection")
    return accumulator.result(native_nll=native_nll), {
        "condition": condition,
        "native_source_projection_call_counts": {
            str(ordinal): counts
            for ordinal, counts in source_projection_calls.items()
        },
        "compiled_layer_calls_per_batch": len(executor.chart_layers),
        "transport_edge_count": len(executor.transport_edges),
        "peak_live_chart_packets": peak_packets,
        "peak_live_chart_packet_bytes": peak_packet_bytes,
    }


def _resource_ledger(
    executor: Gemma3CrossLayerChartGraphExecutor,
    *,
    source_parameters: int,
) -> dict[str, object]:
    layers = tuple(executor.chart_layers.values())
    native_mlp_scalars = sum(
        3 * layer.hidden_width * layer.mode_count for layer in layers
    )
    logical_candidate = executor.stored_coefficient_count
    whole_model_candidate = (
        source_parameters - native_mlp_scalars + logical_candidate
    )
    return {
        "native_target_mlp_parameter_scalars": native_mlp_scalars,
        "logical_compiled_coefficient_scalars": logical_candidate,
        "logical_candidate_whole_model_scalars": whole_model_candidate,
        "logical_whole_model_reduction_fraction": (
            (source_parameters - whole_model_candidate) / source_parameters
        ),
        "estimated_compiled_macs_per_token": (
            executor.estimated_macs_per_token
        ),
        "routing_softmax_logits_per_token": sum(
            layer.chart_count for layer in layers
        ),
        "sparse_edge_index_bytes": executor.stored_sparse_index_bytes,
        "theoretical_sparse_cost_reported": True,
        "realized_kernel_latency_measured": False,
        "shadow_runtime_retains_native_source_weights": True,
        "source_free_serialized_candidate_materialized": False,
        "realized_resident_memory_reduction_claimed": False,
        "compression_success_claimed": False,
    }


def _comparison_gain(
    candidate: Mapping[str, float | int],
    control: Mapping[str, float | int],
) -> dict[str, float]:
    return {
        "delta_nll_improvement": float(
            control["delta_nll_per_token"]
        ) - float(candidate["delta_nll_per_token"]),
        "kl_improvement": float(
            control["native_to_candidate_kl_per_token"]
        ) - float(candidate["native_to_candidate_kl_per_token"]),
        "top1_agreement_improvement": float(
            candidate["top1_agreement_to_native"]
        ) - float(control["top1_agreement_to_native"]),
        "logit_nrmse_improvement": float(
            control["logit_nrmse"]
        ) - float(candidate["logit_nrmse"]),
    }


def run_gemma3_cross_layer_chart_transport_model_experiment(
    *,
    revision: str,
    protocol_path: Path | str = DEFAULT_PROTOCOL_OUTPUT,
    output: Path | str = DEFAULT_OUTPUT,
    panel_path: Path | str = PINNED_DEVELOPMENT_PANEL_PATH,
    model_id: str = PINNED_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 4,
    generator_rank: int = 4,
    maximum_incoming_charts_per_target: int = 2,
    transport_ridge: float = CHART_EDGE_RIDGE,
    include_row_shuffled_control: bool = True,
) -> dict[str, object]:
    """Fit and evaluate the preregistered development-only ladder."""

    protocol = _load_authenticated_protocol(Path(protocol_path))
    if revision != PINNED_MODEL_REVISION or model_id != PINNED_MODEL_ID:
        raise ValueError("worker requires the pinned Gemma checkpoint")
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("worker requires deterministic CPU float32")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if generator_rank not in protocol["generator_rank_ladder"]:
        raise ValueError("generator rank is outside the frozen ladder")
    if (
        maximum_incoming_charts_per_target
        not in protocol["maximum_incoming_charts_per_target_ladder"]
    ):
        raise ValueError("transport indegree is outside the frozen ladder")
    if transport_ridge not in protocol["transport_ridge_ladder"]:
        raise ValueError("transport ridge is outside the frozen ladder")
    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite a transport experiment")
    panel = Path(panel_path)
    if (
        hashlib.sha256(panel.read_bytes()).hexdigest()
        != PINNED_DEVELOPMENT_PANEL_SHA256
    ):
        raise ValueError("development panel bytes differ from the protocol")

    fit_prompts, evaluation_prompts, split = _load_prompt_split(panel)
    fit_ids = _ordered_example_ids(panel, fit_prompts)
    evaluation_ids = _ordered_example_ids(panel, evaluation_prompts)
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
    source_mlps = {
        ordinal: adapter.source_module(adapter.layers[ordinal].id).mlp
        for ordinal in range(len(adapter.layers))
    }
    source_parameters = sum(value.numel() for value in model.parameters())
    model_fingerprint = adapter.model_fingerprint()
    execution_fingerprint = adapter.execution_fingerprint()
    fit_batches = _tokenize_keyed_batches(
        tokenizer,
        fit_prompts,
        fit_ids,
        batch_size=batch_size,
        device=device,
    )
    evaluation_batches = _tokenize_keyed_batches(
        tokenizer,
        evaluation_prompts,
        evaluation_ids,
        batch_size=batch_size,
        device=device,
    )
    native_nll, native_tokens = _evaluate_native(
        model,
        tuple(value.model_inputs for value in evaluation_batches),
    )

    fit_started = time.monotonic()
    stacks: dict[str, tuple[FrozenChartTransportStage, ...]] = {}
    receipts: dict[str, tuple[object, ...]] = {}
    local_metrics: dict[str, dict[str, object]] = {}
    configurations = [
        ("progressive_independent_factorized_chart", False, False),
        ("progressive_factorized_chart_transport", True, False),
    ]
    if include_row_shuffled_control:
        configurations.append(
            ("transport_row_shuffled_negative_control", True, True)
        )
    for arm, transport_enabled, shuffle_packets in configurations:
        _progress(f"fit progressive stack: {arm}")
        stack, arm_receipts, metrics = _fit_progressive_stack(
            adapter=adapter,
            source_mlps=source_mlps,
            fit_batches=fit_batches,
            model_fingerprint=model_fingerprint,
            generator_rank=generator_rank,
            maximum_incoming_charts_per_target=(
                maximum_incoming_charts_per_target
            ),
            transport_ridge=transport_ridge,
            transport_enabled=transport_enabled,
            shuffle_fit_packets=shuffle_packets,
        )
        stacks[arm] = stack
        receipts[arm] = arm_receipts
        local_metrics[arm] = metrics
        _assert_source_restored(
            adapter=adapter,
            source_parameters=source_parameters,
            model_fingerprint=model_fingerprint,
            execution_fingerprint=execution_fingerprint,
        )
        gc.collect()
    fit_seconds = time.monotonic() - fit_started

    evaluation_started = time.monotonic()
    full_metrics: dict[str, object] = {}
    execution_receipts: dict[str, object] = {}
    resources: dict[str, object] = {}
    for arm, stack in stacks.items():
        executor = _executor_from_stages(adapter, stack)
        metrics, execution = _evaluate_executor(
            executor=executor,
            model=model,
            batches=evaluation_batches,
            native_nll=native_nll,
            condition="transport",
        )
        full_metrics[arm] = metrics
        execution_receipts[arm] = execution
        resources[arm] = _resource_ledger(
            executor,
            source_parameters=source_parameters,
        )
        if arm == "progressive_factorized_chart_transport":
            zero_metrics, zero_execution = _evaluate_executor(
                executor=executor,
                model=model,
                batches=evaluation_batches,
                native_nll=native_nll,
                condition="zero_transport",
            )
            full_metrics["transport_zero_edge_control"] = zero_metrics
            execution_receipts["transport_zero_edge_control"] = (
                zero_execution
            )
        _assert_source_restored(
            adapter=adapter,
            source_parameters=source_parameters,
            model_fingerprint=model_fingerprint,
            execution_fingerprint=execution_fingerprint,
        )
    evaluation_seconds = time.monotonic() - evaluation_started
    transport_metrics = full_metrics[
        "progressive_factorized_chart_transport"
    ]
    if not isinstance(transport_metrics, Mapping):
        raise RuntimeError("transport full-model metrics are unavailable")
    comparisons: dict[str, object] = {
        "transport_vs_progressive_independent": _comparison_gain(
            transport_metrics,
            full_metrics["progressive_independent_factorized_chart"],
        ),
        "transport_vs_nested_zero_edge": _comparison_gain(
            transport_metrics,
            full_metrics["transport_zero_edge_control"],
        ),
    }
    if include_row_shuffled_control:
        comparisons["transport_vs_row_shuffled_fit"] = _comparison_gain(
            transport_metrics,
            full_metrics["transport_row_shuffled_negative_control"],
        )
    unexecuted_arms = [
        "exact_all_mode_graph",
        "one_shot_independent_dense_chart_reproduction",
        "budget_matched_factorized_chart_transport",
    ]
    if not include_row_shuffled_control:
        unexecuted_arms.append(
            "transport_row_shuffled_negative_control"
        )

    report: dict[str, object] = {
        "schema": CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_SCHEMA,
        "format_version": (
            CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_FORMAT_VERSION
        ),
        "scientific_status": {
            "development_only": True,
            "partial_protocol_predictive_screen": True,
            "full_preregistered_ladder_executed": False,
            "prompt_disjoint_evaluation": True,
            "family_disjoint_evaluation": False,
            "heldout_confirmation": False,
            "all_modes_retained": True,
            "deletion_or_merging_performed": False,
            "compression_success_claimed": False,
            "dependency_proven_by_prediction_alone": False,
        },
        "execution_scope": {
            "executed_arms": ("native", *tuple(full_metrics)),
            "unexecuted_protocol_arms": tuple(unexecuted_arms),
            "deferred_required_metrics": (
                "prefix_delta_nll_and_kl_after_each_layer",
                "worst_family_metrics_and_bootstrap_intervals",
                "held_direction_jvp_and_finite_displacement",
            ),
            "compression_criterion_evaluated": False,
            "causal_confirmation_evaluated": False,
        },
        "protocol_sha256": protocol["protocol_sha256"],
        "model": {
            "model_id": model_id,
            "revision": revision,
            "model_fingerprint": model_fingerprint,
            "device": device_name,
            "dtype": dtype,
            "source_whole_model_parameters": source_parameters,
        },
        "data": split,
        "configuration": {
            "generator_rank": generator_rank,
            "chart_temperature": CHART_TEMPERATURE,
            "affine_relative_ridge": 1.0e-3,
            "local_chart_relative_ridge": CHART_EDGE_RIDGE,
            "maximum_incoming_charts_per_target": (
                maximum_incoming_charts_per_target
            ),
            "transport_ridge": transport_ridge,
            "row_shuffled_control_executed": (
                include_row_shuffled_control
            ),
            "same_token_adjacent_layer_only": True,
        },
        "native": {
            "nll_per_token": native_nll,
            "supervised_tokens": native_tokens,
        },
        "progressive_layer_fit": local_metrics,
        "progressive_freeze_receipts": {
            arm: [value.metadata() for value in arm_receipts]
            for arm, arm_receipts in receipts.items()
        },
        "full_model_evaluation": full_metrics,
        "full_model_comparisons": comparisons,
        "execution_receipts": execution_receipts,
        "resources": resources,
        "timing": {
            "progressive_fit_seconds": fit_seconds,
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
        description=(
            "Run the predictive-screen rung of the preregistered progressive "
            "cross-layer chart transport experiment."
        )
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_OUTPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=PINNED_DEVELOPMENT_PANEL_PATH,
    )
    parser.add_argument("--model-id", default=PINNED_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generator-rank", type=int, default=4)
    parser.add_argument("--max-incoming-charts", type=int, default=2)
    parser.add_argument("--transport-ridge", type=float, default=1.0e-2)
    parser.add_argument(
        "--skip-row-shuffled-control",
        action="store_true",
    )
    arguments = parser.parse_args(argv)
    run_gemma3_cross_layer_chart_transport_model_experiment(
        revision=arguments.revision,
        protocol_path=arguments.protocol,
        output=arguments.output,
        panel_path=arguments.panel_path,
        model_id=arguments.model_id,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        batch_size=arguments.batch_size,
        generator_rank=arguments.generator_rank,
        maximum_incoming_charts_per_target=(
            arguments.max_incoming_charts
        ),
        transport_ridge=arguments.transport_ridge,
        include_row_shuffled_control=(
            not arguments.skip_row_shuffled_control
        ),
    )
    return 0


__all__ = [
    "CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_FORMAT_VERSION",
    "CROSS_LAYER_CHART_TRANSPORT_EXPERIMENT_SCHEMA",
    "DEFAULT_OUTPUT",
    "run_gemma3_cross_layer_chart_transport_model_experiment",
]


if __name__ == "__main__":
    raise SystemExit(main())
