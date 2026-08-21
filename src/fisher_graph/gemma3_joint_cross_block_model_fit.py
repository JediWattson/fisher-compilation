"""Fit MLP charts and attention-to-MLP transports for the joint Gemma arm.

This is a development fitter, not a held-out authority.  It uses the pinned
prompt-disjoint fit panel to capture native MLP Fisher rows and sequence-aware
attention generator packets.  Every target MLP chart is fitted locally first;
only the remaining modal residual is exposed to strictly earlier attention
events from any prior block.  Layer distance is never a numerical feature.

The resulting artifacts are source-independent modules, but the first v1 fit
is intentionally one-shot on the native trajectory.  Candidate-trajectory
iteration and family-disjoint guard evaluation remain separate later rungs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .gemma3_attention_generator import Gemma3AttentionGenerator
from .gemma3_cross_layer_chart_transport_executor import (
    Gemma3FactorizedChartGeneratorLayer,
)
from .gemma3_cross_layer_chart_transport_fit import (
    factor_chart_edge_bank,
    fit_fisher_weighted_affine_base,
    fit_fisher_weighted_dense_chart_edge_bank,
)
from .gemma3_joint_cross_block_chart_executor import (
    JointGraphCrossBlockTransportEdge,
    JointGraphTransportSpec,
)
from .gemma3_joint_cross_block_transport_fit import (
    fit_greedy_joint_cross_block_transports,
)
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
    CHART_COUNT,
    CHART_EDGE_RIDGE,
    CHART_RANK,
    CHART_TEMPERATURE,
    _LayerCapture,
    _capture_fisher_rows,
    _chart_membership_coordinates,
    _deterministic_chart_geometry,
    _relative_rmse,
)


@dataclass(frozen=True, slots=True)
class JointCrossBlockFitResult:
    """Executable fit-only artifacts for all 18 MLP mutations."""

    mlp_generators: tuple[Gemma3FactorizedChartGeneratorLayer, ...]
    transports: tuple[JointGraphCrossBlockTransportEdge, ...]
    layer_metrics: Mapping[str, object]
    fit_row_identity_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _row_identity(
    batch_row_keys: Sequence[Sequence[tuple[str, int]]],
) -> tuple[tuple[tuple[str, int], ...], str]:
    if isinstance(batch_row_keys, (str, bytes)) or not isinstance(
        batch_row_keys, Sequence
    ):
        raise TypeError("batch_row_keys must be a materialized sequence")
    rows = tuple(row for batch in batch_row_keys for row in tuple(batch))
    if not rows or any(
        not isinstance(example_id, str)
        or not example_id
        or type(position) is not int
        or position < 0
        for example_id, position in rows
    ):
        raise ValueError("fit row identity is invalid")
    if len(set(rows)) != len(rows):
        raise ValueError("fit row identities must be unique")
    digest = hashlib.sha256(
        b"fisher_graph:joint-cross-block-fit-rows:v1\0"
        + _canonical_json_bytes(rows)
    ).hexdigest()
    return rows, digest


def _row_fisher_weights(capture: _LayerCapture) -> Tensor:
    weights = (capture.features * capture.score_gradients).square().mean(dim=1)
    mean = weights.mean()
    if not bool(torch.isfinite(mean)) or float(mean) <= 0.0:
        raise RuntimeError("MLP Fisher row weights are degenerate")
    return (weights / mean).clamp(min=0.1, max=10.0)


def _attention_event_id(block_ordinal: int) -> str:
    return f"mutation.{2 * block_ordinal + 1}"


def _mlp_event_id(block_ordinal: int) -> str:
    return f"mutation.{2 * block_ordinal + 2}"


def _capture_attention_packets(
    *,
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[Mapping[str, Tensor]],
    attention_generators: Sequence[Gemma3AttentionGenerator],
) -> dict[str, Tensor]:
    if len(attention_generators) != len(adapter.layers):
        raise ValueError("attention generator catalog is incomplete")
    capture_sites = tuple(layer.input_site for layer in adapter.layers)
    packets: dict[str, list[Tensor]] = {
        _attention_event_id(ordinal): []
        for ordinal in range(len(adapter.layers))
    }
    for batch in batches:
        with torch.inference_mode():
            observed = adapter.forward(batch, capture_sites=capture_sites)
            for ordinal, generator in enumerate(attention_generators):
                source_state = observed.activations[
                    adapter.layers[ordinal].input_site
                ]
                execution = generator(source_state, observed.sequence)
                packet = execution.emitted_generator_latents
                if packet.shape[:2] != source_state.shape[:2]:
                    raise RuntimeError("attention packet sequence shape drifted")
                packets[_attention_event_id(ordinal)].append(
                    packet[observed.sequence.query_valid_mask]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
    return {
        event_id: torch.cat(values, dim=0).contiguous()
        for event_id, values in packets.items()
    }


def fit_joint_cross_block_mlp_charts(
    *,
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[Mapping[str, Tensor]],
    batch_row_keys: Sequence[Sequence[tuple[str, int]]],
    attention_generators: Sequence[Gemma3AttentionGenerator],
    model_fingerprint: str,
    generator_rank: int = 4,
    maximum_incoming_attention_sources_per_target_chart: int = 1,
    transport_relative_ridge: float = 1.0e-4,
    progress: Callable[[str], None] | None = None,
) -> JointCrossBlockFitResult:
    """Fit all MLP charts plus non-adjacent attention-to-MLP edges."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be Gemma3CausalLMAdapter")
    if len(adapter.layers) != 18:
        raise ValueError("joint Gemma v1 requires all 18 blocks")
    materialized_batches = tuple(batches)
    if not materialized_batches:
        raise ValueError("fit batches cannot be empty")
    rows, row_sha256 = _row_identity(batch_row_keys)
    if sum(
        int(batch["attention_mask"].bool().sum().item())
        for batch in materialized_batches
    ) != len(rows):
        raise ValueError("fit row keys do not match valid token rows")
    if (
        not isinstance(model_fingerprint, str)
        or len(model_fingerprint) != 64
    ):
        raise ValueError("model_fingerprint must be a SHA-256 digest")
    if type(generator_rank) is not int or generator_rank <= 0:
        raise ValueError("generator_rank must be positive")

    layer_ordinals = tuple(range(len(adapter.layers)))
    captures: dict[int, _LayerCapture] = {}
    for ordinal in layer_ordinals:
        if progress is not None:
            progress(
                "capture full-native-suffix Fisher rows for MLP block "
                f"{ordinal}/17"
            )
        # The shared capture helper makes each requested modal feature a leaf.
        # Capturing all 18 targets together would therefore detach later MLP
        # branches from earlier targets and truncate their score gradients.
        # One target per backward pass preserves the complete native suffix for
        # every Fisher row while keeping the source checkpoint frozen.
        observed = _capture_fisher_rows(
            adapter,
            materialized_batches,
            target_layers=(ordinal,),
        )
        if tuple(observed) != (ordinal,):
            raise RuntimeError("single-target Fisher capture inventory drifted")
        captures[ordinal] = observed[ordinal]
    attention_packets = _capture_attention_packets(
        adapter=adapter,
        batches=materialized_batches,
        attention_generators=attention_generators,
    )
    if any(packet.shape[0] != len(rows) for packet in attention_packets.values()):
        raise RuntimeError("attention and MLP fit rows are not aligned")

    prefix_catalog = hashlib.sha256(
        b"fisher_graph:joint-native-fit-trajectory:v1\0"
        + model_fingerprint.encode("ascii")
        + row_sha256.encode("ascii")
    ).hexdigest()
    mlp_generators: list[Gemma3FactorizedChartGeneratorLayer] = []
    local_executions: list[object] = []
    row_weights: list[Tensor] = []
    layer_metrics: dict[str, object] = {}

    for ordinal in layer_ordinals:
        if progress is not None:
            progress(f"fit Fisher activation chart for MLP block {ordinal}/17")
        capture = captures[ordinal]
        if capture.inputs.shape[0] != len(rows):
            raise RuntimeError("MLP capture row count drifted")
        weights = _row_fisher_weights(capture)
        row_weights.append(weights)
        affine_weight, affine_bias, affine_metrics = (
            fit_fisher_weighted_affine_base(
                inputs=capture.inputs,
                modal_states=capture.features,
                row_fisher_weights=weights,
                relative_ridge=1.0e-3,
            )
        )
        centers, bases, scales, occupancy = _deterministic_chart_geometry(
            capture.inputs,
            weights,
            chart_count=CHART_COUNT,
            chart_rank=CHART_RANK,
        )
        memberships, coordinates = _chart_membership_coordinates(
            capture.inputs,
            centers,
            bases,
            scales,
            temperature=CHART_TEMPERATURE,
        )
        base = capture.inputs @ affine_weight.T + affine_bias
        dense, dense_metrics = fit_fisher_weighted_dense_chart_edge_bank(
            memberships=memberships,
            coordinates=coordinates,
            base_modal_residual=capture.features - base,
            row_fisher_weights=weights,
            relative_ridge=CHART_EDGE_RIDGE,
        )
        local_encoder, mode_decoder, factor_metrics = factor_chart_edge_bank(
            dense,
            generator_rank,
        )
        source_mlp = adapter.source_module(adapter.layers[ordinal].id).mlp
        layer = Gemma3FactorizedChartGeneratorLayer(
            affine_weight=affine_weight,
            affine_bias=affine_bias,
            mode_decoder_weight=source_mlp.down_proj.weight.detach(),
            chart_centers=centers,
            chart_bases=bases,
            chart_distance_scales=scales,
            local_generator_encoders=local_encoder,
            chart_mode_decoders=mode_decoder,
            chart_temperature=CHART_TEMPERATURE,
            model_fingerprint=model_fingerprint,
            layer_ordinal=ordinal,
            fit_row_identity_sha256=row_sha256,
            compiled_prefix_catalog_sha256=prefix_catalog,
        )
        with torch.no_grad():
            local = layer.execute_with_incoming(capture.inputs)
        mlp_generators.append(layer)
        local_executions.append(local)
        layer_metrics[str(ordinal)] = {
            "row_count": len(rows),
            "fit_trajectory": "one_shot_native_development_v1",
            "affine": affine_metrics,
            "dense_chart": dense_metrics,
            "factorization": factor_metrics,
            "hard_chart_occupancy": occupancy.tolist(),
            "local_modal_nrmse": _relative_rmse(
                local.modal_states,
                capture.features,
            ),
            "transport": {
                "source_event_count": 0,
                "selected_edge_count": 0,
            },
        }

    transports: list[JointGraphCrossBlockTransportEdge] = []
    for target_ordinal in range(1, len(adapter.layers)):
        if progress is not None:
            progress(
                "fit all-earlier attention transports into MLP block "
                f"{target_ordinal}/17"
            )
        capture = captures[target_ordinal]
        target = mlp_generators[target_ordinal]
        local = local_executions[target_ordinal]
        target_residual = capture.features - local.modal_states
        earlier_attention = {
            _attention_event_id(source): attention_packets[
                _attention_event_id(source)
            ]
            for source in range(target_ordinal)
        }
        fitted, metrics = fit_greedy_joint_cross_block_transports(
            source_packets=earlier_attention,
            target_memberships=local.memberships,
            target_mode_decoders=target.chart_mode_decoders,
            target_modal_residual=target_residual,
            row_fisher_weights=row_weights[target_ordinal],
            maximum_incoming_sources_per_target_chart=(
                maximum_incoming_attention_sources_per_target_chart
            ),
            relative_ridge=transport_relative_ridge,
        )
        for edge_index, edge in enumerate(fitted):
            spec = JointGraphTransportSpec(
                transport_id=(
                    f"transport.{edge.source_event_id}.to."
                    f"{_mlp_event_id(target_ordinal)}."
                    f"c{edge.source_chart_index}.d{edge.target_chart_index}."
                    f"e{edge_index}"
                ),
                source_event_id=edge.source_event_id,
                target_event_id=_mlp_event_id(target_ordinal),
                source_chart_index=edge.source_chart_index,
                target_chart_index=edge.target_chart_index,
            )
            transports.append(
                JointGraphCrossBlockTransportEdge(
                    spec=spec,
                    transport_weight=edge.transport_weight,
                )
            )
        layer_metrics[str(target_ordinal)]["transport"] = metrics

    if not transports:
        raise RuntimeError(
            "joint candidate fit found no positive cross-block transport"
        )
    return JointCrossBlockFitResult(
        mlp_generators=tuple(mlp_generators),
        transports=tuple(transports),
        layer_metrics=layer_metrics,
        fit_row_identity_sha256=row_sha256,
    )


__all__ = [
    "JointCrossBlockFitResult",
    "fit_joint_cross_block_mlp_charts",
]
