"""Source-free factorized chart generators with causal cross-layer messages.

Each layer emits one small latent per active chart.  Sparse, strictly forward
edges transport those generator latents into later charts before the target
mode decoder.  Runtime state belongs to one model forward and is discarded on
success or failure; chart indices have no cross-layer identity by themselves.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import TypeVar

import torch
from torch import Tensor, nn

from .gemma3_cross_layer_chart_transport_protocol import (
    compiled_chart_prefix_catalog_sha256,
    compiled_chart_transport_stage_sha256,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LAYER_HASH_DOMAIN = b"fisher_graph:factorized-chart-generator-layer:v1\0"
_EDGE_HASH_DOMAIN = b"fisher_graph:cross-layer-chart-transport-edge:v1\0"


_CallbackResult = TypeVar("_CallbackResult")


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _artifact_sha256(domain: bytes, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class FactorizedChartGeneratorExecution:
    output: Tensor
    modal_states: Tensor
    memberships: Tensor
    coordinates: Tensor
    emitted_generator_latents: Tensor


class Gemma3FactorizedChartGeneratorLayer(nn.Module):
    """One all-mode layer whose chart edges share low-rank generators."""

    def __init__(
        self,
        *,
        affine_weight: Tensor,
        affine_bias: Tensor,
        mode_decoder_weight: Tensor,
        chart_centers: Tensor,
        chart_bases: Tensor,
        chart_distance_scales: Tensor,
        local_generator_encoders: Tensor,
        chart_mode_decoders: Tensor,
        chart_temperature: float,
        model_fingerprint: str,
        layer_ordinal: int,
        fit_row_identity_sha256: str,
        compiled_prefix_catalog_sha256: str,
    ) -> None:
        super().__init__()
        _require_sha256(model_fingerprint, label="model_fingerprint")
        _require_sha256(
            fit_row_identity_sha256,
            label="fit_row_identity_sha256",
        )
        _require_sha256(
            compiled_prefix_catalog_sha256,
            label="compiled_prefix_catalog_sha256",
        )
        if type(layer_ordinal) is not int or layer_ordinal < 0:
            raise ValueError("layer_ordinal must be nonnegative")
        if affine_weight.ndim != 2 or affine_bias.shape != affine_weight.shape[:1]:
            raise ValueError("affine generator tensors are inconsistent")
        modes, hidden = affine_weight.shape
        if mode_decoder_weight.shape != (hidden, modes):
            raise ValueError("mode decoder shape is inconsistent")
        if chart_centers.ndim != 2 or chart_centers.shape[1] != hidden:
            raise ValueError("chart centers are inconsistent")
        charts = chart_centers.shape[0]
        if (
            chart_bases.ndim != 3
            or chart_bases.shape[:2] != (charts, hidden)
            or chart_distance_scales.shape != (charts,)
        ):
            raise ValueError("chart geometry tensors are inconsistent")
        coordinates = chart_bases.shape[2]
        if (
            local_generator_encoders.ndim != 3
            or local_generator_encoders.shape[:2]
            != (charts, coordinates + 1)
        ):
            raise ValueError("local generator encoders are inconsistent")
        generator_rank = local_generator_encoders.shape[2]
        if chart_mode_decoders.shape != (charts, generator_rank, modes):
            raise ValueError("chart mode decoders are inconsistent")
        if (
            not isinstance(chart_temperature, (int, float))
            or isinstance(chart_temperature, bool)
            or not math.isfinite(float(chart_temperature))
            or float(chart_temperature) <= 0.0
        ):
            raise ValueError("chart_temperature must be finite and positive")
        tensors = (
            affine_weight,
            affine_bias,
            mode_decoder_weight,
            chart_centers,
            chart_bases,
            chart_distance_scales,
            local_generator_encoders,
            chart_mode_decoders,
        )
        if any(not value.is_floating_point() for value in tensors):
            raise ValueError("factorized chart tensors must be floating point")
        if any(
            value.dtype != affine_weight.dtype
            or value.device != affine_weight.device
            for value in tensors
        ):
            raise ValueError("factorized chart tensors must share dtype and device")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("factorized chart tensors must be finite")
        if bool((chart_distance_scales <= 0.0).any()):
            raise ValueError("chart distance scales must be positive")

        self.layer_ordinal = layer_ordinal
        self.model_fingerprint = model_fingerprint
        self.fit_row_identity_sha256 = fit_row_identity_sha256
        self.compiled_prefix_catalog_sha256 = compiled_prefix_catalog_sha256
        self.hidden_width = hidden
        self.mode_count = modes
        self.chart_count = charts
        self.chart_rank = coordinates
        self.generator_rank = generator_rank
        self.chart_temperature = float(chart_temperature)
        self.affine_generators = nn.Linear(hidden, modes, bias=True)
        self.mode_decoder = nn.Linear(modes, hidden, bias=False)
        with torch.no_grad():
            self.affine_generators.weight.copy_(affine_weight.detach())
            self.affine_generators.bias.copy_(affine_bias.detach())
            self.mode_decoder.weight.copy_(mode_decoder_weight.detach())
        self.register_buffer(
            "chart_centers",
            chart_centers.detach().clone().contiguous(),
        )
        self.register_buffer(
            "chart_bases",
            chart_bases.detach().clone().contiguous(),
        )
        self.register_buffer(
            "chart_distance_scales",
            chart_distance_scales.detach().clone().contiguous(),
        )
        self.local_generator_encoders = nn.Parameter(
            local_generator_encoders.detach().clone().contiguous(),
            requires_grad=False,
        )
        self.chart_mode_decoders = nn.Parameter(
            chart_mode_decoders.detach().clone().contiguous(),
            requires_grad=False,
        )
        self.requires_grad_(False)
        self.eval()
        self._artifact_sha256 = _artifact_sha256(
            _LAYER_HASH_DOMAIN,
            self._hash_payload(),
        )

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def gate_proj(self) -> nn.Module:
        return self.affine_generators

    @property
    def up_proj(self) -> nn.Module:
        return self.affine_generators

    @property
    def down_proj(self) -> nn.Module:
        return self.mode_decoder

    @property
    def stored_coefficient_count(self) -> int:
        return sum(
            value.numel()
            for value in (*self.parameters(), *self.buffers())
        )

    @property
    def estimated_macs_per_token(self) -> int:
        return (
            2 * self.hidden_width * self.mode_count
            + self.chart_count * self.hidden_width
            + self.chart_count * self.hidden_width * self.chart_rank
            + self.chart_count * (self.chart_rank + 1) * self.generator_rank
            + self.chart_count * self.generator_rank * self.mode_count
            + self.chart_count * self.generator_rank
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.factorized_chart_generator_layer.v1",
            "model_fingerprint": self.model_fingerprint,
            "layer_ordinal": self.layer_ordinal,
            "hidden_width": self.hidden_width,
            "mode_count": self.mode_count,
            "chart_count": self.chart_count,
            "chart_rank": self.chart_rank,
            "generator_rank": self.generator_rank,
            "chart_temperature": self.chart_temperature,
            "fit_row_identity_sha256": self.fit_row_identity_sha256,
            "compiled_prefix_catalog_sha256": (
                self.compiled_prefix_catalog_sha256
            ),
            "tensor_sha256s": {
                name: _tensor_sha256(value)
                for name, value in self.state_dict().items()
            },
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "all_modes_retained": True,
            "removed_mode_count": 0,
            "merged_mode_count": 0,
            "source_free_runtime": True,
            "stored_coefficient_count": self.stored_coefficient_count,
            "estimated_macs_per_token": self.estimated_macs_per_token,
        }

    def assert_authenticated(self) -> None:
        if (
            self.training
            or any(value.requires_grad for value in self.parameters())
            or _artifact_sha256(_LAYER_HASH_DOMAIN, self._hash_payload())
            != self.artifact_sha256
        ):
            raise RuntimeError("factorized chart layer authentication drifted")

    def encode_local_generator_latents(
        self,
        normalized_input: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if normalized_input.shape[-1] != self.hidden_width:
            raise ValueError("factorized chart input width drifted")
        displacement = normalized_input.unsqueeze(-2) - self.chart_centers
        distance = displacement.square().sum(dim=-1)
        logits = -0.5 * distance / (
            self.chart_distance_scales.square() * self.chart_temperature
        )
        memberships = logits.softmax(dim=-1)
        coordinates = torch.einsum(
            "...ch,chr->...cr",
            displacement,
            self.chart_bases,
        )
        augmented = torch.cat(
            (torch.ones_like(coordinates[..., :1]), coordinates),
            dim=-1,
        )
        local_latents = torch.einsum(
            "...cs,csg->...cg",
            augmented,
            self.local_generator_encoders,
        )
        return memberships, coordinates, local_latents

    def execute_with_incoming(
        self,
        normalized_input: Tensor,
        incoming_generator_latents: Tensor | None = None,
    ) -> FactorizedChartGeneratorExecution:
        memberships, coordinates, local_latents = (
            self.encode_local_generator_latents(normalized_input)
        )
        if incoming_generator_latents is None:
            incoming = torch.zeros_like(local_latents)
        else:
            incoming = incoming_generator_latents
            if incoming.shape != local_latents.shape:
                raise ValueError("incoming generator latent shape drifted")
            if incoming.device != local_latents.device:
                raise ValueError("incoming generator latents are on the wrong device")
            if incoming.dtype != local_latents.dtype:
                raise ValueError("incoming generator latent dtype drifted")
        emitted = memberships.unsqueeze(-1) * (local_latents + incoming)
        modal_states = self.affine_generators(normalized_input) + torch.einsum(
            "...cg,cgk->...k",
            emitted,
            self.chart_mode_decoders,
        )
        return FactorizedChartGeneratorExecution(
            output=self.mode_decoder(modal_states),
            modal_states=modal_states,
            memberships=memberships,
            coordinates=coordinates,
            emitted_generator_latents=emitted,
        )

    def forward(self, normalized_input: Tensor) -> Tensor:
        return self.execute_with_incoming(normalized_input).output


class CrossLayerChartTransportEdge(nn.Module):
    """One sparse source-chart to target-chart generator-latent edge."""

    def __init__(
        self,
        *,
        source_layer_ordinal: int,
        target_layer_ordinal: int,
        source_chart_index: int,
        target_chart_index: int,
        transport_weight: Tensor,
        model_fingerprint: str,
        fit_row_identity_sha256: str,
        source_layer_artifact_sha256: str,
        target_layer_artifact_sha256: str,
    ) -> None:
        super().__init__()
        if (
            type(source_layer_ordinal) is not int
            or type(target_layer_ordinal) is not int
            or source_layer_ordinal < 0
            or target_layer_ordinal != source_layer_ordinal + 1
        ):
            raise ValueError("transport edge must point to the adjacent later layer")
        if (
            type(source_chart_index) is not int
            or type(target_chart_index) is not int
            or source_chart_index < 0
            or target_chart_index < 0
        ):
            raise ValueError("transport chart indices must be nonnegative")
        if (
            transport_weight.ndim != 2
            or transport_weight.shape[0] != transport_weight.shape[1]
            or not transport_weight.shape[0]
            or not transport_weight.is_floating_point()
            or not bool(torch.isfinite(transport_weight).all())
        ):
            raise ValueError("transport weight must be a finite square matrix")
        self.source_layer_ordinal = source_layer_ordinal
        self.target_layer_ordinal = target_layer_ordinal
        self.source_chart_index = source_chart_index
        self.target_chart_index = target_chart_index
        self.generator_rank = transport_weight.shape[0]
        self.model_fingerprint = _require_sha256(
            model_fingerprint,
            label="model_fingerprint",
        )
        self.fit_row_identity_sha256 = _require_sha256(
            fit_row_identity_sha256,
            label="fit_row_identity_sha256",
        )
        self.source_layer_artifact_sha256 = _require_sha256(
            source_layer_artifact_sha256,
            label="source_layer_artifact_sha256",
        )
        self.target_layer_artifact_sha256 = _require_sha256(
            target_layer_artifact_sha256,
            label="target_layer_artifact_sha256",
        )
        self.transport_weight = nn.Parameter(
            transport_weight.detach().clone().contiguous(),
            requires_grad=False,
        )
        self.requires_grad_(False)
        self.eval()
        self._artifact_sha256 = _artifact_sha256(
            _EDGE_HASH_DOMAIN,
            self._hash_payload(),
        )

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def coordinate(self) -> tuple[int, int, int, int]:
        return (
            self.source_layer_ordinal,
            self.source_chart_index,
            self.target_layer_ordinal,
            self.target_chart_index,
        )

    @property
    def stored_coefficient_count(self) -> int:
        return self.transport_weight.numel()

    @property
    def sparse_index_bytes(self) -> int:
        return 4 * 8

    @property
    def estimated_macs_per_token(self) -> int:
        return self.transport_weight.numel()

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.cross_layer_chart_transport_edge.v1",
            "source_layer_ordinal": self.source_layer_ordinal,
            "target_layer_ordinal": self.target_layer_ordinal,
            "source_chart_index": self.source_chart_index,
            "target_chart_index": self.target_chart_index,
            "generator_rank": self.generator_rank,
            "model_fingerprint": self.model_fingerprint,
            "fit_row_identity_sha256": self.fit_row_identity_sha256,
            "source_layer_artifact_sha256": self.source_layer_artifact_sha256,
            "target_layer_artifact_sha256": self.target_layer_artifact_sha256,
            "transport_weight_sha256": _tensor_sha256(self.transport_weight),
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "stored_coefficient_count": self.stored_coefficient_count,
            "estimated_macs_per_token": self.estimated_macs_per_token,
            "sparse_index_bytes": self.sparse_index_bytes,
            "strictly_forward": True,
        }

    def assert_authenticated(self) -> None:
        if (
            self.training
            or any(value.requires_grad for value in self.parameters())
            or _artifact_sha256(_EDGE_HASH_DOMAIN, self._hash_payload())
            != self.artifact_sha256
        ):
            raise RuntimeError("chart transport edge authentication drifted")

    def forward(self, source_generator_latent: Tensor) -> Tensor:
        if source_generator_latent.shape[-1] != self.generator_rank:
            raise ValueError("source generator latent rank drifted")
        return source_generator_latent @ self.transport_weight


class CrossLayerChartMessageSession:
    """Per-forward causal chart-message state with deterministic release."""

    def __init__(
        self,
        *,
        edges: Sequence[CrossLayerChartTransportEdge],
        valid_positions: Tensor,
        condition: str,
        capture_emitted_layer_ordinals: Sequence[int] = (),
    ) -> None:
        if condition not in {"transport", "zero_transport"}:
            raise ValueError("unknown chart transport condition")
        if (
            not isinstance(valid_positions, Tensor)
            or valid_positions.dtype is not torch.bool
        ):
            raise ValueError("valid_positions must be a boolean Tensor")
        self.condition = condition
        self.valid_positions = valid_positions
        self._edges = tuple(edges)
        capture_ordinals = tuple(capture_emitted_layer_ordinals)
        if (
            len(set(capture_ordinals)) != len(capture_ordinals)
            or any(
                type(ordinal) is not int or ordinal < 0
                for ordinal in capture_ordinals
            )
        ):
            raise ValueError("emitted chart capture layers are invalid")
        self._capture_emitted_layer_ordinals = frozenset(capture_ordinals)
        self._captured_emitted_latents: dict[int, Tensor] = {}
        self._messages: dict[int, Tensor] = {}
        self._published_layers: set[int] = set()
        self._target_layers: set[int] = set()
        self._last_consumer = {
            source: max(
                edge.target_layer_ordinal
                for edge in self._edges
                if edge.source_layer_ordinal == source
            )
            for source in {
                edge.source_layer_ordinal for edge in self._edges
            }
        }
        self.peak_live_messages = 0
        self.peak_live_message_bytes = 0
        self.closed = False

    def publish(self, layer_ordinal: int, emitted_latents: Tensor) -> None:
        if self.closed or layer_ordinal in self._published_layers:
            raise RuntimeError("chart layer published more than once")
        if emitted_latents.shape[:-2] != self.valid_positions.shape:
            raise ValueError("emitted chart latents do not match valid positions")
        if emitted_latents.device != self.valid_positions.device:
            raise ValueError(
                "emitted chart latents and valid positions differ in device"
            )
        self._published_layers.add(layer_ordinal)
        if layer_ordinal in self._capture_emitted_layer_ordinals:
            self._captured_emitted_latents[layer_ordinal] = (
                emitted_latents[self.valid_positions]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
        if layer_ordinal not in self._last_consumer:
            return
        masked = emitted_latents.masked_fill(
            ~self.valid_positions[..., None, None],
            0,
        )
        self._messages[layer_ordinal] = masked
        self.peak_live_messages = max(
            self.peak_live_messages,
            len(self._messages),
        )
        self.peak_live_message_bytes = max(
            self.peak_live_message_bytes,
            sum(
                value.numel() * value.element_size()
                for value in self._messages.values()
            ),
        )

    def incoming_for(
        self,
        *,
        target_layer: Gemma3FactorizedChartGeneratorLayer,
        edges: Sequence[CrossLayerChartTransportEdge],
    ) -> Tensor:
        ordinal = target_layer.layer_ordinal
        if self.closed or ordinal in self._target_layers:
            raise RuntimeError("chart target consumed messages more than once")
        self._target_layers.add(ordinal)
        shape = (
            *self.valid_positions.shape,
            target_layer.chart_count,
            target_layer.generator_rank,
        )
        incoming = torch.zeros(
            shape,
            device=target_layer.chart_centers.device,
            dtype=target_layer.chart_centers.dtype,
        )
        for edge in edges:
            if edge.target_layer_ordinal != ordinal:
                raise ValueError("incoming edge targets the wrong layer")
            try:
                source = self._messages[edge.source_layer_ordinal]
            except KeyError as error:
                raise RuntimeError(
                    "target chart executed before its source producer"
                ) from error
            if (
                edge.source_chart_index >= source.shape[-2]
                or edge.target_chart_index >= target_layer.chart_count
                or edge.generator_rank != target_layer.generator_rank
            ):
                raise ValueError("transport edge does not match chart geometry")
            if self.condition == "transport":
                incoming[..., edge.target_chart_index, :] += edge(
                    source[..., edge.source_chart_index, :]
                )
        for source_ordinal, last_consumer in self._last_consumer.items():
            if last_consumer == ordinal:
                self._messages.pop(source_ordinal, None)
        return incoming

    def finish(self, *, expected_layers: Sequence[int]) -> None:
        if self.closed:
            raise RuntimeError("chart message session already closed")
        self.closed = True
        if self._published_layers != set(expected_layers):
            raise RuntimeError("not every chart layer executed exactly once")
        if set(self._captured_emitted_latents) != set(
            self._capture_emitted_layer_ordinals
        ):
            raise RuntimeError(
                "not every requested emitted chart packet was captured"
            )
        if self._messages:
            raise RuntimeError("cross-layer chart messages remained live")

    def abort(self) -> None:
        self._messages.clear()
        self._captured_emitted_latents.clear()
        self.closed = True


class _ChartTransportOverlay(nn.Module):
    def __init__(
        self,
        layer: Gemma3FactorizedChartGeneratorLayer,
        *,
        incoming_edges: Sequence[CrossLayerChartTransportEdge],
        session: CrossLayerChartMessageSession,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.incoming_edges = tuple(incoming_edges)
        object.__setattr__(self, "_session", session)
        self.eval()

    @property
    def gate_proj(self) -> nn.Module:
        return self.layer.gate_proj

    @property
    def up_proj(self) -> nn.Module:
        return self.layer.up_proj

    @property
    def down_proj(self) -> nn.Module:
        return self.layer.down_proj

    def forward(self, normalized_input: Tensor) -> Tensor:
        incoming = self._session.incoming_for(
            target_layer=self.layer,
            edges=self.incoming_edges,
        )
        execution = self.layer.execute_with_incoming(
            normalized_input,
            incoming,
        )
        self._session.publish(
            self.layer.layer_ordinal,
            execution.emitted_generator_latents,
        )
        return execution.output


@dataclass(frozen=True, slots=True)
class CrossLayerChartModelExecution:
    model_output: object
    condition: str
    layer_calls: tuple[int, ...]
    transport_edge_count: int
    peak_live_chart_packets: int
    peak_live_chart_packet_bytes: int


@dataclass(frozen=True, slots=True)
class CrossLayerChartOverlayReceipt:
    """Authenticated accounting for one synchronous prefix overlay call."""

    condition: str
    layer_calls: tuple[int, ...]
    transport_edge_count: int
    peak_live_chart_packets: int
    peak_live_chart_packet_bytes: int
    captured_emitted_generator_latents: Mapping[int, Tensor]


class Gemma3CrossLayerChartGraphExecutor(nn.Module):
    """Temporarily overlay a contiguous causal prefix or complete stack."""

    def __init__(
        self,
        adapter: object,
        *,
        layers: Mapping[int, Gemma3FactorizedChartGeneratorLayer],
        edges: Sequence[CrossLayerChartTransportEdge],
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("chart transport executor requires layers")
        ordered_layers = tuple(sorted(layers))
        if ordered_layers != tuple(range(len(ordered_layers))):
            raise ValueError("chart transport layers must be a contiguous prefix")
        adapter_layers = getattr(adapter, "layers", None)
        if adapter_layers is None or len(ordered_layers) > len(adapter_layers):
            raise ValueError("adapter layer catalog does not contain chart prefix")
        model_fingerprints = {
            layer.model_fingerprint for layer in layers.values()
        }
        if len(model_fingerprints) != 1:
            raise ValueError("chart layers do not share one model fingerprint")
        model_fingerprint = next(iter(model_fingerprints))
        if getattr(adapter, "model_fingerprint")() != model_fingerprint:
            raise ValueError("chart stack does not bind the live model")
        self._model_fingerprint = model_fingerprint
        edge_tuple = tuple(edges)
        coordinates = [edge.coordinate for edge in edge_tuple]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("duplicate cross-layer chart transport edge")
        for edge in edge_tuple:
            source_layer = layers.get(edge.source_layer_ordinal)
            target_layer = layers.get(edge.target_layer_ordinal)
            if (
                source_layer is None
                or target_layer is None
                or edge.model_fingerprint != model_fingerprint
                or edge.source_layer_artifact_sha256
                != source_layer.artifact_sha256
                or edge.target_layer_artifact_sha256
                != target_layer.artifact_sha256
                or edge.fit_row_identity_sha256
                != source_layer.fit_row_identity_sha256
                or edge.fit_row_identity_sha256
                != target_layer.fit_row_identity_sha256
                or edge.generator_rank != source_layer.generator_rank
                or edge.generator_rank != target_layer.generator_rank
                or edge.source_chart_index >= source_layer.chart_count
                or edge.target_chart_index >= target_layer.chart_count
                or edge.transport_weight.dtype
                != source_layer.local_generator_encoders.dtype
                or edge.transport_weight.dtype
                != target_layer.local_generator_encoders.dtype
                or edge.transport_weight.device
                != source_layer.local_generator_encoders.device
                or edge.transport_weight.device
                != target_layer.local_generator_encoders.device
            ):
                raise ValueError("transport edge lineage does not match chart stack")
        prior_stage_artifacts: list[str] = []
        for ordinal in ordered_layers:
            layer = layers[ordinal]
            expected_prefix = compiled_chart_prefix_catalog_sha256(
                prior_stage_artifacts
            )
            if layer.compiled_prefix_catalog_sha256 != expected_prefix:
                raise ValueError("chart layer compiled-prefix lineage drifted")
            inbound = tuple(
                sorted(
                    (
                        edge.artifact_sha256
                        for edge in edge_tuple
                        if edge.target_layer_ordinal == ordinal
                    )
                )
            )
            prior_stage_artifacts.append(
                compiled_chart_transport_stage_sha256(
                    layer_ordinal=ordinal,
                    layer_artifact_sha256=layer.artifact_sha256,
                    incoming_edge_artifact_sha256s=inbound,
                )
            )
        self.adapter = adapter
        self.chart_layers = nn.ModuleDict(
            {str(ordinal): layers[ordinal] for ordinal in ordered_layers}
        )
        self.transport_edges = nn.ModuleList(edge_tuple)
        incoming: defaultdict[int, list[CrossLayerChartTransportEdge]] = (
            defaultdict(list)
        )
        for edge in edge_tuple:
            incoming[edge.target_layer_ordinal].append(edge)
        self._incoming = {
            ordinal: tuple(sorted(values, key=lambda value: value.coordinate))
            for ordinal, values in incoming.items()
        }
        self._ordered_layers = ordered_layers
        self._source_mlp_ids = tuple(
            id(adapter.module.model.layers[ordinal].mlp)
            for ordinal in ordered_layers
        )
        self._active = False
        self.requires_grad_(False)
        self.eval()

    @property
    def stored_coefficient_count(self) -> int:
        return sum(
            layer.stored_coefficient_count
            for layer in self.chart_layers.values()
        ) + sum(
            edge.stored_coefficient_count for edge in self.transport_edges
        )

    @property
    def estimated_macs_per_token(self) -> int:
        return sum(
            layer.estimated_macs_per_token
            for layer in self.chart_layers.values()
        ) + sum(
            edge.estimated_macs_per_token for edge in self.transport_edges
        )

    @property
    def stored_sparse_index_bytes(self) -> int:
        return sum(edge.sparse_index_bytes for edge in self.transport_edges)

    def _assert_runtime_authenticated(self) -> None:
        if self.training:
            raise RuntimeError("chart transport executor must remain in eval mode")
        if self.adapter.model_fingerprint() != self._model_fingerprint:
            raise RuntimeError("live model fingerprint drifted before execution")
        current_source_ids = tuple(
            id(self.adapter.module.model.layers[ordinal].mlp)
            for ordinal in self._ordered_layers
        )
        if current_source_ids != self._source_mlp_ids:
            raise RuntimeError("native Gemma MLP identity drifted")
        for layer in self.chart_layers.values():
            layer.assert_authenticated()
        for edge in self.transport_edges:
            edge.assert_authenticated()

    def run_with_overlay(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        callback: Callable[[], _CallbackResult],
        condition: str = "transport",
        capture_emitted_layer_ordinals: Sequence[int] = (),
    ) -> tuple[_CallbackResult, CrossLayerChartOverlayReceipt]:
        """Run one synchronous callback under this exact compiled prefix.

        The callback must execute the live model exactly once with
        ``model_inputs``.  This keeps instrumentation and backward passes in
        the caller while the executor owns causal message state, restoration,
        and optional valid-token packet capture.
        """

        if self._active:
            raise RuntimeError("chart transport executor is not reentrant")
        if not callable(callback):
            raise TypeError("chart transport overlay callback must be callable")
        capture_ordinals = tuple(capture_emitted_layer_ordinals)
        if (
            len(set(capture_ordinals)) != len(capture_ordinals)
            or any(
                ordinal not in self._ordered_layers
                for ordinal in capture_ordinals
            )
        ):
            raise ValueError(
                "emitted packet capture must select compiled prefix layers"
            )
        self._assert_runtime_authenticated()
        context = self.adapter.prepare_sequence(model_inputs)
        valid_positions = context.query_valid_mask
        model = self.adapter.module
        model_layers = model.model.layers
        originals: dict[int, nn.Module] = {}
        session = CrossLayerChartMessageSession(
            edges=tuple(self.transport_edges),
            valid_positions=valid_positions,
            condition=condition,
            capture_emitted_layer_ordinals=capture_ordinals,
        )
        self._active = True
        succeeded = False
        try:
            for ordinal in self._ordered_layers:
                original = model_layers[ordinal].mlp
                if not isinstance(original, nn.Module):
                    raise TypeError("live Gemma layer MLP is invalid")
                originals[ordinal] = original
                model_layers[ordinal].mlp = _ChartTransportOverlay(
                    self.chart_layers[str(ordinal)],
                    incoming_edges=self._incoming.get(ordinal, ()),
                    session=session,
                )
            callback_result = callback()
            session.finish(expected_layers=self._ordered_layers)
            succeeded = True
        finally:
            for ordinal, original in originals.items():
                model_layers[ordinal].mlp = original
            if not succeeded:
                session.abort()
            self._active = False
        self._assert_runtime_authenticated()
        captured = {
            ordinal: value.clone()
            for ordinal, value in sorted(session._captured_emitted_latents.items())
        }
        return callback_result, CrossLayerChartOverlayReceipt(
            condition=condition,
            layer_calls=tuple(sorted(session._published_layers)),
            transport_edge_count=len(self.transport_edges),
            peak_live_chart_packets=session.peak_live_messages,
            peak_live_chart_packet_bytes=session.peak_live_message_bytes,
            captured_emitted_generator_latents=captured,
        )

    def run(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        condition: str = "transport",
    ) -> CrossLayerChartModelExecution:
        def call_model() -> object:
            call_inputs: dict[str, object] = dict(model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            return self.adapter.module(**call_inputs)

        model_output, receipt = self.run_with_overlay(
            model_inputs,
            callback=call_model,
            condition=condition,
        )
        return CrossLayerChartModelExecution(
            model_output=model_output,
            condition=receipt.condition,
            layer_calls=receipt.layer_calls,
            transport_edge_count=receipt.transport_edge_count,
            peak_live_chart_packets=receipt.peak_live_chart_packets,
            peak_live_chart_packet_bytes=(
                receipt.peak_live_chart_packet_bytes
            ),
        )


__all__ = [
    "CrossLayerChartMessageSession",
    "CrossLayerChartModelExecution",
    "CrossLayerChartOverlayReceipt",
    "CrossLayerChartTransportEdge",
    "FactorizedChartGeneratorExecution",
    "Gemma3CrossLayerChartGraphExecutor",
    "Gemma3FactorizedChartGeneratorLayer",
]
