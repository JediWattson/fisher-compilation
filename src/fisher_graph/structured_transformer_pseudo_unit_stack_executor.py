"""Source-free structured spans with cross-layer carried pseudo-units.

One directed edge reuses an earlier MLP scalar at a later MLP.  The anchor
unit remains an ordinary gate/up/down coordinate in its layer.  The consumer
unit is physically removed from the later gate, up, and down projections; its
removed down column is retained as an explicit decoder applied to the carried
anchor scalar.  Consequently an edge removes exactly the consumer gate and up
rows: ``2 * residual_width`` stored coefficients and MACs per valid token.

The carried value is token-local and only moves forward through layer depth.
It is zeroed on invalid query rows before it can be consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import torch
from torch import Tensor, nn

from .activations import ActivationTrace, record
from .adapters.base import SegmentRun, SequenceContext, module_state_fingerprint
from .compiler.manifest import CompiledSegment
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)


_ARTIFACT_KIND = (
    "fisher_graph.structured_transformer_pseudo_unit_stack_executor"
)
_FORMAT_VERSION = 1
_FINGERPRINT_DOMAIN = (
    b"fisher_graph.structured_transformer_pseudo_unit_stack_executor.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LAYER_ID = re.compile(r"^layer\.([0-9]+)$")
_EDGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_ARTIFACT_FIELDS = {
    "artifact_kind",
    "format_version",
    "contains_source_model_weights",
    "contains_source_fallback",
    "layer_ids",
    "source_layer_configs",
    "source_parent_execution_fingerprints",
    "edges",
    "dtype",
    "model_state_dict",
    "execution_fingerprint",
}


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError(f"unsupported pseudo-unit stack dtype: {dtype}")
    return name


def _storage_pointers(module: nn.Module) -> set[int]:
    pointers = {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel() > 0
    }
    pointers.discard(0)
    return pointers


def _canonical_layer_ids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result:
        raise ValueError("pseudo-unit stack requires at least one layer")
    ordinals: list[int] = []
    for value in result:
        if not isinstance(value, str):
            raise TypeError("layer ids must be strings")
        match = _LAYER_ID.fullmatch(value)
        if match is None:
            raise ValueError(
                "pseudo-unit stack layer ids must use layer.<ordinal>"
            )
        ordinals.append(int(match.group(1)))
    if len(set(result)) != len(result):
        raise ValueError("pseudo-unit stack layer ids must be unique")
    if tuple(ordinals) != tuple(
        range(ordinals[0], ordinals[0] + len(ordinals))
    ):
        raise ValueError("pseudo-unit stack layers must be contiguous")
    return result


@dataclass(frozen=True, slots=True)
class PseudoUnitCarryEdge:
    """One earlier-to-later scalar reuse edge in source-unit coordinates."""

    edge_id: str
    anchor_layer_id: str
    anchor_source_index: int
    consumer_layer_id: str
    consumer_source_index: int
    consumer_decoder_scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.edge_id, str)
            or _EDGE_ID.fullmatch(self.edge_id) is None
        ):
            raise ValueError("edge_id must be a stable identifier")
        for label, value in (
            ("anchor_layer_id", self.anchor_layer_id),
            ("consumer_layer_id", self.consumer_layer_id),
        ):
            if not isinstance(value, str) or _LAYER_ID.fullmatch(value) is None:
                raise ValueError(f"{label} must use layer.<ordinal>")
        for label, value in (
            ("anchor_source_index", self.anchor_source_index),
            ("consumer_source_index", self.consumer_source_index),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be nonnegative")
        if (
            not isinstance(self.consumer_decoder_scale, (float, int))
            or isinstance(self.consumer_decoder_scale, bool)
            or not math.isfinite(float(self.consumer_decoder_scale))
            or float(self.consumer_decoder_scale) == 0.0
        ):
            raise ValueError(
                "consumer_decoder_scale must be finite and nonzero"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "anchor_layer_id": self.anchor_layer_id,
            "anchor_source_index": self.anchor_source_index,
            "consumer_layer_id": self.consumer_layer_id,
            "consumer_source_index": self.consumer_source_index,
            "consumer_decoder_scale": float(
                self.consumer_decoder_scale
            ),
        }

    @classmethod
    def from_metadata(
        cls,
        value: object,
    ) -> PseudoUnitCarryEdge:
        fields = {
            "edge_id",
            "anchor_layer_id",
            "anchor_source_index",
            "consumer_layer_id",
            "consumer_source_index",
            "consumer_decoder_scale",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("pseudo-unit edge metadata fields are invalid")
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StructuredTransformerPseudoUnitStackExecution:
    """Inspectable result of one contiguous stack execution."""

    output: Tensor
    layer_outputs: tuple[Tensor, ...]
    carried_features: tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class StructuredTransformerPseudoUnitStackAccounting:
    """Analytic coefficient and linear-MAC ledger for one stack call."""

    valid_tokens: int
    logical_causal_key_pairs: int
    source_parameter_count: int
    candidate_parameter_count: int
    removed_parameter_count: int
    carried_decoder_parameter_count: int
    source_logical_macs: int
    candidate_logical_macs: int
    removed_logical_macs: int
    carried_decoder_macs: int
    attention_projection_macs: int
    attention_score_macs: int
    attention_value_macs: int
    local_feed_forward_macs: int

    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self),
            "parameter_savings_semantics": (
                "consumer_gate_and_up_rows_only"
            ),
            "mac_savings_semantics": (
                "consumer_gate_and_up_rows_per_valid_token_only"
            ),
            "normalization_activation_masking_additions_and_softmax_excluded": (
                True
            ),
            "latency_measured": False,
            "kernel_speedup_claimed": False,
        }


class StructuredTransformerPseudoUnitStackExecutor(nn.Module):
    """Execute contiguous structured layers with forward-only scalar reuse."""

    def __init__(
        self,
        layer_ids: Sequence[str],
        parent_layers: Sequence[StructuredTransformerLayerExecutor],
        edges: Sequence[PseudoUnitCarryEdge],
    ) -> None:
        super().__init__()
        canonical_ids = _canonical_layer_ids(layer_ids)
        parents = tuple(parent_layers)
        if len(parents) != len(canonical_ids) or any(
            not isinstance(parent, StructuredTransformerLayerExecutor)
            for parent in parents
        ):
            raise TypeError(
                "parent_layers must provide one structured executor per layer"
            )
        if any(
            parent.owns_source_model_weights
            or any(module.training for module in parent.modules())
            for parent in parents
        ):
            raise ValueError(
                "pseudo-unit stack parents must be source-free eval executors"
            )
        dtype = parents[0].dtype
        device = parents[0].device
        width = parents[0].width
        if any(
            parent.dtype != dtype
            or parent.device != device
            or parent.width != width
            for parent in parents
        ):
            raise ValueError(
                "pseudo-unit stack parents must share dtype, device, and "
                "residual width"
            )
        _dtype_name(dtype)

        source_configs = tuple(parent.config for parent in parents)
        for layer_id, config in zip(
            canonical_ids,
            source_configs,
            strict=True,
        ):
            attention_stage, feed_forward_stage = config.transformer.stages
            feed_forward = config.transformer.feed_forward
            if (
                attention_stage.id != f"{layer_id}.attention"
                or attention_stage.input_site != f"{layer_id}.input"
                or feed_forward_stage.id != f"{layer_id}.feed_forward"
                or feed_forward_stage.output_site != f"{layer_id}.output"
                or feed_forward.projection_bias
            ):
                raise ValueError(
                    "parent layer semantics are incompatible with the "
                    "bias-free contiguous pseudo-unit stack"
                )

        parent_fingerprints = tuple(
            parent.execution_fingerprint() for parent in parents
        )
        parent_counts = tuple(
            parent.learned_parameter_count for parent in parents
        )
        source_parameter_count = sum(parent_counts)
        parent_storage = set().union(
            *(_storage_pointers(parent) for parent in parents)
        )

        parsed_edges = self._validate_and_order_edges(
            canonical_ids,
            source_configs,
            edges,
        )
        layer_positions = {
            layer_id: index for index, layer_id in enumerate(canonical_ids)
        }
        consumer_indices: list[list[int]] = [
            [] for _ in canonical_ids
        ]
        for edge in parsed_edges:
            consumer_indices[
                layer_positions[edge.consumer_layer_id]
            ].append(edge.consumer_source_index)
        consumers = tuple(
            tuple(sorted(values)) for values in consumer_indices
        )

        compressed_layers = tuple(
            self._copy_compressed_layer(parent, removed)
            for parent, removed in zip(parents, consumers, strict=True)
        )
        self.layers = nn.ModuleList(compressed_layers)
        if parent_storage & _storage_pointers(self):
            raise RuntimeError(
                "pseudo-unit stack candidate aliases parent tensor storage"
            )

        decoder_rows = []
        for edge in parsed_edges:
            consumer_position = layer_positions[edge.consumer_layer_id]
            parent_down = parents[
                consumer_position
            ].feed_forward.down_proj.weight
            decoder_rows.append(
                parent_down[:, edge.consumer_source_index].detach()
                * float(edge.consumer_decoder_scale)
            )
        decoder = torch.stack(decoder_rows, dim=0).to(
            device=device,
            dtype=dtype,
        )
        self.consumer_decoder_matrix = nn.Parameter(decoder.clone())

        self.layer_ids = canonical_ids
        self.edges = parsed_edges
        self._source_layer_configs = source_configs
        self._source_parent_execution_fingerprints = parent_fingerprints
        self._source_parameter_count = source_parameter_count
        self._source_intermediate_widths = tuple(
            config.transformer.feed_forward.intermediate_width
            for config in source_configs
        )
        self._retained_source_indices = tuple(
            tuple(
                source_index
                for source_index in range(source_width)
                if source_index not in set(removed)
            )
            for source_width, removed in zip(
                self._source_intermediate_widths,
                consumers,
                strict=True,
            )
        )
        self._source_to_local = tuple(
            {
                source_index: local_index
                for local_index, source_index in enumerate(retained)
            }
            for retained in self._retained_source_indices
        )
        self._anchor_edges_by_layer = tuple(
            tuple(
                edge_index
                for edge_index, edge in enumerate(self.edges)
                if edge.anchor_layer_id == layer_id
            )
            for layer_id in self.layer_ids
        )
        self._consumer_edges_by_layer = tuple(
            tuple(
                edge_index
                for edge_index, edge in enumerate(self.edges)
                if edge.consumer_layer_id == layer_id
            )
            for layer_id in self.layer_ids
        )
        expected_candidate_count = (
            source_parameter_count - 2 * width * len(self.edges)
        )
        if self.learned_parameter_count != expected_candidate_count:
            raise RuntimeError(
                "pseudo-unit stack parameter accounting drifted"
            )
        if tuple(
            parent.execution_fingerprint() for parent in parents
        ) != parent_fingerprints:
            raise RuntimeError("pseudo-unit stack mutated a parent executor")
        self.eval()

    @staticmethod
    def _validate_and_order_edges(
        layer_ids: tuple[str, ...],
        source_configs: tuple[
            StructuredTransformerLayerExecutorConfig,
            ...,
        ],
        edges: Sequence[PseudoUnitCarryEdge],
    ) -> tuple[PseudoUnitCarryEdge, ...]:
        values = tuple(edges)
        if not values:
            raise ValueError("pseudo-unit stack requires at least one edge")
        if any(not isinstance(edge, PseudoUnitCarryEdge) for edge in values):
            raise TypeError(
                "edges must contain PseudoUnitCarryEdge values"
            )
        positions = {
            layer_id: index for index, layer_id in enumerate(layer_ids)
        }
        endpoints: set[tuple[str, int]] = set()
        edge_ids: set[str] = set()
        for edge in values:
            if (
                edge.edge_id in edge_ids
                or edge.anchor_layer_id not in positions
                or edge.consumer_layer_id not in positions
            ):
                raise ValueError(
                    "edge ids must be unique and endpoints must be in the "
                    "stack"
                )
            anchor_position = positions[edge.anchor_layer_id]
            consumer_position = positions[edge.consumer_layer_id]
            if anchor_position >= consumer_position:
                raise ValueError(
                    "pseudo-unit carry edges must point strictly forward"
                )
            anchor_width = source_configs[
                anchor_position
            ].transformer.feed_forward.intermediate_width
            consumer_width = source_configs[
                consumer_position
            ].transformer.feed_forward.intermediate_width
            if (
                edge.anchor_source_index >= anchor_width
                or edge.consumer_source_index >= consumer_width
            ):
                raise ValueError("pseudo-unit edge source index is out of range")
            edge_endpoints = (
                (edge.anchor_layer_id, edge.anchor_source_index),
                (edge.consumer_layer_id, edge.consumer_source_index),
            )
            if any(endpoint in endpoints for endpoint in edge_endpoints):
                raise ValueError(
                    "pseudo-unit carry edge endpoints must be disjoint"
                )
            endpoints.update(edge_endpoints)
            edge_ids.add(edge.edge_id)
        return tuple(
            sorted(
                values,
                key=lambda edge: (
                    positions[edge.anchor_layer_id],
                    positions[edge.consumer_layer_id],
                    edge.anchor_source_index,
                    edge.consumer_source_index,
                    edge.edge_id,
                ),
            )
        )

    @staticmethod
    def _copy_compressed_layer(
        parent: StructuredTransformerLayerExecutor,
        removed_indices: tuple[int, ...],
    ) -> StructuredTransformerLayerExecutor:
        source_feed_forward = parent.config.transformer.feed_forward
        retained_width = (
            source_feed_forward.intermediate_width - len(removed_indices)
        )
        if retained_width <= 0:
            raise ValueError(
                "consumer removals must leave at least one local MLP unit"
            )
        config = replace(
            parent.config,
            transformer=replace(
                parent.config.transformer,
                feed_forward=replace(
                    source_feed_forward,
                    intermediate_width=retained_width,
                ),
            ),
        )
        with torch.random.fork_rng(devices=()):
            torch.manual_seed(0)
            candidate = StructuredTransformerLayerExecutor(
                config,
                dtype=parent.dtype,
                device="cpu",
            )
        source_state = parent.state_dict()
        destination_state = candidate.state_dict()
        source_width = source_feed_forward.intermediate_width
        removed = set(removed_indices)
        retained = torch.tensor(
            tuple(index for index in range(source_width) if index not in removed),
            dtype=torch.long,
            device=parent.device,
        )
        copied: dict[str, Tensor] = {}
        for name, destination in destination_state.items():
            source = source_state[name]
            if name in {
                "feed_forward.gate_proj.weight",
                "feed_forward.up_proj.weight",
            }:
                value = source.index_select(0, retained)
            elif name == "feed_forward.down_proj.weight":
                value = source.index_select(1, retained)
            else:
                value = source
            if value.shape != destination.shape:
                raise RuntimeError(
                    f"compressed layer state shape drifted for {name!r}"
                )
            copied[name] = value.detach().to(
                device="cpu",
                dtype=destination.dtype,
            ).clone()
        candidate.load_state_dict(copied, strict=True)
        candidate.to(device=parent.device)
        candidate.eval()
        return candidate

    @property
    def width(self) -> int:
        return self.layers[0].width

    @property
    def dtype(self) -> torch.dtype:
        return self.layers[0].dtype

    @property
    def device(self) -> torch.device:
        return self.layers[0].device

    @property
    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def fixed_runtime_coefficient_count(self) -> int:
        return 0

    @property
    def total_runtime_coefficient_count(self) -> int:
        return self.learned_parameter_count

    @property
    def source_parameter_count(self) -> int:
        return self._source_parameter_count

    @property
    def executor_local_source_free(self) -> bool:
        return True

    @property
    def owns_source_model_weights(self) -> bool:
        return False

    @property
    def owns_source_fallback(self) -> bool:
        return False

    @property
    def source_parent_execution_fingerprints(self) -> tuple[str, ...]:
        return self._source_parent_execution_fingerprints

    @staticmethod
    def carry_site(edge: PseudoUnitCarryEdge) -> str:
        return f"pseudo_unit_stack.edge.{edge.edge_id}.carry"

    @staticmethod
    def consumer_injection_site(edge: PseudoUnitCarryEdge) -> str:
        return (
            f"pseudo_unit_stack.edge.{edge.edge_id}.consumer_injection"
        )

    @property
    def capture_sites(self) -> frozenset[str]:
        edge_sites = {
            site
            for edge in self.edges
            for site in (
                self.carry_site(edge),
                self.consumer_injection_site(edge),
            )
        }
        layer_sites = {
            site
            for config in self._source_layer_configs
            for stage in config.transformer.stages
            for site in (
                stage.input_site,
                stage.normalized_input_site,
                stage.operator_output_site,
                stage.delta_site,
                stage.output_site,
            )
        }
        combined_sites = {
            f"{layer_id}.mlp.carried_injection"
            for layer_id, incoming in zip(
                self.layer_ids,
                self._consumer_edges_by_layer,
                strict=True,
            )
            if incoming
        }
        return frozenset((*edge_sites, *layer_sites, *combined_sites))

    def architecture_manifest(self) -> dict[str, object]:
        return {
            "kind": (
                "structured_transformer_cross_layer_pseudo_unit_stack"
            ),
            "format_version": _FORMAT_VERSION,
            "layer_ids": self.layer_ids,
            "source_layer_configs": tuple(
                config.to_dict() for config in self._source_layer_configs
            ),
            "compressed_layer_configs": tuple(
                layer.config.to_dict() for layer in self.layers
            ),
            "source_parent_execution_fingerprints": (
                self._source_parent_execution_fingerprints
            ),
            "edges": tuple(edge.metadata() for edge in self.edges),
            "source_parameter_count": self.source_parameter_count,
            "candidate_parameter_count": self.learned_parameter_count,
            "parameter_savings_per_edge": 2 * self.width,
            "carry_semantics": (
                "anchor_feature_computed_once_token_local_forward_only"
            ),
            "consumer_semantics": (
                "gate_up_down_unit_removed_then_explicit_decoder_injected"
            ),
            "prefill_only": True,
            "cache_supported": False,
            "contains_source_fallback": False,
            "latency_or_kernel_speed_claim": False,
        }

    def _validate_inputs(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
    ) -> None:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if sequence.phase != "prefill" or sequence.cache_state is not None:
            raise ValueError(
                "pseudo-unit stack supports cache-free prefill only"
            )
        if sequence.cache_positions is not None:
            raise ValueError(
                "pseudo-unit stack rejects cache positions"
            )
        if (
            not sequence.input_origin.attention_mask_supplied
            and sequence.query_length > 1
            and bool(
                (
                    sequence.logical_positions[:, 1:]
                    - sequence.logical_positions[:, :-1]
                    != 1
                ).any()
            )
        ):
            raise ValueError(
                "pseudo-unit stack cannot represent packed or gapped "
                "positions without an explicit attention mask"
            )
        expected = (
            sequence.batch_size,
            sequence.query_length,
            self.width,
        )
        if (
            not isinstance(hidden_states, Tensor)
            or not hidden_states.is_floating_point()
            or tuple(hidden_states.shape) != expected
            or hidden_states.dtype != self.dtype
            or hidden_states.device != self.device
            or sequence.device != self.device
        ):
            raise ValueError(
                "pseudo-unit stack input, sequence, dtype, or device is "
                "incompatible"
            )
        if sequence.key_length != sequence.query_length:
            raise ValueError(
                "pseudo-unit stack requires equal query and key lengths"
            )
        if not torch.equal(
            sequence.logical_positions,
            sequence.key_logical_positions,
        ):
            raise ValueError(
                "pseudo-unit stack requires identical query/key positions"
            )
        if not torch.equal(
            sequence.query_valid_mask,
            sequence.key_valid_mask,
        ):
            raise ValueError(
                "pseudo-unit stack requires identical query/key validity"
            )

    def forward_components(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> StructuredTransformerPseudoUnitStackExecution:
        self._validate_inputs(hidden_states, sequence)
        current = hidden_states
        carries: list[Tensor | None] = [None] * len(self.edges)
        layer_outputs: list[Tensor] = []
        valid = sequence.query_valid_mask.unsqueeze(-1)

        for layer_position, (layer_id, layer) in enumerate(
            zip(self.layer_ids, self.layers, strict=True)
        ):
            attention_stage, feed_forward_stage = (
                layer.config.transformer.stages
            )
            attention_prefix = (
                attention_stage.operator_output_site.rsplit(
                    ".operator_output",
                    1,
                )[0]
            )
            feed_forward_prefix = (
                feed_forward_stage.operator_output_site.rsplit(
                    ".operator_output",
                    1,
                )[0]
            )
            residual = record(trace, attention_stage.input_site, current)
            normalized_attention = record(
                trace,
                attention_stage.normalized_input_site,
                layer.attention_input_norm(residual),
            )
            attention_features = layer.attention_projection_features(
                normalized_attention,
                sequence,
                trace=trace,
                prefix=attention_prefix,
            )
            attention_operator = layer.attention.project(
                attention_features,
                trace=trace,
                prefix=attention_prefix,
            )
            if not layer.config.causal_edges_enabled:
                attention_operator = torch.zeros_like(attention_operator)
            attention_operator = record(
                trace,
                attention_stage.operator_output_site,
                attention_operator,
            )
            attention_delta = layer.attention_output_norm(
                attention_operator
            ).masked_fill(~valid, 0)
            attention_delta = record(
                trace,
                attention_stage.delta_site,
                attention_delta,
            )
            post_attention = record(
                trace,
                attention_stage.output_site,
                residual + attention_delta,
            )
            normalized_feed_forward = record(
                trace,
                feed_forward_stage.normalized_input_site,
                layer.feed_forward_input_norm(post_attention),
            )
            features = layer.feed_forward_projection_features(
                normalized_feed_forward,
                trace=trace,
                prefix=feed_forward_prefix,
            )

            for edge_index in self._anchor_edges_by_layer[layer_position]:
                edge = self.edges[edge_index]
                local_index = self._source_to_local[layer_position][
                    edge.anchor_source_index
                ]
                carried = features[..., local_index].masked_fill(
                    ~sequence.query_valid_mask,
                    0,
                )
                carries[edge_index] = record(
                    trace,
                    self.carry_site(edge),
                    carried,
                )

            feed_forward_operator = layer.feed_forward.down_proj(features)
            incoming = self._consumer_edges_by_layer[layer_position]
            if incoming:
                injections = []
                for edge_index in incoming:
                    carried = carries[edge_index]
                    if carried is None:
                        raise RuntimeError(
                            "consumer executed before its carried anchor"
                        )
                    injection = (
                        carried.unsqueeze(-1)
                        * self.consumer_decoder_matrix[edge_index]
                    )
                    injections.append(
                        record(
                            trace,
                            self.consumer_injection_site(
                                self.edges[edge_index]
                            ),
                            injection,
                        )
                    )
                combined = record(
                    trace,
                    f"{layer_id}.mlp.carried_injection",
                    torch.stack(injections, dim=0).sum(dim=0),
                )
                feed_forward_operator = (
                    feed_forward_operator + combined
                )
            feed_forward_operator = record(
                trace,
                feed_forward_stage.operator_output_site,
                feed_forward_operator,
            )
            feed_forward_delta = layer.feed_forward_output_norm(
                feed_forward_operator
            ).masked_fill(~valid, 0)
            feed_forward_delta = record(
                trace,
                feed_forward_stage.delta_site,
                feed_forward_delta,
            )
            current = record(
                trace,
                feed_forward_stage.output_site,
                post_attention + feed_forward_delta,
            )
            layer_outputs.append(current)

        if any(value is None for value in carries):
            raise RuntimeError("one or more pseudo-unit anchors did not execute")
        return StructuredTransformerPseudoUnitStackExecution(
            output=current,
            layer_outputs=tuple(layer_outputs),
            carried_features=tuple(
                value for value in carries if value is not None
            ),
        )

    def forward(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> Tensor:
        return self.forward_components(
            hidden_states,
            sequence,
            trace=trace,
        ).output

    def run(
        self,
        segment: CompiledSegment,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        if not isinstance(segment, CompiledSegment):
            raise TypeError("segment must be a CompiledSegment")
        first_attention = self._source_layer_configs[
            0
        ].transformer.stages[0]
        last_feed_forward = self._source_layer_configs[
            -1
        ].transformer.stages[1]
        if (
            segment.source_layers != self.layer_ids
            or segment.input_activation != first_attention.input_site
            or segment.output_activation != last_feed_forward.output_site
        ):
            raise ValueError(
                "compiled segment does not bind the complete pseudo-unit "
                "stack window"
            )
        result = self.forward_components(
            hidden_states,
            sequence,
            trace=trace,
        )
        return SegmentRun(
            hidden_states=result.output,
            sequence=sequence,
            raw_output={
                "edge_count": len(self.edges),
                "source_layers": self.layer_ids,
                "executor_local_source_free": True,
                "owns_source_fallback": False,
            },
        )

    def logical_accounting(
        self,
        sequence: SequenceContext,
    ) -> StructuredTransformerPseudoUnitStackAccounting:
        ledgers = tuple(
            layer.logical_accounting(sequence) for layer in self.layers
        )
        valid_tokens = int(sequence.query_valid_mask.sum().item())
        local_macs = sum(ledger.logical_total_macs for ledger in ledgers)
        decoder_parameters = len(self.edges) * self.width
        decoder_macs = valid_tokens * decoder_parameters
        candidate_macs = local_macs + decoder_macs
        removed_macs = valid_tokens * 2 * self.width * len(self.edges)
        candidate_parameters = self.learned_parameter_count
        removed_parameters = (
            self.source_parameter_count - candidate_parameters
        )
        expected_removed = 2 * self.width * len(self.edges)
        if removed_parameters != expected_removed:
            raise RuntimeError(
                "pseudo-unit stack parameter savings drifted"
            )
        return StructuredTransformerPseudoUnitStackAccounting(
            valid_tokens=valid_tokens,
            logical_causal_key_pairs=sum(
                ledger.logical_causal_key_pairs for ledger in ledgers
            ),
            source_parameter_count=self.source_parameter_count,
            candidate_parameter_count=candidate_parameters,
            removed_parameter_count=removed_parameters,
            carried_decoder_parameter_count=decoder_parameters,
            source_logical_macs=candidate_macs + removed_macs,
            candidate_logical_macs=candidate_macs,
            removed_logical_macs=removed_macs,
            carried_decoder_macs=decoder_macs,
            attention_projection_macs=sum(
                ledger.attention_projection_macs for ledger in ledgers
            ),
            attention_score_macs=sum(
                ledger.attention_score_macs for ledger in ledgers
            ),
            attention_value_macs=sum(
                ledger.attention_value_macs for ledger in ledgers
            ),
            local_feed_forward_macs=sum(
                ledger.feed_forward_macs for ledger in ledgers
            ),
        )

    def execution_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_DOMAIN)
        digest.update(
            json.dumps(
                {
                    "architecture": self.architecture_manifest(),
                    "module_training": tuple(
                        (name, module.training)
                        for name, module in self.named_modules()
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(module_state_fingerprint(self).encode("ascii"))
        return digest.hexdigest()

    def artifact_state_dict(self) -> dict[str, object]:
        if any(module.training for module in self.modules()):
            raise RuntimeError(
                "pseudo-unit stack artifacts require every module in eval "
                "mode"
            )
        if self.owns_source_model_weights or self.owns_source_fallback:
            raise RuntimeError(
                "pseudo-unit stack artifacts must remain source-free"
            )
        state = {
            name: value.detach().to(device="cpu").clone()
            for name, value in self.state_dict().items()
        }
        for name, value in state.items():
            if name.endswith("._weight_origin"):
                valid = (
                    value.dtype is torch.uint8
                    and value.ndim == 0
                    and int(value.item()) == 0
                )
            else:
                valid = value.is_floating_point() and bool(
                    torch.isfinite(value).all()
                )
            if not valid:
                raise ValueError(
                    f"pseudo-unit stack state {name!r} is invalid"
                )
        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "contains_source_model_weights": False,
            "contains_source_fallback": False,
            "layer_ids": self.layer_ids,
            "source_layer_configs": tuple(
                config.to_dict() for config in self._source_layer_configs
            ),
            "source_parent_execution_fingerprints": (
                self._source_parent_execution_fingerprints
            ),
            "edges": tuple(edge.metadata() for edge in self.edges),
            "dtype": _dtype_name(self.dtype),
            "model_state_dict": state,
            "execution_fingerprint": self.execution_fingerprint(),
        }

    @classmethod
    def from_artifact_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        map_location: torch.device | str = "cpu",
    ) -> StructuredTransformerPseudoUnitStackExecutor:
        if not isinstance(state, Mapping) or set(state) != _ARTIFACT_FIELDS:
            raise ValueError("pseudo-unit stack artifact fields are invalid")
        dtype_name = state["dtype"]
        if (
            state["artifact_kind"] != _ARTIFACT_KIND
            or state["format_version"] != _FORMAT_VERSION
            or state["contains_source_model_weights"] is not False
            or state["contains_source_fallback"] is not False
            or not isinstance(dtype_name, str)
            or dtype_name not in _DTYPES
            or not isinstance(state["execution_fingerprint"], str)
            or _SHA256.fullmatch(state["execution_fingerprint"]) is None
        ):
            raise ValueError("unsupported pseudo-unit stack artifact")
        layer_ids_raw = state["layer_ids"]
        configs_raw = state["source_layer_configs"]
        fingerprints_raw = state["source_parent_execution_fingerprints"]
        edges_raw = state["edges"]
        if (
            not isinstance(layer_ids_raw, tuple)
            or not isinstance(configs_raw, tuple)
            or not isinstance(fingerprints_raw, tuple)
            or not isinstance(edges_raw, tuple)
            or len(configs_raw) != len(layer_ids_raw)
            or len(fingerprints_raw) != len(layer_ids_raw)
            or any(
                not isinstance(value, str)
                or _SHA256.fullmatch(value) is None
                for value in fingerprints_raw
            )
        ):
            raise ValueError(
                "pseudo-unit stack artifact topology is invalid"
            )
        configs = tuple(
            StructuredTransformerLayerExecutorConfig.from_dict(value)
            for value in configs_raw
        )
        edges = tuple(
            PseudoUnitCarryEdge.from_metadata(value)
            for value in edges_raw
        )
        dtype = _DTYPES[dtype_name]
        with torch.random.fork_rng(devices=()):
            torch.manual_seed(0)
            parents = tuple(
                StructuredTransformerLayerExecutor(
                    config,
                    dtype=dtype,
                    device="cpu",
                ).eval()
                for config in configs
            )
            result = cls(layer_ids_raw, parents, edges)
        result._source_parent_execution_fingerprints = tuple(
            fingerprints_raw
        )

        raw_model_state = state["model_state_dict"]
        if not isinstance(raw_model_state, Mapping):
            raise ValueError(
                "pseudo-unit stack artifact model state is invalid"
            )
        expected = result.state_dict()
        if set(raw_model_state) != set(expected):
            raise ValueError(
                "pseudo-unit stack model-state fields are invalid"
            )
        restored: dict[str, Tensor] = {}
        for name, expected_value in expected.items():
            value = raw_model_state[name]
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.shape != expected_value.shape
                or value.dtype != expected_value.dtype
            ):
                raise ValueError(
                    f"pseudo-unit stack state {name!r} has invalid schema"
                )
            if name.endswith("._weight_origin"):
                valid = (
                    value.dtype is torch.uint8
                    and value.ndim == 0
                    and int(value.item()) == 0
                )
            else:
                valid = value.is_floating_point() and bool(
                    torch.isfinite(value).all()
                )
            if not valid:
                raise ValueError(
                    f"pseudo-unit stack state {name!r} is invalid"
                )
            restored[name] = value.clone()
        result.load_state_dict(restored, strict=True)
        result.to(device=map_location)
        result.eval()
        if result.execution_fingerprint() != state["execution_fingerprint"]:
            raise ValueError(
                "pseudo-unit stack execution fingerprint mismatch"
            )
        return result


__all__ = [
    "PseudoUnitCarryEdge",
    "StructuredTransformerPseudoUnitStackAccounting",
    "StructuredTransformerPseudoUnitStackExecution",
    "StructuredTransformerPseudoUnitStackExecutor",
]
