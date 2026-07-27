"""Fused Gemma executor for dense cross-block fan-in/fan-out groups.

Each fanout group reconstructs several later MLP coordinates from several
earlier, native MLP coordinates.  The analysis-time coordinate mixer has
already been folded through the consumers' native down-projection columns by
the core planner.  Runtime therefore stores only one dense decoder ``[d, A]``
per target layer:

* consumer gate and up rows are removed;
* consumer down columns are removed;
* carried native anchors are appended to the retained MLP features; and
* one contiguous down projection consumes both retained and carried features.

The unchanged Gemma model continues to execute attention, normalization,
residual connections, embeddings, and the language-model head.  Removed
consumer features are never materialized by the production executor.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .structured_mlp_cross_block_bundling import ModeKey
from .structured_mlp_cross_block_fanout import (
    CrossBlockFanoutGroup,
    GlobalCrossBlockFanoutPlan,
)


_SUPPORTED_ACTIVATIONS = frozenset(
    ("gelu", "gelu_pytorch_tanh", "silu", "swish")
)
_CONDITIONS = frozenset(("merged", "deletion"))


def _apply_activation(name: str, values: Tensor) -> Tensor:
    if name == "gelu_pytorch_tanh":
        return F.gelu(values, approximate="tanh")
    if name == "gelu":
        return F.gelu(values)
    if name in ("silu", "swish"):
        return F.silu(values)
    raise ValueError(f"unsupported Gemma MLP activation: {name!r}")


def _validate_source_mlp(
    module: nn.Module,
    *,
    label: str,
) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    down = getattr(module, "down_proj", None)
    if not all(isinstance(value, nn.Linear) for value in (gate, up, down)):
        raise TypeError(
            f"{label} must expose linear gate_proj, up_proj, and down_proj"
        )
    assert isinstance(gate, nn.Linear)
    assert isinstance(up, nn.Linear)
    assert isinstance(down, nn.Linear)
    if any(value.bias is not None for value in (gate, up, down)):
        raise ValueError("Gemma fanout fusion requires bias-free MLPs")
    if (
        gate.weight.shape != up.weight.shape
        or down.in_features != gate.out_features
        or down.out_features != gate.in_features
        or any(
            value.weight.dtype != gate.weight.dtype
            or value.weight.device != gate.weight.device
            for value in (up, down)
        )
    ):
        raise ValueError(f"{label} projection shapes or tensor types disagree")
    return gate, up, down


def _storage_pointers(module: nn.Module) -> set[int]:
    values = {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel()
    }
    values.discard(0)
    return values


def _coordinate(key: ModeKey) -> tuple[int, int]:
    return key.layer_ordinal, key.mode_index


def _target_layer(group: CrossBlockFanoutGroup) -> int:
    targets = {consumer.layer_ordinal for consumer in group.consumers}
    if len(targets) != 1:
        raise ValueError(
            "a fused fanout group must keep every consumer in one layer"
        )
    return next(iter(targets))


def _validate_plan_layer_catalog(
    adapter: Gemma3CausalLMAdapter,
    plan: GlobalCrossBlockFanoutPlan,
) -> None:
    if len(plan.layer_specs) != len(adapter.layers):
        raise ValueError("fanout plan layer catalog does not cover the model")
    for plan_spec, adapter_spec in zip(
        plan.layer_specs,
        adapter.layers,
        strict=True,
    ):
        transformer = adapter_spec.transformer
        if transformer is None or transformer.operator_sites is None:
            raise ValueError("Gemma layer lacks structured operator metadata")
        source_layer = adapter.source_module(adapter_spec.id)
        source_mlp = getattr(source_layer, "mlp", None)
        if not isinstance(source_mlp, nn.Module):
            raise TypeError("Gemma source layer does not expose an MLP")
        gate, _, _ = _validate_source_mlp(
            source_mlp,
            label=f"source layer {adapter_spec.ordinal} MLP",
        )
        if (
            plan_spec.layer_ordinal != adapter_spec.ordinal
            or plan_spec.layer_id != adapter_spec.id
            or plan_spec.activation_site
            != transformer.operator_sites.feed_forward_down_input
            or plan_spec.width != gate.out_features
        ):
            raise ValueError(
                "fanout plan layer catalog does not bind the live adapter"
            )


class Gemma3GlobalFanoutMLP(nn.Module):
    """Copied Gemma MLP with native consumers replaced by carried inputs."""

    def __init__(
        self,
        source_mlp: nn.Module,
        *,
        consumer_source_indices: tuple[int, ...],
        fused_decoder: Tensor,
        activation: str,
    ) -> None:
        super().__init__()
        if (
            not isinstance(activation, str)
            or activation not in _SUPPORTED_ACTIVATIONS
        ):
            raise ValueError("unsupported Gemma MLP activation")
        if source_mlp.training or any(
            parameter.requires_grad for parameter in source_mlp.parameters()
        ):
            raise ValueError(
                "fanout compilation requires a frozen eval source MLP"
            )
        gate, up, down = _validate_source_mlp(
            source_mlp,
            label="source_mlp",
        )
        residual_width = gate.in_features
        intermediate_width = gate.out_features
        consumers = tuple(sorted(consumer_source_indices))
        if (
            len(consumers) != len(set(consumers))
            or any(
                type(index) is not int
                or index < 0
                or index >= intermediate_width
                for index in consumers
            )
            or len(consumers) >= intermediate_width
        ):
            raise ValueError("consumer source indices are invalid")
        if (
            not isinstance(fused_decoder, Tensor)
            or fused_decoder.ndim != 2
            or fused_decoder.shape[0] != residual_width
            or not fused_decoder.is_floating_point()
            or not bool(torch.isfinite(fused_decoder).all())
        ):
            raise ValueError(
                "fused_decoder must be finite floating [residual, anchors]"
            )
        carried_width = fused_decoder.shape[1]
        retained = tuple(
            index
            for index in range(intermediate_width)
            if index not in consumers
        )
        retained_tensor = torch.tensor(
            retained,
            dtype=torch.long,
            device=gate.weight.device,
        )
        source_fingerprint = module_state_fingerprint(source_mlp)
        source_storage = _storage_pointers(source_mlp)

        self.gate_proj = nn.Linear(
            residual_width,
            len(retained),
            bias=False,
            dtype=gate.weight.dtype,
            device=gate.weight.device,
        )
        self.up_proj = nn.Linear(
            residual_width,
            len(retained),
            bias=False,
            dtype=gate.weight.dtype,
            device=gate.weight.device,
        )
        self.down_proj = nn.Linear(
            len(retained) + carried_width,
            residual_width,
            bias=False,
            dtype=gate.weight.dtype,
            device=gate.weight.device,
        )
        self.register_buffer(
            "retained_source_indices",
            retained_tensor,
            persistent=False,
        )
        runtime_decoder = fused_decoder.detach().to(
            device=gate.weight.device,
            dtype=gate.weight.dtype,
        )
        if not bool(torch.isfinite(runtime_decoder).all()):
            raise ValueError(
                "fused decoder is not finite in the runtime model dtype"
            )
        with torch.no_grad():
            self.gate_proj.weight.copy_(
                gate.weight.index_select(0, retained_tensor)
            )
            self.up_proj.weight.copy_(
                up.weight.index_select(0, retained_tensor)
            )
            self.down_proj.weight[:, : len(retained)].copy_(
                down.weight.index_select(1, retained_tensor)
            )
            if carried_width:
                self.down_proj.weight[:, len(retained) :].copy_(
                    runtime_decoder
                )

        self.activation = activation
        self.residual_width = residual_width
        self.source_intermediate_width = intermediate_width
        self.retained_width = len(retained)
        self.carried_width = carried_width
        self.consumer_source_indices = consumers
        self.source_mlp_fingerprint = source_fingerprint
        if source_storage & _storage_pointers(self):
            raise RuntimeError("compiled fanout MLP aliases source storage")
        if module_state_fingerprint(source_mlp) != source_fingerprint:
            raise RuntimeError("compiled fanout MLP mutated its source")
        expected = (
            2 * residual_width * len(retained)
            + residual_width * (len(retained) + carried_width)
        )
        if sum(parameter.numel() for parameter in self.parameters()) != expected:
            raise RuntimeError("compiled fanout MLP accounting drifted")
        if not self.down_proj.weight.is_contiguous():
            raise RuntimeError("fused down projection must be contiguous")
        self.requires_grad_(False)
        self.eval()

    @property
    def dtype(self) -> torch.dtype:
        return self.gate_proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.gate_proj.weight.device

    def retained_features(self, values: Tensor) -> Tensor:
        if (
            not isinstance(values, Tensor)
            or values.ndim != 3
            or values.shape[-1] != self.residual_width
            or values.dtype != self.dtype
            or values.device != self.device
        ):
            raise ValueError(
                "MLP input must be a colocated "
                "[batch, sequence, residual_width] tensor"
            )
        return _apply_activation(
            self.activation,
            self.gate_proj(values),
        ) * self.up_proj(values)

    def down_from_features(
        self,
        retained_features: Tensor,
        carried_features: Tensor,
    ) -> Tensor:
        if (
            retained_features.ndim != 3
            or retained_features.shape[-1] != self.retained_width
            or carried_features.shape
            != (*retained_features.shape[:-1], self.carried_width)
            or retained_features.dtype != self.dtype
            or carried_features.dtype != self.dtype
            or retained_features.device != self.device
            or carried_features.device != self.device
        ):
            raise ValueError("retained and carried feature tensors are invalid")
        fused = torch.cat((retained_features, carried_features), dim=-1)
        return self.down_proj(fused)

    def forward(
        self,
        normalized_hidden_states: Tensor,
        carried_features: Tensor,
    ) -> Tensor:
        retained = self.retained_features(normalized_hidden_states)
        return self.down_from_features(retained, carried_features)


@dataclass(slots=True)
class _ExecutionState:
    valid_positions: Tensor
    condition: str
    root_features: dict[tuple[int, int], Tensor]
    layer_calls: dict[int, int]


class _GlobalFanoutMLPOverlay(nn.Module):
    def __init__(
        self,
        compiled: Gemma3GlobalFanoutMLP,
        *,
        layer_ordinal: int,
        incoming_anchor_coordinates: tuple[tuple[int, int], ...],
        outgoing_root_indices: tuple[int, ...],
        release_root_coordinates: tuple[tuple[int, int], ...],
        state: _ExecutionState,
    ) -> None:
        super().__init__()
        self.compiled = compiled
        self.layer_ordinal = layer_ordinal
        self.incoming_anchor_coordinates = incoming_anchor_coordinates
        self.outgoing_root_indices = outgoing_root_indices
        self.release_root_coordinates = release_root_coordinates
        self._state = state
        self.eval()

    @property
    def gate_proj(self) -> nn.Linear:
        return self.compiled.gate_proj

    @property
    def up_proj(self) -> nn.Linear:
        return self.compiled.up_proj

    @property
    def down_proj(self) -> nn.Linear:
        return self.compiled.down_proj

    def forward(self, normalized_hidden_states: Tensor) -> Tensor:
        calls = self._state.layer_calls.get(self.layer_ordinal, 0)
        if calls:
            raise RuntimeError("each fanout MLP overlay may execute only once")
        retained = self.compiled.retained_features(normalized_hidden_states)
        retained_lookup = {
            int(source_index): retained_index
            for retained_index, source_index in enumerate(
                self.compiled.retained_source_indices.tolist()
            )
        }
        invalid = ~self._state.valid_positions
        for source_index in self.outgoing_root_indices:
            try:
                retained_index = retained_lookup[source_index]
            except KeyError as error:
                raise RuntimeError(
                    "a removed consumer cannot serve as a native root"
                ) from error
            root = retained[..., retained_index].masked_fill(invalid, 0)
            self._state.root_features[
                (self.layer_ordinal, source_index)
            ] = root

        if self.incoming_anchor_coordinates:
            if self._state.condition == "deletion":
                carried = torch.zeros(
                    *retained.shape[:-1],
                    len(self.incoming_anchor_coordinates),
                    dtype=retained.dtype,
                    device=retained.device,
                )
            else:
                roots: list[Tensor] = []
                for coordinate in self.incoming_anchor_coordinates:
                    try:
                        roots.append(self._state.root_features[coordinate])
                    except KeyError as error:
                        raise RuntimeError(
                            "a consumer executed before its native fanout root"
                        ) from error
                carried = torch.stack(roots, dim=-1).masked_fill(
                    invalid.unsqueeze(-1),
                    0,
                )
        else:
            carried = torch.empty(
                *retained.shape[:-1],
                0,
                dtype=retained.dtype,
                device=retained.device,
            )
        output = self.compiled.down_from_features(retained, carried)
        for coordinate in self.release_root_coordinates:
            try:
                del self._state.root_features[coordinate]
            except KeyError as error:
                raise RuntimeError(
                    "a fanout root expired before its final consumer"
                ) from error
        self._state.layer_calls[self.layer_ordinal] = calls + 1
        return output


@dataclass(frozen=True, slots=True)
class Gemma3GlobalFanoutModelExecution:
    model_output: object
    condition: str
    group_count: int
    consumer_count: int
    native_root_count: int
    fused_anchor_input_count: int
    affected_layer_count: int
    source_whole_model_learned_parameters: int
    candidate_whole_model_learned_parameters: int
    native_removed_learned_parameters: int
    fused_decoder_coefficients: int
    net_stored_coefficient_savings: int
    valid_tokens: int
    logical_linear_macs_native_removed: int
    logical_linear_macs_fused_decoder: int
    net_logical_macs_saved: int
    peak_live_root_scalars_per_token: int

    @property
    def removed_learned_parameters(self) -> int:
        return self.native_removed_learned_parameters


class Gemma3GlobalCrossBlockFanoutExecutor(nn.Module):
    """Overlay every affected Gemma MLP during one native model prefill."""

    def __init__(
        self,
        adapter: Gemma3CausalLMAdapter,
        plan: GlobalCrossBlockFanoutPlan,
    ) -> None:
        super().__init__()
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if not isinstance(plan, GlobalCrossBlockFanoutPlan):
            raise TypeError("plan must be a GlobalCrossBlockFanoutPlan")
        plan.validate_integrity()
        if not plan.groups:
            raise ValueError("global fanout execution requires a nonempty plan")
        _validate_plan_layer_catalog(adapter, plan)
        if adapter.model_fingerprint() != plan.source_model_fingerprint:
            raise ValueError("fanout plan does not bind the live Gemma model")
        model = adapter.module
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise ValueError(
                "global fanout execution requires a frozen eval source model"
            )

        groups_by_target: defaultdict[
            int,
            list[CrossBlockFanoutGroup],
        ] = defaultdict(list)
        outgoing: defaultdict[int, set[int]] = defaultdict(set)
        removed_coordinates: set[tuple[int, int]] = set()
        root_keys: dict[tuple[int, int], ModeKey] = {}
        consumer_count = 0
        for group in plan.groups:
            if not isinstance(group, CrossBlockFanoutGroup):
                raise TypeError("fanout plan groups are invalid")
            target = _target_layer(group)
            groups_by_target[target].append(group)
            consumer_count += len(group.consumers)
            for consumer in group.consumers:
                coordinate = _coordinate(consumer)
                if coordinate in removed_coordinates:
                    raise ValueError(
                        "a native consumer may occur in only one fanout group"
                    )
                removed_coordinates.add(coordinate)
            for anchor in group.anchors:
                coordinate = _coordinate(anchor)
                root_keys.setdefault(coordinate, anchor)
                outgoing[anchor.layer_ordinal].add(anchor.mode_index)
                if anchor.layer_ordinal >= target:
                    raise ValueError("fanout roots must point strictly forward")
        if any(coordinate in removed_coordinates for coordinate in root_keys):
            raise ValueError("a removed consumer cannot be a fanout root")

        incoming: dict[int, tuple[tuple[int, int], ...]] = {}
        fused_decoders: dict[int, Tensor] = {}
        for target, groups in groups_by_target.items():
            columns: dict[tuple[int, int], Tensor] = {}
            for group in groups:
                for anchor_index, anchor in enumerate(group.anchors):
                    coordinate = _coordinate(anchor)
                    column = group.fused_decoder[
                        :,
                        anchor_index,
                    ].detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                    if coordinate in columns:
                        columns[coordinate] = columns[coordinate] + column
                    else:
                        columns[coordinate] = column.clone()
            coordinates = tuple(sorted(columns))
            incoming[target] = coordinates
            fused_decoders[target] = torch.stack(
                tuple(columns[coordinate] for coordinate in coordinates),
                dim=1,
            )

        affected = tuple(
            sorted(set(groups_by_target) | set(outgoing))
        )
        compiled: dict[str, Gemma3GlobalFanoutMLP] = {}
        source_fingerprints: dict[int, str] = {}
        residual_width: int | None = None
        intermediate_width: int | None = None
        for ordinal in affected:
            layer_spec = adapter.layers[ordinal]
            if (
                layer_spec.transformer is None
                or layer_spec.transformer.feed_forward is None
            ):
                raise ValueError("Gemma layer lacks structured MLP metadata")
            feed_forward = layer_spec.transformer.feed_forward
            source_layer = adapter.source_module(layer_spec.id)
            source_mlp = getattr(source_layer, "mlp", None)
            if not isinstance(source_mlp, nn.Module):
                raise TypeError("Gemma source layer does not expose an MLP")
            gate, _, _ = _validate_source_mlp(
                source_mlp,
                label=f"source layer {ordinal} MLP",
            )
            layer_consumers = tuple(
                sorted(
                    consumer.mode_index
                    for group in groups_by_target.get(ordinal, ())
                    for consumer in group.consumers
                )
            )
            decoder = fused_decoders.get(
                ordinal,
                torch.empty(
                    gate.in_features,
                    0,
                    dtype=torch.float64,
                ),
            )
            candidate = Gemma3GlobalFanoutMLP(
                source_mlp,
                consumer_source_indices=layer_consumers,
                fused_decoder=decoder,
                activation=feed_forward.activation,
            )
            if residual_width is None:
                residual_width = candidate.residual_width
                intermediate_width = candidate.source_intermediate_width
            elif (
                residual_width != candidate.residual_width
                or intermediate_width != candidate.source_intermediate_width
            ):
                raise ValueError(
                    "global fanout executor requires equal Gemma MLP widths"
                )
            compiled[str(ordinal)] = candidate
            source_fingerprints[ordinal] = module_state_fingerprint(source_mlp)
        assert residual_width is not None
        assert intermediate_width is not None

        last_consumer: dict[tuple[int, int], int] = {}
        for target, coordinates in incoming.items():
            for coordinate in coordinates:
                last_consumer[coordinate] = max(
                    last_consumer.get(coordinate, -1),
                    target,
                )
        release: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for coordinate, target in last_consumer.items():
            release[target].append(coordinate)

        self.adapter = adapter
        self.plan = plan
        self.compiled_mlps = nn.ModuleDict(compiled)
        self._incoming = incoming
        self._outgoing = {
            ordinal: tuple(sorted(values))
            for ordinal, values in outgoing.items()
        }
        self._release = {
            ordinal: tuple(sorted(values))
            for ordinal, values in release.items()
        }
        self._last_consumer = last_consumer
        self._source_fingerprints = source_fingerprints
        self._affected_ordinals = affected
        self.residual_width = residual_width
        self.source_intermediate_width = intermediate_width
        self._consumer_count = consumer_count
        self._native_root_count = len(root_keys)
        self._fused_anchor_input_count = sum(
            len(values) for values in incoming.values()
        )
        if (
            self._consumer_count != plan.consumer_count
            or 3 * residual_width * self._consumer_count
            != plan.native_removed_parameter_count
            or residual_width * self._fused_anchor_input_count
            != plan.fused_decoder_parameter_count
        ):
            raise RuntimeError("fanout plan and runtime accounting disagree")
        self._active = False
        self.requires_grad_(False)
        self.eval()
        self._validate_live_source()

    @property
    def group_count(self) -> int:
        return len(self.plan.groups)

    @property
    def consumer_count(self) -> int:
        return self._consumer_count

    @property
    def native_root_count(self) -> int:
        return self._native_root_count

    @property
    def fused_anchor_input_count(self) -> int:
        return self._fused_anchor_input_count

    def _validate_live_source(self) -> None:
        self.plan.validate_integrity()
        _validate_plan_layer_catalog(self.adapter, self.plan)
        if self.adapter.model_fingerprint() != self.plan.source_model_fingerprint:
            raise ValueError("live Gemma model fingerprint drifted")
        for ordinal, expected in self._source_fingerprints.items():
            layer = self.adapter.source_module(
                self.adapter.layers[ordinal].id
            )
            mlp = getattr(layer, "mlp", None)
            if (
                not isinstance(mlp, nn.Module)
                or module_state_fingerprint(mlp) != expected
            ):
                raise ValueError("live Gemma MLP state differs from compilation")

    def _peak_live_roots(self) -> int:
        peak = 0
        for ordinal in range(len(self.adapter.layers)):
            live = sum(
                anchor_ordinal <= ordinal <= last
                for (anchor_ordinal, _), last in self._last_consumer.items()
            )
            peak = max(peak, live)
        return peak

    def run(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        condition: str = "merged",
    ) -> Gemma3GlobalFanoutModelExecution:
        if condition not in _CONDITIONS:
            raise ValueError("condition must be 'merged' or 'deletion'")
        if self._active:
            raise RuntimeError("global fanout execution is not reentrant")
        self._validate_live_source()
        context = self.adapter.prepare_sequence(model_inputs)
        first_compiled = self.compiled_mlps[str(self._affected_ordinals[0])]
        valid = context.query_valid_mask.to(device=first_compiled.device)
        layers = getattr(getattr(self.adapter.module, "model"), "layers")
        source_parameters = sum(
            parameter.numel()
            for parameter in self.adapter.module.parameters()
        )
        originals: dict[int, nn.Module] = {}
        state = _ExecutionState(
            valid_positions=valid,
            condition=condition,
            root_features={},
            layer_calls={},
        )
        self._active = True
        try:
            for ordinal in self._affected_ordinals:
                original = getattr(layers[ordinal], "mlp")
                if not isinstance(original, nn.Module):
                    raise TypeError("live Gemma layer MLP is invalid")
                originals[ordinal] = original
                layers[ordinal].mlp = _GlobalFanoutMLPOverlay(
                    self.compiled_mlps[str(ordinal)],
                    layer_ordinal=ordinal,
                    incoming_anchor_coordinates=self._incoming.get(
                        ordinal,
                        (),
                    ),
                    outgoing_root_indices=self._outgoing.get(ordinal, ()),
                    release_root_coordinates=self._release.get(ordinal, ()),
                    state=state,
                )
            candidate_parameters = sum(
                parameter.numel()
                for parameter in self.adapter.module.parameters()
            )
            native_removed = (
                3 * self.residual_width * self.consumer_count
            )
            decoder_coefficients = (
                self.residual_width * self.fused_anchor_input_count
            )
            expected = (
                source_parameters - native_removed + decoder_coefficients
            )
            if candidate_parameters != expected:
                raise RuntimeError(
                    "global fanout model parameter accounting drifted"
                )
            call_inputs: dict[str, object] = dict(model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            model_output = self.adapter.module(**call_inputs)
        finally:
            for ordinal, original in originals.items():
                layers[ordinal].mlp = original
            self._active = False
        if set(state.layer_calls) != set(self._affected_ordinals) or any(
            calls != 1 for calls in state.layer_calls.values()
        ):
            raise RuntimeError("not every global fanout overlay executed once")
        if state.root_features:
            raise RuntimeError("fanout root lifetime accounting leaked tensors")
        self._validate_live_source()

        valid_tokens = int(valid.sum().item())
        native_removed = 3 * self.residual_width * self.consumer_count
        decoder_coefficients = (
            self.residual_width * self.fused_anchor_input_count
        )
        native_removed_macs = valid_tokens * native_removed
        decoder_macs = valid_tokens * decoder_coefficients
        return Gemma3GlobalFanoutModelExecution(
            model_output=model_output,
            condition=condition,
            group_count=self.group_count,
            consumer_count=self.consumer_count,
            native_root_count=self.native_root_count,
            fused_anchor_input_count=self.fused_anchor_input_count,
            affected_layer_count=len(self._affected_ordinals),
            source_whole_model_learned_parameters=source_parameters,
            candidate_whole_model_learned_parameters=candidate_parameters,
            native_removed_learned_parameters=native_removed,
            fused_decoder_coefficients=decoder_coefficients,
            net_stored_coefficient_savings=(
                native_removed - decoder_coefficients
            ),
            valid_tokens=valid_tokens,
            logical_linear_macs_native_removed=native_removed_macs,
            logical_linear_macs_fused_decoder=decoder_macs,
            net_logical_macs_saved=native_removed_macs - decoder_macs,
            peak_live_root_scalars_per_token=self._peak_live_roots(),
        )

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
    ) -> Gemma3GlobalFanoutModelExecution:
        return self.run(model_inputs, condition="merged")


__all__ = [
    "Gemma3GlobalCrossBlockFanoutExecutor",
    "Gemma3GlobalFanoutMLP",
    "Gemma3GlobalFanoutModelExecution",
]
