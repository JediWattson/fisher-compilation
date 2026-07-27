"""Native Gemma full-model executor for an uncapped cross-block merge forest.

Every selected consumer physically drops one gate row and one up row.  Its
native down-projection column remains and receives a token-local scalar carried
from an earlier retained native root.  One root may fan out to any number of
later consumers.  The unchanged Gemma model continues to execute attention,
normalization, residuals, embeddings, and the language-model head.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .structured_mlp_global_cross_block_merge import (
    DirectedCrossBlockMerge,
    GlobalCrossBlockMergePlan,
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
        raise ValueError("Gemma global merging requires bias-free MLPs")
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


class Gemma3GlobalMergedMLP(nn.Module):
    """One copied Gemma MLP with an arbitrary set of generator rows removed."""

    def __init__(
        self,
        source_mlp: nn.Module,
        *,
        consumer_source_indices: tuple[int, ...],
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
                "global merge compilation requires a frozen eval source MLP"
            )
        gate, up, down = _validate_source_mlp(
            source_mlp,
            label="source_mlp",
        )
        width = gate.in_features
        intermediate = gate.out_features
        consumers = tuple(sorted(consumer_source_indices))
        if (
            len(consumers) != len(set(consumers))
            or any(
                type(index) is not int
                or index < 0
                or index >= intermediate
                for index in consumers
            )
            or len(consumers) >= intermediate
        ):
            raise ValueError("consumer source indices are invalid")
        retained = tuple(
            index for index in range(intermediate) if index not in consumers
        )
        retained_tensor = torch.tensor(
            retained,
            dtype=torch.long,
            device=gate.weight.device,
        )
        source_fingerprint = module_state_fingerprint(source_mlp)
        source_storage = _storage_pointers(source_mlp)
        self.gate_proj = nn.Linear(
            width,
            len(retained),
            bias=False,
            dtype=gate.weight.dtype,
            device=gate.weight.device,
        )
        self.up_proj = nn.Linear(
            width,
            len(retained),
            bias=False,
            dtype=gate.weight.dtype,
            device=gate.weight.device,
        )
        self.down_proj = nn.Linear(
            intermediate,
            width,
            bias=False,
            dtype=gate.weight.dtype,
            device=gate.weight.device,
        )
        self.register_buffer(
            "retained_source_indices",
            retained_tensor,
            persistent=False,
        )
        with torch.no_grad():
            self.gate_proj.weight.copy_(
                gate.weight.index_select(0, retained_tensor)
            )
            self.up_proj.weight.copy_(
                up.weight.index_select(0, retained_tensor)
            )
            self.down_proj.weight.copy_(down.weight)
        self.activation = activation
        self.residual_width = width
        self.source_intermediate_width = intermediate
        self.consumer_source_indices = consumers
        self.source_mlp_fingerprint = source_fingerprint
        if source_storage & _storage_pointers(self):
            raise RuntimeError("compiled merged MLP aliases source storage")
        if module_state_fingerprint(source_mlp) != source_fingerprint:
            raise RuntimeError("compiled merged MLP mutated its source")
        expected = 3 * width * intermediate - 2 * width * len(consumers)
        if sum(parameter.numel() for parameter in self.parameters()) != expected:
            raise RuntimeError("compiled merged MLP accounting drifted")
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

    def native_width_input(self, retained_features: Tensor) -> Tensor:
        result = torch.zeros(
            *retained_features.shape[:-1],
            self.source_intermediate_width,
            dtype=retained_features.dtype,
            device=retained_features.device,
        )
        result.index_copy_(
            -1,
            self.retained_source_indices,
            retained_features,
        )
        return result


@dataclass(slots=True)
class _ExecutionState:
    valid_positions: Tensor
    condition: str
    root_features: dict[tuple[int, int], Tensor]
    layer_calls: dict[int, int]


class _GlobalMergedMLPOverlay(nn.Module):
    def __init__(
        self,
        compiled: Gemma3GlobalMergedMLP,
        *,
        layer_ordinal: int,
        incoming: tuple[DirectedCrossBlockMerge, ...],
        outgoing_root_indices: tuple[int, ...],
        state: _ExecutionState,
    ) -> None:
        super().__init__()
        self.compiled = compiled
        self.layer_ordinal = layer_ordinal
        self.incoming = incoming
        self.outgoing_root_indices = outgoing_root_indices
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
            raise RuntimeError("each merged MLP overlay may execute only once")
        retained = self.compiled.retained_features(normalized_hidden_states)
        full = self.compiled.native_width_input(retained)
        for merge in self.incoming:
            if self._state.condition == "deletion":
                replacement = torch.zeros_like(
                    self._state.valid_positions,
                    dtype=full.dtype,
                    device=full.device,
                )
            else:
                try:
                    root = self._state.root_features[
                        merge.anchor_coordinate
                    ]
                except KeyError as error:
                    raise RuntimeError(
                        "a consumer executed before its native merge root"
                    ) from error
                replacement = (root * merge.activation_scale).masked_fill(
                    ~self._state.valid_positions,
                    0,
                )
            full[..., merge.consumer.mode_index] = replacement

        retained_lookup = {
            int(source_index): retained_index
            for retained_index, source_index in enumerate(
                self.compiled.retained_source_indices.tolist()
            )
        }
        for source_index in self.outgoing_root_indices:
            try:
                retained_index = retained_lookup[source_index]
            except KeyError as error:
                raise RuntimeError(
                    "a removed consumer cannot serve as a native root"
                ) from error
            root = retained[..., retained_index].masked_fill(
                ~self._state.valid_positions,
                0,
            )
            self._state.root_features[
                (self.layer_ordinal, source_index)
            ] = root
        self._state.layer_calls[self.layer_ordinal] = calls + 1
        return self.compiled.down_proj(full)


@dataclass(frozen=True, slots=True)
class Gemma3GlobalMergeModelExecution:
    model_output: object
    condition: str
    merge_count: int
    native_root_count: int
    affected_layer_count: int
    source_whole_model_learned_parameters: int
    candidate_whole_model_learned_parameters: int
    removed_learned_parameters: int
    fixed_scale_coefficients: int
    net_stored_coefficient_savings: int
    valid_tokens: int
    logical_linear_macs_removed: int
    carry_scale_macs: int
    net_logical_macs_saved: int
    peak_live_root_scalars_per_token: int


class Gemma3GlobalCrossBlockMergeExecutor(nn.Module):
    """Temporarily overlay every affected Gemma MLP in one native prefill."""

    def __init__(
        self,
        adapter: Gemma3CausalLMAdapter,
        plan: GlobalCrossBlockMergePlan,
    ) -> None:
        super().__init__()
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if not isinstance(plan, GlobalCrossBlockMergePlan):
            raise TypeError("plan must be a GlobalCrossBlockMergePlan")
        if plan.merge_count == 0:
            raise ValueError("global model execution requires at least one merge")
        if adapter.model_fingerprint() != plan.source_model_fingerprint:
            raise ValueError("merge plan does not bind the live Gemma model")
        model = adapter.module
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise ValueError(
                "global merge execution requires a frozen eval source model"
            )
        incoming: defaultdict[
            int,
            list[DirectedCrossBlockMerge],
        ] = defaultdict(list)
        outgoing: defaultdict[int, set[int]] = defaultdict(set)
        for merge in plan.merges:
            incoming[merge.consumer.layer_ordinal].append(merge)
            outgoing[merge.anchor.layer_ordinal].add(
                merge.anchor.mode_index
            )
        affected = tuple(sorted(set(incoming) | set(outgoing)))
        compiled: dict[str, Gemma3GlobalMergedMLP] = {}
        source_fingerprints: dict[int, str] = {}
        activation: str | None = None
        residual_width: int | None = None
        intermediate_width: int | None = None
        for ordinal in affected:
            layer_spec = adapter.layers[ordinal]
            if (
                layer_spec.transformer is None
                or layer_spec.transformer.feed_forward is None
            ):
                raise ValueError("Gemma layer lacks structured MLP metadata")
            layer_activation = (
                layer_spec.transformer.feed_forward.activation
            )
            if activation is None:
                activation = layer_activation
            elif activation != layer_activation:
                raise ValueError(
                    "global merge executor requires one Gemma MLP activation"
                )
            source_layer = adapter.source_module(layer_spec.id)
            source_mlp = getattr(source_layer, "mlp", None)
            if not isinstance(source_mlp, nn.Module):
                raise TypeError("Gemma source layer does not expose an MLP")
            consumer_indices = tuple(
                sorted(
                    merge.consumer.mode_index
                    for merge in incoming.get(ordinal, ())
                )
            )
            candidate = Gemma3GlobalMergedMLP(
                source_mlp,
                consumer_source_indices=consumer_indices,
                activation=layer_activation,
            )
            if residual_width is None:
                residual_width = candidate.residual_width
                intermediate_width = candidate.source_intermediate_width
            elif (
                residual_width != candidate.residual_width
                or intermediate_width != candidate.source_intermediate_width
            ):
                raise ValueError(
                    "global merge executor requires equal Gemma MLP widths"
                )
            compiled[str(ordinal)] = candidate
            source_fingerprints[ordinal] = module_state_fingerprint(source_mlp)
        assert residual_width is not None
        assert intermediate_width is not None
        self.adapter = adapter
        self.plan = plan
        self.compiled_mlps = nn.ModuleDict(compiled)
        self._incoming = {
            ordinal: tuple(
                sorted(
                    values,
                    key=lambda merge: merge.consumer.mode_index,
                )
            )
            for ordinal, values in incoming.items()
        }
        self._outgoing = {
            ordinal: tuple(sorted(values))
            for ordinal, values in outgoing.items()
        }
        self._source_fingerprints = source_fingerprints
        self._affected_ordinals = affected
        self.residual_width = residual_width
        self.source_intermediate_width = intermediate_width
        self._active = False
        self.requires_grad_(False)
        self.eval()
        self._validate_live_source()

    @property
    def merge_count(self) -> int:
        return self.plan.merge_count

    @property
    def native_root_count(self) -> int:
        return len(
            {
                merge.anchor_coordinate for merge in self.plan.merges
            }
        )

    def _validate_live_source(self) -> None:
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
        last_consumer: dict[tuple[int, int], int] = {}
        for merge in self.plan.merges:
            last_consumer[merge.anchor_coordinate] = max(
                last_consumer.get(merge.anchor_coordinate, -1),
                merge.consumer.layer_ordinal,
            )
        peak = 0
        for ordinal in range(len(self.adapter.layers)):
            live = sum(
                anchor_ordinal <= ordinal < end
                for (anchor_ordinal, _), end in last_consumer.items()
            )
            peak = max(peak, live)
        return peak

    def run(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        condition: str = "merged",
    ) -> Gemma3GlobalMergeModelExecution:
        if condition not in _CONDITIONS:
            raise ValueError("condition must be 'merged' or 'deletion'")
        if self._active:
            raise RuntimeError("global merge execution is not reentrant")
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
                layers[ordinal].mlp = _GlobalMergedMLPOverlay(
                    self.compiled_mlps[str(ordinal)],
                    layer_ordinal=ordinal,
                    incoming=self._incoming.get(ordinal, ()),
                    outgoing_root_indices=self._outgoing.get(ordinal, ()),
                    state=state,
                )
            candidate_parameters = sum(
                parameter.numel()
                for parameter in self.adapter.module.parameters()
            )
            expected = (
                source_parameters
                - 2 * self.residual_width * self.merge_count
            )
            if candidate_parameters != expected:
                raise RuntimeError(
                    "global in-model learned-parameter accounting drifted"
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
            raise RuntimeError("not every global MLP overlay executed once")
        valid_tokens = int(valid.sum().item())
        removed = 2 * self.residual_width * self.merge_count
        scale_macs = (
            valid_tokens * self.merge_count
            if condition == "merged"
            else 0
        )
        removed_macs = valid_tokens * removed
        return Gemma3GlobalMergeModelExecution(
            model_output=model_output,
            condition=condition,
            merge_count=self.merge_count,
            native_root_count=self.native_root_count,
            affected_layer_count=len(self._affected_ordinals),
            source_whole_model_learned_parameters=source_parameters,
            candidate_whole_model_learned_parameters=candidate_parameters,
            removed_learned_parameters=removed,
            fixed_scale_coefficients=(
                self.merge_count if condition == "merged" else 0
            ),
            net_stored_coefficient_savings=(
                removed
                - (self.merge_count if condition == "merged" else 0)
            ),
            valid_tokens=valid_tokens,
            logical_linear_macs_removed=removed_macs,
            carry_scale_macs=scale_macs,
            net_logical_macs_saved=removed_macs - scale_macs,
            peak_live_root_scalars_per_token=self._peak_live_roots(),
        )

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
    ) -> Gemma3GlobalMergeModelExecution:
        return self.run(model_inputs, condition="merged")


__all__ = [
    "Gemma3GlobalCrossBlockMergeExecutor",
    "Gemma3GlobalMergeModelExecution",
    "Gemma3GlobalMergedMLP",
]
