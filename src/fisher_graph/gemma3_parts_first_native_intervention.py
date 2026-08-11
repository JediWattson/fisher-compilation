"""Ephemeral native part gates for the parts-first supermode pilot.

The exact 39-event executor is the preferred capture path, but its existing
shadow API injects only one residual-width perturbation.  A genuine ordered
finite intervention must gate each selected physical group at its own native
operator boundary and let every later computation rerun on the changed state.

This module installs temporary hooks at exactly those boundaries:

* tied lexical columns at embedding output or language-head input;
* query-head coordinate blocks at the attention ``o_proj`` input; and
* MLP channels at the ``down_proj`` input.

It never edits weights.  Callers must still compare model/execution
fingerprints before and after the session and run an alpha-one literal-parity
control before interpreting a finite gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping, TypeVar

import torch
from torch import Tensor, nn

from .gemma3_parts_first_supermodes import (
    ATTENTION_HEAD_COUNT,
    ATTENTION_HEAD_DIMENSION,
    ExecutionBoundary,
    GEMMA3_LAYER_COUNT,
    HIDDEN_WIDTH,
    MLP_CHANNEL_COUNT,
    PhysicalAtomFamily,
)


NATIVE_GATE_ABI = "fisher_graph.gemma3_parts_first_native_gate.v1"
_RECEIPT_DOMAIN = b"fisher_graph.gemma3_parts_first_native_gate_receipt.v1\0"
_T = TypeVar("_T")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(_RECEIPT_DOMAIN + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenNativePartGate:
    """One frozen union group applied once at one physical boundary."""

    gate_id: str
    family: PhysicalAtomFamily
    boundary: ExecutionBoundary
    part_ordinals: tuple[int, ...]
    alpha: float
    chart_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.gate_id, str) or not self.gate_id:
            raise ValueError("gate_id must be nonempty")
        if not isinstance(self.family, PhysicalAtomFamily):
            raise TypeError("family must be a PhysicalAtomFamily")
        if not isinstance(self.boundary, ExecutionBoundary):
            raise TypeError("boundary must be an ExecutionBoundary")
        if not isinstance(self.chart_id, str) or not self.chart_id:
            raise ValueError("chart_id must be nonempty")
        if (
            not isinstance(self.alpha, float)
            or not math.isfinite(self.alpha)
            or self.alpha < 0.0
            or self.alpha > 2.0
        ):
            raise ValueError("alpha must be an exact float in [0,2]")
        if (
            not self.part_ordinals
            or any(type(value) is not int for value in self.part_ordinals)
            or tuple(sorted(set(self.part_ordinals))) != self.part_ordinals
        ):
            raise ValueError("part_ordinals must be sorted unique exact ints")
        if self.family is PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN:
            if self.boundary not in (
                ExecutionBoundary.lexical_ingress(),
                ExecutionBoundary.lexical_readout(),
            ):
                raise ValueError("lexical columns require a lexical boundary")
            limit = HIDDEN_WIDTH
        elif self.family is PhysicalAtomFamily.ATTENTION_QO_HEAD:
            if self.boundary.boundary_kind.value != "attention":
                raise ValueError("query heads require an attention boundary")
            limit = ATTENTION_HEAD_COUNT
        else:
            if self.boundary.boundary_kind.value != "mlp":
                raise ValueError("MLP channels require an MLP boundary")
            limit = MLP_CHANNEL_COUNT
        if self.part_ordinals[0] < 0 or self.part_ordinals[-1] >= limit:
            raise ValueError("part ordinal is outside its physical family")

    def state(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "family": self.family.value,
            "boundary": self.boundary.to_dict(),
            "part_ordinals": list(self.part_ordinals),
            "alpha": self.alpha,
            "chart_id": self.chart_id,
        }


@dataclass(frozen=True, slots=True)
class NativePartGateExecutionReceipt:
    gate_states: tuple[dict[str, object], ...]
    hook_call_counts: tuple[int, ...]
    hook_execution_order: tuple[int, ...]
    mask_true_counts: tuple[int, ...]
    alpha_one_control: bool
    hooks_removed: bool
    weights_mutated_by_hooks: bool
    later_native_computation_reran: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        count = len(self.gate_states)
        if not count or len(self.hook_call_counts) != count:
            raise ValueError("gate receipt inventory is incomplete")
        if len(self.mask_true_counts) != count:
            raise ValueError("gate mask receipt inventory is incomplete")
        if self.hook_execution_order != tuple(range(count)):
            raise ValueError("native gates did not execute in frozen event order")
        if any(type(value) is not int or value != 1 for value in self.hook_call_counts):
            raise ValueError("every frozen gate must execute exactly once")
        if any(type(value) is not int or value <= 0 for value in self.mask_true_counts):
            raise ValueError("every frozen gate needs nonempty chart support")
        for value in (
            self.alpha_one_control,
            self.hooks_removed,
            self.weights_mutated_by_hooks,
            self.later_native_computation_reran,
        ):
            if type(value) is not bool:
                raise TypeError("gate receipt booleans must be exact bools")
        if not self.hooks_removed or self.weights_mutated_by_hooks:
            raise ValueError("native gate lifecycle did not close")
        if not self.later_native_computation_reran:
            raise ValueError("finite gates must rerun later native computation")
        payload = self._payload()
        computed = _canonical_sha256(payload)
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif self.receipt_sha256 != computed:
            raise ValueError("native gate receipt hash does not close")

    def _payload(self) -> dict[str, object]:
        return {
            "abi": NATIVE_GATE_ABI,
            "gate_states": list(self.gate_states),
            "hook_call_counts": list(self.hook_call_counts),
            "hook_execution_order": list(self.hook_execution_order),
            "mask_true_counts": list(self.mask_true_counts),
            "alpha_one_control": self.alpha_one_control,
            "hooks_removed": self.hooks_removed,
            "weights_mutated_by_hooks": self.weights_mutated_by_hooks,
            "later_native_computation_reran": self.later_native_computation_reran,
        }


def _scaled_part_input(
    value: Tensor,
    *,
    indices: Tensor,
    row_mask: Tensor,
    alpha: float,
) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 3:
        raise ValueError("native gate input must be [batch,sequence,width]")
    if row_mask.shape != value.shape[:2] or row_mask.dtype != torch.bool:
        raise ValueError("frozen chart mask does not align with the native batch")
    if row_mask.device != value.device:
        raise ValueError("frozen chart mask and native value must share a device")
    if not bool(row_mask.any()):
        raise ValueError("frozen chart mask has no support in this batch")
    if alpha == 1.0:
        return value
    result = value.clone()
    selected = result.index_select(-1, indices)
    multiplier = torch.where(
        row_mask.unsqueeze(-1),
        torch.full_like(selected, alpha),
        torch.ones_like(selected),
    )
    result.index_copy_(-1, indices, selected * multiplier)
    return result


class _NativePartGateSession:
    def __init__(
        self,
        model: nn.Module,
        gates: tuple[FrozenNativePartGate, ...],
        row_masks: Mapping[str, Tensor],
        *,
        allow_test_nonproduction_shape: bool,
    ) -> None:
        self._model = model
        self._gates = gates
        self._row_masks = dict(row_masks)
        self._allow_test_nonproduction_shape = allow_test_nonproduction_shape
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._call_counts = [0] * len(gates)
        self._execution_order: list[int] = []
        self._closed = False

    def _remove_hooks(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    @staticmethod
    def _lexical_modules(model: nn.Module) -> tuple[nn.Module, nn.Module]:
        get_input = getattr(model, "get_input_embeddings", None)
        get_output = getattr(model, "get_output_embeddings", None)
        input_module = get_input() if callable(get_input) else None
        output_module = get_output() if callable(get_output) else None
        if not isinstance(input_module, nn.Module) or not isinstance(
            output_module,
            nn.Module,
        ):
            raise TypeError("model must expose input and output lexical modules")
        input_weight = getattr(input_module, "weight", None)
        output_weight = getattr(output_module, "weight", None)
        if (
            not isinstance(input_weight, Tensor)
            or not isinstance(output_weight, Tensor)
            or input_weight.shape != output_weight.shape
            or input_weight.untyped_storage().data_ptr()
            != output_weight.untyped_storage().data_ptr()
        ):
            raise ValueError("input and output lexical storage must be tied")
        if input_module is output_module:
            raise ValueError("lexical role hooks require distinct tied modules")
        return input_module, output_module

    def _layers(self) -> tuple[nn.Module, ...]:
        backbone = getattr(self._model, "model", None)
        layers = getattr(backbone, "layers", None)
        if not isinstance(layers, (nn.ModuleList, list, tuple)):
            raise TypeError("model must expose model.layers")
        result = tuple(layers)
        if (
            not self._allow_test_nonproduction_shape
            and len(result) != GEMMA3_LAYER_COUNT
        ):
            raise ValueError("native gates require the pinned 18-layer model")
        return result

    def _module_and_indices(
        self,
        gate: FrozenNativePartGate,
    ) -> tuple[nn.Module, Tensor, bool]:
        input_lexical, output_lexical = self._lexical_modules(self._model)
        if gate.family is PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN:
            output_role = gate.boundary == ExecutionBoundary.lexical_readout()
            module = output_lexical if output_role else input_lexical
            indices = torch.tensor(gate.part_ordinals, dtype=torch.int64)
            return module, indices, not output_role
        layer_ordinal = gate.boundary.layer_ordinal
        assert layer_ordinal is not None
        layers = self._layers()
        if layer_ordinal >= len(layers):
            raise ValueError("gate layer is outside the model")
        layer = layers[layer_ordinal]
        if gate.family is PhysicalAtomFamily.ATTENTION_QO_HEAD:
            attention = getattr(layer, "self_attn", None)
            module = getattr(attention, "o_proj", None)
            if not isinstance(module, nn.Module):
                raise TypeError("attention layer omitted o_proj")
            coordinates = tuple(
                coordinate
                for head in gate.part_ordinals
                for coordinate in range(
                    head * ATTENTION_HEAD_DIMENSION,
                    (head + 1) * ATTENTION_HEAD_DIMENSION,
                )
            )
            return module, torch.tensor(coordinates, dtype=torch.int64), False
        mlp = getattr(layer, "mlp", None)
        module = getattr(mlp, "down_proj", None)
        if not isinstance(module, nn.Module):
            raise TypeError("MLP layer omitted down_proj")
        return module, torch.tensor(gate.part_ordinals, dtype=torch.int64), False

    def __enter__(self) -> _NativePartGateSession:
        if self._handles or self._closed:
            raise RuntimeError("native gate session is one-use")
        if len({gate.gate_id for gate in self._gates}) != len(self._gates):
            raise ValueError("native gate IDs must be unique")
        boundaries = tuple(gate.boundary for gate in self._gates)
        if len(set(boundaries)) != len(boundaries):
            raise ValueError("selected groups must be unioned once per boundary")
        if tuple(boundary.execution_ordinal for boundary in boundaries) != tuple(
            sorted(boundary.execution_ordinal for boundary in boundaries)
        ):
            raise ValueError("native gates must be frozen in execution order")
        if set(self._row_masks) != {gate.gate_id for gate in self._gates}:
            raise ValueError("one frozen chart mask is required per gate")
        try:
            for index, gate in enumerate(self._gates):
                module, raw_indices, output_hook = self._module_and_indices(gate)
                parameter = next(module.parameters(), None)
                device = (
                    parameter.device
                    if isinstance(parameter, Tensor)
                    else self._row_masks[gate.gate_id].device
                )
                indices = raw_indices.to(device=device)
                row_mask = self._row_masks[gate.gate_id]

                if output_hook:
                    def hook(
                        _module: nn.Module,
                        _arguments: tuple[object, ...],
                        output: Tensor,
                        *,
                        gate_index: int = index,
                        selected: Tensor = indices,
                        mask: Tensor = row_mask,
                        alpha: float = gate.alpha,
                    ) -> Tensor:
                        self._call_counts[gate_index] += 1
                        self._execution_order.append(gate_index)
                        return _scaled_part_input(
                            output,
                            indices=selected,
                            row_mask=mask,
                            alpha=alpha,
                        )

                    self._handles.append(module.register_forward_hook(hook))
                else:
                    def pre_hook(
                        _module: nn.Module,
                        arguments: tuple[object, ...],
                        *,
                        gate_index: int = index,
                        selected: Tensor = indices,
                        mask: Tensor = row_mask,
                        alpha: float = gate.alpha,
                    ) -> tuple[object, ...]:
                        if not arguments or not isinstance(arguments[0], Tensor):
                            raise TypeError("gated module omitted its tensor input")
                        self._call_counts[gate_index] += 1
                        self._execution_order.append(gate_index)
                        scaled = _scaled_part_input(
                            arguments[0],
                            indices=selected,
                            row_mask=mask,
                            alpha=alpha,
                        )
                        return (scaled, *arguments[1:])

                    self._handles.append(module.register_forward_pre_hook(pre_hook))
        except BaseException:
            self._remove_hooks()
            self._closed = True
            raise
        return self

    def close(self) -> NativePartGateExecutionReceipt:
        if self._closed:
            raise RuntimeError("native gate session was already closed")
        self._closed = True
        self._remove_hooks()
        return NativePartGateExecutionReceipt(
            gate_states=tuple(gate.state() for gate in self._gates),
            hook_call_counts=tuple(self._call_counts),
            hook_execution_order=tuple(self._execution_order),
            mask_true_counts=tuple(
                int(self._row_masks[gate.gate_id].sum().item())
                for gate in self._gates
            ),
            alpha_one_control=all(gate.alpha == 1.0 for gate in self._gates),
            hooks_removed=True,
            weights_mutated_by_hooks=False,
            later_native_computation_reran=True,
        )

    def __exit__(self, error_type, error, traceback) -> None:
        if error_type is not None:
            self._remove_hooks()
            self._closed = True


def run_with_native_part_gates(
    model: nn.Module,
    gates: tuple[FrozenNativePartGate, ...],
    row_masks: Mapping[str, Tensor],
    forward: Callable[[], _T],
    *,
    allow_test_nonproduction_shape: bool = False,
) -> tuple[_T, NativePartGateExecutionReceipt]:
    """Execute one forward with all frozen stage gates installed in order."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not gates or any(not isinstance(gate, FrozenNativePartGate) for gate in gates):
        raise ValueError("gates must contain frozen native part gates")
    if not callable(forward):
        raise TypeError("forward must be callable")
    session = _NativePartGateSession(
        model,
        gates,
        row_masks,
        allow_test_nonproduction_shape=allow_test_nonproduction_shape,
    )
    with session:
        result = forward()
    receipt = session.close()
    return result, receipt


__all__ = [
    "FrozenNativePartGate",
    "NATIVE_GATE_ABI",
    "NativePartGateExecutionReceipt",
    "run_with_native_part_gates",
]
