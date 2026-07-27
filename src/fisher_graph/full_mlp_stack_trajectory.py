"""Frozen prefix/suffix trajectories for a complete Gemma MLP stack.

The full-stack replacements are authenticated and lowered exactly once.  A
trajectory run then borrows those frozen compiled modules for an exact subset
of layers while every other layer continues to execute its native MLP.

The streaming visitor API is intentional.  Gemma logits are large enough that
retaining all prefix and suffix outputs at once would make the experiment's
memory use scale with the number of trajectory conditions.  Visitors consume
one output at a time after its overlay has already been restored.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import TypeVar

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .compiler.calibration import CalibrationBatch
from .gemma3_full_mlp_stack_executor import Gemma3FullMLPStackExecutor
from .gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorMLP,
    Gemma3ModalGeneratorReplacement,
)


__all__ = [
    "FrozenFullMLPStackTrajectoryExecutor",
    "FullMLPStackSubsetAccounting",
    "FullMLPStackSubsetExecution",
    "evaluate_full_mlp_stack_trajectory",
]


_ASSESSMENT_ROLE = "open_development_assessment"
_NATIVE_COMPONENTS_RETAINED = (
    "embeddings",
    "attention",
    "normalization",
    "language_model_head",
)
_CallbackResult = TypeVar("_CallbackResult")


def _canonical_layer_subset(
    values: Sequence[int],
    *,
    layer_count: int,
    label: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(values)
    if any(type(value) is not int for value in result):
        raise TypeError(f"{label} must contain exact integers")
    if (
        len(result) != len(set(result))
        or tuple(sorted(result)) != result
        or any(value < 0 or value >= layer_count for value in result)
    ):
        raise ValueError(
            f"{label} must contain unique, increasing, in-range ordinals"
        )
    return result


def _module_runtime_signature(module: nn.Module) -> tuple[object, ...]:
    """Cheaply detect topology, tensor-identity, or in-place state drift."""

    modules = tuple(
        (
            name,
            id(value),
            type(value),
            value.training,
        )
        for name, value in module.named_modules()
    )
    parameters = tuple(
        (
            name,
            id(value),
            value._version,
            value.requires_grad,
        )
        for name, value in module.named_parameters()
    )
    buffers = tuple(
        (
            name,
            id(value),
            value._version,
        )
        for name, value in module.named_buffers()
    )
    return modules, parameters, buffers


@dataclass(frozen=True, slots=True)
class FullMLPStackSubsetAccounting:
    """Exact logical and resident resources for one generated layer subset."""

    generated_layer_ordinals: tuple[int, ...]
    source_layer_count: int
    replaced_layer_count: int
    removed_mode_count: int
    source_whole_model_learned_parameters: int
    logical_native_mlp_stack_learned_parameters: int
    logical_retained_native_non_mlp_learned_parameters: int
    logical_native_mlp_parameters_removed: int
    logical_native_mlp_parameters_retained: int
    logical_generator_subset_learned_parameters: int
    logical_candidate_mlp_learned_parameters: int
    logical_candidate_learned_parameters: int
    logical_net_stored_parameter_savings: int
    experimental_resident_source_learned_parameters: int
    experimental_resident_compiled_full_stack_learned_parameters: int
    experimental_resident_total_learned_parameters: int
    experimental_resident_overhead_vs_logical_candidate: int
    valid_tokens: int
    logical_native_mlp_stack_macs_baseline: int
    logical_native_mlp_macs_removed: int
    logical_native_mlp_macs_retained: int
    logical_generator_macs: int
    logical_candidate_mlp_macs: int
    logical_generator_bias_additions: int
    logical_candidate_mlp_bias_additions: int
    net_logical_macs_saved: int
    replacement_scope: str
    native_components_retained: tuple[str, ...] = (
        _NATIVE_COMPONENTS_RETAINED
    )
    experimental_source_state_retained: bool = True
    full_stack_generator_catalog_compiled_once: bool = True

    def __post_init__(self) -> None:
        ordinals = self.generated_layer_ordinals
        if (
            type(ordinals) is not tuple
            or len(ordinals) != len(set(ordinals))
            or tuple(sorted(ordinals)) != ordinals
            or any(type(value) is not int or value < 0 for value in ordinals)
        ):
            raise ValueError("generated layer ordinals are not canonical")
        integer_values = tuple(
            value
            for value in (
                self.source_layer_count,
                self.replaced_layer_count,
                self.removed_mode_count,
                self.source_whole_model_learned_parameters,
                self.logical_native_mlp_stack_learned_parameters,
                self.logical_retained_native_non_mlp_learned_parameters,
                self.logical_native_mlp_parameters_removed,
                self.logical_native_mlp_parameters_retained,
                self.logical_generator_subset_learned_parameters,
                self.logical_candidate_mlp_learned_parameters,
                self.logical_candidate_learned_parameters,
                self.experimental_resident_source_learned_parameters,
                self.experimental_resident_compiled_full_stack_learned_parameters,
                self.experimental_resident_total_learned_parameters,
                self.experimental_resident_overhead_vs_logical_candidate,
                self.valid_tokens,
                self.logical_native_mlp_stack_macs_baseline,
                self.logical_native_mlp_macs_removed,
                self.logical_native_mlp_macs_retained,
                self.logical_generator_macs,
                self.logical_candidate_mlp_macs,
                self.logical_generator_bias_additions,
                self.logical_candidate_mlp_bias_additions,
            )
        )
        if any(type(value) is not int for value in integer_values):
            raise TypeError("subset accounting must use exact integers")
        if any(value < 0 for value in integer_values):
            raise ValueError("subset accounting must be nonnegative")
        if (
            type(self.logical_net_stored_parameter_savings) is not int
            or type(self.net_logical_macs_saved) is not int
        ):
            raise TypeError("subset savings must use exact signed integers")
        if (
            self.source_layer_count <= 0
            or self.source_whole_model_learned_parameters <= 0
            or self.logical_native_mlp_stack_learned_parameters <= 0
            or self.replaced_layer_count != len(ordinals)
            or any(value >= self.source_layer_count for value in ordinals)
        ):
            raise ValueError("subset layer declaration is inconsistent")
        expected_scope = (
            "native_reference"
            if not ordinals
            else "full_native_mlp_replacement_at_exact_layer_subset"
        )
        if self.replacement_scope != expected_scope:
            raise ValueError("subset replacement scope is inconsistent")
        if (
            self.native_components_retained != _NATIVE_COMPONENTS_RETAINED
            or self.experimental_source_state_retained is not True
            or self.full_stack_generator_catalog_compiled_once is not True
        ):
            raise ValueError("subset execution declarations drifted")

        source = self.source_whole_model_learned_parameters
        native_stack = self.logical_native_mlp_stack_learned_parameters
        non_mlp = self.logical_retained_native_non_mlp_learned_parameters
        removed = self.logical_native_mlp_parameters_removed
        retained = self.logical_native_mlp_parameters_retained
        generator = self.logical_generator_subset_learned_parameters
        candidate_mlp = self.logical_candidate_mlp_learned_parameters
        candidate = self.logical_candidate_learned_parameters
        resident_compiled = (
            self.experimental_resident_compiled_full_stack_learned_parameters
        )
        resident_total = self.experimental_resident_total_learned_parameters
        if (
            non_mlp != source - native_stack
            or retained != native_stack - removed
            or candidate_mlp != retained + generator
            or candidate != non_mlp + candidate_mlp
            or self.logical_net_stored_parameter_savings != source - candidate
            or self.experimental_resident_source_learned_parameters != source
            or resident_total != source + resident_compiled
            or self.experimental_resident_overhead_vs_logical_candidate
            != resident_total - candidate
        ):
            raise ValueError("subset parameter accounting is inconsistent")

        valid = self.valid_tokens
        native_baseline = self.logical_native_mlp_stack_macs_baseline
        removed_macs = self.logical_native_mlp_macs_removed
        retained_macs = self.logical_native_mlp_macs_retained
        generator_macs = self.logical_generator_macs
        candidate_macs = self.logical_candidate_mlp_macs
        bias_additions = self.logical_generator_bias_additions
        if (
            native_baseline != valid * native_stack
            or removed_macs != valid * removed
            or retained_macs != native_baseline - removed_macs
            or candidate_macs != retained_macs + generator_macs
            or self.logical_candidate_mlp_bias_additions != bias_additions
            or self.net_logical_macs_saved
            != native_baseline - candidate_macs
        ):
            raise ValueError("subset compute accounting is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FullMLPStackSubsetExecution:
    """One model output after replacing an exact frozen layer subset."""

    model_output: object
    accounting: FullMLPStackSubsetAccounting

    def __post_init__(self) -> None:
        if not isinstance(self.accounting, FullMLPStackSubsetAccounting):
            raise TypeError("accounting must be FullMLPStackSubsetAccounting")

    @property
    def generated_layer_ordinals(self) -> tuple[int, ...]:
        return self.accounting.generated_layer_ordinals

    @property
    def valid_tokens(self) -> int:
        return self.accounting.valid_tokens


@dataclass(slots=True)
class _SubsetOverlayState:
    calls: dict[int, int]
    expected_forward_calls: int


class _FrozenSubsetMLPOverlay(nn.Module):
    def __init__(
        self,
        compiled: Gemma3ModalGeneratorMLP,
        *,
        layer_ordinal: int,
        state: _SubsetOverlayState,
    ) -> None:
        super().__init__()
        self.compiled = compiled
        self.layer_ordinal = layer_ordinal
        self._state = state
        self.eval()

    def forward(self, normalized_hidden_states: Tensor) -> Tensor:
        calls = self._state.calls.get(self.layer_ordinal, 0)
        if calls >= self._state.expected_forward_calls:
            if self._state.expected_forward_calls == 1:
                raise RuntimeError(
                    "each frozen trajectory overlay may execute only once"
                )
            raise RuntimeError(
                "each frozen trajectory overlay may execute only "
                f"{self._state.expected_forward_calls} times"
            )
        output = self.compiled(
            normalized_hidden_states,
            condition="generated",
        )
        self._state.calls[self.layer_ordinal] = calls + 1
        return output


class FrozenFullMLPStackTrajectoryExecutor(nn.Module):
    """Compile a complete stack once and execute arbitrary exact subsets."""

    def __init__(
        self,
        adapter: Gemma3CausalLMAdapter,
        replacements: Sequence[Gemma3ModalGeneratorReplacement],
    ) -> None:
        super().__init__()
        full_stack = Gemma3FullMLPStackExecutor(adapter, replacements)
        self.full_stack = full_stack
        self._adapter = adapter
        self._layer_count = full_stack.replaced_layer_count
        self._source_model_sha256 = adapter.model_fingerprint()
        self._source_parameters = sum(
            parameter.numel() for parameter in adapter.module.parameters()
        )

        live_layers = self._live_layers()
        source_mlps: list[nn.Module] = []
        source_runtime_signatures: list[tuple[object, ...]] = []
        compiled_sha256: list[str] = []
        compiled_runtime_signatures: list[tuple[object, ...]] = []
        mode_counts: list[int] = []
        native_parameters: list[int] = []
        generator_parameters: list[int] = []
        generator_macs: list[int] = []
        generator_bias_additions: list[int] = []
        for ordinal in range(self._layer_count):
            source = getattr(live_layers[ordinal], "mlp", None)
            compiled = full_stack.compiled_mlps[str(ordinal)]
            if not isinstance(source, nn.Module) or not isinstance(
                compiled,
                Gemma3ModalGeneratorMLP,
            ):
                raise TypeError("trajectory stack contains an invalid MLP")
            if not compiled.is_full_native_replacement:
                raise ValueError(
                    "trajectory stack requires full-channel replacements"
                )
            source_count = sum(
                parameter.numel() for parameter in source.parameters()
            )
            if source_count != compiled.native_removed_parameter_count:
                raise RuntimeError(
                    "trajectory native layer accounting drifted"
                )
            source_mlps.append(source)
            source_runtime_signatures.append(
                _module_runtime_signature(source)
            )
            compiled_sha256.append(module_state_fingerprint(compiled))
            compiled_runtime_signatures.append(
                _module_runtime_signature(compiled)
            )
            mode_counts.append(compiled.removed_mode_count)
            native_parameters.append(source_count)
            generator_parameters.append(compiled.generator_parameter_count)
            generator_macs.append(compiled.generator_macs_per_token)
            generator_bias_additions.append(
                compiled.generator_bias_additions_per_token
            )

        self._source_mlps = tuple(source_mlps)
        self._source_runtime_signatures = tuple(
            source_runtime_signatures
        )
        self._compiled_sha256 = tuple(compiled_sha256)
        self._compiled_runtime_signatures = tuple(
            compiled_runtime_signatures
        )
        self._mode_counts_by_layer = tuple(mode_counts)
        self._native_parameters_by_layer = tuple(native_parameters)
        self._generator_parameters_by_layer = tuple(generator_parameters)
        self._generator_macs_by_layer = tuple(generator_macs)
        self._generator_bias_additions_by_layer = tuple(
            generator_bias_additions
        )
        self._native_mlp_stack_parameters = sum(native_parameters)
        self._compiled_full_stack_parameters = sum(generator_parameters)
        if (
            full_stack.logical_candidate_learned_parameters
            != (
                self._source_parameters
                - self._native_mlp_stack_parameters
                + self._compiled_full_stack_parameters
            )
            or full_stack.experimental_resident_learned_parameters
            != (
                self._source_parameters
                + self._compiled_full_stack_parameters
            )
        ):
            raise RuntimeError("trajectory full-stack accounting drifted")
        self._active = False
        self.requires_grad_(False)
        self.eval()
        self._validate_full_source()

    @property
    def adapter(self) -> Gemma3CausalLMAdapter:
        return self._adapter

    @property
    def compiled_mlps(self) -> nn.ModuleDict:
        return self.full_stack.compiled_mlps

    @property
    def replaced_layer_count(self) -> int:
        return self._layer_count

    @property
    def removed_mode_count(self) -> int:
        return sum(self._mode_counts_by_layer)

    @property
    def mode_counts_by_layer(self) -> tuple[int, ...]:
        return self._mode_counts_by_layer

    def _live_layers(self) -> object:
        model = getattr(self.adapter.module, "model", None)
        layers = getattr(model, "layers", None)
        if layers is None or not hasattr(layers, "__len__") or not hasattr(
            layers,
            "__getitem__",
        ):
            raise TypeError("live Gemma model does not expose indexed layers")
        if hasattr(self, "_layer_count") and len(layers) != self._layer_count:
            raise RuntimeError("live Gemma layer count drifted")
        return layers

    def _validate_affected_layers(
        self,
        ordinals: tuple[int, ...],
    ) -> None:
        layers = self._live_layers()
        for ordinal in ordinals:
            source = self._source_mlps[ordinal]
            compiled = self.compiled_mlps[str(ordinal)]
            if (
                getattr(layers[ordinal], "mlp", None) is not source
                or _module_runtime_signature(source)
                != self._source_runtime_signatures[ordinal]
                or _module_runtime_signature(compiled)
                != self._compiled_runtime_signatures[ordinal]
            ):
                raise RuntimeError(
                    "trajectory execution did not restore an affected MLP "
                    "exactly"
                )

    def _validate_full_source(self) -> None:
        if self.adapter.model_fingerprint() != self._source_model_sha256:
            raise RuntimeError("native Gemma source model state drifted")
        model = self.adapter.module
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise RuntimeError("native Gemma source is not frozen eval")
        layers = self._live_layers()
        for ordinal in range(self._layer_count):
            source = self._source_mlps[ordinal]
            compiled = self.compiled_mlps[str(ordinal)]
            if (
                getattr(layers[ordinal], "mlp", None) is not source
                or _module_runtime_signature(source)
                != self._source_runtime_signatures[ordinal]
                or _module_runtime_signature(compiled)
                != self._compiled_runtime_signatures[ordinal]
                or module_state_fingerprint(compiled)
                != self._compiled_sha256[ordinal]
            ):
                raise RuntimeError(
                    "native or compiled Gemma MLP state drifted"
                )

    def subset_accounting(
        self,
        generated_layer_ordinals: Sequence[int],
        *,
        valid_tokens: int,
    ) -> FullMLPStackSubsetAccounting:
        """Return exact resources without executing or changing the model."""

        subset = _canonical_layer_subset(
            generated_layer_ordinals,
            layer_count=self._layer_count,
            label="generated_layer_ordinals",
        )
        if type(valid_tokens) is not int or valid_tokens < 0:
            raise ValueError("valid_tokens must be a nonnegative integer")
        removed_modes = sum(
            self._mode_counts_by_layer[ordinal] for ordinal in subset
        )
        removed_parameters = sum(
            self._native_parameters_by_layer[ordinal] for ordinal in subset
        )
        generator_parameters = sum(
            self._generator_parameters_by_layer[ordinal]
            for ordinal in subset
        )
        generator_macs_per_token = sum(
            self._generator_macs_by_layer[ordinal] for ordinal in subset
        )
        generator_bias_additions_per_token = sum(
            self._generator_bias_additions_by_layer[ordinal]
            for ordinal in subset
        )
        native_retained = (
            self._native_mlp_stack_parameters - removed_parameters
        )
        non_mlp = self._source_parameters - self._native_mlp_stack_parameters
        candidate_mlp = native_retained + generator_parameters
        candidate = non_mlp + candidate_mlp
        native_baseline_macs = (
            valid_tokens * self._native_mlp_stack_parameters
        )
        removed_macs = valid_tokens * removed_parameters
        retained_macs = native_baseline_macs - removed_macs
        generator_macs = valid_tokens * generator_macs_per_token
        candidate_macs = retained_macs + generator_macs
        bias_additions = (
            valid_tokens * generator_bias_additions_per_token
        )
        resident_total = (
            self._source_parameters + self._compiled_full_stack_parameters
        )
        return FullMLPStackSubsetAccounting(
            generated_layer_ordinals=subset,
            source_layer_count=self._layer_count,
            replaced_layer_count=len(subset),
            removed_mode_count=removed_modes,
            source_whole_model_learned_parameters=self._source_parameters,
            logical_native_mlp_stack_learned_parameters=(
                self._native_mlp_stack_parameters
            ),
            logical_retained_native_non_mlp_learned_parameters=non_mlp,
            logical_native_mlp_parameters_removed=removed_parameters,
            logical_native_mlp_parameters_retained=native_retained,
            logical_generator_subset_learned_parameters=(
                generator_parameters
            ),
            logical_candidate_mlp_learned_parameters=candidate_mlp,
            logical_candidate_learned_parameters=candidate,
            logical_net_stored_parameter_savings=(
                self._source_parameters - candidate
            ),
            experimental_resident_source_learned_parameters=(
                self._source_parameters
            ),
            experimental_resident_compiled_full_stack_learned_parameters=(
                self._compiled_full_stack_parameters
            ),
            experimental_resident_total_learned_parameters=resident_total,
            experimental_resident_overhead_vs_logical_candidate=(
                resident_total - candidate
            ),
            valid_tokens=valid_tokens,
            logical_native_mlp_stack_macs_baseline=native_baseline_macs,
            logical_native_mlp_macs_removed=removed_macs,
            logical_native_mlp_macs_retained=retained_macs,
            logical_generator_macs=generator_macs,
            logical_candidate_mlp_macs=candidate_macs,
            logical_generator_bias_additions=bias_additions,
            logical_candidate_mlp_bias_additions=bias_additions,
            net_logical_macs_saved=native_baseline_macs - candidate_macs,
            replacement_scope=(
                "native_reference"
                if not subset
                else "full_native_mlp_replacement_at_exact_layer_subset"
            ),
        )

    def _call_with_prevalidated_subset(
        self,
        *,
        subset: tuple[int, ...],
        callback: Callable[[], _CallbackResult],
        expected_forward_calls: int,
    ) -> _CallbackResult:
        layers = self._live_layers()
        originals: dict[int, nn.Module] = {}
        state = _SubsetOverlayState(
            calls={},
            expected_forward_calls=expected_forward_calls,
        )
        try:
            for ordinal in subset:
                original = getattr(layers[ordinal], "mlp", None)
                if original is not self._source_mlps[ordinal]:
                    raise RuntimeError("live Gemma MLP identity drifted")
                assert isinstance(original, nn.Module)
                originals[ordinal] = original
                layers[ordinal].mlp = _FrozenSubsetMLPOverlay(
                    self.compiled_mlps[str(ordinal)],
                    layer_ordinal=ordinal,
                    state=state,
                )
            expected_parameters = self.subset_accounting(
                subset,
                valid_tokens=0,
            ).logical_candidate_learned_parameters
            candidate_parameters = sum(
                parameter.numel()
                for parameter in self.adapter.module.parameters()
            )
            if candidate_parameters != expected_parameters:
                raise RuntimeError(
                    "trajectory candidate parameter accounting drifted"
                )
            callback_result = callback()
        finally:
            for ordinal, original in originals.items():
                layers[ordinal].mlp = original
            self._validate_affected_layers(subset)

        if set(state.calls) != set(subset) or any(
            calls != expected_forward_calls
            for calls in state.calls.values()
        ):
            if expected_forward_calls == 1:
                raise RuntimeError(
                    "not every trajectory overlay executed exactly once"
                )
            raise RuntimeError(
                "not every trajectory overlay executed exactly "
                f"{expected_forward_calls} times"
            )
        return callback_result

    def _run_prevalidated_subset(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        subset: tuple[int, ...],
        valid_tokens: int,
    ) -> FullMLPStackSubsetExecution:
        accounting = self.subset_accounting(
            subset,
            valid_tokens=valid_tokens,
        )

        def run_model() -> object:
            call_inputs: dict[str, object] = dict(model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            return self.adapter.module(**call_inputs)

        model_output = self._call_with_prevalidated_subset(
            subset=subset,
            callback=run_model,
            expected_forward_calls=1,
        )
        return FullMLPStackSubsetExecution(
            model_output=model_output,
            accounting=accounting,
        )

    def run_with_subset_overlay(
        self,
        *,
        generated_layer_ordinals: Sequence[int],
        callback: Callable[[], _CallbackResult],
        expected_forward_calls: int = 1,
    ) -> _CallbackResult:
        """Run one synchronous callback under an exact generated subset.

        Every affected generated MLP must run ``expected_forward_calls``
        times.  This permits an instrumented forward/backward stream to be
        consumed while the overlay is live without re-authenticating the full
        source model for every sequence.  Returning a lazy iterator from
        ``callback`` is not sufficient; it must be consumed synchronously
        inside the callback.
        """

        if self._active:
            raise RuntimeError("trajectory execution is not reentrant")
        if not callable(callback):
            raise TypeError("callback must be callable")
        if (
            type(expected_forward_calls) is not int
            or expected_forward_calls <= 0
        ):
            raise ValueError(
                "expected_forward_calls must be a positive integer"
            )
        subset = _canonical_layer_subset(
            generated_layer_ordinals,
            layer_count=self._layer_count,
            label="generated_layer_ordinals",
        )
        self._validate_full_source()
        self._active = True
        try:
            return self._call_with_prevalidated_subset(
                subset=subset,
                callback=callback,
                expected_forward_calls=expected_forward_calls,
            )
        finally:
            self._active = False
            self._validate_full_source()

    def visit_native_and_subsets(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        generated_layer_subsets: Sequence[Sequence[int]],
        native_visitor: Callable[[object], None],
        subset_visitor: Callable[[FullMLPStackSubsetExecution], None],
    ) -> int:
        """Stream one native output and each unique subset output to visitors."""

        if self._active:
            raise RuntimeError("trajectory execution is not reentrant")
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        if (
            isinstance(generated_layer_subsets, (str, bytes))
            or not isinstance(generated_layer_subsets, Sequence)
        ):
            raise TypeError("generated_layer_subsets must be a sequence")
        subsets = tuple(
            _canonical_layer_subset(
                values,
                layer_count=self._layer_count,
                label=f"generated_layer_subsets[{index}]",
            )
            for index, values in enumerate(generated_layer_subsets)
        )
        if not subsets or len(subsets) != len(set(subsets)):
            raise ValueError(
                "generated layer subsets must be nonempty and unique"
            )
        if not callable(native_visitor) or not callable(subset_visitor):
            raise TypeError("trajectory visitors must be callable")

        self._validate_full_source()
        context = self.adapter.prepare_sequence(model_inputs)
        valid_tokens = int(context.query_valid_mask.sum().item())
        call_inputs: dict[str, object] = dict(model_inputs)
        call_inputs["use_cache"] = False
        call_inputs["return_dict"] = True
        self._active = True
        try:
            native_output = self.adapter.module(**call_inputs)
            native_visitor(native_output)
            del native_output
            for subset in subsets:
                execution = self._run_prevalidated_subset(
                    model_inputs,
                    subset=subset,
                    valid_tokens=valid_tokens,
                )
                subset_visitor(execution)
                del execution
        finally:
            self._active = False
            self._validate_full_source()
        return valid_tokens

    def run_subset(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        generated_layer_ordinals: Sequence[int],
    ) -> FullMLPStackSubsetExecution:
        """Run one exact subset through the same fail-safe sweep machinery."""

        if self._active:
            raise RuntimeError("trajectory execution is not reentrant")
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        subset = _canonical_layer_subset(
            generated_layer_ordinals,
            layer_count=self._layer_count,
            label="generated_layer_ordinals",
        )
        self._validate_full_source()
        context = self.adapter.prepare_sequence(model_inputs)
        valid_tokens = int(context.query_valid_mask.sum().item())
        self._active = True
        try:
            return self._run_prevalidated_subset(
                model_inputs,
                subset=subset,
                valid_tokens=valid_tokens,
            )
        finally:
            self._active = False
            self._validate_full_source()

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
    ) -> FullMLPStackSubsetExecution:
        return self.run_subset(
            model_inputs,
            generated_layer_ordinals=tuple(range(self._layer_count)),
        )


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
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError(
            "model output must expose finite [batch, sequence, vocab] logits"
        )
    return logits


def _selected_logits_and_targets(
    logits: Tensor,
    batch: CalibrationBatch,
) -> tuple[Tensor, Tensor]:
    targets = batch.targets.to(device=logits.device)
    if targets.shape != logits.shape[:2]:
        raise ValueError("evaluation targets and logits positions differ")
    supervised = targets != -100
    valid = batch.valid_positions.to(device=logits.device)
    if valid.shape != supervised.shape or bool((supervised & ~valid).any()):
        raise ValueError(
            "supervised targets must be a subset of valid positions"
        )
    if not bool(supervised.any()):
        raise ValueError("evaluation batch has no supervised tokens")
    return (
        logits[supervised].detach().to(device="cpu", dtype=torch.float64),
        targets[supervised].detach().to(device="cpu", dtype=torch.long),
    )


def _nll_terms(logits: Tensor, targets: Tensor) -> list[float]:
    log_normalizer = torch.logsumexp(logits, dim=-1)
    row = torch.arange(targets.shape[0])
    return (-(logits[row, targets] - log_normalizer)).tolist()


def _candidate_terms(
    native_logits: Tensor,
    candidate_logits: Tensor,
    targets: Tensor,
    *,
    vocabulary_chunk_size: int,
) -> tuple[list[float], list[float], int]:
    if candidate_logits.shape != native_logits.shape:
        raise ValueError("native and candidate supervised logits differ")
    native_lse = torch.logsumexp(native_logits, dim=-1)
    candidate_lse = torch.logsumexp(candidate_logits, dim=-1)
    row = torch.arange(targets.shape[0])
    nll = -(candidate_logits[row, targets] - candidate_lse)
    kl_rows = torch.zeros(native_logits.shape[0], dtype=torch.float64)
    for start in range(0, native_logits.shape[1], vocabulary_chunk_size):
        stop = min(start + vocabulary_chunk_size, native_logits.shape[1])
        native_log_probability = (
            native_logits[:, start:stop] - native_lse[:, None]
        )
        candidate_log_probability = (
            candidate_logits[:, start:stop] - candidate_lse[:, None]
        )
        kl_rows += (
            native_log_probability.exp()
            * (native_log_probability - candidate_log_probability)
        ).sum(dim=1)
    matches = int(
        (
            native_logits.argmax(dim=-1)
            == candidate_logits.argmax(dim=-1)
        ).sum().item()
    )
    return nll.tolist(), kl_rows.tolist(), matches


def _trajectory_declarations(
    layer_count: int,
) -> tuple[
    tuple[str, ...],
    dict[str, tuple[int, ...]],
    dict[str, tuple[str, ...]],
]:
    width = max(2, len(str(layer_count)))
    subsets: dict[str, tuple[int, ...]] = {}
    prefix_ids: list[str] = []
    suffix_ids: list[str] = []
    for depth in range(1, layer_count):
        prefix_id = f"prefix_{depth:0{width}d}"
        suffix_id = f"suffix_{depth:0{width}d}"
        subsets[prefix_id] = tuple(range(depth))
        subsets[suffix_id] = tuple(range(layer_count - depth, layer_count))
        prefix_ids.append(prefix_id)
        suffix_ids.append(suffix_id)
    full_stack_id = "full_stack"
    subsets[full_stack_id] = tuple(range(layer_count))
    prefix_ids.append(full_stack_id)
    suffix_ids.append(full_stack_id)
    unique_order = tuple(
        (*prefix_ids[:-1], *suffix_ids[:-1], full_stack_id)
    )
    paths = {
        "prefix": tuple(prefix_ids),
        "suffix": tuple(suffix_ids),
    }
    return unique_order, subsets, paths


def _marginal_changes(
    conditions: Mapping[str, Mapping[str, float]],
    paths: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[dict[str, object], ...]]:
    result: dict[str, tuple[dict[str, object], ...]] = {}
    for trajectory, condition_ids in paths.items():
        previous = conditions["native"]
        rows: list[dict[str, object]] = []
        for depth, condition_id in enumerate(condition_ids, start=1):
            current = conditions[condition_id]
            rows.append(
                {
                    "depth": depth,
                    "condition_id": condition_id,
                    "delta_nll_from_previous_depth": (
                        current["nll_per_token"]
                        - previous["nll_per_token"]
                    ),
                    "delta_native_to_candidate_kl_from_previous_depth": (
                        current["native_to_candidate_kl_per_token"]
                        - previous["native_to_candidate_kl_per_token"]
                    ),
                    "delta_top1_agreement_from_previous_depth": (
                        current["top1_agreement_to_native"]
                        - previous["top1_agreement_to_native"]
                    ),
                }
            )
            previous = current
        result[trajectory] = tuple(rows)
    return result


def evaluate_full_mlp_stack_trajectory(
    adapter: Gemma3CausalLMAdapter,
    executor: FrozenFullMLPStackTrajectoryExecutor,
    batches: Sequence[CalibrationBatch],
    *,
    expected_example_ids: Sequence[str],
    expected_mode_counts_by_layer: Sequence[int],
    vocabulary_chunk_size: int = 16384,
    assessment_role: str = _ASSESSMENT_ROLE,
) -> dict[str, object]:
    """Evaluate every cumulative prefix and suffix of one frozen full stack."""

    if not isinstance(executor, FrozenFullMLPStackTrajectoryExecutor):
        raise TypeError(
            "executor must be FrozenFullMLPStackTrajectoryExecutor"
        )
    if executor.adapter is not adapter:
        raise ValueError("executor and evaluator adapters differ")
    materialized = tuple(batches)
    if (
        not materialized
        or any(not isinstance(batch, CalibrationBatch) for batch in materialized)
    ):
        raise ValueError("batches must contain CalibrationBatch values")
    if (
        isinstance(expected_example_ids, (str, bytes))
        or not isinstance(expected_example_ids, Sequence)
    ):
        raise TypeError("expected_example_ids must be a sequence")
    expected_ids = tuple(expected_example_ids)
    observed_ids = tuple(
        example_id
        for batch in materialized
        for example_id in (
            batch.example_ids if batch.example_ids is not None else ()
        )
    )
    if (
        not expected_ids
        or len(expected_ids) != len(set(expected_ids))
        or any(not isinstance(value, str) or not value for value in expected_ids)
        or any(batch.example_ids is None for batch in materialized)
        or observed_ids != expected_ids
    ):
        raise ValueError(
            "assessment batches do not match the declared example membership"
        )
    if (
        isinstance(expected_mode_counts_by_layer, (str, bytes))
        or not isinstance(expected_mode_counts_by_layer, Sequence)
    ):
        raise TypeError("expected_mode_counts_by_layer must be a sequence")
    mode_counts = tuple(expected_mode_counts_by_layer)
    if (
        not mode_counts
        or any(type(value) is not int or value <= 0 for value in mode_counts)
        or mode_counts != executor.mode_counts_by_layer
        or executor.replaced_layer_count != len(mode_counts)
        or executor.removed_mode_count != sum(mode_counts)
        or tuple(executor.compiled_mlps)
        != tuple(str(index) for index in range(len(mode_counts)))
        or any(
            not executor.compiled_mlps[str(index)].is_full_native_replacement
            or executor.compiled_mlps[str(index)].removed_mode_indices
            != tuple(range(mode_counts[index]))
            for index in range(len(mode_counts))
        )
    ):
        raise ValueError(
            "executor does not exactly cover every declared layer and mode"
        )
    if type(vocabulary_chunk_size) is not int or vocabulary_chunk_size <= 0:
        raise ValueError("vocabulary_chunk_size must be positive")
    if assessment_role != _ASSESSMENT_ROLE:
        raise ValueError(
            "trajectory evaluation cannot relabel open development as heldout"
        )

    condition_order, condition_subsets, paths = _trajectory_declarations(
        len(mode_counts)
    )
    nll_terms: dict[str, list[float]] = {
        condition_id: [] for condition_id in condition_order
    }
    kl_terms: dict[str, list[float]] = {
        condition_id: [] for condition_id in condition_order
    }
    top1_matches = {condition_id: 0 for condition_id in condition_order}
    native_nll_terms: list[float] = []
    supervised_tokens = 0
    logical_valid_tokens = 0

    subset_to_condition = {
        condition_subsets[condition_id]: condition_id
        for condition_id in condition_order
    }
    ordered_subsets = tuple(
        condition_subsets[condition_id] for condition_id in condition_order
    )
    for batch in materialized:
        native_selected: Tensor | None = None
        native_targets: Tensor | None = None
        visited: list[str] = []

        def consume_native(output: object) -> None:
            nonlocal native_selected, native_targets
            native_selected, native_targets = _selected_logits_and_targets(
                _model_logits(output),
                batch,
            )
            native_nll_terms.extend(
                _nll_terms(native_selected, native_targets)
            )

        def consume_subset(
            execution: FullMLPStackSubsetExecution,
        ) -> None:
            if native_selected is None or native_targets is None:
                raise RuntimeError("native trajectory output was not visited")
            condition_id = subset_to_condition.get(
                execution.generated_layer_ordinals
            )
            if condition_id is None:
                raise ValueError("trajectory returned an undeclared subset")
            candidate, targets = _selected_logits_and_targets(
                _model_logits(execution.model_output),
                batch,
            )
            if not torch.equal(targets, native_targets):
                raise ValueError("trajectory evaluation targets drifted")
            expected_accounting = executor.subset_accounting(
                execution.generated_layer_ordinals,
                valid_tokens=execution.valid_tokens,
            )
            if execution.accounting != expected_accounting:
                raise ValueError("trajectory subset accounting drifted")
            candidate_nll, candidate_kl, matches = _candidate_terms(
                native_selected,
                candidate,
                native_targets,
                vocabulary_chunk_size=vocabulary_chunk_size,
            )
            nll_terms[condition_id].extend(candidate_nll)
            kl_terms[condition_id].extend(candidate_kl)
            top1_matches[condition_id] += matches
            visited.append(condition_id)

        with torch.no_grad():
            valid_tokens = executor.visit_native_and_subsets(
                batch.model_inputs,
                generated_layer_subsets=ordered_subsets,
                native_visitor=consume_native,
                subset_visitor=consume_subset,
            )
        if tuple(visited) != condition_order:
            raise RuntimeError("trajectory condition visitation order drifted")
        expected_valid = int(batch.valid_positions.sum().item())
        if valid_tokens != expected_valid:
            raise ValueError("trajectory valid-token accounting drifted")
        if native_targets is None:
            raise RuntimeError("native trajectory output was not consumed")
        supervised_tokens += native_targets.numel()
        logical_valid_tokens += valid_tokens

    if supervised_tokens <= 0:
        raise ValueError("assessment has no supervised tokens")
    native_nll = math.fsum(native_nll_terms) / supervised_tokens
    conditions: dict[str, dict[str, float]] = {
        "native": {
            "nll_per_token": native_nll,
            "delta_nll_per_token": 0.0,
            "native_to_candidate_kl_per_token": 0.0,
            "top1_agreement_to_native": 1.0,
        }
    }
    for condition_id in condition_order:
        nll = math.fsum(nll_terms[condition_id]) / supervised_tokens
        conditions[condition_id] = {
            "nll_per_token": nll,
            "delta_nll_per_token": nll - native_nll,
            "native_to_candidate_kl_per_token": max(
                math.fsum(kl_terms[condition_id]) / supervised_tokens,
                0.0,
            ),
            "top1_agreement_to_native": (
                top1_matches[condition_id] / supervised_tokens
            ),
        }

    declarations: dict[str, dict[str, object]] = {
        "native": {
            "trajectory": "native",
            "depth": 0,
            "generated_layer_ordinals": (),
        }
    }
    for trajectory, condition_ids in paths.items():
        for depth, condition_id in enumerate(condition_ids, start=1):
            existing = declarations.get(condition_id)
            value = {
                "trajectory": (
                    "prefix_and_suffix"
                    if condition_id == "full_stack"
                    else trajectory
                ),
                "depth": depth,
                "generated_layer_ordinals": condition_subsets[condition_id],
            }
            if existing is not None and existing != value:
                raise RuntimeError("trajectory endpoint declaration drifted")
            declarations[condition_id] = value

    all_condition_ids = ("native", *condition_order)
    resources = {
        condition_id: executor.subset_accounting(
            (
                ()
                if condition_id == "native"
                else condition_subsets[condition_id]
            ),
            valid_tokens=logical_valid_tokens,
        ).to_dict()
        for condition_id in all_condition_ids
    }
    return {
        "execution_path": "frozen_full_mlp_stack_prefix_suffix_trajectory",
        "assessment_role": assessment_role,
        "heldout_confirmation": False,
        "assessment_membership_exact": True,
        "assessment_used_for_fitting": False,
        "trajectory_used_for_refitting": False,
        "supervised_tokens": supervised_tokens,
        "logical_valid_tokens": logical_valid_tokens,
        "declared_scope": {
            "layer_count": len(mode_counts),
            "removed_mode_count": sum(mode_counts),
            "mode_counts_by_layer": mode_counts,
            "all_declared_layers_and_modes_compiled_once": True,
        },
        "condition_order": all_condition_ids,
        "condition_declarations": declarations,
        "trajectory_condition_ids": paths,
        "full_stack_endpoint_evaluated_once": True,
        "conditions": conditions,
        "marginal_changes": _marginal_changes(conditions, paths),
        "resource_accounting": resources,
        "interpretation_guard": {
            "marginal_changes_are_path_conditioned": True,
            "marginal_changes_are_not_isolated_layer_causal_effects": True,
        },
        "latency_or_kernel_speed_claim": False,
    }
