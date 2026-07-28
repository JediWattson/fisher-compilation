"""Prepared full-stack Gemma MLP switching for Torch benchmarks.

The ordinary full-stack executor is deliberately defensive: every call
authenticates the live model, installs temporary overlays, and restores the
native stack.  Those costs are useful at an artifact boundary but do not
belong inside a kernel or model latency measurement.

This module separates those phases.  Construction validates every named
replacement catalog through :class:`Gemma3FullMLPStackExecutor` exactly once
and lowers each full replacement to its two runtime linear projections.
Optional fused variants materialize those factors as one affine residual map.
``switch`` changes the complete live MLP stack outside a timed region.
``forward`` then delegates directly to the already-selected model without
hashing, integrity validation, device conversion, or layer mutation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .gemma3_full_mlp_stack_executor import Gemma3FullMLPStackExecutor
from .gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorMLP,
    Gemma3ModalGeneratorReplacement,
)


__all__ = [
    "PreparedGemma3FullMLPStackScopeAccounting",
    "PreparedGemma3FullMLPStackSwitcher",
]


_NATIVE_SCOPE = "native"


@dataclass(frozen=True, slots=True)
class PreparedGemma3FullMLPStackScopeAccounting:
    """Exact resident parameters and linear work for one MLP-stack scope."""

    scope: str
    implementation: str
    layer_count: int
    learned_parameter_count: int
    linear_macs_per_token: int
    linear_bias_additions_per_token: int

    def __post_init__(self) -> None:
        if not isinstance(self.scope, str) or not self.scope:
            raise ValueError("scope must be a nonempty string")
        if self.implementation not in {"native", "factorized", "fused"}:
            raise ValueError("implementation is not recognized")
        for name in (
            "layer_count",
            "learned_parameter_count",
            "linear_macs_per_token",
            "linear_bias_additions_per_token",
        ):
            value = getattr(self, name)
            minimum = 1 if name == "layer_count" else 0
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"{name} must be an integer >= {minimum}"
                )


class _PreparedFullGeneratorMLP(nn.Module):
    """Hash-free hot-path lowering of one validated full replacement."""

    def __init__(self, compiled: Gemma3ModalGeneratorMLP) -> None:
        super().__init__()
        if not isinstance(compiled, Gemma3ModalGeneratorMLP):
            raise TypeError("compiled MLP must be a Gemma3ModalGeneratorMLP")
        if not compiled.is_full_native_replacement:
            raise ValueError(
                "prepared Gemma candidates require full MLP replacements"
            )
        self.input_projection = compiled.generator_input_proj
        self.output_projection = compiled.generator_output_proj
        self.generator_id = compiled.generator_id
        self.generator_artifact_sha256 = (
            compiled.generator_artifact_sha256
        )
        self.requires_grad_(False)
        self.eval()

    def forward(self, normalized_hidden_states: Tensor) -> Tensor:
        return self.output_projection(
            self.input_projection(normalized_hidden_states)
        )


def _fuse_generator_mlp(
    factorized: _PreparedFullGeneratorMLP,
) -> nn.Linear:
    """Materialize one exact affine generator in the runtime dtype."""

    input_projection = factorized.input_projection
    output_projection = factorized.output_projection
    if (
        input_projection.bias is not None
        or input_projection.out_features
        != output_projection.in_features
        or input_projection.in_features
        != output_projection.out_features
    ):
        raise ValueError("factorized generator cannot be fused")
    fused = nn.Linear(
        input_projection.in_features,
        output_projection.out_features,
        bias=output_projection.bias is not None,
        device=input_projection.weight.device,
        dtype=input_projection.weight.dtype,
    )
    # CPU float64 makes this deterministic across runtime accelerators and
    # avoids doing the one-time matrix product in a reduced runtime dtype.
    input_weight = input_projection.weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    output_weight = output_projection.weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    dense_weight = output_weight @ input_weight
    if not bool(torch.isfinite(dense_weight).all()):
        raise ValueError("fused generator weight is not finite")
    with torch.no_grad():
        fused.weight.copy_(
            dense_weight.to(
                device=fused.weight.device,
                dtype=fused.weight.dtype,
            )
        )
        if fused.bias is not None:
            assert output_projection.bias is not None
            fused.bias.copy_(
                output_projection.bias.detach().to(
                    device=fused.bias.device,
                    dtype=fused.bias.dtype,
                )
            )
    fused.requires_grad_(False)
    fused.eval()
    return fused


def _scope_accounting(
    scope: str,
    implementation: str,
    modules: Sequence[nn.Module],
) -> PreparedGemma3FullMLPStackScopeAccounting:
    if not modules:
        raise ValueError("scope modules must be nonempty")
    parameters = sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
    )
    linears = tuple(
        child
        for module in modules
        for child in module.modules()
        if isinstance(child, nn.Linear)
    )
    return PreparedGemma3FullMLPStackScopeAccounting(
        scope=scope,
        implementation=implementation,
        layer_count=len(modules),
        learned_parameter_count=parameters,
        linear_macs_per_token=sum(
            linear.weight.numel() for linear in linears
        ),
        linear_bias_additions_per_token=sum(
            0 if linear.bias is None else linear.bias.numel()
            for linear in linears
        ),
    )


class PreparedGemma3FullMLPStackSwitcher(nn.Module):
    """Validate named Gemma candidates once and switch whole MLP stacks.

    ``native`` is a reserved scope referring to the source model.  Factorized
    scopes are supplied as complete, ordered replacement catalogs; optional
    fused scopes name one of those factorized sources.  ``switch`` performs
    only identity checks and module assignment; it never executes the model.
    ``forward`` performs only a reentrancy guard and direct delegation to the
    live model.

    The object owns the prepared candidates and retains strong references to
    every native MLP.  ``close`` and context-manager exit restore the complete
    native stack.  A closed switcher cannot be reused.
    """

    def __init__(
        self,
        adapter: Gemma3CausalLMAdapter,
        candidates: Mapping[
            str,
            Sequence[Gemma3ModalGeneratorReplacement],
        ],
        *,
        fused_variants: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if not isinstance(candidates, Mapping) or not candidates:
            raise ValueError("candidates must be a nonempty mapping")

        factorized_names = tuple(candidates)
        for name in factorized_names:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "candidate names must be nonempty strings"
                )
            if name == _NATIVE_SCOPE:
                raise ValueError("'native' is a reserved candidate scope")
        if fused_variants is None:
            fused_variants = {}
        elif not isinstance(fused_variants, Mapping):
            raise TypeError("fused_variants must be a mapping")
        fused_names = tuple(fused_variants)
        for name, source in fused_variants.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(source, str)
                or not source
            ):
                raise ValueError(
                    "fused variant names and sources must be nonempty strings"
                )
            if name == _NATIVE_SCOPE:
                raise ValueError("'native' is a reserved candidate scope")
            if name in factorized_names:
                raise ValueError(
                    "fused variant names must not collide with candidates"
                )
            if source not in factorized_names:
                raise ValueError(
                    f"fused variant {name!r} must name a factorized candidate"
                )

        layers = self._model_layers(adapter)
        layer_count = len(layers)
        if layer_count <= 0:
            raise ValueError("Gemma model must expose a nonempty layer stack")
        native_mlps: list[nn.Module] = []
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if not isinstance(mlp, nn.Module):
                raise TypeError("every live Gemma layer must expose an MLP")
            native_mlps.append(mlp)

        factorized_stacks: list[nn.ModuleList] = []
        for name in factorized_names:
            catalog = candidates[name]
            validated = Gemma3FullMLPStackExecutor(adapter, catalog)
            if validated.replaced_layer_count != layer_count:
                raise RuntimeError(
                    f"candidate {name!r} did not compile a complete stack"
                )
            prepared = nn.ModuleList(
                _PreparedFullGeneratorMLP(
                    validated.compiled_mlps[str(ordinal)]
                )
                for ordinal in range(layer_count)
            )
            if len(prepared) != layer_count:
                raise RuntimeError(
                    f"candidate {name!r} lost a compiled Gemma layer"
                )
            factorized_stacks.append(prepared)

        factorized_indices = {
            name: index for index, name in enumerate(factorized_names)
        }
        fused_stacks: list[nn.ModuleList] = []
        for name in fused_names:
            source = fused_variants[name]
            factorized = factorized_stacks[
                factorized_indices[source]
            ]
            fused = nn.ModuleList(
                _fuse_generator_mlp(layer)
                for layer in factorized
            )
            if len(fused) != layer_count:
                raise RuntimeError(
                    f"fused candidate {name!r} lost a Gemma layer"
                )
            fused_stacks.append(fused)

        # Candidate projections are now owned by the prepared modules.  The
        # validating executors are intentionally not retained: their
        # per-execution hashing and temporary-overlay path is not part of this
        # runtime.
        prepared_stacks = (*factorized_stacks, *fused_stacks)
        names = (*factorized_names, *fused_names)
        self.prepared_stacks = nn.ModuleList(prepared_stacks)
        self._adapter = adapter
        self._candidate_names = names
        self._factorized_candidate_names = factorized_names
        self._fused_variant_names = fused_names
        self._fused_variants = MappingProxyType(dict(fused_variants))
        self._candidate_indices = {
            name: index for index, name in enumerate(names)
        }
        self._native_mlps = tuple(native_mlps)
        self._layer_count = layer_count
        self._active_scope = _NATIVE_SCOPE
        self._forward_active = False
        self._entered = False
        self._closed = False
        accounting = {
            _NATIVE_SCOPE: _scope_accounting(
                _NATIVE_SCOPE,
                "native",
                self._native_mlps,
            )
        }
        for name, stack in zip(
            factorized_names,
            factorized_stacks,
            strict=True,
        ):
            accounting[name] = _scope_accounting(
                name,
                "factorized",
                tuple(stack),
            )
        for name, stack in zip(
            fused_names,
            fused_stacks,
            strict=True,
        ):
            accounting[name] = _scope_accounting(
                name,
                "fused",
                tuple(stack),
            )
        self._scope_accounting = MappingProxyType(accounting)
        self._scope_parameter_counts = MappingProxyType(
            {
                name: value.learned_parameter_count
                for name, value in accounting.items()
            }
        )
        self._scope_macs_per_token = MappingProxyType(
            {
                name: value.linear_macs_per_token
                for name, value in accounting.items()
            }
        )
        self.requires_grad_(False)
        self.eval()
        self._assert_live_scope(_NATIVE_SCOPE)

    @staticmethod
    def _model_layers(adapter: Gemma3CausalLMAdapter) -> object:
        model = getattr(adapter.module, "model", None)
        layers = getattr(model, "layers", None)
        if (
            layers is None
            or not hasattr(layers, "__len__")
            or not hasattr(layers, "__getitem__")
        ):
            raise TypeError("live Gemma model does not expose indexed layers")
        return layers

    def _live_layers(self) -> object:
        layers = self._model_layers(self._adapter)
        if len(layers) != self._layer_count:
            raise RuntimeError("live Gemma layer count drifted")
        return layers

    def _stack_for_scope(self, scope: str) -> tuple[nn.Module, ...]:
        if scope == _NATIVE_SCOPE:
            return self._native_mlps
        index = self._candidate_indices.get(scope)
        if index is None:
            available = ", ".join(repr(value) for value in self.scopes)
            raise ValueError(
                f"unknown Gemma MLP scope {scope!r}; expected one of "
                f"{available}"
            )
        return tuple(self.prepared_stacks[index])

    def _assert_live_scope(self, scope: str) -> None:
        layers = self._live_layers()
        expected = self._stack_for_scope(scope)
        if any(
            getattr(layers[ordinal], "mlp", None) is not expected[ordinal]
            for ordinal in range(self._layer_count)
        ):
            raise RuntimeError("live Gemma MLP stack identity drifted")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("prepared Gemma MLP switcher is closed")

    @property
    def adapter(self) -> Gemma3CausalLMAdapter:
        return self._adapter

    @property
    def native_scope(self) -> str:
        return _NATIVE_SCOPE

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return self._candidate_names

    @property
    def factorized_candidate_names(self) -> tuple[str, ...]:
        return self._factorized_candidate_names

    @property
    def fused_variant_names(self) -> tuple[str, ...]:
        return self._fused_variant_names

    @property
    def fused_variants(self) -> Mapping[str, str]:
        return self._fused_variants

    @property
    def scopes(self) -> tuple[str, ...]:
        return (_NATIVE_SCOPE, *self._candidate_names)

    @property
    def scope_accounting(
        self,
    ) -> Mapping[str, PreparedGemma3FullMLPStackScopeAccounting]:
        return self._scope_accounting

    @property
    def scope_parameter_counts(self) -> Mapping[str, int]:
        return self._scope_parameter_counts

    @property
    def scope_macs_per_token(self) -> Mapping[str, int]:
        return self._scope_macs_per_token

    @property
    def active_scope(self) -> str:
        return self._active_scope

    @property
    def closed(self) -> bool:
        return self._closed

    def switch(self, scope: str) -> None:
        """Install one complete prevalidated stack without executing it."""

        self._ensure_open()
        if not isinstance(scope, str):
            raise TypeError("scope must be a string")
        if self._forward_active:
            raise RuntimeError(
                "cannot switch Gemma MLP scope during a forward call"
            )
        target = self._stack_for_scope(scope)
        self._assert_live_scope(self._active_scope)
        if scope == self._active_scope:
            return

        layers = self._live_layers()
        previous = tuple(
            getattr(layers[ordinal], "mlp", None)
            for ordinal in range(self._layer_count)
        )
        try:
            for ordinal, mlp in enumerate(target):
                layers[ordinal].mlp = mlp
        except BaseException:
            for ordinal, mlp in enumerate(previous):
                layers[ordinal].mlp = mlp
            raise
        self._active_scope = scope
        self._assert_live_scope(scope)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Run the already-selected model with no hashing or layer switching."""

        self._ensure_open()
        if self._forward_active:
            raise RuntimeError(
                "prepared Gemma MLP forward is not reentrant"
            )
        self._forward_active = True
        try:
            return self._adapter.module(*args, **kwargs)
        finally:
            self._forward_active = False

    def close(self) -> None:
        """Restore the native MLP stack and permanently close the switcher."""

        if self._closed:
            return
        if self._forward_active:
            raise RuntimeError(
                "cannot close Gemma MLP switcher during a forward call"
            )
        layers = self._live_layers()
        for ordinal, native in enumerate(self._native_mlps):
            layers[ordinal].mlp = native
        self._active_scope = _NATIVE_SCOPE
        self._assert_live_scope(_NATIVE_SCOPE)
        self._closed = True

    def __enter__(self) -> PreparedGemma3FullMLPStackSwitcher:
        self._ensure_open()
        if self._entered:
            raise RuntimeError(
                "prepared Gemma MLP switcher context is not reentrant"
            )
        self._entered = True
        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> bool:
        try:
            self.close()
        finally:
            self._entered = False
        return False

    def train(
        self,
        mode: bool = True,
    ) -> PreparedGemma3FullMLPStackSwitcher:
        if mode:
            raise ValueError(
                "prepared Gemma MLP switching requires frozen eval candidates"
            )
        return super().train(False)
