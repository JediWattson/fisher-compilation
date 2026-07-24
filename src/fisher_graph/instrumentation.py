"""Typed model instrumentation shared by source and compiled runtimes.

Fisher analysis needs only three things from a model runner:

* the module whose train/eval state must be preserved;
* an authoritative catalog of named activation layouts; and
* a forward method that can capture a requested subset of those activations.

``ModelAdapter`` already satisfies :class:`InstrumentedModel`.  Mixed or
compiled runtimes can satisfy the same contract through
:class:`InstrumentedModelBinding`, with backend-owned activation sites supplied
explicitly.  Requiring explicit metadata is important: a tap name alone cannot
establish whether a tensor is a residual-width vector, a modal vector, or a
higher-rank recurrent state.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Protocol

from torch import Tensor, nn

from .activations import ActivationIntervention
from .adapters.base import (
    ActivationSite,
    AdapterRun,
    ExecutionPhase,
    ModelAdapter,
)


class InstrumentedForward(Protocol):
    """A runner that supports adapter-compatible named activation capture."""

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        phase: ExecutionPhase = "prefill",
        cache_state: object | None = None,
        capture_sites: Collection[str] = (),
        interventions: Mapping[str, ActivationIntervention] | None = None,
        retain_gradients: bool = False,
    ) -> AdapterRun: ...


class InstrumentedModel(InstrumentedForward, Protocol):
    """Minimal model surface required by activation-Fisher collection."""

    @property
    def module(self) -> nn.Module: ...

    @property
    def activation_sites(self) -> tuple[ActivationSite, ...]: ...


class InstrumentedModelBinding:
    """Attach authoritative activation metadata to an instrumented runner.

    This is the bridge for runners such as ``MixedModelRuntime`` whose source
    adapter owns standard activation metadata while compiled backends add
    further capture sites.  Callers should include the source adapter's sites
    and one :class:`ActivationSite` for every backend-owned tap they want to
    expose to analysis.
    """

    def __init__(
        self,
        runner: InstrumentedForward,
        *,
        module: nn.Module,
        activation_sites: Collection[ActivationSite],
    ) -> None:
        if not callable(getattr(runner, "forward", None)):
            raise TypeError("runner must provide a callable forward method")
        if not isinstance(module, nn.Module):
            raise TypeError("module must be an nn.Module")
        sites = tuple(activation_sites)
        if any(not isinstance(site, ActivationSite) for site in sites):
            raise TypeError(
                "activation_sites must contain ActivationSite values"
            )
        names = tuple(site.id for site in sites)
        if len(set(names)) != len(names):
            raise ValueError("activation site ids must be unique")

        supported = getattr(runner, "capture_sites", None)
        if supported is not None:
            try:
                unsupported = set(names) - set(supported)
            except TypeError as error:
                raise TypeError(
                    "runner capture_sites must be a collection of names"
                ) from error
            if unsupported:
                raise ValueError(
                    "activation metadata contains sites the runner cannot "
                    f"capture: {sorted(unsupported)}"
                )

        self._runner = runner
        self._module = module
        self._activation_sites = sites
        self._sites_by_id = {site.id: site for site in sites}

    @classmethod
    def from_runtime(
        cls,
        runtime: InstrumentedForward,
        *,
        adapter: ModelAdapter,
        compiled_sites: Collection[ActivationSite] = (),
    ) -> InstrumentedModelBinding:
        """Bind a mixed runtime to its source and compiled site catalogs."""

        if not isinstance(adapter, ModelAdapter):
            raise TypeError("adapter must implement ModelAdapter")
        return cls(
            runtime,
            module=adapter.module,
            activation_sites=(
                *adapter.activation_sites,
                *tuple(compiled_sites),
            ),
        )

    @property
    def module(self) -> nn.Module:
        return self._module

    @property
    def activation_sites(self) -> tuple[ActivationSite, ...]:
        return self._activation_sites

    def activation_site(self, site_id: str) -> ActivationSite:
        try:
            return self._sites_by_id[site_id]
        except KeyError as error:
            raise KeyError(f"unknown activation site: {site_id!r}") from error

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        phase: ExecutionPhase = "prefill",
        cache_state: object | None = None,
        capture_sites: Collection[str] = (),
        interventions: Mapping[str, ActivationIntervention] | None = None,
        retain_gradients: bool = False,
    ) -> AdapterRun:
        requested = tuple(dict.fromkeys(capture_sites))
        if any(not isinstance(name, str) or not name for name in requested):
            raise TypeError("capture site names must be nonempty strings")
        unknown = set(requested) - self._sites_by_id.keys()
        if unknown:
            raise KeyError(
                f"unknown activation capture sites: {sorted(unknown)}"
            )
        run = self._runner.forward(
            model_inputs,
            phase=phase,
            cache_state=cache_state,
            capture_sites=requested,
            interventions=interventions,
            retain_gradients=retain_gradients,
        )
        if not isinstance(run, AdapterRun):
            raise TypeError("instrumented runner must return an AdapterRun")
        return run

    __call__ = forward


def validate_instrumented_model(model: InstrumentedModel) -> None:
    """Validate the small structural contract at a public API boundary."""

    if not callable(getattr(model, "forward", None)):
        raise TypeError(
            "model must implement the InstrumentedModel forward contract"
        )
    if not isinstance(getattr(model, "module", None), nn.Module):
        raise TypeError(
            "instrumented model must expose an nn.Module as 'module'"
        )
    sites = getattr(model, "activation_sites", None)
    if (
        not isinstance(sites, tuple)
        or any(not isinstance(site, ActivationSite) for site in sites)
    ):
        raise TypeError(
            "instrumented model activation_sites must be a tuple of "
            "ActivationSite values"
        )
    names = tuple(site.id for site in sites)
    if len(set(names)) != len(names):
        raise ValueError("instrumented model activation site ids must be unique")


__all__ = [
    "InstrumentedForward",
    "InstrumentedModel",
    "InstrumentedModelBinding",
    "validate_instrumented_model",
]
