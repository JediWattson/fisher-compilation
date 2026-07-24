"""Explicit activation capture without global or module hooks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Collection, Iterator, Mapping

import torch
from torch import Tensor

ActivationIntervention = Callable[[Tensor], Tensor]


class ActivationTrace(Mapping[str, Tensor]):
    """Activations from one forward pass.

    Tensors are kept attached to their autograd graph. Non-leaf tensors retain
    their gradients, so ``trace.gradients()`` is also useful after ``backward``.
    Use ``detached()`` when storing a trace beyond the current training step.
    """

    def __init__(
        self,
        *,
        retain_grad: bool = True,
        interventions: Mapping[str, ActivationIntervention] | None = None,
        store: bool = True,
        capture_sites: Collection[str] | None = None,
    ) -> None:
        self._tensors: OrderedDict[str, Tensor] = OrderedDict()
        self.retain_grad = retain_grad
        self.interventions = dict(interventions or {})
        for name, intervention in self.interventions.items():
            if not isinstance(name, str):
                raise TypeError("activation intervention names must be strings")
            if not callable(intervention):
                raise TypeError(
                    f"intervention for {name!r} must be callable"
                )
        self.store = store
        if capture_sites is None:
            self.capture_sites: frozenset[str] | None = None
        else:
            if any(
                not isinstance(name, str) or not name
                for name in capture_sites
            ):
                raise TypeError(
                    "capture site names must be nonempty strings"
                )
            self.capture_sites = frozenset(capture_sites)
        self._seen: set[str] = set()
        self._applied_interventions: set[str] = set()

    def record(self, name: str, tensor: Tensor) -> Tensor:
        if name in self._seen:
            raise ValueError(f"activation already recorded: {name}")
        self._seen.add(name)
        intervention = self.interventions.get(name)
        if intervention is not None:
            original = tensor
            tensor = intervention(original)
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"intervention for {name!r} must return a Tensor"
                )
            if tensor.shape != original.shape:
                raise ValueError(
                    f"intervention for {name!r} changed shape from "
                    f"{tuple(original.shape)} to {tuple(tensor.shape)}"
                )
            if tensor.dtype != original.dtype:
                raise ValueError(
                    f"intervention for {name!r} changed dtype from "
                    f"{original.dtype} to {tensor.dtype}"
                )
            if tensor.device != original.device:
                raise ValueError(
                    f"intervention for {name!r} changed device from "
                    f"{original.device} to {tensor.device}"
                )
            self._applied_interventions.add(name)
        if self.store and (
            self.capture_sites is None or name in self.capture_sites
        ):
            if self.retain_grad and tensor.requires_grad:
                tensor.retain_grad()
            self._tensors[name] = tensor
        return tensor

    def assert_all_interventions_applied(self) -> None:
        missing = set(self.interventions) - self._applied_interventions
        if missing:
            raise KeyError(f"unknown activation intervention taps: {sorted(missing)}")

    def assert_all_captures_seen(self) -> None:
        if self.capture_sites is None:
            return
        missing = self.capture_sites - self._seen
        if missing:
            raise KeyError(
                f"activation capture sites were not executed: {sorted(missing)}"
            )

    def __getitem__(self, name: str) -> Tensor:
        return self._tensors[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tensors)

    def detached(self, *, cpu: bool = True, clone: bool = True) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        for name, tensor in self._tensors.items():
            value = tensor.detach()
            if cpu:
                value = value.cpu()
            if clone:
                value = value.clone()
            result[name] = value
        return result

    def gradients(
        self, *, strict: bool = False, cpu: bool = False
    ) -> dict[str, Tensor | None]:
        result: dict[str, Tensor | None] = {}
        for name, tensor in self._tensors.items():
            gradient = tensor.grad
            if gradient is None and strict and tensor.requires_grad:
                raise RuntimeError(
                    f"activation {name!r} has no gradient; call backward() first"
                )
            if gradient is not None:
                gradient = gradient.detach()
                if cpu:
                    gradient = gradient.cpu()
            result[name] = gradient
        return result


def record(trace: ActivationTrace | None, name: str, tensor: Tensor) -> Tensor:
    """Record ``tensor`` when tracing is enabled and always return it."""

    return trace.record(name, tensor) if trace is not None else tensor
