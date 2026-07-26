"""Full-width source-independent executors for one residual layer boundary.

The scientific purpose of this executor is deliberately narrower than modal
compression: it asks whether a newly trained graph can replace one source
layer without reading that layer's output.  The residual delta remains at the
complete source width, so a failed run is a generator failure rather than a
rank-selection failure.

The executable graph is the existing
:class:`StaticTransformerSpanExecutor` with an identity residual decoder.  A
storage-matched same-position control retains the identical parameter shapes
but zeros every causal-attention branch output.  The control is diagnostic:
the dense PyTorch reference still issues the attention kernels, while its
result cannot influence the predicted boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .activations import ActivationTrace
from .adapters import SegmentRun, SequenceContext
from .compiler.manifest import CompiledSegment
from .static_transformer_span_executor import (
    StaticTransformerSpanAccounting,
    StaticTransformerSpanExecution,
    StaticTransformerSpanExecutor,
    StaticTransformerSpanExecutorConfig,
)


_ARTIFACT_KIND = "fisher_graph.full_width_single_layer_executor"
_FORMAT_VERSION = 1
_FINGERPRINT_DOMAIN = b"fisher_graph.full_width_single_layer_executor.v1\0"
_ARTIFACT_FIELDS = {
    "artifact_kind",
    "format_version",
    "causal_edges_enabled",
    "executor",
    "execution_fingerprint",
}


@dataclass(frozen=True, slots=True)
class FullWidthSingleLayerExecutorConfig:
    """Architecture and causal-control mode for one replacement layer."""

    residual_width: int
    hidden_width: int
    layer_count: int
    head_count: int
    feed_forward_width: int
    causal_edges_enabled: bool = True

    def __post_init__(self) -> None:
        for name in (
            "residual_width",
            "hidden_width",
            "layer_count",
            "head_count",
            "feed_forward_width",
        ):
            if type(getattr(self, name)) is not int or getattr(
                self,
                name,
            ) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_width % self.head_count:
            raise ValueError("hidden_width must be divisible by head_count")
        if type(self.causal_edges_enabled) is not bool:
            raise TypeError("causal_edges_enabled must be boolean")

    def span_config(self) -> StaticTransformerSpanExecutorConfig:
        return StaticTransformerSpanExecutorConfig(
            residual_width=self.residual_width,
            hidden_width=self.hidden_width,
            layer_count=self.layer_count,
            head_count=self.head_count,
            feed_forward_width=self.feed_forward_width,
            retained_rank=self.residual_width,
        )


class FullWidthSingleLayerExecutor(nn.Module):
    """Wrap a full-width mini-transformer and an optional causal-edge ablation."""

    def __init__(
        self,
        config: FullWidthSingleLayerExecutorConfig,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, FullWidthSingleLayerExecutorConfig):
            raise TypeError(
                "config must be a FullWidthSingleLayerExecutorConfig"
            )
        self.config = config
        decoder = torch.eye(
            config.residual_width,
            dtype=dtype,
            device=device,
        )
        self.executor = StaticTransformerSpanExecutor(
            config.span_config(),
            decoder,
            dtype=dtype,
            device=device,
        )

    @property
    def width(self) -> int:
        return self.config.residual_width

    @property
    def retained_rank(self) -> int:
        return self.width

    @property
    def dtype(self) -> torch.dtype:
        return self.executor.dtype

    @property
    def device(self) -> torch.device:
        return self.executor.device

    @property
    def learned_parameter_count(self) -> int:
        return self.executor.learned_parameter_count

    @property
    def fixed_runtime_coefficient_count(self) -> int:
        return self.executor.fixed_runtime_coefficient_count

    @property
    def total_runtime_coefficient_count(self) -> int:
        return self.executor.total_runtime_coefficient_count

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
    def causal_edge_control(self) -> str:
        return (
            "enabled"
            if self.config.causal_edges_enabled
            else "attention_output_zeroed_storage_matched"
        )

    @contextmanager
    def _causal_edge_mode(self) -> Iterator[None]:
        if self.config.causal_edges_enabled:
            yield
            return

        handles = []
        for block in self.executor.blocks:

            def zero_attention(
                _module: nn.Module,
                _args: tuple[object, ...],
                output: object,
            ) -> Tensor:
                if not isinstance(output, Tensor):
                    raise TypeError(
                        "transformer attention control expected a Tensor"
                    )
                return torch.zeros_like(output)

            handles.append(
                block.attention.register_forward_hook(zero_attention)
            )
        try:
            yield
        finally:
            for handle in reversed(handles):
                handle.remove()

    def forward_components(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        prefix: str = "full_width_single_layer",
    ) -> StaticTransformerSpanExecution:
        with self._causal_edge_mode():
            return self.executor.forward_components(
                hidden_states,
                sequence,
                trace=trace,
                prefix=prefix,
            )

    def forward(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        prefix: str = "full_width_single_layer",
    ) -> Tensor:
        return self.forward_components(
            hidden_states,
            sequence,
            trace=trace,
            prefix=prefix,
        ).output

    def run(
        self,
        segment: CompiledSegment,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        with self._causal_edge_mode():
            return self.executor.run(
                segment,
                hidden_states,
                sequence,
                trace=trace,
            )

    def logical_accounting(
        self,
        sequence: SequenceContext,
    ) -> StaticTransformerSpanAccounting:
        return self.executor.logical_accounting(sequence)

    def execution_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_DOMAIN)
        digest.update(
            json.dumps(
                {
                    "causal_edges_enabled": (
                        self.config.causal_edges_enabled
                    ),
                    "executor": self.executor.execution_fingerprint(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def artifact_state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "causal_edges_enabled": self.config.causal_edges_enabled,
            "executor": self.executor.artifact_state_dict(),
            "execution_fingerprint": self.execution_fingerprint(),
        }

    @classmethod
    def from_artifact_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        map_location: torch.device | str = "cpu",
    ) -> FullWidthSingleLayerExecutor:
        if not isinstance(state, Mapping) or set(state) != _ARTIFACT_FIELDS:
            raise ValueError(
                "full-width single-layer executor fields are invalid"
            )
        if (
            state["artifact_kind"] != _ARTIFACT_KIND
            or state["format_version"] != _FORMAT_VERSION
            or type(state["causal_edges_enabled"]) is not bool
            or not isinstance(state["execution_fingerprint"], str)
        ):
            raise ValueError(
                "unsupported full-width single-layer executor artifact"
            )
        raw_executor = state["executor"]
        if not isinstance(raw_executor, Mapping):
            raise ValueError("full-width executor state is invalid")
        inner = StaticTransformerSpanExecutor.from_artifact_state_dict(
            raw_executor,
            map_location=map_location,
        )
        inner_config = inner.config
        if inner_config.retained_rank != inner_config.residual_width:
            raise ValueError("full-width executor decoder rank is reduced")
        identity = torch.eye(
            inner_config.residual_width,
            dtype=inner.decoder.dtype,
            device=inner.decoder.device,
        )
        if not torch.equal(inner.decoder, identity):
            raise ValueError(
                "full-width executor decoder must be the exact identity"
            )
        config = FullWidthSingleLayerExecutorConfig(
            residual_width=inner_config.residual_width,
            hidden_width=inner_config.hidden_width,
            layer_count=inner_config.layer_count,
            head_count=inner_config.head_count,
            feed_forward_width=inner_config.feed_forward_width,
            causal_edges_enabled=state["causal_edges_enabled"],
        )
        result = cls(
            config,
            dtype=inner.dtype,
            device=map_location,
        )
        result.executor = inner
        if result.execution_fingerprint() != state["execution_fingerprint"]:
            raise ValueError(
                "full-width single-layer execution fingerprint mismatch"
            )
        result.eval()
        return result


__all__ = [
    "FullWidthSingleLayerExecutor",
    "FullWidthSingleLayerExecutorConfig",
]
