"""Source-free transformer execution with a static modal output span.

This executor replaces a residual-width source span with a newly initialized
hidden-width transformer trunk:

```
projected = input_projection(source_prefix)
hidden = transformer_blocks(projected, key_valid_mask)
coordinates = output_head(hidden_norm(hidden[query_valid_mask]))
delta = coordinates @ decoder.T
output[query_valid_mask] = input[query_valid_mask] + delta
```

Only demanded query rows run the final norm, modal head, and decoder.  The
reference transformer trunk remains dense over the batch's longest active key
prefix: later trunk layers need every valid prefix row as a possible key/value
input.  :meth:`logical_accounting` reports ideal valid-row/causal-pair MACs
separately from the dense-prefix work issued by this PyTorch reference.

Non-query rows are returned bit-for-bit unchanged.  That terminal-demand
contract is appropriate when the compiled span ends immediately before a
row-selected model head.  It must not be used when a downstream segment needs
updated non-query residual rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .activations import ActivationTrace, record
from .adapters.base import (
    MaskPolicy,
    SegmentRun,
    SequenceContext,
    SequenceSpec,
)
from .compiler.capabilities import (
    CapabilityValues,
    LengthDomain,
    SequenceCapabilitySet,
)
from .compiler.manifest import CompiledSegment
from .config import TransformerConfig
from .layers import TransformerBlock


_ARTIFACT_KIND = "fisher_graph.static_transformer_span_executor"
_FORMAT_VERSION = 1
_FINGERPRINT_DOMAIN = b"fisher_graph.static_transformer_span_executor.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_FIELDS = {
    "residual_width",
    "hidden_width",
    "layer_count",
    "head_count",
    "feed_forward_width",
    "retained_rank",
}
_ARTIFACT_FIELDS = {
    "artifact_kind",
    "format_version",
    "contains_source_model_weights",
    "contains_source_fallback",
    "config",
    "dtype",
    "model_state_dict",
    "execution_fingerprint",
}
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError(f"unsupported executor dtype: {dtype}")
    return name


@dataclass(frozen=True, slots=True)
class StaticTransformerSpanExecutorConfig:
    """Architecture of one source-independent static modal span."""

    residual_width: int
    hidden_width: int
    layer_count: int
    head_count: int
    feed_forward_width: int
    retained_rank: int

    def __post_init__(self) -> None:
        for label in _CONFIG_FIELDS:
            _positive_integer(getattr(self, label), label=label)
        if self.hidden_width % self.head_count:
            raise ValueError("hidden_width must be divisible by head_count")
        if self.retained_rank > self.residual_width:
            raise ValueError("retained_rank cannot exceed residual_width")


@dataclass(frozen=True, slots=True)
class StaticTransformerSpanExecution:
    """Inspectable result of one terminal-demand span execution."""

    output: Tensor
    demanded_coordinates: Tensor
    demanded_delta: Tensor
    demanded_flat_indices: Tensor
    dense_prefix_length: int


@dataclass(frozen=True, slots=True)
class StaticTransformerSpanAccounting:
    """Logical graph MACs and the dense-prefix reference-kernel estimate.

    Counts exclude additions, bias application, normalization, GELU, masking,
    gathers/scatters, and softmax.  ``logical_total_macs`` assumes projections
    only on valid key rows and attention only on allowed causal key pairs.
    ``reference_dense_prefix_total_macs`` describes the matrix multiplication
    shapes issued by the current dense PyTorch blocks through the longest
    active tensor prefix; it is not a measured latency claim.
    """

    batch_size: int
    sequence_length: int
    dense_prefix_length: int
    valid_key_tokens: int
    demanded_query_tokens: int
    logical_causal_key_pairs: int
    reference_dense_prefix_rows: int
    reference_dense_attention_pairs: int
    input_projection_macs: int
    transformer_qkv_macs: int
    transformer_attention_output_macs: int
    transformer_attention_score_macs: int
    transformer_attention_value_macs: int
    transformer_feed_forward_macs: int
    output_head_macs: int
    decoder_macs: int
    reference_dense_prefix_total_macs: int

    @property
    def transformer_trunk_macs(self) -> int:
        return (
            self.transformer_qkv_macs
            + self.transformer_attention_output_macs
            + self.transformer_attention_score_macs
            + self.transformer_attention_value_macs
            + self.transformer_feed_forward_macs
        )

    @property
    def logical_total_macs(self) -> int:
        return (
            self.input_projection_macs
            + self.transformer_trunk_macs
            + self.output_head_macs
            + self.decoder_macs
        )

    @property
    def dense_reference_to_logical_ratio(self) -> float:
        if self.logical_total_macs == 0:
            return 0.0
        return self.reference_dense_prefix_total_macs / self.logical_total_macs


class StaticTransformerSpanExecutor(nn.Module):
    """A trainable transformer trunk with a fixed rank-limited delta decoder."""

    def __init__(
        self,
        config: StaticTransformerSpanExecutorConfig,
        decoder: Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, StaticTransformerSpanExecutorConfig):
            raise TypeError(
                "config must be a StaticTransformerSpanExecutorConfig"
            )
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("executor dtype must be floating point")
        _dtype_name(dtype)
        if (
            not isinstance(decoder, Tensor)
            or not decoder.is_floating_point()
            or decoder.shape
            != (config.residual_width, config.retained_rank)
            or not torch.isfinite(decoder).all()
        ):
            raise ValueError(
                "decoder must be a finite floating Tensor with shape "
                "[residual_width, retained_rank]"
            )

        self.config = config
        trunk_config = TransformerConfig(
            vocab_size=1,
            max_sequence_length=1,
            d_model=config.hidden_width,
            n_heads=config.head_count,
            n_layers=config.layer_count,
            d_ff=config.feed_forward_width,
            dropout=0.0,
            tie_embeddings=False,
        )
        self.input_projection = nn.Linear(
            config.residual_width,
            config.hidden_width,
            device=device,
            dtype=dtype,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(trunk_config).to(device=device, dtype=dtype)
            for _ in range(config.layer_count)
        )
        self.hidden_norm = nn.LayerNorm(
            config.hidden_width,
            device=device,
            dtype=dtype,
        )
        self.output_head = nn.Linear(
            config.hidden_width,
            config.retained_rank,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "decoder",
            decoder.detach().to(device=device, dtype=dtype).clone(),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.input_projection.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.input_projection.weight.device

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
    def requires_terminal_query_demand(self) -> bool:
        return True

    @property
    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def fixed_runtime_coefficient_count(self) -> int:
        return self.decoder.numel()

    @property
    def total_runtime_coefficient_count(self) -> int:
        return (
            self.learned_parameter_count
            + self.fixed_runtime_coefficient_count
        )

    @property
    def learned_parameter_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters()
        )

    @property
    def fixed_runtime_bytes(self) -> int:
        return self.decoder.numel() * self.decoder.element_size()

    @property
    def expected_learned_parameter_count(self) -> int:
        """Return the closed-form count for the configured architecture."""

        config = self.config
        input_projection = (
            config.residual_width * config.hidden_width
            + config.hidden_width
        )
        per_block = (
            4 * config.hidden_width * config.hidden_width
            + 2 * config.hidden_width * config.feed_forward_width
            + 9 * config.hidden_width
            + config.feed_forward_width
        )
        hidden_norm = 2 * config.hidden_width
        output_head = (
            config.hidden_width * config.retained_rank
            + config.retained_rank
        )
        return (
            input_projection
            + config.layer_count * per_block
            + hidden_norm
            + output_head
        )

    @property
    def sequence_spec(self) -> SequenceSpec:
        return SequenceSpec(
            length_policy="dynamic",
            minimum_length=1,
            maximum_length=None,
            mask=MaskPolicy(
                causal=True,
                padding_side="sparse",
                representation="boolean_valid",
            ),
            position_kind="ordered_logical_tensor_causal",
            supports_prefill=True,
            supports_decode=False,
            cache_kind="none",
        )

    @property
    def capabilities(self) -> SequenceCapabilitySet:
        dtype = _dtype_name(self.dtype)
        return SequenceCapabilitySet(
            length=LengthDomain(1, None),
            executions=CapabilityValues.known("prefill"),
            qk_relations=CapabilityValues.known("equal"),
            position_relations=CapabilityValues.known("equal"),
            mask_origins=CapabilityValues.known("omitted", "provided"),
            mask_patterns=CapabilityValues.known(
                "all_valid",
                "right_padded",
                "left_padded",
                "mixed_padded",
                "sparse",
                "custom",
            ),
            mask_representations=CapabilityValues.known("boolean_valid"),
            visibility_families=CapabilityValues.known("global_causal"),
            position_origins=CapabilityValues.known("omitted", "provided"),
            position_domains=CapabilityValues.known(
                "zero_contiguous",
                "offset_contiguous",
                "arbitrary",
            ),
            cache_kinds=CapabilityValues.known("none"),
            dtypes=CapabilityValues.known(dtype),
            devices=CapabilityValues.known(self.device.type),
            layouts=CapabilityValues.known("contiguous", "strided"),
        )

    def _validate_inputs(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
    ) -> None:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        expected = (
            sequence.batch_size,
            sequence.query_length,
            self.config.residual_width,
        )
        if (
            not isinstance(hidden_states, Tensor)
            or not hidden_states.is_floating_point()
            or hidden_states.shape != expected
        ):
            raise ValueError(
                f"hidden_states must be floating with shape {expected}"
            )
        if hidden_states.device != sequence.device:
            raise ValueError(
                "hidden_states and sequence context must share a device"
            )
        if hidden_states.device != self.device:
            raise ValueError(
                "hidden_states and executor must share a device"
            )
        if sequence.key_length != sequence.query_length:
            raise ValueError(
                "static transformer span requires equal query/key lengths"
            )
        if sequence.phase != "prefill":
            raise ValueError(
                "static transformer span currently supports prefill only"
            )
        if sequence.cache_state is not None or sequence.cache_positions is not None:
            raise ValueError(
                "static transformer span does not accept cache state"
            )
        if not torch.equal(
            sequence.logical_positions,
            sequence.key_logical_positions,
        ):
            raise ValueError(
                "query and key logical-position grids must be equal"
            )
        if (
            sequence.query_valid_mask & ~sequence.key_valid_mask
        ).any():
            raise ValueError(
                "every demanded query row must also be a valid key row"
            )
        if not sequence.key_valid_mask.any():
            raise ValueError("sequence must contain at least one valid key")
        if not torch.isfinite(
            hidden_states[sequence.key_valid_mask]
        ).all():
            raise ValueError("valid key hidden states must be finite")

        for row in range(sequence.batch_size):
            positions = sequence.key_logical_positions[row][
                sequence.key_valid_mask[row]
            ]
            if (positions < 0).any():
                raise ValueError(
                    "valid key logical positions cannot be negative"
                )
            if positions.numel() > 1 and (
                positions[1:] <= positions[:-1]
            ).any():
                raise ValueError(
                    "valid key logical positions must be strictly increasing"
                )

    @staticmethod
    def _dense_prefix_length(key_valid_mask: Tensor) -> int:
        active_columns = key_valid_mask.any(dim=0).nonzero(
            as_tuple=False
        )
        if active_columns.numel() == 0:
            return 0
        return int(active_columns[-1, 0].item()) + 1

    def _execute(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> StaticTransformerSpanExecution:
        self._validate_inputs(hidden_states, sequence)
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a nonempty string")

        prefix_length = self._dense_prefix_length(
            sequence.key_valid_mask
        )
        compute = hidden_states.to(dtype=self.dtype)
        key_mask = sequence.key_valid_mask[:, :prefix_length]
        safe_prefix = torch.where(
            key_mask.unsqueeze(-1),
            compute[:, :prefix_length],
            torch.zeros_like(compute[:, :prefix_length]),
        )
        trunk = self.input_projection(safe_prefix)
        trunk = torch.where(
            key_mask.unsqueeze(-1),
            trunk,
            torch.zeros_like(trunk),
        )
        trunk = record(trace, f"{prefix}.projected", trunk)

        for index, block in enumerate(self.blocks):
            trunk = block(
                trunk,
                attention_mask=key_mask,
                trace=trace,
                prefix=f"{prefix}.trunk.layer.{index}",
            )
            trunk = torch.where(
                key_mask.unsqueeze(-1),
                trunk,
                torch.zeros_like(trunk),
            )
            trunk = record(
                trace,
                f"{prefix}.trunk.layer.{index}.masked_output",
                trunk,
            )

        query_mask = sequence.query_valid_mask[:, :prefix_length]
        demanded_hidden = trunk[query_mask]
        demanded_hidden = record(
            trace,
            f"{prefix}.demanded_hidden",
            demanded_hidden,
        )
        demanded_coordinates = self.output_head(
            self.hidden_norm(demanded_hidden)
        )
        demanded_coordinates = record(
            trace,
            f"{prefix}.modal.delta",
            demanded_coordinates,
        )
        demanded_delta = demanded_coordinates @ self.decoder.T
        demanded_delta = record(
            trace,
            f"{prefix}.decoded_delta",
            demanded_delta,
        )

        demanded_indices = (
            sequence.query_valid_mask.reshape(-1)
            .nonzero(as_tuple=False)
            .flatten()
        )
        flat_input = compute.reshape(-1, self.config.residual_width)
        selected_input = flat_input.index_select(0, demanded_indices)
        selected_output = selected_input + demanded_delta
        flat_output = hidden_states.reshape(
            -1,
            self.config.residual_width,
        ).clone()
        flat_output = flat_output.index_copy(
            0,
            demanded_indices,
            selected_output.to(dtype=hidden_states.dtype),
        )
        return StaticTransformerSpanExecution(
            output=flat_output.reshape_as(hidden_states),
            demanded_coordinates=demanded_coordinates,
            demanded_delta=demanded_delta,
            demanded_flat_indices=demanded_indices,
            dense_prefix_length=prefix_length,
        )

    def forward_components(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        prefix: str = "static_span",
    ) -> StaticTransformerSpanExecution:
        instrumented_input = record(
            trace,
            f"{prefix}.input",
            hidden_states,
        )
        result = self._execute(
            instrumented_input,
            sequence,
            trace=trace,
            prefix=prefix,
        )
        output = record(trace, f"{prefix}.output", result.output)
        return StaticTransformerSpanExecution(
            output=output,
            demanded_coordinates=result.demanded_coordinates,
            demanded_delta=result.demanded_delta,
            demanded_flat_indices=result.demanded_flat_indices,
            dense_prefix_length=result.dense_prefix_length,
        )

    def forward(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        prefix: str = "static_span",
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
        """Implement the backend-neutral compiled-segment protocol."""

        if not isinstance(segment, CompiledSegment):
            raise TypeError("segment must be a CompiledSegment")
        instrumented_input = record(
            trace,
            segment.input_activation,
            hidden_states,
        )
        result = self._execute(
            instrumented_input,
            sequence,
            trace=trace,
            prefix=segment.id,
        )
        output = record(
            trace,
            segment.output_activation,
            result.output,
        )
        return SegmentRun(
            hidden_states=output,
            sequence=sequence,
            raw_output={
                "demanded_coordinates": result.demanded_coordinates,
                "demanded_delta": result.demanded_delta,
                "demanded_flat_indices": result.demanded_flat_indices,
                "dense_prefix_length": result.dense_prefix_length,
                "executor_local_source_free": True,
                "owns_source_fallback": False,
            },
        )

    def logical_accounting(
        self,
        sequence: SequenceContext,
    ) -> StaticTransformerSpanAccounting:
        """Return ideal logical MACs and the dense-prefix reference estimate."""

        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        dummy = torch.zeros(
            sequence.batch_size,
            sequence.query_length,
            self.config.residual_width,
            dtype=self.dtype,
            device=self.device,
        )
        self._validate_inputs(dummy, sequence)
        key_mask = sequence.key_valid_mask
        batch_size, sequence_length = key_mask.shape
        prefix_length = self._dense_prefix_length(key_mask)
        key_tokens = int(key_mask.sum().item())
        query_tokens = int(sequence.query_valid_mask.sum().item())
        causal_pairs = int(
            (
                key_mask.unsqueeze(2)
                & key_mask.unsqueeze(1)
                & torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=key_mask.device,
                )
                .tril()
                .unsqueeze(0)
            ).sum().item()
        )
        config = self.config
        layers = config.layer_count
        hidden = config.hidden_width
        logical_input = key_tokens * config.residual_width * hidden
        qkv = layers * key_tokens * hidden * 3 * hidden
        attention_output = layers * key_tokens * hidden * hidden
        scores = layers * causal_pairs * hidden
        values = layers * causal_pairs * hidden
        feed_forward = (
            layers
            * key_tokens
            * 2
            * hidden
            * config.feed_forward_width
        )
        output_head = query_tokens * hidden * config.retained_rank
        decoder = (
            query_tokens
            * config.retained_rank
            * config.residual_width
        )

        dense_rows = batch_size * prefix_length
        dense_pairs = batch_size * prefix_length * prefix_length
        dense_total = (
            dense_rows * config.residual_width * hidden
            + layers * dense_rows * hidden * 3 * hidden
            + layers * dense_rows * hidden * hidden
            + 2 * layers * dense_pairs * hidden
            + layers
            * dense_rows
            * 2
            * hidden
            * config.feed_forward_width
            + output_head
            + decoder
        )
        return StaticTransformerSpanAccounting(
            batch_size=batch_size,
            sequence_length=sequence_length,
            dense_prefix_length=prefix_length,
            valid_key_tokens=key_tokens,
            demanded_query_tokens=query_tokens,
            logical_causal_key_pairs=causal_pairs,
            reference_dense_prefix_rows=dense_rows,
            reference_dense_attention_pairs=dense_pairs,
            input_projection_macs=logical_input,
            transformer_qkv_macs=qkv,
            transformer_attention_output_macs=attention_output,
            transformer_attention_score_macs=scores,
            transformer_attention_value_macs=values,
            transformer_feed_forward_macs=feed_forward,
            output_head_macs=output_head,
            decoder_macs=decoder,
            reference_dense_prefix_total_macs=dense_total,
        )

    def execution_fingerprint(self) -> str:
        """Hash architecture, execution mode, fixed decoder, and parameters."""

        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_DOMAIN)
        digest.update(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "module_training": tuple(
                        module.training for module in self.modules()
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, value in sorted(self.state_dict().items()):
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def artifact_state_dict(self) -> dict[str, object]:
        """Return a strict weights-only, source-free artifact payload."""

        if any(module.training for module in self.modules()):
            raise ValueError(
                "static transformer span artifact requires eval mode"
            )
        state: dict[str, Tensor] = {}
        for name, value in sorted(self.state_dict().items()):
            if (
                not value.is_floating_point()
                or value.dtype != self.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"executor state {name!r} must be finite {self.dtype}"
                )
            state[name] = value.detach().to(device="cpu").clone()
        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "contains_source_model_weights": False,
            "contains_source_fallback": False,
            "config": asdict(self.config),
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
    ) -> StaticTransformerSpanExecutor:
        """Strictly restore and authenticate a source-free artifact."""

        if not isinstance(state, Mapping) or set(state) != _ARTIFACT_FIELDS:
            raise ValueError(
                "static transformer span artifact fields are invalid"
            )
        if (
            state["artifact_kind"] != _ARTIFACT_KIND
            or type(state["format_version"]) is not int
            or state["format_version"] != _FORMAT_VERSION
            or state["contains_source_model_weights"] is not False
            or state["contains_source_fallback"] is not False
            or not isinstance(state["execution_fingerprint"], str)
            or _SHA256.fullmatch(state["execution_fingerprint"]) is None
        ):
            raise ValueError(
                "unsupported static transformer span artifact"
            )
        raw_config = state["config"]
        if (
            not isinstance(raw_config, Mapping)
            or set(raw_config) != _CONFIG_FIELDS
        ):
            raise ValueError(
                "static transformer span config fields are invalid"
            )
        config = StaticTransformerSpanExecutorConfig(
            residual_width=raw_config["residual_width"],
            hidden_width=raw_config["hidden_width"],
            layer_count=raw_config["layer_count"],
            head_count=raw_config["head_count"],
            feed_forward_width=raw_config["feed_forward_width"],
            retained_rank=raw_config["retained_rank"],
        )
        dtype_name = state["dtype"]
        if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
            raise ValueError(
                "static transformer span artifact dtype is invalid"
            )
        dtype = _DTYPES[dtype_name]
        raw_model_state = state["model_state_dict"]
        if not isinstance(raw_model_state, Mapping):
            raise ValueError(
                "static transformer span model state is invalid"
            )
        decoder = raw_model_state.get("decoder")
        if not isinstance(decoder, Tensor):
            raise TypeError("static transformer span decoder must be a Tensor")
        executor = cls(
            config,
            decoder,
            dtype=dtype,
            device=map_location,
        )
        expected = executor.state_dict()
        if set(raw_model_state) != set(expected):
            raise ValueError(
                "static transformer span model state fields are invalid"
            )
        restored: dict[str, Tensor] = {}
        for name, expected_value in expected.items():
            value = raw_model_state[name]
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"static transformer span state {name!r} must be a Tensor"
                )
            if value.device.type != "cpu":
                raise ValueError(
                    f"static transformer span state {name!r} must be on CPU"
                )
            if value.dtype != dtype:
                raise ValueError(
                    f"static transformer span state {name!r} has wrong dtype"
                )
            if value.shape != expected_value.shape:
                raise ValueError(
                    f"static transformer span state {name!r} has wrong shape"
                )
            if not torch.isfinite(value).all():
                raise ValueError(
                    f"static transformer span state {name!r} must be finite"
                )
            restored[name] = value.detach().to(
                device=map_location
            ).clone()
        executor.load_state_dict(restored, strict=True)
        executor.eval()
        if executor.execution_fingerprint() != state["execution_fingerprint"]:
            raise ValueError(
                "static transformer span execution fingerprint mismatch"
            )
        return executor


__all__ = [
    "StaticTransformerSpanAccounting",
    "StaticTransformerSpanExecution",
    "StaticTransformerSpanExecutor",
    "StaticTransformerSpanExecutorConfig",
]
