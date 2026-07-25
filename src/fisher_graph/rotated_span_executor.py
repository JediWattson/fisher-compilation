"""A causal block executor whose residual delta stays in a locked span.

The codimension-one experiment identifies a unit normal ``q`` whose removal
is behaviorally inexpensive.  This module turns that representation result
into an executable boundary:

```
x_t = (h_t - input_mean) / input_scale
z_t = causal_graph(x_<=t)
h'_t = h_t + z_t @ B.T
```

``B`` is a deterministic orthonormal basis for ``q``'s Euclidean complement.
Consequently every predicted block delta is orthogonal to ``q`` by
construction.  The graph sees the full input residual width: the predecessor
experiment constrained the output span only and did not justify discarding an
input feature.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .activations import ActivationTrace, record
from .adapters import SegmentRun, SequenceContext
from .codimension_projection import canonical_unit_direction
from .compiler.manifest import CompiledSegment
from .gated_executor import ResidualGatedCausalModalExecutor


_ARTIFACT_KIND = "fisher_graph.rotated_span_block_executor"
_FORMAT_VERSION = 1
_FINGERPRINT_DOMAIN = b"fisher_graph.rotated_span_executor.v1\0"


def deterministic_orthogonal_complement(normal: Tensor) -> Tensor:
    """Return a deterministic FP64 basis for a unit normal's complement.

    A Householder reflection maps one coordinate vector to the canonicalized
    normal.  Removing that coordinate's reflected column leaves an
    orthonormal basis for the codimension-one hyperplane.  Choosing the
    normal's largest-magnitude coordinate keeps the construction away from
    its numerically singular case.
    """

    direction = canonical_unit_direction(
        normal,
        label="rotated-span normal",
    )
    width = int(direction.numel())
    if width < 2:
        raise ValueError("rotated span requires width of at least two")
    pivot = int(direction.abs().argmax().item())
    coordinate = torch.zeros_like(direction)
    coordinate[pivot] = 1.0
    difference = coordinate - direction
    squared_norm = difference.square().sum()
    if float(squared_norm.item()) <= 64 * torch.finfo(torch.float64).eps:
        householder = torch.eye(width, dtype=torch.float64)
    else:
        householder = (
            torch.eye(width, dtype=torch.float64)
            - 2.0
            * torch.outer(difference, difference)
            / squared_norm
        )
    columns = [index for index in range(width) if index != pivot]
    basis = householder[:, columns].contiguous()
    identity = torch.eye(width - 1, dtype=torch.float64)
    projector = (
        torch.eye(width, dtype=torch.float64)
        - torch.outer(direction, direction)
    )
    if (
        not torch.allclose(
            basis.T @ basis,
            identity,
            rtol=0.0,
            atol=1e-12,
        )
        or not torch.allclose(
            basis @ basis.T,
            projector,
            rtol=0.0,
            atol=1e-12,
        )
        or float((basis.T @ direction).abs().max().item()) > 1e-12
    ):
        raise RuntimeError("failed to construct the rotated-span basis")
    return basis


@dataclass(frozen=True, slots=True)
class RotatedSpanExecution:
    """Inspectable decomposition of one grouped block execution."""

    output: Tensor
    normalized_input: Tensor
    modal_delta: Tensor
    raw_delta: Tensor


class RotatedSpanBlockExecutor(nn.Module):
    """Decode a causal graph's output inside one locked hyperplane."""

    def __init__(
        self,
        *,
        normal: Tensor,
        input_mean: Tensor,
        input_scale: Tensor,
        graph: ResidualGatedCausalModalExecutor,
    ) -> None:
        super().__init__()
        if not isinstance(graph, ResidualGatedCausalModalExecutor):
            raise TypeError(
                "graph must be a ResidualGatedCausalModalExecutor"
            )
        direction = canonical_unit_direction(
            normal,
            label="rotated-span normal",
        )
        width = int(direction.numel())
        if (
            graph.input_modes != width
            or graph.output_modes != width - 1
            or graph.config.same_position_skip
        ):
            raise ValueError(
                "graph must map full-width inputs to width-minus-one "
                "outputs with same_position_skip disabled"
            )
        for label, value in (
            ("input_mean", input_mean),
            ("input_scale", input_scale),
        ):
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.shape != (width,)
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"{label} must be a finite floating width-vector"
                )
        if (input_scale <= 0).any():
            raise ValueError("input_scale must be strictly positive")
        if graph.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "rotated-span reference graph must use float32 or float64"
            )

        dtype = graph.dtype
        device = graph.device
        basis = deterministic_orthogonal_complement(direction)
        self.graph = graph
        self.register_buffer(
            "normal",
            direction.to(device=device, dtype=torch.float64),
        )
        self.register_buffer(
            "span_basis",
            basis.to(device=device, dtype=dtype),
        )
        self.register_buffer(
            "input_mean",
            input_mean.detach().to(device=device, dtype=dtype).clone(),
        )
        self.register_buffer(
            "input_scale",
            input_scale.detach().to(device=device, dtype=dtype).clone(),
        )

    @property
    def width(self) -> int:
        return int(self.normal.numel())

    @property
    def retained_rank(self) -> int:
        return self.width - 1

    @property
    def learned_parameter_count(self) -> int:
        return self.graph.learned_parameter_count

    @property
    def fixed_runtime_coefficient_count(self) -> int:
        return (
            self.normal.numel()
            + self.span_basis.numel()
            + self.input_mean.numel()
            + self.input_scale.numel()
        )

    def _validate_inputs(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
    ) -> None:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if sequence.phase != "prefill" or sequence.cache_state is not None:
            raise ValueError(
                "rotated-span executor currently supports prefill only"
            )
        expected = (
            sequence.batch_size,
            sequence.query_length,
            self.width,
        )
        if (
            not isinstance(hidden_states, Tensor)
            or hidden_states.shape != expected
            or not hidden_states.is_floating_point()
        ):
            raise ValueError(
                "hidden_states must be floating with shape "
                f"{expected}"
            )
        if hidden_states.device != sequence.device:
            raise ValueError(
                "hidden_states and sequence context must share a device"
            )

    def forward_components(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
    ) -> RotatedSpanExecution:
        """Execute the grouped block and expose its modal/raw delta."""

        self._validate_inputs(hidden_states, sequence)
        compute = hidden_states.to(dtype=self.graph.dtype)
        normalized = (compute - self.input_mean) / self.input_scale
        modal_delta = self.graph(
            normalized,
            query_valid_mask=sequence.query_valid_mask,
            key_valid_mask=sequence.key_valid_mask,
            logical_positions=sequence.logical_positions,
            key_logical_positions=sequence.key_logical_positions,
        )
        raw_delta = modal_delta @ self.span_basis.T
        predicted = compute + raw_delta
        output = torch.where(
            sequence.query_valid_mask.unsqueeze(-1),
            predicted.to(dtype=hidden_states.dtype),
            hidden_states,
        )
        return RotatedSpanExecution(
            output=output,
            normalized_input=normalized,
            modal_delta=modal_delta,
            raw_delta=raw_delta,
        )

    def forward(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
    ) -> Tensor:
        return self.forward_components(hidden_states, sequence).output

    def run(
        self,
        segment: CompiledSegment,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        """Implement the backend-neutral grouped compiled-segment protocol."""

        if not isinstance(segment, CompiledSegment):
            raise TypeError("segment must be a CompiledSegment")
        instrumented_input = record(
            trace,
            segment.input_activation,
            hidden_states,
        )
        components = self.forward_components(
            instrumented_input,
            sequence,
        )
        modal_site = f"{segment.id}.modal_delta"
        modal_delta = record(trace, modal_site, components.modal_delta)
        if modal_delta is not components.modal_delta:
            raw_delta = modal_delta @ self.span_basis.T
            output = torch.where(
                sequence.query_valid_mask.unsqueeze(-1),
                (
                    instrumented_input.to(dtype=self.graph.dtype)
                    + raw_delta
                ).to(dtype=instrumented_input.dtype),
                instrumented_input,
            )
        else:
            output = components.output
            raw_delta = components.raw_delta
        output = record(trace, segment.output_activation, output)
        return SegmentRun(
            hidden_states=output,
            sequence=sequence,
            raw_output={
                "modal_delta": modal_delta,
                "raw_delta": raw_delta,
            },
        )

    def execution_fingerprint(self) -> str:
        """Hash tensor state and graph semantics used by runtime dispatch."""

        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_DOMAIN)
        digest.update(
            json.dumps(
                asdict(self.graph.config),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, value in sorted(self.state_dict().items()):
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(
                json.dumps(tuple(tensor.shape)).encode("ascii")
            )
            digest.update(tensor.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def artifact_state_dict(self) -> dict[str, object]:
        """Return a strict, source-weight-free executor artifact payload."""

        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "normal": self.normal.detach().to(
                device="cpu",
                dtype=torch.float64,
            ),
            "input_mean": self.input_mean.detach().cpu().clone(),
            "input_scale": self.input_scale.detach().cpu().clone(),
            "graph": self.graph.artifact_state_dict(),
            "execution_fingerprint": self.execution_fingerprint(),
        }

    @classmethod
    def from_artifact_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        map_location: torch.device | str = "cpu",
    ) -> RotatedSpanBlockExecutor:
        expected = {
            "artifact_kind",
            "format_version",
            "normal",
            "input_mean",
            "input_scale",
            "graph",
            "execution_fingerprint",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("rotated-span executor fields are invalid")
        if (
            state["artifact_kind"] != _ARTIFACT_KIND
            or state["format_version"] != _FORMAT_VERSION
            or not isinstance(state["execution_fingerprint"], str)
        ):
            raise ValueError("unsupported rotated-span executor artifact")
        graph_state = state["graph"]
        if not isinstance(graph_state, Mapping):
            raise ValueError("rotated-span graph state is invalid")
        graph = ResidualGatedCausalModalExecutor.from_artifact_state_dict(
            graph_state,
            device=map_location,
        )
        normal = state["normal"]
        input_mean = state["input_mean"]
        input_scale = state["input_scale"]
        if not all(
            isinstance(value, Tensor)
            for value in (normal, input_mean, input_scale)
        ):
            raise ValueError(
                "rotated-span fixed state must contain tensors"
            )
        result = cls(
            normal=normal,
            input_mean=input_mean,
            input_scale=input_scale,
            graph=graph,
        )
        if result.execution_fingerprint() != state["execution_fingerprint"]:
            raise ValueError(
                "rotated-span executor execution fingerprint mismatch"
            )
        result.eval()
        return result


__all__ = [
    "RotatedSpanBlockExecutor",
    "RotatedSpanExecution",
    "deterministic_orthogonal_complement",
]
