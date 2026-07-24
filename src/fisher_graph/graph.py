"""A small DAG executor implementing the same layer boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor, nn

from .activations import ActivationTrace, record
from .layers import CausalSelfAttention, FeedForward, LayerExecutor

if TYPE_CHECKING:
    from .layers import TransformerBlock


class GraphOperation(nn.Module):
    """Operation used by a :class:`GraphLayerExecutor` node."""

    def forward(
        self,
        *inputs: Tensor,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        raise NotImplementedError


class UnaryModuleOperation(GraphOperation):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        value: Tensor,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        del trace, prefix
        return self.module(value)


class AttentionOperation(GraphOperation):
    def __init__(self, attention: CausalSelfAttention) -> None:
        super().__init__()
        self.attention = attention

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            trace=trace,
            prefix=f"{prefix}.attention",
        )


class FeedForwardOperation(GraphOperation):
    def __init__(self, mlp: FeedForward) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(
        self,
        hidden_states: Tensor,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        return self.mlp(hidden_states, trace=trace, prefix=f"{prefix}.mlp")


class AddOperation(GraphOperation):
    def forward(
        self,
        left: Tensor,
        right: Tensor,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        del trace, prefix
        return left + right


@dataclass(frozen=True, slots=True)
class GraphNode:
    """Declarative dataflow node.

    ``inputs`` refer to earlier node names or the built-in ``hidden_states`` and
    ``attention_mask`` inputs. ``capture_as`` controls the activation tap name;
    set it to ``None`` when the operation already records its own output.
    """

    name: str
    inputs: tuple[str, ...]
    operation: GraphOperation
    capture_as: str | None = None


class GraphLayerExecutor(LayerExecutor):
    """Execute a transformer layer as an inspectable, ordered DAG."""

    def __init__(self, nodes: list[GraphNode], *, output: str) -> None:
        super().__init__()
        if not nodes:
            raise ValueError("a graph needs at least one node")

        available = {"hidden_states", "attention_mask"}
        operations: dict[str, GraphOperation] = {}
        specs: list[tuple[str, tuple[str, ...], str | None]] = []
        for node in nodes:
            if node.name in available:
                raise ValueError(f"duplicate graph value: {node.name}")
            missing = set(node.inputs) - available
            if missing:
                raise ValueError(
                    f"node {node.name!r} references unavailable inputs: {sorted(missing)}"
                )
            available.add(node.name)
            operations[node.name] = node.operation
            specs.append((node.name, node.inputs, node.capture_as))
        if output not in available:
            raise ValueError(f"unknown graph output: {output}")

        self.operations = nn.ModuleDict(operations)
        self.specs = tuple(specs)
        self.output = output

    @classmethod
    def from_transformer_block(
        cls, block: TransformerBlock
    ) -> GraphLayerExecutor:
        """Transfer a standard block's modules into an equivalent graph."""

        nodes = [
            GraphNode(
                "ln1",
                ("hidden_states",),
                UnaryModuleOperation(block.norm1),
                "ln1",
            ),
            GraphNode(
                "attention",
                ("ln1", "attention_mask"),
                AttentionOperation(block.attention),
            ),
            GraphNode(
                "post_attention",
                ("hidden_states", "attention"),
                AddOperation(),
                "post_attention",
            ),
            GraphNode(
                "ln2",
                ("post_attention",),
                UnaryModuleOperation(block.norm2),
                "ln2",
            ),
            GraphNode(
                "mlp",
                ("ln2",),
                FeedForwardOperation(block.mlp),
            ),
            GraphNode(
                "output",
                ("post_attention", "mlp"),
                AddOperation(),
                "output",
            ),
        ]
        return cls(nodes, output="output")

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        hidden_states = record(trace, f"{prefix}.input", hidden_states)
        # Graph operations receive a tensor mask. The model always normalizes
        # an omitted mask before invoking a layer.
        if attention_mask is None:
            raise ValueError("GraphLayerExecutor requires a normalized attention_mask")
        state = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
        }
        for name, inputs, capture_as in self.specs:
            values = [state[input_name] for input_name in inputs]
            value = self.operations[name](
                *values, trace=trace, prefix=prefix
            )
            if capture_as is not None:
                value = record(trace, f"{prefix}.{capture_as}", value)
            state[name] = value
        return state[self.output]

