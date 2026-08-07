"""Source-free all-mode generator graph execution for one Gemma MLP.

Each natural SwiGLU channel is one scalar graph node and decodes through its
original down-projection column.  Exact and global-affine nodes share one
frontier.  The chart-conditioned graph adds a hidden-state chart frontier and
fused chart-to-mode edges.  No plan deletes or merges a mode, and no plan
retains a source module or tensor-storage alias.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import re

import torch
from torch import Tensor, nn


__all__ = [
    "ALL_MODE_GRAPH_KINDS",
    "AllModeGeneratorGraphExecution",
    "Gemma3AllModeGeneratorGraphMLP",
    "compile_exact_all_mode_generator_graph",
    "compile_fitted_affine_all_mode_generator_graph",
    "compile_chart_conditioned_all_mode_generator_graph",
]


ALL_MODE_GRAPH_KINDS = (
    "exact_swiglu",
    "fitted_affine",
    "chart_conditioned_affine",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH_DOMAIN = b"fisher_graph:gemma3-all-mode-generator-graph:v2\0"


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _storage_pointers(module: nn.Module) -> set[int]:
    return {
        int(value.untyped_storage().data_ptr())
        for value in (*module.parameters(), *module.buffers())
        if value.numel()
    }


def _graph_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(_HASH_DOMAIN + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class AllModeGeneratorGraphExecution:
    output: Tensor
    modal_states: Tensor | None
    node_count: int
    interaction_count: int
    topological_frontier_count: int
    chart_memberships: Tensor | None = None
    chart_coordinates: Tensor | None = None


class Gemma3AllModeGeneratorGraphMLP(nn.Module):
    """Fused traversal of one all-root scalar-generator graph."""

    def __init__(
        self,
        *,
        graph_kind: str,
        decoder_weight: Tensor,
        model_fingerprint: str,
        layer_ordinal: int,
        activation: nn.Module | None = None,
        gate_weight: Tensor | None = None,
        up_weight: Tensor | None = None,
        affine_weight: Tensor | None = None,
        affine_bias: Tensor | None = None,
        chart_centers: Tensor | None = None,
        chart_bases: Tensor | None = None,
        chart_distance_scales: Tensor | None = None,
        chart_edge_weight: Tensor | None = None,
        chart_temperature: float | None = None,
        fit_split_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if graph_kind not in ALL_MODE_GRAPH_KINDS:
            raise ValueError("invalid all-mode graph kind")
        if not isinstance(model_fingerprint, str) or _SHA256.fullmatch(
            model_fingerprint
        ) is None:
            raise ValueError("model_fingerprint must be a SHA-256 digest")
        if type(layer_ordinal) is not int or layer_ordinal < 0:
            raise ValueError("layer_ordinal must be nonnegative")
        if decoder_weight.ndim != 2:
            raise ValueError("decoder_weight must be a matrix")
        hidden, modes = decoder_weight.shape
        self.graph_kind = graph_kind
        self.layer_ordinal = layer_ordinal
        self.model_fingerprint = model_fingerprint
        self.mode_count = modes
        self.hidden_width = hidden
        self.interaction_count = 0
        self.topological_frontier_count = 1
        self.chart_count = 0
        self.chart_rank = 0
        self.chart_temperature = chart_temperature
        self.fit_split_sha256 = fit_split_sha256
        self.mode_decoder = nn.Linear(modes, hidden, bias=False)
        with torch.no_grad():
            self.mode_decoder.weight.copy_(decoder_weight.detach())

        if graph_kind == "exact_swiglu":
            if (
                activation is None
                or gate_weight is None
                or up_weight is None
                or gate_weight.shape != (modes, hidden)
                or up_weight.shape != gate_weight.shape
                or affine_weight is not None
                or affine_bias is not None
                or chart_centers is not None
                or chart_bases is not None
                or chart_distance_scales is not None
                or chart_edge_weight is not None
                or chart_temperature is not None
                or fit_split_sha256 is not None
            ):
                raise ValueError("exact all-mode graph tensors are invalid")
            self.gate_generators: nn.Module | None = nn.Linear(
                hidden, modes, bias=False
            )
            self.up_generators: nn.Module | None = nn.Linear(
                hidden, modes, bias=False
            )
            self.affine_generators: nn.Module | None = None
            self.chart_edge_weight: nn.Parameter | None = None
            with torch.no_grad():
                self.gate_generators.weight.copy_(gate_weight.detach())
                self.up_generators.weight.copy_(up_weight.detach())
            self.activation = copy.deepcopy(activation)
        elif graph_kind == "fitted_affine":
            if (
                affine_weight is None
                or affine_bias is None
                or affine_weight.shape != (modes, hidden)
                or affine_bias.shape != (modes,)
                or gate_weight is not None
                or up_weight is not None
                or activation is not None
                or chart_centers is not None
                or chart_bases is not None
                or chart_distance_scales is not None
                or chart_edge_weight is not None
                or chart_temperature is not None
                or not isinstance(fit_split_sha256, str)
                or _SHA256.fullmatch(fit_split_sha256) is None
            ):
                raise ValueError("affine all-mode graph tensors are invalid")
            self.gate_generators = None
            self.up_generators = None
            self.affine_generators = nn.Linear(hidden, modes, bias=True)
            self.chart_edge_weight = None
            with torch.no_grad():
                self.affine_generators.weight.copy_(affine_weight.detach())
                self.affine_generators.bias.copy_(affine_bias.detach())
            self.activation = None
        else:
            if (
                affine_weight is None
                or affine_bias is None
                or affine_weight.shape != (modes, hidden)
                or affine_bias.shape != (modes,)
                or gate_weight is not None
                or up_weight is not None
                or activation is not None
                or chart_centers is None
                or chart_bases is None
                or chart_distance_scales is None
                or chart_edge_weight is None
                or chart_centers.ndim != 2
                or chart_centers.shape[1] != hidden
                or chart_bases.ndim != 3
                or chart_bases.shape[:2] != chart_centers.shape
                or chart_distance_scales.shape != (chart_centers.shape[0],)
                or chart_edge_weight.shape
                != (
                    chart_centers.shape[0],
                    chart_bases.shape[2] + 1,
                    modes,
                )
                or not isinstance(chart_temperature, (int, float))
                or isinstance(chart_temperature, bool)
                or not math.isfinite(float(chart_temperature))
                or float(chart_temperature) <= 0.0
                or not isinstance(fit_split_sha256, str)
                or _SHA256.fullmatch(fit_split_sha256) is None
            ):
                raise ValueError("chart-conditioned graph tensors are invalid")
            if not bool(torch.isfinite(chart_distance_scales).all()) or bool(
                (chart_distance_scales <= 0.0).any()
            ):
                raise ValueError("chart distance scales must be finite and positive")
            self.gate_generators = None
            self.up_generators = None
            self.affine_generators = nn.Linear(hidden, modes, bias=True)
            with torch.no_grad():
                self.affine_generators.weight.copy_(affine_weight.detach())
                self.affine_generators.bias.copy_(affine_bias.detach())
            self.register_buffer(
                "chart_centers",
                chart_centers.detach().clone().contiguous(),
            )
            self.register_buffer(
                "chart_bases",
                chart_bases.detach().clone().contiguous(),
            )
            self.register_buffer(
                "chart_distance_scales",
                chart_distance_scales.detach().clone().contiguous(),
            )
            self.chart_edge_weight = nn.Parameter(
                chart_edge_weight.detach().clone().contiguous(),
                requires_grad=False,
            )
            self.chart_count = int(chart_centers.shape[0])
            self.chart_rank = int(chart_bases.shape[2])
            self.chart_temperature = float(chart_temperature)
            self.interaction_count = self.chart_count * modes
            self.topological_frontier_count = 2
            self.activation = None

        self.requires_grad_(False)
        self.eval()
        self._artifact_sha256 = _graph_sha256(self._hash_payload())

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def gate_proj(self) -> nn.Module:
        module = self.gate_generators or self.affine_generators
        if module is None:
            raise RuntimeError("all-mode graph has no input generator")
        return module

    @property
    def up_proj(self) -> nn.Module:
        module = self.up_generators or self.affine_generators
        if module is None:
            raise RuntimeError("all-mode graph has no input generator")
        return module

    @property
    def down_proj(self) -> nn.Module:
        return self.mode_decoder

    @property
    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def stored_coefficient_count(self) -> int:
        return sum(
            value.numel()
            for value in (*self.parameters(), *self.buffers())
        )

    @property
    def matrix_macs_per_token(self) -> int:
        if self.graph_kind == "exact_swiglu":
            return 3 * self.hidden_width * self.mode_count
        if self.graph_kind == "fitted_affine":
            return 2 * self.hidden_width * self.mode_count
        return (
            2 * self.hidden_width * self.mode_count
            + self.chart_count * self.hidden_width
            + self.chart_count * self.hidden_width * self.chart_rank
            + self.chart_count * (self.chart_rank + 1) * self.mode_count
        )

    @property
    def routing_blend_macs_per_token(self) -> int:
        if self.graph_kind != "chart_conditioned_affine":
            return 0
        return self.chart_count * self.mode_count

    @property
    def estimated_total_macs_per_token(self) -> int:
        return self.matrix_macs_per_token + self.routing_blend_macs_per_token

    def _hash_payload(self) -> dict[str, object]:
        tensors = {
            name: _tensor_sha256(value)
            for name, value in self.state_dict().items()
        }
        return {
            "schema": "fisher_graph.gemma3_all_mode_generator_graph.v2",
            "graph_kind": self.graph_kind,
            "model_fingerprint": self.model_fingerprint,
            "layer_ordinal": self.layer_ordinal,
            "hidden_width": self.hidden_width,
            "node_count": self.mode_count,
            "total_graph_node_count": self.mode_count + self.chart_count,
            "interaction_count": self.interaction_count,
            "topological_frontier_count": self.topological_frontier_count,
            "chart_count": self.chart_count,
            "chart_rank": self.chart_rank,
            "chart_temperature": self.chart_temperature,
            "fit_split_sha256": self.fit_split_sha256,
            "tensor_sha256s": tensors,
        }

    def graph_metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "source_free_runtime": True,
            "all_modes_retained": True,
            "removed_mode_count": 0,
            "merged_mode_count": 0,
            "routing_cutoff": None,
            "traversal": (
                "hidden_chart_frontier_then_fused_chart_to_mode_edges_"
                "then_decoder"
                if self.graph_kind == "chart_conditioned_affine"
                else "one_fused_mode_frontier_then_decoder"
            ),
            "learned_parameter_count": self.learned_parameter_count,
            "stored_coefficient_count": self.stored_coefficient_count,
            "matrix_macs_per_token": self.matrix_macs_per_token,
            "routing_blend_macs_per_token": (
                self.routing_blend_macs_per_token
            ),
            "estimated_total_macs_per_token": (
                self.estimated_total_macs_per_token
            ),
        }

    def execute_graph(
        self,
        normalized_input: Tensor,
        *,
        capture_modal_states: bool = False,
        capture_chart_states: bool = False,
    ) -> AllModeGeneratorGraphExecution:
        if normalized_input.shape[-1] != self.hidden_width:
            raise ValueError("all-mode graph input width drifted")
        if self.graph_kind == "exact_swiglu":
            if self.gate_generators is None or self.up_generators is None:
                raise RuntimeError("exact all-mode graph is incomplete")
            states = self.activation(self.gate_generators(normalized_input)) * (
                self.up_generators(normalized_input)
            )
            memberships = None
            coordinates = None
        elif self.graph_kind == "fitted_affine":
            if self.affine_generators is None:
                raise RuntimeError("affine all-mode graph is incomplete")
            states = self.affine_generators(normalized_input)
            memberships = None
            coordinates = None
        else:
            if (
                self.affine_generators is None
                or self.chart_edge_weight is None
                or not hasattr(self, "chart_centers")
                or not hasattr(self, "chart_bases")
                or not hasattr(self, "chart_distance_scales")
            ):
                raise RuntimeError("chart-conditioned all-mode graph is incomplete")
            displacement = normalized_input.unsqueeze(-2) - self.chart_centers
            distance = displacement.square().sum(dim=-1)
            scale = self.chart_distance_scales.square()
            logits = -0.5 * distance / (
                scale * float(self.chart_temperature)
            )
            memberships = logits.softmax(dim=-1)
            coordinates = torch.einsum(
                "...ch,chr->...cr",
                displacement,
                self.chart_bases,
            )
            ones = torch.ones_like(coordinates[..., :1])
            augmented = torch.cat((ones, coordinates), dim=-1)
            messages = torch.einsum(
                "...cr,crk->...ck",
                augmented,
                self.chart_edge_weight,
            )
            states = self.affine_generators(normalized_input) + (
                memberships.unsqueeze(-1) * messages
            ).sum(dim=-2)
        output = self.mode_decoder(states)
        return AllModeGeneratorGraphExecution(
            output=output,
            modal_states=states.detach().clone() if capture_modal_states else None,
            node_count=self.mode_count,
            interaction_count=self.interaction_count,
            topological_frontier_count=self.topological_frontier_count,
            chart_memberships=(
                memberships.detach().clone()
                if capture_chart_states and memberships is not None
                else None
            ),
            chart_coordinates=(
                coordinates.detach().clone()
                if capture_chart_states and coordinates is not None
                else None
            ),
        )

    def forward(self, normalized_input: Tensor) -> Tensor:
        return self.execute_graph(normalized_input).output


def compile_exact_all_mode_generator_graph(
    source_mlp: nn.Module,
    *,
    model_fingerprint: str,
    layer_ordinal: int,
) -> Gemma3AllModeGeneratorGraphMLP:
    result = Gemma3AllModeGeneratorGraphMLP(
        graph_kind="exact_swiglu",
        gate_weight=source_mlp.gate_proj.weight.detach().clone().contiguous(),
        up_weight=source_mlp.up_proj.weight.detach().clone().contiguous(),
        decoder_weight=source_mlp.down_proj.weight.detach().clone().contiguous(),
        activation=source_mlp.act_fn,
        model_fingerprint=model_fingerprint,
        layer_ordinal=layer_ordinal,
    )
    if _storage_pointers(source_mlp) & _storage_pointers(result):
        raise RuntimeError("exact all-mode graph aliases source storage")
    return result


def compile_fitted_affine_all_mode_generator_graph(
    *,
    affine_weight: Tensor,
    affine_bias: Tensor,
    decoder_weight: Tensor,
    model_fingerprint: str,
    layer_ordinal: int,
    fit_split_sha256: str,
) -> Gemma3AllModeGeneratorGraphMLP:
    return Gemma3AllModeGeneratorGraphMLP(
        graph_kind="fitted_affine",
        affine_weight=affine_weight.detach().clone().contiguous(),
        affine_bias=affine_bias.detach().clone().contiguous(),
        decoder_weight=decoder_weight.detach().clone().contiguous(),
        model_fingerprint=model_fingerprint,
        layer_ordinal=layer_ordinal,
        fit_split_sha256=fit_split_sha256,
    )


def compile_chart_conditioned_all_mode_generator_graph(
    *,
    affine_weight: Tensor,
    affine_bias: Tensor,
    decoder_weight: Tensor,
    chart_centers: Tensor,
    chart_bases: Tensor,
    chart_distance_scales: Tensor,
    chart_edge_weight: Tensor,
    chart_temperature: float,
    model_fingerprint: str,
    layer_ordinal: int,
    fit_split_sha256: str,
) -> Gemma3AllModeGeneratorGraphMLP:
    return Gemma3AllModeGeneratorGraphMLP(
        graph_kind="chart_conditioned_affine",
        affine_weight=affine_weight.detach().clone().contiguous(),
        affine_bias=affine_bias.detach().clone().contiguous(),
        decoder_weight=decoder_weight.detach().clone().contiguous(),
        chart_centers=chart_centers.detach().clone().contiguous(),
        chart_bases=chart_bases.detach().clone().contiguous(),
        chart_distance_scales=(
            chart_distance_scales.detach().clone().contiguous()
        ),
        chart_edge_weight=chart_edge_weight.detach().clone().contiguous(),
        chart_temperature=chart_temperature,
        model_fingerprint=model_fingerprint,
        layer_ordinal=layer_ordinal,
        fit_split_sha256=fit_split_sha256,
    )
