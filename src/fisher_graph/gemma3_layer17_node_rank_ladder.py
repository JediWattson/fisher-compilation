"""Pure resource planner for the frozen Gemma layer-17 modal topology.

This module deliberately performs no model, tokenizer, prompt, or dataset
access.  It answers one narrow question before a live rate/fidelity ladder is
run: for the already selected four-fragment layer-17 topology, what storage
and matrix-MAC budget follows from a mode-rank cap, a generator-private rank,
and an edge policy?

The existing fragments do not all contain 48 or 64 native modes.  A requested
rank is therefore a *cap* and resolves independently at each node as
``min(cap, native_mode_count)``.  That preserves the frozen fragments and the
441,600-parameter source slice instead of silently changing topology while
testing capacity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal


__all__ = [
    "DEFAULT_LAYER17_NODE_RANK_ARM_SPECS",
    "LAYER17_FRAGMENT_IDS",
    "LAYER17_NATIVE_MODE_COUNTS",
    "LAYER17_SOURCE_MACS_PER_TOKEN",
    "LAYER17_SOURCE_PARAMETERS",
    "LAYER17_TOPOLOGY_SHA256",
    "Layer17EdgePolicy",
    "Layer17NodeRankArmSpec",
    "Layer17NodeRankLadderPlan",
    "Layer17NodeRankResourceRow",
    "build_default_layer17_node_rank_ladder_plan",
    "build_layer17_node_rank_resource_row",
    "resolve_layer17_node_ranks",
    "validate_layer17_node_rank_ladder_plan",
]


Layer17EdgePolicy = Literal["edgeless", "dynamic_q8_top1"]

_SCHEMA = "fisher_graph.gemma3_layer17_node_rank_ladder_plan"
_FORMAT_VERSION = 1
_SPEC_KIND = "fisher_graph.gemma3_layer17_node_rank_arm_spec"
_ROW_KIND = "fisher_graph.gemma3_layer17_node_rank_resource_row"
_SPEC_DOMAIN = b"fisher-graph:gemma3-layer17-node-rank-arm:v1\0"
_ROW_DOMAIN = b"fisher-graph:gemma3-layer17-node-rank-resource:v1\0"
_PLAN_DOMAIN = b"fisher-graph:gemma3-layer17-node-rank-plan:v1\0"
_TOPOLOGY_DOMAIN = b"fisher-graph:gemma3-layer17-frozen-topology:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LABEL = re.compile(r"^[a-z][a-z0-9-]*$")

# Execution order is significant.  The first fragment supplies the source
# state for the three optional conditional edges.
LAYER17_FRAGMENT_IDS = (
    "cluster.0/layer.17",
    "cluster.28/layer.17",
    "cluster.34/layer.17",
    "cluster.54/layer.17",
)
LAYER17_NATIVE_MODE_COUNTS = (54, 38, 85, 53)
_LAYER17_SOURCE_NODE_INDEX = 0
_LAYER17_INPUT_WIDTH = 640
_LAYER17_OUTPUT_WIDTH = 640
_LAYER17_NATIVE_MATRICES_PER_MODE = 3
_LAYER17_QUADRATIC_RANK = 8
_LAYER17_TOP_K = 1

# Every removed Gemma gated-MLP mode owns one input row in each of gate/up and
# one output column in down: three width-640 matrices.  Matrix MACs and stored
# coefficients are equal under this accounting policy.
LAYER17_SOURCE_PARAMETERS = (
    sum(LAYER17_NATIVE_MODE_COUNTS)
    * _LAYER17_NATIVE_MATRICES_PER_MODE
    * _LAYER17_INPUT_WIDTH
)
LAYER17_SOURCE_MACS_PER_TOKEN = LAYER17_SOURCE_PARAMETERS

_SAFETY_METADATA = {
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_activation_tensors": False,
    "contains_gradient_tensors": False,
    "contains_model_or_candidate_weights": False,
    "model_or_tokenizer_accessed": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "scientific_role": "pure_open_development_resource_plan",
}


def _canonical_json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _strict_fields(
    state: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


def _topology_payload() -> dict[str, object]:
    return {
        "layer_ordinal": 17,
        "fragment_ids_in_execution_order": LAYER17_FRAGMENT_IDS,
        "native_mode_counts_in_execution_order": LAYER17_NATIVE_MODE_COUNTS,
        "source_node_index": _LAYER17_SOURCE_NODE_INDEX,
        "input_width": _LAYER17_INPUT_WIDTH,
        "output_width": _LAYER17_OUTPUT_WIDTH,
        "native_matrices_per_mode": _LAYER17_NATIVE_MATRICES_PER_MODE,
        "quadratic_rank": _LAYER17_QUADRATIC_RANK,
        "top_k": _LAYER17_TOP_K,
        "source_parameter_count": LAYER17_SOURCE_PARAMETERS,
        "source_macs_per_token": LAYER17_SOURCE_MACS_PER_TOKEN,
    }


LAYER17_TOPOLOGY_SHA256 = _canonical_json_sha256(
    _topology_payload(),
    domain=_TOPOLOGY_DOMAIN,
)


def resolve_layer17_node_ranks(mode_rank_cap: int) -> tuple[int, ...]:
    """Resolve one requested cap without changing frozen fragment identity."""

    cap = _positive_int(mode_rank_cap, label="mode_rank_cap")
    result = tuple(min(cap, width) for width in LAYER17_NATIVE_MODE_COUNTS)
    if len(result) != len(LAYER17_FRAGMENT_IDS) or any(
        rank <= 0 or rank > native
        for rank, native in zip(result, LAYER17_NATIVE_MODE_COUNTS, strict=True)
    ):
        raise RuntimeError("resolved layer-17 node ranks violate topology")
    return result


@dataclass(frozen=True, slots=True)
class Layer17NodeRankArmSpec:
    """One source-safe declaration of the capacity variable being tested."""

    label: str
    mode_rank_cap: int
    generator_rank: int
    edge_policy: Layer17EdgePolicy
    artifact_sha256: str = ""
    artifact_kind: str = _SPEC_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or _LABEL.fullmatch(self.label) is None:
            raise ValueError("arm label must be canonical lowercase kebab-case")
        _positive_int(self.mode_rank_cap, label="mode_rank_cap")
        generator_rank = _positive_int(
            self.generator_rank,
            label="generator_rank",
        )
        node_ranks = resolve_layer17_node_ranks(self.mode_rank_cap)
        if generator_rank > min(node_ranks):
            raise ValueError(
                "generator_rank cannot exceed the smallest resolved node rank"
            )
        if self.edge_policy not in ("edgeless", "dynamic_q8_top1"):
            raise ValueError("unsupported layer-17 edge policy")
        if (
            self.artifact_kind != _SPEC_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("layer-17 rank arm header is invalid")
        expected = _canonical_json_sha256(
            self._payload(),
            domain=_SPEC_DOMAIN,
        )
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="arm artifact_sha256",
            ) != expected:
                raise ValueError("layer-17 rank arm hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", expected)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "topology_sha256": LAYER17_TOPOLOGY_SHA256,
            "label": self.label,
            "mode_rank_cap": self.mode_rank_cap,
            "generator_rank": self.generator_rank,
            "edge_policy": self.edge_policy,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> "Layer17NodeRankArmSpec":
        expected = {
            "artifact_kind",
            "format_version",
            "topology_sha256",
            "label",
            "mode_rank_cap",
            "generator_rank",
            "edge_policy",
            "artifact_sha256",
        }
        _strict_fields(state, expected, label="layer-17 rank arm")
        if state["topology_sha256"] != LAYER17_TOPOLOGY_SHA256:
            raise ValueError("layer-17 rank arm topology drifted")
        return cls(
            label=state["label"],  # type: ignore[arg-type]
            mode_rank_cap=state["mode_rank_cap"],  # type: ignore[arg-type]
            generator_rank=state["generator_rank"],  # type: ignore[arg-type]
            edge_policy=state["edge_policy"],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


def _node_resources(node_rank: int, generator_rank: int) -> tuple[int, int]:
    # Graph lowering keeps the private reduced-regression factor separate from
    # the public K-dimensional modal state:
    #   [d,R] + [R,K] + [K,d] + latent bias [K] + output bias [d].
    matrix_count = (
        _LAYER17_INPUT_WIDTH * generator_rank
        + generator_rank * node_rank
        + node_rank * _LAYER17_OUTPUT_WIDTH
    )
    parameter_count = matrix_count + node_rank + _LAYER17_OUTPUT_WIDTH
    return parameter_count, matrix_count


def _dynamic_edge_resources(
    source_rank: int,
    target_rank: int,
) -> tuple[int, int, int]:
    # Affine message, message bias, linear gate and gate bias, followed by the
    # rank-q factorized quadratic term.  The dense MAC count includes routing;
    # selected-message MACs intentionally excludes it so the shared routing
    # total can be added exactly once per outgoing edge.
    q = _LAYER17_QUADRATIC_RANK
    parameters = (
        source_rank * target_rank
        + target_rank
        + source_rank
        + 1
        + 2 * source_rank * q
        + q * target_rank
    )
    selected_message_macs = (
        source_rank * target_rank
        + 2 * source_rank * q
        + q * target_rank
    )
    dense_macs = source_rank + selected_message_macs
    return parameters, dense_macs, selected_message_macs


def _expected_resource_values(
    spec: Layer17NodeRankArmSpec,
) -> dict[str, object]:
    node_ranks = resolve_layer17_node_ranks(spec.mode_rank_cap)
    node_resources = tuple(
        _node_resources(rank, spec.generator_rank) for rank in node_ranks
    )
    node_parameters = sum(value[0] for value in node_resources)
    node_macs = sum(value[1] for value in node_resources)

    if spec.edge_policy == "edgeless":
        interaction_count = 0
        interaction_parameters = 0
        routing_macs = 0
        dense_message_macs = 0
        interaction_dense_macs = 0
        selected_message_upper = 0
    else:
        source_rank = node_ranks[_LAYER17_SOURCE_NODE_INDEX]
        targets = tuple(
            rank
            for index, rank in enumerate(node_ranks)
            if index != _LAYER17_SOURCE_NODE_INDEX
        )
        edges = tuple(
            _dynamic_edge_resources(source_rank, target_rank)
            for target_rank in targets
        )
        interaction_count = len(edges)
        interaction_parameters = sum(value[0] for value in edges)
        interaction_dense_macs = sum(value[1] for value in edges)
        routing_macs = interaction_count * source_rank
        dense_message_macs = sum(value[2] for value in edges)
        selected_message_upper = sum(
            sorted((value[2] for value in edges), reverse=True)[:_LAYER17_TOP_K]
        )

    graph_parameters = node_parameters + interaction_parameters
    graph_dense_macs = node_macs + interaction_dense_macs
    executed_upper = node_macs + routing_macs + selected_message_upper
    parameter_savings = LAYER17_SOURCE_PARAMETERS - graph_parameters
    dense_macs_saved = LAYER17_SOURCE_MACS_PER_TOKEN - graph_dense_macs
    executed_macs_saved = LAYER17_SOURCE_MACS_PER_TOKEN - executed_upper
    return {
        "node_ranks": node_ranks,
        "node_count": len(node_ranks),
        "interaction_count": interaction_count,
        "node_parameter_count": node_parameters,
        "interaction_parameter_count": interaction_parameters,
        "graph_parameter_count": graph_parameters,
        "node_macs_per_token": node_macs,
        "interaction_dense_macs_per_token": interaction_dense_macs,
        "graph_dense_macs_per_token": graph_dense_macs,
        "conditional_routing_macs_per_token": routing_macs,
        "conditional_dense_message_macs_per_token": dense_message_macs,
        "conditional_selected_message_macs_per_token_upper_bound": (
            selected_message_upper
        ),
        "executed_graph_macs_per_token_upper_bound": executed_upper,
        "source_parameter_count": LAYER17_SOURCE_PARAMETERS,
        "source_macs_per_token": LAYER17_SOURCE_MACS_PER_TOKEN,
        "net_parameter_savings": parameter_savings,
        "net_dense_macs_saved_per_token": dense_macs_saved,
        "net_executed_macs_saved_per_token": executed_macs_saved,
        "parameter_savings_fraction": (
            parameter_savings / LAYER17_SOURCE_PARAMETERS
        ),
        "dense_mac_savings_fraction": (
            dense_macs_saved / LAYER17_SOURCE_MACS_PER_TOKEN
        ),
        "executed_mac_savings_fraction": (
            executed_macs_saved / LAYER17_SOURCE_MACS_PER_TOKEN
        ),
    }


@dataclass(frozen=True, slots=True)
class Layer17NodeRankResourceRow:
    """Authenticated analytic resource row for one declared arm."""

    spec: Layer17NodeRankArmSpec
    node_ranks: tuple[int, ...]
    node_count: int
    interaction_count: int
    node_parameter_count: int
    interaction_parameter_count: int
    graph_parameter_count: int
    node_macs_per_token: int
    interaction_dense_macs_per_token: int
    graph_dense_macs_per_token: int
    conditional_routing_macs_per_token: int
    conditional_dense_message_macs_per_token: int
    conditional_selected_message_macs_per_token_upper_bound: int
    executed_graph_macs_per_token_upper_bound: int
    source_parameter_count: int
    source_macs_per_token: int
    net_parameter_savings: int
    net_dense_macs_saved_per_token: int
    net_executed_macs_saved_per_token: int
    parameter_savings_fraction: float
    dense_mac_savings_fraction: float
    executed_mac_savings_fraction: float
    artifact_sha256: str = ""
    artifact_kind: str = _ROW_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.spec, Layer17NodeRankArmSpec):
            raise TypeError("resource row spec must be Layer17NodeRankArmSpec")
        if type(self.node_ranks) is not tuple or any(
            type(value) is not int or value <= 0 for value in self.node_ranks
        ):
            raise ValueError("resource row node_ranks are invalid")
        integer_fields = (
            "node_count",
            "interaction_count",
            "node_parameter_count",
            "interaction_parameter_count",
            "graph_parameter_count",
            "node_macs_per_token",
            "interaction_dense_macs_per_token",
            "graph_dense_macs_per_token",
            "conditional_routing_macs_per_token",
            "conditional_dense_message_macs_per_token",
            "conditional_selected_message_macs_per_token_upper_bound",
            "executed_graph_macs_per_token_upper_bound",
            "source_parameter_count",
            "source_macs_per_token",
        )
        for name in integer_fields:
            _nonnegative_int(getattr(self, name), label=name)
        signed_integer_fields = (
            "net_parameter_savings",
            "net_dense_macs_saved_per_token",
            "net_executed_macs_saved_per_token",
        )
        for name in signed_integer_fields:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in (
            "parameter_savings_fraction",
            "dense_mac_savings_fraction",
            "executed_mac_savings_fraction",
        ):
            _finite_float(getattr(self, name), label=name)
        expected = _expected_resource_values(self.spec)
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"resource row {name} violates the analytic invariant"
                )
        if (
            self.artifact_kind != _ROW_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("layer-17 resource row header is invalid")
        expected_sha256 = _canonical_json_sha256(
            self._payload(),
            domain=_ROW_DOMAIN,
        )
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="resource row artifact_sha256",
            ) != expected_sha256:
                raise ValueError("layer-17 resource row hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", expected_sha256)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "topology_sha256": LAYER17_TOPOLOGY_SHA256,
            "spec": self.spec.state_dict(),
            "node_ranks": self.node_ranks,
            "node_count": self.node_count,
            "interaction_count": self.interaction_count,
            "node_parameter_count": self.node_parameter_count,
            "interaction_parameter_count": self.interaction_parameter_count,
            "graph_parameter_count": self.graph_parameter_count,
            "node_macs_per_token": self.node_macs_per_token,
            "interaction_dense_macs_per_token": (
                self.interaction_dense_macs_per_token
            ),
            "graph_dense_macs_per_token": self.graph_dense_macs_per_token,
            "conditional_routing_macs_per_token": (
                self.conditional_routing_macs_per_token
            ),
            "conditional_dense_message_macs_per_token": (
                self.conditional_dense_message_macs_per_token
            ),
            "conditional_selected_message_macs_per_token_upper_bound": (
                self.conditional_selected_message_macs_per_token_upper_bound
            ),
            "executed_graph_macs_per_token_upper_bound": (
                self.executed_graph_macs_per_token_upper_bound
            ),
            "source_parameter_count": self.source_parameter_count,
            "source_macs_per_token": self.source_macs_per_token,
            "net_parameter_savings": self.net_parameter_savings,
            "net_dense_macs_saved_per_token": (
                self.net_dense_macs_saved_per_token
            ),
            "net_executed_macs_saved_per_token": (
                self.net_executed_macs_saved_per_token
            ),
            "parameter_savings_fraction": self.parameter_savings_fraction,
            "dense_mac_savings_fraction": self.dense_mac_savings_fraction,
            "executed_mac_savings_fraction": self.executed_mac_savings_fraction,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> "Layer17NodeRankResourceRow":
        expected = {
            "artifact_kind",
            "format_version",
            "topology_sha256",
            "spec",
            "node_ranks",
            "node_count",
            "interaction_count",
            "node_parameter_count",
            "interaction_parameter_count",
            "graph_parameter_count",
            "node_macs_per_token",
            "interaction_dense_macs_per_token",
            "graph_dense_macs_per_token",
            "conditional_routing_macs_per_token",
            "conditional_dense_message_macs_per_token",
            "conditional_selected_message_macs_per_token_upper_bound",
            "executed_graph_macs_per_token_upper_bound",
            "source_parameter_count",
            "source_macs_per_token",
            "net_parameter_savings",
            "net_dense_macs_saved_per_token",
            "net_executed_macs_saved_per_token",
            "parameter_savings_fraction",
            "dense_mac_savings_fraction",
            "executed_mac_savings_fraction",
            "artifact_sha256",
        }
        _strict_fields(state, expected, label="layer-17 resource row")
        if state["topology_sha256"] != LAYER17_TOPOLOGY_SHA256:
            raise ValueError("layer-17 resource row topology drifted")
        spec_state = state["spec"]
        if not isinstance(spec_state, Mapping):
            raise TypeError("resource row spec state must be a mapping")
        ranks = state["node_ranks"]
        if isinstance(ranks, (str, bytes)) or not isinstance(ranks, Sequence):
            raise TypeError("resource row node_ranks must be a sequence")
        return cls(
            spec=Layer17NodeRankArmSpec.from_state_dict(spec_state),
            node_ranks=tuple(ranks),  # type: ignore[arg-type]
            **{
                name: state[name]
                for name in expected
                if name
                not in {
                    "artifact_kind",
                    "format_version",
                    "topology_sha256",
                    "spec",
                    "node_ranks",
                    "artifact_sha256",
                }
            },  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


def build_layer17_node_rank_resource_row(
    *,
    mode_rank_cap: int,
    generator_rank: int,
    edge_policy: Layer17EdgePolicy,
    label: str,
) -> Layer17NodeRankResourceRow:
    """Build and analytically validate one pure layer-17 resource row."""

    spec = Layer17NodeRankArmSpec(
        label=label,
        mode_rank_cap=mode_rank_cap,
        generator_rank=generator_rank,
        edge_policy=edge_policy,
    )
    return Layer17NodeRankResourceRow(spec=spec, **_expected_resource_values(spec))


DEFAULT_LAYER17_NODE_RANK_ARM_SPECS = (
    Layer17NodeRankArmSpec(
        label="baseline-dynamic",
        mode_rank_cap=32,
        generator_rank=16,
        edge_policy="dynamic_q8_top1",
    ),
    Layer17NodeRankArmSpec(
        label="baseline-edgeless",
        mode_rank_cap=32,
        generator_rank=16,
        edge_policy="edgeless",
    ),
    Layer17NodeRankArmSpec(
        label="latent-lift-edgeless",
        mode_rank_cap=32,
        generator_rank=32,
        edge_policy="edgeless",
    ),
    Layer17NodeRankArmSpec(
        label="cap48-dynamic-diagnostic",
        mode_rank_cap=48,
        generator_rank=16,
        edge_policy="dynamic_q8_top1",
    ),
    Layer17NodeRankArmSpec(
        label="cap64-dynamic-diagnostic",
        mode_rank_cap=64,
        generator_rank=16,
        edge_policy="dynamic_q8_top1",
    ),
)


@dataclass(frozen=True, slots=True)
class Layer17NodeRankLadderPlan:
    """Tensor-free, prompt-free collection of analytic ladder arms."""

    rows: tuple[Layer17NodeRankResourceRow, ...]
    artifact_sha256: str = ""
    schema: str = _SCHEMA
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or not self.rows:
            raise ValueError("layer-17 rank ladder rows must be a nonempty tuple")
        if any(
            not isinstance(row, Layer17NodeRankResourceRow) for row in self.rows
        ):
            raise TypeError("layer-17 rank ladder contains an invalid row")
        labels = tuple(row.spec.label for row in self.rows)
        if len(labels) != len(set(labels)):
            raise ValueError("layer-17 rank ladder labels must be unique")
        if self.schema != _SCHEMA or self.format_version != _FORMAT_VERSION:
            raise ValueError("layer-17 rank ladder header is invalid")
        # Reconstruct every row rather than trusting serialized totals.
        for row in self.rows:
            rebuilt = build_layer17_node_rank_resource_row(
                label=row.spec.label,
                mode_rank_cap=row.spec.mode_rank_cap,
                generator_rank=row.spec.generator_rank,
                edge_policy=row.spec.edge_policy,
            )
            if rebuilt.state_dict() != row.state_dict():
                raise ValueError("layer-17 rank ladder row failed reconstruction")
        expected = _canonical_json_sha256(self._payload(), domain=_PLAN_DOMAIN)
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="rank ladder artifact_sha256",
            ) != expected:
                raise ValueError("layer-17 rank ladder hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", expected)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "topology": _topology_payload(),
            "topology_sha256": LAYER17_TOPOLOGY_SHA256,
            "rows": tuple(row.state_dict() for row in self.rows),
            **_SAFETY_METADATA,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> "Layer17NodeRankLadderPlan":
        expected = {
            "schema",
            "format_version",
            "topology",
            "topology_sha256",
            "rows",
            "artifact_sha256",
            *_SAFETY_METADATA,
        }
        _strict_fields(state, expected, label="layer-17 rank ladder plan")
        topology = state["topology"]
        if not isinstance(topology, Mapping) or _canonical_json_sha256(
            topology,
            domain=_TOPOLOGY_DOMAIN,
        ) != LAYER17_TOPOLOGY_SHA256:
            raise ValueError("layer-17 rank ladder topology payload drifted")
        if state["topology_sha256"] != LAYER17_TOPOLOGY_SHA256:
            raise ValueError("layer-17 rank ladder topology hash drifted")
        for name, value in _SAFETY_METADATA.items():
            if state[name] != value:
                raise ValueError(f"layer-17 rank ladder safety field {name} drifted")
        rows_state = state["rows"]
        if isinstance(rows_state, (str, bytes)) or not isinstance(
            rows_state,
            Sequence,
        ):
            raise TypeError("layer-17 rank ladder rows must be a sequence")
        rows = []
        for value in rows_state:
            if not isinstance(value, Mapping):
                raise TypeError("serialized layer-17 resource row is invalid")
            rows.append(Layer17NodeRankResourceRow.from_state_dict(value))
        return cls(
            rows=tuple(rows),
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            schema=state["schema"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


def build_default_layer17_node_rank_ladder_plan() -> Layer17NodeRankLadderPlan:
    """Build the canonical baseline, latent-lift, and cap diagnostics."""

    return Layer17NodeRankLadderPlan(
        rows=tuple(
            build_layer17_node_rank_resource_row(
                label=spec.label,
                mode_rank_cap=spec.mode_rank_cap,
                generator_rank=spec.generator_rank,
                edge_policy=spec.edge_policy,
            )
            for spec in DEFAULT_LAYER17_NODE_RANK_ARM_SPECS
        )
    )


def validate_layer17_node_rank_ladder_plan(
    value: Layer17NodeRankLadderPlan | Mapping[str, object],
) -> Layer17NodeRankLadderPlan:
    """Strict-load and reconstruct a source-safe ladder plan."""

    if isinstance(value, Layer17NodeRankLadderPlan):
        return Layer17NodeRankLadderPlan.from_state_dict(value.state_dict())
    if not isinstance(value, Mapping):
        raise TypeError("layer-17 node-rank ladder must be a plan or mapping")
    return Layer17NodeRankLadderPlan.from_state_dict(value)
