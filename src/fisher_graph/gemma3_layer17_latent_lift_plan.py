"""Pure planning for a rejected Gemma layer-17 latent-lift arm.

The latent-lift diagnostic keeps the frozen layer-10 parent and the four
selected layer-17 parameter fragments, removes every interaction edge, and
lifts only the private generator rank of the layer-17 nodes from 16 to 32
while retaining their width-32 computational-mode state.

This module does not fit weights, open a corpus, or execute a model.  It binds
an authenticated edgeless source graph to the canonical layer-17 rank-ladder
resource row and can later assemble the executable edgeless graph once fitted
rank-32 layer-17 nodes are supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math

from .gemma3_layer17_node_rank_ladder import (
    LAYER17_FRAGMENT_IDS,
    LAYER17_SOURCE_MACS_PER_TOKEN,
    LAYER17_SOURCE_PARAMETERS,
    Layer17NodeRankLadderPlan,
    Layer17NodeRankResourceRow,
    build_default_layer17_node_rank_ladder_plan,
    validate_layer17_node_rank_ladder_plan,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import ModalGeneratorLowering
from .parameter_cluster_fragments import ParameterClusterLayerFragment


__all__ = [
    "GEMMA3_LAYER17_LATENT_LIFT_EDGE_POLICY",
    "GEMMA3_LAYER17_LATENT_LIFT_MODE_RANK",
    "GEMMA3_LAYER17_LATENT_LIFT_SOURCE_GENERATOR_RANK",
    "GEMMA3_LAYER17_LATENT_LIFT_TARGET_GENERATOR_RANK",
    "Gemma3Layer17LatentLiftMetricTriple",
    "Gemma3Layer17LatentLiftParetoRejection",
    "Gemma3Layer17LatentLiftPlan",
    "Gemma3Layer17LatentLiftResources",
    "build_gemma3_layer17_latent_lift_edgeless_graph",
    "build_gemma3_layer17_latent_lift_pareto_rejection",
    "build_gemma3_layer17_latent_lift_plan",
    "build_gemma3_layer17_latent_lift_plan_from_lowerings",
]


GEMMA3_LAYER17_LATENT_LIFT_MODE_RANK = 32
GEMMA3_LAYER17_LATENT_LIFT_SOURCE_GENERATOR_RANK = 16
GEMMA3_LAYER17_LATENT_LIFT_TARGET_GENERATOR_RANK = 32
GEMMA3_LAYER17_LATENT_LIFT_EDGE_POLICY = "edgeless"

_EXPECTED_LAYERS = (10, 17)
_EXPECTED_NODE_COUNT_PER_LAYER = 4
_PLAN_DOMAIN = b"fisher_graph.gemma3_layer17_latent_lift.plan.v1\0"
_DECISION_DOMAIN = b"fisher_graph.gemma3_layer17_latent_lift.rejection.v1\0"
_SCIENTIFIC_STATUS = "rejected_open_development_diagnostic_arm"
_DECISION_SAFETY = {
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_data_rows": False,
    "contains_activations": False,
    "contains_gradients": False,
    "aggregate_metrics_only": True,
}


def _json_sha256(value: object, *, domain: bytes = _PLAN_DOMAIN) -> str:
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


@dataclass(frozen=True, slots=True)
class Gemma3Layer17LatentLiftResources:
    """Exact static parameter and logical matrix-MAC accounting."""

    layer10_native_removed_parameters: int
    layer10_parent_parameters: int
    layer10_parent_macs_per_token: int
    layer17_native_removed_parameters: int
    layer17_source_parameters: int
    layer17_target_parameters: int
    layer17_source_macs_per_token: int
    layer17_target_macs_per_token: int
    combined_native_removed_parameters: int
    combined_source_parameters: int
    combined_target_parameters: int
    combined_source_macs_per_token: int
    combined_target_macs_per_token: int
    rank_lift_added_parameters: int
    rank_lift_added_macs_per_token: int
    layer17_target_net_parameter_savings: int
    layer17_target_net_macs_saved_per_token: int
    combined_target_net_parameter_savings: int
    combined_target_net_macs_saved_per_token: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(
                "latent-lift diagnostic resources must be nonnegative integers"
            )
        if self.layer17_native_removed_parameters != LAYER17_SOURCE_PARAMETERS:
            raise ValueError("layer-17 native parameter total drifted")
        if self.layer17_native_removed_parameters != LAYER17_SOURCE_MACS_PER_TOKEN:
            raise ValueError("layer-17 native parameter/MAC identity drifted")
        expected = {
            "combined_native_removed_parameters": (
                self.layer10_native_removed_parameters
                + self.layer17_native_removed_parameters
            ),
            "combined_source_parameters": (
                self.layer10_parent_parameters + self.layer17_source_parameters
            ),
            "combined_target_parameters": (
                self.layer10_parent_parameters + self.layer17_target_parameters
            ),
            "combined_source_macs_per_token": (
                self.layer10_parent_macs_per_token
                + self.layer17_source_macs_per_token
            ),
            "combined_target_macs_per_token": (
                self.layer10_parent_macs_per_token
                + self.layer17_target_macs_per_token
            ),
            "rank_lift_added_parameters": (
                self.layer17_target_parameters - self.layer17_source_parameters
            ),
            "rank_lift_added_macs_per_token": (
                self.layer17_target_macs_per_token
                - self.layer17_source_macs_per_token
            ),
            "layer17_target_net_parameter_savings": (
                self.layer17_native_removed_parameters
                - self.layer17_target_parameters
            ),
            "layer17_target_net_macs_saved_per_token": (
                self.layer17_native_removed_parameters
                - self.layer17_target_macs_per_token
            ),
            "combined_target_net_parameter_savings": (
                self.combined_native_removed_parameters
                - self.combined_target_parameters
            ),
            "combined_target_net_macs_saved_per_token": (
                self.combined_native_removed_parameters
                - self.combined_target_macs_per_token
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"latent-lift diagnostic resource field {name} drifted"
                )
        if (
            self.layer17_target_parameters <= self.layer17_source_parameters
            or self.layer17_target_macs_per_token
            <= self.layer17_source_macs_per_token
            or self.layer17_target_net_parameter_savings <= 0
            or self.layer17_target_net_macs_saved_per_token <= 0
            or self.combined_target_net_parameter_savings <= 0
            or self.combined_target_net_macs_saved_per_token <= 0
        ):
            raise ValueError(
                "the latent-lift diagnostic must spend capacity while "
                "retaining positive savings"
            )

    @property
    def layer17_target_parameter_fraction(self) -> float:
        return self.layer17_target_parameters / self.layer17_native_removed_parameters

    @property
    def layer17_target_macs_fraction(self) -> float:
        return (
            self.layer17_target_macs_per_token
            / self.layer17_native_removed_parameters
        )

    @property
    def combined_target_parameter_fraction(self) -> float:
        return self.combined_target_parameters / self.combined_native_removed_parameters

    @property
    def combined_target_macs_fraction(self) -> float:
        return (
            self.combined_target_macs_per_token
            / self.combined_native_removed_parameters
        )

    def state_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        } | {
            "layer17_target_parameter_fraction": (
                self.layer17_target_parameter_fraction
            ),
            "layer17_target_macs_fraction": self.layer17_target_macs_fraction,
            "combined_target_parameter_fraction": (
                self.combined_target_parameter_fraction
            ),
            "combined_target_macs_fraction": self.combined_target_macs_fraction,
        }


@dataclass(frozen=True, slots=True)
class Gemma3Layer17LatentLiftMetricTriple:
    """Aggregate fidelity metrics with no examples or model outputs."""

    delta_nll_per_token: float
    native_to_candidate_kl_per_token: float
    top1_agreement_to_native: float

    def __post_init__(self) -> None:
        values = (
            self.delta_nll_per_token,
            self.native_to_candidate_kl_per_token,
            self.top1_agreement_to_native,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("latent-lift decision metrics must be finite")
        object.__setattr__(
            self,
            "delta_nll_per_token",
            float(self.delta_nll_per_token),
        )
        object.__setattr__(
            self,
            "native_to_candidate_kl_per_token",
            float(self.native_to_candidate_kl_per_token),
        )
        object.__setattr__(
            self,
            "top1_agreement_to_native",
            float(self.top1_agreement_to_native),
        )
        if self.native_to_candidate_kl_per_token < 0.0:
            raise ValueError("native-to-candidate KL must be nonnegative")
        if not 0.0 <= self.top1_agreement_to_native <= 1.0:
            raise ValueError("top-1 agreement must be between zero and one")

    def state_dict(self) -> dict[str, float]:
        return {
            "delta_nll_per_token": self.delta_nll_per_token,
            "native_to_candidate_kl_per_token": (
                self.native_to_candidate_kl_per_token
            ),
            "top1_agreement_to_native": self.top1_agreement_to_native,
        }


@dataclass(frozen=True, slots=True)
class Gemma3Layer17LatentLiftParetoRejection:
    """Source-safe record that rank 32 is worse on all three metrics."""

    latent_lift_plan_sha256: str
    evidence_example_count: int
    rank16: Gemma3Layer17LatentLiftMetricTriple
    rank32: Gemma3Layer17LatentLiftMetricTriple
    delta_nll_rank32_minus_rank16: float
    delta_kl_rank32_minus_rank16: float
    delta_top1_rank32_minus_rank16: float
    pareto_rejected: bool = True
    assessment_role: str = "open_development_assessment"
    scientific_status: str = _SCIENTIFIC_STATUS
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.latent_lift_plan_sha256, str)
            or len(self.latent_lift_plan_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.latent_lift_plan_sha256
            )
        ):
            raise ValueError("latent-lift plan hash must be a SHA-256 digest")
        if type(self.evidence_example_count) is not int or (
            self.evidence_example_count <= 0
        ):
            raise ValueError("evidence_example_count must be positive")
        if not isinstance(
            self.rank16,
            Gemma3Layer17LatentLiftMetricTriple,
        ) or not isinstance(
            self.rank32,
            Gemma3Layer17LatentLiftMetricTriple,
        ):
            raise TypeError("rank metrics must be latent-lift metric triples")
        expected_deltas = (
            self.rank32.delta_nll_per_token - self.rank16.delta_nll_per_token,
            self.rank32.native_to_candidate_kl_per_token
            - self.rank16.native_to_candidate_kl_per_token,
            self.rank32.top1_agreement_to_native
            - self.rank16.top1_agreement_to_native,
        )
        supplied_deltas = (
            self.delta_nll_rank32_minus_rank16,
            self.delta_kl_rank32_minus_rank16,
            self.delta_top1_rank32_minus_rank16,
        )
        if supplied_deltas != expected_deltas:
            raise ValueError("latent-lift decision metric deltas drifted")
        if not (
            expected_deltas[0] > 0.0
            and expected_deltas[1] > 0.0
            and expected_deltas[2] < 0.0
            and self.pareto_rejected is True
        ):
            raise ValueError("rank 32 is not Pareto-worse on all three metrics")
        if (
            self.assessment_role != "open_development_assessment"
            or self.scientific_status != _SCIENTIFIC_STATUS
        ):
            raise ValueError("latent-lift rejection scientific status drifted")
        computed = _json_sha256(self._payload(), domain=_DECISION_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("latent-lift rejection hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "latent_lift_plan_sha256": self.latent_lift_plan_sha256,
            "evidence_example_count": self.evidence_example_count,
            "rank16": self.rank16.state_dict(),
            "rank32": self.rank32.state_dict(),
            "delta_nll_rank32_minus_rank16": (
                self.delta_nll_rank32_minus_rank16
            ),
            "delta_kl_rank32_minus_rank16": (
                self.delta_kl_rank32_minus_rank16
            ),
            "delta_top1_rank32_minus_rank16": (
                self.delta_top1_rank32_minus_rank16
            ),
            "pareto_rejected": self.pareto_rejected,
            "assessment_role": self.assessment_role,
            "scientific_status": self.scientific_status,
            "safety": dict(_DECISION_SAFETY),
        }

    def validate_integrity(self) -> None:
        if (
            _json_sha256(self._payload(), domain=_DECISION_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("latent-lift rejection hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True)
class Gemma3Layer17LatentLiftPlan:
    """Source-safe recipe retained only as a rejected diagnostic arm."""

    source_edgeless_graph_sha256: str
    model_fingerprint: str
    parameter_cluster_plan_sha256: str
    layer10_node_names: tuple[str, ...]
    layer10_node_sha256s: tuple[str, ...]
    layer10_fragment_ids: tuple[str, ...]
    layer10_fragment_sha256s: tuple[str, ...]
    layer17_node_names: tuple[str, ...]
    layer17_source_node_sha256s: tuple[str, ...]
    layer17_fragment_ids: tuple[str, ...]
    layer17_fragment_sha256s: tuple[str, ...]
    layer17_rank_ladder_sha256: str
    layer17_rank_resource_sha256: str
    resources: Gemma3Layer17LatentLiftResources
    mode_rank: int = GEMMA3_LAYER17_LATENT_LIFT_MODE_RANK
    source_generator_rank: int = (
        GEMMA3_LAYER17_LATENT_LIFT_SOURCE_GENERATOR_RANK
    )
    target_generator_rank: int = (
        GEMMA3_LAYER17_LATENT_LIFT_TARGET_GENERATOR_RANK
    )
    edge_policy: str = GEMMA3_LAYER17_LATENT_LIFT_EDGE_POLICY
    interaction_count: int = 0
    scientific_status: str = _SCIENTIFIC_STATUS
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        collections = (
            self.layer10_node_names,
            self.layer10_node_sha256s,
            self.layer10_fragment_ids,
            self.layer10_fragment_sha256s,
            self.layer17_node_names,
            self.layer17_source_node_sha256s,
            self.layer17_fragment_ids,
            self.layer17_fragment_sha256s,
        )
        if any(type(value) is not tuple for value in collections):
            raise TypeError(
                "latent-lift diagnostic identity collections must be tuples"
            )
        if any(
            len(value) != _EXPECTED_NODE_COUNT_PER_LAYER
            for value in collections
        ):
            raise ValueError(
                "the latent-lift diagnostic requires four nodes per layer"
            )
        if any(len(value) != len(set(value)) for value in collections):
            raise ValueError(
                "latent-lift diagnostic identities must be unique per layer"
            )
        if set(self.layer10_node_names) & set(self.layer17_node_names):
            raise ValueError("latent-lift diagnostic layer node names overlap")
        if set(self.layer10_fragment_sha256s) & set(
            self.layer17_fragment_sha256s
        ):
            raise ValueError("latent-lift diagnostic layer fragments overlap")
        if self.layer17_fragment_ids != LAYER17_FRAGMENT_IDS:
            raise ValueError(
                "the latent-lift diagnostic does not preserve the ordered "
                "layer-17 fragments"
            )
        if (
            self.mode_rank != GEMMA3_LAYER17_LATENT_LIFT_MODE_RANK
            or self.source_generator_rank
            != GEMMA3_LAYER17_LATENT_LIFT_SOURCE_GENERATOR_RANK
            or self.target_generator_rank
            != GEMMA3_LAYER17_LATENT_LIFT_TARGET_GENERATOR_RANK
            or self.edge_policy != GEMMA3_LAYER17_LATENT_LIFT_EDGE_POLICY
            or self.interaction_count != 0
            or self.scientific_status != _SCIENTIFIC_STATUS
        ):
            raise ValueError("unsupported latent-lift diagnostic architecture")
        if not isinstance(self.resources, Gemma3Layer17LatentLiftResources):
            raise TypeError(
                "resources must be latent-lift diagnostic resource accounting"
            )
        computed = _json_sha256(self._payload())
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("latent-lift diagnostic plan hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "source_edgeless_graph_sha256": self.source_edgeless_graph_sha256,
            "model_fingerprint": self.model_fingerprint,
            "parameter_cluster_plan_sha256": self.parameter_cluster_plan_sha256,
            "layer10_node_names": self.layer10_node_names,
            "layer10_node_sha256s": self.layer10_node_sha256s,
            "layer10_fragment_ids": self.layer10_fragment_ids,
            "layer10_fragment_sha256s": self.layer10_fragment_sha256s,
            "layer17_node_names": self.layer17_node_names,
            "layer17_source_node_sha256s": self.layer17_source_node_sha256s,
            "layer17_fragment_ids": self.layer17_fragment_ids,
            "layer17_fragment_sha256s": self.layer17_fragment_sha256s,
            "layer17_rank_ladder_sha256": self.layer17_rank_ladder_sha256,
            "layer17_rank_resource_sha256": self.layer17_rank_resource_sha256,
            "mode_rank": self.mode_rank,
            "source_generator_rank": self.source_generator_rank,
            "target_generator_rank": self.target_generator_rank,
            "edge_policy": self.edge_policy,
            "interaction_count": self.interaction_count,
            "scientific_status": self.scientific_status,
            "resources": self.resources.state_dict(),
        }

    def validate_integrity(self) -> None:
        if _json_sha256(self._payload()) != self.artifact_sha256:
            raise ValueError("latent-lift diagnostic plan hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def build_gemma3_layer17_latent_lift_pareto_rejection(
    plan: Gemma3Layer17LatentLiftPlan,
    *,
    evidence_example_count: int,
    rank16: Gemma3Layer17LatentLiftMetricTriple,
    rank32: Gemma3Layer17LatentLiftMetricTriple,
) -> Gemma3Layer17LatentLiftParetoRejection:
    """Record aggregate open-development rejection without retaining rows."""

    if not isinstance(plan, Gemma3Layer17LatentLiftPlan):
        raise TypeError("plan must be Gemma3Layer17LatentLiftPlan")
    plan.validate_integrity()
    if not isinstance(
        rank16,
        Gemma3Layer17LatentLiftMetricTriple,
    ) or not isinstance(rank32, Gemma3Layer17LatentLiftMetricTriple):
        raise TypeError("rank16 and rank32 must be metric triples")
    return Gemma3Layer17LatentLiftParetoRejection(
        latent_lift_plan_sha256=plan.artifact_sha256,
        evidence_example_count=evidence_example_count,
        rank16=rank16,
        rank32=rank32,
        delta_nll_rank32_minus_rank16=(
            rank32.delta_nll_per_token - rank16.delta_nll_per_token
        ),
        delta_kl_rank32_minus_rank16=(
            rank32.native_to_candidate_kl_per_token
            - rank16.native_to_candidate_kl_per_token
        ),
        delta_top1_rank32_minus_rank16=(
            rank32.top1_agreement_to_native
            - rank16.top1_agreement_to_native
        ),
    )


def _canonical_rank_resources() -> tuple[
    Layer17NodeRankLadderPlan,
    Layer17NodeRankResourceRow,
    Layer17NodeRankResourceRow,
]:
    ladder = validate_layer17_node_rank_ladder_plan(
        build_default_layer17_node_rank_ladder_plan()
    )

    def select(generator_rank: int) -> Layer17NodeRankResourceRow:
        matches = tuple(
            row
            for row in ladder.rows
            if row.spec.mode_rank_cap
            == GEMMA3_LAYER17_LATENT_LIFT_MODE_RANK
            and row.spec.generator_rank == generator_rank
            and row.spec.edge_policy == GEMMA3_LAYER17_LATENT_LIFT_EDGE_POLICY
        )
        if len(matches) != 1:
            raise ValueError(
                "rank ladder does not contain one canonical latent-lift row"
            )
        row = matches[0]
        if (
            row.interaction_count != 0
            or row.interaction_parameter_count != 0
            or row.interaction_dense_macs_per_token != 0
            or row.conditional_routing_macs_per_token != 0
            or row.conditional_selected_message_macs_per_token_upper_bound != 0
            or row.graph_parameter_count != row.node_parameter_count
            or row.executed_graph_macs_per_token_upper_bound
            != row.node_macs_per_token
        ):
            raise ValueError(
                "latent-lift diagnostic ladder row is not strictly edgeless"
            )
        return row

    return (
        ladder,
        select(GEMMA3_LAYER17_LATENT_LIFT_SOURCE_GENERATOR_RANK),
        select(GEMMA3_LAYER17_LATENT_LIFT_TARGET_GENERATOR_RANK),
    )


def _canonical_fragments(
    graph: ModalGeneratorGraphPlan,
    fragments_by_node: Mapping[str, ParameterClusterLayerFragment],
) -> tuple[
    tuple[object, ...],
    tuple[ParameterClusterLayerFragment, ...],
    tuple[object, ...],
    tuple[ParameterClusterLayerFragment, ...],
]:
    if not isinstance(graph, ModalGeneratorGraphPlan):
        raise TypeError("source graph must be ModalGeneratorGraphPlan")
    graph.validate_integrity()
    if graph.interactions:
        raise ValueError(
            "the latent-lift diagnostic rejects every dynamic or static edge"
        )
    if not isinstance(fragments_by_node, Mapping):
        raise TypeError("fragments_by_node must be a mapping")
    names = tuple(node.name for node in graph.nodes)
    if set(fragments_by_node) != set(names):
        raise ValueError("fragments_by_node must exactly cover the source graph")
    pairs: dict[int, list[tuple[object, ParameterClusterLayerFragment]]] = {
        layer: [] for layer in _EXPECTED_LAYERS
    }
    for node in graph.nodes:
        fragment = fragments_by_node[node.name]
        if not isinstance(fragment, ParameterClusterLayerFragment):
            raise TypeError("fragments_by_node contains a non-fragment value")
        fragment.validate_integrity()
        if fragment.layer_ordinal not in pairs:
            raise ValueError(
                "the latent-lift diagnostic source graph must contain only "
                "layers 10 and 17"
            )
        if (
            node.weights.parameter_cluster_fragment_sha256
            != fragment.artifact_sha256
            or node.weights.source_model_sha256 != fragment.source_model_sha256
            or node.weights.source_model_sha256 != graph.model_fingerprint
            or node.weights.parameter_cluster_plan_sha256
            != graph.parameter_cluster_plan_sha256
        ):
            raise ValueError("source graph node/fragment binding drifted")
        pairs[fragment.layer_ordinal].append((node, fragment))
    if any(len(value) != _EXPECTED_NODE_COUNT_PER_LAYER for value in pairs.values()):
        raise ValueError(
            "the latent-lift diagnostic requires four source nodes at each layer"
        )
    layer10 = tuple(pairs[10])
    layer17 = tuple(pairs[17])
    return (
        tuple(node for node, _ in layer10),
        tuple(fragment for _, fragment in layer10),
        tuple(node for node, _ in layer17),
        tuple(fragment for _, fragment in layer17),
    )


def build_gemma3_layer17_latent_lift_plan(
    source_edgeless_graph: ModalGeneratorGraphPlan,
    *,
    fragments_by_node: Mapping[str, ParameterClusterLayerFragment],
) -> Gemma3Layer17LatentLiftPlan:
    """Bind the existing L10+L17 nodes to the fixed latent-lift diagnostic rank lift."""

    layer10_nodes, layer10_fragments, layer17_nodes, layer17_fragments = (
        _canonical_fragments(source_edgeless_graph, fragments_by_node)
    )
    if any(
        node.weights.private_width
        != GEMMA3_LAYER17_LATENT_LIFT_SOURCE_GENERATOR_RANK
        or node.weights.latent_width != GEMMA3_LAYER17_LATENT_LIFT_MODE_RANK
        or node.weights.state_factor is None
        for node in layer17_nodes
    ):
        raise ValueError("layer-17 source nodes must be rank-16 into mode rank 32")
    rank_ladder, source_rank_resource, target_rank_resource = (
        _canonical_rank_resources()
    )
    layer17_source_parameters = sum(
        node.weights.parameter_count for node in layer17_nodes
    )
    layer17_source_macs = sum(
        node.weights.macs_per_token for node in layer17_nodes
    )
    layer17_native = sum(
        fragment.native_parameter_count for fragment in layer17_fragments
    )
    if (
        layer17_native != LAYER17_SOURCE_PARAMETERS
        or layer17_source_parameters
        != source_rank_resource.graph_parameter_count
        or layer17_source_macs
        != source_rank_resource.executed_graph_macs_per_token_upper_bound
    ):
        raise ValueError("live layer-17 source resources differ from the rank ladder")
    layer10_native = sum(
        fragment.native_parameter_count for fragment in layer10_fragments
    )
    layer10_parameters = sum(
        node.weights.parameter_count for node in layer10_nodes
    )
    layer10_macs = sum(node.weights.macs_per_token for node in layer10_nodes)
    resources = Gemma3Layer17LatentLiftResources(
        layer10_native_removed_parameters=layer10_native,
        layer10_parent_parameters=layer10_parameters,
        layer10_parent_macs_per_token=layer10_macs,
        layer17_native_removed_parameters=layer17_native,
        layer17_source_parameters=layer17_source_parameters,
        layer17_target_parameters=target_rank_resource.graph_parameter_count,
        layer17_source_macs_per_token=layer17_source_macs,
        layer17_target_macs_per_token=(
            target_rank_resource.executed_graph_macs_per_token_upper_bound
        ),
        combined_native_removed_parameters=layer10_native + layer17_native,
        combined_source_parameters=layer10_parameters + layer17_source_parameters,
        combined_target_parameters=(
            layer10_parameters + target_rank_resource.graph_parameter_count
        ),
        combined_source_macs_per_token=layer10_macs + layer17_source_macs,
        combined_target_macs_per_token=(
            layer10_macs
            + target_rank_resource.executed_graph_macs_per_token_upper_bound
        ),
        rank_lift_added_parameters=(
            target_rank_resource.graph_parameter_count
            - layer17_source_parameters
        ),
        rank_lift_added_macs_per_token=(
            target_rank_resource.executed_graph_macs_per_token_upper_bound
            - layer17_source_macs
        ),
        layer17_target_net_parameter_savings=(
            layer17_native - target_rank_resource.graph_parameter_count
        ),
        layer17_target_net_macs_saved_per_token=(
            layer17_native
            - target_rank_resource.executed_graph_macs_per_token_upper_bound
        ),
        combined_target_net_parameter_savings=(
            layer10_native
            + layer17_native
            - layer10_parameters
            - target_rank_resource.graph_parameter_count
        ),
        combined_target_net_macs_saved_per_token=(
            layer10_native
            + layer17_native
            - layer10_macs
            - target_rank_resource.executed_graph_macs_per_token_upper_bound
        ),
    )
    return Gemma3Layer17LatentLiftPlan(
        source_edgeless_graph_sha256=source_edgeless_graph.artifact_sha256,
        model_fingerprint=source_edgeless_graph.model_fingerprint,
        parameter_cluster_plan_sha256=(
            source_edgeless_graph.parameter_cluster_plan_sha256
        ),
        layer10_node_names=tuple(node.name for node in layer10_nodes),
        layer10_node_sha256s=tuple(node.artifact_sha256 for node in layer10_nodes),
        layer10_fragment_ids=tuple(
            fragment.fragment_id for fragment in layer10_fragments
        ),
        layer10_fragment_sha256s=tuple(
            fragment.artifact_sha256 for fragment in layer10_fragments
        ),
        layer17_node_names=tuple(node.name for node in layer17_nodes),
        layer17_source_node_sha256s=tuple(
            node.artifact_sha256 for node in layer17_nodes
        ),
        layer17_fragment_ids=tuple(
            fragment.fragment_id for fragment in layer17_fragments
        ),
        layer17_fragment_sha256s=tuple(
            fragment.artifact_sha256 for fragment in layer17_fragments
        ),
        layer17_rank_ladder_sha256=rank_ladder.artifact_sha256,
        layer17_rank_resource_sha256=target_rank_resource.artifact_sha256,
        resources=resources,
    )


def build_gemma3_layer17_latent_lift_plan_from_lowerings(
    source_edgeless_graph: ModalGeneratorGraphPlan,
    lowerings: Sequence[ModalGeneratorLowering],
) -> Gemma3Layer17LatentLiftPlan:
    """Resolve authenticated fragments from composition-bundle lowerings."""

    if isinstance(lowerings, (str, bytes)) or not isinstance(lowerings, Sequence):
        raise TypeError("lowerings must be a sequence")
    by_weights: dict[str, list[ModalGeneratorLowering]] = {}
    for lowering in lowerings:
        if not isinstance(lowering, ModalGeneratorLowering):
            raise TypeError("lowerings contains a non-lowering value")
        lowering.validate_integrity()
        by_weights.setdefault(
            lowering.graph_weights.artifact_sha256,
            [],
        ).append(lowering)
    fragments: dict[str, ParameterClusterLayerFragment] = {}
    used_lowerings: set[str] = set()
    for node in source_edgeless_graph.nodes:
        matches = by_weights.get(node.weights.artifact_sha256, ())
        if len(matches) != 1:
            raise ValueError("each source graph node must match one lowering")
        lowering = matches[0]
        if lowering.artifact_sha256 in used_lowerings:
            raise ValueError(
                "a lowering cannot back multiple latent-lift diagnostic nodes"
            )
        used_lowerings.add(lowering.artifact_sha256)
        selected = tuple(
            fragment
            for fragment in lowering.fragment_plan.fragments
            if fragment.artifact_sha256 == lowering.selected_fragment_sha256
        )
        if len(selected) != 1:
            raise ValueError("lowering must bind one selected fragment")
        fragments[node.name] = selected[0]
    if len(used_lowerings) != len(lowerings):
        raise ValueError("lowerings must exactly cover the source graph")
    return build_gemma3_layer17_latent_lift_plan(
        source_edgeless_graph,
        fragments_by_node=fragments,
    )


def build_gemma3_layer17_latent_lift_edgeless_graph(
    plan: Gemma3Layer17LatentLiftPlan,
    *,
    layer10_parent_graph: ModalGeneratorGraphPlan,
    fitted_layer17_graph: ModalGeneratorGraphPlan,
) -> ModalGeneratorGraphPlan:
    """Assemble the rejected diagnostic after fitting, with no edges."""

    if not isinstance(plan, Gemma3Layer17LatentLiftPlan):
        raise TypeError("plan must be Gemma3Layer17LatentLiftPlan")
    plan.validate_integrity()
    for graph, label in (
        (layer10_parent_graph, "layer-10 parent"),
        (fitted_layer17_graph, "fitted layer-17 graph"),
    ):
        if not isinstance(graph, ModalGeneratorGraphPlan):
            raise TypeError(f"{label} must be ModalGeneratorGraphPlan")
        graph.validate_integrity()
        if graph.interactions:
            raise ValueError(f"the latent-lift diagnostic rejects edges in the {label}")
        if (
            graph.model_fingerprint != plan.model_fingerprint
            or graph.parameter_cluster_plan_sha256
            != plan.parameter_cluster_plan_sha256
        ):
            raise ValueError(f"{label} source binding drifted")
    if (
        tuple(node.name for node in layer10_parent_graph.nodes)
        != plan.layer10_node_names
        or tuple(node.artifact_sha256 for node in layer10_parent_graph.nodes)
        != plan.layer10_node_sha256s
        or layer10_parent_graph.parameter_count
        != plan.resources.layer10_parent_parameters
        or layer10_parent_graph.macs_per_token
        != plan.resources.layer10_parent_macs_per_token
    ):
        raise ValueError("layer-10 parent is not preserved exactly")
    if (
        tuple(node.name for node in fitted_layer17_graph.nodes)
        != plan.layer17_node_names
        or tuple(
            node.weights.parameter_cluster_fragment_sha256
            for node in fitted_layer17_graph.nodes
        )
        != plan.layer17_fragment_sha256s
        or any(
            node.weights.private_width
            != GEMMA3_LAYER17_LATENT_LIFT_TARGET_GENERATOR_RANK
            or node.weights.latent_width != GEMMA3_LAYER17_LATENT_LIFT_MODE_RANK
            or node.weights.state_factor is None
            for node in fitted_layer17_graph.nodes
        )
        or fitted_layer17_graph.parameter_count
        != plan.resources.layer17_target_parameters
        or fitted_layer17_graph.macs_per_token
        != plan.resources.layer17_target_macs_per_token
    ):
        raise ValueError(
            "fitted layer-17 graph does not realize the latent-lift diagnostic"
        )
    combined = ModalGeneratorGraphPlan(
        model_fingerprint=plan.model_fingerprint,
        parameter_cluster_plan_sha256=plan.parameter_cluster_plan_sha256,
        nodes=tuple(
            sorted(
                (*layer10_parent_graph.nodes, *fitted_layer17_graph.nodes),
                key=lambda node: (node.causal_order, node.name),
            )
        ),
        interactions=(),
    )
    if (
        combined.parameter_count != plan.resources.combined_target_parameters
        or combined.macs_per_token
        != plan.resources.combined_target_macs_per_token
    ):
        raise RuntimeError("assembled latent-lift diagnostic graph resources drifted")
    return combined
