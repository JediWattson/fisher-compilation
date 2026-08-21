"""Layer-agnostic identities and accounting for whole-model modal graphs.

This module deliberately contains no model or tensor runtime.  It defines the
source-independent artifact vocabulary required by the next compiler rung:

* source modes retain native layer names only as provenance;
* generator families own reusable parameters and may span source layers;
* generator invocations are distinct, causally ordered execution events; and
* transports connect invocations, never native layer objects.

The separation between a family and an invocation is essential.  Reusing one
family at mutation 3 and mutation 19 does not create one self-referential graph
node: it creates two invocations that happen to share parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence


LAYER_AGNOSTIC_MODAL_GRAPH_SCHEMA = (
    "fisher_graph.layer_agnostic_modal_graph.plan.v1"
)
LAYER_AGNOSTIC_MODAL_GRAPH_FORMAT_VERSION = 1

_ARTIFACT_HASH_DOMAIN = b"fisher_graph:layer-agnostic-modal-graph:v1\0"
_EXECUTION_HASH_DOMAIN = (
    b"fisher_graph:layer-agnostic-modal-graph-execution:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
_TOKEN_DEPENDENCIES = frozenset(
    {"token_local", "causal_prefix_mixer", "vocabulary_boundary"}
)
_TOKEN_RELATIONS = frozenset({"same_token", "causal_prefix"})


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CANONICAL_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a bool")
    return value


def _unique_sorted_ids(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(values)
    for value in result:
        _require_id(value, label=label)
    if not result or result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be nonempty, unique, and sorted")
    return result


@dataclass(frozen=True, slots=True)
class SourceModeProvenance:
    """One discovered primitive mode plus its audit-only native provenance."""

    mode_id: str
    mutation_index: int
    coordinate_index: int
    fisher_signature_sha256: str
    geometry_signature_sha256: str
    runtime_signature_sha256: str
    source_boundary_id: str
    source_operator_kind: str
    source_layer_ordinal: int | None = None

    def __post_init__(self) -> None:
        _require_id(self.mode_id, label="mode_id")
        _require_nonnegative_int(self.mutation_index, label="mutation_index")
        _require_nonnegative_int(self.coordinate_index, label="coordinate_index")
        _require_sha256(
            self.fisher_signature_sha256,
            label="fisher_signature_sha256",
        )
        _require_sha256(
            self.geometry_signature_sha256,
            label="geometry_signature_sha256",
        )
        _require_sha256(
            self.runtime_signature_sha256,
            label="runtime_signature_sha256",
        )
        _require_id(self.source_boundary_id, label="source_boundary_id")
        _require_id(self.source_operator_kind, label="source_operator_kind")
        if self.source_layer_ordinal is not None:
            _require_nonnegative_int(
                self.source_layer_ordinal,
                label="source_layer_ordinal",
            )

    def clustering_feature_state(self) -> dict[str, object]:
        """Return the only fields legal as clustering features.

        Native layer, boundary, operator label, and layer distance are excluded.
        ``mode_id`` is also excluded because a human-readable id may encode
        provenance.  The fitter carries it only as an opaque row key beside
        this feature state.
        """

        return {
            "coordinate_index": self.coordinate_index,
            "fisher_signature_sha256": self.fisher_signature_sha256,
            "geometry_signature_sha256": self.geometry_signature_sha256,
            "runtime_signature_sha256": self.runtime_signature_sha256,
        }

    def discovery_state(self) -> dict[str, object]:
        return {
            "opaque_mode_key": self.mode_id,
            "clustering_features": self.clustering_feature_state(),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "mode_id": self.mode_id,
            **self.clustering_feature_state(),
            "mutation_index": self.mutation_index,
            "provenance": {
                "source_boundary_id": self.source_boundary_id,
                "source_operator_kind": self.source_operator_kind,
                "source_layer_ordinal": self.source_layer_ordinal,
            },
        }


@dataclass(frozen=True, slots=True)
class GeneratorFamily:
    """A reusable parameter family discovered across primitive source modes."""

    family_id: str
    member_mode_ids: tuple[str, ...]
    coordinate_rank: int
    basis_sha256: str
    generator_bank_sha256: str
    compatible_runtime_signature_sha256s: tuple[str, ...]
    resource_component_id: str
    basis_scalar_count: int
    generator_bank_scalar_count: int

    def __post_init__(self) -> None:
        _require_id(self.family_id, label="family_id")
        object.__setattr__(
            self,
            "member_mode_ids",
            _unique_sorted_ids(self.member_mode_ids, label="member_mode_ids"),
        )
        _require_positive_int(self.coordinate_rank, label="coordinate_rank")
        _require_sha256(self.basis_sha256, label="basis_sha256")
        _require_sha256(
            self.generator_bank_sha256,
            label="generator_bank_sha256",
        )
        signatures = tuple(self.compatible_runtime_signature_sha256s)
        if (
            not signatures
            or signatures != tuple(sorted(set(signatures)))
        ):
            raise ValueError(
                "compatible runtime signatures must be nonempty and unique"
            )
        for signature in signatures:
            _require_sha256(signature, label="compatible_runtime_signature")
        object.__setattr__(
            self,
            "compatible_runtime_signature_sha256s",
            signatures,
        )
        _require_id(self.resource_component_id, label="resource_component_id")
        _require_nonnegative_int(
            self.basis_scalar_count,
            label="basis_scalar_count",
        )
        _require_nonnegative_int(
            self.generator_bank_scalar_count,
            label="generator_bank_scalar_count",
        )
        if self.stored_scalar_count <= 0:
            raise ValueError("generator family must own stored coefficients")

    @property
    def stored_scalar_count(self) -> int:
        return self.basis_scalar_count + self.generator_bank_scalar_count

    @property
    def artifact_sha256(self) -> str:
        return _sha256(
            b"fisher_graph:layer-agnostic-generator-family:v1\0",
            self.state_dict(),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "member_mode_ids": list(self.member_mode_ids),
            "coordinate_rank": self.coordinate_rank,
            "basis_sha256": self.basis_sha256,
            "generator_bank_sha256": self.generator_bank_sha256,
            "compatible_runtime_signature_sha256s": list(
                self.compatible_runtime_signature_sha256s
            ),
            "resource_component_id": self.resource_component_id,
            "basis_scalar_count": self.basis_scalar_count,
            "generator_bank_scalar_count": self.generator_bank_scalar_count,
            "stored_scalar_count": self.stored_scalar_count,
        }


@dataclass(frozen=True, slots=True)
class CoordinateAdapterContract:
    """One authenticated dense-linear chart-to-family coordinate map.

    The v1 adapter is deliberately concrete: a source chart coordinate of
    rank ``r_s`` is multiplied by a stored ``r_t x r_s`` matrix.  A resource
    id alone is not evidence that this map exists, so the dimensions, runtime
    signatures, coefficient artifact, storage, and per-token MAC count are all
    authenticated here and closed against the deployment ledger below.
    """

    adapter_id: str
    source_region_id: str
    source_runtime_signature_sha256: str
    source_coordinate_rank: int
    target_family_id: str
    target_runtime_signature_sha256: str
    target_coordinate_rank: int
    adapter_artifact_sha256: str
    resource_component_id: str
    stored_scalar_count: int
    macs_per_valid_token: int

    def __post_init__(self) -> None:
        _require_id(self.adapter_id, label="adapter_id")
        _require_id(self.source_region_id, label="source_region_id")
        _require_sha256(
            self.source_runtime_signature_sha256,
            label="source_runtime_signature_sha256",
        )
        _require_positive_int(
            self.source_coordinate_rank,
            label="source_coordinate_rank",
        )
        _require_id(self.target_family_id, label="target_family_id")
        _require_sha256(
            self.target_runtime_signature_sha256,
            label="target_runtime_signature_sha256",
        )
        _require_positive_int(
            self.target_coordinate_rank,
            label="target_coordinate_rank",
        )
        _require_sha256(
            self.adapter_artifact_sha256,
            label="adapter_artifact_sha256",
        )
        _require_id(self.resource_component_id, label="resource_component_id")
        _require_positive_int(
            self.stored_scalar_count,
            label="stored_scalar_count",
        )
        _require_positive_int(
            self.macs_per_valid_token,
            label="macs_per_valid_token",
        )
        dense_linear_count = (
            self.source_coordinate_rank * self.target_coordinate_rank
        )
        if self.stored_scalar_count != dense_linear_count:
            raise ValueError(
                "coordinate adapter scalar count must exactly cover its "
                "dense-linear map"
            )
        if self.macs_per_valid_token != dense_linear_count:
            raise ValueError(
                "coordinate adapter MAC count must exactly cover its "
                "dense-linear map"
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "source_region_id": self.source_region_id,
            "source_runtime_signature_sha256": (
                self.source_runtime_signature_sha256
            ),
            "source_coordinate_rank": self.source_coordinate_rank,
            "target_family_id": self.target_family_id,
            "target_runtime_signature_sha256": (
                self.target_runtime_signature_sha256
            ),
            "target_coordinate_rank": self.target_coordinate_rank,
            "adapter_artifact_sha256": self.adapter_artifact_sha256,
            "resource_component_id": self.resource_component_id,
            "stored_scalar_count": self.stored_scalar_count,
            "macs_per_valid_token": self.macs_per_valid_token,
        }


@dataclass(frozen=True, slots=True)
class GeneratorInvocation:
    """One temporal use of a generator family in the causal graph."""

    invocation_id: str
    family_id: str
    mutation_index: int
    source_mode_ids: tuple[str, ...]
    input_boundary_id: str
    output_boundary_id: str
    token_dependency: str
    operator_role: str
    runtime_signature_sha256: str
    sequence_operator_contract_sha256: str
    source_mutation_id: str
    resource_component_ids: tuple[str, ...]
    local_adapter_scalar_count: int
    expected_macs_per_valid_token: int
    attention_macs_per_causal_pair: int
    non_mac_ops_per_valid_token: int
    coordinate_adapter: CoordinateAdapterContract | None = None

    def __post_init__(self) -> None:
        _require_id(self.invocation_id, label="invocation_id")
        _require_id(self.family_id, label="family_id")
        _require_nonnegative_int(self.mutation_index, label="mutation_index")
        object.__setattr__(
            self,
            "source_mode_ids",
            _unique_sorted_ids(self.source_mode_ids, label="source_mode_ids"),
        )
        _require_id(self.input_boundary_id, label="input_boundary_id")
        _require_id(self.output_boundary_id, label="output_boundary_id")
        if self.token_dependency not in _TOKEN_DEPENDENCIES:
            raise ValueError("token_dependency is invalid")
        _require_id(self.operator_role, label="operator_role")
        _require_sha256(
            self.runtime_signature_sha256,
            label="runtime_signature_sha256",
        )
        _require_sha256(
            self.sequence_operator_contract_sha256,
            label="sequence_operator_contract_sha256",
        )
        _require_id(self.source_mutation_id, label="source_mutation_id")
        object.__setattr__(
            self,
            "resource_component_ids",
            _unique_sorted_ids(
                self.resource_component_ids,
                label="resource_component_ids",
            ),
        )
        if self.coordinate_adapter is not None:
            if not isinstance(
                self.coordinate_adapter,
                CoordinateAdapterContract,
            ):
                raise ValueError(
                    "coordinate_adapter must be a CoordinateAdapterContract"
                )
            if (
                self.coordinate_adapter.resource_component_id
                not in self.resource_component_ids
            ):
                raise ValueError(
                    "coordinate adapter resource must belong to its invocation"
                )
        for label, value in (
            ("local_adapter_scalar_count", self.local_adapter_scalar_count),
            (
                "expected_macs_per_valid_token",
                self.expected_macs_per_valid_token,
            ),
            (
                "attention_macs_per_causal_pair",
                self.attention_macs_per_causal_pair,
            ),
            ("non_mac_ops_per_valid_token", self.non_mac_ops_per_valid_token),
        ):
            _require_nonnegative_int(value, label=label)

    def state_dict(self) -> dict[str, object]:
        return {
            "invocation_id": self.invocation_id,
            "family_id": self.family_id,
            "mutation_index": self.mutation_index,
            "source_mode_ids": list(self.source_mode_ids),
            "input_boundary_id": self.input_boundary_id,
            "output_boundary_id": self.output_boundary_id,
            "token_dependency": self.token_dependency,
            "operator_role": self.operator_role,
            "runtime_signature_sha256": self.runtime_signature_sha256,
            "sequence_operator_contract_sha256": (
                self.sequence_operator_contract_sha256
            ),
            "source_mutation_id": self.source_mutation_id,
            "resource_component_ids": list(self.resource_component_ids),
            "local_adapter_scalar_count": self.local_adapter_scalar_count,
            "expected_macs_per_valid_token": (
                self.expected_macs_per_valid_token
            ),
            "attention_macs_per_causal_pair": (
                self.attention_macs_per_causal_pair
            ),
            "non_mac_ops_per_valid_token": self.non_mac_ops_per_valid_token,
            "coordinate_adapter": (
                None
                if self.coordinate_adapter is None
                else self.coordinate_adapter.state_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class GeneratorTransport:
    """A strictly forward, token-causal interaction between invocations."""

    transport_id: str
    source_invocation_id: str
    target_invocation_id: str
    token_relation: str
    source_token_lag: int
    coefficient_sha256: str
    stored_scalar_count: int
    macs_per_active_token: int
    attention_macs_per_causal_pair: int
    resource_component_id: str

    def __post_init__(self) -> None:
        _require_id(self.transport_id, label="transport_id")
        _require_id(self.source_invocation_id, label="source_invocation_id")
        _require_id(self.target_invocation_id, label="target_invocation_id")
        if self.token_relation not in _TOKEN_RELATIONS:
            raise ValueError("token_relation is invalid")
        _require_nonnegative_int(self.source_token_lag, label="source_token_lag")
        if self.token_relation == "same_token" and self.source_token_lag != 0:
            raise ValueError("same-token transports must have zero token lag")
        _require_sha256(self.coefficient_sha256, label="coefficient_sha256")
        _require_nonnegative_int(
            self.stored_scalar_count,
            label="stored_scalar_count",
        )
        _require_nonnegative_int(
            self.macs_per_active_token,
            label="macs_per_active_token",
        )
        _require_nonnegative_int(
            self.attention_macs_per_causal_pair,
            label="attention_macs_per_causal_pair",
        )
        _require_id(self.resource_component_id, label="resource_component_id")

    def state_dict(self) -> dict[str, object]:
        return {
            "transport_id": self.transport_id,
            "source_invocation_id": self.source_invocation_id,
            "target_invocation_id": self.target_invocation_id,
            "token_relation": self.token_relation,
            "source_token_lag": self.source_token_lag,
            "coefficient_sha256": self.coefficient_sha256,
            "stored_scalar_count": self.stored_scalar_count,
            "macs_per_active_token": self.macs_per_active_token,
            "attention_macs_per_causal_pair": (
                self.attention_macs_per_causal_pair
            ),
            "resource_component_id": self.resource_component_id,
        }


@dataclass(frozen=True, slots=True)
class LayerAgnosticPlanLineage:
    """Hashes binding a fitted plan to its preregistration and fit evidence."""

    protocol_sha256: str
    source_model_sha256: str
    fit_split_manifest_sha256: str
    fit_objective_sha256: str
    exact_control_execution_sha256: str
    arm_id: str = "unbound"
    resource_point_id: str = "unbound"
    resource_point_sha256: str = "0" * 64
    fit_configuration_sha256: str = "0" * 64
    input_contract_sha256: str = "0" * 64
    fit_relative_ridge: float | None = None
    resource_point_fit_receipt_sha256: str = "0" * 64

    def __post_init__(self) -> None:
        for label, value in (
            ("protocol_sha256", self.protocol_sha256),
            ("source_model_sha256", self.source_model_sha256),
            ("fit_split_manifest_sha256", self.fit_split_manifest_sha256),
            ("fit_objective_sha256", self.fit_objective_sha256),
            (
                "exact_control_execution_sha256",
                self.exact_control_execution_sha256,
            ),
        ):
            _require_sha256(value, label=label)
        _require_id(self.arm_id, label="arm_id")
        _require_id(self.resource_point_id, label="resource_point_id")
        for label, value in (
            ("resource_point_sha256", self.resource_point_sha256),
            ("fit_configuration_sha256", self.fit_configuration_sha256),
            ("input_contract_sha256", self.input_contract_sha256),
            (
                "resource_point_fit_receipt_sha256",
                self.resource_point_fit_receipt_sha256,
            ),
        ):
            _require_sha256(value, label=label)
        if self.fit_relative_ridge is not None and (
            not isinstance(self.fit_relative_ridge, float)
            or not math.isfinite(self.fit_relative_ridge)
            or self.fit_relative_ridge <= 0.0
        ):
            raise ValueError(
                "fit_relative_ridge must be finite and positive or None"
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "source_model_sha256": self.source_model_sha256,
            "fit_split_manifest_sha256": self.fit_split_manifest_sha256,
            "fit_objective_sha256": self.fit_objective_sha256,
            "exact_control_execution_sha256": (
                self.exact_control_execution_sha256
            ),
            "arm_id": self.arm_id,
            "resource_point_id": self.resource_point_id,
            "resource_point_sha256": self.resource_point_sha256,
            "fit_configuration_sha256": self.fit_configuration_sha256,
            "input_contract_sha256": self.input_contract_sha256,
            "fit_relative_ridge": self.fit_relative_ridge,
            "resource_point_fit_receipt_sha256": (
                self.resource_point_fit_receipt_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class ModalChartRegion:
    """One authenticated local residual-state chart."""

    region_id: str
    input_width: int
    coordinate_rank: int
    center_sha256: str
    basis_sha256: str
    metric_sha256: str
    compatible_runtime_signature_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id(self.region_id, label="region_id")
        _require_positive_int(self.input_width, label="input_width")
        _require_positive_int(self.coordinate_rank, label="coordinate_rank")
        if self.coordinate_rank > self.input_width:
            raise ValueError("chart rank cannot exceed its input width")
        _require_sha256(self.center_sha256, label="center_sha256")
        _require_sha256(self.basis_sha256, label="basis_sha256")
        _require_sha256(self.metric_sha256, label="metric_sha256")
        signatures = tuple(self.compatible_runtime_signature_sha256s)
        if (
            not signatures
            or signatures != tuple(sorted(set(signatures)))
        ):
            raise ValueError(
                "chart runtime signatures must be nonempty, unique, and sorted"
            )
        for signature in signatures:
            _require_sha256(signature, label="chart_runtime_signature")
        object.__setattr__(
            self,
            "compatible_runtime_signature_sha256s",
            signatures,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "input_width": self.input_width,
            "coordinate_rank": self.coordinate_rank,
            "center_sha256": self.center_sha256,
            "basis_sha256": self.basis_sha256,
            "metric_sha256": self.metric_sha256,
            "compatible_runtime_signature_sha256s": list(
                self.compatible_runtime_signature_sha256s
            ),
            "stored_scalar_count": self.stored_scalar_count,
        }

    @property
    def stored_scalar_count(self) -> int:
        return (
            self.input_width
            + self.input_width * self.coordinate_rank
            + self.coordinate_rank * self.coordinate_rank
        )


@dataclass(frozen=True, slots=True)
class ModalChartAtlas:
    """A pooled chart atlas used by carrier-native generator invocations."""

    atlas_id: str
    input_width: int
    regions: tuple[ModalChartRegion, ...]
    membership_rule: str
    temperature: float
    router_sha256: str
    carrier_runtime_signature_sha256: str
    artifact_resource_component_id: str
    router_resource_component_id: str
    resource_component_ids: tuple[str, ...]
    active_chart_count: int
    router_scalar_count: int
    routing_macs_per_valid_token: int
    routing_non_mac_ops_per_valid_token: int
    local_correction_rank: int
    local_correction_artifact_sha256: str
    local_correction_resource_component_id: str
    local_correction_scalar_count: int
    local_correction_macs_per_valid_token: int

    def __post_init__(self) -> None:
        _require_id(self.atlas_id, label="atlas_id")
        _require_positive_int(self.input_width, label="input_width")
        regions = tuple(self.regions)
        if not regions or len({region.region_id for region in regions}) != len(
            regions
        ):
            raise ValueError("chart regions must be nonempty and unique")
        if any(region.input_width != self.input_width for region in regions):
            raise ValueError("chart region width does not match its atlas")
        object.__setattr__(self, "regions", regions)
        _require_id(self.membership_rule, label="membership_rule")
        if (
            not isinstance(self.temperature, float)
            or not math.isfinite(self.temperature)
            or self.temperature <= 0.0
        ):
            raise ValueError("chart temperature must be finite and positive")
        _require_sha256(self.router_sha256, label="router_sha256")
        _require_sha256(
            self.carrier_runtime_signature_sha256,
            label="carrier_runtime_signature_sha256",
        )
        _require_id(
            self.artifact_resource_component_id,
            label="artifact_resource_component_id",
        )
        _require_id(
            self.router_resource_component_id,
            label="router_resource_component_id",
        )
        object.__setattr__(
            self,
            "resource_component_ids",
            _unique_sorted_ids(
                self.resource_component_ids,
                label="resource_component_ids",
            ),
        )
        if self.artifact_resource_component_id not in self.resource_component_ids:
            raise ValueError("atlas artifact resource must be in its resource set")
        if self.router_resource_component_id not in self.resource_component_ids:
            raise ValueError("atlas router resource must be in its resource set")
        _require_positive_int(
            self.active_chart_count,
            label="active_chart_count",
        )
        if self.active_chart_count > len(self.regions):
            raise ValueError("active chart count cannot exceed atlas regions")
        _require_nonnegative_int(
            self.local_correction_rank,
            label="local_correction_rank",
        )
        if self.local_correction_rank > self.input_width:
            raise ValueError("local correction rank cannot exceed input width")
        _require_sha256(
            self.local_correction_artifact_sha256,
            label="local_correction_artifact_sha256",
        )
        _require_id(
            self.local_correction_resource_component_id,
            label="local_correction_resource_component_id",
        )
        if (
            self.local_correction_resource_component_id
            not in self.resource_component_ids
        ):
            raise ValueError(
                "local correction resource must be in the atlas resource set"
            )
        if len(
            {
                self.artifact_resource_component_id,
                self.router_resource_component_id,
                self.local_correction_resource_component_id,
            }
        ) != 3:
            raise ValueError("atlas resources must have distinct ownership")
        for label, value in (
            ("router_scalar_count", self.router_scalar_count),
            (
                "routing_macs_per_valid_token",
                self.routing_macs_per_valid_token,
            ),
            (
                "routing_non_mac_ops_per_valid_token",
                self.routing_non_mac_ops_per_valid_token,
            ),
            (
                "local_correction_scalar_count",
                self.local_correction_scalar_count,
            ),
            (
                "local_correction_macs_per_valid_token",
                self.local_correction_macs_per_valid_token,
            ),
        ):
            _require_nonnegative_int(value, label=label)
        if self.local_correction_scalar_count != (
            2
            * self.input_width
            * self.local_correction_rank
            * len(self.regions)
        ):
            raise ValueError(
                "local correction scalar count must exactly cover two "
                "dense factors per chart region"
            )
        if self.local_correction_macs_per_valid_token != (
            2
            * self.input_width
            * self.local_correction_rank
            * self.active_chart_count
        ):
            raise ValueError(
                "local correction MAC count must exactly cover every active "
                "chart's two dense factors"
            )

    @property
    def region_scalar_count(self) -> int:
        return sum(region.stored_scalar_count for region in self.regions)

    @property
    def stored_scalar_count(self) -> int:
        return (
            self.region_scalar_count
            + self.router_scalar_count
            + self.local_correction_scalar_count
        )

    @property
    def artifact_sha256(self) -> str:
        return _sha256(
            b"fisher_graph:layer-agnostic-chart-atlas:v1\0",
            self.state_dict(),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "atlas_id": self.atlas_id,
            "input_width": self.input_width,
            "regions": [
                region.state_dict()
                for region in sorted(self.regions, key=lambda item: item.region_id)
            ],
            "membership_rule": self.membership_rule,
            "temperature": self.temperature,
            "router_sha256": self.router_sha256,
            "carrier_runtime_signature_sha256": (
                self.carrier_runtime_signature_sha256
            ),
            "artifact_resource_component_id": (
                self.artifact_resource_component_id
            ),
            "router_resource_component_id": self.router_resource_component_id,
            "local_correction_resource_component_id": (
                self.local_correction_resource_component_id
            ),
            "resource_component_ids": list(self.resource_component_ids),
            "active_chart_count": self.active_chart_count,
            "router_scalar_count": self.router_scalar_count,
            "region_scalar_count": self.region_scalar_count,
            "stored_scalar_count": self.stored_scalar_count,
            "routing_macs_per_valid_token": (
                self.routing_macs_per_valid_token
            ),
            "routing_non_mac_ops_per_valid_token": (
                self.routing_non_mac_ops_per_valid_token
            ),
            "local_correction_rank": self.local_correction_rank,
            "local_correction_artifact_sha256": (
                self.local_correction_artifact_sha256
            ),
            "local_correction_scalar_count": (
                self.local_correction_scalar_count
            ),
            "local_correction_macs_per_valid_token": (
                self.local_correction_macs_per_valid_token
            ),
        }


@dataclass(frozen=True, slots=True)
class ResidualMutationAssembly:
    """One closed carrier transition containing one or more invocations."""

    assembly_index: int
    assembly_role: str
    input_carrier_id: str
    output_carrier_id: str
    invocation_ids: tuple[str, ...]
    source_mutation_id: str
    replacement_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.assembly_index, label="assembly_index")
        _require_id(self.assembly_role, label="assembly_role")
        _require_id(self.input_carrier_id, label="input_carrier_id")
        _require_id(self.output_carrier_id, label="output_carrier_id")
        if self.input_carrier_id == self.output_carrier_id:
            raise ValueError("carrier transition must advance to a new identity")
        object.__setattr__(
            self,
            "invocation_ids",
            _unique_sorted_ids(self.invocation_ids, label="invocation_ids"),
        )
        _require_id(self.source_mutation_id, label="source_mutation_id")
        _require_sha256(
            self.replacement_receipt_sha256,
            label="replacement_receipt_sha256",
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "assembly_index": self.assembly_index,
            "assembly_role": self.assembly_role,
            "input_carrier_id": self.input_carrier_id,
            "output_carrier_id": self.output_carrier_id,
            "invocation_ids": list(self.invocation_ids),
            "source_mutation_id": self.source_mutation_id,
            "replacement_receipt_sha256": self.replacement_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class SharedLexicalBoundaryPlan:
    """One physical lexical factor referenced by ingress and egress."""

    shared_factor_id: str
    factor_artifact_sha256: str
    vocabulary_size: int
    source_width: int
    lexical_rank: int
    ingress_invocation_id: str
    egress_invocation_id: str
    terminal_norm_invocation_id: str
    generator_family_id: str
    factor_resource_component_id: str
    ingress_map_resource_component_id: str
    egress_map_resource_component_id: str
    terminal_norm_resource_component_id: str
    physical_storage_receipt_sha256: str
    ingress_map_artifact_sha256: str
    egress_map_artifact_sha256: str
    terminal_norm_artifact_sha256: str
    ingress_map_scalar_count: int
    egress_map_scalar_count: int
    terminal_norm_scalar_count: int
    storage_multiplicity: int = 1
    physical_storage_alias_required: bool = True
    terminal_rmsnorm_exact: bool = True
    logit_softcap_exact: bool = True

    def __post_init__(self) -> None:
        _require_id(self.shared_factor_id, label="shared_factor_id")
        _require_sha256(
            self.factor_artifact_sha256,
            label="factor_artifact_sha256",
        )
        _require_positive_int(self.vocabulary_size, label="vocabulary_size")
        _require_positive_int(self.source_width, label="source_width")
        _require_positive_int(self.lexical_rank, label="lexical_rank")
        if self.lexical_rank > min(self.vocabulary_size, self.source_width):
            raise ValueError("lexical rank exceeds the matrix rank bound")
        _require_id(self.ingress_invocation_id, label="ingress_invocation_id")
        _require_id(self.egress_invocation_id, label="egress_invocation_id")
        _require_id(
            self.terminal_norm_invocation_id,
            label="terminal_norm_invocation_id",
        )
        _require_id(self.generator_family_id, label="generator_family_id")
        for label, value in (
            ("factor_resource_component_id", self.factor_resource_component_id),
            (
                "ingress_map_resource_component_id",
                self.ingress_map_resource_component_id,
            ),
            (
                "egress_map_resource_component_id",
                self.egress_map_resource_component_id,
            ),
            (
                "terminal_norm_resource_component_id",
                self.terminal_norm_resource_component_id,
            ),
        ):
            _require_id(value, label=label)
        _require_sha256(
            self.physical_storage_receipt_sha256,
            label="physical_storage_receipt_sha256",
        )
        for label, value in (
            ("ingress_map_artifact_sha256", self.ingress_map_artifact_sha256),
            ("egress_map_artifact_sha256", self.egress_map_artifact_sha256),
            (
                "terminal_norm_artifact_sha256",
                self.terminal_norm_artifact_sha256,
            ),
        ):
            _require_sha256(value, label=label)
        for label, value in (
            ("ingress_map_scalar_count", self.ingress_map_scalar_count),
            ("egress_map_scalar_count", self.egress_map_scalar_count),
            ("terminal_norm_scalar_count", self.terminal_norm_scalar_count),
        ):
            _require_nonnegative_int(value, label=label)
        if self.storage_multiplicity != 1:
            raise ValueError("the tied lexical factor must be stored exactly once")
        _require_bool(
            self.physical_storage_alias_required,
            label="physical_storage_alias_required",
        )
        _require_bool(
            self.terminal_rmsnorm_exact,
            label="terminal_rmsnorm_exact",
        )
        _require_bool(self.logit_softcap_exact, label="logit_softcap_exact")
        if not self.physical_storage_alias_required:
            raise ValueError("numerical equality is not sufficient lexical tying")
        if not self.terminal_rmsnorm_exact or not self.logit_softcap_exact:
            raise ValueError("v1 requires exact terminal RMSNorm and softcap")

    @property
    def factorized_scalar_count(self) -> int:
        if self.lexical_rank == self.source_width:
            return self.vocabulary_size * self.source_width
        return self.lexical_rank * (self.vocabulary_size + self.source_width)

    def state_dict(self) -> dict[str, object]:
        return {
            "shared_factor_id": self.shared_factor_id,
            "factor_artifact_sha256": self.factor_artifact_sha256,
            "vocabulary_size": self.vocabulary_size,
            "source_width": self.source_width,
            "lexical_rank": self.lexical_rank,
            "ingress_invocation_id": self.ingress_invocation_id,
            "egress_invocation_id": self.egress_invocation_id,
            "terminal_norm_invocation_id": self.terminal_norm_invocation_id,
            "generator_family_id": self.generator_family_id,
            "factor_resource_component_id": (
                self.factor_resource_component_id
            ),
            "ingress_map_resource_component_id": (
                self.ingress_map_resource_component_id
            ),
            "egress_map_resource_component_id": (
                self.egress_map_resource_component_id
            ),
            "terminal_norm_resource_component_id": (
                self.terminal_norm_resource_component_id
            ),
            "physical_storage_receipt_sha256": (
                self.physical_storage_receipt_sha256
            ),
            "ingress_map_artifact_sha256": (
                self.ingress_map_artifact_sha256
            ),
            "egress_map_artifact_sha256": self.egress_map_artifact_sha256,
            "terminal_norm_artifact_sha256": (
                self.terminal_norm_artifact_sha256
            ),
            "ingress_map_scalar_count": self.ingress_map_scalar_count,
            "egress_map_scalar_count": self.egress_map_scalar_count,
            "terminal_norm_scalar_count": self.terminal_norm_scalar_count,
            "storage_multiplicity": self.storage_multiplicity,
            "physical_storage_alias_required": (
                self.physical_storage_alias_required
            ),
            "terminal_rmsnorm_exact": self.terminal_rmsnorm_exact,
            "logit_softcap_exact": self.logit_softcap_exact,
            "factorized_scalar_count": self.factorized_scalar_count,
        }


@dataclass(frozen=True, slots=True)
class LayerAgnosticResourceComponent:
    """One physically stored candidate component and its logical compute."""

    component_id: str
    component_kind: str
    artifact_sha256: str
    storage_encoding: str
    learned_scalar_count: int
    fixed_scalar_count: int
    serialized_byte_offset: int
    serialized_bytes: int
    expected_macs_per_token: int
    worst_case_macs_per_token: int
    attention_macs_per_causal_pair: int
    causal_pair_contract_sha256: str | None
    non_mac_ops_per_token: int
    serialized_size_audited: bool
    physical_storage_alias_id: str | None = None
    unaccounted_or_live_fallback: bool = False

    def __post_init__(self) -> None:
        _require_id(self.component_id, label="component_id")
        _require_id(self.component_kind, label="component_kind")
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        _require_id(self.storage_encoding, label="storage_encoding")
        for label, value in (
            ("learned_scalar_count", self.learned_scalar_count),
            ("fixed_scalar_count", self.fixed_scalar_count),
            ("serialized_byte_offset", self.serialized_byte_offset),
            ("serialized_bytes", self.serialized_bytes),
            ("expected_macs_per_token", self.expected_macs_per_token),
            ("worst_case_macs_per_token", self.worst_case_macs_per_token),
            (
                "attention_macs_per_causal_pair",
                self.attention_macs_per_causal_pair,
            ),
            ("non_mac_ops_per_token", self.non_mac_ops_per_token),
        ):
            _require_nonnegative_int(value, label=label)
        if self.expected_macs_per_token > self.worst_case_macs_per_token:
            raise ValueError("expected MACs cannot exceed worst-case MACs")
        scalar_count = self.learned_scalar_count + self.fixed_scalar_count
        encoding_bits = {
            "float32": 32,
            "float16": 16,
            "bfloat16": 16,
            "int8": 8,
            "int4_packed": 4,
            "metadata_bytes": 0,
        }.get(self.storage_encoding)
        if encoding_bits is None:
            raise ValueError("storage_encoding is not supported")
        if scalar_count > 0 and encoding_bits == 0:
            raise ValueError("scalar payload cannot use metadata-only encoding")
        minimum_payload_bytes = (
            scalar_count * encoding_bits + 7
        ) // 8
        if self.serialized_bytes < minimum_payload_bytes:
            raise ValueError("serialized span is too small for its scalar payload")
        if scalar_count > 0 and self.serialized_bytes == 0:
            raise ValueError("stored scalars require a nonempty serialized span")
        if self.attention_macs_per_causal_pair > 0:
            _require_sha256(
                self.causal_pair_contract_sha256,
                label="causal_pair_contract_sha256",
            )
        elif self.causal_pair_contract_sha256 is not None:
            raise ValueError("zero pair-MAC components cannot bind a pair contract")
        if self.physical_storage_alias_id is not None:
            _require_id(
                self.physical_storage_alias_id,
                label="physical_storage_alias_id",
            )
        _require_bool(
            self.serialized_size_audited,
            label="serialized_size_audited",
        )
        _require_bool(
            self.unaccounted_or_live_fallback,
            label="unaccounted_or_live_fallback",
        )

    def storage_signature(self) -> tuple[object, ...]:
        return (
            self.artifact_sha256,
            self.storage_encoding,
            self.learned_scalar_count,
            self.fixed_scalar_count,
            self.serialized_byte_offset,
            self.serialized_bytes,
            self.serialized_size_audited,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "component_kind": self.component_kind,
            "artifact_sha256": self.artifact_sha256,
            "storage_encoding": self.storage_encoding,
            "learned_scalar_count": self.learned_scalar_count,
            "fixed_scalar_count": self.fixed_scalar_count,
            "serialized_byte_offset": self.serialized_byte_offset,
            "serialized_bytes": self.serialized_bytes,
            "expected_macs_per_token": self.expected_macs_per_token,
            "worst_case_macs_per_token": self.worst_case_macs_per_token,
            "attention_macs_per_causal_pair": (
                self.attention_macs_per_causal_pair
            ),
            "causal_pair_contract_sha256": (
                self.causal_pair_contract_sha256
            ),
            "non_mac_ops_per_token": self.non_mac_ops_per_token,
            "serialized_size_audited": self.serialized_size_audited,
            "physical_storage_alias_id": self.physical_storage_alias_id,
            "unaccounted_or_live_fallback": (
                self.unaccounted_or_live_fallback
            ),
        }


@dataclass(frozen=True, slots=True)
class CausalPairWorkload:
    """Observed and worst-case valid pair counts for one operator contract."""

    sequence_operator_contract_sha256: str
    valid_causal_pair_count: int
    worst_case_causal_pair_count: int

    def __post_init__(self) -> None:
        _require_sha256(
            self.sequence_operator_contract_sha256,
            label="sequence_operator_contract_sha256",
        )
        _require_nonnegative_int(
            self.valid_causal_pair_count,
            label="valid_causal_pair_count",
        )
        _require_nonnegative_int(
            self.worst_case_causal_pair_count,
            label="worst_case_causal_pair_count",
        )
        if self.worst_case_causal_pair_count < self.valid_causal_pair_count:
            raise ValueError("worst-case pair count cannot be smaller")

    def state_dict(self) -> dict[str, object]:
        return {
            "sequence_operator_contract_sha256": (
                self.sequence_operator_contract_sha256
            ),
            "valid_causal_pair_count": self.valid_causal_pair_count,
            "worst_case_causal_pair_count": (
                self.worst_case_causal_pair_count
            ),
        }


@dataclass(frozen=True, slots=True)
class CausalPairMacRate:
    """Source MAC rate for one authenticated sequence-operator contract."""

    sequence_operator_contract_sha256: str
    macs_per_causal_pair: int

    def __post_init__(self) -> None:
        _require_sha256(
            self.sequence_operator_contract_sha256,
            label="sequence_operator_contract_sha256",
        )
        _require_nonnegative_int(
            self.macs_per_causal_pair,
            label="macs_per_causal_pair",
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "sequence_operator_contract_sha256": (
                self.sequence_operator_contract_sha256
            ),
            "macs_per_causal_pair": self.macs_per_causal_pair,
        }


@dataclass(frozen=True, slots=True)
class LayerAgnosticWorkloadProfile:
    """Frozen token and contract-specific pair counts for total MACs."""

    profile_id: str
    valid_token_count: int
    worst_case_valid_token_count: int
    causal_pair_workloads: tuple[CausalPairWorkload, ...]
    sequence_count: int
    row_identity_sha256: str
    sequence_mask_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_id(self.profile_id, label="profile_id")
        _require_positive_int(self.valid_token_count, label="valid_token_count")
        _require_positive_int(
            self.worst_case_valid_token_count,
            label="worst_case_valid_token_count",
        )
        if self.worst_case_valid_token_count < self.valid_token_count:
            raise ValueError("worst-case token count cannot be smaller")
        workloads = tuple(self.causal_pair_workloads)
        contracts = tuple(
            item.sequence_operator_contract_sha256 for item in workloads
        )
        if not workloads or len(contracts) != len(set(contracts)):
            raise ValueError("causal-pair workloads must be nonempty and unique")
        object.__setattr__(self, "causal_pair_workloads", workloads)
        _require_positive_int(self.sequence_count, label="sequence_count")
        _require_sha256(self.row_identity_sha256, label="row_identity_sha256")
        _require_sha256(
            self.sequence_mask_receipt_sha256,
            label="sequence_mask_receipt_sha256",
        )

    @property
    def valid_causal_pair_count(self) -> int:
        return sum(item.valid_causal_pair_count for item in self.causal_pair_workloads)

    @property
    def worst_case_causal_pair_count(self) -> int:
        return sum(
            item.worst_case_causal_pair_count
            for item in self.causal_pair_workloads
        )

    def pair_count_by_contract(self, *, worst_case: bool) -> dict[str, int]:
        return {
            item.sequence_operator_contract_sha256: (
                item.worst_case_causal_pair_count
                if worst_case
                else item.valid_causal_pair_count
            )
            for item in self.causal_pair_workloads
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "valid_token_count": self.valid_token_count,
            "valid_causal_pair_count": self.valid_causal_pair_count,
            "worst_case_valid_token_count": self.worst_case_valid_token_count,
            "worst_case_causal_pair_count": (
                self.worst_case_causal_pair_count
            ),
            "causal_pair_workloads": [
                item.state_dict()
                for item in sorted(
                    self.causal_pair_workloads,
                    key=lambda value: value.sequence_operator_contract_sha256,
                )
            ],
            "sequence_count": self.sequence_count,
            "row_identity_sha256": self.row_identity_sha256,
            "sequence_mask_receipt_sha256": (
                self.sequence_mask_receipt_sha256
            ),
        }

    @property
    def artifact_sha256(self) -> str:
        return _sha256(
            b"fisher_graph:layer-agnostic-workload-profile:v1\0",
            self.state_dict(),
        )


def deployment_component_span_receipt_sha256(
    components: Sequence[LayerAgnosticResourceComponent],
) -> str:
    """Bind the exact serialized span catalog used by a deployment artifact."""

    catalog = tuple(components)
    if not catalog:
        raise ValueError("deployment component catalog cannot be empty")
    return _sha256(
        b"fisher_graph:deployment-component-spans:v1\0",
        [
            {
                "component_id": component.component_id,
                "artifact_sha256": component.artifact_sha256,
                "serialized_byte_offset": component.serialized_byte_offset,
                "serialized_bytes": component.serialized_bytes,
                "physical_storage_alias_id": (
                    component.physical_storage_alias_id
                ),
            }
            for component in sorted(catalog, key=lambda item: item.component_id)
        ],
    )


@dataclass(frozen=True, slots=True)
class LayerAgnosticResourceLedger:
    """Fail-closed physical-storage and logical-compute accounting."""

    source_unique_learned_scalars: int
    source_serialized_bytes: int
    source_expected_macs_per_token: int
    source_causal_pair_mac_rates: tuple[CausalPairMacRate, ...]
    workload_profile: LayerAgnosticWorkloadProfile
    components: tuple[LayerAgnosticResourceComponent, ...]
    unaccounted_source_checkpoint_present: bool
    native_fallback_reachable: bool
    physical_alias_audit_passed: bool
    physical_storage_receipt_sha256: str
    zero_native_source_calls_audited: bool
    source_call_audit_sha256: str
    deployment_artifact_sha256: str
    deployment_artifact_size_bytes: int
    component_span_receipt_sha256: str
    experiment_executed: bool

    def __post_init__(self) -> None:
        _require_positive_int(
            self.source_unique_learned_scalars,
            label="source_unique_learned_scalars",
        )
        _require_positive_int(
            self.source_serialized_bytes,
            label="source_serialized_bytes",
        )
        _require_positive_int(
            self.source_expected_macs_per_token,
            label="source_expected_macs_per_token",
        )
        if not isinstance(self.workload_profile, LayerAgnosticWorkloadProfile):
            raise TypeError("workload_profile must be authenticated")
        source_rates = tuple(self.source_causal_pair_mac_rates)
        source_contracts = tuple(
            item.sequence_operator_contract_sha256 for item in source_rates
        )
        workload_contracts = tuple(
            item.sequence_operator_contract_sha256
            for item in self.workload_profile.causal_pair_workloads
        )
        if (
            not source_rates
            or len(source_contracts) != len(set(source_contracts))
            or set(source_contracts) != set(workload_contracts)
        ):
            raise ValueError(
                "source pair-MAC rates must close over workload contracts"
            )
        object.__setattr__(
            self,
            "source_causal_pair_mac_rates",
            source_rates,
        )
        components = tuple(self.components)
        if not components:
            raise ValueError("resource ledger must contain candidate components")
        object.__setattr__(self, "components", components)
        ids = tuple(component.component_id for component in components)
        if len(ids) != len(set(ids)):
            raise ValueError("resource component ids must be unique")
        aliases: dict[str, tuple[object, ...]] = {}
        for component in components:
            alias = component.physical_storage_alias_id
            if alias is None:
                continue
            signature = component.storage_signature()
            if alias in aliases and aliases[alias] != signature:
                raise ValueError("aliased components disagree on physical storage")
            aliases[alias] = signature
        for label, value in (
            (
                "unaccounted_source_checkpoint_present",
                self.unaccounted_source_checkpoint_present,
            ),
            ("native_fallback_reachable", self.native_fallback_reachable),
            ("physical_alias_audit_passed", self.physical_alias_audit_passed),
            (
                "zero_native_source_calls_audited",
                self.zero_native_source_calls_audited,
            ),
            ("experiment_executed", self.experiment_executed),
        ):
            _require_bool(value, label=label)
        _require_sha256(
            self.physical_storage_receipt_sha256,
            label="physical_storage_receipt_sha256",
        )
        _require_sha256(
            self.source_call_audit_sha256,
            label="source_call_audit_sha256",
        )
        _require_sha256(
            self.deployment_artifact_sha256,
            label="deployment_artifact_sha256",
        )
        _require_positive_int(
            self.deployment_artifact_size_bytes,
            label="deployment_artifact_size_bytes",
        )
        _require_sha256(
            self.component_span_receipt_sha256,
            label="component_span_receipt_sha256",
        )
        if self.component_span_receipt_sha256 != (
            deployment_component_span_receipt_sha256(components)
        ):
            raise ValueError("deployment component-span receipt is invalid")
        unique_components: list[LayerAgnosticResourceComponent] = []
        seen_aliases: set[str] = set()
        for component in components:
            alias = component.physical_storage_alias_id
            if alias is not None and alias in seen_aliases:
                continue
            if alias is not None:
                seen_aliases.add(alias)
            unique_components.append(component)
        spans = sorted(
            (
                component.serialized_byte_offset,
                component.serialized_bytes,
                component.component_id,
            )
            for component in unique_components
            if component.serialized_bytes > 0
        )
        cursor = 0
        for offset, length, _ in spans:
            if offset != cursor:
                raise ValueError("deployment component spans are not contiguous")
            cursor += length
        if cursor != self.deployment_artifact_size_bytes:
            raise ValueError("deployment artifact size does not close over spans")

    def _unique_storage_components(
        self,
    ) -> tuple[LayerAgnosticResourceComponent, ...]:
        result: list[LayerAgnosticResourceComponent] = []
        seen_aliases: set[str] = set()
        for component in self.components:
            alias = component.physical_storage_alias_id
            if alias is not None:
                if alias in seen_aliases:
                    continue
                seen_aliases.add(alias)
            result.append(component)
        return tuple(result)

    @property
    def candidate_unique_learned_scalars(self) -> int:
        return sum(
            component.learned_scalar_count
            for component in self._unique_storage_components()
        )

    @property
    def candidate_fixed_scalars(self) -> int:
        return sum(
            component.fixed_scalar_count
            for component in self._unique_storage_components()
        )

    @property
    def candidate_unique_stored_scalars(self) -> int:
        return (
            self.candidate_unique_learned_scalars
            + self.candidate_fixed_scalars
        )

    @property
    def candidate_serialized_bytes(self) -> int:
        return sum(
            component.serialized_bytes
            for component in self._unique_storage_components()
        )

    @property
    def candidate_expected_macs_per_token(self) -> int:
        return sum(
            component.expected_macs_per_token for component in self.components
        )

    @property
    def candidate_worst_case_macs_per_token(self) -> int:
        return sum(
            component.worst_case_macs_per_token for component in self.components
        )

    @property
    def candidate_causal_pair_mac_rates(self) -> dict[str, int]:
        rates = {
            item.sequence_operator_contract_sha256: 0
            for item in self.workload_profile.causal_pair_workloads
        }
        for component in self.components:
            if component.attention_macs_per_causal_pair == 0:
                continue
            contract = component.causal_pair_contract_sha256
            if contract not in rates:
                raise ValueError(
                    "candidate pair-MAC component is outside workload contracts"
                )
            if contract is None:
                raise AssertionError("positive pair MACs require a contract")
            rates[contract] += component.attention_macs_per_causal_pair
        return rates

    def source_causal_pair_mac_rate_map(self) -> dict[str, int]:
        return {
            item.sequence_operator_contract_sha256: item.macs_per_causal_pair
            for item in self.source_causal_pair_mac_rates
        }

    def _pair_total_macs(
        self,
        rates: Mapping[str, int],
        *,
        worst_case: bool,
    ) -> int:
        counts = self.workload_profile.pair_count_by_contract(
            worst_case=worst_case
        )
        if set(rates) != set(counts):
            raise ValueError("pair-MAC rates do not close over workload contracts")
        return sum(rates[contract] * counts[contract] for contract in counts)

    @property
    def candidate_non_mac_ops_per_token(self) -> int:
        return sum(
            component.non_mac_ops_per_token for component in self.components
        )

    @property
    def source_expected_total_macs(self) -> int:
        return (
            self.source_expected_macs_per_token
            * self.workload_profile.valid_token_count
            + self._pair_total_macs(
                self.source_causal_pair_mac_rate_map(),
                worst_case=False,
            )
        )

    @property
    def candidate_expected_total_macs(self) -> int:
        return (
            self.candidate_expected_macs_per_token
            * self.workload_profile.valid_token_count
            + self._pair_total_macs(
                self.candidate_causal_pair_mac_rates,
                worst_case=False,
            )
        )

    @property
    def candidate_worst_case_total_macs(self) -> int:
        return (
            self.candidate_worst_case_macs_per_token
            * self.workload_profile.worst_case_valid_token_count
            + self._pair_total_macs(
                self.candidate_causal_pair_mac_rates,
                worst_case=True,
            )
        )

    @property
    def learned_scalar_reduction_fraction(self) -> float:
        return 1.0 - (
            self.candidate_unique_learned_scalars
            / self.source_unique_learned_scalars
        )

    @property
    def stored_scalar_reduction_fraction(self) -> float:
        return 1.0 - (
            self.candidate_unique_stored_scalars
            / self.source_unique_learned_scalars
        )

    @property
    def serialized_byte_reduction_fraction(self) -> float:
        return 1.0 - (
            self.candidate_serialized_bytes / self.source_serialized_bytes
        )

    @property
    def expected_mac_reduction_fraction(self) -> float:
        return 1.0 - (
            self.candidate_expected_total_macs
            / self.source_expected_total_macs
        )

    @property
    def resource_accounting_eligible(self) -> bool:
        """Return generic physical-resource eligibility, not model fidelity."""

        return bool(
            self.experiment_executed
            and not self.unaccounted_source_checkpoint_present
            and not self.native_fallback_reachable
            and self.physical_alias_audit_passed
            and self.zero_native_source_calls_audited
            and not any(
                component.unaccounted_or_live_fallback
                for component in self.components
            )
            and all(
                component.serialized_size_audited
                for component in self.components
            )
            and self.stored_scalar_reduction_fraction > 0.0
            and self.serialized_byte_reduction_fraction > 0.0
        )

    @property
    def compression_credit_eligible(self) -> bool:
        """Never grant model compression credit from a generic ledger alone."""

        return False

    def state_dict(self) -> dict[str, object]:
        return {
            "source_unique_learned_scalars": (
                self.source_unique_learned_scalars
            ),
            "source_serialized_bytes": self.source_serialized_bytes,
            "source_expected_macs_per_token": (
                self.source_expected_macs_per_token
            ),
            "source_causal_pair_mac_rates": [
                item.state_dict()
                for item in sorted(
                    self.source_causal_pair_mac_rates,
                    key=lambda value: (
                        value.sequence_operator_contract_sha256
                    ),
                )
            ],
            "workload_profile": self.workload_profile.state_dict(),
            "workload_profile_sha256": self.workload_profile.artifact_sha256,
            "components": [component.state_dict() for component in self.components],
            "unaccounted_source_checkpoint_present": (
                self.unaccounted_source_checkpoint_present
            ),
            "native_fallback_reachable": self.native_fallback_reachable,
            "physical_alias_audit_passed": self.physical_alias_audit_passed,
            "physical_storage_receipt_sha256": (
                self.physical_storage_receipt_sha256
            ),
            "zero_native_source_calls_audited": (
                self.zero_native_source_calls_audited
            ),
            "source_call_audit_sha256": self.source_call_audit_sha256,
            "deployment_artifact_sha256": self.deployment_artifact_sha256,
            "deployment_artifact_size_bytes": (
                self.deployment_artifact_size_bytes
            ),
            "component_span_receipt_sha256": (
                self.component_span_receipt_sha256
            ),
            "experiment_executed": self.experiment_executed,
            "candidate_unique_learned_scalars": (
                self.candidate_unique_learned_scalars
            ),
            "candidate_fixed_scalars": self.candidate_fixed_scalars,
            "candidate_unique_stored_scalars": (
                self.candidate_unique_stored_scalars
            ),
            "candidate_serialized_bytes": self.candidate_serialized_bytes,
            "candidate_expected_macs_per_token": (
                self.candidate_expected_macs_per_token
            ),
            "candidate_worst_case_macs_per_token": (
                self.candidate_worst_case_macs_per_token
            ),
            "candidate_causal_pair_mac_rates": (
                self.candidate_causal_pair_mac_rates
            ),
            "candidate_non_mac_ops_per_token": (
                self.candidate_non_mac_ops_per_token
            ),
            "source_expected_total_macs": self.source_expected_total_macs,
            "candidate_expected_total_macs": (
                self.candidate_expected_total_macs
            ),
            "candidate_worst_case_total_macs": (
                self.candidate_worst_case_total_macs
            ),
            "learned_scalar_reduction_fraction": (
                self.learned_scalar_reduction_fraction
            ),
            "stored_scalar_reduction_fraction": (
                self.stored_scalar_reduction_fraction
            ),
            "serialized_byte_reduction_fraction": (
                self.serialized_byte_reduction_fraction
            ),
            "expected_mac_reduction_fraction": (
                self.expected_mac_reduction_fraction
            ),
            "resource_accounting_eligible": self.resource_accounting_eligible,
            "compression_credit_eligible": self.compression_credit_eligible,
            "model_specific_bridge_and_fidelity_guard_required": True,
        }


@dataclass(frozen=True, slots=True)
class LayerAgnosticModalGraphPlan:
    """Authenticated setup/fitted topology for a whole-model modal graph."""

    lineage: LayerAgnosticPlanLineage
    source_modes: tuple[SourceModeProvenance, ...]
    generator_families: tuple[GeneratorFamily, ...]
    invocations: tuple[GeneratorInvocation, ...]
    transports: tuple[GeneratorTransport, ...]
    chart_atlas: ModalChartAtlas
    mutation_assemblies: tuple[ResidualMutationAssembly, ...]
    lexical_boundary: SharedLexicalBoundaryPlan
    resource_ledger: LayerAgnosticResourceLedger
    schema: str = LAYER_AGNOSTIC_MODAL_GRAPH_SCHEMA
    format_version: int = LAYER_AGNOSTIC_MODAL_GRAPH_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != LAYER_AGNOSTIC_MODAL_GRAPH_SCHEMA:
            raise ValueError("layer-agnostic graph schema mismatch")
        if self.format_version != LAYER_AGNOSTIC_MODAL_GRAPH_FORMAT_VERSION:
            raise ValueError("layer-agnostic graph format mismatch")
        for field_name in (
            "source_modes",
            "generator_families",
            "invocations",
            "transports",
            "mutation_assemblies",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        self._validate_catalogs()

    def _validate_catalogs(self) -> None:
        modes = {mode.mode_id: mode for mode in self.source_modes}
        families = {
            family.family_id: family for family in self.generator_families
        }
        invocations = {
            invocation.invocation_id: invocation
            for invocation in self.invocations
        }
        if not modes or len(modes) != len(self.source_modes):
            raise ValueError("source mode ids must be nonempty and unique")
        if not families or len(families) != len(self.generator_families):
            raise ValueError("generator family ids must be nonempty and unique")
        if not invocations or len(invocations) != len(self.invocations):
            raise ValueError("invocation ids must be nonempty and unique")
        transport_ids = tuple(edge.transport_id for edge in self.transports)
        if len(transport_ids) != len(set(transport_ids)):
            raise ValueError("transport ids must be unique")

        family_members: list[str] = []
        for family in self.generator_families:
            unknown = set(family.member_mode_ids) - set(modes)
            if unknown:
                raise ValueError("generator family references unknown modes")
            family_members.extend(family.member_mode_ids)
        if sorted(family_members) != sorted(modes):
            raise ValueError("every source mode must belong to exactly one family")

        invoked_modes: list[str] = []
        adapter_ids: list[str] = []
        for invocation in self.invocations:
            family = families.get(invocation.family_id)
            if family is None:
                raise ValueError("invocation references an unknown family")
            if not set(invocation.source_mode_ids).issubset(
                family.member_mode_ids
            ):
                raise ValueError("invocation modes do not belong to its family")
            if (
                invocation.runtime_signature_sha256
                not in family.compatible_runtime_signature_sha256s
            ):
                raise ValueError("invocation runtime is incompatible with family")
            direct_chart_compatible = all(
                region.coordinate_rank == family.coordinate_rank
                and invocation.runtime_signature_sha256
                in region.compatible_runtime_signature_sha256s
                for region in self.chart_atlas.regions
            )
            adapter = invocation.coordinate_adapter
            if adapter is None and not direct_chart_compatible:
                raise ValueError(
                    "every routable chart region must directly match the "
                    "family coordinate rank/runtime or the invocation must "
                    "provide a coordinate adapter contract"
                )
            if adapter is not None:
                region = next(
                    (
                        candidate
                        for candidate in self.chart_atlas.regions
                        if candidate.region_id == adapter.source_region_id
                    ),
                    None,
                )
                if region is None:
                    raise ValueError(
                        "coordinate adapter references an unknown chart region"
                    )
                if (
                    adapter.source_coordinate_rank != region.coordinate_rank
                    or adapter.source_runtime_signature_sha256
                    not in region.compatible_runtime_signature_sha256s
                ):
                    raise ValueError(
                        "coordinate adapter source does not match its chart region"
                    )
                if not all(
                    candidate.coordinate_rank
                    == adapter.source_coordinate_rank
                    and adapter.source_runtime_signature_sha256
                    in candidate.compatible_runtime_signature_sha256s
                    for candidate in self.chart_atlas.regions
                ):
                    raise ValueError(
                        "every routable chart region must match the coordinate "
                        "adapter source rank/runtime"
                    )
                if (
                    adapter.source_runtime_signature_sha256
                    != invocation.runtime_signature_sha256
                ):
                    raise ValueError(
                        "coordinate adapter source runtime does not match invocation"
                    )
                if (
                    adapter.target_family_id != family.family_id
                    or adapter.target_coordinate_rank != family.coordinate_rank
                    or adapter.target_runtime_signature_sha256
                    != invocation.runtime_signature_sha256
                ):
                    raise ValueError(
                        "coordinate adapter target does not match invocation family"
                    )
                adapter_ids.append(adapter.adapter_id)
            for mode_id in invocation.source_mode_ids:
                if modes[mode_id].mutation_index != invocation.mutation_index:
                    raise ValueError("invocation and source-mode order disagree")
                if (
                    modes[mode_id].runtime_signature_sha256
                    != invocation.runtime_signature_sha256
                ):
                    raise ValueError("source mode and invocation runtime disagree")
            invoked_modes.extend(invocation.source_mode_ids)
        if sorted(invoked_modes) != sorted(modes):
            raise ValueError("every source mode must be invoked exactly once")
        if len(adapter_ids) != len(set(adapter_ids)):
            raise ValueError("coordinate adapter ids must be unique")

        edge_pairs: set[tuple[str, str]] = set()
        for edge in self.transports:
            source = invocations.get(edge.source_invocation_id)
            target = invocations.get(edge.target_invocation_id)
            if source is None or target is None:
                raise ValueError("transport references an unknown invocation")
            if source.invocation_id == target.invocation_id:
                raise ValueError("transport self edges are forbidden")
            if source.mutation_index >= target.mutation_index:
                raise ValueError("transport must be strictly forward in time")
            pair = (source.invocation_id, target.invocation_id)
            if pair in edge_pairs:
                raise ValueError("duplicate invocation transports are forbidden")
            edge_pairs.add(pair)
            if (
                edge.token_relation == "causal_prefix"
                and target.token_dependency != "causal_prefix_mixer"
            ):
                raise ValueError(
                    "causal-prefix transport requires a causal-prefix target"
                )

        if (
            self.lexical_boundary.ingress_invocation_id not in invocations
            or self.lexical_boundary.egress_invocation_id not in invocations
            or self.lexical_boundary.terminal_norm_invocation_id not in invocations
        ):
            raise ValueError("lexical boundary references unknown invocations")
        ingress = invocations[self.lexical_boundary.ingress_invocation_id]
        egress = invocations[self.lexical_boundary.egress_invocation_id]
        terminal = invocations[
            self.lexical_boundary.terminal_norm_invocation_id
        ]
        if ingress.token_dependency != "vocabulary_boundary":
            raise ValueError("lexical ingress must be a vocabulary boundary")
        if egress.token_dependency != "vocabulary_boundary":
            raise ValueError("lexical egress must be a vocabulary boundary")
        if terminal.operator_role != "terminal_rmsnorm":
            raise ValueError("terminal RMSNorm invocation is not explicit")
        if terminal.token_dependency != "token_local":
            raise ValueError("terminal RMSNorm must be token-local")
        if not (
            ingress.mutation_index
            < terminal.mutation_index
            < egress.mutation_index
        ):
            raise ValueError("lexical and terminal boundary order is invalid")
        if (
            ingress.family_id != self.lexical_boundary.generator_family_id
            or egress.family_id != self.lexical_boundary.generator_family_id
        ):
            raise ValueError("lexical endpoints must share one generator family")

        self._validate_mutation_assemblies(invocations)
        self._validate_resource_closure(families, invocations)

    def _validate_mutation_assemblies(
        self,
        invocations: Mapping[str, GeneratorInvocation],
    ) -> None:
        assemblies = tuple(
            sorted(
                self.mutation_assemblies,
                key=lambda item: item.assembly_index,
            )
        )
        if len(assemblies) < 4:
            raise ValueError("whole-model carrier chain is incomplete")
        if tuple(item.assembly_index for item in assemblies) != tuple(
            range(len(assemblies))
        ):
            raise ValueError("mutation assembly indexes must be contiguous")
        if assemblies[0].assembly_role != "lexical_ingress":
            raise ValueError("carrier chain must begin at lexical ingress")
        if assemblies[-2].assembly_role != "terminal_rmsnorm":
            raise ValueError("carrier chain must include terminal RMSNorm")
        if assemblies[-1].assembly_role != "lexical_egress":
            raise ValueError("carrier chain must end at lexical egress")
        for previous, current in zip(assemblies, assemblies[1:]):
            if previous.output_carrier_id != current.input_carrier_id:
                raise ValueError("residual carrier boundary chain is not closed")

        assigned_invocations: list[str] = []
        for assembly in assemblies:
            for invocation_id in assembly.invocation_ids:
                invocation = invocations.get(invocation_id)
                if invocation is None:
                    raise ValueError("assembly references an unknown invocation")
                if invocation.mutation_index != assembly.assembly_index:
                    raise ValueError("assembly and invocation order disagree")
                if invocation.operator_role != assembly.assembly_role:
                    raise ValueError("assembly and invocation roles disagree")
                if invocation.input_boundary_id != assembly.input_carrier_id:
                    raise ValueError("invocation reads the wrong carrier boundary")
                if invocation.output_boundary_id != assembly.output_carrier_id:
                    raise ValueError("invocation writes the wrong carrier boundary")
                if invocation.source_mutation_id != assembly.source_mutation_id:
                    raise ValueError("source mutation lineage is inconsistent")
                assigned_invocations.append(invocation_id)
        if sorted(assigned_invocations) != sorted(invocations):
            raise ValueError("every invocation must belong to one assembly")
        if (
            self.lexical_boundary.ingress_invocation_id
            not in assemblies[0].invocation_ids
            or self.lexical_boundary.terminal_norm_invocation_id
            not in assemblies[-2].invocation_ids
            or self.lexical_boundary.egress_invocation_id
            not in assemblies[-1].invocation_ids
        ):
            raise ValueError("lexical boundary does not close the carrier chain")

    def _validate_resource_closure(
        self,
        families: Mapping[str, GeneratorFamily],
        invocations: Mapping[str, GeneratorInvocation],
    ) -> None:
        components = {
            component.component_id: component
            for component in self.resource_ledger.components
        }
        required_ids: set[str] = set(self.chart_atlas.resource_component_ids)
        required_ids.update(
            family.resource_component_id for family in self.generator_families
        )
        for invocation in self.invocations:
            required_ids.update(invocation.resource_component_ids)
        required_ids.update(
            edge.resource_component_id for edge in self.transports
        )
        required_ids.update(
            {
                self.lexical_boundary.factor_resource_component_id,
                self.lexical_boundary.ingress_map_resource_component_id,
                self.lexical_boundary.egress_map_resource_component_id,
                self.lexical_boundary.terminal_norm_resource_component_id,
            }
        )
        family_resource_ids = tuple(
            family.resource_component_id for family in self.generator_families
        )
        if len(family_resource_ids) != len(set(family_resource_ids)):
            raise ValueError("generator families cannot share resource entries")
        invocation_resource_ids = tuple(
            component_id
            for invocation in self.invocations
            for component_id in invocation.resource_component_ids
        )
        if len(invocation_resource_ids) != len(set(invocation_resource_ids)):
            raise ValueError(
                "invocation logical resource entries require unique uses"
            )
        transport_resource_ids = tuple(
            edge.resource_component_id for edge in self.transports
        )
        if len(transport_resource_ids) != len(set(transport_resource_ids)):
            raise ValueError("transport logical resource entries must be unique")
        logical_resource_ids = (
            set(invocation_resource_ids)
            | set(transport_resource_ids)
            | {
                self.chart_atlas.router_resource_component_id,
                self.chart_atlas.local_correction_resource_component_id,
            }
        )
        if len(logical_resource_ids) != (
            len(invocation_resource_ids)
            + len(transport_resource_ids)
            + 2
        ):
            raise ValueError("logical compute resource ownership overlaps")
        if set(components) != required_ids:
            missing = sorted(required_ids - set(components))
            unbound = sorted(set(components) - required_ids)
            raise ValueError(
                "resource ledger does not close over topology: "
                f"missing={missing}, unbound={unbound}"
            )
        for component in components.values():
            carries_logical_work = bool(
                component.expected_macs_per_token
                or component.worst_case_macs_per_token
                or component.attention_macs_per_causal_pair
                or component.non_mac_ops_per_token
            )
            if carries_logical_work and (
                component.component_id not in logical_resource_ids
            ):
                raise ValueError("logical compute ownership is not closed")
        for family in families.values():
            component = components[family.resource_component_id]
            if component.artifact_sha256 != family.artifact_sha256:
                raise ValueError("generator family resource artifact mismatch")
            if (
                component.learned_scalar_count + component.fixed_scalar_count
                != family.stored_scalar_count
            ):
                raise ValueError("generator family scalar receipt mismatch")
        atlas_component = components[
            self.chart_atlas.artifact_resource_component_id
        ]
        if atlas_component.artifact_sha256 != self.chart_atlas.artifact_sha256:
            raise ValueError("chart atlas resource artifact mismatch")
        if (
            atlas_component.learned_scalar_count
            + atlas_component.fixed_scalar_count
            != self.chart_atlas.region_scalar_count
        ):
            raise ValueError("chart region scalar receipt mismatch")
        router_component = components[
            self.chart_atlas.router_resource_component_id
        ]
        if (
            router_component.artifact_sha256 != self.chart_atlas.router_sha256
            or router_component.learned_scalar_count
            + router_component.fixed_scalar_count
            != self.chart_atlas.router_scalar_count
            or router_component.expected_macs_per_token
            != self.chart_atlas.routing_macs_per_valid_token
            or router_component.non_mac_ops_per_token
            != self.chart_atlas.routing_non_mac_ops_per_valid_token
        ):
            raise ValueError("chart router resource receipt mismatch")
        correction_component = components[
            self.chart_atlas.local_correction_resource_component_id
        ]
        if (
            correction_component.component_kind != "chart_local_correction"
            or correction_component.artifact_sha256
            != self.chart_atlas.local_correction_artifact_sha256
        ):
            raise ValueError("chart local-correction resource artifact mismatch")
        if (
            correction_component.learned_scalar_count
            + correction_component.fixed_scalar_count
            != self.chart_atlas.local_correction_scalar_count
        ):
            raise ValueError("chart local-correction scalar receipt mismatch")
        if (
            correction_component.expected_macs_per_token
            != self.chart_atlas.local_correction_macs_per_valid_token
            or correction_component.worst_case_macs_per_token
            != self.chart_atlas.local_correction_macs_per_valid_token
            or correction_component.attention_macs_per_causal_pair != 0
            or correction_component.non_mac_ops_per_token != 0
        ):
            raise ValueError("chart local-correction MAC receipt mismatch")
        for invocation in self.invocations:
            invocation_components = [
                components[component_id]
                for component_id in invocation.resource_component_ids
            ]
            adapter = invocation.coordinate_adapter
            if adapter is not None:
                adapter_component = components[adapter.resource_component_id]
                if (
                    adapter_component.component_kind != "coordinate_adapter"
                    or adapter_component.artifact_sha256
                    != adapter.adapter_artifact_sha256
                ):
                    raise ValueError(
                        "coordinate adapter resource artifact mismatch"
                    )
                if (
                    adapter_component.learned_scalar_count
                    + adapter_component.fixed_scalar_count
                    != adapter.stored_scalar_count
                    or adapter_component.serialized_bytes <= 0
                ):
                    raise ValueError(
                        "coordinate adapter scalar receipt mismatch"
                    )
                if (
                    adapter_component.expected_macs_per_token
                    != adapter.macs_per_valid_token
                    or adapter_component.worst_case_macs_per_token
                    != adapter.macs_per_valid_token
                    or adapter_component.attention_macs_per_causal_pair != 0
                    or adapter_component.non_mac_ops_per_token != 0
                ):
                    raise ValueError(
                        "coordinate adapter MAC receipt mismatch"
                    )
            if invocation.sequence_operator_contract_sha256 not in {
                component.artifact_sha256
                for component in invocation_components
            }:
                raise ValueError("invocation operator artifact is not bound")
            if sum(
                component.learned_scalar_count + component.fixed_scalar_count
                for component in invocation_components
            ) != invocation.local_adapter_scalar_count:
                raise ValueError("invocation adapter scalar receipt mismatch")
            if sum(
                component.expected_macs_per_token
                for component in invocation_components
            ) != invocation.expected_macs_per_valid_token:
                raise ValueError("invocation token-MAC receipt mismatch")
            if sum(
                component.attention_macs_per_causal_pair
                for component in invocation_components
            ) != invocation.attention_macs_per_causal_pair:
                raise ValueError("invocation pair-MAC receipt mismatch")
            if any(
                component.attention_macs_per_causal_pair > 0
                and component.causal_pair_contract_sha256
                != invocation.sequence_operator_contract_sha256
                for component in invocation_components
            ):
                raise ValueError("invocation pair-MAC contract mismatch")
            if sum(
                component.non_mac_ops_per_token
                for component in invocation_components
            ) != invocation.non_mac_ops_per_valid_token:
                raise ValueError("invocation non-MAC receipt mismatch")
        for edge in self.transports:
            component = components[edge.resource_component_id]
            target = invocations[edge.target_invocation_id]
            if component.artifact_sha256 != edge.coefficient_sha256:
                raise ValueError("transport resource artifact mismatch")
            if (
                component.learned_scalar_count
                + component.fixed_scalar_count
                != edge.stored_scalar_count
            ):
                raise ValueError("transport stored-scalar receipt mismatch")
            if component.expected_macs_per_token != edge.macs_per_active_token:
                raise ValueError("transport token-MAC receipt mismatch")
            if (
                component.attention_macs_per_causal_pair
                != edge.attention_macs_per_causal_pair
            ):
                raise ValueError("transport pair-MAC receipt mismatch")
            if (
                component.attention_macs_per_causal_pair > 0
                and component.causal_pair_contract_sha256
                != target.sequence_operator_contract_sha256
            ):
                raise ValueError("transport pair-MAC contract mismatch")
        lexical_component = components[
            self.lexical_boundary.factor_resource_component_id
        ]
        if (
            lexical_component.artifact_sha256
            != self.lexical_boundary.factor_artifact_sha256
            or lexical_component.learned_scalar_count
            != self.lexical_boundary.factorized_scalar_count
            or lexical_component.fixed_scalar_count != 0
            or lexical_component.physical_storage_alias_id
            != self.lexical_boundary.shared_factor_id
        ):
            raise ValueError("shared lexical factor resource receipt mismatch")
        aliased_components = [
            component
            for component in components.values()
            if component.physical_storage_alias_id is not None
        ]
        if (
            len(aliased_components) != 1
            or aliased_components[0].component_id
            != self.lexical_boundary.factor_resource_component_id
            or aliased_components[0].physical_storage_alias_id
            != self.lexical_boundary.shared_factor_id
        ):
            raise ValueError(
                "v1 permits physical storage deduplication only for the "
                "single shared lexical factor"
            )
        lexical_count_expectations = (
            (
                self.lexical_boundary.ingress_map_resource_component_id,
                self.lexical_boundary.ingress_map_scalar_count,
                self.lexical_boundary.ingress_map_artifact_sha256,
            ),
            (
                self.lexical_boundary.egress_map_resource_component_id,
                self.lexical_boundary.egress_map_scalar_count,
                self.lexical_boundary.egress_map_artifact_sha256,
            ),
            (
                self.lexical_boundary.terminal_norm_resource_component_id,
                self.lexical_boundary.terminal_norm_scalar_count,
                self.lexical_boundary.terminal_norm_artifact_sha256,
            ),
        )
        for component_id, expected_count, expected_artifact in (
            lexical_count_expectations
        ):
            component = components[component_id]
            if (
                component.learned_scalar_count + component.fixed_scalar_count
                != expected_count
                or component.artifact_sha256 != expected_artifact
            ):
                raise ValueError("lexical boundary scalar receipt mismatch")
        if not self.resource_ledger.physical_alias_audit_passed:
            raise ValueError("physical lexical storage audit did not pass")
        if (
            self.resource_ledger.physical_storage_receipt_sha256
            != self.lexical_boundary.physical_storage_receipt_sha256
        ):
            raise ValueError("lexical storage receipt lineage mismatch")

    @property
    def communities_spanning_source_layers(self) -> tuple[str, ...]:
        by_mode = {mode.mode_id: mode for mode in self.source_modes}
        spanning: list[str] = []
        for family in self.generator_families:
            layers = {
                by_mode[mode_id].source_layer_ordinal
                for mode_id in family.member_mode_ids
                if by_mode[mode_id].source_layer_ordinal is not None
            }
            if len(layers) > 1:
                spanning.append(family.family_id)
        return tuple(sorted(spanning))

    def execution_state(self) -> dict[str, object]:
        """Return numerical/execution identity without native provenance labels."""

        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "lineage": self.lineage.state_dict(),
            "source_mode_discovery": [
                mode.discovery_state()
                for mode in sorted(self.source_modes, key=lambda item: item.mode_id)
            ],
            "generator_families": [
                family.state_dict()
                for family in sorted(
                    self.generator_families,
                    key=lambda item: item.family_id,
                )
            ],
            "invocations": [
                invocation.state_dict()
                for invocation in sorted(
                    self.invocations,
                    key=lambda item: (item.mutation_index, item.invocation_id),
                )
            ],
            "transports": [
                edge.state_dict()
                for edge in sorted(
                    self.transports,
                    key=lambda item: item.transport_id,
                )
            ],
            "chart_atlas": self.chart_atlas.state_dict(),
            "mutation_assemblies": [
                assembly.state_dict()
                for assembly in sorted(
                    self.mutation_assemblies,
                    key=lambda item: item.assembly_index,
                )
            ],
            "lexical_boundary": self.lexical_boundary.state_dict(),
            "resource_ledger": self.resource_ledger.state_dict(),
        }

    @property
    def execution_sha256(self) -> str:
        return _sha256(_EXECUTION_HASH_DOMAIN, self.execution_state())

    def _artifact_state(self) -> dict[str, object]:
        state = {
            **self.execution_state(),
            "source_mode_provenance": [
                mode.state_dict()
                for mode in sorted(self.source_modes, key=lambda item: item.mode_id)
            ],
            "communities_spanning_source_layers": list(
                self.communities_spanning_source_layers
            ),
            "execution_sha256": self.execution_sha256,
        }
        return state

    @property
    def artifact_sha256(self) -> str:
        return _sha256(_ARTIFACT_HASH_DOMAIN, self._artifact_state())

    def state_dict(self) -> dict[str, object]:
        state = self._artifact_state()
        state["artifact_sha256"] = self.artifact_sha256
        return state


def authenticate_layer_agnostic_modal_graph_state(
    plan: LayerAgnosticModalGraphPlan,
    value: Mapping[str, object],
) -> dict[str, object]:
    """Strictly compare a serialized fitted state with a validated live plan."""

    if not isinstance(plan, LayerAgnosticModalGraphPlan):
        raise TypeError("plan must be a validated layer-agnostic graph")
    if not isinstance(value, Mapping):
        raise TypeError("serialized plan state must be a mapping")
    observed = json.loads(_canonical_json_bytes(dict(value)))
    expected = json.loads(_canonical_json_bytes(plan.state_dict()))
    if observed != expected:
        raise ValueError("serialized layer-agnostic graph state is not authentic")
    return observed


def validate_causal_invocation_graph(
    invocations: Sequence[GeneratorInvocation],
    transports: Sequence[GeneratorTransport],
) -> None:
    """Validate the causal part of a plan without requiring fitted artifacts."""

    catalog = {item.invocation_id: item for item in invocations}
    if not catalog or len(catalog) != len(tuple(invocations)):
        raise ValueError("invocation ids must be nonempty and unique")
    transport_ids = tuple(edge.transport_id for edge in transports)
    if len(transport_ids) != len(set(transport_ids)):
        raise ValueError("transport ids must be unique")
    seen_pairs: set[tuple[str, str]] = set()
    for edge in transports:
        source = catalog.get(edge.source_invocation_id)
        target = catalog.get(edge.target_invocation_id)
        if source is None or target is None:
            raise ValueError("transport references an unknown invocation")
        pair = (source.invocation_id, target.invocation_id)
        if pair in seen_pairs:
            raise ValueError("duplicate invocation transports are forbidden")
        seen_pairs.add(pair)
        if source.invocation_id == target.invocation_id:
            raise ValueError("transport self edges are forbidden")
        if source.mutation_index >= target.mutation_index:
            raise ValueError("transport must be strictly forward in time")
        if (
            edge.token_relation == "causal_prefix"
            and target.token_dependency != "causal_prefix_mixer"
        ):
            raise ValueError(
                "causal-prefix transport requires a causal-prefix target"
            )


@dataclass(frozen=True, slots=True)
class PhysicalStorageReceipt:
    """Runtime-derived storage span used to authenticate lexical tying."""

    storage_handle_sha256: str
    byte_offset: int
    byte_length: int
    dtype: str
    device: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.storage_handle_sha256,
            label="storage_handle_sha256",
        )
        _require_nonnegative_int(self.byte_offset, label="byte_offset")
        _require_positive_int(self.byte_length, label="byte_length")
        _require_id(self.dtype, label="dtype")
        _require_id(self.device, label="device")

    def state_dict(self) -> dict[str, object]:
        return {
            "storage_handle_sha256": self.storage_handle_sha256,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
            "dtype": self.dtype,
            "device": self.device,
        }

    @property
    def receipt_sha256(self) -> str:
        return _sha256(
            b"fisher_graph:physical-storage-receipt:v1\0",
            self.state_dict(),
        )


def require_shared_physical_storage(
    ingress_receipt: PhysicalStorageReceipt,
    egress_receipt: PhysicalStorageReceipt,
) -> str:
    """Require the exact same runtime storage span, not Python identity."""

    if not isinstance(ingress_receipt, PhysicalStorageReceipt) or not isinstance(
        egress_receipt,
        PhysicalStorageReceipt,
    ):
        raise TypeError("physical storage audit requires storage receipts")
    if ingress_receipt.state_dict() != egress_receipt.state_dict():
        raise ValueError(
            "lexical ingress and egress must share one physical storage span"
        )
    return ingress_receipt.receipt_sha256


__all__ = [
    "CausalPairMacRate",
    "CausalPairWorkload",
    "CoordinateAdapterContract",
    "GeneratorFamily",
    "GeneratorInvocation",
    "GeneratorTransport",
    "LAYER_AGNOSTIC_MODAL_GRAPH_FORMAT_VERSION",
    "LAYER_AGNOSTIC_MODAL_GRAPH_SCHEMA",
    "LayerAgnosticModalGraphPlan",
    "LayerAgnosticPlanLineage",
    "LayerAgnosticResourceComponent",
    "LayerAgnosticResourceLedger",
    "LayerAgnosticWorkloadProfile",
    "ModalChartAtlas",
    "ModalChartRegion",
    "PhysicalStorageReceipt",
    "ResidualMutationAssembly",
    "SharedLexicalBoundaryPlan",
    "SourceModeProvenance",
    "authenticate_layer_agnostic_modal_graph_state",
    "deployment_component_span_receipt_sha256",
    "require_shared_physical_storage",
    "validate_causal_invocation_graph",
]
