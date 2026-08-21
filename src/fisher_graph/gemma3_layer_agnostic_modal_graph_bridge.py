"""Bind a fitted generic modal graph to the frozen Gemma whole-model protocol.

The generic graph vocabulary intentionally supports small synthetic models and
future architectures.  It therefore cannot, by itself, prove that a fitted
artifact covers Gemma's exact 39-event ingress/body/norm/egress schedule.  This
module is the model-specific bridge: it authenticates the preregistration,
selection identity, inspectable resource-point fields, and every source
mutation occurrence without loading a model or prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

from .gemma3_layer_agnostic_whole_model_compression_protocol import (
    PINNED_FINAL_RMSNORM_PARAMETER_COUNT,
    PINNED_HIDDEN_WIDTH,
    PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
    build_layer_agnostic_whole_model_compression_protocol,
    validate_layer_agnostic_whole_model_compression_protocol,
)
from .layer_agnostic_modal_graph import LayerAgnosticModalGraphPlan


GEMMA3_LAYER_AGNOSTIC_PLAN_BRIDGE_SCHEMA = (
    "fisher_graph.gemma3_layer_agnostic_modal_graph.bridge_receipt.v1"
)
GEMMA3_LAYER_AGNOSTIC_PRIMARY_ARM_ID = (
    "unrestricted_layer_agnostic_primary"
)
GEMMA3_LAYER_AGNOSTIC_REQUIRED_ASSEMBLY_COUNT = 39

_BRIDGE_HASH_DOMAIN = b"fisher_graph:gemma3-layer-agnostic-plan-bridge:v1\0"
_ZERO_SHA256 = "0" * 64


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _semantic_sha256(label: str, value: object) -> str:
    return hashlib.sha256(
        _BRIDGE_HASH_DOMAIN
        + label.encode("ascii")
        + b"\0"
        + _canonical_json_bytes(value)
    ).hexdigest()


def _require_sha256(value: str, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _validated_protocol(
    protocol: Mapping[str, object] | None,
) -> dict[str, object]:
    value = (
        build_layer_agnostic_whole_model_compression_protocol()
        if protocol is None
        else protocol
    )
    return validate_layer_agnostic_whole_model_compression_protocol(value)


def _resource_point(
    protocol: Mapping[str, object],
    resource_point_id: str,
) -> dict[str, object]:
    ladders = protocol["frozen_ladders"]
    if not isinstance(ladders, Mapping):
        raise ValueError("protocol frozen ladders are invalid")
    points = ladders["ordered_resource_points"]
    if not isinstance(points, list):
        raise ValueError("protocol resource points are invalid")
    matches = [
        point
        for point in points
        if isinstance(point, Mapping)
        and point.get("resource_point_id") == resource_point_id
    ]
    if len(matches) != 1:
        raise ValueError("plan resource point is not frozen by the protocol")
    return dict(matches[0])


def gemma3_layer_agnostic_resource_point_sha256(
    resource_point_id: str,
    protocol: Mapping[str, object] | None = None,
) -> str:
    frozen = _validated_protocol(protocol)
    return _semantic_sha256(
        "resource-point",
        _resource_point(frozen, resource_point_id),
    )


def gemma3_layer_agnostic_fit_configuration_sha256(
    protocol: Mapping[str, object] | None = None,
) -> str:
    frozen = _validated_protocol(protocol)
    return _semantic_sha256(
        "fit-configuration",
        {
            "arm_id": GEMMA3_LAYER_AGNOSTIC_PRIMARY_ARM_ID,
            "frozen_numerics": frozen["frozen_numerics"],
            "frozen_fit_procedure": frozen["frozen_fit_procedure"],
            "fit_and_validation_firewall": frozen[
                "fit_and_validation_firewall"
            ],
        },
    )


def gemma3_layer_agnostic_input_contract_sha256(
    protocol: Mapping[str, object] | None = None,
    *,
    input_mode: str = "input_ids",
) -> str:
    if input_mode != "input_ids":
        raise ValueError("Gemma bridge v1 supports input_ids only")
    frozen = _validated_protocol(protocol)
    return _semantic_sha256(
        "input-contract",
        {
            "input_mode": input_mode,
            "inputs_embeds_supported": False,
            "pinned_source": frozen["pinned_source"],
            "whole_model_boundary_graph": frozen[
                "whole_model_boundary_graph"
            ],
            "shared_lexical_boundary": frozen["shared_lexical_boundary"],
            "causal_and_runtime_invariants": frozen[
                "causal_and_runtime_invariants"
            ],
        },
    )


def _expected_schedule(
    protocol: Mapping[str, object],
) -> tuple[tuple[int, str, str, str, str, str | None], ...]:
    graph = protocol["whole_model_boundary_graph"]
    if not isinstance(graph, Mapping):
        raise ValueError("protocol boundary graph is invalid")
    body = graph["mutation_invocations"]
    if not isinstance(body, list) or len(body) != 36:
        raise ValueError("protocol body schedule is invalid")
    schedule: list[tuple[int, str, str, str, str, str | None]] = [
        (
            0,
            "lexical.ingress",
            "lexical_ingress",
            "source.lexical.ingress",
            "vocabulary_boundary",
            None,
        )
    ]
    for occurrence in body:
        if not isinstance(occurrence, Mapping):
            raise ValueError("protocol mutation occurrence is invalid")
        provenance = occurrence["provenance_only"]
        if not isinstance(provenance, Mapping):
            raise ValueError("protocol mutation provenance is invalid")
        operator_contract = occurrence["operator_contract"]
        role = (
            "attention_mutation"
            if operator_contract == "causal_prefix_mixer"
            else "feed_forward_mutation"
        )
        schedule.append(
            (
                int(occurrence["mutation_index"]),
                str(occurrence["invocation_id"]),
                role,
                str(provenance["source_mutation_id"]),
                str(operator_contract),
                (
                    None
                    if occurrence["attention_contract"] is None
                    else str(occurrence["attention_contract"])
                ),
            )
        )
    schedule.extend(
        (
            (
                37,
                "terminal.rmsnorm",
                "terminal_rmsnorm",
                "source.terminal_rmsnorm",
                "token_local",
                None,
            ),
            (
                38,
                "lexical.egress",
                "lexical_egress",
                "source.lexical.egress",
                "vocabulary_boundary",
                None,
            ),
        )
    )
    return tuple(schedule)


def _schedule_row(
    protocol: Mapping[str, object],
    mutation_index: int,
) -> tuple[int, str, str, str, str, str | None]:
    matches = [
        row for row in _expected_schedule(protocol) if row[0] == mutation_index
    ]
    if len(matches) != 1:
        raise ValueError("mutation index is outside the frozen Gemma schedule")
    return matches[0]


def gemma3_layer_agnostic_sequence_operator_contract_sha256(
    mutation_index: int,
    protocol: Mapping[str, object] | None = None,
) -> str:
    frozen = _validated_protocol(protocol)
    (
        index,
        invocation_id,
        role,
        source_mutation_id,
        dependency,
        attention_kind,
    ) = _schedule_row(frozen, mutation_index)
    graph = frozen["whole_model_boundary_graph"]
    if not isinstance(graph, Mapping):
        raise ValueError("protocol boundary graph is invalid")
    attention_contracts = graph["attention_runtime_contracts"]
    if not isinstance(attention_contracts, Mapping):
        raise ValueError("protocol attention runtime contracts are invalid")
    attention_runtime: object = None
    if attention_kind is not None:
        attention_runtime = {
            "shared": attention_contracts["shared"],
            "kind": attention_kind,
            "kind_contract": attention_contracts[attention_kind],
        }
    return _semantic_sha256(
        "sequence-operator-contract",
        {
            "model_id": frozen["pinned_source"]["model_id"],
            "model_revision": frozen["pinned_source"]["model_revision"],
            "runtime": frozen["pinned_source"]["runtime"],
            "mutation_index": index,
            "invocation_id": invocation_id,
            "operator_role": role,
            "source_mutation_id": source_mutation_id,
            "token_dependency": dependency,
            "attention_runtime": attention_runtime,
        },
    )


def gemma3_layer_agnostic_runtime_signature_sha256(
    mutation_index: int,
    protocol: Mapping[str, object] | None = None,
) -> str:
    frozen = _validated_protocol(protocol)
    row = _schedule_row(frozen, mutation_index)
    return _semantic_sha256(
        "invocation-runtime-signature",
        {
            "mutation_index": mutation_index,
            "operator_role": row[2],
            "token_dependency": row[4],
            "hidden_width": frozen["pinned_source"]["hidden_width"],
            "mlp_width": frozen["pinned_source"]["mlp_width"],
            "vocabulary_size": frozen["pinned_source"]["vocabulary_size"],
            "sequence_operator_contract_sha256": (
                gemma3_layer_agnostic_sequence_operator_contract_sha256(
                    mutation_index,
                    frozen,
                )
            ),
        },
    )


def gemma3_layer_agnostic_source_compute_receipt_sha256(
    *,
    source_expected_macs_per_token: int,
    source_causal_pair_mac_rates: Mapping[str, int],
    workload_profile_sha256: str,
) -> str:
    if (
        type(source_expected_macs_per_token) is not int
        or source_expected_macs_per_token <= 0
    ):
        raise ValueError("source expected MACs per token must be positive")
    _require_sha256(
        workload_profile_sha256,
        label="workload_profile_sha256",
    )
    rates = dict(source_causal_pair_mac_rates)
    if not rates or any(
        not isinstance(contract, str)
        or type(rate) is not int
        or rate <= 0
        for contract, rate in rates.items()
    ):
        raise ValueError("source causal-pair MAC rates must be positive")
    for contract in rates:
        _require_sha256(contract, label="source_pair_contract")
    return _semantic_sha256(
        "source-compute-receipt",
        {
            "source_expected_macs_per_token": source_expected_macs_per_token,
            "source_causal_pair_mac_rates": dict(sorted(rates.items())),
            "workload_profile_sha256": workload_profile_sha256,
        },
    )


def validate_gemma3_layer_agnostic_modal_graph_plan(
    plan: LayerAgnosticModalGraphPlan,
    protocol: Mapping[str, object] | None = None,
    *,
    expected_source_model_sha256: str,
    expected_exact_control_execution_sha256: str,
    expected_fit_split_manifest_sha256: str,
    expected_fit_objective_sha256: str,
    expected_resource_point_fit_receipt_sha256: str,
    expected_workload_profile_sha256: str,
    expected_source_compute_receipt_sha256: str,
    expected_source_serialized_bytes: int,
    input_mode: str = "input_ids",
) -> dict[str, object]:
    """Authenticate one fitted primary-arm plan against the frozen protocol."""

    if not isinstance(plan, LayerAgnosticModalGraphPlan):
        raise TypeError("plan must be a validated layer-agnostic modal graph")
    _require_sha256(
        expected_source_model_sha256,
        label="expected_source_model_sha256",
    )
    _require_sha256(
        expected_exact_control_execution_sha256,
        label="expected_exact_control_execution_sha256",
    )
    for label, digest in (
        (
            "expected_fit_split_manifest_sha256",
            expected_fit_split_manifest_sha256,
        ),
        ("expected_fit_objective_sha256", expected_fit_objective_sha256),
        (
            "expected_resource_point_fit_receipt_sha256",
            expected_resource_point_fit_receipt_sha256,
        ),
        ("expected_workload_profile_sha256", expected_workload_profile_sha256),
        (
            "expected_source_compute_receipt_sha256",
            expected_source_compute_receipt_sha256,
        ),
    ):
        _require_sha256(digest, label=label)
    if (
        type(expected_source_serialized_bytes) is not int
        or expected_source_serialized_bytes <= 0
    ):
        raise ValueError("expected source serialized bytes must be positive")
    if input_mode != "input_ids":
        raise ValueError("Gemma bridge v1 supports input_ids only")
    frozen = _validated_protocol(protocol)
    lineage = plan.lineage
    if lineage.protocol_sha256 != frozen["protocol_sha256"]:
        raise ValueError("plan is not bound to the frozen Gemma protocol")
    arm_ids = {
        arm["arm_id"]
        for arm in frozen["arms"]
        if isinstance(arm, Mapping)
    }
    if lineage.arm_id not in arm_ids:
        raise ValueError("plan arm is not frozen by the protocol")
    if lineage.arm_id != GEMMA3_LAYER_AGNOSTIC_PRIMARY_ARM_ID:
        raise ValueError("bridge v1 accepts only the primary compression arm")
    point = _resource_point(frozen, lineage.resource_point_id)
    if lineage.resource_point_sha256 != _semantic_sha256(
        "resource-point",
        point,
    ):
        raise ValueError("plan resource-point identity is invalid")
    if lineage.fit_configuration_sha256 != (
        gemma3_layer_agnostic_fit_configuration_sha256(frozen)
    ):
        raise ValueError("plan fit-configuration identity is invalid")
    if lineage.input_contract_sha256 != (
        gemma3_layer_agnostic_input_contract_sha256(
            frozen,
            input_mode=input_mode,
        )
    ):
        raise ValueError("plan input-contract identity is invalid")
    if lineage.source_model_sha256 != expected_source_model_sha256:
        raise ValueError("plan source-model identity is not authenticated")
    if (
        lineage.exact_control_execution_sha256
        != expected_exact_control_execution_sha256
    ):
        raise ValueError("plan exact-control identity is not authenticated")
    if lineage.fit_split_manifest_sha256 != expected_fit_split_manifest_sha256:
        raise ValueError("plan fit-split identity is not authenticated")
    if lineage.fit_objective_sha256 != expected_fit_objective_sha256:
        raise ValueError("plan fit-objective identity is not authenticated")
    if (
        lineage.resource_point_fit_receipt_sha256
        != expected_resource_point_fit_receipt_sha256
        or lineage.resource_point_fit_receipt_sha256 == _ZERO_SHA256
    ):
        raise ValueError("plan resource-point fit receipt is not authenticated")
    if (
        lineage.fit_relative_ridge is None
        or not math.isclose(
            lineage.fit_relative_ridge,
            float(point["relative_ridge"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("plan fitted ridge does not match resource point")
    for label, digest in (
        ("source model", lineage.source_model_sha256),
        ("fit split manifest", lineage.fit_split_manifest_sha256),
        ("fit objective", lineage.fit_objective_sha256),
        ("exact control", lineage.exact_control_execution_sha256),
    ):
        if digest == _ZERO_SHA256:
            raise ValueError(f"plan {label} evidence is unbound")

    assemblies = tuple(
        sorted(plan.mutation_assemblies, key=lambda item: item.assembly_index)
    )
    schedule = _expected_schedule(frozen)
    if len(assemblies) != GEMMA3_LAYER_AGNOSTIC_REQUIRED_ASSEMBLY_COUNT:
        raise ValueError("Gemma plan must contain exactly 39 mutation assemblies")
    if assemblies[0].input_carrier_id != "input_ids":
        raise ValueError("Gemma lexical ingress must consume input_ids")
    if assemblies[-1].output_carrier_id != "logits":
        raise ValueError("Gemma lexical egress must produce logits")
    invocations = {
        invocation.invocation_id: invocation for invocation in plan.invocations
    }
    for assembly, expected in zip(
        assemblies,
        schedule,
        strict=True,
    ):
        (
            index,
            invocation_id,
            role,
            source_mutation_id,
            dependency,
            _attention_kind,
        ) = expected
        if (
            assembly.assembly_index != index
            or assembly.assembly_role != role
            or assembly.source_mutation_id != source_mutation_id
        ):
            raise ValueError("Gemma mutation assembly schedule is not exact")
        if assembly.invocation_ids != (invocation_id,):
            raise ValueError("Gemma assembly must own its exact invocation")
        invocation = invocations[invocation_id]
        if (
            invocation.token_dependency != dependency
            or invocation.operator_role != role
            or invocation.source_mutation_id != source_mutation_id
            or invocation.sequence_operator_contract_sha256
            != gemma3_layer_agnostic_sequence_operator_contract_sha256(
                index,
                frozen,
            )
            or invocation.runtime_signature_sha256
            != gemma3_layer_agnostic_runtime_signature_sha256(index, frozen)
        ):
            raise ValueError("Gemma invocation runtime contract is not exact")

    pinned_source = frozen["pinned_source"]
    if not isinstance(pinned_source, Mapping):
        raise ValueError("protocol pinned source is invalid")
    lexical = plan.lexical_boundary
    if (
        lexical.vocabulary_size != pinned_source["vocabulary_size"]
        or lexical.source_width != pinned_source["hidden_width"]
        or lexical.lexical_rank != point["lexical_rank"]
    ):
        raise ValueError("Gemma lexical dimensions do not match resource point")
    atlas = plan.chart_atlas
    if (
        atlas.input_width != PINNED_HIDDEN_WIDTH
        or len(atlas.regions) != point["atlas_regions"]
        or any(
            region.coordinate_rank != point["residual_rank"]
            for region in atlas.regions
        )
        or atlas.active_chart_count != point["active_charts_per_token"]
        or atlas.local_correction_rank
        != point["local_chart_correction_rank"]
        or atlas.membership_rule != "top_k_softmax"
        or not math.isclose(
            atlas.temperature,
            float(point["chart_temperature"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("Gemma chart atlas does not match resource point")
    if (
        lexical.terminal_norm_scalar_count
        != PINNED_FINAL_RMSNORM_PARAMETER_COUNT
    ):
        raise ValueError("Gemma terminal RMSNorm inventory is not exact")
    incoming_counts: dict[str, int] = {}
    for edge in plan.transports:
        incoming_counts[edge.target_invocation_id] = (
            incoming_counts.get(edge.target_invocation_id, 0) + 1
        )
    if incoming_counts and max(incoming_counts.values()) > int(
        point["maximum_incoming_transports"]
    ):
        raise ValueError("Gemma transport fan-in exceeds resource point")
    body_invocations = [
        invocation
        for invocation in plan.invocations
        if 1 <= invocation.mutation_index <= 36
    ]
    body_mode_count = sum(
        len(invocation.source_mode_ids) for invocation in body_invocations
    )
    body_family_count = len(
        {invocation.family_id for invocation in body_invocations}
    )
    if (
        body_mode_count <= 0
        or body_family_count / body_mode_count
        > float(point["generator_community_budget_ratio"])
    ):
        raise ValueError("Gemma generator-community budget is exceeded")
    if (
        plan.resource_ledger.source_unique_learned_scalars
        != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
    ):
        raise ValueError("Gemma source parameter inventory is not exact")
    if any(
        component.storage_encoding != (
            "float32"
            if component.learned_scalar_count + component.fixed_scalar_count > 0
            else "metadata_bytes"
        )
        for component in plan.resource_ledger.components
    ):
        raise ValueError(
            "Gemma primary v1 storage must be float32 without quantization"
        )
    if (
        plan.resource_ledger.source_serialized_bytes
        != expected_source_serialized_bytes
    ):
        raise ValueError("Gemma source serialized-byte baseline is unauthenticated")
    if (
        plan.resource_ledger.workload_profile.artifact_sha256
        != expected_workload_profile_sha256
    ):
        raise ValueError("Gemma workload profile is not authenticated")
    observed_source_compute_receipt = (
        gemma3_layer_agnostic_source_compute_receipt_sha256(
            source_expected_macs_per_token=(
                plan.resource_ledger.source_expected_macs_per_token
            ),
            source_causal_pair_mac_rates=(
                plan.resource_ledger.source_causal_pair_mac_rate_map()
            ),
            workload_profile_sha256=(
                plan.resource_ledger.workload_profile.artifact_sha256
            ),
        )
    )
    if observed_source_compute_receipt != expected_source_compute_receipt_sha256:
        raise ValueError("Gemma source compute baseline is not authenticated")
    expected_attention_contracts = {
        gemma3_layer_agnostic_sequence_operator_contract_sha256(index, frozen)
        for index in range(1, 37, 2)
    }
    workload_contracts = {
        item.sequence_operator_contract_sha256
        for item in plan.resource_ledger.workload_profile.causal_pair_workloads
    }
    source_rate_contracts = {
        item.sequence_operator_contract_sha256
        for item in plan.resource_ledger.source_causal_pair_mac_rates
        if item.macs_per_causal_pair > 0
    }
    if (
        workload_contracts != expected_attention_contracts
        or source_rate_contracts != expected_attention_contracts
    ):
        raise ValueError("Gemma attention workload contracts are not exact")
    if any(
        invocation.attention_macs_per_causal_pair <= 0
        for invocation in body_invocations
        if invocation.token_dependency == "causal_prefix_mixer"
    ):
        raise ValueError("Gemma attention invocation pair MACs are missing")
    if any(
        invocation.attention_macs_per_causal_pair != 0
        for invocation in plan.invocations
        if invocation.token_dependency != "causal_prefix_mixer"
    ):
        raise ValueError("non-attention Gemma invocations cannot own pair MACs")
    if not plan.resource_ledger.experiment_executed:
        raise ValueError("fitted Gemma plan cannot be a setup-only placeholder")

    receipt = {
        "schema": GEMMA3_LAYER_AGNOSTIC_PLAN_BRIDGE_SCHEMA,
        "protocol_sha256": frozen["protocol_sha256"],
        "plan_artifact_sha256": plan.artifact_sha256,
        "arm_id": lineage.arm_id,
        "resource_point_id": lineage.resource_point_id,
        "resource_point_sha256": lineage.resource_point_sha256,
        "assembly_count": len(assemblies),
        "exact_39_event_schedule_bound": True,
        "source_parameter_inventory_bound": True,
        "source_model_sha256": lineage.source_model_sha256,
        "exact_control_execution_sha256": (
            lineage.exact_control_execution_sha256
        ),
        "input_mode": input_mode,
        "attention_runtime_contracts_bound": True,
        "resource_point_operational_fields_bound": True,
        "authoritative_fit_and_workload_receipts_bound": True,
        "authoritative_source_compute_receipt_bound": True,
        "generic_resource_accounting_eligible": (
            plan.resource_ledger.resource_accounting_eligible
        ),
        "compression_credit_authorized": False,
        "compression_credit_status": (
            "pending_family_disjoint_guard_and_assessment"
        ),
        "model_or_prompt_loaded_by_validator": False,
    }
    receipt["receipt_sha256"] = _semantic_sha256("bridge-receipt", receipt)
    return receipt


__all__ = [
    "GEMMA3_LAYER_AGNOSTIC_PLAN_BRIDGE_SCHEMA",
    "GEMMA3_LAYER_AGNOSTIC_PRIMARY_ARM_ID",
    "GEMMA3_LAYER_AGNOSTIC_REQUIRED_ASSEMBLY_COUNT",
    "gemma3_layer_agnostic_fit_configuration_sha256",
    "gemma3_layer_agnostic_input_contract_sha256",
    "gemma3_layer_agnostic_resource_point_sha256",
    "gemma3_layer_agnostic_runtime_signature_sha256",
    "gemma3_layer_agnostic_sequence_operator_contract_sha256",
    "gemma3_layer_agnostic_source_compute_receipt_sha256",
    "validate_gemma3_layer_agnostic_modal_graph_plan",
]
