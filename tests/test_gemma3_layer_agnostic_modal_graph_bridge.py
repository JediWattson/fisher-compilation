from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from fisher_graph.gemma3_layer_agnostic_modal_graph_bridge import (
    GEMMA3_LAYER_AGNOSTIC_PLAN_BRIDGE_SCHEMA,
    GEMMA3_LAYER_AGNOSTIC_PRIMARY_ARM_ID,
    gemma3_layer_agnostic_fit_configuration_sha256,
    gemma3_layer_agnostic_input_contract_sha256,
    gemma3_layer_agnostic_resource_point_sha256,
    gemma3_layer_agnostic_runtime_signature_sha256,
    gemma3_layer_agnostic_sequence_operator_contract_sha256,
    gemma3_layer_agnostic_source_compute_receipt_sha256,
    validate_gemma3_layer_agnostic_modal_graph_plan,
)
from fisher_graph.gemma3_layer_agnostic_whole_model_compression_protocol import (
    PINNED_FINAL_RMSNORM_PARAMETER_COUNT,
    PINNED_HIDDEN_WIDTH,
    PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
    PINNED_VOCABULARY_SIZE,
    build_layer_agnostic_whole_model_compression_protocol,
)
from fisher_graph.layer_agnostic_modal_graph import (
    CausalPairMacRate,
    CausalPairWorkload,
    GeneratorFamily,
    GeneratorInvocation,
    LayerAgnosticModalGraphPlan,
    LayerAgnosticPlanLineage,
    LayerAgnosticResourceComponent,
    LayerAgnosticResourceLedger,
    LayerAgnosticWorkloadProfile,
    ModalChartAtlas,
    ModalChartRegion,
    PhysicalStorageReceipt,
    ResidualMutationAssembly,
    SharedLexicalBoundaryPlan,
    SourceModeProvenance,
    deployment_component_span_receipt_sha256,
)


_SOURCE_MODEL_SHA256 = hashlib.sha256(b"synthetic-source-model").hexdigest()
_EXACT_CONTROL_SHA256 = hashlib.sha256(b"synthetic-exact-control").hexdigest()
_FIT_SPLIT_SHA256 = hashlib.sha256(b"synthetic-fit-split").hexdigest()
_FIT_OBJECTIVE_SHA256 = hashlib.sha256(b"synthetic-fit-objective").hexdigest()
_FIT_RECEIPT_SHA256 = hashlib.sha256(b"synthetic-fit-receipt").hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _schedule() -> tuple[tuple[int, str, str, str, str], ...]:
    protocol = build_layer_agnostic_whole_model_compression_protocol()
    body = protocol["whole_model_boundary_graph"]["mutation_invocations"]
    result: list[tuple[int, str, str, str, str]] = [
        (
            0,
            "lexical.ingress",
            "lexical_ingress",
            "source.lexical.ingress",
            "vocabulary_boundary",
        )
    ]
    for occurrence in body:
        dependency = occurrence["operator_contract"]
        result.append(
            (
                occurrence["mutation_index"],
                occurrence["invocation_id"],
                (
                    "attention_mutation"
                    if dependency == "causal_prefix_mixer"
                    else "feed_forward_mutation"
                ),
                occurrence["provenance_only"]["source_mutation_id"],
                dependency,
            )
        )
    result.extend(
        (
            (
                37,
                "terminal.rmsnorm",
                "terminal_rmsnorm",
                "source.terminal_rmsnorm",
                "token_local",
            ),
            (
                38,
                "lexical.egress",
                "lexical_egress",
                "source.lexical.egress",
                "vocabulary_boundary",
            ),
        )
    )
    return tuple(result)


def _family_id(index: int) -> str:
    if index in (0, 38):
        return "family.lexical"
    if index == 37:
        return "family.terminal_norm"
    return "family.attention" if index % 2 else "family.feed_forward"


def _component(
    *,
    component_id: str,
    component_kind: str,
    artifact_sha256: str,
    scalar_count: int,
    token_macs: int = 0,
    pair_macs: int = 0,
    pair_contract: str | None = None,
    non_mac_ops: int = 0,
    alias: str | None = None,
) -> LayerAgnosticResourceComponent:
    return LayerAgnosticResourceComponent(
        component_id=component_id,
        component_kind=component_kind,
        artifact_sha256=artifact_sha256,
        storage_encoding="float32" if scalar_count else "metadata_bytes",
        learned_scalar_count=scalar_count,
        fixed_scalar_count=0,
        serialized_byte_offset=0,
        serialized_bytes=4 * scalar_count,
        expected_macs_per_token=token_macs,
        worst_case_macs_per_token=token_macs,
        attention_macs_per_causal_pair=pair_macs,
        causal_pair_contract_sha256=pair_contract,
        non_mac_ops_per_token=non_mac_ops,
        serialized_size_audited=True,
        physical_storage_alias_id=alias,
    )


def _pack_components(
    components: list[LayerAgnosticResourceComponent],
) -> tuple[tuple[LayerAgnosticResourceComponent, ...], int, str]:
    cursor = 0
    packed: list[LayerAgnosticResourceComponent] = []
    for component in components:
        packed_component = replace(
            component,
            serialized_byte_offset=cursor,
        )
        packed.append(packed_component)
        cursor += packed_component.serialized_bytes
    catalog = tuple(packed)
    return catalog, cursor, deployment_component_span_receipt_sha256(catalog)


def _gemma_plan() -> LayerAgnosticModalGraphPlan:
    protocol = build_layer_agnostic_whole_model_compression_protocol()
    point = next(
        value
        for value in protocol["frozen_ladders"]["ordered_resource_points"]
        if value["resource_point_id"] == "balanced"
    )
    schedule = _schedule()
    runtime_signatures = tuple(
        sorted(
            gemma3_layer_agnostic_runtime_signature_sha256(index)
            for index, *_ in schedule
        )
    )
    source_modes = tuple(
        SourceModeProvenance(
            mode_id=f"mode.{index}",
            mutation_index=index,
            coordinate_index=0,
            fisher_signature_sha256=_digest(f"fisher.{index}"),
            geometry_signature_sha256=_digest(f"geometry.{index}"),
            runtime_signature_sha256=(
                gemma3_layer_agnostic_runtime_signature_sha256(index)
            ),
            source_boundary_id=f"carrier.{index}",
            source_operator_kind=role,
            source_layer_ordinal=(
                None if index in (0, 37, 38) else (index - 1) // 2
            ),
        )
        for index, _invocation_id, role, _source_id, _dependency in schedule
    )
    family_members = {
        "family.lexical": ("mode.0", "mode.38"),
        "family.attention": tuple(
            sorted(f"mode.{index}" for index in range(1, 37, 2))
        ),
        "family.feed_forward": tuple(
            sorted(f"mode.{index}" for index in range(2, 37, 2))
        ),
        "family.terminal_norm": ("mode.37",),
    }
    families = tuple(
        GeneratorFamily(
            family_id=family_id,
            member_mode_ids=member_ids,
            coordinate_rank=point["residual_rank"],
            basis_sha256=_digest(f"{family_id}.basis"),
            generator_bank_sha256=_digest(f"{family_id}.bank"),
            compatible_runtime_signature_sha256s=tuple(
                sorted(
                    gemma3_layer_agnostic_runtime_signature_sha256(
                        int(mode_id.split(".")[1])
                    )
                    for mode_id in member_ids
                )
            ),
            resource_component_id=f"{family_id}.weights",
            basis_scalar_count=1,
            generator_bank_scalar_count=1,
        )
        for family_id, member_ids in sorted(family_members.items())
    )
    invocations = tuple(
        GeneratorInvocation(
            invocation_id=invocation_id,
            family_id=_family_id(index),
            mutation_index=index,
            source_mode_ids=(f"mode.{index}",),
            input_boundary_id=("input_ids" if index == 0 else f"carrier.{index}"),
            output_boundary_id=(
                "logits" if index == 38 else f"carrier.{index + 1}"
            ),
            token_dependency=dependency,
            operator_role=role,
            runtime_signature_sha256=(
                gemma3_layer_agnostic_runtime_signature_sha256(index)
            ),
            sequence_operator_contract_sha256=(
                gemma3_layer_agnostic_sequence_operator_contract_sha256(index)
            ),
            source_mutation_id=source_id,
            resource_component_ids=(f"invocation.{index}.operator",),
            local_adapter_scalar_count=0,
            expected_macs_per_valid_token=1,
            attention_macs_per_causal_pair=(
                1 if dependency == "causal_prefix_mixer" else 0
            ),
            non_mac_ops_per_valid_token=0,
        )
        for index, invocation_id, role, source_id, dependency in schedule
    )
    assemblies = tuple(
        ResidualMutationAssembly(
            assembly_index=index,
            assembly_role=role,
            input_carrier_id=("input_ids" if index == 0 else f"carrier.{index}"),
            output_carrier_id=(
                "logits" if index == 38 else f"carrier.{index + 1}"
            ),
            invocation_ids=(invocation_id,),
            source_mutation_id=source_id,
            replacement_receipt_sha256=_digest(f"replacement.{index}"),
        )
        for index, invocation_id, role, source_id, _dependency in schedule
    )
    regions = tuple(
        ModalChartRegion(
            region_id=f"chart.{region_index}",
            input_width=PINNED_HIDDEN_WIDTH,
            coordinate_rank=point["residual_rank"],
            center_sha256=_digest(f"chart.{region_index}.center"),
            basis_sha256=_digest(f"chart.{region_index}.basis"),
            metric_sha256=_digest(f"chart.{region_index}.metric"),
            compatible_runtime_signature_sha256s=runtime_signatures,
        )
        for region_index in range(point["atlas_regions"])
    )
    atlas = ModalChartAtlas(
        atlas_id="gemma3.balance.atlas",
        input_width=PINNED_HIDDEN_WIDTH,
        regions=regions,
        membership_rule="top_k_softmax",
        temperature=float(point["chart_temperature"]),
        router_sha256=_digest("atlas.router"),
        carrier_runtime_signature_sha256=_digest("carrier.runtime"),
        artifact_resource_component_id="atlas.regions",
        router_resource_component_id="atlas.router",
        resource_component_ids=(
            "atlas.local_correction",
            "atlas.regions",
            "atlas.router",
        ),
        router_scalar_count=point["atlas_regions"],
        routing_macs_per_valid_token=1,
        routing_non_mac_ops_per_valid_token=1,
        active_chart_count=point["active_charts_per_token"],
        local_correction_rank=point["local_chart_correction_rank"],
        local_correction_artifact_sha256=_digest("atlas.local_correction"),
        local_correction_resource_component_id="atlas.local_correction",
        local_correction_scalar_count=(
            2
            * PINNED_HIDDEN_WIDTH
            * point["local_chart_correction_rank"]
            * point["atlas_regions"]
        ),
        local_correction_macs_per_valid_token=(
            2
            * PINNED_HIDDEN_WIDTH
            * point["local_chart_correction_rank"]
            * point["active_charts_per_token"]
        ),
    )
    factor_count = point["lexical_rank"] * (
        PINNED_VOCABULARY_SIZE + PINNED_HIDDEN_WIDTH
    )
    storage_receipt = PhysicalStorageReceipt(
        storage_handle_sha256=_digest("packed.lexical.C_then_D"),
        byte_offset=0,
        byte_length=4 * factor_count,
        dtype="float32",
        device="cpu",
    )
    lexical = SharedLexicalBoundaryPlan(
        shared_factor_id="lexical.shared_factor",
        factor_artifact_sha256=_digest("lexical.factor"),
        vocabulary_size=PINNED_VOCABULARY_SIZE,
        source_width=PINNED_HIDDEN_WIDTH,
        lexical_rank=point["lexical_rank"],
        ingress_invocation_id="lexical.ingress",
        egress_invocation_id="lexical.egress",
        terminal_norm_invocation_id="terminal.rmsnorm",
        generator_family_id="family.lexical",
        factor_resource_component_id="lexical.factor",
        ingress_map_resource_component_id="lexical.ingress.map",
        egress_map_resource_component_id="lexical.egress.map",
        terminal_norm_resource_component_id="terminal.norm",
        physical_storage_receipt_sha256=storage_receipt.receipt_sha256,
        ingress_map_artifact_sha256=_digest("lexical.ingress.map"),
        egress_map_artifact_sha256=_digest("lexical.egress.map"),
        terminal_norm_artifact_sha256=_digest("terminal.norm"),
        ingress_map_scalar_count=1,
        egress_map_scalar_count=1,
        terminal_norm_scalar_count=PINNED_FINAL_RMSNORM_PARAMETER_COUNT,
    )

    components: list[LayerAgnosticResourceComponent] = []
    for family in families:
        components.append(
            _component(
                component_id=family.resource_component_id,
                component_kind="generator_family",
                artifact_sha256=family.artifact_sha256,
                scalar_count=family.stored_scalar_count,
            )
        )
    for invocation in invocations:
        pair_macs = invocation.attention_macs_per_causal_pair
        components.append(
            _component(
                component_id=invocation.resource_component_ids[0],
                component_kind="invocation_operator",
                artifact_sha256=(
                    invocation.sequence_operator_contract_sha256
                ),
                scalar_count=0,
                token_macs=1,
                pair_macs=pair_macs,
                pair_contract=(
                    invocation.sequence_operator_contract_sha256
                    if pair_macs
                    else None
                ),
            )
        )
    components.extend(
        (
            _component(
                component_id="atlas.regions",
                component_kind="chart_atlas",
                artifact_sha256=atlas.artifact_sha256,
                scalar_count=atlas.region_scalar_count,
            ),
            _component(
                component_id="atlas.router",
                component_kind="chart_router",
                artifact_sha256=atlas.router_sha256,
                scalar_count=atlas.router_scalar_count,
                token_macs=atlas.routing_macs_per_valid_token,
                non_mac_ops=atlas.routing_non_mac_ops_per_valid_token,
            ),
            _component(
                component_id="atlas.local_correction",
                component_kind="chart_local_correction",
                artifact_sha256=atlas.local_correction_artifact_sha256,
                scalar_count=atlas.local_correction_scalar_count,
                token_macs=atlas.local_correction_macs_per_valid_token,
            ),
            _component(
                component_id="lexical.factor",
                component_kind="shared_lexical_factor",
                artifact_sha256=lexical.factor_artifact_sha256,
                scalar_count=lexical.factorized_scalar_count,
                alias=lexical.shared_factor_id,
            ),
            _component(
                component_id="lexical.ingress.map",
                component_kind="lexical_ingress_map",
                artifact_sha256=lexical.ingress_map_artifact_sha256,
                scalar_count=lexical.ingress_map_scalar_count,
            ),
            _component(
                component_id="lexical.egress.map",
                component_kind="lexical_egress_map",
                artifact_sha256=lexical.egress_map_artifact_sha256,
                scalar_count=lexical.egress_map_scalar_count,
            ),
            _component(
                component_id="terminal.norm",
                component_kind="terminal_rmsnorm",
                artifact_sha256=lexical.terminal_norm_artifact_sha256,
                scalar_count=lexical.terminal_norm_scalar_count,
            ),
        )
    )
    packed, deployment_size, span_receipt = _pack_components(components)
    attention_contracts = tuple(
        invocation.sequence_operator_contract_sha256
        for invocation in invocations
        if invocation.token_dependency == "causal_prefix_mixer"
    )
    workload = LayerAgnosticWorkloadProfile(
        profile_id="synthetic.family_disjoint.workload",
        valid_token_count=100,
        worst_case_valid_token_count=120,
        causal_pair_workloads=tuple(
            CausalPairWorkload(
                sequence_operator_contract_sha256=contract,
                valid_causal_pair_count=10,
                worst_case_causal_pair_count=20,
            )
            for contract in attention_contracts
        ),
        sequence_count=4,
        row_identity_sha256=_digest("row.identity"),
        sequence_mask_receipt_sha256=_digest("sequence.mask"),
    )
    ledger = LayerAgnosticResourceLedger(
        source_unique_learned_scalars=PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
        source_serialized_bytes=4 * PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
        source_expected_macs_per_token=1_000,
        source_causal_pair_mac_rates=tuple(
            CausalPairMacRate(
                sequence_operator_contract_sha256=contract,
                macs_per_causal_pair=100,
            )
            for contract in attention_contracts
        ),
        workload_profile=workload,
        components=packed,
        unaccounted_source_checkpoint_present=False,
        native_fallback_reachable=False,
        physical_alias_audit_passed=True,
        physical_storage_receipt_sha256=storage_receipt.receipt_sha256,
        zero_native_source_calls_audited=True,
        source_call_audit_sha256=_digest("source.call.audit"),
        deployment_artifact_sha256=_digest("deployment.artifact"),
        deployment_artifact_size_bytes=deployment_size,
        component_span_receipt_sha256=span_receipt,
        experiment_executed=True,
    )
    lineage = LayerAgnosticPlanLineage(
        protocol_sha256=protocol["protocol_sha256"],
        source_model_sha256=_SOURCE_MODEL_SHA256,
        fit_split_manifest_sha256=_FIT_SPLIT_SHA256,
        fit_objective_sha256=_FIT_OBJECTIVE_SHA256,
        exact_control_execution_sha256=_EXACT_CONTROL_SHA256,
        arm_id=GEMMA3_LAYER_AGNOSTIC_PRIMARY_ARM_ID,
        resource_point_id=point["resource_point_id"],
        resource_point_sha256=gemma3_layer_agnostic_resource_point_sha256(
            point["resource_point_id"]
        ),
        fit_configuration_sha256=(
            gemma3_layer_agnostic_fit_configuration_sha256()
        ),
        input_contract_sha256=gemma3_layer_agnostic_input_contract_sha256(),
        fit_relative_ridge=float(point["relative_ridge"]),
        resource_point_fit_receipt_sha256=_FIT_RECEIPT_SHA256,
    )
    return LayerAgnosticModalGraphPlan(
        lineage=lineage,
        source_modes=source_modes,
        generator_families=families,
        invocations=invocations,
        transports=(),
        chart_atlas=atlas,
        mutation_assemblies=assemblies,
        lexical_boundary=lexical,
        resource_ledger=ledger,
    )


def _expected_source_compute_receipt(
    plan: LayerAgnosticModalGraphPlan,
) -> str:
    return gemma3_layer_agnostic_source_compute_receipt_sha256(
        source_expected_macs_per_token=(
            _gemma_plan_source_expected_macs_per_token()
        ),
        source_causal_pair_mac_rates=(
            _gemma_plan_source_pair_rates()
        ),
        workload_profile_sha256=(
            plan.resource_ledger.workload_profile.artifact_sha256
        ),
    )


def _validate(plan: LayerAgnosticModalGraphPlan) -> dict[str, object]:
    return validate_gemma3_layer_agnostic_modal_graph_plan(
        plan,
        expected_source_model_sha256=_SOURCE_MODEL_SHA256,
        expected_exact_control_execution_sha256=_EXACT_CONTROL_SHA256,
        expected_fit_split_manifest_sha256=_FIT_SPLIT_SHA256,
        expected_fit_objective_sha256=_FIT_OBJECTIVE_SHA256,
        expected_resource_point_fit_receipt_sha256=_FIT_RECEIPT_SHA256,
        expected_workload_profile_sha256=(
            plan.resource_ledger.workload_profile.artifact_sha256
        ),
        expected_source_compute_receipt_sha256=(
            _expected_source_compute_receipt(plan)
        ),
        expected_source_serialized_bytes=(
            4 * PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        ),
    )


def _gemma_plan_source_expected_macs_per_token() -> int:
    return 1_000


def _gemma_plan_source_pair_rates() -> dict[str, int]:
    return {
        gemma3_layer_agnostic_sequence_operator_contract_sha256(index): 100
        for index in range(1, 37, 2)
    }


def test_bridge_authenticates_the_exact_39_event_primary_plan() -> None:
    plan = _gemma_plan()
    receipt = _validate(plan)
    assert receipt["schema"] == GEMMA3_LAYER_AGNOSTIC_PLAN_BRIDGE_SCHEMA
    assert receipt["assembly_count"] == 39
    assert receipt["exact_39_event_schedule_bound"] is True
    assert receipt["attention_runtime_contracts_bound"] is True
    assert receipt["input_mode"] == "input_ids"
    assert receipt["model_or_prompt_loaded_by_validator"] is False
    assert receipt["generic_resource_accounting_eligible"] is True
    assert receipt["compression_credit_authorized"] is False
    assert len(receipt["receipt_sha256"]) == 64


def test_bridge_rejects_unbound_lineage_and_runtime_authority() -> None:
    plan = _gemma_plan()
    wrong_protocol = replace(
        plan,
        lineage=replace(plan.lineage, protocol_sha256=_digest("wrong")),
    )
    with pytest.raises(ValueError, match="frozen Gemma protocol"):
        _validate(wrong_protocol)

    wrong_point = replace(
        plan,
        lineage=replace(
            plan.lineage,
            resource_point_sha256=_digest("wrong.point"),
        ),
    )
    with pytest.raises(ValueError, match="resource-point identity"):
        _validate(wrong_point)

    with pytest.raises(ValueError, match="source-model identity"):
        validate_gemma3_layer_agnostic_modal_graph_plan(
            plan,
            expected_source_model_sha256=_digest("different.source"),
            expected_exact_control_execution_sha256=_EXACT_CONTROL_SHA256,
            expected_fit_split_manifest_sha256=_FIT_SPLIT_SHA256,
            expected_fit_objective_sha256=_FIT_OBJECTIVE_SHA256,
            expected_resource_point_fit_receipt_sha256=_FIT_RECEIPT_SHA256,
            expected_workload_profile_sha256=(
                plan.resource_ledger.workload_profile.artifact_sha256
            ),
            expected_source_compute_receipt_sha256=(
                _expected_source_compute_receipt(plan)
            ),
            expected_source_serialized_bytes=(
                4 * PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
            ),
        )
    with pytest.raises(ValueError, match="input_ids only"):
        validate_gemma3_layer_agnostic_modal_graph_plan(
            plan,
            expected_source_model_sha256=_SOURCE_MODEL_SHA256,
            expected_exact_control_execution_sha256=_EXACT_CONTROL_SHA256,
            expected_fit_split_manifest_sha256=_FIT_SPLIT_SHA256,
            expected_fit_objective_sha256=_FIT_OBJECTIVE_SHA256,
            expected_resource_point_fit_receipt_sha256=_FIT_RECEIPT_SHA256,
            expected_workload_profile_sha256=(
                plan.resource_ledger.workload_profile.artifact_sha256
            ),
            expected_source_compute_receipt_sha256=(
                _expected_source_compute_receipt(plan)
            ),
            expected_source_serialized_bytes=(
                4 * PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
            ),
            input_mode="inputs_embeds",
        )


def test_bridge_rejects_a_semantically_renamed_body_occurrence() -> None:
    plan = _gemma_plan()
    invocation = plan.invocations[1]
    renamed = replace(invocation, invocation_id="renamed.attention")
    invocations = (plan.invocations[0], renamed) + plan.invocations[2:]
    assemblies = list(plan.mutation_assemblies)
    assemblies[1] = replace(
        assemblies[1],
        invocation_ids=(renamed.invocation_id,),
    )
    renamed_plan = replace(
        plan,
        invocations=invocations,
        mutation_assemblies=tuple(assemblies),
    )
    with pytest.raises(ValueError, match="exact invocation"):
        _validate(renamed_plan)


def test_bridge_rejects_a_setup_only_unexecuted_plan() -> None:
    plan = _gemma_plan()
    unexecuted = replace(
        plan,
        resource_ledger=replace(
            plan.resource_ledger,
            experiment_executed=False,
        ),
    )
    with pytest.raises(ValueError, match="setup-only placeholder"):
        _validate(unexecuted)


def test_bridge_binds_ingress_routing_ridge_and_fit_authority() -> None:
    plan = _gemma_plan()
    invocations = list(plan.invocations)
    invocations[0] = replace(invocations[0], input_boundary_id="inputs_embeds")
    assemblies = list(plan.mutation_assemblies)
    assemblies[0] = replace(assemblies[0], input_carrier_id="inputs_embeds")
    inputs_embeds_plan = replace(
        plan,
        invocations=tuple(invocations),
        mutation_assemblies=tuple(assemblies),
    )
    with pytest.raises(ValueError, match="consume input_ids"):
        _validate(inputs_embeds_plan)

    output_invocations = list(plan.invocations)
    output_invocations[-1] = replace(
        output_invocations[-1],
        output_boundary_id="hidden_state",
    )
    output_assemblies = list(plan.mutation_assemblies)
    output_assemblies[-1] = replace(
        output_assemblies[-1],
        output_carrier_id="hidden_state",
    )
    hidden_output_plan = replace(
        plan,
        invocations=tuple(output_invocations),
        mutation_assemblies=tuple(output_assemblies),
    )
    with pytest.raises(ValueError, match="produce logits"):
        _validate(hidden_output_plan)

    changed_atlas = replace(plan.chart_atlas, membership_rule="top99_softmax")
    components = tuple(
        replace(component, artifact_sha256=changed_atlas.artifact_sha256)
        if component.component_id == changed_atlas.artifact_resource_component_id
        else component
        for component in plan.resource_ledger.components
    )
    changed_ledger = replace(
        plan.resource_ledger,
        components=components,
        component_span_receipt_sha256=(
            deployment_component_span_receipt_sha256(components)
        ),
    )
    with pytest.raises(ValueError, match="chart atlas"):
        _validate(
            replace(
                plan,
                chart_atlas=changed_atlas,
                resource_ledger=changed_ledger,
            )
        )

    wrong_fit_split = replace(
        plan,
        lineage=replace(
            plan.lineage,
            fit_split_manifest_sha256=_digest("untrusted.fit.split"),
        ),
    )
    with pytest.raises(ValueError, match="fit-split identity"):
        _validate(wrong_fit_split)
    wrong_ridge = replace(
        plan,
        lineage=replace(plan.lineage, fit_relative_ridge=1.0e-2),
    )
    with pytest.raises(ValueError, match="fitted ridge"):
        _validate(wrong_ridge)


def test_bridge_binds_source_bytes_and_attention_workload_contracts() -> None:
    plan = _gemma_plan()
    inflated = replace(
        plan,
        resource_ledger=replace(
            plan.resource_ledger,
            source_serialized_bytes=(
                100 * plan.resource_ledger.source_serialized_bytes
            ),
        ),
    )
    with pytest.raises(ValueError, match="serialized-byte baseline"):
        _validate(inflated)
    quantized_components = tuple(
        replace(component, storage_encoding="float16")
        if component.learned_scalar_count + component.fixed_scalar_count > 0
        else component
        for component in plan.resource_ledger.components
    )
    quantized = replace(
        plan,
        resource_ledger=replace(
            plan.resource_ledger,
            components=quantized_components,
            component_span_receipt_sha256=(
                deployment_component_span_receipt_sha256(quantized_components)
            ),
        ),
    )
    with pytest.raises(ValueError, match="without quantization"):
        _validate(quantized)
    inflated_compute = replace(
        plan,
        resource_ledger=replace(
            plan.resource_ledger,
            source_expected_macs_per_token=(
                100 * plan.resource_ledger.source_expected_macs_per_token
            ),
        ),
    )
    with pytest.raises(ValueError, match="source compute baseline"):
        _validate(inflated_compute)

    unrelated_contract = _digest("unrelated.workload.contract")
    invocations = tuple(
        replace(invocation, attention_macs_per_causal_pair=0)
        if invocation.token_dependency == "causal_prefix_mixer"
        else invocation
        for invocation in plan.invocations
    )
    invocation_by_component = {
        invocation.resource_component_ids[0]: invocation
        for invocation in invocations
    }
    components = tuple(
        replace(
            component,
            attention_macs_per_causal_pair=0,
            causal_pair_contract_sha256=None,
        )
        if component.component_id in invocation_by_component
        and invocation_by_component[
            component.component_id
        ].token_dependency == "causal_prefix_mixer"
        else component
        for component in plan.resource_ledger.components
    )
    workload = replace(
        plan.resource_ledger.workload_profile,
        causal_pair_workloads=(
            CausalPairWorkload(
                sequence_operator_contract_sha256=unrelated_contract,
                valid_causal_pair_count=10,
                worst_case_causal_pair_count=20,
            ),
        ),
    )
    ledger = replace(
        plan.resource_ledger,
        components=components,
        workload_profile=workload,
        source_causal_pair_mac_rates=(
            CausalPairMacRate(
                sequence_operator_contract_sha256=unrelated_contract,
                macs_per_causal_pair=100,
            ),
        ),
    )
    unrelated = replace(
        plan,
        invocations=invocations,
        resource_ledger=ledger,
    )
    with pytest.raises(
        ValueError,
        match="source compute baseline|workload contracts",
    ):
        _validate(unrelated)
