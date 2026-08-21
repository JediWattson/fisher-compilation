from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from fisher_graph.layer_agnostic_modal_graph import (
    CausalPairMacRate,
    CausalPairWorkload,
    CoordinateAdapterContract,
    GeneratorFamily,
    GeneratorInvocation,
    GeneratorTransport,
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
    authenticate_layer_agnostic_modal_graph_state,
    deployment_component_span_receipt_sha256,
    require_shared_physical_storage,
    validate_causal_invocation_graph,
)


def _digest(character: str) -> str:
    return character * 64


_LEXICAL_SIGNATURE = _digest("a")
_ATTENTION_SIGNATURE = _digest("b")
_TOKEN_SIGNATURE = _digest("c")
_NORM_SIGNATURE = _digest("d")


def _mode(
    mode_id: str,
    mutation_index: int,
    coordinate_index: int,
    runtime_signature: str,
    *,
    layer: int | None,
) -> SourceModeProvenance:
    return SourceModeProvenance(
        mode_id=mode_id,
        mutation_index=mutation_index,
        coordinate_index=coordinate_index,
        fisher_signature_sha256=_digest("1"),
        geometry_signature_sha256=_digest("2"),
        runtime_signature_sha256=runtime_signature,
        source_boundary_id=f"source.boundary.{mutation_index}",
        source_operator_kind="source.operator",
        source_layer_ordinal=layer,
    )


def _invocation(
    invocation_id: str,
    family_id: str,
    mutation_index: int,
    mode_id: str,
    *,
    dependency: str,
    role: str,
    runtime_signature: str,
    source_mutation_id: str,
    coordinate_adapter: CoordinateAdapterContract | None = None,
) -> GeneratorInvocation:
    compute_resource_id = f"compute.{invocation_id}"
    resource_component_ids = tuple(
        sorted(
            {
                compute_resource_id,
                *(
                    ()
                    if coordinate_adapter is None
                    else (coordinate_adapter.resource_component_id,)
                ),
            }
        )
    )
    return GeneratorInvocation(
        invocation_id=invocation_id,
        family_id=family_id,
        mutation_index=mutation_index,
        source_mode_ids=(mode_id,),
        input_boundary_id=(
            "tokens.ids" if mutation_index == 0 else f"carrier.{mutation_index - 1}"
        ),
        output_boundary_id=(
            "logits" if role == "lexical_egress" else f"carrier.{mutation_index}"
        ),
        token_dependency=dependency,
        operator_role=role,
        runtime_signature_sha256=runtime_signature,
        sequence_operator_contract_sha256=_digest("3"),
        source_mutation_id=source_mutation_id,
        resource_component_ids=resource_component_ids,
        local_adapter_scalar_count=(
            0
            if coordinate_adapter is None
            else coordinate_adapter.stored_scalar_count
        ),
        expected_macs_per_valid_token=(
            5
            if coordinate_adapter is None
            else 5 + coordinate_adapter.macs_per_valid_token
        ),
        attention_macs_per_causal_pair=(
            2 if dependency == "causal_prefix_mixer" else 0
        ),
        non_mac_ops_per_valid_token=0,
        coordinate_adapter=coordinate_adapter,
    )


def _family(
    family_id: str,
    member_mode_ids: tuple[str, ...],
    runtime_signatures: tuple[str, ...],
    *,
    resource_id: str,
    stored_scalar_count: int,
) -> GeneratorFamily:
    return GeneratorFamily(
        family_id=family_id,
        member_mode_ids=member_mode_ids,
        coordinate_rank=2,
        basis_sha256=_digest("4"),
        generator_bank_sha256=_digest("5"),
        compatible_runtime_signature_sha256s=runtime_signatures,
        resource_component_id=resource_id,
        basis_scalar_count=0,
        generator_bank_scalar_count=stored_scalar_count,
    )


def _transport(
    edge_id: str,
    source: str,
    target: str,
    *,
    relation: str = "same_token",
) -> GeneratorTransport:
    return GeneratorTransport(
        transport_id=edge_id,
        source_invocation_id=source,
        target_invocation_id=target,
        token_relation=relation,
        source_token_lag=0,
        coefficient_sha256=_digest("6"),
        stored_scalar_count=4,
        macs_per_active_token=4,
        attention_macs_per_causal_pair=(2 if relation == "causal_prefix" else 0),
        resource_component_id=f"transport.{edge_id}",
    )


def _component(
    component_id: str,
    artifact_sha256: str,
    *,
    learned: int = 0,
    fixed: int = 0,
    token_macs: int = 0,
    pair_macs: int = 0,
    pair_contract: str | None = None,
    alias: str | None = None,
    fallback: bool = False,
    kind: str = "test_component",
) -> LayerAgnosticResourceComponent:
    return LayerAgnosticResourceComponent(
        component_id=component_id,
        component_kind=kind,
        artifact_sha256=artifact_sha256,
        storage_encoding="float32",
        learned_scalar_count=learned,
        fixed_scalar_count=fixed,
        serialized_byte_offset=0,
        serialized_bytes=4 * (learned + fixed),
        expected_macs_per_token=token_macs,
        worst_case_macs_per_token=token_macs,
        attention_macs_per_causal_pair=pair_macs,
        causal_pair_contract_sha256=pair_contract,
        non_mac_ops_per_token=0,
        serialized_size_audited=True,
        physical_storage_alias_id=alias,
        unaccounted_or_live_fallback=fallback,
    )


def _repack_components(
    components: list[LayerAgnosticResourceComponent]
    | tuple[LayerAgnosticResourceComponent, ...],
) -> tuple[tuple[LayerAgnosticResourceComponent, ...], int, str]:
    cursor = 0
    packed: list[LayerAgnosticResourceComponent] = []
    for component in components:
        packed.append(replace(component, serialized_byte_offset=cursor))
        cursor += component.serialized_bytes
    result = tuple(packed)
    return result, cursor, deployment_component_span_receipt_sha256(result)


def _replace_ledger_components(
    ledger: LayerAgnosticResourceLedger,
    components: list[LayerAgnosticResourceComponent]
    | tuple[LayerAgnosticResourceComponent, ...],
) -> LayerAgnosticResourceLedger:
    packed, size, receipt = _repack_components(components)
    return replace(
        ledger,
        components=packed,
        deployment_artifact_size_bytes=size,
        component_span_receipt_sha256=receipt,
    )


def _plan(*, experiment_executed: bool = True) -> LayerAgnosticModalGraphPlan:
    modes = (
        _mode("mode.lexical.in", 0, 0, _LEXICAL_SIGNATURE, layer=None),
        _mode("mode.a", 1, 0, _ATTENTION_SIGNATURE, layer=0),
        _mode("mode.b", 2, 1, _TOKEN_SIGNATURE, layer=0),
        _mode("mode.c", 3, 2, _ATTENTION_SIGNATURE, layer=7),
        _mode("mode.norm", 4, 0, _NORM_SIGNATURE, layer=None),
        _mode("mode.lexical.out", 5, 0, _LEXICAL_SIGNATURE, layer=None),
    )
    families = (
        _family(
            "family.lexical",
            ("mode.lexical.in", "mode.lexical.out"),
            (_LEXICAL_SIGNATURE,),
            resource_id="family.lexical.weights",
            stored_scalar_count=40,
        ),
        _family(
            "family.shared_attention",
            ("mode.a", "mode.c"),
            (_ATTENTION_SIGNATURE,),
            resource_id="family.shared_attention.weights",
            stored_scalar_count=5,
        ),
        _family(
            "family.token",
            ("mode.b",),
            (_TOKEN_SIGNATURE,),
            resource_id="family.token.weights",
            stored_scalar_count=3,
        ),
        _family(
            "family.norm",
            ("mode.norm",),
            (_NORM_SIGNATURE,),
            resource_id="family.norm.weights",
            stored_scalar_count=4,
        ),
    )
    invocations = (
        _invocation(
            "invoke.lexical.in",
            "family.lexical",
            0,
            "mode.lexical.in",
            dependency="vocabulary_boundary",
            role="lexical_ingress",
            runtime_signature=_LEXICAL_SIGNATURE,
            source_mutation_id="source.lexical.ingress",
        ),
        _invocation(
            "invoke.a",
            "family.shared_attention",
            1,
            "mode.a",
            dependency="causal_prefix_mixer",
            role="attention_mutation",
            runtime_signature=_ATTENTION_SIGNATURE,
            source_mutation_id="layer.0.attention",
        ),
        _invocation(
            "invoke.b",
            "family.token",
            2,
            "mode.b",
            dependency="token_local",
            role="feed_forward_mutation",
            runtime_signature=_TOKEN_SIGNATURE,
            source_mutation_id="layer.0.feed_forward",
        ),
        _invocation(
            "invoke.c",
            "family.shared_attention",
            3,
            "mode.c",
            dependency="causal_prefix_mixer",
            role="attention_mutation",
            runtime_signature=_ATTENTION_SIGNATURE,
            source_mutation_id="layer.7.attention",
        ),
        _invocation(
            "invoke.norm",
            "family.norm",
            4,
            "mode.norm",
            dependency="token_local",
            role="terminal_rmsnorm",
            runtime_signature=_NORM_SIGNATURE,
            source_mutation_id="source.terminal_rmsnorm",
        ),
        _invocation(
            "invoke.lexical.out",
            "family.lexical",
            5,
            "mode.lexical.out",
            dependency="vocabulary_boundary",
            role="lexical_egress",
            runtime_signature=_LEXICAL_SIGNATURE,
            source_mutation_id="source.lexical.egress",
        ),
    )
    transports = (
        _transport("edge.a.b", "invoke.a", "invoke.b"),
        _transport("edge.a.c", "invoke.a", "invoke.c"),
        _transport(
            "edge.b.c",
            "invoke.b",
            "invoke.c",
            relation="causal_prefix",
        ),
    )
    atlas = ModalChartAtlas(
        atlas_id="atlas.shared",
        input_width=4,
        regions=(
            ModalChartRegion(
                region_id="region.0",
                input_width=4,
                coordinate_rank=2,
                center_sha256=_digest("7"),
                basis_sha256=_digest("8"),
                metric_sha256=_digest("9"),
                compatible_runtime_signature_sha256s=(
                    _LEXICAL_SIGNATURE,
                    _ATTENTION_SIGNATURE,
                    _TOKEN_SIGNATURE,
                    _NORM_SIGNATURE,
                ),
            ),
            ModalChartRegion(
                region_id="region.1",
                input_width=4,
                coordinate_rank=2,
                center_sha256=_digest("a"),
                basis_sha256=_digest("b"),
                metric_sha256=_digest("c"),
                compatible_runtime_signature_sha256s=(
                    _LEXICAL_SIGNATURE,
                    _ATTENTION_SIGNATURE,
                    _TOKEN_SIGNATURE,
                    _NORM_SIGNATURE,
                ),
            ),
        ),
        membership_rule="fisher_mahalanobis_softmax",
        temperature=1.0,
        router_sha256=_digest("d"),
        carrier_runtime_signature_sha256=_digest("e"),
        artifact_resource_component_id="atlas.artifact",
        router_resource_component_id="atlas.router",
        resource_component_ids=(
            "atlas.artifact",
            "atlas.local_correction",
            "atlas.router",
        ),
        active_chart_count=1,
        router_scalar_count=2,
        routing_macs_per_valid_token=3,
        routing_non_mac_ops_per_valid_token=0,
        local_correction_rank=1,
        local_correction_artifact_sha256=_digest("e"),
        local_correction_resource_component_id="atlas.local_correction",
        local_correction_scalar_count=16,
        local_correction_macs_per_valid_token=8,
    )
    assemblies = tuple(
        ResidualMutationAssembly(
            assembly_index=invocation.mutation_index,
            assembly_role=invocation.operator_role,
            input_carrier_id=invocation.input_boundary_id,
            output_carrier_id=invocation.output_boundary_id,
            invocation_ids=(invocation.invocation_id,),
            source_mutation_id=invocation.source_mutation_id,
            replacement_receipt_sha256=_digest("f"),
        )
        for invocation in invocations
    )
    receipt = PhysicalStorageReceipt(
        storage_handle_sha256=_digest("0"),
        byte_offset=0,
        byte_length=160,
        dtype="float32",
        device="cpu",
    )
    lexical = SharedLexicalBoundaryPlan(
        shared_factor_id="lexical.shared",
        factor_artifact_sha256=families[0].artifact_sha256,
        vocabulary_size=16,
        source_width=4,
        lexical_rank=2,
        ingress_invocation_id="invoke.lexical.in",
        egress_invocation_id="invoke.lexical.out",
        terminal_norm_invocation_id="invoke.norm",
        generator_family_id="family.lexical",
        factor_resource_component_id="family.lexical.weights",
        ingress_map_resource_component_id="lexical.ingress.map",
        egress_map_resource_component_id="lexical.egress.map",
        terminal_norm_resource_component_id="family.norm.weights",
        physical_storage_receipt_sha256=receipt.receipt_sha256,
        ingress_map_artifact_sha256=_digest("1"),
        egress_map_artifact_sha256=_digest("2"),
        terminal_norm_artifact_sha256=families[3].artifact_sha256,
        ingress_map_scalar_count=2,
        egress_map_scalar_count=2,
        terminal_norm_scalar_count=4,
    )

    components: list[LayerAgnosticResourceComponent] = []
    for family in families:
        components.append(
            _component(
                family.resource_component_id,
                family.artifact_sha256,
                learned=family.stored_scalar_count,
                alias=(
                    lexical.shared_factor_id
                    if family.family_id == "family.lexical"
                    else None
                ),
            )
        )
    for invocation in invocations:
        compute_resource_id = f"compute.{invocation.invocation_id}"
        components.append(
            _component(
                compute_resource_id,
                invocation.sequence_operator_contract_sha256,
                token_macs=5,
                pair_macs=(
                    2 if invocation.token_dependency == "causal_prefix_mixer" else 0
                ),
                pair_contract=(
                    invocation.sequence_operator_contract_sha256
                    if invocation.token_dependency == "causal_prefix_mixer"
                    else None
                ),
            )
        )
    for edge in transports:
        components.append(
            _component(
                edge.resource_component_id,
                edge.coefficient_sha256,
                learned=edge.stored_scalar_count,
                token_macs=edge.macs_per_active_token,
                pair_macs=edge.attention_macs_per_causal_pair,
                pair_contract=(
                    _digest("3")
                    if edge.attention_macs_per_causal_pair > 0
                    else None
                ),
            )
        )
    components.extend(
        (
            _component(
                "atlas.artifact",
                atlas.artifact_sha256,
                fixed=atlas.region_scalar_count,
            ),
            _component("atlas.router", atlas.router_sha256, learned=2, token_macs=3),
            _component(
                "atlas.local_correction",
                atlas.local_correction_artifact_sha256,
                learned=atlas.local_correction_scalar_count,
                token_macs=atlas.local_correction_macs_per_valid_token,
                kind="chart_local_correction",
            ),
            _component("lexical.ingress.map", _digest("1"), learned=2),
            _component("lexical.egress.map", _digest("2"), learned=2),
        )
    )
    packed_components, deployment_size, span_receipt = _repack_components(
        components
    )
    ledger = LayerAgnosticResourceLedger(
        source_unique_learned_scalars=200,
        source_serialized_bytes=800,
        source_expected_macs_per_token=200,
        source_causal_pair_mac_rates=(
            CausalPairMacRate(
                sequence_operator_contract_sha256=_digest("3"),
                macs_per_causal_pair=20,
            ),
        ),
        workload_profile=LayerAgnosticWorkloadProfile(
            profile_id="workload.test",
            valid_token_count=10,
            worst_case_valid_token_count=12,
            causal_pair_workloads=(
                CausalPairWorkload(
                    sequence_operator_contract_sha256=_digest("3"),
                    valid_causal_pair_count=25,
                    worst_case_causal_pair_count=36,
                ),
            ),
            sequence_count=2,
            row_identity_sha256=_digest("5"),
            sequence_mask_receipt_sha256=_digest("6"),
        ),
        components=packed_components,
        unaccounted_source_checkpoint_present=False,
        native_fallback_reachable=False,
        physical_alias_audit_passed=True,
        physical_storage_receipt_sha256=receipt.receipt_sha256,
        zero_native_source_calls_audited=True,
        source_call_audit_sha256=_digest("3"),
        deployment_artifact_sha256=_digest("4"),
        deployment_artifact_size_bytes=deployment_size,
        component_span_receipt_sha256=span_receipt,
        experiment_executed=experiment_executed,
    )
    return LayerAgnosticModalGraphPlan(
        lineage=LayerAgnosticPlanLineage(
            protocol_sha256=_digest("5"),
            source_model_sha256=_digest("6"),
            fit_split_manifest_sha256=_digest("7"),
            fit_objective_sha256=_digest("8"),
            exact_control_execution_sha256=_digest("9"),
        ),
        source_modes=modes,
        generator_families=families,
        invocations=invocations,
        transports=transports,
        chart_atlas=atlas,
        mutation_assemblies=assemblies,
        lexical_boundary=lexical,
        resource_ledger=ledger,
    )


def test_family_reuse_is_distinct_from_invocation_and_spans_provenance() -> None:
    plan = _plan()
    shared = [
        invocation
        for invocation in plan.invocations
        if invocation.family_id == "family.shared_attention"
    ]
    assert [item.mutation_index for item in shared] == [1, 3]
    assert len({item.invocation_id for item in shared}) == 2
    assert plan.communities_spanning_source_layers == (
        "family.shared_attention",
    )
    assert plan.artifact_sha256 == plan.state_dict()["artifact_sha256"]


def test_provenance_labels_are_authenticated_but_not_clustering_features() -> None:
    plan = _plan()
    renamed_modes = tuple(
        replace(
            mode,
            source_boundary_id=f"renamed.boundary.{index}",
            source_operator_kind="renamed.operator",
            source_layer_ordinal=(
                None
                if mode.source_layer_ordinal is None
                else mode.source_layer_ordinal + 100
            ),
        )
        for index, mode in enumerate(plan.source_modes)
    )
    renamed = replace(plan, source_modes=renamed_modes)
    assert renamed.execution_sha256 == plan.execution_sha256
    assert renamed.artifact_sha256 != plan.artifact_sha256
    for mode in plan.source_modes:
        features = mode.clustering_feature_state()
        assert "mode_id" not in features
        assert "source_layer_ordinal" not in features
        assert "source_boundary_id" not in features


def test_residual_assemblies_close_ingress_body_norm_and_egress() -> None:
    plan = _plan()
    assert [item.assembly_index for item in plan.mutation_assemblies] == list(
        range(6)
    )
    assert plan.mutation_assemblies[0].assembly_role == "lexical_ingress"
    assert plan.mutation_assemblies[-2].assembly_role == "terminal_rmsnorm"
    assert plan.mutation_assemblies[-1].assembly_role == "lexical_egress"
    for previous, current in zip(
        plan.mutation_assemblies,
        plan.mutation_assemblies[1:],
    ):
        assert previous.output_carrier_id == current.input_carrier_id


def test_broken_or_partial_residual_chain_is_rejected() -> None:
    plan = _plan()
    broken = replace(
        plan.mutation_assemblies[2],
        input_carrier_id="carrier.wrong",
    )
    with pytest.raises(ValueError, match="not closed"):
        replace(
            plan,
            mutation_assemblies=(
                plan.mutation_assemblies[:2]
                + (broken,)
                + plan.mutation_assemblies[3:]
            ),
        )
    with pytest.raises(ValueError, match="incomplete"):
        replace(plan, mutation_assemblies=plan.mutation_assemblies[:3])


def test_causal_prefix_semantics_are_bound_to_the_target_operator() -> None:
    plan = _plan()
    validate_causal_invocation_graph(plan.invocations, plan.transports)
    valid = _transport(
        "edge.token.to.attention",
        "invoke.b",
        "invoke.c",
        relation="causal_prefix",
    )
    validate_causal_invocation_graph(plan.invocations, (valid,))
    invalid = _transport(
        "edge.attention.to.token",
        "invoke.a",
        "invoke.b",
        relation="causal_prefix",
    )
    with pytest.raises(ValueError, match="target"):
        validate_causal_invocation_graph(plan.invocations, (invalid,))


@pytest.mark.parametrize(
    ("source", "target"),
    (("invoke.b", "invoke.a"), ("invoke.a", "invoke.a")),
)
def test_causal_graph_rejects_reverse_and_self_edges(
    source: str,
    target: str,
) -> None:
    plan = _plan()
    with pytest.raises(ValueError):
        validate_causal_invocation_graph(
            plan.invocations,
            (_transport("edge.bad", source, target),),
        )


def test_family_runtime_signature_mismatch_is_rejected() -> None:
    plan = _plan()
    incompatible = replace(
        plan.invocations[2],
        runtime_signature_sha256=_ATTENTION_SIGNATURE,
    )
    with pytest.raises(ValueError, match="incompatible with family"):
        replace(
            plan,
            invocations=(
                plan.invocations[:2]
                + (incompatible,)
                + plan.invocations[3:]
            ),
        )


def test_direct_chart_path_rejects_a_mixed_routable_atlas() -> None:
    plan = _plan()
    mixed_region = replace(
        plan.chart_atlas.regions[1],
        compatible_runtime_signature_sha256s=(
            _LEXICAL_SIGNATURE,
            _ATTENTION_SIGNATURE,
            _NORM_SIGNATURE,
        ),
    )
    mixed_atlas = replace(
        plan.chart_atlas,
        regions=(plan.chart_atlas.regions[0], mixed_region),
    )
    with pytest.raises(ValueError, match="every routable chart region"):
        replace(plan, chart_atlas=mixed_atlas)


def test_family_chart_coordinates_require_a_compatible_region_or_adapter() -> None:
    plan = _plan()

    family_index = next(
        index
        for index, family in enumerate(plan.generator_families)
        if family.family_id == "family.shared_attention"
    )
    overwide_family = replace(
        plan.generator_families[family_index],
        coordinate_rank=plan.chart_atlas.input_width + 1,
    )
    overwide_families = (
        plan.generator_families[:family_index]
        + (overwide_family,)
        + plan.generator_families[family_index + 1 :]
    )
    with pytest.raises(ValueError, match="coordinate adapter contract"):
        replace(plan, generator_families=overwide_families)

    adapters = {
        invocation.invocation_id: CoordinateAdapterContract(
            adapter_id=f"adapter.{invocation.invocation_id}",
            source_region_id="region.0",
            source_runtime_signature_sha256=_ATTENTION_SIGNATURE,
            source_coordinate_rank=2,
            target_family_id=overwide_family.family_id,
            target_runtime_signature_sha256=_ATTENTION_SIGNATURE,
            target_coordinate_rank=overwide_family.coordinate_rank,
            adapter_artifact_sha256=(
                _digest("e")
                if invocation.invocation_id == "invoke.a"
                else _digest("f")
            ),
            resource_component_id=f"adapter.{invocation.invocation_id}.weights",
            stored_scalar_count=2 * overwide_family.coordinate_rank,
            macs_per_valid_token=2 * overwide_family.coordinate_rank,
        )
        for invocation in plan.invocations
        if invocation.family_id == overwide_family.family_id
    }
    adapted_invocations = tuple(
        replace(
            invocation,
            coordinate_adapter=adapters[invocation.invocation_id],
            resource_component_ids=tuple(
                sorted(
                    invocation.resource_component_ids
                    + (
                        adapters[
                            invocation.invocation_id
                        ].resource_component_id,
                    )
                )
            ),
            local_adapter_scalar_count=(
                adapters[invocation.invocation_id].stored_scalar_count
            ),
            expected_macs_per_valid_token=(
                invocation.expected_macs_per_valid_token
                + adapters[invocation.invocation_id].macs_per_valid_token
            ),
        )
        if invocation.family_id == overwide_family.family_id
        else invocation
        for invocation in plan.invocations
    )
    adapted_components = [
        replace(component, artifact_sha256=overwide_family.artifact_sha256)
        if component.component_id == overwide_family.resource_component_id
        else component
        for component in plan.resource_ledger.components
    ]
    adapted_components.extend(
        _component(
            adapter.resource_component_id,
            adapter.adapter_artifact_sha256,
            learned=adapter.stored_scalar_count,
            token_macs=adapter.macs_per_valid_token,
            kind="coordinate_adapter",
        )
        for adapter in adapters.values()
    )
    adapted_plan = replace(
        plan,
        generator_families=overwide_families,
        invocations=adapted_invocations,
        resource_ledger=_replace_ledger_components(
            plan.resource_ledger,
            adapted_components,
        ),
    )
    assert (
        next(
            family
            for family in adapted_plan.generator_families
            if family.family_id == overwide_family.family_id
        ).coordinate_rank
        > adapted_plan.chart_atlas.input_width
    )

    adapter_incompatible_region = replace(
        adapted_plan.chart_atlas.regions[1],
        compatible_runtime_signature_sha256s=(
            _LEXICAL_SIGNATURE,
            _TOKEN_SIGNATURE,
            _NORM_SIGNATURE,
        ),
    )
    adapter_incompatible_atlas = replace(
        adapted_plan.chart_atlas,
        regions=(
            adapted_plan.chart_atlas.regions[0],
            adapter_incompatible_region,
        ),
    )
    with pytest.raises(ValueError, match="adapter source rank/runtime"):
        replace(adapted_plan, chart_atlas=adapter_incompatible_atlas)

    planted_invocation = next(
        invocation
        for invocation in adapted_plan.invocations
        if invocation.coordinate_adapter is not None
    )
    planted_adapter = planted_invocation.coordinate_adapter
    assert planted_adapter is not None

    malformed_rank = replace(
        planted_adapter,
        target_coordinate_rank=overwide_family.coordinate_rank - 1,
        stored_scalar_count=(
            planted_adapter.source_coordinate_rank
            * (overwide_family.coordinate_rank - 1)
        ),
        macs_per_valid_token=(
            planted_adapter.source_coordinate_rank
            * (overwide_family.coordinate_rank - 1)
        ),
    )
    with pytest.raises(ValueError, match="target does not match"):
        replace(
            adapted_plan,
            invocations=tuple(
                replace(invocation, coordinate_adapter=malformed_rank)
                if invocation.invocation_id == planted_invocation.invocation_id
                else invocation
                for invocation in adapted_plan.invocations
            ),
        )

    adapter_component = next(
        component
        for component in adapted_plan.resource_ledger.components
        if component.component_id == planted_adapter.resource_component_id
    )
    malformed_artifact_components = tuple(
        replace(component, artifact_sha256=_digest("0"))
        if component.component_id == adapter_component.component_id
        else component
        for component in adapted_plan.resource_ledger.components
    )
    with pytest.raises(ValueError, match="adapter resource artifact"):
        replace(
            adapted_plan,
            resource_ledger=_replace_ledger_components(
                adapted_plan.resource_ledger,
                malformed_artifact_components,
            ),
        )

    malformed_accounting_components = tuple(
        replace(
            component,
            learned_scalar_count=component.learned_scalar_count - 1,
        )
        if component.component_id == adapter_component.component_id
        else component
        for component in adapted_plan.resource_ledger.components
    )
    with pytest.raises(ValueError, match="adapter scalar receipt"):
        replace(
            adapted_plan,
            resource_ledger=_replace_ledger_components(
                adapted_plan.resource_ledger,
                malformed_accounting_components,
            ),
        )

    malformed_mac_components = tuple(
        replace(
            component,
            expected_macs_per_token=component.expected_macs_per_token - 1,
            worst_case_macs_per_token=component.worst_case_macs_per_token - 1,
        )
        if component.component_id == adapter_component.component_id
        else component
        for component in adapted_plan.resource_ledger.components
    )
    with pytest.raises(ValueError, match="adapter MAC receipt"):
        replace(
            adapted_plan,
            resource_ledger=_replace_ledger_components(
                adapted_plan.resource_ledger,
                malformed_mac_components,
            ),
        )


def test_coordinate_adapter_contract_rejects_scalar_or_mac_shortcuts() -> None:
    adapter = CoordinateAdapterContract(
        adapter_id="adapter.valid",
        source_region_id="region.0",
        source_runtime_signature_sha256=_ATTENTION_SIGNATURE,
        source_coordinate_rank=2,
        target_family_id="family.shared_attention",
        target_runtime_signature_sha256=_ATTENTION_SIGNATURE,
        target_coordinate_rank=5,
        adapter_artifact_sha256=_digest("e"),
        resource_component_id="adapter.valid.weights",
        stored_scalar_count=10,
        macs_per_valid_token=10,
    )
    with pytest.raises(ValueError, match="scalar count"):
        replace(adapter, stored_scalar_count=1)
    with pytest.raises(ValueError, match="MAC count"):
        replace(adapter, macs_per_valid_token=1)


def test_chart_activity_and_local_correction_have_exact_formulas() -> None:
    atlas = _plan().chart_atlas
    assert atlas.active_chart_count == 1
    assert atlas.local_correction_scalar_count == (
        2
        * atlas.input_width
        * atlas.local_correction_rank
        * len(atlas.regions)
    )
    assert atlas.local_correction_macs_per_valid_token == (
        2
        * atlas.input_width
        * atlas.local_correction_rank
        * atlas.active_chart_count
    )
    with pytest.raises(ValueError, match="cannot exceed atlas regions"):
        replace(atlas, active_chart_count=len(atlas.regions) + 1)
    with pytest.raises(ValueError, match="scalar count"):
        replace(
            atlas,
            local_correction_scalar_count=(
                atlas.local_correction_scalar_count - 1
            ),
        )
    with pytest.raises(ValueError, match="MAC count"):
        replace(
            atlas,
            local_correction_macs_per_valid_token=(
                atlas.local_correction_macs_per_valid_token - 1
            ),
        )


def test_chart_local_correction_resource_receipt_cannot_drift() -> None:
    plan = _plan()
    correction_id = plan.chart_atlas.local_correction_resource_component_id
    correction = next(
        component
        for component in plan.resource_ledger.components
        if component.component_id == correction_id
    )

    def ledger_with(component: LayerAgnosticResourceComponent):
        return _replace_ledger_components(
            plan.resource_ledger,
            tuple(
                component
                if candidate.component_id == correction_id
                else candidate
                for candidate in plan.resource_ledger.components
            ),
        )

    with pytest.raises(ValueError, match="resource artifact"):
        replace(
            plan,
            resource_ledger=ledger_with(
                replace(correction, artifact_sha256=_digest("0"))
            ),
        )
    with pytest.raises(ValueError, match="scalar receipt"):
        replace(
            plan,
            resource_ledger=ledger_with(
                replace(
                    correction,
                    learned_scalar_count=correction.learned_scalar_count - 1,
                )
            ),
        )
    with pytest.raises(ValueError, match="MAC receipt"):
        replace(
            plan,
            resource_ledger=ledger_with(
                replace(
                    correction,
                    expected_macs_per_token=(
                        correction.expected_macs_per_token - 1
                    ),
                    worst_case_macs_per_token=(
                        correction.worst_case_macs_per_token - 1
                    ),
                )
            ),
        )


def test_lineage_authenticates_ridge_and_resource_point_fit_receipt() -> None:
    lineage = _plan().lineage
    assert lineage.fit_relative_ridge is None
    assert lineage.resource_point_fit_receipt_sha256 == _digest("0")
    bound = replace(
        lineage,
        fit_relative_ridge=1.0e-3,
        resource_point_fit_receipt_sha256=_digest("f"),
    )
    assert bound.state_dict()["fit_relative_ridge"] == 1.0e-3
    assert bound.state_dict()["resource_point_fit_receipt_sha256"] == _digest(
        "f"
    )
    with pytest.raises(ValueError, match="finite and positive"):
        replace(lineage, fit_relative_ridge=0.0)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(lineage, resource_point_fit_receipt_sha256="unbound")


def test_resource_ledger_exactly_closes_over_graph_resources() -> None:
    plan = _plan()
    referenced = {
        component.component_id for component in plan.resource_ledger.components
    }
    assert plan.chart_atlas.artifact_resource_component_id in referenced
    assert plan.lexical_boundary.factor_resource_component_id in referenced
    assert all(
        edge.resource_component_id in referenced for edge in plan.transports
    )
    assert plan.resource_ledger.resource_accounting_eligible is True
    assert plan.resource_ledger.compression_credit_eligible is False

    missing = _replace_ledger_components(
        plan.resource_ledger,
        plan.resource_ledger.components[:-1],
    )
    with pytest.raises(ValueError, match="does not close"):
        replace(plan, resource_ledger=missing)


def test_logical_compute_uses_are_unique_and_pair_macs_are_weighted() -> None:
    plan = _plan()
    ledger = plan.resource_ledger
    assert ledger.source_expected_total_macs == 2_500
    assert ledger.candidate_expected_total_macs == 680
    assert ledger.expected_mac_reduction_fraction == pytest.approx(0.728)

    shared_compute = replace(
        plan.invocations[3],
        resource_component_ids=plan.invocations[1].resource_component_ids,
    )
    removed_id = plan.invocations[3].resource_component_ids[0]
    undercounted_ledger = _replace_ledger_components(
        ledger,
        tuple(
            component
            for component in ledger.components
            if component.component_id != removed_id
        ),
    )
    with pytest.raises(ValueError, match="unique uses"):
        replace(
            plan,
            invocations=plan.invocations[:3]
            + (shared_compute,)
            + plan.invocations[4:],
            resource_ledger=undercounted_ledger,
        )


def test_lexical_scalar_artifact_and_alias_receipts_cannot_drift() -> None:
    plan = _plan()
    components = list(plan.resource_ledger.components)
    index = next(
        index
        for index, component in enumerate(components)
        if component.component_id
        == plan.lexical_boundary.factor_resource_component_id
    )
    components[index] = replace(
        components[index],
        learned_scalar_count=components[index].learned_scalar_count - 1,
        serialized_bytes=components[index].serialized_bytes - 4,
    )
    mismatched = _replace_ledger_components(
        plan.resource_ledger,
        tuple(components),
    )
    with pytest.raises(ValueError, match="scalar receipt"):
        replace(plan, resource_ledger=mismatched)
    with pytest.raises(ValueError, match="matrix rank bound"):
        replace(plan.lexical_boundary, lexical_rank=5)


def test_physical_storage_receipts_accept_views_and_reject_distinct_spans() -> None:
    ingress = PhysicalStorageReceipt(
        storage_handle_sha256=_digest("a"),
        byte_offset=0,
        byte_length=160,
        dtype="float32",
        device="cpu",
    )
    egress_view = replace(ingress)
    assert require_shared_physical_storage(ingress, egress_view) == (
        ingress.receipt_sha256
    )
    with pytest.raises(ValueError, match="storage span"):
        require_shared_physical_storage(
            ingress,
            replace(ingress, byte_offset=4),
        )


def test_resource_flags_are_strict_bools_and_collections_are_frozen() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="must be a bool"):
        replace(plan.resource_ledger, experiment_executed="yes")

    mutable_components = list(plan.resource_ledger.components)
    copied = replace(plan.resource_ledger, components=mutable_components)
    before = copied.candidate_unique_learned_scalars
    mutable_components.clear()
    assert copied.candidate_unique_learned_scalars == before


def test_negative_savings_are_not_clamped_and_fallback_gets_no_credit() -> None:
    plan = _plan()
    oversized_component = replace(
        plan.resource_ledger.components[0],
        learned_scalar_count=300,
        serialized_bytes=1_200,
    )
    oversized = _replace_ledger_components(
        plan.resource_ledger,
        (oversized_component,) + plan.resource_ledger.components[1:],
    )
    assert oversized.learned_scalar_reduction_fraction < 0.0
    assert oversized.serialized_byte_reduction_fraction < 0.0
    assert oversized.resource_accounting_eligible is False
    assert oversized.compression_credit_eligible is False
    fallback_component = replace(
        plan.resource_ledger.components[-1],
        unaccounted_or_live_fallback=True,
    )
    fallback = _replace_ledger_components(
        plan.resource_ledger,
        plan.resource_ledger.components[:-1] + (fallback_component,),
    )
    assert fallback.resource_accounting_eligible is False
    assert fallback.compression_credit_eligible is False


def test_serialized_spans_and_total_stored_scalars_prevent_relabeling() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="too small"):
        replace(
            plan.resource_ledger.components[0],
            serialized_bytes=0,
        )

    components = list(plan.resource_ledger.components)
    index = next(
        index
        for index, component in enumerate(components)
        if component.component_id == "family.shared_attention.weights"
    )
    original = components[index]
    components[index] = replace(
        original,
        learned_scalar_count=0,
        fixed_scalar_count=original.learned_scalar_count,
    )
    relabeled = _replace_ledger_components(plan.resource_ledger, components)
    assert relabeled.candidate_unique_learned_scalars < (
        plan.resource_ledger.candidate_unique_learned_scalars
    )
    assert relabeled.candidate_unique_stored_scalars == (
        plan.resource_ledger.candidate_unique_stored_scalars
    )
    assert relabeled.stored_scalar_reduction_fraction == (
        plan.resource_ledger.stored_scalar_reduction_fraction
    )


def test_serialized_plan_authentication_rejects_rehashed_semantic_drift() -> None:
    plan = _plan()
    authentic = authenticate_layer_agnostic_modal_graph_state(
        plan,
        plan.state_dict(),
    )
    assert authentic["artifact_sha256"] == plan.artifact_sha256
    tampered = deepcopy(authentic)
    tampered["resource_ledger"]["resource_accounting_eligible"] = False
    tampered["artifact_sha256"] = _digest("0")
    with pytest.raises(ValueError, match="not authentic"):
        authenticate_layer_agnostic_modal_graph_state(plan, tampered)
