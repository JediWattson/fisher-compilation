from __future__ import annotations

from dataclasses import replace

import pytest

from fisher_graph.gemma3_parts_first_supermodes import (
    ATTENTION_HEAD_COUNT,
    ATOM_OWNED_SCALAR_COUNT,
    HIDDEN_WIDTH,
    MLP_CHANNEL_COUNT,
    SHARED_SUBSTRATE_SCALAR_COUNT,
    SOURCE_UNIQUE_PARAMETER_COUNT,
    TOTAL_PHYSICAL_ATOM_COUNT,
    VOCABULARY_SIZE,
    AtomGroupDraft,
    ExecutionBoundary,
    FunctionalRole,
    LexicalRowViewSpec,
    PhysicalAtomFamily,
    PhysicalAtomId,
    SupermodePathSpec,
    SupermodeStepSpec,
    build_atom_group_sidecar,
    build_gemma3_parts_catalog,
    validate_atom_group_membership_ref,
    validate_exact_atom_partition,
    validate_gemma3_parts_catalog,
    validate_supermode_path,
)


def lexical(column: int) -> PhysicalAtomId:
    return PhysicalAtomId(
        PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
        column,
    )


def attention(layer: int, head: int) -> PhysicalAtomId:
    return PhysicalAtomId(
        PhysicalAtomFamily.ATTENTION_QO_HEAD,
        head,
        layer,
    )


def mlp(layer: int, channel: int) -> PhysicalAtomId:
    return PhysicalAtomId(
        PhysicalAtomFamily.MLP_CHANNEL,
        channel,
        layer,
    )


def test_catalog_closes_exact_physical_and_shared_accounting() -> None:
    catalog = build_gemma3_parts_catalog()

    assert validate_gemma3_parts_catalog(catalog) is catalog
    assert catalog.total_physical_atom_count == TOTAL_PHYSICAL_ATOM_COUNT == 37_576
    assert catalog.atom_owned_scalar_count == ATOM_OWNED_SCALAR_COUNT == 262_144_000
    assert (
        catalog.shared_substrate_scalar_count
        == SHARED_SUBSTRATE_SCALAR_COUNT
        == 5_954_176
    )
    assert (
        catalog.source_unique_parameter_count
        == SOURCE_UNIQUE_PARAMETER_COUNT
        == 268_098_176
    )
    assert sum(
        row.predecessor_physical_scalar_count
        for row in catalog.refinement_receipts
    ) == SOURCE_UNIQUE_PARAMETER_COUNT
    assert catalog.layer_semantics == "container_only_not_generator"


def test_compact_global_ordinal_domains_are_exact_and_invertible() -> None:
    catalog = build_gemma3_parts_catalog()
    cases = (
        (lexical(0), 0),
        (lexical(HIDDEN_WIDTH - 1), 639),
        (attention(0, 0), 640),
        (attention(17, ATTENTION_HEAD_COUNT - 1), 711),
        (mlp(0, 0), 712),
        (mlp(17, MLP_CHANNEL_COUNT - 1), 37_575),
    )

    for atom_id, global_ordinal in cases:
        assert catalog.global_ordinal(atom_id) == global_ordinal
        assert catalog.atom_id_from_global_ordinal(global_ordinal) == atom_id

    with pytest.raises(ValueError, match="outside the catalog"):
        catalog.atom_id_from_global_ordinal(37_576)


def test_tied_lexical_column_has_two_occurrences_but_one_owner() -> None:
    catalog = build_gemma3_parts_catalog()
    resolved = catalog.resolve(lexical(23))

    assert resolved.scalar_count == VOCABULARY_SIZE
    assert len(resolved.parameter_slices) == 1
    assert resolved.parameter_slices[0].axis == 1
    assert resolved.parameter_slices[0].start == 23
    assert resolved.parameter_slices[0].stop == 24
    assert tuple(item.role for item in resolved.occurrences) == (
        FunctionalRole.LEXICAL_INGRESS,
        FunctionalRole.LEXICAL_READOUT,
    )
    assert resolved.occurrences[0].atom_id == resolved.occurrences[1].atom_id
    assert resolved.predecessor_parent_atom_id == "atom.lexical.tied"


def test_attention_head_owns_only_q_rows_and_o_columns() -> None:
    catalog = build_gemma3_parts_catalog()
    resolved = catalog.resolve(attention(5, 2))

    assert resolved.global_ordinal == 640 + 4 * 5 + 2
    assert resolved.scalar_count == 327_680
    assert tuple(item.axis for item in resolved.parameter_slices) == (0, 1)
    assert tuple(
        (item.start, item.stop) for item in resolved.parameter_slices
    ) == ((512, 768), (512, 768))
    assert resolved.predecessor_parent_family_id == (
        "atom_bank.attention.global.full_rank"
    )

    substrate = next(
        item
        for item in catalog.shared_substrates
        if item.predecessor_parent_atom_id == "atom.attention.layer.5"
    )
    assert substrate.physical_scalar_count == 329_472
    assert substrate.owned_by_physical_atoms is False
    assert {item.component_id for item in substrate.components} >= {
        "k_projection",
        "v_projection",
        "query_rmsnorm",
        "key_rmsnorm",
        "causal_mask",
        "rope",
        "softmax",
    }


def test_mlp_channel_owns_gate_up_rows_and_down_column() -> None:
    catalog = build_gemma3_parts_catalog()
    resolved = catalog.resolve(mlp(9, 517))

    assert resolved.scalar_count == 1_920
    assert tuple(item.axis for item in resolved.parameter_slices) == (0, 0, 1)
    assert all(
        (item.start, item.stop) == (517, 518)
        for item in resolved.parameter_slices
    )
    assert resolved.predecessor_parent_atom_id == "atom.mlp.layer.9"


def test_logical_lexical_row_views_have_zero_credit_and_are_not_atoms() -> None:
    row_view = LexicalRowViewSpec(
        "view.lexical.rows.0_127",
        0,
        128,
        FunctionalRole.LEXICAL_INGRESS,
    )
    assert row_view.storage_credit_scalar_count == 0
    assert row_view.may_enter_physical_supermode_stage is False

    with pytest.raises(ValueError, match="no storage credit"):
        replace(row_view, storage_credit_scalar_count=1)
    with pytest.raises(TypeError, match="lexical row views are logical"):
        SupermodeStepSpec(  # type: ignore[arg-type]
            "step.bad",
            row_view,
            "chart.bad",
            "abi.carrier.in",
            "abi.delta.out",
            "abi.carrier.post",
        )


def test_group_sidecar_requires_one_family_and_one_boundary() -> None:
    catalog = build_gemma3_parts_catalog()
    good = AtomGroupDraft(
        "group.attention.layer0.heads01",
        ExecutionBoundary.attention(0),
        (attention(0, 0), attention(0, 1)),
    )
    blob, (ref,) = build_atom_group_sidecar(catalog, (good,))

    assert validate_atom_group_membership_ref(catalog, ref, blob) == good.atom_ids
    assert ref.member_count == 2
    assert ref.byte_length == 8
    assert ref.storage_credit_scalar_count == 0

    with pytest.raises(ValueError, match="homogeneous"):
        build_atom_group_sidecar(
            catalog,
            (
                AtomGroupDraft(
                    "group.mixed",
                    ExecutionBoundary.attention(0),
                    (attention(0, 0), mlp(0, 0)),
                ),
            ),
        )
    with pytest.raises(ValueError, match="no unique occurrence"):
        build_atom_group_sidecar(
            catalog,
            (
                AtomGroupDraft(
                    "group.wrong_boundary",
                    ExecutionBoundary.attention(1),
                    (attention(0, 0),),
                ),
            ),
        )


def test_group_members_must_be_sorted_unique_and_blob_is_authenticated() -> None:
    catalog = build_gemma3_parts_catalog()
    with pytest.raises(ValueError, match="canonically ordered"):
        build_atom_group_sidecar(
            catalog,
            (
                AtomGroupDraft(
                    "group.noncanonical",
                    ExecutionBoundary.mlp(0),
                    (mlp(0, 2), mlp(0, 1)),
                ),
            ),
        )
    blob, (ref,) = build_atom_group_sidecar(
        catalog,
        (
            AtomGroupDraft(
                "group.mlp.layer0.channels12",
                ExecutionBoundary.mlp(0),
                (mlp(0, 1), mlp(0, 2)),
            ),
        ),
    )
    with pytest.raises(ValueError, match="blob hash mismatch"):
        validate_atom_group_membership_ref(catalog, ref, blob[:-1] + b"\xff")
    with pytest.raises(ValueError, match="another atom universe"):
        validate_atom_group_membership_ref(
            catalog,
            replace(ref, atom_universe_sha256="0" * 64),
            blob,
        )


def test_logical_groups_can_overlap_without_storage_credit() -> None:
    catalog = build_gemma3_parts_catalog()
    drafts = (
        AtomGroupDraft(
            "group.overlap.left",
            ExecutionBoundary.attention(2),
            (attention(2, 0), attention(2, 1)),
        ),
        AtomGroupDraft(
            "group.overlap.right",
            ExecutionBoundary.attention(2),
            (attention(2, 1), attention(2, 2)),
        ),
    )
    blob, refs = build_atom_group_sidecar(catalog, drafts)

    assert all(item.group_semantics == "logical_overlap" for item in refs)
    assert sum(item.storage_credit_scalar_count for item in refs) == 0
    assert set(validate_atom_group_membership_ref(catalog, refs[0], blob)) & set(
        validate_atom_group_membership_ref(catalog, refs[1], blob)
    ) == {attention(2, 1)}


def test_exact_partition_is_separate_disjoint_exhaustive_semantics() -> None:
    catalog = build_gemma3_parts_catalog()
    boundary = ExecutionBoundary.attention(3)
    drafts = (
        AtomGroupDraft(
            "partition.attention.layer3.a",
            boundary,
            (attention(3, 0), attention(3, 1)),
            "exact_partition",
        ),
        AtomGroupDraft(
            "partition.attention.layer3.b",
            boundary,
            (attention(3, 2), attention(3, 3)),
            "exact_partition",
        ),
    )
    blob, refs = build_atom_group_sidecar(catalog, drafts)
    assert validate_exact_atom_partition(
        catalog,
        refs,
        blob,
        atom_family=PhysicalAtomFamily.ATTENTION_QO_HEAD,
        boundary=boundary,
    ) == refs

    with pytest.raises(ValueError, match="not exhaustive"):
        validate_exact_atom_partition(
            catalog,
            refs[:1],
            blob,
            atom_family=PhysicalAtomFamily.ATTENTION_QO_HEAD,
            boundary=boundary,
        )


def test_supermode_path_groups_parts_then_composes_boundaries_in_order() -> None:
    catalog = build_gemma3_parts_catalog()
    drafts = (
        AtomGroupDraft(
            "group.lexical.ingress.columns01",
            ExecutionBoundary.lexical_ingress(),
            (lexical(0), lexical(1)),
        ),
        AtomGroupDraft(
            "group.attention.layer0.heads02",
            ExecutionBoundary.attention(0),
            (attention(0, 0), attention(0, 2)),
        ),
        AtomGroupDraft(
            "group.mlp.layer0.channels57",
            ExecutionBoundary.mlp(0),
            (mlp(0, 5), mlp(0, 7)),
        ),
        AtomGroupDraft(
            "group.lexical.readout.columns01",
            ExecutionBoundary.lexical_readout(),
            (lexical(0), lexical(1)),
        ),
    )
    blob, refs = build_atom_group_sidecar(catalog, drafts)
    path = SupermodePathSpec(
        "supermode.lexical_to_layer0_to_readout",
        catalog.catalog_sha256,
        tuple(
            SupermodeStepSpec(
                f"step.{index}",
                group,
                f"chart.{index}",
                f"abi.carrier.{index}.in",
                f"abi.delta.{index}.out",
                f"abi.carrier.{index}.post",
            )
            for index, group in enumerate(refs)
        ),
    )

    assert validate_supermode_path(path, catalog, blob) is path
    assert tuple(step.boundary.execution_ordinal for step in path.steps) == (
        0,
        1,
        2,
        38,
    )
    assert path.cross_boundary_operator == "ordered_composition"
    assert path.storage_credit_scalar_count == 0


def test_supermode_rejects_cross_boundary_sum_and_whole_layer_generator() -> None:
    catalog = build_gemma3_parts_catalog()
    blob, refs = build_atom_group_sidecar(
        catalog,
        (
            AtomGroupDraft(
                "group.attention.layer0.head0",
                ExecutionBoundary.attention(0),
                (attention(0, 0),),
            ),
            AtomGroupDraft(
                "group.mlp.layer0.channel0",
                ExecutionBoundary.mlp(0),
                (mlp(0, 0),),
            ),
        ),
    )
    steps = (
        SupermodeStepSpec(
            "step.attention",
            refs[0],
            "chart.attention",
            "abi.attention.in",
            "abi.attention.delta",
            "abi.attention.post",
        ),
        SupermodeStepSpec(
            "step.mlp",
            refs[1],
            "chart.mlp",
            "abi.mlp.in",
            "abi.mlp.delta",
            "abi.mlp.post",
        ),
    )

    with pytest.raises(ValueError, match="cannot be summed"):
        SupermodePathSpec(
            "supermode.invalid.sum",
            catalog.catalog_sha256,
            steps,
            cross_boundary_operator="sum",
        )
    with pytest.raises(ValueError, match="whole layer"):
        SupermodePathSpec(
            "supermode.invalid.layer",
            catalog.catalog_sha256,
            steps,
            generator_subject="whole_layer",
        )


def test_supermode_steps_must_strictly_increase_by_boundary() -> None:
    catalog = build_gemma3_parts_catalog()
    blob, refs = build_atom_group_sidecar(
        catalog,
        (
            AtomGroupDraft(
                "group.mlp.layer1.channel0",
                ExecutionBoundary.mlp(1),
                (mlp(1, 0),),
            ),
            AtomGroupDraft(
                "group.attention.layer1.head0",
                ExecutionBoundary.attention(1),
                (attention(1, 0),),
            ),
        ),
    )
    del blob
    with pytest.raises(ValueError, match="strictly increasing"):
        SupermodePathSpec(
            "supermode.invalid.order",
            catalog.catalog_sha256,
            (
                SupermodeStepSpec(
                    "step.late",
                    refs[0],
                    "chart.late",
                    "abi.late.in",
                    "abi.late.delta",
                    "abi.late.post",
                ),
                SupermodeStepSpec(
                    "step.early",
                    refs[1],
                    "chart.early",
                    "abi.early.in",
                    "abi.early.delta",
                    "abi.early.post",
                ),
            ),
        )


def test_whole_layer_cannot_be_constructed_as_a_physical_atom() -> None:
    with pytest.raises(ValueError, match="whole layer"):
        PhysicalAtomId("whole_layer", 0, 0)  # type: ignore[arg-type]


def test_step_requires_chart_and_carrier_abis_and_path_requires_two_steps() -> None:
    catalog = build_gemma3_parts_catalog()
    _, (group,) = build_atom_group_sidecar(
        catalog,
        (
            AtomGroupDraft(
                "group.attention.layer0.head0",
                ExecutionBoundary.attention(0),
                (attention(0, 0),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="chart_id"):
        SupermodeStepSpec(
            "step.chartless",
            group,
            "",
            "abi.in",
            "abi.delta",
            "abi.post",
        )
    step = SupermodeStepSpec(
        "step.valid",
        group,
        "chart.attention.layer0",
        "abi.attention.in",
        "abi.attention.delta",
        "abi.attention.post",
    )
    with pytest.raises(ValueError, match="at least two steps"):
        SupermodePathSpec(
            "supermode.one_step",
            catalog.catalog_sha256,
            (step,),
        )


def test_cross_layer_claim_requires_two_distinct_layer_containers() -> None:
    catalog = build_gemma3_parts_catalog()
    blob, refs = build_atom_group_sidecar(
        catalog,
        (
            AtomGroupDraft(
                "group.attention.layer0.head0",
                ExecutionBoundary.attention(0),
                (attention(0, 0),),
            ),
            AtomGroupDraft(
                "group.mlp.layer0.channel0",
                ExecutionBoundary.mlp(0),
                (mlp(0, 0),),
            ),
            AtomGroupDraft(
                "group.attention.layer1.head0",
                ExecutionBoundary.attention(1),
                (attention(1, 0),),
            ),
        ),
    )
    steps = tuple(
        SupermodeStepSpec(
            f"step.{index}",
            group,
            f"chart.{index}",
            f"abi.{index}.in",
            f"abi.{index}.delta",
            f"abi.{index}.post",
        )
        for index, group in enumerate(refs)
    )
    with pytest.raises(ValueError, match="two distinct layer containers"):
        SupermodePathSpec(
            "supermode.false_cross_layer",
            catalog.catalog_sha256,
            steps[:2],
            cross_layer_claimed=True,
        )
    path = SupermodePathSpec(
        "supermode.true_cross_layer",
        catalog.catalog_sha256,
        steps,
        cross_layer_claimed=True,
    )
    assert validate_supermode_path(path, catalog, blob) is path


def test_catalog_validation_rejects_component_inventory_drift() -> None:
    catalog = build_gemma3_parts_catalog()
    first = catalog.shared_substrates[0]
    changed_component = replace(
        first.components[0],
        component_id="changed_k_projection",
    )
    changed_substrate = replace(
        first,
        components=(changed_component, *first.components[1:]),
    )
    changed_catalog = replace(
        catalog,
        shared_substrates=(changed_substrate, *catalog.shared_substrates[1:]),
        catalog_sha256="",
    )
    with pytest.raises(ValueError, match="component inventory drifted"):
        validate_gemma3_parts_catalog(changed_catalog)


def test_catalog_validation_rejects_self_rehashed_substrate_boundary_drift() -> None:
    catalog = build_gemma3_parts_catalog()
    changed_substrate = replace(
        catalog.shared_substrates[0],
        boundary=ExecutionBoundary.attention(1),
    )
    changed_catalog = replace(
        catalog,
        shared_substrates=(changed_substrate, *catalog.shared_substrates[1:]),
        catalog_sha256="",
    )
    assert changed_catalog.catalog_sha256 != catalog.catalog_sha256
    with pytest.raises(ValueError, match="component inventory drifted"):
        validate_gemma3_parts_catalog(changed_catalog)


def test_catalog_validation_rejects_self_rehashed_refinement_drift() -> None:
    catalog = build_gemma3_parts_catalog()
    changed_receipt = replace(
        catalog.refinement_receipts[1],
        predecessor_parent_family_id="atom_bank.attention.wrong.full_rank",
    )
    changed_catalog = replace(
        catalog,
        refinement_receipts=(
            catalog.refinement_receipts[0],
            changed_receipt,
            *catalog.refinement_receipts[2:],
        ),
        catalog_sha256="",
    )
    assert changed_catalog.catalog_sha256 != catalog.catalog_sha256
    with pytest.raises(ValueError, match="receipt inventory drifted"):
        validate_gemma3_parts_catalog(changed_catalog)
