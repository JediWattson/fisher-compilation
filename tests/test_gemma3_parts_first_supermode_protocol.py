from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fisher_graph.gemma3_parts_first_supermodes import (
    build_gemma3_parts_catalog,
)
from fisher_graph.gemma3_parts_first_supermode_protocol import (
    ATOM_OWNED_SCALARS,
    PHYSICAL_ATOM_COUNT,
    SHARED_SUBSTRATE_SCALARS,
    SOURCE_SCALARS,
    build_parts_first_supermode_protocol,
    protocol_semantic_sha256,
    render_parts_first_supermode_protocol,
    validate_parts_first_supermode_protocol,
)


def test_parts_first_accounting_closes_the_checkpoint() -> None:
    assert PHYSICAL_ATOM_COUNT == 37_576
    assert ATOM_OWNED_SCALARS == 262_144_000
    assert SHARED_SUBSTRATE_SCALARS == 5_954_176
    assert SOURCE_SCALARS == 268_098_176
    state = build_parts_first_supermode_protocol()
    accounting = state["scalar_accounting"]
    assert (
        accounting["atom_owned_scalars"]
        + accounting["shared_substrate_scalars"]
        == accounting["source_unique_stored_scalars"]
    )


def test_physical_families_refine_layers_into_parts() -> None:
    state = build_parts_first_supermode_protocol()
    families = state["physical_atom_universe"]["families"]
    assert [family["atom_count"] for family in families] == [640, 72, 36_864]
    assert [family["family_id"] for family in families] == [
        "tied_lexical_hidden_column",
        "attention_qo_head",
        "mlp_channel",
    ]
    assert families[0]["ordinal_start"] == 0
    assert families[0]["ordinal_end_exclusive"] == 640
    assert families[1]["ordinal_start"] == 640
    assert families[1]["ordinal_end_exclusive"] == 712
    assert families[2]["ordinal_start"] == 712
    assert families[2]["ordinal_end_exclusive"] == 37_576
    assert state["supermode_semantics"]["layer_is_generator_identity"] is False


def test_catalog_and_protocol_close_the_same_physical_universe() -> None:
    catalog = build_gemma3_parts_catalog()
    protocol = build_parts_first_supermode_protocol()
    universe = protocol["physical_atom_universe"]
    accounting = protocol["scalar_accounting"]
    assert catalog.total_physical_atom_count == universe["physical_atom_count"]
    assert catalog.atom_owned_scalar_count == accounting["atom_owned_scalars"]
    assert (
        catalog.shared_substrate_scalar_count
        == accounting["shared_substrate_scalars"]
    )
    assert (
        catalog.source_unique_parameter_count
        == accounting["source_unique_stored_scalars"]
    )
    assert (
        "sorted_unique_uint32_le.v1"
        in protocol["group_ontology"]["membership_encoding"]
    )


def test_lexical_rows_are_views_not_physical_generator_parts() -> None:
    universe = build_parts_first_supermode_protocol()[
        "physical_atom_universe"
    ]
    assert universe["vocabulary_rows_are_semantic_views_only"] is True
    assert universe["vocabulary_row_views_receive_storage_credit"] is False
    assert universe["vocabulary_row_views_may_be_supermode_members"] is False
    lexical = universe["families"][0]
    assert lexical["owned_slice"] == "E[:,g]"
    assert lexical["storage_counted_once_across_roles"] is True


def test_schedule_has_39_events_but_only_38_atom_emitting_roles() -> None:
    schedule = build_parts_first_supermode_protocol()["event_schedule"]
    assert len(schedule) == 39
    assert [event["event_index"] for event in schedule] == list(range(39))
    assert sum(event["atom_emitting"] for event in schedule) == 38
    assert schedule[0]["event_id"] == "lexical.ingress"
    assert schedule[0]["functional_role_id"] == "lexical_ingress"
    assert schedule[1]["physical_atom_family_id"] == "attention_qo_head"
    assert schedule[1]["functional_role_id"] == "attention_qo_contribution"
    assert schedule[37]["event_id"] == "terminal.rmsnorm"
    assert schedule[37]["atom_emitting"] is False
    assert schedule[38]["event_id"] == "lexical.raw_egress"


def test_protocol_requires_ordered_program_not_cross_boundary_sum() -> None:
    state = build_parts_first_supermode_protocol()
    equations = state["exact_operator_equations"]
    semantics = state["supermode_semantics"]
    assert equations["cross_event_outputs_may_be_raw_summed"] is False
    assert semantics["native_post_boundary_and_later_events_rerun_after_each_gate"]
    assert semantics["cross_layer_claim_requires_two_distinct_layer_provenances"]
    assert semantics[
        "attention_chart_state_binds_sequence_KV_mask_position_and_window"
    ]
    assert semantics["primary_intervention_freezes_baseline_chart_activation_mask"]


def test_protocol_is_prospective_and_makes_no_compression_claim() -> None:
    state = build_parts_first_supermode_protocol()
    assert state["status"] == "prospective_not_run"
    boundary = state["claim_boundary"]
    for key in (
        "compression_claim_allowed",
        "parameter_savings_claim_allowed",
        "MAC_savings_claim_allowed",
        "latency_claim_allowed",
        "interchangeable_generator_claim_allowed",
        "invariant_circuit_ontology_claim_allowed",
    ):
        assert boundary[key] is False


def test_render_is_canonical_and_semantically_self_hashed() -> None:
    state = build_parts_first_supermode_protocol()
    rendered = render_parts_first_supermode_protocol(state)
    assert json.loads(rendered) == state
    assert rendered.endswith(b"\n")
    assert state["protocol_sha256"] == protocol_semantic_sha256(state)


def test_coordinated_field_mutation_does_not_validate() -> None:
    state = build_parts_first_supermode_protocol()
    state["claim_boundary"]["compression_claim_allowed"] = True
    state["protocol_sha256"] = protocol_semantic_sha256(state)
    with pytest.raises(ValueError, match="frozen contract"):
        validate_parts_first_supermode_protocol(state)


def test_missing_or_extra_fields_fail_closed() -> None:
    missing = deepcopy(build_parts_first_supermode_protocol())
    missing.pop("event_schedule")
    with pytest.raises(ValueError, match="field inventory"):
        validate_parts_first_supermode_protocol(missing)
    extra = build_parts_first_supermode_protocol()
    extra["unreviewed"] = True
    with pytest.raises(ValueError, match="field inventory"):
        validate_parts_first_supermode_protocol(extra)
