from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
from itertools import combinations

import pytest

import fisher_graph.gemma3_generator_hierarchy_nomination as nomination_module
from fisher_graph.gemma3_generator_causal_map_artifact import (
    GEMMA3_GENERATOR_CAUSAL_MAP_FORMAT_VERSION,
    GEMMA3_GENERATOR_CAUSAL_MAP_SCHEMA,
)
from fisher_graph.gemma3_generator_hierarchy_nomination import (
    FINITE_INTERVENTION_RESPONSE_EVIDENCE,
    CausalGroupNomination,
    SharingFamilyNomination,
    known_v1_gemma3_generator_hierarchy_specs,
    nominate_gemma3_generator_hierarchy,
    nominate_known_v1_gemma3_generator_hierarchy,
)


_AUTHORITY_FIELDS = (
    "authorizes_merge",
    "authorizes_pruning",
    "authorizes_routing",
    "authorizes_compilation",
    "authorizes_execution",
    "authorizes_mutation",
)


@pytest.fixture(autouse=True)
def _unit_fixture_stands_in_for_strict_source_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter's own tests use a reduced map after the auth boundary.

    Production calls still execute the complete public causal-map validator;
    that validator is covered with the full artifact fixture in its own test
    module and by the live integration replay.
    """

    monkeypatch.setattr(
        nomination_module,
        "validate_gemma3_generator_causal_map_payload",
        lambda payload: None,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _no_authority() -> dict[str, bool]:
    return {field: False for field in _AUTHORITY_FIELDS}


def _strict_loaded_map(
    *,
    scientific_payload_sha256: str | None = None,
) -> dict[str, object]:
    nodes = []
    for layer in range(18):
        nodes.append(
            {
                "layer_ordinal": layer,
                "generator_id": f"generator/L{layer:02d}",
                "observational_only": True,
                **_no_authority(),
            }
        )
    edges = []
    for edge_ordinal, (upstream, downstream) in enumerate(
        combinations(range(18), 2)
    ):
        response = 0.001 * (edge_ordinal + 1)
        edges.append(
            {
                "upstream_layer_ordinal": upstream,
                "downstream_layer_ordinal": downstream,
                "mean_directed_response_rms": response,
                "maximum_directed_response_rms": response * 2.0,
                "mean_directed_response_ratio_over_defined": response / 2.0,
                "strict_upstream_invariance_confirmed": True,
                "observational_only": True,
                **_no_authority(),
            }
        )
    return {
        "schema": GEMMA3_GENERATOR_CAUSAL_MAP_SCHEMA,
        "format_version": GEMMA3_GENERATOR_CAUSAL_MAP_FORMAT_VERSION,
        "scientific_payload_sha256": (
            scientific_payload_sha256 or _sha("causal-map")
        ),
        "scientific_status": {
            "observational_metrics_only": True,
            **_no_authority(),
        },
        "safety": {
            "analysis_only": True,
            **_no_authority(),
        },
        "generator_nodes": nodes,
        "directed_edges": edges,
    }


def test_known_v1_nominates_complete_partition_and_separate_family() -> None:
    causal_map = _strict_loaded_map()

    nomination = nominate_known_v1_gemma3_generator_hierarchy(causal_map)

    assert nomination.source_scientific_payload_sha256 == causal_map[
        "scientific_payload_sha256"
    ]
    assert len(nomination.parents) == 17
    assert tuple(
        layer
        for parent in nomination.parents
        for layer in parent.child_layer_ordinals
    ) == tuple(range(18))
    contracted = next(
        parent
        for parent in nomination.parents
        if parent.parent_id == "causal_parent/L03-L04"
    )
    assert contracted.child_layer_ordinals == (3, 4)
    assert contracted.generator_ids == ("generator/L03", "generator/L04")
    assert contracted.kind == "causal_contraction_candidate"
    assert nomination.internal_edge_count == 1
    assert nomination.surfaced_cut_edge_count == 152

    assert len(nomination.sharing_families) == 1
    family = nomination.sharing_families[0]
    assert family.family_id == "sharing_family/L12-L15"
    assert family.child_layer_ordinals == (12, 15)
    assert family.generator_ids == ("generator/L12", "generator/L15")
    assert all(
        12 not in parent.child_layer_ordinals
        or parent.child_layer_ordinals == (12,)
        for parent in nomination.parents
    )
    assert all(
        15 not in parent.child_layer_ordinals
        or parent.child_layer_ordinals == (15,)
        for parent in nomination.parents
    )


def test_classifies_every_source_edge_exactly_once_and_binds_it() -> None:
    nomination = nominate_known_v1_gemma3_generator_hierarchy(
        _strict_loaded_map()
    )

    pairs = tuple(
        (
            edge.upstream_layer_ordinal,
            edge.downstream_layer_ordinal,
        )
        for edge in nomination.directed_edges
    )
    assert pairs == tuple(combinations(range(18), 2))
    assert len(set(pairs)) == 153
    internal = [
        edge
        for edge in nomination.directed_edges
        if edge.disposition == "internal"
    ]
    assert [
        (
            edge.upstream_layer_ordinal,
            edge.downstream_layer_ordinal,
        )
        for edge in internal
    ] == [(3, 4)]
    assert all(
        edge.disposition in {"internal", "surfaced_cut"}
        and len(edge.source_edge_sha256) == 64
        and edge.evidence_semantics
        == FINITE_INTERVENTION_RESPONSE_EVIDENCE
        and edge.jacobian_estimate is False
        and edge.observational_only is True
        and edge.authorizes_execution is False
        for edge in nomination.directed_edges
    )


def test_larger_interval_internalizes_all_and_only_intra_parent_edges() -> None:
    nomination = nominate_gemma3_generator_hierarchy(
        _strict_loaded_map(),
        causal_groups=(
            CausalGroupNomination(
                parent_id="causal_parent/L02-L04",
                child_layer_ordinals=(2, 3, 4),
            ),
        ),
        sharing_families=(
            SharingFamilyNomination(
                family_id="sharing_family/nonlocal",
                child_layer_ordinals=(1, 9, 17),
            ),
        ),
    )

    assert len(nomination.parents) == 16
    assert nomination.internal_edge_count == 3
    assert {
        (
            edge.upstream_layer_ordinal,
            edge.downstream_layer_ordinal,
        )
        for edge in nomination.directed_edges
        if edge.disposition == "internal"
    } == {(2, 3), (2, 4), (3, 4)}
    assert nomination.sharing_families[
        0
    ].child_layer_ordinals == (1, 9, 17)


def test_specs_and_results_are_immutable_hashed_and_deterministic() -> None:
    causal_groups, sharing_families = (
        known_v1_gemma3_generator_hierarchy_specs()
    )
    first = nominate_gemma3_generator_hierarchy(
        _strict_loaded_map(),
        causal_groups=causal_groups,
        sharing_families=sharing_families,
    )
    second = nominate_gemma3_generator_hierarchy(
        _strict_loaded_map(),
        causal_groups=causal_groups,
        sharing_families=sharing_families,
    )
    changed_source = nominate_gemma3_generator_hierarchy(
        _strict_loaded_map(
            scientific_payload_sha256=_sha("different-causal-map")
        ),
        causal_groups=causal_groups,
        sharing_families=sharing_families,
    )

    assert first.nomination_sha256 == second.nomination_sha256
    assert first.nomination_sha256 != changed_source.nomination_sha256
    assert len(causal_groups[0].nomination_sha256) == 64
    assert len(first.parents[0].nomination_sha256) == 64
    assert len(first.sharing_families[0].nomination_sha256) == 64
    assert len(first.directed_edges[0].nomination_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        first.internal_edge_count = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.parents[0].parent_id = "changed"  # type: ignore[misc]


def test_rejects_noncontiguous_or_overlapping_causal_groups() -> None:
    with pytest.raises(ValueError, match="causally contiguous"):
        CausalGroupNomination(
            parent_id="bad/noncontiguous",
            child_layer_ordinals=(2, 4),
        )

    with pytest.raises(ValueError, match="must be disjoint"):
        nominate_gemma3_generator_hierarchy(
            _strict_loaded_map(),
            causal_groups=(
                CausalGroupNomination(
                    parent_id="overlap/a",
                    child_layer_ordinals=(2, 3, 4),
                ),
                CausalGroupNomination(
                    parent_id="overlap/b",
                    child_layer_ordinals=(4, 5),
                ),
            ),
        )


def test_rejects_jacobian_label_or_execution_authority() -> None:
    causal_map = _strict_loaded_map()

    with pytest.raises(ValueError, match="cannot be labeled as Jacobians"):
        nominate_gemma3_generator_hierarchy(
            causal_map,
            evidence_semantics="jacobian",
        )
    with pytest.raises(ValueError, match="cannot be labeled as Jacobians"):
        nominate_gemma3_generator_hierarchy(
            causal_map,
            jacobian_interpretation=True,
        )
    with pytest.raises(ValueError, match="cannot authorize execution"):
        nominate_gemma3_generator_hierarchy(
            causal_map,
            authorizes_execution=True,
        )
    with pytest.raises(ValueError, match="cannot authorize execution"):
        SharingFamilyNomination(
            family_id="unsafe/family",
            child_layer_ordinals=(12, 15),
            authorizes_execution=True,
        )


def test_rejects_source_authority_or_non_observational_rows() -> None:
    authority = _strict_loaded_map()
    authority["directed_edges"][0]["authorizes_execution"] = True
    with pytest.raises(ValueError, match="forbidden optimization authority"):
        nominate_gemma3_generator_hierarchy(authority)

    non_observational = _strict_loaded_map()
    non_observational["generator_nodes"][4]["observational_only"] = False
    with pytest.raises(ValueError, match="node 4 is inconsistent"):
        nominate_gemma3_generator_hierarchy(non_observational)

    bad_catalog = _strict_loaded_map()
    bad_catalog["directed_edges"] = copy.deepcopy(
        bad_catalog["directed_edges"]
    )
    bad_catalog["directed_edges"][0]["downstream_layer_ordinal"] = 2
    with pytest.raises(ValueError, match="edge 0->1 is inconsistent"):
        nominate_gemma3_generator_hierarchy(bad_catalog)
