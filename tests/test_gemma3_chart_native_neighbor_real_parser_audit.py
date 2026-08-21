from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_chart_native_neighbor_experiment as worker
from fisher_graph.chart_energy_neighbor_graph import (
    FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL,
    ChartEnergyNeighborConfig,
    ChartEnergyNeighborInput,
    ChartExampleStats,
    ChartWithinFamilyBootstrapConfig,
    DirectedChartTransportStats,
    canonicalize_chart_energy_neighbor_input,
    parse_chart_energy_neighbor_input_payload,
)
from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import load_manifest
from fisher_graph.gemma3_chart_native_neighbor_setup import build_setup
from fisher_graph.shared_chart_energy_transport import (
    ChartRowKey,
    build_within_example_row_shuffle,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _AuditFixture:
    source: ChartEnergyNeighborInput
    primary: ChartEnergyNeighborInput
    setup: dict[str, object]
    manifest: dict[str, object]
    eligibility: worker.RoleEligibilityCatalog
    runtime_lineage: tuple[worker.RuntimeModeLineage, ...]


@pytest.fixture(scope="module")
def real_parser_audit_fixture() -> _AuditFixture:
    setup = build_setup()
    manifest = load_manifest()
    role = worker.DISCOVERY_ROLE
    eligibility = worker.build_role_eligibility_catalog(manifest, role=role)
    runtime_lineage = worker.derive_runtime_mode_lineage(
        setup_modes=setup["logical_mode_ontology"]["ordered_modes"],
        state_fingerprints_by_mode={
            row["mode_id"]: _sha(f"state:{row['mode_id']}")
            for row in setup["logical_mode_ontology"]["ordered_modes"]
        },
        hidden_width=640,
    )
    modes = worker.derive_core_mode_specs(
        setup=setup,
        mode_lineage=runtime_lineage,
    )
    role_manifest = manifest["roles"][role]
    members = {
        (row["family_id"], row["example_id"]): row
        for row in role_manifest["ordered_members"]
    }
    panel = worker.role_panel_binding_sha256(manifest, role)
    atlas = _sha("A-atlas")
    fit_catalog = _sha("A-transport-catalog")
    split = _sha("A-B-split")
    comparison_sha = modes[0].comparison_abi.sha256

    stats: list[ChartExampleStats] = []
    stats_by_key: dict[tuple[str, str, str], ChartExampleStats] = {}
    row_catalog_by_example: dict[tuple[str, str], str] = {}
    row_count_by_example: dict[tuple[str, str], int] = {}
    all_role_row_keys: list[ChartRowKey] = []
    for family_id, example_ids in eligibility.all_family_example_ids:
        for example_id in example_ids:
            member = members[(family_id, example_id)]
            row_count = int(member["prediction_row_count"])
            assert row_count == int(member["target_token_count"]) - 1
            row_keys = tuple(
                ChartRowKey(family_id, example_id, query, query + 1)
                for query in range(row_count)
            )
            all_role_row_keys.extend(row_keys)
            row_catalog = worker._example_row_catalog_sha256(
                role=role,
                row_keys=row_keys,
            )
            row_catalog_by_example[(family_id, example_id)] = row_catalog
            row_count_by_example[(family_id, example_id)] = row_count
            energy = float(row_count)
            for mode in modes:
                route_required = mode.mode_kind == "attention"
                row = ChartExampleStats(
                    family_id=family_id,
                    example_id=example_id,
                    mode_id=mode.mode_id,
                    comparison_abi_sha256=comparison_sha,
                    chart_atlas_sha256=atlas,
                    aligned_row_catalog_sha256=row_catalog,
                    chart_energy_vector_catalog_sha256=_sha(
                        f"vector:{family_id}:{example_id}:{mode.mode_id}"
                    ),
                    row_count=row_count,
                    occupied_row_count=row_count,
                    minimum_row_fisher_energy=1.0,
                    weighted_activation_energy=energy,
                    weighted_chart_tangent_energy=0.5 * energy,
                    weighted_chart_orthogonal_energy=0.5 * energy,
                    weighted_fisher_energy=energy,
                    supplemental_route_catalog_sha256=(
                        _sha(f"route-catalog:{example_id}:{mode.mode_id}")
                        if route_required
                        else None
                    ),
                    supplemental_route_receipt_sha256=(
                        _sha(f"route-receipt:{example_id}:{mode.mode_id}")
                        if route_required
                        else None
                    ),
                )
                stats.append(row)
                stats_by_key[(family_id, example_id, mode.mode_id)] = row

    shuffle = build_within_example_row_shuffle(
        all_role_row_keys,
        role=role,
    ).catalog_sha256
    directed: list[DirectedChartTransportStats] = []
    for family_id, example_ids in eligibility.all_family_example_ids:
        for example_id in example_ids:
            row_catalog = row_catalog_by_example[(family_id, example_id)]
            row_count = row_count_by_example[(family_id, example_id)]
            energy = float(row_count)
            # Deliberately use target-major setup order.  The public
            # canonicalizer must publish family/example/source/target order.
            for target in modes:
                for source in modes:
                    if source.mode_id == target.mode_id:
                        continue
                    directed.append(
                        DirectedChartTransportStats(
                            family_id=family_id,
                            example_id=example_id,
                            source_mode_id=source.mode_id,
                            target_mode_id=target.mode_id,
                            comparison_abi_sha256=comparison_sha,
                            aligned_row_catalog_sha256=row_catalog,
                            source_chart_stats_sha256=stats_by_key[
                                (family_id, example_id, source.mode_id)
                            ].sha256,
                            target_chart_stats_sha256=stats_by_key[
                                (family_id, example_id, target.mode_id)
                            ].sha256,
                            row_count=row_count,
                            transported_row_count=row_count,
                            transport_fit_role=worker.FIT_ROLE,
                            transport_evaluation_role=role,
                            transport_fit_catalog_sha256=fit_catalog,
                            transport_evaluation_catalog_sha256=row_catalog,
                            transport_evaluation_panel_sha256=panel,
                            transport_split_receipt_sha256=split,
                            transport_fit_receipt_sha256=_sha(
                                f"fit:{source.mode_id}:{target.mode_id}"
                            ),
                            row_shuffle_catalog_sha256=shuffle,
                            transport_protocol=(
                                FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL
                            ),
                            weighted_transport_residual_energy=energy,
                            weighted_predicted_fisher_energy=energy,
                            weighted_target_fisher_energy=energy,
                            weighted_predicted_target_cross_inner_product=(
                                0.5 * energy
                            ),
                            weighted_zero_prediction_residual_energy=energy,
                            weighted_row_shuffled_transport_residual_energy=(
                                2.0 * energy
                            ),
                        )
                    )

    bootstrap_contract = setup["fresh_confirmation_gates"][
        "within_family_bootstrap"
    ]
    graph_contract = setup["neighbor_graph_protocol"]
    config = ChartEnergyNeighborConfig(
        k=int(graph_contract["k"]),
        rank_boundary_tie_absolute_tolerance=float(
            graph_contract["rank_boundary_tie_absolute_tolerance"]
        ),
        maximum_directional_normalized_residual=float(
            graph_contract["maximum_each_directional_normalized_residual"]
        ),
        bootstrap_minimum_mutual_frequency=float(
            bootstrap_contract["minimum_mutual_frequency"]
        ),
        lofo_minimum_mutual_count=int(
            setup["fresh_confirmation_gates"]["leave_one_family_out"][
                "minimum_mutual_count"
            ]
        ),
        lofo_expected_replicate_count=int(
            setup["fresh_confirmation_gates"]["leave_one_family_out"][
                "replicate_count"
            ]
        ),
    )
    raw_source = ChartEnergyNeighborInput(
        panel_role=role,
        panel_binding_sha256=panel,
        mode_specs=modes,
        example_stats=tuple(stats),
        directed_transport_stats=tuple(directed),
        within_family_bootstrap_config=ChartWithinFamilyBootstrapConfig(
            seed=int(bootstrap_contract["seed"]),
            resample_count=int(bootstrap_contract["resample_count"]),
            draws_per_family=8,
            maximum_degenerate_fraction=float(
                bootstrap_contract["maximum_degenerate_fraction"]
            ),
            family_example_ids=eligibility.all_family_example_ids,
            draw_protocol_domain=bytes.fromhex(
                bootstrap_contract["draw_protocol_domain_hex"]
            ),
        ),
        config=config,
    )
    canonical = canonicalize_chart_energy_neighbor_input(raw_source)
    parsed = parse_chart_energy_neighbor_input_payload(
        canonical.canonical_payload()
    )
    primary = parse_chart_energy_neighbor_input_payload(
        worker._primary_input_from_all_eight(
            parsed,
            eligibility,
        ).canonical_payload()
    )
    return _AuditFixture(
        source=parsed,
        primary=primary,
        setup=setup,
        manifest=manifest,
        eligibility=eligibility,
        runtime_lineage=runtime_lineage,
    )


def test_real_public_parser_closes_all8_primary_and_lineage_orders(
    real_parser_audit_fixture: _AuditFixture,
) -> None:
    fixture = real_parser_audit_fixture
    source = fixture.source
    graph_axis = tuple(mode.mode_id for mode in source.mode_specs)
    runtime_axis = tuple(row.mode_id for row in fixture.runtime_lineage)
    assert graph_axis == tuple(sorted(runtime_axis))
    assert graph_axis != runtime_axis
    assert len(source.example_stats) == 38 * 48
    assert len(source.directed_transport_stats) == 38 * 37 * 48
    expected_role_rows = int(
        fixture.manifest["roles"][worker.DISCOVERY_ROLE][
            "prediction_row_count"
        ]
    )
    for mode_id in graph_axis:
        assert sum(
            row.row_count
            for row in source.example_stats
            if row.mode_id == mode_id
        ) == expected_role_rows

    reparsed_primary = fixture.primary
    assert tuple(mode.mode_id for mode in reparsed_primary.mode_specs) == graph_axis
    assert len(reparsed_primary.example_stats) == 38 * 42
    assert len(reparsed_primary.directed_transport_stats) == 38 * 37 * 42
    assert reparsed_primary.config == source.config
    assert reparsed_primary.within_family_bootstrap_config.resample_count == 10_000
    assert reparsed_primary.within_family_bootstrap_config.draws_per_family == 7
    assert (
        reparsed_primary.within_family_bootstrap_config.family_example_ids
        == fixture.eligibility.primary_family_example_ids
    )
    worker._validate_role_graph_input_contract(
        setup=fixture.setup,
        role_manifest=fixture.manifest["roles"][worker.DISCOVERY_ROLE],
        eligibility=fixture.eligibility,
        all_eight_source=source,
        primary_source=reparsed_primary,
        capture_row_count=expected_role_rows,
    )

    qualified = worker.derive_core_mode_specs(
        setup=fixture.setup,
        mode_lineage=fixture.runtime_lineage,
    )
    observed = worker._validate_completed_mode_lineage(
        mode_axis=graph_axis,
        B_source=source,
        C_source=source,
        setup=fixture.setup,
        runtime_qualification=SimpleNamespace(
            mode_lineage=fixture.runtime_lineage
        ),
        runtime_receipt=SimpleNamespace(mode_lineage=fixture.runtime_lineage),
        A_mode_lineage_catalog_sha256=(
            worker._core_mode_lineage_catalog_sha256(qualified)
        ),
    )
    assert observed == qualified


def test_real_parser_evaluation_catalog_is_order_invariant(
    real_parser_audit_fixture: _AuditFixture,
) -> None:
    source = real_parser_audit_fixture.source
    capture = SimpleNamespace(receipt_sha256=_sha("B-capture"))
    expected = worker._recompute_role_evaluation_catalog(
        source=source,
        capture=capture,
    )
    reordered = replace(
        source,
        example_stats=tuple(reversed(source.example_stats)),
        directed_transport_stats=tuple(
            reversed(source.directed_transport_stats)
        ),
    )
    assert worker._recompute_role_evaluation_catalog(
        source=reordered,
        capture=capture,
    ) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("k", 3),
        ("rank_boundary_tie_absolute_tolerance", 1.0e-8),
        ("fisher_energy_epsilon", 1.0e-9),
        ("maximum_directional_normalized_residual", 2.0),
        ("bootstrap_minimum_mutual_frequency", 0.5),
        ("lofo_minimum_mutual_count", 4),
    ),
)
def test_completed_graph_contract_rejects_coordinated_policy_tamper(
    real_parser_audit_fixture: _AuditFixture,
    field: str,
    value: object,
) -> None:
    fixture = real_parser_audit_fixture
    tampered_config = replace(fixture.source.config, **{field: value})
    with pytest.raises(ValueError, match="policy/bootstrap"):
        worker._validate_role_graph_input_contract(
            setup=fixture.setup,
            role_manifest=fixture.manifest["roles"][worker.DISCOVERY_ROLE],
            eligibility=fixture.eligibility,
            all_eight_source=replace(
                fixture.source,
                config=tampered_config,
            ),
            primary_source=replace(
                fixture.primary,
                config=tampered_config,
            ),
            capture_row_count=24_732,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "seed",
        "resample_count",
        "maximum_degenerate_fraction",
        "family_order",
        "example_order",
    ),
)
def test_completed_graph_contract_rejects_coordinated_bootstrap_tamper(
    real_parser_audit_fixture: _AuditFixture,
    tamper: str,
) -> None:
    fixture = real_parser_audit_fixture
    all_changes: dict[str, object] = {}
    primary_changes: dict[str, object] = {}
    if tamper == "seed":
        all_changes["seed"] = fixture.source.within_family_bootstrap_config.seed + 1
        primary_changes.update(all_changes)
    elif tamper == "resample_count":
        all_changes["resample_count"] = 9_999
        primary_changes.update(all_changes)
    elif tamper == "maximum_degenerate_fraction":
        all_changes["maximum_degenerate_fraction"] = 0.5
        primary_changes.update(all_changes)
    elif tamper == "family_order":
        all_changes["family_example_ids"] = tuple(
            reversed(
                fixture.source.within_family_bootstrap_config.family_example_ids
            )
        )
        primary_changes["family_example_ids"] = tuple(
            reversed(
                fixture.primary.within_family_bootstrap_config.family_example_ids
            )
        )
    else:
        assert tamper == "example_order"
        all_changes["family_example_ids"] = tuple(
            (family, tuple(reversed(examples)))
            for family, examples in (
                fixture.source.within_family_bootstrap_config.family_example_ids
            )
        )
        primary_changes["family_example_ids"] = tuple(
            (family, tuple(reversed(examples)))
            for family, examples in (
                fixture.primary.within_family_bootstrap_config.family_example_ids
            )
        )
    tampered_all_bootstrap = replace(
        fixture.source.within_family_bootstrap_config,
        **all_changes,
    )
    tampered_primary_bootstrap = replace(
        fixture.primary.within_family_bootstrap_config,
        **primary_changes,
    )
    with pytest.raises(ValueError, match="policy/bootstrap"):
        worker._validate_role_graph_input_contract(
            setup=fixture.setup,
            role_manifest=fixture.manifest["roles"][worker.DISCOVERY_ROLE],
            eligibility=fixture.eligibility,
            all_eight_source=replace(
                fixture.source,
                within_family_bootstrap_config=tampered_all_bootstrap,
            ),
            primary_source=replace(
                fixture.primary,
                within_family_bootstrap_config=tampered_primary_bootstrap,
            ),
            capture_row_count=24_732,
        )
