from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import fisher_graph.gemma3_chart_native_neighbor_experiment as worker
from fisher_graph.chart_energy_neighbor_graph import (
    CAUSAL_SEQUENCE_NATIVE_ABI,
    FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL,
    ChartEnergyNeighborInput,
    ChartExampleStats,
    DirectedChartTransportStats,
    parse_chart_energy_neighbor_input_payload,
)
from fisher_graph.gemma3_chart_native_neighbor_setup import build_setup
from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
    DEFAULT_MANIFEST,
    load_manifest,
)
from fisher_graph.shared_chart_energy_transport import (
    ChartRowKey,
    build_within_example_row_shuffle,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _production_cardinality_real_input() -> tuple[
    ChartEnergyNeighborInput,
    dict[str, object],
    tuple[worker.RuntimeModeLineage, ...],
    dict[str, object],
]:
    setup = build_setup()
    manifest = load_manifest(DEFAULT_MANIFEST)
    lineage = worker.derive_runtime_mode_lineage(
        setup_modes=setup["logical_mode_ontology"]["ordered_modes"],
        state_fingerprints_by_mode={
            row["mode_id"]: _sha(f"state:{row['mode_id']}")
            for row in setup["logical_mode_ontology"]["ordered_modes"]
        },
        hidden_width=640,
    )
    modes = worker.derive_core_mode_specs(
        setup=setup,
        mode_lineage=lineage,
    )
    role = worker.DISCOVERY_ROLE
    role_manifest = manifest["roles"][role]
    panel = worker.role_panel_binding_sha256(manifest, role)
    atlas = _sha("A-atlas")
    fit_catalog = _sha("A-transport-catalog")
    split = _sha("A-B-split")
    comparison_sha = modes[0].comparison_abi.sha256
    members = tuple(role_manifest["ordered_members"])
    ordered_families = tuple(
        dict.fromkeys(str(member["family_id"]) for member in members)
    )
    families = tuple(
        (
            family_id,
            tuple(
                str(member["example_id"])
                for member in members
                if str(member["family_id"]) == family_id
            ),
        )
        for family_id in ordered_families
    )
    member_by_example = {
        str(member["example_id"]): member for member in members
    }
    row_keys_by_example: dict[str, tuple[ChartRowKey, ...]] = {}
    all_role_row_keys: list[ChartRowKey] = []
    for member in members:
        family_id = str(member["family_id"])
        example_id = str(member["example_id"])
        row_count = int(member["prediction_row_count"])
        keys = tuple(
            ChartRowKey(family_id, example_id, query, query + 1)
            for query in range(row_count)
        )
        row_keys_by_example[example_id] = keys
        all_role_row_keys.extend(keys)
    shuffle = build_within_example_row_shuffle(
        all_role_row_keys,
        role=role,
    ).catalog_sha256

    stats: list[ChartExampleStats] = []
    stats_by_key: dict[tuple[str, str, str], ChartExampleStats] = {}
    for family_id, example_ids in families:
        for example_id in example_ids:
            row_count = int(member_by_example[example_id]["prediction_row_count"])
            row_catalog = worker._example_row_catalog_sha256(
                role=role,
                row_keys=row_keys_by_example[example_id],
            )
            for mode in modes:
                route_required = (
                    mode.native_abi.abi_kind == CAUSAL_SEQUENCE_NATIVE_ABI
                )
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
                    weighted_activation_energy=float(row_count),
                    weighted_chart_tangent_energy=0.5 * row_count,
                    weighted_chart_orthogonal_energy=0.5 * row_count,
                    weighted_fisher_energy=float(row_count),
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

    directed: list[DirectedChartTransportStats] = []
    for family_id, example_ids in families:
        for example_id in example_ids:
            row_count = int(member_by_example[example_id]["prediction_row_count"])
            row_catalog = worker._example_row_catalog_sha256(
                role=role,
                row_keys=row_keys_by_example[example_id],
            )
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
                            weighted_transport_residual_energy=float(row_count),
                            weighted_predicted_fisher_energy=float(row_count),
                            weighted_target_fisher_energy=float(row_count),
                            weighted_predicted_target_cross_inner_product=(
                                0.5 * row_count
                            ),
                            weighted_zero_prediction_residual_energy=float(
                                row_count
                            ),
                            weighted_row_shuffled_transport_residual_energy=(
                                2.0 * row_count
                            ),
                        )
                    )
    graph_input = ChartEnergyNeighborInput(
        panel_role=role,
        panel_binding_sha256=panel,
        # Setup/event order intentionally differs from lexical graph order.
        mode_specs=modes,
        example_stats=tuple(stats),
        directed_transport_stats=tuple(directed),
        within_family_bootstrap_config=worker._frozen_chart_bootstrap_config(
            setup,
            family_example_ids=families,
        ),
        config=worker._frozen_chart_neighbor_config(setup),
    )
    return graph_input, setup, lineage, manifest


def test_real_core_parser_closes_graph_order_and_evaluation_catalog() -> None:
    graph_input, setup, lineage, manifest = _production_cardinality_real_input()
    parsed = parse_chart_energy_neighbor_input_payload(
        graph_input.canonical_payload()
    )
    graph_axis = tuple(mode.mode_id for mode in parsed.mode_specs)
    assert graph_axis == tuple(sorted(row.mode_id for row in lineage))
    assert graph_axis != tuple(row.mode_id for row in lineage)

    qualified = worker.derive_core_mode_specs(
        setup=setup,
        mode_lineage=lineage,
    )
    runtime_receipt = SimpleNamespace(mode_lineage=lineage)
    observed = worker._validate_completed_mode_lineage(
        mode_axis=graph_axis,
        B_source=parsed,
        C_source=parsed,
        setup=setup,
        runtime_qualification=SimpleNamespace(mode_lineage=lineage),
        runtime_receipt=runtime_receipt,
        A_mode_lineage_catalog_sha256=(
            worker._core_mode_lineage_catalog_sha256(qualified)
        ),
    )
    assert observed == qualified

    eligibility = worker.build_role_eligibility_catalog(
        manifest,
        role=worker.DISCOVERY_ROLE,
    )
    primary = worker._primary_input_from_all_eight(parsed, eligibility)
    assert len(primary.example_stats) == 38 * 42
    assert len(primary.directed_transport_stats) == 38 * 37 * 42
    worker._validate_role_graph_input_contract(
        setup=setup,
        role_manifest=manifest["roles"][worker.DISCOVERY_ROLE],
        eligibility=eligibility,
        all_eight_source=parsed,
        primary_source=primary,
        capture_row_count=24_732,
    )
    assert primary.within_family_bootstrap_config.draws_per_family == 7

    capture = SimpleNamespace(receipt_sha256=_sha("B-capture"))
    canonical_catalog = worker._recompute_role_evaluation_catalog(
        source=parsed,
        capture=capture,
    )
    reordered = replace(
        parsed,
        example_stats=tuple(reversed(parsed.example_stats)),
        directed_transport_stats=tuple(
            reversed(parsed.directed_transport_stats)
        ),
    )
    assert worker._recompute_role_evaluation_catalog(
        source=reordered,
        capture=capture,
    ) == canonical_catalog
