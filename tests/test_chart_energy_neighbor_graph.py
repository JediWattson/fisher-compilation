from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import pytest

from fisher_graph.chart_energy_neighbor_graph import (
    CAUSAL_SEQUENCE_NATIVE_ABI,
    DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
    EMBEDDING_CARRIER_H640_ADAPTER,
    FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL,
    LEXICAL_INGRESS_NATIVE_ABI,
    LEXICAL_READOUT_NATIVE_ABI,
    SOFTCAP_READOUT_PULLBACK_H640_ADAPTER,
    TOKEN_LOCAL_NATIVE_ABI,
    ChartComparisonABI,
    ChartEnergyDiagnosticSubsetReceipt,
    ChartEnergyNeighborConfig,
    ChartEnergyNeighborInput,
    ChartExampleStats,
    ChartModeSpec,
    DirectedChartTransportStats,
    NativeChartABI,
    ChartWithinFamilyBootstrapConfig,
    build_chart_energy_neighbor_receipt,
    build_chart_energy_diagnostic_subset_receipt,
    canonicalize_chart_energy_neighbor_input,
    canonical_carrier_comparison_abi,
    load_chart_energy_neighbor_receipt,
    parse_chart_energy_neighbor_input_payload,
    parse_chart_energy_neighbor_receipt_payload,
    parse_chart_energy_diagnostic_subset_receipt_payload,
    save_chart_energy_neighbor_receipt,
    _neighbor_topology,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_KINDS = {
    "mode-a": TOKEN_LOCAL_NATIVE_ABI,
    "mode-b": CAUSAL_SEQUENCE_NATIVE_ABI,
    "mode-c": LEXICAL_INGRESS_NATIVE_ABI,
    "mode-d": LEXICAL_READOUT_NATIVE_ABI,
}

_ADAPTERS = {
    TOKEN_LOCAL_NATIVE_ABI: DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
    CAUSAL_SEQUENCE_NATIVE_ABI: DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
    LEXICAL_INGRESS_NATIVE_ABI: EMBEDDING_CARRIER_H640_ADAPTER,
    LEXICAL_READOUT_NATIVE_ABI: SOFTCAP_READOUT_PULLBACK_H640_ADAPTER,
}

_MODE_KINDS = {
    TOKEN_LOCAL_NATIVE_ABI: "mlp",
    CAUSAL_SEQUENCE_NATIVE_ABI: "attention",
    LEXICAL_INGRESS_NATIVE_ABI: "lexical_ingress",
    LEXICAL_READOUT_NATIVE_ABI: "lexical_egress",
}

_ABI_NAMES = {
    TOKEN_LOCAL_NATIVE_ABI: "mlp.residual_h640.token_local.v1",
    CAUSAL_SEQUENCE_NATIVE_ABI: (
        "attention.residual_h640.causal_sequence.v1"
    ),
    LEXICAL_INGRESS_NATIVE_ABI: "lexical_ingress.lookup_to_h640.v1",
    LEXICAL_READOUT_NATIVE_ABI: (
        "lexical_egress.tied_vocab_softcap_readout.v1"
    ),
}


def _native(kind: str) -> NativeChartABI:
    return NativeChartABI(
        abi_kind=kind,
        abi_name=_ABI_NAMES[kind],
        abi_version=1,
        input_signature=f"{kind}.input",
        output_signature=f"{kind}.output",
        context_signature=f"{kind}.context",
    )


def _mode(mode_id: str, ordinal: int) -> ChartModeSpec:
    kind = _KINDS[mode_id]
    return ChartModeSpec(
        mode_id=mode_id,
        mode_kind=_MODE_KINDS[kind],
        state_fingerprint_sha256=_sha(f"state:{mode_id}"),
        event_id=f"event-{ordinal}",
        event_index=ordinal,
        context_site_id=f"context-site-{ordinal}",
        contribution_site_id=f"contribution-site-{ordinal}",
        site_placement_sha256=_sha(f"placement:{mode_id}"),
        native_abi=_native(kind),
        comparison_abi=canonical_carrier_comparison_abi(),
        comparison_adapter_kind=_ADAPTERS[kind],
        comparison_adapter_receipt_sha256=_sha(f"adapter:{mode_id}"),
    )


def _fixture(
    *,
    positions_by_family: dict[str, dict[str, float]] | None = None,
    examples_by_family: dict[str, tuple[str, ...]] | None = None,
    k: int = 1,
) -> ChartEnergyNeighborInput:
    mode_ids = tuple(_KINDS)
    families = examples_by_family or {
        "family-a": ("example-0", "example-1"),
        "family-b": ("example-0", "example-1"),
    }
    positions = positions_by_family or {
        "family-a": {
            "mode-a": 1.0,
            "mode-b": 1.1,
            "mode-c": 4.0,
            "mode-d": 4.2,
        },
        "family-b": {
            "mode-a": 1.0,
            "mode-b": 1.2,
            "mode-c": 4.0,
            "mode-d": 4.1,
        },
    }
    comparison_sha = canonical_carrier_comparison_abi().sha256
    chart_rows: list[ChartExampleStats] = []
    chart_by_key: dict[tuple[str, str, str], ChartExampleStats] = {}
    for family_id, example_ids in families.items():
        for example_ordinal, example_id in enumerate(example_ids):
            row_catalog = _sha(f"rows:{family_id}:{example_id}")
            example_scale = float(example_ordinal + 1)
            for mode_id in mode_ids:
                value = positions[family_id][mode_id] * example_scale
                energy = value * value
                causal = _KINDS[mode_id] == CAUSAL_SEQUENCE_NATIVE_ABI
                row = ChartExampleStats(
                    family_id=family_id,
                    example_id=example_id,
                    mode_id=mode_id,
                    comparison_abi_sha256=comparison_sha,
                    chart_atlas_sha256=_sha("shared-chart-atlas"),
                    aligned_row_catalog_sha256=row_catalog,
                    chart_energy_vector_catalog_sha256=_sha(
                        f"vectors:{family_id}:{example_id}:{mode_id}"
                    ),
                    row_count=8 + example_ordinal,
                    occupied_row_count=4 + example_ordinal,
                    minimum_row_fisher_energy=(
                        energy / (2.0 * (8 + example_ordinal))
                    ),
                    weighted_activation_energy=2.0 * energy,
                    weighted_chart_tangent_energy=0.75 * energy,
                    weighted_chart_orthogonal_energy=0.25 * energy,
                    weighted_fisher_energy=energy,
                    supplemental_route_catalog_sha256=(
                        _sha(f"route-catalog:{family_id}:{example_id}")
                        if causal
                        else None
                    ),
                    supplemental_route_receipt_sha256=(
                        _sha(
                            f"route-receipt:{family_id}:{example_id}:{mode_id}"
                        )
                        if causal
                        else None
                    ),
                )
                chart_rows.append(row)
                chart_by_key[(family_id, example_id, mode_id)] = row

    transports: list[DirectedChartTransportStats] = []
    for family_id, example_ids in families.items():
        for example_ordinal, example_id in enumerate(example_ids):
            scale = float(example_ordinal + 1)
            row_catalog = _sha(f"rows:{family_id}:{example_id}")
            for source in mode_ids:
                for target in mode_ids:
                    if source == target:
                        continue
                    source_value = positions[family_id][source] * scale
                    target_value = positions[family_id][target] * scale
                    source_energy = source_value**2
                    target_energy = target_value**2
                    cross = source_value * target_value
                    transports.append(
                        DirectedChartTransportStats(
                            family_id=family_id,
                            example_id=example_id,
                            source_mode_id=source,
                            target_mode_id=target,
                            comparison_abi_sha256=comparison_sha,
                            aligned_row_catalog_sha256=row_catalog,
                            source_chart_stats_sha256=chart_by_key[
                                (family_id, example_id, source)
                            ].sha256,
                            target_chart_stats_sha256=chart_by_key[
                                (family_id, example_id, target)
                            ].sha256,
                            row_count=8 + example_ordinal,
                            transported_row_count=8 + example_ordinal,
                            transport_fit_role="A_chart_transport_fit",
                            transport_evaluation_role="B_neighbor_discovery",
                            transport_fit_catalog_sha256=_sha("fit-catalog"),
                            transport_evaluation_catalog_sha256=row_catalog,
                            transport_evaluation_panel_sha256=_sha("panel"),
                            transport_split_receipt_sha256=_sha("split"),
                            transport_fit_receipt_sha256=_sha(
                                f"fit:{source}:{target}"
                            ),
                            row_shuffle_catalog_sha256=_sha(
                                "row-shuffle-catalog"
                            ),
                            transport_protocol=(
                                FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL
                            ),
                            weighted_transport_residual_energy=(
                                source_value - target_value
                            )
                            ** 2,
                            weighted_predicted_fisher_energy=source_energy,
                            weighted_target_fisher_energy=target_energy,
                            weighted_predicted_target_cross_inner_product=(
                                cross
                            ),
                            weighted_zero_prediction_residual_energy=(
                                target_energy
                            ),
                            weighted_row_shuffled_transport_residual_energy=(
                                (source_value - target_value) ** 2
                                + 0.5 * target_energy
                            ),
                        )
                    )
    return ChartEnergyNeighborInput(
        panel_role="B_neighbor_discovery",
        panel_binding_sha256=_sha("panel"),
        mode_specs=tuple(reversed([_mode(item, i) for i, item in enumerate(mode_ids)])),
        example_stats=tuple(reversed(chart_rows)),
        directed_transport_stats=tuple(reversed(transports)),
        within_family_bootstrap_config=ChartWithinFamilyBootstrapConfig(
            seed=7_901,
            resample_count=20,
            draws_per_family=len(next(iter(families.values()))),
            maximum_degenerate_fraction=1.0,
            family_example_ids=tuple(
                (family_id, example_ids)
                for family_id, example_ids in families.items()
            ),
        ),
        config=ChartEnergyNeighborConfig(
            k=k,
            bootstrap_minimum_mutual_frequency=0.0,
            lofo_minimum_mutual_count=1,
            lofo_expected_replicate_count=len(families),
        ),
    )


def _transport_index(
    graph_input: ChartEnergyNeighborInput,
    family: str,
    example: str,
    source: str,
    target: str,
) -> int:
    return next(
        index
        for index, row in enumerate(graph_input.directed_transport_stats)
        if (
            row.family_id,
            row.example_id,
            row.source_mode_id,
            row.target_mode_id,
        )
        == (family, example, source, target)
    )


def _replace_transport(
    graph_input: ChartEnergyNeighborInput,
    key: tuple[str, str, str, str],
    **changes: object,
) -> ChartEnergyNeighborInput:
    rows = list(graph_input.directed_transport_stats)
    index = _transport_index(graph_input, *key)
    rows[index] = replace(rows[index], **changes)
    return replace(graph_input, directed_transport_stats=tuple(rows))


def test_build_is_deterministic_and_cross_native_neighbors_are_allowed() -> None:
    graph_input = _fixture()
    receipt = build_chart_energy_neighbor_receipt(graph_input)
    rebuilt = build_chart_energy_neighbor_receipt(graph_input)
    assert receipt.canonical_payload() == rebuilt.canonical_payload()
    assert receipt.graph_input.mode_specs[0].mode_id == "mode-a"
    assert receipt.analysis["mode_axis"] == [
        "mode-a",
        "mode-b",
        "mode-c",
        "mode-d",
    ]
    topology = receipt.analysis["pooled"]["topology"]
    assert topology["mutual_edges"] == [
        ["mode-a", "mode-b"],
        ["mode-c", "mode-d"],
    ]
    native = receipt.analysis["native_abi_diagnostics"]
    assert len(native["cross_native_abi_mutual_edges"]) == 2
    assert all(
        item["within_partition_mutual_edges"] == []
        for item in native["partitions"]
    )


def test_diagnostic_subset_reuses_point_math_without_authorizing_edges() -> None:
    source = _fixture()
    selected = (
        ("family-a", ("example-0",)),
        ("family-b", ("example-0",)),
    )
    subset = build_chart_energy_diagnostic_subset_receipt(
        source,
        subset_id="boundary-transition-514",
        family_example_ids=selected,
    )
    assert isinstance(subset, ChartEnergyDiagnosticSubsetReceipt)
    assert subset.canonical_payload()["authorizes_edges"] is False
    assert subset.analysis["authorizes_edges"] is False
    assert subset.analysis["stability_status"] == (
        "not_computed_for_non_authorizing_diagnostic_subset"
    )
    assert sum(
        item["example_count"]
        for item in subset.analysis["family_receipts"]
    ) == 2

    selected_main = build_chart_energy_neighbor_receipt(
        _fixture(
            examples_by_family={
                "family-a": ("example-0",),
                "family-b": ("example-0",),
            }
        )
    )
    main_by_pair = {
        (item["left_mode_id"], item["right_mode_id"]): item
        for item in selected_main.analysis["edge_stability_receipts"]
    }
    point_keys = set(subset.analysis["point_gate_receipts"][0])
    for item in subset.analysis["point_gate_receipts"]:
        main = main_by_pair[(item["left_mode_id"], item["right_mode_id"])]
        assert item == {key: main[key] for key in point_keys}
    assert subset.analysis["point_gated_edges"] == selected_main.analysis[
        "point_gated_candidate_edges"
    ]


def test_diagnostic_subset_strict_rebuild_and_tamper_rejection() -> None:
    source = _fixture()
    subset = build_chart_energy_diagnostic_subset_receipt(
        source,
        subset_id="hard-long-veto",
        family_example_ids=(
            ("family-a", ("example-1",)),
            ("family-b", ("example-1",)),
        ),
    )
    payload = subset.canonical_payload()
    parsed = parse_chart_energy_diagnostic_subset_receipt_payload(
        payload,
        source_graph_input=source,
    )
    assert parsed.canonical_payload() == payload

    tampered = json.loads(json.dumps(payload))
    tampered["authorizes_edges"] = True
    with pytest.raises(ValueError, match="cannot authorize"):
        parse_chart_energy_diagnostic_subset_receipt_payload(
            tampered,
            source_graph_input=source,
        )
    tampered = json.loads(json.dumps(payload))
    tampered["analysis"]["point_gated_edges"] = []
    with pytest.raises(ValueError, match="deterministic rebuild"):
        parse_chart_energy_diagnostic_subset_receipt_payload(
            tampered,
            source_graph_input=source,
        )


def test_diagnostic_subset_catalog_is_exact_and_all_ties_are_reportable() -> None:
    source = _fixture()
    with pytest.raises(ValueError, match="include every family"):
        build_chart_energy_diagnostic_subset_receipt(
            source,
            subset_id="short",
            family_example_ids=(("family-a", ("example-0",)),),
        )
    with pytest.raises(ValueError, match="ordered source subset"):
        build_chart_energy_diagnostic_subset_receipt(
            source,
            subset_id="short",
            family_example_ids=(
                ("family-a", ("example-1", "example-0")),
                ("family-b", ("example-0",)),
            ),
        )

    tied_positions = {
        family_id: {mode_id: 1.0 for mode_id in _KINDS}
        for family_id in ("family-a", "family-b")
    }
    tied = build_chart_energy_diagnostic_subset_receipt(
        _fixture(positions_by_family=tied_positions),
        subset_id="transition-tie-diagnostic",
        family_example_ids=(
            ("family-a", ("example-0",)),
            ("family-b", ("example-0",)),
        ),
    )
    assert tied.analysis["point_gated_edges"] == []
    assert all(
        item["status"] == "undefined_rank_boundary_tie"
        for item in tied.analysis["pooled"]["topology"][
            "neighbor_receipts"
        ]
    )


def test_directed_distance_uses_equal_family_mean_and_target_energy() -> None:
    positions = {
        "family-a": {
            "mode-a": 1.0,
            "mode-b": 2.0,
            "mode-c": 8.0,
            "mode-d": 9.0,
        },
        "family-b": {
            "mode-a": 10.0,
            "mode-b": 11.0,
            "mode-c": 30.0,
            "mode-d": 31.0,
        },
    }
    receipt = build_chart_energy_neighbor_receipt(
        _fixture(positions_by_family=positions)
    )
    distance = receipt.analysis["pooled"]["distance"]
    directed = distance["directional_equal_family_distance_matrix"]
    diagnostic = distance[
        "directional_global_pooled_ratio_matrix_diagnostic"
    ]
    expected_a_to_b = 0.5 * (1.0 / 4.0 + 1.0 / 121.0)
    expected_b_to_a = 0.5 * (1.0 / 1.0 + 1.0 / 100.0)
    assert directed[0][1] == pytest.approx(expected_a_to_b)
    assert directed[1][0] == pytest.approx(expected_b_to_a)
    assert distance["symmetric_distance_matrix"][0][1] == pytest.approx(
        0.5 * (expected_a_to_b + expected_b_to_a)
    )
    assert diagnostic[0][1] != pytest.approx(expected_a_to_b)
    assert receipt.analysis["distance_contract"]["global_pooled_ratio_role"] == (
        "diagnostic_only"
    )


def test_receipt_emits_energy_occupancy_and_stability_hooks() -> None:
    receipt = build_chart_energy_neighbor_receipt(_fixture())
    modes = receipt.analysis["mode_diagnostics"]
    assert len(modes) == 4
    attention = next(item for item in modes if item["mode_id"] == "mode-b")
    assert attention["occupancy_fraction"] == pytest.approx(9 / 17)
    assert len(attention["supplemental_route_catalog_sha256s"]) == 4
    assert attention["weighted_fisher_energy"] > 0.0
    stability = receipt.analysis["stability"]
    assert stability["bootstrap"]["resample_count"] == 20
    assert stability["bootstrap"]["valid_resample_count"] > 0
    assert stability["by_family"]["replicate_count"] == 2
    assert stability["leave_one_family_out"]["status"] == "available"
    assert stability["leave_one_example_out"]["replicate_count"] == 4
    edge = next(
        item
        for item in receipt.analysis["edge_stability_receipts"]
        if item["left_mode_id"] == "mode-a"
        and item["right_mode_id"] == "mode-b"
    )
    assert edge["point_mutual_knn"] is True
    assert edge["by_family_joint_defined_count"] == 2
    assert edge["bootstrap_mutual_frequency"] >= 0.0


def test_control_ratios_are_derived_and_point_gate_is_conjunctive() -> None:
    receipt = build_chart_energy_neighbor_receipt(_fixture())
    distance = receipt.analysis["pooled"]["distance"]
    zero = distance["directional_zero_prediction_control_ratio_matrix"]
    assert zero[0][1] == pytest.approx(1.0)
    assert zero[1][0] == pytest.approx(1.0)
    edge = next(
        item
        for item in receipt.analysis["edge_stability_receipts"]
        if item["left_mode_id"] == "mode-a"
        and item["right_mode_id"] == "mode-b"
    )
    assert edge["left_to_right_real_gain_over_zero"] > 0.0
    assert edge["left_to_right_real_gain_over_row_shuffle"] > 0.0
    assert edge["right_to_left_real_gain_over_zero_gate_pass"] is True
    assert edge["point_gate_pass"] is True
    assert ["mode-a", "mode-b"] in receipt.analysis[
        "point_gated_candidate_edges"
    ]
    assert receipt.analysis["confirmation_core_edges"] == []
    assert receipt.analysis["point_gated_candidate_edges_status"] == (
        "applicable"
    )


def test_bootstrap_draw_catalog_is_independently_reproducible() -> None:
    graph_input = _fixture()
    receipt = build_chart_energy_neighbor_receipt(graph_input)
    bootstrap = graph_input.within_family_bootstrap_config
    payload = bootstrap.canonical_payload()
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(payload["draw_catalog_digest_domain_hex"]))
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    for resample in range(bootstrap.resample_count):
        for family_ordinal, (_, examples) in enumerate(
            bootstrap.family_example_ids
        ):
            for draw in range(bootstrap.draws_per_family):
                preimage = b"".join(
                    (
                        bootstrap.draw_protocol_domain,
                        bootstrap.seed.to_bytes(8, "big"),
                        resample.to_bytes(8, "big"),
                        family_ordinal.to_bytes(4, "big"),
                        draw.to_bytes(4, "big"),
                    )
                )
                selected = int.from_bytes(
                    hashlib.sha256(preimage).digest()[:8], "big"
                ) % len(examples)
                digest.update(resample.to_bytes(8, "big"))
                digest.update(family_ordinal.to_bytes(4, "big"))
                digest.update(draw.to_bytes(4, "big"))
                digest.update(selected.to_bytes(4, "big"))
    assert receipt.analysis["stability"]["bootstrap"][
        "draw_catalog_sha256"
    ] == digest.hexdigest()


def test_bootstrap_domain_and_whole_example_catalog_fail_closed() -> None:
    graph_input = _fixture()
    bootstrap = graph_input.within_family_bootstrap_config
    with pytest.raises(ValueError, match="draw protocol domain"):
        replace(bootstrap, draw_protocol_domain=b"wrong-domain\0")
    payload = build_chart_energy_neighbor_receipt(graph_input).canonical_payload()
    payload["input"]["within_family_bootstrap_config"][
        "sampling_unit"
    ] = "example_plus_selected_probes_cluster"
    with pytest.raises(ValueError, match="sampling_unit"):
        parse_chart_energy_neighbor_receipt_payload(payload)


def test_zero_point_candidate_case_is_explicit_and_valid() -> None:
    graph_input = _fixture()
    rows = tuple(
        replace(
            row,
            weighted_row_shuffled_transport_residual_energy=(
                row.weighted_transport_residual_energy
            ),
        )
        for row in graph_input.directed_transport_stats
    )
    receipt = build_chart_energy_neighbor_receipt(
        replace(graph_input, directed_transport_stats=rows)
    )
    assert receipt.analysis["pooled"]["topology"]["mutual_edges"]
    assert receipt.analysis["point_gated_candidate_edges"] == []
    assert receipt.analysis["confirmation_core_edges"] == []


def test_zero_energy_rows_are_retained_but_negative_rows_fail_closed() -> None:
    graph_input = _fixture()
    first = graph_input.example_stats[0]
    zero_minimum = replace(first, minimum_row_fisher_energy=0.0)
    assert zero_minimum.minimum_row_fisher_energy == 0.0

    with pytest.raises(ValueError, match="minimum_row_fisher_energy"):
        replace(first, minimum_row_fisher_energy=-1.0)


def test_single_family_and_single_example_mark_stability_unavailable() -> None:
    receipt = build_chart_energy_neighbor_receipt(
        _fixture(examples_by_family={"family-a": ("example-0",)})
    )
    stability = receipt.analysis["stability"]
    assert stability["leave_one_family_out"]["status"] == (
        "unavailable_single_family"
    )
    assert stability["leave_one_example_out"]["status"] == (
        "unavailable_family_with_single_example"
    )


def test_exact_comparison_abi_is_required_not_native_abi_equality() -> None:
    graph_input = _fixture()
    modes = list(graph_input.mode_specs)
    modes[0] = replace(
        modes[0],
        comparison_abi=ChartComparisonABI(
            abi_name="carrier_chart_energy.h320.v1",
            carrier_width=320,
            row_semantics=(
                "aligned_example_prediction_query_and_target_label_positions"
            ),
            generalized_contribution_semantics=(
                "value_hadamard_summed_next_token_nll_score_gradient"
            ),
            metric_semantics=(
                "shared_membership_spd_tangent_inverse_scale_plus_"
                "isotropic_orthogonal"
            ),
            energy_semantics="membership_weighted_spd_self_cross_mismatch",
        ),
    )
    with pytest.raises(ValueError, match="exact carrier_chart_energy"):
        build_chart_energy_neighbor_receipt(
            replace(graph_input, mode_specs=tuple(modes))
        )


def test_native_adapter_mapping_protects_lexical_readout_pullback() -> None:
    mode = _mode("mode-d", 3)
    with pytest.raises(ValueError, match="incompatible with native ABI"):
        replace(
            mode,
            comparison_adapter_kind=DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
        )
    with pytest.raises(ValueError, match="mode_kind is incompatible"):
        replace(mode, mode_kind="attention")


def test_body_context_and_contribution_sites_cannot_be_collapsed() -> None:
    mode = _mode("mode-a", 0)
    with pytest.raises(ValueError, match="distinct context and contribution"):
        replace(
            mode,
            contribution_site_id=mode.context_site_id,
        )
    lexical = _mode("mode-d", 3)
    same_terminal_site = replace(
        lexical,
        contribution_site_id=lexical.context_site_id,
    )
    assert same_terminal_site.native_abi.abi_kind == LEXICAL_READOUT_NATIVE_ABI


def test_original_event_placement_must_be_one_to_one() -> None:
    graph_input = _fixture()
    modes = list(graph_input.mode_specs)
    modes[0] = replace(
        modes[0],
        event_id=modes[1].event_id,
        event_index=modes[1].event_index,
    )
    with pytest.raises(ValueError, match="unique event_id"):
        build_chart_energy_neighbor_receipt(
            replace(graph_input, mode_specs=tuple(modes))
        )


@pytest.mark.parametrize("remove_kind", ["chart", "transport"])
def test_missing_rows_fail_closed(remove_kind: str) -> None:
    graph_input = _fixture()
    if remove_kind == "chart":
        corrupted = replace(
            graph_input, example_stats=graph_input.example_stats[1:]
        )
        pattern = "complete mode matrix"
    else:
        corrupted = replace(
            graph_input,
            directed_transport_stats=graph_input.directed_transport_stats[1:],
        )
        pattern = "complete ordered mode matrix"
    with pytest.raises(ValueError, match=pattern):
        build_chart_energy_neighbor_receipt(corrupted)


def test_misaligned_rows_and_partial_transport_fail_closed() -> None:
    graph_input = _fixture()
    rows = list(graph_input.example_stats)
    rows[0] = replace(rows[0], aligned_row_catalog_sha256=_sha("wrong-rows"))
    with pytest.raises(ValueError, match="exactly aligned token rows"):
        build_chart_energy_neighbor_receipt(
            replace(graph_input, example_stats=tuple(rows))
        )
    row = graph_input.directed_transport_stats[0]
    corrupted = _replace_transport(
        graph_input,
        (row.family_id, row.example_id, row.source_mode_id, row.target_mode_id),
        transported_row_count=row.row_count - 1,
    )
    with pytest.raises(ValueError, match="cover every aligned"):
        build_chart_energy_neighbor_receipt(corrupted)


def test_attention_route_evidence_is_required_and_partition_specific() -> None:
    graph_input = _fixture()
    attention_rows = [
        index
        for index, row in enumerate(graph_input.example_stats)
        if row.mode_id == "mode-b"
    ]
    rows = list(graph_input.example_stats)
    index = attention_rows[0]
    rows[index] = replace(
        rows[index],
        supplemental_route_catalog_sha256=None,
        supplemental_route_receipt_sha256=None,
    )
    with pytest.raises(ValueError, match="causal-sequence modes require"):
        build_chart_energy_neighbor_receipt(
            replace(graph_input, example_stats=tuple(rows))
        )
    rows = list(graph_input.example_stats)
    index = next(i for i, row in enumerate(rows) if row.mode_id == "mode-a")
    rows[index] = replace(
        rows[index],
        supplemental_route_catalog_sha256=_sha("fake-route-catalog"),
        supplemental_route_receipt_sha256=_sha("fake-route-receipt"),
    )
    with pytest.raises(ValueError, match="other native ABIs"):
        build_chart_energy_neighbor_receipt(
            replace(graph_input, example_stats=tuple(rows))
        )


def test_shared_atlas_and_energy_components_are_authenticated() -> None:
    graph_input = _fixture()
    rows = list(graph_input.example_stats)
    rows[0] = replace(rows[0], chart_atlas_sha256=_sha("other-atlas"))
    with pytest.raises(ValueError, match="one shared chart atlas"):
        build_chart_energy_neighbor_receipt(
            replace(graph_input, example_stats=tuple(rows))
        )
    with pytest.raises(ValueError, match="tangent plus orthogonal"):
        replace(
            graph_input.example_stats[0],
            weighted_chart_tangent_energy=123.0,
        )


def test_quadratic_cauchy_and_chart_stat_bindings_fail_closed() -> None:
    graph_input = _fixture()
    row = graph_input.directed_transport_stats[0]
    with pytest.raises(ValueError, match="shared-SPD energy quadratic"):
        replace(row, weighted_transport_residual_energy=999.0)
    with pytest.raises(ValueError, match="Cauchy"):
        replace(
            row,
            weighted_predicted_fisher_energy=1.0,
            weighted_target_fisher_energy=4.0,
            weighted_predicted_target_cross_inner_product=2.1,
            weighted_transport_residual_energy=0.8,
            weighted_zero_prediction_residual_energy=4.0,
        )
    with pytest.raises(ValueError, match="zero-prediction residual"):
        replace(row, weighted_zero_prediction_residual_energy=123.0)
    corrupted = _replace_transport(
        graph_input,
        (row.family_id, row.example_id, row.source_mode_id, row.target_mode_id),
        source_chart_stats_sha256=_sha("wrong-source-stats"),
    )
    with pytest.raises(ValueError, match="source chart-stat binding"):
        build_chart_energy_neighbor_receipt(corrupted)


def test_predicted_energy_is_not_raw_source_energy_and_reverse_is_independent() -> None:
    graph_input = _fixture()
    rows = list(graph_input.directed_transport_stats)
    for index, row in enumerate(rows):
        if row.source_mode_id == "mode-a" and row.target_mode_id == "mode-b":
            target = row.weighted_target_fisher_energy
            rows[index] = replace(
                row,
                weighted_predicted_fisher_energy=0.0,
                weighted_predicted_target_cross_inner_product=0.0,
                weighted_transport_residual_energy=target,
            )
    receipt = build_chart_energy_neighbor_receipt(
        replace(graph_input, directed_transport_stats=tuple(rows))
    )
    distance = receipt.analysis["pooled"]["distance"][
        "directional_equal_family_distance_matrix"
    ]
    assert distance[0][1] == pytest.approx(1.0)
    assert distance[1][0] != pytest.approx(1.0)


def test_panel_role_and_evaluation_receipts_are_cross_bound() -> None:
    graph_input = _fixture()
    row = graph_input.directed_transport_stats[0]
    corrupted = _replace_transport(
        graph_input,
        (row.family_id, row.example_id, row.source_mode_id, row.target_mode_id),
        transport_evaluation_role="C_neighbor_confirmation",
    )
    with pytest.raises(ValueError, match="does not match graph panel role"):
        build_chart_energy_neighbor_receipt(corrupted)

    confirmation_rows = tuple(
        replace(row, transport_evaluation_role="C_neighbor_confirmation")
        for row in graph_input.directed_transport_stats
    )
    confirmation = replace(
        graph_input,
        panel_role="C_neighbor_confirmation",
        directed_transport_stats=confirmation_rows,
    )
    receipt = build_chart_energy_neighbor_receipt(confirmation)
    assert receipt.analysis["transport_evaluation_role"] == (
        "C_neighbor_confirmation"
    )
    assert receipt.analysis["point_gated_candidate_edges"] == []
    assert receipt.analysis["point_gated_candidate_edges_status"] == (
        "not_applicable_for_panel_role"
    )
    assert receipt.analysis["confirmation_core_edges_status"] == "applicable"
    assert receipt.analysis["confirmation_core_edges"]


def test_directional_fit_receipts_are_independent_but_frozen_per_pair() -> None:
    graph_input = _fixture()
    rows = graph_input.directed_transport_stats
    forward = next(
        row
        for row in rows
        if row.source_mode_id == "mode-a" and row.target_mode_id == "mode-b"
    )
    reverse = next(
        row
        for row in rows
        if row.source_mode_id == "mode-b" and row.target_mode_id == "mode-a"
    )
    assert forward.transport_fit_receipt_sha256 != (
        reverse.transport_fit_receipt_sha256
    )
    key = (
        forward.family_id,
        forward.example_id,
        forward.source_mode_id,
        forward.target_mode_id,
    )
    corrupted = _replace_transport(
        graph_input,
        key,
        transport_fit_receipt_sha256=_sha("drifted-within-pair-fit"),
    )
    with pytest.raises(ValueError, match="one frozen real transport"):
        build_chart_energy_neighbor_receipt(corrupted)


def test_fit_and_evaluation_roles_and_catalogs_are_disjoint() -> None:
    graph_input = _fixture()
    row = graph_input.directed_transport_stats[0]
    with pytest.raises(ValueError, match="fit and evaluation catalogs"):
        replace(
            row,
            transport_fit_catalog_sha256=(
                row.transport_evaluation_catalog_sha256
            ),
        )
    corrupted = _replace_transport(
        graph_input,
        (row.family_id, row.example_id, row.source_mode_id, row.target_mode_id),
        transport_fit_catalog_sha256=_sha("second-fit-catalog"),
    )
    with pytest.raises(ValueError, match="one frozen fit catalog"):
        build_chart_energy_neighbor_receipt(corrupted)


def test_degenerate_family_target_energy_fails_closed() -> None:
    row = _fixture().example_stats[0]
    with pytest.raises(ValueError, match="minimum row Fisher energy"):
        replace(
            row,
            weighted_chart_tangent_energy=0.0,
            weighted_chart_orthogonal_energy=0.0,
            weighted_fisher_energy=0.0,
        )


def test_rank_boundary_tie_fails_closed_for_ambiguous_node() -> None:
    topology = _neighbor_topology(
        (
            (0.0, 1.0, 5.0, 9.0),
            (1.0, 0.0, 1.0, 8.0),
            (5.0, 1.0, 0.0, 7.0),
            (9.0, 8.0, 7.0, 0.0),
        ),
        mode_axis=("mode-a", "mode-b", "mode-c", "mode-d"),
        config=ChartEnergyNeighborConfig(
            k=1,
            lofo_minimum_mutual_count=1,
            lofo_expected_replicate_count=2,
        ),
    )
    neighbor = next(
        item
        for item in topology["neighbor_receipts"]
        if item["mode_id"] == "mode-b"
    )
    assert neighbor["status"] == "undefined_rank_boundary_tie"
    assert neighbor["neighbor_mode_ids"] == []


def test_strict_parser_rebuilds_and_rejects_self_rehashed_analysis_tamper() -> None:
    receipt = build_chart_energy_neighbor_receipt(_fixture())
    payload = receipt.canonical_payload()
    parsed = parse_chart_energy_neighbor_receipt_payload(payload)
    assert parsed.receipt_sha256 == receipt.receipt_sha256
    tampered = json.loads(json.dumps(payload))
    tampered["analysis"]["mode_axis"][0] = "mode-z"
    canonical_without_receipt = {
        key: value for key, value in tampered.items() if key != "receipt_sha256"
    }
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.chart_energy_neighbor_graph.receipt.v1\0")
    digest.update(
        json.dumps(
            canonical_without_receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    tampered["receipt_sha256"] = digest.hexdigest()
    with pytest.raises(ValueError, match="deterministic rebuild"):
        parse_chart_energy_neighbor_receipt_payload(tampered)
    boolean_version = json.loads(json.dumps(payload))
    boolean_version["format_version"] = True
    with pytest.raises(ValueError, match="format version"):
        parse_chart_energy_neighbor_receipt_payload(boolean_version)
    nan_analysis = json.loads(json.dumps(payload))
    nan_analysis["analysis"]["mode_diagnostics"][0][
        "weighted_fisher_energy"
    ] = math.nan
    with pytest.raises(ValueError):
        parse_chart_energy_neighbor_receipt_payload(nan_analysis)


def test_public_input_parser_validates_and_returns_canonical_input() -> None:
    graph_input = _fixture()
    payload = graph_input.canonical_payload()
    parsed = parse_chart_energy_neighbor_input_payload(payload)
    canonical = parsed.canonical_payload()
    assert parse_chart_energy_neighbor_input_payload(canonical) == parsed
    tampered = json.loads(json.dumps(payload))
    tampered["example_stats"][0]["mode_id"] = "mode-unknown"
    with pytest.raises(ValueError, match="unknown mode"):
        parse_chart_energy_neighbor_input_payload(tampered)
    boolean_count = json.loads(json.dumps(payload))
    boolean_count["example_stats"][0]["row_count"] = True
    with pytest.raises((TypeError, ValueError)):
        parse_chart_energy_neighbor_input_payload(boolean_count)


def test_public_typed_input_canonicalizer_matches_strict_parser() -> None:
    graph_input = _fixture()
    canonical = canonicalize_chart_energy_neighbor_input(graph_input)

    assert canonical == parse_chart_energy_neighbor_input_payload(
        canonical.canonical_payload()
    )
    assert tuple(mode.mode_id for mode in canonical.mode_specs) == tuple(
        sorted(mode.mode_id for mode in graph_input.mode_specs)
    )
    assert canonical.example_stats == tuple(
        sorted(
            graph_input.example_stats,
            key=lambda row: (row.family_id, row.example_id, row.mode_id),
        )
    )
    assert canonical.directed_transport_stats == tuple(
        sorted(
            graph_input.directed_transport_stats,
            key=lambda row: (
                row.family_id,
                row.example_id,
                row.source_mode_id,
                row.target_mode_id,
            ),
        )
    )
    with pytest.raises(TypeError, match="ChartEnergyNeighborInput"):
        canonicalize_chart_energy_neighbor_input(object())  # type: ignore[arg-type]


def test_atomic_save_and_strict_load(tmp_path) -> None:
    receipt = build_chart_energy_neighbor_receipt(_fixture())
    path = tmp_path / "chart-neighbors.json"
    save_chart_energy_neighbor_receipt(path, receipt)
    loaded = load_chart_energy_neighbor_receipt(path)
    assert loaded.canonical_payload() == receipt.canonical_payload()
    with pytest.raises(FileExistsError, match="overwrite"):
        save_chart_energy_neighbor_receipt(path, receipt)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(receipt.canonical_payload(), indent=2), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must be canonical"):
        load_chart_energy_neighbor_receipt(noncanonical)
    duplicate = tmp_path / "duplicate.json"
    encoded = path.read_bytes()
    duplicate.write_bytes(
        encoded[:-1]
        + b',"schema":"fisher_graph.chart_energy_neighbor_graph"}'
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_chart_energy_neighbor_receipt(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(encoded.replace(b'"analysis":{', b'"analysis":NaN'))
    with pytest.raises(ValueError, match="nonstandard JSON constant"):
        load_chart_energy_neighbor_receipt(nonfinite)
