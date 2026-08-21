from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from fisher_graph.mode_similarity_atlas import WithinFamilyBootstrapConfig
from fisher_graph.multi_abi_mode_atlas import (
    DownstreamResponseExamplePairStats,
    DownstreamResponsePairStats,
)
from fisher_graph.stable_mutual_knn_graph import (
    STABLE_MUTUAL_KNN_RANK_BOUNDARY_TIE_POLICY,
    StableMutualKNNConfig,
    StableMutualKNNExampleStratum,
    StableMutualKNNInput,
    StableMutualKNNNodeSpec,
    StableMutualKNNPanelInput,
    StableMutualKNNStratumGroup,
    _batch_mutual_masks,
    _maximal_cliques,
    build_stable_mutual_knn_graph_receipt,
    load_stable_mutual_knn_graph_receipt,
    parse_stable_mutual_knn_graph_receipt_payload,
    save_stable_mutual_knn_graph_receipt,
    stable_mutual_knn_candidate_catalog_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixture(*, fresh_d_equals_c: bool = False) -> StableMutualKNNInput:
    nodes = (
        StableMutualKNNNodeSpec("node-a", ("mode-a0", "mode-a1")),
        StableMutualKNNNodeSpec("node-b", ("mode-b",)),
        StableMutualKNNNodeSpec("node-c", ("mode-c",)),
        StableMutualKNNNodeSpec("node-d", ("mode-d",)),
    )
    logical_axis = tuple(
        mode_id for node in nodes for mode_id in node.member_mode_ids
    )
    families = ("family-0", "family-1", "family-2")
    clusters = ("short-raw", "boundary-raw", "long-1", "long-2")
    abi_sha256 = _sha("response-abi")

    def panel(role: str) -> StableMutualKNNPanelInput:
        examples: list[DownstreamResponseExamplePairStats] = []
        aggregates: list[DownstreamResponsePairStats] = []
        for family_id in families:
            rows_by_pair: dict[
                tuple[str, str], list[DownstreamResponseExamplePairStats]
            ] = {}
            for mode_a_index, mode_a_id in enumerate(logical_axis):
                for mode_b_id in logical_axis[mode_a_index + 1 :]:
                    pair_rows: list[DownstreamResponseExamplePairStats] = []
                    for cluster_id in clusters:
                        positions = {
                            "mode-a0": 1.0,
                            "mode-a1": 1.0,
                            "mode-b": 1.1,
                            "mode-c": 4.0,
                            "mode-d": 4.1,
                        }
                        if role == "fresh_confirmation" and fresh_d_equals_c:
                            positions["mode-d"] = positions["mode-c"]
                        elif (
                            role == "fresh_confirmation"
                            and cluster_id.startswith("long")
                        ):
                            positions["mode-d"] = 6.0
                        value_a = positions[mode_a_id] / 2.0
                        value_b = positions[mode_b_id] / 2.0
                        row = DownstreamResponseExamplePairStats(
                            family_id=family_id,
                            example_cluster_id=cluster_id,
                            mode_a_id=mode_a_id,
                            mode_b_id=mode_b_id,
                            response_abi_sha256=abi_sha256,
                            weighted_sse=(value_a - value_b) ** 2,
                            weighted_mode_a_energy=value_a**2,
                            weighted_dot=value_a * value_b,
                            weighted_mode_b_energy=value_b**2,
                            row_count=2,
                        )
                        pair_rows.append(row)
                        examples.append(row)
                    rows_by_pair[(mode_a_id, mode_b_id)] = pair_rows
            for (mode_a_id, mode_b_id), pair_rows in rows_by_pair.items():
                aggregates.append(
                    DownstreamResponsePairStats(
                        family_id=family_id,
                        mode_a_id=mode_a_id,
                        mode_b_id=mode_b_id,
                        response_abi_sha256=abi_sha256,
                        weighted_sse=math.fsum(
                            row.weighted_sse for row in pair_rows
                        ),
                        weighted_mode_a_energy=math.fsum(
                            row.weighted_mode_a_energy for row in pair_rows
                        ),
                        weighted_dot=math.fsum(
                            row.weighted_dot for row in pair_rows
                        ),
                        weighted_mode_b_energy=math.fsum(
                            row.weighted_mode_b_energy for row in pair_rows
                        ),
                        row_count=sum(row.row_count for row in pair_rows),
                    )
                )
        bootstrap = WithinFamilyBootstrapConfig(
            seed=2311,
            resample_count=20,
            draws_per_family=4,
            maximum_degenerate_fraction=0.01,
            family_example_cluster_ids=tuple(
                (family_id, clusters) for family_id in families
            ),
        )
        if role == "fresh_confirmation":
            strata = tuple(
                StableMutualKNNExampleStratum(
                    family_id=family_id,
                    example_cluster_id=cluster_id,
                    stratum_id=cluster_id,
                )
                for family_id in families
                for cluster_id in clusters
            )
            groups = (
                StableMutualKNNStratumGroup("short", ("short-raw",)),
                StableMutualKNNStratumGroup(
                    "boundary", ("boundary-raw",)
                ),
                StableMutualKNNStratumGroup("long", ("long-1", "long-2")),
            )
        else:
            strata = ()
            groups = ()
        return StableMutualKNNPanelInput(
            panel_role=role,
            panel_binding_sha256=_sha(f"panel:{role}"),
            family_pair_stats=tuple(reversed(aggregates)),
            family_example_pair_stats=tuple(reversed(examples)),
            within_family_bootstrap_config=bootstrap,
            example_strata=strata,
            stratum_groups=groups,
        )

    candidates = (("node-a", "node-b"), ("node-c", "node-d"))
    config = StableMutualKNNConfig(
        k=1,
        rank_boundary_tie_absolute_tolerance=1e-10,
        rank_boundary_tie_policy=(
            STABLE_MUTUAL_KNN_RANK_BOUNDARY_TIE_POLICY
        ),
        max_directional_relative_sse=0.20,
        max_absolute_weighted_rmse=1.0,
        bootstrap_min_mutual_frequency=0.0,
        lofo_min_mutual_count=1,
        lofo_replicate_count=3,
        bootstrap_resample_count=20,
        bootstrap_max_degenerate_fraction=0.01,
        minimum_clique_size=2,
        primary_stratum_id="pooled",
        hard_veto_stratum_ids=("long",),
        diagnostic_stratum_ids=("short", "boundary"),
    )
    return StableMutualKNNInput(
        node_specs=nodes,
        candidate_edges=tuple(reversed(candidates)),
        candidate_catalog_sha256=(
            stable_mutual_knn_candidate_catalog_sha256(nodes, candidates)
        ),
        discovery=panel("discovery"),
        fresh_confirmation=panel("fresh_confirmation"),
        config=config,
    )


def _candidate(receipt, left: str, right: str) -> dict[str, object]:
    for item in receipt.analysis["candidate_edge_receipts"]:
        if {item["node_a_id"], item["node_b_id"]} == {left, right}:
            return item
    raise AssertionError("candidate edge not found")


def test_build_recomputes_fresh_gates_and_preserves_atomic_expansion() -> None:
    graph_input = _fixture()
    receipt = build_stable_mutual_knn_graph_receipt(graph_input)
    rebuilt = build_stable_mutual_knn_graph_receipt(graph_input)
    assert receipt.canonical_payload() == rebuilt.canonical_payload()
    assert receipt.graph_input.candidate_edges == (
        ("node-a", "node-b"),
        ("node-c", "node-d"),
    )

    edge_ab = _candidate(receipt, "node-a", "node-b")
    edge_cd = _candidate(receipt, "node-c", "node-d")
    assert edge_ab["discovery_point_mutual"] is True
    assert edge_ab["fresh_pooled_point_mutual"] is True
    assert edge_ab["discovery_symmetric_distance"] == pytest.approx(
        0.009132231404958693
    )
    assert edge_ab["fresh_pooled_symmetric_distance"] == pytest.approx(
        0.009132231404958693
    )
    assert edge_ab["confirmed"] is True
    assert edge_cd["fresh_pooled_point_mutual"] is True
    assert edge_cd["fresh_long_point_mutual"] is True
    assert edge_cd["fresh_long_symmetric_distance"] == pytest.approx(
        0.18055555555555555
    )
    assert edge_cd["long_relative_gate_pass"] is False
    assert edge_cd["confirmed"] is False
    assert edge_cd["fresh_long_asymmetry_ratio"]["status"] == "finite"

    assert receipt.analysis["singleton_node_ids"] == ["node-c", "node-d"]
    assert receipt.analysis["connected_components"] == [
        {
            "member_mode_ids": ["mode-a0", "mode-a1", "mode-b"],
            "node_ids": ["node-a", "node-b"],
            "topology_diagnostic_only": True,
        }
    ]
    assert receipt.analysis["maximal_cliques"][0]["node_ids"] == [
        "node-a",
        "node-b",
    ]
    assert receipt.analysis["claim_flags"] == {
        "cluster_claimed": False,
        "confirmed_neighborhood_edge_claimed": True,
        "deployment_claimed": False,
        "generator_family_claimed": False,
        "merge_or_compression_claimed": False,
        "stable_neighbor_group_claimed": True,
    }


def test_draw_catalog_uses_the_bootstrap_declared_digest_domain() -> None:
    graph_input = _fixture()
    receipt = build_stable_mutual_knn_graph_receipt(graph_input)
    bootstrap = graph_input.fresh_confirmation.within_family_bootstrap_config
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
    for resample_index in range(bootstrap.resample_count):
        for family_ordinal, (_family_id, cluster_ids) in enumerate(
            bootstrap.family_example_cluster_ids
        ):
            for draw_ordinal in range(bootstrap.draws_per_family):
                preimage = b"".join(
                    (
                        bootstrap.draw_protocol_domain,
                        bootstrap.seed.to_bytes(8, "big", signed=False),
                        resample_index.to_bytes(8, "big", signed=False),
                        family_ordinal.to_bytes(4, "big", signed=False),
                        draw_ordinal.to_bytes(4, "big", signed=False),
                    )
                )
                example_ordinal = int.from_bytes(
                    hashlib.sha256(preimage).digest()[:8],
                    "big",
                    signed=False,
                ) % len(cluster_ids)
                digest.update(resample_index.to_bytes(8, "big", signed=False))
                digest.update(family_ordinal.to_bytes(4, "big", signed=False))
                digest.update(draw_ordinal.to_bytes(4, "big", signed=False))
                digest.update(example_ordinal.to_bytes(4, "big", signed=False))
    assert receipt.analysis["fresh_confirmation"]["bootstrap"][
        "draw_catalog_sha256"
    ] == digest.hexdigest()


def test_candidate_hash_matches_pairs_only_and_rejects_wrong_hash() -> None:
    graph_input = _fixture()
    digest = stable_mutual_knn_candidate_catalog_sha256(
        tuple(reversed(graph_input.node_specs)),
        graph_input.candidate_edges,
    )
    assert digest == graph_input.candidate_catalog_sha256
    with pytest.raises(ValueError, match="does not authenticate"):
        build_stable_mutual_knn_graph_receipt(
            replace(graph_input, candidate_catalog_sha256="0" * 64)
        )


def test_declared_equivalence_must_remain_exact_on_fresh_examples() -> None:
    graph_input = _fixture()
    panel = graph_input.fresh_confirmation
    example_rows = list(panel.family_example_pair_stats)
    example_index = next(
        index
        for index, row in enumerate(example_rows)
        if row.family_id == "family-0"
        and row.example_cluster_id == "short-raw"
        and {row.mode_a_id, row.mode_b_id} == {"mode-a0", "mode-a1"}
    )
    row = example_rows[example_index]
    example_rows[example_index] = replace(
        row,
        weighted_sse=0.01,
        weighted_dot=row.weighted_dot - 0.005,
    )
    aggregate_rows = list(panel.family_pair_stats)
    aggregate_index = next(
        index
        for index, aggregate in enumerate(aggregate_rows)
        if aggregate.family_id == "family-0"
        and {aggregate.mode_a_id, aggregate.mode_b_id}
        == {"mode-a0", "mode-a1"}
    )
    aggregate = aggregate_rows[aggregate_index]
    aggregate_rows[aggregate_index] = replace(
        aggregate,
        weighted_sse=aggregate.weighted_sse + 0.01,
        weighted_dot=aggregate.weighted_dot - 0.005,
    )
    corrupted = replace(
        graph_input,
        fresh_confirmation=replace(
            panel,
            family_pair_stats=tuple(aggregate_rows),
            family_example_pair_stats=tuple(example_rows),
        ),
    )
    with pytest.raises(ValueError, match="not an exact identity pair"):
        build_stable_mutual_knn_graph_receipt(corrupted)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bad_sum"])
def test_logical_pair_matrix_and_aggregate_sum_are_complete(
    mutation: str,
) -> None:
    graph_input = _fixture()
    panel = graph_input.discovery
    aggregates = list(panel.family_pair_stats)
    if mutation == "missing":
        aggregates.pop()
    elif mutation == "duplicate":
        aggregates.append(aggregates[-1])
    else:
        row = aggregates[-1]
        aggregates[-1] = replace(
            row,
            weighted_sse=row.weighted_sse + 0.02,
            weighted_dot=row.weighted_dot - 0.01,
        )
    corrupted = replace(
        graph_input,
        discovery=replace(panel, family_pair_stats=tuple(aggregates)),
    )
    expected = {
        "missing": "complete logical-mode unordered matrix",
        "duplicate": "duplicate family pair statistics",
        "bad_sum": "is not the example sum",
    }[mutation]
    with pytest.raises(ValueError, match=expected):
        build_stable_mutual_knn_graph_receipt(corrupted)


def test_fresh_accidental_exact_equality_is_diagnostic_not_collapsed() -> None:
    receipt = build_stable_mutual_knn_graph_receipt(
        _fixture(fresh_d_equals_c=True)
    )
    assert receipt.analysis["node_axis"] == [
        "node-a",
        "node-b",
        "node-c",
        "node-d",
    ]
    pooled = next(
        item
        for item in receipt.analysis["fresh_confirmation"][
            "point_evaluations"
        ]
        if item["stratum_id"] == "pooled"
    )
    accidental = pooled["accidental_exact_equality_edges"]
    assert len(accidental) == 1
    assert {
        accidental[0]["left_node_id"],
        accidental[0]["right_node_id"],
    } == {"node-c", "node-d"}
    assert accidental[0]["node_universe_changed"] is False


def test_stratum_label_tamper_fails_closed() -> None:
    graph_input = _fixture()
    panel = graph_input.fresh_confirmation
    strata = list(panel.example_strata)
    index = next(
        index
        for index, item in enumerate(strata)
        if item.family_id == "family-0" and item.stratum_id == "short-raw"
    )
    strata[index] = replace(strata[index], stratum_id="boundary-raw")
    with pytest.raises(ValueError, match="every family must contribute"):
        build_stable_mutual_knn_graph_receipt(
            replace(
                graph_input,
                fresh_confirmation=replace(
                    panel, example_strata=tuple(strata)
                ),
            )
        )


def test_bool_is_not_accepted_as_a_numeric_config_value() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        StableMutualKNNConfig(k=True)  # type: ignore[arg-type]


def test_rank_boundary_tie_makes_bootstrap_replicate_degenerate() -> None:
    config = replace(
        _fixture().config,
        bootstrap_max_degenerate_fraction=1.0,
    )
    distance = np.asarray(
        [
            [
                [0.0, 0.1, 0.1, 1.0],
                [0.1, 0.0, 0.4, 0.8],
                [0.1, 0.4, 0.0, 0.7],
                [1.0, 0.8, 0.7, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    neighbors, mutual, valid, defined = _batch_mutual_masks(
        distance, config=config
    )
    assert defined[0, 0] is np.False_ or not bool(defined[0, 0])
    assert not bool(valid[0])
    assert not neighbors[0, 0].any()
    assert not mutual[0, 0].any()


def test_components_and_maximal_cliques_do_not_add_chain_transitivity() -> None:
    axis = ("a", "b", "c", "d")
    edges = (("a", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("c", "d"))
    assert _maximal_cliques(axis, edges, minimum_size=2) == (
        ("a", "b", "c"),
        ("b", "c", "d"),
    )


def test_strict_parser_rejects_coordinated_outer_rehash() -> None:
    receipt = build_stable_mutual_knn_graph_receipt(_fixture())
    payload = receipt.canonical_payload()
    parsed = parse_stable_mutual_knn_graph_receipt_payload(payload)
    assert parsed.receipt_sha256 == receipt.receipt_sha256

    tampered = json.loads(json.dumps(payload))
    tampered["analysis"]["candidate_edge_receipts"][0]["confirmed"] = False
    without_sha = dict(tampered)
    without_sha.pop("receipt_sha256")
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.stable_mutual_knn_graph.receipt.v1\0")
    digest.update(
        json.dumps(
            without_sha,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    tampered["receipt_sha256"] = digest.hexdigest()
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        parse_stable_mutual_knn_graph_receipt_payload(tampered)


@pytest.mark.parametrize("mutation", ["seed", "family_order"])
def test_strict_parser_binds_bootstrap_seed_and_family_order(
    mutation: str,
) -> None:
    receipt = build_stable_mutual_knn_graph_receipt(_fixture())
    payload = json.loads(json.dumps(receipt.canonical_payload()))
    bootstrap = payload["input"]["fresh_confirmation"][
        "within_family_bootstrap_config"
    ]
    if mutation == "seed":
        bootstrap["seed"] += 1
    else:
        bootstrap["families"][0], bootstrap["families"][1] = (
            bootstrap["families"][1],
            bootstrap["families"][0],
        )
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        parse_stable_mutual_knn_graph_receipt_payload(payload)


def test_save_load_is_canonical_no_clobber_and_symlink_closed(
    tmp_path: Path,
) -> None:
    receipt = build_stable_mutual_knn_graph_receipt(_fixture())
    path = tmp_path / "graph.json"
    save_stable_mutual_knn_graph_receipt(path, receipt)
    assert path.read_bytes().endswith(b"\n")
    loaded = load_stable_mutual_knn_graph_receipt(path)
    assert loaded.receipt_sha256 == receipt.receipt_sha256
    with pytest.raises(FileExistsError, match="overwrite"):
        save_stable_mutual_knn_graph_receipt(path, receipt)

    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        load_stable_mutual_knn_graph_receipt(link)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":1,"schema":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_stable_mutual_knn_graph_receipt(duplicate)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(receipt.canonical_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical"):
        load_stable_mutual_knn_graph_receipt(noncanonical)


def test_production_shape_unordered_receipt_stays_below_strict_cap() -> None:
    nodes: list[StableMutualKNNNodeSpec] = []
    positions: dict[str, float] = {}
    for node_index in range(37):
        node_id = f"production-node-{node_index:02d}"
        members = (
            ("production-mode-00a", "production-mode-00b")
            if node_index == 0
            else (f"production-mode-{node_index:02d}",)
        )
        nodes.append(StableMutualKNNNodeSpec(node_id, members))
        group = node_index // 2
        position = (
            1000.0
            + group * group * 3.17
            + (0.113 if node_index % 2 else 0.0)
            + node_index * 0.00037
        )
        for member in members:
            positions[member] = position
    node_tuple = tuple(nodes)
    logical_axis = tuple(
        mode_id for node in node_tuple for mode_id in node.member_mode_ids
    )
    families = tuple(f"production-family-{index}" for index in range(6))
    clusters = tuple(f"production-example-{index}" for index in range(10))
    response_abi_sha256 = _sha("production-response-abi")

    def panel(role: str) -> StableMutualKNNPanelInput:
        example_rows: list[DownstreamResponseExamplePairStats] = []
        aggregate_rows: list[DownstreamResponsePairStats] = []
        scale = 1.0 / math.sqrt(10.0)
        for family_id in families:
            for mode_a_index, mode_a_id in enumerate(logical_axis):
                for mode_b_id in logical_axis[mode_a_index + 1 :]:
                    value_a = positions[mode_a_id] * scale
                    value_b = positions[mode_b_id] * scale
                    components: list[
                        DownstreamResponseExamplePairStats
                    ] = []
                    for cluster_id in clusters:
                        row = DownstreamResponseExamplePairStats(
                            family_id=family_id,
                            example_cluster_id=cluster_id,
                            mode_a_id=mode_a_id,
                            mode_b_id=mode_b_id,
                            response_abi_sha256=response_abi_sha256,
                            weighted_sse=(value_a - value_b) ** 2,
                            weighted_mode_a_energy=value_a**2,
                            weighted_dot=value_a * value_b,
                            weighted_mode_b_energy=value_b**2,
                            row_count=4,
                        )
                        components.append(row)
                        example_rows.append(row)
                    aggregate_rows.append(
                        DownstreamResponsePairStats(
                            family_id=family_id,
                            mode_a_id=mode_a_id,
                            mode_b_id=mode_b_id,
                            response_abi_sha256=response_abi_sha256,
                            weighted_sse=math.fsum(
                                row.weighted_sse for row in components
                            ),
                            weighted_mode_a_energy=math.fsum(
                                row.weighted_mode_a_energy
                                for row in components
                            ),
                            weighted_dot=math.fsum(
                                row.weighted_dot for row in components
                            ),
                            weighted_mode_b_energy=math.fsum(
                                row.weighted_mode_b_energy
                                for row in components
                            ),
                            row_count=40,
                        )
                    )
        bootstrap = WithinFamilyBootstrapConfig(
            seed=2311,
            resample_count=10_000,
            draws_per_family=10,
            maximum_degenerate_fraction=0.01,
            family_example_cluster_ids=tuple(
                (family_id, clusters) for family_id in families
            ),
        )
        if role == "fresh_confirmation":
            raw_strata = (
                "short_32",
                "short_64",
                "short_96",
                "boundary_511",
                "boundary_512",
                "transition_513",
                "outside_window_514",
                "outside_window_576",
                "outside_window_640",
                "outside_window_768",
            )
            strata = tuple(
                StableMutualKNNExampleStratum(
                    family_id,
                    cluster_id,
                    stratum_id,
                )
                for family_id in families
                for cluster_id, stratum_id in zip(clusters, raw_strata)
            )
            groups = (
                StableMutualKNNStratumGroup("short", raw_strata[:3]),
                StableMutualKNNStratumGroup("boundary", raw_strata[3:6]),
                StableMutualKNNStratumGroup("long", raw_strata[6:]),
            )
        else:
            strata = ()
            groups = ()
        return StableMutualKNNPanelInput(
            panel_role=role,
            panel_binding_sha256=_sha(f"production:{role}"),
            family_pair_stats=tuple(aggregate_rows),
            family_example_pair_stats=tuple(example_rows),
            within_family_bootstrap_config=bootstrap,
            example_strata=strata,
            stratum_groups=groups,
        )

    candidates = tuple(
        (
            f"production-node-{index:02d}",
            f"production-node-{index + 1:02d}",
        )
        for index in range(0, 28, 2)
    )
    graph_input = StableMutualKNNInput(
        node_specs=node_tuple,
        candidate_edges=candidates,
        candidate_catalog_sha256=(
            stable_mutual_knn_candidate_catalog_sha256(
                node_tuple, candidates
            )
        ),
        discovery=panel("discovery"),
        fresh_confirmation=panel("fresh_confirmation"),
        config=StableMutualKNNConfig(
            hard_veto_stratum_ids=("long",),
            diagnostic_stratum_ids=("short", "boundary"),
        ),
    )
    receipt = build_stable_mutual_knn_graph_receipt(graph_input)
    canonical_bytes = json.dumps(
        receipt.canonical_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    assert len(graph_input.discovery.family_pair_stats) == 4_218
    assert len(graph_input.discovery.family_example_pair_stats) == 42_180
    assert len(canonical_bytes) < 256 * 1024 * 1024
    # This production-shaped synthetic control is about 32 MiB; leave margin
    # for longer real decimal renderings while catching directed-row regressions.
    assert len(canonical_bytes) < 64 * 1024 * 1024
    assert len(receipt.analysis["confirmed_edges"]) == 14
