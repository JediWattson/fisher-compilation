from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_stable_mutual_knn_confirmation_experiment as subject
from fisher_graph.gemma3_stable_mutual_knn_confirmation_panel import (
    confirmation_prompt_records,
    load_panel,
)
from fisher_graph.gemma3_stable_mutual_knn_setup import (
    FRESH_ORDERED_FAMILY_IDS,
    build_gemma3_stable_mutual_knn_setup,
)


def _production_inputs() -> SimpleNamespace:
    predecessor = json.loads(
        Path(subject.PINNED_TYPED_RESPONSE_REPORT_PATH).read_text()
    )
    discovery_family, discovery_examples, discovery_bootstrap = (
        subject._predecessor_graph_statistics(predecessor)
    )
    records = subject._validate_confirmation_records(
        confirmation_prompt_records(
            load_panel(subject.DEFAULT_CORPUS, subject.DEFAULT_PANEL)
        )
    )
    old_families = discovery_bootstrap.ordered_family_ids
    family_map = dict(zip(old_families, FRESH_ORDERED_FAMILY_IDS, strict=True))
    cluster_map: dict[tuple[str, str], str] = {}
    for old_family, old_cluster_ids in (
        discovery_bootstrap.family_example_cluster_ids
    ):
        fresh_family = family_map[old_family]
        fresh_ids = tuple(
            record.prompt_id
            for record in records
            if subject._confirmation_family_id(record) == fresh_family
        )
        assert len(old_cluster_ids) == len(fresh_ids) == 10
        cluster_map.update(
            {
                (old_family, old_cluster): fresh_cluster
                for old_cluster, fresh_cluster in zip(
                    old_cluster_ids, fresh_ids, strict=True
                )
            }
        )
    fresh_family = tuple(
        subject.DownstreamResponsePairStats(
            family_id=family_map[row.family_id],
            mode_a_id=row.mode_a_id,
            mode_b_id=row.mode_b_id,
            response_abi_sha256=row.response_abi_sha256,
            weighted_sse=row.weighted_sse,
            weighted_mode_a_energy=row.weighted_mode_a_energy,
            weighted_dot=row.weighted_dot,
            weighted_mode_b_energy=row.weighted_mode_b_energy,
            row_count=row.row_count,
        )
        for row in discovery_family
    )
    fresh_examples = tuple(
        subject.DownstreamResponseExamplePairStats(
            family_id=family_map[row.family_id],
            example_cluster_id=cluster_map[
                (row.family_id, row.example_cluster_id)
            ],
            mode_a_id=row.mode_a_id,
            mode_b_id=row.mode_b_id,
            response_abi_sha256=row.response_abi_sha256,
            weighted_sse=row.weighted_sse,
            weighted_mode_a_energy=row.weighted_mode_a_energy,
            weighted_dot=row.weighted_dot,
            weighted_mode_b_energy=row.weighted_mode_b_energy,
            row_count=row.row_count,
        )
        for row in discovery_examples
    )
    setup = build_gemma3_stable_mutual_knn_setup()
    nodes, candidates, candidate_sha256 = subject._node_specs_and_candidates(
        setup
    )
    preflight = SimpleNamespace(
        setup=setup,
        predecessor=predecessor,
        confirmation_records=records,
    )
    fresh_bootstrap = subject._fresh_bootstrap_config(preflight)
    fresh_strata = subject._fresh_example_strata(
        example_ids=tuple(record.prompt_id for record in records),
        family_ids=tuple(
            subject._confirmation_family_id(record) for record in records
        ),
        stratum_ids=tuple(str(record.stratum_id) for record in records),
    )
    return SimpleNamespace(
        records=records,
        nodes=nodes,
        candidates=candidates,
        candidate_sha256=candidate_sha256,
        discovery_family=discovery_family,
        discovery_examples=discovery_examples,
        discovery_bootstrap=discovery_bootstrap,
        fresh_family=fresh_family,
        fresh_examples=fresh_examples,
        fresh_bootstrap=fresh_bootstrap,
        fresh_strata=fresh_strata,
        fresh_groups=subject._fresh_stratum_groups(),
    )


def _build_graph(
    inputs: SimpleNamespace,
    *,
    nodes: tuple[subject.StableMutualKNNNodeSpec, ...] | None = None,
    candidates: tuple[tuple[str, str], ...] | None = None,
    candidate_sha256: str | None = None,
    discovery_bootstrap: subject.WithinFamilyBootstrapConfig | None = None,
    fresh_family: tuple[subject.DownstreamResponsePairStats, ...] | None = None,
    fresh_examples: tuple[
        subject.DownstreamResponseExamplePairStats, ...
    ]
    | None = None,
    fresh_panel_binding_sha256: str = "e" * 64,
) -> object:
    return subject.build_stable_mutual_knn_graph_receipt(
        subject.StableMutualKNNInput(
            node_specs=nodes or inputs.nodes,
            candidate_edges=candidates or inputs.candidates,
            candidate_catalog_sha256=(
                candidate_sha256 or inputs.candidate_sha256
            ),
            discovery=subject.StableMutualKNNPanelInput(
                panel_role="discovery",
                panel_binding_sha256=subject.PINNED_MEASUREMENT_INPUT_SHA256,
                family_pair_stats=inputs.discovery_family,
                family_example_pair_stats=inputs.discovery_examples,
                within_family_bootstrap_config=(
                    discovery_bootstrap or inputs.discovery_bootstrap
                ),
            ),
            fresh_confirmation=subject.StableMutualKNNPanelInput(
                panel_role="fresh_confirmation",
                panel_binding_sha256=fresh_panel_binding_sha256,
                family_pair_stats=fresh_family or inputs.fresh_family,
                family_example_pair_stats=(
                    fresh_examples or inputs.fresh_examples
                ),
                within_family_bootstrap_config=inputs.fresh_bootstrap,
                example_strata=inputs.fresh_strata,
                stratum_groups=inputs.fresh_groups,
            ),
            config=subject._graph_config(),
        )
    )


def _outer_bindings(
    inputs: SimpleNamespace, graph: object
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    token_rows = [
        {
            "prompt_id": record.prompt_id,
            "family_id": subject._confirmation_family_id(record),
            "stratum_id": record.stratum_id,
        }
        for record in inputs.records
    ]
    tokenization = {"rows": token_rows}
    actual = subject._graph_input_binding_payload(
        nodes=graph.graph_input.node_specs,
        candidate_edges=graph.graph_input.candidate_edges,
        candidate_catalog_sha256=graph.graph_input.candidate_catalog_sha256,
        config=graph.graph_input.config,
        discovery_family=graph.graph_input.discovery.family_pair_stats,
        discovery_examples=(
            graph.graph_input.discovery.family_example_pair_stats
        ),
        discovery_bootstrap=(
            graph.graph_input.discovery.within_family_bootstrap_config
        ),
        fresh_family=graph.graph_input.fresh_confirmation.family_pair_stats,
        fresh_examples=(
            graph.graph_input.fresh_confirmation.family_example_pair_stats
        ),
        fresh_bootstrap=(
            graph.graph_input.fresh_confirmation.within_family_bootstrap_config
        ),
        fresh_strata=graph.graph_input.fresh_confirmation.example_strata,
        fresh_groups=graph.graph_input.fresh_confirmation.stratum_groups,
    )
    measurement_input = {
        "measurement_input_sha256": "e" * 64,
        "stable_graph_input_binding": actual,
        "stable_graph_input_binding_sha256": actual[
            "graph_input_binding_sha256"
        ],
    }
    measurement = {
        "stable_graph_input_binding_sha256": actual[
            "graph_input_binding_sha256"
        ],
        "fresh_family_pair_stat_count": subject.PINNED_FAMILY_PAIR_STAT_COUNT,
        "fresh_family_stats_catalog_sha256": actual[
            "fresh_family_stats_catalog_sha256"
        ],
        "fresh_example_pair_stat_count": subject.PINNED_EXAMPLE_PAIR_STAT_COUNT,
        "fresh_example_stats_catalog_sha256": actual[
            "fresh_example_stats_catalog_sha256"
        ],
    }
    lineage = {
        "discovery_family_stats_catalog_sha256": (
            subject._PINNED_DISCOVERY_FAMILY_STATS_CATALOG_SHA256
        ),
        "discovery_example_stats_catalog_sha256": (
            subject._PINNED_DISCOVERY_EXAMPLE_STATS_CATALOG_SHA256
        ),
        "discovery_bootstrap_config_sha256": (
            subject._PINNED_DISCOVERY_BOOTSTRAP_CONFIG_SHA256
        ),
        "discovery_stats_binding_sha256": (
            subject._PINNED_DISCOVERY_STATS_BINDING_SHA256
        ),
    }
    return lineage, tokenization, measurement, measurement_input


@pytest.fixture(scope="module")
def production_graph() -> tuple[SimpleNamespace, object]:
    inputs = _production_inputs()
    graph = _build_graph(inputs)
    return inputs, graph


def test_production_shape_survivor_and_exact_graph_binding(
    production_graph: tuple[SimpleNamespace, object],
) -> None:
    inputs, graph = production_graph
    assert len(graph.analysis["confirmed_edges"]) == 1
    bindings = _outer_bindings(inputs, graph)
    subject._validate_graph_input_cross_bindings(
        graph=graph,
        lineage=bindings[0],
        tokenization=bindings[1],
        measurement=bindings[2],
        measurement_input=bindings[3],
    )


def test_coordinated_rebuilt_core_cannot_replace_frozen_node_universe(
    production_graph: tuple[SimpleNamespace, object],
) -> None:
    inputs, original = production_graph
    renamed_nodes = tuple(
        subject.StableMutualKNNNodeSpec(
            node_id=f"renamed.{index:02d}",
            member_mode_ids=node.member_mode_ids,
        )
        for index, node in enumerate(inputs.nodes)
    )
    rename = {
        old.node_id: new.node_id
        for old, new in zip(inputs.nodes, renamed_nodes, strict=True)
    }
    renamed_candidates = tuple(
        (rename[left], rename[right]) for left, right in inputs.candidates
    )
    candidate_sha256 = subject.stable_mutual_knn_candidate_catalog_sha256(
        renamed_nodes, renamed_candidates
    )
    rebuilt = _build_graph(
        inputs,
        nodes=renamed_nodes,
        candidates=renamed_candidates,
        candidate_sha256=candidate_sha256,
    )
    bindings = _outer_bindings(inputs, original)
    with pytest.raises(ValueError, match="input universe"):
        subject._validate_graph_input_cross_bindings(
            graph=rebuilt,
            lineage=bindings[0],
            tokenization=bindings[1],
            measurement=bindings[2],
            measurement_input=bindings[3],
        )


def test_coordinated_rebuilt_core_cannot_replace_discovery_statistics(
    production_graph: tuple[SimpleNamespace, object],
) -> None:
    inputs, original = production_graph
    bootstrap = inputs.discovery_bootstrap
    reversed_bootstrap = subject.WithinFamilyBootstrapConfig(
        seed=bootstrap.seed,
        resample_count=bootstrap.resample_count,
        draws_per_family=bootstrap.draws_per_family,
        maximum_degenerate_fraction=bootstrap.maximum_degenerate_fraction,
        family_example_cluster_ids=tuple(
            (family_id, tuple(reversed(cluster_ids)))
            for family_id, cluster_ids in bootstrap.family_example_cluster_ids
        ),
        draw_protocol_domain=bootstrap.draw_protocol_domain,
    )
    rebuilt = _build_graph(inputs, discovery_bootstrap=reversed_bootstrap)
    bindings = _outer_bindings(inputs, original)
    with pytest.raises(ValueError, match="discovery statistics"):
        subject._validate_graph_input_cross_bindings(
            graph=rebuilt,
            lineage=bindings[0],
            tokenization=bindings[1],
            measurement=bindings[2],
            measurement_input=bindings[3],
        )


def _long_veto_pair_statistics(
    inputs: SimpleNamespace,
) -> tuple[
    tuple[subject.DownstreamResponsePairStats, ...],
    tuple[subject.DownstreamResponseExamplePairStats, ...],
]:
    target_modes = {"mode.mutation.14", "mode.mutation.16"}
    long_ids = {
        record.prompt_id
        for record in inputs.records
        if record.stratum_id in subject._LONG_STRATA
    }
    examples: list[subject.DownstreamResponseExamplePairStats] = []
    for row in inputs.fresh_examples:
        if (
            {row.mode_a_id, row.mode_b_id} == target_modes
            and row.example_cluster_id in long_ids
        ):
            dot = -math.sqrt(
                row.weighted_mode_a_energy * row.weighted_mode_b_energy
            )
            row = subject.DownstreamResponseExamplePairStats(
                family_id=row.family_id,
                example_cluster_id=row.example_cluster_id,
                mode_a_id=row.mode_a_id,
                mode_b_id=row.mode_b_id,
                response_abi_sha256=row.response_abi_sha256,
                weighted_sse=(
                    row.weighted_mode_a_energy
                    + row.weighted_mode_b_energy
                    - 2.0 * dot
                ),
                weighted_mode_a_energy=row.weighted_mode_a_energy,
                weighted_dot=dot,
                weighted_mode_b_energy=row.weighted_mode_b_energy,
                row_count=row.row_count,
            )
        examples.append(row)
    family_rows: list[subject.DownstreamResponsePairStats] = []
    for row in inputs.fresh_family:
        if {row.mode_a_id, row.mode_b_id} == target_modes:
            parts = [
                item
                for item in examples
                if item.family_id == row.family_id
                and item.mode_a_id == row.mode_a_id
                and item.mode_b_id == row.mode_b_id
            ]
            row = subject.DownstreamResponsePairStats(
                family_id=row.family_id,
                mode_a_id=row.mode_a_id,
                mode_b_id=row.mode_b_id,
                response_abi_sha256=row.response_abi_sha256,
                weighted_sse=math.fsum(item.weighted_sse for item in parts),
                weighted_mode_a_energy=math.fsum(
                    item.weighted_mode_a_energy for item in parts
                ),
                weighted_dot=math.fsum(item.weighted_dot for item in parts),
                weighted_mode_b_energy=math.fsum(
                    item.weighted_mode_b_energy for item in parts
                ),
                row_count=sum(item.row_count for item in parts),
            )
        family_rows.append(row)
    return tuple(family_rows), tuple(examples)


def test_zero_survivor_path_and_graph_only_fresh_stats_replacement_rejected(
    production_graph: tuple[SimpleNamespace, object],
) -> None:
    inputs, original = production_graph
    fresh_family, fresh_examples = _long_veto_pair_statistics(inputs)
    rebuilt = _build_graph(
        inputs,
        fresh_family=fresh_family,
        fresh_examples=fresh_examples,
    )
    assert rebuilt.analysis["confirmed_edges"] == []
    assert rebuilt.analysis["connected_components"] == []
    assert rebuilt.analysis["maximal_cliques"] == []
    bindings = _outer_bindings(inputs, original)
    with pytest.raises(ValueError, match="measurement/graph cross-bound"):
        subject._validate_graph_input_cross_bindings(
            graph=rebuilt,
            lineage=bindings[0],
            tokenization=bindings[1],
            measurement=bindings[2],
            measurement_input=bindings[3],
        )


def _report_preflight(
    inputs: SimpleNamespace, output: Path
) -> subject._StaticPreflight:
    root = subject.DEFAULT_REPOSITORY_ROOT.resolve()
    setup_path = (root / subject.DEFAULT_SETUP).resolve()
    predecessor_path = (
        root / subject.PINNED_TYPED_RESPONSE_REPORT_PATH
    ).resolve()
    corpus_path = (root / subject.DEFAULT_CORPUS).resolve()
    panel_path = (root / subject.DEFAULT_PANEL).resolve()
    audit_path = (root / subject.DEFAULT_PANEL_AUDIT).resolve()
    core_path = Path(subject._stable_graph_module.__file__).resolve()
    panel_module_path = Path(subject._panel_module.__file__).resolve()
    worker_source_path = Path(subject.__file__).resolve()
    return subject._StaticPreflight(
        setup=build_gemma3_stable_mutual_knn_setup(),
        predecessor=json.loads(predecessor_path.read_text()),
        confirmation_records=inputs.records,
        setup_path=setup_path,
        predecessor_path=predecessor_path,
        corpus_path=corpus_path,
        panel_path=panel_path,
        panel_audit_path=audit_path,
        output_path=output,
        svg_path=subject._default_svg_path(output),
        receipts={
            "setup_file_sha256": subject.PINNED_SETUP_FILE_SHA256,
            "setup_sha256": subject.PINNED_SETUP_SHA256,
            "confirmation_worker_source_path": str(worker_source_path),
            "confirmation_worker_source_sha256": subject._sha256_file(
                worker_source_path
            ),
            "stable_mutual_knn_core_source_path": str(core_path),
            "stable_mutual_knn_core_source_sha256": (
                subject._PINNED_STABLE_MUTUAL_KNN_CORE_SOURCE_SHA256
            ),
            "confirmation_panel_module_path": str(panel_module_path),
            "confirmation_panel_module_sha256": (
                subject._PINNED_CONFIRMATION_PANEL_MODULE_SHA256
            ),
            "typed_response_report_file_sha256": (
                subject.PINNED_TYPED_RESPONSE_REPORT_FILE_SHA256
            ),
            "typed_response_report_sha256": (
                subject.PINNED_TYPED_RESPONSE_REPORT_SHA256
            ),
            "multi_ABI_receipt_sha256": (
                subject.PINNED_MULTI_ABI_RECEIPT_SHA256
            ),
            "predecessor_measurement_input_sha256": (
                subject.PINNED_MEASUREMENT_INPUT_SHA256
            ),
            "discovery_family_stats_catalog_sha256": (
                subject._PINNED_DISCOVERY_FAMILY_STATS_CATALOG_SHA256
            ),
            "discovery_example_stats_catalog_sha256": (
                subject._PINNED_DISCOVERY_EXAMPLE_STATS_CATALOG_SHA256
            ),
            "discovery_bootstrap_config_sha256": (
                subject._PINNED_DISCOVERY_BOOTSTRAP_CONFIG_SHA256
            ),
            "discovery_stats_binding_sha256": (
                subject._PINNED_DISCOVERY_STATS_BINDING_SHA256
            ),
            "corpus_file_sha256": (
                subject._PINNED_CONFIRMATION_CORPUS_FILE_SHA256
            ),
            "panel_file_sha256": (
                subject._PINNED_CONFIRMATION_PANEL_FILE_SHA256
            ),
            "panel_audit_file_sha256": (
                subject._PINNED_CONFIRMATION_PANEL_AUDIT_FILE_SHA256
            ),
            "panel_audit_payload_sha256": (
                subject._PINNED_CONFIRMATION_PANEL_AUDIT_PAYLOAD_SHA256
            ),
            "confirmation_row_count": 60,
            "reserve_row_count_parsed_and_hashed_only": 12,
            "reserve_rows_returned_by_preflight": 0,
            "free_bytes_before_transformers_tokenizer_or_model_weights": (
                subject._MIN_FREE_BYTES_BEFORE_LIVE * 2
            ),
            "minimum_free_bytes_required": subject._MIN_FREE_BYTES_BEFORE_LIVE,
            "strict_report_size_cap_bytes": subject._MAX_REPORT_BYTES,
            "measured_production_shape_core_estimate_bytes": (
                subject._MEASURED_PRODUCTION_SHAPE_CORE_ESTIMATE_BYTES
            ),
            (
                "all_static_inputs_authenticated_before_transformers_"
                "tokenizer_or_model_weights"
            ): True,
        },
    )


def _report_tokenization(
    inputs: SimpleNamespace,
) -> subject._AuthenticatedTokenization:
    suffix_by_family = {
        "observational_diagnosis": (
            "1a286f03bf293a09d83055f5504354065065817ed28f53e50c954b528449e35c"
        ),
        "quantitative_reasoning": (
            "7cadad253cccb0d4078b27a65178a12ae2cb49b24f95e3bb1e6dba38523147ae"
        ),
        "technical_explanation": (
            "72bdfab476b9c5cf8ba247582ae46e2ddf06d3d6b11d5cfa656c4b155fd30026"
        ),
        "procedural_planning": (
            "9a6ba10ccd1d3a86cc4834d6562fac15868101fbfa07036e41bebe8d7ed429b0"
        ),
        "formal_logic": (
            "a0bff09e72c0758fdb036753241cb4674790db3ca1620ce4c5a12f632025cc76"
        ),
        "causal_evidence_synthesis": (
            "54c5f95819ef81e0b5159c651c55277529aa6c2a86aa3f32936286b3d4c2dc6d"
        ),
    }

    def evidence_token_count(record: object) -> int:
        evidence = record.answer_evidence
        if evidence is None:
            return 0
        return (
            int(evidence["token_end_inclusive"])
            - int(evidence["token_start"])
            + 1
        )

    panel_rows = [
        {
            "prompt_id": record.prompt_id,
            "stratum_id": record.stratum_id,
            "valid_tokens": record.target_valid_tokens,
            "input_ids_sha256": record.input_ids_sha256,
            "choice_token_ids_sha256": (
                subject._panel_module._choice_token_ids_sha256(
                    record.choice_token_ids
                )
            ),
            "choices_sha256": record.choices_sha256,
            "answer_evidence_verified": record.answer_evidence is not None,
            "answer_evidence_token_count": evidence_token_count(record),
            "visible_suffix_token_ids_sha256": (
                suffix_by_family[record.semantic_family]
                if record.answer_evidence is not None
                else None
            ),
            "visible_suffix_leak_free": True,
        }
        for record in inputs.records
    ]
    panel_receipt = {
        "tokenizer_id": subject.PINNED_MODEL_ID,
        "tokenizer_revision": subject.PINNED_MODEL_REVISION,
        "row_count": 60,
        "reserve_tokenized": False,
        "visible_suffix_digest_domain": (
            subject._panel_module._VISIBLE_SUFFIX_TOKEN_IDS_DOMAIN.rstrip(
                b"\0"
            ).decode("ascii")
        ),
        "hard_long_visible_suffixes": [
            {
                "semantic_family": family,
                "row_count": 3,
                "visible_suffix_token_ids_sha256": suffix_by_family[family],
            }
            for family in subject._panel_module.SEMANTIC_FAMILIES
        ],
        "rows": panel_rows,
    }
    assert hashlib.sha256(
        subject._canonical_json_bytes(panel_receipt)
    ).hexdigest() == subject._PINNED_PANEL_TOKENIZATION_CANONICAL_SHA256
    worker_rows = [
        {
            "prompt_id": record.prompt_id,
            "family_id": subject._confirmation_family_id(record),
            "stratum_id": record.stratum_id,
            "valid_token_count": record.target_valid_tokens,
            "input_ids_sha256": record.input_ids_sha256,
            "choices_sha256": record.choices_sha256,
            "choice_token_ids_sha256": subject._manifest_sha256(
                subject._TOKENIZATION_BINDING_DOMAIN,
                list(record.choice_token_ids),
            ),
            "answer_evidence_outside_window_authenticated": (
                record.answer_evidence is not None
            ),
            "answer_evidence_token_count": evidence_token_count(record),
            "visible_suffix_token_ids_sha256": (
                suffix_by_family[record.semantic_family]
                if record.answer_evidence is not None
                else None
            ),
        }
        for record in inputs.records
    ]
    batch_rows: list[dict[str, object]] = []
    for batch_index, rows in enumerate(
        subject.iter_confirmation_length_batches(inputs.records, batch_size=4)
    ):
        batch_rows.append(
            {
                "batch_index": batch_index,
                "stratum_id": rows[0].stratum_id,
                "target_valid_tokens": rows[0].target_valid_tokens,
                "row_count": len(rows),
                "prompt_id_catalog_sha256": subject._manifest_sha256(
                    subject._TOKENIZATION_BINDING_DOMAIN,
                    [row.prompt_id for row in rows],
                ),
                "no_cross_length_padding": True,
            }
        )
    receipt: dict[str, object] = {
        "tokenizer_id": subject.PINNED_MODEL_ID,
        "tokenizer_revision": subject.PINNED_MODEL_REVISION,
        "confirmation_row_count": 60,
        "exact_length_bucket_count": 10,
        "batch_count": 20,
        "batch_size_limit": 4,
        "batch_row_count_catalog": [4, 2] * 10,
        "restart_batching_at_each_exact_length_bucket": True,
        "no_cross_length_padding": True,
        "all_input_ids_attention_masks_and_lengths_authenticated": True,
        "all_choices_single_token_and_distinct": True,
        "choices_already_contained_one_leading_space": True,
        "extra_leading_space_prepended": False,
        "reserve_rows_tokenized": 0,
        "reserve_tokenizer_access_count": 0,
        "panel_audit_payload_sha256": (
            subject._PINNED_CONFIRMATION_PANEL_AUDIT_PAYLOAD_SHA256
        ),
        "panel_tokenization_receipt": panel_receipt,
        "panel_tokenization_receipt_canonical_sha256": (
            subject._PINNED_PANEL_TOKENIZATION_CANONICAL_SHA256
        ),
        "panel_tokenization_receipt_domain_sha256": (
            subject._PINNED_PANEL_TOKENIZATION_DOMAIN_SHA256
        ),
        "rows": worker_rows,
        "batches": batch_rows,
    }
    receipt["tokenization_binding_sha256"] = subject._manifest_sha256(
        subject._TOKENIZATION_BINDING_DOMAIN, receipt
    )
    return subject._AuthenticatedTokenization(batches=(), receipt=receipt)


def test_production_report_assembles_validates_writes_and_strict_loads(
    production_graph: tuple[SimpleNamespace, object],
    tmp_path: Path,
) -> None:
    inputs, _unused_graph = production_graph
    output = tmp_path / "production-confirmation.json"
    preflight = _report_preflight(inputs, output)
    tokenization = _report_tokenization(inputs)
    predecessor_measurement = preflight.predecessor["measurement"]
    state_hashes = dict(
        predecessor_measurement["mode_state_fingerprint_sha256_by_mode"]
    )
    response_hashes = {
        mode_id: hashlib.sha256(f"response:{mode_id}".encode()).hexdigest()
        for mode_id in subject.RESPONSE_MODE_IDS
    }
    response_hashes[subject.LEXICAL_MODE_IDS[1]] = response_hashes[
        subject.LEXICAL_MODE_IDS[0]
    ]
    parity: list[dict[str, object]] = []
    for batch in tokenization.receipt["batches"]:
        batch_index = int(batch["batch_index"])
        rows = tuple(
            subject.iter_confirmation_length_batches(
                inputs.records, batch_size=4
            )
        )[batch_index]
        logits_sha256 = hashlib.sha256(
            f"baseline:{batch_index}".encode()
        ).hexdigest()
        parity.append(
            {
                "batch_index": batch_index,
                "stratum_id": batch["stratum_id"],
                "target_valid_tokens": batch["target_valid_tokens"],
                "example_ids": [row.prompt_id for row in rows],
                "token_id_logits4_sha256": logits_sha256,
                "inputs_embeds_logits4_sha256": logits_sha256,
                "byte_identical": True,
            }
        )
    measurement_input = subject._measurement_input_binding(
        preflight=preflight,
        tokenization=tokenization,
        runtime_weight_catalog_sha256="a" * 64,
        baseline_parity_receipts=parity,
        response_sha256_by_mode=response_hashes,
        mode_state_fingerprint_sha256_by_mode=state_hashes,
        terminal_norm_state_fingerprint_sha256=(
            subject._PINNED_TERMINAL_NORM_STATE_FINGERPRINT_SHA256
        ),
    )
    graph_binding = subject._graph_input_binding_payload(
        nodes=inputs.nodes,
        candidate_edges=inputs.candidates,
        candidate_catalog_sha256=inputs.candidate_sha256,
        config=subject._graph_config(),
        discovery_family=inputs.discovery_family,
        discovery_examples=inputs.discovery_examples,
        discovery_bootstrap=inputs.discovery_bootstrap,
        fresh_family=inputs.fresh_family,
        fresh_examples=inputs.fresh_examples,
        fresh_bootstrap=inputs.fresh_bootstrap,
        fresh_strata=inputs.fresh_strata,
        fresh_groups=inputs.fresh_groups,
    )
    measurement_input.pop("measurement_input_sha256")
    measurement_input["stable_graph_input_binding"] = graph_binding
    measurement_input["stable_graph_input_binding_sha256"] = graph_binding[
        "graph_input_binding_sha256"
    ]
    measurement_input["measurement_input_sha256"] = subject._manifest_sha256(
        subject._MEASUREMENT_BINDING_DOMAIN, measurement_input
    )
    graph = _build_graph(
        inputs,
        fresh_panel_binding_sha256=str(
            measurement_input["measurement_input_sha256"]
        ),
    )
    measurement_receipt = {
        "batch_count": 20,
        "exact_length_bucket_count": 10,
        "example_count": 60,
        "response_width": 4,
        "token_id_baseline_forward_count": 20,
        "inputs_embeds_baseline_forward_count": 20,
        "logical_intervention_forward_count": 760,
        "logical_intervention_forward_count_by_mode": {
            mode_id: 20 for mode_id in subject.RESPONSE_MODE_IDS
        },
        "full_model_forward_count": 800,
        "backward_count": 0,
        "baseline_parity_receipts": parity,
        "all_baseline_parity_receipts_byte_identical": True,
        "primary_response": "C4_times_z_scale1_minus_z_scale0",
        "analytic_output_Fisher_weights": True,
        "runtime_output_Fisher_weight_catalog_sha256": "a" * 64,
        "runtime_output_Fisher_weight_catalog_row_count": 240,
        "response_sha256_by_mode": response_hashes,
        "mode_state_fingerprint_sha256_by_mode": state_hashes,
        "terminal_norm_state_fingerprint_sha256": (
            subject._PINNED_TERMINAL_NORM_STATE_FINGERPRINT_SHA256
        ),
        "measurement_input_binding": measurement_input,
        "measurement_input_sha256": measurement_input[
            "measurement_input_sha256"
        ],
        "fresh_family_pair_stat_count": 4218,
        "fresh_family_stats_catalog_sha256": graph_binding[
            "fresh_family_stats_catalog_sha256"
        ],
        "fresh_example_pair_stat_count": 42180,
        "fresh_example_stats_catalog_sha256": graph_binding[
            "fresh_example_stats_catalog_sha256"
        ],
        "stable_graph_input_binding_sha256": graph_binding[
            "graph_input_binding_sha256"
        ],
        "tied_lexical_shared_state_fingerprint_sha256": state_hashes[
            subject.LEXICAL_MODE_IDS[0]
        ],
        "tied_lexical_distinct_occurrence_interventions_executed": True,
        "tied_lexical_responses_byte_identical": True,
        "target_hook_call_count_by_mutation_mode": {
            mode_id: 20
            for mode_id in (*subject.ATTENTION_MODE_IDS, *subject.MLP_MODE_IDS)
        },
        "lexical_egress_occurrence_hook_call_count": 20,
        "length_bucket_batching_restarted_per_exact_length": True,
        "cross_length_padding_used": False,
        "reserve_model_access_count": 0,
        "reserve_tokenizer_access_count": 0,
        "elapsed_seconds": 0.0,
        "peak_rss_bytes_before": 0,
        "peak_rss_bytes_after": 0,
    }

    class FakeAdapter:
        @staticmethod
        def model_fingerprint() -> str:
            return subject.PINNED_SOURCE_MODEL_FINGERPRINT_SHA256

        @staticmethod
        def execution_fingerprint() -> str:
            return subject.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256

    runtime = SimpleNamespace(
        model_receipt={
            "model_id": subject.PINNED_MODEL_ID,
            "requested_revision": subject.PINNED_MODEL_REVISION,
            "resolved_commit": subject.PINNED_MODEL_REVISION,
            "parameter_count": subject.PINNED_SOURCE_PARAMETER_COUNT,
            "device": "cpu",
            "dtype": "float32",
            "weights_in_report": False,
        },
        source_model_fingerprint_before=(
            subject.PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
        ),
        source_execution_fingerprint_before=(
            subject.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        adapter=FakeAdapter(),
    )
    measurement = subject._ResponseMeasurement(
        example_ids=tuple(record.prompt_id for record in inputs.records),
        family_ids_by_example=tuple(
            subject._confirmation_family_id(record) for record in inputs.records
        ),
        stratum_ids_by_example=tuple(
            str(record.stratum_id) for record in inputs.records
        ),
        responses_by_mode={},
        weights_by_example=(),
        state_fingerprints_by_mode=state_hashes,
        terminal_norm_state_fingerprint_sha256=(
            subject._PINNED_TERMINAL_NORM_STATE_FINGERPRINT_SHA256
        ),
        receipt=measurement_receipt,
    )
    report = subject._assemble_report(
        preflight=preflight,
        tokenization=tokenization,
        runtime=runtime,
        measurement=measurement,
        graph_receipt=graph,
        device_name="cpu",
        dtype="float32",
        batch_size=4,
    )
    validated = subject.validate_gemma3_stable_mutual_knn_confirmation_report(
        report
    )
    loaded = subject.write_gemma3_stable_mutual_knn_confirmation_report(
        validated, output
    )
    assert loaded == validated
    assert loaded["graph_summary"]["confirmed_edge_count"] == 1
    assert output.stat().st_size == loaded["resource_receipt"][
        "outer_report_serialized_bytes"
    ]
    assert subject._default_svg_path(output).is_file()

    def reject_outer_rehash(
        label: str, mutate: object
    ) -> None:
        tampered = copy.deepcopy(loaded)
        mutate(tampered)
        for _attempt in range(12):
            tampered_without_hash = dict(tampered)
            tampered_without_hash.pop("report_sha256", None)
            tampered["report_sha256"] = subject._report_sha256(
                tampered_without_hash
            )
            serialized_bytes = len(subject._compact_report_bytes(tampered))
            if tampered["resource_receipt"][
                "outer_report_serialized_bytes"
            ] == serialized_bytes:
                break
            tampered["resource_receipt"][
                "outer_report_serialized_bytes"
            ] = serialized_bytes
        else:
            raise AssertionError(f"{label} byte receipt did not converge")
        tampered_without_hash = dict(tampered)
        tampered_without_hash.pop("report_sha256", None)
        tampered["report_sha256"] = subject._report_sha256(
            tampered_without_hash
        )
        assert len(subject._compact_report_bytes(tampered)) == tampered[
            "resource_receipt"
        ]["outer_report_serialized_bytes"]
        with pytest.raises(ValueError, match=".+"):
            subject.validate_gemma3_stable_mutual_knn_confirmation_report(
                tampered
            )

    mutations = {
        "lineage": lambda report: report["setup_and_predecessor"].__setitem__(
            "setup_sha256", "0" * 64
        ),
        "panel": lambda report: report["setup_and_predecessor"].__setitem__(
            "panel_file_sha256", "0" * 64
        ),
        "worker_source": lambda report: report[
            "setup_and_predecessor"
        ].__setitem__("confirmation_worker_source_sha256", "0" * 64),
        "tokenization": lambda report: report["panel_firewall"][
            "tokenization"
        ]["rows"][0].__setitem__("prompt_id", "x" * len(inputs.records[0].prompt_id)),
        "measurement": lambda report: report["measurement"].__setitem__(
            "fresh_family_stats_catalog_sha256", "0" * 64
        ),
        "exact_length_bucket_count": lambda report: report[
            "measurement"
        ].__setitem__("exact_length_bucket_count", 11),
        "elapsed_seconds_type": lambda report: report[
            "measurement"
        ].__setitem__("elapsed_seconds", "0"),
        "rss_monotonicity": lambda report: report["measurement"].__setitem__(
            "peak_rss_bytes_before", 1
        ),
        "tied_lexical_fingerprint": lambda report: report[
            "measurement"
        ].__setitem__("tied_lexical_shared_state_fingerprint_sha256", "0" * 64),
        "baseline_digest_shape": lambda report: report["measurement"][
            "baseline_parity_receipts"
        ][0].__setitem__("token_id_logits4_sha256", "g" * 64),
        "fisher_digest_shape": lambda report: report[
            "measurement"
        ].__setitem__("runtime_output_Fisher_weight_catalog_sha256", "g" * 64),
        "intervention_count_type": lambda report: report["measurement"][
            "logical_intervention_forward_count_by_mode"
        ].__setitem__(subject.RESPONSE_MODE_IDS[0], True),
        "process_rss_type": lambda report: report[
            "execution_audit"
        ].__setitem__("process_peak_rss_bytes", False),
        "core": lambda report: report["graph_receipt"]["input"][
            "node_specs"
        ][0].__setitem__("node_id", "tampered.node"),
        "gates": lambda report: report["configuration"].__setitem__("k", 3),
        "components": lambda report: report["graph_summary"].__setitem__(
            "nontrivial_connected_component_count", 9
        ),
        "cliques": lambda report: report["graph_summary"].__setitem__(
            "maximal_stable_neighbor_group_count", 9
        ),
        "claims": lambda report: report["claims"].__setitem__(
            "compression_qualified", True
        ),
        "unknown_claim": lambda report: report["claims"].__setitem__(
            "compression_succeeded", True
        ),
        "resources": lambda report: report["resource_receipt"].__setitem__(
            "actual_embedded_core_canonical_bytes", 0
        ),
        "svg": lambda report: report["visualization"].__setitem__(
            "file_sha256", "0" * 64
        ),
    }
    for label, mutate in mutations.items():
        reject_outer_rehash(label, mutate)

    with pytest.raises(FileExistsError, match="overwrite"):
        subject.write_gemma3_stable_mutual_knn_confirmation_report(
            loaded, output
        )
