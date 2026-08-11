from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest
import torch

import fisher_graph.gemma3_parts_first_attention_edge_experiment as edge_experiment
from fisher_graph.adapters import (
    AttentionSpec,
    NormalizationSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
)
from fisher_graph.gemma3_attention_generator import (
    ExactAttentionGeneratorCoefficients,
    Gemma3AttentionGeneratorConfig,
    compile_exact_attention_generator,
)
from fisher_graph.gemma3_parts_first_attention_edge_experiment import (
    ATTENTION_DISTANCE_BINS,
    ATTENTION_EDGE_ATLAS_ROLE,
    ATTENTION_EDGE_TOP_K,
    AttentionEdgeAtlasConfig,
    AttentionEdgeAtlasLiveConfig,
    AttentionEdgeAtlasRowCapture,
    AttentionEdgeExactControlReceipt,
    RetainedAttentionRoute,
    build_attention_edge_atlas_preflight,
    build_deterministic_attention_edge_fixture,
    compile_attention_edge_atlas,
    load_attention_edge_atlas,
    main,
    parse_attention_edge_atlas_json,
    write_attention_edge_atlas,
)
from fisher_graph.gemma3_parts_first_attention_edges import (
    compile_parts_first_attention_edges,
)
from fisher_graph.gemma3_parts_first_supermodes import (
    SOURCE_UNIQUE_PARAMETER_COUNT,
)


def _tiny_extraction(
    *,
    query_heads: int = 2,
    sequence_length: int = 3,
) -> tuple[object, SequenceContext]:
    norm = NormalizationSpec(
        kind="rms_norm",
        width=4,
        epsilon=1e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )
    qk_norm = replace(norm, width=2)
    config = Gemma3AttentionGeneratorConfig(
        invocation_id="fixture.attention",
        residual_width=4,
        attention=AttentionSpec(
            kind="global_causal",
            query_heads=query_heads,
            key_value_heads=1,
            head_dimension=2,
            query_scale=0.5,
            qk_norm=True,
            window_size=None,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=2,
            ),
            cache_kind="none",
        ),
        input_norm=norm,
        output_norm=norm,
        qk_norm=qk_norm,
        projection_bias=False,
        dropout=0.0,
        logit_softcap=7.0,
    )
    generator_state = torch.Generator().manual_seed(81_744)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator_state) * 0.1

    generator = compile_exact_attention_generator(
        config,
        ExactAttentionGeneratorCoefficients(
            input_norm_weight=random(4),
            q_weight=random(query_heads * 2, 4),
            k_weight=random(2, 4),
            v_weight=random(2, 4),
            o_weight=random(4, query_heads * 2),
            output_norm_weight=random(4),
            q_norm_weight=random(2),
            k_norm_weight=random(2),
        ),
    ).eval()
    mask = torch.ones(1, sequence_length, dtype=torch.bool)
    positions = torch.arange(sequence_length, dtype=torch.int64).reshape(1, -1)
    sequence = SequenceContext(
        query_valid_mask=mask,
        key_valid_mask=mask,
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=False,
        ),
    )
    extraction = compile_parts_first_attention_edges(
        generator,
        random(1, sequence_length, 4),
        random(1, sequence_length, 4),
        sequence,
    )
    return extraction, sequence


def _fixture_source_hash() -> str:
    return "1" * 64


def _rehash_report_dict(value: dict[str, object]) -> None:
    payload = dict(value)
    payload.pop("report_sha256")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    value["report_sha256"] = hashlib.sha256(
        b"fisher_graph:parts-first-attention-edge-atlas:v1\0" + canonical
    ).hexdigest()


def _rehash_row_dict(value: dict[str, object]) -> None:
    payload = dict(value)
    payload.pop("row_sha256")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    value["row_sha256"] = hashlib.sha256(
        b"fisher_graph:parts-first-attention-edge-row:v1\0" + canonical
    ).hexdigest()


def _rehash_implementation_bundle(value: dict[str, object]) -> None:
    canonical = json.dumps(
        {
            "core_implementation_sha256": value[
                "core_implementation_sha256"
            ],
            "runner_implementation_sha256": value[
                "runner_implementation_sha256"
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    value["implementation_bundle_sha256"] = hashlib.sha256(
        b"fisher_graph:parts-first-attention-edge-implementation:v1\0"
        + canonical
    ).hexdigest()


def _exact_control_receipt() -> AttentionEdgeExactControlReceipt:
    digest = "2" * 64
    return AttentionEdgeExactControlReceipt(
        ordered_event_ids=(
            "lexical.ingress",
            *(f"mutation.{index}" for index in range(1, 37)),
            "terminal.rmsnorm",
            "lexical.egress",
        ),
        compiler_metadata_sha256=digest,
        source_independence_audit_sha256=digest,
        native_valid_logits_sha256=digest,
        exact_valid_logits_sha256=digest,
        valid_logit_shape=(1, edge_experiment.VOCABULARY_SIZE),
        literal_native_exact_parity_passed=True,
        source_model_fingerprint_before=(
            edge_experiment.SOURCE_MODEL_FINGERPRINT_SHA256
        ),
        source_model_fingerprint_after_parity=(
            edge_experiment.SOURCE_MODEL_FINGERPRINT_SHA256
        ),
        source_execution_fingerprint_before=(
            edge_experiment.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        source_execution_fingerprint_after_parity=(
            edge_experiment.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        source_parameter_count=SOURCE_UNIQUE_PARAMETER_COUNT,
        exact_parameter_inventory_parity=True,
        source_independence_passed=True,
        native_source_references_released_before_atlas=True,
    )


def _synthetic_model_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    extraction, _ = _tiny_extraction(query_heads=4, sequence_length=4)
    row_catalog_id = "synthetic-smoke-row"
    family_id = "integer_addition"
    tokenization_sha256 = "3" * 64
    row_catalog_sha256 = edge_experiment._json_sha256(  # noqa: SLF001
        [
            {
                "row_ordinal": 0,
                "row_catalog_id": row_catalog_id,
                "family_id": family_id,
            }
        ],
        domain=edge_experiment._CATALOG_DOMAIN,  # noqa: SLF001
    )
    tokenization_catalog_sha256 = (
        edge_experiment._tokenization_catalog_sha256(  # noqa: SLF001
            (tokenization_sha256,)
        )
    )
    monkeypatch.setattr(edge_experiment, "GEMMA3_LAYER_COUNT", 1)
    monkeypatch.setattr(edge_experiment, "ATTENTION_HEAD_COUNT", 4)
    monkeypatch.setattr(
        edge_experiment,
        "PINNED_ATTENTION_EDGE_SMOKE_VALID_TOKEN_COUNT",
        4,
    )
    monkeypatch.setattr(
        edge_experiment,
        "PINNED_ATTENTION_EDGE_SMOKE_ROW_CATALOG_SHA256",
        row_catalog_sha256,
    )
    monkeypatch.setattr(
        edge_experiment,
        "PINNED_ATTENTION_EDGE_SMOKE_TOKENIZATION_CATALOG_SHA256",
        tokenization_catalog_sha256,
    )
    capture = AttentionEdgeAtlasRowCapture(
        row_catalog_id=row_catalog_id,
        family_id=family_id,
        logical_positions=(0, 1, 2, 3),
        supervised_query_ordinal=3,
        tokenization_sha256=tokenization_sha256,
        layer_extractions=(extraction,),
    )
    return compile_attention_edge_atlas(
        (capture,),
        source_kind="model_smoke",
        model_id=edge_experiment.MODEL_ID,
        model_revision=edge_experiment.PINNED_MODEL_REVISION,
        panel_id=edge_experiment.PINNED_PANEL_ID,
        panel_file_sha256=edge_experiment.PINNED_PANEL_FILE_SHA256,
        source_model_fingerprint_sha256=(
            edge_experiment.SOURCE_MODEL_FINGERPRINT_SHA256
        ),
        source_execution_fingerprint_sha256=(
            edge_experiment.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        score_objective_id=edge_experiment.MODEL_SCORE_OBJECTIVE_ID,
        exact_control_receipt=_exact_control_receipt(),
    )


def test_import_is_transformers_and_model_inert() -> None:
    command = (
        "import sys; "
        "import fisher_graph.gemma3_parts_first_attention_edge_experiment; "
        "assert 'transformers' not in sys.modules"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path.cwd() / "src")
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path.cwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_deterministic_fixture_collapses_roles_and_caps_across_heads() -> None:
    first = build_deterministic_attention_edge_fixture()
    second = build_deterministic_attention_edge_fixture()
    assert first == second
    assert first.report_sha256 == second.report_sha256
    assert first.roles_statistically_read == (ATTENTION_EDGE_ATLAS_ROLE,)
    assert first.old_roles_collapsed
    assert first.retrospective_A_only
    assert first.structure_only
    assert not first.attention_edge_confirmation_claimed
    assert not first.attention_promotion_authorized
    assert not first.compression_claimed
    assert len(first.distance_motifs) == 2 * 2 * len(ATTENTION_DISTANCE_BINS)

    by_target: dict[tuple[int, int, int], list[RetainedAttentionRoute]] = {}
    for route in first.retained_routes:
        by_target.setdefault(
            (route.row_ordinal, route.layer_ordinal, route.query_ordinal), []
        ).append(route)
    assert max(len(routes) for routes in by_target.values()) == ATTENTION_EDGE_TOP_K
    assert all(len(routes) <= ATTENTION_EDGE_TOP_K for routes in by_target.values())
    assert any(
        len({route.query_head_ordinal for route in routes}) > 1
        for routes in by_target.values()
        if len(routes) == ATTENTION_EDGE_TOP_K
    )
    assert first.retained_route_count <= sum(
        row.sequence_length * first.layer_count * ATTENTION_EDGE_TOP_K
        for row in first.row_receipts
    )


def test_real_core_extraction_reduces_without_raw_tensor_payloads() -> None:
    extraction, _ = _tiny_extraction()
    capture = AttentionEdgeAtlasRowCapture(
        row_catalog_id="tiny-row",
        family_id="tiny-family",
        logical_positions=(0, 1, 2),
        supervised_query_ordinal=2,
        tokenization_sha256="3" * 64,
        layer_extractions=(extraction,),
    )
    report = compile_attention_edge_atlas(
        (capture,),
        source_kind="model_free_fixture",
        model_id="tiny",
        model_revision="v1",
        panel_id="tiny-panel",
        panel_file_sha256=_fixture_source_hash(),
        source_model_fingerprint_sha256=_fixture_source_hash(),
        source_execution_fingerprint_sha256=_fixture_source_hash(),
        score_objective_id="tiny-score",
        exact_control_receipt=None,
    )
    assert report.full_allowed_route_count == 12
    assert report.retained_route_count == 12
    assert report.row_receipts[0].layer_extraction_sha256s == (
        extraction.sha256,
    )
    assert all(
        math.isfinite(item.softmax_logit_tangent)
        for item in report.retained_routes
    )
    encoded = json.dumps(report.to_dict(), sort_keys=True)
    assert '"prompt"' not in encoded
    assert '"input_ids"' not in encoded
    assert '"token_ids"' not in encoded
    assert '"raw_activations"' not in encoded
    assert '"raw_gradients"' not in encoded


def test_full_route_motifs_and_sparse_energy_ledgers_close() -> None:
    report = build_deterministic_attention_edge_fixture()
    assert sum(item.full_route_count for item in report.distance_motifs) == (
        report.full_allowed_route_count
    )
    assert sum(item.retained_route_count for item in report.distance_motifs) == (
        report.retained_route_count
    )
    assert math.isclose(
        math.fsum(
            item.intrinsic_fisher_energy_sum
            for item in report.distance_motifs
        ),
        report.full_intrinsic_fisher_energy,
        rel_tol=2e-13,
        abs_tol=1e-12,
    )
    assert math.isclose(
        math.fsum(
            item.intrinsic_fisher_energy
            for item in report.retained_routes
        ),
        report.retained_intrinsic_fisher_energy,
        rel_tol=2e-13,
        abs_tol=1e-12,
    )
    assert math.isclose(
        math.fsum(
            item.retained_softmax_logit_tangent_sum
            for item in report.distance_motifs
        ),
        math.fsum(
            item.softmax_logit_tangent for item in report.retained_routes
        ),
        rel_tol=2e-13,
        abs_tol=1e-12,
    )


def test_parser_is_strict_and_replays_hashes() -> None:
    report = build_deterministic_attention_edge_fixture()
    encoded = json.dumps(report.to_dict(), sort_keys=True, allow_nan=False)
    assert parse_attention_edge_atlas_json(encoded) == report
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_attention_edge_atlas_json('{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        parse_attention_edge_atlas_json('{"x":NaN}')

    tampered = report.to_dict()
    routes = tampered["retained_routes"]
    assert isinstance(routes, list) and isinstance(routes[0], dict)
    routes[0]["intrinsic_fisher_energy"] = float(
        routes[0]["intrinsic_fisher_energy"]
    ) + 1.0
    with pytest.raises(ValueError):
        parse_attention_edge_atlas_json(json.dumps(tampered))

    unknown = report.to_dict()
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="field inventory"):
        parse_attention_edge_atlas_json(json.dumps(unknown))

    provenance = report.to_dict()
    provenance["runner_implementation_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="implementation bundle"):
        parse_attention_edge_atlas_json(json.dumps(provenance))


def test_parser_rejects_recomputed_hash_with_broken_sparse_cap() -> None:
    report = build_deterministic_attention_edge_fixture()
    target = next(
        key
        for key in {
            (item.row_ordinal, item.layer_ordinal, item.query_ordinal)
            for item in report.retained_routes
        }
        if sum(
            (item.row_ordinal, item.layer_ordinal, item.query_ordinal) == key
            for item in report.retained_routes
        )
        == ATTENTION_EDGE_TOP_K
    )
    source = next(
        item
        for item in report.retained_routes
        if (item.row_ordinal, item.layer_ordinal, item.query_ordinal) == target
    )
    extra = replace(source, target_rank=ATTENTION_EDGE_TOP_K)
    # Construction alone cannot create rank 9, and report validation also
    # catches duplicated/non-contiguous target ranks before trusting its hash.
    with pytest.raises(ValueError):
        replace(report, retained_routes=(*report.retained_routes, extra))


def test_parser_rejects_recomputed_hash_with_relabelled_causal_distance() -> None:
    report = build_deterministic_attention_edge_fixture()
    tampered = report.to_dict()
    routes = tampered["retained_routes"]
    assert isinstance(routes, list)
    route = next(
        item
        for item in routes
        if isinstance(item, dict) and item.get("distance_bin") == "2-3"
    )
    distance = route["causal_distance"]
    assert type(distance) is int and distance in {2, 3}
    route["causal_distance"] = 3 if distance == 2 else 2
    _rehash_report_dict(tampered)

    with pytest.raises(ValueError, match="causal distance does not match"):
        parse_attention_edge_atlas_json(json.dumps(tampered))


@pytest.mark.parametrize(
    ("field", "bound_field", "message"),
    (
        (
            "signed_response_sum",
            "absolute_signed_response_sum",
            "response algebra",
        ),
        (
            "softmax_logit_tangent_sum",
            "absolute_softmax_logit_tangent_sum",
            "tangent algebra",
        ),
        (
            "maximum_intrinsic_fisher_energy",
            "intrinsic_fisher_energy_sum",
            "maximum energy",
        ),
        (
            "probability_sum",
            "full_route_count",
            "probability exceeds route count",
        ),
    ),
)
def test_parser_rejects_rehashed_motif_algebra_contradictions(
    field: str,
    bound_field: str,
    message: str,
) -> None:
    tampered = build_deterministic_attention_edge_fixture().to_dict()
    motifs = tampered["distance_motifs"]
    assert isinstance(motifs, list)
    motif = next(
        item
        for item in motifs
        if isinstance(item, dict) and int(item["full_route_count"]) > 0
    )
    motif[field] = float(motif[bound_field]) + 1.0
    _rehash_report_dict(tampered)

    with pytest.raises(ValueError, match=message):
        parse_attention_edge_atlas_json(json.dumps(tampered))


def test_parser_rejects_rehashed_empty_motif_and_aggregate_cauchy_attacks() -> None:
    empty_attack = build_deterministic_attention_edge_fixture().to_dict()
    motifs = empty_attack["distance_motifs"]
    assert isinstance(motifs, list)
    empty = next(
        item
        for item in motifs
        if isinstance(item, dict) and item["full_route_count"] == 0
    )
    empty["full_route_count"] = 1
    _rehash_report_dict(empty_attack)
    with pytest.raises(ValueError, match="nonempty full motif"):
        parse_attention_edge_atlas_json(json.dumps(empty_attack))

    cauchy_attack = build_deterministic_attention_edge_fixture().to_dict()
    motifs = cauchy_attack["distance_motifs"]
    assert isinstance(motifs, list)
    motif = next(
        item
        for item in motifs
        if isinstance(item, dict) and int(item["full_route_count"]) > 0
    )
    motif["absolute_signed_response_sum"] = 1.0e9
    _rehash_report_dict(cauchy_attack)
    with pytest.raises(ValueError, match="aggregate Fisher Cauchy"):
        parse_attention_edge_atlas_json(json.dumps(cauchy_attack))


def test_parser_rejects_rehashed_head_probability_and_tangent_attacks() -> None:
    probability_attack = build_deterministic_attention_edge_fixture().to_dict()
    motifs = probability_attack["distance_motifs"]
    assert isinstance(motifs, list)
    motif = next(
        item
        for item in motifs
        if isinstance(item, dict)
        and float(item["full_route_count"])
        - float(item["probability_sum"])
        > 0.1
    )
    motif["probability_sum"] = float(motif["probability_sum"]) + 0.01
    _rehash_report_dict(probability_attack)
    with pytest.raises(ValueError, match="head probability mass"):
        parse_attention_edge_atlas_json(json.dumps(probability_attack))

    tangent_attack = build_deterministic_attention_edge_fixture().to_dict()
    motifs = tangent_attack["distance_motifs"]
    assert isinstance(motifs, list)
    motif = next(
        item
        for item in motifs
        if isinstance(item, dict) and int(item["full_route_count"]) > 0
    )
    motif["softmax_logit_tangent_sum"] = (
        float(motif["softmax_logit_tangent_sum"]) + 1.0
    )
    motif["absolute_softmax_logit_tangent_sum"] = (
        float(motif["absolute_softmax_logit_tangent_sum"]) + 1.0
    )
    _rehash_report_dict(tangent_attack)
    with pytest.raises(ValueError, match="head softmax tangent"):
        parse_attention_edge_atlas_json(json.dumps(tangent_attack))


def test_parser_rejects_rehashed_retained_route_equation_attacks() -> None:
    zero_probability = build_deterministic_attention_edge_fixture().to_dict()
    routes = zero_probability["retained_routes"]
    assert isinstance(routes, list) and isinstance(routes[0], dict)
    routes[0]["probability"] = 0.0
    routes[0]["signed_response"] = 1.0e9
    routes[0]["intrinsic_fisher_energy"] = 1.0
    routes[0]["softmax_logit_tangent"] = 0.0
    _rehash_report_dict(zero_probability)
    with pytest.raises(ValueError, match="zero-probability"):
        parse_attention_edge_atlas_json(json.dumps(zero_probability))

    cauchy_attack = build_deterministic_attention_edge_fixture().to_dict()
    routes = cauchy_attack["retained_routes"]
    assert isinstance(routes, list) and isinstance(routes[0], dict)
    routes[0]["signed_response"] = 1.0e9
    routes[0]["intrinsic_fisher_energy"] = 1.0
    _rehash_report_dict(cauchy_attack)
    with pytest.raises(ValueError, match="Fisher Cauchy"):
        parse_attention_edge_atlas_json(json.dumps(cauchy_attack))


def test_parser_rejects_coordinated_shared_response_and_ledger_attacks() -> None:
    report = build_deterministic_attention_edge_fixture()
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for index, route in enumerate(report.retained_routes):
        groups.setdefault(
            (
                route.row_ordinal,
                route.layer_ordinal,
                route.query_head_ordinal,
                route.query_ordinal,
            ),
            [],
        ).append(index)
    route_index = next(
        index
        for indices in groups.values()
        if len(indices) >= 2
        for index in indices
        if report.retained_routes[index].softmax_logit_tangent != 0.0
    )
    route = report.retained_routes[route_index]
    tampered = report.to_dict()
    routes = tampered["retained_routes"]
    motifs = tampered["distance_motifs"]
    assert isinstance(routes, list) and isinstance(routes[route_index], dict)
    assert isinstance(motifs, list)
    stored = routes[route_index]
    old_tangent = float(stored["softmax_logit_tangent"])
    new_tangent = 0.5 * old_tangent
    stored["softmax_logit_tangent"] = new_tangent
    motif = next(
        item
        for item in motifs
        if isinstance(item, dict)
        and item["layer_ordinal"] == route.layer_ordinal
        and item["query_head_ordinal"] == route.query_head_ordinal
        and item["distance_bin"] == route.distance_bin
    )
    motif["retained_softmax_logit_tangent_sum"] = (
        float(motif["retained_softmax_logit_tangent_sum"])
        + new_tangent
        - old_tangent
    )
    motif["retained_absolute_softmax_logit_tangent_sum"] = (
        float(motif["retained_absolute_softmax_logit_tangent_sum"])
        + abs(new_tangent)
        - abs(old_tangent)
    )
    _rehash_report_dict(tampered)
    with pytest.raises(ValueError, match="shared softmax response"):
        parse_attention_edge_atlas_json(json.dumps(tampered))

    ledger_attack = report.to_dict()
    motifs = ledger_attack["distance_motifs"]
    assert isinstance(motifs, list)
    motif = next(
        item
        for item in motifs
        if isinstance(item, dict)
        and float(item["retained_signed_response_sum"]) > 1e-6
    )
    motif["retained_signed_response_sum"] = (
        float(motif["retained_signed_response_sum"]) - 1e-6
    )
    _rehash_report_dict(ledger_attack)
    with pytest.raises(ValueError, match="scalar ledgers"):
        parse_attention_edge_atlas_json(json.dumps(ledger_attack))


def test_report_rejects_duplicate_retained_route_coordinates() -> None:
    report = build_deterministic_attention_edge_fixture()
    by_target_motif: dict[
        tuple[int, int, int, int, str], list[RetainedAttentionRoute]
    ] = {}
    for route in report.retained_routes:
        by_target_motif.setdefault(
            (
                route.row_ordinal,
                route.layer_ordinal,
                route.query_ordinal,
                route.query_head_ordinal,
                route.distance_bin,
            ),
            [],
        ).append(route)
    source, original = next(
        routes[:2] for routes in by_target_motif.values() if len(routes) >= 2
    )
    assert source.key_ordinal != original.key_ordinal
    duplicate_index = next(
        index
        for index, route in enumerate(report.retained_routes)
        if route is original
    )
    duplicate = replace(
        original,
        key_ordinal=source.key_ordinal,
        causal_distance=source.causal_distance,
    )
    mutated_routes = (
        *report.retained_routes[:duplicate_index],
        duplicate,
        *report.retained_routes[duplicate_index + 1 :],
    )

    with pytest.raises(ValueError, match="coordinate repeats"):
        replace(
            report,
            retained_routes=mutated_routes,
            report_sha256="",
        )


def test_model_smoke_rejects_rehashed_provenance_and_row_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_model_smoke(monkeypatch)

    provenance = report.to_dict()
    provenance["core_implementation_sha256"] = "4" * 64
    provenance["runner_implementation_sha256"] = "5" * 64
    _rehash_implementation_bundle(provenance)
    _rehash_report_dict(provenance)
    with pytest.raises(ValueError, match="implementation provenance"):
        parse_attention_edge_atlas_json(json.dumps(provenance))

    supervised = report.to_dict()
    rows = supervised["row_receipts"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0]["supervised_query_ordinal"] = 2
    _rehash_row_dict(rows[0])
    _rehash_report_dict(supervised)
    with pytest.raises(ValueError, match="row causal geometry"):
        parse_attention_edge_atlas_json(json.dumps(supervised))

    counts = report.to_dict()
    rows = counts["row_receipts"]
    motifs = counts["distance_motifs"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert isinstance(motifs, list) and isinstance(motifs[0], dict)
    rows[0]["full_allowed_route_count"] = (
        int(rows[0]["full_allowed_route_count"]) + 1
    )
    counts["full_allowed_route_count"] = (
        int(counts["full_allowed_route_count"]) + 1
    )
    motifs[0]["full_route_count"] = int(motifs[0]["full_route_count"]) + 1
    _rehash_row_dict(rows[0])
    _rehash_report_dict(counts)
    with pytest.raises(ValueError, match="row causal geometry"):
        parse_attention_edge_atlas_json(json.dumps(counts))


def test_model_smoke_rejects_rehashed_per_bin_full_count_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_model_smoke(monkeypatch)
    tampered = report.to_dict()
    motifs = tampered["distance_motifs"]
    assert isinstance(motifs, list)
    donor = next(
        item
        for item in motifs
        if isinstance(item, dict)
        and item["layer_ordinal"] == 0
        and item["query_head_ordinal"] == 0
        and item["distance_bin"] == "2-3"
    )
    receiver = next(
        item
        for item in motifs
        if isinstance(item, dict)
        and item["layer_ordinal"] == 0
        and item["query_head_ordinal"] == 0
        and item["distance_bin"] == "0"
    )
    donor["full_route_count"] = int(donor["full_route_count"]) - 1
    receiver["full_route_count"] = int(receiver["full_route_count"]) + 1
    _rehash_report_dict(tampered)

    with pytest.raises(ValueError, match="motif causal geometry"):
        parse_attention_edge_atlas_json(json.dumps(tampered))


def test_atomic_writer_is_0600_no_clobber_and_strictly_reloadable(
    tmp_path: Path,
) -> None:
    report = build_deterministic_attention_edge_fixture()
    output = tmp_path / "atlas.json"
    assert write_attention_edge_atlas(report, output) == report
    metadata = output.lstat()
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert load_attention_edge_atlas(output) == report
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_attention_edge_atlas(report, output)
    assert load_attention_edge_atlas(output) == report
    assert not tuple(tmp_path.glob(".atlas.json.*.tmp"))


def test_strict_loader_rejects_symlink_mode_and_multilink(
    tmp_path: Path,
) -> None:
    report = build_deterministic_attention_edge_fixture()

    target = tmp_path / "target.json"
    write_attention_edge_atlas(report, target)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="unavailable or unsafe"):
        load_attention_edge_atlas(symlink)

    target.chmod(0o644)
    with pytest.raises(ValueError, match="identity is unsafe"):
        load_attention_edge_atlas(target)
    target.chmod(0o600)

    alias = tmp_path / "alias.json"
    os.link(target, alias)
    with pytest.raises(ValueError, match="identity is unsafe"):
        load_attention_edge_atlas(target)


def test_strict_loader_rejects_path_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_deterministic_attention_edge_fixture()
    target = tmp_path / "target.json"
    write_attention_edge_atlas(report, target)
    encoded = target.read_bytes()
    real_read = edge_experiment.os.read
    replaced = False

    def replacing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        if not replaced:
            replaced = True
            target.unlink()
            target.write_bytes(encoded)
            target.chmod(0o600)
        return real_read(descriptor, count)

    monkeypatch.setattr(edge_experiment.os, "read", replacing_read)
    with pytest.raises(ValueError, match="changed during read"):
        load_attention_edge_atlas(target)
    assert target.read_bytes() == encoded


@pytest.mark.parametrize("failure_kind", ("directory_fsync", "canonical_reload"))
def test_writer_rolls_back_owned_target_after_post_link_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    report = build_deterministic_attention_edge_fixture()
    target = tmp_path / "target.json"
    if failure_kind == "directory_fsync":
        def fail_fsync(_: Path) -> None:
            raise RuntimeError("injected directory fsync failure")

        monkeypatch.setattr(edge_experiment, "_fsync_directory", fail_fsync)
        message = "injected directory fsync failure"
    else:
        real_load = edge_experiment._load_report_path  # noqa: SLF001

        def fail_canonical(path: Path) -> object:
            if path == target:
                raise RuntimeError("injected canonical reload failure")
            return real_load(path)

        monkeypatch.setattr(edge_experiment, "_load_report_path", fail_canonical)
        message = "injected canonical reload failure"

    with pytest.raises(RuntimeError, match=message):
        write_attention_edge_atlas(report, target)
    assert not target.exists()
    assert not tuple(tmp_path.glob(".target.json.*.tmp"))


def test_writer_preserves_replacement_target_during_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_deterministic_attention_edge_fixture()
    target = tmp_path / "target.json"
    replacement = b"replacement owned by another writer\n"
    real_load = edge_experiment._load_report_path  # noqa: SLF001

    def replace_then_fail(path: Path) -> object:
        if path == target:
            target.unlink()
            target.write_bytes(replacement)
            target.chmod(0o600)
            raise RuntimeError("injected replacement race")
        return real_load(path)

    monkeypatch.setattr(edge_experiment, "_load_report_path", replace_then_fail)
    with pytest.raises(RuntimeError, match="injected replacement race"):
        write_attention_edge_atlas(report, target)
    assert target.read_bytes() == replacement
    assert not tuple(tmp_path.glob(".target.json.*.tmp"))


def test_writer_rejects_non_json_without_creating_parent(tmp_path: Path) -> None:
    report = build_deterministic_attention_edge_fixture()
    output = tmp_path / "missing" / "atlas.txt"
    with pytest.raises(ValueError, match="must use .json"):
        write_attention_edge_atlas(report, output)
    assert not output.parent.exists()


def test_preflight_is_model_free_and_binds_72_by_8_grid() -> None:
    preflight = build_attention_edge_atlas_preflight()
    assert preflight["model_loaded"] is False
    assert preflight["transformers_imported_by_runner"] is False
    assert preflight["panel_row_count"] == 60
    assert preflight["motif_cell_count"] == 72 * 8
    assert preflight["valid_token_count"] == 814
    assert preflight["maximum_retained_route_count"] == 117_216
    assert len(preflight["tokenization_catalog_sha256"]) == 64
    assert preflight["tokenization_catalog_sha256"] != "0" * 64
    assert preflight["supervised_targets_per_row"] == 1
    assert preflight["supervised_target_location"] == "final_valid_query"
    assert preflight["old_roles_collapsed"] is True
    assert preflight["confirmation_claimed"] is False
    assert preflight["promotion_authorized"] is False


def test_preflight_cli_does_not_write_or_load_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--preflight-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_loaded"] is False
    assert payload["panel_row_count"] == 60


def test_demo_cli_is_compact_and_hash_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--demo"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_kind"] == "model_free_fixture"
    assert payload["row_count"] == 2
    assert payload["published"] is False
    assert "retained_routes" not in payload
    assert "distance_motifs" not in payload
    assert len(payload["report_sha256"]) == 64
    assert len(payload["implementation_bundle_sha256"]) == 64


def test_exact_control_receipt_binds_parity_and_release() -> None:
    digest = "2" * 64
    receipt = AttentionEdgeExactControlReceipt(
        ordered_event_ids=(
            "lexical.ingress",
            *(f"mutation.{index}" for index in range(1, 37)),
            "terminal.rmsnorm",
            "lexical.egress",
        ),
        compiler_metadata_sha256=digest,
        source_independence_audit_sha256=digest,
        native_valid_logits_sha256=digest,
        exact_valid_logits_sha256=digest,
        valid_logit_shape=(1, 262_144),
        literal_native_exact_parity_passed=True,
        source_model_fingerprint_before=digest,
        source_model_fingerprint_after_parity=digest,
        source_execution_fingerprint_before=digest,
        source_execution_fingerprint_after_parity=digest,
        source_parameter_count=SOURCE_UNIQUE_PARAMETER_COUNT,
        exact_parameter_inventory_parity=True,
        source_independence_passed=True,
        native_source_references_released_before_atlas=True,
    )
    assert receipt.passed
    assert AttentionEdgeExactControlReceipt.from_dict(receipt.to_dict()) == receipt
    assert not replace(
        receipt,
        native_source_references_released_before_atlas=False,
        receipt_sha256="",
    ).passed


def test_config_and_route_claim_boundaries_are_frozen() -> None:
    with pytest.raises(ValueError, match="top-k"):
        AttentionEdgeAtlasConfig(top_routes_per_target=7)
    with pytest.raises(ValueError, match="frozen at 128"):
        AttentionEdgeAtlasLiveConfig(maximum_prompt_tokens=127)
    with pytest.raises(ValueError, match="local-only"):
        AttentionEdgeAtlasLiveConfig(local_files_only=False)
    with pytest.raises(ValueError, match="distance bin"):
        RetainedAttentionRoute(
            row_ordinal=0,
            layer_ordinal=0,
            query_head_ordinal=0,
            query_ordinal=3,
            key_ordinal=1,
            causal_distance=2,
            distance_bin="1",
            target_rank=1,
            probability=0.5,
            signed_response=1.0,
            intrinsic_fisher_energy=1.0,
            softmax_logit_tangent=0.25,
        )


def test_row_capture_rejects_duplicate_logical_positions() -> None:
    extraction, _ = _tiny_extraction()
    with pytest.raises(ValueError, match="logical positions"):
        AttentionEdgeAtlasRowCapture(
            row_catalog_id="bad",
            family_id="bad",
            logical_positions=(0, 0, 1),
            supervised_query_ordinal=2,
            tokenization_sha256="3" * 64,
            layer_extractions=(extraction,),
        )
