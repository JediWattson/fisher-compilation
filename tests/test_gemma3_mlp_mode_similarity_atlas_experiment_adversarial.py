from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_mlp_mode_similarity_atlas_experiment as worker
from fisher_graph.gemma3_mlp_mode_similarity_atlas_setup import (
    build_gemma3_mlp_mode_similarity_atlas_setup,
)
from fisher_graph.mode_similarity_atlas import (
    parse_mode_similarity_atlas_receipt_payload,
)
from mode_atlas_fixtures import build_production_scale_synthetic_mode_atlas


_ROOT = Path(__file__).resolve().parents[1]


def _canonical_setup() -> dict[str, object]:
    return build_gemma3_mlp_mode_similarity_atlas_setup()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rehash_report(report: dict[str, object]) -> None:
    report.pop("report_sha256", None)
    report["report_sha256"] = worker._report_sha256(report)


def _rehash_atlas(atlas: dict[str, object]) -> None:
    atlas.pop("receipt_sha256", None)
    atlas["receipt_sha256"] = hashlib.sha256(
        b"fisher_graph.mode_similarity_atlas.receipt.v1\0"
        + worker._canonical_json_bytes(atlas)
    ).hexdigest()


def _rehash_atlas_input_statistics(atlas: dict[str, object]) -> None:
    atlas["input_statistics_sha256"] = hashlib.sha256(
        b"fisher_graph.mode_similarity_atlas.stats.v1\0"
        + worker._canonical_json_bytes(
            {
                "directed_family_example_stats": atlas[
                    "directed_family_example_stats"
                ],
                "directed_family_stats": atlas["directed_family_stats"],
            }
        )
    ).hexdigest()


def _synthetic_production_report(destination: Path) -> dict[str, object]:
    """Build one full-inventory, model-free report through the public core."""

    synthetic = build_production_scale_synthetic_mode_atlas()
    setup = synthetic.setup
    mode_ids = tuple(worker.ORDERED_MODE_IDS)
    family_catalog = setup["within_family_bootstrap_config"]["families"]
    family_ids = tuple(row["family_id"] for row in family_catalog)
    directed = synthetic.directed_stats
    directed_examples = synthetic.directed_example_stats
    atlas_receipt = synthetic.receipt
    atlas = atlas_receipt.canonical_payload()
    directed_payload = [row.canonical_payload() for row in directed]
    directed_example_payload = [
        row.canonical_payload() for row in directed_examples
    ]
    mode_catalog = [row.canonical_payload() for row in atlas_receipt.modes]

    diagnostics = [
        {
            "family_id": row.family_id,
            "reference_mode_id": row.reference_mode_id,
            "candidate_mode_id": row.candidate_mode_id,
            "row_count": row.row_count,
            "raw_sse": float(row.weighted_sse),
            "raw_reference_energy": float(row.weighted_reference_energy),
            "raw_candidate_energy": float(row.weighted_candidate_energy),
            "raw_dot": float(row.weighted_dot),
            "raw_relative_sse": (
                float(row.weighted_sse)
                / float(row.weighted_reference_energy)
            ),
            "raw_cosine": (
                float(row.weighted_dot)
                / math.sqrt(
                    float(row.weighted_reference_energy)
                    * float(row.weighted_candidate_energy)
                )
            ),
        }
        for row in directed
    ]

    pair_energy = {
        (reference_mode_id, candidate_mode_id): sum(
            float(row.weighted_sse)
            for row in directed
            if row.reference_mode_id == reference_mode_id
            and row.candidate_mode_id == candidate_mode_id
        )
        for reference_mode_id in mode_ids
        for candidate_mode_id in mode_ids
        if reference_mode_id != candidate_mode_id
    }
    rank_profiles: list[dict[str, object]] = []
    for reference_mode_id in mode_ids:
        for candidate_mode_id in mode_ids:
            if reference_mode_id == candidate_mode_id:
                continue
            energy = pair_energy[(reference_mode_id, candidate_mode_id)]
            rank_profiles.append(
                {
                    "reference_mode_id": reference_mode_id,
                    "candidate_mode_id": candidate_mode_id,
                    "row_count": 120,
                    "weighted_residual_rank_profile": {
                        "weighted_residual_energy": energy,
                        "numerical_rank": 1,
                        "rank_tolerance": max(energy, 1.0) * 1.0e-12,
                        "energy_fraction_captured_by_rank": {
                            str(rank): 1.0
                            for rank in worker.RESIDUAL_RANK_PROFILE_POINTS
                        },
                    },
                }
            )

    probe_catalog: list[dict[str, object]] = []
    probes_by_mode_family: dict[tuple[str, str], list[dict[str, object]]] = {}
    for mode_id in mode_ids:
        for family in family_catalog:
            family_id = family["family_id"]
            selected: list[dict[str, object]] = []
            for example_id in family["example_cluster_ids"]:
                for selection_rank, energy in ((1, 2.0), (2, 1.0)):
                    row = {
                        "origin_mode_id": mode_id,
                        "example_id": example_id,
                        "family_id": family_id,
                        "selection_rank": selection_rank,
                        "token_index": selection_rank - 1,
                        "coordinatewise_output_Fisher_energy": energy,
                        "canonical_probe_row_sha256": _sha(
                            f"probe:{mode_id}:{example_id}:{selection_rank}"
                        ),
                    }
                    probe_catalog.append(row)
                    selected.append(row)
            probes_by_mode_family[(mode_id, family_id)] = selected
    probe_domain = bytes.fromhex(
        setup["probe_sampling"]["probe_catalog_hash_domain_hex"]
    )
    probe_catalog_sha256 = hashlib.sha256(
        probe_domain + worker._canonical_json_bytes(probe_catalog)
    ).hexdigest()

    self_replays = [
        {
            "mode_id": mode_id,
            "family_id": family_id,
            "row_count": 20,
            "maximum_absolute_delta_error": 0.0,
            "delta_NRMSE": 0.0,
            "maximum_absolute_delta_error_limit": 1.0e-6,
            "delta_NRMSE_limit": 1.0e-7,
            "selected_reference_output_sha256": _sha(
                f"self-reference:{mode_id}:{family_id}"
            ),
            "selected_replay_output_sha256": _sha(
                f"self-reference:{mode_id}:{family_id}"
            ),
            "selected_probe_catalog_sha256": worker._manifest_sha256(
                probes_by_mode_family[(mode_id, family_id)]
            ),
            "same_operator_float32_numerical_parity": True,
        }
        for mode_id in mode_ids
        for family_id in family_ids
    ]
    capture_per_mode = {
        mode_id: {
            "attention": 18,
            "mlp": 18,
            "post_feedforward_layernorm": 18,
            "pre_feedforward_layernorm": 18,
        }
        for mode_id in mode_ids
    }
    cross_per_mode = {
        mode_id: {
            "attention": 0,
            "pre_feedforward_layernorm": 9,
            "mlp": 9,
            "gate_proj": 9,
            "up_proj": 9,
            "down_proj": 9,
            "post_feedforward_layernorm": 9,
        }
        for mode_id in mode_ids
    }
    mode_rows = [
        {
            "mode_id": mode_id,
            "row_count": 120,
            "family_row_counts": {family_id: 20 for family_id in family_ids},
            "raw_input_sha256": _sha(f"raw:{mode_id}"),
            "exact_output_sha256": _sha(f"exact:{mode_id}"),
            "output_gradient_sha256": _sha(f"gradient:{mode_id}"),
            "output_fisher_sha256": _sha(f"fisher:{mode_id}"),
        }
        for mode_id in mode_ids
    ]
    capture_source_calls = {
        "per_mode": capture_per_mode,
        "boundary_calls": {
            "embedding": 18,
            "full_model": 18,
            "lexical_head": 18,
        },
        "expected_calls_per_module": 18,
        "passed": True,
    }
    cross_source_calls = {
        "per_mode": cross_per_mode,
        "boundary_calls": {
            "embedding": 0,
            "full_model": 0,
            "lexical_head": 0,
        },
        "expected_chunks_per_candidate_mode": 9,
        "full_atom_cross_executions": 38_880,
        "estimated_projection_MACs": (
            worker.PINNED_PROBE_COUNT
            * worker.PINNED_MODE_COUNT
            * 3
            * worker.PINNED_HIDDEN_WIDTH
            * worker.PINNED_INTERMEDIATE_WIDTH
        ),
        "passed": True,
    }
    capture = {
        "batch_count": 18,
        "backward_count": 18,
        "full_atom_output_count_per_backward": 18,
        "simultaneous_retain_grad_no_graph_detach": True,
        "embedding_output_enabled_as_leaf": True,
        "full_suffix_gradient_preserved_for_every_mode_output": True,
        "coordinatewise_diagonal_output_Fisher_weights": True,
        "Fisher_objective": (
            "summed_gold_cross_entropy_over_restricted_four_choice_"
            "softmax_at_last_valid_prompt_query"
        ),
        "prompt_language_model_NLL_used": False,
        "correct_choice_text_appended_to_model_input": False,
        "choice_tokenization_all_single_token_and_distinct": True,
        "choice_tokenization_catalog_sha256": _sha("choice-catalog"),
        "last_query_catalog_sha256": _sha("last-query-catalog"),
        "source_parameter_gradients_present": False,
        "source_parameter_gradients_checked_after_every_backward": True,
        "row_manifest_sha256": worker._manifest_sha256(mode_rows),
        "mode_rows": mode_rows,
        "source_call_audit": capture_source_calls,
        "elapsed_seconds": 1.0,
        "peak_rss_bytes_before": 10,
        "peak_rss_bytes_after": 20,
    }
    directed_payload.sort(
        key=lambda row: (
            row["family_id"],
            row["reference_mode_id"],
            row["candidate_mode_id"],
        )
    )
    directed_example_payload.sort(
        key=lambda row: (
            row["family_id"],
            row["example_cluster_id"],
            row["reference_mode_id"],
            row["candidate_mode_id"],
        )
    )
    diagnostics.sort(
        key=lambda row: (
            row["family_id"],
            row["reference_mode_id"],
            row["candidate_mode_id"],
        )
    )
    rank_profiles.sort(
        key=lambda row: (row["reference_mode_id"], row["candidate_mode_id"])
    )
    cross = {
        "directed_family_statistic_count": 1_836,
        "directed_example_statistic_count": 18_360,
        "directed_pair_rank_profile_count": 306,
        "ordered_distinct_mode_pair_count": 306,
        "family_count": 6,
        "all_valid_rows_considered_before_probe_selection": True,
        "probes_per_origin_mode_and_example": 2,
        "probe_count": 2_160,
        "probe_count_per_origin_mode_family": 20,
        "probe_selection_rule": (
            "top_coordinatewise_output_Fisher_energy_then_token_index_"
            "then_canonical_row_sha256"
        ),
        "probe_selection_catalog": probe_catalog,
        "probe_selection_catalog_sha256": probe_catalog_sha256,
        "probe_catalog_hash_domain_hex": probe_domain.hex(),
        "shared_concatenated_probe_atlas_evaluated_once_per_candidate_mode": True,
        "cross_evaluation_chunk_size": 256,
        "self_replay_mode_count": 18,
        "self_replay_family_receipt_count": 108,
        "self_replay_receipts": self_replays,
        "candidate_output_sha256_by_mode": {
            mode_id: _sha(f"candidate-output:{mode_id}")
            for mode_id in mode_ids
        },
        "all_self_replay_numerical_parity_gates_passed": True,
        "every_mode_evaluated_on_every_mode_input": True,
        "both_pair_directions_present": True,
        "full_atom_includes_pre_norm_mlp_and_post_norm": True,
        "coordinatewise_output_Fisher_used_in_primary_statistics": True,
        "layer_ordinal_or_distance_used_as_numerical_feature": False,
        "directed_statistics_sha256": worker._manifest_sha256(directed_payload),
        "directed_example_statistics_sha256": worker._manifest_sha256(
            directed_example_payload
        ),
        "diagnostics_sha256": worker._manifest_sha256(diagnostics),
        "directed_pair_rank_profiles_sha256": worker._manifest_sha256(
            rank_profiles
        ),
        "source_call_audit": cross_source_calls,
        "elapsed_seconds": 2.0,
        "peak_rss_bytes_before": 20,
        "peak_rss_bytes_after": 30,
    }
    predecessor = setup["predecessor_chain"]
    report: dict[str, object] = {
        "schema": worker.GEMMA3_MLP_MODE_SIMILARITY_ATLAS_EXPERIMENT_SCHEMA,
        "format_version": 1,
        "experiment_id": "gemma3-mlp-mode-similarity-atlas-v1",
        "diagnostic_outcome": "mode_similarity_discovery_only",
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "setup": {
            "path": str(worker.DEFAULT_SETUP_OUTPUT),
            "setup_file_sha256": worker.PINNED_SETUP_FILE_SHA256,
            "setup_sha256": worker.PINNED_SETUP_SHA256,
            "panel_file_sha256": worker.PINNED_PANEL_FILE_SHA256,
            "predecessor_setup_path": predecessor["setup"]["path"],
            "predecessor_setup_file_sha256": predecessor["setup"][
                "file_sha256"
            ],
            "predecessor_setup_sha256": predecessor["setup"]["setup_sha256"],
            "predecessor_report_path": predecessor["report"]["path"],
            "predecessor_report_file_sha256": predecessor["report"][
                "file_sha256"
            ],
            "predecessor_report_sha256": predecessor["report"][
                "report_sha256"
            ],
            "predecessor_required_outcome_verified": True,
            "all_authenticated_before_live_dependencies": True,
            "all_60_panel_examples_opened_for_discovery": True,
        },
        "model": {
            "model_id": worker.PINNED_MODEL_ID,
            "requested_revision": worker.PINNED_MODEL_REVISION,
            "resolved_commit": worker.PINNED_MODEL_REVISION,
            "parameter_count": worker.PINNED_SOURCE_SCALARS,
            "device": "cpu",
            "dtype": "float32",
            "weights_in_report": False,
        },
        "data": {
            "panel_path": str(worker.DEFAULT_PANEL_PATH),
            "panel_file_sha256": worker.PINNED_PANEL_FILE_SHA256,
            "example_count": 60,
            "ordered_family_ids": list(family_ids),
            "family_count": 6,
            "all_panel_examples_spent_for_discovery": True,
            "future_compression_or_merge_evaluation_requires_prompt_disjoint_panel": True,
        },
        "configuration": {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 4,
            "ordered_mode_ids": list(mode_ids),
            "mode_count": 18,
            "primary_distance": (
                "equal_family_mean_coordinatewise_output_Fisher_relative_SSE"
            ),
            "symmetric_distance": "mean_of_both_directed_distances",
            "raw_cosine_and_weighted_residual_rank_profile_diagnostic_only": True,
        },
        "mode_catalog": mode_catalog,
        "capture": capture,
        "directed_family_statistics": directed_payload,
        "cross_evaluation_diagnostics": diagnostics,
        "directed_pair_residual_rank_profiles": rank_profiles,
        "raw_diagnostic_geometry": worker._raw_diagnostic_geometry(
            diagnostics,
            mode_axis=atlas_receipt.analysis.mode_axis,
            family_ids=family_ids,
        ),
        "cross_evaluation_receipt": cross,
        "atlas": atlas,
        "stable_neighbor_hypotheses": worker._stable_neighbor_hypotheses(
            atlas, setup=setup
        ),
        "visualization": {
            "path": str(worker._default_svg_path(destination)),
            "format": "image/svg+xml",
            "deterministic_stdlib_renderer": True,
        },
        "execution_audit": {
            "preflight_before_tensor_or_model_dependencies": True,
            "source_model_fingerprint_before": (
                worker.PINNED_SOURCE_MODEL_FINGERPRINT
            ),
            "source_model_fingerprint_after": (
                worker.PINNED_SOURCE_MODEL_FINGERPRINT
            ),
            "source_execution_fingerprint_before": (
                worker.PINNED_SOURCE_EXECUTION_FINGERPRINT
            ),
            "source_execution_fingerprint_after": (
                worker.PINNED_SOURCE_EXECUTION_FINGERPRINT
            ),
            "source_model_and_execution_unchanged": True,
            "source_parameters_frozen": True,
            "source_parameter_gradients_present": False,
            "compression_fit_or_merge_performed": False,
            "learned_edge_count": 0,
            "layer_ordinal_or_distance_used_as_numerical_feature": False,
            "full_atom_self_replay_numerical_parity_passed_for_all_108_mode_family_receipts": True,
            "capture_elapsed_seconds": 1.0,
            "cross_evaluation_elapsed_seconds": 2.0,
            "process_peak_rss_bytes": 30,
            "capture_source_call_audit": capture_source_calls,
            "cross_evaluation_source_call_audit": cross_source_calls,
        },
        "claims": {
            "mode_order_discovered": True,
            "mode_family_or_cluster_proven": False,
            "compression_qualified": False,
            "merge_qualified": False,
            "attention_modes_measured": False,
            "lexical_or_embedding_modes_measured": False,
            "MLP_is_source_implementation_provenance_only": True,
            "comparison_ABI": "carrier_mutation.token_local.h640",
        },
    }
    svg_bytes = worker._render_heatmap_svg(report).encode("utf-8")
    report["visualization"]["file_bytes"] = len(svg_bytes)
    report["visualization"]["file_sha256"] = hashlib.sha256(svg_bytes).hexdigest()
    _rehash_report(report)
    return report


@pytest.fixture(scope="session")
def synthetic_valid_report(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path]:
    destination = tmp_path_factory.mktemp("mode-atlas-adversarial") / "atlas.json"
    report = _synthetic_production_report(destination)
    worker.validate_gemma3_mlp_mode_similarity_atlas_report(report)
    return report, destination


def test_synthetic_production_report_validates(
    synthetic_valid_report: tuple[dict[str, object], Path],
) -> None:
    report, _ = synthetic_valid_report
    assert worker.validate_gemma3_mlp_mode_similarity_atlas_report(report) == report


def test_public_run_writes_and_strictly_reloads_model_free_report(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_valid_report: tuple[dict[str, object], Path],
) -> None:
    report, destination = synthetic_valid_report
    phase_token = object()
    monkeypatch.setattr(worker, "_preflight", lambda **kwargs: phase_token)
    monkeypatch.setattr(worker, "_load_live_dependencies", lambda: phase_token)
    monkeypatch.setattr(worker, "_prepare_runtime", lambda **kwargs: phase_token)
    monkeypatch.setattr(worker, "_capture_panel", lambda **kwargs: phase_token)
    monkeypatch.setattr(
        worker, "_cross_evaluate_modes", lambda **kwargs: phase_token
    )
    monkeypatch.setattr(worker, "_build_atlas", lambda **kwargs: phase_token)
    monkeypatch.setattr(worker, "_assemble_report", lambda **kwargs: report)

    result = worker.run_gemma3_mlp_mode_similarity_atlas_experiment(
        output=destination
    )

    assert result == report
    assert worker.load_gemma3_mlp_mode_similarity_atlas_report(destination) == report


def test_real_assemble_validate_write_load_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_valid_report: tuple[dict[str, object], Path],
) -> None:
    template, destination = synthetic_valid_report
    assembled_destination = destination.with_name("assembled-atlas.json")
    atlas_receipt = parse_mode_similarity_atlas_receipt_payload(template["atlas"])
    setup = _canonical_setup()

    class SourceGuard:
        def assert_unchanged(self) -> None:
            return None

    class Adapter:
        def model_fingerprint(self) -> str:
            return worker.PINNED_SOURCE_MODEL_FINGERPRINT

        def execution_fingerprint(self) -> str:
            return worker.PINNED_SOURCE_EXECUTION_FINGERPRINT

    setup_receipt = template["setup"]
    capture_receipt = template["capture"]
    cross_receipt = template["cross_evaluation_receipt"]
    assert isinstance(setup_receipt, dict)
    assert isinstance(capture_receipt, dict)
    assert isinstance(cross_receipt, dict)
    preflight = worker._Preflight(
        setup=setup,
        examples=(),
        family_ids=tuple(worker.ORDERED_FAMILY_IDS),
        setup_path=Path(setup_receipt["path"]),
        panel_path=Path(template["data"]["panel_path"]),
        output_path=assembled_destination,
        svg_path=worker._default_svg_path(assembled_destination),
        receipts={
            key: value for key, value in setup_receipt.items() if key != "path"
        },
    )
    runtime = worker._Runtime(
        tokenizer=object(),
        model=object(),
        adapter=Adapter(),
        device="cpu",
        source_guard=SourceGuard(),
        source_model_fingerprint_before=worker.PINNED_SOURCE_MODEL_FINGERPRINT,
        source_execution_fingerprint_before=(
            worker.PINNED_SOURCE_EXECUTION_FINGERPRINT
        ),
        model_receipt=template["model"],
    )
    capture = worker._PanelCapture(
        modes={},
        ordered_mode_ids=tuple(worker.ORDERED_MODE_IDS),
        ordered_family_ids=tuple(worker.ORDERED_FAMILY_IDS),
        row_manifest_sha256=capture_receipt["row_manifest_sha256"],
        receipt=capture_receipt,
    )
    cross_evaluation = worker._CrossEvaluation(
        directed_stats=atlas_receipt.directed_stats,
        directed_example_stats=atlas_receipt.directed_example_stats,
        diagnostics=tuple(template["cross_evaluation_diagnostics"]),
        directed_pair_rank_profiles=tuple(
            template["directed_pair_residual_rank_profiles"]
        ),
        receipt=cross_receipt,
    )
    monkeypatch.setattr(
        worker, "_build_modes", lambda runtime, dependencies, setup: atlas_receipt.modes
    )

    assembled = worker._assemble_report(
        preflight=preflight,
        runtime=runtime,
        capture=capture,
        cross_evaluation=cross_evaluation,
        atlas=atlas_receipt,
        dependencies=SimpleNamespace(),
        device_name="cpu",
        dtype="float32",
        batch_size=worker.PINNED_BATCH_SIZE,
    )
    validated = worker.validate_gemma3_mlp_mode_similarity_atlas_report(assembled)
    written = worker.write_gemma3_mlp_mode_similarity_atlas_report(
        validated, assembled_destination
    )

    assert written == validated
    assert (
        worker.load_gemma3_mlp_mode_similarity_atlas_report(
            assembled_destination
        )
        == validated
    )


def test_atomic_publish_rolls_back_partial_pair_and_allows_clean_retry(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_valid_report: tuple[dict[str, object], Path],
) -> None:
    template, destination = synthetic_valid_report
    retry_destination = destination.with_name("rollback-atlas.json")
    retry_svg = worker._default_svg_path(retry_destination)
    report = deepcopy(template)
    report["visualization"]["path"] = str(retry_svg)
    _rehash_report(report)
    real_link = worker.os.link

    def fail_second_publish(source: object, target: object) -> None:
        if Path(target) == retry_destination:
            raise OSError("injected JSON publication failure")
        real_link(source, target)

    monkeypatch.setattr(worker.os, "link", fail_second_publish)
    with pytest.raises(OSError, match="injected JSON publication failure"):
        worker.write_gemma3_mlp_mode_similarity_atlas_report(
            report, retry_destination
        )

    assert not retry_destination.exists()
    assert not retry_svg.exists()
    assert not list(
        retry_destination.parent.glob(f".{retry_destination.name}.*.tmp")
    )

    monkeypatch.setattr(worker.os, "link", real_link)
    written = worker.write_gemma3_mlp_mode_similarity_atlas_report(
        report, retry_destination
    )
    assert written == report
    assert worker.load_gemma3_mlp_mode_similarity_atlas_report(
        retry_destination
    ) == report


@pytest.mark.parametrize(
    "attack",
    (
        "atlas_derived_geometry",
        "atlas_config",
        "atlas_bootstrap",
        "atlas_sufficient_statistics",
        "stable_neighbor_claim",
        "visualization_receipt",
        "source_call_audit",
        "resource_cross_binding",
        "compression_claim",
    ),
)
def test_rehashed_semantic_tamper_is_rejected(
    attack: str,
    synthetic_valid_report: tuple[dict[str, object], Path],
) -> None:
    original, _ = synthetic_valid_report
    report = deepcopy(original)
    atlas = report["atlas"]
    cross = report["cross_evaluation_receipt"]
    assert isinstance(atlas, dict)
    assert isinstance(cross, dict)

    if attack == "atlas_derived_geometry":
        matrix = atlas["analysis"]["symmetric_distance"]
        matrix[0][1] += 0.001
        matrix[1][0] += 0.001
        _rehash_atlas(atlas)
    elif attack == "atlas_config":
        atlas["atlas_config"]["energy_epsilon"] = 2.0e-12
        _rehash_atlas(atlas)
    elif attack == "atlas_bootstrap":
        atlas["within_family_bootstrap_config"]["seed"] = 2312
        _rehash_atlas(atlas)
    elif attack == "atlas_sufficient_statistics":
        target = atlas["directed_family_stats"][0]
        family_id = target["family_id"]
        reference_mode_id = target["reference_mode_id"]
        candidate_mode_id = target["candidate_mode_id"]
        target["weighted_sse"] += 0.001
        target["weighted_dot"] -= 0.0005
        for row in atlas["directed_family_example_stats"]:
            if (
                row["family_id"] == family_id
                and row["reference_mode_id"] == reference_mode_id
                and row["candidate_mode_id"] == candidate_mode_id
            ):
                row["weighted_sse"] += 0.0001
                row["weighted_dot"] -= 0.00005
        for row in report["directed_family_statistics"]:
            if (
                row["family_id"] == family_id
                and row["reference_mode_id"] == reference_mode_id
                and row["candidate_mode_id"] == candidate_mode_id
            ):
                row.update(target)
                break
        _rehash_atlas_input_statistics(atlas)
        _rehash_atlas(atlas)
        cross["directed_statistics_sha256"] = worker._manifest_sha256(
            sorted(
                report["directed_family_statistics"],
                key=lambda row: (
                    row["family_id"],
                    row["reference_mode_id"],
                    row["candidate_mode_id"],
                ),
            )
        )
        cross["directed_example_statistics_sha256"] = worker._manifest_sha256(
            sorted(
                atlas["directed_family_example_stats"],
                key=lambda row: (
                    row["family_id"],
                    row["example_cluster_id"],
                    row["reference_mode_id"],
                    row["candidate_mode_id"],
                ),
            )
        )
    elif attack == "stable_neighbor_claim":
        assert report["stable_neighbor_hypotheses"]
        report["stable_neighbor_hypotheses"][0][
            "authorizes_family_merge_or_compression"
        ] = True
    elif attack == "visualization_receipt":
        report["visualization"]["file_sha256"] = "0" * 64
    elif attack == "source_call_audit":
        mode_id = worker.ORDERED_MODE_IDS[0]
        cross["source_call_audit"]["per_mode"][mode_id]["attention"] = 1
    elif attack == "resource_cross_binding":
        report["execution_audit"]["process_peak_rss_bytes"] += 1
    elif attack == "compression_claim":
        report["claims"]["merge_qualified"] = True
    else:  # pragma: no cover - the frozen parameter catalog is exhaustive.
        raise AssertionError(f"unknown attack {attack}")

    _rehash_report(report)
    with pytest.raises(ValueError):
        worker.validate_gemma3_mlp_mode_similarity_atlas_report(report)


@pytest.mark.parametrize(
    ("path", "substitution"),
    (
        (("format_version",), True),
        (("data", "example_count"), True),
        (("cross_evaluation_receipt", "family_count"), True),
        (("capture", "batch_count"), True),
        (("execution_audit", "learned_edge_count"), False),
        (
            (
                "cross_evaluation_receipt",
                "source_call_audit",
                "boundary_calls",
                "embedding",
            ),
            False,
        ),
    ),
)
def test_rehashed_boolean_for_numeric_field_is_rejected(
    path: tuple[str, ...],
    substitution: bool,
    synthetic_valid_report: tuple[dict[str, object], Path],
) -> None:
    original, _ = synthetic_valid_report
    report = deepcopy(original)
    cursor: dict[str, object] = report
    for component in path[:-1]:
        nested = cursor[component]
        assert isinstance(nested, dict)
        cursor = nested
    cursor[path[-1]] = substitution
    _rehash_report(report)

    with pytest.raises(ValueError):
        worker.validate_gemma3_mlp_mode_similarity_atlas_report(report)


def test_public_run_finishes_preflight_before_loading_live_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    tokens = {
        "preflight": object(),
        "dependencies": object(),
        "runtime": object(),
        "capture": object(),
        "cross": object(),
        "atlas": object(),
        "report": {"report_sha256": "f" * 64},
    }

    def phase(name: str, result: object):
        def run(**kwargs: object) -> object:
            events.append(name)
            if name == "live_dependencies":
                assert events == ["preflight", "live_dependencies"]
            return result

        return run

    monkeypatch.setattr(worker, "_preflight", phase("preflight", tokens["preflight"]))
    monkeypatch.setattr(
        worker,
        "_load_live_dependencies",
        lambda: (events.append("live_dependencies") or tokens["dependencies"]),
    )
    monkeypatch.setattr(worker, "_prepare_runtime", phase("runtime", tokens["runtime"]))
    monkeypatch.setattr(worker, "_capture_panel", phase("capture", tokens["capture"]))
    monkeypatch.setattr(worker, "_cross_evaluate_modes", phase("cross", tokens["cross"]))
    monkeypatch.setattr(worker, "_build_atlas", phase("atlas", tokens["atlas"]))
    monkeypatch.setattr(worker, "_assemble_report", phase("report", tokens["report"]))

    def write(report: object, destination: object) -> object:
        events.append("write")
        assert report is tokens["report"]
        assert Path(destination) == tmp_path / "atlas.json"
        return report

    monkeypatch.setattr(
        worker,
        "write_gemma3_mlp_mode_similarity_atlas_report",
        write,
    )

    result = worker.run_gemma3_mlp_mode_similarity_atlas_experiment(
        output=tmp_path / "atlas.json"
    )

    assert result is tokens["report"]
    assert events == [
        "preflight",
        "live_dependencies",
        "runtime",
        "capture",
        "cross",
        "atlas",
        "report",
        "write",
    ]


def test_preflight_rejects_rehashed_setup_panel_and_mode_inventory_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical_setup_path = _ROOT / worker.DEFAULT_SETUP_OUTPUT
    canonical_panel_path = _ROOT / worker.DEFAULT_PANEL_PATH

    setup_tamper = _canonical_setup()
    setup_tamper["setup_sha256"] = "0" * 64
    setup_path = tmp_path / "setup.json"
    setup_path.write_text(
        json.dumps(setup_tamper, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="setup file digest"):
        worker._preflight(
            revision=worker.PINNED_MODEL_REVISION,
            model_id=worker.PINNED_MODEL_ID,
            output=tmp_path / "setup-tamper-report.json",
            setup_path=setup_path,
            panel_path=canonical_panel_path,
            device_name="cpu",
            dtype="float32",
            batch_size=worker.PINNED_BATCH_SIZE,
        )

    panel = json.loads(canonical_panel_path.read_text(encoding="utf-8"))
    panel["examples"][0]["prompt"] += " tampered"
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(
        json.dumps(panel, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "_strict_predecessor_chain", lambda setup: {})
    with pytest.raises(RuntimeError, match="panel file digest"):
        worker._preflight(
            revision=worker.PINNED_MODEL_REVISION,
            model_id=worker.PINNED_MODEL_ID,
            output=tmp_path / "panel-tamper-report.json",
            setup_path=canonical_setup_path,
            panel_path=panel_path,
            device_name="cpu",
            dtype="float32",
            batch_size=worker.PINNED_BATCH_SIZE,
        )

    setup = _canonical_setup()
    examples, family_ids = worker._load_panel(canonical_panel_path)
    setup["scope_freeze"]["directed_off_diagonal_count"] = 304
    with pytest.raises(ValueError, match="mode inventory"):
        worker._validate_setup_bindings(
            setup,
            examples=examples,
            family_ids=family_ids,
        )


def test_predecessor_file_digest_tamper_fails_before_report_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = deepcopy(_canonical_setup())
    setup["predecessor_chain"]["setup"]["file_sha256"] = "0" * 64
    parser_called = False

    def forbidden_parser(*args: object, **kwargs: object) -> object:
        nonlocal parser_called
        parser_called = True
        raise AssertionError("predecessor parser must not open a digest mismatch")

    monkeypatch.setattr(worker, "_sha256_file", lambda path: "1" * 64)
    monkeypatch.setattr(
        "fisher_graph.gemma3_mlp_shared_family_residual_adapter_setup."
        "load_gemma3_mlp_shared_family_residual_adapter_setup",
        forbidden_parser,
    )

    with pytest.raises(RuntimeError, match="predecessor file digest"):
        worker._strict_predecessor_chain(setup)
    assert parser_called is False
