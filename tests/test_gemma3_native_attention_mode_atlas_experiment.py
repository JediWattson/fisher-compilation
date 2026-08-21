from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_native_attention_mode_atlas_experiment as worker
from fisher_graph.adapters import SequenceContext, SequenceInputOrigin
from fisher_graph.gemma3_typed_mode_response_setup import (
    build_gemma3_typed_mode_response_setup,
)
from fisher_graph.mode_similarity_atlas import (
    DirectedFamilyExampleStats,
    DirectedFamilyStats,
    ModeABI,
    ModeSpec,
    build_mode_similarity_atlas_receipt,
    parse_atlas_config_payload,
    parse_mode_similarity_atlas_receipt_payload,
    parse_within_family_bootstrap_config_payload,
)


_ROOT = Path(__file__).resolve().parents[1]


def test_import_is_inert_with_respect_to_transformers_and_model_loading() -> None:
    script = """
import sys
import fisher_graph.gemma3_native_attention_mode_atlas_experiment
assert 'transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=_ROOT, check=True)


def test_worker_binds_the_frozen_typed_setup_and_attention_catalog() -> None:
    setup = build_gemma3_typed_mode_response_setup()

    assert setup["setup_sha256"] == worker.PINNED_SETUP_SHA256
    assert (
        worker._sha256_file(_ROOT / worker.DEFAULT_SETUP_OUTPUT)
        == worker.PINNED_SETUP_FILE_SHA256
    )
    catalog = worker._attention_catalog(setup)
    assert tuple(row["mode_id"] for row in catalog) == worker.ORDERED_MODE_IDS
    assert all(row["operator_kind"] == "attention" for row in catalog)
    assert {
        row["generic_core_mode_kind"] for row in catalog
    } == {"carrier_mutation.sequence_routed.h640"}


def test_preflight_rejects_setup_digest_before_panel_or_predecessor(
    tmp_path: Path,
) -> None:
    setup = tmp_path / "setup.json"
    setup.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="setup file digest"):
        worker._preflight(
            revision=worker.PINNED_MODEL_REVISION,
            model_id=worker.PINNED_MODEL_ID,
            output=tmp_path / "result.json",
            setup_path=setup,
            panel_path=tmp_path / "missing-panel.json",
            mlp_predecessor_path=tmp_path / "missing-predecessor.json",
            device_name="cpu",
            dtype="float32",
            batch_size=worker.PINNED_BATCH_SIZE,
        )


def test_real_preflight_authenticates_all_inert_predecessors_without_model_load(
    tmp_path: Path,
) -> None:
    preflight = worker._preflight(
        revision=worker.PINNED_MODEL_REVISION,
        model_id=worker.PINNED_MODEL_ID,
        output=tmp_path / "attention.json",
        setup_path=_ROOT / worker.DEFAULT_SETUP_OUTPUT,
        panel_path=_ROOT / worker.DEFAULT_PANEL_PATH,
        mlp_predecessor_path=_ROOT / worker.DEFAULT_MLP_PREDECESSOR,
        device_name="cpu",
        dtype="float32",
        batch_size=worker.PINNED_BATCH_SIZE,
    )

    assert len(preflight.examples) == 60
    assert len(preflight.mode_catalog) == 18
    assert preflight.receipts["setup_sha256"] == worker.PINNED_SETUP_SHA256
    assert preflight.receipts["all_authenticated_before_live_dependencies"] is True


def test_repeat_sequence_context_preserves_left_padding_and_logical_positions() -> None:
    torch = pytest.importorskip("torch")
    sequence = worker._BatchSequence(
        query_valid_mask=torch.tensor([[False, True, True]]),
        key_valid_mask=torch.tensor([[False, True, True]]),
        logical_positions=torch.tensor([[0, 0, 1]], dtype=torch.long),
        key_logical_positions=torch.tensor([[0, 0, 1]], dtype=torch.long),
        cache_positions=torch.tensor([0, 1, 2], dtype=torch.long),
        attention_mask_supplied=True,
        position_ids_supplied=False,
        cache_positions_supplied=False,
        example_ids=("example",),
        family_ids=("family",),
    )
    dependencies = SimpleNamespace(
        sequence_context_type=SequenceContext,
        sequence_input_origin_type=SequenceInputOrigin,
    )

    repeated = worker._repeat_sequence_context(
        sequence,
        repeats=3,
        device=torch.device("cpu"),
        dependencies=dependencies,
    )

    assert repeated.query_valid_mask.tolist() == [
        [False, True, True],
        [False, True, True],
        [False, True, True],
    ]
    assert repeated.logical_positions.tolist() == [[0, 0, 1]] * 3
    assert repeated.cache_positions.tolist() == [0, 1, 2]
    assert repeated.input_origin.attention_mask_supplied is True


def test_probe_selection_uses_intrinsic_valid_prompt_indices_and_top_two() -> None:
    torch = pytest.importorskip("torch")
    examples, family_ids = worker._load_panel(_ROOT / worker.DEFAULT_PANEL_PATH)
    sequences = []
    raw_batches = []
    delta_batches = []
    gradient_batches = []
    fisher_batches = []
    for example in examples:
        raw = torch.zeros((1, 4, worker.PINNED_HIDDEN_WIDTH), dtype=torch.float32)
        delta = torch.ones_like(raw)
        gradient = torch.zeros_like(raw)
        fisher = torch.zeros_like(raw, dtype=torch.float64)
        # Left padded: absolute columns 1,2,3 correspond to intrinsic 0,1,2.
        # Intrinsic 1 wins, then intrinsic 2.
        fisher[0, 1, 0] = 1.0
        fisher[0, 2, 0] = 3.0
        fisher[0, 3, 0] = 2.0
        gradient[0, 1:, 0] = fisher[0, 1:, 0].sqrt().float()
        raw_batches.append(raw)
        delta_batches.append(delta)
        gradient_batches.append(gradient)
        fisher_batches.append(fisher)
        sequences.append(
            worker._BatchSequence(
                query_valid_mask=torch.tensor([[False, True, True, True]]),
                key_valid_mask=torch.tensor([[False, True, True, True]]),
                logical_positions=torch.tensor([[0, 0, 1, 2]]),
                key_logical_positions=torch.tensor([[0, 0, 1, 2]]),
                cache_positions=torch.arange(4),
                attention_mask_supplied=True,
                position_ids_supplied=False,
                cache_positions_supplied=False,
                example_ids=(example.example_id,),
                family_ids=(example.family_id,),
            )
        )
    shared = worker._ModeCapture(
        mode_id="placeholder",
        raw_batches=tuple(raw_batches),
        exact_delta_batches=tuple(delta_batches),
        gradient_batches=tuple(gradient_batches),
        fisher_batches=tuple(fisher_batches),
        raw_input_sha256="0" * 64,
        exact_output_sha256="1" * 64,
        output_gradient_sha256="2" * 64,
        output_fisher_sha256="3" * 64,
        state_fingerprint_sha256="4" * 64,
    )
    modes = {
        mode_id: worker._ModeCapture(
            mode_id=mode_id,
            raw_batches=shared.raw_batches,
            exact_delta_batches=shared.exact_delta_batches,
            gradient_batches=shared.gradient_batches,
            fisher_batches=shared.fisher_batches,
            raw_input_sha256=shared.raw_input_sha256,
            exact_output_sha256=shared.exact_output_sha256,
            output_gradient_sha256=shared.output_gradient_sha256,
            output_fisher_sha256=shared.output_fisher_sha256,
            state_fingerprint_sha256=shared.state_fingerprint_sha256,
        )
        for mode_id in worker.ORDERED_MODE_IDS
    }
    capture = worker._PanelCapture(
        modes=modes,
        sequences=tuple(sequences),
        ordered_mode_ids=worker.ORDERED_MODE_IDS,
        ordered_family_ids=tuple(family_ids),
        receipt={},
    )
    preflight = worker._Preflight(
        setup={},
        examples=examples,
        family_ids=tuple(family_ids),
        mode_catalog=(),
        setup_path=Path("setup"),
        panel_path=Path("panel"),
        mlp_predecessor_path=Path("predecessor"),
        output_path=Path("output"),
        svg_path=Path("svg"),
        receipts={},
    )

    probes = worker._build_probe_atlas(
        capture,
        preflight=preflight,
        torch=torch,
    )

    assert len(probes.selection_catalog) == worker.PINNED_PROBE_COUNT
    first = probes.selection_catalog[:2]
    assert [row["valid_prompt_token_index"] for row in first] == [1, 2]
    assert probes.absolute_token_indices[:2] == (2, 3)
    assert [row["selection_rank"] for row in first] == [1, 2]
    assert all("absolute_token_index" not in row for row in first)


def test_main_help_is_model_free() -> None:
    with pytest.raises(SystemExit) as raised:
        worker.main(["--help"])
    assert raised.value.code == 0


def test_production_shape_assemble_validate_write_load_roundtrip(
    tmp_path: Path,
) -> None:
    setup = build_gemma3_typed_mode_response_setup()
    examples, family_ids = worker._load_panel(_ROOT / worker.DEFAULT_PANEL_PATH)
    catalog = worker._attention_catalog(setup)
    abi = ModeABI(**setup["native_abis"]["attention"])
    config = parse_atlas_config_payload(setup["attention_atlas_config"])
    bootstrap = parse_within_family_bootstrap_config_payload(
        setup["attention_within_family_bootstrap_config"]
    )
    positions = {
        mode_id: float(index * index + index / 11.0)
        for index, mode_id in enumerate(worker.ORDERED_MODE_IDS)
    }
    state_by_mode = {
        mode_id: hashlib.sha256(f"state:{mode_id}".encode()).hexdigest()
        for mode_id in worker.ORDERED_MODE_IDS
    }
    modes = tuple(
        ModeSpec(
            mode_id=mode_id,
            mode_kind="carrier_mutation.sequence_routed.h640",
            state_fingerprint_sha256=state_by_mode[mode_id],
            compatibility_abi=abi,
        )
        for mode_id in worker.ORDERED_MODE_IDS
    )
    aggregate = []
    clustered = []
    diagnostics = []
    profiles = []
    for family_index, (family_id, example_ids) in enumerate(
        bootstrap.family_example_cluster_ids
    ):
        for reference, reference_position in positions.items():
            for candidate, candidate_position in positions.items():
                if reference == candidate:
                    continue
                sse = (1.0 + family_index / 20.0) * (
                    reference_position - candidate_position
                ) ** 2
                aggregate.append(
                    DirectedFamilyStats(
                        family_id, reference, candidate, abi.sha256,
                        sse, 1.0, 1.0, 1.0 + sse, 20,
                    )
                )
                diagnostics.append(
                    {
                        "family_id": family_id,
                        "reference_mode_id": reference,
                        "candidate_mode_id": candidate,
                        "row_count": 20,
                        "raw_sse": sse,
                        "raw_reference_energy": 1.0,
                        "raw_candidate_energy": 1.0 + sse,
                        "raw_dot": 1.0,
                        "raw_relative_sse": sse,
                        "raw_NRMSE": sse ** 0.5,
                        "raw_cosine": 1.0 / (1.0 + sse) ** 0.5,
                        "diagnostic_only": True,
                    }
                )
                for example_id in example_ids:
                    clustered.append(
                        DirectedFamilyExampleStats(
                            family_id, example_id, reference, candidate,
                            abi.sha256, sse / 10.0, 0.1, 0.1,
                            (1.0 + sse) / 10.0, 2,
                        )
                    )
    for reference in worker.ORDERED_MODE_IDS:
        for candidate in worker.ORDERED_MODE_IDS:
            if reference != candidate:
                profiles.append(
                    {
                        "reference_mode_id": reference,
                        "candidate_mode_id": candidate,
                        "row_count": 120,
                        "weighted_residual_energy": 1.0,
                        "numerical_rank": 1,
                        "rank_tolerance": 1e-12,
                        "energy_fraction_captured_by_rank": {
                            str(rank): 1.0
                            for rank in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 640)
                        },
                        "diagnostic_only": True,
                    }
                )
    atlas = build_mode_similarity_atlas_receipt(
        modes,
        aggregate,
        config=config,
        directed_example_stats=clustered,
        within_family_bootstrap_config=bootstrap,
    )
    capture_calls = {
        mode_id: {
            "input_layernorm": 18,
            "self_attn": 18,
            "post_attention_layernorm": 18,
            "mlp": 18,
        }
        for mode_id in worker.ORDERED_MODE_IDS
    }
    capture_mode_rows = [
        {
            "mode_id": mode_id,
            "valid_query_row_count": 814,
            "batch_count": 18,
            "raw_input_sha256": "0" * 64,
            "exact_output_sha256": "1" * 64,
            "output_gradient_sha256": "2" * 64,
            "output_fisher_sha256": "3" * 64,
            "state_fingerprint_sha256": state_by_mode[mode_id],
        }
        for mode_id in worker.ORDERED_MODE_IDS
    ]
    capture_receipt = {
        "batch_count": 18,
        "backward_count": 18,
        "valid_query_rows_per_mode": 814,
        "all_valid_query_rows_per_mode": 814,
        "full_sequence_contexts_preserved": True,
        "full_suffix_gradient_preserved_for_every_attention_delta": True,
        "coordinatewise_diagonal_output_Fisher_weights": True,
        "choice_tokenization_all_single_token_and_distinct": True,
        "correct_choice_text_appended_to_model_input": False,
        "prompt_language_model_NLL_used": False,
        "source_parameter_gradients_checked_after_every_backward": True,
        "source_parameter_gradients_present": False,
        "full_atom_output_count_per_backward": 18,
        "Fisher_objective": setup["attention_experiment_contract"][
            "Fisher_objective"
        ],
        "choice_tokenization_catalog_sha256": setup[
            "discovery_data_firewall"
        ]["choice_tokenization_catalog_sha256"],
        "last_query_catalog_sha256": setup[
            "discovery_data_firewall"
        ]["last_query_catalog_sha256"],
        "mode_rows": capture_mode_rows,
        "mode_rows_sha256": worker._manifest_sha256(capture_mode_rows),
        "source_call_audit": {
            "per_mode": capture_calls,
            "boundary_calls": {
                "full_model": 18, "embedding": 18, "lexical_head": 18,
            },
            "expected_calls_per_module": 18,
            "passed": True,
        },
        "elapsed_seconds": 1.0,
        "peak_rss_bytes_before": 100,
        "peak_rss_bytes_after": 200,
    }
    captures = {
        mode_id: worker._ModeCapture(
            mode_id, (), (), (), (), "0" * 64, "1" * 64, "2" * 64,
            "3" * 64, state_by_mode[mode_id],
        )
        for mode_id in worker.ORDERED_MODE_IDS
    }
    capture = worker._PanelCapture(
        captures, (), worker.ORDERED_MODE_IDS, tuple(family_ids), capture_receipt
    )
    probes = []
    for mode_id in worker.ORDERED_MODE_IDS:
        for example in examples:
            for rank in (1, 2):
                probes.append(
                    {
                        "origin_mode_id": mode_id,
                        "example_id": example.example_id,
                        "family_id": example.family_id,
                        "selection_rank": rank,
                        "valid_prompt_token_index": rank - 1,
                        "coordinatewise_output_Fisher_energy": float(3 - rank),
                        "canonical_probe_row_sha256": hashlib.sha256(
                            f"{mode_id}:{example.example_id}:{rank}".encode()
                        ).hexdigest(),
                    }
                )
    self_receipts = [
        {
            "mode_id": mode_id,
            "family_id": family_id,
            "row_count": 20,
            "maximum_absolute_delta_error": 0.0,
            "delta_NRMSE": 0.0,
            "maximum_absolute_delta_error_limit": 1e-6,
            "delta_NRMSE_limit": 1e-7,
            "selected_reference_output_sha256": "4" * 64,
            "selected_replay_output_sha256": "4" * 64,
            "selected_probe_catalog_sha256": worker._manifest_sha256(
                [
                    probe
                    for probe in probes
                    if probe["origin_mode_id"] == mode_id
                    and probe["family_id"] == family_id
                ]
            ),
            "same_operator_float32_numerical_parity": True,
        }
        for mode_id in worker.ORDERED_MODE_IDS
        for family_id in family_ids
    ]
    cross_calls = {
        mode_id: {
            **{
                name: 18
                for name in (
                    "input_layernorm", "self_attn", "q_proj", "k_proj",
                    "v_proj", "q_norm", "k_norm", "o_proj",
                    "post_attention_layernorm", "pre_feedforward_layernorm",
                )
            },
            "mlp": 0,
            "post_feedforward_layernorm": 0,
        }
        for mode_id in worker.ORDERED_MODE_IDS
    }
    sorted_aggregate = sorted(
        (worker._json_safe(value) for value in aggregate),
        key=lambda row: (
            row["family_id"], row["reference_mode_id"], row["candidate_mode_id"]
        ),
    )
    sorted_clustered = sorted(
        (worker._json_safe(value) for value in clustered),
        key=lambda row: (
            row["family_id"], row["example_cluster_id"],
            row["reference_mode_id"], row["candidate_mode_id"],
        ),
    )
    sorted_diagnostics = sorted(
        diagnostics,
        key=lambda row: (
            row["family_id"], row["reference_mode_id"], row["candidate_mode_id"]
        ),
    )
    sorted_profiles = sorted(
        profiles,
        key=lambda row: (row["reference_mode_id"], row["candidate_mode_id"]),
    )
    cross_receipt = {
        "directed_family_statistic_count": 1836,
        "directed_example_statistic_count": 18360,
        "directed_pair_rank_profile_count": 306,
        "probe_count": 2160,
        "probe_count_per_origin_mode_family": 20,
        "probes_per_origin_mode_and_example": 2,
        "all_valid_rows_considered_before_probe_selection": True,
        "probe_selection_rule": "fixture",
        "probe_selection_catalog": probes,
        "probe_selection_catalog_sha256": hashlib.sha256(
            worker._PROBE_CATALOG_HASH_DOMAIN + worker._canonical_json_bytes(probes)
        ).hexdigest(),
        "probe_catalog_hash_domain_hex": worker._PROBE_CATALOG_HASH_DOMAIN.hex(),
        "whole_sequence_native_replay": True,
        "shared_origin_sequences_stacked_once_per_batch_and_candidate": True,
        "self_replay_mode_count": 18,
        "self_replay_family_receipt_count": 108,
        "self_replay_receipts": self_receipts,
        "candidate_output_sha256_by_mode": {
            mode_id: "6" * 64 for mode_id in worker.ORDERED_MODE_IDS
        },
        "all_self_replay_numerical_parity_gates_passed": True,
        "directed_statistics_sha256": worker._manifest_sha256(sorted_aggregate),
        "directed_example_statistics_sha256": worker._manifest_sha256(sorted_clustered),
        "diagnostics_sha256": worker._manifest_sha256(sorted_diagnostics),
        "directed_pair_rank_profiles_sha256": worker._manifest_sha256(sorted_profiles),
        "source_call_audit": {
            "per_mode": cross_calls,
            "boundary_calls": {
                "full_model": 0, "embedding": 0, "lexical_head": 0,
            },
            "prefix_calls_per_candidate_mode": 18,
            "total_prefix_calls": 324,
            "selected_candidate_row_outputs": 38880,
            "full_sequence_valid_token_executions": 263736,
            "estimated_logical_MACs": 436467253248,
            "passed": True,
        },
        "elapsed_seconds": 2.0,
        "peak_rss_bytes_before": 200,
        "peak_rss_bytes_after": 300,
    }
    cross = worker._CrossEvaluation(
        tuple(aggregate), tuple(clustered), tuple(diagnostics), tuple(profiles),
        cross_receipt,
    )
    destination = tmp_path / "attention.json"
    preflight = worker._Preflight(
        setup, examples, tuple(family_ids), catalog,
        _ROOT / worker.DEFAULT_SETUP_OUTPUT,
        _ROOT / worker.DEFAULT_PANEL_PATH,
        _ROOT / worker.DEFAULT_MLP_PREDECESSOR,
        destination,
        worker._default_svg_path(destination),
        {
            "path": str(_ROOT / worker.DEFAULT_SETUP_OUTPUT),
            "setup_sha256": worker.PINNED_SETUP_SHA256,
            "setup_file_sha256": worker.PINNED_SETUP_FILE_SHA256,
            "panel_path": str(_ROOT / worker.DEFAULT_PANEL_PATH),
            "panel_file_sha256": worker.PINNED_PANEL_FILE_SHA256,
            "mlp_predecessor_path": str(_ROOT / worker.DEFAULT_MLP_PREDECESSOR),
            "mlp_predecessor_report_sha256": setup["predecessors"][
                "mlp_native_atlas"
            ]["report_sha256"],
            "mlp_predecessor_file_sha256": setup["predecessors"][
                "mlp_native_atlas"
            ]["report_file_sha256"],
            "mlp_predecessor_required_outcome_verified": True,
            "all_authenticated_before_live_dependencies": True,
        },
    )

    class Guard:
        def assert_unchanged(self) -> None:
            return None

    runtime = SimpleNamespace(
        source_guard=Guard(),
        adapter=SimpleNamespace(
            model_fingerprint=lambda: worker.PINNED_SOURCE_MODEL_FINGERPRINT,
            execution_fingerprint=lambda: worker.PINNED_SOURCE_EXECUTION_FINGERPRINT,
        ),
        source_model_fingerprint_before=worker.PINNED_SOURCE_MODEL_FINGERPRINT,
        source_execution_fingerprint_before=worker.PINNED_SOURCE_EXECUTION_FINGERPRINT,
        model_receipt={
            "model_id": worker.PINNED_MODEL_ID,
            "requested_revision": worker.PINNED_MODEL_REVISION,
            "resolved_commit": worker.PINNED_MODEL_REVISION,
            "parameter_count": worker.PINNED_SOURCE_SCALARS,
            "device": "cpu",
            "dtype": "float32",
            "weights_in_report": False,
        },
    )
    dependencies = SimpleNamespace(
        mode_abi_type=ModeABI,
        mode_spec_type=ModeSpec,
        parse_atlas_receipt=parse_mode_similarity_atlas_receipt_payload,
    )
    report = worker._assemble_report(
        preflight=preflight,
        runtime=runtime,
        capture=capture,
        cross_evaluation=cross,
        atlas=atlas,
        dependencies=dependencies,
        device_name="cpu",
        dtype="float32",
        batch_size=4,
    )
    loaded = worker.write_gemma3_native_attention_mode_atlas_report(
        report, destination
    )

    assert loaded == report
    assert worker.load_gemma3_native_attention_mode_atlas_report(destination) == report
    assert worker._default_svg_path(destination).is_file()

    def rehash(value: dict[str, object]) -> dict[str, object]:
        value.pop("report_sha256", None)
        value["report_sha256"] = worker._report_sha256(value)
        return value

    bool_alias = deepcopy(report)
    bool_alias["format_version"] = True
    with pytest.raises(ValueError, match="identity"):
        worker.validate_gemma3_native_attention_mode_atlas_report(
            rehash(bool_alias)
        )

    nested_atlas = deepcopy(report)
    nested_atlas["atlas"]["analysis"]["symmetric_distance"][0][1] += 0.5
    with pytest.raises(ValueError):
        worker.validate_gemma3_native_attention_mode_atlas_report(
            rehash(nested_atlas)
        )

    probe_tamper = deepcopy(report)
    probe_tamper["cross_evaluation_receipt"]["probe_selection_catalog"][0][
        "coordinatewise_output_Fisher_energy"
    ] = -1.0
    tampered_probes = probe_tamper["cross_evaluation_receipt"][
        "probe_selection_catalog"
    ]
    probe_tamper["cross_evaluation_receipt"][
        "probe_selection_catalog_sha256"
    ] = hashlib.sha256(
        worker._PROBE_CATALOG_HASH_DOMAIN
        + worker._canonical_json_bytes(tampered_probes)
    ).hexdigest()
    with pytest.raises(ValueError, match="probe catalog row"):
        worker.validate_gemma3_native_attention_mode_atlas_report(
            rehash(probe_tamper)
        )

    source_tamper = deepcopy(report)
    changed_calls = source_tamper["cross_evaluation_receipt"][
        "source_call_audit"
    ]
    changed_calls["per_mode"][worker.ORDERED_MODE_IDS[0]]["q_proj"] = 17
    source_tamper["execution_audit"][
        "cross_evaluation_source_call_audit"
    ] = deepcopy(changed_calls)
    with pytest.raises(ValueError, match="source-call"):
        worker.validate_gemma3_native_attention_mode_atlas_report(
            rehash(source_tamper)
        )

    fingerprint_tamper = deepcopy(report)
    fingerprint_tamper["execution_audit"][
        "source_model_fingerprint_before"
    ] = "9" * 64
    fingerprint_tamper["execution_audit"][
        "source_model_fingerprint_after"
    ] = "9" * 64
    with pytest.raises(ValueError, match="scientific boundary"):
        worker.validate_gemma3_native_attention_mode_atlas_report(
            rehash(fingerprint_tamper)
        )

    predecessor_tamper = deepcopy(report)
    predecessor_tamper["setup"]["mlp_predecessor_report_sha256"] = "9" * 64
    predecessor_tamper["setup"]["mlp_predecessor_file_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="setup/predecessor"):
        worker.validate_gemma3_native_attention_mode_atlas_report(
            rehash(predecessor_tamper)
        )

    duplicate_self = deepcopy(report)
    duplicated = deepcopy(
        duplicate_self["cross_evaluation_receipt"]["self_replay_receipts"][0]
    )
    duplicate_self["cross_evaluation_receipt"]["self_replay_receipts"][-1] = duplicated
    with pytest.raises(ValueError, match="coverage"):
        worker.validate_gemma3_native_attention_mode_atlas_report(
            rehash(duplicate_self)
        )

    forged_examples = deepcopy(report)
    forged_catalog = forged_examples["cross_evaluation_receipt"][
        "probe_selection_catalog"
    ]
    for probe in forged_catalog:
        probe["example_id"] = "forged-" + probe["example_id"]
    forged_examples["cross_evaluation_receipt"][
        "probe_selection_catalog_sha256"
    ] = hashlib.sha256(
        worker._PROBE_CATALOG_HASH_DOMAIN
        + worker._canonical_json_bytes(forged_catalog)
    ).hexdigest()
    with pytest.raises(ValueError, match="probe catalog row"):
        worker.validate_gemma3_native_attention_mode_atlas_report(
            rehash(forged_examples)
        )

    claim_tamper = deepcopy(report)
    claim_tamper["claims"]["compression_qualified"] = True
    with pytest.raises(ValueError, match="scientific boundary"):
        worker.validate_gemma3_native_attention_mode_atlas_report(
            rehash(claim_tamper)
        )
