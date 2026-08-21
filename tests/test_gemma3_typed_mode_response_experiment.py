from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_typed_mode_response_experiment as worker
from fisher_graph.gemma3_mlp_mode_similarity_atlas_experiment import (
    load_gemma3_mlp_mode_similarity_atlas_report,
)
from fisher_graph.gemma3_typed_mode_response_setup import (
    LEXICAL_MODE_IDS,
    ORDERED_FAMILY_IDS,
    RESPONSE_MODE_IDS,
    build_gemma3_typed_mode_response_setup,
)
from fisher_graph.mode_similarity_atlas import (
    DirectedFamilyExampleStats,
    DirectedFamilyStats,
    parse_atlas_config_payload,
    parse_mode_similarity_atlas_receipt_payload,
)


_ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _example_catalog() -> tuple[tuple[str, str], ...]:
    prefixes = {
        "integer_addition": "addition",
        "integer_subtraction": "subtraction",
        "integer_comparison": "comparison",
        "lexical_antonym": "antonym",
        "category_completion": "category",
        "world_capitals": "capital",
    }
    return tuple(
        (f"{prefixes[family_id]}-{ordinal:02d}", family_id)
        for family_id in ORDERED_FAMILY_IDS
        for ordinal in range(1, 11)
    )


def _synthetic_measurement() -> worker._ResponseMeasurement:
    examples = _example_catalog()
    responses: dict[
        str, tuple[tuple[float, float, float, float], ...]
    ] = {}
    for mode_index, mode_id in enumerate(RESPONSE_MODE_IDS):
        logical_index = 36 if mode_id in LEXICAL_MODE_IDS else mode_index
        responses[mode_id] = tuple(
            (
                logical_index * 0.01 + row_index * 0.001,
                -logical_index * 0.005,
                row_index * 0.0002,
                -logical_index * 0.005 - row_index * 0.0012,
            )
            for row_index in range(60)
        )
    fingerprints = {
        mode_id: _sha(
            "tied-lexical" if mode_id in LEXICAL_MODE_IDS else mode_id
        )
        for mode_id in RESPONSE_MODE_IDS
    }
    return worker._ResponseMeasurement(
        example_ids=tuple(item[0] for item in examples),
        family_ids_by_example=tuple(item[1] for item in examples),
        responses_by_mode=responses,
        weights_by_example=tuple((0.1, 0.2, 0.3, 0.4) for _ in examples),
        state_fingerprints_by_mode=fingerprints,
        terminal_norm_state_fingerprint_sha256=_sha("terminal"),
        receipt={},
    )


def _fake_core_receipt(
    report: dict[str, object] | None = None,
) -> SimpleNamespace:
    if report is None:
        report = _semantic_report()
    setup = build_gemma3_typed_mode_response_setup()
    measurement = report["measurement"]
    state_hashes = measurement[
        "mode_state_fingerprint_sha256_by_mode"
    ]
    native_abis = setup["native_abis"]
    modes = []
    for raw_mode in setup["typed_mode_catalog"]:
        mode_id = raw_mode["mode_id"]
        kind_name = raw_mode["operator_kind"]
        modes.append(
            worker.TypedModeSpec(
                mode_id=mode_id,
                operator_kind=worker._OPERATOR_KIND[kind_name],
                native_mode_kind=raw_mode["generic_core_mode_kind"],
                state_fingerprint_sha256=state_hashes[mode_id],
                native_abi=worker._mode_abi(
                    native_abis[kind_name], label=f"{kind_name} ABI"
                ),
            )
        )
    terminal = setup["terminal_norm_baseline"]
    modes.append(
        worker.TypedModeSpec(
            mode_id=worker.TERMINAL_NORM_MODE_ID,
            operator_kind=worker.ModeOperatorKind.TERMINAL_NORM,
            native_mode_kind="terminal_boundary.rmsnorm.h640",
            state_fingerprint_sha256=measurement[
                "terminal_norm_state_fingerprint_sha256"
            ],
            native_abi=worker._mode_abi(
                terminal["native_abi"], label="terminal ABI"
            ),
        )
    )
    modes = sorted(
        modes,
        key=lambda mode: (
            mode.state_fingerprint_sha256,
            mode.operator_kind.value,
            mode.mode_id,
        ),
    )
    lexical_class = SimpleNamespace(member_mode_ids=tuple(LEXICAL_MODE_IDS))
    singleton_classes = [
        SimpleNamespace(member_mode_ids=(mode_id,))
        for mode_id in RESPONSE_MODE_IDS
        if mode_id not in LEXICAL_MODE_IDS
    ]
    return SimpleNamespace(
        inventory=SimpleNamespace(attention_mode_count=18, mlp_mode_count=18),
        modes=tuple(modes),
        native_partitions=[
            SimpleNamespace(
                operator_kind=worker.ModeOperatorKind.ATTENTION,
                atlas=SimpleNamespace(
                    receipt_sha256=(
                        worker._PINNED_ATTENTION_NATIVE_RECEIPT_SHA256
                    )
                ),
            ),
            SimpleNamespace(
                operator_kind=worker.ModeOperatorKind.MLP,
                atlas=SimpleNamespace(
                    receipt_sha256=worker._PINNED_MLP_NATIVE_RECEIPT_SHA256
                ),
            ),
            SimpleNamespace(
                operator_kind=worker.ModeOperatorKind.LEXICAL_INGRESS,
                atlas=None,
            ),
            SimpleNamespace(
                operator_kind=worker.ModeOperatorKind.LEXICAL_EGRESS,
                atlas=None,
            ),
            SimpleNamespace(
                operator_kind=worker.ModeOperatorKind.TERMINAL_NORM,
                atlas=None,
            ),
        ],
        downstream_response=SimpleNamespace(
            mode_ids=tuple(
                mode.mode_id
                for mode in modes
                if mode.operator_kind
                is not worker.ModeOperatorKind.TERMINAL_NORM
            ),
            response_abi=worker._response_abi(setup["common_response_abi"]),
            measurement_input_sha256=measurement[
                "measurement_input_sha256"
            ],
            family_pair_stats=[None] * 4_218,
            example_pair_stats=[None] * 42_180,
            equivalence_classes=[lexical_class, *singleton_classes],
        ),
    )


def _semantic_report() -> dict[str, object]:
    response_hashes = {
        mode_id: _sha(
            "lexical-response" if mode_id in LEXICAL_MODE_IDS else mode_id
        )
        for mode_id in RESPONSE_MODE_IDS
    }
    mlp_native_sha = worker._PINNED_MLP_NATIVE_RECEIPT_SHA256
    attention_native_sha = worker._PINNED_ATTENTION_NATIVE_RECEIPT_SHA256
    attention_file_sha = worker._PINNED_ATTENTION_REPORT_FILE_SHA256
    attention_report_sha = worker._PINNED_ATTENTION_REPORT_SHA256
    state_hashes = {
        mode_id: _sha(
            "tied-state" if mode_id in LEXICAL_MODE_IDS else f"state:{mode_id}"
        )
        for mode_id in RESPONSE_MODE_IDS
    }
    terminal_state_hash = _sha("terminal-state")
    ordered_examples = [item[0] for item in _example_catalog()]
    parity_example_batches = [
        ordered_examples[
            family_start + batch_start : min(
                family_start + batch_start + 4, family_start + 10
            )
        ]
        for family_start in range(0, 60, 10)
        for batch_start in (0, 4, 8)
    ]
    baseline_parity = [
        {
            "batch_index": index,
            "example_ids": parity_example_batches[index],
            "token_id_logits4_sha256": _sha(f"baseline-{index}"),
            "inputs_embeds_logits4_sha256": _sha(f"baseline-{index}"),
            "byte_identical": True,
        }
        for index in range(18)
    ]
    measurement_input = worker._measurement_input_binding(
        setup=build_gemma3_typed_mode_response_setup(),
        runtime_weight_catalog_sha256=_sha("weights"),
        baseline_parity_receipts=baseline_parity,
        response_sha256_by_mode=response_hashes,
        mode_state_fingerprint_sha256_by_mode=state_hashes,
        terminal_norm_state_fingerprint_sha256=terminal_state_hash,
    )
    report: dict[str, object] = {
        "schema": worker.GEMMA3_TYPED_MODE_RESPONSE_EXPERIMENT_SCHEMA,
        "format_version": 1,
        "experiment_id": "gemma3-typed-mode-response-v1",
        "diagnostic_outcome": "typed_mode_response_geometry_discovery_only",
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_raw_activations_or_responses": False,
        "setup_and_predecessors": {
            "setup_path": "setup.json",
            "panel_path": "panel.json",
            "mlp_report_path": "mlp.json",
            "attention_report_path": "attention.json",
            "setup_file_sha256": worker.PINNED_SETUP_FILE_SHA256,
            "setup_sha256": worker.PINNED_SETUP_SHA256,
            "panel_file_sha256": worker.PINNED_PANEL_FILE_SHA256,
            "mlp_report_file_sha256": worker.PINNED_MLP_REPORT_FILE_SHA256,
            "mlp_report_sha256": worker.PINNED_MLP_REPORT_SHA256,
            "mlp_native_receipt_sha256": mlp_native_sha,
            "mlp_partition_binding_sha256": (
                worker._native_partition_binding_sha256(
                    operator_kind="mlp",
                    report_file_sha256=worker.PINNED_MLP_REPORT_FILE_SHA256,
                    report_sha256=worker.PINNED_MLP_REPORT_SHA256,
                    native_receipt_sha256=mlp_native_sha,
                )
            ),
            "attention_report_file_sha256": attention_file_sha,
            "attention_report_sha256": attention_report_sha,
            "attention_native_receipt_sha256": attention_native_sha,
            "attention_partition_binding_sha256": (
                worker._native_partition_binding_sha256(
                    operator_kind="attention",
                    report_file_sha256=attention_file_sha,
                    report_sha256=attention_report_sha,
                    native_receipt_sha256=attention_native_sha,
                )
            ),
            "all_authenticated_before_live_dependencies": True,
        },
        "model": {
            "model_id": worker.PINNED_MODEL_ID,
            "requested_revision": worker.PINNED_MODEL_REVISION,
            "resolved_commit": worker.PINNED_MODEL_REVISION,
            "parameter_count": worker.PINNED_SOURCE_PARAMETER_COUNT,
            "device": "cpu",
            "dtype": "float32",
            "weights_in_report": False,
        },
        "configuration": {
            "batch_size": 4,
            "ordered_mode_ids": list(RESPONSE_MODE_IDS),
            "mode_count": 38,
            "family_ids": list(ORDERED_FAMILY_IDS),
            "example_count": 60,
            "response_width": 4,
            "primary_response": "C4_times_z_scale1_minus_z_scale0",
            "coordinate_weight": (
                "squared_analytic_scale1_restricted_choice_CE_gradient"
            ),
            "response_bootstrap_sampling_unit": (
                "example_plus_centered_logits4_response_cluster"
            ),
            "response_bootstrap_config_sha256": (
                worker._PINNED_RESPONSE_BOOTSTRAP_CONFIG_SHA256
            ),
            "generic_core_bootstrap_literal_adapter": (
                "example_plus_selected_probes_cluster"
            ),
            "generic_core_bootstrap_literal_is_not_scientific_sampling_unit": True,
        },
        "measurement": {
            "batch_count": 18,
            "example_count": 60,
            "response_width": 4,
            "token_id_baseline_forward_count": 18,
            "inputs_embeds_baseline_forward_count": 18,
            "logical_intervention_forward_count": 684,
            "full_model_forward_count": 720,
            "backward_count": 0,
            "baseline_parity_receipts": baseline_parity,
            "native_occurrence_intervention_count": 38,
            "all_baseline_parity_receipts_byte_identical": True,
            "analytic_output_Fisher_weights": True,
            "tied_lexical_distinct_occurrence_interventions_executed": True,
            "tied_lexical_responses_byte_identical": True,
            "runtime_output_Fisher_weight_catalog_row_count": 240,
            "runtime_output_Fisher_weight_catalog_sha256": _sha("weights"),
            "primary_response": "C4_times_z_scale1_minus_z_scale0",
            "response_sha256_by_mode": response_hashes,
            "mode_state_fingerprint_sha256_by_mode": state_hashes,
            "terminal_norm_state_fingerprint_sha256": terminal_state_hash,
            "measurement_input_binding": measurement_input,
            "measurement_input_sha256": measurement_input[
                "measurement_input_sha256"
            ],
            "tied_lexical_shared_state_fingerprint_sha256": state_hashes[
                LEXICAL_MODE_IDS[0]
            ],
            "logical_intervention_forward_count_by_mode": {
                mode_id: 18 for mode_id in RESPONSE_MODE_IDS
            },
            "target_hook_call_count_by_mutation_mode": {
                mode_id: 18 for mode_id in RESPONSE_MODE_IDS[:36]
            },
            "lexical_egress_occurrence_hook_call_count": 18,
            "lexical_egress_zero_preserved_by_final_softcap": True,
            "elapsed_seconds": 1.0,
            "peak_rss_bytes_before": 10,
            "peak_rss_bytes_after": 20,
        },
        "measurement_input_binding": measurement_input,
        "multi_abi_atlas": {},
        "response_summary": {
            "unordered_mode_pair_count": 703,
            "family_pair_stat_count": 4_218,
            "example_pair_stat_count": 42_180,
            "exact_response_equivalence_class_count": 37,
            "tied_lexical_exact_equivalence_class_present": True,
            "native_ABI_partition_count": 5,
        },
        "execution_audit": {
            "preflight_completed_before_live_dependencies": True,
            "source_model_fingerprint_before": (
                worker.PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
            ),
            "source_model_fingerprint_after": (
                worker.PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
            ),
            "source_execution_fingerprint_before": (
                worker.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
            ),
            "source_execution_fingerprint_after": (
                worker.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
            ),
            "source_model_and_execution_unchanged": True,
            "source_parameters_frozen": True,
            "source_parameter_gradients_present": False,
            "backward_count": 0,
            "full_model_forward_count": 720,
            "fit_performed": False,
            "learned_edge_count": 0,
            "source_parameter_or_edge_changed": False,
            "process_peak_rss_bytes": 20,
        },
        "claims": {
            "descriptive_geometry_only": True,
            "native_operator_interchangeability_implied_by_response": False,
            "neighbor_hypothesis_claimed": False,
            "cluster_claimed": False,
            "generator_family_claimed": False,
            "merge_qualified": False,
            "compression_qualified": False,
            "deployment_qualified": False,
            "future_selection_requires_fresh_prompt_and_paraphrase_disjoint_data": True,
            "short_context_attention_limitation_carried_forward": True,
        },
        "visualization": {
            "path": "/tmp/typed-mode-response-atlas-v1-heatmap.svg",
            "format": "image/svg+xml",
            "matrix_shape": [38, 38],
            "deterministic_stdlib_renderer": True,
            "file_bytes": 1,
            "file_sha256": _sha("svg"),
        },
    }
    report["report_sha256"] = worker._report_sha256(report)
    return report


def _rebind_report_measurement(report: dict[str, object]) -> None:
    measurement = report["measurement"]
    binding = worker._measurement_input_binding(
        setup=build_gemma3_typed_mode_response_setup(),
        runtime_weight_catalog_sha256=measurement[
            "runtime_output_Fisher_weight_catalog_sha256"
        ],
        baseline_parity_receipts=measurement["baseline_parity_receipts"],
        response_sha256_by_mode=measurement["response_sha256_by_mode"],
        mode_state_fingerprint_sha256_by_mode=measurement[
            "mode_state_fingerprint_sha256_by_mode"
        ],
        terminal_norm_state_fingerprint_sha256=measurement[
            "terminal_norm_state_fingerprint_sha256"
        ],
    )
    measurement["measurement_input_binding"] = binding
    measurement["measurement_input_sha256"] = binding[
        "measurement_input_sha256"
    ]
    report["measurement_input_binding"] = binding
    report.pop("report_sha256", None)
    report["report_sha256"] = worker._report_sha256(report)


def test_import_is_inert_with_respect_to_transformers_and_model_loading() -> None:
    script = """
import sys
import fisher_graph.gemma3_typed_mode_response_experiment
assert 'transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=_ROOT, check=True)


def test_setup_response_abi_parses_verbatim_through_generic_core() -> None:
    setup = build_gemma3_typed_mode_response_setup()
    response = worker._response_abi(setup["common_response_abi"])

    assert response.canonical_payload() == setup["common_response_abi"]
    assert response.response_coordinate_signature.endswith("z1_minus_z0")


def test_canonical_response_bootstrap_projects_exact_generic_schema() -> None:
    setup = build_gemma3_typed_mode_response_setup()
    setup_before = deepcopy(setup)

    generic = worker._generic_core_response_bootstrap_config(setup)
    payload = generic.canonical_payload()

    assert set(payload) == worker._GENERIC_BOOTSTRAP_FIELDS
    assert payload["sampling_unit"] == (
        "example_plus_selected_probes_cluster"
    )
    assert setup == setup_before
    semantic = setup["common_response_within_family_bootstrap_config"]
    assert semantic["sampling_unit"] == (
        "example_plus_centered_logits4_response_cluster"
    )
    assert semantic["response_coordinates_per_example"] == 4
    assert semantic["all_four_coordinates_resampled_together"] is True


def test_analytic_output_fisher_is_raw_float64_gradient_square() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 1.0, -1.0, 0.5]], dtype=torch.float32)
    gold = torch.tensor([1])

    weights = worker._analytic_output_fisher_weights(
        logits, gold, torch=torch
    )
    probabilities = torch.softmax(logits.double(), dim=-1)
    expected = probabilities.clone()
    expected[0, 1] -= 1.0
    expected = expected.square()

    assert weights.dtype == torch.float64
    assert torch.equal(weights, expected)
    assert weights.sum().item() != pytest.approx(1.0)


def test_response_sign_and_centering_are_exactly_scale1_minus_scale0() -> None:
    torch = pytest.importorskip("torch")
    baseline = torch.tensor([[4.0, 2.0, 1.0, -1.0]])
    ablated = torch.tensor([[1.0, 2.0, 0.0, -1.0]])

    response = worker._center_response(
        baseline, ablated, torch=torch
    )

    assert torch.equal(
        response,
        torch.tensor([[2.0, -1.0, 0.0, -1.0]], dtype=torch.float64),
    )
    assert response.sum().item() == 0.0


def test_both_lexical_endpoints_are_distinct_executed_occurrences() -> None:
    source = inspect.getsource(worker._measure_responses_live)

    assert '"inputs_embeds": torch.zeros_like(inputs_embeds)' in source
    assert "output_embedding.register_forward_hook(zero_egress)" in source
    assert "lexical_egress_hook_count += 1" in source


def test_complete_pair_statistics_preserve_tied_lexical_exact_identity() -> None:
    measurement = _synthetic_measurement()

    families, examples = worker._build_pair_statistics(
        measurement,
        response_abi_sha256=_sha("response-abi"),
    )

    assert len(families) == 4_218
    assert len(examples) == 42_180
    tied_families = [
        row
        for row in families
        if {row.mode_a_id, row.mode_b_id} == set(LEXICAL_MODE_IDS)
    ]
    tied_examples = [
        row
        for row in examples
        if {row.mode_a_id, row.mode_b_id} == set(LEXICAL_MODE_IDS)
    ]
    assert len(tied_families) == 6
    assert len(tied_examples) == 60
    assert all(row.weighted_sse == 0.0 for row in tied_families)
    assert all(
        row.weighted_mode_a_energy
        == row.weighted_dot
        == row.weighted_mode_b_energy
        for row in (*tied_families, *tied_examples)
    )
    assert {row.row_count for row in families} == {40}
    assert {row.row_count for row in examples} == {4}


def test_runtime_weight_catalog_hash_binds_exact_float64_bits() -> None:
    base = [
        {
            "family_id": "family",
            "example_id": "example",
            "choice_coordinate": 0,
            "value": 0.25,
        }
    ]
    changed = deepcopy(base)
    changed[0]["value"] = 0.25000000000000006

    first = worker._float64_catalog_sha256(
        base, domain=worker._RUNTIME_WEIGHT_CATALOG_DOMAIN
    )
    second = worker._float64_catalog_sha256(
        changed, domain=worker._RUNTIME_WEIGHT_CATALOG_DOMAIN
    )

    assert first != second


def test_orchestration_completes_preflight_before_live_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    sentinels = {
        name: object()
        for name in (
            "preflight",
            "dependencies",
            "runtime",
            "measurement",
            "atlas",
            "report",
        )
    }

    def phase(name: str, result: object):
        def run(*_args: object, **_kwargs: object) -> object:
            calls.append(name)
            return result

        return run

    monkeypatch.setattr(worker, "_preflight", phase("preflight", sentinels["preflight"]))
    monkeypatch.setattr(
        worker,
        "_load_live_dependencies",
        phase("dependencies", sentinels["dependencies"]),
    )
    monkeypatch.setattr(worker, "_prepare_runtime", phase("runtime", sentinels["runtime"]))
    monkeypatch.setattr(
        worker,
        "_measure_responses",
        phase("measurement", sentinels["measurement"]),
    )
    monkeypatch.setattr(worker, "_build_multi_atlas", phase("atlas", sentinels["atlas"]))
    monkeypatch.setattr(worker, "_assemble_report", phase("report", sentinels["report"]))
    monkeypatch.setattr(worker, "write_gemma3_typed_mode_response_report", phase("write", {"ok": True}))

    result = worker.run_gemma3_typed_mode_response_experiment(
        output=tmp_path / "report.json"
    )

    assert result == {"ok": True}
    assert calls == [
        "preflight",
        "dependencies",
        "runtime",
        "measurement",
        "atlas",
        "report",
        "write",
    ]


def test_failed_preflight_never_resolves_live_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail(**_kwargs: object) -> object:
        raise RuntimeError("preflight failed")

    def live() -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(worker, "_preflight", fail)
    monkeypatch.setattr(worker, "_load_live_dependencies", live)

    with pytest.raises(RuntimeError, match="preflight failed"):
        worker.run_gemma3_typed_mode_response_experiment()
    assert called is False


@pytest.mark.parametrize("attack", ["extra_field", "sampling_unit"])
def test_real_preflight_rejects_bootstrap_adapter_drift_before_live_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attack: str,
) -> None:
    setup = build_gemma3_typed_mode_response_setup()
    bootstrap = setup["common_response_within_family_bootstrap_config"]
    if attack == "extra_field":
        bootstrap["replacement_semantic_metadata"] = True
    else:
        bootstrap["sampling_unit"] = "replacement_sampling_unit"
    live_called = False

    monkeypatch.setattr(
        worker,
        "load_gemma3_typed_mode_response_setup",
        lambda _path: setup,
    )

    def live() -> object:
        nonlocal live_called
        live_called = True
        return object()

    monkeypatch.setattr(worker, "_load_live_dependencies", live)

    with pytest.raises(ValueError, match="bootstrap (schema|sampling unit)"):
        worker.run_gemma3_typed_mode_response_experiment(
            output=tmp_path / f"{attack}.json",
            setup_path=_ROOT / worker.DEFAULT_SETUP_OUTPUT,
            panel_path=_ROOT / worker.PINNED_PANEL_PATH,
            mlp_report_path=_ROOT / worker.DEFAULT_MLP_REPORT,
            attention_report_path=_ROOT / worker.DEFAULT_ATTENTION_REPORT,
        )
    assert live_called is False


def test_rehashed_claim_and_call_count_attacks_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker,
        "parse_multi_abi_mode_atlas_receipt_payload",
        lambda _value: _fake_core_receipt(),
    )
    report = _semantic_report()
    assert worker.validate_gemma3_typed_mode_response_report(report) == report

    claim = deepcopy(report)
    claim["claims"]["cluster_claimed"] = True
    claim.pop("report_sha256")
    claim["report_sha256"] = worker._report_sha256(claim)
    with pytest.raises(ValueError, match="forbidden"):
        worker.validate_gemma3_typed_mode_response_report(claim)

    calls = deepcopy(report)
    calls["measurement"]["full_model_forward_count"] = 719
    calls.pop("report_sha256")
    calls["report_sha256"] = worker._report_sha256(calls)
    with pytest.raises(ValueError, match="call audit"):
        worker.validate_gemma3_typed_mode_response_report(calls)


def test_rehashed_tied_lexical_hash_attack_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker,
        "parse_multi_abi_mode_atlas_receipt_payload",
        lambda _value: _fake_core_receipt(),
    )
    report = _semantic_report()
    report["measurement"]["response_sha256_by_mode"][
        LEXICAL_MODE_IDS[1]
    ] = _sha("different")
    report.pop("report_sha256")
    report["report_sha256"] = worker._report_sha256(report)

    with pytest.raises(ValueError, match="per-mode"):
        worker.validate_gemma3_typed_mode_response_report(report)


def test_coordinated_rehash_cannot_replace_pinned_model_or_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker,
        "parse_multi_abi_mode_atlas_receipt_payload",
        lambda _value: _fake_core_receipt(),
    )
    model = _semantic_report()
    model["execution_audit"]["source_model_fingerprint_before"] = _sha(
        "replacement-model"
    )
    model["execution_audit"]["source_model_fingerprint_after"] = _sha(
        "replacement-model"
    )
    model.pop("report_sha256")
    model["report_sha256"] = worker._report_sha256(model)
    with pytest.raises(ValueError, match="execution safety"):
        worker.validate_gemma3_typed_mode_response_report(model)

    setup = _semantic_report()
    setup["setup_and_predecessors"]["setup_sha256"] = _sha(
        "replacement-setup"
    )
    setup["setup_and_predecessors"]["setup_file_sha256"] = _sha(
        "replacement-setup-file"
    )
    setup.pop("report_sha256")
    setup["report_sha256"] = worker._report_sha256(setup)
    with pytest.raises(ValueError, match="predecessor binding"):
        worker.validate_gemma3_typed_mode_response_report(setup)


def test_native_partition_receipt_is_bound_to_native_report_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _semantic_report()
    replacement_native = _sha("replacement-attention-native")
    replacement_file = _sha("replacement-attention-file")
    replacement_report = _sha("replacement-attention-report")
    fake = _fake_core_receipt(report)
    next(
        item
        for item in fake.native_partitions
        if item.operator_kind is worker.ModeOperatorKind.ATTENTION
    ).atlas.receipt_sha256 = replacement_native
    monkeypatch.setattr(
        worker,
        "parse_multi_abi_mode_atlas_receipt_payload",
        lambda _value: fake,
    )
    report["setup_and_predecessors"][
        "attention_report_file_sha256"
    ] = replacement_file
    report["setup_and_predecessors"][
        "attention_report_sha256"
    ] = replacement_report
    report["setup_and_predecessors"][
        "attention_native_receipt_sha256"
    ] = replacement_native
    report["setup_and_predecessors"][
        "attention_partition_binding_sha256"
    ] = worker._native_partition_binding_sha256(
        operator_kind="attention",
        report_file_sha256=replacement_file,
        report_sha256=replacement_report,
        native_receipt_sha256=replacement_native,
    )
    report.pop("report_sha256")
    report["report_sha256"] = worker._report_sha256(report)

    with pytest.raises(ValueError, match="predecessor binding"):
        worker.validate_gemma3_typed_mode_response_report(report)


@pytest.mark.parametrize("attack", ["weight", "response", "state", "parity"])
def test_coordinated_measurement_rehash_requires_rebuilding_core_receipt(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    report = _semantic_report()
    stale_core = _fake_core_receipt(report)
    measurement = report["measurement"]
    if attack == "weight":
        measurement["runtime_output_Fisher_weight_catalog_sha256"] = _sha(
            "replacement-weights"
        )
    elif attack == "response":
        measurement["response_sha256_by_mode"][RESPONSE_MODE_IDS[0]] = _sha(
            "replacement-response"
        )
    elif attack == "state":
        replacement_state = _sha("replacement-tied-state")
        for mode_id in LEXICAL_MODE_IDS:
            measurement["mode_state_fingerprint_sha256_by_mode"][
                mode_id
            ] = replacement_state
        measurement[
            "tied_lexical_shared_state_fingerprint_sha256"
        ] = replacement_state
    else:
        replacement_parity = _sha("replacement-parity")
        measurement["baseline_parity_receipts"][0][
            "token_id_logits4_sha256"
        ] = replacement_parity
        measurement["baseline_parity_receipts"][0][
            "inputs_embeds_logits4_sha256"
        ] = replacement_parity
    _rebind_report_measurement(report)
    monkeypatch.setattr(
        worker,
        "parse_multi_abi_mode_atlas_receipt_payload",
        lambda _value: stale_core,
    )

    with pytest.raises(
        ValueError,
        match="multi-ABI (receipt scope|mode binding)",
    ):
        worker.validate_gemma3_typed_mode_response_report(report)


@pytest.mark.parametrize("attack", ["lexical_kind", "terminal_abi"])
def test_core_modes_are_bound_to_canonical_setup_kind_and_abi(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    report = _semantic_report()
    fake = _fake_core_receipt(report)
    setup = build_gemma3_typed_mode_response_setup()
    modes = list(fake.modes)
    if attack == "lexical_kind":
        index = next(
            index
            for index, mode in enumerate(modes)
            if mode.mode_id == LEXICAL_MODE_IDS[0]
        )
        modes[index] = replace(
            modes[index],
            native_mode_kind="lexical_boundary.ingress.replacement",
        )
    else:
        index = next(
            index
            for index, mode in enumerate(modes)
            if mode.mode_id == worker.TERMINAL_NORM_MODE_ID
        )
        modes[index] = replace(
            modes[index],
            native_abi=worker._mode_abi(
                setup["native_abis"]["attention"],
                label="replacement terminal ABI",
            ),
        )
    fake.modes = tuple(modes)
    monkeypatch.setattr(
        worker,
        "parse_multi_abi_mode_atlas_receipt_payload",
        lambda _value: fake,
    )

    with pytest.raises(ValueError, match="multi-ABI (mode )?binding"):
        worker.validate_gemma3_typed_mode_response_report(report)


def test_strict_json_reader_rejects_duplicate_keys_nan_and_symlinks(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        worker._strict_json_object_file(duplicate, label="fixture")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported constant"):
        worker._strict_json_object_file(nonfinite, label="fixture")

    canonical = tmp_path / "canonical.json"
    canonical.write_text('{\n  "a": 1\n}\n', encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(canonical)
    with pytest.raises(ValueError, match="symlink"):
        worker._strict_json_object_file(link, label="fixture")


def test_heatmap_is_deterministic_and_has_38_by_38_cells() -> None:
    axis = list(RESPONSE_MODE_IDS)
    matrix = [
        [float(abs(left - right)) for right in range(38)]
        for left in range(38)
    ]
    report = {
        "multi_abi_atlas": {
            "downstream_response": {
                "analysis": {
                    "logical_mode_axis": axis,
                    "logical_seriation_order": list(reversed(axis)),
                    "logical_symmetric_distance": matrix,
                }
            }
        }
    }

    first = worker._render_heatmap_svg(report)
    second = worker._render_heatmap_svg(report)

    assert first == second
    assert first.count("<rect ") == 1 + 38 * 38
    assert ">LI</text>" in first
    assert ">LE</text>" in first


def test_production_shape_actual_core_report_write_load_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise 18+18+2 modes and all 46,398 pair rows model-free."""

    setup = build_gemma3_typed_mode_response_setup()
    mlp_report = load_gemma3_mlp_mode_similarity_atlas_report(
        _ROOT / worker.DEFAULT_MLP_REPORT
    )
    mlp_native = parse_mode_similarity_atlas_receipt_payload(
        mlp_report["atlas"]
    )
    attention_abi = worker._mode_abi(
        setup["native_abis"]["attention"], label="attention ABI"
    )
    id_map = {
        f"mode.mutation.{2 * block + 2}": f"mode.mutation.{2 * block + 1}"
        for block in range(18)
    }
    attention_stats = tuple(
        DirectedFamilyStats(
            family_id=row.family_id,
            reference_mode_id=id_map[row.reference_mode_id],
            candidate_mode_id=id_map[row.candidate_mode_id],
            compatibility_abi_sha256=attention_abi.sha256,
            weighted_sse=row.weighted_sse,
            weighted_reference_energy=row.weighted_reference_energy,
            weighted_dot=row.weighted_dot,
            weighted_candidate_energy=row.weighted_candidate_energy,
            row_count=row.row_count,
        )
        for row in mlp_native.directed_stats
    )
    attention_examples = tuple(
        DirectedFamilyExampleStats(
            family_id=row.family_id,
            example_cluster_id=row.example_cluster_id,
            reference_mode_id=id_map[row.reference_mode_id],
            candidate_mode_id=id_map[row.candidate_mode_id],
            compatibility_abi_sha256=attention_abi.sha256,
            weighted_sse=row.weighted_sse,
            weighted_reference_energy=row.weighted_reference_energy,
            weighted_dot=row.weighted_dot,
            weighted_candidate_energy=row.weighted_candidate_energy,
            row_count=row.row_count,
        )
        for row in mlp_native.directed_example_stats
    )

    base_measurement = _synthetic_measurement()
    fingerprints = dict(base_measurement.state_fingerprints_by_mode)
    fingerprints.update(
        {
            mode.mode_id: mode.state_fingerprint_sha256
            for mode in mlp_native.modes
        }
    )
    fingerprints.update(
        {
            attention_id: _sha(f"attention-state:{attention_id}")
            for attention_id in id_map.values()
        }
    )
    measurement_receipt = deepcopy(_semantic_report()["measurement"])
    measurement_receipt[
        "mode_state_fingerprint_sha256_by_mode"
    ] = fingerprints
    measurement_receipt["terminal_norm_state_fingerprint_sha256"] = (
        base_measurement.terminal_norm_state_fingerprint_sha256
    )
    measurement_receipt[
        "tied_lexical_shared_state_fingerprint_sha256"
    ] = fingerprints[LEXICAL_MODE_IDS[0]]
    measurement_input = worker._measurement_input_binding(
        setup=setup,
        runtime_weight_catalog_sha256=measurement_receipt[
            "runtime_output_Fisher_weight_catalog_sha256"
        ],
        baseline_parity_receipts=measurement_receipt[
            "baseline_parity_receipts"
        ],
        response_sha256_by_mode=measurement_receipt[
            "response_sha256_by_mode"
        ],
        mode_state_fingerprint_sha256_by_mode=fingerprints,
        terminal_norm_state_fingerprint_sha256=(
            base_measurement.terminal_norm_state_fingerprint_sha256
        ),
    )
    measurement_receipt["measurement_input_binding"] = measurement_input
    measurement_receipt["measurement_input_sha256"] = measurement_input[
        "measurement_input_sha256"
    ]
    measurement = replace(
        base_measurement,
        state_fingerprints_by_mode=fingerprints,
        receipt=measurement_receipt,
    )
    output = tmp_path / "typed-response.json"
    attention_file = tmp_path / "attention.json"
    attention_file.write_text("{}\n", encoding="utf-8")
    attention_file_sha = worker._sha256_file(attention_file)
    attention_report_sha = _sha("synthetic-attention-report")
    placeholder_preflight = worker._Preflight(
        setup=setup,
        examples=(),
        family_ids=tuple(ORDERED_FAMILY_IDS),
        setup_path=_ROOT / worker.DEFAULT_SETUP_OUTPUT,
        panel_path=_ROOT / worker.PINNED_PANEL_PATH,
        mlp_report_path=_ROOT / worker.DEFAULT_MLP_REPORT,
        attention_report_path=attention_file,
        output_path=output,
        svg_path=worker._default_svg_path(output),
        mlp_report=mlp_report,
        attention_report={},
        mlp_native_receipt=mlp_native,
        attention_native_receipt=SimpleNamespace(),
        receipts={},
    )
    response_abi = worker._response_abi(setup["common_response_abi"])
    family_stats, example_stats = worker._build_pair_statistics(
        measurement,
        response_abi_sha256=response_abi.sha256,
    )
    modes = worker._typed_modes(placeholder_preflight, measurement)
    multi = worker.build_multi_abi_mode_atlas_receipt(
        worker.MultiABIAtlasInventory(
            attention_mode_count=18, mlp_mode_count=18
        ),
        modes,
        native_atlas_inputs=(
            worker.NativeAtlasInput(
                compatibility_abi_sha256=attention_abi.sha256,
                directed_stats=attention_stats,
                directed_example_stats=attention_examples,
                config=parse_atlas_config_payload(
                    setup["attention_atlas_config"]
                ),
            ),
            worker.NativeAtlasInput.from_receipt(mlp_native),
        ),
        downstream_response_input=worker.DownstreamResponseAtlasInput(
            response_abi=response_abi,
            measurement_input_sha256=measurement_receipt[
                "measurement_input_sha256"
            ],
            mode_ids=tuple(RESPONSE_MODE_IDS),
            family_pair_stats=family_stats,
            example_pair_stats=example_stats,
            config=parse_atlas_config_payload(
                setup["common_response_atlas_config"]
            ),
            within_family_bootstrap_config=(
                worker._generic_core_response_bootstrap_config(setup)
            ),
        ),
    )
    multi = worker.parse_multi_abi_mode_atlas_receipt_payload(
        multi.canonical_payload()
    )
    attention_partition = next(
        item
        for item in multi.native_partitions
        if item.operator_kind is worker.ModeOperatorKind.ATTENTION
    )
    assert attention_partition.atlas is not None
    attention_native_sha = attention_partition.atlas.receipt_sha256
    monkeypatch.setattr(
        worker,
        "_PINNED_ATTENTION_REPORT_FILE_SHA256",
        attention_file_sha,
    )
    monkeypatch.setattr(
        worker,
        "_PINNED_ATTENTION_REPORT_SHA256",
        attention_report_sha,
    )
    monkeypatch.setattr(
        worker,
        "_PINNED_ATTENTION_NATIVE_RECEIPT_SHA256",
        attention_native_sha,
    )
    lineage = {
        "setup_file_sha256": worker.PINNED_SETUP_FILE_SHA256,
        "setup_sha256": worker.PINNED_SETUP_SHA256,
        "panel_file_sha256": worker.PINNED_PANEL_FILE_SHA256,
        "mlp_report_file_sha256": worker.PINNED_MLP_REPORT_FILE_SHA256,
        "mlp_report_sha256": worker.PINNED_MLP_REPORT_SHA256,
        "mlp_native_receipt_sha256": (
            worker._PINNED_MLP_NATIVE_RECEIPT_SHA256
        ),
        "mlp_partition_binding_sha256": (
            worker._native_partition_binding_sha256(
                operator_kind="mlp",
                report_file_sha256=worker.PINNED_MLP_REPORT_FILE_SHA256,
                report_sha256=worker.PINNED_MLP_REPORT_SHA256,
                native_receipt_sha256=(
                    worker._PINNED_MLP_NATIVE_RECEIPT_SHA256
                ),
            )
        ),
        "attention_report_file_sha256": attention_file_sha,
        "attention_report_sha256": attention_report_sha,
        "attention_native_receipt_sha256": attention_native_sha,
        "attention_partition_binding_sha256": (
            worker._native_partition_binding_sha256(
                operator_kind="attention",
                report_file_sha256=attention_file_sha,
                report_sha256=attention_report_sha,
                native_receipt_sha256=attention_native_sha,
            )
        ),
        "all_authenticated_before_live_dependencies": True,
    }
    preflight = replace(placeholder_preflight, receipts=lineage)

    class Guard:
        def assert_unchanged(self) -> None:
            return None

    runtime = SimpleNamespace(
        source_guard=Guard(),
        adapter=SimpleNamespace(
            model_fingerprint=lambda: (
                worker.PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
            ),
            execution_fingerprint=lambda: (
                worker.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
            ),
        ),
        source_model_fingerprint_before=(
            worker.PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
        ),
        source_execution_fingerprint_before=(
            worker.PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        model=SimpleNamespace(parameters=lambda: ()),
        model_receipt={
            "model_id": worker.PINNED_MODEL_ID,
            "requested_revision": worker.PINNED_MODEL_REVISION,
            "resolved_commit": worker.PINNED_MODEL_REVISION,
            "parameter_count": worker.PINNED_SOURCE_PARAMETER_COUNT,
            "device": "cpu",
            "dtype": "float32",
            "weights_in_report": False,
        },
    )
    report = worker._assemble_report(
        preflight=preflight,
        runtime=runtime,
        measurement=measurement,
        multi_atlas=multi,
        device_name="cpu",
        dtype="float32",
        batch_size=4,
    )
    validated = worker.validate_gemma3_typed_mode_response_report(report)
    attention_report_stub = {
        "report_sha256": attention_report_sha,
        "atlas": attention_partition.atlas.canonical_payload(),
    }
    import fisher_graph.gemma3_native_attention_mode_atlas_experiment as attention_worker

    monkeypatch.setattr(
        attention_worker,
        "load_gemma3_native_attention_mode_atlas_report",
        lambda _path: attention_report_stub,
    )
    loaded = worker.write_gemma3_typed_mode_response_report(validated, output)

    assert loaded == validated
    assert loaded["response_summary"]["family_pair_stat_count"] == 4_218
    assert loaded["response_summary"]["example_pair_stat_count"] == 42_180
    assert loaded["response_summary"][
        "exact_response_equivalence_class_count"
    ] == 37


def test_main_help_is_model_free() -> None:
    with pytest.raises(SystemExit) as raised:
        worker.main(["--help"])
    assert raised.value.code == 0


def test_main_prints_only_a_concise_receipt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        worker,
        "run_gemma3_typed_mode_response_experiment",
        lambda **_kwargs: {
            "diagnostic_outcome": "typed_mode_response_geometry_discovery_only",
            "report_sha256": _sha("report"),
            "multi_abi_atlas": {"would_be": "huge"},
        },
    )

    assert worker.main(["--output", "receipt.json"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload == {
        "diagnostic_outcome": "typed_mode_response_geometry_discovery_only",
        "output": "receipt.json",
        "report_sha256": _sha("report"),
    }
    assert len(output) < 512
