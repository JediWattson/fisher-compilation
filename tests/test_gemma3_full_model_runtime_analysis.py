from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from fisher_graph.gemma3_full_model_runtime_analysis import (
    _PROVENANCE_SOURCE_FILES,
    _ideal_linear_speedups,
    _report_sha256,
    _resource_accounting,
    _serialize_benchmark,
    evaluate_prepared_full_model_scopes,
    load_gemma3_full_model_runtime_analysis,
)
from fisher_graph.model_runtime_benchmark import (
    ModelRuntimeBenchmarkReport,
    ModelRuntimeTiming,
)
from fisher_graph.prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)

from test_gemma3_full_mlp_stack_executor import _full_replacements
from test_gemma3_modal_generator_executor import _adapter, _batch


def _switcher() -> PreparedGemma3FullMLPStackSwitcher:
    adapter = _adapter()
    return PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {"factorized_refit": _full_replacements(adapter)},
        fused_variants={"fused_refit": "factorized_refit"},
    )


def test_evaluates_exact_prepared_scopes_and_restores_native() -> None:
    switcher = _switcher()
    batch = _batch().sample(0)

    report = evaluate_prepared_full_model_scopes(
        switcher,
        (batch,),
        vocabulary_chunk_size=5,
    )

    assert switcher.active_scope == "native"
    assert report["assessment_role"] == "open_development_assessment"
    assert report["heldout_confirmation"] is False
    assert report["supervised_tokens"] > 0
    conditions = report["conditions"]
    assert set(conditions) == {
        "native",
        "factorized_refit",
        "fused_refit",
    }
    assert conditions["native"]["delta_nll_per_token"] == 0.0
    assert 0.0 <= conditions["factorized_refit"][
        "top1_agreement_to_native"
    ] <= 1.0
    pair = report["fused_vs_factorized"]
    assert pair["factorized_to_fused_kl_per_token"] >= 0.0
    assert 0.0 <= pair["top1_agreement_to_factorized"] <= 1.0
    assert pair["logit_nrmse"] >= 0.0
    assert -1.0 <= pair["logit_cosine"] <= 1.000001
    assert pair["logit_max_absolute_error"] >= 0.0

    switcher.close()


def test_resource_accounting_separates_logical_and_resident_scopes() -> None:
    switcher = _switcher()
    native_mlp = switcher.scope_parameter_counts["native"]
    source_whole = native_mlp + 1000

    report = _resource_accounting(
        switcher,
        source_whole_model_parameters=source_whole,
    )

    systems = report["systems"]
    assert systems["native"]["logical_whole_model_learned_parameters"] == (
        source_whole
    )
    for scope in ("factorized_refit", "fused_refit"):
        assert systems[scope][
            "logical_whole_model_learned_parameters"
        ] == 1000 + switcher.scope_parameter_counts[scope]
    assert report["experimental_runtime_retains_all_scopes"] is True
    assert report["logical_deployment_retains_only_selected_scope"] is True
    switcher.close()


def test_linear_only_ideal_speedups_include_head_amortization() -> None:
    scope_macs = {
        "native": 100,
        "factorized_refit": 40,
        "fused_refit": 20,
    }
    non_mlp = {
        "attention_projection_macs_per_token": 20,
        "other_linear_macs_per_token": 0,
        "lm_head_macs_per_emitted_logit": 200,
    }

    decode = _ideal_linear_speedups(
        kind="decode",
        batch_size=1,
        context_length=128,
        scope_macs=scope_macs,
        non_mlp_macs=non_mlp,
    )
    prefill = _ideal_linear_speedups(
        kind="prefill",
        batch_size=1,
        context_length=128,
        scope_macs=scope_macs,
        non_mlp_macs=non_mlp,
    )

    assert decode["factorized_refit"] == pytest.approx(320 / 260)
    assert decode["fused_refit"] == pytest.approx(320 / 240)
    assert prefill["factorized_refit"] > decode["factorized_refit"]
    assert prefill["fused_refit"] > decode["fused_refit"]


def test_fidelity_rejects_wrong_scope_and_invalid_chunk() -> None:
    adapter = _adapter()
    wrong = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {"only_candidate": _full_replacements(adapter)},
    )
    with pytest.raises(ValueError, match="exact systems"):
        evaluate_prepared_full_model_scopes(wrong, (_batch().sample(0),))
    wrong.close()

    switcher = _switcher()
    with pytest.raises(ValueError, match="positive"):
        evaluate_prepared_full_model_scopes(
            switcher,
            (_batch().sample(0),),
            vocabulary_chunk_size=0,
        )
    switcher.close()


def test_benchmark_serialization_rejects_nonfinite_or_mismatched_outputs(
) -> None:
    timing = ModelRuntimeTiming(
        cold_seconds=1.0,
        raw_round_seconds=(1.0,),
        median_seconds=1.0,
        p10_seconds=1.0,
        p90_seconds=1.0,
        p95_seconds=1.0,
        processed_token_count=1,
        tokens_per_second=1.0,
    )
    report = ModelRuntimeBenchmarkReport(
        system_names=(
            "native",
            "factorized_refit",
            "fused_refit",
        ),
        rounds=1,
        warmup_calls=0,
        processed_token_count=1,
        timings={
            "native": timing,
            "factorized_refit": timing,
            "fused_refit": timing,
        },
        round_orders=(
            ("native", "factorized_refit", "fused_refit"),
        ),
    )
    valid = {
        scope: torch.zeros((1, 1, 3))
        for scope in report.system_names
    }
    serialized = _serialize_benchmark(report, output_logits=valid)
    assert all(
        row["finite"]
        for row in serialized["output_validation"].values()
    )

    nonfinite = dict(valid)
    nonfinite["fused_refit"] = torch.full((1, 1, 3), torch.nan)
    with pytest.raises(RuntimeError, match="finite same-shape"):
        _serialize_benchmark(report, output_logits=nonfinite)

    mismatched = dict(valid)
    mismatched["factorized_refit"] = torch.zeros((1, 1, 4))
    with pytest.raises(RuntimeError, match="finite same-shape"):
        _serialize_benchmark(report, output_logits=mismatched)


def _valid_saved_payload() -> dict[str, object]:
    def timing(
        raw: list[float],
        *,
        processed_token_count: int,
    ) -> dict[str, object]:
        ordered = sorted(raw)
        median = ordered[1]
        return {
            "cold_seconds": 0.5,
            "raw_round_seconds": raw,
            "median_seconds": median,
            "p10_seconds": ordered[0] * 0.8 + ordered[1] * 0.2,
            "p90_seconds": ordered[1] * 0.2 + ordered[2] * 0.8,
            "p95_seconds": ordered[1] * 0.1 + ordered[2] * 0.9,
            "processed_token_count": processed_token_count,
            "tokens_per_second": processed_token_count / median,
        }

    def benchmark(
        kind: str,
        *,
        context_length: int,
        processed_token_count: int,
    ) -> dict[str, object]:
        return {
            "kind": kind,
            "batch_size": 1,
            "context_length": context_length,
            "system_names": [
                "native",
                "factorized_refit",
                "fused_refit",
            ],
            "rounds": 3,
            "warmup_calls": 1,
            "processed_token_count": processed_token_count,
            "round_orders": [
                ["native", "factorized_refit", "fused_refit"],
                ["factorized_refit", "fused_refit", "native"],
                ["fused_refit", "native", "factorized_refit"],
            ],
            "timings": {
                "native": timing(
                    [1.0, 2.0, 3.0],
                    processed_token_count=processed_token_count,
                ),
                "factorized_refit": timing(
                    [0.5, 1.0, 1.5],
                    processed_token_count=processed_token_count,
                ),
                "fused_refit": timing(
                    [0.25, 0.5, 0.75],
                    processed_token_count=processed_token_count,
                ),
            },
            "speedup_vs_native": {
                "factorized_refit": 2.0,
                "fused_refit": 4.0,
                "fused_vs_factorized": 2.0,
            },
            "output_validation": {
                scope: {
                    "shape": [1, 1, 17],
                    "dtype": "torch.float32",
                    "sha256": character * 64,
                    "finite": True,
                }
                for scope, character in (
                    ("native", "1"),
                    ("factorized_refit", "2"),
                    ("fused_refit", "3"),
                )
            },
        }

    without_digest = {
        "schema": (
            "fisher_graph.gemma3_full_model_runtime_analysis_development"
        ),
        "format_version": 1,
        "protocol": {
            "systems": [
                "native",
                "factorized_refit",
                "fused_refit",
            ],
            "rounds": 3,
            "warmup_calls": 1,
            "batch_sizes": [1],
            "context_lengths": [8, 16],
        },
        "source_code": {
            "binding": "sha256_of_listed_primary_runtime_sources",
            "files_sha256": {
                path: "c" * 64
                for path in _PROVENANCE_SOURCE_FILES
            },
        },
        "benchmark_environment": {"processor": "test-processor"},
        "assessment": {
            "serialized_sha256": "a" * 64,
            "content_sha256": ["b" * 64],
            "example_count": 1,
            "prompt_text_stored": False,
            "token_ids_stored": False,
            "family_disjoint_confirmation": False,
        },
        "resources": {
            "systems": {
                "native": {},
                "factorized_refit": {},
                "fused_refit": {},
            }
        },
        "fidelity": {
            "conditions": {
                "native": {},
                "factorized_refit": {},
                "fused_refit": {},
            }
        },
        "benchmarks": [
            benchmark(
                "prefill_last_logit",
                context_length=8,
                processed_token_count=8,
            ),
            benchmark(
                "cached_single_token_decode",
                context_length=8,
                processed_token_count=1,
            ),
            benchmark(
                "prefill_last_logit",
                context_length=16,
                processed_token_count=16,
            ),
            benchmark(
                "cached_single_token_decode",
                context_length=16,
                processed_token_count=1,
            ),
        ],
        "claim_scope": {
            "logical_parameter_and_linear_mac_counts": True,
            "torch_full_model_latency_measured": True,
            "mlx_full_model_latency_measured": False,
            "fused_scope_is_a_rate_distortion_point": True,
            "fused_scope_is_bit_exact_to_factorized": False,
            "heldout_compression_claim": False,
            "downstream_accuracy_claim": False,
        },
    }
    return {
        **without_digest,
        "report_sha256": _report_sha256(without_digest),
    }


def test_strict_loader_authenticates_digest(
    tmp_path: Path,
) -> None:
    payload = _valid_saved_payload()
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_gemma3_full_model_runtime_analysis(path) == payload

    payload["benchmarks"][0]["speedup_vs_native"][  # type: ignore[index]
        "fused_refit"
    ] = 99.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_gemma3_full_model_runtime_analysis(path)


def _redigest(payload: dict[str, object]) -> None:
    without_digest = {
        key: value
        for key, value in payload.items()
        if key != "report_sha256"
    }
    payload["report_sha256"] = _report_sha256(without_digest)


def _tamper_output_finiteness(payload: dict[str, Any]) -> None:
    payload["benchmarks"][0]["output_validation"]["native"]["finite"] = False


def _tamper_output_shape(payload: dict[str, Any]) -> None:
    payload["benchmarks"][0]["output_validation"]["native"]["shape"] = [1, 2, 17]


def _tamper_quantile(payload: dict[str, Any]) -> None:
    payload["benchmarks"][0]["timings"]["native"]["p95_seconds"] = 99.0


def _tamper_throughput(payload: dict[str, Any]) -> None:
    payload["benchmarks"][0]["timings"]["native"]["tokens_per_second"] = 99.0


def _tamper_round_orders(payload: dict[str, Any]) -> None:
    payload["benchmarks"][0]["round_orders"] = []


def _tamper_speedup(payload: dict[str, Any]) -> None:
    payload["benchmarks"][0]["speedup_vs_native"]["fused_refit"] = 99.0


def _tamper_source_code(payload: dict[str, Any]) -> None:
    payload["source_code"]["files_sha256"][
        _PROVENANCE_SOURCE_FILES[0]
    ] = "not-a-sha256"


def _tamper_claim_scope(payload: dict[str, Any]) -> None:
    payload.pop("claim_scope")


def _tamper_assessment(payload: dict[str, Any]) -> None:
    payload.pop("assessment")


def _tamper_duplicate_matrix_row(payload: dict[str, Any]) -> None:
    payload["benchmarks"].append(copy.deepcopy(payload["benchmarks"][0]))


def _tamper_incomplete_matrix(payload: dict[str, Any]) -> None:
    payload["benchmarks"] = [
        row
        for row in payload["benchmarks"]
        if row["context_length"] != 16
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (_tamper_output_finiteness, "output validation"),
        (_tamper_output_shape, "output validation"),
        (_tamper_quantile, "timing summaries"),
        (_tamper_throughput, "timing throughput"),
        (_tamper_round_orders, "round orders"),
        (_tamper_speedup, "speedup"),
        (_tamper_source_code, "source-code provenance"),
        (_tamper_claim_scope, "top-level structure"),
        (_tamper_assessment, "top-level structure"),
        (_tamper_duplicate_matrix_row, "duplicate row"),
        (_tamper_incomplete_matrix, "matrix is not complete"),
    ),
)
def test_strict_loader_rejects_redigested_scientific_tampering(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    payload: dict[str, Any] = _valid_saved_payload()
    mutation(payload)
    _redigest(payload)
    path = tmp_path / "tampered-runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_gemma3_full_model_runtime_analysis(path)
