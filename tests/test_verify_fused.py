import copy
import json
from pathlib import Path

import pytest
import torch

from fisher_graph.verify import (
    _fused_gate_passed,
    _fused_lazy_benchmark_comparison,
    _verify_fused_benchmark,
)


ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "associative_recall"
)


def _fused_report() -> dict[str, object]:
    return json.loads(
        (ARTIFACTS / "fused_executor_report.json").read_text()
    )


def test_fused_benchmark_verifier_checks_structure_without_rerunning_timing():
    report = _fused_report()

    batch_count, minimum_speedup = _verify_fused_benchmark(
        report["benchmark_environment"],
        report["benchmark"],
    )

    assert batch_count == 4
    assert minimum_speedup > 0


def test_fused_benchmark_verifier_rejects_nonpositive_measurement():
    report = copy.deepcopy(_fused_report())
    report["benchmark"][0]["timings"]["lazy"][  # type: ignore[index]
        "raw_microseconds"
    ][0] = 0.0

    with pytest.raises(ValueError, match="raw_microseconds is invalid"):
        _verify_fused_benchmark(
            report["benchmark_environment"],
            report["benchmark"],
        )


def test_lazy_benchmark_comparison_is_derived_from_recorded_timings():
    report = _fused_report()

    expected = _fused_lazy_benchmark_comparison(report["benchmark"])

    assert expected == report["lazy_vs_monolithic_benchmark"]
    assert expected["hard_latency_gate_applied"] is False
    assert (
        expected["geometric_mean_lazy_to_monolithic_latency_ratio"]
        > 0
    )


def test_lazy_runtime_artifact_is_compact_and_source_bound():
    report = _fused_report()
    artifact = torch.load(
        ARTIFACTS / "fused_modal_runtime.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert set(artifact) == {
        "format_version",
        "artifact_kind",
        "config",
        "state_dict",
        "sidecars",
        "metadata",
    }
    assert artifact["format_version"] == 2
    assert artifact["artifact_kind"] == (
        "lazy_fused_two_layer_modal_stack"
    )
    assert set(artifact["state_dict"]) == {
        "first_input_mean",
        "first_input_kernel",
        "first_hidden_bias",
        "bridge_kernel",
        "bridge_bias",
        "second_fused_output_weight",
        "second_fused_output_bias",
    }
    assert (
        sum(
            tensor.numel() * tensor.element_size()
            for tensor in artifact["state_dict"].values()
        )
        == 199_808
    )
    assert artifact["sidecars"] == report["lazy_fused_artifact"][
        "sidecar_descriptors"
    ]
    for name in (
        "checkpoint_sha256",
        "fisher_sha256",
        "teacher_state_sha256",
    ):
        value = artifact["metadata"][name]
        assert len(value) == 64
        assert value == value.lower()


def test_fused_gate_requires_behavioral_and_numerical_equivalence():
    gate = {
        "maximum_absolute_nll_delta": 1e-6,
        "maximum_mean_answer_kl": 1e-6,
        "maximum_answer_logit_difference": 5e-4,
    }
    comparison = {
        "answer_accuracy_exactly_equal": True,
        "paired_context_accuracy_exactly_equal": True,
        "argmax_predictions_exactly_equal": True,
        "absolute_hard_nll_delta": 1e-7,
        "mean_unfused_to_fused_answer_kl": 1e-8,
        "maximum_answer_logit_difference": 2e-4,
    }
    assert _fused_gate_passed(comparison, gate)

    comparison["argmax_predictions_exactly_equal"] = False
    assert not _fused_gate_passed(comparison, gate)
