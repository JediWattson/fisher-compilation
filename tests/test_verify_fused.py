import copy
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from fisher_graph.associative import (
    AssociativeRecallTaskConfig,
    associative_recall_answer_logits,
    build_associative_recall_splits,
)
from fisher_graph.compiler.manifest import (
    manifest_from_legacy_runtime,
    save_runtime_manifest,
)
from fisher_graph.fused_executor import (
    FusedToyTransformer,
    PackedTriangularFusedTwoLayerModalStack,
    load_lazy_fused_modal_stack,
)
from fisher_graph.training import load_checkpoint
from fisher_graph.verify import (
    _fused_comparison,
    _fused_gate_passed,
    _fused_lazy_benchmark_comparison,
    _fused_triangular_benchmark_comparison,
    _verify_fused_benchmark,
    verify_build,
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


TRIANGULAR_SYSTEMS = (
    "teacher",
    "unfused",
    "monolithic",
    "lazy",
    "triangular",
)

TRIANGULAR_SPEEDUP_PAIRS = {
    "monolithic_vs_unfused": ("unfused", "monolithic"),
    "lazy_vs_unfused": ("unfused", "lazy"),
    "monolithic_vs_teacher": ("teacher", "monolithic"),
    "lazy_vs_teacher": ("teacher", "lazy"),
    "lazy_vs_monolithic": ("monolithic", "lazy"),
    "triangular_vs_unfused": ("unfused", "triangular"),
    "triangular_vs_teacher": ("teacher", "triangular"),
    "triangular_vs_monolithic": ("monolithic", "triangular"),
    "triangular_vs_lazy": ("lazy", "triangular"),
}


def _five_system_benchmark(
    report: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    environment = copy.deepcopy(report["benchmark_environment"])
    environment["benchmark_contract"]["systems"] = list(  # type: ignore[index]
        TRIANGULAR_SYSTEMS
    )
    benchmark = copy.deepcopy(report["benchmark"])
    for batch in benchmark:  # type: ignore[assignment]
        timings = batch["timings"]
        examples = batch["examples_per_second"]
        timings["triangular"] = copy.deepcopy(timings["lazy"])
        examples["triangular"] = examples["lazy"]
        medians = {
            name: timings[name]["median_microseconds"]
            for name in TRIANGULAR_SYSTEMS
        }
        batch["speedup_ratios"] = {
            name: medians[reference] / medians[candidate]
            for name, (reference, candidate) in (
                TRIANGULAR_SPEEDUP_PAIRS.items()
            )
        }
        repeats = timings["triangular"]["repeats"]
        batch["round_orders"] = [
            list(
                TRIANGULAR_SYSTEMS[offset:]
                + TRIANGULAR_SYSTEMS[:offset]
            )
            for repeat in range(repeats)
            for offset in (repeat % len(TRIANGULAR_SYSTEMS),)
        ]
    return environment, benchmark  # type: ignore[return-value]


def test_verify_build_retains_v2_fused_report_compatibility(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    shutil.copytree(ARTIFACTS, root)
    report_path = root / "fused_executor_report.json"
    report = json.loads(report_path.read_text())
    report["format_version"] = 2
    report.pop("triangular_runtime_benchmark")
    report["arithmetic"]["triangular_interpretation"] = (
        "nonzero causal arithmetic available to a sparse or "
        "triangular backend"
    )
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    save_runtime_manifest(
        root / "runtime_manifest.json",
        manifest_from_legacy_runtime(root),
    )

    summary = verify_build(root)["fused_executor"]

    assert summary["format_version"] == 2
    assert "triangular_runtime_present" not in summary


def test_fused_benchmark_verifier_checks_structure_without_rerunning_timing():
    report = _fused_report()

    batch_count, minimum_speedup = _verify_fused_benchmark(
        report["benchmark_environment"],
        report["benchmark"],
    )

    assert batch_count == 4
    assert minimum_speedup > 0


def test_five_system_triangular_benchmark_is_fully_derived_from_raw_timings():
    environment, benchmark = _five_system_benchmark(_fused_report())

    batch_count, minimum_speedup = _verify_fused_benchmark(
        environment,
        benchmark,
        systems=TRIANGULAR_SYSTEMS,
        speedup_pairs=TRIANGULAR_SPEEDUP_PAIRS,
        minimum_speedup_name="triangular_vs_lazy",
    )
    comparison = _fused_triangular_benchmark_comparison(benchmark)

    assert batch_count == 4
    assert minimum_speedup == 1.0
    assert comparison["geometric_mean_triangular_vs_lazy_speedup"] == 1.0
    assert comparison["hard_latency_gate_applied"] is False

    benchmark[0]["speedup_ratios"]["triangular_vs_lazy"] = 2.0
    with pytest.raises(ValueError, match="triangular_vs_lazy mismatch"):
        _verify_fused_benchmark(
            environment,
            benchmark,
            systems=TRIANGULAR_SYSTEMS,
            speedup_pairs=TRIANGULAR_SPEEDUP_PAIRS,
            minimum_speedup_name="triangular_vs_lazy",
        )


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


def test_v3_verifier_reconstructs_ephemeral_triangular_runtime(
    tmp_path: Path,
):
    root = tmp_path / "artifacts"
    shutil.copytree(ARTIFACTS, root)
    report_path = root / "fused_executor_report.json"
    report = json.loads(report_path.read_text())
    environment, benchmark = _five_system_benchmark(report)

    teacher, _ = load_checkpoint(root / "checkpoint.pt")
    teacher.eval()
    lazy_stack, _, _ = load_lazy_fused_modal_stack(
        root / "fused_modal_runtime.pt"
    )
    lazy_model = FusedToyTransformer.from_teacher(
        teacher,
        lazy_stack,
    ).eval()
    status_before = asdict(lazy_stack.instrumentation_status())
    triangular_stack = PackedTriangularFusedTwoLayerModalStack.from_lazy(
        lazy_stack
    )
    triangular_model = FusedToyTransformer.from_teacher(
        teacher,
        triangular_stack,
    ).eval()
    fisher_report = json.loads((root / "fisher_report.json").read_text())
    task = AssociativeRecallTaskConfig(
        **fisher_report["task"]["config"]
    )
    validation = build_associative_recall_splits(task).validation
    lazy_logits = associative_recall_answer_logits(lazy_model, validation)
    triangular_logits = associative_recall_answer_logits(
        triangular_model,
        validation,
    )
    triangular_comparison = _fused_comparison(
        split=validation,
        unfused_logits=lazy_logits,
        fused_logits=triangular_logits,
    )
    status_after = asdict(lazy_stack.instrumentation_status())
    gate = report["protocol"]["validation_gate"]
    runtime_path = root / "fused_modal_runtime.pt"
    runtime_sha256 = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    report["format_version"] = 3
    report["arithmetic"]["triangular_interpretation"] = (
        "packed causal-pair PyTorch reference executes only the "
        "lower-triangular position pairs; wall-clock behavior is "
        "reported in its separate benchmark"
    )
    report["triangular_runtime_benchmark"] = {
        "source_lazy_artifact": {
            "filename": runtime_path.name,
            "sha256": runtime_sha256,
            "artifact_kind": "lazy_fused_two_layer_modal_stack",
            "format_version": 2,
        },
        "runtime_contract": {
            "implementation": "packed_triangular_prefix_v1",
            "serialized_artifact": False,
            "default_backend": False,
            "weights_updated": False,
            "test_used": False,
            "validation_split": "validation_fisher",
            "benchmark_split": "validation_fisher",
            "packed_causal_pair_count": (
                triangular_stack.causal_pair_count
            ),
            "packed_fast_state_tensor_bytes": (
                triangular_stack.packed_state_bytes
            ),
        },
        "source_lazy_status_before": status_before,
        "source_lazy_status_after": status_after,
        "validation": {
            "gate": gate,
            "gate_passed": _fused_gate_passed(
                triangular_comparison,
                gate,
            ),
            "triangular_vs_lazy": triangular_comparison,
        },
        "benchmark_environment": environment,
        "benchmark": benchmark,
        "comparison": _fused_triangular_benchmark_comparison(benchmark),
    }
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    save_runtime_manifest(
        root / "runtime_manifest.json",
        manifest_from_legacy_runtime(root),
    )

    summary = verify_build(root)["fused_executor"]

    assert summary["format_version"] == 3
    assert summary["triangular_runtime_present"] is True
    assert summary["triangular_default_backend"] is False
    assert summary["triangular_serialized_artifact"] is False
    assert summary["triangular_causal_pair_count"] == 36
    assert (
        summary["triangular_fast_stack_resident_tensor_bytes"]
        == 125_120
    )
    assert summary["triangular_validation_gate_passed"] is True
    assert summary["triangular_zero_source_sidecar_loads"] is True
    assert summary["triangular_benchmark_batch_count"] == 4
