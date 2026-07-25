from pathlib import Path

from fisher_graph.verify import verify_build


def test_verify_build_reports_each_modal_layer_and_keeps_layer_zero_aliases():
    artifact_dir = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "associative_recall"
    )

    summary = verify_build(artifact_dir)

    assert summary["modal_executor"] == summary["modal_executors"]["0"]
    assert summary["modal_completion"] == summary["modal_completions"]["0"]
    assert summary["modal_executors"]["1"]["layer_index"] == 1
    assert summary["modal_executors"]["1"]["status"] == "verified"
    assert summary["modal_completions"]["1"]["layer_index"] == 1
    assert summary["modal_completions"]["1"]["status"] == "verified"

    composition = summary["modal_composition"]
    assert composition["present"] is True
    assert composition["layer_count"] == 2
    assert composition["behavior_system_count"] == 9
    assert composition["validation_gate_passed"] is True
    assert composition["compiled_estimated_multiplies"] == 72_384
    assert composition["compiled_multiply_ratio"] < 0.52
    assert composition["validation_shifted_same_input_kl"] < 0.01
    assert composition["test_accuracy"] == 1.0
    assert composition["test_paired_accuracy"] == 1.0
    assert composition["status"] == "verified"

    fused = summary["fused_executor"]
    assert fused["present"] is True
    assert fused["cross_layer_bypass"] is True
    assert fused["parameter_count"] == 0
    assert fused["contains_transformer_block"] is False
    assert fused["format_version"] == 3
    assert fused["triangular_runtime_present"] is True
    assert fused["triangular_causal_pair_count"] == 36
    assert (
        fused["triangular_fast_stack_resident_tensor_bytes"]
        == 125_120
    )
    assert fused["triangular_validation_gate_passed"] is True
    assert fused["triangular_zero_source_sidecar_loads"] is True
    assert fused["lazy_runtime"] is True
    assert fused["fast_state_tensor_count"] == 7
    assert fused["fast_stack_resident_tensor_bytes"] == 199_808
    assert fused["sidecar_resident_tensor_bytes"] == 203_648
    assert (
        fused["default_full_runtime_resident_tensor_bytes"]
        == 205_952
    )
    assert (
        fused["loaded_full_runtime_resident_tensor_bytes"]
        == 409_600
    )
    assert fused["zero_sidecar_loads_during_fast_evaluation"] is True
    assert fused["sidecar_loaded_exactly_once"] is True
    assert fused["instrumentation_cache_reused"] is True
    assert fused["instrumentation_evicted"] is True
    assert fused["validation_gate_passed"] is True
    assert fused["dense_executed_multiplies"] == 49_152
    assert fused["triangular_nonzero_multiplies"] == 30_336
    assert fused["dense_multiply_ratio"] < 0.36
    assert fused["benchmark_batch_count"] == 4
    assert fused["minimum_observed_fused_vs_unfused_speedup"] > 0
    assert fused["geometric_mean_lazy_to_monolithic_latency_ratio"] > 0
    assert fused["maximum_lazy_to_monolithic_latency_ratio"] > 0
    assert fused["benchmark_hard_latency_gate_applied"] is False
    assert fused["test_accuracy"] == 1.0
    assert fused["test_paired_accuracy"] == 1.0
    assert fused["test_maximum_answer_logit_difference"] < 5e-4
    assert fused["status"] == "verified"

    runtime_manifest = summary["runtime_manifest"]
    assert runtime_manifest["present"] is True
    assert runtime_manifest["schema_version"] == 1
    assert runtime_manifest["segment_count"] == 1
    assert runtime_manifest["resource_count"] == 9
    assert runtime_manifest["sequence_policy"] == "fixed"
    assert runtime_manifest["sequence_length"] == 8
    assert runtime_manifest["fallback_segment_count"] == 1
    assert runtime_manifest["status"] == "verified"
