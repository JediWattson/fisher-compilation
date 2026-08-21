from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fisher_graph import gemma3_joint_cross_block_whole_model_experiment as exp


def _events() -> list[str]:
    return [
        "lexical.ingress",
        *(f"mutation.{index}" for index in range(1, 37)),
        "terminal.rmsnorm",
        "lexical.egress",
    ]


def _exact_metrics() -> dict[str, object]:
    return {
        "native": {"supervised_tokens": 2, "nll_per_token": 1.0},
        "candidate": {
            "supervised_tokens": 2,
            "compared_query_rows": 3,
            "nll_per_token": 1.0,
            "delta_nll_per_token": 0.0,
            "native_to_candidate_kl_per_query": 0.0,
            "top1_agreement_to_native": 1.0,
            "logit_nrmse": 0.0,
            "maximum_absolute_logit_error": 0.0,
        },
    }


def _candidate_metrics() -> dict[str, object]:
    return {
        "native": {"supervised_tokens": 2, "nll_per_token": 1.0},
        "candidate": {
            "supervised_tokens": 2,
            "compared_query_rows": 3,
            "nll_per_token": 1.05,
            "delta_nll_per_token": 1.05 - 1.0,
            "native_to_candidate_kl_per_query": 0.04,
            "top1_agreement_to_native": 0.95,
            "logit_nrmse": 0.1,
            "maximum_absolute_logit_error": 0.2,
        },
    }


def _operator_receipts(*, exact: bool = False) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index in range(1, 37):
        attention = index % 2 == 1
        block = (index - 1) // 2
        result.append(
            {
                "event_id": f"mutation.{index}",
                "event_kind": (
                    "attention_mutation" if attention else "mlp_mutation"
                ),
                "endpoint_kind": "compiled",
                "incoming_generator_rank": (
                    640 if exact else 256
                ) if attention else (640 if exact else 4),
                "emitted_generator_rank": (
                    1024 if exact else 64
                ) if attention else (640 if exact else 4),
                "operator_receipt_present": True,
                "generator_kind": (
                    (
                        "exact_full_rank" if exact else "reduced_coordinates"
                    )
                    if attention
                    else None
                ),
                "attention_kind": (
                    "global_causal" if block in (5, 11, 17)
                    else "sliding_causal"
                ) if attention else None,
                "logical_causal_key_pairs": 1 if attention else None,
                "source_projection_call_count": 0 if attention else None,
                "source_fallback_call_count": 0 if attention else None,
            }
        )
    return result


def _audit(*, exact: bool = False) -> dict[str, object]:
    source_names = {
        "source.lexical_ingress",
        "source.terminal_rmsnorm",
        "source.lexical_egress",
        *(f"source.block.{index}" for index in range(18)),
        *(f"source.attention.{index}" for index in range(18)),
        *(f"source.mlp.{index}" for index in range(18)),
    }
    return {
        "batch_count": 1,
        "native_source_calls": {name: 1 for name in source_names},
        "compiled_source_calls": {name: 0 for name in source_names},
        "compiled_source_call_total": 0,
        "zero_live_teacher_calls": True,
        "event_count_per_execution": 39,
        "mutation_count_per_execution": 36,
        "event_receipts": [_events()],
        "operator_receipts": [_operator_receipts(exact=exact)],
        "attention_operator_receipt_count": 18,
        "attention_source_projection_call_total": 0,
        "attention_source_fallback_call_total": 0,
        "peak_live_transport_packets": 0 if exact else 1,
        "peak_live_transport_packet_bytes": 0 if exact else 16,
    }


def _manifest(*, exact: bool) -> dict[str, object]:
    candidate_count = (
        exp.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT if exact else 35
    )
    inventory = {
        "lexical_shared_physical_scalars": 10,
        "mutation_scalars": (
            exp.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT - 11 if exact else 20
        ),
        "transport_scalars": 0 if exact else 4,
        "terminal_norm_scalars": 1,
        "candidate_deployed_scalars": candidate_count,
        "inventory_closes": True,
        "lexical_ingress_egress_factor_counted_once": True,
        "serialized_artifact_written": False,
        "source_unique_parameter_reduction_claimed": False,
        "whole_model_mac_reduction_claimed": False,
        "latency_reduction_claimed": False,
    }
    return {
        "arm_id": "exact" if exact else "candidate",
        "candidate_arm": not exact,
        "exact_control_arm": exact,
        "event_count": 39,
        "ordered_event_ids": _events(),
        "attention_mutation_count": 18,
        "mlp_mutation_count": 18,
        "direct_compiled_attention_mutation_count": 18,
        "lexical_factor_registered_once": True,
        "lexical": {
            "rank": 640 if exact else 512,
            "is_exact_copied_matrix": exact,
            "ingress_and_egress_share_the_same_storage": True,
            "physical_storage_count": 1,
            "second_head_copy_present": False,
        },
        "residual_carrier": {
            "mutation_count": 36,
            "initial_version": 0,
            "final_version": 36,
            "identity_preserved_by_ordered_delta_updates": True,
        },
        "transport_edge_count": 0 if exact else 1,
        "cross_block_transport_count": 0 if exact else 1,
        "cross_component_family_transport_count": 0 if exact else 1,
        "cross_block_cross_component_transport_count": 0 if exact else 1,
        "transports": [] if exact else [
            {
                "transport_id": "transport.1.to.4",
                "source_event_id": "mutation.1",
                "target_event_id": "mutation.4",
                "source_chart_index": 0,
                "target_chart_index": 0,
                "token_relation": "same_token",
                "source_generator_rank": 64,
                "target_generator_rank": 4,
                "stored_scalar_count": 256,
                "strictly_forward": True,
            }
        ],
        "strictly_forward_transports": True,
        "live_teacher_reference_present": False,
        "native_fallback_present": False,
        "source_checkpoint_coefficients_retained_by_mutations": exact,
        "resource_inventory": inventory,
    }


def _candidate_resources() -> dict[str, object]:
    source = exp.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
    candidate = 35
    inventory = _manifest(exact=False)["resource_inventory"]
    return {
        "source_unique_parameter_count": source,
        "candidate_deployed_scalar_count": candidate,
        "parameter_difference": source - candidate,
        "parameter_reduction_fraction": (source - candidate) / source,
        "projected_source_float32_bytes": 4 * source,
        "projected_candidate_float32_bytes": 4 * candidate,
        "projected_byte_reduction_fraction": (source - candidate) / source,
        "lexical_rank": 512,
        "attention_input_rank": 256,
        "attention_output_rank": 64,
        "mlp_generator_rank": 4,
        "executor_inventory": inventory,
        "lexical_factor_counted_once": True,
        "source_independent_in_memory_executor": True,
        "serialized_candidate_artifact_written": False,
        "serialized_byte_reduction_authoritative": False,
        "whole_model_mac_ledger_closed": False,
        "whole_model_mac_reduction_claimed": False,
        "latency_reduction_claimed": False,
    }


def _report() -> dict[str, object]:
    protocol = exp.build_layer_agnostic_whole_model_compression_protocol()
    candidate_resources = _candidate_resources()
    candidate_metrics = _candidate_metrics()
    report: dict[str, object] = {
        "schema": exp.GEMMA3_JOINT_CROSS_BLOCK_EXPERIMENT_SCHEMA,
        "format_version": exp.GEMMA3_JOINT_CROSS_BLOCK_EXPERIMENT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_executor_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "protocol": {
            "protocol_sha256": protocol["protocol_sha256"],
            "relationship": exp.PROTOCOL_RELATIONSHIP,
            "primary_arm_fully_executed": False,
        },
        "scientific_status": {
            "development_only": True,
            "family_disjoint_evaluation": False,
            "heldout_confirmation": False,
            "exact_39_event_control_passed": True,
            "tied_lexical_embedding_and_head_compiled": True,
            "all_18_attention_mutations_compiled": True,
            "all_18_mlp_mutations_compiled": True,
            "single_residual_carrier_compiled": True,
            "cross_block_cross_component_transports_compiled": True,
            "prefill_only": True,
            "cached_decode_tested": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "latency_or_kernel_speed_claimed": False,
        },
        "configuration": {
            "arm_id": exp.DEVELOPMENT_ARM_ID,
            "deterministic_seed": exp.DETERMINISTIC_SEED,
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 4,
            "lexical_rank": 512,
            "attention_input_rank": 256,
            "attention_output_rank": 64,
            "mlp_generator_rank": 4,
            "maximum_attention_sources_per_target_chart": 1,
            "transport_relative_ridge": 1.0e-4,
            "lexical_fit_kind": "deterministic_weight_svd_capacity_control",
            "attention_fit_kind": "deterministic_weight_svd_capacity_control",
            "mlp_fit_kind": (
                "fisher_weighted_activation_charts_full_native_suffix"
            ),
        },
        "model": {
            "model_id": exp.PINNED_MODEL_ID,
            "requested_revision": exp.PINNED_MODEL_REVISION,
            "resolved_commit": exp.PINNED_MODEL_REVISION,
            "parameter_count": exp.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
            "hidden_size": 640,
            "num_hidden_layers": 18,
            "device": "cpu",
            "dtype": "torch.float32",
            "weights_in_artifact": False,
        },
        "data": {
            "prompt_disjoint": True,
            "family_disjoint": False,
            "fit_example_count": 1,
            "evaluation_example_count": 1,
            "panel_sha256": "a" * 64,
            "fit_example_ids_sha256": "b" * 64,
            "evaluation_example_ids_sha256": "c" * 64,
        },
        "compilation": {
            "exact_control": {
                "exact_parameter_inventory_parity": True,
                "source_model_parameter_count": (
                    exp.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
                ),
                "source_independence_audit": {"passed": True},
            },
            "joint_candidate": {
                "exact_parameter_inventory_parity": None,
                "source_model_parameter_count": (
                    exp.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
                ),
                "source_independence_audit": {"passed": True},
            },
            "exact_control_manifest": _manifest(exact=True),
            "candidate_manifest": _manifest(exact=False),
            "fit_row_identity_sha256": "d" * 64,
        },
        "metrics": {
            "exact_control": _exact_metrics(),
            "joint_candidate": candidate_metrics,
        },
        "execution_audit": {
            "exact_control": _audit(exact=True),
            "joint_candidate": _audit(),
            "source_model_fingerprint_before": "e" * 64,
            "source_model_fingerprint_after": "e" * 64,
            "source_execution_fingerprint_before": "f" * 64,
            "source_execution_fingerprint_after": "f" * 64,
            "source_unchanged": True,
        },
        "resources": {
            "exact_control": {
                "source_unique_parameter_count": (
                    exp.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
                ),
                "candidate_deployed_scalar_count": (
                    exp.PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
                ),
                "parameter_difference": 0,
                "parameter_reduction_fraction": 0.0,
                "executor_inventory": _manifest(exact=True)[
                    "resource_inventory"
                ],
            },
            "joint_candidate": candidate_resources,
        },
    }
    report["qualification"] = exp._qualification(
        candidate_metrics,
        candidate_resources,
        exact_control_passed=True,
        cross_block_edge_count=1,
    )
    report["report_sha256"] = exp._report_sha256(report)
    return report


def test_v1_numerics_fail_before_model_or_protocol_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        exp,
        "_protocol_receipt",
        lambda *_args: pytest.fail("protocol must remain unopened"),
    )
    monkeypatch.setattr(
        exp,
        "load_gemma3",
        lambda **_kwargs: pytest.fail("model must remain unopened"),
    )
    with pytest.raises(ValueError, match="numerics are frozen"):
        exp.run_gemma3_joint_cross_block_whole_model_experiment(
            output=tmp_path / "report.json",
            attention_input_rank=128,
        )


def test_exact_control_gate_is_literal_and_precedes_candidate_fit() -> None:
    exp._assert_exact_control_gate(_exact_metrics())
    bad = _exact_metrics()
    bad["candidate"]["maximum_absolute_logit_error"] = 1.0e-8
    with pytest.raises(RuntimeError, match="literal parity"):
        exp._assert_exact_control_gate(bad)


def test_qualification_cannot_authorize_a_development_result() -> None:
    resources = _candidate_resources()
    result = exp._qualification(
        _candidate_metrics(),
        resources,
        exact_control_passed=True,
        cross_block_edge_count=1,
    )
    assert result["available_fidelity_screen_passed"] is True
    assert result["whole_model_parameter_gate_passed"] is True
    assert result["qualified"] is False
    assert result["compression_claim_authorized"] is False
    assert "family_disjoint_guard_not_run" in result["failure_reasons"]
    assert "serialized_source_free_artifact_not_written" in result[
        "failure_reasons"
    ]


def test_strict_loader_rejects_rehashed_semantic_tampering(
    tmp_path: Path,
) -> None:
    report = _report()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = exp.load_gemma3_joint_cross_block_whole_model_report(path)
    assert loaded == report

    tampered = copy.deepcopy(report)
    tampered["execution_audit"]["joint_candidate"][
        "compiled_source_call_total"
    ] = 1
    tampered["report_sha256"] = exp._report_sha256(
        {key: value for key, value in tampered.items() if key != "report_sha256"}
    )
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="source-call or event audit"):
        exp.load_gemma3_joint_cross_block_whole_model_report(path)


def test_strict_loader_rejects_rehashed_graph_and_claim_drift(
    tmp_path: Path,
) -> None:
    def set_batch_zero(value: dict[str, object]) -> None:
        audit = value["execution_audit"]["joint_candidate"]
        audit["batch_count"] = 0
        audit["event_receipts"] = []
        audit["operator_receipts"] = []

    def set_attention_kind(value: dict[str, object]) -> None:
        value["execution_audit"]["joint_candidate"]["operator_receipts"][0][0][
            "attention_kind"
        ] = "fake"

    def set_transport_relation(value: dict[str, object]) -> None:
        value["compilation"]["candidate_manifest"]["transports"][0][
            "token_relation"
        ] = "future_token"

    mutations = (
        lambda value: value.__setitem__("contains_executor_weights", True),
        set_batch_zero,
        lambda value: value["compilation"]["exact_control_manifest"].__setitem__(
            "live_teacher_reference_present", True
        ),
        lambda value: value["compilation"]["exact_control_manifest"][
            "residual_carrier"
        ].__setitem__("initial_version", 9),
        set_attention_kind,
        set_transport_relation,
        lambda value: value["resources"]["joint_candidate"].__setitem__(
            "source_independent_in_memory_executor", False
        ),
        lambda value: value["configuration"].__setitem__("lexical_rank", 384),
    )
    path = tmp_path / "tampered.json"
    for mutate in mutations:
        value = _report()
        mutate(value)
        value["report_sha256"] = exp._report_sha256(
            {key: item for key, item in value.items() if key != "report_sha256"}
        )
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError):
            exp.load_gemma3_joint_cross_block_whole_model_report(path)


class _FakeModel(nn.Module):
    def __init__(self, width: int = 1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width), requires_grad=False)


class _FakeAdapter:
    def __init__(self, model: nn.Module) -> None:
        self.module = model

    def model_fingerprint(self) -> str:
        return "a" * 64

    def execution_fingerprint(self) -> str:
        return "b" * 64


class _FakeGuard:
    def __init__(self, _model: nn.Module) -> None:
        pass

    def assert_unchanged(self) -> None:
        pass


def test_failed_exact_gate_never_starts_candidate_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _FakeModel()
    monkeypatch.setattr(exp, "PINNED_SOURCE_UNIQUE_PARAMETER_COUNT", 1)
    monkeypatch.setattr(exp, "_protocol_receipt", lambda *_args: {})
    monkeypatch.setattr(
        exp,
        "_load_prompt_split",
        lambda _path: (("fit",), ("eval",), {
            "prompt_disjoint": True,
            "family_disjoint": False,
        }),
    )
    monkeypatch.setattr(exp, "resolve_torch_device", lambda _name: torch.device("cpu"))
    monkeypatch.setattr(
        exp,
        "resolve_gemma3_huggingface_paths",
        lambda _path: {"hub_cache": tmp_path},
    )
    monkeypatch.setattr(exp, "load_gemma3", lambda **_kwargs: (object(), model))
    monkeypatch.setattr(exp, "Gemma3CausalLMAdapter", _FakeAdapter)
    monkeypatch.setattr(exp, "_FrozenModelTensorGuard", _FakeGuard)
    monkeypatch.setattr(exp, "_ordered_example_ids", lambda *_args: ("x",))
    keyed = SimpleNamespace(
        model_inputs={"input_ids": torch.ones(1, 1, dtype=torch.long)},
        row_keys=(("x", 0),),
    )
    monkeypatch.setattr(exp, "_tokenize_keyed_batches", lambda *_args, **_kwargs: (keyed,))
    fake_executor = SimpleNamespace(architecture_manifest=lambda: {})
    exact = SimpleNamespace(executor=fake_executor)
    monkeypatch.setattr(exp, "_compile_exact_control", lambda _adapter: exact)
    monkeypatch.setattr(exp, "_resource_report", lambda *_args, **_kwargs: {})
    bad = _exact_metrics()
    bad["candidate"]["delta_nll_per_token"] = 0.01
    monkeypatch.setattr(
        exp,
        "_evaluate_joint_executor",
        lambda **_kwargs: (bad, {}),
    )
    monkeypatch.setattr(
        exp,
        "_compile_joint_executors",
        lambda **_kwargs: pytest.fail("candidate fit must not start"),
    )
    with pytest.raises(RuntimeError, match="literal parity"):
        exp.run_gemma3_joint_cross_block_whole_model_experiment(
            output=tmp_path / "result.json"
        )


def test_mocked_public_runner_executes_exact_before_candidate_and_writes_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_count = 100
    model = _FakeModel(source_count)
    monkeypatch.setattr(exp, "PINNED_SOURCE_UNIQUE_PARAMETER_COUNT", source_count)
    protocol = exp.build_layer_agnostic_whole_model_compression_protocol()
    monkeypatch.setattr(
        exp,
        "_protocol_receipt",
        lambda *_args: {
            "protocol_sha256": protocol["protocol_sha256"],
            "relationship": exp.PROTOCOL_RELATIONSHIP,
            "primary_arm_fully_executed": False,
        },
    )
    split = {
        "prompt_disjoint": True,
        "family_disjoint": False,
        "fit_example_count": 1,
        "evaluation_example_count": 1,
        "panel_sha256": "a" * 64,
        "fit_example_ids_sha256": "b" * 64,
        "evaluation_example_ids_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        exp,
        "_load_prompt_split",
        lambda _path: (("fit",), ("eval",), split),
    )
    monkeypatch.setattr(exp, "resolve_torch_device", lambda _name: torch.device("cpu"))
    monkeypatch.setattr(
        exp,
        "resolve_gemma3_huggingface_paths",
        lambda _path: {"hub_cache": tmp_path},
    )
    monkeypatch.setattr(exp, "load_gemma3", lambda **_kwargs: (object(), model))
    monkeypatch.setattr(exp, "Gemma3CausalLMAdapter", _FakeAdapter)
    monkeypatch.setattr(exp, "_FrozenModelTensorGuard", _FakeGuard)
    monkeypatch.setattr(exp, "_ordered_example_ids", lambda *_args: ("x",))
    keyed = SimpleNamespace(
        model_inputs={"input_ids": torch.ones(1, 1, dtype=torch.long)},
        row_keys=(("x", 0),),
    )
    monkeypatch.setattr(
        exp,
        "_tokenize_keyed_batches",
        lambda *_args, **_kwargs: (keyed,),
    )
    monkeypatch.setattr(
        exp,
        "_model_provenance",
        lambda *_args, **_kwargs: {
            "model_id": exp.PINNED_MODEL_ID,
            "requested_revision": exp.PINNED_MODEL_REVISION,
            "resolved_commit": exp.PINNED_MODEL_REVISION,
            "parameter_count": source_count,
            "hidden_size": 640,
            "num_hidden_layers": 18,
            "device": "cpu",
            "dtype": "torch.float32",
            "weights_in_artifact": False,
        },
    )

    exact_manifest = _manifest(exact=True)
    candidate_manifest = _manifest(exact=False)
    exact_metadata = {
        "exact_parameter_inventory_parity": True,
        "source_model_parameter_count": source_count,
        "source_independence_audit": {"passed": True},
    }
    candidate_metadata = {
        "exact_parameter_inventory_parity": None,
        "source_model_parameter_count": source_count,
        "source_independence_audit": {"passed": True},
    }
    exact_executor = SimpleNamespace(
        architecture_manifest=lambda: exact_manifest
    )
    candidate_executor = SimpleNamespace(
        architecture_manifest=lambda: candidate_manifest
    )
    exact = SimpleNamespace(
        executor=exact_executor,
        metadata=lambda: exact_metadata,
    )
    candidate = SimpleNamespace(
        executor=candidate_executor,
        metadata=lambda: candidate_metadata,
    )
    monkeypatch.setattr(exp, "_compile_exact_control", lambda _adapter: exact)
    order: list[str] = []

    def evaluate(**kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        if kwargs["executor"] is exact_executor:
            order.append("exact_evaluate")
            return _exact_metrics(), _audit(exact=True)
        order.append("candidate_evaluate")
        return _candidate_metrics(), _audit()

    monkeypatch.setattr(exp, "_evaluate_joint_executor", evaluate)

    def compile_candidate(**_kwargs: object) -> object:
        order.append("candidate_compile")
        fit = SimpleNamespace(
            fit_row_identity_sha256="d" * 64,
            layer_metrics={},
        )
        return SimpleNamespace(compilation=candidate, fit=fit)

    monkeypatch.setattr(exp, "_compile_joint_executors", compile_candidate)
    exact_resources = {
        "source_unique_parameter_count": source_count,
        "candidate_deployed_scalar_count": source_count,
        "parameter_difference": 0,
        "parameter_reduction_fraction": 0.0,
        "executor_inventory": exact_manifest["resource_inventory"],
    }

    def resources(compilation: object, **_kwargs: object) -> dict[str, object]:
        return (
            exact_resources
            if compilation is exact
            else _candidate_resources()
        )

    monkeypatch.setattr(exp, "_resource_report", resources)
    output = tmp_path / "joint.json"
    report = exp.run_gemma3_joint_cross_block_whole_model_experiment(
        output=output
    )

    assert order == ["exact_evaluate", "candidate_compile", "candidate_evaluate"]
    assert output.exists()
    assert report["scientific_status"][
        "all_18_attention_mutations_compiled"
    ] is True
    assert exp.load_gemma3_joint_cross_block_whole_model_report(output) == report


def test_main_dispatches_the_model_facing_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        exp,
        "run_gemma3_joint_cross_block_whole_model_experiment",
        lambda **kwargs: observed.update(kwargs) or {},
    )
    output = tmp_path / "main.json"
    assert exp.main(["--output", str(output)]) == 0
    assert observed["output"] == output
    assert observed["attention_input_rank"] == 256
    assert observed["attention_output_rank"] == 64
