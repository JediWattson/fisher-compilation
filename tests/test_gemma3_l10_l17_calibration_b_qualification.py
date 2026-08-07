from __future__ import annotations

import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_l10_l17_calibration_b_qualification as qualification
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_v8_heldout_authority import Gemma3V8HeldoutManifest


def _digest(index: int) -> str:
    return f"{index:064x}"


def _manifest() -> Gemma3V8HeldoutManifest:
    return Gemma3V8HeldoutManifest(
        role="calibration_b",
        prompt_sha256s=tuple(_digest(index + 1) for index in range(96)),
        normalized_prompt_sha256s=tuple(
            _digest(index + 1001) for index in range(96)
        ),
        family_ids=tuple(f"family.{index % 8}" for index in range(96)),
        length_bands=tuple(
            ("micro", "compact", "medium", "long")[index % 4]
            for index in range(96)
        ),
        approximate_word_counts=tuple(10 + index % 7 for index in range(96)),
        declared_policy="one_shot_frozen_candidate_selection",
    )


def _bundle_binding() -> dict[str, object]:
    return {
        "bundle_file_sha256": "1" * 64,
        "composition_payload_sha256": "2" * 64,
        "combined_dynamic_graph_sha256": "3" * 64,
        "combined_edgeless_graph_sha256": "4" * 64,
        "model_fingerprint": "5" * 64,
        "parameter_cluster_plan_sha256": "6" * 64,
        "model_id": "google/gemma-3-270m",
        "requested_revision": "revision",
        "parents": (
            {"role": "layer10", "lineage": ("a", "b")},
            {"role": "layer17", "lineage": ("c", "d")},
        ),
        "resources": dict(qualification._EXPECTED_RESOURCES),
    }


def _metric(native: float, delta: float, kl: float, top1: float) -> dict[str, float]:
    return {
        "nll_per_token": native + delta,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _assessment() -> dict[str, object]:
    native = 2.0
    conditions = {
        "layer10_dynamic": _metric(native, 0.02, 0.03, 0.90),
        "layer17_dynamic": _metric(native, 0.03, 0.04, 0.89),
        "composed_edgeless": _metric(native, 0.06, 0.06, 0.85),
        "composed_dynamic": _metric(native, 0.05, 0.05, 0.86),
        "matched_deletion": _metric(native, 0.30, 0.30, 0.55),
    }
    families = {
        f"family_{index:02d}": {
            "supervised_tokens": 10,
            "native": {"nll_per_token": native},
            "conditions": copy.deepcopy(conditions),
        }
        for index in range(8)
    }
    macro = qualification._average_condition_metrics(families)
    source_parameters = 268_000_000
    dynamic_candidate_parameters = source_parameters - 817_658
    resource_record = {
        "replacement_scope": "synthetic_scope",
        "replaced_layer_count": 2,
        "graph_node_count": 8,
        "fragment_count": 8,
        "removed_mode_count": 564,
        "source_whole_model_learned_parameters": source_parameters,
        "candidate_whole_model_learned_parameters": (
            dynamic_candidate_parameters
        ),
        "native_removed_learned_parameters": 1_082_880,
        "modal_graph_learned_parameters": 265_222,
        "net_stored_parameter_savings": 817_658,
        "graph_runtime_storage": "synthetic_runtime",
        "logical_linear_macs_native_removed": 1_082_880 * 120,
        "logical_modal_graph_macs": 259_648 * 120,
        "logical_executed_modal_graph_macs": 253_248 * 120,
        "logical_modal_graph_additions": 100 * 120,
        "logical_executed_modal_graph_additions": 90 * 120,
        "net_logical_macs_saved": (1_082_880 - 253_248) * 120,
        "executed_peak_live_modal_width": 64,
    }
    edgeless_resource = dict(resource_record)
    edgeless_resource.update(
        {
            "candidate_whole_model_learned_parameters": (
                dynamic_candidate_parameters - 42
            ),
            "modal_graph_learned_parameters": 265_180,
            "net_stored_parameter_savings": 817_700,
        }
    )
    resources = {
        name: dict(resource_record) for name in qualification._CONDITIONS
    }
    resources["composed_edgeless"] = edgeless_resource
    return {
        "execution_path": "combined_modal_generator_graph_executor",
        "assessment_role": "claimed_closed_expansion_qualification",
        "heldout_confirmation": False,
        "example_count": 96,
        "family_count": 8,
        "supervised_tokens": 80,
        "logical_valid_tokens": 120,
        "native": {"nll_per_token": native},
        "conditions": conditions,
        "equal_family_macro": macro,
        "families": families,
        "graph_comparison": {
            "node_count": 8,
            "interacting_edge_count": 6,
            "edgeless_edge_count": 0,
            "node_artifacts_identical": True,
            "deletion_paths_agree": True,
            "deletion_equivalence_atol": 0.0,
            "deletion_equivalence_rtol": 0.0,
            "deletion_max_abs_logit_difference": 0.0,
            "interaction_parameter_delta": 42,
            "layer10_gain": 0.25,
            "layer17_gain": 0.5,
        },
        "resource_accounting": resources,
        "observed_resources": dict(qualification._EXPECTED_RESOURCES),
        "latency_or_kernel_speed_claim": False,
    }


def _result() -> dict[str, object]:
    assessment = _assessment()
    decision = qualification.qualification_decision(assessment)
    payload: dict[str, object] = {
        "schema": qualification._RESULT_SCHEMA,
        "format_version": 1,
        "scientific_role": "claimed_closed_expansion_qualification",
        "heldout_confirmation": False,
        "protocol_sha256": "7" * 64,
        "bundle": {
            "bundle_file_sha256": "1" * 64,
            "composition_payload_sha256": "2" * 64,
            "combined_dynamic_graph_sha256": "3" * 64,
            "combined_edgeless_graph_sha256": "4" * 64,
            "model_fingerprint": "5" * 64,
            "parameter_cluster_plan_sha256": "6" * 64,
        },
        "manifest_sha256": "8" * 64,
        "claim": {
            "role": "calibration_b",
            "protocol_sha256": "7" * 64,
            "manifest_sha256": "8" * 64,
            "challenger_receipt_sha256": "2" * 64,
            "claim_sha256": "9" * 64,
            "claim_file_sha256": "a" * 64,
            "state": "claimed_before_aggregate_prompt_source_open",
        },
        "export": {
            "export_sha256": "b" * 64,
            "export_file_sha256": "c" * 64,
        },
        "thresholds": dict(qualification._THRESHOLDS),
        "expected_resources": dict(qualification._EXPECTED_RESOURCES),
        "assessment": assessment,
        "decision": decision,
        "candidate_changed": False,
        "bundle_file_sha256_after": "1" * 64,
        "calibration_b_claimed": True,
        "calibration_b_evaluated": True,
        "validation_opened": False,
        "test_opened": False,
        "safety": dict(qualification._SAFETY),
    }
    return {
        **payload,
        "result_sha256": qualification._domain_sha256(
            qualification._RESULT_DOMAIN,
            payload,
        ),
    }


def test_protocol_json_roundtrip_preserves_tuple_bound_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    binding = _bundle_binding()
    monkeypatch.setattr(
        qualification,
        "_bundle_binding",
        lambda path: ({}, binding),
    )
    monkeypatch.setattr(
        qualification,
        "load_gemma3_v8_heldout_manifest",
        lambda role, **kwargs: manifest,
    )
    monkeypatch.setattr(
        qualification,
        "_tokenizer_contract",
        lambda: {
            "max_length": 256,
            "local_files_only": True,
            "tokenization_batch_size": 1,
        },
    )
    source_hashes = {
        name: "d" * 64
        for name in (
            "fisher_graph.gemma3_l10_l17_calibration_b_qualification",
            "fisher_graph.gemma3_modal_graph_composition_bundle",
            "fisher_graph.gemma3_v8_heldout_authority",
            "fisher_graph.gemma3_modal_generator_graph_executor",
            "fisher_graph.modal_graph_rung_evaluation",
            "fisher_graph.gemma3_state_conditioned_shape_flow_experiment",
        )
    }
    monkeypatch.setattr(qualification, "_source_hashes", lambda: source_hashes)
    protocol_path = tmp_path / "protocol.json"

    qualification.freeze_gemma3_l10_l17_calibration_b_protocol(
        bundle_path=tmp_path / "bundle.pt",
        audit_path=tmp_path / "audit.json",
        family_path=tmp_path / "families.json",
        output=protocol_path,
    )
    loaded = qualification.load_gemma3_l10_l17_calibration_b_protocol(
        protocol_path
    )

    assert qualification._canonical_equal(loaded["bundle"], binding)
    assert qualification._canonical_equal(loaded["manifest"], manifest.metadata())


def test_claim_export_returns_only_source_safe_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    binding = _bundle_binding()
    protocol = qualification._protocol_payload(
        bundle_binding=binding,
        manifest=manifest,
    )
    protocol["protocol_sha256"] = qualification._domain_sha256(
        qualification._PROTOCOL_DOMAIN,
        protocol,
    )
    monkeypatch.setattr(
        qualification,
        "load_gemma3_l10_l17_calibration_b_protocol",
        lambda path: json.loads(json.dumps(protocol)),
    )
    monkeypatch.setattr(
        qualification,
        "_bundle_binding",
        lambda path: ({}, binding),
    )
    monkeypatch.setattr(
        qualification,
        "load_gemma3_v8_heldout_manifest",
        lambda role, **kwargs: manifest,
    )

    class Claim:
        def metadata(self) -> dict[str, object]:
            return {
                "role": "calibration_b",
                "claim_sha256": "e" * 64,
            }

    monkeypatch.setattr(
        qualification,
        "claim_gemma3_v8_heldout_role",
        lambda *args, **kwargs: Claim(),
    )
    output = tmp_path / "claimed.json"

    def fake_export(*args: object, **kwargs: object) -> dict[str, object]:
        output.write_text('{"prompt":"secret"}\n', encoding="utf-8")
        return {"export_sha256": "f" * 64, "examples": ["secret"]}

    monkeypatch.setattr(
        qualification,
        "export_claimed_gemma3_v8_role",
        fake_export,
    )
    receipt = qualification.claim_export_gemma3_l10_l17_calibration_b(
        output=output,
    )

    assert receipt["receipt_contains_prompt_text"] is False
    assert "examples" not in receipt
    assert "secret" not in json.dumps(receipt)


def test_claim_export_rejects_output_collision_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-present.json"
    output.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(
        qualification,
        "claim_gemma3_v8_heldout_role",
        lambda *args, **kwargs: pytest.fail("claim must not run"),
    )

    with pytest.raises(FileExistsError, match="preexisting"):
        qualification.claim_export_gemma3_l10_l17_calibration_b(output=output)


def test_claim_export_rejects_source_drift_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _bundle_binding()
    tokenizer_contract = {
        "max_length": 256,
        "local_files_only": True,
        "tokenization_batch_size": 1,
    }
    protocol = {
        "bundle": json.loads(json.dumps(binding)),
        "runtime": {
            "source_sha256s": {"fisher_graph.tree": "1" * 64},
            "tokenizer_contract": tokenizer_contract,
        },
    }
    monkeypatch.setattr(
        qualification,
        "load_gemma3_l10_l17_calibration_b_protocol",
        lambda path: protocol,
    )
    monkeypatch.setattr(
        qualification,
        "_bundle_binding",
        lambda path: ({}, binding),
    )
    monkeypatch.setattr(
        qualification,
        "_source_hashes",
        lambda: {"fisher_graph.tree": "2" * 64},
    )
    monkeypatch.setattr(
        qualification,
        "_tokenizer_contract",
        lambda: tokenizer_contract,
    )
    monkeypatch.setattr(
        qualification,
        "claim_gemma3_v8_heldout_role",
        lambda *args, **kwargs: pytest.fail("claim must not run"),
    )

    with pytest.raises(ValueError, match="runtime changed"):
        qualification.claim_export_gemma3_l10_l17_calibration_b(
            output=tmp_path / "claimed.json",
        )


def test_qualification_decision_passes_all_declared_gates() -> None:
    decision = qualification.qualification_decision(_assessment())

    assert decision["qualification_passed"] is True
    assert all(decision["checks"].values())
    assert decision["derived_metrics"]["passing_family_count"] == 8


def test_qualification_decision_rejects_nonreproducible_macro() -> None:
    assessment = _assessment()
    assessment["equal_family_macro"]["conditions"]["composed_dynamic"][
        "top1_agreement_to_native"
    ] = 0.99

    with pytest.raises(ValueError, match="macro"):
        qualification.qualification_decision(assessment)


def test_result_validator_recomputes_decision_and_rejects_prompt_fields() -> None:
    result = _result()
    validated = qualification.validate_gemma3_l10_l17_calibration_b_result(
        result
    )
    assert validated["decision"]["qualification_passed"] is True

    tampered = copy.deepcopy(result)
    tampered["decision"]["qualification_passed"] = False
    payload = {key: value for key, value in tampered.items() if key != "result_sha256"}
    tampered["result_sha256"] = qualification._domain_sha256(
        qualification._RESULT_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="reproducible"):
        qualification.validate_gemma3_l10_l17_calibration_b_result(tampered)

    leaked = copy.deepcopy(result)
    leaked["assessment"]["resource_accounting"]["composed_dynamic"][
        "prompt"
    ] = "heldout text"
    payload = {key: value for key, value in leaked.items() if key != "result_sha256"}
    leaked["result_sha256"] = qualification._domain_sha256(
        qualification._RESULT_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="resources|forbidden field"):
        qualification.validate_gemma3_l10_l17_calibration_b_result(leaked)


class _SpyNativeModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.forward_count = 0

    def __call__(self, **inputs: object) -> SimpleNamespace:
        assert inputs["use_cache"] is False
        assert inputs["return_dict"] is True
        self.forward_count += 1
        return SimpleNamespace(logits=self.logits.clone())


class _SpyGraphExecutor:
    def __init__(
        self,
        *,
        name: str,
        plan: SimpleNamespace,
        generated_logits: torch.Tensor,
        deletion_logits: torch.Tensor,
        replaced_layer_count: int,
        removed_mode_count: int,
        native_removed_parameters: int,
        graph_parameters: int,
        executed_macs_per_token: int,
    ) -> None:
        self.name = name
        self.graph_plan = plan
        self.generated_logits = generated_logits
        self.deletion_logits = deletion_logits
        self.replaced_layer_count = replaced_layer_count
        self.removed_mode_count = removed_mode_count
        self.native_removed_parameters = native_removed_parameters
        self.graph_parameters = graph_parameters
        self.executed_macs_per_token = executed_macs_per_token
        self.generated_forward_count = 0
        self.deletion_forward_count = 0
        self.transaction_entry_count = 0
        self.transaction_exit_count = 0
        self._transaction_active = False

    @contextmanager
    def validated_transaction(self):
        assert self._transaction_active is False
        self._transaction_active = True
        self.transaction_entry_count += 1
        try:
            yield
        finally:
            self.transaction_exit_count += 1
            self._transaction_active = False

    def run(
        self,
        model_inputs: object,
        *,
        condition: str,
    ) -> SimpleNamespace:
        assert self._transaction_active is True
        assert isinstance(model_inputs, dict)
        valid_tokens = int(model_inputs["attention_mask"].sum().item())
        generated = condition == "generated"
        if generated:
            self.generated_forward_count += 1
        else:
            assert condition == "deletion"
            self.deletion_forward_count += 1

        source_parameters = 268_098_176
        net_savings = self.native_removed_parameters - self.graph_parameters
        executed_macs = (
            self.executed_macs_per_token * valid_tokens if generated else 0
        )
        removed_macs = self.native_removed_parameters * valid_tokens
        return SimpleNamespace(
            model_output=SimpleNamespace(
                logits=(
                    self.generated_logits if generated else self.deletion_logits
                ).clone(),
            ),
            graph_execution=SimpleNamespace(
                traversal_order=(self.graph_plan.traversal_order if generated else ()),
            ),
            condition=condition,
            replacement_scope="partial_native_mlp_mode_replacement",
            replaced_layer_count=self.replaced_layer_count,
            graph_node_count=len(self.graph_plan.nodes),
            fragment_count=len(self.graph_plan.nodes),
            removed_mode_count=self.removed_mode_count,
            source_whole_model_learned_parameters=source_parameters,
            candidate_whole_model_learned_parameters=(
                source_parameters - net_savings
            ),
            native_removed_learned_parameters=self.native_removed_parameters,
            modal_graph_learned_parameters=self.graph_parameters,
            net_stored_parameter_savings=net_savings,
            graph_runtime_storage=(
                "registered_copied_device_local_graph_parameters"
            ),
            valid_tokens=valid_tokens,
            logical_linear_macs_native_removed=removed_macs,
            logical_modal_graph_macs=(
                self.executed_macs_per_token * valid_tokens
            ),
            logical_executed_modal_graph_macs=executed_macs,
            logical_modal_graph_additions=valid_tokens,
            logical_executed_modal_graph_additions=(
                valid_tokens if generated else 0
            ),
            net_logical_macs_saved=removed_macs - executed_macs,
            peak_live_modal_width=32 if generated else 0,
        )


def _spy_plan(
    name: str,
    *,
    node_count: int,
    edge_count: int,
    node_hashes: tuple[str, ...] | None = None,
) -> SimpleNamespace:
    hashes = node_hashes or tuple(
        qualification._sha256_bytes(f"{name}.node.{index}".encode())
        for index in range(node_count)
    )
    nodes = tuple(SimpleNamespace(artifact_sha256=value) for value in hashes)
    return SimpleNamespace(
        nodes=nodes,
        interactions=tuple(object() for _ in range(edge_count)),
        traversal_order=tuple(f"{name}.node.{index}" for index in range(node_count)),
    )


def _one_token_batch(family_index: int) -> CalibrationBatch:
    return CalibrationBatch(
        model_inputs={
            "input_ids": torch.tensor([[family_index + 1, 99]]),
            "attention_mask": torch.tensor([[True, True]]),
        },
        targets=torch.tensor([[0, -100]]),
        valid_positions=torch.tensor([[True, True]]),
        example_ids=(f"synthetic.family.{family_index:02d}",),
    )


def test_synthetic_scorer_runs_exact_seven_forward_transaction() -> None:
    def logits(target_logit: float) -> torch.Tensor:
        value = torch.zeros(1, 2, 3, dtype=torch.float32)
        value[..., 0] = target_logit
        return value

    native_model = _SpyNativeModel(logits(3.0))
    adapter = SimpleNamespace(module=native_model)
    combined_hashes = tuple(
        qualification._sha256_bytes(f"combined.node.{index}".encode())
        for index in range(8)
    )
    layer10_plan = _spy_plan("layer10", node_count=4, edge_count=3)
    layer17_plan = _spy_plan("layer17", node_count=4, edge_count=3)
    edgeless_plan = _spy_plan(
        "composed",
        node_count=8,
        edge_count=0,
        node_hashes=combined_hashes,
    )
    dynamic_plan = _spy_plan(
        "composed",
        node_count=8,
        edge_count=6,
        node_hashes=combined_hashes,
    )
    deletion_logits = logits(0.0)
    layer10 = _SpyGraphExecutor(
        name="layer10",
        plan=layer10_plan,
        generated_logits=logits(2.90),
        deletion_logits=deletion_logits,
        replaced_layer_count=1,
        removed_mode_count=334,
        native_removed_parameters=641_280,
        graph_parameters=132_035,
        executed_macs_per_token=125_000,
    )
    layer17 = _SpyGraphExecutor(
        name="layer17",
        plan=layer17_plan,
        generated_logits=logits(2.85),
        deletion_logits=deletion_logits,
        replaced_layer_count=1,
        removed_mode_count=230,
        native_removed_parameters=441_600,
        graph_parameters=133_187,
        executed_macs_per_token=128_248,
    )
    edgeless = _SpyGraphExecutor(
        name="composed_edgeless",
        plan=edgeless_plan,
        generated_logits=logits(2.70),
        deletion_logits=deletion_logits,
        replaced_layer_count=2,
        removed_mode_count=564,
        native_removed_parameters=1_082_880,
        graph_parameters=255_232,
        executed_macs_per_token=249_856,
    )
    dynamic = _SpyGraphExecutor(
        name="composed_dynamic",
        plan=dynamic_plan,
        generated_logits=logits(2.70),
        deletion_logits=deletion_logits,
        replaced_layer_count=2,
        removed_mode_count=564,
        native_removed_parameters=1_082_880,
        graph_parameters=265_222,
        executed_macs_per_token=253_248,
    )
    family_batches = tuple(
        (f"family_{index:02d}", (_one_token_batch(index),))
        for index in range(8)
    )

    assessment = qualification._score_claimed_calibration_b(
        adapter=adapter,
        parent_executors={"layer10": layer10, "layer17": layer17},
        parent_plans={"layer10": layer10_plan, "layer17": layer17_plan},
        parent_gains={"layer10": 0.25, "layer17": 0.5},
        edgeless_executor=edgeless,
        dynamic_executor=dynamic,
        family_batches=family_batches,
    )

    executors = (layer10, layer17, edgeless, dynamic)
    assert native_model.forward_count == 8
    assert sum(value.generated_forward_count for value in executors) == 32
    assert sum(value.deletion_forward_count for value in executors) == 16
    assert sum(value.transaction_entry_count for value in executors) == 4
    assert sum(value.transaction_exit_count for value in executors) == 4
    assert all(value._transaction_active is False for value in executors)
    assert assessment["graph_comparison"]["deletion_paths_agree"] is True
    assert assessment["graph_comparison"][
        "deletion_max_abs_logit_difference"
    ] == 0.0
    assert assessment["observed_resources"] == qualification._EXPECTED_RESOURCES
    assert assessment["resource_accounting"]["composed_dynamic"][
        "logical_executed_modal_graph_macs"
    ] == 16 * 253_248
