from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_layer17_open_a_capacity_evaluation as capacity
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_l3_l4_progressive_a_corpus import (
    build_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


class _SpyNativeModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.forward_count = 0

    def __call__(self, **inputs: object) -> SimpleNamespace:
        assert inputs["use_cache"] is False
        assert inputs["return_dict"] is True
        self.forward_count += 1
        return SimpleNamespace(logits=self.logits.clone())


class _SpyExecutor:
    def __init__(
        self,
        *,
        plan: SimpleNamespace,
        generated_logits: torch.Tensor,
        deletion_logits: torch.Tensor,
    ) -> None:
        self.graph_plan = plan
        self.generated_logits = generated_logits
        self.deletion_logits = deletion_logits
        self.generated_forward_count = 0
        self.deletion_forward_count = 0
        self.transaction_entries = 0
        self.transaction_exits = 0
        self._active = False

    @contextmanager
    def validated_transaction(self):
        assert self._active is False
        self._active = True
        self.transaction_entries += 1
        try:
            yield
        finally:
            self.transaction_exits += 1
            self._active = False

    def run(self, model_inputs: object, *, condition: str) -> SimpleNamespace:
        assert self._active is True
        assert isinstance(model_inputs, dict)
        valid = int(model_inputs["attention_mask"].sum().item())
        generated = condition == "generated"
        if generated:
            self.generated_forward_count += 1
        else:
            assert condition == "deletion"
            self.deletion_forward_count += 1
        removed = 441_600
        graph_parameters = self.graph_plan.parameter_count
        graph_macs = self.graph_plan.macs_per_token
        graph_additions = self.graph_plan.accounting.elementwise_additions_per_token
        source_parameters = 268_098_176
        return SimpleNamespace(
            model_output=SimpleNamespace(
                logits=(
                    self.generated_logits if generated else self.deletion_logits
                ).clone()
            ),
            graph_execution=SimpleNamespace(
                traversal_order=(
                    self.graph_plan.traversal_order if generated else ()
                )
            ),
            condition=condition,
            replacement_scope="partial_native_mlp_mode_replacement",
            replaced_layer_count=1,
            graph_node_count=4,
            fragment_count=4,
            removed_mode_count=230,
            source_whole_model_learned_parameters=source_parameters,
            candidate_whole_model_learned_parameters=(
                source_parameters - removed + graph_parameters
            ),
            native_removed_learned_parameters=removed,
            modal_graph_learned_parameters=graph_parameters,
            net_stored_parameter_savings=removed - graph_parameters,
            graph_runtime_storage="registered_copied_device_local_graph_parameters",
            valid_tokens=valid,
            logical_linear_macs_native_removed=removed * valid,
            logical_modal_graph_macs=graph_macs * valid,
            logical_executed_modal_graph_macs=(graph_macs * valid if generated else 0),
            logical_modal_graph_additions=graph_additions * valid,
            logical_executed_modal_graph_additions=(
                graph_additions * valid if generated else 0
            ),
            net_logical_macs_saved=(
                removed - (graph_macs if generated else 0)
            )
            * valid,
            peak_live_modal_width=(32 if generated else 0),
        )


def _plan(*, rank: int) -> SimpleNamespace:
    nodes = tuple(
        SimpleNamespace(
            name=f"node.{index}",
            causal_order=index,
            input_boundary="layer.17.in",
            output_boundary="layer.17.out",
        )
        for index in range(4)
    )
    return SimpleNamespace(
        nodes=nodes,
        interactions=(),
        traversal_order=tuple(node.name for node in nodes),
        parameter_count=100 + rank,
        macs_per_token=80 + rank,
        accounting=SimpleNamespace(elementwise_additions_per_token=4 + rank // 16),
    )


def _family_batch(family_index: int) -> CalibrationBatch:
    examples = 32
    return CalibrationBatch(
        model_inputs={
            "input_ids": torch.full((examples, 2), family_index + 1),
            "attention_mask": torch.ones(examples, 2, dtype=torch.bool),
        },
        targets=torch.tensor([[0, -100]]).repeat(examples, 1),
        valid_positions=torch.ones(examples, 2, dtype=torch.bool),
        example_ids=tuple(
            f"synthetic.family.{family_index:02d}.{index:02d}"
            for index in range(examples)
        ),
    )


def test_synthetic_capacity_scorer_runs_paired_edgeless_and_deletion_controls() -> None:
    def logits(target_logit: float) -> torch.Tensor:
        value = torch.zeros(32, 2, 3, dtype=torch.float32)
        value[..., 0] = target_logit
        return value

    native = _SpyNativeModel(logits(3.0))
    rank16 = _SpyExecutor(
        plan=_plan(rank=16),
        generated_logits=logits(2.80),
        deletion_logits=logits(0.0),
    )
    rank32 = _SpyExecutor(
        plan=_plan(rank=32),
        generated_logits=logits(2.90),
        deletion_logits=logits(0.0),
    )
    families = tuple(
        (f"family_{index:02d}", (_family_batch(index),)) for index in range(4)
    )

    result = capacity._score_capacity_panel(
        adapter=SimpleNamespace(module=native),
        rank16_executor=rank16,
        rank32_executor=rank32,
        family_batches=families,
    )

    assert native.forward_count == 4
    assert rank16.generated_forward_count == rank32.generated_forward_count == 4
    assert rank16.deletion_forward_count == rank32.deletion_forward_count == 4
    assert rank16.transaction_entries == rank32.transaction_entries == 1
    assert rank16.transaction_exits == rank32.transaction_exits == 1
    assert result["supervised_tokens"] == 128
    assert result["logical_valid_tokens"] == 256
    assert result["graph_comparison"]["deletion_paths_agree"] is True
    assert result["graph_comparison"]["deletion_max_abs_logit_difference"] == 0.0
    assert result["resource_accounting"]["rank32_edgeless"][
        "graph_macs_per_token"
    ] == 112
    assert result["capacity_delta"]["rank32_added_graph_parameters"] == 16


def test_synthetic_capacity_scorer_supports_named_capped_arms() -> None:
    def logits(target_logit: float) -> torch.Tensor:
        value = torch.zeros(32, 2, 3, dtype=torch.float32)
        value[..., 0] = target_logit
        return value

    baseline = _SpyExecutor(
        plan=_plan(rank=48),
        generated_logits=logits(2.80),
        deletion_logits=logits(0.0),
    )
    challenger = _SpyExecutor(
        plan=_plan(rank=64),
        generated_logits=logits(2.90),
        deletion_logits=logits(0.0),
    )
    result = capacity._score_capacity_panel(
        adapter=SimpleNamespace(module=_SpyNativeModel(logits(3.0))),
        rank16_executor=baseline,
        rank32_executor=challenger,
        family_batches=tuple(
            (f"family_{index:02d}", (_family_batch(index),))
            for index in range(4)
        ),
        baseline_label="cap48",
        challenger_label="cap64",
    )

    assert set(result["conditions"]) == {
        "cap48_edgeless",
        "cap64_edgeless",
        "matched_deletion",
    }
    assert set(result["resource_accounting"]) == set(result["conditions"])
    assert result["capacity_delta"]["cap64_added_graph_parameters"] == 16


def _authority(
    *,
    label: str,
    rank: int,
    topology: tuple[tuple[object, ...], ...] | None = None,
) -> capacity._CandidateAuthority:
    scope = {
        "layer_ordinal": 17,
        "fragment_ids": ("a", "b", "c", "d"),
        "execution_order_sha256s": tuple(_digest(index + 1) for index in range(4)),
        "removed_mode_indices": (1, 2, 3),
        "source_fragment_plan_sha256": _digest(10),
        "source_model_sha256": _digest(11),
    }
    removal_hash = capacity._domain_sha256(capacity._REMOVAL_SCOPE_DOMAIN, scope)
    return capacity._CandidateAuthority(
        label=label,
        path=Path(f"{label}.pt"),
        file_sha256=_digest(rank),
        binding={
            "candidate_role": label,
            "model_id": "google/gemma-3-270m",
            "requested_revision": "revision",
            "model_fingerprint": _digest(12),
            "parameter_cluster_plan_sha256": _digest(13),
            "mode_rank": 32,
            "generator_rank": rank,
            "node_count": 4,
            "interaction_count": 0,
            "removal_scope_sha256": removal_hash,
            "graph_parameters": 100 + rank,
            "graph_macs_per_token": 80 + rank,
        },
        topology=topology
        or tuple((f"node.{index}", index, "in", "out") for index in range(4)),
        removal_scope=scope,
        lowerings=(),
        edgeless_graph=SimpleNamespace(),
    )


def test_candidate_pair_rejects_topology_drift() -> None:
    rank16 = _authority(label="rank16", rank=16)
    changed = list(rank16.topology)
    changed[-1] = ("different.node", 3, "in", "out")
    rank32 = _authority(label="rank32", rank=32, topology=tuple(changed))
    with pytest.raises(ValueError, match="topology"):
        capacity._validate_candidate_pair(rank16, rank32)


def _fixed_capacity_authorities(
    *,
    lofo_report_file_sha256: str = _digest(310),
) -> tuple[capacity._CandidateAuthority, capacity._CandidateAuthority]:
    baseline_source = _authority(label="frozen_v9", rank=16)
    baseline_file = _digest(301)
    baseline_scientific = _digest(302)
    authority_sha256 = _digest(305)
    report_sha256 = _digest(306)
    baseline = replace(
        baseline_source,
        path=Path("frozen-v9.pt"),
        file_sha256=baseline_file,
        binding={
            **baseline_source.binding,
            "candidate_kind": "capped_node_edgeless_candidate",
            "candidate_artifact_schema": (
                capacity.GEMMA3_LAYER17_CAPPED_NODE_SCHEMA
            ),
            "tensor_file": "frozen-v9.pt",
            "tensor_file_sha256": baseline_file,
            "scientific_payload_sha256": baseline_scientific,
            "mode_rank": 48,
            "mode_rank_cap": 48,
            "resolved_node_ranks": (48, 38, 48, 48),
            "generator_rank": 16,
            "graph_parameters": 148,
            "graph_macs_per_token": 128,
            "edgeless_graph_sha256": _digest(320),
        },
    )
    challenger_source = _authority(label="adaptive_a_fit", rank=16)
    challenger_file = _digest(303)
    challenger_scientific = _digest(304)
    metadata = {
        "experiment": {
            "experiment_kind": "gemma3_layer17_v8_fit_all_family_refit_v1",
            "scientific_role": (
                "calibration_a_fit_all_family_refit_candidate"
            ),
            "full_eight_family_refit_completed": True,
            "fit_family_count": 8,
            "fit_example_count": 256,
            "selection_opened": False,
            "heldout_confirmation": False,
            "assessment_metrics_present": False,
            "serving_authorized": False,
            "lofo_report_sha256": report_sha256,
            "lofo_protocol_sha256": (
                capacity.FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
            ),
            "lofo_authority_sha256": authority_sha256,
            "frozen_v9_candidate_file_sha256": baseline_file,
            "frozen_v9_candidate_scientific_sha256": baseline_scientific,
        },
        "lineage": {
            "lofo_report_file_sha256": lofo_report_file_sha256,
            "lofo_report_sha256": report_sha256,
            "lofo_protocol_sha256": (
                capacity.FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
            ),
            "lofo_all_required_gates_pass": True,
            "lofo_authorized_next_action": capacity._LOFO_PASS_NEXT_ACTION,
            "lofo_authority_sha256": authority_sha256,
            "frozen_v9_candidate_file_sha256": baseline_file,
            "frozen_v9_candidate_scientific_sha256": baseline_scientific,
        },
        "fit_receipt": {
            "family_count": 8,
            "fit_example_count": 256,
            "fit_split_sha256": _digest(307),
            "diagnostic_split_sha256": _digest(308),
            "all_coefficients_fit_on_all_normalized_rows": True,
            "diagnostic_subset_within_fit": True,
            "diagnostic_used_for_selection": False,
            "diagnostic_supports_assessment_claim": False,
        },
        "protocol": {
            "artifact_sha256": (
                capacity.FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
            ),
            "authorized_by_passing_outer_lofo": True,
        },
        "authority": {"authority_sha256": authority_sha256},
    }
    challenger = replace(
        challenger_source,
        path=Path("adaptive-a-fit.pt"),
        file_sha256=challenger_file,
        binding={
            **challenger_source.binding,
            "candidate_kind": (
                "v8_all_family_fixed_capacity_refit_candidate"
            ),
            "candidate_artifact_schema": (
                capacity.GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA
            ),
            "tensor_file": "adaptive-a-fit.pt",
            "tensor_file_sha256": challenger_file,
            "scientific_payload_sha256": challenger_scientific,
            "mode_rank": 48,
            "mode_rank_cap": 48,
            "resolved_node_ranks": (48, 38, 48, 48),
            "generator_rank": 16,
            "graph_parameters": 148,
            "graph_macs_per_token": 128,
            "edgeless_graph_sha256": _digest(321),
            "refit_provenance": {
                "lofo_report_file_sha256": lofo_report_file_sha256,
                "lofo_report_sha256": report_sha256,
                "lofo_protocol_sha256": (
                    capacity.FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
                ),
                "lofo_authority_sha256": authority_sha256,
                "frozen_v9_candidate_file_sha256": baseline_file,
                "frozen_v9_candidate_scientific_sha256": (
                    baseline_scientific
                ),
                "fit_split_sha256": _digest(307),
                "diagnostic_split_sha256": _digest(308),
                "fit_family_count": 8,
                "fit_example_count": 256,
                "full_eight_family_refit_completed": True,
                "diagnostic_subset_within_fit": True,
                "diagnostic_used_for_selection": False,
            },
        },
        private_metadata=metadata,
    )
    return baseline, challenger


def _passing_lofo_report(
    baseline: capacity._CandidateAuthority,
) -> dict[str, object]:
    return {
        "report_sha256": _digest(306),
        "decision": {
            "all_required_gates_pass": True,
            "next_action": capacity._LOFO_PASS_NEXT_ACTION,
            "protocol_sha256": (
                capacity.FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
            ),
        },
        "protocol": {
            "artifact_sha256": (
                capacity.FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
            )
        },
        "authority": {"authority_sha256": _digest(305)},
        "lineage": {
            "frozen_v9_candidate_file_sha256": baseline.binding[
                "tensor_file_sha256"
            ],
            "frozen_v9_candidate_scientific_sha256": baseline.binding[
                "scientific_payload_sha256"
            ],
        },
        "scope": {
            "compiled_layer_count": 1,
            "compiled_layer_ordinal": 17,
            "whole_model_compiled": False,
        },
        "safety": {
            "selection_opened": False,
            "guard_opened": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "heldout_confirmation": False,
        "serving_authorized": False,
        "compression_claim": False,
    }


def test_candidate_pair_classifies_equal_resources_as_fixed_capacity_refit() -> None:
    baseline, challenger = _fixed_capacity_authorities()

    pair = capacity._validate_candidate_pair(baseline, challenger)

    assert pair["comparison_kind"] == "fixed_capacity_refit"
    assert pair["graph_parameter_delta"] == 0
    assert pair["graph_macs_per_token_delta"] == 0


def test_candidate_pair_rejects_mixed_zero_and_positive_resource_delta() -> None:
    baseline, challenger = _fixed_capacity_authorities()
    challenger = replace(
        challenger,
        binding={**challenger.binding, "graph_macs_per_token": 129},
    )

    with pytest.raises(ValueError, match="one class"):
        capacity._validate_candidate_pair(baseline, challenger)


def test_candidate_pair_rejects_same_scientific_payload_at_fixed_capacity() -> None:
    baseline, challenger = _fixed_capacity_authorities()
    challenger = replace(
        challenger,
        binding={
            **challenger.binding,
            "scientific_payload_sha256": baseline.binding[
                "scientific_payload_sha256"
            ],
        },
    )

    with pytest.raises(ValueError, match="distinct scientific"):
        capacity._validate_candidate_pair(baseline, challenger)


def test_capped_candidate_adapter_normalizes_strict_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "cap64.pt"
    candidate_path.write_bytes(b"synthetic-capped-candidate")
    execution_hashes = tuple(_digest(index + 1) for index in range(4))
    resolved_ranks = (54, 38, 64, 53)
    nodes = tuple(
        SimpleNamespace(
            name=f"node.{index}",
            causal_order=index,
            input_boundary="layer.17.in",
            output_boundary="layer.17.out",
        )
        for index in range(4)
    )
    graph = SimpleNamespace(
        nodes=nodes,
        interactions=(),
        traversal_order=tuple(node.name for node in nodes),
        parameter_cluster_plan_sha256=_digest(20),
        artifact_sha256=_digest(21),
        parameter_count=200_000,
        macs_per_token=190_000,
    )
    lowerings = {
        node.name: SimpleNamespace(
            selected_fragment_sha256=execution_hashes[index],
            computational_mode_basis=SimpleNamespace(rank=resolved_ranks[index]),
            coordinate_generator_plan=SimpleNamespace(rank=16),
        )
        for index, node in enumerate(nodes)
    }
    raw = {
        "schema": capacity.GEMMA3_LAYER17_CAPPED_NODE_SCHEMA,
        "experiment": {
            "model_id": "google/gemma-3-270m",
            "requested_revision": "revision",
            "adapter_model_fingerprint": _digest(22),
            "heldout_confirmation": False,
        },
        "config": {
            "mode_rank_cap": 64,
            "resolved_node_ranks": resolved_ranks,
            "generator_rank": 16,
            "edge_policy": "edgeless",
        },
        "fragment_selection": {
            "layer_ordinal": 17,
            "fragment_ids": tuple(f"fragment.{index}" for index in range(4)),
            "execution_order_sha256s": execution_hashes,
            "removed_mode_indices": (1, 2, 3),
            "source_fragment_plan_sha256": _digest(20),
            "source_model_sha256": _digest(22),
        },
        "scientific_payload_sha256": _digest(23),
        "safety": {
            "guard_used": False,
            "calibration_b_used": False,
            "validation_used": False,
            "test_used": False,
            "heldout_confirmation": False,
        },
    }
    monkeypatch.setattr(capacity.torch, "load", lambda *args, **kwargs: raw)
    monkeypatch.setattr(
        capacity,
        "restore_gemma3_layer17_capped_node_runtime",
        lambda value: (graph, lowerings, None),
    )

    authority = capacity._candidate_authority(
        candidate_path,
        label="cap64",
        expected_generator_rank=16,
    )

    assert authority.binding["candidate_kind"] == "capped_node_edgeless_candidate"
    assert authority.binding["mode_rank_cap"] == 64
    assert authority.binding["resolved_node_ranks"] == resolved_ranks
    assert authority.binding["candidate_artifact_schema"] == (
        capacity.GEMMA3_LAYER17_CAPPED_NODE_SCHEMA
    )


def test_fixed_capacity_authorization_cross_binds_lofo_and_full_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lofo_path = tmp_path / "lofo.json"
    lofo_path.write_bytes(b"strict-lofo-report")
    lofo_file_sha256 = capacity._file_sha256(lofo_path)
    baseline, challenger = _fixed_capacity_authorities(
        lofo_report_file_sha256=lofo_file_sha256
    )
    monkeypatch.setattr(
        capacity,
        "load_gemma3_layer17_v8_fit_lofo_report",
        lambda path: _passing_lofo_report(baseline),
    )

    receipt = capacity._authorize_fixed_capacity_adaptive_selection(
        lofo_report_path=lofo_path,
        baseline=baseline,
        challenger=challenger,
    )

    assert receipt["selection_access_authorized"] is True
    assert receipt["authorization_completed_before_selection_open"] is True
    assert receipt["lofo_report_file_sha256"] == lofo_file_sha256
    assert receipt["fit_family_count"] == 8
    assert receipt["diagnostic_used_for_selection"] is False
    capacity._reject_forbidden_output_fields(receipt)


def test_failed_lofo_blocks_before_selection_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lofo_path = tmp_path / "lofo.json"
    lofo_path.write_bytes(b"strict-lofo-report")
    baseline, challenger = _fixed_capacity_authorities(
        lofo_report_file_sha256=capacity._file_sha256(lofo_path)
    )
    failed = _passing_lofo_report(baseline)
    failed["decision"] = {
        **failed["decision"],  # type: ignore[arg-type]
        "all_required_gates_pass": False,
    }
    monkeypatch.setattr(
        capacity,
        "load_gemma3_layer17_v8_fit_lofo_report",
        lambda path: failed,
    )
    candidates = iter((baseline, challenger))
    monkeypatch.setattr(
        capacity,
        "_candidate_authority",
        lambda *args, **kwargs: next(candidates),
    )
    selection_opened = False

    def forbidden_selection(**kwargs: object) -> None:
        nonlocal selection_opened
        selection_opened = True
        raise AssertionError("selection must remain closed")

    monkeypatch.setattr(
        capacity,
        "_load_open_selection_authority",
        forbidden_selection,
    )

    with pytest.raises(ValueError, match="does not authorize"):
        capacity.evaluate_gemma3_layer17_open_a_capacity(
            rank16_candidate_path="baseline.pt",
            rank32_candidate_path="challenger.pt",
            baseline_label="frozen_v9",
            challenger_label="adaptive_a_fit",
            lofo_report_path=lofo_path,
            output=tmp_path / "never-written.json",
        )
    assert selection_opened is False


def test_fixed_capacity_authorization_rejects_baseline_lineage_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lofo_path = tmp_path / "lofo.json"
    lofo_path.write_bytes(b"strict-lofo-report")
    baseline, challenger = _fixed_capacity_authorities(
        lofo_report_file_sha256=capacity._file_sha256(lofo_path)
    )
    report = _passing_lofo_report(baseline)
    report["lineage"] = {
        **report["lineage"],  # type: ignore[arg-type]
        "frozen_v9_candidate_file_sha256": _digest(999),
    }
    monkeypatch.setattr(
        capacity,
        "load_gemma3_layer17_v8_fit_lofo_report",
        lambda path: report,
    )

    with pytest.raises(ValueError, match="supplied frozen baseline"):
        capacity._authorize_fixed_capacity_adaptive_selection(
            lofo_report_path=lofo_path,
            baseline=baseline,
            challenger=challenger,
        )


def test_fixed_capacity_authorization_rejects_missing_challenger_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lofo_path = tmp_path / "lofo.json"
    lofo_path.write_bytes(b"strict-lofo-report")
    baseline, challenger = _fixed_capacity_authorities(
        lofo_report_file_sha256=capacity._file_sha256(lofo_path)
    )
    challenger = replace(challenger, private_metadata=None)
    monkeypatch.setattr(
        capacity,
        "load_gemma3_layer17_v8_fit_lofo_report",
        lambda path: _passing_lofo_report(baseline),
    )

    with pytest.raises(TypeError, match="private metadata"):
        capacity._authorize_fixed_capacity_adaptive_selection(
            lofo_report_path=lofo_path,
            baseline=baseline,
            challenger=challenger,
        )


def test_selection_authority_opens_only_selection_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        "calibration_a_fit": tmp_path / "fit.json",
        "calibration_a_selection": tmp_path / "selection.json",
        "calibration_a_guard": tmp_path / "guard.json",
    }
    role_rows = {
        "calibration_a_fit": (("synthetic fit",), ("fit.family",)),
        "calibration_a_selection": (
            tuple(f"synthetic selection {index}" for index in range(128)),
            tuple(f"selection.family.{index % 4}" for index in range(128)),
        ),
        "calibration_a_guard": (("synthetic guard",), ("guard.family",)),
    }
    for role, path in paths.items():
        prompts, families = role_rows[role]
        write_gemma3_l3_l4_progressive_a_role_input(
            path,
            corpus_id="synthetic-open-a",
            profile="full",
            role=role,  # type: ignore[arg-type]
            prompts=prompts,
            family_ids=families,
        )
    artifact = build_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_id="synthetic-open-a",
        profile="full",
        tokenizer_contract=capacity._tokenizer_contract(),
        role_input_paths=paths,  # type: ignore[arg-type]
    )
    corpus_path = tmp_path / "corpus.json"
    artifact_file_sha256 = write_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_path, artifact
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    selection_view = artifact.role_view("calibration_a_selection")
    receipt = {
        "receipt_sha256": _digest(99),
        "corpus": {
            "artifact_sha256": artifact.artifact_sha256,
            "artifact_file_sha256": artifact_file_sha256,
            "tokenizer_contract_sha256": artifact.tokenizer_contract_sha256,
        },
        "roles": {
            "calibration_a_selection": {
                "manifest_sha256": selection_view.manifest_sha256,
                "role_input_file_sha256": selection_view.role_input_file_sha256,
                "example_count": 128,
                "family_count": 4,
            }
        },
    }
    monkeypatch.setattr(
        capacity,
        "load_gemma3_layer10_v8_corpus_receipt",
        lambda path: receipt,
    )
    paths["calibration_a_fit"].unlink()
    paths["calibration_a_guard"].unlink()

    authority = capacity._load_open_selection_authority(
        corpus_artifact_path=corpus_path,
        selection_path=paths["calibration_a_selection"],
        receipt_path=receipt_path,
    )

    assert len(authority.role.prompts) == 128
    assert authority.binding["family_count"] == 4
    assert not paths["calibration_a_fit"].exists()
    assert not paths["calibration_a_guard"].exists()


def _valid_result() -> dict[str, object]:
    family_metric = {
        "supervised_tokens": 32,
        "native": {"nll_per_token": 2.0},
        "conditions": {
            name: {
                "nll_per_token": 2.1,
                "delta_nll_per_token": 2.1 - 2.0,
                "native_to_candidate_kl_per_token": 0.2,
                "top1_agreement_to_native": 0.8,
            }
            for name in capacity._CONDITIONS
        },
    }
    families = {
        f"family_{index:02d}": copy.deepcopy(family_metric) for index in range(4)
    }
    resources = {
        name: {
            "modal_graph_learned_parameters": 100,
            "native_removed_learned_parameters": 200,
            "net_stored_parameter_savings": 100,
            "graph_macs_per_token": 80,
            "native_removed_macs_per_token": 200,
            "executed_graph_macs_per_token": 0 if name == "matched_deletion" else 80,
            "net_logical_macs_saved_per_token": 200 if name == "matched_deletion" else 120,
        }
        for name in capacity._CONDITIONS
    }
    assessment = {
        "execution_path": "paired_layer17_edgeless_modal_graph_executors",
        "assessment_role": "open_development_capacity_comparison",
        "heldout_confirmation": False,
        "example_count": 128,
        "family_count": 4,
        "supervised_tokens": 128,
        "logical_valid_tokens": 256,
        "native": family_metric["native"],
        "conditions": family_metric["conditions"],
        "equal_family_macro": {
            "native": family_metric["native"],
            "conditions": family_metric["conditions"],
        },
        "families": families,
        "capacity_delta": {},
        "graph_comparison": {
            "deletion_paths_agree": True,
            "deletion_equivalence_atol": 0.0,
            "deletion_equivalence_rtol": 0.0,
            "deletion_max_abs_logit_difference": 0.0,
        },
        "resource_accounting": resources,
        "latency_or_kernel_speed_claim": False,
    }
    candidates = {
        name: {
            "candidate_role": name,
            "mode_rank": 32,
            "generator_rank": rank,
            "interaction_count": 0,
            "node_count": 4,
            "tensor_file_sha256": _digest(rank),
        }
        for name, rank in (("rank16", 16), ("rank32", 32))
    }
    payload: dict[str, object] = {
        "schema": capacity._SCHEMA,
        "format_version": 1,
        "scientific_role": "open_development_capacity_comparison",
        "heldout_confirmation": False,
        "candidates": candidates,
        "candidate_pair": {
            "identical_model": True,
            "identical_fragment_topology": True,
            "identical_native_removal_scope": True,
            "both_graphs_edgeless": True,
        },
        "corpus": {
            "example_count": 128,
            "family_count": 4,
            "assessment_role": "already_open_calibration_a_selection",
        },
        "runtime": {},
        "tokenization": {"example_count": 128, "family_stream_count": 4},
        "assessment": assessment,
        "candidate_changed": False,
        "candidate_tensor_file_sha256s_after": {
            name: candidate["tensor_file_sha256"]
            for name, candidate in candidates.items()
        },
        "selection_opened": True,
        "fit_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "safety": dict(capacity._SAFETY),
    }
    return {
        **payload,
        "result_sha256": capacity._domain_sha256(capacity._RESULT_DOMAIN, payload),
    }


def _adaptive_assessment_metrics(
    *,
    family_deltas: tuple[float, float, float, float] = (0.08, 0.08, 0.08, 0.08),
    baseline_micro_delta: float = 0.09,
    challenger_micro_delta: float = 0.08,
    baseline_kl: float = 0.09,
    challenger_kl: float = 0.09,
    baseline_top1: float = 0.84,
    challenger_top1: float = 0.84,
) -> dict[str, object]:
    baseline_condition = "frozen_v9_edgeless"
    challenger_condition = "adaptive_a_fit_edgeless"

    def metric(delta: float, kl: float, top1: float) -> dict[str, float]:
        # Native NLL is zero in this synthetic boundary fixture, making the
        # serialized DeltaNLL exactly identical to nll-native.
        return {
            "nll_per_token": delta,
            "delta_nll_per_token": delta,
            "native_to_candidate_kl_per_token": kl,
            "top1_agreement_to_native": top1,
        }

    families = {
        f"family_{index:02d}": {
            "supervised_tokens": 32,
            "native": {"nll_per_token": 0.0},
            "conditions": {
                baseline_condition: metric(
                    0.09,
                    baseline_kl,
                    baseline_top1,
                ),
                challenger_condition: metric(
                    delta,
                    challenger_kl,
                    challenger_top1,
                ),
                "matched_deletion": metric(0.2, 0.2, 0.5),
            },
        }
        for index, delta in enumerate(family_deltas)
    }
    conditions = {
        baseline_condition: metric(
            baseline_micro_delta,
            baseline_kl,
            baseline_top1,
        ),
        challenger_condition: metric(
            challenger_micro_delta,
            challenger_kl,
            challenger_top1,
        ),
        "matched_deletion": metric(0.2, 0.2, 0.5),
    }
    return {
        "native": {"nll_per_token": 0.0},
        "conditions": conditions,
        "equal_family_macro": capacity._equal_family_macro(
            families,
            conditions=tuple(conditions),
        ),
        "families": families,
    }


def _adaptive_authorization_receipt(
    baseline: capacity._CandidateAuthority,
    challenger: capacity._CandidateAuthority,
) -> dict[str, object]:
    provenance = challenger.binding["refit_provenance"]
    assert isinstance(provenance, dict)
    return {
        "authorization_kind": "passing_lofo_then_full_eight_family_refit",
        "selection_access_authorized": True,
        "authorization_completed_before_selection_open": True,
        "lofo_report_file": "lofo.json",
        "lofo_report_file_sha256": provenance["lofo_report_file_sha256"],
        "lofo_report_sha256": provenance["lofo_report_sha256"],
        "lofo_protocol_sha256": provenance["lofo_protocol_sha256"],
        "lofo_authority_sha256": provenance["lofo_authority_sha256"],
        "lofo_completed_fold_count": 8,
        "lofo_all_required_gates_pass": True,
        "lofo_authorized_next_action": capacity._LOFO_PASS_NEXT_ACTION,
        "baseline_tensor_file_sha256": baseline.binding["tensor_file_sha256"],
        "baseline_scientific_payload_sha256": baseline.binding[
            "scientific_payload_sha256"
        ],
        "challenger_tensor_file_sha256": challenger.binding[
            "tensor_file_sha256"
        ],
        "challenger_scientific_payload_sha256": challenger.binding[
            "scientific_payload_sha256"
        ],
        "full_eight_family_refit_completed": True,
        "fit_family_count": 8,
        "fit_example_count": 256,
        "fit_split_sha256": provenance["fit_split_sha256"],
        "diagnostic_split_sha256": provenance["diagnostic_split_sha256"],
        "diagnostic_subset_within_fit": True,
        "diagnostic_used_for_selection": False,
        "claim_role": "already_open_adaptive_development_selection",
        "heldout_confirmation": False,
        "serving_authorized": False,
        "compression_claim": False,
        "source_safe": True,
    }


def _valid_adaptive_result() -> dict[str, object]:
    result = _valid_result()
    baseline, challenger = _fixed_capacity_authorities()
    pair = capacity._validate_candidate_pair(baseline, challenger)
    result["scientific_role"] = (
        "already_open_adaptive_development_fixed_capacity_refit"
    )
    result["candidates"] = {
        baseline.label: copy.deepcopy(baseline.binding),
        challenger.label: copy.deepcopy(challenger.binding),
    }
    result["candidate_pair"] = pair
    result["candidate_tensor_file_sha256s_after"] = {
        baseline.label: baseline.binding["tensor_file_sha256"],
        challenger.label: challenger.binding["tensor_file_sha256"],
    }
    assessment = result["assessment"]
    assert isinstance(assessment, dict)
    metrics = _adaptive_assessment_metrics()
    assessment.update(metrics)
    assessment["assessment_role"] = result["scientific_role"]
    assessment["capacity_delta"] = {
        "adaptive_a_fit_minus_frozen_v9_nll_per_token": -0.01,
        "adaptive_a_fit_minus_frozen_v9_native_kl_per_token": 0.0,
        "adaptive_a_fit_minus_frozen_v9_top1_agreement": 0.0,
        "adaptive_a_fit_added_graph_parameters": 0,
        "adaptive_a_fit_added_graph_macs_per_token": 0,
    }
    assessment["resource_accounting"] = {
        name: {
            "modal_graph_learned_parameters": 148,
            "native_removed_learned_parameters": 441_600,
            "net_stored_parameter_savings": 441_452,
            "graph_macs_per_token": 128,
            "native_removed_macs_per_token": 441_600,
            "executed_graph_macs_per_token": (
                0 if name == "matched_deletion" else 128
            ),
            "net_logical_macs_saved_per_token": (
                441_600 if name == "matched_deletion" else 441_472
            ),
        }
        for name in (
            "frozen_v9_edgeless",
            "adaptive_a_fit_edgeless",
            "matched_deletion",
        )
    }
    result["authorization"] = _adaptive_authorization_receipt(
        baseline,
        challenger,
    )
    result["adaptive_selection"] = (
        capacity._evaluate_fixed_capacity_adaptive_gates(
            assessment=assessment,
            candidate_pair=pair,
            baseline_label=baseline.label,
            challenger_label=challenger.label,
        )
    )
    payload = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    result["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )
    return result


def test_adaptive_gate_policy_passes_exact_absolute_thresholds() -> None:
    baseline, challenger = _fixed_capacity_authorities()
    pair = capacity._validate_candidate_pair(baseline, challenger)
    assessment = _adaptive_assessment_metrics()

    decision = capacity._evaluate_fixed_capacity_adaptive_gates(
        assessment=assessment,
        candidate_pair=pair,
        baseline_label=baseline.label,
        challenger_label=challenger.label,
    )

    assert decision["all_required_gates_pass"] is True
    assert decision["passing_family_count"] == 4
    assert decision["heldout_or_serving_authorized"] is False

    three_family_decision = capacity._evaluate_fixed_capacity_adaptive_gates(
        assessment=_adaptive_assessment_metrics(
            family_deltas=(0.05, 0.05, 0.05, 0.11),
        ),
        candidate_pair=pair,
        baseline_label=baseline.label,
        challenger_label=challenger.label,
    )
    assert three_family_decision["passing_family_count"] == 3
    assert three_family_decision["all_required_gates_pass"] is True


@pytest.mark.parametrize(
    ("family_deltas", "baseline_micro_delta", "failed_gate"),
    (
        ((0.05, 0.05, 0.11, 0.11), 0.09, "candidate_family_delta_nll_pass_count"),
        ((0.08, 0.08, 0.08, 0.08), 0.08, "strict_micro_delta_nll_improvement_over_baseline"),
    ),
)
def test_adaptive_gate_policy_fails_family_count_or_no_improvement(
    family_deltas: tuple[float, float, float, float],
    baseline_micro_delta: float,
    failed_gate: str,
) -> None:
    baseline, challenger = _fixed_capacity_authorities()
    decision = capacity._evaluate_fixed_capacity_adaptive_gates(
        assessment=_adaptive_assessment_metrics(
            family_deltas=family_deltas,
            baseline_micro_delta=baseline_micro_delta,
        ),
        candidate_pair=capacity._validate_candidate_pair(baseline, challenger),
        baseline_label=baseline.label,
        challenger_label=challenger.label,
    )

    assert decision["all_required_gates_pass"] is False
    assert next(
        row for row in decision["gate_table"] if row["gate"] == failed_gate
    )["passed"] is False


def test_result_validator_replays_adaptive_decision_and_provenance() -> None:
    result = _valid_adaptive_result()

    validated = capacity.validate_gemma3_layer17_open_a_capacity_result(result)

    assert validated["candidate_pair"]["comparison_kind"] == (
        "fixed_capacity_refit"
    )
    assert validated["adaptive_selection"]["all_required_gates_pass"] is True


def test_metric_validator_accepts_macro_delta_roundoff() -> None:
    native_nll = 5.798526560826362
    nll = 5.894171490720028
    delta_from_family_macro = 0.09564492989366591
    assert delta_from_family_macro != nll - native_nll
    assert abs(delta_from_family_macro - (nll - native_nll)) < 1e-12

    capacity._validate_metric_container(
        {
            "native": {"nll_per_token": native_nll},
            "conditions": {
                "candidate": {
                    "nll_per_token": nll,
                    "delta_nll_per_token": delta_from_family_macro,
                    "native_to_candidate_kl_per_token": 0.07228037908457986,
                    "top1_agreement_to_native": 0.8493901350003648,
                }
            },
        },
        label="equal-family macro",
        conditions=("candidate",),
        identity_ulps=capacity._MACRO_METRIC_IDENTITY_ULPS,
    )


def test_metric_validator_uses_narrow_scale_aware_ulp_budgets() -> None:
    native_nll = 5.0
    nll = 6.0
    expected = nll - native_nll
    operand_ulp = max(math.ulp(native_nll), math.ulp(nll))

    def metrics(delta: float) -> dict[str, object]:
        return {
            "native": {"nll_per_token": native_nll},
            "conditions": {
                "candidate": {
                    "nll_per_token": nll,
                    "delta_nll_per_token": delta,
                    "native_to_candidate_kl_per_token": 0.1,
                    "top1_agreement_to_native": 0.9,
                }
            },
        }

    capacity._validate_metric_container(
        metrics(expected + 16 * operand_ulp),
        label="aggregate",
        conditions=("candidate",),
        identity_ulps=capacity._MACRO_METRIC_IDENTITY_ULPS,
    )
    with pytest.raises(ValueError, match="beyond 2 ULPs"):
        capacity._validate_metric_container(
            metrics(expected + 3 * operand_ulp),
            label="micro",
            conditions=("candidate",),
            identity_ulps=capacity._MICRO_FAMILY_METRIC_IDENTITY_ULPS,
        )
    with pytest.raises(ValueError, match="beyond 32 ULPs"):
        capacity._validate_metric_container(
            metrics(expected + 33 * operand_ulp),
            label="aggregate",
            conditions=("candidate",),
            identity_ulps=capacity._MACRO_METRIC_IDENTITY_ULPS,
        )


def test_result_validator_rejects_rehashed_adaptive_decision_tamper() -> None:
    result = _valid_adaptive_result()
    result["adaptive_selection"]["all_required_gates_pass"] = False  # type: ignore[index]
    payload = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    result["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )

    with pytest.raises(ValueError, match="does not replay"):
        capacity.validate_gemma3_layer17_open_a_capacity_result(result)


def test_result_validator_rejects_rehashed_delta_or_macro_contradiction() -> None:
    result = _valid_adaptive_result()
    family = result["assessment"]["families"]["family_00"]  # type: ignore[index]
    family["conditions"]["adaptive_a_fit_edgeless"][  # type: ignore[index]
        "delta_nll_per_token"
    ] = 0.079
    payload = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    result["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="delta NLL .* contradicts"):
        capacity.validate_gemma3_layer17_open_a_capacity_result(result)

    result = _valid_adaptive_result()
    macro = result["assessment"]["equal_family_macro"]["conditions"][  # type: ignore[index]
        "adaptive_a_fit_edgeless"
    ]
    macro["nll_per_token"] = 0.07  # type: ignore[index]
    macro["delta_nll_per_token"] = 0.07  # type: ignore[index]
    payload = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    result["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="macro does not replay"):
        capacity.validate_gemma3_layer17_open_a_capacity_result(result)


def test_result_validator_rejects_rehashed_resource_or_provenance_drift() -> None:
    result = _valid_adaptive_result()
    result["assessment"]["resource_accounting"][  # type: ignore[index]
        "adaptive_a_fit_edgeless"
    ]["modal_graph_learned_parameters"] = 149
    payload = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    result["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="candidate/runtime resource"):
        capacity.validate_gemma3_layer17_open_a_capacity_result(result)

    result = _valid_adaptive_result()
    result["authorization"]["lofo_report_sha256"] = _digest(999)  # type: ignore[index]
    payload = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    result["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="provenance does not bind"):
        capacity.validate_gemma3_layer17_open_a_capacity_result(result)


def test_result_validator_rejects_unknown_adaptive_candidate_field() -> None:
    result = _valid_adaptive_result()
    result["candidates"]["adaptive_a_fit"]["opaque_blob"] = "not allowed"  # type: ignore[index]
    payload = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    result["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="candidate binding fields"):
        capacity.validate_gemma3_layer17_open_a_capacity_result(result)


def test_prevalidation_checkpoint_is_derived_and_cleaned_after_publish(
    tmp_path: Path,
) -> None:
    output = tmp_path / "adaptive-result.json"
    checkpoint_path = (
        tmp_path / "adaptive-result.prevalidation-checkpoint.json"
    )
    assert capacity._prevalidation_checkpoint_path(output) == checkpoint_path

    published = capacity._publish_with_prevalidation_checkpoint(
        _valid_result(),
        output=output,
    )

    assert output.exists()
    assert not checkpoint_path.exists()
    assert capacity.load_gemma3_layer17_open_a_capacity_result(output) == published


def test_prevalidation_checkpoint_accepts_adaptive_result_shape(
    tmp_path: Path,
) -> None:
    checkpoint = capacity._build_prevalidation_checkpoint(
        _valid_adaptive_result(),
        final_output=tmp_path / "adaptive-result.json",
    )

    assert checkpoint["status"] == "unvalidated"
    assert checkpoint["unvalidated_result"]["authorization"][  # type: ignore[index]
        "selection_access_authorized"
    ] is True


@pytest.mark.parametrize("existing_kind", ("output", "checkpoint"))
def test_evaluate_preflights_output_and_checkpoint_before_candidate_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_kind: str,
) -> None:
    output = tmp_path / "adaptive-result.json"
    existing = (
        output
        if existing_kind == "output"
        else capacity._prevalidation_checkpoint_path(output)
    )
    existing.write_text("occupied", encoding="utf-8")
    candidate_accessed = False

    def forbidden_candidate(*args: object, **kwargs: object) -> None:
        nonlocal candidate_accessed
        candidate_accessed = True
        raise AssertionError("candidate access must not occur")

    monkeypatch.setattr(capacity, "_candidate_authority", forbidden_candidate)
    with pytest.raises(FileExistsError):
        capacity.evaluate_gemma3_layer17_open_a_capacity(output=output)
    assert candidate_accessed is False


def test_prevalidation_checkpoint_survives_strict_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "adaptive-result.json"
    checkpoint_path = capacity._prevalidation_checkpoint_path(output)

    def fail_validation(value: object) -> dict[str, object]:
        raise ValueError("synthetic late validation failure")

    monkeypatch.setattr(
        capacity,
        "validate_gemma3_layer17_open_a_capacity_result",
        fail_validation,
    )
    with pytest.raises(ValueError, match="synthetic late"):
        capacity._publish_with_prevalidation_checkpoint(
            _valid_result(),
            output=output,
        )

    assert not output.exists()
    assert checkpoint_path.exists()
    checkpoint = capacity.load_gemma3_layer17_open_a_prevalidation_checkpoint(
        checkpoint_path
    )
    assert checkpoint["status"] == "unvalidated"
    assert checkpoint["intended_final_output_file"] == output.name


def test_prevalidation_checkpoint_model_free_finalize_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "adaptive-result.json"
    checkpoint_path = capacity._prevalidation_checkpoint_path(output)
    checkpoint = capacity._build_prevalidation_checkpoint(
        _valid_result(),
        final_output=output,
    )
    capacity._write_exclusive(checkpoint_path, checkpoint)
    for name in (
        "load_gemma3",
        "_candidate_authority",
        "_load_open_selection_authority",
    ):
        monkeypatch.setattr(
            capacity,
            name,
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("checkpoint recovery must not open model or data")
            ),
        )

    finalized = (
        capacity.finalize_gemma3_layer17_open_a_prevalidation_checkpoint(
            checkpoint_path
        )
    )

    assert output.exists()
    assert not checkpoint_path.exists()
    assert capacity.load_gemma3_layer17_open_a_capacity_result(output) == finalized


def test_prevalidation_finalizer_rejects_alternate_directory_or_name(
    tmp_path: Path,
) -> None:
    output = tmp_path / "adaptive-result.json"
    checkpoint_path = capacity._prevalidation_checkpoint_path(output)
    checkpoint = capacity._build_prevalidation_checkpoint(
        _valid_result(),
        final_output=output,
    )
    capacity._write_exclusive(checkpoint_path, checkpoint)
    alternate = tmp_path / "alternate" / output.name

    with pytest.raises(ValueError, match="output path differs"):
        capacity.finalize_gemma3_layer17_open_a_prevalidation_checkpoint(
            checkpoint_path,
            output=alternate,
        )
    assert checkpoint_path.exists()
    assert not output.exists()
    assert not alternate.exists()

    renamed_checkpoint = tmp_path / "renamed-checkpoint.json"
    checkpoint_path.rename(renamed_checkpoint)
    with pytest.raises(ValueError, match="checkpoint path differs"):
        capacity.finalize_gemma3_layer17_open_a_prevalidation_checkpoint(
            renamed_checkpoint
        )
    assert renamed_checkpoint.exists()
    assert not output.exists()


def test_prevalidation_checkpoint_rejects_tamper_and_source_leak(
    tmp_path: Path,
) -> None:
    output = tmp_path / "adaptive-result.json"
    checkpoint_path = capacity._prevalidation_checkpoint_path(output)
    checkpoint = capacity._build_prevalidation_checkpoint(
        _valid_result(),
        final_output=output,
    )
    tampered = copy.deepcopy(checkpoint)
    tampered["status"] = "validated"
    checkpoint_path.write_bytes(capacity._canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        capacity.finalize_gemma3_layer17_open_a_prevalidation_checkpoint(
            checkpoint_path
        )
    assert checkpoint_path.exists()
    assert not output.exists()

    leaked = _valid_result()
    leaked["runtime"]["prompt_text"] = "must never persist"  # type: ignore[index]
    payload = {
        key: value for key, value in leaked.items() if key != "result_sha256"
    }
    leaked["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="forbidden source field"):
        capacity._build_prevalidation_checkpoint(
            leaked,
            final_output=tmp_path / "leaked.json",
        )

    unknown = _valid_result()
    unknown["runtime"]["raw_material"] = "opaque source data"  # type: ignore[index]
    payload = {
        key: value for key, value in unknown.items() if key != "result_sha256"
    }
    unknown["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="checkpoint runtime fields"):
        capacity._build_prevalidation_checkpoint(
            unknown,
            final_output=tmp_path / "unknown.json",
        )

    nested = _valid_result()
    nested["runtime"]["model_id"] = {  # type: ignore[index]
        "raw_material": "opaque source data"
    }
    payload = {
        key: value for key, value in nested.items() if key != "result_sha256"
    }
    nested["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )
    with pytest.raises(TypeError, match="runtime model_id must be scalar"):
        capacity._build_prevalidation_checkpoint(
            nested,
            final_output=tmp_path / "nested.json",
        )


def test_result_validator_rejects_family_identity_leak() -> None:
    result = _valid_result()
    capacity.validate_gemma3_layer17_open_a_capacity_result(result)
    leaked = copy.deepcopy(result)
    leaked["assessment"]["family_ids"] = ["secret.family"]  # type: ignore[index]
    payload = {key: value for key, value in leaked.items() if key != "result_sha256"}
    leaked["result_sha256"] = capacity._domain_sha256(capacity._RESULT_DOMAIN, payload)
    with pytest.raises(ValueError, match="forbidden source field"):
        capacity.validate_gemma3_layer17_open_a_capacity_result(leaked)


def test_result_validator_roundtrips_canonical_json_path_and_tuples(
    tmp_path: Path,
) -> None:
    result = _valid_result()
    result["corpus"]["source_manifest_chain"] = (  # type: ignore[index]
        _digest(201),
        _digest(202),
    )
    payload = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    result["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )
    in_memory = capacity.validate_gemma3_layer17_open_a_capacity_result(result)
    assert in_memory["corpus"]["source_manifest_chain"] == [  # type: ignore[index]
        _digest(201),
        _digest(202),
    ]

    output = tmp_path / "capacity-result.json"
    output.write_bytes(capacity._canonical_json_bytes(result))
    from_path = capacity.validate_gemma3_layer17_open_a_capacity_result(output)
    from_loader = capacity.load_gemma3_layer17_open_a_capacity_result(output)

    assert from_path == in_memory
    assert from_loader == in_memory


def test_result_validator_accepts_generic_capped_candidate_labels() -> None:
    result = _valid_result()
    candidates = result["candidates"]
    after = result["candidate_tensor_file_sha256s_after"]
    assessment = result["assessment"]
    pair = result["candidate_pair"]
    assert isinstance(candidates, dict)
    assert isinstance(after, dict)
    assert isinstance(assessment, dict)
    assert isinstance(pair, dict)
    renamed_candidates = {}
    renamed_after = {}
    for source, target, cap, ranks in (
        ("rank16", "cap48", 48, (48, 38, 48, 48)),
        ("rank32", "cap64", 64, (54, 38, 64, 53)),
    ):
        candidate = dict(candidates[source])
        candidate.update(
            {
                "candidate_role": target,
                "candidate_kind": "capped_node_edgeless_candidate",
                "candidate_artifact_schema": (
                    capacity.GEMMA3_LAYER17_CAPPED_NODE_SCHEMA
                ),
                "mode_rank": cap,
                "mode_rank_cap": cap,
                "resolved_node_ranks": ranks,
                "generator_rank": 16,
            }
        )
        renamed_candidates[target] = candidate
        renamed_after[target] = after[source]
    result["candidates"] = renamed_candidates
    result["candidate_tensor_file_sha256s_after"] = renamed_after
    pair["baseline_label"] = "cap48"
    pair["challenger_label"] = "cap64"

    condition_renames = {
        "rank16_edgeless": "cap48_edgeless",
        "rank32_edgeless": "cap64_edgeless",
        "matched_deletion": "matched_deletion",
    }
    assessment["conditions"] = {
        condition_renames[name]: value
        for name, value in assessment["conditions"].items()
    }
    assessment["equal_family_macro"]["conditions"] = {
        condition_renames[name]: value
        for name, value in assessment["equal_family_macro"]["conditions"].items()
    }
    for family in assessment["families"].values():
        family["conditions"] = {
            condition_renames[name]: value
            for name, value in family["conditions"].items()
        }
    assessment["resource_accounting"] = {
        condition_renames[name]: value
        for name, value in assessment["resource_accounting"].items()
    }
    payload = {
        key: value for key, value in result.items() if key != "result_sha256"
    }
    result["result_sha256"] = capacity._domain_sha256(
        capacity._RESULT_DOMAIN,
        payload,
    )

    validated = capacity.validate_gemma3_layer17_open_a_capacity_result(result)
    assert set(validated["candidates"]) == {"cap48", "cap64"}
    assert set(validated["assessment"]["conditions"]) == {
        "cap48_edgeless",
        "cap64_edgeless",
        "matched_deletion",
    }


def test_cli_exposes_configurable_batch_size() -> None:
    args = capacity.build_parser().parse_args(["--tokenization-batch-size", "7"])
    assert args.tokenization_batch_size == 7


def test_cli_accepts_generic_candidate_aliases_and_labels() -> None:
    args = capacity.build_parser().parse_args(
        [
            "--baseline-candidate",
            "cap48.pt",
            "--baseline-label",
            "cap48",
            "--challenger-candidate",
            "cap64.pt",
            "--challenger-label",
            "cap64",
        ]
    )
    assert args.baseline_candidate == Path("cap48.pt")
    assert args.challenger_candidate == Path("cap64.pt")
    assert args.baseline_label == "cap48"
    assert args.challenger_label == "cap64"
