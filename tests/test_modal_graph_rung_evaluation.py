from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.modal_graph_rung_evaluation import (
    DevelopmentInteractionPartitionPlan,
    evaluate_modal_graph_rung_conditions,
    partition_development_export_for_interactions,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _development_export(count: int = 40) -> object:
    prompts = tuple(f"development prompt {index}" for index in range(count))
    return SimpleNamespace(
        prompts=prompts,
        prompt_sha256s=tuple(_sha(prompt) for prompt in prompts),
        family_ids=tuple(f"family.{index % 4}" for index in range(count)),
        fit_positions=tuple(range(40, 40 + count)),
        source_prompt_indices=tuple(range(1_040, 1_040 + count)),
        source_corpus_id="structured-strong-v9",
        source_fit_prompt_index_sha256="a" * 64,
        artifact_sha256="b" * 64,
    )


def test_development_partition_is_deterministic_disjoint_and_source_safe() -> None:
    export = _development_export()
    first = partition_development_export_for_interactions(export)
    second = partition_development_export_for_interactions(export)

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.selection.artifact_sha256 == second.selection.artifact_sha256
    assert first.assessment.artifact_sha256 == second.assessment.artifact_sha256
    assert first.selection.prompt_count == 20
    assert first.assessment.prompt_count == 20
    assert set(first.selection.prompt_sha256s).isdisjoint(
        first.assessment.prompt_sha256s
    )
    assert set(first.selection.source_prompt_indices).isdisjoint(
        first.assessment.source_prompt_indices
    )
    assert (
        set(first.selection.prompt_sha256s)
        | set(first.assessment.prompt_sha256s)
    ) == set(export.prompt_sha256s)
    assert (
        set(first.selection.source_prompt_indices)
        | set(first.assessment.source_prompt_indices)
    ) == set(export.source_prompt_indices)

    metadata = first.metadata()
    serialized = json.dumps(metadata, sort_keys=True)
    assert all(prompt not in serialized for prompt in export.prompts)
    assert metadata["assessment_status"] == (
        "open_development_not_closed_guard"
    )
    assert metadata["prompt_membership_disjoint"] is True
    assert metadata["source_prompt_index_membership_disjoint"] is True
    # This deterministic membership split does not pretend to be family-heldout.
    assert metadata["family_disjoint"] is False
    assert metadata["membership_externally_authenticated"] is False


def test_development_partition_rejects_wrong_size_and_changes_with_salt() -> None:
    with pytest.raises(ValueError, match="exactly 40"):
        partition_development_export_for_interactions(
            _development_export(39)
        )

    export = _development_export()
    first = partition_development_export_for_interactions(export)
    alternate = partition_development_export_for_interactions(
        export,
        partition_salt="four-node-fanin-rung-alternate-v1",
    )
    assert first.artifact_sha256 != alternate.artifact_sha256
    assert set(first.selection.prompt_sha256s) != set(
        alternate.selection.prompt_sha256s
    )

    drifted_assessment = replace(
        first.assessment,
        partition_salt="drifted-partition-v1",
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="member provenance"):
        DevelopmentInteractionPartitionPlan(
            source_export_sha256=first.source_export_sha256,
            selection=first.selection,
            assessment=drifted_assessment,
            expected_prompt_count=40,
        )


class _NativeModel(nn.Module):
    def forward(
        self,
        *,
        logits: Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> object:
        assert use_cache is False
        assert return_dict is True
        return SimpleNamespace(logits=logits)


class _Adapter:
    def __init__(self) -> None:
        self.module = _NativeModel()


def _plan(*, interactions: int, node_hashes: tuple[str, ...] | None = None):
    hashes = node_hashes or tuple(f"{index + 1:064x}" for index in range(4))
    nodes = tuple(
        SimpleNamespace(
            name=f"node.{index}",
            artifact_sha256=digest,
        )
        for index, digest in enumerate(hashes)
    )
    return SimpleNamespace(
        model_fingerprint="c" * 64,
        parameter_cluster_plan_sha256="d" * 64,
        nodes=nodes,
        interactions=tuple(object() for _ in range(interactions)),
        traversal_order=tuple(node.name for node in nodes),
    )


class _GraphExecutor:
    def __init__(
        self,
        *,
        generated_logits: Tensor,
        deletion_logits: Tensor,
        interactions: int,
        node_hashes: tuple[str, ...] | None = None,
    ) -> None:
        self.graph_plan = _plan(
            interactions=interactions,
            node_hashes=node_hashes,
        )
        self.generated_logits = generated_logits
        self.deletion_logits = deletion_logits
        self.calls: list[str] = []
        self.graph_parameters = 40 + 3 * interactions
        self.graph_macs = 80 + 6 * interactions
        self.graph_additions = 16 + 2 * interactions

    def run(
        self,
        model_inputs: dict[str, Tensor],
        *,
        condition: str,
    ) -> object:
        assert condition in {"generated", "deletion"}
        self.calls.append(condition)
        generated = condition == "generated"
        logits = (
            self.generated_logits
            if generated
            else self.deletion_logits
        )
        executed_macs = self.graph_macs * 2 if generated else 0
        executed_additions = self.graph_additions * 2 if generated else 0
        return SimpleNamespace(
            model_output=SimpleNamespace(logits=logits.clone()),
            condition=condition,
            graph_execution=SimpleNamespace(
                traversal_order=(
                    self.graph_plan.traversal_order if generated else ()
                )
            ),
            replacement_scope="partial_native_mlp_mode_replacement",
            replaced_layer_count=4,
            graph_node_count=4,
            fragment_count=4,
            removed_mode_count=250,
            source_whole_model_learned_parameters=1_000,
            candidate_whole_model_learned_parameters=(
                1_000 - 120 + self.graph_parameters
            ),
            native_removed_learned_parameters=120,
            modal_graph_learned_parameters=self.graph_parameters,
            net_stored_parameter_savings=120 - self.graph_parameters,
            graph_runtime_storage=(
                "registered_copied_device_local_graph_parameters"
            ),
            valid_tokens=2,
            logical_linear_macs_native_removed=240,
            logical_modal_graph_macs=self.graph_macs * 2,
            logical_executed_modal_graph_macs=executed_macs,
            logical_modal_graph_additions=self.graph_additions * 2,
            logical_executed_modal_graph_additions=executed_additions,
            net_logical_macs_saved=240 - executed_macs,
            peak_live_modal_width=128 if generated else 0,
        )


class _DenseExecutor:
    def __init__(self, logits: Tensor) -> None:
        self.logits = logits
        self.calls: list[str] = []

    def run(
        self,
        model_inputs: dict[str, Tensor],
        *,
        condition: str,
    ) -> object:
        assert condition == "generated"
        self.calls.append(condition)
        return SimpleNamespace(
            model_output=SimpleNamespace(logits=self.logits.clone()),
            replacement_scope="partial_native_mlp_mode_replacement",
            replaced_layer_count=4,
            removed_mode_count=250,
            source_whole_model_learned_parameters=1_000,
            candidate_whole_model_learned_parameters=910,
            native_removed_learned_parameters=120,
            modal_generator_learned_parameters=30,
            net_stored_parameter_savings=90,
            valid_tokens=2,
            logical_linear_macs_native_removed=240,
            logical_modal_generator_macs=140,
            logical_executed_modal_generator_macs=140,
            logical_modal_generator_bias_additions=16,
            logical_executed_modal_generator_bias_additions=16,
            net_logical_macs_saved=100,
        )


def _evaluation_fixture() -> tuple[
    CalibrationBatch,
    Tensor,
    Tensor,
    Tensor,
]:
    native = torch.tensor(
        [[[5.0, 0.0, -1.0], [0.0, 4.0, -2.0]]],
        dtype=torch.float32,
    )
    edgeless = native + torch.tensor(
        [[[0.0, 0.15, -0.05], [0.1, 0.0, -0.05]]]
    )
    interacting = native + torch.tensor(
        [[[0.0, 0.02, -0.01], [0.01, 0.0, -0.01]]]
    )
    deletion = native.roll(shifts=1, dims=-1)
    batch = CalibrationBatch(
        model_inputs={"logits": native},
        targets=torch.tensor([[0, 1]]),
        valid_positions=torch.ones((1, 2), dtype=torch.bool),
        example_ids=("assessment.0",),
    )
    return batch, edgeless, interacting, deletion


def test_unified_evaluator_reports_every_control_and_exact_accounting() -> None:
    batch, edgeless_logits, interacting_logits, deletion_logits = (
        _evaluation_fixture()
    )
    interacting = _GraphExecutor(
        generated_logits=interacting_logits,
        deletion_logits=deletion_logits,
        interactions=3,
    )
    edgeless = _GraphExecutor(
        generated_logits=edgeless_logits,
        deletion_logits=deletion_logits,
        interactions=0,
    )
    dense = _DenseExecutor(edgeless_logits + 1e-7)

    report = evaluate_modal_graph_rung_conditions(
        _Adapter(),
        interacting,
        edgeless,
        (batch,),
        nodewise_dense_executor=dense,
        dense_equivalence_atol=1e-6,
        dense_equivalence_rtol=0.0,
        expected_example_ids=("assessment.0",),
    )

    assert report["assessment_role"] == "open_development_assessment"
    assert report["heldout_confirmation"] is False
    assert set(report["conditions"]) == {
        "interacting_graph",
        "edgeless_graph",
        "matched_deletion",
        "nodewise_dense_fused",
    }
    assert report["conditions"]["interacting_graph"][
        "native_to_candidate_kl_per_token"
    ] < report["conditions"]["edgeless_graph"][
        "native_to_candidate_kl_per_token"
    ]
    assert report["conditions"]["matched_deletion"][
        "top1_agreement_to_native"
    ] == 0.0
    comparison = report["graph_comparison"]
    assert comparison["node_count"] == 4
    assert comparison["interacting_edge_count"] == 3
    assert comparison["interaction_parameter_delta"] == 9
    assert comparison["deletion_paths_agree"] is True
    assert comparison["nodewise_dense_agrees_with_edgeless"] is True
    assert comparison["nodewise_dense_max_abs_logit_difference"] < 1e-6

    resources = report["resource_accounting"]
    assert resources["interacting_graph"][
        "modal_graph_learned_parameters"
    ] == 49
    assert resources["edgeless_graph"][
        "modal_graph_learned_parameters"
    ] == 40
    assert resources["interacting_graph"][
        "logical_executed_modal_graph_macs"
    ] == 196
    assert resources["matched_deletion"][
        "logical_executed_modal_graph_macs"
    ] == 0
    assert resources["nodewise_dense_fused"][
        "modal_generator_learned_parameters"
    ] == 30
    assert interacting.calls == ["generated", "deletion"]
    assert edgeless.calls == ["generated", "deletion"]
    assert dense.calls == ["generated"]


def test_unified_evaluator_rejects_node_or_deletion_control_drift() -> None:
    batch, edgeless_logits, interacting_logits, deletion_logits = (
        _evaluation_fixture()
    )
    interacting = _GraphExecutor(
        generated_logits=interacting_logits,
        deletion_logits=deletion_logits,
        interactions=3,
    )
    wrong_nodes = _GraphExecutor(
        generated_logits=edgeless_logits,
        deletion_logits=deletion_logits,
        interactions=0,
        node_hashes=(
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "f" * 64,
        ),
    )
    with pytest.raises(ValueError, match="identical modal nodes"):
        evaluate_modal_graph_rung_conditions(
            _Adapter(),
            interacting,
            wrong_nodes,
            (batch,),
            expected_example_ids=("assessment.0",),
        )

    mismatched_deletion = _GraphExecutor(
        generated_logits=edgeless_logits,
        deletion_logits=deletion_logits + 0.01,
        interactions=0,
    )
    with pytest.raises(ValueError, match="deletion.*exceed"):
        evaluate_modal_graph_rung_conditions(
            _Adapter(),
            interacting,
            mismatched_deletion,
            (batch,),
            expected_example_ids=("assessment.0",),
        )


def test_unified_evaluator_rejects_dense_fused_drift() -> None:
    batch, edgeless_logits, interacting_logits, deletion_logits = (
        _evaluation_fixture()
    )
    with pytest.raises(ValueError, match="nodewise dense.*exceed"):
        evaluate_modal_graph_rung_conditions(
            _Adapter(),
            _GraphExecutor(
                generated_logits=interacting_logits,
                deletion_logits=deletion_logits,
                interactions=3,
            ),
            _GraphExecutor(
                generated_logits=edgeless_logits,
                deletion_logits=deletion_logits,
                interactions=0,
            ),
            (batch,),
            nodewise_dense_executor=_DenseExecutor(
                edgeless_logits + 0.1
            ),
            dense_equivalence_atol=1e-6,
            dense_equivalence_rtol=0.0,
            expected_example_ids=("assessment.0",),
        )


def test_unified_evaluator_rejects_membership_and_invalid_supervision() -> None:
    batch, edgeless_logits, interacting_logits, deletion_logits = (
        _evaluation_fixture()
    )
    interacting = _GraphExecutor(
        generated_logits=interacting_logits,
        deletion_logits=deletion_logits,
        interactions=3,
    )
    edgeless = _GraphExecutor(
        generated_logits=edgeless_logits,
        deletion_logits=deletion_logits,
        interactions=0,
    )
    with pytest.raises(ValueError, match="declared example membership"):
        evaluate_modal_graph_rung_conditions(
            _Adapter(),
            interacting,
            edgeless,
            (batch, batch),
            expected_example_ids=("assessment.0",),
        )

    invalid = CalibrationBatch(
        model_inputs={"logits": batch.model_inputs["logits"]},
        targets=batch.targets,
        valid_positions=torch.tensor([[True, False]]),
        example_ids=("assessment.0",),
    )
    with pytest.raises(ValueError, match="subset of valid positions"):
        evaluate_modal_graph_rung_conditions(
            _Adapter(),
            interacting,
            edgeless,
            (invalid,),
            expected_example_ids=("assessment.0",),
        )


def test_unified_evaluator_rejects_missing_traversal_evidence() -> None:
    batch, edgeless_logits, interacting_logits, deletion_logits = (
        _evaluation_fixture()
    )
    interacting = _GraphExecutor(
        generated_logits=interacting_logits,
        deletion_logits=deletion_logits,
        interactions=3,
    )
    edgeless = _GraphExecutor(
        generated_logits=edgeless_logits,
        deletion_logits=deletion_logits,
        interactions=0,
    )
    original = interacting.run

    def without_traversal(
        model_inputs: dict[str, Tensor],
        *,
        condition: str,
    ) -> object:
        result = original(model_inputs, condition=condition)
        result.graph_execution = SimpleNamespace(traversal_order=())
        return result

    interacting.run = without_traversal  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="traversal order drifted"):
        evaluate_modal_graph_rung_conditions(
            _Adapter(),
            interacting,
            edgeless,
            (batch,),
            expected_example_ids=("assessment.0",),
        )
