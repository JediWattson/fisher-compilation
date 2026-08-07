from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from fisher_graph.gemma3_state_conditioned_modal_graph_artifact import (
    GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_FORMAT_VERSION,
    GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA,
    build_gemma3_state_conditioned_modal_graph_candidate,
    build_gemma3_state_conditioned_modal_graph_report,
    load_gemma3_state_conditioned_modal_graph_candidate,
    save_gemma3_state_conditioned_modal_graph_candidate,
)
from fisher_graph.modal_generator_graph import ModalGeneratorGraphPlan
from test_modal_compiler_pipeline import _conditional_compiled


def _arguments(*, promoted: bool = True) -> dict[str, object]:
    pipeline, promotion = _conditional_compiled()
    dynamic = pipeline.graph_plan
    edgeless = ModalGeneratorGraphPlan(
        model_fingerprint=dynamic.model_fingerprint,
        parameter_cluster_plan_sha256=(
            dynamic.parameter_cluster_plan_sha256
        ),
        nodes=dynamic.nodes,
        interactions=(),
    )
    return {
        "experiment": {
            "experiment_id": "gemma3.same-layer.state-conditioned.v1",
            "model_id": "google/gemma-3-270m",
        },
        "config": {
            "routing_group": "next_mode",
            "temperature": 0.75,
            "top_k": 1,
        },
        "splits": {
            "fit_split_sha256": pipeline.fit_split_sha256,
            "selection_split_sha256": pipeline.eval_split_sha256,
        },
        "selection": {
            "dynamic_graph_sha256": dynamic.artifact_sha256,
            "promotion_status": "promoted" if promoted else "unpromoted",
            "promotion_passed": promoted,
            "compiler_pipeline_sha256": (
                pipeline.artifact_sha256 if promoted else None
            ),
            "interaction_promotion_sha256": (
                promotion.artifact_sha256 if promoted else None
            ),
        },
        "resources": {
            "dynamic_parameter_count": dynamic.parameter_count,
            "edgeless_parameter_count": edgeless.parameter_count,
            "dynamic_dense_macs_per_token": dynamic.macs_per_token,
            "edgeless_dense_macs_per_token": edgeless.macs_per_token,
        },
        "lowerings_by_node": {
            node.node_name: node.lowering for node in pipeline.nodes
        },
        "edgeless_graph": edgeless,
        "dynamic_graph": dynamic,
        "compiler_pipeline": pipeline if promoted else None,
    }


def _payload(*, promoted: bool = True) -> dict[str, object]:
    return build_gemma3_state_conditioned_modal_graph_candidate(
        **_arguments(promoted=promoted),  # type: ignore[arg-type]
    )


def _save_raw(path: Path, payload: dict[str, object]) -> None:
    with path.open("wb") as handle:
        torch.save(payload, handle)


def test_build_report_save_load_and_refuse_overwrite(tmp_path: Path) -> None:
    arguments = _arguments()
    payload = build_gemma3_state_conditioned_modal_graph_candidate(
        **arguments,  # type: ignore[arg-type]
    )
    report = build_gemma3_state_conditioned_modal_graph_report(
        payload,
        tensor_file="candidate.pt",
    )

    assert payload["schema"] == GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA
    assert payload["format_version"] == (
        GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_FORMAT_VERSION
    )
    assert payload["safety"]["contains_prompt_text"] is False
    assert payload["safety"]["contains_token_ids"] is False
    assert payload["safety"]["contains_source_model_weights"] is False
    assert payload["safety"]["contains_promoted_compiler_pipeline"] is True
    assert payload["lineage"]["state_conditioned_interaction_count"] == 2
    assert payload["lineage"]["dynamic_graph_sha256"] == (
        arguments["dynamic_graph"].artifact_sha256
    )
    assert len(payload["lowering_records"]) == 3
    assert "lowering_records" not in report
    assert report["artifact"]["tensor_file"] == "candidate.pt"
    json.dumps(report, allow_nan=False)

    output = tmp_path / "candidate.pt"
    saved_report = save_gemma3_state_conditioned_modal_graph_candidate(
        output,
        **arguments,
    )
    loaded = load_gemma3_state_conditioned_modal_graph_candidate(output)
    with output.with_suffix(".json").open(encoding="utf-8") as handle:
        disk_report = json.load(handle)

    assert loaded["scientific_payload_sha256"] == (
        payload["scientific_payload_sha256"]
    )
    assert saved_report == disk_report
    assert disk_report["artifact"]["scientific_payload_sha256"] == (
        payload["scientific_payload_sha256"]
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        save_gemma3_state_conditioned_modal_graph_candidate(
            output,
            **arguments,
        )


def test_unpromoted_candidate_is_strict_and_source_safe() -> None:
    payload = _payload(promoted=False)

    assert payload["compiler_pipeline"] is None
    assert payload["lineage"]["compiler_pipeline_sha256"] is None
    assert payload["lineage"]["interaction_promotion_sha256"] is None
    assert payload["safety"]["contains_promoted_compiler_pipeline"] is False
    build_gemma3_state_conditioned_modal_graph_report(
        payload,
        tensor_file="unpromoted.pt",
    )


def test_strict_loader_rejects_nested_and_outer_tampering(
    tmp_path: Path,
) -> None:
    payload = _payload()

    lowering = copy.deepcopy(payload)
    lowering["lowering_records"][0]["lowering"]["graph_weights"][
        "input_factor"
    ][0, 0] += 1.0
    lowering_path = tmp_path / "lowering.pt"
    _save_raw(lowering_path, lowering)
    with pytest.raises(ValueError, match="input_factor hash mismatch"):
        load_gemma3_state_conditioned_modal_graph_candidate(lowering_path)

    gate = copy.deepcopy(payload)
    gate["dynamic_graph"]["interactions"][0]["gate_weight"][0] += 1.0
    gate_path = tmp_path / "gate.pt"
    _save_raw(gate_path, gate)
    with pytest.raises(ValueError, match="gate_weight hash mismatch"):
        load_gemma3_state_conditioned_modal_graph_candidate(gate_path)

    metadata = copy.deepcopy(payload)
    metadata["config"]["top_k"] = 2
    metadata_path = tmp_path / "metadata.pt"
    _save_raw(metadata_path, metadata)
    with pytest.raises(ValueError, match="scientific payload hash mismatch"):
        load_gemma3_state_conditioned_modal_graph_candidate(metadata_path)

    unknown_root = copy.deepcopy(payload)
    unknown_root["shadow"] = {}
    root_path = tmp_path / "unknown-root.pt"
    _save_raw(root_path, unknown_root)
    with pytest.raises(ValueError, match="fields are invalid"):
        load_gemma3_state_conditioned_modal_graph_candidate(root_path)

    unknown_record = copy.deepcopy(payload)
    unknown_record["lowering_records"][0]["shadow"] = "forbidden"
    record_path = tmp_path / "unknown-record.pt"
    _save_raw(record_path, unknown_record)
    with pytest.raises(ValueError, match="lowering_records.*fields"):
        load_gemma3_state_conditioned_modal_graph_candidate(record_path)


def test_builder_rejects_graph_lowering_and_pipeline_lineage_drift() -> None:
    arguments = _arguments()
    dynamic = arguments["dynamic_graph"]

    altered_node = replace(
        dynamic.nodes[0],
        input_boundary="different.input",
        artifact_sha256="",
    )
    altered_edgeless = ModalGeneratorGraphPlan(
        model_fingerprint=dynamic.model_fingerprint,
        parameter_cluster_plan_sha256=(
            dynamic.parameter_cluster_plan_sha256
        ),
        nodes=(altered_node, *dynamic.nodes[1:]),
        interactions=(),
    )
    with pytest.raises(ValueError, match="identical nodes"):
        build_gemma3_state_conditioned_modal_graph_candidate(
            **{
                **arguments,
                "edgeless_graph": altered_edgeless,
            },  # type: ignore[arg-type]
        )

    lowerings = dict(arguments["lowerings_by_node"])
    lowerings["node.0"], lowerings["node.1"] = (
        lowerings["node.1"],
        lowerings["node.0"],
    )
    with pytest.raises(ValueError, match="reconstruct the exact graph node"):
        build_gemma3_state_conditioned_modal_graph_candidate(
            **{
                **arguments,
                "lowerings_by_node": lowerings,
            },  # type: ignore[arg-type]
        )

    other_pipeline, _ = _conditional_compiled(mixed=True)
    with pytest.raises(ValueError, match="does not contain the dynamic graph"):
        build_gemma3_state_conditioned_modal_graph_candidate(
            **{
                **arguments,
                "compiler_pipeline": other_pipeline,
            },  # type: ignore[arg-type]
        )


def test_metadata_is_json_only_source_safe_and_bound_to_artifacts() -> None:
    arguments = _arguments()

    with pytest.raises(ValueError, match="forbidden field.*prompt_text"):
        build_gemma3_state_conditioned_modal_graph_candidate(
            **{
                **arguments,
                "experiment": {"prompt_text": "secret"},
            },  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Tensor rows or weights"):
        build_gemma3_state_conditioned_modal_graph_candidate(
            **{
                **arguments,
                "config": {"hidden": torch.ones(1)},
            },  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-JSON-safe"):
        build_gemma3_state_conditioned_modal_graph_candidate(
            **{
                **arguments,
                "selection": {"audit_note": "free text is not retained"},
            },  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="contradicts nested artifacts"):
        build_gemma3_state_conditioned_modal_graph_candidate(
            **{
                **arguments,
                "resources": {"dynamic_parameter_count": 1},
            },  # type: ignore[arg-type]
        )


def test_promotion_metadata_cannot_contradict_pipeline() -> None:
    promoted = _arguments(promoted=True)
    with pytest.raises(
        ValueError,
        match="promotion_passed contradicts compiler pipeline",
    ):
        build_gemma3_state_conditioned_modal_graph_candidate(
            **{
                **promoted,
                "selection": {
                    **promoted["selection"],
                    "promotion_passed": False,
                },
            },  # type: ignore[arg-type]
        )

    unpromoted = _arguments(promoted=False)
    with pytest.raises(
        ValueError,
        match="promotion_passed contradicts compiler pipeline",
    ):
        build_gemma3_state_conditioned_modal_graph_candidate(
            **{
                **unpromoted,
                "selection": {
                    **unpromoted["selection"],
                    "promotion_passed": True,
                },
            },  # type: ignore[arg-type]
        )

    missing_hash = dict(promoted["selection"])
    del missing_hash["compiler_pipeline_sha256"]
    with pytest.raises(ValueError, match="promotion hashes.*together"):
        build_gemma3_state_conditioned_modal_graph_candidate(
            **{
                **promoted,
                "selection": missing_hash,
            },  # type: ignore[arg-type]
        )


def test_report_preserves_optional_control_availability() -> None:
    arguments = _arguments(promoted=False)
    selection = {
        **arguments["selection"],
        "variants": (
            {
                "behavior": {
                    "conditions": {
                        "matched_deletion": {"nll_per_token": 5.25},
                    },
                    "graph_comparison": {
                        "deletion_paths_agree": True,
                        "nodewise_dense_supplied": False,
                        "nodewise_dense_agrees_with_edgeless": None,
                        "nodewise_dense_equivalence_scope": None,
                    },
                },
                "flow": {
                    "conditions": {
                        "dense_all_target": {"weighted_nrmse": 1.1},
                        "constant_oracle_majority": {
                            "weighted_nrmse": 1.0,
                        },
                    },
                },
            },
        ),
    }
    payload = build_gemma3_state_conditioned_modal_graph_candidate(
        **{
            **arguments,
            "selection": selection,
        },  # type: ignore[arg-type]
    )
    report = build_gemma3_state_conditioned_modal_graph_report(
        payload,
        tensor_file="candidate.pt",
    )

    variant = report["selection"]["variants"][0]
    comparison = variant["behavior"]["graph_comparison"]
    assert comparison["deletion_paths_agree"] is True
    assert comparison["nodewise_dense_supplied"] is False
    assert comparison["nodewise_dense_agrees_with_edgeless"] is None
    assert comparison["nodewise_dense_equivalence_scope"] is None
    assert variant["behavior"]["conditions"]["matched_deletion"] == {
        "nll_per_token": 5.25,
    }
    assert variant["flow"]["conditions"]["dense_all_target"] == {
        "weighted_nrmse": 1.1,
    }
    assert variant["flow"]["conditions"]["constant_oracle_majority"] == {
        "weighted_nrmse": 1.0,
    }


def test_report_and_file_suffixes_are_strict(tmp_path: Path) -> None:
    payload = _payload()

    with pytest.raises(ValueError, match="source-safe .pt basename"):
        build_gemma3_state_conditioned_modal_graph_report(
            payload,
            tensor_file="nested/candidate.pt",
        )
    with pytest.raises(ValueError, match="must use .pt"):
        save_gemma3_state_conditioned_modal_graph_candidate(
            tmp_path / "candidate.bin",
            **_arguments(),
        )
    with pytest.raises(ValueError, match="must use .pt"):
        load_gemma3_state_conditioned_modal_graph_candidate(
            tmp_path / "candidate.bin"
        )
