from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from fisher_graph import gemma3_l10_l17_a5d_report as report


def _digest(seed: int) -> str:
    return f"{seed:064x}"


def _patch_receipt_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        report,
        "validate_a5d_source_anchored_residual_receipt",
        lambda value: copy.deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        report,
        "validate_a5d_family_residual_cv_receipt",
        lambda value: copy.deepcopy(dict(value)),
    )


def _target_receipt() -> dict[str, object]:
    return {
        "schema": report.GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_SCHEMA,
        "receipt_sha256": _digest(100),
        "construction": {
            "observations": 1_798,
            "sequences": 28,
            "residual_width": 640,
            "joint_coordinate_width": 182,
            "node_order": list(report._EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE),
            "source_mapping_preserved_at_alpha_zero": True,
            "projection_uses_affine_means": False,
        },
        "rows": {"source_affine_means_injected": False},
    }


def _cv_receipt(*, positive: bool) -> dict[str, object]:
    target = _target_receipt()
    lowerings = {
        name: _digest(200 + index)
        for index, name in enumerate(
            report._EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE
        )
    }
    return {
        "schema": report.GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_SCHEMA,
        "receipt_sha256": _digest(101),
        "source": {
            "residual_target_receipt_sha256": target["receipt_sha256"],
            "source_graph_sha256": report._EXPECTED_LAYER17_GRAPH_SHA256,
        },
        "configuration": {
            "ridge_grid": list(report.A5D_RIDGE_GRID),
            "alpha_grid": list(report.A5D_ALPHA_GRID),
            "inner_fold_count": 7,
        },
        "ownership": {"outer_held_family_states_or_rows_accessed": False},
        "selection": {
            "selected_alpha": 0.25 if positive else 0.0,
            "selected_ridge": 1.0e-4 if positive else None,
            "use_frozen_fallback": not positive,
        },
        "final_refit": {
            "fit": (
                {
                    "graph_sha256": _digest(102),
                    "lowering_sha256_by_node": lowerings,
                    "parameter_count": 160_534,
                    "macs_per_token": 160_352,
                    "removed_source_affine_mean_parameters": 2_560,
                }
                if positive
                else None
            )
        },
    }


def _resource(
    *,
    nodes: int,
    interactions: int,
    parameters: int,
    macs: int,
    additions: int,
    peak: int,
) -> dict[str, int]:
    return {
        "node_count": nodes,
        "interaction_count": interactions,
        "parameter_count": parameters,
        "macs_per_token": macs,
        "additions_per_token": additions,
        "peak_live_modal_width": peak,
    }


def _source_owner() -> dict[str, object]:
    return {
        "layer17_graph_sha256": report._EXPECTED_LAYER17_GRAPH_SHA256,
        "layer17_lowering_sha256_by_node": dict(
            report._EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE
        ),
        "composition_graph_sha256": report._EXPECTED_COMPOSITION_GRAPH_SHA256,
        "layer17_resources": _resource(
            nodes=4,
            interactions=0,
            parameters=163_094,
            macs=160_352,
            additions=620,
            peak=48,
        ),
        "composition_resources": _resource(
            nodes=8,
            interactions=3,
            parameters=295_129,
            macs=289_600,
            additions=1_120,
            peak=48,
        ),
    }


def _lineage() -> dict[str, object]:
    return {
        "a5c_report_sha256": report._EXPECTED_A5C_REPORT_SHA256,
        "capture_sha256": report._EXPECTED_A5C_CAPTURE["capture_sha256"],
        "target_solve_receipt_sha256": (
            report._EXPECTED_TARGET_SOLVE_RECEIPT_SHA256
        ),
        "coordinate_row_bank_receipt_sha256": (
            report._EXPECTED_COORDINATE_ROW_BANK_RECEIPT_SHA256
        ),
        "breadth_split_receipt_sha256": (
            report._EXPECTED_BREADTH_SPLIT_RECEIPT_SHA256
        ),
        "source_anchored_residual_receipt_sha256": _digest(100),
        "residual_cv_receipt_sha256": _digest(101),
        "layer10_graph_sha256": report._EXPECTED_LAYER10_GRAPH_SHA256,
        "layer10_lowering_sha256_by_node": dict(
            report._EXPECTED_LAYER10_LOWERING_SHA256_BY_NODE
        ),
        "matched_double_deletion_graph_sha256": (
            report._EXPECTED_COMPOSITION_GRAPH_SHA256
        ),
    }


def _executable(*, positive: bool) -> dict[str, object]:
    cv = report.compact_a5d_family_residual_cv_receipt(
        _cv_receipt(positive=positive)
    )
    source = _source_owner()
    source_layer17 = source["layer17_resources"]
    source_composition = source["composition_resources"]
    assert isinstance(source_layer17, dict) and isinstance(source_composition, dict)
    additive = None
    if positive:
        additive = {
            "graph_sha256": cv["final_residual_graph_sha256"],
            "lowering_sha256_by_node": dict(
                cv["final_residual_lowering_sha256_by_node"]
            ),
            "application_layer_ordinal": 17,
            "basis_means_exactly_zero": True,
            "source_decoders_reused": True,
            "source_affine_means_reinjected": False,
            "resources": _resource(
                nodes=4,
                interactions=0,
                parameters=160_534,
                macs=160_352,
                additions=620,
                peak=48,
            ),
        }
    additive_resources = None if additive is None else additive["resources"]
    assert additive_resources is None or isinstance(additive_resources, dict)
    totals = {
        "layer17_scope": report._sum_resource(
            source_layer17, additive_resources
        ),
        "composition_scope": report._sum_resource(
            source_composition, additive_resources
        ),
    }
    value: dict[str, object] = {
        "kind": report._ADDITIVE if positive else report._FROZEN,
        "selected_alpha": 0.25 if positive else 0.0,
        "selected_alpha_hex": (0.25 if positive else 0.0).hex(),
        "selected_ridge": 1.0e-4 if positive else None,
        "selected_ridge_hex": (1.0e-4).hex() if positive else None,
        "application_boundary": report._OUTPUT_BOUNDARY,
        "application_order": report._APPLICATION_ORDER,
        "source_ownership_preserved": True,
        "source_affine_means_reinjected": False,
        "lineage": _lineage(),
        "source_owner": source,
        "additive_residual": additive,
        "selected_resources": totals,
    }
    value["selection_freeze_sha256"] = report.a5d_selection_freeze_sha256(
        kind=value["kind"],
        selected_alpha=value["selected_alpha"],
        selected_ridge=value["selected_ridge"],
        application_boundary=value["application_boundary"],
        application_order=value["application_order"],
        source_ownership_preserved=value["source_ownership_preserved"],
        source_affine_means_reinjected=value[
            "source_affine_means_reinjected"
        ],
        lineage=value["lineage"],
        source_owner=value["source_owner"],
        additive_residual=value["additive_residual"],
        selected_resources=value["selected_resources"],
    )
    return value


def _metric(native: float, delta: float, *, kl: float, top1: float) -> dict[str, float]:
    return {
        "nll_per_token": native + delta,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _execution_row(
    *,
    source_parameters: int,
    owner: dict[str, int],
    additive: dict[str, int] | None,
    replaced: int,
    removed: int,
    deletion: bool = False,
) -> dict[str, int]:
    total = report._sum_resource(owner, additive)
    executed_macs = 0 if deletion else total["macs_per_token"]
    executed_additions = 0 if deletion else total["additions_per_token"]
    return {
        "replaced_layer_count": replaced,
        "owning_graph_node_count": owner["node_count"],
        "additive_graph_node_count": 0 if additive is None else additive["node_count"],
        "total_graph_node_count": total["node_count"],
        "owning_interaction_count": owner["interaction_count"],
        "additive_interaction_count": (
            0 if additive is None else additive["interaction_count"]
        ),
        "total_interaction_count": total["interaction_count"],
        "native_removed_parameters": removed,
        "owning_graph_parameters": owner["parameter_count"],
        "additive_graph_parameters": (
            0 if additive is None else additive["parameter_count"]
        ),
        "total_graph_parameters": total["parameter_count"],
        "candidate_whole_model_learned_parameters": (
            source_parameters - removed + total["parameter_count"]
        ),
        "net_parameter_savings": removed - total["parameter_count"],
        "owning_dense_graph_macs_per_token": owner["macs_per_token"],
        "additive_dense_graph_macs_per_token": (
            0 if additive is None else additive["macs_per_token"]
        ),
        "total_dense_graph_macs_per_token": total["macs_per_token"],
        "executed_graph_macs_per_token": executed_macs,
        "net_executed_macs_saved_per_token": removed - executed_macs,
        "owning_dense_graph_additions_per_token": owner["additions_per_token"],
        "additive_dense_graph_additions_per_token": (
            0 if additive is None else additive["additions_per_token"]
        ),
        "total_dense_graph_additions_per_token": total["additions_per_token"],
        "executed_graph_additions_per_token": executed_additions,
        "executed_peak_live_modal_width": (
            0 if deletion else total["peak_live_modal_width"]
        ),
    }


def _evaluation(executable: dict[str, object], *, positive: bool) -> dict[str, object]:
    source_parameters = 170_000_000
    source = executable["source_owner"]
    assert isinstance(source, dict)
    layer17 = source["layer17_resources"]
    composition = source["composition_resources"]
    assert isinstance(layer17, dict) and isinstance(composition, dict)
    additive_mapping = executable["additive_residual"]
    additive = None
    additive_graph = None
    if additive_mapping is not None:
        assert isinstance(additive_mapping, dict)
        additive = additive_mapping["resources"]
        additive_graph = additive_mapping["graph_sha256"]
        assert isinstance(additive, dict)
    layer10 = _resource(
        nodes=4,
        interactions=3,
        parameters=132_035,
        macs=129_248,
        additions=500,
        peak=32,
    )
    native = 7.0
    frozen_metric = _metric(native, 0.14, kl=0.10, top1=0.75)
    selected_metric = (
        _metric(native, 0.08, kl=0.05, top1=0.85)
        if positive
        else dict(frozen_metric)
    )

    def condition(
        metric: dict[str, float], owner_graph: str, residual_graph: object = None
    ) -> dict[str, object]:
        return {
            **metric,
            "owning_graph_sha256": owner_graph,
            "additive_graph_sha256": residual_graph,
        }

    selected_composition_resource = _execution_row(
        source_parameters=source_parameters,
        owner=composition,
        additive=additive,
        replaced=2,
        removed=1_082_880,
    )
    return {
        "assessment_role": report._ASSESSMENT_ROLE,
        "outer_fold_index": 0,
        "logical_valid_tokens": 36,
        "supervised_tokens": 35,
        "source_whole_model_learned_parameters": source_parameters,
        "native": {"nll_per_token": native},
        "conditions": {
            "layer10_only": condition(
                _metric(native, 0.09, kl=0.05, top1=0.85),
                report._EXPECTED_LAYER10_GRAPH_SHA256,
            ),
            "selected_layer17_only": condition(
                _metric(native, 0.04, kl=0.02, top1=0.9),
                report._EXPECTED_LAYER17_GRAPH_SHA256,
                additive_graph,
            ),
            "frozen_uncorrected_composition": condition(
                frozen_metric, report._EXPECTED_COMPOSITION_GRAPH_SHA256
            ),
            "selected_composition": condition(
                selected_metric,
                report._EXPECTED_COMPOSITION_GRAPH_SHA256,
                additive_graph,
            ),
            "matched_double_deletion": condition(
                _metric(native, 0.9, kl=1.0, top1=0.4),
                report._EXPECTED_COMPOSITION_GRAPH_SHA256,
                additive_graph,
            ),
        },
        "resource_accounting": {
            "layer10_only": _execution_row(
                source_parameters=source_parameters,
                owner=layer10,
                additive=None,
                replaced=1,
                removed=641_280,
            ),
            "selected_layer17_only": _execution_row(
                source_parameters=source_parameters,
                owner=layer17,
                additive=additive,
                replaced=1,
                removed=441_600,
            ),
            "frozen_uncorrected_composition": _execution_row(
                source_parameters=source_parameters,
                owner=composition,
                additive=None,
                replaced=2,
                removed=1_082_880,
            ),
            "selected_composition": selected_composition_resource,
            "matched_double_deletion": {
                **selected_composition_resource,
                "executed_graph_macs_per_token": 0,
                "net_executed_macs_saved_per_token": 1_082_880,
                "executed_graph_additions_per_token": 0,
                "executed_peak_live_modal_width": 0,
            },
        },
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "heldout_confirmation": False,
        "exact_resources_match_frozen_executable": True,
        "latency_or_kernel_speed_claim": False,
    }


def _inputs(*, positive: bool) -> dict[str, object]:
    target_receipt = _target_receipt()
    cv_receipt = _cv_receipt(positive=positive)
    target = report.compact_a5d_source_anchored_residual_receipt(target_receipt)
    cv = report.compact_a5d_family_residual_cv_receipt(cv_receipt)
    executable = _executable(positive=positive)
    evaluation = _evaluation(executable, positive=positive)
    return {
        "source_bindings": {
            "a5c_file_sha256": report._EXPECTED_A5C_FILE_SHA256,
            "a5c_report_sha256": report._EXPECTED_A5C_REPORT_SHA256,
        },
        "runtime": {
            "model_id": report._EXPECTED_MODEL_ID,
            "requested_revision": report._EXPECTED_MODEL_REVISION,
            "model_fingerprint": report._EXPECTED_MODEL_FINGERPRINT,
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
        },
        "configuration": {
            "outer_fold_index": 0,
            "training_family_count": 7,
            "training_examples_per_family": 4,
            "row_selection_policy": "all_valid_captured_rows_per_example",
            "generator_rank": report.A5D_FIXED_GENERATOR_RANK,
            "ridge_grid": list(report.A5D_RIDGE_GRID),
            "alpha_grid": list(report.A5D_ALPHA_GRID),
            "held_examples_scored": 1,
            "output_boundary": report._OUTPUT_BOUNDARY,
            "final_head_chunk_rows": 8,
        },
        "capture": dict(report._EXPECTED_A5C_CAPTURE),
        "source_anchored_residual": target,
        "residual_cv": cv,
        "evidence_receipts": {
            "source_anchored_residual": target_receipt,
            "residual_cv": cv_receipt,
        },
        "selected_executable": executable,
        "chronology": {
            "residual_cv_completed_event": 1,
            "executable_frozen_event": 2,
            "outer_held_batch_selected_event": 3,
            "outer_held_model_evaluated_event": 4,
            "outer_held_batch_selected_or_scored_before_freeze": False,
            "executable_frozen_before_outer_held_batch_selection": True,
            "executable_frozen_before_outer_held_model_evaluation": True,
            "residual_cv_receipt_sha256": cv["receipt_sha256"],
            "selection_freeze_sha256": executable["selection_freeze_sha256"],
            "outer_evaluation_sha256": report.a5d_outer_evaluation_sha256(
                evaluation
            ),
        },
        "outer_evaluation": evaluation,
        "comparison_to_a5c": {
            "a5c_file_sha256": report._EXPECTED_A5C_FILE_SHA256,
            "a5c_report_sha256": report._EXPECTED_A5C_REPORT_SHA256,
            "same_outer_fold": True,
            "same_held_example_policy": True,
            "a5c_frozen_composition": dict(
                report._EXPECTED_A5C_COMPOSITION_METRIC
            ),
        },
    }


def _build(monkeypatch: pytest.MonkeyPatch, *, positive: bool) -> dict[str, object]:
    _patch_receipt_validators(monkeypatch)
    return report.build_gemma3_l10_l17_a5d_report(**_inputs(positive=positive))


def _rehash(value: dict[str, object]) -> None:
    value.pop("report_sha256", None)
    value["report_sha256"] = report._sha256(report._REPORT_DOMAIN, value)


def _refresh_freeze(value: dict[str, object]) -> None:
    executable = value["selected_executable"]
    assert isinstance(executable, dict)
    executable["selection_freeze_sha256"] = report.a5d_selection_freeze_sha256(
        kind=executable["kind"],
        selected_alpha=executable["selected_alpha"],
        selected_ridge=executable["selected_ridge"],
        application_boundary=executable["application_boundary"],
        application_order=executable["application_order"],
        source_ownership_preserved=executable["source_ownership_preserved"],
        source_affine_means_reinjected=executable[
            "source_affine_means_reinjected"
        ],
        lineage=executable["lineage"],
        source_owner=executable["source_owner"],
        additive_residual=executable["additive_residual"],
        selected_resources=executable["selected_resources"],
    )
    chronology = value["chronology"]
    assert isinstance(chronology, dict)
    chronology["selection_freeze_sha256"] = executable[
        "selection_freeze_sha256"
    ]
    _rehash(value)


@pytest.mark.parametrize("positive", [False, True])
def test_report_builds_both_executable_branches(
    monkeypatch: pytest.MonkeyPatch, positive: bool
) -> None:
    value = _build(monkeypatch, positive=positive)

    assert value["schema"] == report.GEMMA3_L10_L17_A5D_REPORT_SCHEMA
    executable = value["selected_executable"]
    assert isinstance(executable, dict)
    assert (executable["additive_residual"] is not None) is positive
    assert value["conclusion"]["additive_residual_deployed"] is positive
    assert report.validate_gemma3_l10_l17_a5d_report(value) == value


def test_alpha_zero_requires_exact_omitted_additive_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _build(monkeypatch, positive=False)
    executable = value["selected_executable"]
    assert isinstance(executable, dict)
    assert executable["selected_alpha"] == 0.0
    assert executable["selected_ridge"] is None
    assert executable["additive_residual"] is None
    assert executable["selected_resources"] == {
        "layer17_scope": executable["source_owner"]["layer17_resources"],
        "composition_scope": executable["source_owner"]["composition_resources"],
    }
    assert (
        value["outer_evaluation"]["conditions"]["selected_composition"]
        == value["outer_evaluation"]["conditions"][
            "frozen_uncorrected_composition"
        ]
    )

    tampered = copy.deepcopy(value)
    tampered["selected_executable"]["additive_residual"] = {
        "graph_sha256": _digest(999)
    }
    _rehash(tampered)
    with pytest.raises(ValueError, match="fields|fallback|additive"):
        report.validate_gemma3_l10_l17_a5d_report(tampered)


def test_positive_residual_requires_exact_owner_plus_additive_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _build(monkeypatch, positive=True)
    executable = value["selected_executable"]
    assert isinstance(executable, dict)
    selected = executable["selected_resources"]
    owner = executable["source_owner"]
    additive = executable["additive_residual"]
    assert isinstance(selected, dict)
    assert isinstance(owner, dict)
    assert isinstance(additive, dict)
    assert selected["composition_scope"]["parameter_count"] == (
        owner["composition_resources"]["parameter_count"]
        + additive["resources"]["parameter_count"]
    )

    tampered = copy.deepcopy(value)
    tampered["selected_executable"]["selected_resources"][
        "composition_scope"
    ]["parameter_count"] += 1
    with pytest.raises(ValueError, match="sum owner plus additive"):
        _refresh_freeze(tampered)


def test_self_rehashed_semantic_and_chronology_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _build(monkeypatch, positive=True)
    tampered = copy.deepcopy(value)
    tampered["selected_executable"]["source_ownership_preserved"] = False
    _rehash(tampered)
    with pytest.raises(ValueError, match="application semantics"):
        report.validate_gemma3_l10_l17_a5d_report(tampered)

    tampered = copy.deepcopy(value)
    tampered["chronology"]["outer_held_batch_selected_event"] = 1
    _rehash(tampered)
    with pytest.raises(ValueError, match="chronology"):
        report.validate_gemma3_l10_l17_a5d_report(tampered)


def test_compact_receipts_are_crosslinked_to_full_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _build(monkeypatch, positive=True)
    tampered = copy.deepcopy(value)
    tampered["source_anchored_residual"]["receipt_sha256"] = _digest(999)
    _rehash(tampered)
    with pytest.raises(ValueError, match="compact receipts"):
        report.validate_gemma3_l10_l17_a5d_report(tampered)


def test_tensor_sensitive_and_canonical_source_tamper_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_receipt_validators(monkeypatch)
    inputs = _inputs(positive=False)
    inputs["capture"] = {**inputs["capture"], "tensor": torch.zeros(1)}
    with pytest.raises(TypeError, match="tensor payload"):
        report.build_gemma3_l10_l17_a5d_report(**inputs)

    value = _build(monkeypatch, positive=False)
    tampered = copy.deepcopy(value)
    tampered["source_bindings"]["a5c_report_sha256"] = _digest(999)
    _rehash(tampered)
    with pytest.raises(ValueError, match="canonical A5c"):
        report.validate_gemma3_l10_l17_a5d_report(tampered)


def test_save_load_roundtrip_is_atomic_and_refuses_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = _build(monkeypatch, positive=True)
    destination = tmp_path / "a5d.json"

    assert report.save_gemma3_l10_l17_a5d_report(destination, value) == value
    assert report.load_gemma3_l10_l17_a5d_report(destination) == value
    with pytest.raises(FileExistsError, match="overwrite"):
        report.save_gemma3_l10_l17_a5d_report(destination, value)


def test_load_rejects_nonfinite_json(tmp_path: Path) -> None:
    destination = tmp_path / "a5d.json"
    destination.write_text(json.dumps({"value": float("nan")}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        report.load_gemma3_l10_l17_a5d_report(destination)
