from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_development as development,
)
from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_plan as plan_module,
)
from fisher_graph.gemma3_l3_l4_iterative_generator_innovation import (
    GENERATOR_INNOVATION_TANGENT_ORDER,
)
from fisher_graph.gemma3_l3_l4_iterative_generator_innovation_edges import (
    GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER,
)
from fisher_graph.gemma3_l3_l4_progressive_worker import (
    gemma_progressive_panel_membership_receipt_sha256,
)
from fisher_graph.token_loss_fisher import (
    COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES,
    build_token_loss_fisher_prompt_record,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _frozen_plan(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(
        plan_module,
        "validate_gemma_iterative_token_fisher_development_report",
        lambda _value: None,
    )
    monkeypatch.setattr(
        plan_module,
        "validate_gemma_iterative_fisher_corrective_development_report",
        lambda _value: None,
    )
    random = torch.Generator().manual_seed(391)
    latent = torch.randn(256, 2, generator=random, dtype=torch.float64)
    scores = 0.08 * torch.randn(
        256,
        8,
        generator=random,
        dtype=torch.float64,
    )
    scores[:, 0] += latent[:, 0]
    scores[:, 2] += 0.55 * latent[:, 0]
    scores[:, 4] -= 0.70 * latent[:, 0]
    scores[:, 1] += latent[:, 1]
    scores[:, 3] += 0.45 * latent[:, 1]
    scores[:, 5] -= 0.65 * latent[:, 1]
    records = tuple(
        build_token_loss_fisher_prompt_record(
            example_id=f"basis-example-{index:02d}",
            family_id=f"basis-family-{index % 8}",
            coordinate_names=(
                COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES
            ),
            token_scores=scores,
            compensation_target=torch.zeros(256, dtype=torch.float64),
        )
        for index in range(16)
    )
    selected = torch.tensor((0, 1, 2, 3, 4, 5), dtype=torch.int64)
    fisher = torch.tensor(
        records[0].fisher_second_moment,
        dtype=torch.float64,
    ).index_select(0, selected).index_select(1, selected)
    token_report: dict[str, object] = {
        "schema": "test.token-fisher",
        "report_sha256": _sha("token-report"),
        "lineage": {
            "parent_artifact_sha256": _sha("parent"),
            "parent_h4_head_sha256": _sha("h4"),
            "accepted_x4_head_sha256": _sha("x4"),
            "bridge_binding_sha256": _sha("bridge"),
            "model_sha256": _sha("model"),
            "adapter_execution_sha256": _sha("adapter"),
            "fit_manifest_sha256": _sha("parent-fit-manifest"),
            "factorial_report_sha256": _sha("factorial"),
            "factorial_report_file_sha256": _sha("factorial-file"),
        },
        "prompt_fisher_records": tuple(row.to_dict() for row in records),
        "analysis": {
            "cumulative_coupling_graph": {
                "coordinate_names": (
                    COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES[:6]
                ),
                "family_balanced_fisher_second_moment": tuple(
                    tuple(float(value) for value in row) for row in fisher
                ),
            }
        },
    }
    corrective_report: dict[str, object] = {
        "schema": "test.corrective",
        "report_sha256": _sha("corrective-report"),
        "lineage": {
            "token_fisher_report_sha256": token_report["report_sha256"],
            "token_fisher_report_file_sha256": _sha("token-report-file"),
        },
        "decision": {
            "next_step": (
                "collect_preregistered_new_causal_feature_on_"
                "new_family_disjoint_data"
            ),
            "provider_compiled": False,
            "runtime_claim_authorized": False,
            "fresh_confirmation_authorized": False,
        },
    }
    return plan_module.build_gemma_iterative_generator_innovation_plan(
        token_fisher_report=token_report,
        token_fisher_report_file_sha256=_sha("token-report-file"),
        corrective_report=corrective_report,
        corrective_report_file_sha256=_sha("corrective-report-file"),
    )


def _new_panel_records() -> tuple[tuple[object, ...], tuple[object, ...]]:
    random = torch.Generator().manual_seed(509)
    legacy_records: list[object] = []
    generator_records: list[object] = []
    for index in range(16):
        generator_scores = torch.randn(
            48,
            4,
            generator=random,
            dtype=torch.float64,
        )
        generator_scores[:, 2:] += 0.2 * generator_scores[:, :2]
        target = (
            generator_scores
            @ torch.tensor(
                (0.04, -0.03, 0.025, -0.02),
                dtype=torch.float64,
            )
            + 0.01
            * torch.randn(48, generator=random, dtype=torch.float64)
        )
        legacy_scores = torch.randn(
            48,
            6,
            generator=random,
            dtype=torch.float64,
        )
        legacy_scores[:, :2] = generator_scores[:, :2]
        example_id = f"new-example-{index:02d}"
        family_id = f"new-family-{index % 8}"
        legacy_records.append(
            build_token_loss_fisher_prompt_record(
                example_id=example_id,
                family_id=family_id,
                coordinate_names=(
                    GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER
                ),
                token_scores=legacy_scores,
                compensation_target=target,
            )
        )
        generator_records.append(
            build_token_loss_fisher_prompt_record(
                example_id=example_id,
                family_id=family_id,
                coordinate_names=GENERATOR_INNOVATION_TANGENT_ORDER,
                token_scores=generator_scores,
                compensation_target=target,
            )
        )
    return tuple(legacy_records), tuple(generator_records)


def _feature_summary(example_id: str) -> dict[str, object]:
    return {
        "active_activation_row_count": 48,
        "mean_by_channel": (0.0, 0.0),
        "second_moment_by_channel": (0.25, 0.16),
        "mean_absolute_by_channel": (0.5, 0.4),
        "maximum_absolute_by_channel": (0.5, 0.4),
        "positive_count_by_channel": (24, 24),
        "negative_count_by_channel": (24, 24),
        "zero_count_by_channel": (0, 0),
        "bounded_innovation_trace_sha256": _sha(
            f"bounded-feature-{example_id}"
        ),
        "whole_sequence_equals_two_chunks": True,
        "prior_excludes_current_activation": True,
        "padding_updates_state": False,
    }


def _build_report(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object], str]:
    plan = _frozen_plan(monkeypatch)
    plan_file_sha256 = _sha("plan-file")
    legacy, generator = _new_panel_records()
    family_by_example = {
        row.example_id: row.family_id for row in generator
    }
    manifest_sha256 = _sha("new-panel-manifest")
    membership_sha256 = (
        gemma_progressive_panel_membership_receipt_sha256(
            role="calibration_a_fit",
            manifest_sha256=manifest_sha256,
            family_by_example=family_by_example,
        )
    )
    basis = plan["frozen_generator_basis"]
    assert isinstance(basis, Mapping)
    collection_lineage = {
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": plan_file_sha256,
        "basis_sha256": basis["basis_sha256"],
        "collection_role_input_file_sha256": _sha("private-role"),
        "collection_manifest_sha256": manifest_sha256,
        "collection_membership_receipt_sha256": membership_sha256,
        "prompt_free_panel_artifact_receipt_sha256": _sha(
            "prompt-free-receipt"
        ),
    }
    planned_parent = plan["lineage"]
    assert isinstance(planned_parent, Mapping)
    planned_parent = planned_parent[
        "token_fisher_model_and_parent_lineage"
    ]
    assert isinstance(planned_parent, Mapping)
    lineage = {
        **planned_parent,
        **collection_lineage,
    }
    example_ids = tuple(row.example_id for row in generator)
    report = (
        development
        .build_gemma_iterative_generator_innovation_development_report(
            legacy_records=legacy,
            generator_records=generator,
            plan=plan,
            plan_file_sha256=plan_file_sha256,
            feature_summary_by_example={
                example_id: _feature_summary(example_id)
                for example_id in example_ids
            },
            top_mode_receipt_by_example={
                example_id: {
                    "top_mode_indices": (0, 1),
                    "top_mode_norms": (1.25, 0.75),
                }
                for example_id in example_ids
            },
            token_vjp_artifact_sha256_by_example={
                example_id: _sha(f"vjp-{example_id}")
                for example_id in example_ids
            },
            source_tangent_record_sha256_by_example={
                example_id: _sha(f"tangent-{example_id}")
                for example_id in example_ids
            },
            total_backward_call_count=16 * 6,
            vjp_chunk_size=8,
            lineage=lineage,
            collection_lineage=collection_lineage,
        )
    )
    return report, plan, plan_file_sha256


def _all_mapping_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        result.update(str(key) for key in value)
        for child in value.values():
            result.update(_all_mapping_keys(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            result.update(_all_mapping_keys(child))
    return result


def test_development_report_builds_validates_and_exactly_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, plan, plan_file_sha256 = _build_report(monkeypatch)

    development.validate_gemma_iterative_generator_innovation_development_report(
        report
    )
    assert (
        development
        .replay_gemma_iterative_generator_innovation_development_report(
            report=report,
            plan=plan,
            plan_file_sha256=plan_file_sha256,
        )
        == report
    )
    assert report["resources"] == {
        "fit_example_count": 16,
        "fit_family_count": 8,
        "examples_per_family": 2,
        "supervised_token_count": 16 * 48,
        "source_forward_count": 16,
        "retained_parent_token_vjp_forward_count": 16,
        "total_model_forward_count": 32,
        "model_forward_count_per_example": 2,
        "token_vjp_backward_call_count": 16 * 6,
        "token_vjp_chunk_size": 8,
        "candidate_forward_count": 0,
        "finite_displacement_forward_count": 0,
        "fresh_shadow_forward_count": 0,
    }
    assert report["feature_audit"]["bounded_feature_rows_retained"] is False
    forbidden_raw_keys = {
        "prompt_text",
        "token_ids",
        "logits",
        "modal_rows",
        "feature_rows",
        "activation_rows",
        "gradient_rows",
        "token_score_rows",
    }
    assert _all_mapping_keys(report).isdisjoint(forbidden_raw_keys)


def test_report_privacy_and_forward_accounting_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _, _ = _build_report(monkeypatch)
    privacy_tamper = copy.deepcopy(report)
    privacy_tamper["audit"]["raw_token_ids_retained"] = True
    with pytest.raises(ValueError, match="safety audit differs"):
        development.validate_gemma_iterative_generator_innovation_development_report(
            privacy_tamper
        )

    accounting_tamper = copy.deepcopy(report)
    accounting_tamper["resources"]["total_model_forward_count"] = 31
    with pytest.raises(ValueError, match="resources differ"):
        development.validate_gemma_iterative_generator_innovation_development_report(
            accounting_tamper
        )


def test_frozen_plan_and_fitter_gate_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _frozen_plan(monkeypatch)
    changed = copy.deepcopy(plan)
    changed["nested_family_screen"]["gates"][
        "minimum_parent_family_win_count"
    ] = 7

    with pytest.raises(ValueError, match="drifted"):
        development._validate_plan_fitter_compatibility(changed)
