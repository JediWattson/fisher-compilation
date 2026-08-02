from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_plan as plan_module,
)
from fisher_graph.token_loss_fisher import (
    COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES,
    build_token_loss_fisher_prompt_record,
)


def _sources(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
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
    generator = torch.Generator().manual_seed(31)
    latent = torch.randn(256, 2, generator=generator, dtype=torch.float64)
    noise = 0.08 * torch.randn(
        256,
        8,
        generator=generator,
        dtype=torch.float64,
    )
    scores = noise
    scores[:, 0] += latent[:, 0]
    scores[:, 2] += 0.55 * latent[:, 0]
    scores[:, 4] -= 0.70 * latent[:, 0]
    scores[:, 1] += latent[:, 1]
    scores[:, 3] += 0.45 * latent[:, 1]
    scores[:, 5] -= 0.65 * latent[:, 1]
    records = tuple(
        build_token_loss_fisher_prompt_record(
            example_id=f"example-{index:02d}",
            family_id=f"family-{index % 8}",
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
        "report_sha256": "a" * 64,
        "lineage": {
            "parent_artifact_sha256": "1" * 64,
            "parent_h4_head_sha256": "2" * 64,
            "accepted_x4_head_sha256": "3" * 64,
            "bridge_binding_sha256": "4" * 64,
            "model_sha256": "5" * 64,
            "adapter_execution_sha256": "6" * 64,
            "fit_manifest_sha256": "7" * 64,
            "factorial_report_sha256": "8" * 64,
            "factorial_report_file_sha256": "9" * 64,
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
        "report_sha256": "b" * 64,
        "lineage": {
            "token_fisher_report_sha256": "a" * 64,
            "token_fisher_report_file_sha256": "c" * 64,
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
    return token_report, corrective_report


def _build(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    token, corrective = _sources(monkeypatch)
    plan = plan_module.build_gemma_iterative_generator_innovation_plan(
        token_fisher_report=token,
        token_fisher_report_file_sha256="c" * 64,
        corrective_report=corrective,
        corrective_report_file_sha256="d" * 64,
    )
    return token, corrective, plan


def test_build_validate_and_replay_fixed_generator_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, corrective, plan = _build(monkeypatch)
    plan_module.validate_gemma_iterative_generator_innovation_plan(plan)

    basis = plan["frozen_generator_basis"]
    matrix = torch.tensor(
        basis["basis_matrix_source_coordinates_by_generator"],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        matrix.T @ matrix,
        torch.eye(2, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert torch.count_nonzero(matrix[1::2, 0]) == 0
    assert torch.count_nonzero(matrix[0::2, 1]) == 0
    assert basis["fisher_trace_coverage_fraction"] > 0.5
    assert plan["decision"]["new_family_disjoint_panel_opened"] is False
    assert plan["resources"]["source_model_forwards"] == 0
    assert (
        plan_module.replay_gemma_iterative_generator_innovation_plan(
            token_fisher_report=token,
            token_fisher_report_file_sha256="c" * 64,
            corrective_report=corrective,
            corrective_report_file_sha256="d" * 64,
            plan=plan,
        )
        == plan
    )


def test_plan_is_order_invariant_and_basis_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, corrective, plan = _build(monkeypatch)
    reordered = copy.deepcopy(token)
    reordered["prompt_fisher_records"] = tuple(
        reversed(reordered["prompt_fisher_records"])
    )
    replay = plan_module.build_gemma_iterative_generator_innovation_plan(
        token_fisher_report=reordered,
        token_fisher_report_file_sha256="c" * 64,
        corrective_report=corrective,
        corrective_report_file_sha256="d" * 64,
    )
    assert replay == plan

    changed = copy.deepcopy(plan)
    changed_basis = [
        list(row)
        for row in changed["frozen_generator_basis"][
            "basis_matrix_source_coordinates_by_generator"
        ]
    ]
    changed_basis[0][0] += 0.01
    changed["frozen_generator_basis"][
        "basis_matrix_source_coordinates_by_generator"
    ] = changed_basis
    with pytest.raises(ValueError, match="does not reconstruct"):
        plan_module.validate_gemma_iterative_generator_innovation_plan(
            changed
        )


def test_plan_rejects_corrective_source_not_bound_to_token_fisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, corrective = _sources(monkeypatch)
    corrective["lineage"]["token_fisher_report_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="not bound"):
        plan_module.build_gemma_iterative_generator_innovation_plan(
            token_fisher_report=token,
            token_fisher_report_file_sha256="c" * 64,
            corrective_report=corrective,
            corrective_report_file_sha256="d" * 64,
        )
