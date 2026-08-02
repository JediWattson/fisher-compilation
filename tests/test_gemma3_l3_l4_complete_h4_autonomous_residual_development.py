from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest
import torch

from fisher_graph.complete_h4_autonomous_residual import (
    AutonomousCompleteH4ResidualProvider,
    load_autonomous_complete_h4_residual_provider,
)

from fisher_graph import (
    gemma3_l3_l4_complete_h4_autonomous_residual_development as development,
)


_SHA = "a" * 64


def _fidelity(passed: bool) -> dict[str, object]:
    return {
        ledger: {
            "gates": {"passed": passed},
            "manifest": {
                "expected_examples": 16 if ledger == "ordinary" else 8,
                "observed_examples": 16 if ledger == "ordinary" else 8,
                "complete": True,
                "family_count": 8,
            },
            "family_summary": {"family_count": 8},
        }
        for ledger in development._ALL_LEDGERS
    }


def _recipe_rows(*, passing_id: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, recipe in enumerate(development.DEFAULT_RECIPES):
        rows.append(
            {
                "recipe_id": recipe.recipe_id,
                "fidelity": _fidelity(recipe.recipe_id == passing_id),
                "serving_resources": {
                    "prepared_float_scalar_count": 10_000 + index,
                    "logical_macs_per_token_upper_bound": 20_000 + index,
                },
            }
        )
    return rows


def test_fixed_recipe_bank_is_a_real_capacity_and_objective_ladder() -> None:
    recipes = development.DEFAULT_RECIPES
    assert tuple(value.recipe_id for value in recipes) == (
        "r64_l8_hidden",
        "r256_l8_hidden",
        "r256_l8_reverse_vjp",
        "r320_l8_reverse_vjp",
    )
    assert tuple(value.rank for value in recipes) == (64, 256, 256, 320)
    assert {value.fit_objective for value in recipes} == {
        "hidden_residual_ridge",
        "reverse_vjp_row_weighted_ridge_v1",
    }
    assert len({value.artifact_sha256 for value in recipes}) == 4
    assert all(value.lag_count == 8 for value in recipes)


def test_outer_lofo_splits_hold_exactly_one_of_eight_families() -> None:
    families = tuple(f"family-{index}" for index in reversed(range(8)))
    folds = development.build_outer_lofo_splits(families)
    assert len(folds) == 8
    assert tuple(value["held_family_id"] for value in folds) == tuple(
        sorted(families)
    )
    for fold in folds:
        held = fold["held_family_id"]
        training = fold["training_family_ids"]
        assert len(training) == 7
        assert held not in training
        assert set(training) | {held} == set(families)


def test_recipe_selection_requires_every_frozen_ledger_and_uses_exact_cost() -> None:
    rows = _recipe_rows()
    rows[1]["fidelity"] = _fidelity(True)
    rows[2]["fidelity"] = _fidelity(True)
    rows[1]["serving_resources"] = {
        "prepared_float_scalar_count": 90,
        "logical_macs_per_token_upper_bound": 100,
    }
    rows[2]["serving_resources"] = {
        "prepared_float_scalar_count": 80,
        "logical_macs_per_token_upper_bound": 100,
    }
    assert development.choose_passing_recipe(tuple(reversed(rows))) == (
        development.DEFAULT_RECIPES[2].recipe_id
    )

    rows[2]["fidelity"]["graph_core"]["gates"]["passed"] = False  # type: ignore[index]
    assert development.choose_passing_recipe(rows) == (
        development.DEFAULT_RECIPES[1].recipe_id
    )

    rows[1]["fidelity"]["graph_core"]["manifest"]["family_count"] = 7  # type: ignore[index]
    assert development.choose_passing_recipe(rows) is None


def test_scalar_report_builder_publishes_no_candidate_when_oof_fails() -> None:
    families = tuple(f"family-{index}" for index in range(8))
    kwargs = {
        "artifact_path": development.DEFAULT_OUTPUT,
        "panel": {"manifest_sha256": _SHA, "prompt_count": 16},
        "bridge_binding_sha256": _SHA,
        "recipes": development.DEFAULT_RECIPES,
        "folds": development.build_outer_lofo_splits(families),
        "fit_collection": {
            "prompt_count": 16,
            "family_count": 8,
            "native_h4_and_reverse_vjp_fit_only": True,
            "raw_tensor_serialization": False,
        },
        "base_fidelity": _fidelity(False),
        "recipe_rows": _recipe_rows(),
        "candidate": None,
        "integrity": {
            "fit_provider_count": 32,
            "guard_opened": False,
            "calibration_b_opened": False,
        },
    }
    left = development.build_autonomous_residual_development_report(**kwargs)
    right = development.build_autonomous_residual_development_report(**kwargs)
    assert left == right
    assert left["candidate"] is None
    assert left["passed"] is False
    assert left["selection"]["selected_recipe_id"] is None
    assert left["integrity"]["guard_opened"] is False
    assert left["integrity"]["calibration_b_opened"] is False
    assert left["execution_scope"]["full_vocabulary_logits_evaluated"] is True
    assert left["execution_scope"]["whole_model_compiled"] is False
    assert left["execution_scope"]["layer_4_computation_deleted"] is False
    json.dumps(left, sort_keys=True, allow_nan=False)


def test_report_builder_rejects_candidate_without_a_matching_oof_pass() -> None:
    families = tuple(f"family-{index}" for index in range(8))
    with pytest.raises(ValueError, match="does not match OOF"):
        development.build_autonomous_residual_development_report(
            artifact_path=development.DEFAULT_OUTPUT,
            panel={"manifest_sha256": _SHA},
            bridge_binding_sha256=_SHA,
            recipes=development.DEFAULT_RECIPES,
            folds=development.build_outer_lofo_splits(families),
            fit_collection={"prompt_count": 16},
            base_fidelity=_fidelity(False),
            recipe_rows=_recipe_rows(),
            candidate={"recipe_id": development.DEFAULT_RECIPES[0].recipe_id},
            integrity={"guard_opened": False, "calibration_b_opened": False},
        )


def test_report_builder_rejects_tensor_serialization() -> None:
    families = tuple(f"family-{index}" for index in range(8))
    with pytest.raises(TypeError, match="non-scalar data Tensor"):
        development.build_autonomous_residual_development_report(
            artifact_path=development.DEFAULT_OUTPUT,
            panel={"manifest_sha256": _SHA},
            bridge_binding_sha256=_SHA,
            recipes=development.DEFAULT_RECIPES,
            folds=development.build_outer_lofo_splits(families),
            fit_collection={"forbidden": torch.ones(1)},
            base_fidelity=_fidelity(False),
            recipe_rows=_recipe_rows(),
            candidate=None,
            integrity={"guard_opened": False, "calibration_b_opened": False},
        )


def test_required_ledger_manifest_coverage_fails_closed() -> None:
    sequence = object.__new__(development.AutonomousCompleteH4TrainingSequence)
    object.__setattr__(sequence, "example_id", "example-0")
    object.__setattr__(sequence, "family_id", "family-0")
    record = object.__new__(development._FitRecord)
    record.example = object()
    record.sequence = sequence
    record.ledger_indices = {
        name: torch.tensor([0], dtype=torch.int64)
        for name in development._ALL_LEDGERS
    }
    with pytest.raises(RuntimeError, match="all eight families"):
        development._ledger_manifests((record,))


def test_exact_full_model_work_accounting_includes_conditional_fit() -> None:
    without_candidate = development._work_accounting(
        prompt_count=16,
        recipe_count=4,
        outer_fold_count=8,
        full_provider_fitted=False,
    )
    with_candidate = development._work_accounting(
        prompt_count=16,
        recipe_count=4,
        outer_fold_count=8,
        full_provider_fitted=True,
    )
    assert without_candidate["full_model_forward_count"] == 128
    assert without_candidate["backward_vjp_traversal_count"] == 16
    assert without_candidate["fit_provider_count"] == 32
    assert with_candidate["fit_provider_count"] == 33
    assert with_candidate["full_model_work_breakdown"] == {
        "fit_native_source_forwards": 16,
        "fit_base_vjp_forwards": 16,
        "fit_base_vjp_backward_traversals": 16,
        "evaluation_native_source_forwards": 16,
        "evaluation_base_forwards": 16,
        "evaluation_recipe_forwards": 64,
        "total_forwards": 128,
        "total_backward_vjp_traversals": 16,
    }


def test_publish_atomically_binds_runtime_provider_sidecar(tmp_path: Path) -> None:
    decoder = torch.zeros((1, 640), dtype=torch.float64)
    decoder[0, 0] = 1.0
    provider = AutonomousCompleteH4ResidualProvider(
        bridge_binding_sha256=_SHA,
        output_decoder=decoder,
        lag_source_kernel=torch.zeros((1, 64, 1), dtype=torch.float64),
        state_kernel=torch.zeros((1, 1), dtype=torch.float64),
        bias=torch.zeros(1, dtype=torch.float64),
        ridge=1.0,
        fit_objective="hidden_residual_ridge",
        fit_row_count=1,
        fit_family_ids=("family",),
        fit_sequence_sha256s=("b" * 64,),
        weighted_residual_rmse=0.0,
        fit_weight_sha256="c" * 64,
    )
    families = tuple(f"family-{index}" for index in range(8))
    rows = _recipe_rows(passing_id=development.DEFAULT_RECIPES[0].recipe_id)
    report = development.build_autonomous_residual_development_report(
        artifact_path=tmp_path / ".local-runs" / "report.json",
        panel={"manifest_sha256": _SHA},
        bridge_binding_sha256=_SHA,
        recipes=development.DEFAULT_RECIPES,
        folds=development.build_outer_lofo_splits(families),
        fit_collection={"prompt_count": 16},
        base_fidelity=_fidelity(False),
        recipe_rows=rows,
        candidate={
            "recipe_id": development.DEFAULT_RECIPES[0].recipe_id,
            "provider_artifact_sha256": provider.artifact_sha256,
        },
        integrity={"guard_opened": False, "calibration_b_opened": False},
    )
    output = tmp_path / ".local-runs" / "report.json"
    provider_output = tmp_path / ".local-runs" / "report.provider.pt"
    published = development._publish(
        report,
        output=output,
        provider=provider,
        provider_output=provider_output,
    )
    receipt = published["candidate"]["provider_tensor_artifact"]
    assert output.exists() and provider_output.exists()
    assert stat.S_IMODE(provider_output.stat().st_mode) == 0o600
    restored = load_autonomous_complete_h4_residual_provider(
        provider_output,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_file_sha256=receipt["file_sha256"],
        expected_bridge_binding_sha256=_SHA,
    )
    assert restored.metadata() == provider.metadata()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["candidate"]["provider_tensor_artifact"] == receipt
    assert all(
        not isinstance(value, torch.Tensor)
        for value in persisted["candidate"].values()
    )


def test_existing_provider_sidecar_fails_before_live_context_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".local-runs"
    root.mkdir()
    output = root / "report.json"
    provider_output = root / "report.provider.pt"
    provider_output.write_bytes(b"occupied")
    monkeypatch.setattr(
        development,
        "prepare_complete_h4_rank320_live_context",
        lambda **_kwargs: pytest.fail("live context should not be loaded"),
    )
    with pytest.raises(FileExistsError, match="V14 provider"):
        development.run_gemma3_l3_l4_complete_h4_autonomous_residual_development(
            output=output,
            provider_output=provider_output,
        )
