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
    gemma3_l3_l4_complete_h4_autonomous_capacity_sentinel as sentinel,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_autonomous_residual_development as v14,
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
        for ledger in v14._ALL_LEDGERS
    }


def _recipe_row(passed: bool) -> dict[str, object]:
    return {
        **sentinel.K640_CAPACITY_RECIPE.metadata(),
        "recipe_sha256": sentinel.K640_CAPACITY_RECIPE.artifact_sha256,
        "outer_fold_count": 8,
        "every_fold_fit_family_count": 7,
        "fold_provider_artifact_sha256s": {
            f"held-{index}": f"{index:x}" * 64 for index in range(8)
        },
        "serving_resources": {
            "prepared_float_scalar_count": 1_147_520,
            "logical_macs_per_token_upper_bound": 1_556_480,
        },
        "fidelity": _fidelity(passed),
    }


def _folds() -> tuple[dict[str, object], ...]:
    return v14.build_outer_lofo_splits(
        tuple(f"family-{index}" for index in range(8))
    )


def _report_kwargs(*, passed: bool) -> dict[str, object]:
    return {
        "artifact_path": sentinel.DEFAULT_OUTPUT,
        "panel": {"manifest_sha256": _SHA, "prompt_count": 16},
        "bridge_binding_sha256": _SHA,
        "folds": _folds(),
        "prerequisites": {
            "v14": sentinel._expected_v14_prerequisite_receipt(),
        },
        "fit_collection": {
            "prompt_count": 16,
            "family_count": 8,
            "native_h4_and_reverse_vjp_fit_only": True,
            "raw_fit_trace_tensor_serialization": False,
        },
        "base_fidelity": _fidelity(False),
        "recipe_row": _recipe_row(passed),
        "candidate": (
            {
                "recipe_id": sentinel.K640_CAPACITY_RECIPE.recipe_id,
                "provider_artifact_sha256": _SHA,
            }
            if passed
            else None
        ),
        "integrity": {
            "outer_fold_provider_fit_count": 8,
            "full_model_forward_count": 80,
            "backward_vjp_traversal_count": 16,
            "causal_off_support_execution_checks": 16,
            "guard_opened": False,
            "calibration_b_opened": False,
        },
    }


def _small_provider() -> AutonomousCompleteH4ResidualProvider:
    decoder = torch.zeros((1, 640), dtype=torch.float64)
    decoder[0, 0] = 1.0
    return AutonomousCompleteH4ResidualProvider(
        bridge_binding_sha256=_SHA,
        output_decoder=decoder,
        lag_source_kernel=torch.zeros((1, 64, 1), dtype=torch.float64),
        state_kernel=torch.zeros((1, 1), dtype=torch.float64),
        bias=torch.zeros(1, dtype=torch.float64),
        ridge=1.0,
        fit_objective="hidden_residual_ridge",
        fit_row_count=1,
        fit_family_ids=tuple(f"family-{index}" for index in range(8)),
        fit_sequence_sha256s=("b" * 64,),
        weighted_residual_rmse=0.0,
        fit_weight_sha256="c" * 64,
    )


def test_k640_recipe_and_resource_ceiling_are_frozen() -> None:
    recipe = sentinel.K640_CAPACITY_RECIPE
    assert recipe.recipe_id == "r640_l8_reverse_vjp_capacity"
    assert recipe.rank == 640
    assert recipe.lag_count == 8
    assert recipe.ridge == 1.0e-4
    assert recipe.fit_objective == "reverse_vjp_row_weighted_ridge_v1"
    assert 640 * 640 + 8 * 64 * 640 + 640 * 640 + 640 == 1_147_520
    assert (
        8 * 64 * 640 + 640 * 640 + 640 * 640 + 640 * 640
        == 1_556_480
    )
    assert sentinel._EXPECTED_FULL_SUPPORT_ROWS == 819
    assert sentinel._EXPECTED_MINIMUM_TRAINING_SUPPORT_ROWS == 703


def test_exact_single_recipe_work_accounting() -> None:
    without_candidate = sentinel._work_accounting(
        prompt_count=16,
        outer_fold_count=8,
        full_provider_fitted=False,
    )
    with_candidate = sentinel._work_accounting(
        prompt_count=16,
        outer_fold_count=8,
        full_provider_fitted=True,
    )
    assert without_candidate["fit_provider_count"] == 8
    assert with_candidate["fit_provider_count"] == 9
    assert without_candidate["full_model_forward_count"] == 80
    assert without_candidate["backward_vjp_traversal_count"] == 16
    assert without_candidate["full_model_work_breakdown"] == {
        "fit_native_source_forwards": 16,
        "fit_base_vjp_forwards": 16,
        "fit_base_vjp_backward_traversals": 16,
        "evaluation_native_source_forwards": 16,
        "evaluation_base_forwards": 16,
        "evaluation_recipe_forwards": 16,
        "total_forwards": 80,
        "total_backward_vjp_traversals": 16,
    }
    with pytest.raises(RuntimeError, match="work geometry"):
        sentinel._work_accounting(
            prompt_count=15,
            outer_fold_count=8,
            full_provider_fitted=False,
        )


def test_report_is_single_recipe_scalar_only_and_fail_closed() -> None:
    left = sentinel.build_autonomous_capacity_sentinel_report(
        **_report_kwargs(passed=False)
    )
    right = sentinel.build_autonomous_capacity_sentinel_report(
        **_report_kwargs(passed=False)
    )
    assert left == right
    assert left["format_version"] == 15
    assert left["capacity_sentinel"]["single_fixed_recipe"] is True
    assert left["capacity_sentinel"]["rank_selection_performed"] is False
    assert left["prerequisites"] == {
        "v14": sentinel._expected_v14_prerequisite_receipt(),
    }
    assert left["candidate"] is None
    assert left["passed"] is False
    assert left["classification"] == (
        "autonomous_complete_h4_k640_oof_capacity_ceiling_insufficient"
    )
    assert left["success_authorizes"].startswith("enlarge_or_nonlinearize")
    assert left["fresh_guard_authorized"] is False
    assert left["execution_scope"]["whole_model_compiled"] is False
    json.dumps(left, sort_keys=True, allow_nan=False)


def test_report_pass_authorizes_distillation_but_not_guard_or_serving() -> None:
    report = sentinel.build_autonomous_capacity_sentinel_report(
        **_report_kwargs(passed=True)
    )
    assert report["passed"] is True
    assert report["classification"] == (
        "autonomous_complete_h4_k640_oof_capacity_ceiling_reached"
    )
    assert report["success_authorizes"] == (
        "distill_k640_capacity_ceiling_on_reusable_calibration_a"
    )
    assert report["fresh_guard_authorized"] is False
    assert report["serving_authorized"] is False
    assert report["compression_claim"] is False

    bad = _report_kwargs(passed=True)
    bad["candidate"] = None
    with pytest.raises(ValueError, match="does not match"):
        sentinel.build_autonomous_capacity_sentinel_report(**bad)


def test_report_rejects_non_k640_row_and_tensor_serialization() -> None:
    bad_recipe = _report_kwargs(passed=False)
    bad_recipe["recipe_row"] = {
        **_recipe_row(False),
        "recipe_id": "not-k640",
    }
    with pytest.raises(ValueError, match="frozen K640"):
        sentinel.build_autonomous_capacity_sentinel_report(**bad_recipe)

    tensor_report = _report_kwargs(passed=False)
    tensor_report["fit_collection"] = {"forbidden": torch.ones(1)}
    with pytest.raises(TypeError, match="non-scalar data Tensor"):
        sentinel.build_autonomous_capacity_sentinel_report(**tensor_report)

    prerequisite_report = _report_kwargs(passed=False)
    prerequisite_report["prerequisites"]["v14"]["passed"] = True
    with pytest.raises(ValueError, match="prerequisite V14 receipt differs"):
        sentinel.build_autonomous_capacity_sentinel_report(**prerequisite_report)


def test_publish_uses_v15_domain_and_atomically_binds_conditional_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _small_provider()
    kwargs = _report_kwargs(passed=True)
    kwargs["candidate"] = {
        "recipe_id": sentinel.K640_CAPACITY_RECIPE.recipe_id,
        "provider_artifact_sha256": provider.artifact_sha256,
    }
    output = tmp_path / ".local-runs" / "v15.json"
    provider_output = output.with_suffix(".provider.pt")
    kwargs["artifact_path"] = output
    report = sentinel.build_autonomous_capacity_sentinel_report(**kwargs)
    monkeypatch.setattr(
        sentinel,
        "_validate_sentinel_provider",
        lambda _provider, *, expected_fit_family_count: None,
    )
    published = sentinel._publish(
        report,
        output=output,
        provider=provider,
        provider_output=provider_output,
    )
    assert output.exists() and provider_output.exists()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(provider_output.stat().st_mode) == 0o600
    assert published["report_sha256"] == json.loads(
        output.read_text(encoding="utf-8")
    )["report_sha256"]
    receipt = published["candidate"]["provider_tensor_artifact"]
    restored = load_autonomous_complete_h4_residual_provider(
        provider_output,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_file_sha256=receipt["file_sha256"],
        expected_bridge_binding_sha256=_SHA,
    )
    assert restored.metadata() == provider.metadata()


def test_existing_output_fails_before_live_context_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".local-runs"
    root.mkdir()
    output = root / "v15.json"
    output.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(
        sentinel,
        "prepare_complete_h4_rank320_live_context",
        lambda **_kwargs: pytest.fail("live context should not be loaded"),
    )
    with pytest.raises(FileExistsError, match="V15 report"):
        sentinel.run_gemma3_l3_l4_complete_h4_autonomous_capacity_sentinel(
            output=output
        )


@pytest.mark.parametrize(
    "drift",
    ("file_hash", "passed", "candidate", "guard", "calibration_b"),
)
def test_v14_prerequisite_drift_fails_before_live_context_load(
    drift: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".local-runs"
    root.mkdir()
    prerequisite = root / "v14.json"
    integrity = {
        "guard_opened": False,
        "calibration_b_opened": False,
    }
    payload = {
        "format_version": 14,
        "report_sha256": sentinel._V14_LOGICAL_SHA256,
        "classification": sentinel._V14_CLASSIFICATION,
        "passed": False,
        "candidate": None,
        "integrity": integrity,
    }
    if drift == "passed":
        payload["passed"] = True
    elif drift == "candidate":
        payload["candidate"] = {"recipe_id": "unexpected"}
    elif drift == "guard":
        integrity["guard_opened"] = True
    elif drift == "calibration_b":
        integrity["calibration_b_opened"] = True
    prerequisite.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sentinel, "_V14_REPORT", prerequisite)
    monkeypatch.setattr(
        sentinel,
        "_V14_FILE_SHA256",
        (
            "f" * 64
            if drift == "file_hash"
            else sentinel._v14._file_sha256(prerequisite)
        ),
    )
    monkeypatch.setattr(
        sentinel,
        "prepare_complete_h4_rank320_live_context",
        lambda **_kwargs: pytest.fail("live context should not be loaded"),
    )

    expected_drift = "file" if drift == "file_hash" else "semantics"
    with pytest.raises(RuntimeError, match=f"V14 {expected_drift} drifted"):
        sentinel.run_gemma3_l3_l4_complete_h4_autonomous_capacity_sentinel(
            output=root / "v15.json"
        )


def test_cli_defaults_to_write_once_v15_paths() -> None:
    args = sentinel.build_parser().parse_args([])
    assert args.output == sentinel.DEFAULT_OUTPUT
    assert args.provider_output is None
    assert args.cache_dir is None
    assert sentinel.DEFAULT_PROVIDER_OUTPUT == sentinel.DEFAULT_OUTPUT.with_suffix(
        ".provider.pt"
    )
