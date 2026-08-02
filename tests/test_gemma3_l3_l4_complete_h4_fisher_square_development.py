from __future__ import annotations

import copy
import json
from pathlib import Path
import stat

import pytest
import torch

from fisher_graph.complete_h4_autonomous_residual import (
    AutonomousCompleteH4ResidualProvider,
)
from fisher_graph.complete_h4_fisher_conditional_residual import (
    AutonomousCompleteH4FisherXYProvider,
    load_autonomous_complete_h4_fisher_xy_provider,
    summarize_fisher_xy_bounded_coordinate_geometry,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_fisher_square_development as development,
)


_SHA = "a" * 64
_FAMILIES = tuple(f"family-{index}" for index in range(8))
_SEQUENCE_SHA256S = {
    family: (f"{2 * index + 1:064x}", f"{2 * index + 2:064x}")
    for index, family in enumerate(_FAMILIES)
}
_PARENT_ARTIFACTS = {
    family: f"{0x100 + index:064x}" for index, family in enumerate(_FAMILIES)
}
_RUNTIME_INPUT_FIELDS = (
    "source_modes",
    "logical_positions",
    "valid_mask",
    "source_mask",
    "support_mask",
    "base_h4",
)
_CHILD_RESOURCES = {
    "scope": "incremental_fisher_square_provider_including_k256_parent",
    "prepared_float_scalar_count": 377_604,
    "runtime_parameter_bytes_float64": 377_604 * 8,
    "logical_macs_per_token_upper_bound": 541_184,
    "incremental_child_prepared_float_scalar_count": 16_900,
    "incremental_child_runtime_parameter_bytes_float64": 16_900 * 8,
    "incremental_child_logical_macs_per_token_upper_bound": 16_896,
    "retained_gemma_parameters_excluded": True,
    "base_bridge_and_full_suffix_macs_excluded": True,
    "end_to_end_model_parameter_or_flop_claim": False,
}


def _fit_diagnostics(objective: str) -> tuple[dict[str, str], dict[str, object]]:
    offset = 0x200 if objective == "reverse_vjp_fisher" else 0x300
    artifacts = {
        family: f"{offset + index:064x}" for index, family in enumerate(_FAMILIES)
    }
    diagnostics = {
        family: {
            "provider_artifact_sha256": artifacts[family],
            "coordinate_objective": objective,
            "bounded_coordinate_geometry_sha256": "f" * 64,
            "bounded_coordinate_covariance_eigenvalues": (1.0, 0.25),
            "bounded_coordinate_lambda2_over_lambda1": 0.25,
            "bounded_coordinate_abs_correlation": 0.40,
            "bounded_coordinate_target_r2": (0.30, 0.20),
            "residual_second_coordinate_energy_fraction": 0.15,
        }
        for family in _FAMILIES
    }
    return artifacts, diagnostics


def _ownership_receipt(
    held: str,
    *,
    provider_artifact_sha256: str,
    parent_provider_artifact_sha256: str | None = None,
    coordinate_objective: str | None = None,
) -> dict[str, object]:
    fit_families = tuple(family for family in _FAMILIES if family != held)
    fit_sequences = tuple(
        sorted(
            sequence
            for family in fit_families
            for sequence in _SEQUENCE_SHA256S[family]
        )
    )
    receipt: dict[str, object] = {
        "held_family_id": held,
        "provider_artifact_sha256": provider_artifact_sha256,
        "fit_family_ids": fit_families,
        "fit_sequence_sha256s": fit_sequences,
        "held_sequence_sha256s": _SEQUENCE_SHA256S[held],
    }
    if parent_provider_artifact_sha256 is not None:
        receipt["parent_provider_artifact_sha256"] = (
            parent_provider_artifact_sha256
        )
    if coordinate_objective is not None:
        receipt["coordinate_objective"] = coordinate_objective
    return receipt


def _held_runtime_diagnostic(
    held: str,
    *,
    provider_artifact_sha256: str,
    parent_provider_artifact_sha256: str,
    coordinate_objective: str,
    coordinates: torch.Tensor | None = None,
) -> dict[str, object]:
    if coordinates is None:
        coordinates = torch.tensor(
            ((-0.6, -0.2), (-0.2, 0.7), (0.3, -0.6), (0.8, 0.4)),
            dtype=torch.float64,
        )
    geometry = summarize_fisher_xy_bounded_coordinate_geometry(coordinates)
    held_sequences = _SEQUENCE_SHA256S[held]
    return {
        "provider_artifact_sha256": provider_artifact_sha256,
        "parent_provider_artifact_sha256": parent_provider_artifact_sha256,
        "coordinate_objective": coordinate_objective,
        "held_family_id": held,
        "held_sequence_sha256s": held_sequences,
        "sequence_coordinate_receipts": tuple(
            {
                "sequence_sha256": sequence,
                "row_count": 2,
                "bounded_coordinates_sha256": f"{0x400 + index:064x}",
            }
            for index, sequence in enumerate(held_sequences)
        ),
        "weighting_semantics": "equal_sequences_then_equal_supported_rows",
        "runtime_input_fields": _RUNTIME_INPUT_FIELDS,
        "row_count": geometry.row_count,
        "bounded_coordinates_sha256": geometry.bounded_coordinates_sha256,
        "row_weight_sha256": geometry.row_weight_sha256,
        "bounded_coordinate_covariance_eigenvalues": geometry.covariance_eigenvalues,
        "bounded_coordinate_lambda2_over_lambda1": geometry.lambda2_over_lambda1,
        "bounded_coordinate_abs_correlation": geometry.abs_correlation,
        "residual_second_coordinate_energy_fraction": (
            geometry.residual_second_coordinate_energy_fraction
        ),
        "geometry_artifact_sha256": geometry.artifact_sha256,
    }


def _fidelity(
    *,
    passed: bool,
    absolute_delta_nll: float,
    kl: float,
    top1: float,
    family_absolute_delta_nll: float | None = None,
) -> dict[str, object]:
    family_value = (
        absolute_delta_nll
        if family_absolute_delta_nll is None
        else family_absolute_delta_nll
    )
    result: dict[str, object] = {}
    for ledger in development._v14._ALL_LEDGERS:
        expected = 16 if ledger == "ordinary" else 8
        result[ledger] = {
            "gates": {"passed": passed},
            "manifest": {
                "expected_examples": expected,
                "observed_examples": expected,
                "complete": True,
                "family_count": 8,
            },
            "aggregate": {
                "delta_nll_per_token": absolute_delta_nll,
                "absolute_delta_nll_per_token": absolute_delta_nll,
                "source_to_candidate_kl_per_token": kl,
                "top1_agreement_to_source": top1,
            },
            "family_summary": {
                "family_count": 8,
                "macro": {
                    "absolute_delta_nll_per_token": absolute_delta_nll,
                    "source_to_candidate_kl_per_token": kl,
                    "top1_agreement_to_source": top1,
                },
                "families": [
                    {
                        "family_id": family,
                        "absolute_delta_nll_per_token": family_value,
                    }
                    for family in _FAMILIES
                ],
            },
        }
    return result


def _arm_rows(
    *,
    fisher_passed: bool = True,
    fisher_absolute_delta_nll: float = 0.90,
) -> dict[str, dict[str, object]]:
    fisher_artifacts, fisher_diagnostics = _fit_diagnostics("reverse_vjp_fisher")
    pca_artifacts, pca_diagnostics = _fit_diagnostics("activation_pca")
    parent = _fidelity(
        passed=False,
        absolute_delta_nll=1.00,
        kl=1.00,
        top1=0.60,
    )
    fisher = _fidelity(
        passed=fisher_passed,
        absolute_delta_nll=fisher_absolute_delta_nll,
        kl=0.90,
        top1=0.63,
    )
    pca = _fidelity(
        passed=False,
        absolute_delta_nll=0.95,
        kl=0.92,
        top1=0.62,
    )
    parent_ownership = {
        held: _ownership_receipt(
            held,
            provider_artifact_sha256=_PARENT_ARTIFACTS[held],
        )
        for held in _FAMILIES
    }

    def child_protocol(
        artifacts: dict[str, str],
        *,
        objective: str,
    ) -> dict[str, object]:
        return {
            "fold_parent_provider_artifact_sha256s": dict(_PARENT_ARTIFACTS),
            "fold_ownership_receipts": {
                held: _ownership_receipt(
                    held,
                    provider_artifact_sha256=artifacts[held],
                    parent_provider_artifact_sha256=_PARENT_ARTIFACTS[held],
                    coordinate_objective=objective,
                )
                for held in _FAMILIES
            },
            "held_runtime_coordinate_diagnostics": {
                held: _held_runtime_diagnostic(
                    held,
                    provider_artifact_sha256=artifacts[held],
                    parent_provider_artifact_sha256=_PARENT_ARTIFACTS[held],
                    coordinate_objective=objective,
                )
                for held in _FAMILIES
            },
        }

    return {
        development.PARENT_ID: {
            "arm_id": development.PARENT_ID,
            "coordinate_objective": "reverse_vjp_weighted_shared_parent",
            "fold_provider_artifact_sha256s": dict(_PARENT_ARTIFACTS),
            "fold_ownership_receipts": parent_ownership,
            "serving_resources": {
                "prepared_float_scalar_count": 360_704,
                "logical_macs_per_token_upper_bound": 524_288,
            },
            "fidelity": parent,
        },
        development.FISHER_CHILD_ID: {
            "arm_id": development.FISHER_CHILD_ID,
            "coordinate_objective": "reverse_vjp_fisher",
            "serving_resources": dict(_CHILD_RESOURCES),
            "fold_provider_artifact_sha256s": fisher_artifacts,
            **child_protocol(fisher_artifacts, objective="reverse_vjp_fisher"),
            "fit_diagnostics": fisher_diagnostics,
            "fidelity": fisher,
        },
        development.PCA_CHILD_ID: {
            "arm_id": development.PCA_CHILD_ID,
            "coordinate_objective": "activation_pca",
            "serving_resources": dict(_CHILD_RESOURCES),
            "fold_provider_artifact_sha256s": pca_artifacts,
            **child_protocol(pca_artifacts, objective="activation_pca"),
            "fit_diagnostics": pca_diagnostics,
            "fidelity": pca,
        },
    }


def _report_kwargs(
    *,
    arm_rows: dict[str, dict[str, object]],
    candidate: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "artifact_path": development.DEFAULT_OUTPUT,
        "panel": {"manifest_sha256": _SHA, "prompt_count": 16},
        "bridge_binding_sha256": _SHA,
        "folds": development._v14.build_outer_lofo_splits(_FAMILIES),
        "prerequisites": {
            "v14": {"report_sha256": "b" * 64},
            "v15": {"report_sha256": "c" * 64},
        },
        "fit_collection": {
            "prompt_count": 16,
            "family_count": 8,
            "native_h4_and_reverse_vjp_fit_only": True,
        },
        "base_fidelity": _fidelity(
            passed=False,
            absolute_delta_nll=2.0,
            kl=2.0,
            top1=0.40,
        ),
        "arm_rows": arm_rows,
        "candidate": candidate,
        "integrity": {
            "guard_opened": False,
            "calibration_b_opened": False,
        },
    }


def _candidate_row(
    *,
    arm_id: str = development.FISHER_CHILD_ID,
    provider_artifact_sha256: str = _SHA,
) -> dict[str, object]:
    return {
        "arm_id": arm_id,
        "provider_artifact_sha256": provider_artifact_sha256,
        "full_provider_fit_geometry_qualification": {
            "provider_artifact_sha256": provider_artifact_sha256,
            "coordinate_objective": "reverse_vjp_fisher",
            "bounded_coordinate_geometry_sha256": "f" * 64,
            "bounded_coordinate_covariance_eigenvalues": (1.0, 0.25),
            "bounded_coordinate_lambda2_over_lambda1": 0.25,
            "bounded_coordinate_abs_correlation": 0.40,
            "bounded_coordinate_target_r2": (0.30, 0.20),
            "residual_second_coordinate_energy_fraction": 0.15,
        },
    }


def _full_provider(*, degenerate: bool = False) -> AutonomousCompleteH4FisherXYProvider:
    fit_sequences = tuple(
        sorted(sequence for values in _SEQUENCE_SHA256S.values() for sequence in values)
    )
    decoder = torch.eye(640, dtype=torch.float64)[:256].contiguous()
    parent = AutonomousCompleteH4ResidualProvider(
        bridge_binding_sha256=_SHA,
        output_decoder=decoder,
        lag_source_kernel=torch.zeros((8, 64, 256), dtype=torch.float64),
        state_kernel=torch.zeros((256, 256), dtype=torch.float64),
        bias=torch.zeros(256, dtype=torch.float64),
        ridge=1.0e-4,
        fit_objective="reverse_vjp_row_weighted_ridge_v1",
        fit_row_count=16,
        fit_family_ids=_FAMILIES,
        fit_sequence_sha256s=fit_sequences,
        weighted_residual_rmse=0.0,
        fit_weight_sha256="d" * 64,
    )
    coordinates = (
        torch.zeros((4, 2), dtype=torch.float64)
        if degenerate
        else torch.tensor(
            ((-0.6, -0.2), (-0.2, 0.7), (0.3, -0.6), (0.8, 0.4)),
            dtype=torch.float64,
        )
    )
    geometry = summarize_fisher_xy_bounded_coordinate_geometry(coordinates)
    return AutonomousCompleteH4FisherXYProvider(
        parent_provider=parent,
        router_weight=torch.zeros((256, 2), dtype=torch.float64),
        router_bias=torch.zeros(2, dtype=torch.float64),
        coordinate_scales=torch.ones(2, dtype=torch.float64),
        conditional_left=torch.zeros((3 * 256, 16), dtype=torch.float64),
        conditional_right=torch.zeros((16, 256), dtype=torch.float64),
        router_ridge=1.0e-4,
        conditional_ridge=1.0e-4,
        operator_norm_bound=0.25,
        fit_row_count=16,
        fit_family_ids=_FAMILIES,
        fit_sequence_sha256s=fit_sequences,
        coordinate_objective="reverse_vjp_fisher",
        coordinate_axes_sha256="e" * 64,
        coordinate_axis_values=(1.0, 0.5),
        fit_weight_sha256="f" * 64,
        coordinate_target_weighted_rmse=0.0,
        bounded_coordinate_geometry_sha256=geometry.artifact_sha256,
        bounded_coordinate_covariance_eigenvalues=geometry.covariance_eigenvalues,
        bounded_coordinate_lambda2_over_lambda1=geometry.lambda2_over_lambda1,
        bounded_coordinate_abs_correlation=geometry.abs_correlation,
        bounded_coordinate_target_r2=(0.30, 0.20),
        residual_second_coordinate_energy_fraction=(
            geometry.residual_second_coordinate_energy_fraction
        ),
        weighted_residual_rmse_before=0.0,
        weighted_residual_rmse_after=0.0,
        pre_projection_corner_operator_norms=(0.0, 0.0, 0.0, 0.0),
        post_projection_corner_operator_norms=(0.0, 0.0, 0.0, 0.0),
        trust_projection_scale=1.0,
        fit_receipt_sha256="1" * 64,
    )


def _full_candidate(
    provider: AutonomousCompleteH4FisherXYProvider,
) -> dict[str, object]:
    return {
        "arm_id": development.FISHER_CHILD_ID,
        "provider_artifact_sha256": provider.artifact_sha256,
        "parent_provider_artifact_sha256": (
            provider.parent_provider.artifact_sha256
        ),
        "fit_family_ids": provider.fit_family_ids,
        "fit_sequence_sha256s": provider.fit_sequence_sha256s,
        "full_provider_fit_geometry_qualification": (
            development._selected_full_provider_fit_geometry(provider)
        ),
    }


def test_fixed_protocol_and_exact_resource_and_work_geometry() -> None:
    recipe = development.PARENT_RECIPE
    assert recipe.recipe_id == development.PARENT_ID
    assert recipe.rank == 256
    assert recipe.lag_count == 8
    assert recipe.ridge == 1.0e-4
    assert recipe.fit_objective == "reverse_vjp_row_weighted_ridge_v1"
    assert development._CONDITIONAL_RANK == 16
    assert development.FISHER_XY_COORDINATE_COUNT == 2

    parent_scalars = 256 * 640 + 8 * 64 * 256 + 256 * 256 + 256
    parent_macs = 2 * 256 * 640 + 8 * 64 * 256 + 256 * 256
    child_scalars = 2 * 256 + 2 + 2 + 3 * 256 * 16 + 16 * 256
    child_macs = 2 * 256 + 3 * 256 * 16 + 16 * 256
    assert parent_scalars == development._EXPECTED_PARENT_SCALARS == 360_704
    assert parent_macs == development._EXPECTED_PARENT_MACS == 524_288
    assert child_scalars == development._EXPECTED_CHILD_INCREMENTAL_SCALARS == 16_900
    assert child_macs == development._EXPECTED_CHILD_INCREMENTAL_MACS == 16_896

    without_candidate = development._work_accounting(
        prompt_count=16,
        outer_fold_count=8,
        full_provider_fitted=False,
    )
    with_candidate = development._work_accounting(
        prompt_count=16,
        outer_fold_count=8,
        full_provider_fitted=True,
    )
    assert without_candidate["full_model_forward_count"] == 112
    assert without_candidate["backward_vjp_traversal_count"] == 16
    assert without_candidate["outer_provider_fit_count"] == 24
    assert without_candidate["fit_provider_count"] == 24
    assert with_candidate["fit_provider_count"] == 26
    assert with_candidate["conditional_full_panel_provider_fit_count"] == 2


def test_mechanism_gates_pass_only_for_material_fisher_specific_gain() -> None:
    rows = _arm_rows()
    # Top-1 is deliberately aggregate-weighted, not an equal-family macro.
    rows[development.PARENT_ID]["fidelity"]["ordinary"]["family_summary"][
        "macro"
    ]["top1_agreement_to_source"] = 0.99
    rows[development.FISHER_CHILD_ID]["fidelity"]["ordinary"][
        "family_summary"
    ]["macro"]["top1_agreement_to_source"] = 0.10
    rows[development.PCA_CHILD_ID]["fidelity"]["ordinary"]["family_summary"][
        "macro"
    ]["top1_agreement_to_source"] = 0.20
    result = development.evaluate_mechanism_gates(
        parent=rows[development.PARENT_ID],
        fisher_child=rows[development.FISHER_CHILD_ID],
        pca_child=rows[development.PCA_CHILD_ID],
    )
    assert result["passed"] is True
    assert result["observations"][
        "ordinary_family_absolute_delta_nll_win_count"
    ] == 8
    assert result["observations"][
        "ordinary_family_macro_absolute_delta_nll_relative_improvement"
    ] == pytest.approx(0.10)
    assert result["observations"]["ordinary_aggregate_top1_gain"] == pytest.approx(
        0.03
    )
    assert all(result["gates"].values())


def test_mechanism_gates_fail_closed_on_pca_tie_and_worst_family_regression() -> None:
    rows = _arm_rows()
    fisher_ordinary = rows[development.FISHER_CHILD_ID]["fidelity"]["ordinary"]
    pca_ordinary = rows[development.PCA_CHILD_ID]["fidelity"]["ordinary"]
    pca_ordinary["family_summary"]["macro"][
        "absolute_delta_nll_per_token"
    ] = fisher_ordinary["family_summary"]["macro"][
        "absolute_delta_nll_per_token"
    ]
    result = development.evaluate_mechanism_gates(
        parent=rows[development.PARENT_ID],
        fisher_child=rows[development.FISHER_CHILD_ID],
        pca_child=rows[development.PCA_CHILD_ID],
    )
    assert result["passed"] is False
    assert result["gates"]["fisher_strictly_beats_pca_absolute_delta_nll"] is False

    rows = _arm_rows()
    rows[development.FISHER_CHILD_ID]["fidelity"]["ordinary"]["family_summary"][
        "families"
    ][0]["absolute_delta_nll_per_token"] = 1.021
    result = development.evaluate_mechanism_gates(
        parent=rows[development.PARENT_ID],
        fisher_child=rows[development.FISHER_CHILD_ID],
        pca_child=rows[development.PCA_CHILD_ID],
    )
    assert result["passed"] is False
    assert result["gates"]["worst_family_regression_floor"] is False


def test_coordinate_geometry_requires_nondegenerate_folds_in_both_arms() -> None:
    rows = _arm_rows()
    result = development.evaluate_coordinate_geometry_gates(
        fisher_child=rows[development.FISHER_CHILD_ID],
        pca_child=rows[development.PCA_CHILD_ID],
    )
    assert result["passed"] is True
    runtime = rows[development.FISHER_CHILD_ID][
        "held_runtime_coordinate_diagnostics"
    ][_FAMILIES[0]]
    assert result["worst_case_across_both_arms"] == {
        "fit_fold_bounded_coordinate_target_r2_min_observed": 0.20,
        "held_family_runtime_bounded_coordinate_lambda2_over_lambda1_min_observed": runtime[
            "bounded_coordinate_lambda2_over_lambda1"
        ],
        "held_family_runtime_bounded_coordinate_abs_correlation_max_observed": runtime[
            "bounded_coordinate_abs_correlation"
        ],
        "held_family_runtime_residual_second_coordinate_energy_fraction_min_observed": runtime[
            "residual_second_coordinate_energy_fraction"
        ],
    }
    assert result["fit_diagnostics_authenticated_by_provider_artifact_sha256"] is True
    assert (
        result["held_runtime_geometry_authenticated_by_geometry_artifact_sha256"]
        is True
    )
    for arm_id in (development.FISHER_CHILD_ID, development.PCA_CHILD_ID):
        arm = result["arms"][arm_id]
        assert len(arm["fit_fold_predictability"]["per_fold"]) == 8
        assert len(arm["held_family_runtime_geometry"]["per_fold"]) == 8
        assert len(arm["ownership_receipts"]) == 8

    failing = copy.deepcopy(rows)
    pca = failing[development.PCA_CHILD_ID]
    pca["held_runtime_coordinate_diagnostics"][_FAMILIES[-1]] = (
        _held_runtime_diagnostic(
            _FAMILIES[-1],
            provider_artifact_sha256=pca["fold_provider_artifact_sha256s"][
                _FAMILIES[-1]
            ],
            parent_provider_artifact_sha256=_PARENT_ARTIFACTS[_FAMILIES[-1]],
            coordinate_objective="activation_pca",
            coordinates=torch.tensor(
                ((-0.8, -0.8), (-0.2, -0.2), (0.2, 0.2), (0.8, 0.8)),
                dtype=torch.float64,
            ),
        )
    )
    result = development.evaluate_coordinate_geometry_gates(
        fisher_child=failing[development.FISHER_CHILD_ID],
        pca_child=failing[development.PCA_CHILD_ID],
    )
    assert result["passed"] is False
    pca = result["arms"][development.PCA_CHILD_ID]
    held = pca["held_family_runtime_geometry"]
    assert held["failed_fold_ids"] == (_FAMILIES[-1],)
    assert held["per_fold"][_FAMILIES[-1]]["gates"][
        "residual_has_second_coordinate_energy"
    ] is False


def test_coordinate_geometry_threshold_boundaries_are_inclusive() -> None:
    rows = _arm_rows()
    for arm_id in (development.FISHER_CHILD_ID, development.PCA_CHILD_ID):
        diagnostic = rows[arm_id]["fit_diagnostics"][_FAMILIES[0]]
        diagnostic["bounded_coordinate_covariance_eigenvalues"] = (1.0, 0.01)
        diagnostic["bounded_coordinate_lambda2_over_lambda1"] = 0.01
        diagnostic["bounded_coordinate_abs_correlation"] = 0.99
        diagnostic["bounded_coordinate_target_r2"] = (0.01, 0.01)
        diagnostic["residual_second_coordinate_energy_fraction"] = 0.01
    result = development.evaluate_coordinate_geometry_gates(
        fisher_child=rows[development.FISHER_CHILD_ID],
        pca_child=rows[development.PCA_CHILD_ID],
    )
    assert result["passed"] is True
    assert result["thresholds"] == {
        "fit_fold_bounded_coordinate_target_r2_each_min": 0.01,
        "held_family_runtime_bounded_coordinate_lambda2_over_lambda1_min": 0.01,
        "held_family_runtime_bounded_coordinate_abs_correlation_max": 0.99,
        "held_family_runtime_residual_second_coordinate_energy_fraction_min": 0.01,
        "bounded_coordinate_lambda2_over_lambda1_min": 0.01,
        "bounded_coordinate_abs_correlation_max": 0.99,
        "bounded_coordinate_target_r2_each_min": 0.01,
        "residual_second_coordinate_energy_fraction_min": 0.01,
        "required_on_every_outer_fold": True,
        "required_for_coordinate_objectives": (
            "reverse_vjp_fisher",
            "activation_pca",
        ),
    }


def test_fit_predictability_and_held_runtime_geometry_are_separate_gates() -> None:
    rows = _arm_rows()
    diagnostic = rows[development.FISHER_CHILD_ID]["fit_diagnostics"][_FAMILIES[0]]
    diagnostic["bounded_coordinate_target_r2"] = (0.009, 0.20)
    result = development.evaluate_coordinate_geometry_gates(
        fisher_child=rows[development.FISHER_CHILD_ID],
        pca_child=rows[development.PCA_CHILD_ID],
    )
    assert result["passed"] is False
    fisher = result["arms"][development.FISHER_CHILD_ID]
    assert fisher["fit_fold_predictability"]["failed_fold_ids"] == (
        _FAMILIES[0],
    )
    assert fisher["held_family_runtime_geometry"]["passed"] is True


def test_coordinate_geometry_rejects_unauthenticated_diagnostics() -> None:
    rows = _arm_rows()
    rows[development.FISHER_CHILD_ID]["fit_diagnostics"][_FAMILIES[0]][
        "provider_artifact_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="not bound to the fold provider"):
        development.evaluate_coordinate_geometry_gates(
            fisher_child=rows[development.FISHER_CHILD_ID],
            pca_child=rows[development.PCA_CHILD_ID],
        )


def test_fold_ownership_and_runtime_coordinate_receipts_are_exact() -> None:
    rows = _arm_rows()
    result = development.evaluate_coordinate_geometry_gates(
        fisher_child=rows[development.FISHER_CHILD_ID],
        pca_child=rows[development.PCA_CHILD_ID],
    )
    fisher = result["arms"][development.FISHER_CHILD_ID]
    ownership = fisher["ownership_receipts"][_FAMILIES[0]]
    assert ownership["provider_artifact_sha256"] == rows[
        development.FISHER_CHILD_ID
    ]["fold_provider_artifact_sha256s"][_FAMILIES[0]]
    assert ownership["parent_provider_artifact_sha256"] == _PARENT_ARTIFACTS[
        _FAMILIES[0]
    ]
    assert ownership["fit_family_ids"] == _FAMILIES[1:]
    assert ownership["held_sequence_sha256s"] == _SEQUENCE_SHA256S[_FAMILIES[0]]
    assert len(ownership["fit_sequence_sha256s"]) == 14

    runtime = fisher["held_family_runtime_geometry"]["per_fold"][_FAMILIES[0]]
    assert tuple(
        receipt["sequence_sha256"]
        for receipt in runtime["sequence_coordinate_receipts"]
    ) == _SEQUENCE_SHA256S[_FAMILIES[0]]
    assert sum(
        receipt["row_count"]
        for receipt in runtime["sequence_coordinate_receipts"]
    ) == runtime["row_count"]

    wrong_sequence = copy.deepcopy(rows)
    wrong_sequence[development.FISHER_CHILD_ID]["fold_ownership_receipts"][
        _FAMILIES[0]
    ]["held_sequence_sha256s"] = _SEQUENCE_SHA256S[_FAMILIES[1]]
    with pytest.raises(ValueError, match="ownership|held runtime"):
        development.evaluate_coordinate_geometry_gates(
            fisher_child=wrong_sequence[development.FISHER_CHILD_ID],
            pca_child=wrong_sequence[development.PCA_CHILD_ID],
        )

    wrong_geometry = copy.deepcopy(rows)
    wrong_geometry[development.PCA_CHILD_ID][
        "held_runtime_coordinate_diagnostics"
    ][_FAMILIES[0]]["geometry_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="geometry artifact hash mismatch"):
        development.evaluate_coordinate_geometry_gates(
            fisher_child=wrong_geometry[development.FISHER_CHILD_ID],
            pca_child=wrong_geometry[development.PCA_CHILD_ID],
        )


def test_zero_coordinate_covariance_is_a_scientific_failure_not_bad_input() -> None:
    rows = _arm_rows()
    fisher = rows[development.FISHER_CHILD_ID]
    fisher["held_runtime_coordinate_diagnostics"][_FAMILIES[0]] = (
        _held_runtime_diagnostic(
            _FAMILIES[0],
            provider_artifact_sha256=fisher["fold_provider_artifact_sha256s"][
                _FAMILIES[0]
            ],
            parent_provider_artifact_sha256=_PARENT_ARTIFACTS[_FAMILIES[0]],
            coordinate_objective="reverse_vjp_fisher",
            coordinates=torch.zeros((4, 2), dtype=torch.float64),
        )
    )
    result = development.evaluate_coordinate_geometry_gates(
        fisher_child=rows[development.FISHER_CHILD_ID],
        pca_child=rows[development.PCA_CHILD_ID],
    )
    assert result["passed"] is False
    fold = result["arms"][development.FISHER_CHILD_ID][
        "held_family_runtime_geometry"
    ]["per_fold"][_FAMILIES[0]]
    assert fold["gates"]["bounded_covariance_rank2"] is False


def test_report_separates_absolute_and_mechanism_qualification() -> None:
    absolute_failure = development.build_fisher_square_development_report(
        **_report_kwargs(arm_rows=_arm_rows(fisher_passed=False), candidate=None)
    )
    assert absolute_failure["absolute_qualification"]["passed"] is False
    assert absolute_failure["mechanism_retention"]["passed"] is True
    assert absolute_failure["classification"] == (
        "fisher_square_absolute_fidelity_insufficient"
    )
    assert absolute_failure["candidate"] is None

    mechanism_failure = development.build_fisher_square_development_report(
        **_report_kwargs(
            arm_rows=_arm_rows(fisher_absolute_delta_nll=0.96),
            candidate=None,
        )
    )
    assert mechanism_failure["absolute_qualification"]["passed"] is True
    assert mechanism_failure["mechanism_retention"]["passed"] is False
    assert mechanism_failure["classification"] == (
        "fisher_square_mechanism_retention_insufficient"
    )

    geometry_rows = _arm_rows()
    geometry_rows[development.FISHER_CHILD_ID]["fit_diagnostics"][_FAMILIES[0]][
        "bounded_coordinate_target_r2"
    ] = (0.20, 0.009)
    geometry_failure = development.build_fisher_square_development_report(
        **_report_kwargs(arm_rows=geometry_rows, candidate=None)
    )
    assert geometry_failure["coordinate_geometry_qualification"]["passed"] is False
    assert geometry_failure["absolute_qualification"]["passed"] is True
    assert geometry_failure["mechanism_retention"]["passed"] is True
    assert geometry_failure["classification"] == (
        "fisher_square_coordinate_geometry_insufficient"
    )
    assert geometry_failure["candidate"] is None

    candidate = _candidate_row()
    passed = development.build_fisher_square_development_report(
        **_report_kwargs(arm_rows=_arm_rows(), candidate=candidate)
    )
    assert passed["passed"] is True
    assert passed["selection"]["selected_arm_id"] == development.FISHER_CHILD_ID
    assert passed["fresh_guard_authorized"] is False
    assert passed["calibration_b_authorized"] is False
    assert passed["execution_scope"]["full_vocabulary_logits_evaluated"] is True
    assert passed["execution_scope"]["whole_model_compiled"] is False
    json.dumps(passed, sort_keys=True, allow_nan=False)


def test_report_rejects_candidate_mismatch_tensor_and_unmatched_control() -> None:
    with pytest.raises(ValueError, match="candidate does not match"):
        development.build_fisher_square_development_report(
            **_report_kwargs(
                arm_rows=_arm_rows(),
                candidate=_candidate_row(arm_id=development.PCA_CHILD_ID),
            )
        )

    tensor_kwargs = _report_kwargs(
        arm_rows=_arm_rows(fisher_passed=False),
        candidate=None,
    )
    tensor_kwargs["fit_collection"] = {"forbidden": torch.ones(1)}
    with pytest.raises(TypeError, match="non-scalar data Tensor"):
        development.build_fisher_square_development_report(**tensor_kwargs)

    rows = _arm_rows(fisher_passed=False)
    rows[development.PCA_CHILD_ID]["serving_resources"] = {
        **_CHILD_RESOURCES,
        "prepared_float_scalar_count": 1,
    }
    with pytest.raises(ValueError, match="control resources differ"):
        development.build_fisher_square_development_report(
            **_report_kwargs(arm_rows=rows, candidate=None)
        )


def test_selected_full_provider_must_pass_its_own_fit_geometry() -> None:
    candidate = _candidate_row()
    diagnostic = candidate["full_provider_fit_geometry_qualification"]
    diagnostic["bounded_coordinate_covariance_eigenvalues"] = (0.0, 0.0)
    diagnostic["bounded_coordinate_lambda2_over_lambda1"] = 0.0
    diagnostic["residual_second_coordinate_energy_fraction"] = 0.0
    with pytest.raises(
        ValueError,
        match="selected full-panel provider fit geometry is insufficient",
    ):
        development.build_fisher_square_development_report(
            **_report_kwargs(arm_rows=_arm_rows(), candidate=candidate)
        )


def test_publish_binds_and_roundtrips_selected_full_provider(tmp_path: Path) -> None:
    provider = _full_provider()
    candidate = _full_candidate(provider)
    output = tmp_path / ".local-runs" / "v16.json"
    provider_output = tmp_path / ".local-runs" / "v16.provider.pt"
    kwargs = _report_kwargs(arm_rows=_arm_rows(), candidate=candidate)
    kwargs["artifact_path"] = output
    report = development.build_fisher_square_development_report(**kwargs)

    published = development._publish(
        report,
        output=output,
        provider=provider,
        provider_output=provider_output,
    )
    receipt = published["candidate"]["provider_tensor_artifact"]
    assert output.exists() and provider_output.exists()
    assert stat.S_IMODE(provider_output.stat().st_mode) == 0o600
    assert receipt["provider_artifact_sha256"] == provider.artifact_sha256
    restored = load_autonomous_complete_h4_fisher_xy_provider(
        provider_output,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_file_sha256=receipt["file_sha256"],
        expected_bridge_binding_sha256=_SHA,
    )
    assert restored.metadata() == provider.metadata()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["candidate"]["provider_tensor_artifact"] == receipt
    assert persisted["candidate"]["full_provider_fit_geometry_qualification"] == (
        json.loads(
            json.dumps(candidate["full_provider_fit_geometry_qualification"])
        )
    )


def test_publish_rejects_degenerate_full_refit_without_artifacts(
    tmp_path: Path,
) -> None:
    valid = _full_provider()
    candidate = _full_candidate(valid)
    kwargs = _report_kwargs(arm_rows=_arm_rows(), candidate=candidate)
    output = tmp_path / ".local-runs" / "v16.json"
    provider_output = tmp_path / ".local-runs" / "v16.provider.pt"
    kwargs["artifact_path"] = output
    report = development.build_fisher_square_development_report(**kwargs)

    with pytest.raises(
        RuntimeError,
        match="selected full-panel provider fit geometry is insufficient",
    ):
        development._publish(
            report,
            output=output,
            provider=_full_provider(degenerate=True),
            provider_output=provider_output,
        )
    assert not output.exists()
    assert not provider_output.exists()


def test_prerequisite_binding_checks_hash_and_semantics(tmp_path: Path) -> None:
    path = tmp_path / "v14.json"
    payload = {
        "format_version": 14,
        "report_sha256": "b" * 64,
        "classification": "expected-classification",
        "passed": False,
        "candidate": None,
        "integrity": {
            "guard_opened": False,
            "calibration_b_opened": False,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    file_sha256 = development._v14._file_sha256(path)
    receipt = development._validate_prerequisite_report(
        path,
        logical_sha256="b" * 64,
        file_sha256=file_sha256,
        classification="expected-classification",
        format_version=14,
    )
    assert receipt["file_sha256"] == file_sha256
    assert receipt["guard_opened"] is False

    drifted = copy.deepcopy(payload)
    drifted["passed"] = True
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="semantics drifted"):
        development._validate_prerequisite_report(
            path,
            logical_sha256="b" * 64,
            file_sha256=development._v14._file_sha256(path),
            classification="expected-classification",
            format_version=14,
        )


def test_preflight_failures_happen_before_live_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".local-runs"
    root.mkdir()
    existing = root / "existing.json"
    existing.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(
        development,
        "prepare_complete_h4_rank320_live_context",
        lambda **_kwargs: pytest.fail("live context should not be loaded"),
    )
    with pytest.raises(FileExistsError, match="V16 report"):
        development.run_gemma3_l3_l4_complete_h4_fisher_square_development(
            output=existing
        )

    destination = root / "new.json"
    monkeypatch.setattr(
        development,
        "_validate_prerequisites",
        lambda: (_ for _ in ()).throw(RuntimeError("prerequisite drift")),
    )
    with pytest.raises(RuntimeError, match="prerequisite drift"):
        development.run_gemma3_l3_l4_complete_h4_fisher_square_development(
            output=destination
        )


def test_cli_defaults_to_write_once_v16_artifacts() -> None:
    arguments = development.build_parser().parse_args([])
    assert arguments.output == development.DEFAULT_OUTPUT
    assert arguments.provider_output is None
    assert arguments.cache_dir is None
