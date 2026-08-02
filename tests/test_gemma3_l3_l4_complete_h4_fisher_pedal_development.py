from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import stat

import pytest
import torch

from fisher_graph.complete_h4_fisher_conditional_pedal import (
    AutonomousCompleteH4FisherXYPedalProvider,
    FisherXYPedalRuntimeReplay,
    fit_autonomous_complete_h4_fisher_xy_pedal,
    load_autonomous_complete_h4_fisher_xy_pedal_provider,
)
from fisher_graph.complete_h4_autonomous_residual import (
    AutonomousCompleteH4ResidualProvider,
    AutonomousCompleteH4TrainingSequence,
    fit_autonomous_complete_h4_residual,
)
from fisher_graph.complete_h4_fisher_conditional_residual import (
    summarize_fisher_xy_bounded_coordinate_geometry,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_fisher_pedal_development as development,
)


_SHA = "a" * 64
_FAMILIES = tuple(f"family-{index}" for index in range(8))
_SEQUENCES = {
    family: (f"{2 * index + 1:064x}", f"{2 * index + 2:064x}")
    for index, family in enumerate(_FAMILIES)
}
_PARENT_ARTIFACTS = {
    family: f"{0x100 + index:064x}" for index, family in enumerate(_FAMILIES)
}
_CHILD_RESOURCES = {
    "scope": "incremental_fisher_pedal_provider_including_k256_parent",
    "prepared_float_scalar_count": 377_608,
    "runtime_parameter_bytes_float64": 377_608 * 8,
    "logical_macs_per_token_upper_bound": 541_187,
    "incremental_child_prepared_float_scalar_count": 16_904,
    "incremental_child_runtime_parameter_bytes_float64": 16_904 * 8,
    "incremental_child_logical_macs_per_token_upper_bound": 16_899,
    "retained_gemma_parameters_excluded": True,
    "base_bridge_and_full_suffix_macs_excluded": True,
    "elementwise_norm_clamp_and_scaling_ops_excluded_from_matrix_macs": True,
    "end_to_end_model_parameter_or_flop_claim": False,
}


def _fidelity(
    *,
    passed: bool,
    absolute_delta_nll: float,
    kl: float,
    top1: float,
) -> dict[str, object]:
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
                        "absolute_delta_nll_per_token": absolute_delta_nll,
                    }
                    for family in _FAMILIES
                ],
            },
        }
    return result


def _ownership(
    held: str,
    *,
    provider_artifact: str,
    parent_artifact: str | None = None,
    objective: str | None = None,
    mode: str | None = None,
) -> dict[str, object]:
    fit_families = tuple(family for family in _FAMILIES if family != held)
    receipt: dict[str, object] = {
        "held_family_id": held,
        "provider_artifact_sha256": provider_artifact,
        "fit_family_ids": fit_families,
        "fit_sequence_sha256s": tuple(
            sorted(sequence for family in fit_families for sequence in _SEQUENCES[family])
        ),
        "held_sequence_sha256s": _SEQUENCES[held],
    }
    if parent_artifact is not None:
        receipt["parent_provider_artifact_sha256"] = parent_artifact
    if objective is not None:
        receipt["coordinate_objective"] = objective
    if mode is not None:
        receipt["pedal_mode"] = mode
    return receipt


def _fit_diagnostic(
    *,
    artifact: str,
    objective: str,
    mode: str,
) -> dict[str, object]:
    if mode == "unit":
        effective_mean = effective_min = effective_max = 1.0
        effective_std = slope_norm = 0.0
        rmse_after = 0.95
    elif mode == "constant_optimal":
        effective_mean = effective_min = effective_max = 0.5
        effective_std = slope_norm = 0.0
        rmse_after = 0.90
    else:
        effective_mean = 0.5
        effective_std = 0.2
        effective_min = 0.2
        effective_max = 0.8
        slope_norm = 0.3
        rmse_after = 0.80
    factor_payload = {
        "coordinate_objective": objective,
        "tensor_sha256s": {
            "router_weight": "1" * 64,
            "router_bias": "2" * 64,
            "coordinate_scales": "3" * 64,
            "direction_left": "4" * 64,
            "direction_right": "5" * 64,
        },
        "coordinate_axes_sha256": "6" * 64,
        "coordinate_axis_values": (2.0, 1.0),
        "bounded_coordinate_geometry_sha256": "f" * 64,
        "pedal_unclipped_target_sha256": "7" * 64,
        "pedal_fit_weight_sha256": "8" * 64,
        "pedal_support_mask_sha256": "9" * 64,
    }
    return {
        "provider_artifact_sha256": artifact,
        "coordinate_objective": objective,
        "bounded_coordinate_geometry_sha256": "f" * 64,
        "bounded_coordinate_covariance_eigenvalues": (1.0, 0.25),
        "bounded_coordinate_lambda2_over_lambda1": 0.25,
        "bounded_coordinate_abs_correlation": 0.40,
        "bounded_coordinate_target_r2": (0.30, 0.20),
        "residual_second_coordinate_energy_fraction": 0.15,
        "pedal_mode": mode,
        "trust_fraction": 0.25,
        "fit_max_bounded_direction_ratio": 0.25,
        "fit_max_emitted_delta_ratio": 0.20,
        "pedal_effective_weighted_mean": effective_mean,
        "pedal_effective_weighted_std": effective_std,
        "pedal_effective_min": effective_min,
        "pedal_effective_max": effective_max,
        "pedal_slope_l2_norm": slope_norm,
        "weighted_residual_rmse_constant": 0.90,
        "weighted_residual_rmse_after": rmse_after,
        "shared_factor_bundle": {
            **factor_payload,
            "artifact_sha256": development._v14._sha256(
                factor_payload, domain=development._FACTOR_BUNDLE_DOMAIN
            ),
        },
    }


def _held_diagnostic(
    held: str,
    *,
    artifact: str,
    parent_artifact: str,
    objective: str,
    mode: str,
    pedal_rows: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    replay_sequence_override: str | None = None,
    parent_scale_rows: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    direction_scale_rows: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> dict[str, object]:
    if pedal_rows is None:
        if mode == "unit":
            pedal_rows = ((1.0, 1.0), (1.0, 1.0))
        elif mode == "constant_optimal":
            pedal_rows = ((0.5, 0.5), (0.5, 0.5))
        else:
            pedal_rows = ((0.0, 0.2), (0.8, 1.0))
    coordinate_bank = torch.tensor(
        ((-0.6, -0.2), (-0.2, 0.7), (0.3, -0.6), (0.8, 0.4)),
        dtype=torch.float64,
    )
    split = len(pedal_rows[0])
    if split + len(pedal_rows[1]) != len(coordinate_bank):
        raise ValueError("test pedal rows must contain four values")
    coordinate_rows = (coordinate_bank[:split], coordinate_bank[split:])
    if parent_scale_rows is None:
        parent_scale_rows = tuple(
            tuple(1.0 for _ in values) for values in pedal_rows
        )  # type: ignore[assignment]
    if direction_scale_rows is None:
        direction_scale_rows = tuple(
            tuple(0.10 for _ in values) for values in pedal_rows
        )  # type: ignore[assignment]
    replays: list[FisherXYPedalRuntimeReplay] = []
    receipts: list[dict[str, object]] = []
    weights: list[torch.Tensor] = []
    effective_pedals: list[torch.Tensor] = []
    effective_weights: list[torch.Tensor] = []
    effective_counts: list[int] = []
    for sequence, coordinates, pedals_value, parent_scales, direction_scales in zip(
        _SEQUENCES[held],
        coordinate_rows,
        pedal_rows,
        parent_scale_rows,
        direction_scale_rows,
        strict=True,
    ):
        pedals = torch.tensor(pedals_value, dtype=torch.float64)
        if (
            len(parent_scales) != len(pedals_value)
            or len(direction_scales) != len(pedals_value)
        ):
            raise ValueError("test modal scales must match pedal rows")
        parent = torch.tensor(parent_scales, dtype=torch.float64).unsqueeze(1).expand(
            -1, 4
        ).contiguous()
        direction = (
            torch.tensor(direction_scales, dtype=torch.float64)
            .unsqueeze(1)
            .expand(-1, 4)
            .contiguous()
        )
        bounded = direction.clone()
        replay = FisherXYPedalRuntimeReplay(
            provider_artifact_sha256=artifact,
            parent_provider_artifact_sha256=parent_artifact,
            sequence_artifact_sha256=(replay_sequence_override or sequence),
            trust_fraction=0.25,
            parent_modal=parent,
            bounded_coordinates=coordinates,
            unbounded_direction=direction,
            bounded_direction=bounded,
            pedal=pedals,
            emitted_delta=pedals.unsqueeze(1) * bounded,
        )
        metadata = replay.metadata()
        replays.append(replay)
        sequence_row_weights = torch.full(
            (replay.row_count,),
            1.0 / (len(_SEQUENCES[held]) * replay.row_count),
            dtype=torch.float64,
        )
        weights.append(sequence_row_weights)
        effective_support = development.fisher_xy_pedal_fit_support_mask(
            replay.parent_modal,
            replay.bounded_direction,
        )
        effective_count = int(effective_support.sum())
        effective_counts.append(effective_count)
        if effective_count:
            effective_pedals.append(replay.pedal[effective_support])
            effective_weights.append(
                sequence_row_weights[effective_support]
                * replay.bounded_direction[effective_support].square().sum(dim=1)
            )
        receipts.append(
            {
                "sequence_sha256": sequence,
                "row_count": replay.row_count,
                "effective_row_count": effective_count,
                "effective_support_mask_sha256": development._v14._tensor_sha256(
                    effective_support
                ),
                "bounded_coordinates_sha256": metadata["tensor_sha256s"][
                    "bounded_coordinates"
                ],
                "runtime_replay_artifact_sha256": replay.artifact_sha256,
                "runtime_replay": metadata,
            }
        )
    coordinates = torch.cat([value.bounded_coordinates for value in replays])
    pedals = torch.cat([value.pedal for value in replays])
    row_weights = torch.cat(weights)
    geometry = summarize_fisher_xy_bounded_coordinate_geometry(
        coordinates, row_weights
    )
    parent_norm = torch.cat(
        [torch.linalg.vector_norm(value.parent_modal, dim=1) for value in replays]
    )
    bounded_norm = torch.cat(
        [
            torch.linalg.vector_norm(value.bounded_direction, dim=1)
            for value in replays
        ]
    )
    emitted_norm = torch.cat(
        [torch.linalg.vector_norm(value.emitted_delta, dim=1) for value in replays]
    )
    bounded_ratio = float((bounded_norm / parent_norm).max())
    emitted_ratio = float((emitted_norm / parent_norm).max())
    every_sequence_has_effective_rows = all(value > 0 for value in effective_counts)
    if every_sequence_has_effective_rows:
        selected_effective_pedals = torch.cat(effective_pedals)
        raw_effective_weights = torch.cat(effective_weights)
        if mode == "unit":
            effective_min = effective_mean = effective_max = 1.0
            effective_std = 0.0
        elif mode == "constant_optimal":
            exact_pedal = float(pedals[0])
            effective_min = effective_mean = effective_max = exact_pedal
            effective_std = 0.0
        else:
            effective_min, effective_mean, effective_std, effective_max = (
                development._realized_convex_weighted_stats(
                    selected_effective_pedals,
                    raw_effective_weights,
                    label="test held conditional effective pedal",
                )
            )
    else:
        effective_mean = effective_std = effective_min = effective_max = 0.0
    if mode == "unit":
        pedal_min = pedal_mean = pedal_max = 1.0
        zero_fraction = 0.0
        unit_fraction = 1.0
    elif mode == "constant_optimal":
        exact_pedal = float(pedals[0])
        pedal_min = pedal_mean = pedal_max = exact_pedal
        zero_fraction = 1.0 if exact_pedal == 0.0 else 0.0
        unit_fraction = 1.0 if exact_pedal == 1.0 else 0.0
    else:
        pedal_min, pedal_mean, _pedal_std, pedal_max = (
            development._realized_convex_weighted_stats(
                pedals,
                row_weights,
                label="test held conditional pedal",
            )
        )
        realized_row_mass = row_weights.sum()
        zero_fraction = float(
            (row_weights * (pedals == 0.0)).sum() / realized_row_mass
        )
        unit_fraction = float(
            (row_weights * (pedals == 1.0)).sum() / realized_row_mass
        )
    payload = {
        "provider_artifact_sha256": artifact,
        "parent_provider_artifact_sha256": parent_artifact,
        "coordinate_objective": objective,
        "pedal_mode": mode,
        "trust_fraction": 0.25,
        "held_family_id": held,
        "held_sequence_sha256s": _SEQUENCES[held],
        "sequence_coordinate_receipts": tuple(receipts),
        "weighting_semantics": "equal_sequences_then_equal_supported_rows",
        "runtime_input_fields": development._RUNTIME_INPUT_FIELDS,
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
        "pedal_distribution": {
            "weighting_semantics": "equal_sequences_then_equal_supported_rows",
            "pedal_min": pedal_min,
            "pedal_weighted_mean": pedal_mean,
            "pedal_max": pedal_max,
            "pedal_zero_weight_fraction": zero_fraction,
            "pedal_unit_weight_fraction": unit_fraction,
        },
        "pedal_effective_distribution": {
            "weighting_semantics": (
                "normalized_equal_sequence_row_weight_times_bounded_direction_"
                "energy_on_effective_support"
            ),
            "effective_row_count": sum(effective_counts),
            "effective_sequence_count": sum(value > 0 for value in effective_counts),
            "every_sequence_has_effective_rows": (
                every_sequence_has_effective_rows
            ),
            "pedal_effective_min": effective_min,
            "pedal_effective_weighted_mean": effective_mean,
            "pedal_effective_weighted_std": effective_std,
            "pedal_effective_max": effective_max,
        },
        "runtime_trust": {
            "max_bounded_direction_to_parent_norm_ratio": bounded_ratio,
            "max_emitted_delta_to_parent_norm_ratio": emitted_ratio,
            "pointwise_trust_certificate_passed": True,
        },
    }
    return {
        **payload,
        "held_diagnostic_sha256": development._v14._sha256(
            payload, domain=development._HELD_DIAGNOSTIC_DOMAIN
        ),
    }


def _arm_rows(*, fisher_absolute_passed: bool = True) -> dict[str, dict[str, object]]:
    metrics = {
        development.PARENT_ID: (False, 1.00, 1.00, 0.60),
        development.FISHER_UNIT_ID: (False, 0.96, 0.95, 0.61),
        development.FISHER_CONSTANT_ID: (False, 0.92, 0.91, 0.62),
        development.FISHER_PEDAL_ID: (
            fisher_absolute_passed,
            0.90,
            0.90,
            0.63,
        ),
        development.PCA_PEDAL_ID: (False, 0.91, 0.905, 0.625),
    }
    result: dict[str, dict[str, object]] = {
        development.PARENT_ID: {
            "arm_id": development.PARENT_ID,
            "coordinate_objective": development._COORDINATE_OBJECTIVES[
                development.PARENT_ID
            ],
            "fold_provider_artifact_sha256s": dict(_PARENT_ARTIFACTS),
            "fold_ownership_receipts": {
                family: _ownership(
                    family, provider_artifact=_PARENT_ARTIFACTS[family]
                )
                for family in _FAMILIES
            },
            "serving_resources": {
                "prepared_float_scalar_count": 360_704,
                "logical_macs_per_token_upper_bound": 524_288,
            },
            "fidelity": _fidelity(
                passed=metrics[development.PARENT_ID][0],
                absolute_delta_nll=metrics[development.PARENT_ID][1],
                kl=metrics[development.PARENT_ID][2],
                top1=metrics[development.PARENT_ID][3],
            ),
        }
    }
    for arm_index, arm_id in enumerate(development._CHILD_IDS):
        objective = development._COORDINATE_OBJECTIVES[arm_id]
        mode = development._PEDAL_MODES[arm_id]
        artifacts = {
            family: f"{0x200 + 0x20 * arm_index + index:064x}"
            for index, family in enumerate(_FAMILIES)
        }
        held = {
            family: _held_diagnostic(
                family,
                artifact=artifacts[family],
                parent_artifact=_PARENT_ARTIFACTS[family],
                objective=objective,
                mode=mode,
            )
            for family in _FAMILIES
        }
        passed, delta, kl, top1 = metrics[arm_id]
        result[arm_id] = {
            "arm_id": arm_id,
            "coordinate_objective": objective,
            "pedal_mode": mode,
            "fold_provider_artifact_sha256s": artifacts,
            "fold_parent_provider_artifact_sha256s": dict(_PARENT_ARTIFACTS),
            "fold_ownership_receipts": {
                family: _ownership(
                    family,
                    provider_artifact=artifacts[family],
                    parent_artifact=_PARENT_ARTIFACTS[family],
                    objective=objective,
                    mode=mode,
                )
                for family in _FAMILIES
            },
            "fit_diagnostics": {
                family: _fit_diagnostic(
                    artifact=artifacts[family], objective=objective, mode=mode
                )
                for family in _FAMILIES
            },
            "held_runtime_diagnostics": held,
            "held_runtime_coordinate_diagnostics": held,
            "serving_resources": dict(_CHILD_RESOURCES),
            "fidelity": _fidelity(
                passed=passed,
                absolute_delta_nll=delta,
                kl=kl,
                top1=top1,
            ),
        }
    return result


def _full_qualification(*, artifact: str = _SHA, passed: bool = True) -> dict[str, object]:
    variation_floor = math.sqrt(torch.finfo(torch.float64).eps)
    geometry = development._v16._coordinate_fold_diagnostics(
        _fit_diagnostic(
            artifact=artifact,
            objective="reverse_vjp_fisher",
            mode="conditional",
        ),
        expected_artifact_sha256=artifact,
        expected_coordinate_objective="reverse_vjp_fisher",
    )
    trust_gates = {
        "fit_bounded_direction_within_relative_trust_ball": True,
        "fit_emitted_delta_within_relative_trust_ball": True,
        "conditional_pedal_mode": True,
        "fixed_trust_fraction": True,
        "conditional_pedal_has_nonzero_slope": passed,
        "effective_pedal_lies_in_closed_unit_interval": True,
        "conditional_pedal_varies_on_fit_effective_rows": passed,
        "conditional_fit_not_worse_than_constant": True,
        "conditional_fit_not_worse_than_parent": True,
    }
    return {
        "provider_artifact_sha256": artifact,
        "coordinate_objective": "reverse_vjp_fisher",
        "pedal_mode": "conditional",
        "qualification_scope": "fresh_full_fit_output_diagnostic",
        "provider_metadata_alone_is_candidate_certificate": False,
        "coordinate_geometry": geometry,
        "trust": {
            "trust_fraction": 0.25,
            "fit_max_bounded_direction_ratio": 0.25,
            "fit_max_emitted_delta_ratio": 0.20,
            "pedal_slope_l2_norm": 0.1 if passed else 0.0,
            "pedal_effective_weighted_mean": 0.5 if passed else 0.2,
            "pedal_effective_weighted_std": 0.1 if passed else 0.0,
            "pedal_effective_min": 0.2,
            "pedal_effective_max": 0.8 if passed else 0.2,
            "weighted_residual_rmse_before": 1.0,
            "weighted_residual_rmse_constant": 0.9,
            "weighted_residual_rmse_after": 0.8,
            "variation_floor": variation_floor,
            "gates": trust_gates,
            "passed": passed,
        },
        "passed": passed,
    }


def _candidate(qualification: dict[str, object]) -> dict[str, object]:
    return {
        "arm_id": development.FISHER_PEDAL_ID,
        "provider_artifact_sha256": qualification["provider_artifact_sha256"],
        "full_provider_fit_qualification": qualification,
    }


def _report_kwargs(
    *,
    rows: dict[str, dict[str, object]],
    qualification: dict[str, object] | None,
    candidate: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "artifact_path": development.DEFAULT_OUTPUT,
        "panel": {"manifest_sha256": _SHA, "prompt_count": 16},
        "bridge_binding_sha256": _SHA,
        "folds": development._v14.build_outer_lofo_splits(_FAMILIES),
        "prerequisites": {"v16": {"report_sha256": "b" * 64}},
        "fit_collection": {"prompt_count": 16, "family_count": 8},
        "base_fidelity": _fidelity(
            passed=False, absolute_delta_nll=2.0, kl=2.0, top1=0.4
        ),
        "arm_rows": rows,
        "full_refit_qualification": qualification,
        "candidate": candidate,
        "integrity": {"guard_opened": False, "calibration_b_opened": False},
    }


def _small_sequence(
    index: int,
    *,
    length: int,
    family_id: str,
    example_prefix: str,
) -> AutonomousCompleteH4TrainingSequence:
    decoder = torch.eye(640, dtype=torch.float64)[:4].contiguous()
    generator = torch.Generator().manual_seed(54000 + index)
    source = torch.randn((length, 64), generator=generator, dtype=torch.float64)
    base = torch.randn((length, 640), generator=generator, dtype=torch.float64)
    p = 0.15 * base[:, :4]
    raw_c1 = 1.2 * p[:, 0] + 0.4 * p[:, 2]
    raw_c2 = -0.9 * p[:, 1] + 0.3 * p[:, 3]
    c1 = raw_c1 / (0.10 + raw_c1.abs())
    c2 = raw_c2 / (0.10 + raw_c2.abs())
    direction = torch.stack(
        (
            0.6 * c1 * p[:, 0] - 0.2 * c2 * p[:, 1],
            -0.4 * c1 * p[:, 1] + 0.3 * c2 * p[:, 2],
            0.5 * c2 * p[:, 2] + 0.2 * c1 * c2 * p[:, 0],
            -0.3 * c2 * p[:, 3] + 0.2 * c1 * c2 * p[:, 1],
        ),
        dim=1,
    )
    pedal = (0.5 + 0.25 * c1 - 0.20 * c2 + 0.15 * c1 * c2).clamp(
        0.0, 1.0
    )
    native = base + (p + pedal.unsqueeze(1) * direction) @ decoder
    gradients = torch.zeros_like(base)
    gradients[:, :4] = torch.stack(
        (p[:, 0] + 0.2 * p[:, 2], p[:, 1], p[:, 2], p[:, 3]), dim=1
    )
    mask = torch.ones(length, dtype=torch.bool)
    return AutonomousCompleteH4TrainingSequence(
        example_id=f"{example_prefix}-{index}",
        family_id=family_id,
        source_modes=source,
        logical_positions=torch.arange(length, dtype=torch.int64),
        valid_mask=mask,
        source_mask=mask,
        support_mask=mask,
        base_h4=base,
        native_h4=native,
        reverse_vjp_gradients=gradients,
    )


def _variable_length_lofo_fixture(
    mode: str,
) -> tuple[
    AutonomousCompleteH4FisherXYPedalProvider,
    tuple[AutonomousCompleteH4TrainingSequence, ...],
]:
    decoder = torch.eye(640, dtype=torch.float64)[:4].contiguous()
    training = tuple(
        _small_sequence(
            index,
            length=8 + index,
            family_id=f"train-family-{index // 2}",
            example_prefix="train-example",
        )
        for index in range(14)
    )
    held = tuple(
        _small_sequence(
            100 + index,
            length=length,
            family_id="held-variable-family",
            example_prefix="held-example",
        )
        for index, length in enumerate((7, 23))
    )
    parent = AutonomousCompleteH4ResidualProvider(
        bridge_binding_sha256=_SHA,
        output_decoder=decoder,
        lag_source_kernel=torch.zeros((1, 64, 4), dtype=torch.float64),
        state_kernel=0.15 * torch.eye(4, dtype=torch.float64),
        bias=torch.zeros(4, dtype=torch.float64),
        ridge=1.0e-4,
        fit_objective="reverse_vjp_row_weighted_ridge_v1",
        fit_row_count=sum(value.source_modes.shape[0] for value in training),
        fit_family_ids=tuple(sorted({value.family_id for value in training})),
        fit_sequence_sha256s=tuple(
            sorted(value.artifact_sha256 for value in training)
        ),
        weighted_residual_rmse=0.0,
        fit_weight_sha256="d" * 64,
    )
    provider = fit_autonomous_complete_h4_fisher_xy_pedal(
        sequences=training,
        parent_provider=parent,
        conditional_rank=4,
        coordinate_objective="reverse_vjp_fisher",
        pedal_mode=mode,
        router_ridge=1.0e-6,
        direction_ridge=1.0e-6,
        pedal_ridge=1.0e-6,
    )
    return provider, held


def _small_full_provider():
    decoder = torch.eye(640, dtype=torch.float64)[:4].contiguous()
    sequences: list[AutonomousCompleteH4TrainingSequence] = []
    for index in range(16):
        generator = torch.Generator().manual_seed(44000 + index)
        length = 10
        source = torch.randn((length, 64), generator=generator, dtype=torch.float64)
        base = torch.randn((length, 640), generator=generator, dtype=torch.float64)
        p = 0.15 * base[:, :4]
        raw_c1 = 1.2 * p[:, 0] + 0.4 * p[:, 2]
        raw_c2 = -0.9 * p[:, 1] + 0.3 * p[:, 3]
        c1 = raw_c1 / (0.10 + raw_c1.abs())
        c2 = raw_c2 / (0.10 + raw_c2.abs())
        direction = torch.stack(
            (
                0.6 * c1 * p[:, 0] - 0.2 * c2 * p[:, 1],
                -0.4 * c1 * p[:, 1] + 0.3 * c2 * p[:, 2],
                0.5 * c2 * p[:, 2] + 0.2 * c1 * c2 * p[:, 0],
                -0.3 * c2 * p[:, 3] + 0.2 * c1 * c2 * p[:, 1],
            ),
            dim=1,
        )
        pedal = (0.5 + 0.25 * c1 - 0.20 * c2 + 0.15 * c1 * c2).clamp(
            0.0, 1.0
        )
        native = base + (p + pedal.unsqueeze(1) * direction) @ decoder
        gradients = torch.zeros_like(base)
        gradients[:, :4] = torch.stack(
            (p[:, 0] + 0.2 * p[:, 2], p[:, 1], p[:, 2], p[:, 3]), dim=1
        )
        mask = torch.ones(length, dtype=torch.bool)
        sequences.append(
            AutonomousCompleteH4TrainingSequence(
                example_id=f"full-example-{index}",
                family_id=f"full-family-{index // 2}",
                source_modes=source,
                logical_positions=torch.arange(length, dtype=torch.int64),
                valid_mask=mask,
                source_mask=mask,
                support_mask=mask,
                base_h4=base,
                native_h4=native,
                reverse_vjp_gradients=gradients,
            )
        )
    sequence_ids = tuple(sorted(value.artifact_sha256 for value in sequences))
    families = tuple(sorted({value.family_id for value in sequences}))
    parent = AutonomousCompleteH4ResidualProvider(
        bridge_binding_sha256=_SHA,
        output_decoder=decoder,
        lag_source_kernel=torch.zeros((1, 64, 4), dtype=torch.float64),
        state_kernel=0.15 * torch.eye(4, dtype=torch.float64),
        bias=torch.zeros(4, dtype=torch.float64),
        ridge=1.0e-4,
        fit_objective="reverse_vjp_row_weighted_ridge_v1",
        fit_row_count=160,
        fit_family_ids=families,
        fit_sequence_sha256s=sequence_ids,
        weighted_residual_rmse=0.0,
        fit_weight_sha256="d" * 64,
    )
    provider = fit_autonomous_complete_h4_fisher_xy_pedal(
        sequences=tuple(sequences),
        parent_provider=parent,
        conditional_rank=4,
        coordinate_objective="reverse_vjp_fisher",
        pedal_mode="conditional",
        router_ridge=1.0e-6,
        direction_ridge=1.0e-6,
        pedal_ridge=1.0e-6,
    )
    return provider


def test_fixed_protocol_resource_and_work_counts() -> None:
    assert development.DEFAULT_OUTPUT.name.endswith("dev-v18.json")
    assert development.DEFAULT_OUTPUT != development._V17_OUTPUT
    assert development._SCHEMA.endswith(".v18")
    assert development._REPORT_DOMAIN.endswith(b":v18\0")
    assert development._HELD_DIAGNOSTIC_DOMAIN.endswith(b":v18\0")
    assert development._FACTOR_BUNDLE_DOMAIN.endswith(b":v18\0")
    assert development._TRUST_FRACTION == 0.25
    assert development._EXPECTED_CHILD_INCREMENTAL_SCALARS == 16_904
    assert development._EXPECTED_CHILD_INCREMENTAL_MACS == 16_899
    without = development._work_accounting(
        prompt_count=16, outer_fold_count=8, full_provider_fitted=False
    )
    with_candidate = development._work_accounting(
        prompt_count=16, outer_fold_count=8, full_provider_fitted=True
    )
    assert without["full_model_forward_count"] == 144
    assert without["backward_vjp_traversal_count"] == 16
    assert without["outer_provider_fit_count"] == 40
    assert without["fit_provider_count"] == 40
    assert with_candidate["fit_provider_count"] == 42


@pytest.mark.parametrize(
    ("mode", "arm_id"),
    (
        ("unit", development.FISHER_UNIT_ID),
        ("constant_optimal", development.FISHER_CONSTANT_ID),
        ("conditional", development.FISHER_PEDAL_ID),
    ),
)
def test_variable_length_held_diagnostics_use_realized_mass_or_exact_control(
    mode: str,
    arm_id: str,
) -> None:
    provider, held_sequences = _variable_length_lofo_fixture(mode)
    diagnostic = development._held_runtime_diagnostics(
        provider,
        held_family_id="held-variable-family",
        held_sequences=held_sequences,
    )
    assert sorted(
        receipt["row_count"]
        for receipt in diagnostic["sequence_coordinate_receipts"]
    ) == [7, 23]
    distribution = diagnostic["pedal_distribution"]
    effective = diagnostic["pedal_effective_distribution"]
    if mode in ("unit", "constant_optimal"):
        expected = 1.0 if mode == "unit" else float(provider.pedal_bias[0])
        assert distribution["pedal_min"] == expected
        assert distribution["pedal_weighted_mean"] == expected
        assert distribution["pedal_max"] == expected
        assert effective["pedal_effective_min"] == expected
        assert effective["pedal_effective_weighted_mean"] == expected
        assert effective["pedal_effective_weighted_std"] == 0.0
        assert effective["pedal_effective_max"] == expected
    else:
        assert (
            distribution["pedal_min"]
            <= distribution["pedal_weighted_mean"]
            <= distribution["pedal_max"]
        )
        assert (
            effective["pedal_effective_min"]
            <= effective["pedal_effective_weighted_mean"]
            <= effective["pedal_effective_max"]
        )
    parsed = development._validate_held_trust_diagnostic(
        diagnostic,
        expected_arm_id=arm_id,
        expected_family_id="held-variable-family",
        expected_provider_artifact_sha256=provider.artifact_sha256,
        expected_parent_artifact_sha256=provider.parent_provider.artifact_sha256,
        expected_held_sequence_sha256s=tuple(
            sorted(value.artifact_sha256 for value in held_sequences)
        ),
    )
    if mode in ("unit", "constant_optimal"):
        assert parsed["passed"] is True


def test_mechanism_requires_parent_constant_and_pca_comparisons() -> None:
    rows = _arm_rows()
    result = development.evaluate_mechanism_gates(
        parent=rows[development.PARENT_ID],
        fisher_unit=rows[development.FISHER_UNIT_ID],
        fisher_constant=rows[development.FISHER_CONSTANT_ID],
        fisher_pedal=rows[development.FISHER_PEDAL_ID],
        pca_pedal=rows[development.PCA_PEDAL_ID],
    )
    assert result["passed"] is True
    assert result["observations"][
        "versus_parent_family_absolute_delta_nll_win_count"
    ] == 8
    assert result["observations"][
        "versus_constant_family_absolute_delta_nll_win_count"
    ] == 8

    tied = copy.deepcopy(rows)
    tied[development.PCA_PEDAL_ID]["fidelity"]["ordinary"]["family_summary"][
        "macro"
    ]["absolute_delta_nll_per_token"] = 0.90
    result = development.evaluate_mechanism_gates(
        parent=tied[development.PARENT_ID],
        fisher_unit=tied[development.FISHER_UNIT_ID],
        fisher_constant=tied[development.FISHER_CONSTANT_ID],
        fisher_pedal=tied[development.FISHER_PEDAL_ID],
        pca_pedal=tied[development.PCA_PEDAL_ID],
    )
    assert result["gates"]["fisher_strictly_beats_pca_absolute_delta_nll"] is False


def test_trust_authenticates_all_32_held_replays_and_equal_sequence_stats() -> None:
    rows = _arm_rows()
    result = development.evaluate_trust_gates(arm_rows=rows)
    assert result["passed"] is True
    assert result["held_runtime_diagnostic_count"] == 32
    assert all(len(value["per_fold"]) == 8 for value in result["arms"].values())

    diagnostic = rows[development.FISHER_PEDAL_ID]["held_runtime_diagnostics"][
        _FAMILIES[0]
    ]
    assert diagnostic["pedal_distribution"]["pedal_weighted_mean"] == pytest.approx(
        0.5
    )
    tampered = copy.deepcopy(rows)
    receipt = tampered[development.FISHER_PEDAL_ID]["held_runtime_diagnostics"][
        _FAMILIES[0]
    ]["sequence_coordinate_receipts"][0]
    receipt["runtime_replay"]["pedal_mean"] = 0.15
    payload = tampered[development.FISHER_PEDAL_ID]["held_runtime_diagnostics"][
        _FAMILIES[0]
    ]
    payload["held_diagnostic_sha256"] = development._v14._sha256(
        {key: value for key, value in payload.items() if key != "held_diagnostic_sha256"},
        domain=development._HELD_DIAGNOSTIC_DOMAIN,
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        development.evaluate_trust_gates(arm_rows=tampered)


def test_conditional_pedal_must_vary_on_fit_and_held_effective_rows() -> None:
    rows = _arm_rows()
    family = _FAMILIES[0]
    artifact = rows[development.FISHER_PEDAL_ID][
        "fold_provider_artifact_sha256s"
    ][family]
    rows[development.FISHER_PEDAL_ID]["held_runtime_diagnostics"][family] = (
        _held_diagnostic(
            family,
            artifact=artifact,
            parent_artifact=_PARENT_ARTIFACTS[family],
            objective="reverse_vjp_fisher",
            mode="conditional",
            pedal_rows=((0.5, 0.0), (0.5, 1.0)),
            parent_scale_rows=((1.0, 1.0e-10), (1.0, 1.0e-10)),
            direction_scale_rows=((0.10, 1.0e-12), (0.10, 1.0e-12)),
        )
    )
    held_result = development.evaluate_trust_gates(arm_rows=rows)
    held_fold = held_result["arms"][development.FISHER_PEDAL_ID]["per_fold"][
        family
    ]
    assert held_result["passed"] is False
    assert held_fold["held_runtime"]["gates"][
        "conditional_pedal_varies_on_held_effective_rows"
    ] is False
    assert held_fold["held_runtime"]["pedal_effective_max"] == 1.0
    assert held_fold["held_runtime"]["pedal_effective_min"] == 0.0
    assert held_fold["held_runtime"]["pedal_effective_weighted_std"] <= math.sqrt(
        torch.finfo(torch.float64).eps
    )

    rows = _arm_rows()
    fit = rows[development.FISHER_PEDAL_ID]["fit_diagnostics"][family]
    fit["pedal_effective_weighted_std"] = 0.0
    fit["pedal_effective_min"] = fit["pedal_effective_max"] = 0.5
    fit_result = development.evaluate_trust_gates(arm_rows=rows)
    fit_fold = fit_result["arms"][development.FISHER_PEDAL_ID]["per_fold"][family]
    assert fit_result["passed"] is False
    assert fit_fold["gates"][
        "conditional_pedal_varies_on_fit_effective_rows"
    ] is False

    rows = _arm_rows()
    held_diagnostic = rows[development.FISHER_PEDAL_ID][
        "held_runtime_diagnostics"
    ][family]
    held_diagnostic["pedal_effective_distribution"].update(
        {
            "pedal_effective_min": -1.0,
            "pedal_effective_weighted_mean": 9.0,
            "pedal_effective_weighted_std": 0.1,
            "pedal_effective_max": 2.0,
        }
    )
    held_diagnostic["held_diagnostic_sha256"] = development._v14._sha256(
        {
            key: value
            for key, value in held_diagnostic.items()
            if key != "held_diagnostic_sha256"
        },
        domain=development._HELD_DIAGNOSTIC_DOMAIN,
    )
    interval_result = development.evaluate_trust_gates(arm_rows=rows)
    interval_fold = interval_result["arms"][development.FISHER_PEDAL_ID][
        "per_fold"
    ][family]
    assert interval_result["passed"] is False
    assert interval_fold["held_runtime"]["gates"][
        "effective_pedal_lies_in_closed_unit_interval"
    ] is False


def test_held_replay_must_be_bound_to_its_exact_sequence_artifact() -> None:
    rows = _arm_rows()
    family = _FAMILIES[0]
    artifact = rows[development.FISHER_PEDAL_ID][
        "fold_provider_artifact_sha256s"
    ][family]
    rows[development.FISHER_PEDAL_ID]["held_runtime_diagnostics"][family] = (
        _held_diagnostic(
            family,
            artifact=artifact,
            parent_artifact=_PARENT_ARTIFACTS[family],
            objective="reverse_vjp_fisher",
            mode="conditional",
            replay_sequence_override="e" * 64,
        )
    )
    with pytest.raises(ValueError, match="replay artifact or binding differs"):
        development.evaluate_trust_gates(arm_rows=rows)


def test_held_effective_support_requires_rows_from_every_sequence() -> None:
    rows = _arm_rows()
    family = _FAMILIES[0]
    artifact = rows[development.FISHER_PEDAL_ID][
        "fold_provider_artifact_sha256s"
    ][family]
    rows[development.FISHER_PEDAL_ID]["held_runtime_diagnostics"][family] = (
        _held_diagnostic(
            family,
            artifact=artifact,
            parent_artifact=_PARENT_ARTIFACTS[family],
            objective="reverse_vjp_fisher",
            mode="conditional",
            direction_scale_rows=((0.0, 0.0), (0.10, 0.10)),
        )
    )
    result = development.evaluate_trust_gates(arm_rows=rows)
    held = result["arms"][development.FISHER_PEDAL_ID]["per_fold"][family][
        "held_runtime"
    ]
    assert result["passed"] is False
    assert held["effective_sequence_count"] == 1
    assert held["gates"][
        "every_held_sequence_has_effective_direction_rows"
    ] is False


def test_report_authenticated_fisher_controls_must_share_factor_bundle() -> None:
    rows = _arm_rows()
    family = _FAMILIES[0]
    bundle = rows[development.FISHER_PEDAL_ID]["fit_diagnostics"][family][
        "shared_factor_bundle"
    ]
    bundle["tensor_sha256s"]["direction_right"] = "a" * 64
    bundle["artifact_sha256"] = development._v14._sha256(
        {key: value for key, value in bundle.items() if key != "artifact_sha256"},
        domain=development._FACTOR_BUNDLE_DOMAIN,
    )
    with pytest.raises(ValueError, match="do not share one factor bundle"):
        development.evaluate_trust_gates(arm_rows=rows)


def test_equal_sequence_pedal_weighting_is_not_raw_row_weighting() -> None:
    diagnostic = _held_diagnostic(
        _FAMILIES[0],
        artifact="7" * 64,
        parent_artifact="8" * 64,
        objective="reverse_vjp_fisher",
        mode="conditional",
        pedal_rows=((0.0,), (1.0, 1.0, 1.0)),
    )
    distribution = diagnostic["pedal_distribution"]
    assert distribution["pedal_weighted_mean"] == pytest.approx(0.5)
    assert distribution["pedal_zero_weight_fraction"] == pytest.approx(0.5)
    assert distribution["pedal_unit_weight_fraction"] == pytest.approx(0.5)
    assert torch.tensor((0.0, 1.0, 1.0, 1.0)).mean().item() == pytest.approx(0.75)


def test_fisher_controls_must_share_exact_router_and_direction_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = SimpleNamespace(artifact_sha256="1" * 64)

    def provider(mode: str):
        return SimpleNamespace(
            parent_provider=parent,
            router_weight=torch.arange(8, dtype=torch.float64).reshape(4, 2),
            router_bias=torch.tensor((0.1, -0.2), dtype=torch.float64),
            coordinate_scales=torch.tensor((1.0, 2.0), dtype=torch.float64),
            direction_left=torch.arange(24, dtype=torch.float64).reshape(6, 4),
            direction_right=torch.arange(16, dtype=torch.float64).reshape(4, 4),
            coordinate_axes_sha256="2" * 64,
            coordinate_axis_values=(2.0, 1.0),
            bounded_coordinate_geometry_sha256="3" * 64,
            bounded_coordinate_covariance_eigenvalues=(1.0, 0.5),
            bounded_coordinate_lambda2_over_lambda1=0.5,
            bounded_coordinate_abs_correlation=0.1,
            bounded_coordinate_target_r2=(0.8, 0.7),
            residual_second_coordinate_energy_fraction=0.9,
            bounded_target_clipped_fraction=0.2,
            bounded_target_mean_clip_scale=0.8,
            direction_clipped_fraction=0.1,
            direction_mean_clip_scale=0.9,
            pedal_unclipped_target_sha256="4" * 64,
            pedal_target_sha256="5" * 64,
            pedal_fit_weight_sha256="6" * 64,
            pedal_support_mask_sha256="7" * 64,
            pedal_supported_row_count=10,
            pedal_unclipped_target_weighted_mean=0.5,
            pedal_target_weighted_mean=0.5,
            pedal_mode=mode,
        )

    values = (
        provider("unit"),
        provider("constant_optimal"),
        provider("conditional"),
        provider("conditional"),
    )
    monkeypatch.setattr(development, "_child_resources", lambda _value: {"x": 1})
    development._validate_parameter_matched_children(values)  # type: ignore[arg-type]
    values[2].direction_right[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="not exactly resource matched"):
        development._validate_parameter_matched_children(values)  # type: ignore[arg-type]


def test_report_separates_mechanism_support_absolute_and_full_refit() -> None:
    absolute_failure = development.build_fisher_pedal_development_report(
        **_report_kwargs(
            rows=_arm_rows(fisher_absolute_passed=False),
            qualification=None,
            candidate=None,
        )
    )
    assert absolute_failure["mechanism_support"]["passed"] is True
    assert absolute_failure["absolute_readiness"]["passed"] is False
    assert absolute_failure["passed"] is False
    assert absolute_failure["classification"] == (
        "fisher_pedal_mechanism_supported_absolute_fidelity_insufficient"
    )

    qualification = _full_qualification()
    passed = development.build_fisher_pedal_development_report(
        **_report_kwargs(
            rows=_arm_rows(),
            qualification=qualification,
            candidate=_candidate(qualification),
        )
    )
    assert passed["mechanism_support"]["passed"] is True
    assert passed["format_version"] == 18
    assert passed["schema"].endswith(".v18")
    assert passed["fixed_protocol"]["write_once_predecessor_preserved"] is True
    assert passed["fixed_protocol"]["v17_invalid_aggregation_receipt"][
        "file_sha256"
    ] == development._V17_FILE_SHA256
    assert passed["absolute_readiness"]["passed"] is True
    assert passed["candidate_readiness"]["passed"] is True
    assert passed["passed"] is True
    assert passed["fresh_guard_authorized"] is False
    json.dumps(passed, sort_keys=True, allow_nan=False)


def test_full_refit_rejects_constant_collapse_without_candidate() -> None:
    qualification = _full_qualification(passed=False)
    report = development.build_fisher_pedal_development_report(
        **_report_kwargs(
            rows=_arm_rows(), qualification=qualification, candidate=None
        )
    )
    assert report["mechanism_support"]["passed"] is True
    assert report["full_refit_qualification"]["passed"] is False
    assert report["candidate"] is None
    assert report["classification"] == (
        "fisher_pedal_full_refit_qualification_insufficient"
    )

    regression = _full_qualification()
    regression["trust"]["weighted_residual_rmse_after"] = 0.95
    regression["trust"]["gates"][
        "conditional_fit_not_worse_than_constant"
    ] = False
    regression["trust"]["passed"] = False
    regression["passed"] = False
    report = development.build_fisher_pedal_development_report(
        **_report_kwargs(rows=_arm_rows(), qualification=regression, candidate=None)
    )
    assert report["full_refit_qualification"]["passed"] is False
    assert report["candidate"] is None


def test_cross_arm_ownership_and_preflight_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _arm_rows(fisher_absolute_passed=False)
    rows[development.FISHER_CONSTANT_ID]["fold_ownership_receipts"][_FAMILIES[0]][
        "fit_sequence_sha256s"
    ] = rows[development.FISHER_CONSTANT_ID]["fold_ownership_receipts"][
        _FAMILIES[1]
    ]["fit_sequence_sha256s"]
    with pytest.raises(ValueError, match="ownership"):
        development.build_fisher_pedal_development_report(
            **_report_kwargs(rows=rows, qualification=None, candidate=None)
        )

    root = tmp_path / ".local-runs"
    root.mkdir()
    destination = root / "v17.json"
    monkeypatch.setattr(
        development,
        "prepare_complete_h4_rank320_live_context",
        lambda **_kwargs: pytest.fail("live model must not load"),
    )
    monkeypatch.setattr(
        development,
        "_validate_prerequisites",
        lambda: (_ for _ in ()).throw(RuntimeError("V16 drift")),
    )
    with pytest.raises(RuntimeError, match="V16 drift"):
        development.run_gemma3_l3_l4_complete_h4_fisher_pedal_development(
            output=destination
        )


def test_v18_runner_refuses_the_write_once_v17_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        development,
        "prepare_complete_h4_rank320_live_context",
        lambda **_kwargs: pytest.fail("V18 must reject V17 before model load"),
    )
    with pytest.raises(ValueError, match="preserve the write-once V17 rung"):
        development.run_gemma3_l3_l4_complete_h4_fisher_pedal_development(
            output=development._V17_OUTPUT
        )
    v17_alias = (
        development._V17_OUTPUT.parent
        / ".."
        / development._V17_OUTPUT.parent.name
        / development._V17_OUTPUT.name
    )
    assert v17_alias != development._V17_OUTPUT
    with pytest.raises(ValueError, match="preserve the write-once V17 rung"):
        development.run_gemma3_l3_l4_complete_h4_fisher_pedal_development(
            output=v17_alias
        )
    with pytest.raises(ValueError, match="preserve the write-once V17 rung"):
        development.run_gemma3_l3_l4_complete_h4_fisher_pedal_development(
            output=development._V17_OUTPUT.parent / "v18-alias-guard.json",
            provider_output=v17_alias.with_suffix(".provider.pt"),
        )


def test_v18_output_validation_rejects_paths_that_escape_local_runs() -> None:
    escaping_report = Path(".local-runs") / ".." / "escaped-v18.json"
    escaping_provider = Path(".local-runs") / ".." / "escaped-v18.provider.pt"
    with pytest.raises(ValueError, match="under .local-runs"):
        development._validate_output(escaping_report)
    with pytest.raises(ValueError, match="under .local-runs"):
        development._validate_provider_output(escaping_provider)


def test_publish_binds_and_roundtrips_selected_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _small_full_provider()
    qualification = development._full_provider_fit_qualification(provider)
    assert qualification["passed"] is True
    candidate = {
        "arm_id": development.FISHER_PEDAL_ID,
        "provider_artifact_sha256": provider.artifact_sha256,
        "parent_provider_artifact_sha256": (
            provider.parent_provider.artifact_sha256
        ),
        "fit_family_ids": provider.fit_family_ids,
        "fit_sequence_sha256s": provider.fit_sequence_sha256s,
        "full_provider_fit_qualification": qualification,
    }
    report = development.build_fisher_pedal_development_report(
        **_report_kwargs(
            rows=_arm_rows(), qualification=qualification, candidate=candidate
        )
    )
    # The publication machinery is rank-agnostic; only the live V17 runner's
    # fixed K256 validator is bypassed for this compact real-tensor fixture.
    monkeypatch.setattr(development, "_validate_child", lambda *_args, **_kwargs: None)
    output = tmp_path / ".local-runs" / "v17.json"
    provider_output = tmp_path / ".local-runs" / "v17.provider.pt"
    report["artifact"]["path"] = output.as_posix()
    published = development._publish(
        report,
        output=output,
        provider=provider,
        provider_output=provider_output,
    )
    receipt = published["candidate"]["provider_tensor_artifact"]
    assert output.exists() and provider_output.exists()
    assert stat.S_IMODE(provider_output.stat().st_mode) == 0o600
    restored = load_autonomous_complete_h4_fisher_xy_pedal_provider(
        provider_output,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_file_sha256=receipt["file_sha256"],
        expected_bridge_binding_sha256=_SHA,
    )
    assert restored.metadata() == provider.metadata()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["candidate"]["provider_tensor_artifact"] == receipt
    report_sha = persisted.pop("report_sha256")
    assert report_sha == development._v14._sha256(
        persisted, domain=development._REPORT_DOMAIN
    )


def test_publish_fail_closed_report_without_candidate_writes_no_provider(
    tmp_path: Path,
) -> None:
    report = development.build_fisher_pedal_development_report(
        **_report_kwargs(
            rows=_arm_rows(fisher_absolute_passed=False),
            qualification=None,
            candidate=None,
        )
    )
    output = tmp_path / ".local-runs" / "v17.json"
    provider_output = tmp_path / ".local-runs" / "v17.provider.pt"
    report["artifact"]["path"] = output.as_posix()
    published = development._publish(
        report,
        output=output,
        provider=None,
        provider_output=provider_output,
    )
    assert output.exists()
    assert not provider_output.exists()
    assert published["candidate"] is None
    persisted = json.loads(output.read_text(encoding="utf-8"))
    report_sha = persisted.pop("report_sha256")
    assert report_sha == development._v14._sha256(
        persisted, domain=development._REPORT_DOMAIN
    )


def test_publish_rejects_candidate_provider_mismatch_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _small_full_provider()
    qualification = development._full_provider_fit_qualification(provider)
    candidate = {
        "arm_id": development.FISHER_PEDAL_ID,
        "provider_artifact_sha256": provider.artifact_sha256,
        "parent_provider_artifact_sha256": "f" * 64,
        "fit_family_ids": provider.fit_family_ids,
        "fit_sequence_sha256s": provider.fit_sequence_sha256s,
        "full_provider_fit_qualification": qualification,
    }
    report = development.build_fisher_pedal_development_report(
        **_report_kwargs(
            rows=_arm_rows(), qualification=qualification, candidate=candidate
        )
    )
    monkeypatch.setattr(development, "_validate_child", lambda *_args, **_kwargs: None)
    output = tmp_path / ".local-runs" / "v17.json"
    provider_output = tmp_path / ".local-runs" / "v17.provider.pt"
    report["artifact"]["path"] = output.as_posix()
    with pytest.raises(ValueError, match="candidate and provider differ"):
        development._publish(
            report,
            output=output,
            provider=provider,
            provider_output=provider_output,
        )
    assert not output.exists()
    assert not provider_output.exists()
