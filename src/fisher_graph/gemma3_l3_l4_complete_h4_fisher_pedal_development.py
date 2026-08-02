"""Fixed V18 bounded Fisher-pedal outer-LOFO development screen.

V18 keeps the V16 K256/L8 reverse-VJP parent and two-coordinate square, but
replaces its global four-corner projection with a per-row relative trust ball
and a learned confidence pedal.  Five fixed arms are evaluated on the opened
A16 panel: parent, Fisher unit pedal, Fisher fit-optimal constant pedal,
Fisher conditional pedal, and an exactly parameter-matched activation-PCA
conditional pedal.  Every provider is fitted inside its train-seven outer
fold and the full Gemma suffix and vocabulary remain source-authoritative.

Mechanism support is reported separately from absolute executor readiness.
A full-panel provider is fitted and serialized only when coordinate geometry,
pointwise trust, absolute fidelity, and every preregistered mechanism gate
pass.  This runner never opens a fresh guard or Calibration B.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from .complete_h4_autonomous_residual import (
    AutonomousCompleteH4ResidualProvider,
    AutonomousCompleteH4TrainingSequence,
)
from .complete_h4_fisher_conditional_pedal import (
    FISHER_XY_PEDAL_ABSOLUTE_ENERGY_FLOOR,
    FISHER_XY_PEDAL_RELATIVE_ENERGY_FLOOR,
    FISHER_XY_PEDAL_TRUST_FRACTION,
    AutonomousCompleteH4FisherXYPedalProvider,
    FisherXYPedalRuntimeReplay,
    autonomous_complete_h4_fisher_xy_pedal_provider_state_dict,
    fit_autonomous_complete_h4_fisher_xy_pedal,
    fisher_xy_pedal_fit_support_mask,
    load_autonomous_complete_h4_fisher_xy_pedal_provider,
    replay_autonomous_complete_h4_fisher_xy_pedal,
    validate_fisher_xy_pedal_runtime_replay_metadata,
)
from .complete_h4_fisher_conditional_residual import (
    FISHER_XY_COORDINATE_COUNT,
    summarize_fisher_xy_bounded_coordinate_geometry,
)
from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from . import gemma3_l3_l4_complete_h4_fisher_square_development as _v16
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassExecution,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    SourceAuthoritativeShadowFidelityAccumulator,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_PROVIDER_OUTPUT",
    "PARENT_ID",
    "FISHER_UNIT_ID",
    "FISHER_CONSTANT_ID",
    "FISHER_PEDAL_ID",
    "PCA_PEDAL_ID",
    "PARENT_RECIPE",
    "build_fisher_pedal_development_report",
    "evaluate_coordinate_geometry_gates",
    "evaluate_mechanism_gates",
    "evaluate_trust_gates",
    "run_gemma3_l3_l4_complete_h4_fisher_pedal_development",
    "build_parser",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_V17_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-fisher-pedal-r16-k256-"
    "outer-lofo-a-fit16-dev-v17.json"
)
_V17_LOGICAL_SHA256 = (
    "f59ed94966ebda89296ac1ef26ead49f464b410ac9fb8221f9e44cdfa105d1b7"
)
_V17_FILE_SHA256 = (
    "1439ffe8439e47d37d8e610d0094912cfa86e006558a222cfd10b087c037c401"
)
_V17_CLASSIFICATION = "fisher_pedal_pointwise_trust_insufficient"
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-fisher-pedal-r16-k256-"
    "outer-lofo-a-fit16-dev-v18.json"
)
DEFAULT_PROVIDER_OUTPUT = DEFAULT_OUTPUT.with_suffix(".provider.pt")
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_fisher_pedal_"
    "outer_lofo_development.v18"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-fisher-pedal-dev:v18\0"
_HELD_DIAGNOSTIC_DOMAIN = b"fisher-graph:fisher-pedal-held-runtime:v18\0"
_FACTOR_BUNDLE_DOMAIN = b"fisher-graph:fisher-pedal-factor-bundle:v18\0"

_EXPECTED_PROMPTS = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_HELD_SEQUENCES_PER_FOLD = 2
_EXPECTED_TRAINING_PROMPTS_PER_FOLD = 14
_EXPECTED_OUTER_PROVIDER_FITS = 40
_EXPECTED_FULL_MODEL_FORWARDS = 144
_EXPECTED_BACKWARD_VJP_TRAVERSALS = 16
_EXPECTED_CAUSAL_CHECKS = 80
_EXPECTED_HELD_RUNTIME_DIAGNOSTICS = 32
_EXPECTED_PARENT_SCALARS = 360_704
_EXPECTED_PARENT_MACS = 524_288
_EXPECTED_CHILD_INCREMENTAL_SCALARS = 16_904
_EXPECTED_CHILD_INCREMENTAL_MACS = 16_899
_EXPECTED_CHILD_SCALARS = 377_608
_EXPECTED_CHILD_MACS = 541_187

_V16_REPORT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-fisher-square-r16-k256-"
    "outer-lofo-held-runtime-geometry-a-fit16-dev-v16.json"
)
_V16_LOGICAL_SHA256 = (
    "fa6d89d49cd2b041c361a50efeb8b606d6fa0b72be74c83235950a9cdd7ef2ff"
)
_V16_FILE_SHA256 = (
    "14a2eb93cda810cd68ff859c4731f0991daa65084b3917eb8442537e0b54ad31"
)
_V16_CLASSIFICATION = "fisher_square_absolute_fidelity_insufficient"

_CONDITIONAL_RANK = 16
_ROUTER_RIDGE = 1.0e-4
_DIRECTION_RIDGE = 1.0e-4
_PEDAL_RIDGE = 1.0e-4
_TRUST_FRACTION = FISHER_XY_PEDAL_TRUST_FRACTION

PARENT_ID = "r256_l8_reverse_vjp_parent"
FISHER_UNIT_ID = "r256_l8_reverse_vjp_fisher_unit_pedal_r16"
FISHER_CONSTANT_ID = "r256_l8_reverse_vjp_fisher_constant_pedal_r16"
FISHER_PEDAL_ID = "r256_l8_reverse_vjp_fisher_conditional_pedal_r16"
PCA_PEDAL_ID = "r256_l8_reverse_vjp_activation_pca_conditional_pedal_r16"
_CHILD_IDS = (
    FISHER_UNIT_ID,
    FISHER_CONSTANT_ID,
    FISHER_PEDAL_ID,
    PCA_PEDAL_ID,
)
_ARM_IDS = (PARENT_ID, *_CHILD_IDS)
_PEDAL_MODES = {
    FISHER_UNIT_ID: "unit",
    FISHER_CONSTANT_ID: "constant_optimal",
    FISHER_PEDAL_ID: "conditional",
    PCA_PEDAL_ID: "conditional",
}
_COORDINATE_OBJECTIVES = {
    PARENT_ID: "reverse_vjp_weighted_shared_parent",
    FISHER_UNIT_ID: "reverse_vjp_fisher",
    FISHER_CONSTANT_ID: "reverse_vjp_fisher",
    FISHER_PEDAL_ID: "reverse_vjp_fisher",
    PCA_PEDAL_ID: "activation_pca",
}

PARENT_RECIPE = _v14.AutonomousResidualRecipe(
    recipe_id=PARENT_ID,
    rank=256,
    lag_count=8,
    ridge=1.0e-4,
    fit_objective="reverse_vjp_row_weighted_ridge_v1",
)

_MECHANISM_THRESHOLDS = {
    "versus_parent_macro_absolute_delta_nll_relative_improvement_min": 0.05,
    "versus_parent_macro_kl_relative_improvement_min": 0.05,
    "versus_parent_aggregate_top1_gain_min": 0.02,
    "versus_parent_family_absolute_delta_nll_win_count_min": 6,
    "versus_parent_worst_family_relative_regression_max": 0.02,
    "versus_constant_macro_absolute_delta_nll_relative_improvement_min": 0.01,
    "versus_constant_kl_rule": "not_higher",
    "versus_constant_top1_rule": "not_lower",
    "versus_constant_family_absolute_delta_nll_win_count_min": 5,
    "versus_pca_absolute_delta_nll_rule": "strictly_lower",
    "versus_pca_kl_rule": "not_higher",
    "versus_pca_top1_rule": "not_lower",
    "support_and_graph_core_rule": "no_aggregate_delta_nll_kl_or_top1_regression",
}
_RUNTIME_INPUT_FIELDS = (
    "source_modes",
    "logical_positions",
    "valid_mask",
    "source_mask",
    "support_mask",
    "base_h4",
)


def _is_under_local_runs(path: Path) -> bool:
    """Require a real ``.local-runs`` ancestor after resolving aliases."""

    return ".local-runs" in path.resolve(strict=False).parts


def _validate_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".json" or not _is_under_local_runs(output):
        raise ValueError("V18 output must be JSON under .local-runs")
    return output


def _validate_provider_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".pt" or not _is_under_local_runs(output):
        raise ValueError("V18 provider output must be PT under .local-runs")
    return output


def _same_destination(left: Path, right: Path) -> bool:
    """Compare normalized destinations so ``..`` aliases cannot bypass guards."""

    return left.resolve(strict=False) == right.resolve(strict=False)


def _validate_prerequisite_report(
    path: Path,
    *,
    logical_sha256: str,
    file_sha256: str,
    classification: str,
    format_version: int,
) -> dict[str, object]:
    return _v16._validate_prerequisite_report(
        path,
        logical_sha256=logical_sha256,
        file_sha256=file_sha256,
        classification=classification,
        format_version=format_version,
    )


def _validate_prerequisites() -> dict[str, object]:
    return {
        "v16": _validate_prerequisite_report(
            _V16_REPORT,
            logical_sha256=_V16_LOGICAL_SHA256,
            file_sha256=_V16_FILE_SHA256,
            classification=_V16_CLASSIFICATION,
            format_version=16,
        ),
        "v17_invalid_held_aggregation_receipt": _validate_prerequisite_report(
            _V17_OUTPUT,
            logical_sha256=_V17_LOGICAL_SHA256,
            file_sha256=_V17_FILE_SHA256,
            classification=_V17_CLASSIFICATION,
            format_version=17,
        ),
    }


def _relative_improvement(reference: float, candidate: float, *, label: str) -> float:
    if reference <= 0.0:
        raise ValueError(f"{label} reference must be positive")
    return (reference - candidate) / reference


def _work_accounting(
    *,
    prompt_count: int,
    outer_fold_count: int,
    full_provider_fitted: bool,
) -> dict[str, object]:
    if (
        prompt_count != _EXPECTED_PROMPTS
        or outer_fold_count != _EXPECTED_FAMILIES
        or type(full_provider_fitted) is not bool
    ):
        raise RuntimeError("V17 work geometry differs")
    breakdown = {
        "fit_native_source_forwards": prompt_count,
        "fit_base_vjp_forwards": prompt_count,
        "fit_base_vjp_backward_traversals": prompt_count,
        "evaluation_native_source_forwards": prompt_count,
        "evaluation_base_forwards": prompt_count,
        "evaluation_parent_forwards": prompt_count,
        "evaluation_fisher_unit_forwards": prompt_count,
        "evaluation_fisher_constant_forwards": prompt_count,
        "evaluation_fisher_pedal_forwards": prompt_count,
        "evaluation_pca_pedal_forwards": prompt_count,
    }
    total_forwards = sum(
        int(value)
        for name, value in breakdown.items()
        if name != "fit_base_vjp_backward_traversals"
    )
    total_backwards = int(breakdown["fit_base_vjp_backward_traversals"])
    full_fit_count = 2 * int(full_provider_fitted)
    outer_fit_count = outer_fold_count * len(_ARM_IDS)
    if (
        total_forwards != _EXPECTED_FULL_MODEL_FORWARDS
        or total_backwards != _EXPECTED_BACKWARD_VJP_TRAVERSALS
        or outer_fit_count != _EXPECTED_OUTER_PROVIDER_FITS
    ):
        raise RuntimeError("V17 exact work count differs")
    return {
        "outer_parent_fit_count": outer_fold_count,
        "outer_child_fit_count": outer_fold_count * len(_CHILD_IDS),
        "outer_provider_fit_count": outer_fit_count,
        "expected_outer_provider_fit_count": _EXPECTED_OUTER_PROVIDER_FITS,
        "conditional_full_panel_provider_fit_count": full_fit_count,
        "fit_provider_count": outer_fit_count + full_fit_count,
        "expected_fit_provider_count": _EXPECTED_OUTER_PROVIDER_FITS + full_fit_count,
        "full_model_forward_count": total_forwards,
        "expected_full_model_forward_count": _EXPECTED_FULL_MODEL_FORWARDS,
        "backward_vjp_traversal_count": total_backwards,
        "expected_backward_vjp_traversal_count": _EXPECTED_BACKWARD_VJP_TRAVERSALS,
        "full_model_work_breakdown": {
            **breakdown,
            "total_forwards": total_forwards,
            "total_backward_vjp_traversals": total_backwards,
        },
    }


def _parent_resources(
    provider: AutonomousCompleteH4ResidualProvider,
) -> dict[str, object]:
    resources = _v16._parent_resources(provider)
    if (
        resources["prepared_float_scalar_count"] != _EXPECTED_PARENT_SCALARS
        or resources["logical_macs_per_token_upper_bound"] != _EXPECTED_PARENT_MACS
    ):
        raise RuntimeError("V17 parent resource geometry differs")
    return resources


def _child_resources(
    provider: AutonomousCompleteH4FisherXYPedalProvider,
) -> dict[str, object]:
    metadata = provider.metadata()
    resources = {
        "scope": "incremental_fisher_pedal_provider_including_k256_parent",
        "prepared_float_scalar_count": metadata["prepared_float_scalar_count"],
        "runtime_parameter_bytes_float64": metadata[
            "runtime_parameter_bytes_float64"
        ],
        "logical_macs_per_token_upper_bound": metadata[
            "logical_macs_per_token_upper_bound"
        ],
        "incremental_child_prepared_float_scalar_count": metadata[
            "incremental_prepared_float_scalar_count"
        ],
        "incremental_child_runtime_parameter_bytes_float64": metadata[
            "incremental_runtime_parameter_bytes_float64"
        ],
        "incremental_child_logical_macs_per_token_upper_bound": metadata[
            "incremental_logical_macs_per_token_upper_bound"
        ],
        "retained_gemma_parameters_excluded": True,
        "base_bridge_and_full_suffix_macs_excluded": True,
        "elementwise_norm_clamp_and_scaling_ops_excluded_from_matrix_macs": True,
        "end_to_end_model_parameter_or_flop_claim": False,
    }
    if (
        resources["prepared_float_scalar_count"] != _EXPECTED_CHILD_SCALARS
        or resources["runtime_parameter_bytes_float64"]
        != _EXPECTED_CHILD_SCALARS * 8
        or resources["logical_macs_per_token_upper_bound"]
        != _EXPECTED_CHILD_MACS
        or resources["incremental_child_prepared_float_scalar_count"]
        != _EXPECTED_CHILD_INCREMENTAL_SCALARS
        or resources["incremental_child_runtime_parameter_bytes_float64"]
        != _EXPECTED_CHILD_INCREMENTAL_SCALARS * 8
        or resources["incremental_child_logical_macs_per_token_upper_bound"]
        != _EXPECTED_CHILD_INCREMENTAL_MACS
    ):
        raise RuntimeError("V17 child resource geometry differs")
    return resources


def _validate_parent(
    provider: AutonomousCompleteH4ResidualProvider,
    *,
    expected_fit_family_count: int,
) -> None:
    _v16._validate_parent(
        provider,
        expected_fit_family_count=expected_fit_family_count,
    )


def _validate_child(
    provider: AutonomousCompleteH4FisherXYPedalProvider,
    *,
    coordinate_objective: str,
    pedal_mode: str,
    expected_parent_artifact_sha256: str,
    expected_fit_family_count: int,
) -> None:
    if not isinstance(provider, AutonomousCompleteH4FisherXYPedalProvider):
        raise TypeError("V17 child must be a Fisher-XY pedal provider")
    provider.validate_integrity()
    if (
        provider.coordinate_objective != coordinate_objective
        or provider.pedal_mode != pedal_mode
        or provider.conditional_rank != _CONDITIONAL_RANK
        or provider.router_ridge != _ROUTER_RIDGE
        or provider.direction_ridge != _DIRECTION_RIDGE
        or provider.pedal_ridge != _PEDAL_RIDGE
        or provider.trust_fraction != _TRUST_FRACTION
        or provider.parent_provider.artifact_sha256
        != expected_parent_artifact_sha256
        or len(provider.fit_family_ids) != expected_fit_family_count
        or provider.fit_sequence_sha256s
        != provider.parent_provider.fit_sequence_sha256s
    ):
        raise RuntimeError("V17 child protocol differs")
    _child_resources(provider)


def _validate_parameter_matched_children(
    providers: Sequence[AutonomousCompleteH4FisherXYPedalProvider],
) -> None:
    selected = tuple(providers)
    if len(selected) != len(_CHILD_IDS):
        raise RuntimeError("V17 parameter-match set differs")
    resources = tuple(_child_resources(provider) for provider in selected)
    fisher = selected[:3]
    shared_tensors = (
        "router_weight",
        "router_bias",
        "coordinate_scales",
        "direction_left",
        "direction_right",
    )
    shared_receipts = (
        "coordinate_axes_sha256",
        "coordinate_axis_values",
        "bounded_coordinate_geometry_sha256",
        "bounded_coordinate_covariance_eigenvalues",
        "bounded_coordinate_lambda2_over_lambda1",
        "bounded_coordinate_abs_correlation",
        "bounded_coordinate_target_r2",
        "residual_second_coordinate_energy_fraction",
        "bounded_target_clipped_fraction",
        "bounded_target_mean_clip_scale",
        "direction_clipped_fraction",
        "direction_mean_clip_scale",
        "pedal_unclipped_target_sha256",
        "pedal_target_sha256",
        "pedal_fit_weight_sha256",
        "pedal_support_mask_sha256",
        "pedal_supported_row_count",
        "pedal_unclipped_target_weighted_mean",
        "pedal_target_weighted_mean",
    )
    if (
        len({provider.parent_provider.artifact_sha256 for provider in selected}) != 1
        or any(value != resources[0] for value in resources[1:])
        or any(
            not all(
                torch.equal(
                    getattr(provider, tensor_name),
                    getattr(fisher[0], tensor_name),
                )
                for provider in fisher[1:]
            )
            for tensor_name in shared_tensors
        )
        or any(
            not all(
                getattr(provider, receipt_name)
                == getattr(fisher[0], receipt_name)
                for provider in fisher[1:]
            )
            for receipt_name in shared_receipts
        )
    ):
        raise RuntimeError("V17 child controls are not exactly resource matched")


def _authenticated_fit_diagnostics(
    provider: AutonomousCompleteH4FisherXYPedalProvider,
) -> dict[str, object]:
    provider.validate_integrity()
    factor_payload = {
        "coordinate_objective": provider.coordinate_objective,
        "tensor_sha256s": {
            name: _v14._tensor_sha256(getattr(provider, name))
            for name in (
                "router_weight",
                "router_bias",
                "coordinate_scales",
                "direction_left",
                "direction_right",
            )
        },
        "coordinate_axes_sha256": provider.coordinate_axes_sha256,
        "coordinate_axis_values": provider.coordinate_axis_values,
        "bounded_coordinate_geometry_sha256": (
            provider.bounded_coordinate_geometry_sha256
        ),
        "pedal_unclipped_target_sha256": provider.pedal_unclipped_target_sha256,
        "pedal_fit_weight_sha256": provider.pedal_fit_weight_sha256,
        "pedal_support_mask_sha256": provider.pedal_support_mask_sha256,
    }
    return {
        "provider_artifact_sha256": provider.artifact_sha256,
        "coordinate_objective": provider.coordinate_objective,
        "bounded_coordinate_geometry_sha256": (
            provider.bounded_coordinate_geometry_sha256
        ),
        "bounded_coordinate_covariance_eigenvalues": (
            provider.bounded_coordinate_covariance_eigenvalues
        ),
        "bounded_coordinate_lambda2_over_lambda1": (
            provider.bounded_coordinate_lambda2_over_lambda1
        ),
        "bounded_coordinate_abs_correlation": (
            provider.bounded_coordinate_abs_correlation
        ),
        "bounded_coordinate_target_r2": provider.bounded_coordinate_target_r2,
        "residual_second_coordinate_energy_fraction": (
            provider.residual_second_coordinate_energy_fraction
        ),
        "pedal_mode": provider.pedal_mode,
        "pedal_unclipped_target_sha256": provider.pedal_unclipped_target_sha256,
        "pedal_target_sha256": provider.pedal_target_sha256,
        "pedal_fit_weight_sha256": provider.pedal_fit_weight_sha256,
        "pedal_support_mask_sha256": provider.pedal_support_mask_sha256,
        "pedal_supported_row_count": provider.pedal_supported_row_count,
        "pedal_unclipped_target_weighted_mean": (
            provider.pedal_unclipped_target_weighted_mean
        ),
        "pedal_unclipped_target_weighted_rmse": (
            provider.pedal_unclipped_target_weighted_rmse
        ),
        "bounded_target_clipped_fraction": provider.bounded_target_clipped_fraction,
        "bounded_target_mean_clip_scale": provider.bounded_target_mean_clip_scale,
        "direction_clipped_fraction": provider.direction_clipped_fraction,
        "direction_mean_clip_scale": provider.direction_mean_clip_scale,
        "pedal_target_weighted_rmse": provider.pedal_target_weighted_rmse,
        "pedal_weighted_mean": provider.pedal_weighted_mean,
        "pedal_weighted_std": provider.pedal_weighted_std,
        "pedal_min": provider.pedal_min,
        "pedal_max": provider.pedal_max,
        "pedal_effective_weighted_mean": provider.pedal_effective_weighted_mean,
        "pedal_effective_weighted_std": provider.pedal_effective_weighted_std,
        "pedal_effective_min": provider.pedal_effective_min,
        "pedal_effective_max": provider.pedal_effective_max,
        "pedal_slope_l2_norm": float(
            torch.linalg.vector_norm(provider.pedal_weight)
        ),
        "pedal_zero_fraction": provider.pedal_zero_fraction,
        "pedal_one_fraction": provider.pedal_one_fraction,
        "pedal_target_clipped_fraction": provider.pedal_target_clipped_fraction,
        "weighted_bounded_target_rmse_before": (
            provider.weighted_bounded_target_rmse_before
        ),
        "weighted_bounded_target_rmse_after": (
            provider.weighted_bounded_target_rmse_after
        ),
        "weighted_residual_rmse_before": provider.weighted_residual_rmse_before,
        "weighted_residual_rmse_unit": provider.weighted_residual_rmse_unit,
        "weighted_residual_rmse_constant": (
            provider.weighted_residual_rmse_constant
        ),
        "weighted_residual_rmse_oracle": provider.weighted_residual_rmse_oracle,
        "weighted_residual_rmse_after": provider.weighted_residual_rmse_after,
        "fit_bounded_direction_ratio_quantiles": (
            provider.fit_bounded_direction_ratio_quantiles
        ),
        "fit_emitted_delta_ratio_quantiles": (
            provider.fit_emitted_delta_ratio_quantiles
        ),
        "fit_max_bounded_direction_ratio": (
            provider.fit_max_bounded_direction_ratio
        ),
        "fit_max_emitted_delta_ratio": provider.fit_max_emitted_delta_ratio,
        "trust_fraction": provider.trust_fraction,
        "shared_factor_bundle": {
            **factor_payload,
            "artifact_sha256": _v14._sha256(
                factor_payload, domain=_FACTOR_BUNDLE_DOMAIN
            ),
        },
    }


def _validate_factor_bundle(
    value: object,
    *,
    expected_coordinate_objective: str,
) -> dict[str, object]:
    row = _v16._mapping(value, label="V17 shared factor bundle")
    artifact = _v16._sha256_identifier(
        row.get("artifact_sha256"), label="V17 shared factor bundle artifact"
    )
    payload = {key: item for key, item in row.items() if key != "artifact_sha256"}
    if (
        payload.get("coordinate_objective") != expected_coordinate_objective
        or _v14._sha256(payload, domain=_FACTOR_BUNDLE_DOMAIN) != artifact
    ):
        raise ValueError("V17 shared factor bundle receipt differs")
    tensors = _v16._mapping(
        payload.get("tensor_sha256s"), label="V17 shared factor tensor hashes"
    )
    expected_tensor_names = {
        "router_weight",
        "router_bias",
        "coordinate_scales",
        "direction_left",
        "direction_right",
    }
    if set(tensors) != expected_tensor_names:
        raise ValueError("V17 shared factor tensor fields differ")
    for name in expected_tensor_names:
        _v16._sha256_identifier(tensors[name], label=f"V17 {name} tensor")
    for name in (
        "coordinate_axes_sha256",
        "bounded_coordinate_geometry_sha256",
        "pedal_unclipped_target_sha256",
        "pedal_fit_weight_sha256",
        "pedal_support_mask_sha256",
    ):
        _v16._sha256_identifier(payload.get(name), label=f"V17 {name}")
    _v16._pair(payload.get("coordinate_axis_values"), label="V17 coordinate axes")
    return {**payload, "artifact_sha256": artifact}


def _provider_fold_ownership_receipt(
    provider: AutonomousCompleteH4ResidualProvider
    | AutonomousCompleteH4FisherXYPedalProvider,
    *,
    held_family_id: str,
    held_sequences: Sequence[AutonomousCompleteH4TrainingSequence],
) -> dict[str, object]:
    held_family = _v14._identifier(held_family_id, label="held family")
    selected_sequences = tuple(
        sorted(held_sequences, key=lambda value: value.artifact_sha256)
    )
    held_sequence_sha256s = tuple(
        value.artifact_sha256 for value in selected_sequences
    )
    if (
        len(selected_sequences) != _EXPECTED_HELD_SEQUENCES_PER_FOLD
        or {value.family_id for value in selected_sequences} != {held_family}
    ):
        raise RuntimeError("V17 held sequence ownership differs")
    receipt: dict[str, object] = {
        "held_family_id": held_family,
        "provider_artifact_sha256": provider.artifact_sha256,
        "fit_family_ids": provider.fit_family_ids,
        "fit_sequence_sha256s": provider.fit_sequence_sha256s,
        "held_sequence_sha256s": held_sequence_sha256s,
    }
    is_child = isinstance(provider, AutonomousCompleteH4FisherXYPedalProvider)
    if is_child:
        receipt.update(
            {
                "parent_provider_artifact_sha256": (
                    provider.parent_provider.artifact_sha256
                ),
                "coordinate_objective": provider.coordinate_objective,
                "pedal_mode": provider.pedal_mode,
            }
        )
    # V16 authenticates the common LOFO ownership fields.  Pedal mode is an
    # additional V17 binding checked by this harness.
    common = {key: value for key, value in receipt.items() if key != "pedal_mode"}
    _v16._fold_ownership_receipt(
        common,
        expected_held_family_id=held_family,
        expected_provider_artifact_sha256=provider.artifact_sha256,
        expected_parent_provider_artifact_sha256=(
            provider.parent_provider.artifact_sha256 if is_child else None
        ),
        expected_coordinate_objective=(
            provider.coordinate_objective if is_child else None
        ),
    )
    return receipt


def _replay_payload(replay: FisherXYPedalRuntimeReplay) -> dict[str, object]:
    replay.validate_integrity()
    return replay.metadata()


def _realized_convex_weighted_stats(
    values: Tensor,
    weights: Tensor,
    *,
    label: str,
) -> tuple[float, float, float, float]:
    weight_total = weights.sum()
    if not bool(torch.isfinite(weight_total)) or not bool(weight_total > 0.0):
        raise RuntimeError(f"{label} weight total became invalid")
    minimum = float(values.min())
    maximum = float(values.max())
    mean = float((weights * values).sum() / weight_total)
    tolerance = 128.0 * torch.finfo(torch.float64).eps * max(
        1.0,
        abs(minimum),
        abs(maximum),
    )
    if mean < minimum - tolerance or mean > maximum + tolerance:
        raise RuntimeError(f"{label} escaped its observed convex range")
    mean = min(max(mean, minimum), maximum)
    std = float(
        torch.sqrt((weights * (values - mean).square()).sum() / weight_total)
    )
    return minimum, mean, std, maximum


def _held_runtime_diagnostics(
    provider: AutonomousCompleteH4FisherXYPedalProvider,
    *,
    held_family_id: str,
    held_sequences: Sequence[AutonomousCompleteH4TrainingSequence],
) -> dict[str, object]:
    ownership = _provider_fold_ownership_receipt(
        provider,
        held_family_id=held_family_id,
        held_sequences=held_sequences,
    )
    selected_sequences = tuple(
        sorted(held_sequences, key=lambda value: value.artifact_sha256)
    )
    replays: list[FisherXYPedalRuntimeReplay] = []
    sequence_receipts: list[dict[str, object]] = []
    weights: list[Tensor] = []
    effective_pedals: list[Tensor] = []
    effective_weights: list[Tensor] = []
    effective_counts: list[int] = []
    for sequence in selected_sequences:
        replay = replay_autonomous_complete_h4_fisher_xy_pedal(provider, sequence)
        metadata = _replay_payload(replay)
        if (
            validate_fisher_xy_pedal_runtime_replay_metadata(metadata)[
                "artifact_sha256"
            ]
            != replay.artifact_sha256
        ):
            raise RuntimeError("held pedal replay artifact authentication failed")
        replays.append(replay)
        sequence_row_weights = torch.full(
            (replay.row_count,),
            1.0 / (len(selected_sequences) * replay.row_count),
            dtype=torch.float64,
        )
        weights.append(sequence_row_weights)
        effective_support = fisher_xy_pedal_fit_support_mask(
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
        sequence_receipts.append(
            {
                "sequence_sha256": sequence.artifact_sha256,
                "row_count": replay.row_count,
                "effective_row_count": effective_count,
                "effective_support_mask_sha256": _v14._tensor_sha256(
                    effective_support
                ),
                "bounded_coordinates_sha256": metadata["tensor_sha256s"][
                    "bounded_coordinates"
                ],
                "runtime_replay_artifact_sha256": replay.artifact_sha256,
                "runtime_replay": metadata,
            }
        )
    coordinates = torch.cat([value.bounded_coordinates for value in replays], dim=0)
    pedals = torch.cat([value.pedal for value in replays], dim=0)
    row_weights = torch.cat(weights, dim=0)
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
    positive_parent = parent_norm > 0.0
    safe_parent = torch.where(positive_parent, parent_norm, torch.ones_like(parent_norm))
    bounded_ratios = torch.where(
        positive_parent, bounded_norm / safe_parent, torch.zeros_like(parent_norm)
    )
    emitted_ratios = torch.where(
        positive_parent, emitted_norm / safe_parent, torch.zeros_like(parent_norm)
    )
    if provider.pedal_mode == "unit":
        pedal_min = pedal_mean = pedal_max = 1.0
        pedal_zero_fraction = 0.0
        pedal_unit_fraction = 1.0
    elif provider.pedal_mode == "constant_optimal":
        exact_pedal = float(provider.pedal_bias[0])
        pedal_min = pedal_mean = pedal_max = exact_pedal
        pedal_zero_fraction = 1.0 if exact_pedal == 0.0 else 0.0
        pedal_unit_fraction = 1.0 if exact_pedal == 1.0 else 0.0
    else:
        pedal_min, pedal_mean, _pedal_std, pedal_max = (
            _realized_convex_weighted_stats(
                pedals,
                row_weights,
                label="held conditional pedal",
            )
        )
        realized_row_weight_total = row_weights.sum()
        pedal_zero_fraction = min(
            max(
                float(
                    (row_weights * (pedals == 0.0)).sum()
                    / realized_row_weight_total
                ),
                0.0,
            ),
            1.0,
        )
        pedal_unit_fraction = min(
            max(
                float(
                    (row_weights * (pedals == 1.0)).sum()
                    / realized_row_weight_total
                ),
                0.0,
            ),
            1.0,
        )
    pedal_distribution = {
        "weighting_semantics": "equal_sequences_then_equal_supported_rows",
        "pedal_min": pedal_min,
        "pedal_weighted_mean": pedal_mean,
        "pedal_max": pedal_max,
        "pedal_zero_weight_fraction": pedal_zero_fraction,
        "pedal_unit_weight_fraction": pedal_unit_fraction,
    }
    every_sequence_has_effective_rows = all(value > 0 for value in effective_counts)
    if every_sequence_has_effective_rows:
        selected_effective_pedals = torch.cat(effective_pedals)
        raw_effective_weights = torch.cat(effective_weights)
        if provider.pedal_mode == "unit":
            effective_min = effective_mean = effective_max = 1.0
            effective_std = 0.0
        elif provider.pedal_mode == "constant_optimal":
            exact_pedal = float(provider.pedal_bias[0])
            effective_min = effective_mean = effective_max = exact_pedal
            effective_std = 0.0
        else:
            effective_min, effective_mean, effective_std, effective_max = (
                _realized_convex_weighted_stats(
                    selected_effective_pedals,
                    raw_effective_weights,
                    label="held conditional effective pedal",
                )
            )
    else:
        # Keep the report finite and explicitly fail closed.  Zero placeholders
        # are not interpreted as an observed pedal distribution because the
        # every-sequence gate below remains false.
        effective_mean = effective_std = effective_min = effective_max = 0.0
    pedal_effective_distribution = {
        "weighting_semantics": (
            "normalized_equal_sequence_row_weight_times_bounded_direction_"
            "energy_on_effective_support"
        ),
        "effective_row_count": sum(effective_counts),
        "effective_sequence_count": sum(value > 0 for value in effective_counts),
        "every_sequence_has_effective_rows": every_sequence_has_effective_rows,
        "pedal_effective_min": effective_min,
        "pedal_effective_weighted_mean": effective_mean,
        "pedal_effective_weighted_std": effective_std,
        "pedal_effective_max": effective_max,
    }
    runtime_trust = {
        "max_bounded_direction_to_parent_norm_ratio": float(
            bounded_ratios.max()
        ),
        "max_emitted_delta_to_parent_norm_ratio": float(emitted_ratios.max()),
        "pointwise_trust_certificate_passed": bool(
            float(bounded_ratios.max()) <= provider.trust_fraction + 1.0e-14
            and float(emitted_ratios.max()) <= provider.trust_fraction + 1.0e-14
        ),
    }
    geometry = summarize_fisher_xy_bounded_coordinate_geometry(
        coordinates,
        row_weights,
    )
    payload = {
        "provider_artifact_sha256": provider.artifact_sha256,
        "parent_provider_artifact_sha256": provider.parent_provider.artifact_sha256,
        "coordinate_objective": provider.coordinate_objective,
        "pedal_mode": provider.pedal_mode,
        "trust_fraction": provider.trust_fraction,
        "held_family_id": held_family_id,
        "held_sequence_sha256s": ownership["held_sequence_sha256s"],
        "sequence_coordinate_receipts": tuple(sequence_receipts),
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
        "pedal_distribution": pedal_distribution,
        "pedal_effective_distribution": pedal_effective_distribution,
        "runtime_trust": runtime_trust,
    }
    return {
        **payload,
        "held_diagnostic_sha256": _v14._sha256(
            payload,
            domain=_HELD_DIAGNOSTIC_DOMAIN,
        ),
    }


def evaluate_coordinate_geometry_gates(
    *,
    fisher_child: Mapping[str, object],
    pca_child: Mapping[str, object],
) -> dict[str, object]:
    """Reuse the corrected V16 fit/held coordinate qualification exactly."""

    fisher = dict(fisher_child)
    pca = dict(pca_child)
    fisher["arm_id"] = _v16.FISHER_CHILD_ID
    pca["arm_id"] = _v16.PCA_CHILD_ID
    fisher["held_runtime_coordinate_diagnostics"] = fisher.get(
        "held_runtime_diagnostics"
    )
    pca["held_runtime_coordinate_diagnostics"] = pca.get(
        "held_runtime_diagnostics"
    )
    qualification = _v16.evaluate_coordinate_geometry_gates(
        fisher_child=fisher,
        pca_child=pca,
    )
    arms = dict(qualification["arms"])
    fisher_result = dict(arms.pop(_v16.FISHER_CHILD_ID))
    pca_result = dict(arms.pop(_v16.PCA_CHILD_ID))
    if arms:
        raise RuntimeError("V16 coordinate adapter returned unexpected arms")
    fisher_result["arm_id"] = FISHER_PEDAL_ID
    pca_result["arm_id"] = PCA_PEDAL_ID
    return {
        **qualification,
        "arms": {
            FISHER_PEDAL_ID: fisher_result,
            PCA_PEDAL_ID: pca_result,
        },
        "protocol_reused_from": "authoritative_v16_held_runtime_geometry",
    }


def _validate_held_trust_diagnostic(
    value: object,
    *,
    expected_arm_id: str,
    expected_family_id: str,
    expected_provider_artifact_sha256: str,
    expected_parent_artifact_sha256: str,
    expected_held_sequence_sha256s: tuple[str, ...],
) -> dict[str, object]:
    row = _v16._mapping(value, label="held pedal runtime diagnostic")
    diagnostic_sha = _v16._sha256_identifier(
        row.get("held_diagnostic_sha256"),
        label="held pedal diagnostic",
    )
    payload = {key: value for key, value in row.items() if key != "held_diagnostic_sha256"}
    if _v14._sha256(payload, domain=_HELD_DIAGNOSTIC_DOMAIN) != diagnostic_sha:
        raise ValueError("held pedal diagnostic hash mismatch")
    provider_artifact = _v16._sha256_identifier(
        row.get("provider_artifact_sha256"),
        label="held pedal provider artifact",
    )
    parent_artifact = _v16._sha256_identifier(
        row.get("parent_provider_artifact_sha256"),
        label="held pedal parent artifact",
    )
    family = _v14._identifier(row.get("held_family_id"), label="held pedal family")
    held_sequences = _v16._sha256_tuple(
        row.get("held_sequence_sha256s"),
        count=_EXPECTED_HELD_SEQUENCES_PER_FOLD,
        label="held pedal sequences",
    )
    objective = _v14._identifier(
        row.get("coordinate_objective"),
        label="held pedal coordinate objective",
    )
    mode = _v14._identifier(row.get("pedal_mode"), label="held pedal mode")
    runtime_fields_value = row.get("runtime_input_fields")
    runtime_fields = (
        tuple(runtime_fields_value)
        if isinstance(runtime_fields_value, Sequence)
        and not isinstance(runtime_fields_value, (str, bytes))
        else ()
    )
    trust_fraction = _v16._finite(
        row.get("trust_fraction"),
        label="held trust fraction",
    )
    if (
        provider_artifact != expected_provider_artifact_sha256
        or parent_artifact != expected_parent_artifact_sha256
        or family != expected_family_id
        or held_sequences != expected_held_sequence_sha256s
        or objective != _COORDINATE_OBJECTIVES[expected_arm_id]
        or mode != _PEDAL_MODES[expected_arm_id]
        or runtime_fields != _RUNTIME_INPUT_FIELDS
        or row.get("weighting_semantics")
        != "equal_sequences_then_equal_supported_rows"
        or trust_fraction != _TRUST_FRACTION
    ):
        raise ValueError("held pedal runtime ownership or protocol differs")

    receipts_value = row.get("sequence_coordinate_receipts")
    if (
        not isinstance(receipts_value, Sequence)
        or isinstance(receipts_value, (str, bytes))
        or len(receipts_value) != _EXPECTED_HELD_SEQUENCES_PER_FOLD
    ):
        raise ValueError("held pedal replay receipts differ")
    replay_artifacts: list[str] = []
    replay_rows = 0
    replay_effective_rows = 0
    replay_effective_sequences = 0
    replay_sequences: list[str] = []
    for receipt_value in receipts_value:
        receipt = _v16._mapping(receipt_value, label="held pedal replay receipt")
        sequence_sha = _v16._sha256_identifier(
            receipt.get("sequence_sha256"), label="held pedal replay sequence"
        )
        replay_metadata = _v16._mapping(
            receipt.get("runtime_replay"), label="held pedal replay metadata"
        )
        replay_artifact = _v16._sha256_identifier(
            replay_metadata.get("artifact_sha256"),
            label="held pedal replay artifact",
        )
        embedded_artifact = _v16._sha256_identifier(
            receipt.get("runtime_replay_artifact_sha256"),
            label="held pedal receipt replay artifact",
        )
        validated_replay = validate_fisher_xy_pedal_runtime_replay_metadata(
            replay_metadata
        )
        if (
            replay_artifact != embedded_artifact
            or validated_replay["artifact_sha256"] != replay_artifact
            or validated_replay.get("sequence_artifact_sha256") != sequence_sha
            or validated_replay.get("provider_artifact_sha256")
            != provider_artifact
            or validated_replay.get("parent_provider_artifact_sha256")
            != parent_artifact
            or validated_replay.get("trust_fraction") != trust_fraction
        ):
            raise ValueError("held pedal replay artifact or binding differs")
        _v16._sha256_identifier(
            receipt.get("effective_support_mask_sha256"),
            label="held effective support mask",
        )
        replay_row_count = validated_replay.get("row_count")
        if (
            type(replay_row_count) is not int
            or replay_row_count <= 0
            or replay_row_count != receipt.get("row_count")
        ):
            raise ValueError("held pedal replay row count differs")
        effective_row_count = receipt.get("effective_row_count")
        if (
            type(effective_row_count) is not int
            or effective_row_count < 0
            or effective_row_count > replay_row_count
        ):
            raise ValueError("held pedal effective row count differs")
        replay_rows += replay_row_count
        replay_effective_rows += effective_row_count
        replay_effective_sequences += int(effective_row_count > 0)
        replay_sequences.append(sequence_sha)
        replay_artifacts.append(replay_artifact)
    if (
        tuple(sorted(replay_sequences)) != held_sequences
        or replay_rows != row.get("row_count")
    ):
        raise ValueError("held pedal replay aggregate ownership differs")

    distribution = _v16._mapping(
        row.get("pedal_distribution"), label="held pedal distribution"
    )
    if (
        distribution.get("weighting_semantics")
        != "equal_sequences_then_equal_supported_rows"
    ):
        raise ValueError("held pedal distribution weighting differs")
    pedal_min = _v16._finite(
        distribution.get("pedal_min"), label="held pedal minimum"
    )
    pedal_mean = _v16._finite(
        distribution.get("pedal_weighted_mean"), label="held pedal weighted mean"
    )
    pedal_max = _v16._finite(
        distribution.get("pedal_max"), label="held pedal maximum"
    )
    zero_fraction = _v16._finite(
        distribution.get("pedal_zero_weight_fraction"),
        label="held zero-pedal weight fraction",
    )
    unit_fraction = _v16._finite(
        distribution.get("pedal_unit_weight_fraction"),
        label="held unit-pedal weight fraction",
    )
    effective_distribution = _v16._mapping(
        row.get("pedal_effective_distribution"),
        label="held effective pedal distribution",
    )
    if (
        effective_distribution.get("weighting_semantics")
        != (
            "normalized_equal_sequence_row_weight_times_bounded_direction_"
            "energy_on_effective_support"
        )
        or effective_distribution.get("effective_row_count")
        != replay_effective_rows
        or effective_distribution.get("effective_sequence_count")
        != replay_effective_sequences
        or effective_distribution.get("every_sequence_has_effective_rows")
        is not (replay_effective_sequences == _EXPECTED_HELD_SEQUENCES_PER_FOLD)
    ):
        raise ValueError("held effective pedal distribution ownership differs")
    effective_min = _v16._finite(
        effective_distribution.get("pedal_effective_min"),
        label="held effective pedal minimum",
    )
    effective_mean = _v16._finite(
        effective_distribution.get("pedal_effective_weighted_mean"),
        label="held effective pedal weighted mean",
    )
    effective_std = _v16._finite(
        effective_distribution.get("pedal_effective_weighted_std"),
        label="held effective pedal weighted std",
    )
    effective_max = _v16._finite(
        effective_distribution.get("pedal_effective_max"),
        label="held effective pedal maximum",
    )
    runtime_trust = _v16._mapping(
        row.get("runtime_trust"), label="held pedal runtime trust"
    )
    bounded_ratio = _v16._finite(
        runtime_trust.get("max_bounded_direction_to_parent_norm_ratio"),
        label="held bounded-direction ratio",
    )
    emitted_ratio = _v16._finite(
        runtime_trust.get("max_emitted_delta_to_parent_norm_ratio"),
        label="held emitted-delta ratio",
    )
    tolerance = 1.0e-12
    variation_floor = math.sqrt(torch.finfo(torch.float64).eps)
    scalar_protocol = True
    if mode == "unit":
        scalar_protocol = (
            pedal_min == pedal_mean == pedal_max == 1.0
            and effective_min == effective_mean == effective_max == 1.0
            and effective_std == 0.0
        )
    elif mode == "constant_optimal":
        scalar_protocol = math.isclose(
            pedal_min, pedal_max, rel_tol=0.0, abs_tol=1.0e-12
        ) and math.isclose(
            effective_min, effective_max, rel_tol=0.0, abs_tol=1.0e-12
        ) and effective_std <= variation_floor
    conditional_variation = True
    if mode == "conditional":
        conditional_variation = (
            effective_std > variation_floor
            and effective_max - effective_min > variation_floor
        )
    gates = {
        "pedal_lies_in_closed_unit_interval": (
            0.0 <= pedal_min <= pedal_mean <= pedal_max <= 1.0
            and 0.0 <= zero_fraction <= 1.0
            and 0.0 <= unit_fraction <= 1.0
        ),
        "bounded_direction_within_relative_trust_ball": (
            0.0 <= bounded_ratio <= trust_fraction + tolerance
        ),
        "emitted_delta_within_relative_trust_ball": (
            0.0 <= emitted_ratio <= trust_fraction + tolerance
            and runtime_trust.get("pointwise_trust_certificate_passed") is True
        ),
        "declared_pedal_mode_runtime_semantics": scalar_protocol,
        "effective_pedal_lies_in_closed_unit_interval": (
            0.0 <= effective_min <= effective_mean <= effective_max <= 1.0
            and effective_std >= 0.0
        ),
        "every_held_sequence_has_effective_direction_rows": (
            replay_effective_sequences == _EXPECTED_HELD_SEQUENCES_PER_FOLD
        ),
        "conditional_pedal_varies_on_held_effective_rows": conditional_variation,
    }
    return {
        "held_family_id": family,
        "provider_artifact_sha256": provider_artifact,
        "parent_provider_artifact_sha256": parent_artifact,
        "coordinate_objective": objective,
        "pedal_mode": mode,
        "held_sequence_sha256s": held_sequences,
        "runtime_replay_artifact_sha256s": tuple(replay_artifacts),
        "pedal_min": pedal_min,
        "pedal_weighted_mean": pedal_mean,
        "pedal_max": pedal_max,
        "pedal_zero_weight_fraction": zero_fraction,
        "pedal_unit_weight_fraction": unit_fraction,
        "effective_row_count": replay_effective_rows,
        "effective_sequence_count": replay_effective_sequences,
        "pedal_effective_min": effective_min,
        "pedal_effective_weighted_mean": effective_mean,
        "pedal_effective_weighted_std": effective_std,
        "pedal_effective_max": effective_max,
        "max_bounded_direction_to_parent_norm_ratio": bounded_ratio,
        "max_emitted_delta_to_parent_norm_ratio": emitted_ratio,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_trust_gates(
    *,
    arm_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Authenticate fit and held pointwise trust for all four child arms."""

    rows = _validate_arm_rows(arm_rows)
    per_arm: dict[str, object] = {}
    for arm_id in _CHILD_IDS:
        row = rows[arm_id]
        artifacts = _v16._mapping(
            row.get("fold_provider_artifact_sha256s"),
            label=f"{arm_id} fold artifacts",
        )
        parent_artifacts = _v16._mapping(
            row.get("fold_parent_provider_artifact_sha256s"),
            label=f"{arm_id} parent artifacts",
        )
        ownership_rows = _v16._mapping(
            row.get("fold_ownership_receipts"),
            label=f"{arm_id} ownership receipts",
        )
        fit_rows = _v16._mapping(
            row.get("fit_diagnostics"),
            label=f"{arm_id} fit diagnostics",
        )
        held_rows = _v16._mapping(
            row.get("held_runtime_diagnostics"),
            label=f"{arm_id} held diagnostics",
        )
        if (
            len(artifacts) != _EXPECTED_FAMILIES
            or set(artifacts) != set(parent_artifacts)
            or set(artifacts) != set(ownership_rows)
            or set(artifacts) != set(fit_rows)
            or set(artifacts) != set(held_rows)
        ):
            raise ValueError("V17 trust diagnostics must cover all outer folds")
        per_fold: dict[str, object] = {}
        for family in sorted(artifacts):
            selected_family = _v14._identifier(family, label="held family")
            artifact = _v16._sha256_identifier(
                artifacts[family], label="V17 child artifact"
            )
            parent_artifact = _v16._sha256_identifier(
                parent_artifacts[family], label="V17 parent artifact"
            )
            ownership_value = dict(
                _v16._mapping(
                    ownership_rows[family], label="V17 child ownership"
                )
            )
            if ownership_value.pop("pedal_mode", None) != _PEDAL_MODES[arm_id]:
                raise ValueError("V17 ownership pedal mode differs")
            ownership = _v16._fold_ownership_receipt(
                ownership_value,
                expected_held_family_id=selected_family,
                expected_provider_artifact_sha256=artifact,
                expected_parent_provider_artifact_sha256=parent_artifact,
                expected_coordinate_objective=_COORDINATE_OBJECTIVES[arm_id],
            )
            fit = _v16._mapping(fit_rows[family], label="V17 fit diagnostic")
            fit_artifact = _v16._sha256_identifier(
                fit.get("provider_artifact_sha256"),
                label="V17 fit provider artifact",
            )
            fit_mode = _v14._identifier(fit.get("pedal_mode"), label="V17 fit mode")
            fit_trust = _v16._finite(
                fit.get("trust_fraction"), label="V17 fit trust fraction"
            )
            fit_bounded = _v16._finite(
                fit.get("fit_max_bounded_direction_ratio"),
                label="V17 fit bounded ratio",
            )
            fit_emitted = _v16._finite(
                fit.get("fit_max_emitted_delta_ratio"),
                label="V17 fit emitted ratio",
            )
            fit_effective_mean = _v16._finite(
                fit.get("pedal_effective_weighted_mean"),
                label="V17 fit effective pedal mean",
            )
            fit_effective_std = _v16._finite(
                fit.get("pedal_effective_weighted_std"),
                label="V17 fit effective pedal std",
            )
            fit_effective_min = _v16._finite(
                fit.get("pedal_effective_min"),
                label="V17 fit effective pedal minimum",
            )
            fit_effective_max = _v16._finite(
                fit.get("pedal_effective_max"),
                label="V17 fit effective pedal maximum",
            )
            fit_slope_norm = _v16._finite(
                fit.get("pedal_slope_l2_norm"),
                label="V17 fit pedal slope norm",
            )
            fit_rmse_constant = _v16._finite(
                fit.get("weighted_residual_rmse_constant"),
                label="V17 fit constant residual RMSE",
            )
            fit_rmse_after = _v16._finite(
                fit.get("weighted_residual_rmse_after"),
                label="V17 fit selected residual RMSE",
            )
            factor_bundle = _validate_factor_bundle(
                fit.get("shared_factor_bundle"),
                expected_coordinate_objective=_COORDINATE_OBJECTIVES[arm_id],
            )
            if (
                fit_artifact != artifact
                or fit_mode != _PEDAL_MODES[arm_id]
                or fit_trust != _TRUST_FRACTION
            ):
                raise ValueError("V17 fit trust receipt differs")
            held = _validate_held_trust_diagnostic(
                held_rows[family],
                expected_arm_id=arm_id,
                expected_family_id=selected_family,
                expected_provider_artifact_sha256=artifact,
                expected_parent_artifact_sha256=parent_artifact,
                expected_held_sequence_sha256s=ownership["held_sequence_sha256s"],  # type: ignore[arg-type]
            )
            variation_floor = math.sqrt(torch.finfo(torch.float64).eps)
            rmse_tolerance = 64.0 * torch.finfo(torch.float64).eps * max(
                fit_rmse_constant,
                1.0,
            )
            fit_scalar_protocol = True
            fit_conditional_noncollapse = True
            fit_conditional_nonregression = True
            if fit_mode == "unit":
                fit_scalar_protocol = (
                    fit_effective_min
                    == fit_effective_mean
                    == fit_effective_max
                    == 1.0
                    and fit_effective_std == 0.0
                    and fit_slope_norm == 0.0
                )
            elif fit_mode == "constant_optimal":
                fit_scalar_protocol = (
                    math.isclose(
                        fit_effective_min,
                        fit_effective_max,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    and fit_effective_std <= variation_floor
                    and fit_slope_norm == 0.0
                )
            else:
                fit_conditional_noncollapse = (
                    fit_slope_norm > variation_floor
                    and fit_effective_std > variation_floor
                    and fit_effective_max - fit_effective_min > variation_floor
                )
                fit_conditional_nonregression = (
                    fit_rmse_after <= fit_rmse_constant + rmse_tolerance
                )
            gates = {
                "fit_bounded_direction_within_relative_trust_ball": (
                    0.0 <= fit_bounded <= _TRUST_FRACTION + 1.0e-12
                ),
                "fit_emitted_delta_within_relative_trust_ball": (
                    0.0 <= fit_emitted <= _TRUST_FRACTION + 1.0e-12
                ),
                "fit_effective_pedal_lies_in_closed_unit_interval": (
                    0.0
                    <= fit_effective_min
                    <= fit_effective_mean
                    <= fit_effective_max
                    <= 1.0
                    and fit_effective_std >= 0.0
                ),
                "declared_pedal_mode_fit_semantics": fit_scalar_protocol,
                "conditional_pedal_varies_on_fit_effective_rows": (
                    fit_conditional_noncollapse
                ),
                "conditional_fit_not_worse_than_constant": (
                    fit_conditional_nonregression
                ),
                "held_runtime_trust": held["passed"] is True,
            }
            per_fold[selected_family] = {
                "provider_artifact_sha256": artifact,
                "parent_provider_artifact_sha256": parent_artifact,
                "fit_max_bounded_direction_ratio": fit_bounded,
                "fit_max_emitted_delta_ratio": fit_emitted,
                "fit_pedal_slope_l2_norm": fit_slope_norm,
                "fit_pedal_effective_weighted_mean": fit_effective_mean,
                "fit_pedal_effective_weighted_std": fit_effective_std,
                "fit_pedal_effective_min": fit_effective_min,
                "fit_pedal_effective_max": fit_effective_max,
                "fit_weighted_residual_rmse_constant": fit_rmse_constant,
                "fit_weighted_residual_rmse_after": fit_rmse_after,
                "shared_factor_bundle_sha256": factor_bundle["artifact_sha256"],
                "held_runtime": held,
                "gates": gates,
                "passed": all(gates.values()),
            }
        failed = tuple(
            family for family, value in per_fold.items() if value["passed"] is not True  # type: ignore[index]
        )
        per_arm[arm_id] = {
            "pedal_mode": _PEDAL_MODES[arm_id],
            "coordinate_objective": _COORDINATE_OBJECTIVES[arm_id],
            "per_fold": per_fold,
            "failed_fold_ids": failed,
            "failed_fold_count": len(failed),
            "passed": not failed,
        }
    for family in sorted(
        _v16._mapping(
            _v16._mapping(per_arm[FISHER_PEDAL_ID], label="Fisher pedal trust").get(
                "per_fold"
            ),
            label="Fisher pedal trust folds",
        )
    ):
        fisher_bundle_hashes = {
            _v16._mapping(
                _v16._mapping(per_arm[arm_id], label=f"{arm_id} trust").get(
                    "per_fold"
                ),
                label=f"{arm_id} trust folds",
            )[family]["shared_factor_bundle_sha256"]  # type: ignore[index]
            for arm_id in (
                FISHER_UNIT_ID,
                FISHER_CONSTANT_ID,
                FISHER_PEDAL_ID,
            )
        }
        if len(fisher_bundle_hashes) != 1:
            raise ValueError("V17 Fisher controls do not share one factor bundle")
    return {
        "trust_fraction": _TRUST_FRACTION,
        "certificate_scope": (
            "pointwise_modal_correction_amplitude_not_decoded_h4_"
            "jacobian_or_lipschitz"
        ),
        "raw_direction_inside_ball_is_never_amplified": True,
        "arm_count": len(_CHILD_IDS),
        "held_runtime_diagnostic_count": _EXPECTED_HELD_RUNTIME_DIAGNOSTICS,
        "arms": per_arm,
        "fisher_unit_constant_conditional_factor_bundles_identical": True,
        "passed": all(value["passed"] is True for value in per_arm.values()),  # type: ignore[index]
    }


def evaluate_mechanism_gates(
    *,
    parent: Mapping[str, object],
    fisher_unit: Mapping[str, object],
    fisher_constant: Mapping[str, object],
    fisher_pedal: Mapping[str, object],
    pca_pedal: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the frozen V17 parent/control/conditional comparisons."""

    arms = {
        PARENT_ID: parent,
        FISHER_UNIT_ID: fisher_unit,
        FISHER_CONSTANT_ID: fisher_constant,
        FISHER_PEDAL_ID: fisher_pedal,
        PCA_PEDAL_ID: pca_pedal,
    }
    macro = {
        name: _v16._family_macro(row, "ordinary") for name, row in arms.items()
    }
    aggregate = {
        name: _v16._aggregate(row, "ordinary") for name, row in arms.items()
    }
    absolute_nll = {
        name: _v16._finite(
            value.get("absolute_delta_nll_per_token"),
            label=f"{name} macro absolute delta NLL",
        )
        for name, value in macro.items()
    }
    kl = {
        name: _v16._finite(
            value.get("source_to_candidate_kl_per_token"),
            label=f"{name} macro KL",
        )
        for name, value in macro.items()
    }
    top1 = {
        name: _v16._finite(
            value.get("top1_agreement_to_source"),
            label=f"{name} aggregate top1",
        )
        for name, value in aggregate.items()
    }
    family_rows = {name: _v16._family_rows(row) for name, row in arms.items()}
    family_sets = {frozenset(value) for value in family_rows.values()}
    if len(family_sets) != 1 or len(next(iter(family_sets))) != _EXPECTED_FAMILIES:
        raise ValueError("V17 mechanism family membership differs")

    parent_family_improvements: dict[str, float] = {}
    constant_family_improvements: dict[str, float] = {}
    for family, pedal_row in family_rows[FISHER_PEDAL_ID].items():
        pedal_value = _v16._finite(
            pedal_row.get("absolute_delta_nll_per_token"),
            label="Fisher-pedal family absolute delta NLL",
        )
        parent_value = _v16._finite(
            family_rows[PARENT_ID][family].get("absolute_delta_nll_per_token"),
            label="parent family absolute delta NLL",
        )
        constant_value = _v16._finite(
            family_rows[FISHER_CONSTANT_ID][family].get(
                "absolute_delta_nll_per_token"
            ),
            label="constant family absolute delta NLL",
        )
        parent_family_improvements[family] = _relative_improvement(
            parent_value, pedal_value, label="parent family absolute delta NLL"
        )
        constant_family_improvements[family] = _relative_improvement(
            constant_value, pedal_value, label="constant family absolute delta NLL"
        )

    support_core_observations: dict[str, object] = {}
    support_core_passed: dict[str, bool] = {}
    for ledger in ("complete_h4_support", "graph_core"):
        parent_ledger = _v16._aggregate(parent, ledger)
        pedal_ledger = _v16._aggregate(fisher_pedal, ledger)
        parent_delta = abs(
            _v16._finite(
                parent_ledger.get("delta_nll_per_token"),
                label=f"parent {ledger} delta NLL",
            )
        )
        pedal_delta = abs(
            _v16._finite(
                pedal_ledger.get("delta_nll_per_token"),
                label=f"pedal {ledger} delta NLL",
            )
        )
        parent_kl = _v16._finite(
            parent_ledger.get("source_to_candidate_kl_per_token"),
            label=f"parent {ledger} KL",
        )
        pedal_kl = _v16._finite(
            pedal_ledger.get("source_to_candidate_kl_per_token"),
            label=f"pedal {ledger} KL",
        )
        parent_top1 = _v16._finite(
            parent_ledger.get("top1_agreement_to_source"),
            label=f"parent {ledger} top1",
        )
        pedal_top1 = _v16._finite(
            pedal_ledger.get("top1_agreement_to_source"),
            label=f"pedal {ledger} top1",
        )
        passed = (
            pedal_delta <= parent_delta
            and pedal_kl <= parent_kl
            and pedal_top1 >= parent_top1
        )
        support_core_passed[ledger] = passed
        support_core_observations[ledger] = {
            "parent_absolute_delta_nll_per_token": parent_delta,
            "fisher_pedal_absolute_delta_nll_per_token": pedal_delta,
            "parent_kl_per_token": parent_kl,
            "fisher_pedal_kl_per_token": pedal_kl,
            "parent_top1": parent_top1,
            "fisher_pedal_top1": pedal_top1,
            "passed": passed,
        }

    parent_abs_improvement = _relative_improvement(
        absolute_nll[PARENT_ID],
        absolute_nll[FISHER_PEDAL_ID],
        label="parent macro absolute delta NLL",
    )
    parent_kl_improvement = _relative_improvement(
        kl[PARENT_ID], kl[FISHER_PEDAL_ID], label="parent macro KL"
    )
    constant_abs_improvement = _relative_improvement(
        absolute_nll[FISHER_CONSTANT_ID],
        absolute_nll[FISHER_PEDAL_ID],
        label="constant macro absolute delta NLL",
    )
    parent_win_count = sum(value > 0.0 for value in parent_family_improvements.values())
    constant_win_count = sum(
        value > 0.0 for value in constant_family_improvements.values()
    )
    gates = {
        "versus_parent_absolute_delta_nll_materiality": parent_abs_improvement
        >= 0.05,
        "versus_parent_kl_materiality": parent_kl_improvement >= 0.05,
        "versus_parent_top1_materiality": (
            top1[FISHER_PEDAL_ID] - top1[PARENT_ID]
        )
        >= 0.02,
        "versus_parent_family_win_count": parent_win_count >= 6,
        "versus_parent_worst_family_regression_floor": min(
            parent_family_improvements.values()
        )
        >= -0.02,
        "versus_constant_absolute_delta_nll_materiality": (
            constant_abs_improvement >= 0.01
        ),
        "versus_constant_kl_no_regression": (
            kl[FISHER_PEDAL_ID] <= kl[FISHER_CONSTANT_ID]
        ),
        "versus_constant_top1_no_regression": (
            top1[FISHER_PEDAL_ID] >= top1[FISHER_CONSTANT_ID]
        ),
        "versus_constant_family_win_count": constant_win_count >= 5,
        "fisher_strictly_beats_pca_absolute_delta_nll": (
            absolute_nll[FISHER_PEDAL_ID] < absolute_nll[PCA_PEDAL_ID]
        ),
        "fisher_does_not_regress_pca_kl": (
            kl[FISHER_PEDAL_ID] <= kl[PCA_PEDAL_ID]
        ),
        "fisher_does_not_regress_pca_top1": (
            top1[FISHER_PEDAL_ID] >= top1[PCA_PEDAL_ID]
        ),
        "complete_h4_support_no_regression": support_core_passed[
            "complete_h4_support"
        ],
        "graph_core_no_regression": support_core_passed["graph_core"],
    }
    return {
        "thresholds": dict(_MECHANISM_THRESHOLDS),
        "observations": {
            "versus_parent_macro_absolute_delta_nll_relative_improvement": (
                parent_abs_improvement
            ),
            "versus_parent_macro_kl_relative_improvement": parent_kl_improvement,
            "versus_parent_aggregate_top1_gain": (
                top1[FISHER_PEDAL_ID] - top1[PARENT_ID]
            ),
            "versus_parent_family_absolute_delta_nll_win_count": parent_win_count,
            "versus_parent_worst_family_relative_improvement": min(
                parent_family_improvements.values()
            ),
            "versus_constant_macro_absolute_delta_nll_relative_improvement": (
                constant_abs_improvement
            ),
            "versus_constant_family_absolute_delta_nll_win_count": (
                constant_win_count
            ),
            "ordinary_family_macro_absolute_delta_nll": absolute_nll,
            "ordinary_family_macro_kl": kl,
            "ordinary_aggregate_top1": top1,
            "required_ledger_no_regression": support_core_observations,
            "unit_pedal_control_is_diagnostic_only": True,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _validate_arm_rows(
    arm_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if not isinstance(arm_rows, Mapping) or set(arm_rows) != set(_ARM_IDS):
        raise ValueError("V17 report requires exactly five fixed arms")
    rows = {
        name: _v16._mapping(arm_rows[name], label=f"{name} arm")
        for name in _ARM_IDS
    }
    for name, row in rows.items():
        if (
            row.get("arm_id") != name
            or row.get("coordinate_objective") != _COORDINATE_OBJECTIVES[name]
        ):
            raise ValueError("V17 arm identity differs")
        if name != PARENT_ID and row.get("pedal_mode") != _PEDAL_MODES[name]:
            raise ValueError("V17 child pedal mode differs")
        fidelity = _v16._mapping(row.get("fidelity"), label=f"{name} fidelity")
        if set(fidelity) != set(_v14._ALL_LEDGERS):
            raise ValueError("V17 arm fidelity ledgers differ")
    resources = [
        _v16._mapping(rows[name].get("serving_resources"), label=f"{name} resources")
        for name in _CHILD_IDS
    ]
    if any(value != resources[0] for value in resources[1:]):
        raise ValueError("V17 child control resources differ")
    return rows


def _validate_cross_arm_fold_ownership(
    rows: Mapping[str, Mapping[str, object]],
    held_families: set[str],
) -> dict[str, object]:
    parent_artifacts = _v16._mapping(
        rows[PARENT_ID].get("fold_provider_artifact_sha256s"),
        label="parent fold provider artifacts",
    )
    parent_receipts = _v16._mapping(
        rows[PARENT_ID].get("fold_ownership_receipts"),
        label="parent fold ownership receipts",
    )
    if set(parent_artifacts) != held_families or set(parent_receipts) != held_families:
        raise ValueError("V17 parent ownership does not match outer folds")
    validated: dict[str, object] = {}
    for family in sorted(held_families):
        parent_artifact = _v16._sha256_identifier(
            parent_artifacts[family], label="parent fold provider artifact"
        )
        parent_receipt = _v16._fold_ownership_receipt(
            parent_receipts[family],
            expected_held_family_id=family,
            expected_provider_artifact_sha256=parent_artifact,
            expected_parent_provider_artifact_sha256=None,
            expected_coordinate_objective=None,
        )
        child_receipts: dict[str, object] = {}
        for child_id in _CHILD_IDS:
            receipts = _v16._mapping(
                rows[child_id].get("fold_ownership_receipts"),
                label=f"{child_id} ownership receipts",
            )
            raw = dict(
                _v16._mapping(receipts.get(family), label=f"{child_id} ownership")
            )
            mode = raw.pop("pedal_mode", None)
            if mode != _PEDAL_MODES[child_id]:
                raise ValueError("V17 ownership pedal mode differs")
            child_artifacts = _v16._mapping(
                rows[child_id].get("fold_provider_artifact_sha256s"),
                label=f"{child_id} provider artifacts",
            )
            child_artifact = _v16._sha256_identifier(
                child_artifacts.get(family), label=f"{child_id} provider artifact"
            )
            parsed = _v16._fold_ownership_receipt(
                raw,
                expected_held_family_id=family,
                expected_provider_artifact_sha256=child_artifact,
                expected_parent_provider_artifact_sha256=parent_artifact,
                expected_coordinate_objective=_COORDINATE_OBJECTIVES[child_id],
            )
            if (
                parsed["fit_family_ids"] != parent_receipt["fit_family_ids"]
                or parsed["fit_sequence_sha256s"]
                != parent_receipt["fit_sequence_sha256s"]
                or parsed["held_sequence_sha256s"]
                != parent_receipt["held_sequence_sha256s"]
            ):
                raise ValueError("V17 cross-arm fold ownership differs")
            child_receipts[child_id] = {**parsed, "pedal_mode": mode}
        validated[family] = {
            "parent": parent_receipt,
            "children": child_receipts,
        }
    return validated


def _absolute_passed(fisher_pedal: Mapping[str, object]) -> bool:
    return _v16._absolute_passed(fisher_pedal)


def _full_provider_fit_qualification(
    provider: AutonomousCompleteH4FisherXYPedalProvider,
) -> dict[str, object]:
    diagnostic = _authenticated_fit_diagnostics(provider)
    geometry = _v16._coordinate_fold_diagnostics(
        diagnostic,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_coordinate_objective="reverse_vjp_fisher",
    )
    fit_bounded = _v16._finite(
        diagnostic["fit_max_bounded_direction_ratio"],
        label="full fit bounded direction ratio",
    )
    fit_emitted = _v16._finite(
        diagnostic["fit_max_emitted_delta_ratio"],
        label="full fit emitted delta ratio",
    )
    slope_norm = float(torch.linalg.vector_norm(provider.pedal_weight))
    pedal_mean = _v16._finite(
        provider.pedal_effective_weighted_mean,
        label="full fit effective pedal weighted mean",
    )
    pedal_std = _v16._finite(
        provider.pedal_effective_weighted_std,
        label="full fit effective pedal weighted std",
    )
    pedal_min = _v16._finite(
        provider.pedal_effective_min,
        label="full fit effective pedal minimum",
    )
    pedal_max = _v16._finite(
        provider.pedal_effective_max,
        label="full fit effective pedal maximum",
    )
    rmse_before = _v16._finite(
        provider.weighted_residual_rmse_before,
        label="full fit residual RMSE before",
    )
    rmse_constant = _v16._finite(
        provider.weighted_residual_rmse_constant,
        label="full fit constant residual RMSE",
    )
    rmse_after = _v16._finite(
        provider.weighted_residual_rmse_after,
        label="full fit conditional residual RMSE",
    )
    variation_floor = math.sqrt(torch.finfo(torch.float64).eps)
    rmse_tolerance = 64.0 * torch.finfo(torch.float64).eps * max(
        rmse_before, rmse_constant, 1.0
    )
    trust_gates = {
        "fit_bounded_direction_within_relative_trust_ball": (
            0.0 <= fit_bounded <= _TRUST_FRACTION + 1.0e-12
        ),
        "fit_emitted_delta_within_relative_trust_ball": (
            0.0 <= fit_emitted <= _TRUST_FRACTION + 1.0e-12
        ),
        "conditional_pedal_mode": provider.pedal_mode == "conditional",
        "fixed_trust_fraction": provider.trust_fraction == _TRUST_FRACTION,
        "conditional_pedal_has_nonzero_slope": slope_norm > variation_floor,
        "effective_pedal_lies_in_closed_unit_interval": (
            0.0 <= pedal_min <= pedal_mean <= pedal_max <= 1.0
            and pedal_std >= 0.0
        ),
        "conditional_pedal_varies_on_fit_effective_rows": (
            pedal_std > variation_floor
            and pedal_max - pedal_min > variation_floor
        ),
        "conditional_fit_not_worse_than_constant": (
            rmse_after <= rmse_constant + rmse_tolerance
        ),
        "conditional_fit_not_worse_than_parent": (
            rmse_after <= rmse_before + rmse_tolerance
        ),
    }
    return {
        "provider_artifact_sha256": provider.artifact_sha256,
        "coordinate_objective": provider.coordinate_objective,
        "pedal_mode": provider.pedal_mode,
        "qualification_scope": "fresh_full_fit_output_diagnostic",
        "provider_metadata_alone_is_candidate_certificate": False,
        "coordinate_geometry": geometry,
        "trust": {
            "trust_fraction": provider.trust_fraction,
            "fit_max_bounded_direction_ratio": fit_bounded,
            "fit_max_emitted_delta_ratio": fit_emitted,
            "pedal_slope_l2_norm": slope_norm,
            "pedal_effective_weighted_mean": pedal_mean,
            "pedal_effective_weighted_std": pedal_std,
            "pedal_effective_min": pedal_min,
            "pedal_effective_max": pedal_max,
            "weighted_residual_rmse_before": rmse_before,
            "weighted_residual_rmse_constant": rmse_constant,
            "weighted_residual_rmse_after": rmse_after,
            "variation_floor": variation_floor,
            "gates": trust_gates,
            "passed": all(trust_gates.values()),
        },
        "passed": geometry["passed"] is True and all(trust_gates.values()),
    }


def _validate_full_provider_fit_qualification(
    value: object,
    *,
    expected_artifact_sha256: str,
) -> dict[str, object]:
    row = _v16._mapping(value, label="full V17 provider qualification")
    artifact = _v16._sha256_identifier(
        row.get("provider_artifact_sha256"),
        label="full V17 provider artifact",
    )
    if artifact != expected_artifact_sha256:
        raise ValueError("full V17 qualification artifact differs")
    if (
        row.get("coordinate_objective") != "reverse_vjp_fisher"
        or row.get("pedal_mode") != "conditional"
        or row.get("qualification_scope")
        != "fresh_full_fit_output_diagnostic"
        or row.get("provider_metadata_alone_is_candidate_certificate") is not False
    ):
        raise ValueError("full V17 provider protocol differs")
    geometry = _v16._coordinate_fold_diagnostics(
        row.get("coordinate_geometry"),
        expected_artifact_sha256=artifact,
        expected_coordinate_objective="reverse_vjp_fisher",
    )
    trust = _v16._mapping(row.get("trust"), label="full V17 trust")
    trust_fraction = _v16._finite(
        trust.get("trust_fraction"), label="full V17 trust fraction"
    )
    bounded = _v16._finite(
        trust.get("fit_max_bounded_direction_ratio"),
        label="full V17 bounded ratio",
    )
    emitted = _v16._finite(
        trust.get("fit_max_emitted_delta_ratio"),
        label="full V17 emitted ratio",
    )
    slope_norm = _v16._finite(
        trust.get("pedal_slope_l2_norm"), label="full V17 pedal slope norm"
    )
    pedal_mean = _v16._finite(
        trust.get("pedal_effective_weighted_mean"),
        label="full V17 effective pedal weighted mean",
    )
    pedal_std = _v16._finite(
        trust.get("pedal_effective_weighted_std"),
        label="full V17 effective pedal weighted std",
    )
    pedal_min = _v16._finite(
        trust.get("pedal_effective_min"),
        label="full V17 effective pedal minimum",
    )
    pedal_max = _v16._finite(
        trust.get("pedal_effective_max"),
        label="full V17 effective pedal maximum",
    )
    rmse_before = _v16._finite(
        trust.get("weighted_residual_rmse_before"),
        label="full V17 residual RMSE before",
    )
    rmse_constant = _v16._finite(
        trust.get("weighted_residual_rmse_constant"),
        label="full V17 constant residual RMSE",
    )
    rmse_after = _v16._finite(
        trust.get("weighted_residual_rmse_after"),
        label="full V17 conditional residual RMSE",
    )
    variation_floor = _v16._finite(
        trust.get("variation_floor"), label="full V17 variation floor"
    )
    expected_variation_floor = math.sqrt(torch.finfo(torch.float64).eps)
    rmse_tolerance = 64.0 * torch.finfo(torch.float64).eps * max(
        rmse_before, rmse_constant, 1.0
    )
    gates = _v16._mapping(trust.get("gates"), label="full V17 trust gates")
    recomputed = {
        "fit_bounded_direction_within_relative_trust_ball": (
            0.0 <= bounded <= _TRUST_FRACTION + 1.0e-12
        ),
        "fit_emitted_delta_within_relative_trust_ball": (
            0.0 <= emitted <= _TRUST_FRACTION + 1.0e-12
        ),
        "conditional_pedal_mode": True,
        "fixed_trust_fraction": trust_fraction == _TRUST_FRACTION,
        "conditional_pedal_has_nonzero_slope": slope_norm > variation_floor,
        "effective_pedal_lies_in_closed_unit_interval": (
            0.0 <= pedal_min <= pedal_mean <= pedal_max <= 1.0
            and pedal_std >= 0.0
        ),
        "conditional_pedal_varies_on_fit_effective_rows": (
            pedal_std > variation_floor
            and pedal_max - pedal_min > variation_floor
        ),
        "conditional_fit_not_worse_than_constant": (
            rmse_after <= rmse_constant + rmse_tolerance
        ),
        "conditional_fit_not_worse_than_parent": (
            rmse_after <= rmse_before + rmse_tolerance
        ),
    }
    passed = geometry["passed"] is True and all(recomputed.values())
    if (
        variation_floor != expected_variation_floor
        or dict(gates) != recomputed
        or trust.get("passed") is not all(recomputed.values())
    ):
        raise ValueError("full V17 trust qualification differs")
    if row.get("passed") is not passed:
        raise ValueError("full V17 provider qualification result differs")
    return {
        "provider_artifact_sha256": artifact,
        "coordinate_objective": "reverse_vjp_fisher",
        "pedal_mode": "conditional",
        "qualification_scope": "fresh_full_fit_output_diagnostic",
        "provider_metadata_alone_is_candidate_certificate": False,
        "coordinate_geometry": geometry,
        "trust": {
            "trust_fraction": trust_fraction,
            "fit_max_bounded_direction_ratio": bounded,
            "fit_max_emitted_delta_ratio": emitted,
            "pedal_slope_l2_norm": slope_norm,
            "pedal_effective_weighted_mean": pedal_mean,
            "pedal_effective_weighted_std": pedal_std,
            "pedal_effective_min": pedal_min,
            "pedal_effective_max": pedal_max,
            "weighted_residual_rmse_before": rmse_before,
            "weighted_residual_rmse_constant": rmse_constant,
            "weighted_residual_rmse_after": rmse_after,
            "variation_floor": variation_floor,
            "gates": recomputed,
            "passed": all(recomputed.values()),
        },
        "passed": passed,
    }


def build_fisher_pedal_development_report(
    *,
    artifact_path: Path | str,
    panel: Mapping[str, object],
    bridge_binding_sha256: str,
    folds: Sequence[Mapping[str, object]],
    prerequisites: Mapping[str, object],
    fit_collection: Mapping[str, object],
    base_fidelity: Mapping[str, object],
    arm_rows: Mapping[str, Mapping[str, object]],
    full_refit_qualification: Mapping[str, object] | None,
    candidate: Mapping[str, object] | None,
    integrity: Mapping[str, object],
) -> dict[str, object]:
    """Build one deterministic scalar-only V18 report."""

    rows = _validate_arm_rows(arm_rows)
    if len(folds) != _EXPECTED_FAMILIES:
        raise ValueError("V18 requires exactly eight outer folds")
    held_families = {
        _v14._identifier(
            _v16._mapping(fold, label="outer fold").get("held_family_id"),
            label="held family",
        )
        for fold in folds
    }
    if len(held_families) != _EXPECTED_FAMILIES:
        raise ValueError("V18 outer folds must hold eight unique families")
    ownership = _validate_cross_arm_fold_ownership(rows, held_families)
    coordinate_geometry = evaluate_coordinate_geometry_gates(
        fisher_child=rows[FISHER_PEDAL_ID],
        pca_child=rows[PCA_PEDAL_ID],
    )
    trust = evaluate_trust_gates(arm_rows=rows)
    mechanism = evaluate_mechanism_gates(
        parent=rows[PARENT_ID],
        fisher_unit=rows[FISHER_UNIT_ID],
        fisher_constant=rows[FISHER_CONSTANT_ID],
        fisher_pedal=rows[FISHER_PEDAL_ID],
        pca_pedal=rows[PCA_PEDAL_ID],
    )
    absolute = _absolute_passed(rows[FISHER_PEDAL_ID])
    mechanism_supported = (
        coordinate_geometry["passed"] is True
        and trust["passed"] is True
        and mechanism["passed"] is True
    )
    outer_candidate_eligible = mechanism_supported and absolute

    validated_full: dict[str, object] | None = None
    if full_refit_qualification is not None:
        artifact = _v16._sha256_identifier(
            full_refit_qualification.get("provider_artifact_sha256"),
            label="full V17 provider artifact",
        )
        validated_full = _validate_full_provider_fit_qualification(
            full_refit_qualification,
            expected_artifact_sha256=artifact,
        )
    if outer_candidate_eligible != (full_refit_qualification is not None):
        raise ValueError("V17 full refit does not match outer qualification")
    full_refit_passed = validated_full is not None and validated_full["passed"] is True
    ready = outer_candidate_eligible and full_refit_passed
    if (candidate is not None) != ready:
        raise ValueError("V17 candidate does not match complete readiness")
    if candidate is not None:
        candidate_artifact = _v16._sha256_identifier(
            candidate.get("provider_artifact_sha256"),
            label="V17 candidate provider artifact",
        )
        if (
            candidate.get("arm_id") != FISHER_PEDAL_ID
            or validated_full is None
            or candidate_artifact != validated_full["provider_artifact_sha256"]
            or candidate.get("full_provider_fit_qualification")
            != full_refit_qualification
        ):
            raise ValueError("V17 candidate differs from selected full provider")

    classification = (
        "fisher_pedal_oof_candidate_ready_for_fresh_protocol"
        if ready
        else (
            "fisher_pedal_coordinate_geometry_insufficient"
            if coordinate_geometry["passed"] is not True
            else (
                "fisher_pedal_pointwise_trust_insufficient"
                if trust["passed"] is not True
                else (
                    "fisher_pedal_mechanism_support_insufficient"
                    if mechanism["passed"] is not True
                    else (
                        "fisher_pedal_mechanism_supported_absolute_fidelity_insufficient"
                        if not absolute
                        else "fisher_pedal_full_refit_qualification_insufficient"
                    )
                )
            )
        )
    )
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": 18,
        "scientific_status": (
            "opened_calibration_a_fixed_fisher_pedal_outer_lofo_"
            "v18_realized_mass_development"
        ),
        "artifact": {"path": Path(artifact_path).as_posix()},
        "panel": dict(panel),
        "prerequisites": dict(prerequisites),
        "bridge_binding_sha256": _v14._identifier(
            bridge_binding_sha256, label="bridge binding"
        ),
        "fixed_protocol": {
            "recipe_grid": False,
            "corrected_rung": "v18_realized_mass_held_pedal_aggregation",
            "write_once_predecessor_report_path": _V17_OUTPUT.as_posix(),
            "write_once_predecessor_preserved": True,
            "v17_invalid_aggregation_receipt": {
                "report_sha256": _V17_LOGICAL_SHA256,
                "file_sha256": _V17_FILE_SHA256,
                "classification": _V17_CLASSIFICATION,
                "failure": (
                    "held_unit_and_constant_pedal_summaries_used_unnormalized_"
                    "floating_weight_sums_and_failed_exact_runtime_semantics"
                ),
            },
            "v18_correction": (
                "exact_control_stats_from_serving_scalar_and_conditional_"
                "moments_fractions_divided_by_realized_float64_mass"
            ),
            "parent": {
                **PARENT_RECIPE.metadata(),
                "recipe_sha256": PARENT_RECIPE.artifact_sha256,
            },
            "conditional_rank": _CONDITIONAL_RANK,
            "coordinate_count": FISHER_XY_COORDINATE_COUNT,
            "router_ridge": _ROUTER_RIDGE,
            "direction_ridge": _DIRECTION_RIDGE,
            "pedal_ridge": _PEDAL_RIDGE,
            "trust_fraction": _TRUST_FRACTION,
            "effective_direction_relative_energy_floor": (
                FISHER_XY_PEDAL_RELATIVE_ENERGY_FLOOR
            ),
            "effective_direction_absolute_energy_floor": (
                FISHER_XY_PEDAL_ABSOLUTE_ENERGY_FLOOR
            ),
            "conditional_variation_scope": (
                "fit_and_held_rows_above_effective_direction_energy_floor"
            ),
            "held_pedal_summary_semantics": (
                "controls_from_exact_serving_scalar_conditional_moments_and_"
                "fractions_divided_by_realized_float64_mass"
            ),
            "provider_metadata_alone_is_qualification_certificate": False,
            "qualification_requires": (
                "fresh_fit_output_plus_outer_held_energy_weighted_runtime_"
                "variation_plus_full_suffix_behavior"
            ),
            "trust_or_budget_ladder": False,
            "unit_pedal": 1.0,
            "constant_control": "fit_optimal_clamped_analytic_scalar",
            "conditional_pedal_features": "c1_c2_c1_times_c2_plus_bias",
            "pedal_range": "closed_interval_0_1",
            "direction_fit_target": "per_row_trust_ball_clipped_residual",
            "runtime_direction_rule": (
                "clip_only_when_direction_exceeds_beta_times_parent_norm_"
                "never_amplify"
            ),
        },
        "outer_lofo": {
            "folds": [dict(value) for value in folds],
            "ownership_receipts": ownership,
            "every_provider_fit_inside_training_fold": True,
            "held_family_excluded_from_parent_axes_direction_pedal_and_scales": True,
            "training_family_count_per_fold": _EXPECTED_FAMILIES - 1,
        },
        "execution_scope": {
            "semantics": (
                "full_vocabulary_full_suffix_behavioral_shadow_from_one_"
                "complete_h4_correction_boundary"
            ),
            "replacement_boundary": "layer.4.output",
            "replacement_boundary_count": 1,
            "full_vocabulary_logits_evaluated": True,
            "untouched_downstream_gemma_layers_final_norm_and_lm_head_executed": True,
            "whole_model_compiled": False,
            "layer_4_computation_deleted": False,
            "source_model_parameters_retained": True,
        },
        "fit_collection": dict(fit_collection),
        "base_fidelity": dict(base_fidelity),
        "arms": {name: dict(rows[name]) for name in _ARM_IDS},
        "coordinate_geometry_qualification": coordinate_geometry,
        "pointwise_trust_qualification": trust,
        "mechanism_retention": mechanism,
        "mechanism_support": {
            "coordinate_geometry_passed": coordinate_geometry["passed"],
            "pointwise_trust_passed": trust["passed"],
            "mechanism_comparison_passed": mechanism["passed"],
            "passed": mechanism_supported,
        },
        "absolute_readiness": {
            "required_ledgers": _v14._REQUIRED_LEDGERS,
            "gates": dict(ESTABLISHED_SHADOW_FIDELITY_GATES.metadata()),
            "passed": absolute,
        },
        "full_refit_qualification": validated_full,
        "candidate_readiness": {
            "outer_candidate_eligible": outer_candidate_eligible,
            "full_refit_passed": full_refit_passed,
            "selected_arm_id": FISHER_PEDAL_ID if ready else None,
            "passed": ready,
        },
        "candidate": None if candidate is None else dict(candidate),
        "integrity": dict(integrity),
        "passed": ready,
        "classification": classification,
        "success_authorizes": (
            "freeze_fisher_pedal_candidate_for_new_candidate_bound_protocol"
            if ready
            else None
        ),
        "fresh_guard_authorized": False,
        "calibration_b_authorized": False,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
    }
    _v14._scalar_report(report)
    return report


def _publish(
    report: dict[str, object],
    *,
    output: Path,
    provider: AutonomousCompleteH4FisherXYPedalProvider | None,
    provider_output: Path,
) -> dict[str, object]:
    if _same_destination(output, _V17_OUTPUT) or _same_destination(
        provider_output,
        _V17_OUTPUT.with_suffix(".provider.pt"),
    ):
        raise ValueError("V18 publication must preserve the write-once V17 rung")
    candidate = report.get("candidate")
    if (provider is None) != (candidate is None):
        raise ValueError("V18 published provider must match the selected candidate")
    destinations = (output,) if provider is None else (output, provider_output)
    reservation = _v14._reserve_outputs(destinations)
    report_stage: Path | None = None
    provider_stage: Path | None = None
    try:
        if provider is not None:
            if not isinstance(candidate, Mapping):
                raise TypeError("V18 selected candidate must be a mapping")
            _validate_child(
                provider,
                coordinate_objective="reverse_vjp_fisher",
                pedal_mode="conditional",
                expected_parent_artifact_sha256=(
                    provider.parent_provider.artifact_sha256
                ),
                expected_fit_family_count=_EXPECTED_FAMILIES,
            )
            qualification = _full_provider_fit_qualification(provider)
            if qualification["passed"] is not True:
                raise RuntimeError("selected V18 full provider qualification is insufficient")
            if (
                candidate.get("provider_artifact_sha256")
                != provider.artifact_sha256
                or candidate.get("parent_provider_artifact_sha256")
                != provider.parent_provider.artifact_sha256
                or candidate.get("fit_family_ids") != provider.fit_family_ids
                or candidate.get("fit_sequence_sha256s")
                != provider.fit_sequence_sha256s
                or candidate.get("full_provider_fit_qualification")
                != qualification
            ):
                raise ValueError("V18 candidate and provider differ")
            provider_stage = _v14._stage_torch(
                autonomous_complete_h4_fisher_xy_pedal_provider_state_dict(provider),
                provider_output,
            )
            provider_file_sha256 = _v14._file_sha256(provider_stage)
            restored = load_autonomous_complete_h4_fisher_xy_pedal_provider(
                provider_stage,
                expected_artifact_sha256=provider.artifact_sha256,
                expected_file_sha256=provider_file_sha256,
                expected_bridge_binding_sha256=provider.bridge_binding_sha256,
            )
            if restored.metadata() != provider.metadata():
                raise RuntimeError("staged V18 provider roundtrip drifted")
            report["candidate"] = {
                **dict(candidate),
                "provider_tensor_artifact": {
                    "path": provider_output.as_posix(),
                    "file_sha256": provider_file_sha256,
                    "file_bytes": provider_stage.stat().st_size,
                    "provider_artifact_sha256": provider.artifact_sha256,
                    "bridge_binding_sha256": provider.bridge_binding_sha256,
                    "write_once": True,
                    "file_mode": "0600",
                    "contains_runtime_provider_tensors_only": True,
                    "contains_native_h4_logits_targets_gradients_or_coordinate_axes": False,
                },
            }
        _v14._scalar_report(report)
        report["report_sha256"] = _v14._sha256(report, domain=_REPORT_DOMAIN)
        report_stage = _v14._stage_json(report, output)
        staged = (
            (report_stage,)
            if provider_stage is None
            else (report_stage, provider_stage)
        )
        reservation.publish(staged)
        if provider is not None:
            receipt = report["candidate"]["provider_tensor_artifact"]  # type: ignore[index]
            load_autonomous_complete_h4_fisher_xy_pedal_provider(
                provider_output,
                expected_artifact_sha256=provider.artifact_sha256,
                expected_file_sha256=str(receipt["file_sha256"]),  # type: ignore[index]
                expected_bridge_binding_sha256=provider.bridge_binding_sha256,
            )
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": _v14._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        for stage in (report_stage, provider_stage):
            if stage is not None:
                stage.unlink(missing_ok=True)


def _fit_parent(
    sequences: Sequence[object],
    *,
    bridge_binding_sha256: str,
) -> AutonomousCompleteH4ResidualProvider:
    return _v14._fit_provider(
        sequences,  # type: ignore[arg-type]
        PARENT_RECIPE,
        bridge_binding_sha256=bridge_binding_sha256,
    )


def _fit_child(
    sequences: Sequence[object],
    *,
    parent: AutonomousCompleteH4ResidualProvider,
    coordinate_objective: str,
    pedal_mode: str,
) -> AutonomousCompleteH4FisherXYPedalProvider:
    return fit_autonomous_complete_h4_fisher_xy_pedal(
        sequences=sequences,  # type: ignore[arg-type]
        parent_provider=parent,
        conditional_rank=_CONDITIONAL_RANK,
        coordinate_objective=coordinate_objective,
        pedal_mode=pedal_mode,
        router_ridge=_ROUTER_RIDGE,
        direction_ridge=_DIRECTION_RIDGE,
        pedal_ridge=_PEDAL_RIDGE,
        trust_fraction=_TRUST_FRACTION,
        vjp_weight_floor=0.5,
        vjp_weight_ceiling=2.0,
    )


def run_gemma3_l3_l4_complete_h4_fisher_pedal_development(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    provider_output: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the corrected A16 V18 parent/control/Fisher/PCA outer-LOFO screen."""

    destination = _validate_output(output)
    provider_destination = _validate_provider_output(
        destination.with_suffix(".provider.pt")
        if provider_output is None
        else provider_output
    )
    if _same_destination(destination, _V17_OUTPUT) or _same_destination(
        provider_destination,
        _V17_OUTPUT.with_suffix(".provider.pt"),
    ):
        raise ValueError("V18 runner must preserve the write-once V17 rung")
    if provider_destination == destination:
        raise ValueError("V18 report and provider outputs must differ")
    if destination.exists():
        raise FileExistsError("refusing to overwrite V18 report")
    if provider_destination.exists():
        raise FileExistsError("refusing to overwrite V18 provider")

    # Bind the exact corrected V16 result before any live model is loaded.
    prerequisites = _validate_prerequisites()
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        records = _v14._collect_fit_records(context)
        bridge_binding = _v14._identifier(
            context.bridge.bridge_binding_sha256,
            label="bridge binding",
        )
        families = tuple(sorted({record.sequence.family_id for record in records}))
        folds = _v14.build_outer_lofo_splits(families)
        parents: dict[str, AutonomousCompleteH4ResidualProvider] = {}
        children: dict[
            str, dict[str, AutonomousCompleteH4FisherXYPedalProvider]
        ] = {arm_id: {} for arm_id in _CHILD_IDS}
        held_sequences_by_family: dict[
            str, tuple[AutonomousCompleteH4TrainingSequence, ...]
        ] = {}
        for fold in folds:
            held = str(fold["held_family_id"])
            held_sequences = tuple(
                record.sequence
                for record in records
                if record.sequence.family_id == held
            )
            training = tuple(
                record.sequence
                for record in records
                if record.sequence.family_id != held
            )
            if (
                len(training) != _EXPECTED_TRAINING_PROMPTS_PER_FOLD
                or len(held_sequences) != _EXPECTED_HELD_SEQUENCES_PER_FOLD
                or held in {value.family_id for value in training}
            ):
                raise RuntimeError("V17 outer-LOFO training ownership differs")
            parent = _fit_parent(training, bridge_binding_sha256=bridge_binding)
            _validate_parent(
                parent,
                expected_fit_family_count=_EXPECTED_FAMILIES - 1,
            )
            fold_children = {
                arm_id: _fit_child(
                    training,
                    parent=parent,
                    coordinate_objective=_COORDINATE_OBJECTIVES[arm_id],
                    pedal_mode=_PEDAL_MODES[arm_id],
                )
                for arm_id in _CHILD_IDS
            }
            for arm_id, child in fold_children.items():
                _validate_child(
                    child,
                    coordinate_objective=_COORDINATE_OBJECTIVES[arm_id],
                    pedal_mode=_PEDAL_MODES[arm_id],
                    expected_parent_artifact_sha256=parent.artifact_sha256,
                    expected_fit_family_count=_EXPECTED_FAMILIES - 1,
                )
                children[arm_id][held] = child
            _validate_parameter_matched_children(tuple(fold_children.values()))
            parents[held] = parent
            held_sequences_by_family[held] = held_sequences

        manifests = _v14._ledger_manifests(records)
        ledger_coverage = _v14._ledger_coverage(manifests)
        accumulators = {
            arm: {
                ledger: SourceAuthoritativeShadowFidelityAccumulator(
                    manifest,
                    gates=ESTABLISHED_SHADOW_FIDELITY_GATES,
                )
                for ledger, manifest in manifests.items()
            }
            for arm in ("base", *_ARM_IDS)
        }
        causal_checks = 0
        for record in records:
            model_inputs, supervised_indices, supervised_targets = _v14._retokenize(
                context.tokenize,
                record.example,
            )
            if (
                gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
                != record.model_inputs_sha256
                or _v14._tensor_sha256(supervised_indices)
                != record.supervised_indices_sha256
                or _v14._tensor_sha256(supervised_targets)
                != record.supervised_targets_sha256
            ):
                raise RuntimeError("V17 shadow retokenization drifted")
            source_logits, native_h4, native_positions, native_valid = (
                _v14._native_boundary(context.adapter, model_inputs)
            )
            base = context.bridge.execute(context.adapter, model_inputs)
            if (
                not isinstance(base, Gemma3L3L4OnePassExecution)
                or not _v14._bitwise_equal(
                    native_positions, base.prefix.logical_positions
                )
                or not _v14._bitwise_equal(
                    native_valid, base.prefix.valid_target_mask
                )
            ):
                raise RuntimeError("V17 shadow base sequence differs")
            _v14._add_shadow_rows(
                accumulators["base"],
                record=record,
                source_logits=source_logits,
                candidate_logits=base.logits,
                supervised_indices=supervised_indices,
                supervised_targets=supervised_targets,
            )
            held = record.sequence.family_id
            providers: tuple[
                tuple[
                    str,
                    AutonomousCompleteH4ResidualProvider
                    | AutonomousCompleteH4FisherXYPedalProvider,
                ],
                ...,
            ] = (
                (PARENT_ID, parents[held]),
                *((arm_id, children[arm_id][held]) for arm_id in _CHILD_IDS),
            )
            support = base.prefix.complete_h4_causal_support_mask()
            for arm_id, provider in providers:
                if held in set(provider.fit_family_ids):
                    raise RuntimeError("held family leaked into V17 provider")
                candidate_execution = context.bridge.execute(
                    context.adapter,
                    model_inputs,
                    h4_head=provider,
                )
                if (
                    candidate_execution.h4_head_sha256 != provider.artifact_sha256
                    or candidate_execution.prefix.artifact_sha256
                    != base.prefix.artifact_sha256
                    or not _v14._bitwise_equal(
                        candidate_execution.candidate_h4[
                            ~support.to(candidate_execution.candidate_h4.device)
                        ],
                        base.candidate_h4[
                            ~support.to(base.candidate_h4.device)
                        ],
                    )
                ):
                    raise RuntimeError("V17 provider escaped causal support")
                _v14._add_shadow_rows(
                    accumulators[arm_id],
                    record=record,
                    source_logits=source_logits,
                    candidate_logits=candidate_execution.logits,
                    supervised_indices=supervised_indices,
                    supervised_targets=supervised_targets,
                )
                causal_checks += 1
                del candidate_execution
            del model_inputs, source_logits, native_h4, base

        fidelity = {
            arm: {
                ledger: accumulator.finalize()
                for ledger, accumulator in ledgers.items()
            }
            for arm, ledgers in accumulators.items()
        }
        parent_resource_rows = [_parent_resources(value) for value in parents.values()]
        child_resource_rows = {
            arm_id: [_child_resources(value) for value in providers.values()]
            for arm_id, providers in children.items()
        }
        if (
            len({tuple(sorted(value.items())) for value in parent_resource_rows}) != 1
            or any(
                len({json.dumps(value, sort_keys=True) for value in rows}) != 1
                for rows in child_resource_rows.values()
            )
            or any(
                rows[0] != child_resource_rows[FISHER_PEDAL_ID][0]
                for rows in child_resource_rows.values()
            )
        ):
            raise RuntimeError("V17 outer-fold resource geometry differs")

        arm_rows: dict[str, Mapping[str, object]] = {
            PARENT_ID: {
                "arm_id": PARENT_ID,
                "coordinate_objective": _COORDINATE_OBJECTIVES[PARENT_ID],
                "recipe": {
                    **PARENT_RECIPE.metadata(),
                    "recipe_sha256": PARENT_RECIPE.artifact_sha256,
                },
                "outer_fold_count": len(parents),
                "every_fold_fit_family_count": _EXPECTED_FAMILIES - 1,
                "fold_provider_artifact_sha256s": {
                    held: provider.artifact_sha256
                    for held, provider in sorted(parents.items())
                },
                "fold_ownership_receipts": {
                    held: _provider_fold_ownership_receipt(
                        provider,
                        held_family_id=held,
                        held_sequences=held_sequences_by_family[held],
                    )
                    for held, provider in sorted(parents.items())
                },
                "serving_resources": parent_resource_rows[0],
                "fidelity": fidelity[PARENT_ID],
            }
        }
        for arm_id in _CHILD_IDS:
            providers = children[arm_id]
            held_diagnostics = {
                held: _held_runtime_diagnostics(
                    provider,
                    held_family_id=held,
                    held_sequences=held_sequences_by_family[held],
                )
                for held, provider in sorted(providers.items())
            }
            arm_rows[arm_id] = {
                "arm_id": arm_id,
                "coordinate_objective": _COORDINATE_OBJECTIVES[arm_id],
                "pedal_mode": _PEDAL_MODES[arm_id],
                "conditional_rank": _CONDITIONAL_RANK,
                "coordinate_count": FISHER_XY_COORDINATE_COUNT,
                "outer_fold_count": len(providers),
                "every_fold_fit_family_count": _EXPECTED_FAMILIES - 1,
                "fold_provider_artifact_sha256s": {
                    held: provider.artifact_sha256
                    for held, provider in sorted(providers.items())
                },
                "fold_parent_provider_artifact_sha256s": {
                    held: provider.parent_provider.artifact_sha256
                    for held, provider in sorted(providers.items())
                },
                "fold_ownership_receipts": {
                    held: _provider_fold_ownership_receipt(
                        provider,
                        held_family_id=held,
                        held_sequences=held_sequences_by_family[held],
                    )
                    for held, provider in sorted(providers.items())
                },
                "fit_diagnostics": {
                    held: _authenticated_fit_diagnostics(provider)
                    for held, provider in sorted(providers.items())
                },
                "held_runtime_diagnostics": held_diagnostics,
                "serving_resources": child_resource_rows[arm_id][0],
                "fidelity": fidelity[arm_id],
            }

        coordinate_geometry = evaluate_coordinate_geometry_gates(
            fisher_child=arm_rows[FISHER_PEDAL_ID],
            pca_child=arm_rows[PCA_PEDAL_ID],
        )
        trust = evaluate_trust_gates(arm_rows=arm_rows)
        mechanism = evaluate_mechanism_gates(
            parent=arm_rows[PARENT_ID],
            fisher_unit=arm_rows[FISHER_UNIT_ID],
            fisher_constant=arm_rows[FISHER_CONSTANT_ID],
            fisher_pedal=arm_rows[FISHER_PEDAL_ID],
            pca_pedal=arm_rows[PCA_PEDAL_ID],
        )
        absolute = _absolute_passed(arm_rows[FISHER_PEDAL_ID])
        outer_candidate_eligible = (
            coordinate_geometry["passed"] is True
            and trust["passed"] is True
            and mechanism["passed"] is True
            and absolute
        )
        fitted_full_provider: AutonomousCompleteH4FisherXYPedalProvider | None = None
        publish_provider: AutonomousCompleteH4FisherXYPedalProvider | None = None
        full_qualification: dict[str, object] | None = None
        if outer_candidate_eligible:
            all_sequences = tuple(record.sequence for record in records)
            full_parent = _fit_parent(
                all_sequences,
                bridge_binding_sha256=bridge_binding,
            )
            _validate_parent(full_parent, expected_fit_family_count=_EXPECTED_FAMILIES)
            fitted_full_provider = _fit_child(
                all_sequences,
                parent=full_parent,
                coordinate_objective="reverse_vjp_fisher",
                pedal_mode="conditional",
            )
            _validate_child(
                fitted_full_provider,
                coordinate_objective="reverse_vjp_fisher",
                pedal_mode="conditional",
                expected_parent_artifact_sha256=full_parent.artifact_sha256,
                expected_fit_family_count=_EXPECTED_FAMILIES,
            )
            full_qualification = _full_provider_fit_qualification(
                fitted_full_provider
            )
            if full_qualification["passed"] is True:
                publish_provider = fitted_full_provider
        candidate = (
            None
            if publish_provider is None
            else {
                "arm_id": FISHER_PEDAL_ID,
                "provider_artifact_sha256": publish_provider.artifact_sha256,
                "parent_provider_artifact_sha256": (
                    publish_provider.parent_provider.artifact_sha256
                ),
                "provider": publish_provider.metadata(),
                "serving_resources": _child_resources(publish_provider),
                "fit_family_count": _EXPECTED_FAMILIES,
                "fit_family_ids": publish_provider.fit_family_ids,
                "fit_sequence_sha256s": publish_provider.fit_sequence_sha256s,
                "full_provider_fit_qualification": full_qualification,
                "native_h4_logits_targets_gradients_or_coordinate_axes_required_at_runtime": False,
            }
        )
        context.validate_immutable_inputs()
        work = _work_accounting(
            prompt_count=len(records),
            outer_fold_count=len(folds),
            full_provider_fitted=fitted_full_provider is not None,
        )
        integrity = {
            "outer_fold_count": len(folds),
            "ledger_coverage": ledger_coverage,
            **work,
            "outer_fold_ownership_receipt_count": len(folds) * len(_ARM_IDS),
            "expected_outer_fold_ownership_receipt_count": (
                _EXPECTED_FAMILIES * len(_ARM_IDS)
            ),
            "held_runtime_diagnostic_count": (
                len(_CHILD_IDS) * len(folds)
            ),
            "expected_held_runtime_diagnostic_count": (
                _EXPECTED_HELD_RUNTIME_DIAGNOSTICS
            ),
            "parameter_matched_child_checks": len(folds),
            "expected_parameter_matched_child_checks": _EXPECTED_FAMILIES,
            "causal_off_support_execution_checks": causal_checks,
            "expected_causal_off_support_execution_checks": _EXPECTED_CAUSAL_CHECKS,
            "source_native_data_entered_serving_provider": False,
            "full_provider_fit_was_conditional_on_all_outer_gates": True,
            "full_provider_fit_qualification_required_before_candidate_and_publication": True,
            "guard_opened": False,
            "calibration_b_opened": False,
        }
        if (
            len(parents) * len(_ARM_IDS) != _EXPECTED_OUTER_PROVIDER_FITS
            or causal_checks != _EXPECTED_CAUSAL_CHECKS
            or len(held_sequences_by_family) != _EXPECTED_FAMILIES
            or sum(len(value) for value in children.values())
            != _EXPECTED_HELD_RUNTIME_DIAGNOSTICS
            or work["full_model_forward_count"] != _EXPECTED_FULL_MODEL_FORWARDS
        ):
            raise RuntimeError("V17 exact execution geometry differs")
        report = build_fisher_pedal_development_report(
            artifact_path=destination,
            panel=context.panel_receipt,
            bridge_binding_sha256=bridge_binding,
            folds=folds,
            prerequisites=prerequisites,
            fit_collection={
                "prompt_count": len(records),
                "family_count": len(families),
                "supervised_token_count": sum(
                    value.supervised_token_count for value in records
                ),
                "trace_receipt_sha256s": [value.receipt_sha256 for value in records],
                "native_h4_and_reverse_vjp_fit_only": True,
                "raw_fit_trace_tensor_serialization": False,
                "coordinate_axes_and_analytic_pedal_targets_fit_only_not_serialized": True,
                "conditional_runtime_provider_tensor_sidecar": candidate is not None,
            },
            base_fidelity=fidelity["base"],
            arm_rows=arm_rows,
            full_refit_qualification=full_qualification,
            candidate=candidate,
            integrity=integrity,
        )
        return _publish(
            report,
            output=destination,
            provider=publish_provider,
            provider_output=provider_destination,
        )
    finally:
        context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider-output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_fisher_pedal_development(
        output=arguments.output,
        provider_output=arguments.provider_output,
        cache_dir=arguments.cache_dir,
    )
    print(
        json.dumps(
            {
                "path": report["artifact"]["path"],  # type: ignore[index]
                "report_sha256": report["report_sha256"],
                "classification": report["classification"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
