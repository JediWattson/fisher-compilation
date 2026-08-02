"""Fixed V16 Fisher-square outer-LOFO development screen.

V16 compares one K256/L8 reverse-VJP parent with two exactly
parameter-matched rank-16 conditional children.  The children differ only in
the fit-only target used to train their two-coordinate runtime router:
reverse-VJP Fisher axes or activation-PCA axes.  Every decoder, parent,
router, and conditional child is fitted inside its train-seven outer fold.

The opened A16 panel remains development data.  Source logits stay
authoritative, the full Gemma suffix and vocabulary are evaluated, and a
full-panel Fisher provider is serialized only when the preregistered
coordinate-geometry, established absolute-fidelity, and mechanism-retention
gates all pass.  This runner never opens a fresh guard or Calibration B.
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
from .complete_h4_fisher_conditional_residual import (
    COORDINATE_OBJECTIVES,
    FISHER_XY_COORDINATE_COUNT,
    FISHER_XY_OPERATOR_NORM_BOUND,
    AutonomousCompleteH4FisherXYProvider,
    FisherXYBoundedCoordinateGeometry,
    autonomous_complete_h4_fisher_xy_provider_state_dict,
    fit_autonomous_complete_h4_fisher_xy_residual,
    load_autonomous_complete_h4_fisher_xy_provider,
    replay_autonomous_complete_h4_fisher_xy_bounded_coordinates,
    summarize_fisher_xy_bounded_coordinate_geometry,
)
from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
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
    "FISHER_CHILD_ID",
    "PARENT_ID",
    "PCA_CHILD_ID",
    "PARENT_RECIPE",
    "build_fisher_square_development_report",
    "evaluate_coordinate_geometry_gates",
    "evaluate_mechanism_gates",
    "run_gemma3_l3_l4_complete_h4_fisher_square_development",
    "build_parser",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-fisher-square-r16-k256-"
    "outer-lofo-held-runtime-geometry-a-fit16-dev-v16.json"
)
DEFAULT_PROVIDER_OUTPUT = DEFAULT_OUTPUT.with_suffix(".provider.pt")
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_fisher_square_"
    "outer_lofo_development.v16"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-fisher-square-dev:v16\0"

_EXPECTED_PROMPTS = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_HELD_SEQUENCES_PER_FOLD = 2
_EXPECTED_TRAINING_PROMPTS_PER_FOLD = 14
_EXPECTED_OUTER_PARENT_FITS = 8
_EXPECTED_OUTER_CHILD_FITS = 16
_EXPECTED_OUTER_PROVIDER_FITS = 24
_EXPECTED_FULL_MODEL_FORWARDS = 112
_EXPECTED_BACKWARD_VJP_TRAVERSALS = 16
_EXPECTED_CAUSAL_CHECKS = 48
_EXPECTED_PARENT_SCALARS = 360_704
_EXPECTED_PARENT_MACS = 524_288
_EXPECTED_CHILD_INCREMENTAL_SCALARS = 16_900
_EXPECTED_CHILD_INCREMENTAL_MACS = 16_896
_EXPECTED_CHILD_SCALARS = 377_604
_EXPECTED_CHILD_MACS = 541_184

_V14_REPORT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-autonomous-residual-"
    "outer-lofo-a-fit16-dev-v14.json"
)
_V14_LOGICAL_SHA256 = (
    "01803d62e106de05acafcd000308ae2f861f2be9c6bb879fd2d7f4c9e611f906"
)
_V14_FILE_SHA256 = (
    "fc78f37790898dd3acbded89f6da8a2fa9ee466217d1eba3f123e570859384ad"
)
_V14_CLASSIFICATION = "autonomous_complete_h4_oof_recipes_insufficient"
_V15_REPORT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-autonomous-residual-k640-capacity-"
    "outer-lofo-a-fit16-dev-v15.json"
)
_V15_LOGICAL_SHA256 = (
    "8518dab697e78ecb210a1fb99e173f486f8939893b081ba0d28d970c82e86ff4"
)
_V15_FILE_SHA256 = (
    "4c6033385f278437849da770c18058720c6e3dcb075e3d0844c631480994de18"
)
_V15_CLASSIFICATION = (
    "autonomous_complete_h4_k640_oof_capacity_ceiling_insufficient"
)

_CONDITIONAL_RANK = 16
_ROUTER_RIDGE = 1.0e-4
_CONDITIONAL_RIDGE = 1.0e-4

PARENT_ID = "r256_l8_reverse_vjp_parent"
FISHER_CHILD_ID = "r256_l8_reverse_vjp_fisher_square_r16"
PCA_CHILD_ID = "r256_l8_reverse_vjp_activation_pca_square_r16"
_CHILD_IDS = (FISHER_CHILD_ID, PCA_CHILD_ID)
_ARM_IDS = (PARENT_ID, FISHER_CHILD_ID, PCA_CHILD_ID)

PARENT_RECIPE = _v14.AutonomousResidualRecipe(
    recipe_id=PARENT_ID,
    rank=256,
    lag_count=8,
    ridge=1.0e-4,
    fit_objective="reverse_vjp_row_weighted_ridge_v1",
)

_MECHANISM_THRESHOLDS = {
    "ordinary_family_macro_absolute_delta_nll_relative_improvement_min": 0.05,
    "ordinary_family_macro_kl_relative_improvement_min": 0.05,
    "ordinary_aggregate_top1_gain_min": 0.02,
    "ordinary_family_absolute_delta_nll_win_count_min": 6,
    "worst_family_absolute_delta_nll_relative_regression_max": 0.02,
    "fisher_vs_pca_absolute_delta_nll_rule": "strictly_lower",
    "fisher_vs_pca_kl_rule": "not_higher",
    "fisher_vs_pca_top1_rule": "not_lower",
    "support_and_graph_core_rule": "no_aggregate_delta_nll_kl_or_top1_regression",
}

_COORDINATE_GEOMETRY_THRESHOLDS = {
    "fit_fold_bounded_coordinate_target_r2_each_min": 0.01,
    "held_family_runtime_bounded_coordinate_lambda2_over_lambda1_min": 0.01,
    "held_family_runtime_bounded_coordinate_abs_correlation_max": 0.99,
    "held_family_runtime_residual_second_coordinate_energy_fraction_min": 0.01,
    # Retain the fit-fold geometry thresholds as authenticated diagnostics and
    # as the full-panel provider's final fail-closed qualification.  Held-fold
    # qualification below does not substitute fit rows for actual held rows.
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
_COORDINATE_DIAGNOSTIC_FIELDS = (
    "bounded_coordinate_geometry_sha256",
    "bounded_coordinate_covariance_eigenvalues",
    "bounded_coordinate_lambda2_over_lambda1",
    "bounded_coordinate_abs_correlation",
    "bounded_coordinate_target_r2",
    "residual_second_coordinate_energy_fraction",
)
_ADDITIONAL_FIT_DIAGNOSTIC_FIELDS = (
    "coordinate_axis_values",
    "coordinate_target_weighted_rmse",
    "weighted_residual_rmse_before",
    "weighted_residual_rmse_after",
    "post_projection_corner_operator_norms",
    "trust_projection_scale",
)
_RUNTIME_COORDINATE_INPUT_FIELDS = (
    "source_modes",
    "logical_positions",
    "valid_mask",
    "source_mask",
    "support_mask",
    "base_h4",
)
_HELD_RUNTIME_COORDINATE_DIAGNOSTIC_FIELDS = (
    "bounded_coordinate_covariance_eigenvalues",
    "bounded_coordinate_lambda2_over_lambda1",
    "bounded_coordinate_abs_correlation",
    "residual_second_coordinate_energy_fraction",
)


def _validate_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".json" or ".local-runs" not in output.parts:
        raise ValueError("V16 output must be JSON under .local-runs")
    return output


def _validate_provider_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".pt" or ".local-runs" not in output.parts:
        raise ValueError("V16 provider output must be PT under .local-runs")
    return output


def _validate_prerequisite_report(
    path: Path,
    *,
    logical_sha256: str,
    file_sha256: str,
    classification: str,
    format_version: int,
) -> dict[str, object]:
    if not path.is_file() or _v14._file_sha256(path) != file_sha256:
        raise RuntimeError(f"V16 prerequisite V{format_version} file drifted")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"V16 prerequisite V{format_version} report is unreadable"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("format_version") != format_version
        or payload.get("report_sha256") != logical_sha256
        or payload.get("classification") != classification
        or payload.get("passed") is not False
        or payload.get("candidate") is not None
        or _mapping(payload.get("integrity"), label="prerequisite integrity").get(
            "guard_opened"
        )
        is not False
        or _mapping(payload.get("integrity"), label="prerequisite integrity").get(
            "calibration_b_opened"
        )
        is not False
    ):
        raise RuntimeError(f"V16 prerequisite V{format_version} semantics drifted")
    return {
        "path": path.as_posix(),
        "format_version": format_version,
        "report_sha256": logical_sha256,
        "file_sha256": file_sha256,
        "classification": classification,
        "passed": False,
        "candidate": None,
        "guard_opened": False,
        "calibration_b_opened": False,
    }


def _validate_prerequisites() -> dict[str, object]:
    return {
        "v14": _validate_prerequisite_report(
            _V14_REPORT,
            logical_sha256=_V14_LOGICAL_SHA256,
            file_sha256=_V14_FILE_SHA256,
            classification=_V14_CLASSIFICATION,
            format_version=14,
        ),
        "v15": _validate_prerequisite_report(
            _V15_REPORT,
            logical_sha256=_V15_LOGICAL_SHA256,
            file_sha256=_V15_FILE_SHA256,
            classification=_V15_CLASSIFICATION,
            format_version=15,
        ),
    }


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _sha256_identifier(value: object, *, label: str) -> str:
    selected = _v14._identifier(value, label=label)
    if len(selected) != 64 or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return selected


def _ledger(arm: Mapping[str, object], name: str) -> Mapping[str, object]:
    fidelity = _mapping(arm.get("fidelity"), label="arm fidelity")
    return _mapping(fidelity.get(name), label=f"{name} fidelity")


def _family_macro(arm: Mapping[str, object], ledger: str) -> Mapping[str, object]:
    summary = _mapping(
        _ledger(arm, ledger).get("family_summary"),
        label=f"{ledger} family summary",
    )
    return _mapping(summary.get("macro"), label=f"{ledger} family macro")


def _aggregate(arm: Mapping[str, object], ledger: str) -> Mapping[str, object]:
    return _mapping(
        _ledger(arm, ledger).get("aggregate"),
        label=f"{ledger} aggregate",
    )


def _family_rows(arm: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    summary = _mapping(
        _ledger(arm, "ordinary").get("family_summary"),
        label="ordinary family summary",
    )
    rows = summary.get("families")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("ordinary family rows must be a sequence")
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        selected = _mapping(row, label="ordinary family row")
        family = _v14._identifier(selected.get("family_id"), label="family_id")
        if family in result:
            raise ValueError("ordinary family rows must be unique")
        result[family] = selected
    if len(result) != _EXPECTED_FAMILIES:
        raise ValueError("mechanism gates require all eight families")
    return result


def _relative_improvement(parent: float, candidate: float, *, label: str) -> float:
    if parent <= 0.0:
        raise ValueError(f"{label} parent must be positive")
    return (parent - candidate) / parent


def _pair(value: object, *, label: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{label} must contain exactly two values")
    return (
        _finite(value[0], label=f"{label}[0]"),
        _finite(value[1], label=f"{label}[1]"),
    )


def _coordinate_fold_diagnostics(
    value: object,
    *,
    expected_artifact_sha256: str,
    expected_coordinate_objective: str,
) -> dict[str, object]:
    row = _mapping(value, label="coordinate fold diagnostics")
    artifact = _sha256_identifier(
        row.get("provider_artifact_sha256"),
        label="coordinate diagnostic provider artifact",
    )
    if artifact != expected_artifact_sha256:
        raise ValueError("coordinate diagnostics are not bound to the fold provider")
    objective = _v14._identifier(
        row.get("coordinate_objective"),
        label="coordinate diagnostic objective",
    )
    if objective != expected_coordinate_objective:
        raise ValueError("coordinate diagnostic objective differs from its arm")
    geometry_artifact = _sha256_identifier(
        row.get("bounded_coordinate_geometry_sha256"),
        label="fit bounded coordinate geometry artifact",
    )

    eigenvalues = _pair(
        row.get("bounded_coordinate_covariance_eigenvalues"),
        label="bounded coordinate covariance eigenvalues",
    )
    ratio = _finite(
        row.get("bounded_coordinate_lambda2_over_lambda1"),
        label="bounded coordinate lambda2/lambda1",
    )
    correlation = _finite(
        row.get("bounded_coordinate_abs_correlation"),
        label="bounded coordinate absolute correlation",
    )
    target_r2 = _pair(
        row.get("bounded_coordinate_target_r2"),
        label="bounded coordinate target R2",
    )
    residual_energy = _finite(
        row.get("residual_second_coordinate_energy_fraction"),
        label="residual second-coordinate energy fraction",
    )
    expected_ratio = (
        0.0 if eigenvalues[0] == 0.0 else eigenvalues[1] / eigenvalues[0]
    )
    if (
        eigenvalues[0] < 0.0
        or eigenvalues[1] < 0.0
        or eigenvalues[1] > eigenvalues[0]
        or ratio < 0.0
        or ratio > 1.0
        or not math.isclose(
            ratio,
            expected_ratio,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        or correlation < 0.0
        or correlation > 1.0
        or any(value > 1.0 for value in target_r2)
        or residual_energy < 0.0
        or residual_energy > 1.0
    ):
        raise ValueError("coordinate diagnostics have invalid scalar geometry")
    gates = {
        "bounded_covariance_rank2": ratio
        >= _COORDINATE_GEOMETRY_THRESHOLDS[
            "bounded_coordinate_lambda2_over_lambda1_min"
        ],
        "bounded_coordinates_not_collinear": correlation
        <= _COORDINATE_GEOMETRY_THRESHOLDS[
            "bounded_coordinate_abs_correlation_max"
        ],
        "both_bounded_targets_predictable": min(target_r2)
        >= _COORDINATE_GEOMETRY_THRESHOLDS[
            "bounded_coordinate_target_r2_each_min"
        ],
        "residual_has_second_coordinate_energy": residual_energy
        >= _COORDINATE_GEOMETRY_THRESHOLDS[
            "residual_second_coordinate_energy_fraction_min"
        ],
    }
    return {
        "provider_artifact_sha256": artifact,
        "coordinate_objective": objective,
        "bounded_coordinate_geometry_sha256": geometry_artifact,
        "bounded_coordinate_covariance_eigenvalues": eigenvalues,
        "bounded_coordinate_lambda2_over_lambda1": ratio,
        "bounded_coordinate_abs_correlation": correlation,
        "bounded_coordinate_target_r2": target_r2,
        "residual_second_coordinate_energy_fraction": residual_energy,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _sha256_tuple(
    value: object,
    *,
    count: int,
    label: str,
) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != count
    ):
        raise ValueError(f"{label} must contain exactly {count} SHA-256 values")
    result = tuple(
        _sha256_identifier(item, label=f"{label} SHA-256") for item in value
    )
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be sorted and unique")
    return result


def _fold_ownership_receipt(
    value: object,
    *,
    expected_held_family_id: str,
    expected_provider_artifact_sha256: str,
    expected_parent_provider_artifact_sha256: str | None,
    expected_coordinate_objective: str | None,
) -> dict[str, object]:
    row = _mapping(value, label="fold ownership receipt")
    held_family = _v14._identifier(
        row.get("held_family_id"),
        label="ownership held family",
    )
    provider_artifact = _sha256_identifier(
        row.get("provider_artifact_sha256"),
        label="ownership provider artifact",
    )
    fit_families_value = row.get("fit_family_ids")
    if (
        not isinstance(fit_families_value, Sequence)
        or isinstance(fit_families_value, (str, bytes))
    ):
        raise TypeError("ownership fit families must be a sequence")
    fit_families = tuple(
        _v14._identifier(item, label="ownership fit family")
        for item in fit_families_value
    )
    fit_sequences = _sha256_tuple(
        row.get("fit_sequence_sha256s"),
        count=_EXPECTED_TRAINING_PROMPTS_PER_FOLD,
        label="ownership fit sequences",
    )
    held_sequences = _sha256_tuple(
        row.get("held_sequence_sha256s"),
        count=_EXPECTED_HELD_SEQUENCES_PER_FOLD,
        label="ownership held sequences",
    )
    if (
        held_family != expected_held_family_id
        or provider_artifact != expected_provider_artifact_sha256
        or fit_families != tuple(sorted(set(fit_families)))
        or len(fit_families) != _EXPECTED_FAMILIES - 1
        or held_family in set(fit_families)
        or set(fit_sequences) & set(held_sequences)
    ):
        raise ValueError("fold ownership receipt differs from outer LOFO ownership")
    result: dict[str, object] = {
        "held_family_id": held_family,
        "provider_artifact_sha256": provider_artifact,
        "fit_family_ids": fit_families,
        "fit_sequence_sha256s": fit_sequences,
        "held_sequence_sha256s": held_sequences,
    }
    if expected_parent_provider_artifact_sha256 is None:
        if "parent_provider_artifact_sha256" in row:
            raise ValueError("parent ownership receipt cannot name a parent provider")
    else:
        parent_artifact = _sha256_identifier(
            row.get("parent_provider_artifact_sha256"),
            label="ownership parent provider artifact",
        )
        if parent_artifact != expected_parent_provider_artifact_sha256:
            raise ValueError("fold ownership parent provider differs")
        result["parent_provider_artifact_sha256"] = parent_artifact
    if expected_coordinate_objective is None:
        if "coordinate_objective" in row:
            raise ValueError("parent ownership receipt cannot name coordinates")
    else:
        objective = _v14._identifier(
            row.get("coordinate_objective"),
            label="ownership coordinate objective",
        )
        if objective != expected_coordinate_objective:
            raise ValueError("fold ownership coordinate objective differs")
        result["coordinate_objective"] = objective
    return result


def _held_runtime_coordinate_fold_diagnostics(
    value: object,
    *,
    expected_artifact_sha256: str,
    expected_parent_artifact_sha256: str,
    expected_coordinate_objective: str,
    expected_held_family_id: str,
    expected_held_sequence_sha256s: tuple[str, ...],
) -> dict[str, object]:
    row = _mapping(value, label="held runtime coordinate diagnostics")
    artifact = _sha256_identifier(
        row.get("provider_artifact_sha256"),
        label="held runtime provider artifact",
    )
    parent_artifact = _sha256_identifier(
        row.get("parent_provider_artifact_sha256"),
        label="held runtime parent provider artifact",
    )
    objective = _v14._identifier(
        row.get("coordinate_objective"),
        label="held runtime coordinate objective",
    )
    held_family = _v14._identifier(
        row.get("held_family_id"),
        label="held runtime family",
    )
    held_sequences = _sha256_tuple(
        row.get("held_sequence_sha256s"),
        count=_EXPECTED_HELD_SEQUENCES_PER_FOLD,
        label="held runtime sequences",
    )
    runtime_fields_value = row.get("runtime_input_fields")
    runtime_fields = (
        tuple(runtime_fields_value)
        if isinstance(runtime_fields_value, Sequence)
        and not isinstance(runtime_fields_value, (str, bytes))
        else ()
    )
    if (
        artifact != expected_artifact_sha256
        or parent_artifact != expected_parent_artifact_sha256
        or objective != expected_coordinate_objective
        or held_family != expected_held_family_id
        or held_sequences != expected_held_sequence_sha256s
        or runtime_fields != _RUNTIME_COORDINATE_INPUT_FIELDS
        or row.get("weighting_semantics")
        != "equal_sequences_then_equal_supported_rows"
    ):
        raise ValueError("held runtime coordinate ownership differs")

    sequence_receipts_value = row.get("sequence_coordinate_receipts")
    if (
        not isinstance(sequence_receipts_value, Sequence)
        or isinstance(sequence_receipts_value, (str, bytes))
        or len(sequence_receipts_value) != _EXPECTED_HELD_SEQUENCES_PER_FOLD
    ):
        raise ValueError("held runtime sequence receipts differ")
    sequence_receipts: list[dict[str, object]] = []
    for receipt_value in sequence_receipts_value:
        receipt = _mapping(receipt_value, label="held runtime sequence receipt")
        sequence_sha = _sha256_identifier(
            receipt.get("sequence_sha256"),
            label="held runtime sequence",
        )
        coordinate_sha = _sha256_identifier(
            receipt.get("bounded_coordinates_sha256"),
            label="held runtime sequence coordinates",
        )
        row_count = receipt.get("row_count")
        if type(row_count) is not int or row_count <= 0:
            raise ValueError("held runtime sequence row count must be positive")
        sequence_receipts.append(
            {
                "sequence_sha256": sequence_sha,
                "row_count": row_count,
                "bounded_coordinates_sha256": coordinate_sha,
            }
        )
    sequence_receipts.sort(key=lambda item: str(item["sequence_sha256"]))
    if tuple(item["sequence_sha256"] for item in sequence_receipts) != held_sequences:
        raise ValueError("held runtime sequence receipt ownership differs")

    geometry = FisherXYBoundedCoordinateGeometry(
        row_count=row.get("row_count"),  # type: ignore[arg-type]
        bounded_coordinates_sha256=row.get("bounded_coordinates_sha256"),  # type: ignore[arg-type]
        row_weight_sha256=row.get("row_weight_sha256"),  # type: ignore[arg-type]
        covariance_eigenvalues=_pair(
            row.get("bounded_coordinate_covariance_eigenvalues"),
            label="held runtime covariance eigenvalues",
        ),
        lambda2_over_lambda1=_finite(
            row.get("bounded_coordinate_lambda2_over_lambda1"),
            label="held runtime coordinate ratio",
        ),
        abs_correlation=_finite(
            row.get("bounded_coordinate_abs_correlation"),
            label="held runtime coordinate correlation",
        ),
        residual_second_coordinate_energy_fraction=_finite(
            row.get("residual_second_coordinate_energy_fraction"),
            label="held runtime residual second-coordinate energy",
        ),
        artifact_sha256=row.get("geometry_artifact_sha256"),  # type: ignore[arg-type]
    )
    geometry.validate_integrity()
    if sum(int(item["row_count"]) for item in sequence_receipts) != geometry.row_count:
        raise ValueError("held runtime aggregate row count differs")
    gates = {
        "bounded_covariance_rank2": geometry.lambda2_over_lambda1
        >= _COORDINATE_GEOMETRY_THRESHOLDS[
            "held_family_runtime_bounded_coordinate_lambda2_over_lambda1_min"
        ],
        "bounded_coordinates_not_collinear": geometry.abs_correlation
        <= _COORDINATE_GEOMETRY_THRESHOLDS[
            "held_family_runtime_bounded_coordinate_abs_correlation_max"
        ],
        "residual_has_second_coordinate_energy": (
            geometry.residual_second_coordinate_energy_fraction
            >= _COORDINATE_GEOMETRY_THRESHOLDS[
                "held_family_runtime_residual_second_coordinate_energy_fraction_min"
            ]
        ),
    }
    return {
        "provider_artifact_sha256": artifact,
        "parent_provider_artifact_sha256": parent_artifact,
        "coordinate_objective": objective,
        "held_family_id": held_family,
        "held_sequence_sha256s": held_sequences,
        "sequence_coordinate_receipts": tuple(sequence_receipts),
        "weighting_semantics": "equal_sequences_then_equal_supported_rows",
        "runtime_input_fields": _RUNTIME_COORDINATE_INPUT_FIELDS,
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
        "gates": gates,
        "passed": all(gates.values()),
    }


def _coordinate_arm_geometry(
    arm: Mapping[str, object],
    *,
    expected_arm_id: str,
    expected_coordinate_objective: str,
) -> dict[str, object]:
    if arm.get("arm_id") != expected_arm_id:
        raise ValueError("coordinate geometry arm identity differs")
    artifact_rows = _mapping(
        arm.get("fold_provider_artifact_sha256s"),
        label="coordinate fold provider artifacts",
    )
    parent_artifact_rows = _mapping(
        arm.get("fold_parent_provider_artifact_sha256s"),
        label="coordinate fold parent provider artifacts",
    )
    fit_rows = _mapping(
        arm.get("fit_diagnostics"),
        label="coordinate fold fit diagnostics",
    )
    held_runtime_rows = _mapping(
        arm.get("held_runtime_coordinate_diagnostics"),
        label="held runtime coordinate diagnostics",
    )
    ownership_rows = _mapping(
        arm.get("fold_ownership_receipts"),
        label="coordinate fold ownership receipts",
    )
    if (
        len(artifact_rows) != _EXPECTED_FAMILIES
        or set(artifact_rows) != set(parent_artifact_rows)
        or set(artifact_rows) != set(fit_rows)
        or set(artifact_rows) != set(held_runtime_rows)
        or set(artifact_rows) != set(ownership_rows)
    ):
        raise ValueError("coordinate diagnostics must cover all eight outer folds")
    fit_per_fold: dict[str, object] = {}
    held_per_fold: dict[str, object] = {}
    ownership_per_fold: dict[str, object] = {}
    for family in sorted(artifact_rows):
        selected_family = _v14._identifier(family, label="held family")
        artifact = _sha256_identifier(
            artifact_rows[family],
            label="fold provider artifact",
        )
        parent_artifact = _sha256_identifier(
            parent_artifact_rows[family],
            label="fold parent provider artifact",
        )
        ownership = _fold_ownership_receipt(
            ownership_rows[family],
            expected_held_family_id=selected_family,
            expected_provider_artifact_sha256=artifact,
            expected_parent_provider_artifact_sha256=parent_artifact,
            expected_coordinate_objective=expected_coordinate_objective,
        )
        fit = _coordinate_fold_diagnostics(
            fit_rows[family],
            expected_artifact_sha256=artifact,
            expected_coordinate_objective=expected_coordinate_objective,
        )
        target_r2 = _pair(
            fit.get("bounded_coordinate_target_r2"),
            label="fit bounded coordinate target R2",
        )
        fit_gate = {
            "both_bounded_targets_predictable": min(target_r2)
            >= _COORDINATE_GEOMETRY_THRESHOLDS[
                "fit_fold_bounded_coordinate_target_r2_each_min"
            ]
        }
        fit_per_fold[selected_family] = {
            **fit,
            "gates": fit_gate,
            "passed": all(fit_gate.values()),
        }
        held_per_fold[selected_family] = _held_runtime_coordinate_fold_diagnostics(
            held_runtime_rows[family],
            expected_artifact_sha256=artifact,
            expected_parent_artifact_sha256=parent_artifact,
            expected_coordinate_objective=expected_coordinate_objective,
            expected_held_family_id=selected_family,
            expected_held_sequence_sha256s=ownership["held_sequence_sha256s"],  # type: ignore[arg-type]
        )
        ownership_per_fold[selected_family] = ownership

    fit_selected = tuple(
        _mapping(value, label="fit coordinate fold")
        for value in fit_per_fold.values()
    )
    held_selected = tuple(
        _mapping(value, label="held runtime coordinate fold")
        for value in held_per_fold.values()
    )
    fit_failed_folds = tuple(
        family
        for family, value in fit_per_fold.items()
        if _mapping(value, label="fit coordinate fold").get("passed") is not True
    )
    held_failed_folds = tuple(
        family
        for family, value in held_per_fold.items()
        if _mapping(value, label="held runtime coordinate fold").get("passed")
        is not True
    )
    fit_qualification = {
        "semantics": "training_fold_router_prediction_of_fit_only_bounded_targets",
        "per_fold": fit_per_fold,
        "worst_case": {
            "bounded_coordinate_target_r2_min_observed": min(
                min(
                    _pair(
                        value.get("bounded_coordinate_target_r2"),
                        label="fit bounded coordinate target R2",
                    )
                )
                for value in fit_selected
            )
        },
        "failed_fold_ids": fit_failed_folds,
        "failed_fold_count": len(fit_failed_folds),
        "passed": not fit_failed_folds,
    }
    held_qualification = {
        "semantics": (
            "actual_runtime_allowed_bounded_coordinates_on_two_unseen_held_"
            "family_sequences"
        ),
        "per_fold": held_per_fold,
        "worst_case": {
            "bounded_coordinate_lambda2_over_lambda1_min_observed": min(
                _finite(
                    value.get("bounded_coordinate_lambda2_over_lambda1"),
                    label="held runtime bounded coordinate ratio",
                )
                for value in held_selected
            ),
            "bounded_coordinate_abs_correlation_max_observed": max(
                _finite(
                    value.get("bounded_coordinate_abs_correlation"),
                    label="held runtime bounded coordinate correlation",
                )
                for value in held_selected
            ),
            "residual_second_coordinate_energy_fraction_min_observed": min(
                _finite(
                    value.get("residual_second_coordinate_energy_fraction"),
                    label="held runtime residual second-coordinate energy",
                )
                for value in held_selected
            ),
        },
        "failed_fold_ids": held_failed_folds,
        "failed_fold_count": len(held_failed_folds),
        "passed": not held_failed_folds,
    }
    return {
        "arm_id": expected_arm_id,
        "coordinate_objective": expected_coordinate_objective,
        "fold_count": len(fit_per_fold),
        "ownership_receipts": ownership_per_fold,
        "fit_fold_predictability": fit_qualification,
        "held_family_runtime_geometry": held_qualification,
        "passed": (
            fit_qualification["passed"] is True
            and held_qualification["passed"] is True
        ),
    }


def evaluate_coordinate_geometry_gates(
    *,
    fisher_child: Mapping[str, object],
    pca_child: Mapping[str, object],
) -> dict[str, object]:
    """Validate two-dimensional coordinate support in both child arms."""

    fisher = _coordinate_arm_geometry(
        fisher_child,
        expected_arm_id=FISHER_CHILD_ID,
        expected_coordinate_objective="reverse_vjp_fisher",
    )
    pca = _coordinate_arm_geometry(
        pca_child,
        expected_arm_id=PCA_CHILD_ID,
        expected_coordinate_objective="activation_pca",
    )
    arms = {FISHER_CHILD_ID: fisher, PCA_CHILD_ID: pca}
    fisher_folds = set(
        _mapping(
            _mapping(
                fisher.get("held_family_runtime_geometry"),
                label="Fisher held runtime geometry",
            ).get("per_fold"),
            label="Fisher held runtime folds",
        )
    )
    pca_folds = set(
        _mapping(
            _mapping(
                pca.get("held_family_runtime_geometry"),
                label="PCA held runtime geometry",
            ).get("per_fold"),
            label="PCA held runtime folds",
        )
    )
    if fisher_folds != pca_folds:
        raise ValueError("Fisher and PCA coordinate folds differ")

    def worst(
        value: object,
        section: str,
        key: str,
        *,
        label: str,
    ) -> float:
        arm = _mapping(value, label="coordinate arm")
        qualification = _mapping(
            arm.get(section),
            label=f"coordinate arm {section}",
        )
        observations = _mapping(
            qualification.get("worst_case"),
            label="coordinate arm worst case",
        )
        return _finite(observations.get(key), label=label)

    worst_case = {
        "fit_fold_bounded_coordinate_target_r2_min_observed": min(
            worst(
                value,
                "fit_fold_predictability",
                "bounded_coordinate_target_r2_min_observed",
                label="worst fit bounded coordinate target R2",
            )
            for value in arms.values()
        ),
        "held_family_runtime_bounded_coordinate_lambda2_over_lambda1_min_observed": min(
            worst(
                value,
                "held_family_runtime_geometry",
                "bounded_coordinate_lambda2_over_lambda1_min_observed",
                label="worst held runtime coordinate ratio",
            )
            for value in arms.values()
        ),
        "held_family_runtime_bounded_coordinate_abs_correlation_max_observed": max(
            worst(
                value,
                "held_family_runtime_geometry",
                "bounded_coordinate_abs_correlation_max_observed",
                label="worst held runtime coordinate correlation",
            )
            for value in arms.values()
        ),
        "held_family_runtime_residual_second_coordinate_energy_fraction_min_observed": min(
            worst(
                value,
                "held_family_runtime_geometry",
                "residual_second_coordinate_energy_fraction_min_observed",
                label="worst held runtime residual second-coordinate energy",
            )
            for value in arms.values()
        ),
    }
    return {
        "thresholds": dict(_COORDINATE_GEOMETRY_THRESHOLDS),
        "fit_provider_metadata_fields": _COORDINATE_DIAGNOSTIC_FIELDS,
        "held_runtime_geometry_fields": _HELD_RUNTIME_COORDINATE_DIAGNOSTIC_FIELDS,
        "fit_diagnostics_authenticated_by_provider_artifact_sha256": True,
        "held_runtime_geometry_authenticated_by_geometry_artifact_sha256": True,
        "arms": arms,
        "worst_case_across_both_arms": worst_case,
        "passed": all(value["passed"] is True for value in arms.values()),
    }


def evaluate_mechanism_gates(
    *,
    parent: Mapping[str, object],
    fisher_child: Mapping[str, object],
    pca_child: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the preregistered V16 mechanism-retention gates."""

    parent_macro = _family_macro(parent, "ordinary")
    fisher_macro = _family_macro(fisher_child, "ordinary")
    pca_macro = _family_macro(pca_child, "ordinary")
    parent_ordinary_aggregate = _aggregate(parent, "ordinary")
    fisher_ordinary_aggregate = _aggregate(fisher_child, "ordinary")
    pca_ordinary_aggregate = _aggregate(pca_child, "ordinary")
    parent_abs = _finite(
        parent_macro.get("absolute_delta_nll_per_token"),
        label="parent ordinary macro absolute delta NLL",
    )
    fisher_abs = _finite(
        fisher_macro.get("absolute_delta_nll_per_token"),
        label="Fisher ordinary macro absolute delta NLL",
    )
    pca_abs = _finite(
        pca_macro.get("absolute_delta_nll_per_token"),
        label="PCA ordinary macro absolute delta NLL",
    )
    parent_kl = _finite(
        parent_macro.get("source_to_candidate_kl_per_token"),
        label="parent ordinary macro KL",
    )
    fisher_kl = _finite(
        fisher_macro.get("source_to_candidate_kl_per_token"),
        label="Fisher ordinary macro KL",
    )
    pca_kl = _finite(
        pca_macro.get("source_to_candidate_kl_per_token"),
        label="PCA ordinary macro KL",
    )
    parent_top1 = _finite(
        parent_ordinary_aggregate.get("top1_agreement_to_source"),
        label="parent ordinary aggregate top1",
    )
    fisher_top1 = _finite(
        fisher_ordinary_aggregate.get("top1_agreement_to_source"),
        label="Fisher ordinary aggregate top1",
    )
    pca_top1 = _finite(
        pca_ordinary_aggregate.get("top1_agreement_to_source"),
        label="PCA ordinary aggregate top1",
    )

    parent_families = _family_rows(parent)
    fisher_families = _family_rows(fisher_child)
    pca_families = _family_rows(pca_child)
    if set(parent_families) != set(fisher_families) or set(parent_families) != set(
        pca_families
    ):
        raise ValueError("mechanism family membership differs")
    family_improvements: dict[str, float] = {}
    for family, parent_row in parent_families.items():
        parent_value = _finite(
            parent_row.get("absolute_delta_nll_per_token"),
            label="parent family absolute delta NLL",
        )
        fisher_value = _finite(
            fisher_families[family].get("absolute_delta_nll_per_token"),
            label="Fisher family absolute delta NLL",
        )
        family_improvements[family] = _relative_improvement(
            parent_value,
            fisher_value,
            label="family absolute delta NLL",
        )
    win_count = sum(value > 0.0 for value in family_improvements.values())
    worst_improvement = min(family_improvements.values())

    no_required_ledger_regression: dict[str, bool] = {}
    required_ledger_observations: dict[str, object] = {}
    for ledger in ("complete_h4_support", "graph_core"):
        parent_aggregate = _aggregate(parent, ledger)
        fisher_aggregate = _aggregate(fisher_child, ledger)
        parent_delta = abs(
            _finite(
                parent_aggregate.get("delta_nll_per_token"),
                label=f"parent {ledger} delta NLL",
            )
        )
        fisher_delta = abs(
            _finite(
                fisher_aggregate.get("delta_nll_per_token"),
                label=f"Fisher {ledger} delta NLL",
            )
        )
        parent_ledger_kl = _finite(
            parent_aggregate.get("source_to_candidate_kl_per_token"),
            label=f"parent {ledger} KL",
        )
        fisher_ledger_kl = _finite(
            fisher_aggregate.get("source_to_candidate_kl_per_token"),
            label=f"Fisher {ledger} KL",
        )
        parent_ledger_top1 = _finite(
            parent_aggregate.get("top1_agreement_to_source"),
            label=f"parent {ledger} top1",
        )
        fisher_ledger_top1 = _finite(
            fisher_aggregate.get("top1_agreement_to_source"),
            label=f"Fisher {ledger} top1",
        )
        passed = (
            fisher_delta <= parent_delta
            and fisher_ledger_kl <= parent_ledger_kl
            and fisher_ledger_top1 >= parent_ledger_top1
        )
        no_required_ledger_regression[ledger] = passed
        required_ledger_observations[ledger] = {
            "parent_absolute_delta_nll_per_token": parent_delta,
            "fisher_absolute_delta_nll_per_token": fisher_delta,
            "parent_kl_per_token": parent_ledger_kl,
            "fisher_kl_per_token": fisher_ledger_kl,
            "parent_top1": parent_ledger_top1,
            "fisher_top1": fisher_ledger_top1,
            "passed": passed,
        }

    absolute_improvement = _relative_improvement(
        parent_abs, fisher_abs, label="ordinary macro absolute delta NLL"
    )
    kl_improvement = _relative_improvement(
        parent_kl, fisher_kl, label="ordinary macro KL"
    )
    top1_gain = fisher_top1 - parent_top1
    gates = {
        "ordinary_absolute_delta_nll_materiality": absolute_improvement
        >= 0.05,
        "ordinary_kl_materiality": kl_improvement >= 0.05,
        "ordinary_top1_materiality": top1_gain >= 0.02,
        "ordinary_family_win_count": win_count >= 6,
        "worst_family_regression_floor": worst_improvement >= -0.02,
        "complete_h4_support_no_regression": no_required_ledger_regression[
            "complete_h4_support"
        ],
        "graph_core_no_regression": no_required_ledger_regression["graph_core"],
        "fisher_strictly_beats_pca_absolute_delta_nll": fisher_abs < pca_abs,
        "fisher_does_not_regress_pca_kl": fisher_kl <= pca_kl,
        "fisher_does_not_regress_pca_top1": fisher_top1 >= pca_top1,
    }
    return {
        "thresholds": dict(_MECHANISM_THRESHOLDS),
        "observations": {
            "ordinary_family_macro_absolute_delta_nll_relative_improvement": absolute_improvement,
            "ordinary_family_macro_kl_relative_improvement": kl_improvement,
            "ordinary_aggregate_top1_gain": top1_gain,
            "ordinary_family_absolute_delta_nll_win_count": win_count,
            "worst_family_absolute_delta_nll_relative_improvement": worst_improvement,
            "fisher_ordinary_family_macro_absolute_delta_nll": fisher_abs,
            "pca_ordinary_family_macro_absolute_delta_nll": pca_abs,
            "fisher_ordinary_family_macro_kl": fisher_kl,
            "pca_ordinary_family_macro_kl": pca_kl,
            "fisher_ordinary_aggregate_top1": fisher_top1,
            "pca_ordinary_aggregate_top1": pca_top1,
            "required_ledger_no_regression": required_ledger_observations,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _absolute_passed(fisher_child: Mapping[str, object]) -> bool:
    fidelity = _mapping(fisher_child.get("fidelity"), label="Fisher fidelity")
    return all(
        _v14._required_ledger_passed(ledger, fidelity.get(ledger))
        for ledger in _v14._REQUIRED_LEDGERS
    )


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
        raise RuntimeError("V16 work geometry differs")
    breakdown = {
        "fit_native_source_forwards": prompt_count,
        "fit_base_vjp_forwards": prompt_count,
        "fit_base_vjp_backward_traversals": prompt_count,
        "evaluation_native_source_forwards": prompt_count,
        "evaluation_base_forwards": prompt_count,
        "evaluation_parent_forwards": prompt_count,
        "evaluation_fisher_child_forwards": prompt_count,
        "evaluation_pca_child_forwards": prompt_count,
    }
    total_forwards = sum(
        int(value)
        for name, value in breakdown.items()
        if name != "fit_base_vjp_backward_traversals"
    )
    total_backwards = int(breakdown["fit_base_vjp_backward_traversals"])
    full_fit_count = 2 * int(full_provider_fitted)
    if (
        total_forwards != _EXPECTED_FULL_MODEL_FORWARDS
        or total_backwards != _EXPECTED_BACKWARD_VJP_TRAVERSALS
        or outer_fold_count * 3 != _EXPECTED_OUTER_PROVIDER_FITS
    ):
        raise RuntimeError("V16 exact work count differs")
    return {
        "outer_parent_fit_count": outer_fold_count,
        "expected_outer_parent_fit_count": _EXPECTED_OUTER_PARENT_FITS,
        "outer_child_fit_count": outer_fold_count * 2,
        "expected_outer_child_fit_count": _EXPECTED_OUTER_CHILD_FITS,
        "outer_provider_fit_count": outer_fold_count * 3,
        "expected_outer_provider_fit_count": _EXPECTED_OUTER_PROVIDER_FITS,
        "conditional_full_panel_provider_fit_count": full_fit_count,
        "fit_provider_count": outer_fold_count * 3 + full_fit_count,
        "expected_fit_provider_count": _EXPECTED_OUTER_PROVIDER_FITS
        + full_fit_count,
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
    resources = _v14._provider_resources(provider)
    if (
        resources["prepared_float_scalar_count"] != _EXPECTED_PARENT_SCALARS
        or resources["logical_macs_per_token_upper_bound"]
        != _EXPECTED_PARENT_MACS
    ):
        raise RuntimeError("V16 parent resource geometry differs")
    return resources


def _child_resources(
    provider: AutonomousCompleteH4FisherXYProvider,
) -> dict[str, object]:
    provider.validate_integrity()
    if (
        provider.rank != PARENT_RECIPE.rank
        or provider.conditional_rank != _CONDITIONAL_RANK
        or provider.incremental_prepared_float_scalar_count
        != _EXPECTED_CHILD_INCREMENTAL_SCALARS
        or provider.prepared_float_scalar_count != _EXPECTED_CHILD_SCALARS
        or provider.incremental_logical_macs_per_token_upper_bound
        != _EXPECTED_CHILD_INCREMENTAL_MACS
        or provider.logical_macs_per_token_upper_bound != _EXPECTED_CHILD_MACS
    ):
        raise RuntimeError("V16 child resource geometry differs")
    return {
        "scope": "incremental_fisher_square_provider_including_k256_parent",
        "prepared_float_scalar_count": provider.prepared_float_scalar_count,
        "runtime_parameter_bytes_float64": provider.prepared_float_scalar_count
        * 8,
        "logical_macs_per_token_upper_bound": (
            provider.logical_macs_per_token_upper_bound
        ),
        "incremental_child_prepared_float_scalar_count": (
            provider.incremental_prepared_float_scalar_count
        ),
        "incremental_child_runtime_parameter_bytes_float64": (
            provider.incremental_prepared_float_scalar_count * 8
        ),
        "incremental_child_logical_macs_per_token_upper_bound": (
            provider.incremental_logical_macs_per_token_upper_bound
        ),
        "retained_gemma_parameters_excluded": True,
        "base_bridge_and_full_suffix_macs_excluded": True,
        "end_to_end_model_parameter_or_flop_claim": False,
    }


def _authenticated_coordinate_diagnostics(
    provider: AutonomousCompleteH4FisherXYProvider,
) -> dict[str, object]:
    metadata = provider.metadata()
    if metadata.get("artifact_sha256") != provider.artifact_sha256:
        raise RuntimeError("coordinate metadata is not authenticated by provider")
    result = {
        "provider_artifact_sha256": provider.artifact_sha256,
        "coordinate_objective": provider.coordinate_objective,
        **{
            name: metadata.get(name)
            for name in (
                *_COORDINATE_DIAGNOSTIC_FIELDS,
                *_ADDITIONAL_FIT_DIAGNOSTIC_FIELDS,
            )
        },
    }
    _coordinate_fold_diagnostics(
        result,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_coordinate_objective=provider.coordinate_objective,
    )
    return result


def _selected_full_provider_fit_geometry(
    provider: AutonomousCompleteH4FisherXYProvider,
) -> dict[str, object]:
    diagnostic = _coordinate_fold_diagnostics(
        _authenticated_coordinate_diagnostics(provider),
        expected_artifact_sha256=provider.artifact_sha256,
        expected_coordinate_objective="reverse_vjp_fisher",
    )
    if diagnostic["passed"] is not True:
        raise RuntimeError(
            "selected full-panel provider fit geometry is insufficient"
        )
    return diagnostic


def _provider_fold_ownership_receipt(
    provider: AutonomousCompleteH4ResidualProvider | AutonomousCompleteH4FisherXYProvider,
    *,
    held_family_id: str,
    held_sequences: Sequence[AutonomousCompleteH4TrainingSequence],
) -> dict[str, object]:
    held_family = _v14._identifier(held_family_id, label="held family")
    selected_sequences = tuple(sorted(held_sequences, key=lambda value: value.artifact_sha256))
    held_sequence_sha256s = tuple(
        value.artifact_sha256 for value in selected_sequences
    )
    if (
        len(selected_sequences) != _EXPECTED_HELD_SEQUENCES_PER_FOLD
        or {value.family_id for value in selected_sequences} != {held_family}
    ):
        raise RuntimeError("V16 held sequence ownership differs")
    receipt: dict[str, object] = {
        "held_family_id": held_family,
        "provider_artifact_sha256": provider.artifact_sha256,
        "fit_family_ids": provider.fit_family_ids,
        "fit_sequence_sha256s": provider.fit_sequence_sha256s,
        "held_sequence_sha256s": held_sequence_sha256s,
    }
    if isinstance(provider, AutonomousCompleteH4FisherXYProvider):
        receipt.update(
            {
                "parent_provider_artifact_sha256": (
                    provider.parent_provider.artifact_sha256
                ),
                "coordinate_objective": provider.coordinate_objective,
            }
        )
    _fold_ownership_receipt(
        receipt,
        expected_held_family_id=held_family,
        expected_provider_artifact_sha256=provider.artifact_sha256,
        expected_parent_provider_artifact_sha256=(
            provider.parent_provider.artifact_sha256
            if isinstance(provider, AutonomousCompleteH4FisherXYProvider)
            else None
        ),
        expected_coordinate_objective=(
            provider.coordinate_objective
            if isinstance(provider, AutonomousCompleteH4FisherXYProvider)
            else None
        ),
    )
    return receipt


def _held_runtime_coordinate_diagnostics(
    provider: AutonomousCompleteH4FisherXYProvider,
    *,
    held_family_id: str,
    held_sequences: Sequence[AutonomousCompleteH4TrainingSequence],
) -> dict[str, object]:
    ownership = _provider_fold_ownership_receipt(
        provider,
        held_family_id=held_family_id,
        held_sequences=held_sequences,
    )
    selected_sequences = tuple(sorted(held_sequences, key=lambda value: value.artifact_sha256))
    coordinate_rows: list[Tensor] = []
    weight_rows: list[Tensor] = []
    sequence_receipts: list[dict[str, object]] = []
    for sequence in selected_sequences:
        coordinates = replay_autonomous_complete_h4_fisher_xy_bounded_coordinates(
            provider,
            sequence,
        )
        per_sequence = summarize_fisher_xy_bounded_coordinate_geometry(coordinates)
        coordinate_rows.append(coordinates)
        weight_rows.append(
            torch.full(
                (coordinates.shape[0],),
                1.0
                / (_EXPECTED_HELD_SEQUENCES_PER_FOLD * coordinates.shape[0]),
                dtype=torch.float64,
            )
        )
        sequence_receipts.append(
            {
                "sequence_sha256": sequence.artifact_sha256,
                "row_count": per_sequence.row_count,
                "bounded_coordinates_sha256": (
                    per_sequence.bounded_coordinates_sha256
                ),
            }
        )
    coordinates = torch.cat(coordinate_rows, dim=0).contiguous()
    weights = torch.cat(weight_rows, dim=0).contiguous()
    geometry = summarize_fisher_xy_bounded_coordinate_geometry(coordinates, weights)
    result = {
        "provider_artifact_sha256": provider.artifact_sha256,
        "parent_provider_artifact_sha256": provider.parent_provider.artifact_sha256,
        "coordinate_objective": provider.coordinate_objective,
        "held_family_id": held_family_id,
        "held_sequence_sha256s": ownership["held_sequence_sha256s"],
        "sequence_coordinate_receipts": tuple(sequence_receipts),
        "weighting_semantics": "equal_sequences_then_equal_supported_rows",
        "runtime_input_fields": _RUNTIME_COORDINATE_INPUT_FIELDS,
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
    _held_runtime_coordinate_fold_diagnostics(
        result,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_parent_artifact_sha256=provider.parent_provider.artifact_sha256,
        expected_coordinate_objective=provider.coordinate_objective,
        expected_held_family_id=held_family_id,
        expected_held_sequence_sha256s=ownership["held_sequence_sha256s"],  # type: ignore[arg-type]
    )
    return result


def _validate_parent(
    provider: AutonomousCompleteH4ResidualProvider,
    *,
    expected_fit_family_count: int,
) -> None:
    if not isinstance(provider, AutonomousCompleteH4ResidualProvider):
        raise TypeError("V16 parent must be autonomous complete-H4")
    provider.validate_integrity()
    if (
        provider.rank != PARENT_RECIPE.rank
        or provider.state_rank != PARENT_RECIPE.rank
        or provider.lag_count != PARENT_RECIPE.lag_count
        or provider.fit_objective != PARENT_RECIPE.fit_objective
        or provider.state_encoder is not None
        or not math.isclose(
            provider.ridge, PARENT_RECIPE.ridge, rel_tol=0.0, abs_tol=0.0
        )
        or len(provider.fit_family_ids) != expected_fit_family_count
    ):
        raise RuntimeError("V16 K256 parent geometry differs")
    _parent_resources(provider)


def _validate_child(
    provider: AutonomousCompleteH4FisherXYProvider,
    *,
    coordinate_objective: str,
    expected_parent_artifact_sha256: str,
    expected_fit_family_count: int,
) -> None:
    if not isinstance(provider, AutonomousCompleteH4FisherXYProvider):
        raise TypeError("V16 child must be Fisher-XY complete-H4")
    provider.validate_integrity()
    if (
        coordinate_objective not in COORDINATE_OBJECTIVES
        or provider.coordinate_objective != coordinate_objective
        or provider.parent_provider.artifact_sha256
        != expected_parent_artifact_sha256
        or provider.conditional_rank != _CONDITIONAL_RANK
        or provider.router_ridge != _ROUTER_RIDGE
        or provider.conditional_ridge != _CONDITIONAL_RIDGE
        or provider.operator_norm_bound != FISHER_XY_OPERATOR_NORM_BOUND
        or len(provider.fit_family_ids) != expected_fit_family_count
        or provider.fit_family_ids != provider.parent_provider.fit_family_ids
        or provider.fit_sequence_sha256s
        != provider.parent_provider.fit_sequence_sha256s
        or max(provider.post_projection_corner_operator_norms)
        > FISHER_XY_OPERATOR_NORM_BOUND + 1.0e-12
    ):
        raise RuntimeError("V16 conditional child geometry differs")
    _child_resources(provider)
    _authenticated_coordinate_diagnostics(provider)


def _validate_parameter_matched_children(
    fisher: AutonomousCompleteH4FisherXYProvider,
    pca: AutonomousCompleteH4FisherXYProvider,
) -> None:
    fisher_resources = _child_resources(fisher)
    pca_resources = _child_resources(pca)
    resource_keys = (
        "prepared_float_scalar_count",
        "runtime_parameter_bytes_float64",
        "logical_macs_per_token_upper_bound",
        "incremental_child_prepared_float_scalar_count",
        "incremental_child_runtime_parameter_bytes_float64",
        "incremental_child_logical_macs_per_token_upper_bound",
    )
    if (
        fisher.parent_provider.artifact_sha256
        != pca.parent_provider.artifact_sha256
        or any(fisher_resources[key] != pca_resources[key] for key in resource_keys)
    ):
        raise RuntimeError("V16 PCA control is not exactly parameter matched")


def _validate_arm_rows(
    arm_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if not isinstance(arm_rows, Mapping) or set(arm_rows) != set(_ARM_IDS):
        raise ValueError("V16 report requires exactly parent, Fisher, and PCA arms")
    result = {name: _mapping(arm_rows[name], label=f"{name} arm") for name in _ARM_IDS}
    expected_objectives = {
        PARENT_ID: "reverse_vjp_weighted_shared_parent",
        FISHER_CHILD_ID: "reverse_vjp_fisher",
        PCA_CHILD_ID: "activation_pca",
    }
    for name, row in result.items():
        if (
            row.get("arm_id") != name
            or row.get("coordinate_objective") != expected_objectives[name]
        ):
            raise ValueError("V16 arm identity differs")
        fidelity = _mapping(row.get("fidelity"), label=f"{name} fidelity")
        if set(fidelity) != set(_v14._ALL_LEDGERS):
            raise ValueError("V16 arm fidelity ledgers differ")
    fisher_resources = _mapping(
        result[FISHER_CHILD_ID].get("serving_resources"),
        label="Fisher resources",
    )
    pca_resources = _mapping(
        result[PCA_CHILD_ID].get("serving_resources"),
        label="PCA resources",
    )
    if fisher_resources != pca_resources:
        raise ValueError("V16 PCA control resources differ from Fisher child")
    return result


def _validate_cross_arm_fold_ownership(
    rows: Mapping[str, Mapping[str, object]],
    held_families: set[str],
) -> dict[str, object]:
    parent_artifacts = _mapping(
        rows[PARENT_ID].get("fold_provider_artifact_sha256s"),
        label="parent fold provider artifacts",
    )
    parent_receipts = _mapping(
        rows[PARENT_ID].get("fold_ownership_receipts"),
        label="parent fold ownership receipts",
    )
    if set(parent_artifacts) != held_families or set(parent_receipts) != held_families:
        raise ValueError("parent ownership receipts do not match outer folds")
    validated: dict[str, object] = {}
    for family in sorted(held_families):
        parent_artifact = _sha256_identifier(
            parent_artifacts[family],
            label="parent fold provider artifact",
        )
        parent_receipt = _fold_ownership_receipt(
            parent_receipts[family],
            expected_held_family_id=family,
            expected_provider_artifact_sha256=parent_artifact,
            expected_parent_provider_artifact_sha256=None,
            expected_coordinate_objective=None,
        )
        for child_id in _CHILD_IDS:
            child_receipts = _mapping(
                rows[child_id].get("fold_ownership_receipts"),
                label=f"{child_id} ownership receipts",
            )
            child_receipt = _mapping(
                child_receipts.get(family),
                label=f"{child_id} ownership receipt",
            )
            if (
                child_receipt.get("parent_provider_artifact_sha256")
                != parent_artifact
                or tuple(child_receipt.get("fit_family_ids", ()))
                != parent_receipt["fit_family_ids"]
                or tuple(child_receipt.get("fit_sequence_sha256s", ()))
                != parent_receipt["fit_sequence_sha256s"]
                or tuple(child_receipt.get("held_sequence_sha256s", ()))
                != parent_receipt["held_sequence_sha256s"]
            ):
                raise ValueError("cross-arm fold ownership receipts differ")
        validated[family] = parent_receipt
    return validated


def build_fisher_square_development_report(
    *,
    artifact_path: Path | str,
    panel: Mapping[str, object],
    bridge_binding_sha256: str,
    folds: Sequence[Mapping[str, object]],
    prerequisites: Mapping[str, object],
    fit_collection: Mapping[str, object],
    base_fidelity: Mapping[str, object],
    arm_rows: Mapping[str, Mapping[str, object]],
    candidate: Mapping[str, object] | None,
    integrity: Mapping[str, object],
) -> dict[str, object]:
    """Build the deterministic scalar-only V16 report."""

    rows = _validate_arm_rows(arm_rows)
    if len(folds) != _EXPECTED_FAMILIES:
        raise ValueError("V16 requires exactly eight outer folds")
    held_families = {
        _v14._identifier(
            _mapping(fold, label="outer fold").get("held_family_id"),
            label="held family",
        )
        for fold in folds
    }
    if len(held_families) != _EXPECTED_FAMILIES:
        raise ValueError("V16 outer folds must hold eight unique families")
    ownership_receipts = _validate_cross_arm_fold_ownership(rows, held_families)
    coordinate_geometry = evaluate_coordinate_geometry_gates(
        fisher_child=rows[FISHER_CHILD_ID],
        pca_child=rows[PCA_CHILD_ID],
    )
    coordinate_arms = _mapping(
        coordinate_geometry.get("arms"),
        label="coordinate geometry arms",
    )
    for arm_id in _CHILD_IDS:
        arm_geometry = _mapping(
            coordinate_arms.get(arm_id),
            label="coordinate geometry arm",
        )
        for section in ("fit_fold_predictability", "held_family_runtime_geometry"):
            qualification = _mapping(
                arm_geometry.get(section),
                label=f"coordinate geometry {section}",
            )
            if set(
                _mapping(
                    qualification.get("per_fold"),
                    label=f"coordinate geometry {section} folds",
                )
            ) != held_families:
                raise ValueError("coordinate geometry does not match the outer folds")
    absolute = _absolute_passed(rows[FISHER_CHILD_ID])
    mechanism = evaluate_mechanism_gates(
        parent=rows[PARENT_ID],
        fisher_child=rows[FISHER_CHILD_ID],
        pca_child=rows[PCA_CHILD_ID],
    )
    passed = (
        coordinate_geometry["passed"] is True
        and absolute
        and mechanism["passed"] is True
    )
    candidate_id = None if candidate is None else candidate.get("arm_id")
    if candidate is not None:
        candidate_artifact = _sha256_identifier(
            candidate.get("provider_artifact_sha256"),
            label="candidate provider artifact",
        )
        full_fit_geometry = _coordinate_fold_diagnostics(
            candidate.get("full_provider_fit_geometry_qualification"),
            expected_artifact_sha256=candidate_artifact,
            expected_coordinate_objective="reverse_vjp_fisher",
        )
        if full_fit_geometry["passed"] is not True:
            raise ValueError("selected full-panel provider fit geometry is insufficient")
    if (not passed) != (candidate is None) or (
        candidate is not None and candidate_id != FISHER_CHILD_ID
    ):
        raise ValueError("V16 candidate does not match absolute and mechanism gates")
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": 16,
        "scientific_status": (
            "opened_calibration_a_fixed_fisher_square_outer_lofo_development"
        ),
        "artifact": {"path": Path(artifact_path).as_posix()},
        "panel": dict(panel),
        "prerequisites": dict(prerequisites),
        "bridge_binding_sha256": _v14._identifier(
            bridge_binding_sha256, label="bridge binding"
        ),
        "fixed_protocol": {
            "recipe_grid": False,
            "parent": {
                **PARENT_RECIPE.metadata(),
                "recipe_sha256": PARENT_RECIPE.artifact_sha256,
            },
            "conditional_rank": _CONDITIONAL_RANK,
            "coordinate_count": FISHER_XY_COORDINATE_COUNT,
            "router_ridge": _ROUTER_RIDGE,
            "conditional_ridge": _CONDITIONAL_RIDGE,
            "operator_norm_bound": FISHER_XY_OPERATOR_NORM_BOUND,
            "candidate_coordinate_objective": "reverse_vjp_fisher",
            "parameter_matched_control_coordinate_objective": "activation_pca",
        },
        "outer_lofo": {
            "folds": [dict(value) for value in folds],
            "ownership_receipts": ownership_receipts,
            "decoder_parent_router_and_child_fit_inside_training_fold": True,
            "held_family_excluded_from_every_fit": True,
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
        "absolute_qualification": {
            "required_ledgers": _v14._REQUIRED_LEDGERS,
            "gates": dict(ESTABLISHED_SHADOW_FIDELITY_GATES.metadata()),
            "passed": absolute,
        },
        "coordinate_geometry_qualification": coordinate_geometry,
        "mechanism_retention": mechanism,
        "selection": {
            "rule": (
                "fisher_child_only_if_coordinate_geometry_absolute_and_"
                "preregistered_mechanism_gates_all_pass"
            ),
            "selected_arm_id": FISHER_CHILD_ID if passed else None,
            "passed": passed,
        },
        "candidate": None if candidate is None else dict(candidate),
        "integrity": dict(integrity),
        "passed": passed,
        "classification": (
            "fisher_square_oof_candidate_ready_for_fresh_protocol"
            if passed
            else (
                "fisher_square_coordinate_geometry_insufficient"
                if coordinate_geometry["passed"] is not True
                else (
                    "fisher_square_absolute_fidelity_insufficient"
                    if not absolute
                    else "fisher_square_mechanism_retention_insufficient"
                )
            )
        ),
        "success_authorizes": (
            "freeze_fisher_square_candidate_for_new_candidate_bound_protocol"
            if passed
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
    provider: AutonomousCompleteH4FisherXYProvider | None,
    provider_output: Path,
) -> dict[str, object]:
    candidate = report.get("candidate")
    if (provider is None) != (candidate is None):
        raise ValueError("V16 published provider must match the selected candidate")
    destinations = (output,) if provider is None else (output, provider_output)
    reservation = _v14._reserve_outputs(destinations)
    report_stage: Path | None = None
    provider_stage: Path | None = None
    try:
        if provider is not None:
            if not isinstance(candidate, Mapping):
                raise TypeError("V16 selected candidate must be a mapping")
            _validate_child(
                provider,
                coordinate_objective="reverse_vjp_fisher",
                expected_parent_artifact_sha256=(
                    provider.parent_provider.artifact_sha256
                ),
                expected_fit_family_count=_EXPECTED_FAMILIES,
            )
            full_fit_geometry = _selected_full_provider_fit_geometry(provider)
            if (
                candidate.get("provider_artifact_sha256")
                != provider.artifact_sha256
                or candidate.get("fit_family_ids") != provider.fit_family_ids
                or candidate.get("fit_sequence_sha256s")
                != provider.fit_sequence_sha256s
                or candidate.get("full_provider_fit_geometry_qualification")
                != full_fit_geometry
            ):
                raise ValueError("V16 candidate and provider differ")
            provider_stage = _v14._stage_torch(
                autonomous_complete_h4_fisher_xy_provider_state_dict(provider),
                provider_output,
            )
            provider_file_sha256 = _v14._file_sha256(provider_stage)
            restored = load_autonomous_complete_h4_fisher_xy_provider(
                provider_stage,
                expected_artifact_sha256=provider.artifact_sha256,
                expected_file_sha256=provider_file_sha256,
                expected_bridge_binding_sha256=provider.bridge_binding_sha256,
            )
            if restored.metadata() != provider.metadata():
                raise RuntimeError("staged V16 provider roundtrip drifted")
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
            load_autonomous_complete_h4_fisher_xy_provider(
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
    provider = _v14._fit_provider(
        sequences,  # type: ignore[arg-type]
        PARENT_RECIPE,
        bridge_binding_sha256=bridge_binding_sha256,
    )
    return provider


def _fit_child(
    sequences: Sequence[object],
    *,
    parent: AutonomousCompleteH4ResidualProvider,
    coordinate_objective: str,
) -> AutonomousCompleteH4FisherXYProvider:
    return fit_autonomous_complete_h4_fisher_xy_residual(
        sequences=sequences,  # type: ignore[arg-type]
        parent_provider=parent,
        conditional_rank=_CONDITIONAL_RANK,
        coordinate_objective=coordinate_objective,
        router_ridge=_ROUTER_RIDGE,
        conditional_ridge=_CONDITIONAL_RIDGE,
        operator_norm_bound=FISHER_XY_OPERATOR_NORM_BOUND,
        vjp_weight_floor=0.5,
        vjp_weight_ceiling=2.0,
    )


def run_gemma3_l3_l4_complete_h4_fisher_square_development(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    provider_output: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the fixed A16 V16 parent/Fisher/PCA outer-LOFO comparison."""

    destination = _validate_output(output)
    provider_destination = _validate_provider_output(
        destination.with_suffix(".provider.pt")
        if provider_output is None
        else provider_output
    )
    if provider_destination == destination:
        raise ValueError("V16 report and provider outputs must differ")
    if destination.exists():
        raise FileExistsError("refusing to overwrite V16 report")
    if provider_destination.exists():
        raise FileExistsError("refusing to overwrite V16 provider")

    # Bind the two consumed development outcomes before any model is loaded.
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
        fisher_children: dict[str, AutonomousCompleteH4FisherXYProvider] = {}
        pca_children: dict[str, AutonomousCompleteH4FisherXYProvider] = {}
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
                raise RuntimeError("V16 outer-LOFO training ownership differs")
            parent = _fit_parent(
                training,
                bridge_binding_sha256=bridge_binding,
            )
            _validate_parent(
                parent,
                expected_fit_family_count=_EXPECTED_FAMILIES - 1,
            )
            fisher = _fit_child(
                training,
                parent=parent,
                coordinate_objective="reverse_vjp_fisher",
            )
            pca = _fit_child(
                training,
                parent=parent,
                coordinate_objective="activation_pca",
            )
            _validate_child(
                fisher,
                coordinate_objective="reverse_vjp_fisher",
                expected_parent_artifact_sha256=parent.artifact_sha256,
                expected_fit_family_count=_EXPECTED_FAMILIES - 1,
            )
            _validate_child(
                pca,
                coordinate_objective="activation_pca",
                expected_parent_artifact_sha256=parent.artifact_sha256,
                expected_fit_family_count=_EXPECTED_FAMILIES - 1,
            )
            _validate_parameter_matched_children(fisher, pca)
            parents[held] = parent
            fisher_children[held] = fisher
            pca_children[held] = pca
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
                raise RuntimeError("V16 shadow retokenization drifted")
            source_logits, native_h4, native_positions, native_valid = (
                _v14._native_boundary(context.adapter, model_inputs)
            )
            base = context.bridge.execute(context.adapter, model_inputs)
            if (
                not isinstance(base, Gemma3L3L4OnePassExecution)
                or not _v14._bitwise_equal(
                    native_positions,
                    base.prefix.logical_positions,
                )
                or not _v14._bitwise_equal(
                    native_valid,
                    base.prefix.valid_target_mask,
                )
            ):
                raise RuntimeError("V16 shadow base sequence differs")
            _v14._add_shadow_rows(
                accumulators["base"],
                record=record,
                source_logits=source_logits,
                candidate_logits=base.logits,
                supervised_indices=supervised_indices,
                supervised_targets=supervised_targets,
            )
            held = record.sequence.family_id
            providers = (
                (PARENT_ID, parents[held]),
                (FISHER_CHILD_ID, fisher_children[held]),
                (PCA_CHILD_ID, pca_children[held]),
            )
            support = base.prefix.complete_h4_causal_support_mask()
            for arm, provider in providers:
                if held in set(provider.fit_family_ids):
                    raise RuntimeError("held family leaked into V16 provider")
                candidate = context.bridge.execute(
                    context.adapter,
                    model_inputs,
                    h4_head=provider,
                )
                if (
                    candidate.h4_head_sha256 != provider.artifact_sha256
                    or candidate.prefix.artifact_sha256
                    != base.prefix.artifact_sha256
                    or not _v14._bitwise_equal(
                        candidate.candidate_h4[
                            ~support.to(candidate.candidate_h4.device)
                        ],
                        base.candidate_h4[
                            ~support.to(base.candidate_h4.device)
                        ],
                    )
                ):
                    raise RuntimeError("V16 provider escaped causal support")
                _v14._add_shadow_rows(
                    accumulators[arm],
                    record=record,
                    source_logits=source_logits,
                    candidate_logits=candidate.logits,
                    supervised_indices=supervised_indices,
                    supervised_targets=supervised_targets,
                )
                causal_checks += 1
                del candidate
            del model_inputs, source_logits, native_h4, base

        fidelity = {
            arm: {
                ledger: accumulator.finalize()
                for ledger, accumulator in ledgers.items()
            }
            for arm, ledgers in accumulators.items()
        }
        parent_resource_rows = [_parent_resources(value) for value in parents.values()]
        fisher_resource_rows = [
            _child_resources(value) for value in fisher_children.values()
        ]
        pca_resource_rows = [_child_resources(value) for value in pca_children.values()]
        if (
            len({tuple(sorted(value.items())) for value in parent_resource_rows}) != 1
            or len({tuple(sorted(value.items())) for value in fisher_resource_rows}) != 1
            or len({tuple(sorted(value.items())) for value in pca_resource_rows}) != 1
            or fisher_resource_rows[0] != pca_resource_rows[0]
        ):
            raise RuntimeError("V16 outer-fold resource geometry differs")

        arm_rows: dict[str, Mapping[str, object]] = {
            PARENT_ID: {
                "arm_id": PARENT_ID,
                "coordinate_objective": "reverse_vjp_weighted_shared_parent",
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
            },
            FISHER_CHILD_ID: {
                "arm_id": FISHER_CHILD_ID,
                "coordinate_objective": "reverse_vjp_fisher",
                "conditional_rank": _CONDITIONAL_RANK,
                "coordinate_count": FISHER_XY_COORDINATE_COUNT,
                "outer_fold_count": len(fisher_children),
                "every_fold_fit_family_count": _EXPECTED_FAMILIES - 1,
                "fold_provider_artifact_sha256s": {
                    held: provider.artifact_sha256
                    for held, provider in sorted(fisher_children.items())
                },
                "fold_parent_provider_artifact_sha256s": {
                    held: provider.parent_provider.artifact_sha256
                    for held, provider in sorted(fisher_children.items())
                },
                "fold_ownership_receipts": {
                    held: _provider_fold_ownership_receipt(
                        provider,
                        held_family_id=held,
                        held_sequences=held_sequences_by_family[held],
                    )
                    for held, provider in sorted(fisher_children.items())
                },
                "fit_diagnostics": {
                    held: _authenticated_coordinate_diagnostics(provider)
                    for held, provider in sorted(fisher_children.items())
                },
                "held_runtime_coordinate_diagnostics": {
                    held: _held_runtime_coordinate_diagnostics(
                        provider,
                        held_family_id=held,
                        held_sequences=held_sequences_by_family[held],
                    )
                    for held, provider in sorted(fisher_children.items())
                },
                "serving_resources": fisher_resource_rows[0],
                "fidelity": fidelity[FISHER_CHILD_ID],
            },
            PCA_CHILD_ID: {
                "arm_id": PCA_CHILD_ID,
                "coordinate_objective": "activation_pca",
                "conditional_rank": _CONDITIONAL_RANK,
                "coordinate_count": FISHER_XY_COORDINATE_COUNT,
                "parameter_matched_to": FISHER_CHILD_ID,
                "outer_fold_count": len(pca_children),
                "every_fold_fit_family_count": _EXPECTED_FAMILIES - 1,
                "fold_provider_artifact_sha256s": {
                    held: provider.artifact_sha256
                    for held, provider in sorted(pca_children.items())
                },
                "fold_parent_provider_artifact_sha256s": {
                    held: provider.parent_provider.artifact_sha256
                    for held, provider in sorted(pca_children.items())
                },
                "fold_ownership_receipts": {
                    held: _provider_fold_ownership_receipt(
                        provider,
                        held_family_id=held,
                        held_sequences=held_sequences_by_family[held],
                    )
                    for held, provider in sorted(pca_children.items())
                },
                "fit_diagnostics": {
                    held: _authenticated_coordinate_diagnostics(provider)
                    for held, provider in sorted(pca_children.items())
                },
                "held_runtime_coordinate_diagnostics": {
                    held: _held_runtime_coordinate_diagnostics(
                        provider,
                        held_family_id=held,
                        held_sequences=held_sequences_by_family[held],
                    )
                    for held, provider in sorted(pca_children.items())
                },
                "serving_resources": pca_resource_rows[0],
                "fidelity": fidelity[PCA_CHILD_ID],
            },
        }
        coordinate_geometry = evaluate_coordinate_geometry_gates(
            fisher_child=arm_rows[FISHER_CHILD_ID],
            pca_child=arm_rows[PCA_CHILD_ID],
        )
        absolute = _absolute_passed(arm_rows[FISHER_CHILD_ID])
        mechanism = evaluate_mechanism_gates(
            parent=arm_rows[PARENT_ID],
            fisher_child=arm_rows[FISHER_CHILD_ID],
            pca_child=arm_rows[PCA_CHILD_ID],
        )
        selected = (
            coordinate_geometry["passed"] is True
            and absolute
            and mechanism["passed"] is True
        )
        full_provider: AutonomousCompleteH4FisherXYProvider | None = None
        full_provider_fit_geometry: dict[str, object] | None = None
        if selected:
            all_sequences = tuple(record.sequence for record in records)
            full_parent = _fit_parent(
                all_sequences,
                bridge_binding_sha256=bridge_binding,
            )
            _validate_parent(
                full_parent,
                expected_fit_family_count=_EXPECTED_FAMILIES,
            )
            full_provider = _fit_child(
                all_sequences,
                parent=full_parent,
                coordinate_objective="reverse_vjp_fisher",
            )
            _validate_child(
                full_provider,
                coordinate_objective="reverse_vjp_fisher",
                expected_parent_artifact_sha256=full_parent.artifact_sha256,
                expected_fit_family_count=_EXPECTED_FAMILIES,
            )
            full_provider_fit_geometry = _selected_full_provider_fit_geometry(
                full_provider
            )
        candidate = (
            None
            if full_provider is None
            else {
                "arm_id": FISHER_CHILD_ID,
                "provider_artifact_sha256": full_provider.artifact_sha256,
                "parent_provider_artifact_sha256": (
                    full_provider.parent_provider.artifact_sha256
                ),
                "provider": full_provider.metadata(),
                "serving_resources": _child_resources(full_provider),
                "fit_family_count": _EXPECTED_FAMILIES,
                "fit_family_ids": full_provider.fit_family_ids,
                "fit_sequence_sha256s": full_provider.fit_sequence_sha256s,
                "full_provider_fit_geometry_qualification": (
                    full_provider_fit_geometry
                ),
                "native_h4_logits_targets_gradients_or_coordinate_axes_required_at_runtime": False,
            }
        )
        context.validate_immutable_inputs()
        work = _work_accounting(
            prompt_count=len(records),
            outer_fold_count=len(folds),
            full_provider_fitted=full_provider is not None,
        )
        integrity = {
            "outer_fold_count": len(folds),
            "ledger_coverage": ledger_coverage,
            **work,
            "outer_fold_ownership_receipt_count": len(folds) * len(_ARM_IDS),
            "expected_outer_fold_ownership_receipt_count": (
                _EXPECTED_FAMILIES * len(_ARM_IDS)
            ),
            "held_runtime_coordinate_diagnostic_count": (
                len(fisher_children) + len(pca_children)
            ),
            "expected_held_runtime_coordinate_diagnostic_count": (
                _EXPECTED_FAMILIES * len(_CHILD_IDS)
            ),
            "parameter_matched_child_checks": len(folds),
            "expected_parameter_matched_child_checks": _EXPECTED_FAMILIES,
            "causal_off_support_execution_checks": causal_checks,
            "expected_causal_off_support_execution_checks": _EXPECTED_CAUSAL_CHECKS,
            "source_native_data_entered_serving_provider": False,
            "full_provider_fit_was_conditional_on_coordinate_absolute_and_mechanism_pass": True,
            "full_provider_fit_geometry_check_count": int(
                full_provider_fit_geometry is not None
            ),
            "full_provider_fit_geometry_required_before_candidate_and_publication": True,
            "guard_opened": False,
            "calibration_b_opened": False,
        }
        if (
            len(parents) != _EXPECTED_OUTER_PARENT_FITS
            or len(fisher_children) + len(pca_children)
            != _EXPECTED_OUTER_CHILD_FITS
            or causal_checks != _EXPECTED_CAUSAL_CHECKS
            or len(held_sequences_by_family) != _EXPECTED_FAMILIES
            or len(fisher_children) + len(pca_children)
            != _EXPECTED_FAMILIES * len(_CHILD_IDS)
            or work["full_model_forward_count"] != _EXPECTED_FULL_MODEL_FORWARDS
        ):
            raise RuntimeError("V16 exact execution geometry differs")
        report = build_fisher_square_development_report(
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
                "coordinate_axes_fit_only_and_not_serialized": True,
                "conditional_runtime_provider_tensor_sidecar": (
                    full_provider is not None
                ),
            },
            base_fidelity=fidelity["base"],
            arm_rows=arm_rows,
            candidate=candidate,
            integrity=integrity,
        )
        return _publish(
            report,
            output=destination,
            provider=full_provider,
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
    report = run_gemma3_l3_l4_complete_h4_fisher_square_development(
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
