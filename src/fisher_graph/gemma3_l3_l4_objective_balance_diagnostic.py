"""Run the fit-only Gemma L3-to-L4 objective-balance diagnostic.

This additive diagnostic replays the authenticated C2 calibration and measures
only the C2 fit panel.  It never materializes the consumed C2 selection panel.
All candidates use the same rank-16 architecture and cold-start primary seed;
the only changes are the preregistered objective-balance treatments.

The raw Fisher-gauge D0 control always runs.  D1-D3 then run in order.  Each
primary treatment that passes the complete fit-side gates is replicated from
a second seed; a failed replication advances to the next treatment.  The
schedule stops only at the first two-seed pass or after exhausting D3.  The
result is diagnostic evidence about optimization, not held-out selection or a
deployable compression claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol

import torch
from torch import Tensor

from .adapters import module_state_fingerprint
from .contrast_objective_balancing import (
    UnitRmsFisherGauge,
    audit_objective_contributions,
)
from .external_models import find_git_worktree
from .gemma3_experiment import DEFAULT_MODEL_ID
from .gemma3_l3_l4_basis_package import DEFAULT_BASIS_PACKAGE
from . import gemma3_l3_l4_contrast_provider_development as c2
from .gemma3_l3_l4_contrast_provider_development_protocol import (
    DEFAULT_DEVELOPMENT_FIT_PANEL_SHA256,
    DEFAULT_DEVELOPMENT_PILOT_PANEL_SHA256,
    DEFAULT_DEVELOPMENT_PROTOCOL_SHA256,
    ContrastProviderDevelopmentProtocol,
    DevelopmentCalibrationBinding,
    default_contrast_provider_development_protocol,
    select_global_calibration_amplitude,
)
from .gemma3_l3_l4_reference_provider_experiment import (
    DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    _deferred_collision_gates,
    _fisher_metric_weight,
    _load_live_dependencies,
)
from .gemma3_l3_l4_objective_balance_diagnostic_protocol import (
    DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256,
    DIAGNOSTIC_EXECUTION_DEVICE,
    DIAGNOSTIC_EXECUTION_DTYPE,
    ObjectiveBalanceDiagnosticGates,
    ObjectiveBalanceDiagnosticProtocol,
    ObjectiveBalanceRecipe,
    default_objective_balance_diagnostic_protocol,
    family_balance_copy_binding_sha256,
    family_balance_copy_id,
)
from .gemma3_l3_l4_spectral_mapping_experiment import DEFAULT_REVISION
from .gemma3_l3_l4_synthetic_reference_protocol import (
    SyntheticReferenceGates,
)
from .state_conditioned_contrast_assessment import (
    ContrastAssessmentGates,
    ContrastAssessmentResult,
    ContrastDefinition,
    ContrastObservation,
    assess_state_conditioned_contrasts,
)
from .state_conditioned_contrast_fit import (
    ContrastAwareObjective,
    ContrastAwareReferenceProviderPlan,
    ReferenceProviderContrastPair,
    fit_contrast_aware_reference_provider,
)
from . import state_conditioned_reference_selection as reference_selection
from .state_conditioned_reference_selection import (
    FullWidthCandidatePrediction,
    FullWidthCandidateScore,
    FullWidthReferenceCandidate,
    FullWidthReferenceControls,
    FullWidthReferenceProbe,
    fit_full_width_reference_controls,
    full_width_reference_gates_sha256,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "LoadedObjectiveBalanceDiagnosticArtifact",
    "build_parser",
    "describe_objective_balance_diagnostic",
    "load_objective_balance_diagnostic_artifact",
    "main",
    "run_objective_balance_diagnostic",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-objective-balance-diagnostic-d0-d3.pt"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_objective_balance_diagnostic.d0_d3"
_FORMAT_VERSION = 1
_MODAL_WIDTH = 64
_LATENT_RANK = 16
_EXPECTED_C2_CALIBRATION_AMPLITUDE = 8.0
_EXPECTED_C2_CALIBRATION_SHA256 = (
    "aedb23de65ed6a37d645539001311ddb415cd2713400777dac448cb96bd5bfa8"
)
_ALLOWED_MEASUREMENT_ROLES = frozenset({"pilot", "fit"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_DOMAIN = b"fisher-graph:objective-balance-diagnostic:tensor:v1\0"
_BINDING_DOMAIN = b"fisher-graph:objective-balance-diagnostic:binding:v1\0"
_GAUGE_DOMAIN = b"fisher-graph:objective-balance-diagnostic:gauge:v1\0"
_CODE_DOMAIN = b"fisher-graph:objective-balance-diagnostic:code:v1\0"
_ARTIFACT_DOMAIN = b"fisher-graph:objective-balance-diagnostic:artifact:v1\0"
_REPORT_DOMAIN = b"fisher-graph:objective-balance-diagnostic:report:v1\0"
_CODE_FILES = (
    "adapters/__init__.py",
    "adapters/base.py",
    "adapters/gemma3.py",
    "contrast_objective_balancing.py",
    "external_models.py",
    "gated_executor.py",
    "gemma3_experiment.py",
    "gemma3_l3_l4_basis_package.py",
    "gemma3_l3_l4_contrast_provider_development.py",
    "gemma3_l3_l4_contrast_provider_development_materialization.py",
    "gemma3_l3_l4_contrast_provider_development_protocol.py",
    "gemma3_l3_l4_manifold_lift.py",
    "gemma3_l3_l4_objective_balance_diagnostic.py",
    "gemma3_l3_l4_objective_balance_diagnostic_protocol.py",
    "gemma3_l3_l4_reference_provider_experiment.py",
    "gemma3_l3_l4_spectral_mapping_experiment.py",
    "gemma3_l3_l4_synthetic_materialization.py",
    "gemma3_l3_l4_synthetic_reference_protocol.py",
    "state_conditioned_contrast_assessment.py",
    "state_conditioned_contrast_fit.py",
    "state_conditioned_reference_provider.py",
    "state_conditioned_reference_selection.py",
)


class _RecipeLike(Protocol):
    recipe_id: str
    training_metric: str
    signed_pair_multiplicity: int
    direction_weight: float
    steps: int
    learning_rate: float
    advancement_eligible: bool
    artifact_sha256: str

    def state_dict(self) -> dict[str, object]: ...


class _DiagnosticProtocolLike(Protocol):
    protocol_sha256: str
    primary_seed: int
    replication_seed: int
    recipes: tuple[_RecipeLike, ...]

    def state_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class _RecipeEvaluation:
    recipe_id: str
    seed_role: str
    seed: int
    combined_pass: bool
    row: dict[str, object]
    plan: ContrastAwareReferenceProviderPlan


@dataclass(frozen=True, slots=True)
class LoadedObjectiveBalanceDiagnosticArtifact:
    """Authenticated views of one published fit-only diagnostic."""

    state: Mapping[str, object]
    report: Mapping[str, object]
    manifest: Mapping[str, object]
    artifact_sha256: str
    tensor_file_sha256: str
    report_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {
            "dtype": str(canonical.dtype),
            "shape": tuple(int(width) for width in canonical.shape),
        }
    )
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + header
        + b"\0"
        + canonical.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256s() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    result = {
        name: _file_sha256(root / name)
        for name in _CODE_FILES
    }
    if set(result) != set(_CODE_FILES):
        raise RuntimeError("objective-balance code manifest is incomplete")
    return result


def _code_bundle_sha256(values: Mapping[str, str]) -> str:
    if set(values) != set(_CODE_FILES):
        raise ValueError("objective-balance code manifest is incomplete")
    for name, value in values.items():
        _require_sha256(value, label=f"code digest {name}")
    return _json_sha256(dict(values), domain=_CODE_DOMAIN)


def _measure_c2_role(
    *,
    role: str,
    **kwargs: object,
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Measure only an explicitly permitted consumed-C2 role."""

    if role not in _ALLOWED_MEASUREMENT_ROLES:
        raise PermissionError(
            "objective-balance diagnostic forbids C2 selection materialization"
        )
    measured, report = c2._measure_role(role=role, **kwargs)  # type: ignore[arg-type]
    return measured, report


def _fit_contrast_assessment(
    *,
    protocol: ContrastProviderDevelopmentProtocol,
    measured: Sequence[object],
    predictions: Mapping[str, Tensor],
    metric_weight: Tensor,
    gates: ContrastAssessmentGates,
    required_null_candidate_pass_count: int,
) -> tuple[
    ContrastAssessmentResult,
    dict[str, dict[str, str]],
    dict[str, object],
]:
    """Assess every consumed-C2 fit contrast under the canonical raw metric."""

    measured_by_id = {
        value.probe.probe_id: value for value in measured  # type: ignore[attr-defined]
    }
    if set(predictions) != set(measured_by_id):
        raise ValueError("contrast predictions do not cover the fit panel")
    observations: list[ContrastObservation] = []
    identities: dict[str, dict[str, str]] = {}
    for group in protocol.groups_for_role("fit"):
        for index, (left_id, right_id) in enumerate(
            group.canonical_variant_pairs
        ):
            contrast_id = f"{group.group_id}.pair.{index:02d}"
            left = measured_by_id[left_id]
            right = measured_by_id[right_id]
            teacher = (
                left.target_replays[0]  # type: ignore[attr-defined]
                * metric_weight.view(1, 1, -1),
                right.target_replays[0]  # type: ignore[attr-defined]
                * metric_weight.view(1, 1, -1),
            )
            repeated_teacher = (
                left.target_replays[1]  # type: ignore[attr-defined]
                * metric_weight.view(1, 1, -1),
                right.target_replays[1]  # type: ignore[attr-defined]
                * metric_weight.view(1, 1, -1),
            )
            candidate = (
                predictions[left_id] * metric_weight.view(1, 1, -1),
                predictions[right_id] * metric_weight.view(1, 1, -1),
            )
            observations.append(
                ContrastObservation(
                    definition=ContrastDefinition(
                        contrast_id=contrast_id,
                        family=group.family,
                        role=(
                            "expected_sensitivity"
                            if group.intent == "sensitivity"
                            else "intended_null"
                        ),
                        coefficients=(-1.0, 1.0),
                    ),
                    teacher_endpoints=teacher,
                    repeated_teacher_endpoints=repeated_teacher,
                    candidate_endpoints=candidate,
                    repeated_candidate_endpoints=tuple(
                        value.clone() for value in candidate
                    ),
                )
            )
            identities[contrast_id] = {
                "group_id": group.group_id,
                "family": group.family,
                "intent": group.intent,
                "rank_band": group.rank_band,
                "left_probe_id": left_id,
                "right_probe_id": right_id,
            }
    result = assess_state_conditioned_contrasts(
        observations,
        gates=gates,
    )
    scores = {value.contrast_id: value for value in result.contrast_scores}
    if set(scores) != set(identities):
        raise RuntimeError("fit contrast score identities drifted")
    family_coverage: dict[str, dict[str, object]] = {}
    for family, intent in (
        ("radial_sensitivity", "sensitivity"),
        ("signed_sensitivity", "sensitivity"),
        ("null_invariance", "invariance"),
    ):
        ids = tuple(
            key
            for key, identity in identities.items()
            if identity["family"] == family
        )
        qualified_status = (
            "eligible_sensitivity"
            if intent == "sensitivity"
            else "valid_intended_null"
        )
        qualified = tuple(
            key
            for key in ids
            if scores[key].teacher_status == qualified_status
        )
        candidate_passes = tuple(
            key
            for key in qualified
            if scores[key].decision_status == "pass"
        )
        bands = tuple(
            sorted(
                {
                    identities[key]["rank_band"]
                    for key in qualified
                }
            )
        )
        family_coverage[family] = {
            "intent": intent,
            "planned_contrast_count": len(ids),
            "teacher_qualified_contrast_count": len(qualified),
            "candidate_pass_count": len(candidate_passes),
            "every_teacher_qualified_contrast_passed": (
                len(candidate_passes) == len(qualified)
            ),
            "qualified_rank_bands": bands,
            "all_four_rank_bands_covered": len(bands) == 4,
        }
    coverage = {
        "family_coverage": family_coverage,
        "all_families_cover_all_four_rank_bands": all(
            bool(value["all_four_rank_bands_covered"])
            for value in family_coverage.values()
        ),
        "every_teacher_qualified_contrast_passed": all(
            bool(value["every_teacher_qualified_contrast_passed"])
            for value in family_coverage.values()
        ),
        "required_null_contrasts_valid_and_passed": (
            family_coverage["null_invariance"][
                "teacher_qualified_contrast_count"
            ]
            == required_null_candidate_pass_count
            and family_coverage["null_invariance"][
                "candidate_pass_count"
            ]
            == required_null_candidate_pass_count
        ),
        "required_null_candidate_pass_count": (
            required_null_candidate_pass_count
        ),
    }
    return result, identities, coverage


def _candidate_failure_reasons(
    *,
    ordinary_score: FullWidthCandidateScore,
    contrast_result: ContrastAssessmentResult,
    coverage: Mapping[str, object],
    balance_gate_passed: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not ordinary_score.passed:
        reasons.extend(
            f"ordinary:{name}"
            for name, passed in ordinary_score.gate_flags.state_dict().items()
            if name != "all_passed" and passed is False
        )
    if contrast_result.overall_status != "pass":
        reasons.append(f"contrast:{contrast_result.overall_status}")
        reasons.extend(
            f"contrast:{value}" for value in contrast_result.reason_codes
        )
    if not bool(coverage["all_families_cover_all_four_rank_bands"]):
        reasons.append("contrast:teacher_coverage_missing_rank_band")
    if not bool(coverage["every_teacher_qualified_contrast_passed"]):
        reasons.append("contrast:not_every_qualified_contrast_passed")
    if not bool(coverage["required_null_contrasts_valid_and_passed"]):
        reasons.append("contrast:required_null_count_not_passed")
    if not balance_gate_passed:
        reasons.append("objective:contribution_balance_gate_failed")
    return tuple(sorted(set(reasons)))


def _fit_teacher_weighted_energy(
    measured: Sequence[object],
    *,
    target_center: Tensor,
    target_scale: Tensor,
    metric_weight: Tensor,
) -> float:
    squared = 0.0
    count = 0
    for value in measured:
        target = (
            value.target_modes  # type: ignore[attr-defined]
            - target_center.view(1, 1, -1)
        ) / target_scale.view(1, 1, -1)
        weighted = target * metric_weight.view(1, 1, -1)
        mask = value.valid_mask  # type: ignore[attr-defined]
        squared += float(weighted[mask].square().sum())
        count += int(mask.sum()) * _MODAL_WIDTH
    if count <= 0:
        raise ValueError("fit panel has no valid teacher scalars")
    energy = squared / count
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("fit teacher weighted energy is invalid")
    return energy


def _standardized_gauge_sha256(
    *,
    basis_payload_sha256: str,
    source_model_sha256: str,
    c2_protocol_sha256: str,
    calibration_sha256: str,
    target_center: Tensor,
    target_scale: Tensor,
    canonical_metric_weight: Tensor,
) -> str:
    return _json_sha256(
        {
            "basis_payload_sha256": basis_payload_sha256,
            "source_model_sha256": source_model_sha256,
            "c2_protocol_sha256": c2_protocol_sha256,
            "calibration_sha256": calibration_sha256,
            "target_center_sha256": _tensor_sha256(target_center),
            "target_scale_sha256": _tensor_sha256(target_scale),
            "canonical_metric_weight_sha256": _tensor_sha256(
                canonical_metric_weight
            ),
            "metric_semantics": (
                "canonical_raw_l4_sqrt_fisher_metric_for_all_fit_scoring"
            ),
        },
        domain=_GAUGE_DOMAIN,
    )


def _provider_binding_sha256(
    *,
    diagnostic_protocol_sha256: str,
    recipe: _RecipeLike,
    c2_protocol_sha256: str,
    calibration_sha256: str,
    basis_payload_sha256: str,
    source_model_sha256: str,
    norm_sha256: str,
    training_metric_weight: Tensor,
    target_center: Tensor,
    target_scale: Tensor,
) -> str:
    return _json_sha256(
        {
            "schema": "fisher_graph.objective_balance_provider_binding.v1",
            "diagnostic_protocol_sha256": diagnostic_protocol_sha256,
            "recipe_sha256": recipe.artifact_sha256,
            "c2_protocol_sha256": c2_protocol_sha256,
            "calibration_sha256": calibration_sha256,
            "basis_payload_sha256": basis_payload_sha256,
            "source_model_sha256": source_model_sha256,
            "norm_sha256": norm_sha256,
            "training_metric_weight_sha256": _tensor_sha256(
                training_metric_weight
            ),
            "target_center_sha256": _tensor_sha256(target_center),
            "target_scale_sha256": _tensor_sha256(target_scale),
            "latent_rank": _LATENT_RANK,
            "visible_source_modes": _MODAL_WIDTH,
            "visible_target_modes": _MODAL_WIDTH,
        },
        domain=_BINDING_DOMAIN,
    )


def _fit_data_binding_sha256(
    *,
    basis: object,
    c2_protocol: ContrastProviderDevelopmentProtocol,
    calibration: DevelopmentCalibrationBinding,
    norm_sha256: str,
    canonical_metric_weight: Tensor,
) -> str:
    """Return the exact recipe-independent C2 rank-16 fit-data binding."""

    return c2._provider_binding_sha256(  # type: ignore[arg-type]
        basis=basis,
        protocol=c2_protocol,
        calibration=calibration,
        objective=c2._objective(),
        norm_sha256=norm_sha256,
        metric_weight=canonical_metric_weight,
    )


_PAIR_TENSOR_FIELDS = (
    "teacher_midpoint_jvp",
    "provider_chart_modal_primal",
    "provider_chart_null_primal",
    "provider_chart_row_rms_primal",
    "provider_chart_modal_tangent",
    "provider_chart_null_tangent",
    "provider_chart_row_rms_tangent",
)


def _clone_contrast_pair(
    pair: ReferenceProviderContrastPair,
    *,
    pair_id: str,
) -> ReferenceProviderContrastPair:
    return ReferenceProviderContrastPair(
        pair_id=pair_id,
        family=pair.family,
        role=pair.role,
        left_endpoint_id=pair.left_endpoint_id,
        right_endpoint_id=pair.right_endpoint_id,
        rank_stratum=pair.rank_stratum,
        **{
            name: getattr(pair, name)
            for name in _PAIR_TENSOR_FIELDS
        },
    )


def _balance_training_pairs(
    pairs: Sequence[ReferenceProviderContrastPair],
    *,
    recipe: _RecipeLike,
) -> tuple[
    tuple[ReferenceProviderContrastPair, ...],
    dict[str, object],
]:
    natural = tuple(sorted(pairs, key=lambda value: value.pair_id))
    if len({value.pair_id for value in natural}) != len(natural):
        raise ValueError("natural fit pair ids must be unique")
    family_counts = {
        family: sum(value.family == family for value in natural)
        for family in (
            "radial_sensitivity",
            "signed_sensitivity",
            "null_invariance",
        )
    }
    if family_counts != {
        "radial_sensitivity": 16,
        "signed_sensitivity": 8,
        "null_invariance": 24,
    }:
        raise ValueError("natural C2 fit pair family counts drifted")
    multiplicity = int(recipe.signed_pair_multiplicity)
    if multiplicity not in (1, 2):
        raise ValueError("signed pair multiplicity must be one or two")
    duplicates: list[ReferenceProviderContrastPair] = []
    duplicate_bindings: list[dict[str, str]] = []
    if multiplicity == 2:
        for pair in natural:
            if pair.family != "signed_sensitivity":
                continue
            copy_binding = family_balance_copy_binding_sha256(
                pair.pair_id,
                pair.artifact_sha256,
            )
            duplicate = _clone_contrast_pair(
                pair,
                pair_id=family_balance_copy_id(
                    pair.pair_id,
                    pair.artifact_sha256,
                ),
            )
            duplicates.append(duplicate)
            duplicate_bindings.append(
                {
                    "source_pair_id": pair.pair_id,
                    "source_pair_sha256": pair.artifact_sha256,
                    "copy_binding_sha256": copy_binding,
                    "duplicate_pair_id": duplicate.pair_id,
                    "duplicate_pair_sha256": duplicate.artifact_sha256,
                }
            )
    balanced = tuple(
        sorted((*natural, *duplicates), key=lambda value: value.pair_id)
    )
    balanced_family_counts = {
        family: sum(value.family == family for value in balanced)
        for family in family_counts
    }
    expected_signed = 8 * multiplicity
    if balanced_family_counts != {
        "radial_sensitivity": 16,
        "signed_sensitivity": expected_signed,
        "null_invariance": 24,
    }:
        raise RuntimeError("balanced fit pair family counts are invalid")
    return balanced, {
        "semantics": (
            "natural_pair_mean"
            if multiplicity == 1
            else "equal_radial_signed_mass_by_authenticated_signed_duplicate"
        ),
        "natural_family_counts": family_counts,
        "balanced_family_counts": balanced_family_counts,
        "signed_pair_multiplicity": multiplicity,
        "duplicate_bindings": tuple(duplicate_bindings),
        "natural_pair_sha256s": tuple(
            value.artifact_sha256 for value in natural
        ),
        "balanced_pair_sha256s": tuple(
            value.artifact_sha256 for value in balanced
        ),
    }


def _teacher_signal_diagnostics(
    measured: Sequence[object],
    pairs: Sequence[ReferenceProviderContrastPair],
    *,
    target_center: Tensor,
    target_scale: Tensor,
    metric_weight: Tensor,
) -> dict[str, object]:
    measured_by_id = {
        value.probe.probe_id: value for value in measured  # type: ignore[attr-defined]
    }
    delta_mses: list[float] = []
    jvp_mses: list[float] = []
    sensitivity_pair_ids: list[str] = []
    jvp_pair_ids: list[str] = []
    for pair in pairs:
        if pair.role != "expected_sensitivity":
            continue
        left = measured_by_id[pair.left_endpoint_id]
        right = measured_by_id[pair.right_endpoint_id]
        left_target = (
            left.target_modes  # type: ignore[attr-defined]
            - target_center.view(1, 1, -1)
        ) / target_scale.view(1, 1, -1)
        right_target = (
            right.target_modes  # type: ignore[attr-defined]
            - target_center.view(1, 1, -1)
        ) / target_scale.view(1, 1, -1)
        mask = left.valid_mask  # type: ignore[attr-defined]
        if not torch.equal(mask, right.valid_mask):  # type: ignore[attr-defined]
            raise ValueError("teacher signal pair masks differ")
        delta = (
            right_target - left_target
        ) * metric_weight.view(1, 1, -1)
        delta_mses.append(float(delta[mask].square().mean()))
        sensitivity_pair_ids.append(pair.pair_id)
        if pair.teacher_midpoint_jvp is not None:
            jvp = (
                pair.teacher_midpoint_jvp
                / target_scale.view(1, -1)
                * metric_weight.view(1, -1)
            )
            jvp_mses.append(float(jvp.square().mean()))
            jvp_pair_ids.append(pair.pair_id)
    if not delta_mses or not jvp_mses:
        raise ValueError("fit teacher signal diagnostics are empty")
    if len(delta_mses) != 24 or len(jvp_mses) != 24:
        raise ValueError("natural C2 fit sensitivity/JVP counts drifted")
    return {
        "sensitivity_pair_count": len(delta_mses),
        "jvp_pair_count": len(jvp_mses),
        "minimum_teacher_delta_mse": min(delta_mses),
        "maximum_teacher_delta_mse": max(delta_mses),
        "minimum_teacher_jvp_mse": min(jvp_mses),
        "maximum_teacher_jvp_mse": max(jvp_mses),
        "sensitivity_pair_ids_sha256": _json_sha256(
            tuple(sensitivity_pair_ids),
            domain=_BINDING_DOMAIN,
        ),
        "jvp_pair_ids_sha256": _json_sha256(
            tuple(jvp_pair_ids),
            domain=_BINDING_DOMAIN,
        ),
        "metric_weight_sha256": _tensor_sha256(metric_weight),
        "raw_teacher_delta_or_jvp_tensors_published": False,
    }


def _validate_output_path(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("diagnostic output must use a .pt suffix")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite diagnostic output")
    worktree = find_git_worktree(Path(__file__))
    resolved = destination.expanduser().resolve()
    if worktree is not None:
        root = worktree.resolve()
        if resolved == root or root in resolved.parents:
            relative = resolved.relative_to(root)
            if not relative.parts or relative.parts[0] not in {
                ".local-runs",
                "local-runs",
            }:
                raise ValueError(
                    "worktree outputs must remain under ignored local-runs"
                )
    return resolved


def _stage_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _assert_tensor_free_report(value: object, *, path: str = "report") -> None:
    if isinstance(value, Tensor):
        raise ValueError(f"{path} must not contain tensors")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_tensor_free_report(
                nested,
                path=f"{path}.{key}",
            )
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_tensor_free_report(
                nested,
                path=f"{path}[{index}]",
            )


def _assert_safe_artifact_tree(value: object, *, path: str = "state") -> None:
    forbidden_exact = {
        "target_modes",
        "teacher_midpoint_jvp",
        "provider_chart_modal_primal",
        "provider_chart_null_primal",
        "provider_chart_row_rms_primal",
        "provider_chart_modal_tangent",
        "provider_chart_null_tangent",
        "provider_chart_row_rms_tangent",
        "prompt_text",
        "prompt_texts",
        "token_ids",
        "input_ids",
        "source_model_state_dict",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in forbidden_exact:
                raise ValueError(
                    f"{path}.{key} is forbidden in diagnostic artifacts"
                )
            _assert_safe_artifact_tree(nested, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_safe_artifact_tree(
                nested,
                path=f"{path}[{index}]",
            )


def _publish_artifact(
    state: Mapping[str, object],
    report_payload: Mapping[str, object],
    *,
    output: Path,
) -> dict[str, object]:
    _assert_safe_artifact_tree(state)
    _assert_safe_artifact_tree(report_payload, path="report")
    _assert_tensor_free_report(report_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite diagnostic output")
    tensor_stage = _stage_path(output)
    report_stage = _stage_path(report_path)
    published: list[Path] = []
    try:
        torch.save(dict(state), tensor_stage)
        report = {
            **dict(report_payload),
            "artifact": {
                "tensor_file": str(output),
                "tensor_file_sha256": _file_sha256(tensor_stage),
                "tensor_file_bytes": tensor_stage.stat().st_size,
                "report_file": str(report_path),
                "committable": False,
            },
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        _assert_tensor_free_report(report)
        with report_stage.open("w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tensor_stage, output)
        published.append(output)
        os.link(report_stage, report_path)
        published.append(report_path)
        return report
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        tensor_stage.unlink(missing_ok=True)
        report_stage.unlink(missing_ok=True)


def _read_regular_file(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.read_bytes()


def _restore_calibration_binding(
    raw: object,
) -> DevelopmentCalibrationBinding:
    if not isinstance(raw, Mapping) or set(raw) != {
        "protocol_sha256",
        "pilot_panel_sha256",
        "calibration_rule_sha256",
        "selected_amplitude",
        "pilot_metric_sha256s",
        "artifact_sha256",
    }:
        raise ValueError("diagnostic calibration state is invalid")
    pilot_metric_sha256s = raw["pilot_metric_sha256s"]
    if not isinstance(pilot_metric_sha256s, (tuple, list)):
        raise TypeError("calibration pilot metric hashes must be a sequence")
    return DevelopmentCalibrationBinding(
        protocol_sha256=str(raw["protocol_sha256"]),
        pilot_panel_sha256=str(raw["pilot_panel_sha256"]),
        calibration_rule_sha256=str(raw["calibration_rule_sha256"]),
        selected_amplitude=float(raw["selected_amplitude"]),
        pilot_metric_sha256s=tuple(
            str(value) for value in pilot_metric_sha256s
        ),
        artifact_sha256=str(raw["artifact_sha256"]),
    )


def _restore_full_width_controls(
    raw: object,
) -> FullWidthReferenceControls:
    expected = {
        "artifact_kind",
        "format_version",
        "fit_target_center_sha256",
        "normalized_position_bin_centers_sha256",
        "normalized_position_bin_counts",
        "fit_probe_ids",
        "fit_probe_sha256s",
        "standardized_gauge_sha256",
        "position_semantics",
        "fit_target_center",
        "normalized_position_bin_centers",
        "artifact_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("diagnostic controls state is invalid")
    fit_target_center = raw["fit_target_center"]
    position_centers = raw["normalized_position_bin_centers"]
    counts = raw["normalized_position_bin_counts"]
    probe_ids = raw["fit_probe_ids"]
    probe_sha256s = raw["fit_probe_sha256s"]
    if (
        not isinstance(fit_target_center, Tensor)
        or not isinstance(position_centers, Tensor)
        or not isinstance(counts, (tuple, list))
        or not isinstance(probe_ids, (tuple, list))
        or not isinstance(probe_sha256s, (tuple, list))
    ):
        raise TypeError("diagnostic controls fields have invalid types")
    controls = FullWidthReferenceControls(
        fit_target_center=fit_target_center,
        normalized_position_bin_centers=position_centers,
        normalized_position_bin_counts=tuple(
            int(value) for value in counts
        ),
        fit_probe_ids=tuple(str(value) for value in probe_ids),
        fit_probe_sha256s=tuple(str(value) for value in probe_sha256s),
        standardized_gauge_sha256=str(
            raw["standardized_gauge_sha256"]
        ),
        position_semantics=str(raw["position_semantics"]),
        artifact_sha256=str(raw["artifact_sha256"]),
    )
    state = controls.state_dict()
    for name in expected - {
        "fit_target_center",
        "normalized_position_bin_centers",
    }:
        if _canonical_json_bytes(raw[name]) != _canonical_json_bytes(
            state[name]
        ):
            raise ValueError(
                f"stored diagnostic controls field {name!r} is invalid"
            )
    return controls


def load_objective_balance_diagnostic_artifact(
    path: Path | str,
) -> LoadedObjectiveBalanceDiagnosticArtifact:
    """Load and authenticate one diagnostic tensor/report pair.

    Loading is weights-only.  The validator treats the tensor file, adjacent
    JSON report, logical manifest, live code bundle, protocol, gauge, controls,
    candidate rows, and plan table as one bound publication.
    """

    source = Path(path).expanduser().resolve()
    if source.suffix != ".pt":
        raise ValueError("diagnostic artifact must use a .pt suffix")
    tensor_payload = _read_regular_file(source, label="diagnostic artifact")
    report_path = source.with_suffix(".json")
    report_payload = _read_regular_file(
        report_path,
        label="diagnostic report",
    )
    tensor_file_sha256 = hashlib.sha256(tensor_payload).hexdigest()
    raw = torch.load(
        io.BytesIO(tensor_payload),
        map_location="cpu",
        weights_only=True,
    )
    state_keys = {
        "manifest",
        "artifact_sha256",
        "protocol_state",
        "calibration_state",
        "unit_rms_gauge_state",
        "canonical_metric_weight",
        "controls_state",
        "plan_states",
        "candidate_results",
    }
    if not isinstance(raw, Mapping) or set(raw) != state_keys:
        raise ValueError(
            "diagnostic tensor fields do not match the frozen format"
        )
    manifest = raw["manifest"]
    if not isinstance(manifest, Mapping):
        raise TypeError("diagnostic manifest must be a mapping")
    logical_sha256 = _json_sha256(
        manifest,
        domain=_ARTIFACT_DOMAIN,
    )
    if (
        raw["artifact_sha256"] != logical_sha256
        or manifest.get("schema") != _SCHEMA
        or manifest.get("format_version") != _FORMAT_VERSION
    ):
        raise ValueError("diagnostic logical artifact binding mismatch")

    try:
        report = json.loads(report_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("diagnostic report is not canonical JSON") from exc
    if not isinstance(report, Mapping):
        raise TypeError("diagnostic report must be a mapping")
    report_without_hash = dict(report)
    supplied_report_sha256 = report_without_hash.pop(
        "report_sha256",
        None,
    )
    computed_report_sha256 = _json_sha256(
        report_without_hash,
        domain=_REPORT_DOMAIN,
    )
    if (
        supplied_report_sha256 != computed_report_sha256
        or report.get("artifact_sha256") != logical_sha256
    ):
        raise ValueError("diagnostic report SHA-256 mismatch")
    report_artifact = report.get("artifact")
    if not isinstance(report_artifact, Mapping) or set(report_artifact) != {
        "tensor_file",
        "tensor_file_sha256",
        "tensor_file_bytes",
        "report_file",
        "committable",
    }:
        raise ValueError("diagnostic report artifact binding is invalid")
    if (
        report_artifact.get("tensor_file") != str(source)
        or report_artifact.get("report_file") != str(report_path)
        or report_artifact.get("tensor_file_sha256")
        != tensor_file_sha256
        or report_artifact.get("tensor_file_bytes")
        != len(tensor_payload)
        or report_artifact.get("committable") is not False
    ):
        raise ValueError("diagnostic report does not bind the tensor file")
    report_extra_keys = {
        "artifact_sha256",
        "protocol",
        "calibration",
        "pilot_metrics",
        "pilot_measurement",
        "fit_measurement",
        "fit_provider_chart_mismatch_diagnostics",
        "teacher_signal_diagnostics",
        "gauge",
        "candidate_results",
        "interpretation",
        "safety",
        "artifact",
        "report_sha256",
    }
    if set(report) != set(manifest) | report_extra_keys:
        raise ValueError(
            "diagnostic report fields do not match the frozen format"
        )
    for name, value in manifest.items():
        if _canonical_json_bytes(report.get(name)) != (
            _canonical_json_bytes(value)
        ):
            raise ValueError(
                f"diagnostic report manifest field {name!r} drifted"
            )

    code_sha256s = manifest.get("code_sha256s")
    if (
        not isinstance(code_sha256s, Mapping)
        or dict(code_sha256s) != _code_sha256s()
        or manifest.get("code_bundle_sha256")
        != _code_bundle_sha256(dict(code_sha256s))
    ):
        raise ValueError("diagnostic code binding differs from live code")
    protocol_state = raw["protocol_state"]
    if not isinstance(protocol_state, Mapping):
        raise TypeError("diagnostic protocol state must be a mapping")
    protocol = ObjectiveBalanceDiagnosticProtocol.from_state_dict(
        protocol_state
    )
    if (
        protocol.artifact_sha256
        != DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256
        or manifest.get("protocol_sha256") != protocol.artifact_sha256
        or manifest.get("requested_execution_device")
        != protocol.execution_device
        or manifest.get("actual_execution_device")
        != protocol.execution_device
        or manifest.get("requested_execution_dtype")
        != protocol.execution_dtype
        or manifest.get("actual_execution_dtype")
        != protocol.execution_dtype
        or _canonical_json_bytes(report.get("protocol"))
        != _canonical_json_bytes(protocol_state)
    ):
        raise ValueError("diagnostic protocol binding mismatch")
    calibration_state = raw["calibration_state"]
    controls_state = raw["controls_state"]
    calibration = _restore_calibration_binding(calibration_state)
    controls = _restore_full_width_controls(controls_state)
    if (
        calibration.artifact_sha256
        != manifest.get("c2_calibration_sha256")
        or calibration.selected_amplitude
        != manifest.get("selected_calibration_amplitude")
        or calibration.protocol_sha256
        != manifest.get("c2_protocol_sha256")
        or calibration.pilot_panel_sha256
        != manifest.get("c2_pilot_panel_sha256")
        or _canonical_json_bytes(report.get("calibration"))
        != _canonical_json_bytes(calibration_state)
    ):
        raise ValueError("diagnostic calibration binding mismatch")
    if (
        controls.artifact_sha256
        != manifest.get("controls_sha256")
        or controls.standardized_gauge_sha256
        != manifest.get("standardized_gauge_sha256")
    ):
        raise ValueError("diagnostic controls binding mismatch")

    gauge_state = raw["unit_rms_gauge_state"]
    gauge = UnitRmsFisherGauge.from_state_dict(gauge_state)
    metric_weight = raw["canonical_metric_weight"]
    gauge.validate_source(metric_weight)
    if (
        gauge.artifact_sha256 != manifest.get("unit_rms_gauge_sha256")
        or not isinstance(metric_weight, Tensor)
        or metric_weight.shape != (_MODAL_WIDTH,)
        or not bool(torch.isfinite(metric_weight).all())
        or _tensor_sha256(metric_weight)
        != manifest.get("canonical_metric_weight_sha256")
    ):
        raise ValueError("diagnostic Fisher gauge binding mismatch")

    plan_states = raw["plan_states"]
    candidate_results = raw["candidate_results"]
    executed_ids_raw = manifest.get("executed_candidate_ids")
    plan_sha256s = manifest.get("candidate_plan_sha256s")
    result_sha256s = manifest.get("candidate_result_sha256s")
    if (
        not isinstance(executed_ids_raw, (tuple, list))
        or not isinstance(plan_states, Mapping)
        or not isinstance(candidate_results, Mapping)
        or not isinstance(plan_sha256s, Mapping)
        or not isinstance(result_sha256s, Mapping)
    ):
        raise TypeError("diagnostic candidate tables must be mappings")
    executed_ids = tuple(executed_ids_raw)
    if (
        not executed_ids
        or any(not isinstance(value, str) or not value for value in executed_ids)
        or len(set(executed_ids)) != len(executed_ids)
        or set(plan_states) != set(executed_ids)
        or set(candidate_results) != set(executed_ids)
        or set(plan_sha256s) != set(executed_ids)
        or set(result_sha256s) != set(executed_ids)
    ):
        raise ValueError("diagnostic candidate tables are incomplete")
    for candidate_id in executed_ids:
        plan_state = plan_states[candidate_id]
        candidate_row = candidate_results[candidate_id]
        if not isinstance(plan_state, Mapping):
            raise TypeError("diagnostic plan state must be a mapping")
        plan = ContrastAwareReferenceProviderPlan.from_state_dict(plan_state)
        if (
            not isinstance(candidate_row, Mapping)
            or plan.artifact_sha256
            != plan_sha256s[candidate_id]
            or candidate_row.get("candidate_id") != candidate_id
            or _json_sha256(candidate_row, domain=_ARTIFACT_DOMAIN)
            != result_sha256s[candidate_id]
        ):
            raise ValueError(
                f"diagnostic candidate {candidate_id!r} binding mismatch"
            )
    report_rows = report.get("candidate_results")
    if (
        not isinstance(report_rows, list)
        or _canonical_json_bytes(report_rows)
        != _canonical_json_bytes(
            tuple(candidate_results[value] for value in executed_ids)
        )
    ):
        raise ValueError("diagnostic report candidate rows drifted")
    if (
        manifest.get("selection_materialized") is not False
        or manifest.get("selection_measured") is not False
        or manifest.get("selection_scored") is not False
        or manifest.get("c2_artifact_loaded") is not False
    ):
        raise ValueError("diagnostic artifact violates the C2 firewall")
    return LoadedObjectiveBalanceDiagnosticArtifact(
        state=dict(raw),
        report=dict(report),
        manifest=dict(manifest),
        artifact_sha256=logical_sha256,
        tensor_file_sha256=tensor_file_sha256,
        report_sha256=computed_report_sha256,
    )


def _publish_and_authenticate_artifact(
    state: Mapping[str, object],
    report_payload: Mapping[str, object],
    *,
    output: Path,
) -> LoadedObjectiveBalanceDiagnosticArtifact:
    _publish_artifact(state, report_payload, output=output)
    try:
        return load_objective_balance_diagnostic_artifact(output)
    except BaseException:
        output.unlink(missing_ok=True)
        output.with_suffix(".json").unlink(missing_ok=True)
        raise


def _round_trip_plan_state(
    plan: ContrastAwareReferenceProviderPlan,
) -> dict[str, object]:
    state = plan.state_dict()
    restored = ContrastAwareReferenceProviderPlan.from_state_dict(state)
    if (
        restored.artifact_sha256 != plan.artifact_sha256
        or not torch.equal(restored.encoder_weight, plan.encoder_weight)
        or not torch.equal(restored.decoder_weight, plan.decoder_weight)
    ):
        raise RuntimeError("diagnostic plan round trip changed the candidate")
    return state


def _execute_recipe_schedule(
    protocol: _DiagnosticProtocolLike,
    *,
    evaluate: Callable[[_RecipeLike, str, int], _RecipeEvaluation],
) -> tuple[
    tuple[_RecipeEvaluation, ...],
    _RecipeEvaluation | None,
    _RecipeEvaluation | None,
]:
    recipes = tuple(protocol.recipes)
    if len(recipes) != 4:
        raise ValueError("objective-balance protocol must declare D0-D3")
    results: list[_RecipeEvaluation] = []
    first_passer: _RecipeEvaluation | None = None
    replication: _RecipeEvaluation | None = None
    for index, recipe in enumerate(recipes):
        primary = evaluate(recipe, "primary", protocol.primary_seed)
        if (
            primary.recipe_id != recipe.recipe_id
            or primary.seed_role != "primary"
            or primary.seed != protocol.primary_seed
        ):
            raise RuntimeError("primary recipe evaluation identity drifted")
        results.append(primary)
        # D0 is a required raw-gauge control, never an advancement treatment.
        if index == 0 or not recipe.advancement_eligible:
            continue
        if primary.combined_pass:
            attempted_replication = evaluate(
                recipe,
                "replication",
                protocol.replication_seed,
            )
            if (
                attempted_replication.recipe_id != recipe.recipe_id
                or attempted_replication.seed_role != "replication"
                or attempted_replication.seed != protocol.replication_seed
            ):
                raise RuntimeError(
                    "replication recipe evaluation identity drifted"
                )
            results.append(attempted_replication)
            if attempted_replication.combined_pass:
                first_passer = primary
                replication = attempted_replication
                break
    return tuple(results), first_passer, replication


def _diagnostic_decision(
    protocol: ObjectiveBalanceDiagnosticProtocol,
    evaluations: Sequence[_RecipeEvaluation],
    *,
    two_seed_primary: _RecipeEvaluation | None,
    two_seed_replication: _RecipeEvaluation | None,
) -> dict[str, object]:
    primary_treatment_passers = tuple(
        value
        for value in evaluations
        if value.seed_role == "primary"
        and value.combined_pass
        and protocol.recipe(value.recipe_id).advancement_eligible
    )
    replicated_pass = (
        two_seed_replication is not None
        and two_seed_replication.combined_pass
    )
    authorized_recipe_id = (
        two_seed_primary.recipe_id
        if two_seed_primary is not None and replicated_pass
        else None
    )
    if authorized_recipe_id is not None:
        outcome = "two_seed_fit_pass_may_declare_fresh_c3"
    elif primary_treatment_passers:
        outcome = "primary_fit_passes_failed_replication_no_c3_authority"
    else:
        outcome = "no_primary_treatment_passed_fit_gates"
    return {
        "primary_treatment_passing_recipe_ids": tuple(
            value.recipe_id for value in primary_treatment_passers
        ),
        "first_primary_passing_recipe_id": (
            None
            if not primary_treatment_passers
            else primary_treatment_passers[0].recipe_id
        ),
        "replication_passed": replicated_pass,
        "authorized_fresh_c3_recipe_id": authorized_recipe_id,
        "outcome": outcome,
    }


def _objective_for_recipe(
    recipe: ObjectiveBalanceRecipe,
) -> ContrastAwareObjective:
    return ContrastAwareObjective(
        pointwise_weight=recipe.pointwise_weight,
        sensitivity_relative_delta_weight=(
            recipe.sensitivity_relative_delta_weight
        ),
        sensitivity_direction_weight=recipe.direction_weight,
        midpoint_jvp_weight=recipe.midpoint_jvp_weight,
        intended_null_weight=recipe.intended_null_weight,
        sensitivity_relative_floor=1e-6,
        direction_norm_floor=1e-8,
        jvp_relative_floor=1e-6,
    )


def _contribution_balance_gate(
    plan: ContrastAwareReferenceProviderPlan,
    *,
    recipe: ObjectiveBalanceRecipe,
    gates: ObjectiveBalanceDiagnosticGates,
    training_teacher_energy: float,
    raw_teacher_energy: float,
    teacher_signal_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    audit = audit_objective_contributions(
        plan.initial_metrics,
        plan.objective,
    )
    active_nonnull = {
        "pointwise": audit.pointwise_fraction,
        "sensitivity_relative_delta": (
            audit.sensitivity_relative_delta / audit.total
            if audit.total > 0.0
            else 0.0
        ),
        "sensitivity_direction": (
            audit.sensitivity_direction / audit.total
            if audit.total > 0.0
            else 0.0
        ),
        "midpoint_jvp": (
            audit.midpoint_jvp / audit.total
            if audit.total > 0.0
            else 0.0
        ),
    }
    raw_energy_pass = (
        raw_teacher_energy
        > gates.minimum_gauge_energy
    )
    if plan.objective.pointwise_weight <= 0.0:
        raise RuntimeError("diagnostic pointwise objective became inactive")
    if plan.fisher_metric_supplied is not True:
        raise RuntimeError("diagnostic plan lost its explicit Fisher metric")
    normalized_energy_error = abs(training_teacher_energy - 1.0)
    if plan.fisher_metric_weight.square().mean().sqrt().item() == 0.0:
        raise RuntimeError("diagnostic training metric is degenerate")
    if (
        plan.fisher_metric_weight.shape != (_MODAL_WIDTH,)
        or not bool(torch.isfinite(plan.fisher_metric_weight).all())
    ):
        raise RuntimeError("diagnostic training metric is invalid")
    if (
        plan.objective.artifact_sha256
        != _objective_for_recipe(recipe).artifact_sha256
    ):
        raise RuntimeError("diagnostic plan objective semantics drifted")
    delta_floor = (
        gates.minimum_teacher_mse_floor_multiple
        * plan.objective.sensitivity_relative_floor**2
    )
    jvp_floor = (
        gates.minimum_teacher_mse_floor_multiple
        * plan.objective.jvp_relative_floor**2
    )
    teacher_delta_pass = (
        float(teacher_signal_diagnostics["minimum_teacher_delta_mse"])
        > delta_floor
    )
    teacher_jvp_pass = (
        float(teacher_signal_diagnostics["minimum_teacher_jvp_mse"])
        > jvp_floor
    )
    pointwise_pass = (
        gates.minimum_initial_pointwise_share
        <= audit.pointwise_fraction
        <= gates.maximum_initial_pointwise_share
    )
    contrast_pass = (
        audit.contrast_fraction
        >= gates.minimum_initial_contrast_share
    )
    largest_active = max(active_nonnull.values())
    single_component_pass = (
        largest_active
        <= gates.maximum_initial_active_component_share
    )
    unit_energy_pass = (
        normalized_energy_error
        <= gates.normalized_energy_absolute_tolerance
    )
    flags = {
        "reported_total_matches": audit.reported_total_matches,
        "raw_teacher_energy_above_floor": raw_energy_pass,
        "unit_rms_teacher_energy": unit_energy_pass,
        "teacher_delta_above_relative_floor": teacher_delta_pass,
        "teacher_jvp_above_relative_floor": teacher_jvp_pass,
        "initial_pointwise_fraction": pointwise_pass,
        "initial_contrast_fraction": contrast_pass,
        "maximum_single_active_nonnull_fraction": single_component_pass,
    }
    return {
        "passed": all(flags.values()),
        "flags": flags,
        "initial_contribution_audit": audit.state_dict(),
        "active_nonnull_component_fractions": active_nonnull,
        "maximum_active_nonnull_component_fraction": largest_active,
        "raw_fit_teacher_weighted_energy": raw_teacher_energy,
        "training_fit_teacher_weighted_energy": training_teacher_energy,
        "unit_rms_energy_absolute_error": normalized_energy_error,
        "minimum_raw_teacher_energy": (
            gates.minimum_gauge_energy
        ),
        "maximum_unit_rms_energy_absolute_error": (
            gates.normalized_energy_absolute_tolerance
        ),
        "teacher_delta_mse_floor": delta_floor,
        "teacher_jvp_mse_floor": jvp_floor,
    }


def _fit_execution_accounting(
    plan: ContrastAwareReferenceProviderPlan,
    fit_batches: Sequence[object],
) -> dict[str, object]:
    raw = c2._execution_accounting(plan, fit_batches)  # type: ignore[arg-type]
    return {
        key.replace("selection_panel", "fit_panel"): value
        for key, value in raw.items()
    }


def _training_metric_for_recipe(
    recipe: ObjectiveBalanceRecipe,
    *,
    raw_metric_weight: Tensor,
    unit_rms_gauge: UnitRmsFisherGauge,
) -> Tensor:
    if recipe.training_metric == "canonical_fisher":
        return raw_metric_weight
    if recipe.training_metric == "fit_teacher_weighted_rms":
        return unit_rms_gauge.metric_weight
    raise ValueError("diagnostic recipe training metric is invalid")


def _score_fit_only_ordinary_candidate(
    *,
    controls: FullWidthReferenceControls,
    fit_probes: Sequence[FullWidthReferenceProbe],
    candidate: FullWidthReferenceCandidate,
    gates: SyntheticReferenceGates,
) -> FullWidthCandidateScore:
    """Score a candidate on the same authenticated fit panel as its controls.

    The shared reference scorer normally enforces disjoint selection or
    assessment panels.  This diagnostic deliberately makes no generalization
    claim, so it uses the scorer's validated metric core after enforcing a
    stricter fit-only identity contract here.  No fit endpoint is relabeled as
    selection data.
    """

    if not isinstance(controls, FullWidthReferenceControls):
        raise TypeError("controls must be FullWidthReferenceControls")
    if not isinstance(candidate, FullWidthReferenceCandidate):
        raise TypeError("candidate must be FullWidthReferenceCandidate")
    if not isinstance(gates, SyntheticReferenceGates):
        raise TypeError("gates must be SyntheticReferenceGates")
    if isinstance(fit_probes, (str, bytes)) or not isinstance(
        fit_probes,
        Sequence,
    ):
        raise TypeError("fit probes must be a sequence")
    probes = tuple(fit_probes)
    if not probes:
        raise ValueError("fit probes must not be empty")
    if any(not isinstance(value, FullWidthReferenceProbe) for value in probes):
        raise TypeError(
            "fit probes must contain FullWidthReferenceProbe values"
        )
    if any(value.split != "fit" for value in probes):
        raise ValueError("fit-only scoring requires only 'fit' probes")
    probe_ids = tuple(value.probe_id for value in probes)
    if len(set(probe_ids)) != len(probe_ids):
        raise ValueError("fit-only scoring probes contain duplicate ids")
    if tuple(sorted(probe_ids)) != controls.fit_probe_ids:
        raise ValueError("fit-only scoring probe ids differ from controls")
    probe_sha256s = tuple(
        sorted(value.artifact_sha256 for value in probes)
    )
    if probe_sha256s != controls.fit_probe_sha256s:
        raise ValueError("fit-only scoring probe hashes differ from controls")
    gauges = {
        controls.standardized_gauge_sha256,
        candidate.standardized_gauge_sha256,
        *(value.standardized_gauge_sha256 for value in probes),
    }
    if len(gauges) != 1:
        raise ValueError(
            "controls, fit probes, and candidate use multiple gauges"
        )
    deferred_gates = _deferred_collision_gates(gates)
    if deferred_gates.minimum_collision_target_relative_difference != 0.0:
        raise RuntimeError("fit-only collision gate was not deferred")
    return reference_selection._score_validated_full_width_reference_candidate(
        controls=controls,
        probes=probes,
        collision_probes=(),
        collision_gate_deferred=True,
        candidate=candidate,
        gates=deferred_gates,
    )


def _fit_only_ordinary_candidate_and_score(
    *,
    candidate_id: str,
    plan: ContrastAwareReferenceProviderPlan,
    measured: Sequence[object],
    ordinary_probes: Sequence[FullWidthReferenceProbe],
    controls: FullWidthReferenceControls,
    metric_weight: Tensor,
    standardized_gauge_sha256: str,
    support_radius: float,
    gates: SyntheticReferenceGates,
) -> tuple[
    FullWidthReferenceCandidate,
    FullWidthCandidateScore,
    dict[str, Tensor],
    dict[str, object],
]:
    """Build and score one ordinary candidate without changing split labels."""

    raw64 = c2._runtime_predictions(  # type: ignore[arg-type]
        plan,
        measured,
        dtype=torch.float64,
    )
    raw32 = c2._runtime_predictions(  # type: ignore[arg-type]
        plan,
        measured,
        dtype=torch.float32,
    )
    structural, structural_metadata = c2._structural_metrics(  # type: ignore[arg-type]
        plan,
        measured,
        support_radius=support_radius,
        raw64=raw64,
        raw32=raw32,
    )
    ordinary_ids = {value.probe_id for value in ordinary_probes}
    predictions = tuple(
        FullWidthCandidatePrediction(
            probe_id=probe_id,
            retained_standardized_prediction=(
                raw64[probe_id] * metric_weight.view(1, 1, -1)
            ),
            standardized_gauge_sha256=standardized_gauge_sha256,
        )
        for probe_id in sorted(ordinary_ids)
    )
    candidate = FullWidthReferenceCandidate(
        candidate_id=candidate_id,
        source_rank=_MODAL_WIDTH,
        target_rank=_MODAL_WIDTH,
        stored_scalar_count=plan.accounting().total_stored_scalar_count,
        predictions=predictions,
        structural_metrics=structural,
        candidate_binding_sha256=plan.artifact_sha256,
    )
    score = _score_fit_only_ordinary_candidate(
        controls=controls,
        fit_probes=ordinary_probes,
        candidate=candidate,
        gates=gates,
    )
    return candidate, score, raw64, structural_metadata


def _evaluate_recipe(
    recipe: ObjectiveBalanceRecipe,
    *,
    seed_role: str,
    seed: int,
    diagnostic_protocol: ObjectiveBalanceDiagnosticProtocol,
    c2_protocol: ContrastProviderDevelopmentProtocol,
    calibration: DevelopmentCalibrationBinding,
    basis: object,
    norm_sha256: str,
    epsilon: float,
    fit: Sequence[object],
    modal_center: Tensor,
    gain_log_center: float,
    gain_log_scale: float,
    target_center: Tensor,
    target_scale: Tensor,
    raw_metric_weight: Tensor,
    unit_rms_gauge: UnitRmsFisherGauge,
    raw_teacher_energy: float,
    natural_pairs: Sequence[ReferenceProviderContrastPair],
    fit_data_binding_sha256: str,
    ordinary_probes: Sequence[object],
    controls: object,
    standardized_gauge_sha256: str,
    fidelity_gates: SyntheticReferenceGates,
    contrast_gates: ContrastAssessmentGates,
) -> _RecipeEvaluation:
    if seed_role not in {"primary", "replication"}:
        raise ValueError("diagnostic seed role is invalid")
    objective = _objective_for_recipe(recipe)
    training_metric = _training_metric_for_recipe(
        recipe,
        raw_metric_weight=raw_metric_weight,
        unit_rms_gauge=unit_rms_gauge,
    )
    training_teacher_energy = _fit_teacher_weighted_energy(
        fit,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=training_metric,
    )
    training_teacher_signal_diagnostics = _teacher_signal_diagnostics(
        fit,
        natural_pairs,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=training_metric,
    )
    candidate_binding_sha256 = _provider_binding_sha256(
        diagnostic_protocol_sha256=diagnostic_protocol.protocol_sha256,
        recipe=recipe,
        c2_protocol_sha256=c2_protocol.protocol_sha256,
        calibration_sha256=calibration.artifact_sha256,
        basis_payload_sha256=basis.basis_payload_sha256,  # type: ignore[attr-defined]
        source_model_sha256=basis.source_model_sha256,  # type: ignore[attr-defined]
        norm_sha256=norm_sha256,
        training_metric_weight=training_metric,
        target_center=target_center,
        target_scale=target_scale,
    )
    fit_batches = c2._indexed_batches(
        fit,
        split="fit",
        binding_sha256=fit_data_binding_sha256,
    )
    fit_pairs, pair_balance = _balance_training_pairs(
        natural_pairs,
        recipe=recipe,
    )
    plan = fit_contrast_aware_reference_provider(
        modal_center=modal_center,
        gain_log_center=gain_log_center,
        gain_log_scale=gain_log_scale,
        residual_width=basis.residual_width,  # type: ignore[attr-defined]
        rms_epsilon=epsilon,
        target_center=target_center,
        target_scale=target_scale,
        fit_batches=fit_batches,
        contrast_pairs=fit_pairs,
        executor_config=c2._executor_config(
            diagnostic_protocol.latent_rank
        ),
        objective=objective,
        fisher_metric_weight=training_metric,
        steps=recipe.steps,
        learning_rate=recipe.learning_rate,
        seed=seed,
    )
    plan_state = _round_trip_plan_state(plan)
    support_radius = c2._feature_radius(plan, fit)
    candidate_id = f"{recipe.recipe_id}.{seed_role}"
    (
        candidate,
        ordinary_score,
        predictions,
        structural_metadata,
    ) = _fit_only_ordinary_candidate_and_score(
        candidate_id=candidate_id,
        plan=plan,
        measured=fit,
        ordinary_probes=ordinary_probes,
        controls=controls,
        metric_weight=raw_metric_weight,
        standardized_gauge_sha256=standardized_gauge_sha256,
        support_radius=support_radius,
        gates=fidelity_gates,
    )
    contrast_result, identities, coverage = _fit_contrast_assessment(
        protocol=c2_protocol,
        measured=fit,
        predictions=predictions,
        metric_weight=raw_metric_weight,
        gates=contrast_gates,
        required_null_candidate_pass_count=(
            diagnostic_protocol.gates.required_null_candidate_pass_count
        ),
    )
    balance_gate = _contribution_balance_gate(
        plan,
        recipe=recipe,
        gates=diagnostic_protocol.gates,
        training_teacher_energy=training_teacher_energy,
        raw_teacher_energy=raw_teacher_energy,
        teacher_signal_diagnostics=training_teacher_signal_diagnostics,
    )
    ordinary_gate_state = ordinary_score.gate_flags.state_dict()
    ordinary_gate_values = tuple(
        passed
        for name, passed in ordinary_gate_state.items()
        if name != "all_passed"
    )
    ordinary_gate_contract_pass = (
        len(ordinary_gate_values)
        == diagnostic_protocol.gates.required_ordinary_gate_count
        and all(ordinary_gate_values)
    )
    sensitivity_contract_pass = bool(
        coverage["every_teacher_qualified_contrast_passed"]
    )
    family_contract_pass = contrast_result.overall_status == "pass"
    fidelity_combined_pass = (
        (
            ordinary_gate_contract_pass
            if diagnostic_protocol.gates.require_all_ordinary_gates
            else ordinary_score.passed
        )
        and (
            sensitivity_contract_pass
            if diagnostic_protocol.gates.require_every_eligible_sensitivity_contrast
            else True
        )
        and (
            family_contract_pass
            if diagnostic_protocol.gates.require_all_contrast_families
            else True
        )
        and bool(coverage["all_families_cover_all_four_rank_bands"])
        and bool(coverage["required_null_contrasts_valid_and_passed"])
    )
    advancement_pass = (
        recipe.advancement_eligible
        and fidelity_combined_pass
        and bool(balance_gate["passed"])
    )
    final_audit = audit_objective_contributions(
        plan.final_metrics,
        objective,
    )
    row = {
        "candidate_id": candidate_id,
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": recipe.artifact_sha256,
        "recipe": recipe.state_dict(),
        "seed_role": seed_role,
        "seed": seed,
        "latent_rank": plan.latent_rank,
        "training_metric": recipe.training_metric,
        "training_metric_weight_sha256": _tensor_sha256(training_metric),
        "canonical_scoring_metric_weight_sha256": _tensor_sha256(
            raw_metric_weight
        ),
        "pair_balance": pair_balance,
        "training_teacher_signal_diagnostics": (
            training_teacher_signal_diagnostics
        ),
        "provider_binding_sha256": candidate_binding_sha256,
        "fit_data_binding_sha256": fit_data_binding_sha256,
        "fit_data_binding_recipe_independent": True,
        "plan_sha256": plan.artifact_sha256,
        "plan_round_trip_passed": True,
        "accounting": asdict(plan.accounting()),
        "execution_accounting": _fit_execution_accounting(
            plan,
            fit_batches,
        ),
        "initial_training_metrics": plan.initial_metrics.state_dict(),
        "final_training_metrics": plan.final_metrics.state_dict(),
        "final_contribution_audit": final_audit.state_dict(),
        "objective_balance_gate": balance_gate,
        "protocol_gate_contract": {
            "ordinary_gate_count": len(ordinary_gate_values),
            "required_ordinary_gate_count": (
                diagnostic_protocol.gates.required_ordinary_gate_count
            ),
            "all_ordinary_gates_passed": ordinary_gate_contract_pass,
            "every_eligible_sensitivity_passed": (
                sensitivity_contract_pass
            ),
            "all_contrast_families_formally_passed": (
                family_contract_pass
            ),
            "required_null_contrasts_passed": bool(
                coverage["required_null_contrasts_valid_and_passed"]
            ),
        },
        "ordinary_score": ordinary_score.state_dict(),
        "contrast_result": contrast_result.state_dict(),
        "contrast_coverage": coverage,
        "contrast_identities": identities,
        "structural_metadata": structural_metadata,
        "mode_packing": c2._mode_packing_diagnostics(plan),
        "fit_fidelity_combined_pass": fidelity_combined_pass,
        "advancement_fit_gate_pass": advancement_pass,
        "failure_reasons": _candidate_failure_reasons(
            ordinary_score=ordinary_score,
            contrast_result=contrast_result,
            coverage=coverage,
            balance_gate_passed=bool(balance_gate["passed"]),
        ),
        "candidate_binding_sha256": candidate.artifact_sha256,
        "contains_raw_fit_targets": False,
        "contains_teacher_jvp_tensors": False,
        "contains_provider_chart_tensors": False,
        "_plan_state": plan_state,
    }
    return _RecipeEvaluation(
        recipe_id=recipe.recipe_id,
        seed_role=seed_role,
        seed=seed,
        combined_pass=advancement_pass,
        row=row,
        plan=plan,
    )


def _authenticated_protocols() -> tuple[
    ObjectiveBalanceDiagnosticProtocol,
    ContrastProviderDevelopmentProtocol,
]:
    protocol = default_objective_balance_diagnostic_protocol()
    if (
        protocol.artifact_sha256
        != DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256
    ):
        raise ValueError("objective-balance protocol trust anchor drifted")
    c2_protocol = default_contrast_provider_development_protocol()
    provenance = protocol.c2_provenance
    if (
        c2_protocol.protocol_sha256 != DEFAULT_DEVELOPMENT_PROTOCOL_SHA256
        or provenance.protocol_sha256 != c2_protocol.protocol_sha256
        or provenance.pilot_panel_sha256
        != c2_protocol.panel_sha256("pilot")
        or provenance.fit_panel_sha256 != c2_protocol.panel_sha256("fit")
        or provenance.pilot_panel_sha256
        != DEFAULT_DEVELOPMENT_PILOT_PANEL_SHA256
        or provenance.fit_panel_sha256
        != DEFAULT_DEVELOPMENT_FIT_PANEL_SHA256
        or provenance.objective_sha256 != c2._objective().artifact_sha256
        or provenance.objective_sha256
        != _objective_for_recipe(protocol.recipes[0]).artifact_sha256
        or provenance.training_sha256 != c2._training_sha256()
        or provenance.selection_materialization_allowed
        or provenance.c2_artifact_loading_allowed
    ):
        raise ValueError("objective-balance C2 provenance drifted")
    if protocol.latent_rank != _LATENT_RANK:
        raise ValueError("objective-balance rank drifted")
    gates = protocol.gates
    if not (
        gates.require_all_ordinary_gates
        and gates.require_every_eligible_sensitivity_contrast
        and gates.require_all_contrast_families
        and gates.require_two_seed_pass
    ):
        raise ValueError(
            "objective-balance required decision booleans drifted"
        )
    return protocol, c2_protocol


def _actual_model_execution(adapter: object) -> tuple[str, str]:
    module = getattr(adapter, "module", None)
    if module is None or not hasattr(module, "parameters"):
        raise TypeError("live adapter has no parameterized model module")
    first_parameter = next(module.parameters(), None)
    if first_parameter is None or not first_parameter.is_floating_point():
        raise TypeError("live model has no floating parameters")
    return (
        str(first_parameter.device),
        str(first_parameter.dtype).removeprefix("torch."),
    )


def describe_objective_balance_diagnostic() -> dict[str, object]:
    """Describe D0-D3 without loading a model or materializing any panel."""

    protocol, c2_protocol = _authenticated_protocols()
    code_sha256s = _code_sha256s()
    report = {
        "schema": f"{_SCHEMA}.description",
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.artifact_sha256,
        "protocol_trust_anchor": (
            DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256
        ),
        "protocol": protocol.state_dict(),
        "c2_protocol_sha256": c2_protocol.protocol_sha256,
        "allowed_c2_role_panel_sha256s": {
            "pilot": c2_protocol.panel_sha256("pilot"),
            "fit": c2_protocol.panel_sha256("fit"),
        },
        "allowed_c2_role_probe_counts": {
            "pilot": len(c2_protocol.probes_for_role("pilot")),
            "fit": len(c2_protocol.probes_for_role("fit")),
        },
        "selection_role_allowed": False,
        "selection_materialization_allowed": False,
        "selection_measurement_allowed": False,
        "c2_artifact_loading_allowed": False,
        "recipe_ids": tuple(
            value.recipe_id for value in protocol.recipes
        ),
        "advancement_recipe_ids": protocol.advancement_recipe_ids,
        "latent_rank": protocol.latent_rank,
        "execution_device": protocol.execution_device,
        "execution_dtype": protocol.execution_dtype,
        "primary_seed": protocol.primary_seed,
        "replication_seed": protocol.replication_seed,
        "decision_rule": (
            "run_D0_then_D1_D3_in_order_replicate_each_primary_passer_"
            "continue_after_failed_replication_stop_at_first_two_seed_pass_"
            "or_after_exhausting_D3"
        ),
        "d3_treatment_semantics": (
            "composite_direction_weight_and_doubled_steps_"
            "no_component_attribution"
        ),
        "code_sha256s": code_sha256s,
        "code_bundle_sha256": _code_bundle_sha256(code_sha256s),
        "model_loaded": False,
        "pilot_materialized": False,
        "fit_materialized": False,
        "teacher_target_opened": False,
        "selection_materialized": False,
        "selection_target_opened": False,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
    }
    _assert_tensor_free_report(report)
    return report


def run_objective_balance_diagnostic(
    *,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    basis_package_file_sha256: str = DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    basis_package_payload_sha256: str = (
        DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
    ),
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Run the authenticated D0-D3 fit-only diagnostic and publish locally."""

    protocol, c2_protocol = _authenticated_protocols()
    if (
        device_name != protocol.execution_device
        or dtype != protocol.execution_dtype
    ):
        raise ValueError(
            "objective-balance execution is frozen to cpu/float32"
        )
    destination = _validate_output_path(output)
    code_sha256s = _code_sha256s()
    code_bundle_sha256 = _code_bundle_sha256(code_sha256s)
    fidelity_gates = SyntheticReferenceGates()
    contrast_gates = ContrastAssessmentGates()
    scoring_gates_sha256 = full_width_reference_gates_sha256(
        _deferred_collision_gates(fidelity_gates)
    )
    if (
        scoring_gates_sha256 != protocol.gates.ordinary_gates_sha256
        or contrast_gates.artifact_sha256
        != protocol.gates.contrast_gates_sha256
    ):
        raise ValueError("objective-balance scoring gates drifted")

    (
        basis,
        adapter,
        pre_ff3,
        post_ff3,
        epsilon,
    ) = _load_live_dependencies(
        basis_package_path=basis_package_path,
        basis_package_file_sha256=basis_package_file_sha256,
        basis_package_payload_sha256=basis_package_payload_sha256,
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_REVISION,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    actual_execution_device, actual_execution_dtype = (
        _actual_model_execution(adapter)
    )
    if (
        actual_execution_device != protocol.execution_device
        or actual_execution_dtype != protocol.execution_dtype
    ):
        raise ValueError(
            "live model execution device or dtype differs from protocol"
        )
    model_before = adapter.model_fingerprint()
    norm_sha256 = module_state_fingerprint(pre_ff3)
    raw_metric_weight = _fisher_metric_weight(basis)

    pilot, pilot_measurement = _measure_c2_role(
        role="pilot",
        protocol=c2_protocol,
        calibration=None,
        frozen_candidates=None,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
        replay_count=2,
    )
    pilot_metrics = c2._calibration_metrics(
        protocol=c2_protocol,
        measured=pilot,
        metric_weight=raw_metric_weight,
    )
    calibration = select_global_calibration_amplitude(
        c2_protocol,
        pilot_metrics,
    )
    provenance = protocol.c2_provenance
    calibrated_fit_panel_sha256 = c2_protocol.calibrated_panel_sha256(
        "fit",
        calibration,
    )
    if (
        calibration.selected_amplitude
        != _EXPECTED_C2_CALIBRATION_AMPLITUDE
        or calibration.selected_amplitude != provenance.selected_amplitude
        or calibration.artifact_sha256
        != _EXPECTED_C2_CALIBRATION_SHA256
        or calibration.artifact_sha256 != provenance.calibration_sha256
        or calibrated_fit_panel_sha256
        != provenance.calibrated_fit_panel_sha256
    ):
        raise ValueError("C2 pilot replay did not authenticate amplitude h=8")

    fit, fit_measurement = _measure_c2_role(
        role="fit",
        protocol=c2_protocol,
        calibration=calibration,
        frozen_candidates=None,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
        replay_count=2,
    )
    forbidden_prefixes = provenance.forbidden_probe_prefixes
    if any(
        value.probe.probe_id.startswith(  # type: ignore[attr-defined]
            forbidden_prefixes
        )
        for value in fit
    ):
        raise RuntimeError("C2 selection identity entered the fit measurement")
    (
        modal_center,
        gain_log_center,
        gain_log_scale,
        target_center,
        target_scale,
    ) = c2._fit_gauges(
        fit,
        residual_width=basis.residual_width,
        epsilon=epsilon,
    )
    raw_teacher_energy = _fit_teacher_weighted_energy(
        fit,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=raw_metric_weight,
    )
    unit_rms_gauge = UnitRmsFisherGauge.from_metric_weight(
        raw_metric_weight
    )
    unit_teacher_energy = _fit_teacher_weighted_energy(
        fit,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=unit_rms_gauge.metric_weight,
    )
    if (
        raw_teacher_energy <= protocol.gates.minimum_gauge_energy
        or abs(unit_teacher_energy - 1.0)
        > protocol.gates.normalized_energy_absolute_tolerance
    ):
        raise ValueError("fit teacher Fisher gauge failed normalization gates")
    natural_pairs, chart_mismatch_diagnostics = c2._training_contrast_pairs(
        protocol=c2_protocol,
        measured=fit,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
    )
    teacher_signal_diagnostics = _teacher_signal_diagnostics(
        fit,
        natural_pairs,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=raw_metric_weight,
    )
    minimum_teacher_mse = min(
        float(teacher_signal_diagnostics["minimum_teacher_delta_mse"]),
        float(teacher_signal_diagnostics["minimum_teacher_jvp_mse"]),
    )
    c2_objective = c2._objective()
    teacher_floor = (
        protocol.gates.minimum_teacher_mse_floor_multiple
        * max(
            c2_objective.sensitivity_relative_floor**2,
            c2_objective.jvp_relative_floor**2,
        )
    )
    if minimum_teacher_mse <= teacher_floor:
        raise ValueError("fit teacher contrast signal is too close to a floor")

    fit_data_binding_sha256 = _fit_data_binding_sha256(
        basis=basis,
        c2_protocol=c2_protocol,
        calibration=calibration,
        norm_sha256=norm_sha256,
        canonical_metric_weight=raw_metric_weight,
    )
    standardized_gauge_sha256 = _standardized_gauge_sha256(
        basis_payload_sha256=basis.basis_payload_sha256,
        source_model_sha256=basis.source_model_sha256,
        c2_protocol_sha256=c2_protocol.protocol_sha256,
        calibration_sha256=calibration.artifact_sha256,
        target_center=target_center,
        target_scale=target_scale,
        canonical_metric_weight=raw_metric_weight,
    )
    fit_ordinary_probes = c2._ordinary_full_width_probes(
        fit,
        split="fit",
        metric_weight=raw_metric_weight,
        standardized_gauge_sha256=standardized_gauge_sha256,
    )
    controls = fit_full_width_reference_controls(
        fit_probes=fit_ordinary_probes,
        position_bin_count=16,
    )

    def evaluate(
        recipe: _RecipeLike,
        seed_role: str,
        seed: int,
    ) -> _RecipeEvaluation:
        if not isinstance(recipe, ObjectiveBalanceRecipe):
            raise TypeError("diagnostic schedule yielded an invalid recipe")
        if seed_role == "primary" and seed != recipe.primary_seed:
            raise ValueError("recipe primary seed differs from declaration")
        return _evaluate_recipe(
            recipe,
            seed_role=seed_role,
            seed=seed,
            diagnostic_protocol=protocol,
            c2_protocol=c2_protocol,
            calibration=calibration,
            basis=basis,
            norm_sha256=norm_sha256,
            epsilon=epsilon,
            fit=fit,
            modal_center=modal_center,
            gain_log_center=gain_log_center,
            gain_log_scale=gain_log_scale,
            target_center=target_center,
            target_scale=target_scale,
            raw_metric_weight=raw_metric_weight,
            unit_rms_gauge=unit_rms_gauge,
            raw_teacher_energy=raw_teacher_energy,
            natural_pairs=natural_pairs,
            fit_data_binding_sha256=fit_data_binding_sha256,
            ordinary_probes=fit_ordinary_probes,
            controls=controls,
            standardized_gauge_sha256=standardized_gauge_sha256,
            fidelity_gates=fidelity_gates,
            contrast_gates=contrast_gates,
        )

    evaluations, first_passer, replication = _execute_recipe_schedule(
        protocol,
        evaluate=evaluate,
    )
    candidate_rows: list[dict[str, object]] = []
    plan_states: dict[str, dict[str, object]] = {}
    for evaluation in evaluations:
        row = dict(evaluation.row)
        plan_state = row.pop("_plan_state")
        if not isinstance(plan_state, dict):
            raise RuntimeError("diagnostic candidate lost its plan state")
        candidate_id = str(row["candidate_id"])
        candidate_rows.append(row)
        plan_states[candidate_id] = plan_state
    decision = _diagnostic_decision(
        protocol,
        evaluations,
        two_seed_primary=first_passer,
        two_seed_replication=replication,
    )
    authorized_recipe_id = decision["authorized_fresh_c3_recipe_id"]
    outcome = str(decision["outcome"])

    if (
        adapter.model_fingerprint() != model_before
        or module_state_fingerprint(pre_ff3) != norm_sha256
        or _code_sha256s() != code_sha256s
    ):
        raise RuntimeError(
            "model, normalization, or code changed during diagnostic"
        )
    candidate_result_sha256s = {
        str(value["candidate_id"]): _json_sha256(
            value,
            domain=_ARTIFACT_DOMAIN,
        )
        for value in candidate_rows
    }
    manifest = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.artifact_sha256,
        "c2_protocol_sha256": c2_protocol.protocol_sha256,
        "c2_pilot_panel_sha256": c2_protocol.panel_sha256("pilot"),
        "c2_fit_panel_sha256": c2_protocol.panel_sha256("fit"),
        "c2_calibrated_fit_panel_sha256": calibrated_fit_panel_sha256,
        "c2_calibration_sha256": calibration.artifact_sha256,
        "selected_calibration_amplitude": calibration.selected_amplitude,
        "basis_package_file_sha256": basis_package_file_sha256,
        "basis_package_payload_sha256": basis.basis_payload_sha256,
        "source_model_sha256": basis.source_model_sha256,
        "requested_execution_device": device_name,
        "requested_execution_dtype": dtype,
        "actual_execution_device": actual_execution_device,
        "actual_execution_dtype": actual_execution_dtype,
        "pre_feedforward_norm_sha256": norm_sha256,
        "canonical_metric_weight_sha256": _tensor_sha256(
            raw_metric_weight
        ),
        "fit_data_binding_sha256": fit_data_binding_sha256,
        "fit_data_binding_semantics": (
            "exact_recipe_independent_C2_rank16_raw_control_binding_"
            "reused_for_all_D0_D3_recipes_to_freeze_batch_order"
        ),
        "unit_rms_gauge_sha256": unit_rms_gauge.artifact_sha256,
        "standardized_gauge_sha256": standardized_gauge_sha256,
        "controls_sha256": controls.artifact_sha256,
        "ordinary_gates_sha256": scoring_gates_sha256,
        "contrast_gates_sha256": contrast_gates.artifact_sha256,
        "executed_candidate_ids": tuple(
            str(value["candidate_id"]) for value in candidate_rows
        ),
        "candidate_plan_sha256s": {
            evaluation.row["candidate_id"]: evaluation.plan.artifact_sha256
            for evaluation in evaluations
        },
        "candidate_result_sha256s": candidate_result_sha256s,
        **decision,
        "selection_materialized": False,
        "selection_measured": False,
        "selection_scored": False,
        "selection_data_changed_training": False,
        "c2_artifact_loaded": False,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "code_sha256s": code_sha256s,
        "code_bundle_sha256": code_bundle_sha256,
        "scientific_scope": (
            "fit_only_rank16_optimization_diagnostic_not_generalization"
        ),
        "d3_treatment_semantics": (
            "composite_direction_weight_and_doubled_steps_"
            "no_component_attribution"
        ),
        "ordinary_scoring_panel_semantics": (
            "same_authenticated_C2_fit_endpoints_scored_with_truthful_fit_"
            "labels_via_validated_12_gate_metric_core_not_held_out"
        ),
    }
    logical_artifact_sha256 = _json_sha256(
        manifest,
        domain=_ARTIFACT_DOMAIN,
    )
    state = {
        "manifest": manifest,
        "artifact_sha256": logical_artifact_sha256,
        "protocol_state": protocol.state_dict(),
        "calibration_state": calibration.state_dict(),
        "unit_rms_gauge_state": unit_rms_gauge.state_dict(),
        "canonical_metric_weight": raw_metric_weight,
        "controls_state": controls.state_dict(),
        "plan_states": plan_states,
        "candidate_results": {
            str(value["candidate_id"]): value
            for value in candidate_rows
        },
    }
    gauge_state = unit_rms_gauge.state_dict()
    del gauge_state["metric_weight"]
    report_payload = {
        **manifest,
        "artifact_sha256": logical_artifact_sha256,
        "protocol": protocol.state_dict(),
        "calibration": calibration.state_dict(),
        "pilot_metrics": tuple(
            value.state_dict() for value in pilot_metrics
        ),
        "pilot_measurement": pilot_measurement,
        "fit_measurement": fit_measurement,
        "fit_provider_chart_mismatch_diagnostics": (
            chart_mismatch_diagnostics
        ),
        "teacher_signal_diagnostics": teacher_signal_diagnostics,
        "gauge": {
            **gauge_state,
            "raw_fit_teacher_weighted_energy": raw_teacher_energy,
            "unit_fit_teacher_weighted_energy": unit_teacher_energy,
            "target_center_sha256": _tensor_sha256(target_center),
            "target_scale_sha256": _tensor_sha256(target_scale),
        },
        "candidate_results": candidate_rows,
        "outcome": outcome,
        "interpretation": {
            "fit_side_only": True,
            "held_out_selection_evidence": False,
            "fresh_c3_required": True,
            "scalar_contribution_balance_proves_gradient_balance": False,
            "scalar_contribution_balance_proves_adam_update_balance": False,
            "balance_evidence_semantics": (
                "tests_fit_capability_consequence_under_dimensionless_"
                "objective_not_per_term_gradient_or_optimizer_share"
            ),
            "v4_assessment_opened": False,
            "natural_prompt_fidelity_claim": False,
            "whole_model_replacement_claim": False,
            "wall_clock_speed_claim": False,
            "whole_model_compression_claim": False,
            "requested_and_actual_execution_match_frozen_protocol": True,
            "provider_fit_numeric_dtype": "torch.float64",
            "d3_is_composite_treatment": True,
            "d3_changes_direction_weight_and_step_count": True,
            "d3_only_pass_supports_component_attribution": False,
            "relative_component_scale_invariance_proved": False,
            "d1_changes_only_absolute_terms_proved": False,
            "relative_component_scale_invariance_boundary": (
                "common_metric_rescaling_is_invariant_only_while_fixed_"
                "numerical_floors_are_inactive;teacher_delta_and_jvp_floors_"
                "are_gated_but_candidate_direction_norm_floor_is_not"
            ),
        },
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_provider_parameters": True,
            "contains_raw_teacher_targets": False,
            "contains_teacher_jvp_tensors": False,
            "contains_provider_chart_jvp_tensors": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_c2_selection_data": False,
            "committable": False,
        },
    }
    authenticated = _publish_and_authenticate_artifact(
        state,
        report_payload,
        output=destination,
    )
    return dict(authenticated.report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "describe",
        help="describe D0-D3 without loading or materializing live data",
    )
    run_parser = commands.add_parser(
        "run",
        help="replay C2 pilot and run the fit-only rank-16 diagnostic",
    )
    run_parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--cache-dir", type=Path)
    run_parser.add_argument(
        "--device",
        choices=(DIAGNOSTIC_EXECUTION_DEVICE,),
        default=DIAGNOSTIC_EXECUTION_DEVICE,
    )
    run_parser.add_argument(
        "--dtype",
        choices=(DIAGNOSTIC_EXECUTION_DTYPE,),
        default=DIAGNOSTIC_EXECUTION_DTYPE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        report = describe_objective_balance_diagnostic()
    else:
        report = run_objective_balance_diagnostic(
            basis_package_path=args.basis_package,
            output=args.output,
            cache_dir=args.cache_dir,
            device_name=args.device,
            dtype=args.dtype,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
