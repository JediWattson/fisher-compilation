"""Run the matched fit-only Gemma L3/L4 rank-64 capacity control.

This rung changes exactly one scientific variable from the authenticated D3
objective-balance primary: the learned outer modal packer is widened from
``64 -> 16 -> 64`` to ``64 -> 64 -> 64``.  The causal expert rank remains 16,
and the D3 objective, C2 pilot/fit rows, batch order, optimizer, seeds, gates,
execution device, and measurement dtype remain fixed.

The source D3 artifact is loaded through its strict authenticated loader only
to bind the failed baseline and its fit-data identities.  Its provider
parameters are never supplied to the rank-64 fitter.  A primary capability
pass authorizes only the frozen replication seed.  A two-seed pass opens a
separately preregistered compressed-width ladder; this control never opens C3.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch
from torch import Tensor

from .adapters import module_state_fingerprint
from .contrast_objective_balancing import (
    UnitRmsFisherGauge,
    audit_objective_contributions,
)
from .external_models import find_git_worktree
from .gated_executor import GatedCausalModalExecutorConfig
from .gemma3_experiment import DEFAULT_MODEL_ID
from .gemma3_l3_l4_basis_package import DEFAULT_BASIS_PACKAGE
from . import gemma3_l3_l4_contrast_provider_development as c2
from .gemma3_l3_l4_contrast_provider_development_protocol import (
    ContrastProviderDevelopmentProtocol,
    DevelopmentCalibrationBinding,
    default_contrast_provider_development_protocol,
    select_global_calibration_amplitude,
)
from . import gemma3_l3_l4_objective_balance_diagnostic as d0d3
from .gemma3_l3_l4_objective_balance_diagnostic_protocol import (
    DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256,
    ObjectiveBalanceDiagnosticProtocol,
    ObjectiveBalanceRecipe,
    default_objective_balance_diagnostic_protocol,
)
from .gemma3_l3_l4_rank64_capacity_control_protocol import (
    CAPACITY_CONTROL_EXECUTION_DEVICE,
    CAPACITY_CONTROL_EXECUTION_DTYPE,
    DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256,
    MatchedD3TrainingSpec,
    ObjectiveBalanceResultBinding,
    Rank64CapacityControlProtocol,
    default_rank64_capacity_control_protocol,
    source_replay_binding_sha256,
    source_sequence_binding_sha256,
)
from .gemma3_l3_l4_reference_provider_experiment import (
    DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    _deferred_collision_gates,
    _fisher_metric_weight,
    _load_live_dependencies,
)
from .gemma3_l3_l4_spectral_mapping_experiment import DEFAULT_REVISION
from .gemma3_l3_l4_synthetic_reference_protocol import (
    SyntheticReferenceGates,
)
from .state_conditioned_contrast_assessment import (
    ContrastAssessmentGates,
)
from . import state_conditioned_contrast_assessment as contrast_assessment
from .state_conditioned_contrast_fit import (
    ContrastAwareObjective,
    ContrastAwareReferenceProviderPlan,
    ReferenceProviderContrastPair,
    fit_contrast_aware_reference_provider,
)
from .state_conditioned_reference_selection import (
    FullWidthCandidateScore,
    FullWidthReferenceControls,
    fit_full_width_reference_controls,
    full_width_reference_gates_sha256,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE_DIAGNOSTIC",
    "LoadedRank64CapacityControlArtifact",
    "build_parser",
    "describe_rank64_capacity_control",
    "load_rank64_capacity_control_artifact",
    "main",
    "run_rank64_capacity_control",
]


DEFAULT_SOURCE_DIAGNOSTIC = d0d3.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-objective-balance-r64-capacity-control.pt"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_rank64_capacity_control"
_FORMAT_VERSION = 1
_MODAL_WIDTH = 64
_SOURCE_D3_CANDIDATE_ID = (
    "d3_unit_rms_family_balanced_direction.primary"
)
_EXPECTED_C2_CALIBRATION_AMPLITUDE = 8.0
_EXPECTED_C2_CALIBRATION_SHA256 = (
    "aedb23de65ed6a37d645539001311ddb415cd2713400777dac448cb96bd5bfa8"
)
_SOURCE_PLAN_SEQUENCE_FIELDS = (
    "fit_batch_sha256s",
    "fit_batch_content_sha256s",
    "fit_indexed_batch_sha256s",
    "fit_endpoint_sha256s",
    "fit_pair_sha256s",
)
_SOURCE_SEQUENCE_ANCHOR_FIELDS = {
    "fit_batch_sha256s": "d3_fit_batch_sequence_sha256",
    "fit_batch_content_sha256s": (
        "d3_fit_batch_content_sequence_sha256"
    ),
    "fit_indexed_batch_sha256s": (
        "d3_fit_indexed_batch_sequence_sha256"
    ),
    "fit_endpoint_sha256s": "d3_fit_endpoint_sequence_sha256",
    "fit_pair_sha256s": "d3_fit_pair_sequence_sha256",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_DOMAIN = b"fisher-graph:rank64-capacity-control:tensor:v1\0"
_BINDING_DOMAIN = b"fisher-graph:rank64-capacity-control:binding:v1\0"
_CODE_DOMAIN = b"fisher-graph:rank64-capacity-control:code:v1\0"
_ARTIFACT_DOMAIN = b"fisher-graph:rank64-capacity-control:artifact:v1\0"
_REPORT_DOMAIN = b"fisher-graph:rank64-capacity-control:report:v1\0"
_DECISION_FIELDS = {
    "outcome",
    "primary_treatment_valid",
    "primary_fit_capability_passed",
    "replication_executed",
    "replication_treatment_valid",
    "replication_fit_capability_passed",
    "two_seed_fit_capability_passed",
    "compressed_width_ladder_preregistration_supported",
    "fresh_c3_authorized",
    "compression_claim_authorized",
}
_MANIFEST_FIELDS = {
    "schema",
    "format_version",
    "protocol_sha256",
    "source_objective_protocol_sha256",
    "source_objective_result_binding_sha256",
    "source_objective_logical_artifact_sha256",
    "source_objective_tensor_file_sha256",
    "source_objective_report_sha256",
    "source_objective_code_bundle_sha256",
    "source_d3_recipe_sha256",
    "source_d3_primary_plan_sha256",
    "source_d3_primary_result_sha256",
    "source_d3_fit_data_binding_sha256",
    "c2_protocol_sha256",
    "c2_pilot_panel_sha256",
    "c2_fit_panel_sha256",
    "c2_calibrated_fit_panel_sha256",
    "c2_calibration_sha256",
    "selected_calibration_amplitude",
    "basis_package_file_sha256",
    "basis_package_payload_sha256",
    "source_model_sha256",
    "requested_execution_device",
    "requested_execution_dtype",
    "actual_execution_device",
    "actual_execution_dtype",
    "pre_feedforward_norm_sha256",
    "canonical_metric_weight_sha256",
    "fit_data_binding_sha256",
    "unit_rms_gauge_sha256",
    "standardized_gauge_sha256",
    "controls_sha256",
    "ordinary_gates_sha256",
    "contrast_gates_sha256",
    "baseline_latent_rank",
    "latent_rank",
    "expert_rank",
    "controlled_change",
    "executed_candidate_ids",
    "candidate_plan_sha256s",
    "candidate_result_sha256s",
    "selection_materialized",
    "selection_measured",
    "selection_scored",
    "selection_data_changed_training",
    "c2_provider_artifact_loaded",
    "authenticated_source_result_artifact_loaded",
    "source_d3_parameters_used_for_initialization",
    "cold_start",
    "v2_targets_loaded",
    "v3_targets_loaded",
    "prompt_text_loaded",
    "token_ids_loaded",
    "tokenizer_loaded",
    "natural_activation_rows_loaded",
    "code_sha256s",
    "code_bundle_sha256",
    "scientific_scope",
} | _DECISION_FIELDS
_CANDIDATE_ROW_FIELDS = {
    "candidate_id",
    "treatment_id",
    "source_recipe_id",
    "source_recipe_sha256",
    "training_sha256",
    "seed_role",
    "seed",
    "baseline_latent_rank",
    "latent_rank",
    "expert_rank",
    "cold_start",
    "source_plan_parameters_used_for_initialization",
    "training_metric",
    "training_metric_weight_sha256",
    "canonical_scoring_metric_weight_sha256",
    "pair_balance",
    "training_teacher_signal_diagnostics",
    "provider_binding_sha256",
    "fit_data_binding_sha256",
    "fit_data_binding_recipe_independent",
    "source_fit_sequence_comparison",
    "plan_sha256",
    "plan_round_trip_passed",
    "accounting",
    "execution_accounting",
    "initial_training_metrics",
    "final_training_metrics",
    "final_contribution_audit",
    "objective_balance_gate",
    "treatment_validity",
    "fit_capability_contract",
    "ordinary_score",
    "contrast_result",
    "contrast_coverage",
    "contrast_identities",
    "structural_metadata",
    "mode_packing",
    "fit_capability_pass",
    "combined_pass",
    "failure_reasons",
    "candidate_binding_sha256",
    "contains_raw_fit_targets",
    "contains_teacher_jvp_tensors",
    "contains_provider_chart_tensors",
}
_CODE_FILES = tuple(
    dict.fromkeys(
        (
            *d0d3._CODE_FILES,
            "gemma3_l3_l4_rank64_capacity_control.py",
            "gemma3_l3_l4_rank64_capacity_control_protocol.py",
        )
    )
)


@dataclass(frozen=True, slots=True)
class _SourceD3Bindings:
    fit_data_binding_sha256: str
    sequence_sha256s: Mapping[str, tuple[str, ...]]
    balanced_pair_sha256s: tuple[str, ...]
    natural_pair_sha256s: tuple[str, ...]
    shared_manifest: Mapping[str, object]
    source_plan_parameters_used: bool = False


@dataclass(frozen=True, slots=True)
class _CapacityEvaluation:
    candidate_id: str
    seed_role: str
    seed: int
    treatment_valid: bool
    fit_capability_pass: bool
    combined_pass: bool
    row: dict[str, object]
    plan: ContrastAwareReferenceProviderPlan


@dataclass(frozen=True, slots=True)
class LoadedRank64CapacityControlArtifact:
    """Authenticated views of one local rank-64 capacity publication."""

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
    result = {name: _file_sha256(root / name) for name in _CODE_FILES}
    if set(result) != set(_CODE_FILES):
        raise RuntimeError("rank-64 capacity code manifest is incomplete")
    return result


def _code_bundle_sha256(values: Mapping[str, str]) -> str:
    if set(values) != set(_CODE_FILES):
        raise ValueError("rank-64 capacity code manifest is incomplete")
    for name, value in values.items():
        _require_sha256(value, label=f"code digest {name}")
    return _json_sha256(dict(values), domain=_CODE_DOMAIN)


def _executor_config(
    protocol: Rank64CapacityControlProtocol,
) -> GatedCausalModalExecutorConfig:
    spec = protocol.executor
    config = GatedCausalModalExecutorConfig(
        input_modes=spec.input_modes,
        output_modes=spec.output_modes,
        expert_count=spec.expert_count,
        expert_rank=spec.expert_rank,
        router_width=spec.router_width,
        same_position_skip=spec.same_position_skip,
        max_positive_lag=spec.max_positive_lag,
        router_activation=spec.router_activation,  # type: ignore[arg-type]
        source_normalized_routing=spec.source_normalized_routing,
    )
    if (
        config.input_modes != protocol.latent_rank + 2
        or config.output_modes != protocol.latent_rank
        or config.expert_rank != 16
    ):
        raise RuntimeError("rank-64 executor config drifted")
    return config


def _objective(training: MatchedD3TrainingSpec) -> ContrastAwareObjective:
    return ContrastAwareObjective(
        pointwise_weight=training.pointwise_weight,
        sensitivity_relative_delta_weight=(
            training.sensitivity_relative_delta_weight
        ),
        sensitivity_direction_weight=(
            training.sensitivity_direction_weight
        ),
        midpoint_jvp_weight=training.midpoint_jvp_weight,
        intended_null_weight=training.intended_null_weight,
        sensitivity_relative_floor=training.sensitivity_relative_floor,
        direction_norm_floor=training.direction_norm_floor,
        jvp_relative_floor=training.jvp_relative_floor,
    )


def _provider_binding_sha256(
    *,
    protocol: Rank64CapacityControlProtocol,
    c2_protocol_sha256: str,
    calibration_sha256: str,
    basis_payload_sha256: str,
    source_model_sha256: str,
    norm_sha256: str,
    training_metric_weight: Tensor,
    target_center: Tensor,
    target_scale: Tensor,
) -> str:
    """Bind the rank-64 candidate without inheriting rank-16 constants."""

    return _json_sha256(
        {
            "schema": "fisher_graph.rank64_capacity_provider_binding.v1",
            "protocol_sha256": protocol.protocol_sha256,
            "source_result_binding_sha256": (
                protocol.source_result.artifact_sha256
            ),
            "training_sha256": protocol.training.artifact_sha256,
            "executor_sha256": protocol.executor.artifact_sha256,
            "c2_protocol_sha256": c2_protocol_sha256,
            "calibration_sha256": calibration_sha256,
            "basis_payload_sha256": basis_payload_sha256,
            "source_model_sha256": source_model_sha256,
            "norm_sha256": norm_sha256,
            "training_metric_weight_sha256": _tensor_sha256(
                training_metric_weight
            ),
            "target_center_sha256": d0d3._tensor_sha256(target_center),
            "target_scale_sha256": d0d3._tensor_sha256(target_scale),
            "baseline_latent_rank": protocol.baseline_latent_rank,
            "latent_rank": protocol.latent_rank,
            "visible_source_modes": protocol.visible_source_modes,
            "visible_target_modes": protocol.visible_target_modes,
            "cold_start": True,
            "source_plan_parameters_used": False,
        },
        domain=_BINDING_DOMAIN,
    )


def _authenticated_declarations() -> tuple[
    Rank64CapacityControlProtocol,
    ObjectiveBalanceDiagnosticProtocol,
    ContrastProviderDevelopmentProtocol,
    ObjectiveBalanceRecipe,
]:
    protocol = default_rank64_capacity_control_protocol()
    objective_protocol = default_objective_balance_diagnostic_protocol()
    c2_protocol = default_contrast_provider_development_protocol()
    d3 = objective_protocol.recipe(protocol.training.recipe_id)
    training = protocol.training
    frozen = {
        "training_metric": training.training_metric,
        "signed_pair_multiplicity": training.signed_pair_multiplicity,
        "pointwise_weight": training.pointwise_weight,
        "sensitivity_relative_delta_weight": (
            training.sensitivity_relative_delta_weight
        ),
        "direction_weight": training.sensitivity_direction_weight,
        "midpoint_jvp_weight": training.midpoint_jvp_weight,
        "intended_null_weight": training.intended_null_weight,
        "steps": training.steps,
        "learning_rate": training.learning_rate,
    }
    if (
        protocol.protocol_sha256
        != DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256
        or objective_protocol.protocol_sha256
        != DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256
        or d3.artifact_sha256
        != protocol.source_result.d3_recipe_sha256
        or any(getattr(d3, name) != value for name, value in frozen.items())
        or d3.primary_seed != protocol.primary_seed
        or objective_protocol.replication_seed != protocol.replication_seed
        or objective_protocol.gates.ordinary_gates_sha256
        != training.ordinary_gates_sha256
        or objective_protocol.gates.contrast_gates_sha256
        != training.contrast_gates_sha256
        or objective_protocol.c2_provenance.protocol_sha256
        != c2_protocol.protocol_sha256
        or objective_protocol.c2_provenance.calibration_sha256
        != _EXPECTED_C2_CALIBRATION_SHA256
    ):
        raise ValueError("rank-64 capacity declaration drifted")
    config = _executor_config(protocol)
    if asdict(config) != {
        name: getattr(protocol.executor, name)
        for name in asdict(config)
    }:
        raise ValueError("rank-64 executor declaration drifted")
    return protocol, objective_protocol, c2_protocol, d3


def _authenticate_source_d3(
    path: Path | str,
    *,
    protocol: Rank64CapacityControlProtocol,
) -> _SourceD3Bindings:
    """Strictly authenticate D3 and retain only tensor-free identities."""

    loaded = d0d3.load_objective_balance_diagnostic_artifact(path)
    binding = protocol.source_result
    manifest = loaded.manifest
    candidate_results = loaded.state.get("candidate_results")
    plan_states = loaded.state.get("plan_states")
    if (
        loaded.artifact_sha256 != binding.logical_artifact_sha256
        or loaded.tensor_file_sha256 != binding.tensor_sha256
        or loaded.report_sha256 != binding.report_sha256
        or manifest.get("protocol_sha256") != binding.protocol_sha256
        or manifest.get("code_bundle_sha256") != binding.code_bundle_sha256
        or manifest.get("outcome") != "no_primary_treatment_passed_fit_gates"
        or manifest.get("authorized_fresh_c3_recipe_id") is not None
        or manifest.get("candidate_plan_sha256s", {}).get(
            _SOURCE_D3_CANDIDATE_ID
        )
        != binding.d3_primary_plan_sha256
        or manifest.get("candidate_result_sha256s", {}).get(
            _SOURCE_D3_CANDIDATE_ID
        )
        != binding.d3_primary_result_sha256
        or not isinstance(candidate_results, Mapping)
        or not isinstance(plan_states, Mapping)
    ):
        raise ValueError("authenticated objective-balance source drifted")
    row = candidate_results.get(_SOURCE_D3_CANDIDATE_ID)
    source_plan = plan_states.get(_SOURCE_D3_CANDIDATE_ID)
    if not isinstance(row, Mapping) or not isinstance(source_plan, Mapping):
        raise ValueError("source D3 candidate is absent")
    pair_balance = row.get("pair_balance")
    if (
        row.get("recipe_sha256") != binding.d3_recipe_sha256
        or row.get("plan_sha256") != binding.d3_primary_plan_sha256
        or row.get("fit_data_binding_sha256")
        != binding.fit_data_binding_sha256
        or row.get("latent_rank") != protocol.baseline_latent_rank
        or row.get("seed") != protocol.primary_seed
        or row.get("seed_role") != "primary"
        or row.get("advancement_fit_gate_pass") is not False
        or not isinstance(pair_balance, Mapping)
    ):
        raise ValueError("source D3 primary identity drifted")
    sequence_sha256s: dict[str, tuple[str, ...]] = {}
    for name in _SOURCE_PLAN_SEQUENCE_FIELDS:
        values = source_plan.get(name)
        if (
            not isinstance(values, (tuple, list))
            or not values
            or any(_SHA256.fullmatch(str(value)) is None for value in values)
        ):
            raise ValueError(f"source D3 {name} is invalid")
        sequence_sha256s[name] = tuple(str(value) for value in values)
    natural = pair_balance.get("natural_pair_sha256s")
    balanced = pair_balance.get("balanced_pair_sha256s")
    if (
        not isinstance(natural, (tuple, list))
        or not isinstance(balanced, (tuple, list))
        or tuple(str(value) for value in balanced)
        != sequence_sha256s["fit_pair_sha256s"]
        or len(natural) != 48
        or len(balanced) != 56
    ):
        raise ValueError("source D3 pair sequence drifted")
    _validate_source_sequence_anchors(
        binding,
        sequences=sequence_sha256s,
        natural=tuple(str(value) for value in natural),
        balanced=tuple(str(value) for value in balanced),
    )
    shared_names = (
        "c2_protocol_sha256",
        "c2_pilot_panel_sha256",
        "c2_fit_panel_sha256",
        "c2_calibrated_fit_panel_sha256",
        "c2_calibration_sha256",
        "selected_calibration_amplitude",
        "basis_package_file_sha256",
        "basis_package_payload_sha256",
        "source_model_sha256",
        "pre_feedforward_norm_sha256",
        "canonical_metric_weight_sha256",
        "unit_rms_gauge_sha256",
        "standardized_gauge_sha256",
        "controls_sha256",
        "ordinary_gates_sha256",
        "contrast_gates_sha256",
    )
    shared_manifest = {name: manifest.get(name) for name in shared_names}
    for name, value in shared_manifest.items():
        if name == "selected_calibration_amplitude":
            if value != _EXPECTED_C2_CALIBRATION_AMPLITUDE:
                raise ValueError("source calibration amplitude drifted")
        else:
            _require_sha256(value, label=f"source {name}")
    if source_replay_binding_sha256(
        sequence_sha256s=sequence_sha256s,
        natural_pair_sha256s=tuple(str(value) for value in natural),
        balanced_pair_sha256s=tuple(str(value) for value in balanced),
        shared_manifest=shared_manifest,
    ) != binding.d3_source_replay_binding_sha256:
        raise ValueError("source D3 replay aggregate binding drifted")
    # Do not restore or return the source plan.  In particular, its encoder,
    # executor, and decoder tensors have no data path into the new fitter.
    return _SourceD3Bindings(
        fit_data_binding_sha256=str(row["fit_data_binding_sha256"]),
        sequence_sha256s=sequence_sha256s,
        natural_pair_sha256s=tuple(str(value) for value in natural),
        balanced_pair_sha256s=tuple(str(value) for value in balanced),
        shared_manifest=shared_manifest,
    )


def _validate_source_sequence_anchors(
    binding: ObjectiveBalanceResultBinding,
    *,
    sequences: Mapping[str, Sequence[str]],
    natural: Sequence[str],
    balanced: Sequence[str],
) -> None:
    if set(sequences) != set(_SOURCE_PLAN_SEQUENCE_FIELDS):
        raise ValueError("source D3 sequence table is incomplete")
    for sequence_name, field_name in _SOURCE_SEQUENCE_ANCHOR_FIELDS.items():
        computed = source_sequence_binding_sha256(
            sequence_name,
            tuple(sequences[sequence_name]),
        )
        if computed != getattr(binding, field_name):
            raise ValueError(
                f"source D3 {sequence_name} aggregate binding drifted"
            )
    if source_sequence_binding_sha256(
        "natural_pair_sha256s",
        tuple(natural),
    ) != binding.d3_natural_pair_sequence_sha256:
        raise ValueError("source D3 natural-pair aggregate binding drifted")
    if source_sequence_binding_sha256(
        "balanced_pair_sha256s",
        tuple(balanced),
    ) != binding.d3_balanced_pair_sequence_sha256:
        raise ValueError("source D3 balanced-pair aggregate binding drifted")


def describe_rank64_capacity_control() -> dict[str, object]:
    """Describe the frozen rung without opening a model or result artifact."""

    protocol, objective_protocol, c2_protocol, d3 = (
        _authenticated_declarations()
    )
    code_sha256s = _code_sha256s()
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "protocol_trust_anchor": (
            DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256
        ),
        "protocol": protocol.state_dict(),
        "source_objective_protocol_sha256": (
            objective_protocol.protocol_sha256
        ),
        "source_d3_recipe_sha256": d3.artifact_sha256,
        "c2_protocol_sha256": c2_protocol.protocol_sha256,
        "allowed_c2_roles": ("pilot", "fit"),
        "selection_role_allowed": False,
        "selection_materialization_allowed": False,
        "selection_measurement_allowed": False,
        "source_diagnostic_loading_in_describe": False,
        "source_d3_parameters_used_for_initialization": False,
        "cold_start": True,
        "controlled_change": "outer_latent_rank_16_to_64_only",
        "baseline_latent_rank": protocol.baseline_latent_rank,
        "latent_rank": protocol.latent_rank,
        "expert_rank": protocol.executor.expert_rank,
        "primary_seed": protocol.primary_seed,
        "replication_seed": protocol.replication_seed,
        "decision_rule": (
            "run_primary;stop_on_invalid_or_capability_failure;"
            "replicate_only_complete_primary_pass;"
            "require_two_seed_pass_to_open_separate_width_ladder"
        ),
        "fresh_c3_authorized": False,
        "compression_claim_authorized": False,
        "model_loaded": False,
        "pilot_materialized": False,
        "fit_materialized": False,
        "selection_materialized": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "code_sha256s": code_sha256s,
        "code_bundle_sha256": _code_bundle_sha256(code_sha256s),
    }
    d0d3._assert_tensor_free_report(report)
    return report


def _plan_sequence_comparison(
    plan: ContrastAwareReferenceProviderPlan,
    *,
    source: _SourceD3Bindings,
) -> dict[str, object]:
    flags = {
        name: tuple(getattr(plan, name)) == source.sequence_sha256s[name]
        for name in _SOURCE_PLAN_SEQUENCE_FIELDS
    }
    return {
        "passed": all(flags.values()),
        "flags": flags,
        "source_sequence_sha256s": dict(source.sequence_sha256s),
        "candidate_sequence_sha256s": {
            name: tuple(getattr(plan, name))
            for name in _SOURCE_PLAN_SEQUENCE_FIELDS
        },
        "comparison_semantics": (
            "exact_ordered_batch_content_index_endpoint_and_pair_hashes"
        ),
    }


def _evaluate_capacity_candidate(
    *,
    seed_role: str,
    seed: int,
    protocol: Rank64CapacityControlProtocol,
    objective_protocol: ObjectiveBalanceDiagnosticProtocol,
    d3_recipe: ObjectiveBalanceRecipe,
    source: _SourceD3Bindings,
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
    controls: FullWidthReferenceControls,
    standardized_gauge_sha256: str,
    fidelity_gates: SyntheticReferenceGates,
    contrast_gates: ContrastAssessmentGates,
) -> _CapacityEvaluation:
    if seed_role not in {"primary", "replication"}:
        raise ValueError("rank-64 capacity seed role is invalid")
    expected_seed = (
        protocol.primary_seed
        if seed_role == "primary"
        else protocol.replication_seed
    )
    if seed != expected_seed:
        raise ValueError("rank-64 capacity seed differs from declaration")
    training = protocol.training
    objective = _objective(training)
    if objective.artifact_sha256 != d0d3._objective_for_recipe(
        d3_recipe
    ).artifact_sha256:
        raise RuntimeError("rank-64 objective differs from source D3")
    training_metric = unit_rms_gauge.metric_weight
    training_teacher_energy = d0d3._fit_teacher_weighted_energy(
        fit,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=training_metric,
    )
    training_teacher_signal_diagnostics = (
        d0d3._teacher_signal_diagnostics(
            fit,
            natural_pairs,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=training_metric,
        )
    )
    fit_batches = c2._indexed_batches(
        fit,
        split="fit",
        binding_sha256=fit_data_binding_sha256,
    )
    fit_pairs, pair_balance = d0d3._balance_training_pairs(
        natural_pairs,
        recipe=d3_recipe,
    )
    if (
        tuple(pair_balance["natural_pair_sha256s"])
        != source.natural_pair_sha256s
        or tuple(pair_balance["balanced_pair_sha256s"])
        != source.balanced_pair_sha256s
        or tuple(value.artifact_sha256 for value in fit_pairs)
        != source.balanced_pair_sha256s
    ):
        raise ValueError("current D3 pair order differs from source")
    provider_binding_sha256 = _provider_binding_sha256(
        protocol=protocol,
        c2_protocol_sha256=c2_protocol.protocol_sha256,
        calibration_sha256=calibration.artifact_sha256,
        basis_payload_sha256=basis.basis_payload_sha256,  # type: ignore[attr-defined]
        source_model_sha256=basis.source_model_sha256,  # type: ignore[attr-defined]
        norm_sha256=norm_sha256,
        training_metric_weight=training_metric,
        target_center=target_center,
        target_scale=target_scale,
    )
    # There is intentionally no source-plan or initialization argument here.
    # ``fit_contrast_aware_reference_provider`` creates a fresh model from the
    # frozen seed, making source D3 weights structurally unable to enter.
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
        executor_config=_executor_config(protocol),
        objective=objective,
        fisher_metric_weight=training_metric,
        steps=training.steps,
        learning_rate=training.learning_rate,
        seed=seed,
    )
    plan_state = d0d3._round_trip_plan_state(plan)
    sequence_comparison = _plan_sequence_comparison(plan, source=source)
    support_radius = c2._feature_radius(plan, fit)
    candidate_id = f"r64_d3_capacity_control.{seed_role}"
    (
        candidate,
        ordinary_score,
        predictions,
        structural_metadata,
    ) = d0d3._fit_only_ordinary_candidate_and_score(
        candidate_id=candidate_id,
        plan=plan,
        measured=fit,
        ordinary_probes=ordinary_probes,  # type: ignore[arg-type]
        controls=controls,
        metric_weight=raw_metric_weight,
        standardized_gauge_sha256=standardized_gauge_sha256,
        support_radius=support_radius,
        gates=fidelity_gates,
    )
    contrast_result, identities, coverage = d0d3._fit_contrast_assessment(
        protocol=c2_protocol,
        measured=fit,
        predictions=predictions,
        metric_weight=raw_metric_weight,
        gates=contrast_gates,
        required_null_candidate_pass_count=(
            objective_protocol.gates.required_null_candidate_pass_count
        ),
    )
    balance_gate = d0d3._contribution_balance_gate(
        plan,
        recipe=d3_recipe,
        gates=objective_protocol.gates,
        training_teacher_energy=training_teacher_energy,
        raw_teacher_energy=raw_teacher_energy,
        teacher_signal_diagnostics=(
            training_teacher_signal_diagnostics
        ),
    )
    executor_config = plan.executor_config
    exact_config = asdict(executor_config) == asdict(
        _executor_config(protocol)
    )
    treatment_flags = {
        "objective_contribution_balance": bool(balance_gate["passed"]),
        "exact_source_fit_sequences": bool(
            sequence_comparison["passed"]
        ),
        "exact_rank64_executor_config": exact_config,
        "exact_d3_objective": (
            plan.objective.artifact_sha256 == objective.artifact_sha256
        ),
        "exact_steps": plan.training_steps == training.steps,
        "exact_learning_rate": (
            plan.learning_rate == training.learning_rate
        ),
        "exact_seed": plan.seed == seed,
        "exact_unit_rms_training_metric": (
            plan.fisher_metric_supplied
            and torch.equal(
                plan.fisher_metric_weight,
                training_metric.to(dtype=torch.float64),
            )
        ),
        "exact_fit_data_binding": (
            plan.synthetic_binding_sha256
            == source.fit_data_binding_sha256
            == fit_data_binding_sha256
        ),
        "cold_start_source_weights_unused": (
            source.source_plan_parameters_used is False
        ),
    }
    treatment_valid = all(treatment_flags.values())

    ordinary_gate_state = ordinary_score.gate_flags.state_dict()
    ordinary_gate_values = tuple(
        passed
        for name, passed in ordinary_gate_state.items()
        if name != "all_passed"
    )
    ordinary_contract = (
        len(ordinary_gate_values)
        == objective_protocol.gates.required_ordinary_gate_count
        and all(ordinary_gate_values)
    )
    sensitivity_contract = bool(
        coverage["every_teacher_qualified_contrast_passed"]
    )
    family_contract = contrast_result.overall_status == "pass"
    fit_capability_pass = (
        ordinary_contract
        and sensitivity_contract
        and family_contract
        and bool(coverage["all_families_cover_all_four_rank_bands"])
        and bool(coverage["required_null_contrasts_valid_and_passed"])
    )
    combined_pass = treatment_valid and fit_capability_pass
    final_audit = audit_objective_contributions(
        plan.final_metrics,
        objective,
    )
    failure_reasons = list(
        d0d3._candidate_failure_reasons(
            ordinary_score=ordinary_score,
            contrast_result=contrast_result,
            coverage=coverage,
            balance_gate_passed=bool(balance_gate["passed"]),
        )
    )
    failure_reasons.extend(
        f"treatment_validity:{name}"
        for name, passed in treatment_flags.items()
        if not passed
    )
    row = {
        "candidate_id": candidate_id,
        "treatment_id": "r64_d3_capacity_control",
        "source_recipe_id": d3_recipe.recipe_id,
        "source_recipe_sha256": d3_recipe.artifact_sha256,
        "training_sha256": training.artifact_sha256,
        "seed_role": seed_role,
        "seed": seed,
        "baseline_latent_rank": protocol.baseline_latent_rank,
        "latent_rank": plan.latent_rank,
        "expert_rank": executor_config.expert_rank,
        "cold_start": True,
        "source_plan_parameters_used_for_initialization": False,
        "training_metric": training.training_metric,
        "training_metric_weight_sha256": _tensor_sha256(
            training_metric
        ),
        "canonical_scoring_metric_weight_sha256": _tensor_sha256(
            raw_metric_weight
        ),
        "pair_balance": pair_balance,
        "training_teacher_signal_diagnostics": (
            training_teacher_signal_diagnostics
        ),
        "provider_binding_sha256": provider_binding_sha256,
        "fit_data_binding_sha256": fit_data_binding_sha256,
        "fit_data_binding_recipe_independent": True,
        "source_fit_sequence_comparison": sequence_comparison,
        "plan_sha256": plan.artifact_sha256,
        "plan_round_trip_passed": True,
        "accounting": asdict(plan.accounting()),
        "execution_accounting": d0d3._fit_execution_accounting(
            plan,
            fit_batches,
        ),
        "initial_training_metrics": plan.initial_metrics.state_dict(),
        "final_training_metrics": plan.final_metrics.state_dict(),
        "final_contribution_audit": final_audit.state_dict(),
        "objective_balance_gate": balance_gate,
        "treatment_validity": {
            "passed": treatment_valid,
            "flags": treatment_flags,
            "failure_semantics": (
                "invalid_rank_comparison_no_capacity_conclusion"
            ),
        },
        "fit_capability_contract": {
            "ordinary_gate_count": len(ordinary_gate_values),
            "required_ordinary_gate_count": (
                objective_protocol.gates.required_ordinary_gate_count
            ),
            "all_ordinary_gates_passed": ordinary_contract,
            "every_eligible_sensitivity_passed": sensitivity_contract,
            "all_contrast_families_formally_passed": family_contract,
            "all_families_cover_all_four_rank_bands": bool(
                coverage["all_families_cover_all_four_rank_bands"]
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
        "fit_capability_pass": fit_capability_pass,
        "combined_pass": combined_pass,
        "failure_reasons": tuple(sorted(set(failure_reasons))),
        "candidate_binding_sha256": candidate.artifact_sha256,
        "contains_raw_fit_targets": False,
        "contains_teacher_jvp_tensors": False,
        "contains_provider_chart_tensors": False,
        "_plan_state": plan_state,
    }
    return _CapacityEvaluation(
        candidate_id=candidate_id,
        seed_role=seed_role,
        seed=seed,
        treatment_valid=treatment_valid,
        fit_capability_pass=fit_capability_pass,
        combined_pass=combined_pass,
        row=row,
        plan=plan,
    )


def _execute_capacity_schedule(
    protocol: Rank64CapacityControlProtocol,
    *,
    evaluate: object,
) -> tuple[_CapacityEvaluation, _CapacityEvaluation | None]:
    if not callable(evaluate):
        raise TypeError("rank-64 evaluator must be callable")
    primary = evaluate("primary", protocol.primary_seed)
    if (
        not isinstance(primary, _CapacityEvaluation)
        or primary.seed_role != "primary"
        or primary.seed != protocol.primary_seed
        or primary.combined_pass
        is not (
            primary.treatment_valid and primary.fit_capability_pass
        )
    ):
        raise RuntimeError("rank-64 primary evaluation identity drifted")
    if not (primary.treatment_valid and primary.fit_capability_pass):
        return primary, None
    replication = evaluate("replication", protocol.replication_seed)
    if (
        not isinstance(replication, _CapacityEvaluation)
        or replication.seed_role != "replication"
        or replication.seed != protocol.replication_seed
        or replication.combined_pass
        is not (
            replication.treatment_valid
            and replication.fit_capability_pass
        )
    ):
        raise RuntimeError("rank-64 replication identity drifted")
    return primary, replication


def _capacity_decision(
    primary: _CapacityEvaluation,
    replication: _CapacityEvaluation | None,
) -> dict[str, object]:
    if not primary.treatment_valid:
        outcome = "invalid_rank_comparison_primary_treatment_validity_failed"
    elif not primary.fit_capability_pass:
        outcome = "rank64_primary_fit_capability_failed"
    elif replication is None:
        raise RuntimeError("passing primary did not run replication")
    elif not replication.treatment_valid:
        outcome = (
            "invalid_rank_comparison_replication_treatment_validity_failed"
        )
    elif not replication.fit_capability_pass:
        outcome = "rank64_replication_fit_capability_failed"
    else:
        outcome = "rank64_two_seed_fit_capability_pass"
    two_seed_pass = outcome == "rank64_two_seed_fit_capability_pass"
    return {
        "outcome": outcome,
        "primary_treatment_valid": primary.treatment_valid,
        "primary_fit_capability_passed": primary.fit_capability_pass,
        "replication_executed": replication is not None,
        "replication_treatment_valid": (
            None if replication is None else replication.treatment_valid
        ),
        "replication_fit_capability_passed": (
            None if replication is None else replication.fit_capability_pass
        ),
        "two_seed_fit_capability_passed": two_seed_pass,
        "compressed_width_ladder_preregistration_supported": two_seed_pass,
        "fresh_c3_authorized": False,
        "compression_claim_authorized": False,
    }


def _validate_output_path(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("rank-64 capacity output must use a .pt suffix")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite rank-64 capacity output")
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


def _publish_artifact(
    state: Mapping[str, object],
    report_payload: Mapping[str, object],
    *,
    output: Path,
) -> dict[str, object]:
    d0d3._assert_safe_artifact_tree(state)
    d0d3._assert_safe_artifact_tree(report_payload, path="report")
    d0d3._assert_tensor_free_report(report_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite rank-64 capacity output")
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
        d0d3._assert_tensor_free_report(report)
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


def _restore_and_validate_contrast_score(
    raw: Mapping[str, object],
    *,
    gates: ContrastAssessmentGates,
) -> object:
    tuple_fields = {
        "teacher_endpoint_sha256s",
        "repeated_teacher_endpoint_sha256s",
        "candidate_endpoint_sha256s",
        "repeated_candidate_endpoint_sha256s",
        "reason_codes",
    }
    values: dict[str, object] = {}
    for name in contrast_assessment.ContrastScore.__dataclass_fields__:
        value = raw[name]
        if name in tuple_fields:
            value = tuple(value)  # type: ignore[arg-type]
        elif name == "candidate_gate_flags":
            value = tuple(
                (str(pair[0]), bool(pair[1]))
                for pair in value  # type: ignore[union-attr]
            )
        values[name] = value
    score = contrast_assessment.ContrastScore(**values)
    if (
        score.gates_sha256 != gates.artifact_sha256
        or score.role not in {
            "expected_sensitivity",
            "intended_null",
        }
        or score.teacher_effective_noise_l2
        != max(
            score.teacher_repeat_noise_l2,
            score.teacher_numeric_floor_l2,
        )
    ):
        raise ValueError("rank-64 contrast score gates drifted")
    candidate_optional = (
        "candidate_contrast_l2",
        "candidate_repeat_noise_l2",
        "candidate_numeric_floor_l2",
        "candidate_effective_noise_l2",
        "candidate_contrast_relative_error",
        "candidate_direction_cosine",
        "candidate_projection_gain",
        "candidate_orthogonal_leakage",
        "candidate_magnitude_ratio",
        "candidate_null_relative_effect_upper",
        "candidate_null_relative_error_upper",
    )
    if score.candidate_scored:
        if (
            score.candidate_contrast_l2 is None
            or score.candidate_repeat_noise_l2 is None
            or score.candidate_numeric_floor_l2
            != score.teacher_numeric_floor_l2
            or score.candidate_effective_noise_l2
            != max(
                score.candidate_repeat_noise_l2,
                score.candidate_numeric_floor_l2,
            )
        ):
            raise ValueError("rank-64 candidate noise accounting drifted")
    elif (
        any(getattr(score, name) is not None for name in candidate_optional)
        or score.candidate_gate_flags
    ):
        raise ValueError("unscored rank-64 contrast contains candidate metrics")

    uncertainty = (
        gates.repeat_noise_multiplier * score.teacher_effective_noise_l2
    )
    invalid_baseline = score.teacher_baseline_l2 <= max(
        score.teacher_numeric_floor_l2,
        torch.finfo(torch.float64).tiny,
    )
    reasons: list[str] = []
    expected_scored = False
    if invalid_baseline:
        expected_teacher_status = "invalid"
        expected_decision = "invalid"
        expected_candidate_status = "not_scored"
        reasons.append("teacher_baseline_is_numerically_unresolved")
    else:
        lower = max(
            score.teacher_contrast_l2 - uncertainty,
            0.0,
        ) / score.teacher_baseline_l2
        upper = (
            score.teacher_contrast_l2 + uncertainty
        ) / score.teacher_baseline_l2
        if (
            score.teacher_relative_effect_lower != lower
            or score.teacher_relative_effect_upper != upper
        ):
            raise ValueError("rank-64 teacher effect interval drifted")
        if score.role == "expected_sensitivity":
            if score.teacher_contrast_l2 < uncertainty:
                expected_teacher_status = (
                    "numerically_unresolved_sensitivity"
                )
                expected_decision = "panel_inconclusive"
                reasons.append(
                    "teacher_sensitivity_not_numerically_resolved"
                )
            elif lower >= gates.minimum_sensitivity_relative_effect:
                expected_teacher_status = "eligible_sensitivity"
                expected_decision = "pass"
                expected_scored = True
            elif upper < gates.minimum_sensitivity_relative_effect:
                expected_teacher_status = "underpowered_sensitivity"
                expected_decision = "panel_inconclusive"
                reasons.append("teacher_sensitivity_below_effect_floor")
            else:
                expected_teacher_status = (
                    "boundary_inconclusive_sensitivity"
                )
                expected_decision = "panel_inconclusive"
                reasons.append(
                    "teacher_sensitivity_interval_crosses_effect_floor"
                )
        elif upper <= gates.maximum_teacher_null_relative_effect:
            expected_teacher_status = "valid_intended_null"
            expected_decision = "pass"
            expected_scored = True
        elif lower > gates.maximum_teacher_null_relative_effect:
            expected_teacher_status = "violated_intended_null"
            expected_decision = "teacher_null_failure"
            reasons.append("teacher_null_effect_exceeds_ceiling")
        else:
            expected_teacher_status = "boundary_inconclusive_null"
            expected_decision = "panel_inconclusive"
            reasons.append("teacher_null_interval_crosses_ceiling")
        expected_candidate_status = "not_scored"

    expected_flags: tuple[tuple[str, bool], ...] = ()
    if expected_scored and score.role == "expected_sensitivity":
        expected_candidate_status = "pass"
        resolved = (
            score.candidate_contrast_l2 is not None
            and score.candidate_effective_noise_l2 is not None
            and score.candidate_contrast_l2
            > gates.repeat_noise_multiplier
            * score.candidate_effective_noise_l2
        )
        expected_flags = tuple(
            sorted(
                {
                    "contrast_relative_error": (
                        score.candidate_contrast_relative_error is not None
                        and score.candidate_contrast_relative_error
                        <= gates.maximum_sensitivity_contrast_relative_error
                    ),
                    "direction_cosine": (
                        resolved
                        and score.candidate_direction_cosine is not None
                        and score.candidate_direction_cosine
                        >= gates.minimum_sensitivity_direction_cosine
                    ),
                    "projection_gain": (
                        score.candidate_projection_gain is not None
                        and gates.minimum_sensitivity_projection_gain
                        <= score.candidate_projection_gain
                        <= gates.maximum_sensitivity_projection_gain
                    ),
                    "orthogonal_leakage": (
                        score.candidate_orthogonal_leakage is not None
                        and score.candidate_orthogonal_leakage
                        <= gates.maximum_sensitivity_orthogonal_leakage
                    ),
                }.items()
            )
        )
        if not resolved:
            reasons.append("candidate_sensitivity_not_numerically_resolved")
        if not all(value for _, value in expected_flags):
            expected_candidate_status = "fail"
            expected_decision = "candidate_fail"
            reasons.extend(
                f"candidate_failed_{name}"
                for name, passed in expected_flags
                if not passed
            )
    elif expected_scored:
        expected_candidate_status = "pass"
        expected_flags = tuple(
            sorted(
                {
                    "null_relative_effect": (
                        score.candidate_null_relative_effect_upper is not None
                        and score.candidate_null_relative_effect_upper
                        <= gates.maximum_candidate_null_relative_effect
                    ),
                    "null_relative_error": (
                        score.candidate_null_relative_error_upper is not None
                        and score.candidate_null_relative_error_upper
                        <= gates.maximum_candidate_null_relative_error
                    ),
                }.items()
            )
        )
        if not all(value for _, value in expected_flags):
            expected_candidate_status = "fail"
            expected_decision = "candidate_fail"
            reasons.extend(
                f"candidate_failed_{name}"
                for name, passed in expected_flags
                if not passed
            )
    if (
        score.teacher_status != expected_teacher_status
        or score.candidate_scored is not expected_scored
        or score.candidate_gate_flags != expected_flags
        or score.candidate_status != expected_candidate_status
        or score.decision_status != expected_decision
        or score.reason_codes != tuple(sorted(set(reasons)))
    ):
        raise ValueError("rank-64 contrast score decision drifted")
    return score


def _validate_contrast_result_state(
    raw: object,
    *,
    gates: ContrastAssessmentGates,
) -> tuple[Mapping[str, object], tuple[object, ...]]:
    if not isinstance(raw, Mapping):
        raise TypeError("rank-64 contrast result must be a mapping")
    state = dict(raw)
    scores = state.get("contrast_scores")
    families = state.get("family_scores")
    if not isinstance(scores, (tuple, list)) or not isinstance(
        families,
        (tuple, list),
    ):
        raise TypeError("rank-64 contrast result rows must be sequences")
    score_hashes: list[str] = []
    restored_scores: list[object] = []
    for score in scores:
        if not isinstance(score, Mapping):
            raise TypeError("rank-64 contrast score must be a mapping")
        payload = dict(score)
        supplied = payload.pop("artifact_sha256", None)
        computed = contrast_assessment._digest(
            payload,
            domain=contrast_assessment._SCORE_DOMAIN,
        )
        if supplied != computed:
            raise ValueError("rank-64 contrast score hash mismatch")
        score_hashes.append(computed)
        restored_scores.append(
            _restore_and_validate_contrast_score(score, gates=gates)
        )
    family_hashes: list[str] = []
    for family in families:
        if not isinstance(family, Mapping):
            raise TypeError("rank-64 contrast family must be a mapping")
        payload = dict(family)
        supplied = payload.pop("artifact_sha256", None)
        computed = contrast_assessment._digest(
            payload,
            domain=contrast_assessment._FAMILY_DOMAIN,
        )
        if supplied != computed:
            raise ValueError("rank-64 contrast family hash mismatch")
        family_hashes.append(computed)
    payload = dict(state)
    supplied = payload.pop("artifact_sha256", None)
    payload.pop("contrast_scores", None)
    payload.pop("family_scores", None)
    if (
        tuple(payload.get("contrast_score_sha256s", ()))
        != tuple(score_hashes)
        or tuple(payload.get("family_score_sha256s", ()))
        != tuple(family_hashes)
        or supplied
        != contrast_assessment._digest(
            payload,
            domain=contrast_assessment._ASSESSMENT_DOMAIN,
        )
    ):
        raise ValueError("rank-64 contrast assessment hash mismatch")
    by_family: dict[str, list[object]] = {}
    for score in restored_scores:
        by_family.setdefault(score.family, []).append(score)  # type: ignore[attr-defined]
    recomputed_families = tuple(
        contrast_assessment._family_score(
            family,
            tuple(
                sorted(
                    values,
                    key=lambda value: value.contrast_id,  # type: ignore[attr-defined]
                )
            ),
            gates=gates,
        )
        for family, values in sorted(by_family.items())
    )
    if _canonical_json_bytes(
        tuple(value.state_dict() for value in recomputed_families)
    ) != _canonical_json_bytes(families):
        raise ValueError("rank-64 contrast family aggregation drifted")
    counts = {
        status: sum(
            value.decision_status == status
            for value in recomputed_families
        )
        for status in contrast_assessment._DECISION_PRIORITY
    }
    overall = max(
        (value.decision_status for value in recomputed_families),
        key=lambda status: contrast_assessment._DECISION_PRIORITY[status],
    )
    reasons = tuple(
        sorted(
            {
                f"{family.family}:{reason}"
                for family in recomputed_families
                for reason in family.reason_codes
            }
        )
    )
    expected_aggregate = {
        "gates_sha256": gates.artifact_sha256,
        "invalid_family_count": counts["invalid"],
        "teacher_null_failure_family_count": counts[
            "teacher_null_failure"
        ],
        "panel_inconclusive_family_count": counts["panel_inconclusive"],
        "candidate_failed_family_count": counts["candidate_fail"],
        "passed_family_count": counts["pass"],
        "overall_status": overall,
        "reason_codes": reasons,
        "weak_teacher_contrasts_entered_candidate_relative_metrics": False,
        "intended_null_contrasts_entered_direction_metrics": False,
    }
    for name, expected in expected_aggregate.items():
        if (
            _canonical_json_bytes(state.get(name))
            != _canonical_json_bytes(expected)
        ):
            raise ValueError(
                "rank-64 contrast assessment aggregation drifted"
            )
    return state, tuple(restored_scores)


def _recompute_contrast_coverage(
    scores: Sequence[object],
    identities: object,
    *,
    required_null_candidate_pass_count: int,
) -> dict[str, object]:
    if not isinstance(identities, Mapping):
        raise TypeError("rank-64 contrast identities must be a mapping")
    by_id = {
        value.contrast_id: value  # type: ignore[attr-defined]
        for value in scores
    }
    if set(by_id) != set(identities):
        raise ValueError("rank-64 contrast identities differ from scores")
    family_coverage: dict[str, dict[str, object]] = {}
    for family, intent, planned in (
        ("radial_sensitivity", "sensitivity", 16),
        ("signed_sensitivity", "sensitivity", 8),
        ("null_invariance", "invariance", 24),
    ):
        ids = tuple(
            key
            for key, identity in identities.items()
            if isinstance(identity, Mapping)
            and identity.get("family") == family
        )
        if len(ids) != planned:
            raise ValueError("rank-64 contrast family count drifted")
        qualified_status = (
            "eligible_sensitivity"
            if intent == "sensitivity"
            else "valid_intended_null"
        )
        qualified = tuple(
            key
            for key in ids
            if by_id[key].teacher_status == qualified_status  # type: ignore[attr-defined]
        )
        candidate_passes = tuple(
            key
            for key in qualified
            if by_id[key].decision_status == "pass"  # type: ignore[attr-defined]
        )
        bands = tuple(
            sorted(
                {
                    identities[key]["rank_band"]  # type: ignore[index]
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
    return {
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
            and family_coverage["null_invariance"]["candidate_pass_count"]
            == required_null_candidate_pass_count
        ),
        "required_null_candidate_pass_count": (
            required_null_candidate_pass_count
        ),
    }


def _expected_contrast_identities(
    protocol: ContrastProviderDevelopmentProtocol,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for group in protocol.groups_for_role("fit"):
        for index, (left_id, right_id) in enumerate(
            group.canonical_variant_pairs
        ):
            contrast_id = f"{group.group_id}.pair.{index:02d}"
            result[contrast_id] = {
                "group_id": group.group_id,
                "family": group.family,
                "intent": group.intent,
                "rank_band": group.rank_band,
                "left_probe_id": left_id,
                "right_probe_id": right_id,
            }
    return result


def _recompute_ordinary_gate_flags(
    score: FullWidthCandidateScore,
    gates: SyntheticReferenceGates,
) -> dict[str, bool]:
    structural = score.structural_metrics
    return {
        "fisher_weighted_relative_error": (
            score.fisher_weighted_relative_error
            <= gates.maximum_fisher_weighted_relative_error
        ),
        "reference_cosine": (
            score.reference_cosine >= gates.minimum_reference_cosine
        ),
        "error_reduction_vs_constant": (
            score.error_reduction_vs_constant
            >= gates.minimum_error_reduction_vs_constant
        ),
        "error_reduction_vs_position_only": (
            score.error_reduction_vs_position_only
            >= gates.minimum_error_reduction_vs_position_only
        ),
        "per_probe_p90_relative_error": (
            score.maximum_per_probe_p90_relative_error
            <= gates.maximum_per_probe_p90_relative_error
        ),
        "worst_family_relative_error": (
            score.worst_family_relative_error
            <= gates.maximum_worst_panel_relative_error
        ),
        "prepared_vs_analytic_relative_error": (
            structural.prepared_vs_analytic_relative_error
            <= gates.maximum_prepared_vs_analytic_relative_error
        ),
        "causality_violation": (
            structural.causality_violation
            <= gates.maximum_causality_violation
        ),
        "padding_violation": (
            structural.padding_violation <= gates.maximum_padding_violation
        ),
        "repeat_relative_error": (
            structural.repeat_relative_error
            <= gates.maximum_repeat_relative_error
        ),
        "collision_target_relative_difference": (
            score.minimum_collision_target_relative_difference
            >= gates.minimum_collision_target_relative_difference
        ),
        "in_support_fraction": (
            structural.in_support_fraction
            >= gates.minimum_in_support_fraction
        ),
    }


def load_rank64_capacity_control_artifact(
    path: Path | str,
) -> LoadedRank64CapacityControlArtifact:
    """Load and authenticate a tensor/report pair with weights-only loading."""

    source = Path(path).expanduser().resolve()
    if source.suffix != ".pt":
        raise ValueError("rank-64 capacity artifact must use a .pt suffix")
    tensor_payload = _read_regular_file(
        source,
        label="rank-64 capacity artifact",
    )
    report_path = source.with_suffix(".json")
    report_payload = _read_regular_file(
        report_path,
        label="rank-64 capacity report",
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
            "rank-64 capacity tensor fields differ from frozen format"
        )
    d0d3._assert_safe_artifact_tree(raw)
    manifest = raw["manifest"]
    if not isinstance(manifest, Mapping):
        raise TypeError("rank-64 capacity manifest must be a mapping")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("rank-64 capacity manifest fields drifted")
    logical_sha256 = _json_sha256(manifest, domain=_ARTIFACT_DOMAIN)
    if (
        raw["artifact_sha256"] != logical_sha256
        or manifest.get("schema") != _SCHEMA
        or manifest.get("format_version") != _FORMAT_VERSION
    ):
        raise ValueError("rank-64 capacity logical artifact binding mismatch")
    try:
        report = json.loads(report_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "rank-64 capacity report is not canonical JSON"
        ) from exc
    if not isinstance(report, Mapping):
        raise TypeError("rank-64 capacity report must be a mapping")
    d0d3._assert_safe_artifact_tree(report, path="report")
    d0d3._assert_tensor_free_report(report)
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
        raise ValueError("rank-64 capacity report SHA-256 mismatch")
    report_artifact = report.get("artifact")
    if not isinstance(report_artifact, Mapping) or set(report_artifact) != {
        "tensor_file",
        "tensor_file_sha256",
        "tensor_file_bytes",
        "report_file",
        "committable",
    }:
        raise ValueError("rank-64 capacity report artifact binding is invalid")
    if (
        report_artifact.get("tensor_file") != str(source)
        or report_artifact.get("report_file") != str(report_path)
        or report_artifact.get("tensor_file_sha256")
        != tensor_file_sha256
        or report_artifact.get("tensor_file_bytes") != len(tensor_payload)
        or report_artifact.get("committable") is not False
    ):
        raise ValueError("rank-64 report does not bind its tensor file")
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
        "source_d3_binding",
        "candidate_results",
        "interpretation",
        "safety",
        "artifact",
        "report_sha256",
    }
    if set(report) != set(manifest) | report_extra_keys:
        raise ValueError(
            "rank-64 capacity report fields differ from frozen format"
        )
    for name, value in manifest.items():
        if _canonical_json_bytes(report.get(name)) != (
            _canonical_json_bytes(value)
        ):
            raise ValueError(
                f"rank-64 report manifest field {name!r} drifted"
            )

    code_sha256s = manifest.get("code_sha256s")
    if (
        not isinstance(code_sha256s, Mapping)
        or dict(code_sha256s) != _code_sha256s()
        or manifest.get("code_bundle_sha256")
        != _code_bundle_sha256(dict(code_sha256s))
    ):
        raise ValueError("rank-64 capacity code binding differs from live code")
    protocol_state = raw["protocol_state"]
    if not isinstance(protocol_state, Mapping):
        raise TypeError("rank-64 protocol state must be a mapping")
    protocol = Rank64CapacityControlProtocol.from_state_dict(protocol_state)
    if (
        protocol.protocol_sha256
        != DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256
        or manifest.get("protocol_sha256") != protocol.protocol_sha256
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
        raise ValueError("rank-64 capacity protocol binding mismatch")
    source_result = protocol.source_result
    source_manifest_bindings = {
        "source_objective_protocol_sha256": (
            source_result.protocol_sha256
        ),
        "source_objective_result_binding_sha256": (
            source_result.artifact_sha256
        ),
        "source_objective_logical_artifact_sha256": (
            source_result.logical_artifact_sha256
        ),
        "source_objective_tensor_file_sha256": source_result.tensor_sha256,
        "source_objective_report_sha256": source_result.report_sha256,
        "source_objective_code_bundle_sha256": (
            source_result.code_bundle_sha256
        ),
        "source_d3_recipe_sha256": source_result.d3_recipe_sha256,
        "source_d3_primary_plan_sha256": (
            source_result.d3_primary_plan_sha256
        ),
        "source_d3_primary_result_sha256": (
            source_result.d3_primary_result_sha256
        ),
        "source_d3_fit_data_binding_sha256": (
            source_result.fit_data_binding_sha256
        ),
    }
    if any(
        manifest.get(name) != expected
        for name, expected in source_manifest_bindings.items()
    ):
        raise ValueError("rank-64 source-D3 manifest binding mismatch")
    (
        live_protocol,
        objective_protocol,
        live_c2_protocol,
        d3_recipe,
    ) = _authenticated_declarations()
    if (
        live_protocol.protocol_sha256 != protocol.protocol_sha256
        or live_c2_protocol.protocol_sha256
        != manifest.get("c2_protocol_sha256")
    ):
        raise ValueError("rank-64 live declarations differ from artifact")
    ordinary_gates = _deferred_collision_gates(
        SyntheticReferenceGates()
    )
    contrast_gates = ContrastAssessmentGates()
    if (
        full_width_reference_gates_sha256(ordinary_gates)
        != protocol.training.ordinary_gates_sha256
        or manifest.get("ordinary_gates_sha256")
        != protocol.training.ordinary_gates_sha256
        or contrast_gates.artifact_sha256
        != protocol.training.contrast_gates_sha256
        or manifest.get("contrast_gates_sha256")
        != protocol.training.contrast_gates_sha256
        or manifest.get("fit_data_binding_sha256")
        != protocol.training.fit_data_binding_sha256
    ):
        raise ValueError("rank-64 fit gate or data declaration drifted")
    calibration_state = raw["calibration_state"]
    controls_state = raw["controls_state"]
    calibration = d0d3._restore_calibration_binding(calibration_state)
    controls = d0d3._restore_full_width_controls(controls_state)
    if (
        calibration.artifact_sha256
        != manifest.get("c2_calibration_sha256")
        or calibration.selected_amplitude
        != manifest.get("selected_calibration_amplitude")
        or controls.artifact_sha256 != manifest.get("controls_sha256")
        or controls.standardized_gauge_sha256
        != manifest.get("standardized_gauge_sha256")
        or _canonical_json_bytes(report.get("calibration"))
        != _canonical_json_bytes(calibration_state)
    ):
        raise ValueError("rank-64 capacity calibration/control binding mismatch")
    gauge_state = raw["unit_rms_gauge_state"]
    gauge = UnitRmsFisherGauge.from_state_dict(gauge_state)
    metric_weight = raw["canonical_metric_weight"]
    if not isinstance(metric_weight, Tensor):
        raise TypeError("rank-64 canonical metric must be a tensor")
    gauge.validate_source(metric_weight)
    if (
        gauge.artifact_sha256 != manifest.get("unit_rms_gauge_sha256")
        or metric_weight.shape != (_MODAL_WIDTH,)
        or not bool(torch.isfinite(metric_weight).all())
        or d0d3._tensor_sha256(metric_weight)
        != manifest.get("canonical_metric_weight_sha256")
    ):
        raise ValueError("rank-64 capacity Fisher gauge binding mismatch")

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
        raise TypeError("rank-64 capacity candidate tables are invalid")
    executed_ids = tuple(executed_ids_raw)
    if (
        executed_ids
        not in (
            ("r64_d3_capacity_control.primary",),
            (
                "r64_d3_capacity_control.primary",
                "r64_d3_capacity_control.replication",
            ),
        )
        or set(plan_states) != set(executed_ids)
        or set(candidate_results) != set(executed_ids)
        or set(plan_sha256s) != set(executed_ids)
        or set(result_sha256s) != set(executed_ids)
    ):
        raise ValueError("rank-64 capacity candidate tables are incomplete")
    for candidate_id in executed_ids:
        plan_state = plan_states[candidate_id]
        row = candidate_results[candidate_id]
        if not isinstance(plan_state, Mapping) or not isinstance(row, Mapping):
            raise TypeError("rank-64 candidate state has invalid types")
        if set(row) != _CANDIDATE_ROW_FIELDS:
            raise ValueError("rank-64 candidate row fields drifted")
        plan = ContrastAwareReferenceProviderPlan.from_state_dict(plan_state)
        plan.validate_integrity()
        expected_role = (
            "primary" if candidate_id.endswith(".primary") else "replication"
        )
        expected_seed = (
            protocol.primary_seed
            if expected_role == "primary"
            else protocol.replication_seed
        )
        if (
            plan.artifact_sha256 != plan_sha256s[candidate_id]
            or row.get("plan_sha256") != plan.artifact_sha256
            or plan.latent_rank != protocol.latent_rank
            or asdict(plan.executor_config) != asdict(_executor_config(protocol))
            or row.get("candidate_id") != candidate_id
            or row.get("seed_role") != expected_role
            or row.get("seed") != expected_seed
            or row.get("source_plan_parameters_used_for_initialization")
            is not False
            or row.get("cold_start") is not True
            or row.get("fit_data_binding_recipe_independent") is not True
            or row.get("plan_round_trip_passed") is not True
            or row.get("contains_raw_fit_targets") is not False
            or row.get("contains_teacher_jvp_tensors") is not False
            or row.get("contains_provider_chart_tensors") is not False
            or _json_sha256(row, domain=_ARTIFACT_DOMAIN)
            != result_sha256s[candidate_id]
        ):
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} binding mismatch"
            )
        if (
            row.get("baseline_latent_rank")
            != protocol.baseline_latent_rank
            or row.get("latent_rank") != protocol.latent_rank
            or row.get("expert_rank") != protocol.executor.expert_rank
            or row.get("source_recipe_sha256")
            != source_result.d3_recipe_sha256
            or row.get("training_sha256")
            != protocol.training.artifact_sha256
            or row.get("fit_data_binding_sha256")
            != source_result.fit_data_binding_sha256
            or row.get("training_metric")
            != protocol.training.training_metric
            or row.get("training_metric_weight_sha256")
            != _tensor_sha256(plan.fisher_metric_weight)
            or row.get("canonical_scoring_metric_weight_sha256")
            != _tensor_sha256(metric_weight)
            or row.get("accounting") != asdict(plan.accounting())
            or _canonical_json_bytes(row.get("initial_training_metrics"))
            != _canonical_json_bytes(plan.initial_metrics.state_dict())
            or _canonical_json_bytes(row.get("final_training_metrics"))
            != _canonical_json_bytes(plan.final_metrics.state_dict())
            or row.get("provider_binding_sha256")
            != _provider_binding_sha256(
                protocol=protocol,
                c2_protocol_sha256=str(
                    manifest["c2_protocol_sha256"]
                ),
                calibration_sha256=str(
                    manifest["c2_calibration_sha256"]
                ),
                basis_payload_sha256=str(
                    manifest["basis_package_payload_sha256"]
                ),
                source_model_sha256=str(
                    manifest["source_model_sha256"]
                ),
                norm_sha256=str(
                    manifest["pre_feedforward_norm_sha256"]
                ),
                training_metric_weight=plan.fisher_metric_weight,
                target_center=plan.target_center,
                target_scale=plan.target_scale,
            )
        ):
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} scientific binding drifted"
            )
        ordinary_raw = row.get("ordinary_score")
        if not isinstance(ordinary_raw, Mapping):
            raise TypeError("rank-64 ordinary score must be a mapping")
        ordinary_score = FullWidthCandidateScore.from_state_dict(
            ordinary_raw
        )
        expected_ordinary_flags = _recompute_ordinary_gate_flags(
            ordinary_score,
            ordinary_gates,
        )
        if (
            ordinary_score.candidate_id != candidate_id
            or ordinary_score.candidate_artifact_sha256
            != row.get("candidate_binding_sha256")
            or ordinary_score.source_rank != protocol.visible_source_modes
            or ordinary_score.target_rank != protocol.visible_target_modes
            or ordinary_score.stored_scalar_count
            != plan.accounting().total_stored_scalar_count
            or ordinary_score.controls_artifact_sha256
            != controls.artifact_sha256
            or ordinary_score.gates_sha256
            != protocol.training.ordinary_gates_sha256
            or ordinary_score.gate_flags.state_dict()
            != expected_ordinary_flags
            or ordinary_score.passed is not all(
                expected_ordinary_flags.values()
            )
        ):
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} ordinary score drifted"
            )
        contrast_state, restored_contrast_scores = (
            _validate_contrast_result_state(
                row.get("contrast_result"),
                gates=contrast_gates,
            )
        )
        coverage = row.get("contrast_coverage")
        balance_gate = row.get("objective_balance_gate")
        training_diagnostics = row.get(
            "training_teacher_signal_diagnostics"
        )
        if (
            not isinstance(coverage, Mapping)
            or not isinstance(balance_gate, Mapping)
            or not isinstance(training_diagnostics, Mapping)
        ):
            raise TypeError("rank-64 candidate diagnostics are invalid")
        contrast_identities = row.get("contrast_identities")
        expected_identities = _expected_contrast_identities(
            live_c2_protocol
        )
        if _canonical_json_bytes(contrast_identities) != (
            _canonical_json_bytes(expected_identities)
        ):
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} identities drifted"
            )
        recomputed_coverage = _recompute_contrast_coverage(
            restored_contrast_scores,
            contrast_identities,
            required_null_candidate_pass_count=(
                objective_protocol.gates.required_null_candidate_pass_count
            ),
        )
        if _canonical_json_bytes(coverage) != _canonical_json_bytes(
            recomputed_coverage
        ):
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} coverage drifted"
            )
        recomputed_balance = d0d3._contribution_balance_gate(
            plan,
            recipe=d3_recipe,
            gates=objective_protocol.gates,
            training_teacher_energy=float(
                balance_gate["training_fit_teacher_weighted_energy"]
            ),
            raw_teacher_energy=float(
                balance_gate["raw_fit_teacher_weighted_energy"]
            ),
            teacher_signal_diagnostics=training_diagnostics,
        )
        if _canonical_json_bytes(recomputed_balance) != (
            _canonical_json_bytes(balance_gate)
        ):
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} balance gate drifted"
            )
        source_comparison = row.get("source_fit_sequence_comparison")
        pair_balance = row.get("pair_balance")
        if not isinstance(source_comparison, Mapping) or not isinstance(
            pair_balance,
            Mapping,
        ):
            raise TypeError("rank-64 source comparison is invalid")
        if set(source_comparison) != {
            "passed",
            "flags",
            "source_sequence_sha256s",
            "candidate_sequence_sha256s",
            "comparison_semantics",
        } or set(pair_balance) != {
            "semantics",
            "natural_family_counts",
            "balanced_family_counts",
            "signed_pair_multiplicity",
            "duplicate_bindings",
            "natural_pair_sha256s",
            "balanced_pair_sha256s",
        }:
            raise ValueError("rank-64 source/pair schema drifted")
        source_sequences = source_comparison.get(
            "source_sequence_sha256s"
        )
        candidate_sequences = source_comparison.get(
            "candidate_sequence_sha256s"
        )
        sequence_flags = source_comparison.get("flags")
        if (
            not isinstance(source_sequences, Mapping)
            or not isinstance(candidate_sequences, Mapping)
            or not isinstance(sequence_flags, Mapping)
            or set(source_sequences) != set(_SOURCE_PLAN_SEQUENCE_FIELDS)
            or set(candidate_sequences) != set(_SOURCE_PLAN_SEQUENCE_FIELDS)
            or set(sequence_flags) != set(_SOURCE_PLAN_SEQUENCE_FIELDS)
        ):
            raise ValueError("rank-64 source sequence comparison is invalid")
        expected_sequence_flags = {
            name: tuple(getattr(plan, name))
            == tuple(source_sequences[name])  # type: ignore[arg-type]
            for name in _SOURCE_PLAN_SEQUENCE_FIELDS
        }
        if (
            {
                name: tuple(getattr(plan, name))
                for name in _SOURCE_PLAN_SEQUENCE_FIELDS
            }
            != {
                name: tuple(candidate_sequences[name])  # type: ignore[arg-type]
                for name in _SOURCE_PLAN_SEQUENCE_FIELDS
            }
            or dict(sequence_flags) != expected_sequence_flags
            or source_comparison.get("passed")
            is not all(expected_sequence_flags.values())
            or tuple(pair_balance.get("balanced_pair_sha256s", ()))
            != plan.fit_pair_sha256s
            or tuple(pair_balance.get("balanced_pair_sha256s", ()))
            != tuple(source_sequences["fit_pair_sha256s"])  # type: ignore[arg-type]
        ):
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} source sequence drifted"
            )
        treatment = row.get("treatment_validity")
        capability = row.get("fit_capability_contract")
        if not isinstance(treatment, Mapping) or not isinstance(
            capability,
            Mapping,
        ):
            raise TypeError("rank-64 candidate gate records are invalid")
        if set(treatment) != {
            "passed",
            "flags",
            "failure_semantics",
        } or set(capability) != {
            "ordinary_gate_count",
            "required_ordinary_gate_count",
            "all_ordinary_gates_passed",
            "every_eligible_sensitivity_passed",
            "all_contrast_families_formally_passed",
            "all_families_cover_all_four_rank_bands",
            "required_null_contrasts_passed",
        }:
            raise ValueError("rank-64 candidate gate schema drifted")
        treatment_flags = treatment.get("flags")
        if (
            not isinstance(treatment_flags, Mapping)
            or not treatment_flags
            or any(type(value) is not bool for value in treatment_flags.values())
        ):
            raise TypeError("rank-64 treatment-validity flags are invalid")
        expected_treatment_flags = {
            "objective_contribution_balance": bool(
                recomputed_balance["passed"]
            ),
            "exact_source_fit_sequences": bool(
                source_comparison["passed"]
            ),
            "exact_rank64_executor_config": (
                asdict(plan.executor_config)
                == asdict(_executor_config(protocol))
            ),
            "exact_d3_objective": (
                plan.objective.artifact_sha256
                == _objective(protocol.training).artifact_sha256
            ),
            "exact_steps": plan.training_steps == protocol.training.steps,
            "exact_learning_rate": (
                plan.learning_rate == protocol.training.learning_rate
            ),
            "exact_seed": plan.seed == expected_seed,
            "exact_unit_rms_training_metric": (
                plan.fisher_metric_supplied
                and torch.equal(
                    plan.fisher_metric_weight,
                    gauge.metric_weight.to(dtype=torch.float64),
                )
            ),
            "exact_fit_data_binding": (
                plan.synthetic_binding_sha256
                == source_result.fit_data_binding_sha256
                == manifest.get("fit_data_binding_sha256")
            ),
            "cold_start_source_weights_unused": (
                row.get(
                    "source_plan_parameters_used_for_initialization"
                )
                is False
            ),
        }
        if dict(treatment_flags) != expected_treatment_flags:
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} treatment flags drifted"
            )
        recomputed_treatment = all(expected_treatment_flags.values())
        ordinary_flags = expected_ordinary_flags
        ordinary_values = tuple(
            value
            for name, value in ordinary_flags.items()
            if name != "all_passed"
        )
        expected_capability = {
            "ordinary_gate_count": len(ordinary_values),
            "required_ordinary_gate_count": (
                objective_protocol.gates.required_ordinary_gate_count
            ),
            "all_ordinary_gates_passed": (
                len(ordinary_values)
                == objective_protocol.gates.required_ordinary_gate_count
                and all(ordinary_values)
            ),
            "every_eligible_sensitivity_passed": bool(
                coverage["every_teacher_qualified_contrast_passed"]
            ),
            "all_contrast_families_formally_passed": (
                contrast_state["overall_status"] == "pass"
            ),
            "all_families_cover_all_four_rank_bands": bool(
                coverage["all_families_cover_all_four_rank_bands"]
            ),
            "required_null_contrasts_passed": bool(
                coverage["required_null_contrasts_valid_and_passed"]
            ),
        }
        if dict(capability) != expected_capability:
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} capability contract drifted"
            )
        recomputed_capability = all(
            (
                expected_capability["all_ordinary_gates_passed"],
                expected_capability[
                    "every_eligible_sensitivity_passed"
                ],
                expected_capability[
                    "all_contrast_families_formally_passed"
                ],
                expected_capability[
                    "all_families_cover_all_four_rank_bands"
                ],
                expected_capability["required_null_contrasts_passed"],
            )
        )
        if (
            treatment.get("passed") is not recomputed_treatment
            or row.get("fit_capability_pass") is not recomputed_capability
            or row.get("combined_pass")
            is not (recomputed_treatment and recomputed_capability)
        ):
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} gate decision drifted"
            )
        expected_failures: list[str] = []
        if not ordinary_score.passed:
            expected_failures.extend(
                f"ordinary:{name}"
                for name, passed in ordinary_flags.items()
                if name != "all_passed" and passed is False
            )
        if contrast_state["overall_status"] != "pass":
            expected_failures.append(
                f"contrast:{contrast_state['overall_status']}"
            )
            expected_failures.extend(
                f"contrast:{value}"
                for value in contrast_state["reason_codes"]  # type: ignore[union-attr]
            )
        if not bool(
            coverage["all_families_cover_all_four_rank_bands"]
        ):
            expected_failures.append(
                "contrast:teacher_coverage_missing_rank_band"
            )
        if not bool(coverage["every_teacher_qualified_contrast_passed"]):
            expected_failures.append(
                "contrast:not_every_qualified_contrast_passed"
            )
        if not bool(
            coverage["required_null_contrasts_valid_and_passed"]
        ):
            expected_failures.append(
                "contrast:required_null_count_not_passed"
            )
        if not bool(recomputed_balance["passed"]):
            expected_failures.append(
                "objective:contribution_balance_gate_failed"
            )
        expected_failures.extend(
            f"treatment_validity:{name}"
            for name, passed in expected_treatment_flags.items()
            if not passed
        )
        if tuple(sorted(set(expected_failures))) != tuple(
            row.get("failure_reasons", ())
        ):
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} failures drifted"
            )
    report_rows = report.get("candidate_results")
    if (
        not isinstance(report_rows, list)
        or _canonical_json_bytes(report_rows)
        != _canonical_json_bytes(
            tuple(candidate_results[value] for value in executed_ids)
        )
    ):
        raise ValueError("rank-64 report candidate rows drifted")
    primary_row = candidate_results["r64_d3_capacity_control.primary"]
    assert isinstance(primary_row, Mapping)
    replication_row = candidate_results.get(
        "r64_d3_capacity_control.replication"
    )
    if replication_row is not None and not isinstance(
        replication_row,
        Mapping,
    ):
        raise TypeError("rank-64 replication row is invalid")
    primary_valid = bool(
        primary_row["treatment_validity"]["passed"]  # type: ignore[index]
    )
    primary_capability = bool(primary_row["fit_capability_pass"])
    replication_valid = (
        None
        if replication_row is None
        else bool(
            replication_row["treatment_validity"]["passed"]  # type: ignore[index]
        )
    )
    replication_capability = (
        None
        if replication_row is None
        else bool(replication_row["fit_capability_pass"])
    )
    if (replication_row is not None) is not (
        primary_valid and primary_capability
    ):
        raise ValueError(
            "rank-64 executed candidate schedule differs from protocol"
        )
    if not primary_valid:
        recomputed_outcome = (
            "invalid_rank_comparison_primary_treatment_validity_failed"
        )
    elif not primary_capability:
        recomputed_outcome = "rank64_primary_fit_capability_failed"
    elif replication_row is None:
        raise ValueError(
            "rank-64 passing primary lacks required replication"
        )
    elif not replication_valid:
        recomputed_outcome = (
            "invalid_rank_comparison_replication_treatment_validity_failed"
        )
    elif not replication_capability:
        recomputed_outcome = "rank64_replication_fit_capability_failed"
    else:
        recomputed_outcome = "rank64_two_seed_fit_capability_pass"
    two_seed_pass = (
        recomputed_outcome == "rank64_two_seed_fit_capability_pass"
    )
    recomputed_decision = {
        "outcome": recomputed_outcome,
        "primary_treatment_valid": primary_valid,
        "primary_fit_capability_passed": primary_capability,
        "replication_executed": replication_row is not None,
        "replication_treatment_valid": replication_valid,
        "replication_fit_capability_passed": replication_capability,
        "two_seed_fit_capability_passed": two_seed_pass,
        "compressed_width_ladder_preregistration_supported": two_seed_pass,
        "fresh_c3_authorized": False,
        "compression_claim_authorized": False,
    }
    if any(
        manifest.get(name) != expected
        for name, expected in recomputed_decision.items()
    ):
        raise ValueError("rank-64 manifest decision differs from candidate rows")
    source_d3_binding = report.get("source_d3_binding")
    if not isinstance(source_d3_binding, Mapping) or set(
        source_d3_binding
    ) != {
        "source_result",
        "source_fit_sequence_sha256s",
        "source_natural_pair_sha256s",
        "source_balanced_pair_sha256s",
        "source_shared_manifest",
        "source_plan_parameters_used",
    }:
        raise ValueError("rank-64 source-D3 report binding is invalid")
    source_sequence_report = source_d3_binding[
        "source_fit_sequence_sha256s"
    ]
    source_comparison = primary_row.get("source_fit_sequence_comparison")
    pair_balance = primary_row.get("pair_balance")
    if (
        _canonical_json_bytes(source_d3_binding["source_result"])
        != _canonical_json_bytes(source_result.state_dict())
        or source_d3_binding.get("source_plan_parameters_used") is not False
        or not isinstance(source_comparison, Mapping)
        or not isinstance(pair_balance, Mapping)
        or _canonical_json_bytes(source_sequence_report)
        != _canonical_json_bytes(
            source_comparison.get("source_sequence_sha256s")
        )
        or _canonical_json_bytes(
            source_d3_binding["source_natural_pair_sha256s"]
        )
        != _canonical_json_bytes(pair_balance.get("natural_pair_sha256s"))
        or _canonical_json_bytes(
            source_d3_binding["source_balanced_pair_sha256s"]
        )
        != _canonical_json_bytes(pair_balance.get("balanced_pair_sha256s"))
        or not isinstance(
            source_d3_binding["source_shared_manifest"],
            Mapping,
        )
    ):
        raise ValueError("rank-64 source-D3 report identity drifted")
    source_shared = source_d3_binding["source_shared_manifest"]
    assert isinstance(source_shared, Mapping)
    if any(
        manifest.get(name) != value
        for name, value in source_shared.items()
    ):
        raise ValueError("rank-64 shared source-D3 manifest drifted")
    if not isinstance(source_sequence_report, Mapping):
        raise TypeError("rank-64 source sequence report must be a mapping")
    source_natural = tuple(
        source_d3_binding["source_natural_pair_sha256s"]  # type: ignore[arg-type]
    )
    source_balanced = tuple(
        source_d3_binding["source_balanced_pair_sha256s"]  # type: ignore[arg-type]
    )
    _validate_source_sequence_anchors(
        source_result,
        sequences={
            str(name): tuple(values)  # type: ignore[arg-type]
            for name, values in source_sequence_report.items()
        },
        natural=source_natural,
        balanced=source_balanced,
    )
    if source_replay_binding_sha256(
        sequence_sha256s=source_sequence_report,
        natural_pair_sha256s=source_natural,
        balanced_pair_sha256s=source_balanced,
        shared_manifest=source_shared,
    ) != source_result.d3_source_replay_binding_sha256:
        raise ValueError("rank-64 source replay aggregate binding drifted")
    for candidate_id in executed_ids:
        row = candidate_results[candidate_id]
        assert isinstance(row, Mapping)
        comparison = row["source_fit_sequence_comparison"]
        pair_balance = row["pair_balance"]
        if (
            not isinstance(comparison, Mapping)
            or not isinstance(pair_balance, Mapping)
            or _canonical_json_bytes(
                comparison.get("source_sequence_sha256s")
            )
            != _canonical_json_bytes(source_sequence_report)
            or _canonical_json_bytes(
                pair_balance.get("natural_pair_sha256s")
            )
            != _canonical_json_bytes(source_natural)
            or _canonical_json_bytes(
                pair_balance.get("balanced_pair_sha256s")
            )
            != _canonical_json_bytes(source_balanced)
        ):
            raise ValueError(
                f"rank-64 candidate {candidate_id!r} source replay drifted"
            )
    false_firewall_fields = (
        "selection_materialized",
        "selection_measured",
        "selection_scored",
        "selection_data_changed_training",
        "c2_provider_artifact_loaded",
        "source_d3_parameters_used_for_initialization",
        "v2_targets_loaded",
        "v3_targets_loaded",
        "prompt_text_loaded",
        "token_ids_loaded",
        "tokenizer_loaded",
        "natural_activation_rows_loaded",
        "fresh_c3_authorized",
        "compression_claim_authorized",
    )
    if (
        any(manifest.get(name) is not False for name in false_firewall_fields)
        or manifest.get("authenticated_source_result_artifact_loaded")
        is not True
        or manifest.get("cold_start") is not True
        or manifest.get("baseline_latent_rank")
        != protocol.baseline_latent_rank
        or manifest.get("latent_rank") != protocol.latent_rank
        or manifest.get("expert_rank") != protocol.executor.expert_rank
        or manifest.get("controlled_change")
        != "outer_latent_rank_16_to_64_only"
        or manifest.get("scientific_scope")
        != (
            "fit_only_rank64_capacity_control_not_compression_or_"
            "generalization"
        )
    ):
        raise ValueError("rank-64 artifact violates its scientific firewall")
    expected_interpretation = {
        "fit_side_only": True,
        "held_out_selection_evidence": False,
        "rank64_is_capacity_oracle_not_compression": True,
        "two_seed_pass_implicates_outer_packing_bottleneck": True,
        "two_seed_pass_proves_rank16_is_only_bottleneck": False,
        "two_seed_pass_opens_only_separate_width_ladder": True,
        "valid_failure_leaves_executor_objective_optimization_entangled": True,
        "rank64_failure_proves_insufficient_capacity": False,
        "fresh_c3_authorized": False,
        "natural_prompt_fidelity_claim": False,
        "whole_model_replacement_claim": False,
        "wall_clock_speed_claim": False,
        "whole_model_compression_claim": False,
        "provider_fit_numeric_dtype": "torch.float64",
        "live_measurement_device": protocol.execution_device,
        "live_measurement_dtype": protocol.execution_dtype,
    }
    expected_safety = {
        "contains_source_model_state_dict": False,
        "contains_rank64_provider_parameters": True,
        "contains_source_d3_provider_parameters": False,
        "contains_raw_teacher_targets": False,
        "contains_teacher_jvp_tensors": False,
        "contains_provider_chart_jvp_tensors": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_c2_selection_data": False,
        "committable": False,
    }
    if (
        report.get("interpretation") != expected_interpretation
        or report.get("safety") != expected_safety
    ):
        raise ValueError("rank-64 report scientific boundary drifted")
    return LoadedRank64CapacityControlArtifact(
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
) -> LoadedRank64CapacityControlArtifact:
    _publish_artifact(state, report_payload, output=output)
    try:
        return load_rank64_capacity_control_artifact(output)
    except BaseException:
        output.unlink(missing_ok=True)
        output.with_suffix(".json").unlink(missing_ok=True)
        raise


def run_rank64_capacity_control(
    *,
    source_diagnostic_path: Path | str = DEFAULT_SOURCE_DIAGNOSTIC,
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
    """Run the authenticated matched rank-64 fit-only capacity control."""

    (
        protocol,
        objective_protocol,
        c2_protocol,
        d3_recipe,
    ) = _authenticated_declarations()
    if (
        device_name != protocol.execution_device
        or dtype != protocol.execution_dtype
    ):
        raise ValueError(
            "rank-64 capacity execution is frozen to cpu/float32"
        )
    destination = _validate_output_path(output)
    source = _authenticate_source_d3(
        source_diagnostic_path,
        protocol=protocol,
    )
    code_sha256s = _code_sha256s()
    code_bundle_sha256 = _code_bundle_sha256(code_sha256s)
    fidelity_gates = SyntheticReferenceGates()
    contrast_gates = ContrastAssessmentGates()
    ordinary_gates_sha256 = full_width_reference_gates_sha256(
        _deferred_collision_gates(fidelity_gates)
    )
    if (
        ordinary_gates_sha256
        != protocol.training.ordinary_gates_sha256
        or contrast_gates.artifact_sha256
        != protocol.training.contrast_gates_sha256
    ):
        raise ValueError("rank-64 capacity scoring gates drifted")

    basis, adapter, pre_ff3, post_ff3, epsilon = _load_live_dependencies(
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
        d0d3._actual_model_execution(adapter)
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

    pilot, pilot_measurement = d0d3._measure_c2_role(
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
    calibrated_fit_panel_sha256 = c2_protocol.calibrated_panel_sha256(
        "fit",
        calibration,
    )
    if (
        calibration.selected_amplitude
        != _EXPECTED_C2_CALIBRATION_AMPLITUDE
        or calibration.artifact_sha256
        != _EXPECTED_C2_CALIBRATION_SHA256
        or calibrated_fit_panel_sha256
        != objective_protocol.c2_provenance.calibrated_fit_panel_sha256
    ):
        raise ValueError("C2 pilot replay did not authenticate amplitude h=8")

    fit, fit_measurement = d0d3._measure_c2_role(
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
    forbidden_prefixes = (
        objective_protocol.c2_provenance.forbidden_probe_prefixes
    )
    if any(
        value.probe.probe_id.startswith(forbidden_prefixes)  # type: ignore[attr-defined]
        for value in fit
    ):
        raise RuntimeError("C2 selection identity entered rank-64 fit data")
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
    raw_teacher_energy = d0d3._fit_teacher_weighted_energy(
        fit,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=raw_metric_weight,
    )
    unit_rms_gauge = UnitRmsFisherGauge.from_metric_weight(
        raw_metric_weight
    )
    unit_teacher_energy = d0d3._fit_teacher_weighted_energy(
        fit,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=unit_rms_gauge.metric_weight,
    )
    if (
        raw_teacher_energy
        <= objective_protocol.gates.minimum_gauge_energy
        or abs(unit_teacher_energy - 1.0)
        > objective_protocol.gates.normalized_energy_absolute_tolerance
    ):
        raise ValueError("rank-64 Fisher gauge failed normalization gates")
    natural_pairs, chart_mismatch_diagnostics = (
        c2._training_contrast_pairs(
            protocol=c2_protocol,
            measured=fit,
            basis=basis,
            adapter=adapter,
            pre_ff3=pre_ff3,
            post_ff3=post_ff3,
            epsilon=epsilon,
        )
    )
    teacher_signal_diagnostics = d0d3._teacher_signal_diagnostics(
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
    teacher_floor = (
        objective_protocol.gates.minimum_teacher_mse_floor_multiple
        * max(
            protocol.training.sensitivity_relative_floor**2,
            protocol.training.jvp_relative_floor**2,
        )
    )
    if minimum_teacher_mse <= teacher_floor:
        raise ValueError("rank-64 teacher contrast signal is near a floor")

    fit_data_binding_sha256 = d0d3._fit_data_binding_sha256(
        basis=basis,
        c2_protocol=c2_protocol,
        calibration=calibration,
        norm_sha256=norm_sha256,
        canonical_metric_weight=raw_metric_weight,
    )
    if (
        fit_data_binding_sha256
        != source.fit_data_binding_sha256
        or fit_data_binding_sha256
        != protocol.training.fit_data_binding_sha256
    ):
        raise ValueError("rank-64 fit-data binding differs from source D3")
    standardized_gauge_sha256 = d0d3._standardized_gauge_sha256(
        basis_payload_sha256=basis.basis_payload_sha256,
        source_model_sha256=basis.source_model_sha256,
        c2_protocol_sha256=c2_protocol.protocol_sha256,
        calibration_sha256=calibration.artifact_sha256,
        target_center=target_center,
        target_scale=target_scale,
        canonical_metric_weight=raw_metric_weight,
    )
    ordinary_probes = c2._ordinary_full_width_probes(
        fit,
        split="fit",
        metric_weight=raw_metric_weight,
        standardized_gauge_sha256=standardized_gauge_sha256,
    )
    controls = fit_full_width_reference_controls(
        fit_probes=ordinary_probes,
        position_bin_count=16,
    )
    current_shared_manifest = {
        "c2_protocol_sha256": c2_protocol.protocol_sha256,
        "c2_pilot_panel_sha256": c2_protocol.panel_sha256("pilot"),
        "c2_fit_panel_sha256": c2_protocol.panel_sha256("fit"),
        "c2_calibrated_fit_panel_sha256": calibrated_fit_panel_sha256,
        "c2_calibration_sha256": calibration.artifact_sha256,
        "selected_calibration_amplitude": calibration.selected_amplitude,
        "basis_package_file_sha256": basis_package_file_sha256,
        "basis_package_payload_sha256": basis.basis_payload_sha256,
        "source_model_sha256": basis.source_model_sha256,
        "pre_feedforward_norm_sha256": norm_sha256,
        "canonical_metric_weight_sha256": d0d3._tensor_sha256(
            raw_metric_weight
        ),
        "unit_rms_gauge_sha256": unit_rms_gauge.artifact_sha256,
        "standardized_gauge_sha256": standardized_gauge_sha256,
        "controls_sha256": controls.artifact_sha256,
        "ordinary_gates_sha256": ordinary_gates_sha256,
        "contrast_gates_sha256": contrast_gates.artifact_sha256,
    }
    if current_shared_manifest != dict(source.shared_manifest):
        changed = tuple(
            name
            for name, value in current_shared_manifest.items()
            if source.shared_manifest.get(name) != value
        )
        raise ValueError(
            "rank-64 shared D3 problem drifted: "
            + ",".join(changed)
        )

    def evaluate(seed_role: str, seed: int) -> _CapacityEvaluation:
        return _evaluate_capacity_candidate(
            seed_role=seed_role,
            seed=seed,
            protocol=protocol,
            objective_protocol=objective_protocol,
            d3_recipe=d3_recipe,
            source=source,
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
            ordinary_probes=ordinary_probes,
            controls=controls,
            standardized_gauge_sha256=standardized_gauge_sha256,
            fidelity_gates=fidelity_gates,
            contrast_gates=contrast_gates,
        )

    primary, replication = _execute_capacity_schedule(
        protocol,
        evaluate=evaluate,
    )
    evaluations = tuple(
        value for value in (primary, replication) if value is not None
    )
    candidate_rows: list[dict[str, object]] = []
    plan_states: dict[str, dict[str, object]] = {}
    for evaluation in evaluations:
        row = dict(evaluation.row)
        plan_state = row.pop("_plan_state")
        if not isinstance(plan_state, dict):
            raise RuntimeError("rank-64 candidate lost its plan state")
        candidate_rows.append(row)
        plan_states[evaluation.candidate_id] = plan_state
    decision = _capacity_decision(primary, replication)

    if (
        adapter.model_fingerprint() != model_before
        or module_state_fingerprint(pre_ff3) != norm_sha256
        or _code_sha256s() != code_sha256s
    ):
        raise RuntimeError(
            "model, normalization, or code changed during rank-64 run"
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
        "protocol_sha256": protocol.protocol_sha256,
        "source_objective_protocol_sha256": (
            objective_protocol.protocol_sha256
        ),
        "source_objective_result_binding_sha256": (
            protocol.source_result.artifact_sha256
        ),
        "source_objective_logical_artifact_sha256": (
            protocol.source_result.logical_artifact_sha256
        ),
        "source_objective_tensor_file_sha256": (
            protocol.source_result.tensor_sha256
        ),
        "source_objective_report_sha256": (
            protocol.source_result.report_sha256
        ),
        "source_objective_code_bundle_sha256": (
            protocol.source_result.code_bundle_sha256
        ),
        "source_d3_recipe_sha256": d3_recipe.artifact_sha256,
        "source_d3_primary_plan_sha256": (
            protocol.source_result.d3_primary_plan_sha256
        ),
        "source_d3_primary_result_sha256": (
            protocol.source_result.d3_primary_result_sha256
        ),
        "source_d3_fit_data_binding_sha256": (
            protocol.source_result.fit_data_binding_sha256
        ),
        "c2_protocol_sha256": c2_protocol.protocol_sha256,
        "c2_pilot_panel_sha256": c2_protocol.panel_sha256("pilot"),
        "c2_fit_panel_sha256": c2_protocol.panel_sha256("fit"),
        "c2_calibrated_fit_panel_sha256": (
            calibrated_fit_panel_sha256
        ),
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
        "canonical_metric_weight_sha256": d0d3._tensor_sha256(
            raw_metric_weight
        ),
        "fit_data_binding_sha256": fit_data_binding_sha256,
        "unit_rms_gauge_sha256": unit_rms_gauge.artifact_sha256,
        "standardized_gauge_sha256": standardized_gauge_sha256,
        "controls_sha256": controls.artifact_sha256,
        "ordinary_gates_sha256": ordinary_gates_sha256,
        "contrast_gates_sha256": contrast_gates.artifact_sha256,
        "baseline_latent_rank": protocol.baseline_latent_rank,
        "latent_rank": protocol.latent_rank,
        "expert_rank": protocol.executor.expert_rank,
        "controlled_change": "outer_latent_rank_16_to_64_only",
        "executed_candidate_ids": tuple(
            str(value["candidate_id"]) for value in candidate_rows
        ),
        "candidate_plan_sha256s": {
            value.candidate_id: value.plan.artifact_sha256
            for value in evaluations
        },
        "candidate_result_sha256s": candidate_result_sha256s,
        **decision,
        "selection_materialized": False,
        "selection_measured": False,
        "selection_scored": False,
        "selection_data_changed_training": False,
        "c2_provider_artifact_loaded": False,
        "authenticated_source_result_artifact_loaded": True,
        "source_d3_parameters_used_for_initialization": False,
        "cold_start": True,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "code_sha256s": code_sha256s,
        "code_bundle_sha256": code_bundle_sha256,
        "scientific_scope": (
            "fit_only_rank64_capacity_control_not_compression_or_"
            "generalization"
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
            "target_center_sha256": d0d3._tensor_sha256(target_center),
            "target_scale_sha256": d0d3._tensor_sha256(target_scale),
        },
        "source_d3_binding": {
            "source_result": protocol.source_result.state_dict(),
            "source_fit_sequence_sha256s": dict(source.sequence_sha256s),
            "source_natural_pair_sha256s": source.natural_pair_sha256s,
            "source_balanced_pair_sha256s": source.balanced_pair_sha256s,
            "source_shared_manifest": dict(source.shared_manifest),
            "source_plan_parameters_used": False,
        },
        "candidate_results": candidate_rows,
        "interpretation": {
            "fit_side_only": True,
            "held_out_selection_evidence": False,
            "rank64_is_capacity_oracle_not_compression": True,
            "two_seed_pass_implicates_outer_packing_bottleneck": True,
            "two_seed_pass_proves_rank16_is_only_bottleneck": False,
            "two_seed_pass_opens_only_separate_width_ladder": True,
            "valid_failure_leaves_executor_objective_optimization_entangled": (
                True
            ),
            "rank64_failure_proves_insufficient_capacity": False,
            "fresh_c3_authorized": False,
            "natural_prompt_fidelity_claim": False,
            "whole_model_replacement_claim": False,
            "wall_clock_speed_claim": False,
            "whole_model_compression_claim": False,
            "provider_fit_numeric_dtype": "torch.float64",
            "live_measurement_device": protocol.execution_device,
            "live_measurement_dtype": protocol.execution_dtype,
        },
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_rank64_provider_parameters": True,
            "contains_source_d3_provider_parameters": False,
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
        help="describe rank-64 control without opening model or artifacts",
    )
    run_parser = commands.add_parser(
        "run",
        help="replay C2 pilot/fit and run the matched rank-64 control",
    )
    run_parser.add_argument(
        "--source-diagnostic",
        type=Path,
        default=DEFAULT_SOURCE_DIAGNOSTIC,
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
        choices=(CAPACITY_CONTROL_EXECUTION_DEVICE,),
        default=CAPACITY_CONTROL_EXECUTION_DEVICE,
    )
    run_parser.add_argument(
        "--dtype",
        choices=(CAPACITY_CONTROL_EXECUTION_DTYPE,),
        default=CAPACITY_CONTROL_EXECUTION_DTYPE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        report = describe_rank64_capacity_control()
    else:
        report = run_rank64_capacity_control(
            source_diagnostic_path=args.source_diagnostic,
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
