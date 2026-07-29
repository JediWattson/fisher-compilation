"""Run the paired function-preserving Gemma L3/L4 width control.

The primary pair starts an exact rank-16 D3 replay and a gradient-open rank-64
lift from the same observable function and provider-chart JVP.  A rank-64
primary pass beside a rank-16 failure authorizes only the frozen paired
replication seed.  This fit-side control never opens C2 selection, C3, or a
compression claim.
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
    default_contrast_provider_development_protocol,
    select_global_calibration_amplitude,
)
from . import gemma3_l3_l4_objective_balance_diagnostic as d0d3
from .gemma3_l3_l4_objective_balance_diagnostic_protocol import (
    DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256,
    default_objective_balance_diagnostic_protocol,
)
from . import gemma3_l3_l4_rank64_capacity_control as r64
from .gemma3_l3_l4_rank64_capacity_control_protocol import (
    DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256,
    default_rank64_capacity_control_protocol,
)
from .gemma3_l3_l4_function_preserving_width_control_protocol import (
    DEFAULT_FUNCTION_PRESERVING_WIDTH_CONTROL_PROTOCOL_SHA256,
    FunctionPreservingWidthControlProtocol,
    WidthExecutorSpec,
    default_function_preserving_width_control_protocol,
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
from . import state_conditioned_contrast_fit as contrast_fit
from .state_conditioned_contrast_fit import (
    ContrastAwareObjective,
    ContrastAwareReferenceProviderPlan,
)
from .state_conditioned_reference_selection import (
    FullWidthCandidateScore,
    fit_full_width_reference_controls,
    full_width_reference_gates_sha256,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE_DIAGNOSTIC",
    "DEFAULT_SOURCE_RANK64",
    "LoadedFunctionPreservingWidthControlArtifact",
    "build_parser",
    "describe_function_preserving_width_control",
    "load_function_preserving_width_control_artifact",
    "main",
    "run_function_preserving_width_control",
]


DEFAULT_SOURCE_DIAGNOSTIC = d0d3.DEFAULT_OUTPUT
DEFAULT_SOURCE_RANK64 = r64.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-function-preserving-width-control.pt"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_function_preserving_width_control"
_FORMAT_VERSION = 1
_PRIMARY_D3_ID = "d3_unit_rms_family_balanced_direction.primary"
_EXPECTED_CALIBRATION_SHA256 = (
    "aedb23de65ed6a37d645539001311ddb415cd2713400777dac448cb96bd5bfa8"
)
_D3_PRIMARY_FINAL_METRICS_SHA256 = (
    "80b841300d25b8576b1ee99ab60b2770a3f9773c61794c78661985c04e599b66"
)
_EXPECTED_SOURCE_MODEL_SHA256 = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
_EXPECTED_PRE_FEEDFORWARD_NORM_SHA256 = (
    "53a898a10dd76575f0d603f06ef33a9af73b91335e0f1a3c4fec1d475e2706f9"
)
_EXPECTED_CANONICAL_METRIC_WEIGHT_SHA256 = (
    "e9d643a297c583a4dfe4a264ef49bd525e9d33fd4786e1ad1f4272a8d8ccf5ac"
)
_EXPECTED_UNIT_RMS_GAUGE_SHA256 = (
    "4a553347335815c56643fdde56c32247e32153ed20e8c713626e35a9a072c312"
)
_EXPECTED_STANDARDIZED_GAUGE_SHA256 = (
    "e1bd1659b762476aee4622d3473fa12d6efcca70f7e72bd4b4b45ee86a8413b7"
)
_EXPECTED_CONTROLS_SHA256 = (
    "7c150b9d051abea9a3f9dbbc934465932aacd4a62b73558b22b1bf89075c964d"
)
_ARTIFACT_DOMAIN = b"fisher-graph:function-preserving-width:artifact:v1\0"
_REPORT_DOMAIN = b"fisher-graph:function-preserving-width:report:v1\0"
_CODE_DOMAIN = b"fisher-graph:function-preserving-width:code:v1\0"
_RESULT_DOMAIN = b"fisher-graph:function-preserving-width:result:v1\0"
_INITIALIZATION_DOMAIN = (
    b"fisher-graph:function-preserving-width:initialization:v1\0"
)
_MEASUREMENT_EVIDENCE_DOMAIN = (
    b"fisher-graph:function-preserving-width:measurement-evidence:v1\0"
)
_MEASUREMENT_EVIDENCE_FIELDS = (
    "pilot_metrics",
    "pilot_measurement",
    "fit_measurement",
    "fit_provider_chart_mismatch_diagnostics",
    "teacher_signal_diagnostics",
    "training_teacher_signal_diagnostics",
    "pair_balance",
    "gauge",
)


@dataclass(frozen=True, slots=True)
class _AuthenticatedSources:
    source_d3: r64._SourceD3Bindings
    d3_primary_row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ArmEvaluation:
    candidate_id: str
    pair_role: str
    arm: str
    fit_capability_pass: bool
    row: dict[str, object]
    plan: ContrastAwareReferenceProviderPlan


@dataclass(frozen=True, slots=True)
class _PairEvaluation:
    pair_role: str
    seed: int
    rank16: _ArmEvaluation
    rank64: _ArmEvaluation
    treatment_valid: bool
    validity_flags: Mapping[str, bool]
    comparison_status: str


@dataclass(frozen=True, slots=True)
class LoadedFunctionPreservingWidthControlArtifact:
    """Authenticated tensor/report views of one local publication."""

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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measurement_evidence_sha256(
    evidence: Mapping[str, object],
) -> str:
    if any(name not in evidence for name in _MEASUREMENT_EVIDENCE_FIELDS):
        raise ValueError("paired width measurement evidence is incomplete")
    if (
        not isinstance(evidence["pilot_metrics"], (tuple, list))
        or not isinstance(evidence["pilot_measurement"], Mapping)
        or not isinstance(evidence["fit_measurement"], Mapping)
        or not isinstance(
            evidence["fit_provider_chart_mismatch_diagnostics"],
            (tuple, list),
        )
        or not isinstance(evidence["teacher_signal_diagnostics"], Mapping)
        or not isinstance(
            evidence["training_teacher_signal_diagnostics"],
            Mapping,
        )
        or not isinstance(evidence["pair_balance"], Mapping)
        or not isinstance(evidence["gauge"], Mapping)
    ):
        raise TypeError("paired width measurement evidence types drifted")
    payload = {
        "artifact_kind": (
            "fisher_graph.function_preserving_width_measurement_evidence"
        ),
        "format_version": _FORMAT_VERSION,
        **{
            name: evidence[name]
            for name in _MEASUREMENT_EVIDENCE_FIELDS
        },
    }
    d0d3._assert_safe_artifact_tree(payload, path="measurement_evidence")
    d0d3._assert_tensor_free_report(payload)
    return _json_sha256(
        payload,
        domain=_MEASUREMENT_EVIDENCE_DOMAIN,
    )


def _code_sha256s() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    values = dict(r64._code_sha256s())
    for name in (
        "gemma3_l3_l4_function_preserving_width_control.py",
        "gemma3_l3_l4_function_preserving_width_control_protocol.py",
    ):
        values[name] = _file_sha256(root / name)
    return dict(sorted(values.items()))


def _code_bundle_sha256(values: Mapping[str, str]) -> str:
    return _json_sha256(dict(values), domain=_CODE_DOMAIN)


def _executor_config(spec: WidthExecutorSpec) -> GatedCausalModalExecutorConfig:
    return GatedCausalModalExecutorConfig(
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


def _objective(protocol: FunctionPreservingWidthControlProtocol) -> (
    ContrastAwareObjective
):
    training = protocol.training
    return ContrastAwareObjective(
        pointwise_weight=training.pointwise_weight,
        sensitivity_relative_delta_weight=training.relative_delta_weight,
        sensitivity_direction_weight=training.direction_weight,
        midpoint_jvp_weight=training.midpoint_jvp_weight,
        intended_null_weight=training.intended_null_weight,
        sensitivity_relative_floor=training.sensitivity_relative_floor,
        direction_norm_floor=training.direction_norm_floor,
        jvp_relative_floor=training.jvp_relative_floor,
    )


def _source_replay_exact(
    *,
    pair_role: str,
    arm: str,
    plan: ContrastAwareReferenceProviderPlan,
    protocol: FunctionPreservingWidthControlProtocol,
) -> bool:
    """Recompute the one row that can be an exact authenticated D3 replay."""

    if pair_role != "primary" or arm != "rank16":
        return False
    return (
        plan.artifact_sha256 == protocol.sources.d3_primary_plan_sha256
        and plan.initial_metrics.artifact_sha256
        == protocol.training.primary_initial_metrics_sha256
        and plan.final_metrics.artifact_sha256
        == _D3_PRIMARY_FINAL_METRICS_SHA256
    )


def _authenticated_declarations() -> tuple[
    FunctionPreservingWidthControlProtocol,
    object,
    object,
    object,
]:
    protocol = default_function_preserving_width_control_protocol()
    objective_protocol = default_objective_balance_diagnostic_protocol()
    c2_protocol = default_contrast_provider_development_protocol()
    rank64_protocol = default_rank64_capacity_control_protocol()
    d3 = objective_protocol.recipe(protocol.training.recipe_id)
    training = protocol.training
    if (
        protocol.protocol_sha256
        != DEFAULT_FUNCTION_PRESERVING_WIDTH_CONTROL_PROTOCOL_SHA256
        or objective_protocol.protocol_sha256
        != DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256
        or rank64_protocol.protocol_sha256
        != DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256
        or protocol.sources.d3_protocol_sha256
        != objective_protocol.protocol_sha256
        or protocol.sources.rank64_protocol_sha256
        != rank64_protocol.protocol_sha256
        or d3.artifact_sha256 != protocol.sources.d3_recipe_sha256
        or d3.primary_seed != training.primary_seed
        or objective_protocol.replication_seed != training.replication_seed
        or d3.signed_pair_multiplicity
        != training.signed_pair_multiplicity
        or d3.pointwise_weight != training.pointwise_weight
        or d3.sensitivity_relative_delta_weight
        != training.relative_delta_weight
        or d3.direction_weight != training.direction_weight
        or d3.midpoint_jvp_weight != training.midpoint_jvp_weight
        or d3.intended_null_weight != training.intended_null_weight
        or d3.steps != training.steps
        or d3.learning_rate != training.learning_rate
        or objective_protocol.gates.ordinary_gates_sha256
        != training.ordinary_gates_sha256
        or objective_protocol.gates.contrast_gates_sha256
        != training.contrast_gates_sha256
    ):
        raise ValueError("function-preserving declarations drifted")
    return protocol, objective_protocol, c2_protocol, d3


def _authenticate_sources(
    *,
    source_diagnostic_path: Path | str,
    source_rank64_path: Path | str,
    protocol: FunctionPreservingWidthControlProtocol,
) -> _AuthenticatedSources:
    d3_loaded = d0d3.load_objective_balance_diagnostic_artifact(
        source_diagnostic_path
    )
    source = protocol.sources
    if (
        d3_loaded.artifact_sha256 != source.d3_logical_artifact_sha256
        or d3_loaded.tensor_file_sha256 != source.d3_tensor_file_sha256
        or d3_loaded.report_sha256 != source.d3_report_sha256
        or d3_loaded.manifest.get("protocol_sha256")
        != source.d3_protocol_sha256
        or d3_loaded.manifest.get("code_bundle_sha256")
        != source.d3_code_bundle_sha256
        or d3_loaded.manifest.get("candidate_plan_sha256s", {}).get(
            _PRIMARY_D3_ID
        )
        != source.d3_primary_plan_sha256
        or d3_loaded.manifest.get("candidate_result_sha256s", {}).get(
            _PRIMARY_D3_ID
        )
        != source.d3_primary_result_sha256
    ):
        raise ValueError("authenticated D3 source drifted")
    rows = d3_loaded.state.get("candidate_results")
    if not isinstance(rows, Mapping) or not isinstance(
        rows.get(_PRIMARY_D3_ID),
        Mapping,
    ):
        raise ValueError("authenticated D3 primary row is absent")
    d3_row = rows[_PRIMARY_D3_ID]
    assert isinstance(d3_row, Mapping)
    if (
        d3_row.get("plan_sha256") != source.d3_primary_plan_sha256
        or d3_row.get("recipe_sha256") != source.d3_recipe_sha256
        or d3_row.get("fit_data_binding_sha256")
        != protocol.training.fit_data_binding_sha256
        or d3_row.get("latent_rank") != 16
        or d3_row.get("seed") != protocol.training.primary_seed
        or not isinstance(d3_row.get("final_training_metrics"), Mapping)
        or d3_row["final_training_metrics"].get("artifact_sha256")
        != _D3_PRIMARY_FINAL_METRICS_SHA256
    ):
        raise ValueError("authenticated D3 primary identity drifted")

    rank64_loaded = r64.load_rank64_capacity_control_artifact(
        source_rank64_path
    )
    if (
        rank64_loaded.artifact_sha256
        != source.rank64_logical_artifact_sha256
        or rank64_loaded.tensor_file_sha256
        != source.rank64_tensor_file_sha256
        or rank64_loaded.report_sha256 != source.rank64_report_sha256
        or rank64_loaded.manifest.get("protocol_sha256")
        != source.rank64_protocol_sha256
        or rank64_loaded.manifest.get("code_bundle_sha256")
        != source.rank64_code_bundle_sha256
        or rank64_loaded.manifest.get("candidate_plan_sha256s", {}).get(
            "r64_d3_capacity_control.primary"
        )
        != source.rank64_primary_plan_sha256
        or rank64_loaded.manifest.get("candidate_result_sha256s", {}).get(
            "r64_d3_capacity_control.primary"
        )
        != source.rank64_primary_result_sha256
        or rank64_loaded.manifest.get("outcome")
        != "invalid_rank_comparison_primary_treatment_validity_failed"
    ):
        raise ValueError("authenticated rank-64 predecessor drifted")

    rank64_protocol = default_rank64_capacity_control_protocol()
    source_d3 = r64._authenticate_source_d3(
        source_diagnostic_path,
        protocol=rank64_protocol,
    )
    return _AuthenticatedSources(
        source_d3=source_d3,
        d3_primary_row=d3_row,
    )


def describe_function_preserving_width_control() -> dict[str, object]:
    """Describe the sealed rung without loading a model or source artifact."""

    protocol, objective_protocol, c2_protocol, d3 = (
        _authenticated_declarations()
    )
    code = _code_sha256s()
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "protocol_trust_anchor": (
            DEFAULT_FUNCTION_PRESERVING_WIDTH_CONTROL_PROTOCOL_SHA256
        ),
        "protocol": protocol.state_dict(),
        "source_objective_protocol_sha256": (
            objective_protocol.protocol_sha256
        ),
        "source_rank64_protocol_sha256": (
            protocol.sources.rank64_protocol_sha256
        ),
        "source_d3_recipe_sha256": d3.artifact_sha256,
        "c2_protocol_sha256": c2_protocol.protocol_sha256,
        "source_artifacts_loaded": False,
        "model_loaded": False,
        "allowed_c2_roles": ("pilot", "fit"),
        "selection_allowed": False,
        "fresh_c3_authorized": False,
        "compression_claim_authorized": False,
        "post_run_scientific_authority_requires_external_trust_anchor_triple": (
            True
        ),
        "run_generated_receipt_verifies_publication_integrity_only": True,
        "run_generated_receipt_is_external_scientific_trust_root": False,
        "durable_scientific_authority_requires_recording_exact_receipt_triple_"
        "outside_artifact": True,
        "self_hashes_sufficient_for_scientific_authority": False,
        "primary_arms": (
            "rank16_exact_d3_replay",
            "rank64_function_preserving_lift",
        ),
        "decision_rule": (
            "paired_primary;replicate_only_valid_rank16_fail_rank64_pass;"
            "require_same_pattern_on_replication_for_width_attribution"
        ),
        "code_sha256s": code,
        "code_bundle_sha256": _code_bundle_sha256(code),
    }
    d0d3._assert_tensor_free_report(report)
    return report


def _new_training_model(
    *,
    modal_center: Tensor,
    gain_log_center: float,
    gain_log_scale: float,
    residual_width: int,
    rms_epsilon: float,
    target_center: Tensor,
    target_scale: Tensor,
    config: GatedCausalModalExecutorConfig,
    seed: int,
) -> contrast_fit._PackedTrainingModule:
    return contrast_fit._PackedTrainingModule(
        modal_center=modal_center,
        gain_log_center=gain_log_center,
        gain_log_scale=gain_log_scale,
        residual_width=residual_width,
        rms_epsilon=rms_epsilon,
        target_center=target_center,
        target_scale=target_scale,
        executor_config=config,
        seed=seed,
    )


def _lift_rank16_model(
    source: contrast_fit._PackedTrainingModule,
    *,
    modal_center: Tensor,
    gain_log_center: float,
    gain_log_scale: float,
    residual_width: int,
    rms_epsilon: float,
    target_center: Tensor,
    target_scale: Tensor,
    config: GatedCausalModalExecutorConfig,
    seed: int,
) -> contrast_fit._PackedTrainingModule:
    """Create the exact nested path and a decoder-gated active extra bank."""

    if source.encoder_weight.shape != (64, 16):
        raise ValueError("nested lift source must have rank 16")
    lifted = _new_training_model(
        modal_center=modal_center,
        gain_log_center=gain_log_center,
        gain_log_scale=gain_log_scale,
        residual_width=residual_width,
        rms_epsilon=rms_epsilon,
        target_center=target_center,
        target_scale=target_scale,
        config=config,
        seed=seed,
    )
    source_state = source.executor.state_dict()
    lifted_state = lifted.executor.state_dict()
    with torch.no_grad():
        lifted.encoder_weight.copy_(
            torch.eye(
                64,
                dtype=lifted.encoder_weight.dtype,
                device=lifted.encoder_weight.device,
            )
        )
        lifted.decoder_weight.zero_()
        lifted.decoder_weight[:16].copy_(source.decoder_weight)
        for tensor in lifted_state.values():
            tensor.zero_()

        # Active source inputs are constant, first 16 packed coordinates, gain.
        lifted_state["same_position_weight"][0, :16].copy_(
            source_state["same_position_weight"][0]
        )
        lifted_state["same_position_weight"][1:17, :16].copy_(
            source_state["same_position_weight"][1:17]
        )
        lifted_state["same_position_weight"][65, :16].copy_(
            source_state["same_position_weight"][17]
        )
        lifted_state["same_position_bias"][:16].copy_(
            source_state["same_position_bias"]
        )
        for index in range(48):
            lifted_state["same_position_weight"][
                17 + index,
                16 + index,
            ] = 1.0

        lifted_state["expert_input_weight"][:, 0].copy_(
            source_state["expert_input_weight"][:, 0]
        )
        lifted_state["expert_input_weight"][:, 1:17].copy_(
            source_state["expert_input_weight"][:, 1:17]
        )
        lifted_state["expert_input_weight"][:, 65].copy_(
            source_state["expert_input_weight"][:, 17]
        )
        lifted_state["expert_output_weight"][:, :, :16].copy_(
            source_state["expert_output_weight"]
        )
        for name in ("router_query_weight", "router_key_weight"):
            lifted_state[name][0].copy_(source_state[name][0])
            lifted_state[name][1:17].copy_(source_state[name][1:17])
            lifted_state[name][65].copy_(source_state[name][17])
        for name in (
            "router_output_weight",
            "router_bias",
            "router_lag_weight",
            "source_score_weight",
        ):
            lifted_state[name].copy_(source_state[name])
    return lifted


def _initial_parameter_hashes(
    model: contrast_fit._PackedTrainingModule,
) -> dict[str, str]:
    return {
        "encoder_sha256": d0d3._tensor_sha256(
            model.encoder_weight.detach()
        ),
        "executor_sha256": module_state_fingerprint(model.executor),
        "decoder_sha256": d0d3._tensor_sha256(
            model.decoder_weight.detach()
        ),
    }


def _reconstruct_initial_parameter_bindings(
    protocol: FunctionPreservingWidthControlProtocol,
    *,
    pair_role: str,
) -> dict[str, object]:
    if pair_role not in {"primary", "replication"}:
        raise ValueError("initial parameter seed role is invalid")
    seed = (
        protocol.training.primary_seed
        if pair_role == "primary"
        else protocol.training.replication_seed
    )
    modal_center = torch.zeros(64, dtype=torch.float64)
    target_center = torch.zeros(64, dtype=torch.float64)
    target_scale = torch.ones(64, dtype=torch.float64)
    rank16 = _new_training_model(
        modal_center=modal_center,
        gain_log_center=0.0,
        gain_log_scale=1.0,
        residual_width=64,
        rms_epsilon=1e-6,
        target_center=target_center,
        target_scale=target_scale,
        config=_executor_config(protocol.rank16_executor),
        seed=seed,
    )
    rank64 = _lift_rank16_model(
        rank16,
        modal_center=modal_center,
        gain_log_center=0.0,
        gain_log_scale=1.0,
        residual_width=64,
        rms_epsilon=1e-6,
        target_center=target_center,
        target_scale=target_scale,
        config=_executor_config(protocol.rank64_executor),
        seed=seed,
    )
    observed = {
        "seed_role": pair_role,
        "seed": seed,
        "protocol_initialization_sha256": (
            protocol.initialization.artifact_sha256
        ),
        "rank16": _initial_parameter_hashes(rank16),
        "rank64": _initial_parameter_hashes(rank64),
    }
    if (
        observed["rank16"]
        != protocol.initialization.hashes_for(
            seed_role=pair_role,
            arm="rank16",
        )
        or observed["rank64"]
        != protocol.initialization.hashes_for(
            seed_role=pair_role,
            arm="rank64",
        )
    ):
        raise RuntimeError("deterministic initial parameter binding drifted")
    return observed


def _difference_metrics(left: Tensor, right: Tensor) -> tuple[float, float]:
    difference = left - right
    maximum = float(difference.detach().abs().max()) if difference.numel() else 0.0
    denominator = max(
        float(torch.linalg.vector_norm(left.detach())),
        float(torch.linalg.vector_norm(right.detach())),
        torch.finfo(left.dtype).tiny,
    )
    relative = float(torch.linalg.vector_norm(difference.detach())) / denominator
    return maximum, relative


def _initial_equivalence(
    rank16: contrast_fit._PackedTrainingModule,
    rank64: contrast_fit._PackedTrainingModule,
    *,
    data: contrast_fit._PreparedFitData,
    target_center: Tensor,
    target_scale: Tensor,
    metric_weight: Tensor,
    objective: ContrastAwareObjective,
    protocol: FunctionPreservingWidthControlProtocol,
    pair_role: str,
    seed: int,
) -> dict[str, object]:
    expected_parameters = _reconstruct_initial_parameter_bindings(
        protocol,
        pair_role=pair_role,
    )
    if (
        seed != expected_parameters["seed"]
        or _initial_parameter_hashes(rank16)
        != expected_parameters["rank16"]
        or _initial_parameter_hashes(rank64)
        != expected_parameters["rank64"]
    ):
        raise ValueError("live initial parameters differ from frozen binding")
    maximum_output_absolute = 0.0
    maximum_output_relative = 0.0
    for indexed in data.batches:
        batch = indexed.batch
        left = rank16.forward_standardized(
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
            batch.logical_positions,
        )
        right = rank64.forward_standardized(
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
            batch.logical_positions,
        )
        absolute, relative = _difference_metrics(left, right)
        maximum_output_absolute = max(maximum_output_absolute, absolute)
        maximum_output_relative = max(maximum_output_relative, relative)

    maximum_jvp_absolute = 0.0
    maximum_jvp_relative = 0.0
    jvp_count = 0
    for pair in data.pairs:
        if pair.teacher_midpoint_jvp is None:
            continue
        indexed, row = contrast_fit._pair_row(
            data,
            pair.left_endpoint_id,
        )
        batch = indexed.batch
        chart = tuple(
            getattr(pair, name)
            for name in contrast_fit._PROVIDER_CHART_FIELDS
        )
        if any(value is None for value in chart):
            raise RuntimeError("provider-chart JVP input is incomplete")
        (
            modal_primal,
            null_primal,
            rms_primal,
            modal_tangent,
            null_tangent,
            rms_tangent,
        ) = chart
        assert all(isinstance(value, Tensor) for value in chart)
        mask = batch.valid_mask[row : row + 1]
        positions = batch.logical_positions[row : row + 1]

        def apply(
            model: contrast_fit._PackedTrainingModule,
            modal: Tensor,
            null: Tensor,
            rms: Tensor,
        ) -> Tensor:
            return model.forward_standardized(
                modal,
                null,
                rms,
                mask,
                positions,
            )

        args = (
            modal_primal.unsqueeze(0),  # type: ignore[union-attr]
            null_primal.unsqueeze(0),  # type: ignore[union-attr]
            rms_primal.unsqueeze(0),  # type: ignore[union-attr]
        )
        tangents = (
            modal_tangent.unsqueeze(0),  # type: ignore[union-attr]
            null_tangent.unsqueeze(0),  # type: ignore[union-attr]
            rms_tangent.unsqueeze(0),  # type: ignore[union-attr]
        )
        _, left_jvp = torch.func.jvp(
            lambda modal, null, rms: apply(
                rank16,
                modal,
                null,
                rms,
            ),
            args,
            tangents,
        )
        _, right_jvp = torch.func.jvp(
            lambda modal, null, rms: apply(
                rank64,
                modal,
                null,
                rms,
            ),
            args,
            tangents,
        )
        absolute, relative = _difference_metrics(left_jvp, right_jvp)
        maximum_jvp_absolute = max(maximum_jvp_absolute, absolute)
        maximum_jvp_relative = max(maximum_jvp_relative, relative)
        jvp_count += 1

    initial16 = contrast_fit._materialize_metrics(
        contrast_fit._loss_components(
            rank16,
            data=data,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=metric_weight,
            objective=objective,
        ),
        data=data,
    )
    initial64 = contrast_fit._materialize_metrics(
        contrast_fit._loss_components(
            rank64,
            data=data,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=metric_weight,
            objective=objective,
        ),
        data=data,
    )
    tolerance_abs = protocol.lift.equivalence_absolute_tolerance
    tolerance_rel = protocol.lift.equivalence_relative_tolerance
    flags = {
        "observable_absolute": maximum_output_absolute <= tolerance_abs,
        "observable_relative": maximum_output_relative <= tolerance_rel,
        "jvp_absolute": maximum_jvp_absolute <= tolerance_abs,
        "jvp_relative": maximum_jvp_relative <= tolerance_rel,
        "initial_metrics_exact": (
            _canonical_json_bytes(initial16.state_dict())
            == _canonical_json_bytes(initial64.state_dict())
        ),
        "all_expected_jvps_compared": jvp_count == 32,
    }
    state = {
        "artifact_kind": "fisher_graph.initial_width_equivalence",
        "format_version": _FORMAT_VERSION,
        "maximum_observable_absolute_error": maximum_output_absolute,
        "maximum_observable_relative_error": maximum_output_relative,
        "maximum_jvp_absolute_error": maximum_jvp_absolute,
        "maximum_jvp_relative_error": maximum_jvp_relative,
        "jvp_pair_count": jvp_count,
        "rank16_initial_metrics_sha256": initial16.artifact_sha256,
        "rank64_initial_metrics_sha256": initial64.artifact_sha256,
        "initial_parameter_bindings": expected_parameters,
        "flags": flags,
        "passed": all(flags.values()),
    }
    state["artifact_sha256"] = _json_sha256(
        state,
        domain=_INITIALIZATION_DOMAIN,
    )
    return state


def _fit_from_initialized_model(
    model: contrast_fit._PackedTrainingModule,
    *,
    data: contrast_fit._PreparedFitData,
    target_center: Tensor,
    target_scale: Tensor,
    metric_weight: Tensor,
    objective: ContrastAwareObjective,
    steps: int,
    learning_rate: float,
    seed: int,
    synthetic_binding_sha256: str,
    audit_added_width: bool,
    protocol: FunctionPreservingWidthControlProtocol,
) -> tuple[ContrastAwareReferenceProviderPlan, dict[str, object]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    initial = contrast_fit._materialize_metrics(
        contrast_fit._loss_components(
            model,
            data=data,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=metric_weight,
            objective=objective,
        ),
        data=data,
    )
    decoder_step1 = 0.0
    encoder_step2 = 0.0
    executor_step2 = 0.0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        components = contrast_fit._loss_components(
            model,
            data=data,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=metric_weight,
            objective=objective,
        )
        if not bool(torch.isfinite(components.weighted_total)):
            raise ValueError("paired width fit produced a nonfinite loss")
        components.weighted_total.backward()
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise ValueError("paired width fit produced a nonfinite gradient")
        if audit_added_width and step == 0:
            gradient = model.decoder_weight.grad
            assert gradient is not None
            decoder_step1 = float(
                torch.linalg.vector_norm(gradient[16:]).detach()
            )
        if audit_added_width and step == 1:
            encoder_gradient = model.encoder_weight.grad
            executor_gradient = model.executor.same_position_weight.grad
            assert encoder_gradient is not None
            assert executor_gradient is not None
            encoder_step2 = float(
                torch.linalg.vector_norm(
                    encoder_gradient[:, 16:]
                ).detach()
            )
            executor_step2 = float(
                torch.linalg.vector_norm(
                    executor_gradient[17:65, 16:]
                ).detach()
            )
        optimizer.step()
    model.eval()
    final = contrast_fit._materialize_metrics(
        contrast_fit._loss_components(
            model,
            data=data,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=metric_weight,
            objective=objective,
        ),
        data=data,
    )
    bindings = {
        value.batch.synthetic_binding_sha256 for value in data.batches
    }
    if bindings != {synthetic_binding_sha256}:
        raise ValueError("fit batches lost their synthetic binding")
    plan = ContrastAwareReferenceProviderPlan(
        modal_center=model.modal_center,
        gain_log_center=model.gain_log_center,
        gain_log_scale=model.gain_log_scale,
        residual_width=model.residual_width,
        rms_epsilon=model.rms_epsilon,
        target_center=model.target_center,
        target_scale=model.target_scale,
        encoder_weight=model.encoder_weight.detach(),
        executor_artifact=model.executor.artifact_state_dict(),
        decoder_weight=model.decoder_weight.detach(),
        fisher_metric_weight=metric_weight,
        fisher_metric_supplied=True,
        synthetic_binding_sha256=synthetic_binding_sha256,
        fit_batch_sha256s=tuple(
            value.batch.artifact_sha256 for value in data.batches
        ),
        fit_batch_content_sha256s=tuple(
            value.batch.content_sha256 for value in data.batches
        ),
        fit_indexed_batch_sha256s=tuple(
            value.artifact_sha256 for value in data.batches
        ),
        fit_endpoint_sha256s=tuple(
            location.endpoint_sha256
            for _, location in sorted(data.endpoints.items())
        ),
        fit_pair_sha256s=tuple(
            value.artifact_sha256 for value in data.pairs
        ),
        objective=objective,
        training_steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        initial_metrics=initial,
        final_metrics=final,
    )
    floor = protocol.lift.gradient_norm_floor
    flags = {
        "extra_decoder_gradient_step1": (
            not audit_added_width or decoder_step1 > floor
        ),
        "extra_encoder_gradient_step2": (
            not audit_added_width or encoder_step2 > floor
        ),
        "extra_executor_gradient_step2": (
            not audit_added_width or executor_step2 > floor
        ),
    }
    audit = {
        "artifact_kind": "fisher_graph.width_gradient_openness",
        "format_version": _FORMAT_VERSION,
        "applicable": audit_added_width,
        "extra_decoder_gradient_norm_step1": decoder_step1,
        "extra_encoder_gradient_norm_step2": encoder_step2,
        "extra_executor_gradient_norm_step2": executor_step2,
        "gradient_norm_floor": floor,
        "flags": flags,
        "passed": all(flags.values()),
    }
    audit["artifact_sha256"] = _json_sha256(
        audit,
        domain=_INITIALIZATION_DOMAIN,
    )
    return plan, audit


def _gradient_audit_contract(
    protocol: FunctionPreservingWidthControlProtocol,
    *,
    pair_role: str,
    arm: str,
) -> dict[str, object]:
    if pair_role not in {"primary", "replication"}:
        raise ValueError("gradient audit seed role is invalid")
    if arm not in {"rank16", "rank64"}:
        raise ValueError("gradient audit arm is invalid")
    applicable = arm == "rank64"
    prefix = pair_role
    decoder = (
        float(
            getattr(
                protocol.training,
                f"{prefix}_extra_decoder_gradient_norm_step1",
            )
        )
        if applicable
        else 0.0
    )
    encoder = (
        float(
            getattr(
                protocol.training,
                f"{prefix}_extra_encoder_gradient_norm_step2",
            )
        )
        if applicable
        else 0.0
    )
    executor = (
        float(
            getattr(
                protocol.training,
                f"{prefix}_extra_executor_gradient_norm_step2",
            )
        )
        if applicable
        else 0.0
    )
    floor = protocol.lift.gradient_norm_floor
    flags = {
        "extra_decoder_gradient_step1": (
            not applicable or decoder > floor
        ),
        "extra_encoder_gradient_step2": (
            not applicable or encoder > floor
        ),
        "extra_executor_gradient_step2": (
            not applicable or executor > floor
        ),
    }
    state = {
        "artifact_kind": "fisher_graph.width_gradient_openness",
        "format_version": _FORMAT_VERSION,
        "applicable": applicable,
        "extra_decoder_gradient_norm_step1": decoder,
        "extra_encoder_gradient_norm_step2": encoder,
        "extra_executor_gradient_norm_step2": executor,
        "gradient_norm_floor": floor,
        "flags": flags,
        "passed": all(flags.values()),
    }
    state["artifact_sha256"] = _json_sha256(
        state,
        domain=_INITIALIZATION_DOMAIN,
    )
    if applicable and state["artifact_sha256"] != getattr(
        protocol.training,
        f"{prefix}_gradient_audit_sha256",
    ):
        raise RuntimeError("frozen real-fit gradient audit drifted")
    return state


def _score_plan(
    *,
    candidate_id: str,
    pair_role: str,
    arm: str,
    plan: ContrastAwareReferenceProviderPlan,
    gradient_audit: Mapping[str, object],
    initialization_audit: Mapping[str, object],
    protocol: FunctionPreservingWidthControlProtocol,
    objective_protocol: object,
    d3_recipe: object,
    source: _AuthenticatedSources,
    fit: Sequence[object],
    raw_metric_weight: Tensor,
    raw_teacher_energy: float,
    training_teacher_energy: float,
    teacher_signal_diagnostics: Mapping[str, object],
    ordinary_probes: Sequence[object],
    controls: object,
    standardized_gauge_sha256: str,
    fidelity_gates: SyntheticReferenceGates,
    contrast_gates: ContrastAssessmentGates,
) -> _ArmEvaluation:
    support_radius = c2._feature_radius(plan, fit)
    candidate, ordinary, predictions, structural = (
        d0d3._fit_only_ordinary_candidate_and_score(
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
    )
    contrast, identities, coverage = d0d3._fit_contrast_assessment(
        protocol=default_contrast_provider_development_protocol(),
        measured=fit,
        predictions=predictions,
        metric_weight=raw_metric_weight,
        gates=contrast_gates,
        required_null_candidate_pass_count=(
            objective_protocol.gates.required_null_candidate_pass_count
        ),
    )
    balance = d0d3._contribution_balance_gate(
        plan,
        recipe=d3_recipe,
        gates=objective_protocol.gates,
        training_teacher_energy=training_teacher_energy,
        raw_teacher_energy=raw_teacher_energy,
        teacher_signal_diagnostics=teacher_signal_diagnostics,
    )
    ordinary_flags = ordinary.gate_flags.state_dict()
    ordinary_values = tuple(
        passed
        for name, passed in ordinary_flags.items()
        if name != "all_passed"
    )
    ordinary_pass = (
        len(ordinary_values)
        == objective_protocol.gates.required_ordinary_gate_count
        and all(ordinary_values)
    )
    fit_pass = (
        ordinary_pass
        and contrast.overall_status == "pass"
        and bool(coverage["every_teacher_qualified_contrast_passed"])
        and bool(coverage["all_families_cover_all_four_rank_bands"])
        and bool(coverage["required_null_contrasts_valid_and_passed"])
    )
    sequence = r64._plan_sequence_comparison(
        plan,
        source=source.source_d3,
    )
    source_replay = _source_replay_exact(
        pair_role=pair_role,
        arm=arm,
        plan=plan,
        protocol=protocol,
    )
    row = {
        "candidate_id": candidate_id,
        "pair_role": pair_role,
        "arm": arm,
        "seed": plan.seed,
        "latent_rank": plan.latent_rank,
        "expert_rank": plan.executor_config.expert_rank,
        "plan_sha256": plan.artifact_sha256,
        "candidate_binding_sha256": candidate.artifact_sha256,
        "source_replay_exact": source_replay,
        "source_sequence_comparison": sequence,
        "initialization_equivalence": dict(initialization_audit),
        "gradient_openness": dict(gradient_audit),
        "initial_training_metrics": plan.initial_metrics.state_dict(),
        "final_training_metrics": plan.final_metrics.state_dict(),
        "final_contribution_audit": audit_objective_contributions(
            plan.final_metrics,
            plan.objective,
        ).state_dict(),
        "objective_balance_gate": balance,
        "ordinary_score": ordinary.state_dict(),
        "contrast_result": contrast.state_dict(),
        "contrast_identities": identities,
        "contrast_coverage": coverage,
        "structural_metadata": structural,
        "accounting": asdict(plan.accounting()),
        "fit_capability_contract": {
            "ordinary_gate_count": len(ordinary_values),
            "all_ordinary_gates_passed": ordinary_pass,
            "all_contrast_families_passed": (
                contrast.overall_status == "pass"
            ),
            "every_qualified_contrast_passed": bool(
                coverage["every_teacher_qualified_contrast_passed"]
            ),
            "all_four_rank_bands_covered": bool(
                coverage["all_families_cover_all_four_rank_bands"]
            ),
            "required_null_contrasts_passed": bool(
                coverage["required_null_contrasts_valid_and_passed"]
            ),
        },
        "fit_capability_pass": fit_pass,
    }
    return _ArmEvaluation(
        candidate_id=candidate_id,
        pair_role=pair_role,
        arm=arm,
        fit_capability_pass=fit_pass,
        row=row,
        plan=plan,
    )


def _pair_status(rank16_pass: bool, rank64_pass: bool) -> str:
    if not rank16_pass and not rank64_pass:
        return "both_fail"
    if not rank16_pass and rank64_pass:
        return "rank16_fail_rank64_pass"
    if rank16_pass and rank64_pass:
        return "both_pass"
    return "rank16_pass_rank64_fail"


def _evaluate_pair(
    *,
    pair_role: str,
    seed: int,
    protocol: FunctionPreservingWidthControlProtocol,
    objective_protocol: object,
    d3_recipe: object,
    source: _AuthenticatedSources,
    fit: Sequence[object],
    fit_batches: Sequence[object],
    fit_pairs: Sequence[object],
    modal_center: Tensor,
    gain_log_center: float,
    gain_log_scale: float,
    residual_width: int,
    rms_epsilon: float,
    target_center: Tensor,
    target_scale: Tensor,
    unit_metric_weight: Tensor,
    raw_metric_weight: Tensor,
    raw_teacher_energy: float,
    training_teacher_energy: float,
    teacher_signal_diagnostics: Mapping[str, object],
    fit_data_binding_sha256: str,
    ordinary_probes: Sequence[object],
    controls: object,
    standardized_gauge_sha256: str,
    fidelity_gates: SyntheticReferenceGates,
    contrast_gates: ContrastAssessmentGates,
) -> _PairEvaluation:
    if pair_role not in {"primary", "replication"}:
        raise ValueError("paired width role is invalid")
    expected_seed = (
        protocol.training.primary_seed
        if pair_role == "primary"
        else protocol.training.replication_seed
    )
    if seed != expected_seed:
        raise ValueError("paired width seed drifted")
    objective = _objective(protocol)
    data = contrast_fit._prepare_fit_data(
        fit_batches=fit_batches,
        contrast_pairs=fit_pairs,
        require_fit_split=True,
    )
    rank16_model = _new_training_model(
        modal_center=modal_center,
        gain_log_center=gain_log_center,
        gain_log_scale=gain_log_scale,
        residual_width=residual_width,
        rms_epsilon=rms_epsilon,
        target_center=target_center,
        target_scale=target_scale,
        config=_executor_config(protocol.rank16_executor),
        seed=seed,
    )
    rank64_model = _lift_rank16_model(
        rank16_model,
        modal_center=modal_center,
        gain_log_center=gain_log_center,
        gain_log_scale=gain_log_scale,
        residual_width=residual_width,
        rms_epsilon=rms_epsilon,
        target_center=target_center,
        target_scale=target_scale,
        config=_executor_config(protocol.rank64_executor),
        seed=seed,
    )
    initial = _initial_equivalence(
        rank16_model,
        rank64_model,
        data=data,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=unit_metric_weight,
        objective=objective,
        protocol=protocol,
        pair_role=pair_role,
        seed=seed,
    )
    plan16, gradient16 = _fit_from_initialized_model(
        rank16_model,
        data=data,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=unit_metric_weight,
        objective=objective,
        steps=protocol.training.steps,
        learning_rate=protocol.training.learning_rate,
        seed=seed,
        synthetic_binding_sha256=fit_data_binding_sha256,
        audit_added_width=False,
        protocol=protocol,
    )
    plan64, gradient64 = _fit_from_initialized_model(
        rank64_model,
        data=data,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=unit_metric_weight,
        objective=objective,
        steps=protocol.training.steps,
        learning_rate=protocol.training.learning_rate,
        seed=seed,
        synthetic_binding_sha256=fit_data_binding_sha256,
        audit_added_width=True,
        protocol=protocol,
    )
    expected_gradient16 = _gradient_audit_contract(
        protocol,
        pair_role=pair_role,
        arm="rank16",
    )
    expected_gradient64 = _gradient_audit_contract(
        protocol,
        pair_role=pair_role,
        arm="rank64",
    )
    if (
        _canonical_json_bytes(gradient16)
        != _canonical_json_bytes(expected_gradient16)
        or _canonical_json_bytes(gradient64)
        != _canonical_json_bytes(expected_gradient64)
    ):
        raise RuntimeError("real-fit gradient audit differs from protocol")
    prefix = f"fp_width.{pair_role}"
    arm16 = _score_plan(
        candidate_id=f"{prefix}.rank16",
        pair_role=pair_role,
        arm="rank16",
        plan=plan16,
        gradient_audit=gradient16,
        initialization_audit=initial,
        protocol=protocol,
        objective_protocol=objective_protocol,
        d3_recipe=d3_recipe,
        source=source,
        fit=fit,
        raw_metric_weight=raw_metric_weight,
        raw_teacher_energy=raw_teacher_energy,
        training_teacher_energy=training_teacher_energy,
        teacher_signal_diagnostics=teacher_signal_diagnostics,
        ordinary_probes=ordinary_probes,
        controls=controls,
        standardized_gauge_sha256=standardized_gauge_sha256,
        fidelity_gates=fidelity_gates,
        contrast_gates=contrast_gates,
    )
    arm64 = _score_plan(
        candidate_id=f"{prefix}.rank64",
        pair_role=pair_role,
        arm="rank64",
        plan=plan64,
        gradient_audit=gradient64,
        initialization_audit=initial,
        protocol=protocol,
        objective_protocol=objective_protocol,
        d3_recipe=d3_recipe,
        source=source,
        fit=fit,
        raw_metric_weight=raw_metric_weight,
        raw_teacher_energy=raw_teacher_energy,
        training_teacher_energy=training_teacher_energy,
        teacher_signal_diagnostics=teacher_signal_diagnostics,
        ordinary_probes=ordinary_probes,
        controls=controls,
        standardized_gauge_sha256=standardized_gauge_sha256,
        fidelity_gates=fidelity_gates,
        contrast_gates=contrast_gates,
    )
    flags = {
        "initial_observable_and_jvp_equivalence": bool(initial["passed"]),
        "gradient_open_rank64": bool(gradient64["passed"]),
        "rank16_balance": bool(
            arm16.row["objective_balance_gate"]["passed"]  # type: ignore[index]
        ),
        "rank64_balance": bool(
            arm64.row["objective_balance_gate"]["passed"]  # type: ignore[index]
        ),
        "rank16_source_sequences": bool(
            arm16.row["source_sequence_comparison"]["passed"]  # type: ignore[index]
        ),
        "rank64_source_sequences": bool(
            arm64.row["source_sequence_comparison"]["passed"]  # type: ignore[index]
        ),
        "rank16_primary_replay": (
            arm16.row["source_replay_exact"] is (pair_role == "primary")
            and arm64.row["source_replay_exact"] is False
        ),
        "initial_metrics_match": (
            plan16.initial_metrics.artifact_sha256
            == plan64.initial_metrics.artifact_sha256
        ),
        "exact_configs": (
            asdict(plan16.executor_config)
            == asdict(_executor_config(protocol.rank16_executor))
            and asdict(plan64.executor_config)
            == asdict(_executor_config(protocol.rank64_executor))
        ),
        "exact_training_contract": (
            plan16.training_steps
            == plan64.training_steps
            == protocol.training.steps
            and plan16.learning_rate
            == plan64.learning_rate
            == protocol.training.learning_rate
            and plan16.objective.artifact_sha256
            == plan64.objective.artifact_sha256
            == objective.artifact_sha256
        ),
    }
    valid = all(flags.values())
    status = _pair_status(
        arm16.fit_capability_pass,
        arm64.fit_capability_pass,
    )
    for arm in (arm16, arm64):
        arm.row["pair_treatment_validity"] = {
            "passed": valid,
            "flags": flags,
            "failure_semantics": (
                "invalid_paired_width_comparison_no_capacity_conclusion"
            ),
        }
        arm.row["pair_comparison_status"] = status
    return _PairEvaluation(
        pair_role=pair_role,
        seed=seed,
        rank16=arm16,
        rank64=arm64,
        treatment_valid=valid,
        validity_flags=flags,
        comparison_status=status,
    )


def _decision(
    primary: _PairEvaluation,
    replication: _PairEvaluation | None,
) -> dict[str, object]:
    if not primary.treatment_valid:
        outcome = "invalid_primary_pair"
    elif primary.comparison_status != "rank16_fail_rank64_pass":
        outcome = f"primary_{primary.comparison_status}"
    elif replication is None:
        raise RuntimeError("authorized replication was not executed")
    elif not replication.treatment_valid:
        outcome = "invalid_replication_pair"
    elif replication.comparison_status == "rank16_fail_rank64_pass":
        outcome = "two_seed_outer_width_support"
    else:
        outcome = f"replication_{replication.comparison_status}"
    return {
        "outcome": outcome,
        "primary_treatment_valid": primary.treatment_valid,
        "primary_comparison_status": primary.comparison_status,
        "replication_executed": replication is not None,
        "replication_treatment_valid": (
            None if replication is None else replication.treatment_valid
        ),
        "replication_comparison_status": (
            None if replication is None else replication.comparison_status
        ),
        "two_seed_outer_width_supported": (
            outcome == "two_seed_outer_width_support"
        ),
        "expert_core_control_authorized": (
            outcome in {"primary_both_fail", "replication_both_fail"}
        ),
        "compressed_width_ladder_authorized": (
            outcome == "two_seed_outer_width_support"
        ),
        "fresh_c3_authorized": False,
        "compression_claim_authorized": False,
    }


def _validate_output_path(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("paired width output must use a .pt suffix")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite paired width output")
    resolved = destination.expanduser().resolve()
    worktree = find_git_worktree(Path(__file__))
    if worktree is not None:
        root = worktree.resolve()
        if resolved == root or root in resolved.parents:
            relative = resolved.relative_to(root)
            if (
                not relative.parts
                or relative.parts[0] not in {".local-runs", "local-runs"}
            ):
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
) -> dict[str, str]:
    d0d3._assert_safe_artifact_tree(state)
    d0d3._assert_safe_artifact_tree(report_payload, path="report")
    d0d3._assert_tensor_free_report(report_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite paired width output")
    tensor_stage = _stage_path(output)
    report_stage = _stage_path(report_path)
    published: list[Path] = []
    receipt: dict[str, str] | None = None
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
        receipt = {
            "expected_artifact_sha256": _require_sha256(
                state.get("artifact_sha256"),
                label="published logical artifact SHA-256",
            ),
            "expected_tensor_file_sha256": str(
                report["artifact"]["tensor_file_sha256"]
            ),
            "expected_report_sha256": str(report["report_sha256"]),
        }
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        tensor_stage.unlink(missing_ok=True)
        report_stage.unlink(missing_ok=True)
    if receipt is None:
        raise RuntimeError("paired width publication produced no receipt")
    return receipt


def _read_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.read_bytes()


_STATE_FIELDS = {
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
_MANIFEST_FIELDS = {
    "schema",
    "format_version",
    "protocol_sha256",
    "source_d3_logical_artifact_sha256",
    "source_d3_tensor_file_sha256",
    "source_d3_report_sha256",
    "source_d3_primary_plan_sha256",
    "source_d3_primary_result_sha256",
    "source_rank64_logical_artifact_sha256",
    "source_rank64_tensor_file_sha256",
    "source_rank64_report_sha256",
    "source_rank64_primary_plan_sha256",
    "source_rank64_primary_result_sha256",
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
    "measurement_evidence_sha256",
    "executed_candidate_ids",
    "candidate_plan_sha256s",
    "candidate_result_sha256s",
    "outcome",
    "primary_treatment_valid",
    "primary_comparison_status",
    "replication_executed",
    "replication_treatment_valid",
    "replication_comparison_status",
    "two_seed_outer_width_supported",
    "expert_core_control_authorized",
    "compressed_width_ladder_authorized",
    "fresh_c3_authorized",
    "compression_claim_authorized",
    "selection_materialized",
    "selection_measured",
    "selection_scored",
    "c2_provider_artifact_loaded",
    "authenticated_d3_source_loaded",
    "authenticated_rank64_source_loaded",
    "source_final_parameters_used_for_initialization",
    "v2_targets_loaded",
    "v3_targets_loaded",
    "prompt_text_loaded",
    "token_ids_loaded",
    "tokenizer_loaded",
    "natural_activation_rows_loaded",
    "code_sha256s",
    "code_bundle_sha256",
    "scientific_scope",
}
_REPORT_EXTRA_FIELDS = {
    "artifact_sha256",
    "protocol",
    "calibration",
    "pilot_metrics",
    "pilot_measurement",
    "fit_measurement",
    "fit_provider_chart_mismatch_diagnostics",
    "teacher_signal_diagnostics",
    "training_teacher_signal_diagnostics",
    "pair_balance",
    "gauge",
    "candidate_results",
    "interpretation",
    "safety",
    "artifact",
    "report_sha256",
}
_ROW_FIELDS = {
    "candidate_id",
    "pair_role",
    "arm",
    "seed",
    "latent_rank",
    "expert_rank",
    "plan_sha256",
    "candidate_binding_sha256",
    "source_replay_exact",
    "source_sequence_comparison",
    "initialization_equivalence",
    "gradient_openness",
    "initial_training_metrics",
    "final_training_metrics",
    "final_contribution_audit",
    "objective_balance_gate",
    "ordinary_score",
    "contrast_result",
    "contrast_identities",
    "contrast_coverage",
    "structural_metadata",
    "accounting",
    "fit_capability_contract",
    "fit_capability_pass",
    "pair_treatment_validity",
    "pair_comparison_status",
}
_INITIALIZATION_EQUIVALENCE_FIELDS = {
    "artifact_kind",
    "format_version",
    "maximum_observable_absolute_error",
    "maximum_observable_relative_error",
    "maximum_jvp_absolute_error",
    "maximum_jvp_relative_error",
    "jvp_pair_count",
    "rank16_initial_metrics_sha256",
    "rank64_initial_metrics_sha256",
    "initial_parameter_bindings",
    "flags",
    "passed",
    "artifact_sha256",
}
_PAIR_TREATMENT_VALIDITY_FIELDS = {
    "passed",
    "flags",
    "failure_semantics",
}
_SOURCE_SEQUENCE_COMPARISON_FIELDS = {
    "passed",
    "flags",
    "source_sequence_sha256s",
    "candidate_sequence_sha256s",
    "comparison_semantics",
}
_STRUCTURAL_METADATA_FIELDS = {
    "support_radius",
    "support_rule",
    "invalid_padding_rows_tested",
    "nonvacuous_padding_test",
}


def _recompute_decision_from_rows(
    rows: Mapping[str, object],
) -> dict[str, object]:
    def pair(role: str) -> tuple[bool, str] | None:
        rank16 = rows.get(f"fp_width.{role}.rank16")
        rank64 = rows.get(f"fp_width.{role}.rank64")
        if rank16 is None and rank64 is None:
            return None
        if not isinstance(rank16, Mapping) or not isinstance(rank64, Mapping):
            raise ValueError("paired rows are incomplete")
        validity16 = rank16.get("pair_treatment_validity")
        validity64 = rank64.get("pair_treatment_validity")
        if not isinstance(validity16, Mapping) or not isinstance(
            validity64,
            Mapping,
        ):
            raise ValueError("paired treatment validity is absent")
        if _canonical_json_bytes(validity16) != _canonical_json_bytes(
            validity64
        ):
            raise ValueError("paired treatment validity differs by arm")
        if (
            set(validity16) != _PAIR_TREATMENT_VALIDITY_FIELDS
            or validity16.get("failure_semantics")
            != "invalid_paired_width_comparison_no_capacity_conclusion"
        ):
            raise ValueError("paired treatment validity schema drifted")
        initial16 = rank16.get("initialization_equivalence")
        initial64 = rank64.get("initialization_equivalence")
        gradient64 = rank64.get("gradient_openness")
        balance16 = rank16.get("objective_balance_gate")
        balance64 = rank64.get("objective_balance_gate")
        sequences16 = rank16.get("source_sequence_comparison")
        sequences64 = rank64.get("source_sequence_comparison")
        metrics16 = rank16.get("initial_training_metrics")
        metrics64 = rank64.get("initial_training_metrics")
        for name, value in (
            ("rank16 initialization", initial16),
            ("rank64 initialization", initial64),
            ("rank64 gradient audit", gradient64),
            ("rank16 balance", balance16),
            ("rank64 balance", balance64),
            ("rank16 sequences", sequences16),
            ("rank64 sequences", sequences64),
            ("rank16 initial metrics", metrics16),
            ("rank64 initial metrics", metrics64),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
        assert isinstance(initial16, Mapping)
        assert isinstance(initial64, Mapping)
        assert isinstance(gradient64, Mapping)
        assert isinstance(balance16, Mapping)
        assert isinstance(balance64, Mapping)
        assert isinstance(sequences16, Mapping)
        assert isinstance(sequences64, Mapping)
        assert isinstance(metrics16, Mapping)
        assert isinstance(metrics64, Mapping)
        expected_flags = {
            "initial_observable_and_jvp_equivalence": (
                initial16.get("passed") is True
                and _canonical_json_bytes(initial16)
                == _canonical_json_bytes(initial64)
            ),
            "gradient_open_rank64": gradient64.get("passed") is True,
            "rank16_balance": balance16.get("passed") is True,
            "rank64_balance": balance64.get("passed") is True,
            "rank16_source_sequences": sequences16.get("passed") is True,
            "rank64_source_sequences": sequences64.get("passed") is True,
            "rank16_primary_replay": (
                rank16.get("source_replay_exact") is (role == "primary")
                and rank64.get("source_replay_exact") is False
            ),
            "initial_metrics_match": (
                _canonical_json_bytes(metrics16)
                == _canonical_json_bytes(metrics64)
            ),
            "exact_configs": True,
            "exact_training_contract": True,
        }
        supplied_flags = validity16.get("flags")
        if (
            not isinstance(supplied_flags, Mapping)
            or dict(supplied_flags) != expected_flags
        ):
            raise ValueError("paired treatment flags drifted")
        valid = all(expected_flags.values())
        if validity16.get("passed") is not valid:
            raise ValueError("paired treatment validity decision drifted")
        status = _pair_status(
            bool(rank16.get("fit_capability_pass")),
            bool(rank64.get("fit_capability_pass")),
        )
        if (
            rank16.get("pair_comparison_status") != status
            or rank64.get("pair_comparison_status") != status
        ):
            raise ValueError("paired comparison status drifted")
        return valid, status

    primary_state = pair("primary")
    replication_state = pair("replication")
    if primary_state is None:
        raise ValueError("primary pair is absent")
    primary_valid, primary_status = primary_state
    if replication_state is not None and not (
        primary_valid
        and primary_status == "rank16_fail_rank64_pass"
    ):
        raise ValueError(
            "replication pair was not authorized by the primary result"
        )
    if not primary_valid:
        outcome = "invalid_primary_pair"
    elif primary_status != "rank16_fail_rank64_pass":
        outcome = f"primary_{primary_status}"
    elif replication_state is None:
        raise ValueError("authorized replication pair is absent")
    elif not replication_state[0]:
        outcome = "invalid_replication_pair"
    elif replication_state[1] == "rank16_fail_rank64_pass":
        outcome = "two_seed_outer_width_support"
    else:
        outcome = f"replication_{replication_state[1]}"
    return {
        "outcome": outcome,
        "primary_treatment_valid": primary_valid,
        "primary_comparison_status": primary_status,
        "replication_executed": replication_state is not None,
        "replication_treatment_valid": (
            None if replication_state is None else replication_state[0]
        ),
        "replication_comparison_status": (
            None if replication_state is None else replication_state[1]
        ),
        "two_seed_outer_width_supported": (
            outcome == "two_seed_outer_width_support"
        ),
        "expert_core_control_authorized": (
            outcome in {"primary_both_fail", "replication_both_fail"}
        ),
        "compressed_width_ladder_authorized": (
            outcome == "two_seed_outer_width_support"
        ),
        "fresh_c3_authorized": False,
        "compression_claim_authorized": False,
    }


def _interpretation_from_evidence(
    rows: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    """Materialize only observed claims or explicitly negative scope claims."""

    primary = rows.get("fp_width.primary.rank16")
    if not isinstance(primary, Mapping):
        raise ValueError("primary rank16 evidence is absent")
    validity = primary.get("pair_treatment_validity")
    if not isinstance(validity, Mapping):
        raise ValueError("primary treatment validity is absent")
    flags = validity.get("flags")
    if not isinstance(flags, Mapping):
        raise ValueError("primary treatment validity flags are absent")
    return {
        "fit_side_only": True,
        "initial_function_and_jvp_matched": (
            flags.get("initial_observable_and_jvp_equivalence") is True
        ),
        "gradient_open_width_control": (
            flags.get("gradient_open_rank64") is True
        ),
        "two_seed_support_implicates_outer_width": (
            decision.get("two_seed_outer_width_supported") is True
        ),
        "two_seed_support_proves_only_bottleneck": False,
        "both_fail_opens_expert_core_control": (
            decision.get("expert_core_control_authorized") is True
        ),
        "fresh_c3_authorized": False,
        "compression_claim_authorized": False,
        "natural_prompt_fidelity_claim": False,
        "whole_model_replacement_claim": False,
        "wall_clock_speed_claim": False,
        "provider_fit_numeric_dtype": "torch.float64",
        "post_run_scientific_authority_requires_external_trust_anchor_triple": (
            True
        ),
        "run_generated_receipt_verifies_publication_integrity_only": True,
        "run_generated_receipt_is_external_scientific_trust_root": False,
        "durable_scientific_authority_requires_recording_exact_receipt_triple_"
        "outside_artifact": True,
        "self_hashes_sufficient_for_scientific_authority": False,
    }


def _safety_contract() -> dict[str, bool]:
    return {
        "contains_source_model_state_dict": False,
        "contains_rank16_provider_parameters": True,
        "contains_rank64_provider_parameters": True,
        "contains_loaded_source_final_provider_parameters": False,
        "contains_regenerated_d3_equivalent_parameters": True,
        "contains_raw_teacher_targets": False,
        "contains_teacher_jvp_tensors": False,
        "contains_provider_chart_jvp_tensors": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_c2_selection_data": False,
        "external_trust_anchor_embedded_in_artifact": False,
        "scientific_outcome_requires_external_trust_anchor": True,
        "run_generated_receipt_verifies_publication_integrity_only": True,
        "run_generated_receipt_is_external_scientific_trust_root": False,
        "durable_scientific_authority_requires_recording_exact_receipt_triple_"
        "outside_artifact": True,
        "committable": False,
    }


def load_function_preserving_width_control_artifact(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_tensor_file_sha256: str,
    expected_report_sha256: str,
) -> LoadedFunctionPreservingWidthControlArtifact:
    """Strictly load, authenticate, and recompute a paired result."""

    source = Path(path).expanduser().resolve()
    if source.suffix != ".pt":
        raise ValueError("paired width artifact must use a .pt suffix")
    tensor_payload = _read_regular(source, label="paired width artifact")
    report_path = source.with_suffix(".json")
    report_payload = _read_regular(report_path, label="paired width report")
    tensor_sha256 = hashlib.sha256(tensor_payload).hexdigest()
    trusted_artifact_sha256 = _require_sha256(
        expected_artifact_sha256,
        label="expected logical artifact SHA-256",
    )
    trusted_tensor_sha256 = _require_sha256(
        expected_tensor_file_sha256,
        label="expected tensor-file SHA-256",
    )
    trusted_report_sha256 = _require_sha256(
        expected_report_sha256,
        label="expected report SHA-256",
    )
    if tensor_sha256 != trusted_tensor_sha256:
        raise ValueError(
            "paired width external trust anchor mismatch: tensor file"
        )
    raw = torch.load(
        io.BytesIO(tensor_payload),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw, Mapping) or set(raw) != _STATE_FIELDS:
        raise ValueError("paired width tensor fields drifted")
    d0d3._assert_safe_artifact_tree(raw)
    manifest = raw["manifest"]
    if not isinstance(manifest, Mapping):
        raise TypeError("paired width manifest must be a mapping")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("paired width manifest fields drifted")
    logical = _json_sha256(manifest, domain=_ARTIFACT_DOMAIN)
    if (
        raw["artifact_sha256"] != logical
        or manifest.get("schema") != _SCHEMA
        or manifest.get("format_version") != _FORMAT_VERSION
    ):
        raise ValueError("paired width logical binding mismatch")
    try:
        report = json.loads(report_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("paired width report is invalid JSON") from exc
    if not isinstance(report, Mapping):
        raise TypeError("paired width report must be a mapping")
    d0d3._assert_safe_artifact_tree(report, path="report")
    d0d3._assert_tensor_free_report(report)
    without_hash = dict(report)
    supplied_report_sha = without_hash.pop("report_sha256", None)
    computed_report_sha = _json_sha256(
        without_hash,
        domain=_REPORT_DOMAIN,
    )
    if (
        logical != trusted_artifact_sha256
        or tensor_sha256 != trusted_tensor_sha256
        or computed_report_sha != trusted_report_sha256
    ):
        raise ValueError("paired width external trust anchor mismatch")
    artifact_binding = report.get("artifact")
    if (
        supplied_report_sha != computed_report_sha
        or report.get("artifact_sha256") != logical
        or not isinstance(artifact_binding, Mapping)
        or artifact_binding.get("tensor_file") != str(source)
        or artifact_binding.get("report_file") != str(report_path)
        or artifact_binding.get("tensor_file_sha256") != tensor_sha256
        or artifact_binding.get("tensor_file_bytes") != len(tensor_payload)
        or artifact_binding.get("committable") is not False
    ):
        raise ValueError("paired width report binding mismatch")
    if set(report) != _MANIFEST_FIELDS | _REPORT_EXTRA_FIELDS:
        raise ValueError("paired width report fields drifted")
    for name, value in manifest.items():
        if _canonical_json_bytes(report.get(name)) != _canonical_json_bytes(
            value
        ):
            raise ValueError(f"paired width report field {name!r} drifted")

    code = manifest.get("code_sha256s")
    if (
        not isinstance(code, Mapping)
        or dict(code) != _code_sha256s()
        or manifest.get("code_bundle_sha256")
        != _code_bundle_sha256(dict(code))
    ):
        raise ValueError("paired width code binding drifted")
    protocol = FunctionPreservingWidthControlProtocol.from_state_dict(
        raw["protocol_state"]
    )
    if (
        protocol.protocol_sha256
        != DEFAULT_FUNCTION_PRESERVING_WIDTH_CONTROL_PROTOCOL_SHA256
        or manifest.get("protocol_sha256") != protocol.protocol_sha256
        or _canonical_json_bytes(report.get("protocol"))
        != _canonical_json_bytes(protocol.state_dict())
    ):
        raise ValueError("paired width protocol binding drifted")
    measurement_evidence_sha256 = _measurement_evidence_sha256(report)
    if (
        measurement_evidence_sha256
        != protocol.training.measurement_evidence_sha256
        or manifest.get("measurement_evidence_sha256")
        != measurement_evidence_sha256
    ):
        raise ValueError("paired width measurement evidence drifted")
    source_bindings = {
        "source_d3_logical_artifact_sha256": (
            protocol.sources.d3_logical_artifact_sha256
        ),
        "source_d3_tensor_file_sha256": (
            protocol.sources.d3_tensor_file_sha256
        ),
        "source_d3_report_sha256": protocol.sources.d3_report_sha256,
        "source_d3_primary_plan_sha256": (
            protocol.sources.d3_primary_plan_sha256
        ),
        "source_d3_primary_result_sha256": (
            protocol.sources.d3_primary_result_sha256
        ),
        "source_rank64_logical_artifact_sha256": (
            protocol.sources.rank64_logical_artifact_sha256
        ),
        "source_rank64_tensor_file_sha256": (
            protocol.sources.rank64_tensor_file_sha256
        ),
        "source_rank64_report_sha256": (
            protocol.sources.rank64_report_sha256
        ),
        "source_rank64_primary_plan_sha256": (
            protocol.sources.rank64_primary_plan_sha256
        ),
        "source_rank64_primary_result_sha256": (
            protocol.sources.rank64_primary_result_sha256
        ),
    }
    if any(
        manifest.get(name) != expected
        for name, expected in source_bindings.items()
    ):
        raise ValueError("paired width predecessor binding drifted")

    calibration = d0d3._restore_calibration_binding(
        raw["calibration_state"]
    )
    controls = d0d3._restore_full_width_controls(raw["controls_state"])
    gauge = UnitRmsFisherGauge.from_state_dict(
        raw["unit_rms_gauge_state"]
    )
    metric = raw["canonical_metric_weight"]
    if not isinstance(metric, Tensor):
        raise TypeError("paired width canonical metric is not a tensor")
    gauge.validate_source(metric)
    if (
        calibration.artifact_sha256
        != manifest.get("c2_calibration_sha256")
        or controls.artifact_sha256 != manifest.get("controls_sha256")
        or gauge.artifact_sha256
        != manifest.get("unit_rms_gauge_sha256")
        or d0d3._tensor_sha256(metric)
        != manifest.get("canonical_metric_weight_sha256")
    ):
        raise ValueError("paired width gauge/control binding drifted")
    c2_protocol = default_contrast_provider_development_protocol()
    fidelity_gates = SyntheticReferenceGates()
    ordinary_gates = _deferred_collision_gates(fidelity_gates)
    contrast_gates = ContrastAssessmentGates()
    manifest_firewall = {
        "c2_protocol_sha256": c2_protocol.protocol_sha256,
        "c2_pilot_panel_sha256": c2_protocol.panel_sha256("pilot"),
        "c2_fit_panel_sha256": c2_protocol.panel_sha256("fit"),
        "c2_calibrated_fit_panel_sha256": (
            c2_protocol.calibrated_panel_sha256("fit", calibration)
        ),
        "c2_calibration_sha256": _EXPECTED_CALIBRATION_SHA256,
        "selected_calibration_amplitude": 8.0,
        "basis_package_file_sha256": DEFAULT_BASIS_PACKAGE_FILE_SHA256,
        "basis_package_payload_sha256": (
            DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        ),
        "source_model_sha256": _EXPECTED_SOURCE_MODEL_SHA256,
        "requested_execution_device": protocol.execution_device,
        "requested_execution_dtype": protocol.execution_dtype,
        "actual_execution_device": protocol.execution_device,
        "actual_execution_dtype": protocol.execution_dtype,
        "pre_feedforward_norm_sha256": (
            _EXPECTED_PRE_FEEDFORWARD_NORM_SHA256
        ),
        "canonical_metric_weight_sha256": (
            _EXPECTED_CANONICAL_METRIC_WEIGHT_SHA256
        ),
        "fit_data_binding_sha256": (
            protocol.training.fit_data_binding_sha256
        ),
        "unit_rms_gauge_sha256": _EXPECTED_UNIT_RMS_GAUGE_SHA256,
        "standardized_gauge_sha256": (
            _EXPECTED_STANDARDIZED_GAUGE_SHA256
        ),
        "controls_sha256": _EXPECTED_CONTROLS_SHA256,
        "ordinary_gates_sha256": (
            protocol.training.ordinary_gates_sha256
        ),
        "contrast_gates_sha256": (
            protocol.training.contrast_gates_sha256
        ),
        "measurement_evidence_sha256": (
            protocol.training.measurement_evidence_sha256
        ),
        "selection_materialized": False,
        "selection_measured": False,
        "selection_scored": False,
        "c2_provider_artifact_loaded": False,
        "authenticated_d3_source_loaded": True,
        "authenticated_rank64_source_loaded": True,
        "source_final_parameters_used_for_initialization": False,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "scientific_scope": (
            "fit_only_paired_function_preserving_width_control"
        ),
    }
    if any(
        _canonical_json_bytes(manifest.get(name))
        != _canonical_json_bytes(expected)
        for name, expected in manifest_firewall.items()
    ):
        raise ValueError("paired width manifest firewall drifted")

    plan_states = raw["plan_states"]
    rows = raw["candidate_results"]
    executed = manifest.get("executed_candidate_ids")
    plan_hashes = manifest.get("candidate_plan_sha256s")
    result_hashes = manifest.get("candidate_result_sha256s")
    if (
        not isinstance(plan_states, Mapping)
        or not isinstance(rows, Mapping)
        or not isinstance(executed, (tuple, list))
        or not isinstance(plan_hashes, Mapping)
        or not isinstance(result_hashes, Mapping)
        or set(plan_states) != set(executed)
        or set(rows) != set(executed)
        or set(plan_hashes) != set(executed)
        or set(result_hashes) != set(executed)
    ):
        raise ValueError("paired width candidate tables drifted")
    allowed_orders = (
        (
            "fp_width.primary.rank16",
            "fp_width.primary.rank64",
        ),
        (
            "fp_width.primary.rank16",
            "fp_width.primary.rank64",
            "fp_width.replication.rank16",
            "fp_width.replication.rank64",
        ),
    )
    if tuple(executed) not in allowed_orders:
        raise ValueError("paired width candidate order drifted")

    canonical_state = plan_states.get("fp_width.primary.rank16")
    if not isinstance(canonical_state, Mapping):
        raise ValueError("paired width canonical rank16 plan is absent")
    canonical_plan = ContrastAwareReferenceProviderPlan.from_state_dict(
        canonical_state
    )
    canonical_plan.validate_integrity()
    if (
        canonical_plan.artifact_sha256
        != protocol.sources.d3_primary_plan_sha256
        or canonical_plan.synthetic_binding_sha256
        != protocol.training.fit_data_binding_sha256
        or not torch.equal(
            canonical_plan.fisher_metric_weight,
            gauge.metric_weight,
        )
    ):
        raise ValueError("paired width canonical D3 geometry drifted")

    objective_protocol = default_objective_balance_diagnostic_protocol()
    d3_recipe = objective_protocol.recipe(protocol.training.recipe_id)
    report_gauge = report.get("gauge")
    training_teacher_signal = report.get(
        "training_teacher_signal_diagnostics"
    )
    if not isinstance(report_gauge, Mapping) or not isinstance(
        training_teacher_signal,
        Mapping,
    ):
        raise TypeError("paired width report lacks balance evidence")
    if (
        report_gauge.get("target_center_sha256")
        != d0d3._tensor_sha256(canonical_plan.target_center)
        or report_gauge.get("target_scale_sha256")
        != d0d3._tensor_sha256(canonical_plan.target_scale)
    ):
        raise ValueError("paired width report target geometry drifted")
    for candidate_id in executed:
        state = plan_states[candidate_id]
        row = rows[candidate_id]
        if not isinstance(state, Mapping) or not isinstance(row, Mapping):
            raise TypeError("paired width candidate entry is invalid")
        if set(row) != _ROW_FIELDS:
            raise ValueError("paired width candidate fields drifted")
        plan = ContrastAwareReferenceProviderPlan.from_state_dict(state)
        plan.validate_integrity()
        arm = row.get("arm")
        role = row.get("pair_role")
        expected_config = (
            _executor_config(protocol.rank16_executor)
            if arm == "rank16"
            else _executor_config(protocol.rank64_executor)
        )
        if (
            plan.artifact_sha256 != plan_hashes[candidate_id]
            or row.get("plan_sha256") != plan.artifact_sha256
            or row.get("candidate_id") != candidate_id
            or role not in {"primary", "replication"}
            or arm not in {"rank16", "rank64"}
            or candidate_id != f"fp_width.{role}.{arm}"
            or asdict(plan.executor_config) != asdict(expected_config)
            or row.get("seed") != plan.seed
            or plan.training_steps != protocol.training.steps
            or plan.learning_rate != protocol.training.learning_rate
            or plan.seed
            != (
                protocol.training.primary_seed
                if role == "primary"
                else protocol.training.replication_seed
            )
            or plan.objective.artifact_sha256
            != _objective(protocol).artifact_sha256
            or _canonical_json_bytes(row.get("initial_training_metrics"))
            != _canonical_json_bytes(plan.initial_metrics.state_dict())
            or _canonical_json_bytes(row.get("final_training_metrics"))
            != _canonical_json_bytes(plan.final_metrics.state_dict())
            or result_hashes[candidate_id]
            != _json_sha256(row, domain=_RESULT_DOMAIN)
        ):
            raise ValueError("paired width candidate plan/result drifted")
        if (
            plan.synthetic_binding_sha256
            != protocol.training.fit_data_binding_sha256
            or plan.fisher_metric_supplied is not True
            or not torch.equal(
                plan.fisher_metric_weight,
                gauge.metric_weight,
            )
            or not torch.equal(
                plan.modal_center,
                canonical_plan.modal_center,
            )
            or plan.gain_log_center != canonical_plan.gain_log_center
            or plan.gain_log_scale != canonical_plan.gain_log_scale
            or plan.residual_width != canonical_plan.residual_width
            or plan.rms_epsilon != canonical_plan.rms_epsilon
            or not torch.equal(
                plan.target_center,
                canonical_plan.target_center,
            )
            or not torch.equal(
                plan.target_scale,
                canonical_plan.target_scale,
            )
        ):
            raise ValueError("paired width shared fit geometry drifted")
        source_replay = _source_replay_exact(
            pair_role=str(role),
            arm=str(arm),
            plan=plan,
            protocol=protocol,
        )
        if row.get("source_replay_exact") is not source_replay:
            raise ValueError("paired width source replay decision drifted")
        pair_validity = row["pair_treatment_validity"]
        if (
            not isinstance(pair_validity, Mapping)
            or set(pair_validity) != _PAIR_TREATMENT_VALIDITY_FIELDS
            or pair_validity.get("failure_semantics")
            != "invalid_paired_width_comparison_no_capacity_conclusion"
        ):
            raise ValueError("paired width treatment validity schema drifted")
        structural = row["structural_metadata"]
        if (
            not isinstance(structural, Mapping)
            or set(structural) != _STRUCTURAL_METADATA_FIELDS
            or structural.get("support_rule")
            != (
                "fit_max_l2_radius_of_encoded_nonconstant_features_plus_"
                "margin"
            )
            or not isinstance(structural.get("support_radius"), float)
            or not math.isfinite(float(structural["support_radius"]))
            or float(structural["support_radius"]) < 0.0
            or not isinstance(
                structural.get("invalid_padding_rows_tested"),
                int,
            )
            or isinstance(
                structural.get("invalid_padding_rows_tested"),
                bool,
            )
            or int(structural["invalid_padding_rows_tested"]) <= 0
            or structural.get("nonvacuous_padding_test") is not True
        ):
            raise ValueError("paired width structural metadata drifted")
        for audit_name in (
            "initialization_equivalence",
            "gradient_openness",
        ):
            audit = row.get(audit_name)
            if not isinstance(audit, Mapping):
                raise TypeError(f"paired width {audit_name} is invalid")
            audit_payload = dict(audit)
            supplied = audit_payload.pop("artifact_sha256", None)
            if supplied != _json_sha256(
                audit_payload,
                domain=_INITIALIZATION_DOMAIN,
            ):
                raise ValueError(f"paired width {audit_name} hash drifted")
            flags = audit.get("flags")
            if (
                not isinstance(flags, Mapping)
                or audit.get("passed") is not all(
                    value is True for value in flags.values()
                )
            ):
                raise ValueError(
                    f"paired width {audit_name} decision drifted"
                )
        initialization = row["initialization_equivalence"]
        assert isinstance(initialization, Mapping)
        initialization_error_fields = (
            "maximum_observable_absolute_error",
            "maximum_observable_relative_error",
            "maximum_jvp_absolute_error",
            "maximum_jvp_relative_error",
        )
        initialization_metric_fields = (
            "rank16_initial_metrics_sha256",
            "rank64_initial_metrics_sha256",
        )
        initialization_flags = initialization.get("flags")
        if (
            set(initialization) != _INITIALIZATION_EQUIVALENCE_FIELDS
            or initialization.get("artifact_kind")
            != "fisher_graph.initial_width_equivalence"
            or type(initialization.get("format_version")) is not int
            or initialization.get("format_version") != _FORMAT_VERSION
            or any(
                type(initialization.get(name)) is not float
                or not math.isfinite(initialization[name])
                or initialization[name] < 0.0
                for name in initialization_error_fields
            )
            or type(initialization.get("jvp_pair_count")) is not int
            or initialization.get("jvp_pair_count") != 32
            or any(
                not isinstance(initialization.get(name), str)
                or len(initialization[name]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in initialization[name]
                )
                for name in initialization_metric_fields
            )
            or not isinstance(initialization_flags, Mapping)
            or any(
                type(value) is not bool
                for value in initialization_flags.values()
            )
            or type(initialization.get("passed")) is not bool
        ):
            raise ValueError(
                "paired width initialization equivalence schema or "
                "semantics drifted"
            )
        expected_initial_flags = {
            "observable_absolute": (
                float(initialization["maximum_observable_absolute_error"])
                <= protocol.lift.equivalence_absolute_tolerance
            ),
            "observable_relative": (
                float(initialization["maximum_observable_relative_error"])
                <= protocol.lift.equivalence_relative_tolerance
            ),
            "jvp_absolute": (
                float(initialization["maximum_jvp_absolute_error"])
                <= protocol.lift.equivalence_absolute_tolerance
            ),
            "jvp_relative": (
                float(initialization["maximum_jvp_relative_error"])
                <= protocol.lift.equivalence_relative_tolerance
            ),
            "initial_metrics_exact": (
                initialization["rank16_initial_metrics_sha256"]
                == initialization["rank64_initial_metrics_sha256"]
            ),
            "all_expected_jvps_compared": (
                initialization["jvp_pair_count"] == 32
            ),
        }
        if dict(initialization["flags"]) != expected_initial_flags:
            raise ValueError("paired width initialization flags drifted")
        expected_initial_parameters = (
            _reconstruct_initial_parameter_bindings(
                protocol,
                pair_role=str(role),
            )
        )
        paired_initial_metrics: dict[str, str] = {}
        for paired_arm in ("rank16", "rank64"):
            paired_state = plan_states.get(
                f"fp_width.{role}.{paired_arm}"
            )
            if not isinstance(paired_state, Mapping):
                raise ValueError("paired width initial metric arm is absent")
            paired_plan = (
                ContrastAwareReferenceProviderPlan.from_state_dict(
                    paired_state
                )
            )
            paired_initial_metrics[paired_arm] = (
                paired_plan.initial_metrics.artifact_sha256
            )
        if _canonical_json_bytes(
            initialization.get("initial_parameter_bindings")
        ) != _canonical_json_bytes(
            expected_initial_parameters
        ) or (
            initialization.get("rank16_initial_metrics_sha256")
            != paired_initial_metrics["rank16"]
            or initialization.get("rank64_initial_metrics_sha256")
            != paired_initial_metrics["rank64"]
        ):
            raise ValueError(
                "paired width initial parameter binding drifted"
            )
        gradient = row["gradient_openness"]
        assert isinstance(gradient, Mapping)
        expected_gradient = _gradient_audit_contract(
            protocol,
            pair_role=str(role),
            arm=str(arm),
        )
        if _canonical_json_bytes(gradient) != _canonical_json_bytes(
            expected_gradient
        ):
            raise ValueError("paired width gradient-open flags drifted")
        recomputed_balance = d0d3._contribution_balance_gate(
            plan,
            recipe=d3_recipe,
            gates=objective_protocol.gates,
            training_teacher_energy=float(
                report_gauge["unit_fit_teacher_weighted_energy"]
            ),
            raw_teacher_energy=float(
                report_gauge["raw_fit_teacher_weighted_energy"]
            ),
            teacher_signal_diagnostics=training_teacher_signal,
        )
        if _canonical_json_bytes(recomputed_balance) != (
            _canonical_json_bytes(row.get("objective_balance_gate"))
        ):
            raise ValueError("paired width contribution balance drifted")
        recomputed_final_audit = audit_objective_contributions(
            plan.final_metrics,
            plan.objective,
        )
        if _canonical_json_bytes(recomputed_final_audit.state_dict()) != (
            _canonical_json_bytes(row.get("final_contribution_audit"))
        ):
            raise ValueError("paired width final contribution audit drifted")
        sequence = row["source_sequence_comparison"]
        if not isinstance(sequence, Mapping):
            raise TypeError("paired width source sequence audit is invalid")
        if (
            set(sequence) != _SOURCE_SEQUENCE_COMPARISON_FIELDS
            or sequence.get("comparison_semantics")
            != (
                "exact_ordered_batch_content_index_endpoint_and_pair_hashes"
            )
        ):
            raise ValueError("paired width source sequence schema drifted")
        candidate_sequences = sequence.get("candidate_sequence_sha256s")
        source_sequences = sequence.get("source_sequence_sha256s")
        sequence_flags = sequence.get("flags")
        if (
            not isinstance(candidate_sequences, Mapping)
            or not isinstance(source_sequences, Mapping)
            or not isinstance(sequence_flags, Mapping)
            or set(candidate_sequences) != set(r64._SOURCE_PLAN_SEQUENCE_FIELDS)
            or set(source_sequences) != set(r64._SOURCE_PLAN_SEQUENCE_FIELDS)
        ):
            raise ValueError("paired width source sequence table drifted")
        expected_sequence_flags = {
            name: tuple(getattr(plan, name))
            == tuple(source_sequences[name])
            for name in r64._SOURCE_PLAN_SEQUENCE_FIELDS
        }
        predecessor_sequence_binding = (
            default_rank64_capacity_control_protocol().source_result
        )
        frozen_source_sequences = all(
            r64.source_sequence_binding_sha256(
                sequence_name,
                tuple(source_sequences[sequence_name]),
            )
            == getattr(predecessor_sequence_binding, field_name)
            for sequence_name, field_name in (
                r64._SOURCE_SEQUENCE_ANCHOR_FIELDS.items()
            )
        )
        if (
            dict(sequence_flags) != expected_sequence_flags
            or sequence.get("passed") is not all(
                expected_sequence_flags.values()
            )
            or not frozen_source_sequences
            or any(
                tuple(candidate_sequences[name])
                != tuple(getattr(plan, name))
                for name in r64._SOURCE_PLAN_SEQUENCE_FIELDS
            )
        ):
            raise ValueError("paired width source sequence decision drifted")
        score = FullWidthCandidateScore.from_state_dict(
            row["ordinary_score"]
        )
        accounting = plan.accounting()
        if (
            score.candidate_id != candidate_id
            or score.candidate_artifact_sha256
            != row.get("candidate_binding_sha256")
            or score.source_rank != 64
            or score.target_rank != 64
            or score.stored_scalar_count
            != accounting.total_stored_scalar_count
            or score.controls_artifact_sha256
            != controls.artifact_sha256
            or score.controls_artifact_sha256
            != manifest.get("controls_sha256")
            or score.gates_sha256
            != protocol.training.ordinary_gates_sha256
            or score.gates_sha256
            != manifest.get("ordinary_gates_sha256")
            or row.get("latent_rank") != plan.latent_rank
            or row.get("latent_rank")
            != (16 if arm == "rank16" else 64)
            or row.get("expert_rank") != expected_config.expert_rank
            or _canonical_json_bytes(row.get("accounting"))
            != _canonical_json_bytes(asdict(accounting))
        ):
            raise ValueError("paired width ordinary/row binding drifted")
        recomputed_flags = r64._recompute_ordinary_gate_flags(
            score,
            ordinary_gates,
        )
        supplied_flags = score.gate_flags.state_dict()
        if any(
            supplied_flags.get(name) != value
            for name, value in recomputed_flags.items()
        ) or score.passed is not all(recomputed_flags.values()):
            raise ValueError("paired width ordinary gates drifted")
        _, contrast_scores = r64._validate_contrast_result_state(
            row["contrast_result"],
            gates=contrast_gates,
        )
        coverage = r64._recompute_contrast_coverage(
            contrast_scores,
            row["contrast_identities"],
            required_null_candidate_pass_count=24,
        )
        if _canonical_json_bytes(coverage) != _canonical_json_bytes(
            row["contrast_coverage"]
        ):
            raise ValueError("paired width contrast coverage drifted")
        if _canonical_json_bytes(row["contrast_identities"]) != (
            _canonical_json_bytes(
                r64._expected_contrast_identities(
                    default_contrast_provider_development_protocol()
                )
            )
        ):
            raise ValueError("paired width contrast identities drifted")
        ordinary_pass = all(recomputed_flags.values())
        contrast_state = row["contrast_result"]
        assert isinstance(contrast_state, Mapping)
        fit_pass = (
            ordinary_pass
            and contrast_state.get("overall_status") == "pass"
            and bool(coverage["every_teacher_qualified_contrast_passed"])
            and bool(coverage["all_families_cover_all_four_rank_bands"])
            and bool(coverage["required_null_contrasts_valid_and_passed"])
        )
        expected_fit_contract = {
            "ordinary_gate_count": len(recomputed_flags),
            "all_ordinary_gates_passed": ordinary_pass,
            "all_contrast_families_passed": (
                contrast_state.get("overall_status") == "pass"
            ),
            "every_qualified_contrast_passed": bool(
                coverage["every_teacher_qualified_contrast_passed"]
            ),
            "all_four_rank_bands_covered": bool(
                coverage["all_families_cover_all_four_rank_bands"]
            ),
            "required_null_contrasts_passed": bool(
                coverage["required_null_contrasts_valid_and_passed"]
            ),
        }
        if (
            row.get("fit_capability_pass") is not fit_pass
            or _canonical_json_bytes(row.get("fit_capability_contract"))
            != _canonical_json_bytes(expected_fit_contract)
        ):
            raise ValueError("paired width fit decision drifted")

    recomputed = _recompute_decision_from_rows(rows)
    for name, expected in recomputed.items():
        if manifest.get(name) != expected:
            raise ValueError("paired width final decision drifted")
    expected_interpretation = _interpretation_from_evidence(
        rows,
        recomputed,
    )
    if (
        _canonical_json_bytes(report.get("candidate_results"))
        != _canonical_json_bytes(
            [rows[candidate_id] for candidate_id in executed]
        )
        or _canonical_json_bytes(report.get("interpretation"))
        != _canonical_json_bytes(expected_interpretation)
        or _canonical_json_bytes(report.get("safety"))
        != _canonical_json_bytes(_safety_contract())
    ):
        raise ValueError("paired width report safety semantics drifted")
    return LoadedFunctionPreservingWidthControlArtifact(
        state=raw,
        report=report,
        manifest=manifest,
        artifact_sha256=logical,
        tensor_file_sha256=tensor_sha256,
        report_sha256=computed_report_sha,
    )


def run_function_preserving_width_control(
    *,
    source_diagnostic_path: Path | str = DEFAULT_SOURCE_DIAGNOSTIC,
    source_rank64_path: Path | str = DEFAULT_SOURCE_RANK64,
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
    """Execute the authenticated paired fit-only width control."""

    protocol, objective_protocol, c2_protocol, d3_recipe = (
        _authenticated_declarations()
    )
    if (
        device_name != protocol.execution_device
        or dtype != protocol.execution_dtype
        or str(basis_package_file_sha256)
        != DEFAULT_BASIS_PACKAGE_FILE_SHA256
        or str(basis_package_payload_sha256)
        != DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
    ):
        raise ValueError(
            "paired width execution and basis bindings are frozen"
        )
    destination = _validate_output_path(output)
    sources = _authenticate_sources(
        source_diagnostic_path=source_diagnostic_path,
        source_rank64_path=source_rank64_path,
        protocol=protocol,
    )
    code = _code_sha256s()
    code_bundle = _code_bundle_sha256(code)
    fidelity_gates = SyntheticReferenceGates()
    contrast_gates = ContrastAssessmentGates()
    ordinary_gate_sha = full_width_reference_gates_sha256(
        _deferred_collision_gates(fidelity_gates)
    )
    if (
        ordinary_gate_sha != protocol.training.ordinary_gates_sha256
        or contrast_gates.artifact_sha256
        != protocol.training.contrast_gates_sha256
    ):
        raise ValueError("paired width scoring gates drifted")

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
    actual_device, actual_dtype = d0d3._actual_model_execution(adapter)
    if (
        actual_device != protocol.execution_device
        or actual_dtype != protocol.execution_dtype
    ):
        raise ValueError("live model execution differs from paired protocol")
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
        calibration.selected_amplitude != 8.0
        or calibration.artifact_sha256 != _EXPECTED_CALIBRATION_SHA256
        or calibrated_fit_panel_sha256
        != objective_protocol.c2_provenance.calibrated_fit_panel_sha256
    ):
        raise ValueError("paired width C2 calibration replay drifted")
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
    if any(
        value.probe.probe_id.startswith(
            objective_protocol.c2_provenance.forbidden_probe_prefixes
        )
        for value in fit
    ):
        raise RuntimeError("C2 selection identity entered paired fit data")
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
    unit_gauge = UnitRmsFisherGauge.from_metric_weight(raw_metric_weight)
    raw_teacher_energy = d0d3._fit_teacher_weighted_energy(
        fit,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=raw_metric_weight,
    )
    unit_teacher_energy = d0d3._fit_teacher_weighted_energy(
        fit,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=unit_gauge.metric_weight,
    )
    if (
        raw_teacher_energy <= objective_protocol.gates.minimum_gauge_energy
        or abs(unit_teacher_energy - 1.0)
        > objective_protocol.gates.normalized_energy_absolute_tolerance
    ):
        raise ValueError("paired width Fisher gauge drifted")
    natural_pairs, chart_mismatch = c2._training_contrast_pairs(
        protocol=c2_protocol,
        measured=fit,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
    )
    fit_pairs, pair_balance = d0d3._balance_training_pairs(
        natural_pairs,
        recipe=d3_recipe,
    )
    fit_batches = c2._indexed_batches(
        fit,
        split="fit",
        binding_sha256=protocol.training.fit_data_binding_sha256,
    )
    teacher_signal = d0d3._teacher_signal_diagnostics(
        fit,
        natural_pairs,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=raw_metric_weight,
    )
    training_teacher_signal = d0d3._teacher_signal_diagnostics(
        fit,
        natural_pairs,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=unit_gauge.metric_weight,
    )
    fit_data_binding = d0d3._fit_data_binding_sha256(
        basis=basis,
        c2_protocol=c2_protocol,
        calibration=calibration,
        norm_sha256=norm_sha256,
        canonical_metric_weight=raw_metric_weight,
    )
    if (
        fit_data_binding != protocol.training.fit_data_binding_sha256
        or fit_data_binding != sources.source_d3.fit_data_binding_sha256
        or tuple(value.artifact_sha256 for value in fit_pairs)
        != sources.source_d3.balanced_pair_sha256s
    ):
        raise ValueError("paired width fit problem differs from source D3")
    standardized_gauge_sha = d0d3._standardized_gauge_sha256(
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
        standardized_gauge_sha256=standardized_gauge_sha,
    )
    controls = fit_full_width_reference_controls(
        fit_probes=ordinary_probes,
        position_bin_count=16,
    )

    pair_kwargs = {
        "protocol": protocol,
        "objective_protocol": objective_protocol,
        "d3_recipe": d3_recipe,
        "source": sources,
        "fit": fit,
        "fit_batches": fit_batches,
        "fit_pairs": fit_pairs,
        "modal_center": modal_center,
        "gain_log_center": gain_log_center,
        "gain_log_scale": gain_log_scale,
        "residual_width": basis.residual_width,
        "rms_epsilon": epsilon,
        "target_center": target_center,
        "target_scale": target_scale,
        "unit_metric_weight": unit_gauge.metric_weight,
        "raw_metric_weight": raw_metric_weight,
        "raw_teacher_energy": raw_teacher_energy,
        "training_teacher_energy": unit_teacher_energy,
        "teacher_signal_diagnostics": training_teacher_signal,
        "fit_data_binding_sha256": fit_data_binding,
        "ordinary_probes": ordinary_probes,
        "controls": controls,
        "standardized_gauge_sha256": standardized_gauge_sha,
        "fidelity_gates": fidelity_gates,
        "contrast_gates": contrast_gates,
    }
    primary = _evaluate_pair(
        pair_role="primary",
        seed=protocol.training.primary_seed,
        **pair_kwargs,
    )
    replication = None
    if (
        primary.treatment_valid
        and primary.comparison_status == "rank16_fail_rank64_pass"
    ):
        replication = _evaluate_pair(
            pair_role="replication",
            seed=protocol.training.replication_seed,
            **pair_kwargs,
        )
    pairs = tuple(
        value for value in (primary, replication) if value is not None
    )
    arms = tuple(
        arm
        for pair in pairs
        for arm in (pair.rank16, pair.rank64)
    )
    decision = _decision(primary, replication)
    if (
        adapter.model_fingerprint() != model_before
        or module_state_fingerprint(pre_ff3) != norm_sha256
        or _code_sha256s() != code
    ):
        raise RuntimeError("model, normalization, or code changed during run")

    rows = [dict(arm.row) for arm in arms]
    row_map = {
        str(row["candidate_id"]): row
        for row in rows
    }
    plan_states = {
        arm.candidate_id: arm.plan.state_dict() for arm in arms
    }
    result_hashes = {
        str(row["candidate_id"]): _json_sha256(
            row,
            domain=_RESULT_DOMAIN,
        )
        for row in rows
    }
    gauge_state = unit_gauge.state_dict()
    del gauge_state["metric_weight"]
    measurement_evidence = {
        "pilot_metrics": tuple(
            value.state_dict() for value in pilot_metrics
        ),
        "pilot_measurement": pilot_measurement,
        "fit_measurement": fit_measurement,
        "fit_provider_chart_mismatch_diagnostics": chart_mismatch,
        "teacher_signal_diagnostics": teacher_signal,
        "training_teacher_signal_diagnostics": training_teacher_signal,
        "pair_balance": pair_balance,
        "gauge": {
            **gauge_state,
            "raw_fit_teacher_weighted_energy": raw_teacher_energy,
            "unit_fit_teacher_weighted_energy": unit_teacher_energy,
            "target_center_sha256": d0d3._tensor_sha256(target_center),
            "target_scale_sha256": d0d3._tensor_sha256(target_scale),
        },
    }
    measurement_evidence_sha256 = _measurement_evidence_sha256(
        measurement_evidence
    )
    if (
        measurement_evidence_sha256
        != protocol.training.measurement_evidence_sha256
    ):
        raise ValueError("paired width measurement evidence drifted")
    manifest = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "source_d3_logical_artifact_sha256": (
            protocol.sources.d3_logical_artifact_sha256
        ),
        "source_d3_tensor_file_sha256": (
            protocol.sources.d3_tensor_file_sha256
        ),
        "source_d3_report_sha256": protocol.sources.d3_report_sha256,
        "source_d3_primary_plan_sha256": (
            protocol.sources.d3_primary_plan_sha256
        ),
        "source_d3_primary_result_sha256": (
            protocol.sources.d3_primary_result_sha256
        ),
        "source_rank64_logical_artifact_sha256": (
            protocol.sources.rank64_logical_artifact_sha256
        ),
        "source_rank64_tensor_file_sha256": (
            protocol.sources.rank64_tensor_file_sha256
        ),
        "source_rank64_report_sha256": (
            protocol.sources.rank64_report_sha256
        ),
        "source_rank64_primary_plan_sha256": (
            protocol.sources.rank64_primary_plan_sha256
        ),
        "source_rank64_primary_result_sha256": (
            protocol.sources.rank64_primary_result_sha256
        ),
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
        "actual_execution_device": actual_device,
        "actual_execution_dtype": actual_dtype,
        "pre_feedforward_norm_sha256": norm_sha256,
        "canonical_metric_weight_sha256": d0d3._tensor_sha256(
            raw_metric_weight
        ),
        "fit_data_binding_sha256": fit_data_binding,
        "unit_rms_gauge_sha256": unit_gauge.artifact_sha256,
        "standardized_gauge_sha256": standardized_gauge_sha,
        "controls_sha256": controls.artifact_sha256,
        "ordinary_gates_sha256": ordinary_gate_sha,
        "contrast_gates_sha256": contrast_gates.artifact_sha256,
        "measurement_evidence_sha256": measurement_evidence_sha256,
        "executed_candidate_ids": tuple(
            str(row["candidate_id"]) for row in rows
        ),
        "candidate_plan_sha256s": {
            arm.candidate_id: arm.plan.artifact_sha256 for arm in arms
        },
        "candidate_result_sha256s": result_hashes,
        **decision,
        "selection_materialized": False,
        "selection_measured": False,
        "selection_scored": False,
        "c2_provider_artifact_loaded": False,
        "authenticated_d3_source_loaded": True,
        "authenticated_rank64_source_loaded": True,
        "source_final_parameters_used_for_initialization": False,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "code_sha256s": code,
        "code_bundle_sha256": code_bundle,
        "scientific_scope": (
            "fit_only_paired_function_preserving_width_control"
        ),
    }
    logical = _json_sha256(manifest, domain=_ARTIFACT_DOMAIN)
    state = {
        "manifest": manifest,
        "artifact_sha256": logical,
        "protocol_state": protocol.state_dict(),
        "calibration_state": calibration.state_dict(),
        "unit_rms_gauge_state": unit_gauge.state_dict(),
        "canonical_metric_weight": raw_metric_weight,
        "controls_state": controls.state_dict(),
        "plan_states": plan_states,
        "candidate_results": row_map,
    }
    report_payload = {
        **manifest,
        "artifact_sha256": logical,
        "protocol": protocol.state_dict(),
        "calibration": calibration.state_dict(),
        **measurement_evidence,
        "candidate_results": rows,
        "interpretation": _interpretation_from_evidence(
            row_map,
            decision,
        ),
        "safety": _safety_contract(),
    }
    publication_receipt = _publish_artifact(
        state,
        report_payload,
        output=destination,
    )
    try:
        authenticated = load_function_preserving_width_control_artifact(
            destination,
            **publication_receipt,
        )
    except BaseException:
        destination.unlink(missing_ok=True)
        destination.with_suffix(".json").unlink(missing_ok=True)
        raise
    return dict(authenticated.report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "describe",
        help="print the sealed declaration without loading artifacts",
    )
    run = commands.add_parser(
        "run",
        help="execute the authenticated paired control",
    )
    run.add_argument(
        "--source-diagnostic",
        type=Path,
        default=DEFAULT_SOURCE_DIAGNOSTIC,
    )
    run.add_argument(
        "--source-rank64",
        type=Path,
        default=DEFAULT_SOURCE_RANK64,
    )
    run.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--device", default="cpu", choices=("cpu",))
    run.add_argument("--dtype", default="float32", choices=("float32",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        report = describe_function_preserving_width_control()
    else:
        report = run_function_preserving_width_control(
            source_diagnostic_path=args.source_diagnostic,
            source_rank64_path=args.source_rank64,
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
