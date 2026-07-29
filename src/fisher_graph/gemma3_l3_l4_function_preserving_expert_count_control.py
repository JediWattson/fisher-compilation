"""Run the fixed-rank function-preserving expert-count control.

The paired fit-side capacity oracle holds the outer latent width, expert rank,
router feature width, objective, data, optimizer, and fit budget fixed.  It
compares the authenticated two-expert/rank-64 replay with a four-expert/rank-64
treatment that starts at the same observable function and provider-chart JVP.

This module intentionally does not authorize selection, C3, compression,
generalization, or speed claims.
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
import struct
import tempfile

import torch
from torch import Tensor, nn

from .adapters import module_state_fingerprint
from .contrast_objective_balancing import (
    UnitRmsFisherGauge,
    audit_objective_contributions,
)
from .external_models import find_git_worktree
from .gated_executor import (
    GatedCausalModalExecutorConfig,
    ResidualGatedCausalModalExecutor,
)
from .gemma3_experiment import DEFAULT_MODEL_ID
from .gemma3_l3_l4_basis_package import DEFAULT_BASIS_PACKAGE
from . import gemma3_l3_l4_contrast_provider_development as c2
from .gemma3_l3_l4_contrast_provider_development_protocol import (
    default_contrast_provider_development_protocol,
    select_global_calibration_amplitude,
)
from . import gemma3_l3_l4_function_preserving_width_control as width
from .gemma3_l3_l4_function_preserving_width_control_protocol import (
    default_function_preserving_width_control_protocol,
)
from . import (
    gemma3_l3_l4_function_preserving_expert_rank_control as expert_rank
)
from .gemma3_l3_l4_function_preserving_expert_rank_control_protocol import (
    default_function_preserving_expert_rank_control_protocol,
)
from . import (
    gemma3_l3_l4_function_preserving_expert_count_control_protocol
    as expert_protocol_module,
)
from .gemma3_l3_l4_function_preserving_expert_count_control_protocol import (
    DEFAULT_FUNCTION_PRESERVING_EXPERT_COUNT_CONTROL_PROTOCOL_SHA256,
    ExpertCountExecutorSpec,
    FunctionPreservingExpertCountControlProtocol,
    default_function_preserving_expert_count_control_protocol,
)
from . import gemma3_l3_l4_objective_balance_diagnostic as d0d3
from .gemma3_l3_l4_objective_balance_diagnostic_protocol import (
    default_objective_balance_diagnostic_protocol,
)
from . import gemma3_l3_l4_rank64_capacity_control as r64
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
    IndexedReferenceBatch,
    ReferenceProviderContrastPair,
    evaluate_contrast_aware_reference_provider,
)
from . import state_conditioned_reference_selection as reference_selection
from .state_conditioned_reference_selection import (
    FullWidthCandidatePrediction,
    FullWidthCandidateScore,
    FullWidthReferenceCandidate,
    FullWidthReferenceControls,
    FullWidthReferenceProbe,
    FullWidthStructuralMetrics,
    fit_full_width_reference_controls,
    full_width_reference_gates_sha256,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE_EXPERT_RANK",
    "LoadedFunctionPreservingExpertCountControlArtifact",
    "build_parser",
    "describe_function_preserving_expert_count_control",
    "load_function_preserving_expert_count_control_artifact",
    "main",
    "run_function_preserving_expert_count_preflight",
    "run_function_preserving_expert_count_control",
]


DEFAULT_SOURCE_EXPERT_RANK = expert_rank.DEFAULT_OUTPUT
DEFAULT_SOURCE_DIAGNOSTIC = width.DEFAULT_SOURCE_DIAGNOSTIC
DEFAULT_SOURCE_RANK64 = width.DEFAULT_SOURCE_RANK64
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-function-preserving-expert-count-control.pt"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_function_preserving_expert_count_control"
)
_FORMAT_VERSION = 1
_SOURCE_CANDIDATE_ID = "fp_expert_rank.primary.expert64"
_EXPECTED_CALIBRATION_SHA256 = width._EXPECTED_CALIBRATION_SHA256
_EXPECTED_SOURCE_MODEL_SHA256 = width._EXPECTED_SOURCE_MODEL_SHA256
_EXPECTED_PRE_FEEDFORWARD_NORM_SHA256 = (
    width._EXPECTED_PRE_FEEDFORWARD_NORM_SHA256
)
_EXPECTED_CANONICAL_METRIC_WEIGHT_SHA256 = (
    width._EXPECTED_CANONICAL_METRIC_WEIGHT_SHA256
)
_EXPECTED_UNIT_RMS_GAUGE_SHA256 = width._EXPECTED_UNIT_RMS_GAUGE_SHA256
_EXPECTED_STANDARDIZED_GAUGE_SHA256 = (
    width._EXPECTED_STANDARDIZED_GAUGE_SHA256
)
_EXPECTED_CONTROLS_SHA256 = width._EXPECTED_CONTROLS_SHA256

_ARTIFACT_DOMAIN = b"fisher-graph:function-preserving-expert-count:artifact:v1\0"
_REPORT_DOMAIN = b"fisher-graph:function-preserving-expert-count:report:v1\0"
_CODE_DOMAIN = b"fisher-graph:function-preserving-expert-count:code:v1\0"
_RESULT_DOMAIN = b"fisher-graph:function-preserving-expert-count:result:v1\0"
_AUDIT_DOMAIN = b"fisher-graph:function-preserving-expert-count:audit:v1\0"
_MEASUREMENT_DOMAIN = (
    b"fisher-graph:function-preserving-expert-count:measurement:v1\0"
)
_MEASUREMENT_FIELDS = width._MEASUREMENT_EVIDENCE_FIELDS


@dataclass(frozen=True, slots=True)
class _AuthenticatedSources:
    expert_rank_result: (
        expert_rank.LoadedFunctionPreservingExpertRankControlArtifact
    )
    expert_rank_plan: ContrastAwareReferenceProviderPlan
    source_d3: r64._SourceD3Bindings


@dataclass(frozen=True, slots=True)
class _ArmEvaluation:
    candidate_id: str
    pair_role: str
    arm: str
    fit_capability_pass: bool
    row: dict[str, object]
    plan: ContrastAwareReferenceProviderPlan
    ordinary_candidate: FullWidthReferenceCandidate


@dataclass(frozen=True, slots=True)
class _PairEvaluation:
    pair_role: str
    seed: int
    control: _ArmEvaluation
    treatment: _ArmEvaluation
    treatment_valid: bool
    validity_flags: Mapping[str, bool]
    comparison_status: str


@dataclass(frozen=True, slots=True)
class _RestoredFitEndpoint:
    """Minimal prompt-free row used to replay fit-side evidence."""

    probe: object
    modal_coordinates: Tensor
    null_coordinates: Tensor
    row_rms: Tensor
    target_modes: Tensor
    target_replays: tuple[Tensor, Tensor]
    logical_positions: Tensor
    valid_mask: Tensor


@dataclass(frozen=True, slots=True)
class LoadedFunctionPreservingExpertCountControlArtifact:
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
    return width._require_sha256(value, label=label)


def _file_sha256(path: Path | str) -> str:
    return width._file_sha256(path)


def _measurement_evidence_sha256(
    evidence: Mapping[str, object],
) -> str:
    return width._measurement_evidence_sha256(evidence)


def _code_sha256s() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    values = dict(width._code_sha256s())
    for name in (
        "gemma3_l3_l4_function_preserving_expert_rank_control.py",
        "gemma3_l3_l4_function_preserving_expert_rank_control_protocol.py",
        "gemma3_l3_l4_function_preserving_expert_count_control.py",
        "gemma3_l3_l4_function_preserving_expert_count_control_protocol.py",
    ):
        values[name] = _file_sha256(root / name)
    return dict(sorted(values.items()))


def _code_bundle_sha256(values: Mapping[str, str]) -> str:
    return _json_sha256(dict(values), domain=_CODE_DOMAIN)


def _executor_config(
    spec: ExpertCountExecutorSpec,
) -> GatedCausalModalExecutorConfig:
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


def _objective(
    protocol: FunctionPreservingExpertCountControlProtocol,
) -> ContrastAwareObjective:
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


class _ResidualExpertCountLift(nn.Module):
    """Four-expert split-rank bank equivalent to a standard E4/R64 executor."""

    def __init__(
        self,
        source: expert_rank._ResidualExpertRankLift,
        *,
        target_config: GatedCausalModalExecutorConfig,
    ) -> None:
        super().__init__()
        source_base = source.base
        if (
            source_base.config.input_modes != 66
            or source_base.config.output_modes != 64
            or source_base.config.expert_count != 2
            or source_base.config.expert_rank != 16
            or source.config.expert_count != 2
            or source.config.expert_rank != 64
            or target_config.input_modes != 66
            or target_config.output_modes != 64
            or target_config.expert_count != 4
            or target_config.expert_rank != 64
            or (
                asdict(source.config) | {"expert_count": 4}
                != asdict(target_config)
            )
        ):
            raise ValueError("expert-count split-bank geometry drifted")
        base_config = GatedCausalModalExecutorConfig(
            **(
                asdict(source_base.config)
                | {"expert_count": target_config.expert_count}
            )
        )
        self.base = ResidualGatedCausalModalExecutor(
            base_config,
            dtype=source_base.dtype,
            device=source_base.device,
        )
        self.config = target_config
        self.extra_input_weight = nn.Parameter(
            torch.zeros(
                4,
                66,
                48,
                dtype=source_base.dtype,
                device=source_base.device,
            )
        )
        self.extra_output_weight = nn.Parameter(
            torch.zeros(
                4,
                48,
                64,
                dtype=source_base.dtype,
                device=source_base.device,
            )
        )
        with torch.no_grad():
            source_state = source_base.state_dict()
            target_state = self.base.state_dict()
            for name, target in target_state.items():
                value = source_state[name]
                if name == "expert_input_weight":
                    target[:2].copy_(value)
                    target[2:].copy_(value)
                elif name == "expert_output_weight":
                    target[:2].copy_(2.0 * value)
                    target[2:].zero_()
                elif name == "router_output_weight":
                    target[:, :2].copy_(value)
                    target[:, 2:].copy_(value)
                elif name == "router_bias":
                    target[:2].copy_(value)
                    target[2:].copy_(value)
                else:
                    target.copy_(value)
            self.extra_input_weight[:2].copy_(
                source.extra_input_weight
            )
            self.extra_input_weight[2:].copy_(
                source.extra_input_weight
            )
            self.extra_output_weight[:2].copy_(
                2.0 * source.extra_output_weight
            )
            self.extra_output_weight[2:].zero_()

    def forward(
        self,
        coordinates: Tensor,
        *,
        query_valid_mask: Tensor | None = None,
        key_valid_mask: Tensor | None = None,
        logical_positions: Tensor | None = None,
        key_logical_positions: Tensor | None = None,
    ) -> Tensor:
        components = self.base.forward_components(
            coordinates,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )
        source_latent = torch.einsum(
            "bsi,eir->bser",
            coordinates,
            self.extra_input_weight,
        )
        mixture = (
            components.router_probabilities
            * components.source_probabilities.unsqueeze(-1)
        )
        expert_state = torch.einsum(
            "bqse,bser->bqer",
            mixture,
            source_latent,
        )
        extra = torch.einsum(
            "bqer,ero->bqo",
            expert_state,
            self.extra_output_weight,
        )
        return components.output + extra

    def concatenated_executor(
        self,
    ) -> ResidualGatedCausalModalExecutor:
        executor = ResidualGatedCausalModalExecutor(
            self.config,
            dtype=self.base.dtype,
            device=self.base.device,
        )
        source = self.base.state_dict()
        target = executor.state_dict()
        with torch.no_grad():
            for name in target:
                if name == "expert_input_weight":
                    target[name].copy_(
                        torch.cat(
                            (
                                source[name],
                                self.extra_input_weight.detach(),
                            ),
                            dim=2,
                        )
                    )
                elif name == "expert_output_weight":
                    target[name].copy_(
                        torch.cat(
                            (
                                source[name],
                                self.extra_output_weight.detach(),
                            ),
                            dim=1,
                        )
                    )
                else:
                    target[name].copy_(source[name])
        return executor

    def artifact_state_dict(self) -> dict[str, object]:
        return self.concatenated_executor().artifact_state_dict()


def _new_control_model(
    *,
    modal_center: Tensor,
    gain_log_center: float,
    gain_log_scale: float,
    residual_width: int,
    rms_epsilon: float,
    target_center: Tensor,
    target_scale: Tensor,
    seed: int,
) -> contrast_fit._PackedTrainingModule:
    source_protocol = default_function_preserving_expert_rank_control_protocol()
    rank16 = expert_rank._new_control_model(
        modal_center=modal_center,
        gain_log_center=gain_log_center,
        gain_log_scale=gain_log_scale,
        residual_width=residual_width,
        rms_epsilon=rms_epsilon,
        target_center=target_center,
        target_scale=target_scale,
        seed=seed,
    )
    return expert_rank._new_treatment_model(
        control=rank16,
        protocol=source_protocol,
        seed=seed,
    )


def _new_treatment_model(
    *,
    control: contrast_fit._PackedTrainingModule,
    protocol: FunctionPreservingExpertCountControlProtocol,
    seed: int,
) -> contrast_fit._PackedTrainingModule:
    treatment = _new_control_model(
        modal_center=control.modal_center,
        gain_log_center=control.gain_log_center,
        gain_log_scale=control.gain_log_scale,
        residual_width=control.residual_width,
        rms_epsilon=control.rms_epsilon,
        target_center=control.target_center,
        target_scale=control.target_scale,
        seed=seed,
    )
    if any(
        not torch.equal(left, right)
        for left, right in zip(
            control.parameters(),
            treatment.parameters(),
            strict=True,
        )
    ):
        raise RuntimeError("expert-count treatment base initialization drifted")
    source_executor = treatment.executor
    if not isinstance(source_executor, expert_rank._ResidualExpertRankLift):
        raise TypeError("expert-count source lacks split rank-64 executor")
    treatment.executor = _ResidualExpertCountLift(
        source_executor,
        target_config=_executor_config(protocol.e4_executor),
    )
    return treatment


def _parameter_bindings(model: nn.Module) -> dict[str, str]:
    packed = model
    if not isinstance(packed, contrast_fit._PackedTrainingModule):
        raise TypeError("parameter binding requires packed training model")
    executor = packed.executor
    if isinstance(executor, _ResidualExpertCountLift):
        executor_state = executor.artifact_state_dict()
        split = {
            "base_executor_sha256": module_state_fingerprint(executor.base),
            "extra_input_sha256": d0d3._tensor_sha256(
                executor.extra_input_weight.detach()
            ),
            "extra_output_sha256": d0d3._tensor_sha256(
                executor.extra_output_weight.detach()
            ),
        }
    else:
        executor_state = executor.artifact_state_dict()
        split = {}
    return {
        "encoder_sha256": d0d3._tensor_sha256(
            packed.encoder_weight.detach()
        ),
        "executor_artifact_sha256": (
            contrast_fit._executor_artifact_sha256(executor_state)
        ),
        "decoder_sha256": d0d3._tensor_sha256(
            packed.decoder_weight.detach()
        ),
        **split,
    }


def _difference_metrics(left: Tensor, right: Tensor) -> tuple[float, float]:
    return width._difference_metrics(left, right)


def _packed_apply_with_executor(
    model: contrast_fit._PackedTrainingModule,
    executor: nn.Module,
    modal: Tensor,
    null: Tensor,
    rms: Tensor,
    mask: Tensor,
    positions: Tensor,
) -> Tensor:
    features = model.encode_features(modal, null, rms, mask)
    latent = executor(
        features,
        query_valid_mask=mask,
        key_valid_mask=mask,
        logical_positions=positions,
        key_logical_positions=positions,
    )
    output = latent @ model.decoder_weight
    return torch.where(
        mask.unsqueeze(-1),
        output,
        torch.zeros_like(output),
    )


def _concatenated_packed_model(
    model: contrast_fit._PackedTrainingModule,
) -> contrast_fit._PackedTrainingModule:
    split = model.executor
    if not isinstance(split, _ResidualExpertCountLift):
        raise TypeError("concatenated view requires split expert-count lift")
    packed = width._new_training_model(
        modal_center=model.modal_center,
        gain_log_center=model.gain_log_center,
        gain_log_scale=model.gain_log_scale,
        residual_width=model.residual_width,
        rms_epsilon=model.rms_epsilon,
        target_center=model.target_center,
        target_scale=model.target_scale,
        config=split.config,
        seed=0,
    )
    with torch.no_grad():
        packed.encoder_weight.copy_(model.encoder_weight)
        packed.decoder_weight.copy_(model.decoder_weight)
    packed.executor = split.concatenated_executor()
    return packed


def _wrapper_concat_parity(
    model: contrast_fit._PackedTrainingModule,
    *,
    data: contrast_fit._PreparedFitData,
    target_center: Tensor,
    target_scale: Tensor,
    metric_weight: Tensor,
    objective: ContrastAwareObjective,
    protocol: FunctionPreservingExpertCountControlProtocol,
    pair_role: str,
    stage: str,
) -> tuple[dict[str, object], object]:
    if stage not in {"initial", "post_fit"}:
        raise ValueError("wrapper/concatenated parity stage is invalid")
    packed = _concatenated_packed_model(model)
    output_absolute = 0.0
    output_relative = 0.0
    for indexed in data.batches:
        batch = indexed.batch
        wrapper_output = model.forward_standardized(
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
            batch.logical_positions,
        )
        packed_output = packed.forward_standardized(
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
            batch.logical_positions,
        )
        absolute, relative = _difference_metrics(
            wrapper_output,
            packed_output,
        )
        output_absolute = max(output_absolute, absolute)
        output_relative = max(output_relative, relative)

    jvp_absolute = 0.0
    jvp_relative = 0.0
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
        mask = batch.valid_mask[row : row + 1]
        positions = batch.logical_positions[row : row + 1]

        def apply(
            candidate: contrast_fit._PackedTrainingModule,
            modal: Tensor,
            null: Tensor,
            rms: Tensor,
        ) -> Tensor:
            return candidate.forward_standardized(
                modal,
                null,
                rms,
                mask,
                positions,
            )

        _, wrapper_jvp = torch.func.jvp(
            lambda modal, null, rms: apply(
                model,
                modal,
                null,
                rms,
            ),
            args,
            tangents,
        )
        _, packed_jvp = torch.func.jvp(
            lambda modal, null, rms: apply(
                packed,
                modal,
                null,
                rms,
            ),
            args,
            tangents,
        )
        absolute, relative = _difference_metrics(
            wrapper_jvp,
            packed_jvp,
        )
        jvp_absolute = max(jvp_absolute, absolute)
        jvp_relative = max(jvp_relative, relative)
        jvp_count += 1
    wrapper_metrics = contrast_fit._materialize_metrics(
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
    packed_metrics = contrast_fit._materialize_metrics(
        contrast_fit._loss_components(
            packed,
            data=data,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=metric_weight,
            objective=objective,
        ),
        data=data,
    )
    metric_error = abs(
        wrapper_metrics.weighted_total - packed_metrics.weighted_total
    )
    absolute_tolerance = protocol.lift.equivalence_absolute_tolerance
    relative_tolerance = protocol.lift.equivalence_relative_tolerance
    flags = {
        "output_absolute": output_absolute <= absolute_tolerance,
        "output_relative": output_relative <= relative_tolerance,
        "jvp_absolute": jvp_absolute <= absolute_tolerance,
        "jvp_relative": jvp_relative <= relative_tolerance,
        "weighted_total_absolute": metric_error <= absolute_tolerance,
        "all_expected_jvps_compared": jvp_count == 32,
    }
    state = {
        "artifact_kind": "fisher_graph.expert_count_wrapper_concat_parity",
        "format_version": _FORMAT_VERSION,
        "pair_role": pair_role,
        "stage": stage,
        "maximum_output_absolute_error": output_absolute,
        "maximum_output_relative_error": output_relative,
        "maximum_jvp_absolute_error": jvp_absolute,
        "maximum_jvp_relative_error": jvp_relative,
        "weighted_total_absolute_error": metric_error,
        "jvp_pair_count": jvp_count,
        "concatenated_metrics_sha256": packed_metrics.artifact_sha256,
        "concatenated_executor_sha256": (
            contrast_fit._executor_artifact_sha256(
                packed.executor.artifact_state_dict()
            )
        ),
        "flags": flags,
        "passed": all(flags.values()),
    }
    state["artifact_sha256"] = _json_sha256(
        state,
        domain=_AUDIT_DOMAIN,
    )
    return state, packed_metrics


def _initial_equivalence(
    control: contrast_fit._PackedTrainingModule,
    treatment: contrast_fit._PackedTrainingModule,
    *,
    data: contrast_fit._PreparedFitData,
    target_center: Tensor,
    target_scale: Tensor,
    metric_weight: Tensor,
    objective: ContrastAwareObjective,
    protocol: FunctionPreservingExpertCountControlProtocol,
    pair_role: str,
    seed: int,
) -> dict[str, object]:
    if not isinstance(treatment.executor, _ResidualExpertCountLift):
        raise TypeError("expert-count treatment lacks the split lift")
    concatenated = treatment.executor.concatenated_executor()
    output_absolute = 0.0
    output_relative = 0.0
    concat_output_absolute = 0.0
    concat_output_relative = 0.0
    route_mass_absolute = 0.0
    route_mass_relative = 0.0
    sibling_route_absolute = 0.0
    source_probability_absolute = 0.0
    allowed_route_sum_absolute = 0.0
    route_masks_exact = True
    route_outside_edges_zero = True
    control_split = control.executor
    treatment_split = treatment.executor
    if not isinstance(
        control_split,
        expert_rank._ResidualExpertRankLift,
    ):
        raise TypeError("expert-count control lacks split rank-64 executor")
    assert isinstance(treatment_split, _ResidualExpertCountLift)
    control_base = control_split.base
    treatment_base = treatment_split.base
    lift_parameter_flags = {
        "base_input_active_copy": torch.equal(
            treatment_base.expert_input_weight[:2],
            control_base.expert_input_weight,
        ),
        "base_input_dormant_copy": torch.equal(
            treatment_base.expert_input_weight[2:],
            control_base.expert_input_weight,
        ),
        "base_output_active_double": torch.equal(
            treatment_base.expert_output_weight[:2],
            2.0 * control_base.expert_output_weight,
        ),
        "base_output_dormant_zero": bool(
            (treatment_base.expert_output_weight[2:] == 0).all()
        ),
        "extra_input_active_copy": torch.equal(
            treatment_split.extra_input_weight[:2],
            control_split.extra_input_weight,
        ),
        "extra_input_dormant_copy": torch.equal(
            treatment_split.extra_input_weight[2:],
            control_split.extra_input_weight,
        ),
        "extra_output_active_double": torch.equal(
            treatment_split.extra_output_weight[:2],
            2.0 * control_split.extra_output_weight,
        ),
        "extra_output_dormant_zero": bool(
            (treatment_split.extra_output_weight[2:] == 0).all()
        ),
        "router_output_active_copy": torch.equal(
            treatment_base.router_output_weight[:, :2],
            control_base.router_output_weight,
        ),
        "router_output_dormant_copy": torch.equal(
            treatment_base.router_output_weight[:, 2:],
            control_base.router_output_weight,
        ),
        "router_bias_active_copy": torch.equal(
            treatment_base.router_bias[:2],
            control_base.router_bias,
        ),
        "router_bias_dormant_copy": torch.equal(
            treatment_base.router_bias[2:],
            control_base.router_bias,
        ),
    }
    for name in (
        "same_position_weight",
        "same_position_bias",
        "router_query_weight",
        "router_key_weight",
        "router_lag_weight",
        "source_score_weight",
    ):
        control_value = getattr(control_base, name)
        treatment_value = getattr(treatment_base, name)
        lift_parameter_flags[f"{name}_copy"] = (
            control_value is None
            and treatment_value is None
            or isinstance(control_value, Tensor)
            and isinstance(treatment_value, Tensor)
            and torch.equal(control_value, treatment_value)
        )
    for indexed in data.batches:
        batch = indexed.batch
        control_features = control.encode_features(
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
        )
        treatment_features = treatment.encode_features(
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
        )
        if not torch.equal(control_features, treatment_features):
            raise RuntimeError("paired expert-count encoders drifted")
        control_route = control_split.base.forward_components(
            control_features,
            query_valid_mask=batch.valid_mask,
            key_valid_mask=batch.valid_mask,
            logical_positions=batch.logical_positions,
            key_logical_positions=batch.logical_positions,
        )
        treatment_route = treatment_split.base.forward_components(
            treatment_features,
            query_valid_mask=batch.valid_mask,
            key_valid_mask=batch.valid_mask,
            logical_positions=batch.logical_positions,
            key_logical_positions=batch.logical_positions,
        )
        aggregated = (
            treatment_route.router_probabilities[..., :2]
            + treatment_route.router_probabilities[..., 2:]
        )
        absolute, relative = _difference_metrics(
            control_route.router_probabilities,
            aggregated,
        )
        route_mass_absolute = max(route_mass_absolute, absolute)
        route_mass_relative = max(route_mass_relative, relative)
        sibling_route_absolute = max(
            sibling_route_absolute,
            float(
                (
                    treatment_route.router_probabilities[..., :2]
                    - treatment_route.router_probabilities[..., 2:]
                )
                .abs()
                .max()
                .detach()
            ),
        )
        source_probability_absolute = max(
            source_probability_absolute,
            float(
                (
                    control_route.source_probabilities
                    - treatment_route.source_probabilities
                )
                .abs()
                .max()
                .detach()
            ),
        )
        route_sum = treatment_route.router_probabilities.sum(dim=-1)
        if bool(treatment_route.positive_lag_mask.any()):
            allowed_route_sum_absolute = max(
                allowed_route_sum_absolute,
                float(
                    (
                        route_sum[
                            treatment_route.positive_lag_mask
                        ]
                        - 1.0
                    )
                    .abs()
                    .max()
                    .detach()
                ),
            )
        route_masks_exact = route_masks_exact and torch.equal(
            control_route.positive_lag_mask,
            treatment_route.positive_lag_mask,
        )
        allowed = treatment_route.positive_lag_mask.unsqueeze(-1)
        route_outside_edges_zero = route_outside_edges_zero and bool(
            (
                treatment_route.router_probabilities.masked_select(~allowed)
                == 0
            ).all()
        )
        left = control.forward_standardized(
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
            batch.logical_positions,
        )
        right = treatment.forward_standardized(
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
            batch.logical_positions,
        )
        packed = _packed_apply_with_executor(
            treatment,
            concatenated,
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
            batch.logical_positions,
        )
        absolute, relative = _difference_metrics(left, right)
        output_absolute = max(output_absolute, absolute)
        output_relative = max(output_relative, relative)
        absolute, relative = _difference_metrics(right, packed)
        concat_output_absolute = max(concat_output_absolute, absolute)
        concat_output_relative = max(concat_output_relative, relative)

    jvp_absolute = 0.0
    jvp_relative = 0.0
    concat_jvp_absolute = 0.0
    concat_jvp_relative = 0.0
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

        _, left_jvp = torch.func.jvp(
            lambda modal, null, rms: apply(
                control,
                modal,
                null,
                rms,
            ),
            args,
            tangents,
        )
        _, right_jvp = torch.func.jvp(
            lambda modal, null, rms: apply(
                treatment,
                modal,
                null,
                rms,
            ),
            args,
            tangents,
        )
        _, packed_jvp = torch.func.jvp(
            lambda modal, null, rms: _packed_apply_with_executor(
                treatment,
                concatenated,
                modal,
                null,
                rms,
                mask,
                positions,
            ),
            args,
            tangents,
        )
        absolute, relative = _difference_metrics(left_jvp, right_jvp)
        jvp_absolute = max(jvp_absolute, absolute)
        jvp_relative = max(jvp_relative, relative)
        absolute, relative = _difference_metrics(right_jvp, packed_jvp)
        concat_jvp_absolute = max(concat_jvp_absolute, absolute)
        concat_jvp_relative = max(concat_jvp_relative, relative)
        jvp_count += 1

    control_metrics = contrast_fit._materialize_metrics(
        contrast_fit._loss_components(
            control,
            data=data,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=metric_weight,
            objective=objective,
        ),
        data=data,
    )
    treatment_metrics = contrast_fit._materialize_metrics(
        contrast_fit._loss_components(
            treatment,
            data=data,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=metric_weight,
            objective=objective,
        ),
        data=data,
    )
    metric_delta = abs(
        control_metrics.weighted_total
        - treatment_metrics.weighted_total
    )
    tolerance = protocol.lift.equivalence_absolute_tolerance
    relative_tolerance = protocol.lift.equivalence_relative_tolerance
    flags = {
        "observable_absolute": output_absolute <= tolerance,
        "observable_relative": output_relative <= relative_tolerance,
        "jvp_absolute": jvp_absolute <= tolerance,
        "jvp_relative": jvp_relative <= relative_tolerance,
        "wrapper_concat_observable_absolute": (
            concat_output_absolute <= tolerance
        ),
        "wrapper_concat_observable_relative": (
            concat_output_relative <= relative_tolerance
        ),
        "wrapper_concat_jvp_absolute": (
            concat_jvp_absolute <= tolerance
        ),
        "wrapper_concat_jvp_relative": (
            concat_jvp_relative <= relative_tolerance
        ),
        "initial_weighted_total_absolute": metric_delta <= tolerance,
        "parent_route_mass_absolute": route_mass_absolute <= tolerance,
        "parent_route_mass_relative": (
            route_mass_relative <= relative_tolerance
        ),
        "sibling_route_probabilities_equal": (
            sibling_route_absolute <= tolerance
        ),
        "source_probabilities_equal": (
            source_probability_absolute <= tolerance
        ),
        "causal_masks_exact": route_masks_exact,
        "outside_edge_routes_zero": route_outside_edges_zero,
        "allowed_edge_route_mass_one": (
            allowed_route_sum_absolute <= tolerance
        ),
        "dormant_lift_parameter_identity": all(
            lift_parameter_flags.values()
        ),
        "all_expected_jvps_compared": jvp_count == 32,
    }
    state = {
        "artifact_kind": "fisher_graph.initial_expert_count_equivalence",
        "format_version": _FORMAT_VERSION,
        "pair_role": pair_role,
        "seed": seed,
        "maximum_observable_absolute_error": output_absolute,
        "maximum_observable_relative_error": output_relative,
        "maximum_jvp_absolute_error": jvp_absolute,
        "maximum_jvp_relative_error": jvp_relative,
        "maximum_wrapper_concat_observable_absolute_error": (
            concat_output_absolute
        ),
        "maximum_wrapper_concat_observable_relative_error": (
            concat_output_relative
        ),
        "maximum_wrapper_concat_jvp_absolute_error": concat_jvp_absolute,
        "maximum_wrapper_concat_jvp_relative_error": concat_jvp_relative,
        "initial_weighted_total_absolute_error": metric_delta,
        "maximum_parent_route_mass_absolute_error": route_mass_absolute,
        "maximum_parent_route_mass_relative_error": route_mass_relative,
        "maximum_sibling_route_absolute_error": sibling_route_absolute,
        "maximum_source_probability_absolute_error": (
            source_probability_absolute
        ),
        "causal_masks_exact": route_masks_exact,
        "outside_edge_routes_zero": route_outside_edges_zero,
        "maximum_allowed_edge_route_sum_absolute_error": (
            allowed_route_sum_absolute
        ),
        "lift_parameter_flags": lift_parameter_flags,
        "jvp_pair_count": jvp_count,
        "control_initial_metrics_sha256": (
            control_metrics.artifact_sha256
        ),
        "treatment_initial_metrics_sha256": (
            treatment_metrics.artifact_sha256
        ),
        "control_initial_parameter_bindings": _parameter_bindings(control),
        "treatment_initial_parameter_bindings": (
            _parameter_bindings(treatment)
        ),
        "flags": flags,
        "passed": all(flags.values()),
    }
    state["artifact_sha256"] = _json_sha256(
        state,
        domain=_AUDIT_DOMAIN,
    )
    return state


def _gradient_audit(
    *,
    applicable: bool,
    pair_role: str,
    values: Mapping[str, float],
    protocol: FunctionPreservingExpertCountControlProtocol,
) -> dict[str, object]:
    floor = protocol.lift.gradient_norm_floor
    zero_tolerance = 0.0
    expected_names = {
        f"step{step}_{bank}_{measure}_norm"
        for step, bank, measure in (
            (1, "active_base_input", "gradient"),
            (1, "active_base_output", "gradient"),
            (1, "dormant_base_input", "gradient"),
            (1, "dormant_base_output", "gradient"),
            (1, "active_extra_input", "gradient"),
            (1, "active_extra_output", "gradient"),
            (1, "dormant_extra_input", "gradient"),
            (1, "dormant_extra_output", "gradient"),
            (1, "router_sibling", "gradient"),
            (1, "active_base_input", "delta"),
            (1, "active_base_output", "delta"),
            (1, "dormant_base_input", "delta"),
            (1, "dormant_base_output", "delta"),
            (1, "active_extra_input", "delta"),
            (1, "active_extra_output", "delta"),
            (1, "dormant_extra_input", "delta"),
            (1, "dormant_extra_output", "delta"),
            (1, "router_sibling", "delta"),
            (2, "dormant_base_input", "gradient"),
            (2, "active_extra_input", "gradient"),
            (2, "dormant_extra_input", "gradient"),
            (2, "dormant_base_input", "delta"),
            (2, "active_extra_input", "delta"),
            (2, "dormant_extra_input", "delta"),
        )
    }
    if set(values) != expected_names:
        raise ValueError("expert-count gradient audit fields drifted")
    zero_names = {
        "step1_dormant_base_input_gradient_norm",
        "step1_active_extra_input_gradient_norm",
        "step1_dormant_extra_input_gradient_norm",
        "step1_dormant_base_input_delta_norm",
        "step1_active_extra_input_delta_norm",
        "step1_dormant_extra_input_delta_norm",
    }
    flags: dict[str, bool] = {}
    for name in sorted(expected_names):
        if name in zero_names:
            passed = values[name] <= zero_tolerance
            suffix = "zero"
        else:
            passed = values[name] > floor
            suffix = "open"
        flags[f"{name}_{suffix}"] = not applicable or passed
    state = {
        "artifact_kind": "fisher_graph.expert_count_gradient_openness",
        "format_version": _FORMAT_VERSION,
        "pair_role": pair_role,
        "applicable": applicable,
        **dict(sorted(values.items())),
        "gradient_norm_floor": floor,
        "zero_gradient_absolute_tolerance": zero_tolerance,
        "flags": flags,
        "passed": all(flags.values()),
    }
    state["artifact_sha256"] = _json_sha256(
        state,
        domain=_AUDIT_DOMAIN,
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
    pair_role: str,
    synthetic_binding_sha256: str,
    audit_added_rank: bool,
    protocol: FunctionPreservingExpertCountControlProtocol,
) -> tuple[
    ContrastAwareReferenceProviderPlan,
    dict[str, object],
    dict[str, object],
]:
    split = model.executor
    if audit_added_rank and not isinstance(split, _ResidualExpertCountLift):
        raise TypeError("expert-count treatment lacks split-bank executor")
    if not audit_added_rank and isinstance(split, _ResidualExpertCountLift):
        raise TypeError("expert-count control unexpectedly has split bank")
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
    if audit_added_rank:
        initial_parity, _ = _wrapper_concat_parity(
            model,
            data=data,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=metric_weight,
            objective=objective,
            protocol=protocol,
            pair_role=pair_role,
            stage="initial",
        )
        if initial_parity["passed"] is not True:
            raise RuntimeError(
                "initial split/concatenated expert-count parity failed"
            )
    plan_initial_metrics = initial
    values: dict[str, float] = {}

    def joined_norm(*tensors: Tensor) -> float:
        return float(
            torch.linalg.vector_norm(
                torch.cat(tuple(value.reshape(-1) for value in tensors))
            )
        )

    if audit_added_rank:
        assert isinstance(split, _ResidualExpertCountLift)
        initial_parameters = {
            "base_input": (
                split.base.expert_input_weight.detach().clone()
            ),
            "base_output": (
                split.base.expert_output_weight.detach().clone()
            ),
            "extra_input": split.extra_input_weight.detach().clone(),
            "extra_output": split.extra_output_weight.detach().clone(),
            "router_output": (
                split.base.router_output_weight.detach().clone()
            ),
            "router_bias": split.base.router_bias.detach().clone(),
        }
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
            raise ValueError("expert-count fit produced a nonfinite loss")
        components.weighted_total.backward()
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise ValueError("expert-count fit produced a nonfinite gradient")
        if audit_added_rank and step in (0, 1):
            assert isinstance(split, _ResidualExpertCountLift)
            base_input_gradient = split.base.expert_input_weight.grad
            base_output_gradient = split.base.expert_output_weight.grad
            extra_input_gradient = split.extra_input_weight.grad
            extra_output_gradient = split.extra_output_weight.grad
            router_output_gradient = split.base.router_output_weight.grad
            router_bias_gradient = split.base.router_bias.grad
            assert all(
                value is not None
                for value in (
                    base_input_gradient,
                    base_output_gradient,
                    extra_input_gradient,
                    extra_output_gradient,
                    router_output_gradient,
                    router_bias_gradient,
                )
            )
            assert base_input_gradient is not None
            assert base_output_gradient is not None
            assert extra_input_gradient is not None
            assert extra_output_gradient is not None
            assert router_output_gradient is not None
            assert router_bias_gradient is not None
            prefix = f"step{step + 1}"
            if step == 0:
                values.update(
                    {
                        f"{prefix}_active_base_input_gradient_norm": (
                            joined_norm(base_input_gradient[:2])
                        ),
                        f"{prefix}_active_base_output_gradient_norm": (
                            joined_norm(base_output_gradient[:2])
                        ),
                        f"{prefix}_dormant_base_input_gradient_norm": (
                            joined_norm(base_input_gradient[2:])
                        ),
                        f"{prefix}_dormant_base_output_gradient_norm": (
                            joined_norm(base_output_gradient[2:])
                        ),
                        f"{prefix}_active_extra_input_gradient_norm": (
                            joined_norm(extra_input_gradient[:2])
                        ),
                        f"{prefix}_active_extra_output_gradient_norm": (
                            joined_norm(extra_output_gradient[:2])
                        ),
                        f"{prefix}_dormant_extra_input_gradient_norm": (
                            joined_norm(extra_input_gradient[2:])
                        ),
                        f"{prefix}_dormant_extra_output_gradient_norm": (
                            joined_norm(extra_output_gradient[2:])
                        ),
                        f"{prefix}_router_sibling_gradient_norm": (
                            joined_norm(
                                (
                                    router_output_gradient[:, :2]
                                    - router_output_gradient[:, 2:]
                                ),
                                (
                                    router_bias_gradient[:2]
                                    - router_bias_gradient[2:]
                                ),
                            )
                        ),
                    }
                )
            else:
                values.update(
                    {
                        f"{prefix}_dormant_base_input_gradient_norm": (
                            joined_norm(base_input_gradient[2:])
                        ),
                        f"{prefix}_active_extra_input_gradient_norm": (
                            joined_norm(extra_input_gradient[:2])
                        ),
                        f"{prefix}_dormant_extra_input_gradient_norm": (
                            joined_norm(extra_input_gradient[2:])
                        ),
                    }
                )
        optimizer.step()
        if audit_added_rank and step in (0, 1):
            assert isinstance(split, _ResidualExpertCountLift)
            prefix = f"step{step + 1}"
            deltas = {
                "base_input": (
                    split.base.expert_input_weight.detach()
                    - initial_parameters["base_input"]
                ),
                "base_output": (
                    split.base.expert_output_weight.detach()
                    - initial_parameters["base_output"]
                ),
                "extra_input": (
                    split.extra_input_weight.detach()
                    - initial_parameters["extra_input"]
                ),
                "extra_output": (
                    split.extra_output_weight.detach()
                    - initial_parameters["extra_output"]
                ),
                "router_output": (
                    split.base.router_output_weight.detach()
                    - initial_parameters["router_output"]
                ),
                "router_bias": (
                    split.base.router_bias.detach()
                    - initial_parameters["router_bias"]
                ),
            }
            if step == 0:
                values.update(
                    {
                        f"{prefix}_active_base_input_delta_norm": (
                            joined_norm(deltas["base_input"][:2])
                        ),
                        f"{prefix}_active_base_output_delta_norm": (
                            joined_norm(deltas["base_output"][:2])
                        ),
                        f"{prefix}_dormant_base_input_delta_norm": (
                            joined_norm(deltas["base_input"][2:])
                        ),
                        f"{prefix}_dormant_base_output_delta_norm": (
                            joined_norm(deltas["base_output"][2:])
                        ),
                        f"{prefix}_active_extra_input_delta_norm": (
                            joined_norm(deltas["extra_input"][:2])
                        ),
                        f"{prefix}_active_extra_output_delta_norm": (
                            joined_norm(deltas["extra_output"][:2])
                        ),
                        f"{prefix}_dormant_extra_input_delta_norm": (
                            joined_norm(deltas["extra_input"][2:])
                        ),
                        f"{prefix}_dormant_extra_output_delta_norm": (
                            joined_norm(deltas["extra_output"][2:])
                        ),
                        f"{prefix}_router_sibling_delta_norm": (
                            joined_norm(
                                (
                                    deltas["router_output"][:, :2]
                                    - deltas["router_output"][:, 2:]
                                ),
                                (
                                    deltas["router_bias"][:2]
                                    - deltas["router_bias"][2:]
                                ),
                            )
                        ),
                    }
                )
            else:
                values.update(
                    {
                        f"{prefix}_dormant_base_input_delta_norm": (
                            joined_norm(deltas["base_input"][2:])
                        ),
                        f"{prefix}_active_extra_input_delta_norm": (
                            joined_norm(deltas["extra_input"][:2])
                        ),
                        f"{prefix}_dormant_extra_input_delta_norm": (
                            joined_norm(deltas["extra_input"][2:])
                        ),
                    }
                )
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
    if audit_added_rank:
        postfit_parity, plan_final_metrics = _wrapper_concat_parity(
            model,
            data=data,
            target_center=target_center,
            target_scale=target_scale,
            metric_weight=metric_weight,
            objective=objective,
            protocol=protocol,
            pair_role=pair_role,
            stage="post_fit",
        )
        if postfit_parity["passed"] is not True:
            raise RuntimeError(
                "post-fit split/concatenated expert-count parity failed"
            )
    else:
        plan_final_metrics = final
        postfit_parity = {
            "artifact_kind": (
                "fisher_graph.expert_count_wrapper_concat_parity"
            ),
            "format_version": _FORMAT_VERSION,
            "pair_role": pair_role,
            "stage": "not_applicable_control",
            "applicable": False,
            "passed": True,
        }
        postfit_parity["artifact_sha256"] = _json_sha256(
            postfit_parity,
            domain=_AUDIT_DOMAIN,
        )
    bindings = {
        value.batch.synthetic_binding_sha256 for value in data.batches
    }
    if bindings != {synthetic_binding_sha256}:
        raise ValueError("expert-count fit batches lost their binding")
    executor = model.executor
    if isinstance(executor, _ResidualExpertCountLift):
        executor_artifact = executor.artifact_state_dict()
    else:
        executor_artifact = executor.artifact_state_dict()
    plan = ContrastAwareReferenceProviderPlan(
        modal_center=model.modal_center,
        gain_log_center=model.gain_log_center,
        gain_log_scale=model.gain_log_scale,
        residual_width=model.residual_width,
        rms_epsilon=model.rms_epsilon,
        target_center=model.target_center,
        target_scale=model.target_scale,
        encoder_weight=model.encoder_weight.detach(),
        executor_artifact=executor_artifact,
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
        initial_metrics=plan_initial_metrics,
        final_metrics=plan_final_metrics,
    )
    audit = _gradient_audit(
        applicable=audit_added_rank,
        pair_role=pair_role,
        values=(
            values
            if audit_added_rank
            else {
                name: 0.0
                for name in {
                    f"step{step}_{bank}_{measure}_norm"
                    for step, bank, measure in (
                        (1, "active_base_input", "gradient"),
                        (1, "active_base_output", "gradient"),
                        (1, "dormant_base_input", "gradient"),
                        (1, "dormant_base_output", "gradient"),
                        (1, "active_extra_input", "gradient"),
                        (1, "active_extra_output", "gradient"),
                        (1, "dormant_extra_input", "gradient"),
                        (1, "dormant_extra_output", "gradient"),
                        (1, "router_sibling", "gradient"),
                        (1, "active_base_input", "delta"),
                        (1, "active_base_output", "delta"),
                        (1, "dormant_base_input", "delta"),
                        (1, "dormant_base_output", "delta"),
                        (1, "active_extra_input", "delta"),
                        (1, "active_extra_output", "delta"),
                        (1, "dormant_extra_input", "delta"),
                        (1, "dormant_extra_output", "delta"),
                        (1, "router_sibling", "delta"),
                        (2, "dormant_base_input", "gradient"),
                        (2, "active_extra_input", "gradient"),
                        (2, "dormant_extra_input", "gradient"),
                        (2, "dormant_base_input", "delta"),
                        (2, "active_extra_input", "delta"),
                        (2, "dormant_extra_input", "delta"),
                    )
                }
            }
        ),
        protocol=protocol,
    )
    return plan, audit, postfit_parity


def _authenticated_declarations(
    *,
    protocol_override: (
        FunctionPreservingExpertCountControlProtocol | None
    ) = None,
) -> tuple[
    FunctionPreservingExpertCountControlProtocol,
    object,
    object,
    object,
]:
    protocol = (
        default_function_preserving_expert_count_control_protocol()
        if protocol_override is None
        else protocol_override
    )
    objective_protocol = default_objective_balance_diagnostic_protocol()
    c2_protocol = default_contrast_provider_development_protocol()
    d3_recipe = objective_protocol.recipe(protocol.training.recipe_id)
    source_expert_rank_protocol = (
        default_function_preserving_expert_rank_control_protocol()
    )
    training = protocol.training
    if (
        (
            protocol_override is None
            and protocol.protocol_sha256
            != DEFAULT_FUNCTION_PRESERVING_EXPERT_COUNT_CONTROL_PROTOCOL_SHA256
        )
        or d3_recipe.artifact_sha256
        != default_function_preserving_width_control_protocol().sources
        .d3_recipe_sha256
        or d3_recipe.primary_seed != training.primary_seed
        or d3_recipe.signed_pair_multiplicity
        != training.signed_pair_multiplicity
        or d3_recipe.pointwise_weight != training.pointwise_weight
        or d3_recipe.sensitivity_relative_delta_weight
        != training.relative_delta_weight
        or d3_recipe.direction_weight != training.direction_weight
        or d3_recipe.midpoint_jvp_weight != training.midpoint_jvp_weight
        or d3_recipe.intended_null_weight != training.intended_null_weight
        or d3_recipe.steps != training.steps
        or d3_recipe.learning_rate != training.learning_rate
        or objective_protocol.gates.ordinary_gates_sha256
        != training.ordinary_gates_sha256
        or objective_protocol.gates.contrast_gates_sha256
        != training.contrast_gates_sha256
        or source_expert_rank_protocol.protocol_sha256
        != protocol.source.expert_rank_protocol_sha256
    ):
        raise ValueError("expert-count declarations drifted")
    return protocol, objective_protocol, c2_protocol, d3_recipe


def _authenticate_sources(
    *,
    source_expert_rank_path: Path | str,
    source_diagnostic_path: Path | str,
    source_rank64_path: Path | str,
    protocol: FunctionPreservingExpertCountControlProtocol,
) -> _AuthenticatedSources:
    source = protocol.source
    loaded = expert_rank.load_function_preserving_expert_rank_control_artifact(
        source_expert_rank_path,
        expected_artifact_sha256=(
            source.expert_rank_logical_artifact_sha256
        ),
        expected_tensor_file_sha256=(
            source.expert_rank_tensor_file_sha256
        ),
        expected_report_sha256=source.expert_rank_report_sha256,
    )
    plans = loaded.state.get("plan_states")
    rows = loaded.state.get("candidate_results")
    if not isinstance(plans, Mapping) or not isinstance(rows, Mapping):
        raise ValueError("authenticated expert-rank source tables are absent")
    raw_plan = plans.get(_SOURCE_CANDIDATE_ID)
    row = rows.get(_SOURCE_CANDIDATE_ID)
    if not isinstance(raw_plan, Mapping) or not isinstance(row, Mapping):
        raise ValueError("authenticated expert-rank E2/R64 source is absent")
    plan = ContrastAwareReferenceProviderPlan.from_state_dict(raw_plan)
    plan.validate_integrity()
    if (
        loaded.manifest.get("protocol_sha256")
        != source.expert_rank_protocol_sha256
        or loaded.manifest.get("code_bundle_sha256")
        != source.expert_rank_code_bundle_sha256
        or loaded.manifest.get("outcome") != source.expert_rank_outcome
        or loaded.manifest.get("primary_comparison_status")
        != source.expert_rank_primary_comparison_status
        or loaded.manifest.get("primary_treatment_valid")
        is not source.expert_rank_primary_treatment_valid
        or loaded.manifest.get("expert_count_control_authorized")
        is not source.expert_rank_expert_count_control_authorized
        or loaded.manifest.get("replication_executed") is not False
        or loaded.manifest.get("two_seed_inner_expert_rank_supported")
        is not False
        or loaded.manifest.get("descending_expert_rank_ladder_authorized")
        is not False
        or loaded.manifest.get("fresh_c3_authorized") is not False
        or loaded.manifest.get("candidate_plan_sha256s", {}).get(
            _SOURCE_CANDIDATE_ID
        )
        != source.expert_rank_primary_e2r64_plan_sha256
        or loaded.manifest.get("candidate_result_sha256s", {}).get(
            _SOURCE_CANDIDATE_ID
        )
        != source.expert_rank_primary_e2r64_result_sha256
        or plan.artifact_sha256
        != source.expert_rank_primary_e2r64_plan_sha256
        or plan.initial_metrics.artifact_sha256
        != source.expert_rank_primary_e2r64_initial_metrics_sha256
        or plan.final_metrics.artifact_sha256
        != source.expert_rank_primary_e2r64_final_metrics_sha256
        or row.get("fit_capability_pass") is not False
        or row.get("pair_comparison_status") != "both_fail"
        or plan.latent_rank != 64
        or plan.executor_config.expert_count != 2
        or plan.executor_config.expert_rank != 64
    ):
        raise ValueError("authenticated expert-rank source identity drifted")
    width_protocol = default_function_preserving_width_control_protocol()
    predecessor = width._authenticate_sources(
        source_diagnostic_path=source_diagnostic_path,
        source_rank64_path=source_rank64_path,
        protocol=width_protocol,
    )
    return _AuthenticatedSources(
        expert_rank_result=loaded,
        expert_rank_plan=plan,
        source_d3=predecessor.source_d3,
    )


def describe_function_preserving_expert_count_control() -> dict[str, object]:
    """Describe the sealed rung without loading a model or result artifact."""

    protocol, objective_protocol, c2_protocol, d3_recipe = (
        _authenticated_declarations()
    )
    code = _code_sha256s()
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "protocol": protocol.state_dict(),
        "objective_protocol_sha256": objective_protocol.protocol_sha256,
        "c2_protocol_sha256": c2_protocol.protocol_sha256,
        "d3_recipe_sha256": d3_recipe.artifact_sha256,
        "source_artifacts_loaded": False,
        "model_loaded": False,
        "outer_latent_rank": 64,
        "expert_rank": 64,
        "control_expert_count": 2,
        "treatment_expert_count": 4,
        "valid_primary_both_fail_next_rung": (
            "separately_preregister_e8_expert_count_control_only"
        ),
        "fit_only_split_bank_parameterization": True,
        "published_plan_uses_concatenated_standard_executor": True,
        "selection_allowed": False,
        "fresh_c3_authorized": False,
        "fresh_c3_authorized_before_descending_ladder": False,
        "compression_claim_authorized": False,
        "speed_claim_authorized": False,
        "run_generated_receipt_verifies_publication_integrity_only": True,
        "run_generated_receipt_is_external_scientific_trust_root": False,
        "durable_scientific_authority_requires_recording_exact_receipt_"
        "triple_outside_artifact": True,
        "code_sha256s": code,
        "code_bundle_sha256": _code_bundle_sha256(code),
    }
    d0d3._assert_tensor_free_report(report)
    return report


@dataclass(slots=True)
class _LiveFitProblem:
    protocol: FunctionPreservingExpertCountControlProtocol
    objective_protocol: object
    c2_protocol: object
    d3_recipe: object
    sources: _AuthenticatedSources
    basis: object
    adapter: object
    pre_ff3: nn.Module
    norm_sha256: str
    model_before_sha256: str
    epsilon: float
    fit: Sequence[object]
    fit_batches: Sequence[object]
    fit_pairs: Sequence[object]
    modal_center: Tensor
    gain_log_center: float
    gain_log_scale: float
    target_center: Tensor
    target_scale: Tensor
    raw_metric_weight: Tensor
    unit_gauge: UnitRmsFisherGauge
    raw_teacher_energy: float
    unit_teacher_energy: float
    training_teacher_signal: Mapping[str, object]
    fit_data_binding_sha256: str
    ordinary_probes: Sequence[object]
    controls: object
    standardized_gauge_sha256: str
    fidelity_gates: SyntheticReferenceGates
    contrast_gates: ContrastAssessmentGates
    calibration: object
    measurement_evidence: Mapping[str, object]
    actual_device: str
    actual_dtype: str


def _prepare_live_fit_problem(
    *,
    source_expert_rank_path: Path | str,
    source_diagnostic_path: Path | str,
    source_rank64_path: Path | str,
    basis_package_path: Path | str,
    basis_package_file_sha256: str,
    basis_package_payload_sha256: str,
    cache_dir: Path | str | None,
    device_name: str,
    dtype: str,
    _protocol_override: (
        FunctionPreservingExpertCountControlProtocol | None
    ) = None,
) -> _LiveFitProblem:
    protocol, objective_protocol, c2_protocol, d3_recipe = (
        _authenticated_declarations(protocol_override=_protocol_override)
    )
    if (
        device_name != protocol.execution_device
        or dtype != protocol.execution_dtype
        or basis_package_file_sha256
        != DEFAULT_BASIS_PACKAGE_FILE_SHA256
        or basis_package_payload_sha256
        != DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
    ):
        raise ValueError("expert-count execution and basis bindings are frozen")
    sources = _authenticate_sources(
        source_expert_rank_path=source_expert_rank_path,
        source_diagnostic_path=source_diagnostic_path,
        source_rank64_path=source_rank64_path,
        protocol=protocol,
    )
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
        raise ValueError("expert-count gates drifted")
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
        raise ValueError("live execution differs from expert-count protocol")
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
    if (
        calibration.selected_amplitude != 8.0
        or calibration.artifact_sha256 != _EXPECTED_CALIBRATION_SHA256
    ):
        raise ValueError("expert-count calibration replay drifted")
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
        raise RuntimeError("selection identity entered expert-count fit")
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
        raise ValueError("expert-count Fisher gauge drifted")
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
        or fit_data_binding
        != sources.expert_rank_plan.synthetic_binding_sha256
        or tuple(value.artifact_sha256 for value in fit_pairs)
        != sources.source_d3.balanced_pair_sha256s
    ):
        raise ValueError("expert-count fit problem differs from source")
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
    return _LiveFitProblem(
        protocol=protocol,
        objective_protocol=objective_protocol,
        c2_protocol=c2_protocol,
        d3_recipe=d3_recipe,
        sources=sources,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        norm_sha256=norm_sha256,
        model_before_sha256=model_before,
        epsilon=epsilon,
        fit=fit,
        fit_batches=fit_batches,
        fit_pairs=fit_pairs,
        modal_center=modal_center,
        gain_log_center=gain_log_center,
        gain_log_scale=gain_log_scale,
        target_center=target_center,
        target_scale=target_scale,
        raw_metric_weight=raw_metric_weight,
        unit_gauge=unit_gauge,
        raw_teacher_energy=raw_teacher_energy,
        unit_teacher_energy=unit_teacher_energy,
        training_teacher_signal=training_teacher_signal,
        fit_data_binding_sha256=fit_data_binding,
        ordinary_probes=ordinary_probes,
        controls=controls,
        standardized_gauge_sha256=standardized_gauge_sha,
        fidelity_gates=fidelity_gates,
        contrast_gates=contrast_gates,
        calibration=calibration,
        measurement_evidence=measurement_evidence,
        actual_device=actual_device,
        actual_dtype=actual_dtype,
    )


def _run_fit_only_preflight(
    problem: _LiveFitProblem,
) -> dict[str, object]:
    """Run only the frozen two-step openness audit; never score or publish."""

    protocol = problem.protocol
    data = contrast_fit._prepare_fit_data(
        fit_batches=problem.fit_batches,
        contrast_pairs=problem.fit_pairs,
        require_fit_split=True,
    )
    objective = _objective(protocol)
    by_role: dict[str, object] = {}
    for pair_role, seed in (
        ("primary", protocol.training.primary_seed),
        ("replication", protocol.training.replication_seed),
    ):
        control = _new_control_model(
            modal_center=problem.modal_center,
            gain_log_center=problem.gain_log_center,
            gain_log_scale=problem.gain_log_scale,
            residual_width=problem.basis.residual_width,
            rms_epsilon=problem.epsilon,
            target_center=problem.target_center,
            target_scale=problem.target_scale,
            seed=seed,
        )
        treatment = _new_treatment_model(
            control=control,
            protocol=protocol,
            seed=seed,
        )
        initial = _initial_equivalence(
            control,
            treatment,
            data=data,
            target_center=problem.target_center,
            target_scale=problem.target_scale,
            metric_weight=problem.unit_gauge.metric_weight,
            objective=objective,
            protocol=protocol,
            pair_role=pair_role,
            seed=seed,
        )
        (
            control_plan,
            control_gradient,
            control_postfit_parity,
        ) = _fit_from_initialized_model(
            control,
            data=data,
            target_center=problem.target_center,
            target_scale=problem.target_scale,
            metric_weight=problem.unit_gauge.metric_weight,
            objective=objective,
            steps=2,
            learning_rate=protocol.training.learning_rate,
            seed=seed,
            pair_role=pair_role,
            synthetic_binding_sha256=(
                problem.fit_data_binding_sha256
            ),
            audit_added_rank=False,
            protocol=protocol,
        )
        (
            treatment_plan,
            treatment_gradient,
            treatment_postfit_parity,
        ) = (
            _fit_from_initialized_model(
                treatment,
                data=data,
                target_center=problem.target_center,
                target_scale=problem.target_scale,
                metric_weight=problem.unit_gauge.metric_weight,
                objective=objective,
                steps=2,
                learning_rate=protocol.training.learning_rate,
                seed=seed,
                pair_role=pair_role,
                synthetic_binding_sha256=(
                    problem.fit_data_binding_sha256
                ),
                audit_added_rank=True,
                protocol=protocol,
            )
        )
        if (
            initial["passed"] is not True
            or control_gradient["passed"] is not True
            or treatment_gradient["passed"] is not True
        ):
            raise RuntimeError("expert-count fit-only preflight failed")
        state = {
            "artifact_kind": (
                "fisher_graph.expert_count_fit_only_preflight_seed"
            ),
            "format_version": _FORMAT_VERSION,
            "pair_role": pair_role,
            "seed": seed,
            "steps": 2,
            "scoring_executed": False,
            "publication_executed": False,
            "initialization_equivalence": initial,
            "control_gradient_openness": control_gradient,
            "treatment_gradient_openness": treatment_gradient,
            "control_postfit_wrapper_concat_parity": (
                control_postfit_parity
            ),
            "treatment_postfit_wrapper_concat_parity": (
                treatment_postfit_parity
            ),
            "control_two_step_initial_metrics_sha256": (
                control_plan.initial_metrics.artifact_sha256
            ),
            "control_two_step_final_metrics_sha256": (
                control_plan.final_metrics.artifact_sha256
            ),
            "treatment_two_step_initial_metrics_sha256": (
                treatment_plan.initial_metrics.artifact_sha256
            ),
            "treatment_two_step_final_metrics_sha256": (
                treatment_plan.final_metrics.artifact_sha256
            ),
            "control_two_step_executor_sha256": (
                contrast_fit._executor_artifact_sha256(
                    control_plan.executor_artifact
                )
            ),
            "treatment_two_step_executor_sha256": (
                contrast_fit._executor_artifact_sha256(
                    treatment_plan.executor_artifact
                )
            ),
        }
        state["artifact_sha256"] = _json_sha256(
            state,
            domain=_AUDIT_DOMAIN,
        )
        by_role[pair_role] = state
    preflight = {
        "artifact_kind": (
            "fisher_graph.expert_count_fit_only_preflight"
        ),
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "roles": by_role,
        "scoring_executed": False,
        "publication_executed": False,
    }
    preflight["artifact_sha256"] = _json_sha256(
        preflight,
        domain=_AUDIT_DOMAIN,
    )
    return preflight


def _validate_fit_only_preflight(
    preflight: object,
    *,
    protocol: FunctionPreservingExpertCountControlProtocol,
) -> Mapping[str, object]:
    """Require the live two-step audit to match every frozen binding."""

    top_fields = {
        "artifact_kind",
        "format_version",
        "protocol_sha256",
        "roles",
        "scoring_executed",
        "publication_executed",
        "artifact_sha256",
    }
    role_fields = {
        "artifact_kind",
        "format_version",
        "pair_role",
        "seed",
        "steps",
        "scoring_executed",
        "publication_executed",
        "initialization_equivalence",
        "control_gradient_openness",
        "treatment_gradient_openness",
        "control_postfit_wrapper_concat_parity",
        "treatment_postfit_wrapper_concat_parity",
        "control_two_step_initial_metrics_sha256",
        "control_two_step_final_metrics_sha256",
        "treatment_two_step_initial_metrics_sha256",
        "treatment_two_step_final_metrics_sha256",
        "control_two_step_executor_sha256",
        "treatment_two_step_executor_sha256",
        "artifact_sha256",
    }
    if not isinstance(preflight, Mapping) or set(preflight) != top_fields:
        raise ValueError("expert-count live preflight schema drifted")
    unhashed = dict(preflight)
    supplied = unhashed.pop("artifact_sha256", None)
    roles = preflight.get("roles")
    if (
        supplied != _json_sha256(unhashed, domain=_AUDIT_DOMAIN)
        or preflight.get("artifact_kind")
        != "fisher_graph.expert_count_fit_only_preflight"
        or type(preflight.get("format_version")) is not int
        or preflight.get("format_version") != _FORMAT_VERSION
        or preflight.get("protocol_sha256")
        != protocol.protocol_sha256
        or preflight.get("scoring_executed") is not False
        or preflight.get("publication_executed") is not False
        or not isinstance(roles, Mapping)
        or set(roles) != {"primary", "replication"}
    ):
        raise ValueError("expert-count live preflight binding drifted")
    assert isinstance(roles, Mapping)
    for pair_role in ("primary", "replication"):
        role = roles[pair_role]
        frozen = _preflight_binding_for_role(protocol, pair_role)
        if not isinstance(role, Mapping) or set(role) != role_fields:
            raise ValueError("expert-count live preflight role schema drifted")
        role_unhashed = dict(role)
        role_sha = role_unhashed.pop("artifact_sha256", None)
        if (
            role_sha != _json_sha256(role_unhashed, domain=_AUDIT_DOMAIN)
            or role.get("artifact_kind")
            != "fisher_graph.expert_count_fit_only_preflight_seed"
            or role.get("format_version") != _FORMAT_VERSION
            or type(role.get("format_version")) is not int
            or role.get("pair_role") != pair_role
            or role.get("seed") != frozen["seed"]
            or type(role.get("seed")) is not int
            or role.get("steps") != frozen["preflight_steps"]
            or type(role.get("steps")) is not int
            or role.get("scoring_executed") is not False
            or role.get("publication_executed") is not False
        ):
            raise ValueError("expert-count live preflight role drifted")
        initial = _validate_initial_audit(
            role.get("initialization_equivalence"),
            protocol=protocol,
            pair_role=pair_role,
        )
        control_gradient = _validate_gradient_audit(
            role.get("control_gradient_openness"),
            protocol=protocol,
            pair_role=pair_role,
            arm="expert2",
        )
        treatment_gradient = _validate_gradient_audit(
            role.get("treatment_gradient_openness"),
            protocol=protocol,
            pair_role=pair_role,
            arm="expert4",
        )
        control_parity = {
            "artifact_kind": (
                "fisher_graph.expert_count_wrapper_concat_parity"
            ),
            "format_version": _FORMAT_VERSION,
            "pair_role": pair_role,
            "stage": "not_applicable_control",
            "applicable": False,
            "passed": True,
        }
        control_parity["artifact_sha256"] = _json_sha256(
            control_parity,
            domain=_AUDIT_DOMAIN,
        )
        treatment_parity = _validate_hashed_audit(
            role.get("treatment_postfit_wrapper_concat_parity"),
            fields=_PARITY_FIELDS,
            label="two-step postfit parity",
        )
        expected_parity = frozen["two_step_postfit_parity"]
        expected_control_two_step = frozen["control_two_step"]
        if not isinstance(expected_parity, Mapping):
            raise TypeError("frozen two-step parity is invalid")
        if not isinstance(expected_control_two_step, Mapping):
            raise TypeError("frozen control two-step binding is invalid")
        control_bindings = initial.get(
            "control_initial_parameter_bindings"
        )
        treatment_bindings = initial.get(
            "treatment_initial_parameter_bindings"
        )
        if not isinstance(control_bindings, Mapping) or not isinstance(
            treatment_bindings,
            Mapping,
        ):
            raise TypeError("live preflight initial bindings are invalid")
        expected_treatment_parity = {
            "maximum_output_absolute_error": expected_parity[
                "maximum_output_absolute_error"
            ],
            "maximum_output_relative_error": expected_parity[
                "maximum_output_relative_error"
            ],
            "maximum_jvp_absolute_error": expected_parity[
                "maximum_jvp_absolute_error"
            ],
            "maximum_jvp_relative_error": expected_parity[
                "maximum_jvp_relative_error"
            ],
            "weighted_total_absolute_error": expected_parity[
                "weighted_total_absolute_error"
            ],
            "concatenated_metrics_sha256": expected_parity[
                "metrics_sha256"
            ],
            "concatenated_executor_sha256": expected_parity[
                "concatenated_executor_sha256"
            ],
        }
        if (
            initial.get("artifact_sha256")
            != frozen["initialization_audit_sha256"]
            or control_bindings.get("executor_artifact_sha256")
            != frozen["control_initial_executor_sha256"]
            or treatment_bindings.get("executor_artifact_sha256")
            != frozen["treatment_initial_executor_sha256"]
            or treatment_bindings.get("base_executor_sha256")
            != frozen["treatment_base_executor_sha256"]
            or treatment_bindings.get("extra_input_sha256")
            != frozen["extra_u_initial_sha256"]
            or treatment_bindings.get("extra_output_sha256")
            != frozen["extra_v_initial_sha256"]
            or treatment_gradient.get("artifact_sha256")
            != frozen["treatment_gradient_audit_sha256"]
            or treatment_parity.get("artifact_sha256")
            != frozen["two_step_postfit_parity_sha256"]
            or any(
                treatment_parity.get(name) != value
                for name, value in expected_treatment_parity.items()
            )
            or treatment_parity.get("jvp_pair_count")
            != frozen["expected_jvp_pair_count"]
            or treatment_parity.get("passed") is not True
            or _canonical_json_bytes(
                role.get("control_postfit_wrapper_concat_parity")
            )
            != _canonical_json_bytes(control_parity)
            or role.get("control_two_step_initial_metrics_sha256")
            != initial["control_initial_metrics_sha256"]
            or role.get("control_two_step_final_metrics_sha256")
            != expected_control_two_step["metrics_sha256"]
            or role.get("control_two_step_executor_sha256")
            != expected_control_two_step["executor_sha256"]
            or role.get("treatment_two_step_initial_metrics_sha256")
            != initial["treatment_initial_metrics_sha256"]
            or role.get("treatment_two_step_final_metrics_sha256")
            != expected_parity["metrics_sha256"]
            or role.get("treatment_two_step_executor_sha256")
            != expected_parity["concatenated_executor_sha256"]
        ):
            raise ValueError("expert-count frozen live preflight drifted")
        if (
            control_gradient.get("passed") is not True
            or treatment_gradient.get("passed") is not True
        ):
            raise ValueError("expert-count live preflight gradient failed")
    return preflight


def _source_replay_exact(
    *,
    pair_role: str,
    arm: str,
    plan: ContrastAwareReferenceProviderPlan,
    protocol: FunctionPreservingExpertCountControlProtocol,
) -> bool:
    if pair_role != "primary" or arm != "expert2":
        return False
    return (
        plan.artifact_sha256
        == protocol.source.expert_rank_primary_e2r64_plan_sha256
        and plan.initial_metrics.artifact_sha256
        == protocol.source.expert_rank_primary_e2r64_initial_metrics_sha256
        and plan.final_metrics.artifact_sha256
        == protocol.source.expert_rank_primary_e2r64_final_metrics_sha256
    )


def _score_plan(
    *,
    candidate_id: str,
    pair_role: str,
    arm: str,
    plan: ContrastAwareReferenceProviderPlan,
    initialization_audit: Mapping[str, object],
    gradient_audit: Mapping[str, object],
    postfit_parity: Mapping[str, object],
    protocol: FunctionPreservingExpertCountControlProtocol,
    objective_protocol: object,
    d3_recipe: object,
    source: _AuthenticatedSources,
    fit: Sequence[object],
    fit_batches: Sequence[object],
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
    accounting = asdict(plan.accounting())
    execution = d0d3._fit_execution_accounting(plan, fit_batches)
    expected_stored, expected_macs = (
        (31_492, 5_555_776)
        if arm == "expert2"
        else (48_166, 8_985_792)
    )
    if (
        accounting["total_stored_scalar_count"] != expected_stored
        or execution["canonical_total_mac_count"] != expected_macs
    ):
        raise RuntimeError("expert-count frozen accounting drifted")
    row = {
        "candidate_id": candidate_id,
        "pair_role": pair_role,
        "arm": arm,
        "seed": plan.seed,
        "outer_rank": plan.latent_rank,
        "expert_count": plan.executor_config.expert_count,
        "expert_rank": plan.executor_config.expert_rank,
        "plan_sha256": plan.artifact_sha256,
        "candidate_binding_sha256": candidate.artifact_sha256,
        "source_replay_exact": _source_replay_exact(
            pair_role=pair_role,
            arm=arm,
            plan=plan,
            protocol=protocol,
        ),
        "source_sequence_comparison": sequence,
        "initialization_equivalence": dict(initialization_audit),
        "gradient_openness": dict(gradient_audit),
        "postfit_wrapper_concat_parity": dict(postfit_parity),
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
        "accounting": accounting,
        "execution_accounting": execution,
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
        ordinary_candidate=candidate,
    )


def _pair_status(control_pass: bool, treatment_pass: bool) -> str:
    if not control_pass and not treatment_pass:
        return "both_fail"
    if not control_pass and treatment_pass:
        return "expert2_fail_expert4_pass"
    if control_pass and treatment_pass:
        return "both_pass"
    return "expert2_pass_expert4_fail"


def _preflight_binding_for_role(
    protocol: FunctionPreservingExpertCountControlProtocol,
    pair_role: str,
) -> Mapping[str, object]:
    preflight = getattr(protocol, "preflight", None)
    if preflight is None or not hasattr(preflight, "for_role"):
        raise RuntimeError("expert-count preflight binding is absent")
    binding = preflight.for_role(pair_role)
    if not isinstance(binding, Mapping):
        raise TypeError("expert-count preflight role binding is invalid")
    return binding


def _evaluate_pair(
    *,
    pair_role: str,
    seed: int,
    problem: _LiveFitProblem,
) -> _PairEvaluation:
    protocol = problem.protocol
    if pair_role not in {"primary", "replication"}:
        raise ValueError("expert-count pair role is invalid")
    expected_seed = (
        protocol.training.primary_seed
        if pair_role == "primary"
        else protocol.training.replication_seed
    )
    if seed != expected_seed:
        raise ValueError("expert-count seed drifted")
    objective = _objective(protocol)
    data = contrast_fit._prepare_fit_data(
        fit_batches=problem.fit_batches,
        contrast_pairs=problem.fit_pairs,
        require_fit_split=True,
    )
    control_model = _new_control_model(
        modal_center=problem.modal_center,
        gain_log_center=problem.gain_log_center,
        gain_log_scale=problem.gain_log_scale,
        residual_width=problem.basis.residual_width,
        rms_epsilon=problem.epsilon,
        target_center=problem.target_center,
        target_scale=problem.target_scale,
        seed=seed,
    )
    treatment_model = _new_treatment_model(
        control=control_model,
        protocol=protocol,
        seed=seed,
    )
    initial = _initial_equivalence(
        control_model,
        treatment_model,
        data=data,
        target_center=problem.target_center,
        target_scale=problem.target_scale,
        metric_weight=problem.unit_gauge.metric_weight,
        objective=objective,
        protocol=protocol,
        pair_role=pair_role,
        seed=seed,
    )
    frozen = _preflight_binding_for_role(protocol, pair_role)
    if (
        initial["passed"] is not True
        or initial["artifact_sha256"]
        != frozen["initialization_audit_sha256"]
    ):
        raise RuntimeError("expert-count initial preflight binding drifted")
    control_plan, control_gradient, control_parity = (
        _fit_from_initialized_model(
            control_model,
            data=data,
            target_center=problem.target_center,
            target_scale=problem.target_scale,
            metric_weight=problem.unit_gauge.metric_weight,
            objective=objective,
            steps=protocol.training.steps,
            learning_rate=protocol.training.learning_rate,
            seed=seed,
            pair_role=pair_role,
            synthetic_binding_sha256=problem.fit_data_binding_sha256,
            audit_added_rank=False,
            protocol=protocol,
        )
    )
    treatment_plan, treatment_gradient, treatment_parity = (
        _fit_from_initialized_model(
            treatment_model,
            data=data,
            target_center=problem.target_center,
            target_scale=problem.target_scale,
            metric_weight=problem.unit_gauge.metric_weight,
            objective=objective,
            steps=protocol.training.steps,
            learning_rate=protocol.training.learning_rate,
            seed=seed,
            pair_role=pair_role,
            synthetic_binding_sha256=problem.fit_data_binding_sha256,
            audit_added_rank=True,
            protocol=protocol,
        )
    )
    if (
        treatment_gradient["artifact_sha256"]
        != frozen["treatment_gradient_audit_sha256"]
        or treatment_gradient["passed"] is not True
        or treatment_parity["passed"] is not True
    ):
        raise RuntimeError("expert-count live gradient/parity audit drifted")
    prefix = f"fp_expert_count.{pair_role}"
    common = {
        "initialization_audit": initial,
        "protocol": protocol,
        "objective_protocol": problem.objective_protocol,
        "d3_recipe": problem.d3_recipe,
        "source": problem.sources,
        "fit": problem.fit,
        "fit_batches": problem.fit_batches,
        "raw_metric_weight": problem.raw_metric_weight,
        "raw_teacher_energy": problem.raw_teacher_energy,
        "training_teacher_energy": problem.unit_teacher_energy,
        "teacher_signal_diagnostics": problem.training_teacher_signal,
        "ordinary_probes": problem.ordinary_probes,
        "controls": problem.controls,
        "standardized_gauge_sha256": (
            problem.standardized_gauge_sha256
        ),
        "fidelity_gates": problem.fidelity_gates,
        "contrast_gates": problem.contrast_gates,
    }
    control = _score_plan(
        candidate_id=f"{prefix}.expert2",
        pair_role=pair_role,
        arm="expert2",
        plan=control_plan,
        gradient_audit=control_gradient,
        postfit_parity=control_parity,
        **common,
    )
    treatment = _score_plan(
        candidate_id=f"{prefix}.expert4",
        pair_role=pair_role,
        arm="expert4",
        plan=treatment_plan,
        gradient_audit=treatment_gradient,
        postfit_parity=treatment_parity,
        **common,
    )
    flags = {
        "initial_observable_and_jvp_equivalence": (
            initial["passed"] is True
        ),
        "gradient_open_expert4": treatment_gradient["passed"] is True,
        "postfit_wrapper_concat_parity": (
            treatment_parity["passed"] is True
        ),
        "expert2_balance": (
            control.row["objective_balance_gate"]["passed"] is True  # type: ignore[index]
        ),
        "expert4_balance": (
            treatment.row["objective_balance_gate"]["passed"] is True  # type: ignore[index]
        ),
        "expert2_source_sequences": (
            control.row["source_sequence_comparison"]["passed"] is True  # type: ignore[index]
        ),
        "expert4_source_sequences": (
            treatment.row["source_sequence_comparison"]["passed"] is True  # type: ignore[index]
        ),
        "expert2_primary_replay": (
            control.row["source_replay_exact"] is (pair_role == "primary")
            and treatment.row["source_replay_exact"] is False
        ),
        "fixed_outer_encoder_decoder_router": True,
        "exact_training_contract": (
            control_plan.training_steps
            == treatment_plan.training_steps
            == protocol.training.steps
            and control_plan.learning_rate
            == treatment_plan.learning_rate
            == protocol.training.learning_rate
            and control_plan.objective.artifact_sha256
            == treatment_plan.objective.artifact_sha256
            == objective.artifact_sha256
        ),
    }
    valid = all(flags.values())
    status = _pair_status(
        control.fit_capability_pass,
        treatment.fit_capability_pass,
    )
    validity = {
        "passed": valid,
        "flags": flags,
        "failure_semantics": (
            "invalid_paired_expert_count_comparison_no_capacity_conclusion"
        ),
    }
    for arm in (control, treatment):
        arm.row["pair_treatment_validity"] = validity
        arm.row["pair_comparison_status"] = status
    return _PairEvaluation(
        pair_role=pair_role,
        seed=seed,
        control=control,
        treatment=treatment,
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
    elif primary.comparison_status != (
        "expert2_fail_expert4_pass"
    ):
        outcome = f"primary_{primary.comparison_status}"
    elif replication is None:
        raise RuntimeError("authorized expert-count replication was not run")
    elif not replication.treatment_valid:
        outcome = "invalid_replication_pair"
    elif replication.comparison_status == (
        "expert2_fail_expert4_pass"
    ):
        outcome = "two_seed_routed_expert_count_support"
    else:
        outcome = (
            "inconsistent_replication_"
            f"{replication.comparison_status}"
        )
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
        "two_seed_routed_expert_count_supported": (
            outcome == "two_seed_routed_expert_count_support"
        ),
        "descending_expert_count_ladder_authorized": (
            outcome == "two_seed_routed_expert_count_support"
        ),
        "e8_expert_count_control_authorized": (
            outcome == "primary_both_fail"
        ),
        "fresh_c3_authorized": False,
        "fresh_c3_after_descending_ladder_only": (
            outcome == "two_seed_routed_expert_count_support"
        ),
        "compression_claim_authorized": False,
        "speed_claim_authorized": False,
    }


def _interpretation_from_evidence(
    rows: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    primary = rows.get("fp_expert_count.primary.expert4")
    if not isinstance(primary, Mapping):
        raise ValueError("expert-count primary treatment row is absent")
    validity = primary.get("pair_treatment_validity")
    gradient = primary.get("gradient_openness")
    parity = primary.get("postfit_wrapper_concat_parity")
    if not isinstance(validity, Mapping):
        raise TypeError("expert-count validity must be a mapping")
    if not isinstance(gradient, Mapping) or not isinstance(parity, Mapping):
        raise TypeError("expert-count audits must be mappings")
    return {
        "outcome": decision.get("outcome"),
        "fit_only_capacity_oracle": True,
        "fixed_outer_rank": 64,
        "controlled_expert_count_change": "2_to_4",
        "primary_treatment_valid": (
            decision.get("primary_treatment_valid") is True
        ),
        "initial_function_and_jvp_matched": (
            validity.get("flags", {}).get(  # type: ignore[union-attr]
                "initial_observable_and_jvp_equivalence"
            )
            is True
        ),
        "added_expert_bank_gradient_open": gradient.get("passed") is True,
        "postfit_split_concat_parity": parity.get("passed") is True,
        "two_seed_routed_expert_count_supported": (
            decision.get("two_seed_routed_expert_count_supported") is True
        ),
        "expert_count_alone_proves_only_bottleneck": False,
        "both_fail_opens_e8_expert_count_control": (
            decision.get("e8_expert_count_control_authorized") is True
        ),
        "descending_ladder_authorized": (
            decision.get("descending_expert_count_ladder_authorized") is True
        ),
        "fresh_c3_authorized": False,
        "compression_claim_authorized": False,
        "speed_claim_authorized": False,
        "natural_prompt_fidelity_claim": False,
        "whole_model_replacement_claim": False,
        "post_run_scientific_authority_requires_external_trust_anchor_"
        "triple": True,
        "run_generated_receipt_verifies_publication_integrity_only": True,
        "run_generated_receipt_is_external_scientific_trust_root": False,
        "self_hashes_sufficient_for_scientific_authority": False,
    }


def _safety_contract() -> dict[str, bool]:
    return {
        "contains_source_model_state_dict": False,
        "contains_loaded_expert_rank_source_final_provider_parameters": (
            False
        ),
        "contains_regenerated_expert_rank_final_equivalent_provider_"
        "parameters": True,
        "contains_regenerated_expert_rank_initial_parameter_hashes": True,
        "contains_expert2_provider_parameters": True,
        "contains_expert4_provider_parameters": True,
        "contains_ordinary_provider_prediction_tensors": True,
        "contains_synthetic_fit_teacher_target_tensors": True,
        "contains_raw_fit_targets": True,
        "contains_raw_model_teacher_hidden_states": False,
        "contains_teacher_jvp_tensors": True,
        "contains_provider_chart_jvp_tensors": True,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_c2_selection_data": False,
        "external_trust_anchor_embedded_in_artifact": False,
        "scientific_outcome_requires_external_trust_anchor": True,
        "run_generated_receipt_verifies_publication_integrity_only": True,
        "run_generated_receipt_is_external_scientific_trust_root": False,
        "durable_scientific_authority_requires_recording_exact_receipt_"
        "triple_outside_artifact": True,
        "committable": False,
    }


_FIT_PAIR_TENSOR_FIELDS = {
    "teacher_midpoint_jvp",
    "provider_chart_modal_primal",
    "provider_chart_null_primal",
    "provider_chart_row_rms_primal",
    "provider_chart_modal_tangent",
    "provider_chart_null_tangent",
    "provider_chart_row_rms_tangent",
}


def _assert_safe_expert_count_artifact_tree(
    value: object,
    *,
    path: str = "state",
) -> None:
    """Allow only the preregistered fit replay tensors generic D0-D3 forbids."""

    always_forbidden = {
        "prompt",
        "prompt_text",
        "prompt_texts",
        "token_ids",
        "input_ids",
        "tokenizer_state",
        "hidden_states",
        "raw_hidden_states",
        "source_model_state_dict",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_name = str(key)
            nested_path = f"{path}.{key_name}"
            if key_name in always_forbidden:
                raise ValueError(
                    f"{nested_path} is forbidden in expert-count artifacts"
                )
            if key_name == "target_modes":
                allowed = (
                    (
                        path.startswith(
                            "state.ordinary_scoring_batch_states["
                        )
                        and path.endswith(".batch_state")
                        and isinstance(nested, Tensor)
                        and nested.dtype is torch.float64
                        and nested.ndim == 3
                        and nested.shape[-1] == 64
                    )
                    or (
                        path.endswith(".accounting")
                        and type(nested) is int
                        and nested == 64
                    )
                )
                if not allowed:
                    raise ValueError(
                        f"{nested_path} is forbidden in expert-count artifacts"
                    )
                continue
            if key_name in _FIT_PAIR_TENSOR_FIELDS:
                allowed = (
                    path.startswith("state.fit_contrast_pair_states[")
                    and "." not in path.removeprefix("state.")
                    and (nested is None or isinstance(nested, Tensor))
                )
                if not allowed:
                    raise ValueError(
                        f"{nested_path} is forbidden in expert-count artifacts"
                    )
                continue
            _assert_safe_expert_count_artifact_tree(
                nested,
                path=nested_path,
            )
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_safe_expert_count_artifact_tree(
                nested,
                path=f"{path}[{index}]",
            )


def _validate_output_path(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("expert-count output must use a .pt suffix")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite expert-count output")
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


def _canonical_torch_state_tree(value: object) -> object:
    """Normalize identity-sensitive pickle details without changing values."""

    text_pool: dict[str, str] = {}

    def canonical_text(text: str) -> str:
        if text not in text_pool:
            text_pool[text] = (
                memoryview(
                    text.encode("utf-8", errors="surrogatepass")
                )
                .tobytes()
                .decode("utf-8", errors="surrogatepass")
            )
        return text_pool[text]

    def canonical_tree(nested: object) -> object:
        if isinstance(nested, Tensor):
            return (
                nested.detach()
                .to(device="cpu")
                .contiguous()
                .clone()
            )
        if isinstance(nested, Mapping):
            return {
                (
                    canonical_text(key)
                    if isinstance(key, str)
                    else key
                ): canonical_tree(item)
                for key, item in nested.items()
            }
        if isinstance(nested, list):
            return [canonical_tree(item) for item in nested]
        if isinstance(nested, tuple):
            return tuple(canonical_tree(item) for item in nested)
        if isinstance(nested, str):
            return canonical_text(nested)
        if isinstance(nested, bytes):
            return memoryview(nested).tobytes()
        return nested

    return canonical_tree(value)


def _canonical_torch_payload(state: Mapping[str, object]) -> bytes:
    """Serialize a normalized state through Torch's canonical stream name."""

    buffer = io.BytesIO()
    canonical = _canonical_torch_state_tree(dict(state))
    torch.save(canonical, buffer)
    return buffer.getvalue()


def _publish_artifact(
    state: Mapping[str, object],
    report_payload: Mapping[str, object],
    *,
    output: Path,
) -> dict[str, str]:
    _assert_safe_expert_count_artifact_tree(state)
    d0d3._assert_safe_artifact_tree(report_payload, path="report")
    d0d3._assert_tensor_free_report(report_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite expert-count output")
    tensor_stage = _stage_path(output)
    report_stage = _stage_path(report_path)
    published: list[Path] = []
    receipt: dict[str, str] | None = None
    try:
        tensor_payload = _canonical_torch_payload(state)
        _require_canonical_torch_zip_framing(tensor_payload)
        with tensor_stage.open("wb") as handle:
            handle.write(tensor_payload)
            handle.flush()
            os.fsync(handle.fileno())
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
        raise RuntimeError("expert-count publication produced no receipt")
    return receipt


def _read_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.read_bytes()


def _central_directory_is_exact(
    payload: bytes,
    *,
    offset: int,
    size: int,
    entry_count: int,
) -> bool:
    """Validate the complete central directory and its local-header offsets."""

    if (
        offset < 0
        or size < 0
        or entry_count <= 0
        or offset + size > len(payload)
    ):
        return False
    cursor = offset
    local_offsets: list[int] = []
    for _ in range(entry_count):
        if (
            cursor + 46 > offset + size
            or payload[cursor : cursor + 4] != b"PK\x01\x02"
        ):
            return False
        filename_length = struct.unpack_from("<H", payload, cursor + 28)[0]
        extra_length = struct.unpack_from("<H", payload, cursor + 30)[0]
        comment_length = struct.unpack_from("<H", payload, cursor + 32)[0]
        local_offset = struct.unpack_from("<L", payload, cursor + 42)[0]
        if (
            local_offset == 0xFFFFFFFF
            or local_offset + 4 > offset
            or payload[local_offset : local_offset + 4] != b"PK\x03\x04"
        ):
            return False
        local_offsets.append(local_offset)
        cursor += 46 + filename_length + extra_length + comment_length
    return (
        cursor == offset + size
        and len(set(local_offsets)) == entry_count
        and min(local_offsets) == 0
    )


def _canonical_torch_zip_archive_end(payload: bytes) -> int:
    """Return the first complete Torch ZIP64 boundary in ``payload``."""

    if len(payload) < 98 or payload[:4] != b"PK\x03\x04":
        raise ValueError("expert-count tensor file is not a Torch ZIP archive")
    candidates: list[int] = []
    search_from = 0
    while True:
        eocd_offset = payload.find(b"PK\x05\x06", search_from)
        if eocd_offset < 0:
            break
        search_from = eocd_offset + 1
        if eocd_offset + 22 > len(payload):
            continue
        (
            _signature,
            disk_number,
            central_disk,
            entries_on_disk,
            entries_total,
            central_size_32,
            central_offset_32,
            comment_length,
        ) = struct.unpack_from("<4s4H2LH", payload, eocd_offset)
        archive_end = eocd_offset + 22 + comment_length
        locator_offset = eocd_offset - 20
        if (
            archive_end > len(payload)
            or disk_number != 0
            or central_disk != 0
            or entries_on_disk != entries_total
            or locator_offset < 0
            or payload[locator_offset : locator_offset + 4]
            != b"PK\x06\x07"
        ):
            continue
        (
            _locator_signature,
            zip64_disk,
            zip64_offset,
            total_disks,
        ) = struct.unpack_from("<4sLQL", payload, locator_offset)
        if (
            zip64_disk != 0
            or total_disks != 1
            or zip64_offset + 56 > locator_offset
            or payload[zip64_offset : zip64_offset + 4]
            != b"PK\x06\x06"
        ):
            continue
        zip64_size = struct.unpack_from("<Q", payload, zip64_offset + 4)[0]
        if (
            zip64_size < 44
            or zip64_offset + 12 + zip64_size != locator_offset
        ):
            continue
        (
            _zip64_signature,
            _zip64_size,
            _version_made,
            _version_needed,
            zip64_disk_number,
            zip64_central_disk,
            zip64_entries_on_disk,
            zip64_entries_total,
            central_size,
            central_offset,
        ) = struct.unpack_from("<4sQ2H2L4Q", payload, zip64_offset)
        if (
            zip64_disk_number != 0
            or zip64_central_disk != 0
            or zip64_entries_on_disk != zip64_entries_total
            or zip64_entries_total <= 0
            or central_offset + central_size != zip64_offset
            or (
                entries_total != 0xFFFF
                and entries_total != zip64_entries_total
            )
            or (
                central_size_32 != 0xFFFFFFFF
                and central_size_32 != central_size
            )
            or (
                central_offset_32 != 0xFFFFFFFF
                and central_offset_32 != central_offset
            )
            or not _central_directory_is_exact(
                payload,
                offset=central_offset,
                size=central_size,
                entry_count=zip64_entries_total,
            )
        ):
            continue
        candidates.append(archive_end)
    if not candidates:
        raise ValueError("expert-count Torch ZIP framing is invalid")
    return min(candidates)


def _require_canonical_torch_zip_framing(payload: bytes) -> None:
    """Reject data hidden after the canonical Torch ZIP archive."""

    if _canonical_torch_zip_archive_end(payload) != len(payload):
        raise ValueError(
            "expert-count tensor file contains trailing bytes"
        )


def _restore_ordinary_candidate(
    raw: object,
) -> FullWidthReferenceCandidate:
    candidate_fields = {
        "artifact_kind",
        "format_version",
        "candidate_id",
        "source_rank",
        "target_rank",
        "stored_scalar_count",
        "prediction_sha256s",
        "structural_metrics",
        "candidate_binding_sha256",
        "predictions",
        "artifact_sha256",
    }
    prediction_fields = {
        "artifact_kind",
        "format_version",
        "probe_id",
        "retained_standardized_prediction_sha256",
        "standardized_gauge_sha256",
        "retained_standardized_prediction",
        "artifact_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != candidate_fields:
        raise ValueError("expert-count ordinary candidate schema drifted")
    predictions_raw = raw.get("predictions")
    if not isinstance(predictions_raw, (tuple, list)):
        raise TypeError("expert-count ordinary predictions are invalid")
    predictions: list[FullWidthCandidatePrediction] = []
    for value in predictions_raw:
        if not isinstance(value, Mapping) or set(value) != prediction_fields:
            raise ValueError(
                "expert-count ordinary prediction schema drifted"
            )
        prediction = FullWidthCandidatePrediction(
            probe_id=str(value["probe_id"]),
            retained_standardized_prediction=value[
                "retained_standardized_prediction"
            ],  # type: ignore[arg-type]
            standardized_gauge_sha256=str(
                value["standardized_gauge_sha256"]
            ),
            artifact_sha256=str(value["artifact_sha256"]),
        )
        if (
            value.get("artifact_kind")
            != "fisher_graph.full_width_candidate_prediction"
            or type(value.get("format_version")) is not int
            or value.get("format_version") != 1
            or value.get("retained_standardized_prediction_sha256")
                != reference_selection._tensor_sha256(
                prediction.retained_standardized_prediction
            )
        ):
            raise ValueError(
                "expert-count ordinary prediction binding drifted"
            )
        predictions.append(prediction)
    structural = FullWidthStructuralMetrics.from_state_dict(
        raw.get("structural_metrics")
    )
    candidate = FullWidthReferenceCandidate(
        candidate_id=str(raw["candidate_id"]),
        source_rank=raw["source_rank"],  # type: ignore[arg-type]
        target_rank=raw["target_rank"],  # type: ignore[arg-type]
        stored_scalar_count=raw["stored_scalar_count"],  # type: ignore[arg-type]
        predictions=tuple(predictions),
        structural_metrics=structural,
        candidate_binding_sha256=str(
            raw["candidate_binding_sha256"]
        ),
        artifact_sha256=str(raw["artifact_sha256"]),
    )
    if (
        raw.get("artifact_kind")
        != "fisher_graph.full_width_reference_candidate"
        or type(raw.get("format_version")) is not int
        or raw.get("format_version") != 1
        or list(raw.get("prediction_sha256s", ()))
        != sorted(value.artifact_sha256 for value in predictions)
    ):
        raise ValueError("expert-count ordinary candidate binding drifted")
    return candidate


def _ordered_ordinary_scoring_batches(
    values: Sequence[IndexedReferenceBatch],
) -> tuple[IndexedReferenceBatch, ...]:
    """Return the exact plan-bound fit-batch order used for replay."""

    batches = tuple(values)
    if not batches or any(
        not isinstance(value, IndexedReferenceBatch) for value in batches
    ):
        raise TypeError(
            "ordinary scoring batches must contain IndexedReferenceBatch "
            "values"
        )
    ordered = tuple(sorted(batches, key=lambda value: value.artifact_sha256))
    if len({value.artifact_sha256 for value in ordered}) != len(ordered):
        raise ValueError("ordinary scoring batches contain duplicate hashes")
    if any(
        value.batch.split != "fit"
        or value.batch.modal_modes != 64
        or value.batch.target_mode_count != 64
        or value.batch.null_modes != 1
        for value in ordered
    ):
        raise ValueError("ordinary scoring batch geometry drifted")
    bindings = {
        value.batch.synthetic_binding_sha256 for value in ordered
    }
    if len(bindings) != 1:
        raise ValueError("ordinary scoring batches use multiple bindings")
    return ordered


def _ordinary_scoring_endpoint_locations(
    batches: Sequence[IndexedReferenceBatch],
) -> dict[str, tuple[IndexedReferenceBatch, int, str]]:
    """Recompute the plan's exact endpoint identities from fit batches."""

    ordered = _ordered_ordinary_scoring_batches(batches)
    result: dict[str, tuple[IndexedReferenceBatch, int, str]] = {}
    for indexed in ordered:
        for row_index, endpoint_id in enumerate(indexed.endpoint_ids):
            if endpoint_id in result:
                raise ValueError(
                    "ordinary scoring endpoint ids are not unique"
                )
            endpoint_sha256 = contrast_fit._json_sha256(
                {
                    "endpoint_id": endpoint_id,
                    "indexed_batch_sha256": indexed.artifact_sha256,
                    "batch_content_sha256": indexed.batch.content_sha256,
                    "row_index": row_index,
                },
                domain=contrast_fit._ENDPOINT_DOMAIN,
            )
            result[endpoint_id] = (
                indexed,
                row_index,
                endpoint_sha256,
            )
    return result


def _ordinary_scoring_probes(
    *,
    batches: Sequence[IndexedReferenceBatch],
    controls: FullWidthReferenceControls,
    metric_weight: Tensor,
    c2_protocol: object,
) -> tuple[FullWidthReferenceProbe, ...]:
    """Rebuild the exact target-bearing fit probes used by the scorer."""

    endpoints = _ordinary_scoring_endpoint_locations(batches)
    probes_for_role = getattr(c2_protocol, "probes_for_role", None)
    if not callable(probes_for_role):
        raise TypeError("C2 protocol does not expose fit probes")
    probe_ids = tuple(
        value.probe_id
        for value in probes_for_role("fit")
        if value.family in {"multitone", "block_sparse"}
    )
    if (
        len(probe_ids) != 16
        or set(probe_ids) != set(controls.fit_probe_ids)
    ):
        raise ValueError("ordinary scoring probe order drifted")
    probes: list[FullWidthReferenceProbe] = []
    for probe_id in probe_ids:
        location = endpoints.get(probe_id)
        if location is None:
            raise ValueError("ordinary scoring probe endpoint is absent")
        indexed, row_index, _ = location
        batch = indexed.batch
        probes.append(
            FullWidthReferenceProbe(
                probe_id=probe_id,
                split="fit",
                family=probe_id.rsplit(".", 2)[-2],
                standardized_target=(
                    batch.target_modes[row_index : row_index + 1]
                    * metric_weight.view(1, 1, -1)
                ),
                logical_positions=(
                    batch.logical_positions[row_index : row_index + 1]
                ),
                valid_mask=batch.valid_mask[row_index : row_index + 1],
                standardized_gauge_sha256=(
                    controls.standardized_gauge_sha256
                ),
            )
        )
    result = tuple(probes)
    if (
        set(value.probe_id for value in result)
        != set(controls.fit_probe_ids)
        or tuple(sorted(value.artifact_sha256 for value in result))
        != controls.fit_probe_sha256s
    ):
        raise ValueError("ordinary scoring probe bindings drifted")
    return result


def _restored_fit_endpoints(
    *,
    batches: Sequence[IndexedReferenceBatch],
    c2_protocol: object,
) -> tuple[_RestoredFitEndpoint, ...]:
    """Rebuild the original ordinal fit-panel view without prompt material."""

    probes_for_role = getattr(c2_protocol, "probes_for_role", None)
    if not callable(probes_for_role):
        raise TypeError("C2 protocol does not expose fit probes")
    declared = tuple(probes_for_role("fit"))
    endpoints = _ordinary_scoring_endpoint_locations(batches)
    declared_ids = tuple(value.probe_id for value in declared)
    if (
        len(declared) != 80
        or len(set(declared_ids)) != len(declared_ids)
        or set(declared_ids) != set(endpoints)
    ):
        raise ValueError("restored fit endpoint identities drifted")
    result: list[_RestoredFitEndpoint] = []
    for probe in declared:
        indexed, row_index, _ = endpoints[probe.probe_id]
        batch = indexed.batch
        target = batch.target_modes[row_index : row_index + 1]
        result.append(
            _RestoredFitEndpoint(
                probe=probe,
                modal_coordinates=(
                    batch.modal_coordinates[row_index : row_index + 1]
                ),
                null_coordinates=(
                    batch.null_coordinates[row_index : row_index + 1]
                ),
                row_rms=batch.row_rms[row_index : row_index + 1],
                target_modes=target,
                target_replays=(target, target.clone()),
                logical_positions=(
                    batch.logical_positions[row_index : row_index + 1]
                ),
                valid_mask=batch.valid_mask[row_index : row_index + 1],
            )
        )
    return tuple(result)


def _restored_runtime_predictions(
    plan: ContrastAwareReferenceProviderPlan,
    measured: Sequence[_RestoredFitEndpoint],
    *,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    runtime = plan.prepare(dtype=dtype, device="cpu")
    predictions: dict[str, Tensor] = {}
    with torch.no_grad():
        for value in measured:
            probe_id = str(getattr(value.probe, "probe_id"))
            predictions[probe_id] = (
                runtime(
                    value.modal_coordinates.to(dtype=dtype),
                    value.null_coordinates.to(dtype=dtype),
                    value.row_rms.to(dtype=dtype),
                    valid_mask=value.valid_mask,
                    logical_positions=value.logical_positions,
                )
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            )
    return predictions


def _recompute_ordinary_candidate_and_score(
    *,
    candidate_id: str,
    plan: ContrastAwareReferenceProviderPlan,
    published_candidate: FullWidthReferenceCandidate,
    published_score: FullWidthCandidateScore,
    batches: Sequence[IndexedReferenceBatch],
    measured: Sequence[_RestoredFitEndpoint],
    probes: Sequence[FullWidthReferenceProbe],
    controls: FullWidthReferenceControls,
    metric_weight: Tensor,
    gates: SyntheticReferenceGates,
    c2_protocol: object,
    contrast_gates: ContrastAssessmentGates,
    required_null_candidate_pass_count: int,
    published_structural_metadata: object,
    published_contrast_result: object,
    published_contrast_identities: object,
    published_contrast_coverage: object,
) -> tuple[FullWidthReferenceCandidate, FullWidthCandidateScore]:
    """Numerically replay a plan and exactly reproduce its ordinary score."""

    _ordinary_scoring_endpoint_locations(batches)
    raw64 = _restored_runtime_predictions(
        plan,
        measured,
        dtype=torch.float64,
    )
    raw32 = _restored_runtime_predictions(
        plan,
        measured,
        dtype=torch.float32,
    )
    support_radius = c2._feature_radius(plan, measured)  # type: ignore[arg-type]
    structural, structural_metadata = c2._structural_metrics(  # type: ignore[arg-type]
        plan,
        measured,
        support_radius=support_radius,
        raw64=raw64,
        raw32=raw32,
    )
    if (
        structural.state_dict()
        != published_candidate.structural_metrics.state_dict()
        or _canonical_json_bytes(structural_metadata)
        != _canonical_json_bytes(published_structural_metadata)
    ):
        raise ValueError(
            "expert-count ordinary structural replay drifted"
        )
    predictions: list[FullWidthCandidatePrediction] = []
    for probe_id in controls.fit_probe_ids:
        predictions.append(
            FullWidthCandidatePrediction(
                probe_id=probe_id,
                retained_standardized_prediction=(
                    raw64[probe_id]
                    * metric_weight.view(1, 1, -1)
                ),
                standardized_gauge_sha256=(
                    controls.standardized_gauge_sha256
                ),
            )
        )
    recomputed_candidate = FullWidthReferenceCandidate(
        candidate_id=candidate_id,
        source_rank=plan.latent_rank,
        target_rank=64,
        stored_scalar_count=(
            plan.accounting().total_stored_scalar_count
        ),
        predictions=tuple(predictions),
        structural_metrics=structural,
        candidate_binding_sha256=plan.artifact_sha256,
    )
    published_predictions = {
        value.probe_id: value
        for value in published_candidate.predictions
    }
    if (
        set(published_predictions) != set(controls.fit_probe_ids)
        or any(
            value.artifact_sha256
            != published_predictions[value.probe_id].artifact_sha256
            or not torch.equal(
                value.retained_standardized_prediction,
                published_predictions[
                    value.probe_id
                ].retained_standardized_prediction,
            )
            for value in recomputed_candidate.predictions
        )
        or
        recomputed_candidate.artifact_sha256
        != published_candidate.artifact_sha256
    ):
        raise ValueError(
            "expert-count ordinary candidate numerical replay drifted"
        )
    recomputed_score = d0d3._score_fit_only_ordinary_candidate(
        controls=controls,
        fit_probes=probes,
        candidate=recomputed_candidate,
        gates=gates,
    )
    if (
        recomputed_score.artifact_sha256
        != published_score.artifact_sha256
        or _canonical_json_bytes(recomputed_score.state_dict())
        != _canonical_json_bytes(published_score.state_dict())
    ):
        recomputed_score_state = recomputed_score.state_dict()
        published_score_state = published_score.state_dict()
        differing_fields = tuple(
            name
            for name in recomputed_score_state
            if _canonical_json_bytes(recomputed_score_state[name])
            != _canonical_json_bytes(published_score_state.get(name))
        )
        raise ValueError(
            "expert-count ordinary score replay drifted: "
            f"{differing_fields}"
        )
    (
        recomputed_contrast,
        recomputed_identities,
        recomputed_coverage,
    ) = d0d3._fit_contrast_assessment(
        protocol=c2_protocol,
        measured=measured,
        predictions=raw64,
        metric_weight=metric_weight,
        gates=contrast_gates,
        required_null_candidate_pass_count=(
            required_null_candidate_pass_count
        ),
    )
    if (
        _canonical_json_bytes(recomputed_contrast.state_dict())
        != _canonical_json_bytes(published_contrast_result)
        or _canonical_json_bytes(recomputed_identities)
        != _canonical_json_bytes(published_contrast_identities)
        or _canonical_json_bytes(recomputed_coverage)
        != _canonical_json_bytes(published_contrast_coverage)
    ):
        raise ValueError("expert-count contrast replay drifted")
    return recomputed_candidate, recomputed_score


_STATE_FIELDS = {
    "manifest",
    "artifact_sha256",
    "protocol_state",
    "calibration_state",
    "unit_rms_gauge_state",
    "canonical_metric_weight",
    "controls_state",
    "ordinary_scoring_batch_states",
    "fit_contrast_pair_states",
    "plan_states",
    "ordinary_candidate_states",
    "candidate_results",
}
_DECISION_FIELDS = {
    "outcome",
    "primary_treatment_valid",
    "primary_comparison_status",
    "replication_executed",
    "replication_treatment_valid",
    "replication_comparison_status",
    "two_seed_routed_expert_count_supported",
    "descending_expert_count_ladder_authorized",
    "e8_expert_count_control_authorized",
    "fresh_c3_authorized",
    "fresh_c3_after_descending_ladder_only",
    "compression_claim_authorized",
    "speed_claim_authorized",
}
_MANIFEST_FIELDS = {
    "schema",
    "format_version",
    "protocol_sha256",
    "source_expert_rank_protocol_sha256",
    "source_expert_rank_code_bundle_sha256",
    "source_expert_rank_logical_artifact_sha256",
    "source_expert_rank_tensor_file_sha256",
    "source_expert_rank_report_sha256",
    "source_expert_rank_primary_e2r64_plan_sha256",
    "source_expert_rank_primary_e2r64_result_sha256",
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
    "ordinary_candidate_sha256s",
    "ordinary_scoring_indexed_batch_sha256s",
    "fit_contrast_pair_sha256s",
    "candidate_result_sha256s",
    *_DECISION_FIELDS,
    "selection_materialized",
    "selection_measured",
    "selection_scored",
    "c2_provider_artifact_loaded",
    "authenticated_expert_rank_source_loaded",
    "source_final_parameters_used_for_initialization",
    "fit_split_wrapper_used",
    "published_plans_use_concatenated_executor",
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
    *_MEASUREMENT_FIELDS,
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
    "outer_rank",
    "expert_count",
    "expert_rank",
    "plan_sha256",
    "candidate_binding_sha256",
    "source_replay_exact",
    "source_sequence_comparison",
    "initialization_equivalence",
    "gradient_openness",
    "postfit_wrapper_concat_parity",
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
    "execution_accounting",
    "fit_capability_contract",
    "fit_capability_pass",
    "pair_treatment_validity",
    "pair_comparison_status",
}
_VALIDITY_FIELDS = {"passed", "flags", "failure_semantics"}
_STRUCTURAL_FIELDS = width._STRUCTURAL_METADATA_FIELDS
_SEQUENCE_FIELDS = width._SOURCE_SEQUENCE_COMPARISON_FIELDS
_INITIAL_FIELDS = {
    "artifact_kind",
    "format_version",
    "pair_role",
    "seed",
    "maximum_observable_absolute_error",
    "maximum_observable_relative_error",
    "maximum_jvp_absolute_error",
    "maximum_jvp_relative_error",
    "maximum_wrapper_concat_observable_absolute_error",
    "maximum_wrapper_concat_observable_relative_error",
    "maximum_wrapper_concat_jvp_absolute_error",
    "maximum_wrapper_concat_jvp_relative_error",
    "initial_weighted_total_absolute_error",
    "maximum_parent_route_mass_absolute_error",
    "maximum_parent_route_mass_relative_error",
    "maximum_sibling_route_absolute_error",
    "maximum_source_probability_absolute_error",
    "maximum_allowed_edge_route_sum_absolute_error",
    "causal_masks_exact",
    "outside_edge_routes_zero",
    "lift_parameter_flags",
    "jvp_pair_count",
    "control_initial_metrics_sha256",
    "treatment_initial_metrics_sha256",
    "control_initial_parameter_bindings",
    "treatment_initial_parameter_bindings",
    "flags",
    "passed",
    "artifact_sha256",
}
_GRADIENT_FIELDS = {
    "artifact_kind",
    "format_version",
    "pair_role",
    "applicable",
    "step1_active_base_input_gradient_norm",
    "step1_active_base_output_gradient_norm",
    "step1_dormant_base_input_gradient_norm",
    "step1_dormant_base_output_gradient_norm",
    "step1_active_extra_input_gradient_norm",
    "step1_active_extra_output_gradient_norm",
    "step1_dormant_extra_input_gradient_norm",
    "step1_dormant_extra_output_gradient_norm",
    "step1_router_sibling_gradient_norm",
    "step1_active_base_input_delta_norm",
    "step1_active_base_output_delta_norm",
    "step1_dormant_base_input_delta_norm",
    "step1_dormant_base_output_delta_norm",
    "step1_active_extra_input_delta_norm",
    "step1_active_extra_output_delta_norm",
    "step1_dormant_extra_input_delta_norm",
    "step1_dormant_extra_output_delta_norm",
    "step1_router_sibling_delta_norm",
    "step2_dormant_base_input_gradient_norm",
    "step2_active_extra_input_gradient_norm",
    "step2_dormant_extra_input_gradient_norm",
    "step2_dormant_base_input_delta_norm",
    "step2_active_extra_input_delta_norm",
    "step2_dormant_extra_input_delta_norm",
    "gradient_norm_floor",
    "zero_gradient_absolute_tolerance",
    "flags",
    "passed",
    "artifact_sha256",
}
_PARITY_FIELDS = {
    "artifact_kind",
    "format_version",
    "pair_role",
    "stage",
    "maximum_output_absolute_error",
    "maximum_output_relative_error",
    "maximum_jvp_absolute_error",
    "maximum_jvp_relative_error",
    "weighted_total_absolute_error",
    "jvp_pair_count",
    "concatenated_metrics_sha256",
    "concatenated_executor_sha256",
    "flags",
    "passed",
    "artifact_sha256",
}
_EXECUTION_ACCOUNTING_FIELDS = {
    "fit_panel_valid_rows",
    "fit_panel_core_mac_count",
    "fit_panel_encoder_mac_count",
    "fit_panel_decoder_mac_count",
    "fit_panel_target_destandardization_mac_count",
    "fit_panel_total_mac_count",
    "macs_per_valid_row_over_fit_panel",
    "canonical_sequence_length",
    "canonical_batch_size",
    "canonical_total_mac_count",
    "canonical_core_mac_count",
    "canonical_encoder_mac_count",
    "canonical_decoder_mac_count",
    "canonical_target_destandardization_mac_count",
    "semantics",
}


def _recompute_decision_from_rows(
    rows: Mapping[str, object],
) -> dict[str, object]:
    def pair(role: str) -> tuple[bool, str] | None:
        control = rows.get(f"fp_expert_count.{role}.expert2")
        treatment = rows.get(f"fp_expert_count.{role}.expert4")
        if control is None and treatment is None:
            return None
        if not isinstance(control, Mapping) or not isinstance(
            treatment,
            Mapping,
        ):
            raise ValueError("expert-count paired rows are incomplete")
        left = control.get("pair_treatment_validity")
        right = treatment.get("pair_treatment_validity")
        if (
            not isinstance(left, Mapping)
            or not isinstance(right, Mapping)
            or set(left) != _VALIDITY_FIELDS
            or _canonical_json_bytes(left) != _canonical_json_bytes(right)
            or left.get("failure_semantics")
            != (
                "invalid_paired_expert_count_comparison_no_capacity_"
                "conclusion"
            )
        ):
            raise ValueError("expert-count treatment validity drifted")
        initial = control.get("initialization_equivalence")
        gradient = treatment.get("gradient_openness")
        parity = treatment.get("postfit_wrapper_concat_parity")
        for name, value in (
            ("initialization", initial),
            ("gradient", gradient),
            ("parity", parity),
            ("control balance", control.get("objective_balance_gate")),
            ("treatment balance", treatment.get("objective_balance_gate")),
            ("control sequences", control.get("source_sequence_comparison")),
            (
                "treatment sequences",
                treatment.get("source_sequence_comparison"),
            ),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"expert-count {name} must be a mapping")
        assert isinstance(initial, Mapping)
        assert isinstance(gradient, Mapping)
        assert isinstance(parity, Mapping)
        expected_flags = {
            "initial_observable_and_jvp_equivalence": (
                initial.get("passed") is True
                and _canonical_json_bytes(initial)
                == _canonical_json_bytes(
                    treatment.get("initialization_equivalence")
                )
            ),
            "gradient_open_expert4": gradient.get("passed") is True,
            "postfit_wrapper_concat_parity": parity.get("passed") is True,
            "expert2_balance": (
                control["objective_balance_gate"].get("passed") is True  # type: ignore[union-attr]
            ),
            "expert4_balance": (
                treatment["objective_balance_gate"].get("passed") is True  # type: ignore[union-attr]
            ),
            "expert2_source_sequences": (
                control["source_sequence_comparison"].get("passed") is True  # type: ignore[union-attr]
            ),
            "expert4_source_sequences": (
                treatment["source_sequence_comparison"].get("passed") is True  # type: ignore[union-attr]
            ),
            "expert2_primary_replay": (
                control.get("source_replay_exact") is (role == "primary")
                and treatment.get("source_replay_exact") is False
            ),
            "fixed_outer_encoder_decoder_router": True,
            "exact_training_contract": True,
        }
        supplied = left.get("flags")
        if (
            not isinstance(supplied, Mapping)
            or dict(supplied) != expected_flags
        ):
            raise ValueError("expert-count treatment flags drifted")
        valid = all(expected_flags.values())
        if left.get("passed") is not valid:
            raise ValueError("expert-count validity decision drifted")
        status = _pair_status(
            control.get("fit_capability_pass") is True,
            treatment.get("fit_capability_pass") is True,
        )
        if (
            control.get("pair_comparison_status") != status
            or treatment.get("pair_comparison_status") != status
        ):
            raise ValueError("expert-count pair status drifted")
        return valid, status

    primary = pair("primary")
    replication = pair("replication")
    if primary is None:
        raise ValueError("expert-count primary pair is absent")
    primary_valid, primary_status = primary
    if replication is not None and not (
        primary_valid
        and primary_status == "expert2_fail_expert4_pass"
    ):
        raise ValueError("expert-count replication was unauthorized")
    if not primary_valid:
        outcome = "invalid_primary_pair"
    elif primary_status != "expert2_fail_expert4_pass":
        outcome = f"primary_{primary_status}"
    elif replication is None:
        raise ValueError("authorized expert-count replication is absent")
    elif not replication[0]:
        outcome = "invalid_replication_pair"
    elif replication[1] == "expert2_fail_expert4_pass":
        outcome = "two_seed_routed_expert_count_support"
    else:
        outcome = f"inconsistent_replication_{replication[1]}"
    return {
        "outcome": outcome,
        "primary_treatment_valid": primary_valid,
        "primary_comparison_status": primary_status,
        "replication_executed": replication is not None,
        "replication_treatment_valid": (
            None if replication is None else replication[0]
        ),
        "replication_comparison_status": (
            None if replication is None else replication[1]
        ),
        "two_seed_routed_expert_count_supported": (
            outcome == "two_seed_routed_expert_count_support"
        ),
        "descending_expert_count_ladder_authorized": (
            outcome == "two_seed_routed_expert_count_support"
        ),
        "e8_expert_count_control_authorized": (
            outcome == "primary_both_fail"
        ),
        "fresh_c3_authorized": False,
        "fresh_c3_after_descending_ladder_only": (
            outcome == "two_seed_routed_expert_count_support"
        ),
        "compression_claim_authorized": False,
        "speed_claim_authorized": False,
    }


def _validate_hashed_audit(
    audit: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(audit, Mapping) or set(audit) != fields:
        raise ValueError(f"expert-count {label} schema drifted")
    payload = dict(audit)
    supplied = payload.pop("artifact_sha256", None)
    if supplied != _json_sha256(payload, domain=_AUDIT_DOMAIN):
        raise ValueError(f"expert-count {label} hash drifted")
    return audit


def _reconstruct_initial_bindings(
    protocol: FunctionPreservingExpertCountControlProtocol,
    *,
    pair_role: str,
) -> tuple[dict[str, str], dict[str, str]]:
    if pair_role not in {"primary", "replication"}:
        raise ValueError("initial binding role is invalid")
    seed = (
        protocol.training.primary_seed
        if pair_role == "primary"
        else protocol.training.replication_seed
    )
    modal_center = torch.zeros(64, dtype=torch.float64)
    target_center = torch.zeros(64, dtype=torch.float64)
    target_scale = torch.ones(64, dtype=torch.float64)
    control = _new_control_model(
        modal_center=modal_center,
        gain_log_center=0.0,
        gain_log_scale=1.0,
        residual_width=64,
        rms_epsilon=1e-6,
        target_center=target_center,
        target_scale=target_scale,
        seed=seed,
    )
    treatment = _new_treatment_model(
        control=control,
        protocol=protocol,
        seed=seed,
    )
    return _parameter_bindings(control), _parameter_bindings(treatment)


def _initial_audit_contract(
    protocol: FunctionPreservingExpertCountControlProtocol,
    *,
    pair_role: str,
    control_initial_metrics_sha256: str,
    treatment_initial_metrics_sha256: str,
) -> dict[str, object]:
    frozen = _preflight_binding_for_role(protocol, pair_role)
    seed = int(frozen["seed"])
    values = frozen["initial_equivalence"]
    if not isinstance(values, Mapping):
        raise TypeError("frozen initial equivalence must be a mapping")
    control, treatment = _reconstruct_initial_bindings(
        protocol,
        pair_role=pair_role,
    )
    state = {
        "artifact_kind": "fisher_graph.initial_expert_count_equivalence",
        "format_version": _FORMAT_VERSION,
        "pair_role": pair_role,
        "seed": seed,
        **dict(values),
        "initial_weighted_total_absolute_error": 0.0,
        "jvp_pair_count": 32,
        "control_initial_metrics_sha256": (
            control_initial_metrics_sha256
        ),
        "treatment_initial_metrics_sha256": (
            treatment_initial_metrics_sha256
        ),
        "control_initial_parameter_bindings": control,
        "treatment_initial_parameter_bindings": treatment,
        "flags": {
            "observable_absolute": True,
            "observable_relative": True,
            "jvp_absolute": True,
            "jvp_relative": True,
            "wrapper_concat_observable_absolute": True,
            "wrapper_concat_observable_relative": True,
            "wrapper_concat_jvp_absolute": True,
            "wrapper_concat_jvp_relative": True,
            "initial_weighted_total_absolute": True,
            "parent_route_mass_absolute": True,
            "parent_route_mass_relative": True,
            "sibling_route_probabilities_equal": True,
            "source_probabilities_equal": True,
            "causal_masks_exact": True,
            "outside_edge_routes_zero": True,
            "allowed_edge_route_mass_one": True,
            "dormant_lift_parameter_identity": True,
            "all_expected_jvps_compared": True,
        },
        "passed": True,
    }
    state["artifact_sha256"] = _json_sha256(
        state,
        domain=_AUDIT_DOMAIN,
    )
    if (
        state["artifact_sha256"]
        != frozen["initialization_audit_sha256"]
    ):
        raise ValueError("frozen initial audit reconstruction drifted")
    return state


def _validate_initial_audit(
    audit: object,
    *,
    protocol: FunctionPreservingExpertCountControlProtocol,
    pair_role: str,
) -> Mapping[str, object]:
    state = _validate_hashed_audit(
        audit,
        fields=_INITIAL_FIELDS,
        label="initial equivalence",
    )
    seed = (
        protocol.training.primary_seed
        if pair_role == "primary"
        else protocol.training.replication_seed
    )
    numeric_fields = (
        "maximum_observable_absolute_error",
        "maximum_observable_relative_error",
        "maximum_jvp_absolute_error",
        "maximum_jvp_relative_error",
        "maximum_wrapper_concat_observable_absolute_error",
        "maximum_wrapper_concat_observable_relative_error",
        "maximum_wrapper_concat_jvp_absolute_error",
        "maximum_wrapper_concat_jvp_relative_error",
        "initial_weighted_total_absolute_error",
        "maximum_parent_route_mass_absolute_error",
        "maximum_parent_route_mass_relative_error",
        "maximum_sibling_route_absolute_error",
        "maximum_source_probability_absolute_error",
        "maximum_allowed_edge_route_sum_absolute_error",
    )
    flags = state.get("flags")
    lift_flags = state.get("lift_parameter_flags")
    if (
        state.get("artifact_kind")
        != "fisher_graph.initial_expert_count_equivalence"
        or type(state.get("format_version")) is not int
        or state.get("format_version") != _FORMAT_VERSION
        or state.get("pair_role") != pair_role
        or type(state.get("seed")) is not int
        or state.get("seed") != seed
        or type(state.get("jvp_pair_count")) is not int
        or state.get("jvp_pair_count") != 32
        or any(
            type(state.get(name)) is not float
            or not math.isfinite(state[name])
            or state[name] < 0.0
            for name in numeric_fields
        )
        or not isinstance(flags, Mapping)
        or any(type(value) is not bool for value in flags.values())
        or not isinstance(lift_flags, Mapping)
        or not lift_flags
        or any(type(value) is not bool for value in lift_flags.values())
        or type(state.get("causal_masks_exact")) is not bool
        or type(state.get("outside_edge_routes_zero")) is not bool
        or type(state.get("passed")) is not bool
    ):
        raise ValueError("expert-count initial semantics drifted")
    tolerance = protocol.lift.equivalence_absolute_tolerance
    relative = protocol.lift.equivalence_relative_tolerance
    expected_flags = {
        "observable_absolute": (
            state["maximum_observable_absolute_error"] <= tolerance
        ),
        "observable_relative": (
            state["maximum_observable_relative_error"] <= relative
        ),
        "jvp_absolute": state["maximum_jvp_absolute_error"] <= tolerance,
        "jvp_relative": state["maximum_jvp_relative_error"] <= relative,
        "wrapper_concat_observable_absolute": (
            state["maximum_wrapper_concat_observable_absolute_error"]
            <= tolerance
        ),
        "wrapper_concat_observable_relative": (
            state["maximum_wrapper_concat_observable_relative_error"]
            <= relative
        ),
        "wrapper_concat_jvp_absolute": (
            state["maximum_wrapper_concat_jvp_absolute_error"]
            <= tolerance
        ),
        "wrapper_concat_jvp_relative": (
            state["maximum_wrapper_concat_jvp_relative_error"]
            <= relative
        ),
        "initial_weighted_total_absolute": (
            state["initial_weighted_total_absolute_error"] <= tolerance
        ),
        "parent_route_mass_absolute": (
            state["maximum_parent_route_mass_absolute_error"] <= tolerance
        ),
        "parent_route_mass_relative": (
            state["maximum_parent_route_mass_relative_error"] <= relative
        ),
        "sibling_route_probabilities_equal": (
            state["maximum_sibling_route_absolute_error"] <= tolerance
        ),
        "source_probabilities_equal": (
            state["maximum_source_probability_absolute_error"] <= tolerance
        ),
        "causal_masks_exact": state["causal_masks_exact"] is True,
        "outside_edge_routes_zero": (
            state["outside_edge_routes_zero"] is True
        ),
        "allowed_edge_route_mass_one": (
            state["maximum_allowed_edge_route_sum_absolute_error"]
            <= tolerance
        ),
        "dormant_lift_parameter_identity": all(lift_flags.values()),
        "all_expected_jvps_compared": state["jvp_pair_count"] == 32,
    }
    frozen = _preflight_binding_for_role(protocol, pair_role)
    control, treatment = _reconstruct_initial_bindings(
        protocol,
        pair_role=pair_role,
    )
    if (
        dict(flags) != expected_flags
        or state.get("passed") is not all(expected_flags.values())
        or state.get("artifact_sha256")
        != frozen["initialization_audit_sha256"]
        or _canonical_json_bytes(
            state.get("control_initial_parameter_bindings")
        )
        != _canonical_json_bytes(control)
        or _canonical_json_bytes(
            state.get("treatment_initial_parameter_bindings")
        )
        != _canonical_json_bytes(treatment)
    ):
        raise ValueError("expert-count initial binding drifted")
    return state


def _validate_gradient_audit(
    audit: object,
    *,
    protocol: FunctionPreservingExpertCountControlProtocol,
    pair_role: str,
    arm: str,
) -> Mapping[str, object]:
    state = _validate_hashed_audit(
        audit,
        fields=_GRADIENT_FIELDS,
        label="gradient openness",
    )
    applicable = arm == "expert4"
    value_fields = {
        name
        for name in _GRADIENT_FIELDS
        if name.startswith("step")
    }
    frozen_values = _preflight_binding_for_role(
        protocol,
        pair_role,
    )["treatment_gradient"]
    if not isinstance(frozen_values, Mapping) or set(
        frozen_values
    ) != value_fields:
        raise ValueError("frozen expert-count gradient fields drifted")
    expected = _gradient_audit(
        applicable=applicable,
        pair_role=pair_role,
        protocol=protocol,
        values=(
            dict(frozen_values)
            if applicable
            else {name: 0.0 for name in value_fields}
        ),
    )
    if _canonical_json_bytes(state) != _canonical_json_bytes(expected):
        raise ValueError("expert-count gradient binding drifted")
    return state


def _validate_postfit_parity(
    audit: object,
    *,
    protocol: FunctionPreservingExpertCountControlProtocol,
    pair_role: str,
    arm: str,
    plan: ContrastAwareReferenceProviderPlan,
) -> Mapping[str, object]:
    if arm == "expert2":
        expected = {
            "artifact_kind": (
                "fisher_graph.expert_count_wrapper_concat_parity"
            ),
            "format_version": _FORMAT_VERSION,
            "pair_role": pair_role,
            "stage": "not_applicable_control",
            "applicable": False,
            "passed": True,
        }
        expected["artifact_sha256"] = _json_sha256(
            expected,
            domain=_AUDIT_DOMAIN,
        )
        if _canonical_json_bytes(audit) != _canonical_json_bytes(expected):
            raise ValueError("expert-count control parity drifted")
        return expected
    state = _validate_hashed_audit(
        audit,
        fields=_PARITY_FIELDS,
        label="postfit parity",
    )
    numeric = (
        "maximum_output_absolute_error",
        "maximum_output_relative_error",
        "maximum_jvp_absolute_error",
        "maximum_jvp_relative_error",
        "weighted_total_absolute_error",
    )
    flags = state.get("flags")
    if (
        state.get("artifact_kind")
        != "fisher_graph.expert_count_wrapper_concat_parity"
        or state.get("format_version") != _FORMAT_VERSION
        or type(state.get("format_version")) is not int
        or state.get("pair_role") != pair_role
        or state.get("stage") != "post_fit"
        or type(state.get("jvp_pair_count")) is not int
        or state.get("jvp_pair_count") != 32
        or any(
            type(state.get(name)) is not float
            or not math.isfinite(state[name])
            or state[name] < 0.0
            for name in numeric
        )
        or not isinstance(flags, Mapping)
        or any(type(value) is not bool for value in flags.values())
    ):
        raise ValueError("expert-count postfit parity semantics drifted")
    tolerance = protocol.lift.equivalence_absolute_tolerance
    relative = protocol.lift.equivalence_relative_tolerance
    expected_flags = {
        "output_absolute": (
            state["maximum_output_absolute_error"] <= tolerance
        ),
        "output_relative": (
            state["maximum_output_relative_error"] <= relative
        ),
        "jvp_absolute": state["maximum_jvp_absolute_error"] <= tolerance,
        "jvp_relative": state["maximum_jvp_relative_error"] <= relative,
        "weighted_total_absolute": (
            state["weighted_total_absolute_error"] <= tolerance
        ),
        "all_expected_jvps_compared": state["jvp_pair_count"] == 32,
    }
    if (
        dict(flags) != expected_flags
        or state.get("passed") is not all(expected_flags.values())
        or state.get("concatenated_executor_sha256")
        != contrast_fit._executor_artifact_sha256(plan.executor_artifact)
        or state.get("concatenated_metrics_sha256")
        != plan.final_metrics.artifact_sha256
    ):
        raise ValueError("expert-count postfit parity binding drifted")
    return state


def load_function_preserving_expert_count_control_artifact(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_tensor_file_sha256: str,
    expected_report_sha256: str,
) -> LoadedFunctionPreservingExpertCountControlArtifact:
    """Strictly load, authenticate, and recompute an expert-count result."""

    source = Path(path).expanduser().resolve()
    if source.suffix != ".pt":
        raise ValueError("expert-count artifact must use a .pt suffix")
    tensor_payload = _read_regular(source, label="expert-count artifact")
    report_path = source.with_suffix(".json")
    report_payload = _read_regular(report_path, label="expert-count report")
    tensor_sha256 = hashlib.sha256(tensor_payload).hexdigest()
    trusted_artifact = _require_sha256(
        expected_artifact_sha256,
        label="expected logical artifact SHA-256",
    )
    trusted_tensor = _require_sha256(
        expected_tensor_file_sha256,
        label="expected tensor-file SHA-256",
    )
    trusted_report = _require_sha256(
        expected_report_sha256,
        label="expected report SHA-256",
    )
    if tensor_sha256 != trusted_tensor:
        raise ValueError(
            "expert-count external trust anchor mismatch: tensor file"
        )
    try:
        report = json.loads(report_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("expert-count report is invalid JSON") from exc
    if not isinstance(report, Mapping):
        raise TypeError("expert-count report must be a mapping")
    d0d3._assert_safe_artifact_tree(report, path="report")
    d0d3._assert_tensor_free_report(report)
    without_hash = dict(report)
    supplied_report_sha = without_hash.pop("report_sha256", None)
    computed_report_sha = _json_sha256(
        without_hash,
        domain=_REPORT_DOMAIN,
    )
    artifact = report.get("artifact")
    if (
        computed_report_sha != trusted_report
        or supplied_report_sha != computed_report_sha
        or report.get("artifact_sha256") != trusted_artifact
        or not isinstance(artifact, Mapping)
        or set(artifact)
        != {
            "tensor_file",
            "tensor_file_sha256",
            "tensor_file_bytes",
            "report_file",
            "committable",
        }
        or artifact.get("tensor_file") != str(source)
        or artifact.get("report_file") != str(report_path)
        or artifact.get("tensor_file_sha256") != tensor_sha256
        or artifact.get("tensor_file_bytes") != len(tensor_payload)
        or artifact.get("committable") is not False
    ):
        raise ValueError(
            "expert-count external trust anchor or report binding mismatch"
        )
    _require_canonical_torch_zip_framing(tensor_payload)
    raw = torch.load(
        io.BytesIO(tensor_payload),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw, Mapping):
        raise ValueError("expert-count tensor fields drifted")
    if _canonical_torch_payload(raw) != tensor_payload:
        raise ValueError(
            "expert-count tensor file is not the canonical serialization "
            "of its consumed state"
        )
    if set(raw) != _STATE_FIELDS:
        raise ValueError("expert-count tensor fields drifted")
    _assert_safe_expert_count_artifact_tree(raw)
    manifest = raw["manifest"]
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("expert-count manifest fields drifted")
    logical = _json_sha256(manifest, domain=_ARTIFACT_DOMAIN)
    if (
        raw.get("artifact_sha256") != logical
        or manifest.get("schema") != _SCHEMA
        or manifest.get("format_version") != _FORMAT_VERSION
        or logical != trusted_artifact
    ):
        raise ValueError("expert-count logical binding drifted")
    if report.get("artifact_sha256") != logical:
        raise ValueError("expert-count report binding drifted")
    if set(report) != _MANIFEST_FIELDS | _REPORT_EXTRA_FIELDS:
        raise ValueError("expert-count report fields drifted")
    for name, value in manifest.items():
        if _canonical_json_bytes(report.get(name)) != _canonical_json_bytes(
            value
        ):
            raise ValueError(f"expert-count report field {name!r} drifted")
    code = manifest.get("code_sha256s")
    if (
        not isinstance(code, Mapping)
        or dict(code) != _code_sha256s()
        or manifest.get("code_bundle_sha256")
        != _code_bundle_sha256(dict(code))
    ):
        raise ValueError("expert-count code binding drifted")
    protocol = FunctionPreservingExpertCountControlProtocol.from_state_dict(
        raw["protocol_state"]
    )
    if (
        protocol.protocol_sha256
        != DEFAULT_FUNCTION_PRESERVING_EXPERT_COUNT_CONTROL_PROTOCOL_SHA256
        or manifest.get("protocol_sha256") != protocol.protocol_sha256
        or _canonical_json_bytes(report.get("protocol"))
        != _canonical_json_bytes(protocol.state_dict())
    ):
        raise ValueError("expert-count protocol binding drifted")
    measurement_sha = _measurement_evidence_sha256(report)
    if (
        measurement_sha != protocol.training.measurement_evidence_sha256
        or manifest.get("measurement_evidence_sha256") != measurement_sha
    ):
        raise ValueError("expert-count measurement evidence drifted")
    source_firewall = {
        "source_expert_rank_protocol_sha256": (
            protocol.source.expert_rank_protocol_sha256
        ),
        "source_expert_rank_code_bundle_sha256": (
            protocol.source.expert_rank_code_bundle_sha256
        ),
        "source_expert_rank_logical_artifact_sha256": (
            protocol.source.expert_rank_logical_artifact_sha256
        ),
        "source_expert_rank_tensor_file_sha256": (
            protocol.source.expert_rank_tensor_file_sha256
        ),
        "source_expert_rank_report_sha256": (
            protocol.source.expert_rank_report_sha256
        ),
        "source_expert_rank_primary_e2r64_plan_sha256": (
            protocol.source.expert_rank_primary_e2r64_plan_sha256
        ),
        "source_expert_rank_primary_e2r64_result_sha256": (
            protocol.source.expert_rank_primary_e2r64_result_sha256
        ),
    }
    if any(
        manifest.get(name) != value
        for name, value in source_firewall.items()
    ):
        raise ValueError("expert-count predecessor binding drifted")

    calibration = d0d3._restore_calibration_binding(
        raw["calibration_state"]
    )
    controls = d0d3._restore_full_width_controls(raw["controls_state"])
    gauge = UnitRmsFisherGauge.from_state_dict(
        raw["unit_rms_gauge_state"]
    )
    metric = raw["canonical_metric_weight"]
    if not isinstance(metric, Tensor):
        raise TypeError("expert-count canonical metric is not a tensor")
    gauge.validate_source(metric)
    c2_protocol = default_contrast_provider_development_protocol()
    ordinary_gates = _deferred_collision_gates(
        SyntheticReferenceGates()
    )
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
        "basis_package_file_sha256": (
            DEFAULT_BASIS_PACKAGE_FILE_SHA256
        ),
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
        "authenticated_expert_rank_source_loaded": True,
        "source_final_parameters_used_for_initialization": False,
        "fit_split_wrapper_used": True,
        "published_plans_use_concatenated_executor": True,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "scientific_scope": (
            "fit_only_paired_function_preserving_expert_count_control"
        ),
    }
    if (
        calibration.artifact_sha256
        != manifest.get("c2_calibration_sha256")
        or controls.artifact_sha256 != manifest.get("controls_sha256")
        or gauge.artifact_sha256
        != manifest.get("unit_rms_gauge_sha256")
        or d0d3._tensor_sha256(metric)
        != manifest.get("canonical_metric_weight_sha256")
        or any(
            _canonical_json_bytes(manifest.get(name))
            != _canonical_json_bytes(expected)
            for name, expected in manifest_firewall.items()
        )
    ):
        raise ValueError("expert-count manifest firewall drifted")

    batch_states = raw["ordinary_scoring_batch_states"]
    batch_hashes = manifest.get(
        "ordinary_scoring_indexed_batch_sha256s"
    )
    if (
        type(batch_states) is not tuple
        or not batch_states
        or any(not isinstance(value, Mapping) for value in batch_states)
        or type(batch_hashes) is not tuple
    ):
        raise ValueError("expert-count ordinary scoring batches drifted")
    ordinary_scoring_batches = tuple(
        IndexedReferenceBatch.from_state_dict(value)
        for value in batch_states
    )
    ordered_ordinary_scoring_batches = (
        _ordered_ordinary_scoring_batches(ordinary_scoring_batches)
    )
    ordinary_scoring_hashes = tuple(
        value.artifact_sha256
        for value in ordinary_scoring_batches
    )
    if (
        ordinary_scoring_hashes
        != tuple(
            value.artifact_sha256
            for value in ordered_ordinary_scoring_batches
        )
        or ordinary_scoring_hashes != batch_hashes
    ):
        raise ValueError(
            "expert-count ordinary scoring batch order drifted"
        )
    pair_states = raw["fit_contrast_pair_states"]
    pair_hashes = manifest.get("fit_contrast_pair_sha256s")
    if (
        type(pair_states) is not tuple
        or not pair_states
        or any(not isinstance(value, Mapping) for value in pair_states)
        or type(pair_hashes) is not tuple
    ):
        raise ValueError("expert-count fit contrast pairs drifted")
    fit_contrast_pairs = tuple(
        ReferenceProviderContrastPair.from_state_dict(value)
        for value in pair_states
    )
    fit_contrast_pair_hashes = tuple(
        value.artifact_sha256 for value in fit_contrast_pairs
    )
    if (
        tuple(value.pair_id for value in fit_contrast_pairs)
        != tuple(
            sorted(value.pair_id for value in fit_contrast_pairs)
        )
        or len({value.pair_id for value in fit_contrast_pairs})
        != len(fit_contrast_pairs)
        or fit_contrast_pair_hashes != pair_hashes
    ):
        raise ValueError("expert-count fit contrast pair order drifted")
    replay_sequences = {
        "fit_batch_sha256s": tuple(
            value.batch.artifact_sha256
            for value in ordinary_scoring_batches
        ),
        "fit_batch_content_sha256s": tuple(
            value.batch.content_sha256
            for value in ordinary_scoring_batches
        ),
        "fit_indexed_batch_sha256s": ordinary_scoring_hashes,
        "fit_pair_sha256s": fit_contrast_pair_hashes,
    }
    if any(
        getattr(
            protocol.training,
            f"{name}_sequence_sha256",
        )
        != expert_protocol_module.fit_replay_sequence_sha256(
            name,
            values,
        )
        for name, values in replay_sequences.items()
    ):
        raise ValueError(
            "expert-count fit replay protocol commitment drifted"
        )
    ordinary_endpoint_locations = (
        _ordinary_scoring_endpoint_locations(
            ordinary_scoring_batches
        )
    )
    ordinary_probes = _ordinary_scoring_probes(
        batches=ordinary_scoring_batches,
        controls=controls,
        metric_weight=metric,
        c2_protocol=c2_protocol,
    )
    restored_fit_endpoints = _restored_fit_endpoints(
        batches=ordinary_scoring_batches,
        c2_protocol=c2_protocol,
    )
    recomputed_controls = fit_full_width_reference_controls(
        fit_probes=ordinary_probes,
        position_bin_count=controls.position_bin_count,
    )
    if (
        recomputed_controls.artifact_sha256
        != controls.artifact_sha256
        or not torch.equal(
            recomputed_controls.fit_target_center,
            controls.fit_target_center,
        )
        or not torch.equal(
            recomputed_controls.normalized_position_bin_centers,
            controls.normalized_position_bin_centers,
        )
    ):
        raise ValueError("expert-count ordinary controls replay drifted")

    plans = raw["plan_states"]
    ordinary_candidates = raw["ordinary_candidate_states"]
    rows = raw["candidate_results"]
    executed = manifest.get("executed_candidate_ids")
    plan_hashes = manifest.get("candidate_plan_sha256s")
    ordinary_candidate_hashes = manifest.get(
        "ordinary_candidate_sha256s"
    )
    result_hashes = manifest.get("candidate_result_sha256s")
    if (
        not isinstance(plans, Mapping)
        or not isinstance(ordinary_candidates, Mapping)
        or not isinstance(rows, Mapping)
        or not isinstance(executed, (tuple, list))
        or not isinstance(plan_hashes, Mapping)
        or not isinstance(ordinary_candidate_hashes, Mapping)
        or not isinstance(result_hashes, Mapping)
        or set(plans) != set(executed)
        or set(ordinary_candidates) != set(executed)
        or set(rows) != set(executed)
        or set(plan_hashes) != set(executed)
        or set(ordinary_candidate_hashes) != set(executed)
        or set(result_hashes) != set(executed)
    ):
        raise ValueError("expert-count candidate tables drifted")
    allowed_orders = (
        (
            "fp_expert_count.primary.expert2",
            "fp_expert_count.primary.expert4",
        ),
        (
            "fp_expert_count.primary.expert2",
            "fp_expert_count.primary.expert4",
            "fp_expert_count.replication.expert2",
            "fp_expert_count.replication.expert4",
        ),
    )
    if tuple(executed) not in allowed_orders:
        raise ValueError("expert-count candidate order drifted")
    canonical_state = plans.get("fp_expert_count.primary.expert2")
    if not isinstance(canonical_state, Mapping):
        raise ValueError("expert-count canonical control is absent")
    canonical = ContrastAwareReferenceProviderPlan.from_state_dict(
        canonical_state
    )
    if (
        canonical.artifact_sha256
        != protocol.source.expert_rank_primary_e2r64_plan_sha256
        or canonical.synthetic_binding_sha256
        != protocol.training.fit_data_binding_sha256
        or not torch.equal(
            canonical.fisher_metric_weight,
            gauge.metric_weight,
        )
        or canonical.fit_batch_sha256s
        != tuple(
            value.batch.artifact_sha256
            for value in ordinary_scoring_batches
        )
        or canonical.fit_batch_content_sha256s
        != tuple(
            value.batch.content_sha256
            for value in ordinary_scoring_batches
        )
        or canonical.fit_indexed_batch_sha256s
        != tuple(
            value.artifact_sha256
            for value in ordinary_scoring_batches
        )
        or canonical.fit_endpoint_sha256s
        != tuple(
            location[2]
            for _, location in sorted(
                ordinary_endpoint_locations.items()
            )
        )
        or canonical.fit_pair_sha256s
        != fit_contrast_pair_hashes
    ):
        raise ValueError("expert-count canonical source replay drifted")
    objective_protocol = default_objective_balance_diagnostic_protocol()
    d3_recipe = objective_protocol.recipe(protocol.training.recipe_id)
    report_gauge = report.get("gauge")
    training_signal = report.get(
        "training_teacher_signal_diagnostics"
    )
    if not isinstance(report_gauge, Mapping) or not isinstance(
        training_signal,
        Mapping,
    ):
        raise TypeError("expert-count balance evidence is absent")
    if (
        report_gauge.get("target_center_sha256")
        != d0d3._tensor_sha256(canonical.target_center)
        or report_gauge.get("target_scale_sha256")
        != d0d3._tensor_sha256(canonical.target_scale)
    ):
        raise ValueError("expert-count target geometry drifted")

    for candidate_id in executed:
        state = plans[candidate_id]
        row = rows[candidate_id]
        if not isinstance(state, Mapping) or not isinstance(row, Mapping):
            raise TypeError("expert-count candidate entry is invalid")
        if set(row) != _ROW_FIELDS:
            raise ValueError("expert-count candidate fields drifted")
        plan = ContrastAwareReferenceProviderPlan.from_state_dict(state)
        plan.validate_integrity()
        ordinary_candidate = _restore_ordinary_candidate(
            ordinary_candidates[candidate_id]
        )
        arm = row.get("arm")
        role = row.get("pair_role")
        expected_config = (
            _executor_config(protocol.e2_executor)
            if arm == "expert2"
            else _executor_config(protocol.e4_executor)
        )
        seed = (
            protocol.training.primary_seed
            if role == "primary"
            else protocol.training.replication_seed
        )
        if (
            role not in {"primary", "replication"}
            or arm not in {"expert2", "expert4"}
            or candidate_id != f"fp_expert_count.{role}.{arm}"
            or plan.artifact_sha256 != plan_hashes[candidate_id]
            or ordinary_candidate.artifact_sha256
            != ordinary_candidate_hashes[candidate_id]
            or row.get("plan_sha256") != plan.artifact_sha256
            or row.get("candidate_id") != candidate_id
            or asdict(plan.executor_config) != asdict(expected_config)
            or plan.latent_rank != 64
            or row.get("outer_rank") != 64
            or row.get("expert_count") != expected_config.expert_count
            or row.get("expert_rank") != expected_config.expert_rank
            or row.get("seed") != plan.seed
            or plan.seed != seed
            or plan.training_steps != protocol.training.steps
            or plan.learning_rate != protocol.training.learning_rate
            or plan.objective.artifact_sha256
            != _objective(protocol).artifact_sha256
            or _canonical_json_bytes(row.get("initial_training_metrics"))
            != _canonical_json_bytes(plan.initial_metrics.state_dict())
            or _canonical_json_bytes(row.get("final_training_metrics"))
            != _canonical_json_bytes(plan.final_metrics.state_dict())
            or result_hashes[candidate_id]
            != _json_sha256(row, domain=_RESULT_DOMAIN)
        ):
            raise ValueError("expert-count candidate plan/result drifted")
        if (
            plan.synthetic_binding_sha256
            != protocol.training.fit_data_binding_sha256
            or plan.fisher_metric_supplied is not True
            or not torch.equal(plan.fisher_metric_weight, gauge.metric_weight)
            or not torch.equal(plan.modal_center, canonical.modal_center)
            or plan.gain_log_center != canonical.gain_log_center
            or plan.gain_log_scale != canonical.gain_log_scale
            or plan.residual_width != canonical.residual_width
            or plan.rms_epsilon != canonical.rms_epsilon
            or not torch.equal(plan.target_center, canonical.target_center)
            or not torch.equal(plan.target_scale, canonical.target_scale)
            or plan.fit_batch_sha256s
            != canonical.fit_batch_sha256s
            or plan.fit_batch_content_sha256s
            != canonical.fit_batch_content_sha256s
            or plan.fit_indexed_batch_sha256s
            != canonical.fit_indexed_batch_sha256s
            or plan.fit_endpoint_sha256s
            != canonical.fit_endpoint_sha256s
            or plan.fit_pair_sha256s
            != canonical.fit_pair_sha256s
        ):
            raise ValueError("expert-count shared fit geometry drifted")
        if row.get("source_replay_exact") is not _source_replay_exact(
            pair_role=str(role),
            arm=str(arm),
            plan=plan,
            protocol=protocol,
        ):
            raise ValueError("expert-count source replay drifted")
        initial = _validate_initial_audit(
            row.get("initialization_equivalence"),
            protocol=protocol,
            pair_role=str(role),
        )
        paired_initial = rows.get(
            f"fp_expert_count.{role}."
            f"{'expert4' if arm == 'expert2' else 'expert2'}"
        )
        if (
            not isinstance(paired_initial, Mapping)
            or _canonical_json_bytes(initial)
            != _canonical_json_bytes(
                paired_initial.get("initialization_equivalence")
            )
        ):
            raise ValueError("expert-count paired initialization drifted")
        control_state = plans.get(f"fp_expert_count.{role}.expert2")
        treatment_state = plans.get(f"fp_expert_count.{role}.expert4")
        if not isinstance(control_state, Mapping) or not isinstance(
            treatment_state,
            Mapping,
        ):
            raise ValueError("expert-count paired plans are absent")
        control_initial_hash = (
            ContrastAwareReferenceProviderPlan.from_state_dict(
                control_state
            ).initial_metrics.artifact_sha256
        )
        treatment_initial_hash = (
            ContrastAwareReferenceProviderPlan.from_state_dict(
                treatment_state
            ).initial_metrics.artifact_sha256
        )
        if (
            initial.get("control_initial_metrics_sha256")
            != control_initial_hash
            or initial.get("treatment_initial_metrics_sha256")
            != treatment_initial_hash
        ):
            raise ValueError("expert-count initial metrics binding drifted")
        _validate_gradient_audit(
            row.get("gradient_openness"),
            protocol=protocol,
            pair_role=str(role),
            arm=str(arm),
        )
        _validate_postfit_parity(
            row.get("postfit_wrapper_concat_parity"),
            protocol=protocol,
            pair_role=str(role),
            arm=str(arm),
            plan=plan,
        )
        structural = row.get("structural_metadata")
        if (
            not isinstance(structural, Mapping)
            or set(structural) != _STRUCTURAL_FIELDS
            or structural.get("support_rule")
            != (
                "fit_max_l2_radius_of_encoded_nonconstant_features_plus_"
                "margin"
            )
            or type(structural.get("support_radius")) is not float
            or not math.isfinite(structural["support_radius"])
            or structural["support_radius"] < 0.0
            or type(structural.get("invalid_padding_rows_tested")) is not int
            or structural["invalid_padding_rows_tested"] <= 0
            or structural.get("nonvacuous_padding_test") is not True
        ):
            raise ValueError("expert-count structural metadata drifted")
        sequence = row.get("source_sequence_comparison")
        if (
            not isinstance(sequence, Mapping)
            or set(sequence) != _SEQUENCE_FIELDS
            or sequence.get("comparison_semantics")
            != (
                "exact_ordered_batch_content_index_endpoint_and_pair_hashes"
            )
        ):
            raise ValueError("expert-count source sequence schema drifted")
        candidate_sequences = sequence.get("candidate_sequence_sha256s")
        source_sequences = sequence.get("source_sequence_sha256s")
        sequence_flags = sequence.get("flags")
        if (
            not isinstance(candidate_sequences, Mapping)
            or not isinstance(source_sequences, Mapping)
            or not isinstance(sequence_flags, Mapping)
            or set(candidate_sequences)
            != set(r64._SOURCE_PLAN_SEQUENCE_FIELDS)
            or set(source_sequences)
            != set(r64._SOURCE_PLAN_SEQUENCE_FIELDS)
        ):
            raise ValueError("expert-count source sequences drifted")
        expected_sequence_flags = {
            name: tuple(getattr(plan, name))
            == tuple(source_sequences[name])
            for name in r64._SOURCE_PLAN_SEQUENCE_FIELDS
        }
        if (
            dict(sequence_flags) != expected_sequence_flags
            or sequence.get("passed")
            is not all(expected_sequence_flags.values())
            or any(
                tuple(candidate_sequences[name])
                != tuple(getattr(plan, name))
                for name in r64._SOURCE_PLAN_SEQUENCE_FIELDS
            )
        ):
            raise ValueError("expert-count source sequence decision drifted")
        balance = d0d3._contribution_balance_gate(
            plan,
            recipe=d3_recipe,
            gates=objective_protocol.gates,
            training_teacher_energy=float(
                report_gauge["unit_fit_teacher_weighted_energy"]
            ),
            raw_teacher_energy=float(
                report_gauge["raw_fit_teacher_weighted_energy"]
            ),
            teacher_signal_diagnostics=training_signal,
        )
        final_audit = audit_objective_contributions(
            plan.final_metrics,
            plan.objective,
        ).state_dict()
        recomputed_final_metrics = (
            evaluate_contrast_aware_reference_provider(
                plan,
                batches=ordinary_scoring_batches,
                contrast_pairs=fit_contrast_pairs,
            )
        )
        if (
            _canonical_json_bytes(
                recomputed_final_metrics.state_dict()
            )
            != _canonical_json_bytes(plan.final_metrics.state_dict())
        ):
            raise ValueError("expert-count final metrics replay drifted")
        if (
            _canonical_json_bytes(balance)
            != _canonical_json_bytes(row.get("objective_balance_gate"))
            or _canonical_json_bytes(final_audit)
            != _canonical_json_bytes(row.get("final_contribution_audit"))
        ):
            raise ValueError("expert-count objective audit drifted")
        score = FullWidthCandidateScore.from_state_dict(
            row["ordinary_score"]
        )
        _recompute_ordinary_candidate_and_score(
            candidate_id=str(candidate_id),
            plan=plan,
            published_candidate=ordinary_candidate,
            published_score=score,
            batches=ordinary_scoring_batches,
            measured=restored_fit_endpoints,
            probes=ordinary_probes,
            controls=controls,
            metric_weight=metric,
            gates=SyntheticReferenceGates(),
            c2_protocol=c2_protocol,
            contrast_gates=contrast_gates,
            required_null_candidate_pass_count=(
                objective_protocol.gates
                .required_null_candidate_pass_count
            ),
            published_structural_metadata=row.get(
                "structural_metadata"
            ),
            published_contrast_result=row.get("contrast_result"),
            published_contrast_identities=row.get(
                "contrast_identities"
            ),
            published_contrast_coverage=row.get(
                "contrast_coverage"
            ),
        )
        accounting = asdict(plan.accounting())
        expected_stored, expected_macs = (
            (31_492, 5_555_776)
            if arm == "expert2"
            else (48_166, 8_985_792)
        )
        execution = row.get("execution_accounting")
        recomputed_execution = d0d3._fit_execution_accounting(
            plan,
            ordinary_scoring_batches,
        )
        if (
            isinstance(execution, Mapping)
            and set(execution) == _EXECUTION_ACCOUNTING_FIELDS
            and _canonical_json_bytes(execution)
            != _canonical_json_bytes(recomputed_execution)
        ):
            raise ValueError(
                "expert-count execution accounting replay drifted"
            )
        candidate_probe_ids = tuple(
            value.probe_id for value in ordinary_candidate.predictions
        )
        score_probe_ids = tuple(
            value.probe_id for value in score.probe_metrics
        )
        expected_probe_ids = tuple(controls.fit_probe_ids)
        score_families_are_bound = all(
            metric.probe_id.startswith("development_c2.fit.ordinary.")
            and metric.family == metric.probe_id.rsplit(".", 2)[-2]
            for metric in score.probe_metrics
        )
        expected_canonical_execution = {
            "canonical_sequence_length": 128,
            "canonical_batch_size": 1,
            "canonical_total_mac_count": expected_macs,
            "canonical_core_mac_count": (
                4_499_008 if arm == "expert2" else 7_929_024
            ),
            "canonical_encoder_mac_count": 524_288,
            "canonical_decoder_mac_count": 524_288,
            "canonical_target_destandardization_mac_count": 8_192,
        }
        if (
            score.candidate_id != candidate_id
            or score.candidate_artifact_sha256
            != ordinary_candidate.artifact_sha256
            or score.candidate_artifact_sha256
            != row.get("candidate_binding_sha256")
            or ordinary_candidate.candidate_id != candidate_id
            or ordinary_candidate.candidate_binding_sha256
            != plan.artifact_sha256
            or ordinary_candidate.source_rank != 64
            or ordinary_candidate.target_rank != 64
            or ordinary_candidate.stored_scalar_count != expected_stored
            or candidate_probe_ids != expected_probe_ids
            or score_probe_ids != expected_probe_ids
            or not score_families_are_bound
            or any(
                value.standardized_gauge_sha256
                != manifest.get("standardized_gauge_sha256")
                for value in ordinary_candidate.predictions
            )
            or _canonical_json_bytes(
                ordinary_candidate.structural_metrics.state_dict()
            )
            != _canonical_json_bytes(score.structural_metrics.state_dict())
            or score.source_rank != 64
            or score.target_rank != 64
            or score.stored_scalar_count != expected_stored
            or accounting["total_stored_scalar_count"] != expected_stored
            or _canonical_json_bytes(row.get("accounting"))
            != _canonical_json_bytes(accounting)
            or not isinstance(execution, Mapping)
            or set(execution) != _EXECUTION_ACCOUNTING_FIELDS
            or score.controls_artifact_sha256
            != controls.artifact_sha256
            or score.gates_sha256
            != protocol.training.ordinary_gates_sha256
        ):
            raise ValueError("expert-count accounting/score binding drifted")
        assert isinstance(execution, Mapping)
        count_fields = _EXECUTION_ACCOUNTING_FIELDS - {
            "macs_per_valid_row_over_fit_panel",
            "semantics",
        }
        if (
            any(
                type(execution.get(name)) is not int
                or execution[name] < 0
                for name in count_fields
            )
            or any(
                execution.get(name) != value
                for name, value in expected_canonical_execution.items()
            )
            or execution.get("fit_panel_valid_rows", 0) <= 0
            or type(execution.get("macs_per_valid_row_over_fit_panel"))
            is not float
            or not math.isfinite(
                execution["macs_per_valid_row_over_fit_panel"]
            )
            or execution["macs_per_valid_row_over_fit_panel"] < 0.0
            or execution["macs_per_valid_row_over_fit_panel"]
            != (
                execution["fit_panel_total_mac_count"]
                / execution["fit_panel_valid_rows"]
            )
            or execution["fit_panel_total_mac_count"]
            != (
                execution["fit_panel_core_mac_count"]
                + execution["fit_panel_encoder_mac_count"]
                + execution["fit_panel_decoder_mac_count"]
                + execution[
                    "fit_panel_target_destandardization_mac_count"
                ]
            )
            or execution["canonical_total_mac_count"]
            != (
                execution["canonical_core_mac_count"]
                + execution["canonical_encoder_mac_count"]
                + execution["canonical_decoder_mac_count"]
                + execution[
                    "canonical_target_destandardization_mac_count"
                ]
            )
            or execution.get("semantics")
            != (
                "ideal_sparse_mathematical_MACs_not_wall_clock_or_"
                "kernel_latency"
            )
        ):
            raise ValueError("expert-count execution accounting drifted")
        recomputed_flags = r64._recompute_ordinary_gate_flags(
            score,
            ordinary_gates,
        )
        supplied_flags = score.gate_flags.state_dict()
        if any(
            supplied_flags.get(name) != value
            for name, value in recomputed_flags.items()
        ) or score.passed is not all(recomputed_flags.values()):
            raise ValueError("expert-count ordinary gates drifted")
        _, contrast_scores = r64._validate_contrast_result_state(
            row["contrast_result"],
            gates=contrast_gates,
        )
        coverage = r64._recompute_contrast_coverage(
            contrast_scores,
            row["contrast_identities"],
            required_null_candidate_pass_count=24,
        )
        if (
            _canonical_json_bytes(coverage)
            != _canonical_json_bytes(row["contrast_coverage"])
            or _canonical_json_bytes(row["contrast_identities"])
            != _canonical_json_bytes(
                r64._expected_contrast_identities(c2_protocol)
            )
        ):
            raise ValueError("expert-count contrast evidence drifted")
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
            raise ValueError("expert-count fit decision drifted")
        validity = row.get("pair_treatment_validity")
        if (
            not isinstance(validity, Mapping)
            or set(validity) != _VALIDITY_FIELDS
            or validity.get("failure_semantics")
            != (
                "invalid_paired_expert_count_comparison_no_capacity_"
                "conclusion"
            )
        ):
            raise ValueError("expert-count validity schema drifted")

    recomputed = _recompute_decision_from_rows(rows)
    for name, expected in recomputed.items():
        if manifest.get(name) != expected:
            raise ValueError("expert-count final decision drifted")
    if (
        _canonical_json_bytes(report.get("candidate_results"))
        != _canonical_json_bytes(
            [rows[candidate_id] for candidate_id in executed]
        )
        or _canonical_json_bytes(report.get("interpretation"))
        != _canonical_json_bytes(
            _interpretation_from_evidence(rows, recomputed)
        )
        or _canonical_json_bytes(report.get("safety"))
        != _canonical_json_bytes(_safety_contract())
    ):
        raise ValueError("expert-count report safety semantics drifted")
    return LoadedFunctionPreservingExpertCountControlArtifact(
        state=raw,
        report=report,
        manifest=manifest,
        artifact_sha256=logical,
        tensor_file_sha256=tensor_sha256,
        report_sha256=computed_report_sha,
    )


def run_function_preserving_expert_count_control(
    *,
    source_expert_rank_path: Path | str = DEFAULT_SOURCE_EXPERT_RANK,
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
    """Execute the authenticated paired fit-only expert-count control."""

    destination = _validate_output_path(output)
    problem = _prepare_live_fit_problem(
        source_expert_rank_path=source_expert_rank_path,
        source_diagnostic_path=source_diagnostic_path,
        source_rank64_path=source_rank64_path,
        basis_package_path=basis_package_path,
        basis_package_file_sha256=basis_package_file_sha256,
        basis_package_payload_sha256=basis_package_payload_sha256,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    protocol = problem.protocol
    code = _code_sha256s()
    code_bundle = _code_bundle_sha256(code)
    measurement_sha = _measurement_evidence_sha256(
        problem.measurement_evidence
    )
    if measurement_sha != protocol.training.measurement_evidence_sha256:
        raise ValueError("expert-count measurement evidence drifted")
    live_preflight = _run_fit_only_preflight(problem)
    _validate_fit_only_preflight(
        live_preflight,
        protocol=protocol,
    )
    protocol_state = protocol.state_dict()
    if (
        protocol.preflight.bindings_finalized is not True
        or protocol_state.get("outcome_run_allowed") is not True
    ):
        raise RuntimeError(
            "expert-count outcome run is blocked until the exact two-seed "
            "fit-only preflight is frozen"
        )
    primary = _evaluate_pair(
        pair_role="primary",
        seed=protocol.training.primary_seed,
        problem=problem,
    )
    replication = None
    if (
        primary.treatment_valid
        and primary.comparison_status
        == "expert2_fail_expert4_pass"
    ):
        replication = _evaluate_pair(
            pair_role="replication",
            seed=protocol.training.replication_seed,
            problem=problem,
        )
    pairs = tuple(
        value for value in (primary, replication) if value is not None
    )
    arms = tuple(
        arm
        for pair in pairs
        for arm in (pair.control, pair.treatment)
    )
    decision = _decision(primary, replication)
    if (
        problem.adapter.model_fingerprint()
        != problem.model_before_sha256
        or module_state_fingerprint(problem.pre_ff3)
        != problem.norm_sha256
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
    ordinary_candidate_states = {
        arm.candidate_id: arm.ordinary_candidate.state_dict()
        for arm in arms
    }
    ordinary_scoring_batches = _ordered_ordinary_scoring_batches(
        problem.fit_batches
    )
    ordinary_scoring_batch_states = tuple(
        value.state_dict() for value in ordinary_scoring_batches
    )
    fit_contrast_pairs = tuple(
        sorted(problem.fit_pairs, key=lambda value: value.pair_id)
    )
    fit_contrast_pair_states = tuple(
        value.state_dict() for value in fit_contrast_pairs
    )
    result_hashes = {
        str(row["candidate_id"]): _json_sha256(
            row,
            domain=_RESULT_DOMAIN,
        )
        for row in rows
    }
    source = protocol.source
    c2_protocol = problem.c2_protocol
    calibrated_fit_panel = c2_protocol.calibrated_panel_sha256(
        "fit",
        problem.calibration,
    )
    ordinary_gate_sha = full_width_reference_gates_sha256(
        _deferred_collision_gates(problem.fidelity_gates)
    )
    manifest = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "source_expert_rank_protocol_sha256": (
            source.expert_rank_protocol_sha256
        ),
        "source_expert_rank_code_bundle_sha256": (
            source.expert_rank_code_bundle_sha256
        ),
        "source_expert_rank_logical_artifact_sha256": (
            source.expert_rank_logical_artifact_sha256
        ),
        "source_expert_rank_tensor_file_sha256": (
            source.expert_rank_tensor_file_sha256
        ),
        "source_expert_rank_report_sha256": (
            source.expert_rank_report_sha256
        ),
        "source_expert_rank_primary_e2r64_plan_sha256": (
            source.expert_rank_primary_e2r64_plan_sha256
        ),
        "source_expert_rank_primary_e2r64_result_sha256": (
            source.expert_rank_primary_e2r64_result_sha256
        ),
        "c2_protocol_sha256": c2_protocol.protocol_sha256,
        "c2_pilot_panel_sha256": c2_protocol.panel_sha256("pilot"),
        "c2_fit_panel_sha256": c2_protocol.panel_sha256("fit"),
        "c2_calibrated_fit_panel_sha256": calibrated_fit_panel,
        "c2_calibration_sha256": (
            problem.calibration.artifact_sha256
        ),
        "selected_calibration_amplitude": (
            problem.calibration.selected_amplitude
        ),
        "basis_package_file_sha256": basis_package_file_sha256,
        "basis_package_payload_sha256": (
            problem.basis.basis_payload_sha256
        ),
        "source_model_sha256": problem.basis.source_model_sha256,
        "requested_execution_device": device_name,
        "requested_execution_dtype": dtype,
        "actual_execution_device": problem.actual_device,
        "actual_execution_dtype": problem.actual_dtype,
        "pre_feedforward_norm_sha256": problem.norm_sha256,
        "canonical_metric_weight_sha256": d0d3._tensor_sha256(
            problem.raw_metric_weight
        ),
        "fit_data_binding_sha256": problem.fit_data_binding_sha256,
        "unit_rms_gauge_sha256": problem.unit_gauge.artifact_sha256,
        "standardized_gauge_sha256": (
            problem.standardized_gauge_sha256
        ),
        "controls_sha256": problem.controls.artifact_sha256,
        "ordinary_gates_sha256": ordinary_gate_sha,
        "contrast_gates_sha256": (
            problem.contrast_gates.artifact_sha256
        ),
        "measurement_evidence_sha256": measurement_sha,
        "executed_candidate_ids": tuple(
            str(row["candidate_id"]) for row in rows
        ),
        "candidate_plan_sha256s": {
            arm.candidate_id: arm.plan.artifact_sha256 for arm in arms
        },
        "ordinary_candidate_sha256s": {
            arm.candidate_id: arm.ordinary_candidate.artifact_sha256
            for arm in arms
        },
        "ordinary_scoring_indexed_batch_sha256s": tuple(
            value.artifact_sha256
            for value in ordinary_scoring_batches
        ),
        "fit_contrast_pair_sha256s": tuple(
            value.artifact_sha256 for value in fit_contrast_pairs
        ),
        "candidate_result_sha256s": result_hashes,
        **decision,
        "selection_materialized": False,
        "selection_measured": False,
        "selection_scored": False,
        "c2_provider_artifact_loaded": False,
        "authenticated_expert_rank_source_loaded": True,
        "source_final_parameters_used_for_initialization": False,
        "fit_split_wrapper_used": True,
        "published_plans_use_concatenated_executor": True,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "code_sha256s": code,
        "code_bundle_sha256": code_bundle,
        "scientific_scope": (
            "fit_only_paired_function_preserving_expert_count_control"
        ),
    }
    logical = _json_sha256(manifest, domain=_ARTIFACT_DOMAIN)
    state = {
        "manifest": manifest,
        "artifact_sha256": logical,
        "protocol_state": protocol.state_dict(),
        "calibration_state": problem.calibration.state_dict(),
        "unit_rms_gauge_state": problem.unit_gauge.state_dict(),
        "canonical_metric_weight": problem.raw_metric_weight,
        "controls_state": problem.controls.state_dict(),
        "ordinary_scoring_batch_states": (
            ordinary_scoring_batch_states
        ),
        "fit_contrast_pair_states": fit_contrast_pair_states,
        "plan_states": plan_states,
        "ordinary_candidate_states": ordinary_candidate_states,
        "candidate_results": row_map,
    }
    report_payload = {
        **manifest,
        "artifact_sha256": logical,
        "protocol": protocol.state_dict(),
        "calibration": problem.calibration.state_dict(),
        **problem.measurement_evidence,
        "candidate_results": rows,
        "interpretation": _interpretation_from_evidence(
            row_map,
            decision,
        ),
        "safety": _safety_contract(),
    }
    receipt = _publish_artifact(
        state,
        report_payload,
        output=destination,
    )
    try:
        loaded = load_function_preserving_expert_count_control_artifact(
            destination,
            **receipt,
        )
    except BaseException:
        destination.unlink(missing_ok=True)
        destination.with_suffix(".json").unlink(missing_ok=True)
        raise
    return dict(loaded.report)


def run_function_preserving_expert_count_preflight(
    *,
    source_expert_rank_path: Path | str = DEFAULT_SOURCE_EXPERT_RANK,
    source_diagnostic_path: Path | str = DEFAULT_SOURCE_DIAGNOSTIC,
    source_rank64_path: Path | str = DEFAULT_SOURCE_RANK64,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    basis_package_file_sha256: str = DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    basis_package_payload_sha256: str = (
        DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
    ),
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Replay only the frozen two-step fit-side implementation preflight.

    This path cannot execute a 600-step outcome fit, score a candidate, or
    publish an artifact.
    """

    problem = _prepare_live_fit_problem(
        source_expert_rank_path=source_expert_rank_path,
        source_diagnostic_path=source_diagnostic_path,
        source_rank64_path=source_rank64_path,
        basis_package_path=basis_package_path,
        basis_package_file_sha256=basis_package_file_sha256,
        basis_package_payload_sha256=basis_package_payload_sha256,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    code = _code_sha256s()
    preflight = _run_fit_only_preflight(problem)
    _validate_fit_only_preflight(preflight, protocol=problem.protocol)
    if (
        problem.adapter.model_fingerprint()
        != problem.model_before_sha256
        or module_state_fingerprint(problem.pre_ff3)
        != problem.norm_sha256
        or _code_sha256s() != code
    ):
        raise RuntimeError(
            "model, normalization, or code changed during preflight"
        )
    return dict(preflight)


def _add_live_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-expert-rank",
        type=Path,
        default=DEFAULT_SOURCE_EXPERT_RANK,
    )
    parser.add_argument(
        "--source-diagnostic",
        type=Path,
        default=DEFAULT_SOURCE_DIAGNOSTIC,
    )
    parser.add_argument(
        "--source-rank64",
        type=Path,
        default=DEFAULT_SOURCE_RANK64,
    )
    parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--dtype", default="float32", choices=("float32",))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "describe",
        help="print the sealed declaration without loading artifacts",
    )
    preflight = commands.add_parser(
        "preflight",
        help=(
            "replay the frozen two-seed, two-step fit-only implementation "
            "audit without scoring or publication"
        ),
    )
    run = commands.add_parser(
        "run",
        help="execute the authenticated expert-count control",
    )
    _add_live_arguments(preflight)
    _add_live_arguments(run)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        report = describe_function_preserving_expert_count_control()
    elif args.command == "preflight":
        report = run_function_preserving_expert_count_preflight(
            source_expert_rank_path=args.source_expert_rank,
            source_diagnostic_path=args.source_diagnostic,
            source_rank64_path=args.source_rank64,
            basis_package_path=args.basis_package,
            cache_dir=args.cache_dir,
            device_name=args.device,
            dtype=args.dtype,
        )
    else:
        report = run_function_preserving_expert_count_control(
            source_expert_rank_path=args.source_expert_rank,
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
