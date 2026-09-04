"""Mandatory synthetic diagnostics for compact-message completion.

This module implements the post-representation, model-free diagnostics that
were deliberately kept out of the campaign/orchestration layer:

* A_mid-conditioned MLP q/v codecs and the M2q/M2v ceilings;
* the fixed p=32 compact ambient and chart oracles; and
* the standardized full-rank identity smoke and later frozen-tolerance null.

Every entry point accepts only the deterministic synthetic campaign objects.
It never opens a corpus, a ``.local-runs`` directory, or a D panel, and none of
its results is eligible for a scientific or guard claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import json
import struct
from typing import Final, Literal

import numpy as np
import torch
from torch import Tensor

from .fisher_compact_message_completion_qualification import (
    CODEC_LADDER,
    FittedCodecCandidate,
    SelectionGeometry,
    evaluate_fidelity,
    fidelity_gate_records,
    row_keys,
    subset_partition_by_units,
)
from .fisher_compact_message_completion_science import (
    FidelityMetrics,
    FisherCodec,
    HierarchicalRowKey,
    QContributionCodec,
    VFlowCodec,
    combined_qv_second_moment,
    evaluate_five_fidelity_gates,
    fit_q_contribution_codec,
    fit_v_flow_codec,
    hierarchical_row_weights,
    tensor_weighted_median,
    weighted_relative_l2,
)
from .fisher_compact_message_completion_synthetic import (
    SYNTHETIC_EVIDENCE_SCOPE,
    SyntheticCampaignData,
    SyntheticPartition,
    exact_synthetic_mlp_blocks,
    synthetic_rmsnorm,
)


__all__ = [
    "CompactOracleDiagnostics",
    "DiagnosticContractError",
    "DirectFullRankIdentityNull",
    "FidelityPanelResult",
    "FittedMlpCodecCandidate",
    "FourBlockCaptureValidation",
    "FullRankCalibration",
    "FullRankCoefficientCalibration",
    "FullRankHeadCalibration",
    "FullRankIdentitySmoke",
    "FullRankNormalization",
    "FullRankPanelValidation",
    "FullRankValidationRecord",
    "LocalOracleMetrics",
    "M2DiagnosticResults",
    "OraclePanelResult",
    "build_full_rank_identity_smoke",
    "calibrate_full_rank_identity",
    "evaluate_compact_oracles",
    "evaluate_direct_full_rank_identity_null",
    "evaluate_m2_diagnostics",
    "fit_mlp_codec_candidates",
    "validate_four_block_capture",
    "validate_full_rank_identity",
]


_FULL_RANK_VARIANCE_FLOOR: Final = 1e-6
_FULL_RANK_REFERENCE_FLOOR: Final = 1e-8
_FULL_RANK_TOLERANCE_FLOOR: Final = 1e-12
_FULL_RANK_CAP_MULTIPLIER: Final = 1e-5
_FULL_RANK_CAP_OFFSET: Final = 1e-7
_FULL_RANK_ERROR_MULTIPLIER: Final = 10.0
_FULL_RANK_ROUNDOFF_MULTIPLIER: Final = 100.0
_FULL_RANK_ADVANTAGE_MULTIPLIER: Final = 10.0
_FOUR_BLOCK_COUNT: Final = 4
_FULL_PROMPT_LABEL: Final = (
    "source-authoritative, teacher-forced full-prompt shadow evaluation"
)


class DiagnosticContractError(RuntimeError):
    """A synthetic measurement violates a frozen diagnostic contract."""


def _tensor(value: np.ndarray | Tensor) -> Tensor:
    if isinstance(value, Tensor):
        result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    else:
        result = torch.from_numpy(np.asarray(value).copy()).to(dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise DiagnosticContractError("diagnostic tensors must be finite")
    return result


def _fit_train(campaign: SyntheticCampaignData) -> SyntheticPartition:
    return subset_partition_by_units(
        campaign.fit,
        campaign.fit_split.right_units,
    )


def _fit_monitor(campaign: SyntheticCampaignData) -> SyntheticPartition:
    return subset_partition_by_units(
        campaign.fit,
        campaign.fit_split.left_units,
    )


def _partition_keys(partition: SyntheticPartition) -> tuple[HierarchicalRowKey, ...]:
    return row_keys(partition)


def _counts(
    partition: SyntheticPartition,
    mask: Tensor | np.ndarray,
) -> tuple[int, int]:
    selected = np.asarray(mask, dtype=bool)
    row_count = int(np.count_nonzero(selected))
    units = {
        metadata.analysis_unit_commitment
        for metadata, keep in zip(partition.metadata, selected, strict=True)
        if keep
    }
    return row_count, len(units)


def _masked_keys(
    keys: Sequence[HierarchicalRowKey],
    mask: Tensor | np.ndarray,
) -> tuple[HierarchicalRowKey, ...]:
    selected = np.asarray(mask, dtype=bool)
    return tuple(key for key, keep in zip(keys, selected, strict=True) if keep)


def _hash_tensors(domain: bytes, tensors: Sequence[Tensor]) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for tensor in tensors:
        value = _tensor(tensor).numpy().astype("<f8", copy=False)
        digest.update(struct.pack(">I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack(">Q", dimension))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _hash_json(domain: bytes, value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain + b"\0" + encoded).hexdigest()


def _apply_chart_codecs(
    values: Tensor,
    chart_ids: Tensor,
    codecs: Sequence[FisherCodec],
    operation: Literal["encode", "decode", "project"],
) -> Tensor:
    rows = _tensor(values)
    charts = chart_ids.detach().to(device="cpu", dtype=torch.int64).flatten()
    if rows.shape[0] != charts.numel() or not codecs:
        raise DiagnosticContractError("codec rows, chart IDs, and codec inventory differ")
    if bool((charts < 0).any()) or bool((charts >= len(codecs)).any()):
        raise DiagnosticContractError("chart assignment is outside the codec inventory")
    output_width = (
        codecs[0].rank
        if operation == "encode"
        else codecs[0].decoder.shape[0]
    )
    result = torch.empty(rows.shape[:-1] + (output_width,), dtype=torch.float64)
    for chart, codec in enumerate(codecs):
        mask = charts == chart
        if bool(mask.any()):
            result[mask] = getattr(codec, operation)(rows[mask])
    return result.contiguous()


def _fidelity_panel(
    partition_role: Literal["fit_monitor", "tune", "development_confirm"],
    metrics: FidelityMetrics,
) -> "FidelityPanelResult":
    return FidelityPanelResult(partition_role, metrics)


@dataclass(frozen=True, slots=True)
class FidelityPanelResult:
    partition_role: Literal["fit_monitor", "tune", "development_confirm"]
    metrics: FidelityMetrics
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE

    @property
    def all_five_passed(self) -> bool:
        return evaluate_five_fidelity_gates(self.metrics).passed

    def as_artifact_panel(self) -> dict[str, object]:
        return {
            "partition_role": self.partition_role,
            "gate_results": fidelity_gate_records(self.metrics),
            "all_five_passed": self.all_five_passed,
            "evaluation_label": _FULL_PROMPT_LABEL,
        }


@dataclass(frozen=True, slots=True)
class FourBlockCaptureValidation:
    partition_role: str
    block_boundaries: tuple[tuple[int, int], ...]
    q_block_sum_max_abs_error: float
    q_block_sum_weighted_relative_l2: float
    q_exact_capture_max_abs_error: float
    q_exact_capture_weighted_relative_l2: float
    v_block_sum_max_abs_error: float
    v_block_sum_weighted_relative_l2: float
    v_exact_capture_max_abs_error: float
    v_exact_capture_weighted_relative_l2: float
    post_sum_rmsnorm_max_abs_error: float
    post_sum_rmsnorm_weighted_relative_l2: float
    q_absolute_cap: float
    v_absolute_cap: float
    relative_cap: float
    passed: bool
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE


def _error_pair(
    reference: Tensor,
    observed: Tensor,
    weights: Tensor,
) -> tuple[float, float]:
    if reference.shape != observed.shape:
        raise DiagnosticContractError("diagnostic seam tensors have different shapes")
    return (
        float((observed - reference).abs().max()),
        weighted_relative_l2(reference, observed, weights),
    )


def validate_four_block_capture(
    partition: SyntheticPartition,
    campaign: SyntheticCampaignData,
) -> FourBlockCaptureValidation:
    """Validate the contiguous-floor four-block MLP capture and common norm.

    The check is measurement-validity machinery, not an M2 pass/fail result.
    Any M2 evaluation refuses to run when this check fails.
    """

    required = ("h_A", "q_M", "q_M_block", "v_M", "v_M_block")
    if any(name not in partition.tensors for name in required):
        raise DiagnosticContractError("four-block MLP capture tensors are incomplete")
    h_a = _tensor(partition.tensors["h_A"])
    q_m = _tensor(partition.tensors["q_M"])
    q_blocks = _tensor(partition.tensors["q_M_block"])
    v_m = _tensor(partition.tensors["v_M"])
    v_blocks = _tensor(partition.tensors["v_M_block"])
    hidden = campaign.config.hidden_dimension
    if q_m.shape != (h_a.shape[0], hidden) or v_m.shape != q_m.shape:
        raise DiagnosticContractError("MLP total capture shapes violate the contract")
    expected_block_shape = (h_a.shape[0], _FOUR_BLOCK_COUNT, hidden)
    if q_blocks.shape != expected_block_shape or v_blocks.shape != expected_block_shape:
        raise DiagnosticContractError(
            "MLP block capture must have exactly four contiguous-floor blocks"
        )
    width = int(campaign.source.mlp_gate.shape[0])
    boundaries = tuple(
        ((block * width) // _FOUR_BLOCK_COUNT, ((block + 1) * width) // _FOUR_BLOCK_COUNT)
        for block in range(_FOUR_BLOCK_COUNT)
    )
    if boundaries[0][0] != 0 or boundaries[-1][1] != width:
        raise AssertionError("contiguous-floor block construction is incomplete")
    if sum(end - start for start, end in boundaries) != width:
        raise AssertionError("contiguous-floor blocks do not partition every channel")

    keys = _partition_keys(partition)
    weights = hierarchical_row_weights(keys)
    q_sum_error, q_sum_relative = _error_pair(q_m, q_blocks.sum(dim=1), weights)
    exact_blocks = _tensor(
        exact_synthetic_mlp_blocks(h_a.numpy(), campaign.source)
    )
    q_exact_error, q_exact_relative = _error_pair(
        q_blocks.reshape(q_blocks.shape[0], -1),
        exact_blocks.reshape(exact_blocks.shape[0], -1),
        weights,
    )
    v_sum_error, v_sum_relative = _error_pair(v_m, v_blocks.sum(dim=1), weights)
    q_sum = q_blocks.sum(dim=1)
    common_denominator = torch.sqrt(
        q_sum.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    exact_v_blocks = (
        q_blocks
        / common_denominator[:, None, :]
        * _tensor(campaign.source.post_mlp_norm_weight)[None, None, :]
    )
    v_exact_error, v_exact_relative = _error_pair(
        v_blocks.reshape(v_blocks.shape[0], -1),
        exact_v_blocks.reshape(exact_v_blocks.shape[0], -1),
        weights,
    )
    post_sum = _tensor(
        synthetic_rmsnorm(
            q_sum.numpy(),
            campaign.source.post_mlp_norm_weight,
        )
    )
    post_error, post_relative = _error_pair(v_m, post_sum, weights)
    q_scale = max(
        _FULL_RANK_REFERENCE_FLOOR,
        tensor_weighted_median(q_m, weights, keys, absolute=True),
    )
    v_scale = max(
        _FULL_RANK_REFERENCE_FLOOR,
        tensor_weighted_median(v_m, weights, keys, absolute=True),
    )
    # These are the F5 calibration hard caps.  The actual O1 artifact freezes
    # tighter seam tolerances; this capture-level precondition only determines
    # whether it is valid to compute an M2 diagnostic at all.
    q_cap = 1e-4 * q_scale + 1e-6
    v_cap = 1e-4 * v_scale + 1e-6
    relative_cap = 1e-4
    passed = (
        max(q_sum_error, q_exact_error) <= q_cap
        and max(q_sum_relative, q_exact_relative) <= relative_cap
        and max(v_sum_error, v_exact_error, post_error) <= v_cap
        and max(v_sum_relative, v_exact_relative, post_relative) <= relative_cap
    )
    return FourBlockCaptureValidation(
        partition.partition_role,
        boundaries,
        q_sum_error,
        q_sum_relative,
        q_exact_error,
        q_exact_relative,
        v_sum_error,
        v_sum_relative,
        v_exact_error,
        v_exact_relative,
        post_error,
        post_relative,
        q_cap,
        v_cap,
        relative_cap,
        passed,
    )


@dataclass(frozen=True, slots=True)
class FittedMlpCodecCandidate:
    codec_id: str
    q_rank: int
    v_rank: int
    regularization_ratio: float
    q_codecs: tuple[QContributionCodec, ...]
    v_codecs: tuple[VFlowCodec, ...]
    mid_locator_sha256: str
    fit_train_capture_validation: FourBlockCaptureValidation
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE

    def __post_init__(self) -> None:
        if not self.q_codecs or len(self.q_codecs) != len(self.v_codecs):
            raise DiagnosticContractError(
                "MLP Q-contribution and V-flow codec inventories must align"
            )
        if any(not isinstance(codec, QContributionCodec) for codec in self.q_codecs):
            raise DiagnosticContractError(
                "q_codecs must contain only QContributionCodec values"
            )
        if any(not isinstance(codec, VFlowCodec) for codec in self.v_codecs):
            raise DiagnosticContractError(
                "v_codecs must contain only VFlowCodec values"
            )
        if any(codec.rank != self.q_rank for codec in self.q_codecs):
            raise DiagnosticContractError("MLP Q codec rank is inconsistent")
        if any(codec.rank != self.v_rank for codec in self.v_codecs):
            raise DiagnosticContractError("MLP V codec rank is inconsistent")

    def project_q(self, values: Tensor, chart_ids: Tensor) -> Tensor:
        return _apply_chart_codecs(values, chart_ids, self.q_codecs, "project")

    def project_v(self, values: Tensor, chart_ids: Tensor) -> Tensor:
        return _apply_chart_codecs(values, chart_ids, self.v_codecs, "project")

    @property
    def identity_sha256(self) -> str:
        tensors: list[Tensor] = []
        for codec in self.q_codecs + self.v_codecs:
            tensors.extend((codec.encoder, codec.decoder, codec.projector))
        digest = _hash_tensors(
            f"fisher-cmc/synthetic-mlp-codec/{self.codec_id}/v1".encode("ascii"),
            tensors,
        )
        return hashlib.sha256(
            b"fisher-cmc/synthetic-mlp-codec-binding/v1\0"
            + bytes.fromhex(digest)
            + bytes.fromhex(self.mid_locator_sha256)
        ).hexdigest()


def fit_mlp_codec_candidates(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
) -> tuple[FittedMlpCodecCandidate, ...]:
    """Fit all four mirrored MLP q/v codecs from fit-train only."""

    fit_train = _fit_train(campaign)
    capture_validation = validate_four_block_capture(fit_train, campaign)
    if not capture_validation.passed:
        raise DiagnosticContractError(
            "invalid_campaign_measurement_contract: four-block MLP capture failed"
        )
    h_a = _tensor(fit_train.tensors["h_A"])
    charts = geometry.mid_locator.assign(h_a)
    q_total = _tensor(fit_train.tensors["q_M"])
    q_blocks = _tensor(fit_train.tensors["q_M_block"])
    v_total = _tensor(fit_train.tensors["v_M"])
    v_blocks = _tensor(fit_train.tensors["v_M_block"])
    keys = _partition_keys(fit_train)
    carrier_vectors = _tensor(campaign.source.q_carrier)
    carrier_eigenvalues = _tensor(campaign.source.h4_fisher_diagonal)
    candidates: list[FittedMlpCodecCandidate] = []
    for codec_id, q_rank, v_rank, ratio in CODEC_LADDER:
        q_codecs: list[QContributionCodec] = []
        v_codecs: list[VFlowCodec] = []
        for chart in range(geometry.mid_locator.chart_count):
            mask = charts == chart
            if not bool(mask.any()):
                raise DiagnosticContractError(
                    "development_complete_insufficient_support: empty A_mid fit-train chart"
                )
            chart_keys = _masked_keys(keys, mask)
            weights = hierarchical_row_weights(chart_keys)
            q_moment = combined_qv_second_moment(
                q_total[mask], q_blocks[mask], weights
            )
            v_moment = combined_qv_second_moment(
                v_total[mask], v_blocks[mask], weights
            )
            q_codecs.append(
                fit_q_contribution_codec(
                    q_moment,
                    carrier_vectors,
                    carrier_eigenvalues,
                    rank=q_rank,
                    regularization_ratio=ratio,
                )
            )
            v_codecs.append(
                fit_v_flow_codec(
                    v_moment,
                    carrier_vectors,
                    carrier_eigenvalues,
                    rank=v_rank,
                    regularization_ratio=ratio,
                )
            )
        candidates.append(
            FittedMlpCodecCandidate(
                codec_id,
                q_rank,
                v_rank,
                ratio,
                tuple(q_codecs),
                tuple(v_codecs),
                geometry.mid_locator.identity_sha256,
                capture_validation,
            )
        )
    return tuple(candidates)


def _mlp_candidate(
    candidates: Sequence[FittedMlpCodecCandidate], codec_id: str
) -> FittedMlpCodecCandidate:
    matches = tuple(candidate for candidate in candidates if candidate.codec_id == codec_id)
    if len(matches) != 1:
        raise DiagnosticContractError("selected MLP codec is absent or duplicated")
    return matches[0]


def _m2_panel(
    partition: SyntheticPartition,
    partition_role: Literal["fit_monitor", "tune"],
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    candidate: FittedMlpCodecCandidate,
    *,
    flow: Literal["q", "v"],
) -> FidelityPanelResult:
    validation = validate_four_block_capture(partition, campaign)
    if not validation.passed:
        raise DiagnosticContractError(
            "invalid_campaign_measurement_contract: M2 panel has an invalid four-block capture"
        )
    if candidate.mid_locator_sha256 != geometry.mid_locator.identity_sha256:
        raise DiagnosticContractError("M2 codec is bound to another A_mid locator")
    h_a = _tensor(partition.tensors["h_A"])
    charts = geometry.mid_locator.assign(h_a)
    if flow == "q":
        q_hat = candidate.project_q(_tensor(partition.tensors["q_M"]), charts)
        v_hat = _tensor(
            synthetic_rmsnorm(q_hat.numpy(), campaign.source.post_mlp_norm_weight)
        )
    else:
        v_hat = candidate.project_v(_tensor(partition.tensors["v_M"]), charts)
    h_plus = h_a + v_hat
    return _fidelity_panel(
        partition_role,
        evaluate_fidelity(partition, h_plus, campaign),
    )


@dataclass(frozen=True, slots=True)
class M2DiagnosticResults:
    codec_id: str
    mlp_codec_sha256: str
    fit_train_capture_validation: FourBlockCaptureValidation
    m2q_panels: tuple[FidelityPanelResult, ...]
    m2v_panels: tuple[FidelityPanelResult, ...]
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE

    def artifact_body(self, ceiling: Literal["M2q", "M2v"]) -> dict[str, object]:
        panels = self.m2q_panels if ceiling == "M2q" else self.m2v_panels
        return {
            "ceiling": ceiling,
            "primary_mlp_partition": "contiguous_floor_four_block",
            "exact_teacher_attention": True,
            "exact_h_A": True,
            "learned_mlp_replacement": False,
            "diagnostic_only": True,
            "panels": [panel.as_artifact_panel() for panel in panels],
        }


def evaluate_m2_diagnostics(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    candidates: Sequence[FittedMlpCodecCandidate],
    codec_id: str,
) -> M2DiagnosticResults:
    """Evaluate selected M2q/M2v on fit-monitor and tune.

    M2 is diagnostic-only and never changes the selected attention codec or
    the eligibility of a later O7 evaluation.
    """

    candidate = _mlp_candidate(candidates, codec_id)
    fit_monitor = _fit_monitor(campaign)
    panels = (("fit_monitor", fit_monitor), ("tune", campaign.tune))
    q_results = tuple(
        _m2_panel(
            partition,
            role,
            campaign,
            geometry,
            candidate,
            flow="q",
        )
        for role, partition in panels
    )
    v_results = tuple(
        _m2_panel(
            partition,
            role,
            campaign,
            geometry,
            candidate,
            flow="v",
        )
        for role, partition in panels
    )
    return M2DiagnosticResults(
        codec_id,
        candidate.identity_sha256,
        candidate.fit_train_capture_validation,
        q_results,
        v_results,
    )


@dataclass(frozen=True, slots=True)
class LocalOracleMetrics:
    q_total_max_abs_error: float
    q_total_weighted_relative_l2: float
    h_a_max_abs_error: float
    h_a_weighted_relative_l2: float
    h_plus_max_abs_error: float
    h_plus_weighted_relative_l2: float

    def as_dict(self) -> dict[str, float]:
        return {
            "q_total_max_abs_error": self.q_total_max_abs_error,
            "q_total_weighted_relative_l2": self.q_total_weighted_relative_l2,
            "h_A_max_abs_error": self.h_a_max_abs_error,
            "h_A_weighted_relative_l2": self.h_a_weighted_relative_l2,
            "h_plus_max_abs_error": self.h_plus_max_abs_error,
            "h_plus_weighted_relative_l2": self.h_plus_weighted_relative_l2,
        }


@dataclass(frozen=True, slots=True)
class OraclePanelResult:
    partition_role: Literal["fit_monitor", "tune"]
    local: LocalOracleMetrics
    full_prompt: FidelityPanelResult
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE

    def as_protocol_record(self) -> dict[str, object]:
        return {
            "partition_role": self.partition_role,
            "local": self.local.as_dict(),
            "full_prompt_shadow": self.full_prompt.as_artifact_panel(),
        }


def _oracle_panel(
    partition: SyntheticPartition,
    partition_role: Literal["fit_monitor", "tune"],
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    codec: FittedCodecCandidate,
    *,
    chart_projected: bool,
) -> OraclePanelResult:
    messages = _tensor(partition.tensors["message"])
    retained = geometry.message_sketch.transform(messages)
    reconstructed = geometry.message_sketch.reconstruct(retained)
    q_heads = torch.einsum(
        "ahm,nam->nah",
        _tensor(campaign.source.output_weights),
        reconstructed,
    )
    h = _tensor(partition.tensors["h"])
    charts = geometry.locator.assign(h)
    if chart_projected:
        q_heads = codec.project_q(q_heads, charts)
    q_total = q_heads.sum(dim=1)
    v_a = _tensor(
        synthetic_rmsnorm(q_total.numpy(), campaign.source.post_attention_norm_weight)
    )
    h_a = h + v_a
    exact_blocks = _tensor(exact_synthetic_mlp_blocks(h_a.numpy(), campaign.source))
    q_m = exact_blocks.sum(dim=1)
    v_m = _tensor(
        synthetic_rmsnorm(q_m.numpy(), campaign.source.post_mlp_norm_weight)
    )
    h_plus = h_a + v_m
    keys = _partition_keys(partition)
    weights = hierarchical_row_weights(keys)
    q_reference = _tensor(partition.tensors["q_A"])
    h_a_reference = _tensor(partition.tensors["h_A"])
    h_plus_reference = _tensor(partition.tensors["h_plus"])
    q_error, q_relative = _error_pair(q_reference, q_total, weights)
    h_a_error, h_a_relative = _error_pair(h_a_reference, h_a, weights)
    h_plus_error, h_plus_relative = _error_pair(
        h_plus_reference, h_plus, weights
    )
    return OraclePanelResult(
        partition_role,
        LocalOracleMetrics(
            q_error,
            q_relative,
            h_a_error,
            h_a_relative,
            h_plus_error,
            h_plus_relative,
        ),
        _fidelity_panel(
            partition_role,
            evaluate_fidelity(partition, h_plus, campaign),
        ),
    )


@dataclass(frozen=True, slots=True)
class CompactOracleDiagnostics:
    codec_id: str
    representation_sha256: str
    message_rank_per_head: int
    ambient_panels: tuple[OraclePanelResult, ...]
    chart_panels: tuple[OraclePanelResult, ...]
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE

    def artifact_body(
        self, oracle: Literal["C-Ambient", "C-Chart"]
    ) -> dict[str, object]:
        is_chart = oracle == "C-Chart"
        panels = self.chart_panels if is_chart else self.ambient_panels
        return {
            "oracle": oracle,
            "message_rank_per_head": self.message_rank_per_head,
            "q_codec_projection": is_chart,
            "selection_eligible": False,
            "panels": [panel.as_protocol_record() for panel in panels],
        }

    def as_protocol_dict(
        self, oracle: Literal["C-Ambient", "C-Chart"]
    ) -> dict[str, object]:
        """Return the complete Section 19 local-plus-shadow report.

        The top-level claim markers live in the artifact envelope; this richer
        view is useful to callers that need a standalone protocol record.
        """

        is_chart = oracle == "C-Chart"
        panels = self.chart_panels if is_chart else self.ambient_panels
        return {
            "evidence_scope": self.evidence_scope,
            "scientific_claim_eligible": False,
            "guard_claim_eligible": False,
            "oracle": oracle,
            "message_rank_per_head": self.message_rank_per_head,
            "q_codec_projection": is_chart,
            "selection_eligible": False,
            "panels": [panel.as_protocol_record() for panel in panels],
        }


def evaluate_compact_oracles(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    codec_id: str,
) -> CompactOracleDiagnostics:
    """Evaluate the mandatory fixed-p=32 ambient and chart oracle paths."""

    if geometry.message_sketch.rank != 32:
        raise DiagnosticContractError(
            "compact oracle is frozen at p=32 per query head"
        )
    codec = geometry.codec(codec_id)
    fit_monitor = _fit_monitor(campaign)
    panels = (("fit_monitor", fit_monitor), ("tune", campaign.tune))
    ambient = tuple(
        _oracle_panel(
            partition,
            role,
            campaign,
            geometry,
            codec,
            chart_projected=False,
        )
        for role, partition in panels
    )
    chart = tuple(
        _oracle_panel(
            partition,
            role,
            campaign,
            geometry,
            codec,
            chart_projected=True,
        )
        for role, partition in panels
    )
    return CompactOracleDiagnostics(
        codec_id,
        geometry.identity_sha256,
        32,
        ambient,
        chart,
    )


@dataclass(frozen=True, slots=True)
class FullRankNormalization:
    message_centers: Tensor
    coordinate_mean: Tensor
    coordinate_scale: Tensor
    variance_floor: float = _FULL_RANK_VARIANCE_FLOOR

    def roundtrip(self, messages: np.ndarray | Tensor) -> Tensor:
        values = _tensor(messages)
        if values.ndim != 3 or values.shape[1:] != self.message_centers.shape:
            raise DiagnosticContractError(
                "full-rank messages must have shape [rows, heads, message_dimension]"
            )
        rho = values - self.message_centers
        standardized = (rho - self.coordinate_mean) / self.coordinate_scale
        rho_roundtrip = self.coordinate_mean + self.coordinate_scale * standardized
        return (self.message_centers + rho_roundtrip).contiguous()

    @property
    def identity_sha256(self) -> str:
        return _hash_tensors(
            b"fisher-cmc/synthetic-full-rank-normalization/v1",
            (self.message_centers, self.coordinate_mean, self.coordinate_scale),
        )


@dataclass(frozen=True, slots=True)
class FullRankHeadCalibration:
    head_id: int
    fit_train_row_count: int
    fit_train_analysis_unit_count: int
    message_reference_scale: float
    q_reference_scale: float
    observed_max_error_m: float
    observed_max_error_q: float
    cap_m: float
    cap_q: float
    tau_full_m: float
    tau_full_q: float
    passed: bool


@dataclass(frozen=True, slots=True)
class FullRankCoefficientCalibration:
    codec_id: str
    chart_id: int
    head_id: int
    fit_train_row_count: int
    fit_train_analysis_unit_count: int
    coefficient_reference_scale: float
    observed_max_error_c: float
    cap_c: float
    tau_full_c: float
    tau_full_advantage: float
    passed: bool


@dataclass(frozen=True, slots=True)
class FullRankCalibration:
    representation_sha256: str
    normalization: FullRankNormalization
    head_calibrations: tuple[FullRankHeadCalibration, ...]
    coefficient_calibrations: tuple[FullRankCoefficientCalibration, ...]
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.head_calibrations) and all(
            item.passed for item in self.coefficient_calibrations
        )

    def head(self, head_id: int) -> FullRankHeadCalibration:
        matches = tuple(item for item in self.head_calibrations if item.head_id == head_id)
        if len(matches) != 1:
            raise DiagnosticContractError("full-rank head tolerance is absent or duplicated")
        return matches[0]

    def coefficient(
        self, codec_id: str, chart_id: int, head_id: int
    ) -> FullRankCoefficientCalibration:
        matches = tuple(
            item
            for item in self.coefficient_calibrations
            if (item.codec_id, item.chart_id, item.head_id)
            == (codec_id, chart_id, head_id)
        )
        if len(matches) != 1:
            raise DiagnosticContractError(
                "codec/chart/head full-rank tolerance is absent or duplicated"
            )
        return matches[0]

    @property
    def identity_sha256(self) -> str:
        body: dict[str, object] = {
            "representation_sha256": self.representation_sha256,
            "normalization_sha256": self.normalization.identity_sha256,
            "head_calibrations": [
                {
                    field: getattr(item, field)
                    for field in (descriptor.name for descriptor in fields(item))
                }
                for item in self.head_calibrations
            ],
            "coefficient_calibrations": [
                {
                    field: getattr(item, field)
                    for field in (descriptor.name for descriptor in fields(item))
                }
                for item in self.coefficient_calibrations
            ],
        }
        return _hash_json(b"fisher-cmc/synthetic-full-rank-tolerances/v1", body)


def _validate_head_shapes(
    partition: SyntheticPartition,
    campaign: SyntheticCampaignData,
) -> None:
    if "message" not in partition.tensors or "q_A_head" not in partition.tensors:
        raise DiagnosticContractError(
            "missing_head_in_validation_rows: message/q head capture is absent"
        )
    messages = partition.tensors["message"]
    q_heads = partition.tensors["q_A_head"]
    expected_heads = campaign.config.head_count
    if (
        messages.ndim != 3
        or q_heads.ndim != 3
        or messages.shape[0] != q_heads.shape[0]
        or messages.shape[1] != expected_heads
        or q_heads.shape[1] != expected_heads
    ):
        raise DiagnosticContractError(
            "missing_head_in_validation_rows: every valid row must retain every query head"
        )
    if messages.shape[2] != campaign.config.message_dimension:
        raise DiagnosticContractError("transported-message tensor shape is invalid")
    if q_heads.shape[2] != campaign.config.hidden_dimension:
        raise DiagnosticContractError("attention q-head tensor shape is invalid")


def _fit_full_rank_normalization(
    fit_train: SyntheticPartition,
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
) -> FullRankNormalization:
    _validate_head_shapes(fit_train, campaign)
    messages = _tensor(fit_train.tensors["message"])
    keys = _partition_keys(fit_train)
    weights = hierarchical_row_weights(keys)
    centers = torch.stack(
        tuple(
            representation.mean
            for representation in geometry.message_sketch.representations
        ),
        dim=0,
    ).to(dtype=torch.float64)
    if centers.shape != messages.shape[1:]:
        raise DiagnosticContractError(
            "full-rank centers do not match the transported-message shape"
        )
    rho = messages - centers
    coordinate_mean = torch.einsum("n,nam->am", weights, rho)
    centered = rho - coordinate_mean
    variance = torch.einsum("n,nam,nam->am", weights, centered, centered)
    scale = torch.sqrt(torch.clamp(variance, min=_FULL_RANK_VARIANCE_FLOOR))
    return FullRankNormalization(
        centers.contiguous(),
        coordinate_mean.contiguous(),
        scale.contiguous(),
    )


def _reference_scale(
    values: Tensor,
    weights: Tensor,
    keys: Sequence[HierarchicalRowKey],
) -> float:
    return max(
        _FULL_RANK_REFERENCE_FLOOR,
        tensor_weighted_median(values, weights, keys, absolute=True),
    )


def _full_rank_tolerance(error: float, reference_scale: float) -> tuple[float, float]:
    cap = _FULL_RANK_CAP_MULTIPLIER * reference_scale + _FULL_RANK_CAP_OFFSET
    tau = min(
        cap,
        max(
            _FULL_RANK_ERROR_MULTIPLIER * error,
            _FULL_RANK_ROUNDOFF_MULTIPLIER
            * float(torch.finfo(torch.float64).eps)
            * reference_scale,
            _FULL_RANK_TOLERANCE_FLOOR,
        ),
    )
    return cap, tau


def _full_rank_roundtrip(
    partition: SyntheticPartition,
    campaign: SyntheticCampaignData,
    normalization: FullRankNormalization,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_head_shapes(partition, campaign)
    messages = _tensor(partition.tensors["message"])
    q_reference = _tensor(partition.tensors["q_A_head"])
    message_roundtrip = normalization.roundtrip(messages)
    q_roundtrip = torch.einsum(
        "ahm,nam->nah",
        _tensor(campaign.source.output_weights),
        message_roundtrip,
    )
    return messages, q_reference, message_roundtrip, q_roundtrip


def calibrate_full_rank_identity(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
) -> FullRankCalibration:
    """Derive all full-rank tolerances from every fit-train row only."""

    fit_train = _fit_train(campaign)
    normalization = _fit_full_rank_normalization(fit_train, campaign, geometry)
    messages, q_reference, message_roundtrip, q_roundtrip = _full_rank_roundtrip(
        fit_train, campaign, normalization
    )
    keys = _partition_keys(fit_train)
    weights = hierarchical_row_weights(keys)
    row_count, unit_count = _counts(
        fit_train, np.ones(len(fit_train.metadata), dtype=bool)
    )
    head_calibrations: list[FullRankHeadCalibration] = []
    for head in range(campaign.config.head_count):
        m_error = float(
            (message_roundtrip[:, head] - messages[:, head]).abs().max()
        )
        q_error = float(
            (q_roundtrip[:, head] - q_reference[:, head]).abs().max()
        )
        m_scale = _reference_scale(messages[:, head], weights, keys)
        q_scale = _reference_scale(q_reference[:, head], weights, keys)
        cap_m, tau_m = _full_rank_tolerance(m_error, m_scale)
        cap_q, tau_q = _full_rank_tolerance(q_error, q_scale)
        head_calibrations.append(
            FullRankHeadCalibration(
                head,
                row_count,
                unit_count,
                m_scale,
                q_scale,
                m_error,
                q_error,
                cap_m,
                cap_q,
                tau_m,
                tau_q,
                m_error <= cap_m and q_error <= cap_q,
            )
        )

    h = _tensor(fit_train.tensors["h"])
    charts = geometry.locator.assign(h)
    coefficient_calibrations: list[FullRankCoefficientCalibration] = []
    for candidate in geometry.codec_candidates:
        for chart in range(geometry.locator.chart_count):
            mask = charts == chart
            chart_row_count, chart_unit_count = _counts(fit_train, mask)
            if chart_row_count == 0:
                raise DiagnosticContractError(
                    "development_complete_insufficient_support: active chart has no fit-train full-rank rows"
                )
            chart_keys = _masked_keys(keys, mask)
            chart_weights = hierarchical_row_weights(chart_keys)
            codec = candidate.q_codecs[chart]
            for head in range(campaign.config.head_count):
                c_reference = codec.encode(q_reference[mask, head])
                c_roundtrip = codec.encode(q_roundtrip[mask, head])
                error = float((c_roundtrip - c_reference).abs().max())
                scale = _reference_scale(c_reference, chart_weights, chart_keys)
                cap, tau = _full_rank_tolerance(error, scale)
                coefficient_calibrations.append(
                    FullRankCoefficientCalibration(
                        candidate.codec_id,
                        chart,
                        head,
                        chart_row_count,
                        chart_unit_count,
                        scale,
                        error,
                        cap,
                        tau,
                        _FULL_RANK_ADVANTAGE_MULTIPLIER * tau * tau,
                        error <= cap,
                    )
                )
    return FullRankCalibration(
        geometry.identity_sha256,
        normalization,
        tuple(head_calibrations),
        tuple(coefficient_calibrations),
    )


@dataclass(frozen=True, slots=True)
class FullRankValidationRecord:
    codec_id: str
    chart_id: int
    head_id: int
    fit_train_row_count: int
    fit_train_analysis_unit_count: int
    validation_row_count: int
    validation_analysis_unit_count: int
    observed_max_error_m: float | None
    observed_max_error_q: float | None
    observed_max_error_c: float | None
    coefficient_completion_mse: float | None
    tau_full_m: float
    tau_full_q: float
    tau_full_c: float
    tau_full_advantage: float
    status: Literal["passed", "failed", "invalid_zero_validation_support"]
    zero_row_reason: str | None

    def as_artifact_record(self) -> dict[str, object]:
        # The strict artifact schema uses numeric zero sentinels for a zero-row
        # record.  ``zero_row_reason`` remains available in the richer protocol
        # representation below and must not be interpreted as an observed zero.
        return {
            "codec_id": self.codec_id,
            "chart_id": self.chart_id,
            "head_id": self.head_id,
            "fit_train_row_count": self.fit_train_row_count,
            "fit_train_analysis_unit_count": self.fit_train_analysis_unit_count,
            "fit_monitor_row_count": self.validation_row_count,
            "fit_monitor_analysis_unit_count": self.validation_analysis_unit_count,
            "max_error_m": self.observed_max_error_m or 0.0,
            "max_error_q": self.observed_max_error_q or 0.0,
            "max_error_c": self.observed_max_error_c or 0.0,
            "tau_full_m": self.tau_full_m,
            "tau_full_q": self.tau_full_q,
            "tau_full_c": self.tau_full_c,
            "status": self.status,
        }

    def as_protocol_record(self) -> dict[str, object]:
        return {
            "codec_id": self.codec_id,
            "chart_id": self.chart_id,
            "head_id": self.head_id,
            "assigned_row_count": self.validation_row_count,
            "distinct_analysis_unit_count": self.validation_analysis_unit_count,
            "observed_max_error_m": self.observed_max_error_m,
            "observed_max_error_q": self.observed_max_error_q,
            "observed_max_error_c": self.observed_max_error_c,
            "coefficient_completion_mse": self.coefficient_completion_mse,
            "tau_full_m": self.tau_full_m,
            "tau_full_q": self.tau_full_q,
            "tau_full_c": self.tau_full_c,
            "tau_full_advantage": self.tau_full_advantage,
            "validation_status": self.status,
            "zero_row_reason": self.zero_row_reason,
        }


@dataclass(frozen=True, slots=True)
class FullRankPanelValidation:
    partition_role: str
    records: tuple[FullRankValidationRecord, ...]
    representation_candidate_status: Literal[
        "eligible", "failed", "invalid_zero_validation_support"
    ]
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE

    @property
    def passed(self) -> bool:
        return self.representation_candidate_status == "eligible" and all(
            record.status == "passed" for record in self.records
        )


def _weighted_coefficient_mse(error: Tensor, weights: Tensor) -> float:
    per_row = error.square().mean(dim=-1)
    return float(torch.dot(weights, per_row))


def validate_full_rank_identity(
    partition: SyntheticPartition,
    partition_role: str,
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    calibration: FullRankCalibration,
    *,
    codec_ids: Sequence[str] | None = None,
) -> FullRankPanelValidation:
    """Apply frozen full-rank tolerances to a panel without refitting anything."""

    if calibration.representation_sha256 != geometry.identity_sha256:
        raise DiagnosticContractError(
            "full-rank tolerances are bound to another frozen representation"
        )
    messages, q_reference, message_roundtrip, q_roundtrip = _full_rank_roundtrip(
        partition, campaign, calibration.normalization
    )
    h = _tensor(partition.tensors["h"])
    charts = geometry.locator.assign(h)
    keys = _partition_keys(partition)
    selected_ids = (
        tuple(candidate.codec_id for candidate in geometry.codec_candidates)
        if codec_ids is None
        else tuple(codec_ids)
    )
    if len(set(selected_ids)) != len(selected_ids):
        raise DiagnosticContractError("full-rank codec selection contains duplicates")
    records: list[FullRankValidationRecord] = []
    saw_zero = False
    for codec_id in selected_ids:
        candidate = geometry.codec(codec_id)
        for chart in range(geometry.locator.chart_count):
            mask = charts == chart
            row_count, unit_count = _counts(partition, mask)
            for head in range(campaign.config.head_count):
                head_calibration = calibration.head(head)
                coefficient_calibration = calibration.coefficient(
                    codec_id, chart, head
                )
                if row_count == 0:
                    saw_zero = True
                    records.append(
                        FullRankValidationRecord(
                            codec_id,
                            chart,
                            head,
                            coefficient_calibration.fit_train_row_count,
                            coefficient_calibration.fit_train_analysis_unit_count,
                            0,
                            0,
                            None,
                            None,
                            None,
                            None,
                            head_calibration.tau_full_m,
                            head_calibration.tau_full_q,
                            coefficient_calibration.tau_full_c,
                            coefficient_calibration.tau_full_advantage,
                            "invalid_zero_validation_support",
                            "no_valid_fit_monitor_rows_assigned_to_active_source_chart",
                        )
                    )
                    continue
                chart_keys = _masked_keys(keys, mask)
                chart_weights = hierarchical_row_weights(chart_keys)
                m_error = float(
                    (message_roundtrip[mask, head] - messages[mask, head])
                    .abs()
                    .max()
                )
                q_error = float(
                    (q_roundtrip[mask, head] - q_reference[mask, head])
                    .abs()
                    .max()
                )
                codec = candidate.q_codecs[chart]
                c_reference = codec.encode(q_reference[mask, head])
                c_roundtrip = codec.encode(q_roundtrip[mask, head])
                c_difference = c_roundtrip - c_reference
                c_error = float(c_difference.abs().max())
                completion_mse = _weighted_coefficient_mse(
                    c_difference, chart_weights
                )
                passed = (
                    m_error <= head_calibration.tau_full_m
                    and q_error <= head_calibration.tau_full_q
                    and c_error <= coefficient_calibration.tau_full_c
                    and completion_mse
                    <= coefficient_calibration.tau_full_advantage
                )
                records.append(
                    FullRankValidationRecord(
                        codec_id,
                        chart,
                        head,
                        coefficient_calibration.fit_train_row_count,
                        coefficient_calibration.fit_train_analysis_unit_count,
                        row_count,
                        unit_count,
                        m_error,
                        q_error,
                        c_error,
                        completion_mse,
                        head_calibration.tau_full_m,
                        head_calibration.tau_full_q,
                        coefficient_calibration.tau_full_c,
                        coefficient_calibration.tau_full_advantage,
                        "passed" if passed else "failed",
                        None,
                    )
                )
    if saw_zero:
        candidate_status: Literal[
            "eligible", "failed", "invalid_zero_validation_support"
        ] = "invalid_zero_validation_support"
    elif all(record.status == "passed" for record in records):
        candidate_status = "eligible"
    else:
        candidate_status = "failed"
    return FullRankPanelValidation(
        partition_role,
        tuple(records),
        candidate_status,
    )


@dataclass(frozen=True, slots=True)
class FullRankIdentitySmoke:
    calibration_fit_train: FullRankCalibration
    validation_fit_monitor: FullRankPanelValidation
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE

    @property
    def passed(self) -> bool:
        return self.calibration_fit_train.passed and self.validation_fit_monitor.passed

    @property
    def frozen_tolerances_sha256(self) -> str:
        return self.calibration_fit_train.identity_sha256

    def as_artifact_body(self) -> dict[str, object]:
        return {
            "message_rank": "full_message_dimension",
            "standardization_roundtrip": True,
            "calibration_fit_train": {
                "normalization_sha256": (
                    self.calibration_fit_train.normalization.identity_sha256
                ),
                "head_tolerances": [
                    {
                        field: getattr(item, field)
                        for field in (
                            descriptor.name for descriptor in fields(item)
                        )
                    }
                    for item in self.calibration_fit_train.head_calibrations
                ],
                "coefficient_tolerances": [
                    {
                        field: getattr(item, field)
                        for field in (
                            descriptor.name for descriptor in fields(item)
                        )
                    }
                    for item in self.calibration_fit_train.coefficient_calibrations
                ],
            },
            "validation_fit_monitor": {
                "representation_candidate_status": (
                    self.validation_fit_monitor.representation_candidate_status
                ),
                "records": [
                    record.as_protocol_record()
                    for record in self.validation_fit_monitor.records
                ],
            },
            "frozen_tolerances_sha256": self.frozen_tolerances_sha256,
            "all_eligible_records_passed": self.passed,
        }

    def as_protocol_dict(self) -> dict[str, object]:
        """Return the richer H7/I6 split view, including zero-row reasons."""

        return {
            "evidence_scope": self.evidence_scope,
            "scientific_claim_eligible": False,
            "guard_claim_eligible": False,
            "standardization_roundtrip": True,
            "calibration_fit_train": {
                "normalization_sha256": self.calibration_fit_train.normalization.identity_sha256,
                "head_tolerances": [
                    {
                        field: getattr(item, field)
                        for field in (descriptor.name for descriptor in fields(item))
                    }
                    for item in self.calibration_fit_train.head_calibrations
                ],
                "coefficient_tolerances": [
                    {
                        field: getattr(item, field)
                        for field in (descriptor.name for descriptor in fields(item))
                    }
                    for item in self.calibration_fit_train.coefficient_calibrations
                ],
            },
            "validation_fit_monitor": {
                "representation_candidate_status": self.validation_fit_monitor.representation_candidate_status,
                "records": [
                    record.as_protocol_record()
                    for record in self.validation_fit_monitor.records
                ],
            },
            "frozen_tolerances_sha256": self.frozen_tolerances_sha256,
            "passed": self.passed,
        }


def build_full_rank_identity_smoke(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
) -> FullRankIdentitySmoke:
    """Build the mandatory pre-tune P4 smoke from fit-train/fit-monitor only."""

    calibration = calibrate_full_rank_identity(campaign, geometry)
    validation = validate_full_rank_identity(
        _fit_monitor(campaign),
        "fit_monitor",
        campaign,
        geometry,
        calibration,
    )
    return FullRankIdentitySmoke(calibration, validation)


@dataclass(frozen=True, slots=True)
class DirectFullRankIdentityNull:
    selected_codec_id: str
    selected_codec_sha256: str
    frozen_tolerances_sha256: str
    panel_validation: FullRankPanelValidation
    representation_refit_performed: bool = False
    evidence_scope: str = SYNTHETIC_EVIDENCE_SCOPE

    @property
    def message_roundtrip_passed(self) -> bool:
        return all(
            record.observed_max_error_m is not None
            and record.observed_max_error_m <= record.tau_full_m
            for record in self.panel_validation.records
        )

    @property
    def q_roundtrip_passed(self) -> bool:
        return all(
            record.observed_max_error_q is not None
            and record.observed_max_error_q <= record.tau_full_q
            for record in self.panel_validation.records
        )

    @property
    def coefficient_roundtrip_passed(self) -> bool:
        return all(
            record.observed_max_error_c is not None
            and record.observed_max_error_c <= record.tau_full_c
            for record in self.panel_validation.records
        )

    @property
    def completion_target_null_passed(self) -> bool:
        return self.coefficient_roundtrip_passed

    @property
    def coefficient_advantage_null_passed(self) -> bool:
        return all(
            record.coefficient_completion_mse is not None
            and record.coefficient_completion_mse <= record.tau_full_advantage
            for record in self.panel_validation.records
        )

    @property
    def passed(self) -> bool:
        return (
            self.panel_validation.passed
            and self.message_roundtrip_passed
            and self.q_roundtrip_passed
            and self.coefficient_roundtrip_passed
            and self.completion_target_null_passed
            and self.coefficient_advantage_null_passed
        )

    def as_artifact_body(self) -> dict[str, object]:
        return {
            "selected_codec_sha256": self.selected_codec_sha256,
            "frozen_tolerances_sha256": self.frozen_tolerances_sha256,
            "representation_refit_performed": self.representation_refit_performed,
            "message_roundtrip_passed": self.message_roundtrip_passed,
            "q_roundtrip_passed": self.q_roundtrip_passed,
            "coefficient_roundtrip_passed": self.coefficient_roundtrip_passed,
            "completion_target_null_passed": self.completion_target_null_passed,
            "coefficient_advantage_null_passed": self.coefficient_advantage_null_passed,
            "downstream_vanishing_required": False,
        }


def evaluate_direct_full_rank_identity_null(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    smoke: FullRankIdentitySmoke,
    codec_id: str,
    *,
    partition: SyntheticPartition | None = None,
    partition_role: str = "all_fit_post_refit_self_check",
) -> DirectFullRankIdentityNull:
    """Run the mandatory post-refit null without changing frozen quantities.

    The optional ``partition`` is an already-captured synthetic panel.  It is
    provided for deterministic self-tests; the default is the complete fit
    capture.  No normalization, center, codec, or tolerance is refitted here.
    """

    if not smoke.passed:
        raise DiagnosticContractError(
            "direct full-rank identity null requires a passing frozen P4 smoke"
        )
    candidate = geometry.codec(codec_id)
    validation = validate_full_rank_identity(
        partition or campaign.fit,
        partition_role,
        campaign,
        geometry,
        smoke.calibration_fit_train,
        codec_ids=(codec_id,),
    )
    return DirectFullRankIdentityNull(
        codec_id,
        candidate.identity_sha256,
        smoke.frozen_tolerances_sha256,
        validation,
    )
