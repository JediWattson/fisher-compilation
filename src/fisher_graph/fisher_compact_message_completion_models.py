"""Paired J-model bank fitting and evaluation for synthetic qualification.

The production scientific contract treats a model family as one optimizer-owned
bank over all active chart/head cells.  This module performs that orchestration
on already-prepared tensors while preserving the common J2N/J4 initialization,
hierarchical row sampler, coefficient-only objective, and weights-only bundle
boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Final

import numpy as np
import torch
from torch import Tensor

from .fisher_compact_message_completion_capture import (
    decode_safetensors,
    encode_safetensors,
)
from .fisher_compact_message_completion_qualification import (
    FittedCodecCandidate,
    PreparedCompletionPanel,
    QualificationError,
    SelectionGeometry,
    evaluate_fidelity,
    prepare_completion_panel,
    row_keys as partition_row_keys,
    subset_partition_by_units,
)
from .fisher_compact_message_completion_science import (
    ChartHeadCompletionBank,
    CompletionBankTrainingRows,
    CompletionTrainingConfig,
    CompletionTrainingResult,
    DownstreamMetric,
    FidelityMetrics,
    HierarchicalRowKey,
    J4Model,
    OperatorAggregateSummary,
    OperatorObservation,
    PairedBootstrapResult,
    aggregate_operator_pair_summaries,
    bank_coefficient_loss,
    bootstrap_paired_comparison,
    completion_response_operator,
    evaluate_completion_bank_loss,
    hierarchical_row_weights,
    iter_stratified_bootstrap_indices,
    make_frozen_completion_bank,
    make_paired_j2n_j4_banks,
    percentile_interval,
    primary_downstream_metric,
    select_restart,
    stable_rank,
    summarize_operator_pairs,
    train_completion_bank,
    weighted_median,
)
from .fisher_compact_message_completion_synthetic import (
    SyntheticCampaignData,
    exact_synthetic_mlp,
    synthetic_rmsnorm,
    synthetic_row_derivation_protocol_sha256,
)


__all__ = [
    "FULL_INITIAL_TRAINING_CONFIG",
    "FULL_DIAGNOSTIC_TRAINING_CONFIG",
    "FULL_REFIT_CONFIG",
    "InitialBankFit",
    "FrozenInteractionMeans",
    "InteractionMeanResult",
    "ModelBankEvaluation",
    "ModelBankFit",
    "ModelComparison",
    "RefitBankFit",
    "InteractionBootstrapResult",
    "LoadedRawExecution",
    "LoadedWeightsOnlyBundles",
    "RAW_EXECUTION_PROTOCOL_SHA256",
    "WeightsOnlyBundleSelfTest",
    "bootstrap_interaction_statistics",
    "build_weights_only_bundles",
    "evaluate_bank",
    "evaluate_interaction_mean",
    "fit_initial_model_banks",
    "fit_refit_model_banks",
    "freeze_epsilon_down",
    "freeze_epsilon_down_from_fit_train",
    "model_comparisons",
    "load_weights_only_bundles",
    "self_test_weights_only_bundles",
    "trained_j4_operator_aggregate",
]


FULL_INITIAL_TRAINING_CONFIG: Final = CompletionTrainingConfig()
FULL_DIAGNOSTIC_TRAINING_CONFIG: Final = CompletionTrainingConfig(
    maximum_steps=1_000,
)
FULL_REFIT_CONFIG: Final = CompletionTrainingConfig(
    maximum_steps=1_000,
    minimum_steps=1_000,
    evaluation_interval=25,
    early_stopping_patience=1_000,
    early_stopping=False,
)
_PRIMARY_FAMILIES: Final = ("J2N", "J2+", "J3", "J4")
_ALL_REPORTED_FAMILIES: Final = ("J0", "J1", "J2N", "J2", "J2+", "J3", "J4")
_RAW_EXECUTION_PROTOCOL: Final = (
    "fisher-cmc/synthetic-frozen-raw-execution/v1\n"
    "assign=A_in minimum support-distance chart\n"
    "z=A_in chart-local standardized coordinate\n"
    "r=per-head p32 message-sketch standardized coordinate\n"
    "message_hat=per-head p32 message-sketch reconstruction\n"
    "q_base_head=W_O_head(message_hat_head)\n"
    "c_base=selected chart-conditioned q encoder(q_base_head)\n"
    "c_hat=selected J-family bank(c_base,z,r)\n"
    "q_hat_head=selected chart-conditioned q decoder(c_hat)\n"
    "q_hat=sum_heads(q_hat_head)\n"
    "v_A=RMSNorm(q_hat; source post-attention norm, epsilon=1e-6)\n"
    "h_A=h+v_A\n"
    "q_M=W_down(silu(W_gate h_A) * (W_up h_A))\n"
    "v_M=RMSNorm(q_M; source post-MLP norm, epsilon=1e-6)\n"
    "h_plus=h_A+v_M\n"
    "logits=lm_head(h_plus)\n"
)
RAW_EXECUTION_PROTOCOL_SHA256: Final = hashlib.sha256(
    _RAW_EXECUTION_PROTOCOL.encode("ascii")
).hexdigest()


def _training_rows(
    panel: PreparedCompletionPanel,
    *,
    full_coefficients: bool,
) -> CompletionBankTrainingRows:
    target = panel.c_teacher if full_coefficients else panel.delta_c
    epsilon = panel.epsilon_j0 if full_coefficients else panel.epsilon_completion
    return CompletionBankTrainingRows(
        c_base=panel.c_base.to(torch.float32),
        z=panel.z.to(torch.float32),
        r=panel.r.to(torch.float32),
        chart_ids=panel.chart_ids,
        target=target.to(torch.float32),
        epsilon=epsilon.to(torch.float32),
        row_keys=panel.row_keys,
    )

def _state_sha256(state: Mapping[str, Tensor], *, domain: bytes) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(len(value.shape).to_bytes(4, "big"))
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _architecture_sha256(bank: ChartHeadCompletionBank, model_id: str) -> str:
    payload = (
        f"{model_id}|charts={bank.chart_ids}|heads={bank.head_count}|"
        f"dz={bank.activation_dimension}|dr={bank.message_dimension}|"
        f"dc={bank.coefficient_dimension}|params={bank.trainable_parameter_count}"
    ).encode("ascii")
    return hashlib.sha256(b"fisher-cmc/synthetic-bank-architecture/v1\0" + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelBankFit:
    model_id: str
    bank: ChartHeadCompletionBank
    selected_seed: int | None
    selected_step: int | None
    coefficient_loss: float
    architecture_sha256: str
    weights_sha256: str
    restart_results: tuple[CompletionTrainingResult, ...]


@dataclass(frozen=True, slots=True)
class InitialBankFit:
    representation_sha256: str
    train_panel: PreparedCompletionPanel
    monitor_panel: PreparedCompletionPanel
    tune_panel: PreparedCompletionPanel
    fits: Mapping[str, ModelBankFit]
    optimizer_updates: int


def _fit_one_family(
    model_id: str,
    train_rows: CompletionBankTrainingRows,
    monitor_rows: CompletionBankTrainingRows,
    *,
    chart_ids: Sequence[int],
    head_count: int,
    coefficient_dimension: int,
    seeds: Sequence[int],
    config: CompletionTrainingConfig,
    prebuilt: Mapping[int, ChartHeadCompletionBank] | None = None,
) -> tuple[ModelBankFit, int]:
    target_kind = "full_coefficients" if model_id == "J0" else "completion_delta"
    candidates: list[tuple[ChartHeadCompletionBank, CompletionTrainingResult]] = []
    for seed in seeds:
        bank = (
            prebuilt[seed]
            if prebuilt is not None
            else make_frozen_completion_bank(
                model_id,
                chart_ids,
                head_count,
                coefficient_dimension,
                initialization_seed=seed,
            )
        )
        result = train_completion_bank(
            bank,
            train_rows,
            monitor_rows,
            seed=seed,
            config=CompletionTrainingConfig(
                learning_rate=config.learning_rate,
                beta1=config.beta1,
                beta2=config.beta2,
                optimizer_epsilon=config.optimizer_epsilon,
                weight_decay=config.weight_decay,
                gradient_clip_norm=config.gradient_clip_norm,
                warmup_steps=config.warmup_steps,
                batch_size=config.batch_size,
                maximum_steps=config.maximum_steps,
                minimum_steps=config.minimum_steps,
                evaluation_interval=config.evaluation_interval,
                early_stopping_patience=config.early_stopping_patience,
                target_kind=target_kind,
                early_stopping=config.early_stopping,
            ),
        )
        candidates.append((bank, result))
    selected_result = select_restart([result for _, result in candidates])
    bank = next(bank for bank, result in candidates if result is selected_result)
    state = {name: value.detach().cpu() for name, value in bank.state_dict().items()}
    coefficient = evaluate_completion_bank_loss(
        bank, monitor_rows, target_kind=target_kind
    )
    return (
        ModelBankFit(
            model_id=model_id,
            bank=bank,
            selected_seed=selected_result.seed,
            selected_step=(
                selected_result.best_step
                if config.early_stopping
                else selected_result.final_step
            ),
            coefficient_loss=coefficient,
            architecture_sha256=_architecture_sha256(bank, model_id),
            weights_sha256=_state_sha256(
                state,
                domain=f"fisher-cmc/synthetic-{model_id}-weights/v1".encode(),
            ),
            restart_results=tuple(result for _, result in candidates),
        ),
        sum(result.final_step for _, result in candidates),
    )


def fit_initial_model_banks(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    codec_id: str,
    *,
    config: CompletionTrainingConfig = FULL_INITIAL_TRAINING_CONFIG,
    diagnostic_config: CompletionTrainingConfig = FULL_DIAGNOSTIC_TRAINING_CONFIG,
    restart_seeds: Sequence[int] = (1701, 2718, 3141),
    include_optional_j2: bool = True,
) -> InitialBankFit:
    """Fit the paired initial J family without tuning any representation."""

    train_partition = subset_partition_by_units(
        campaign.fit, campaign.fit_split.right_units, partition_role="fit"
    )
    monitor_partition = subset_partition_by_units(
        campaign.fit, campaign.fit_split.left_units, partition_role="fit"
    )
    train_panel = prepare_completion_panel(
        train_partition, campaign, geometry, codec_id
    )
    monitor_panel = prepare_completion_panel(
        monitor_partition, campaign, geometry, codec_id
    )
    tune_panel = prepare_completion_panel(campaign.tune, campaign, geometry, codec_id)
    completion_train = _training_rows(train_panel, full_coefficients=False)
    completion_monitor = _training_rows(monitor_panel, full_coefficients=False)
    full_train = _training_rows(train_panel, full_coefficients=True)
    full_monitor = _training_rows(monitor_panel, full_coefficients=True)
    charts = tuple(range(geometry.locator.chart_count))
    heads = campaign.config.head_count
    coefficient = geometry.codec(codec_id).q_rank
    fits: dict[str, ModelBankFit] = {}
    updates = 0

    # J1 is analytic and carries no optimizer state.
    j1 = make_frozen_completion_bank(
        "J1", charts, heads, coefficient, initialization_seed=1701
    )
    j1_loss = evaluate_completion_bank_loss(
        j1, completion_monitor, target_kind="completion_delta"
    )
    fits["J1"] = ModelBankFit(
        "J1",
        j1,
        None,
        None,
        j1_loss,
        _architecture_sha256(j1, "J1"),
        _state_sha256(j1.state_dict(), domain=b"fisher-cmc/synthetic-J1-state/v1"),
        (),
    )

    j0_fit, spent = _fit_one_family(
        "J0",
        full_train,
        full_monitor,
        chart_ids=charts,
        head_count=heads,
        coefficient_dimension=coefficient,
        seeds=(min(restart_seeds),),
        config=diagnostic_config,
    )
    fits["J0"] = j0_fit
    updates += spent

    paired_j2n: dict[int, ChartHeadCompletionBank] = {}
    paired_j4: dict[int, ChartHeadCompletionBank] = {}
    for seed in restart_seeds:
        j2n, j4 = make_paired_j2n_j4_banks(
            charts, heads, coefficient, initialization_seed=seed
        )
        paired_j2n[seed] = j2n
        paired_j4[seed] = j4
    for model_id, prebuilt in (("J2N", paired_j2n), ("J4", paired_j4)):
        fit, spent = _fit_one_family(
            model_id,
            completion_train,
            completion_monitor,
            chart_ids=charts,
            head_count=heads,
            coefficient_dimension=coefficient,
            seeds=restart_seeds,
            config=config,
            prebuilt=prebuilt,
        )
        fits[model_id] = fit
        updates += spent

    for model_id in ("J2+", "J3"):
        fit, spent = _fit_one_family(
            model_id,
            completion_train,
            completion_monitor,
            chart_ids=charts,
            head_count=heads,
            coefficient_dimension=coefficient,
            seeds=restart_seeds,
            config=config,
        )
        fits[model_id] = fit
        updates += spent
    if include_optional_j2:
        fit, spent = _fit_one_family(
            "J2",
            completion_train,
            completion_monitor,
            chart_ids=charts,
            head_count=heads,
            coefficient_dimension=coefficient,
            seeds=(min(restart_seeds),),
            config=diagnostic_config,
        )
        fits["J2"] = fit
        updates += spent
    return InitialBankFit(
        geometry.identity_sha256,
        train_panel,
        monitor_panel,
        tune_panel,
        fits,
        updates,
    )


def _freeze_epsilon_down_rows(
    displacement: Tensor,
    keys: Sequence[HierarchicalRowKey],
    fisher: Tensor,
) -> float:
    row_energy = torch.einsum(
        "ni,ij,nj->n", displacement, fisher.to(torch.float64), displacement
    )
    by_sequence: dict[tuple[bytes, bytes], list[int]] = {}
    for index, key in enumerate(keys):
        by_sequence.setdefault(
            (key.analysis_unit_commitment, key.canonical_sequence_hash), []
        ).append(index)
    by_unit: dict[bytes, list[float]] = {}
    for (unit, _), indexes in by_sequence.items():
        by_unit.setdefault(unit, []).append(float(row_energy[indexes].sum()))
    unit_values = [
        math.fsum(values) / len(values) for _, values in sorted(by_unit.items())
    ]
    reference = weighted_median(unit_values, [1.0] * len(unit_values))
    return max(1e-12, 1e-6 * reference)


def freeze_epsilon_down(panel: PreparedCompletionPanel, fisher: Tensor) -> float:
    return _freeze_epsilon_down_rows(
        panel.h_plus - panel.h,
        panel.row_keys,
        fisher,
    )


def freeze_epsilon_down_from_fit_train(campaign: SyntheticCampaignData) -> float:
    """Freeze the downstream denominator constant before any tune access.

    The constant depends only on exact source ``h``/``h_plus`` differences and
    the H4 Fisher carrier, so representation or codec selection is neither
    needed nor permitted here.
    """

    fit_train = subset_partition_by_units(
        campaign.fit,
        campaign.fit_split.right_units,
        partition_role="fit",
    )
    h = torch.from_numpy(np.asarray(fit_train.tensors["h"]).copy()).to(
        torch.float64
    )
    h_plus = torch.from_numpy(
        np.asarray(fit_train.tensors["h_plus"]).copy()
    ).to(torch.float64)
    fisher = torch.diag(
        torch.from_numpy(
            np.asarray(campaign.source.h4_fisher_diagonal).copy()
        ).to(torch.float64)
    )
    return _freeze_epsilon_down_rows(
        h_plus - h,
        partition_row_keys(fit_train),
        fisher,
    )


def _predict_coefficients(
    bank: ChartHeadCompletionBank, panel: PreparedCompletionPanel
) -> Tensor:
    parameter = next(bank.parameters(), None)
    dtype = torch.float32 if parameter is None else parameter.dtype
    with torch.no_grad():
        value = bank(
            panel.c_base.to(dtype),
            panel.z.to(dtype),
            panel.r.to(dtype),
            panel.chart_ids,
        )
    return value.to(torch.float64)


def _h_plus_from_coefficients(
    coefficients: Tensor,
    panel: PreparedCompletionPanel,
    codec: FittedCodecCandidate,
    campaign: SyntheticCampaignData,
) -> Tensor:
    q_heads = codec.decode_q(coefficients, panel.chart_ids)
    q_total = q_heads.sum(dim=1)
    v_a = synthetic_rmsnorm(
        q_total.detach().cpu().numpy(),
        campaign.source.post_attention_norm_weight,
    )
    h_a = panel.h + torch.from_numpy(v_a).to(torch.float64)
    _, v_m = exact_synthetic_mlp(h_a.detach().cpu().numpy(), campaign.source)
    return h_a + torch.from_numpy(v_m).to(torch.float64)


@dataclass(frozen=True, slots=True)
class ModelBankEvaluation:
    model_id: str
    coefficient_loss: float
    downstream: DownstreamMetric
    fidelity: FidelityMetrics | None


def evaluate_bank(
    fit: ModelBankFit,
    panel: PreparedCompletionPanel,
    codec: FittedCodecCandidate,
    campaign: SyntheticCampaignData,
    *,
    epsilon_down: float,
) -> ModelBankEvaluation:
    target_kind = "full_coefficients" if fit.model_id == "J0" else "completion_delta"
    rows = _training_rows(panel, full_coefficients=fit.model_id == "J0")
    coefficient_loss = evaluate_completion_bank_loss(
        fit.bank, rows, target_kind=target_kind
    )
    predicted = _predict_coefficients(fit.bank, panel)
    h_plus = _h_plus_from_coefficients(predicted, panel, codec, campaign)
    fisher = torch.diag(torch.from_numpy(campaign.source.h4_fisher_diagonal)).to(
        torch.float64
    )
    downstream = primary_downstream_metric(
        h_plus - panel.h_plus,
        panel.h_plus - panel.h,
        fisher,
        panel.row_keys,
        epsilon_down=epsilon_down,
    )
    partition = campaign.partition(panel.partition_role)
    # Fit subsets have the same role but not the full partition; fidelity is
    # only requested for complete tune/development panels in orchestration.
    fidelity: FidelityMetrics | None = None
    if len(partition.metadata) == len(panel.row_keys):
        fidelity = evaluate_fidelity(partition, h_plus, campaign)
    return ModelBankEvaluation(
        fit.model_id, coefficient_loss, downstream, fidelity
    )


def trained_j4_operator_aggregate(
    fit: ModelBankFit,
    panel: PreparedCompletionPanel,
    geometry: SelectionGeometry,
    campaign: SyntheticCampaignData,
    *,
    codec_id: str,
    tau_r: float,
) -> OperatorAggregateSummary:
    """Measure the realized J4 message-response operator on one frozen panel.

    Each supported causal row contributes one observation per retained head.
    ``summarize_operator_pairs`` then applies the frozen equal-analysis-unit
    weighting within chart/head pairs, and
    ``aggregate_operator_pair_summaries`` gives supported pairs equal weight.
    """

    if fit.model_id != "J4" or fit.bank.model_family is not J4Model:
        raise QualificationError("trained operator reporting requires a J4 bank")
    if panel.partition_role != "development_confirm":
        raise QualificationError(
            "trained operator reporting requires the development-confirm panel"
        )
    codec = geometry.codec(codec_id)
    if codec.q_rank != fit.bank.coefficient_dimension:
        raise QualificationError("J4 bank and selected q codec do not align")
    expected_charts = tuple(range(geometry.locator.chart_count))
    if (
        fit.bank.chart_ids != expected_charts
        or panel.active_chart_ids != expected_charts
        or len(codec.q_codecs) != len(expected_charts)
        or geometry.message_sketch.head_count != fit.bank.head_count
        or campaign.source.output_weights.shape[0] != fit.bank.head_count
    ):
        raise QualificationError(
            "J4 bank, panel, locator, codec, message sketch, and source universes differ"
        )
    tolerance = float(tau_r)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise QualificationError("tau_r must be finite and strictly positive")
    if (
        len(panel.row_keys) != panel.chart_ids.numel()
        or panel.support_mask.shape != panel.chart_ids.shape
        or panel.z.shape[0] != len(panel.row_keys)
    ):
        raise QualificationError("operator panel rows are not aligned")
    if any(int(chart_id) not in fit.bank.chart_ids for chart_id in panel.chart_ids):
        raise QualificationError("operator panel contains an inactive chart")

    expected_pairs = tuple(
        (chart_id, head_id)
        for chart_id in fit.bank.chart_ids
        for head_id in range(fit.bank.head_count)
    )
    pair_constants: dict[
        tuple[int, int], tuple[Tensor, Tensor, Tensor, Tensor]
    ] = {}
    for chart_id, head_id in expected_pairs:
        q_codec = codec.q_codecs[chart_id]
        message = geometry.message_sketch.representations[head_id]
        scale = message.coordinate_scale.to(torch.float64)
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0.0).all()):
            raise QualificationError("message-coordinate scales must be positive")
        basis = message.basis.to(torch.float64)
        output_weight = torch.from_numpy(
            np.asarray(campaign.source.output_weights[head_id]).copy()
        ).to(torch.float64)
        analytic_standardized = (
            q_codec.encoder.to(torch.float64)
            @ output_weight
            @ basis
            @ torch.diag(scale)
        )
        pair_constants[(chart_id, head_id)] = (
            analytic_standardized,
            torch.diag(torch.reciprocal(scale)),
            basis,
            q_codec.decoder.to(torch.float64),
        )

    observations: list[OperatorObservation] = []
    with torch.no_grad():
        for row_index, row_key in enumerate(panel.row_keys):
            if not bool(panel.support_mask[row_index]):
                continue
            chart_id = int(panel.chart_ids[row_index])
            z = panel.z[row_index].to(torch.float64)
            for head_id in range(fit.bank.head_count):
                cell = fit.bank.cell(chart_id, head_id)
                if not isinstance(cell, J4Model):
                    raise QualificationError("J4 bank contains a non-J4 cell")
                analytic, inverse_scale, basis, decoder = pair_constants[
                    (chart_id, head_id)
                ]
                response_standardized = completion_response_operator(
                    analytic,
                    cell.message_weight.detach().to(torch.float64),
                    cell.interaction_output.detach().to(torch.float64),
                    cell.activation_interaction.detach().to(torch.float64) @ z,
                    cell.message_interaction.detach().to(torch.float64),
                )
                response_raw = response_standardized @ inverse_scale
                ambient = decoder @ response_raw @ basis.T
                q_codec = codec.q_codecs[chart_id]
                h4_surrogate = (
                    q_codec.carrier_eigenvalues.to(torch.float64).sqrt().unsqueeze(1)
                    * (q_codec.carrier_vectors.to(torch.float64).T @ ambient)
                )
                observations.append(
                    OperatorObservation(
                        row_key=row_key,
                        chart_id=chart_id,
                        head_id=head_id,
                        ambient_operator_norm=float(
                            torch.linalg.matrix_norm(ambient, ord=2)
                        ),
                        h4_surrogate_operator_norm=float(
                            torch.linalg.matrix_norm(h4_surrogate, ord=2)
                        ),
                        ambient_operator_stable_rank=stable_rank(
                            ambient,
                            tolerance=tolerance,
                        ),
                    )
                )
    return aggregate_operator_pair_summaries(
        summarize_operator_pairs(observations, expected_pairs=expected_pairs)
    )


@dataclass(frozen=True, slots=True)
class ModelComparison:
    baseline: str
    result: PairedBootstrapResult
    passed: bool


def model_comparisons(
    evaluations: Mapping[str, ModelBankEvaluation],
    *,
    strata: Sequence[str],
    replicates: int = 10_000,
) -> tuple[ModelComparison, ...]:
    j4 = evaluations["J4"].downstream
    j4_losses = [item.loss for item in j4.units]
    unit_ids = [item.analysis_unit_commitment for item in j4.units]
    result: list[ModelComparison] = []
    for baseline in ("J1", "J2N", "J2+", "J3"):
        base = evaluations[baseline].downstream
        base_by_unit = {
            item.analysis_unit_commitment: item.loss for item in base.units
        }
        base_losses = [base_by_unit[unit] for unit in unit_ids]
        comparison = bootstrap_paired_comparison(
            j4_losses,
            base_losses,
            strata,
            unit_ids=unit_ids,
            replicates=replicates,
        )
        passed = (
            j4.mean_loss < base.mean_loss
            and comparison.rho_point.relative_screen_passed
            and comparison.paired_effect_interval[0] > 0.0
            and comparison.valid_rho_replicate_fraction >= 0.95
        )
        result.append(ModelComparison(baseline, comparison, passed))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class InteractionMeanResult:
    downstream: DownstreamMetric
    gamma: float | None
    advantage: float
    advantage_floor: float
    paired_interval: tuple[float, float]
    gamma_interval: tuple[float, float] | None
    valid_fraction: float
    passed: bool


@dataclass(frozen=True, slots=True)
class InteractionBootstrapResult:
    paired_effect_interval: tuple[float, float]
    gamma_interval: tuple[float, float] | None
    valid_gamma_replicate_fraction: float


def bootstrap_interaction_statistics(
    baseline_unit_losses: Sequence[float],
    j4_unit_losses: Sequence[float],
    ablated_unit_losses: Sequence[float],
    strata: Sequence[str],
    *,
    unit_ids: Sequence[bytes],
    advantage_floor: float,
    replicates: int = 10_000,
) -> InteractionBootstrapResult:
    """Bootstrap the paired ablation effect and E21 three-way gamma."""

    baseline = tuple(float(value) for value in baseline_unit_losses)
    j4 = tuple(float(value) for value in j4_unit_losses)
    ablated = tuple(float(value) for value in ablated_unit_losses)
    if (
        not baseline
        or len(baseline) != len(j4)
        or len(baseline) != len(ablated)
        or len(baseline) != len(strata)
        or len(baseline) != len(unit_ids)
    ):
        raise QualificationError("interaction bootstrap inputs are misaligned")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in baseline + j4 + ablated
    ):
        raise QualificationError("interaction bootstrap losses must be finite and nonnegative")
    if not math.isfinite(advantage_floor) or advantage_floor <= 0.0:
        raise QualificationError("interaction advantage floor must be positive")
    effect_replicates: list[float] = []
    gamma_replicates: list[float] = []
    for indexes in iter_stratified_bootstrap_indices(
        strata,
        unit_ids=unit_ids,
        replicates=replicates,
    ):
        count = len(indexes)
        base_mean = math.fsum(baseline[index] for index in indexes) / count
        j4_mean = math.fsum(j4[index] for index in indexes) / count
        ablated_mean = math.fsum(ablated[index] for index in indexes) / count
        effect = ablated_mean - j4_mean
        effect_replicates.append(effect)
        replicate_advantage = base_mean - j4_mean
        if replicate_advantage > advantage_floor:
            gamma_replicates.append(effect / replicate_advantage)
    valid_fraction = len(gamma_replicates) / replicates
    return InteractionBootstrapResult(
        paired_effect_interval=percentile_interval(effect_replicates),
        gamma_interval=(
            percentile_interval(gamma_replicates)
            if valid_fraction >= 0.95
            else None
        ),
        valid_gamma_replicate_fraction=valid_fraction,
    )


@dataclass(frozen=True, slots=True)
class FrozenInteractionMeans:
    """All-fit chart/head means of ``G_z z``, frozen before later panels."""

    values: Tensor
    identity_sha256: str


def _fit_interaction_means(
    bank: ChartHeadCompletionBank,
    all_fit_panel: PreparedCompletionPanel,
) -> FrozenInteractionMeans:
    if bank.model_family is not J4Model:
        raise QualificationError("interaction-mean requires a J4 bank")
    first = bank.cell(bank.chart_ids[0], 0)
    assert isinstance(first, J4Model)
    values = torch.empty(
        (len(bank.chart_ids), bank.head_count, first.interaction_rank),
        dtype=torch.float32,
    )
    for chart in bank.chart_ids:
        fit_mask = all_fit_panel.chart_ids == chart
        if not bool(fit_mask.any()):
            raise QualificationError(
                f"active chart {chart} has no all-fit rows for interaction mean"
            )
        chart_keys = tuple(
            key
            for key, keep in zip(all_fit_panel.row_keys, fit_mask, strict=True)
            if bool(keep)
        )
        # F13 requires chart-conditioned weights, not globally fitted weights
        # filtered after normalization.
        chart_weights = hierarchical_row_weights(chart_keys).to(torch.float32)
        z_fit = all_fit_panel.z[fit_mask].to(torch.float32)
        for head in range(bank.head_count):
            cell = bank.cell(chart, head)
            assert isinstance(cell, J4Model)
            values[chart, head] = torch.einsum(
                "n,nb->b",
                chart_weights,
                z_fit @ cell.activation_interaction.T,
            )
    frozen = values.detach().cpu().contiguous()
    return FrozenInteractionMeans(
        frozen,
        _state_sha256(
            {"interaction_mean": frozen},
            domain=b"fisher-cmc/synthetic-frozen-interaction-mean/v1",
        ),
    )


def _interaction_mean_coefficients(
    bank: ChartHeadCompletionBank,
    panel: PreparedCompletionPanel,
    frozen_means: FrozenInteractionMeans,
) -> Tensor:
    if bank.model_family is not J4Model:
        raise QualificationError("interaction-mean requires a J4 bank")
    expected = (len(bank.chart_ids), bank.head_count)
    if frozen_means.values.ndim != 3 or frozen_means.values.shape[:2] != expected:
        raise QualificationError("frozen interaction means do not match the J4 bank")
    output = torch.empty_like(panel.c_base, dtype=torch.float32)
    for chart in bank.chart_ids:
        dev_mask = panel.chart_ids == chart
        if not bool(dev_mask.any()):
            continue
        for head in range(bank.head_count):
            cell = bank.cell(chart, head)
            assert isinstance(cell, J4Model)
            output[dev_mask, head, :] = cell.forward_interaction_mean(
                panel.c_base[dev_mask, head, :].to(torch.float32),
                panel.z[dev_mask].to(torch.float32),
                panel.r[dev_mask, head, :].to(torch.float32),
                frozen_means.values[chart, head].to(torch.float32),
            )
    return output.to(torch.float64)


def evaluate_interaction_mean(
    j4_fit: ModelBankFit,
    baseline: ModelBankEvaluation,
    j4: ModelBankEvaluation,
    development_panel: PreparedCompletionPanel,
    frozen_means: FrozenInteractionMeans,
    codec: FittedCodecCandidate,
    campaign: SyntheticCampaignData,
    *,
    epsilon_down: float,
    strata: Sequence[str],
    replicates: int = 10_000,
) -> InteractionMeanResult:
    coefficients = _interaction_mean_coefficients(
        j4_fit.bank, development_panel, frozen_means
    )
    h_plus = _h_plus_from_coefficients(
        coefficients, development_panel, codec, campaign
    )
    fisher = torch.diag(torch.from_numpy(campaign.source.h4_fisher_diagonal)).to(
        torch.float64
    )
    metric = primary_downstream_metric(
        h_plus - development_panel.h_plus,
        development_panel.h_plus - development_panel.h,
        fisher,
        development_panel.row_keys,
        epsilon_down=epsilon_down,
    )
    advantage = baseline.downstream.mean_loss - j4.downstream.mean_loss
    floor = max(1e-12, 1e-6 * baseline.downstream.mean_loss)
    gamma = (
        None
        if advantage <= floor
        else (metric.mean_loss - j4.downstream.mean_loss) / advantage
    )
    mean_by_unit = {
        item.analysis_unit_commitment: item.loss for item in metric.units
    }
    j4_by_unit = {
        item.analysis_unit_commitment: item.loss for item in j4.downstream.units
    }
    unit_ids = [item.analysis_unit_commitment for item in j4.downstream.units]
    base_by_unit = {
        item.analysis_unit_commitment: item.loss
        for item in baseline.downstream.units
    }
    bootstrap = bootstrap_interaction_statistics(
        [base_by_unit[unit] for unit in unit_ids],
        [j4_by_unit[unit] for unit in unit_ids],
        [mean_by_unit[unit] for unit in unit_ids],
        strata,
        unit_ids=unit_ids,
        advantage_floor=floor,
        replicates=replicates,
    )
    passed = (
        gamma is not None
        and gamma >= 0.25
        and bootstrap.paired_effect_interval[0] > 0.0
        and bootstrap.valid_gamma_replicate_fraction >= 0.95
    )
    return InteractionMeanResult(
        metric,
        gamma,
        advantage,
        floor,
        bootstrap.paired_effect_interval,
        bootstrap.gamma_interval,
        bootstrap.valid_gamma_replicate_fraction,
        passed,
    )


@dataclass(frozen=True, slots=True)
class RefitBankFit:
    representation_sha256: str
    fits: Mapping[str, ModelBankFit]
    all_fit_panel: PreparedCompletionPanel
    tune_panel: PreparedCompletionPanel
    interaction_means: FrozenInteractionMeans
    optimizer_updates: int


def fit_refit_model_banks(
    campaign: SyntheticCampaignData,
    geometry: SelectionGeometry,
    codec_id: str,
    *,
    config: CompletionTrainingConfig = FULL_REFIT_CONFIG,
    seed: int = 8_675_309,
) -> RefitBankFit:
    """Run the symmetric all-fit weight-only refit on frozen geometry."""

    all_fit = prepare_completion_panel(campaign.fit, campaign, geometry, codec_id)
    train_rows = _training_rows(all_fit, full_coefficients=False)
    charts = tuple(range(geometry.locator.chart_count))
    heads = campaign.config.head_count
    coefficient = geometry.codec(codec_id).q_rank
    paired_j2n, paired_j4 = make_paired_j2n_j4_banks(
        charts, heads, coefficient, initialization_seed=seed
    )
    prebuilt = {"J2N": {seed: paired_j2n}, "J4": {seed: paired_j4}}
    fits: dict[str, ModelBankFit] = {}
    updates = 0
    for model_id in _PRIMARY_FAMILIES:
        fit, spent = _fit_one_family(
            model_id,
            train_rows,
            train_rows,
            chart_ids=charts,
            head_count=heads,
            coefficient_dimension=coefficient,
            seeds=(seed,),
            config=config,
            prebuilt=prebuilt.get(model_id),
        )
        fits[model_id] = fit
        updates += spent
    interaction_means = _fit_interaction_means(fits["J4"].bank, all_fit)
    # Only after all gradients and fit-derived constants are frozen do we
    # prepare tune for the one
    # non-selecting replay.  Tune is never a refit monitor or checkpoint source.
    tune = prepare_completion_panel(campaign.tune, campaign, geometry, codec_id)
    return RefitBankFit(
        geometry.identity_sha256,
        fits,
        all_fit,
        tune,
        interaction_means,
        updates,
    )


def _bundle_arrays(fits: Mapping[str, ModelBankFit]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for model_id in sorted(fits):
        for name, value in fits[model_id].bank.state_dict().items():
            # Safetensors names remain explicit and do not rely on pickle keys.
            arrays[f"{model_id}.{name}"] = value.detach().cpu().numpy()
    return arrays


@dataclass(frozen=True, slots=True)
class WeightsOnlyBundleSelfTest:
    """Deterministic result of reloading and executing frozen bundle bytes."""

    output_sha256: str
    model_max_abs_errors: tuple[tuple[str, float], ...]
    row_count: int
    passed: bool


@dataclass(frozen=True, slots=True)
class LoadedRawExecution:
    """All frozen intermediates from raw inputs through the O7 continuation."""

    model_id: str
    chart_ids: Tensor
    support_mask: Tensor
    z: Tensor
    r: Tensor
    reconstructed_message: Tensor
    q_base_heads: Tensor
    c_base: Tensor
    coefficients: Tensor
    q_heads: Tensor
    q_total: Tensor
    v_attention: Tensor
    h_a: Tensor
    q_mlp: Tensor
    v_mlp: Tensor
    h_plus: Tensor
    logits: Tensor


def _raw32(value: str) -> np.ndarray:
    if len(value) != 64:
        raise QualificationError("bundle identity must be a SHA-256")
    try:
        content = bytes.fromhex(value)
    except ValueError as error:
        raise QualificationError("bundle identity must be a SHA-256") from error
    return np.frombuffer(content, dtype=np.uint8).copy()


def _utf8(value: str) -> np.ndarray:
    if not value:
        raise QualificationError("bundle text identity cannot be empty")
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode_utf8(arrays: Mapping[str, np.ndarray], name: str) -> str:
    value = arrays.get(name)
    if value is None or value.ndim != 1 or value.dtype != np.dtype("uint8"):
        raise QualificationError(f"bundle tensor {name!r} is missing or malformed")
    try:
        decoded = value.tobytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise QualificationError(f"bundle tensor {name!r} is not UTF-8") from error
    if not decoded:
        raise QualificationError(f"bundle tensor {name!r} is empty")
    return decoded


def _numpy_arrays_sha256(
    arrays: Mapping[str, np.ndarray], *, domain: bytes
) -> str:
    if not arrays:
        raise QualificationError("cannot hash an empty bundle tensor set")
    return hashlib.sha256(domain + b"\0" + encode_safetensors(arrays)).hexdigest()


_SOURCE_EXECUTION_NAMES: Final = (
    "source.attention_output.weight",
    "source.carrier.vectors",
    "source.h4_fisher_diagonal",
    "source.lm_head.weight",
    "source.mlp.down.weight",
    "source.mlp.gate.weight",
    "source.mlp.up.weight",
    "source.post_attention_norm.weight",
    "source.post_mlp_norm.weight",
    "source.rmsnorm_epsilon",
)
_SYNTHETIC_SOURCE_PARAMETER_NAMES: Final = (
    "activation_basis",
    "chart_offsets",
    "message_base_basis",
    "message_interaction_basis",
    "output_weights",
    "q_carrier",
    "interaction_z",
    "interaction_r",
    "post_attention_norm_weight",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "post_mlp_norm_weight",
    "lm_head",
    "h4_fisher_diagonal",
)


def _source_execution_arrays(campaign: SyntheticCampaignData) -> dict[str, np.ndarray]:
    source = campaign.source
    return {
        "source.attention_output.weight": np.asarray(source.output_weights).copy(),
        "source.carrier.vectors": np.asarray(source.q_carrier).copy(),
        "source.h4_fisher_diagonal": np.asarray(
            source.h4_fisher_diagonal
        ).copy(),
        "source.lm_head.weight": np.asarray(source.lm_head).copy(),
        "source.mlp.down.weight": np.asarray(source.mlp_down).copy(),
        "source.mlp.gate.weight": np.asarray(source.mlp_gate).copy(),
        "source.mlp.up.weight": np.asarray(source.mlp_up).copy(),
        "source.post_attention_norm.weight": np.asarray(
            source.post_attention_norm_weight
        ).copy(),
        "source.post_mlp_norm.weight": np.asarray(
            source.post_mlp_norm_weight
        ).copy(),
        "source.rmsnorm_epsilon": np.asarray([1e-6], dtype=np.float64),
    }


def _selection_execution_identity_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    prefixes = (
        "representation.",
        "support.",
        "message_sketch.",
        "codec.",
        "interaction_mean.",
    )
    exact = {"execution.chart_ids", "execution.codec_ranks"}
    return {
        name: value
        for name, value in arrays.items()
        if name in exact or name.startswith(prefixes)
    }


def _model_identity_arrays(
    arrays: Mapping[str, np.ndarray], model_id: str
) -> dict[str, np.ndarray]:
    prefix = f"{model_id}."
    return {
        name[len(prefix) :]: value
        for name, value in arrays.items()
        if name.startswith(prefix) and name != "J1.analytic_c_base_passthrough"
    }


def _deterministic_self_test_inputs(
    *,
    geometry: SelectionGeometry,
    codec_id: str,
    campaign: SyntheticCampaignData,
) -> dict[str, np.ndarray]:
    """Create corpus-independent raw tensors covering every active chart."""

    locator = geometry.locator
    charts = tuple(range(locator.chart_count))
    hidden = campaign.config.hidden_dimension
    raw_h = np.empty((len(charts), hidden), dtype=np.float64)
    for row, chart in enumerate(charts):
        # Frozen k-means centers are public fitted execution state rather than
        # corpus rows.  Reconstructing them supplies one deterministic witness
        # for each chart without embedding any panel payload in the bundle.
        center = locator.centers[chart : chart + 1]
        candidate = locator.global_representation.mean + (
            center @ locator.global_representation.basis.T
        )[0]
        assigned = int(locator.assign(candidate[None, :])[0])
        if assigned != chart:
            candidate = locator.local_representations[chart].mean
            assigned = int(locator.assign(candidate[None, :])[0])
        if assigned != chart:
            raise QualificationError(
                f"cannot derive corpus-independent self-test witness for chart {chart}"
            )
        raw_h[row] = candidate.detach().cpu().numpy()

    r_index = np.arange(
        len(charts) * campaign.config.head_count * campaign.config.compact_dimension,
        dtype=np.float64,
    ).reshape(
        len(charts),
        campaign.config.head_count,
        campaign.config.compact_dimension,
    )
    intended_r = torch.from_numpy(0.1 * np.sin(r_index + 0.75)).to(torch.float64)
    raw_message = geometry.message_sketch.reconstruct(intended_r)
    chart_ids, z = locator.transform(torch.from_numpy(raw_h))
    expected_charts = torch.tensor(charts, dtype=torch.int64)
    if not torch.equal(chart_ids, expected_charts):
        raise QualificationError("bundle self-test chart witnesses are not stable")
    r = geometry.message_sketch.transform(raw_message)
    reconstructed = geometry.message_sketch.reconstruct(r)
    q_base = torch.einsum(
        "ahm,nam->nah",
        torch.from_numpy(campaign.source.output_weights).to(torch.float64),
        reconstructed,
    )
    c_base = geometry.codec(codec_id).encode_q(q_base, chart_ids)
    return {
        "execution.self_test.raw_h": raw_h,
        "execution.self_test.raw_message": raw_message.detach().cpu().numpy(),
        "execution.self_test.c_base": c_base.to(torch.float32).cpu().numpy(),
        "execution.self_test.z": z.to(torch.float32).cpu().numpy(),
        "execution.self_test.r": r.to(torch.float32).cpu().numpy(),
        "execution.self_test.chart_ids": expected_charts.cpu().numpy(),
    }


def _reference_raw_execution(
    model_id: str,
    fits: Mapping[str, ModelBankFit],
    inputs: Mapping[str, np.ndarray],
    geometry: SelectionGeometry,
    codec_id: str,
    campaign: SyntheticCampaignData,
) -> dict[str, np.ndarray]:
    h = torch.from_numpy(inputs["execution.self_test.raw_h"].copy()).to(
        torch.float64
    )
    message = torch.from_numpy(
        inputs["execution.self_test.raw_message"].copy()
    ).to(torch.float64)
    chart_ids, z, _ = geometry.locator.transform_with_support(h)
    r = geometry.message_sketch.transform(message)
    reconstructed = geometry.message_sketch.reconstruct(r)
    q_base = torch.einsum(
        "ahm,nam->nah",
        torch.from_numpy(campaign.source.output_weights).to(torch.float64),
        reconstructed,
    )
    codec = geometry.codec(codec_id)
    c_base = codec.encode_q(q_base, chart_ids)
    if model_id == "J1":
        coefficients = c_base.to(torch.float32).to(torch.float64)
    else:
        bank = fits[model_id].bank
        parameter = next(bank.parameters(), None)
        dtype = torch.float32 if parameter is None else parameter.dtype
        with torch.no_grad():
            coefficients = bank(
                c_base.to(dtype),
                z.to(dtype),
                r.to(dtype),
                chart_ids,
            ).to(torch.float64)
    q_heads = codec.decode_q(coefficients, chart_ids)
    q_total = q_heads.sum(dim=1)
    v_attention = torch.from_numpy(
        synthetic_rmsnorm(
            q_total.detach().cpu().numpy(),
            campaign.source.post_attention_norm_weight,
        )
    ).to(torch.float64)
    h_a = h + v_attention
    q_mlp_array, v_mlp_array = exact_synthetic_mlp(
        h_a.detach().cpu().numpy(), campaign.source
    )
    q_mlp = torch.from_numpy(q_mlp_array).to(torch.float64)
    v_mlp = torch.from_numpy(v_mlp_array).to(torch.float64)
    h_plus = h_a + v_mlp
    logits = h_plus @ torch.from_numpy(campaign.source.lm_head).to(torch.float64).T
    return {
        "z": z.detach().cpu().numpy(),
        "r": r.detach().cpu().numpy(),
        "reconstructed_message": reconstructed.detach().cpu().numpy(),
        "q_base_heads": q_base.detach().cpu().numpy(),
        "c_base": c_base.detach().cpu().numpy(),
        "coefficients": coefficients.detach().cpu().numpy(),
        "q_heads": q_heads.detach().cpu().numpy(),
        "q_total": q_total.detach().cpu().numpy(),
        "h_A": h_a.detach().cpu().numpy(),
        "q_M": q_mlp.detach().cpu().numpy(),
        "h_plus": h_plus.detach().cpu().numpy(),
        "logits": logits.detach().cpu().numpy(),
    }


def _self_test_expected_arrays(
    fits: Mapping[str, ModelBankFit],
    inputs: Mapping[str, np.ndarray],
    model_ids: Sequence[str],
    *,
    geometry: SelectionGeometry,
    codec_id: str,
    campaign: SyntheticCampaignData,
) -> dict[str, np.ndarray]:
    arrays = dict(inputs)
    for model_id in model_ids:
        expected = _reference_raw_execution(
            model_id, fits, inputs, geometry, codec_id, campaign
        )
        # Retain the original coefficient-output tensor name for old readers.
        arrays[f"execution.self_test.expected.{model_id}"] = expected[
            "coefficients"
        ]
        for stage, value in expected.items():
            arrays[f"execution.self_test.expected.{stage}.{model_id}"] = value
    return arrays


def _selection_bundle_arrays(
    geometry: SelectionGeometry,
    codec_id: str,
    interaction_means: FrozenInteractionMeans,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for locator_name, locator in (
        ("A_in", geometry.locator),
        ("A_mid", geometry.mid_locator),
        ("A_out", geometry.out_locator),
    ):
        global_prefix = f"representation.{locator_name}.global"
        arrays[f"{global_prefix}.mean"] = locator.global_representation.mean.numpy()
        arrays[f"{global_prefix}.basis"] = locator.global_representation.basis.numpy()
        arrays[f"{global_prefix}.coordinate_mean"] = (
            locator.global_representation.coordinate_mean.numpy()
        )
        arrays[f"{global_prefix}.coordinate_scale"] = (
            locator.global_representation.coordinate_scale.numpy()
        )
        arrays[f"representation.{locator_name}.kmeans_centers"] = (
            locator.centers.numpy()
        )
        for chart, representation in enumerate(locator.local_representations):
            prefix = f"representation.{locator_name}.chart{chart}"
            arrays[f"{prefix}.mean"] = representation.mean.numpy()
            arrays[f"{prefix}.basis"] = representation.basis.numpy()
            arrays[f"{prefix}.coordinate_mean"] = representation.coordinate_mean.numpy()
            arrays[f"{prefix}.coordinate_scale"] = representation.coordinate_scale.numpy()
            support = locator.support_covariances[chart]
            support_prefix = f"support.{locator_name}.chart{chart}"
            arrays[f"{support_prefix}.covariance"] = support.symmetric_covariance.numpy()
            arrays[f"{support_prefix}.pseudoinverse"] = support.pseudoinverse.numpy()
            arrays[f"{support_prefix}.active"] = support.active.numpy()
            arrays[f"{support_prefix}.sigma_perp_squared"] = np.asarray(
                [float(locator.sigma_perp_squared[chart])], dtype=np.float64
            )
            arrays[f"{support_prefix}.parallel_limit"] = np.asarray(
                [float(locator.parallel_support_limits[chart])], dtype=np.float64
            )
            arrays[f"{support_prefix}.perpendicular_limit"] = np.asarray(
                [float(locator.perpendicular_support_limits[chart])], dtype=np.float64
            )
    for head, representation in enumerate(geometry.message_sketch.representations):
        prefix = f"message_sketch.head{head}"
        arrays[f"{prefix}.mean"] = representation.mean.numpy()
        arrays[f"{prefix}.basis"] = representation.basis.numpy()
        arrays[f"{prefix}.coordinate_mean"] = representation.coordinate_mean.numpy()
        arrays[f"{prefix}.coordinate_scale"] = representation.coordinate_scale.numpy()
    codec = geometry.codec(codec_id)
    for kind, codecs in (("q", codec.q_codecs), ("v", codec.v_codecs)):
        for chart, fitted in enumerate(codecs):
            prefix = f"codec.{kind}.chart{chart}"
            arrays[f"{prefix}.encoder"] = fitted.encoder.numpy()
            arrays[f"{prefix}.decoder"] = fitted.decoder.numpy()
            arrays[f"{prefix}.projector"] = fitted.projector.numpy()
    arrays["codec.epsilon_completion_by_chart_head"] = (
        codec.epsilon_completion_by_chart_head.numpy()
    )
    arrays["codec.epsilon_j0_by_chart_head"] = codec.epsilon_j0_by_chart_head.numpy()
    arrays["interaction_mean.values"] = interaction_means.values.numpy()
    arrays["execution.chart_ids"] = np.asarray(
        tuple(range(geometry.locator.chart_count)), dtype=np.int64
    )
    arrays["execution.codec_ranks"] = np.asarray(
        [codec.q_rank, codec.v_rank], dtype=np.int64
    )
    arrays["execution.representation_identity_raw32"] = np.frombuffer(
        bytes.fromhex(geometry.identity_sha256), dtype=np.uint8
    ).copy()
    arrays["execution.interaction_mean_identity_raw32"] = np.frombuffer(
        bytes.fromhex(interaction_means.identity_sha256), dtype=np.uint8
    ).copy()
    return arrays


def build_weights_only_bundles(
    fits: Mapping[str, ModelBankFit],
    *,
    baseline_id: str,
    geometry: SelectionGeometry,
    codec_id: str,
    interaction_means: FrozenInteractionMeans,
    campaign: SyntheticCampaignData,
) -> tuple[bytes, bytes]:
    """Return complete deterministic candidate/comparator safetensors.

    The candidate contains every frozen tensor needed to begin with raw
    ``h``/per-head messages and continue through the exact source MLP and LM
    head.  The comparator contains every required control bank and binds the
    candidate's shared execution identities.  No Python object serialization
    or implicit access to the source campaign is required after this call.
    """

    required_comparators = ("J2N", "J2+", "J3")
    if (
        baseline_id not in {"J1", *required_comparators}
        or "J4" not in fits
        or any(name not in fits for name in required_comparators)
    ):
        raise QualificationError("candidate/comparator fits are incomplete")
    reference_bank = fits["J4"].bank
    inputs = _deterministic_self_test_inputs(
        geometry=geometry,
        codec_id=codec_id,
        campaign=campaign,
    )
    candidate_arrays = _selection_bundle_arrays(
        geometry, codec_id, interaction_means
    )
    source_arrays = _source_execution_arrays(campaign)
    candidate_arrays.update(source_arrays)
    candidate_arrays.update(_bundle_arrays({"J4": fits["J4"]}))
    candidate_arrays.update(
        _self_test_expected_arrays(
            fits,
            inputs,
            ("J4",),
            geometry=geometry,
            codec_id=codec_id,
            campaign=campaign,
        )
    )

    codec = geometry.codec(codec_id)
    dimensions = np.asarray(
        [
            geometry.locator.chart_count,
            reference_bank.head_count,
            campaign.config.hidden_dimension,
            campaign.config.message_dimension,
            campaign.config.compact_dimension,
            geometry.message_sketch.rank,
            codec.q_rank,
            codec.v_rank,
            campaign.config.interaction_rank,
            campaign.config.vocabulary_size,
        ],
        dtype=np.int64,
    )
    source_identity = _numpy_arrays_sha256(
        source_arrays,
        domain=b"fisher-cmc/synthetic-source-execution-state/v1",
    )
    source_model_identity = hashlib.sha256(
        encode_safetensors(
            {
                name: np.asarray(getattr(campaign.source, name))
                for name in _SYNTHETIC_SOURCE_PARAMETER_NAMES
            }
        )
    ).hexdigest()
    carrier_artifact_identity = hashlib.sha256(
        encode_safetensors(
            {
                "h4_fisher_diagonal": np.asarray(
                    campaign.source.h4_fisher_diagonal, dtype=np.float64
                ),
                "h4_fisher_eigenvectors": np.asarray(
                    campaign.source.q_carrier, dtype=np.float64
                ),
            }
        )
    ).hexdigest()
    carrier_identity = _numpy_arrays_sha256(
        {
            "source.carrier.vectors": source_arrays["source.carrier.vectors"],
            "source.h4_fisher_diagonal": source_arrays[
                "source.h4_fisher_diagonal"
            ],
        },
        domain=b"fisher-cmc/synthetic-source-carrier/v1",
    )
    selection_identity = _numpy_arrays_sha256(
        _selection_execution_identity_arrays(candidate_arrays),
        domain=b"fisher-cmc/synthetic-selection-execution-state/v1",
    )
    shared_metadata = {
        "metadata.bundle_format_version": np.asarray([2], dtype=np.int64),
        "metadata.codec_id_utf8": _utf8(codec_id),
        "metadata.dimensions": dimensions,
        "metadata.execution_protocol_sha256_raw32": _raw32(
            RAW_EXECUTION_PROTOCOL_SHA256
        ),
        "metadata.row_derivation_protocol_sha256_raw32": _raw32(
            synthetic_row_derivation_protocol_sha256()
        ),
        "metadata.representation_identity_raw32": _raw32(
            geometry.identity_sha256
        ),
        "metadata.locator_A_in_identity_raw32": _raw32(
            geometry.locator.identity_sha256
        ),
        "metadata.message_sketch_identity_raw32": _raw32(
            geometry.message_sketch.identity_sha256
        ),
        "metadata.selected_codec_identity_raw32": _raw32(codec.identity_sha256),
        "metadata.interaction_mean_identity_raw32": _raw32(
            interaction_means.identity_sha256
        ),
        "metadata.selection_execution_identity_raw32": _raw32(
            selection_identity
        ),
        "metadata.source_execution_identity_raw32": _raw32(source_identity),
        "metadata.source_model_identity_raw32": _raw32(source_model_identity),
        "metadata.carrier_identity_raw32": _raw32(carrier_identity),
        "metadata.carrier_artifact_identity_raw32": _raw32(
            carrier_artifact_identity
        ),
        "metadata.tokenizer_identity_raw32": _raw32(
            hashlib.sha256(
                b"fisher-cmc/synthetic-tokenizer/no-tokenizer/v1"
            ).hexdigest()
        ),
    }
    candidate_arrays.update(shared_metadata)
    candidate_arrays["metadata.bundle_role_utf8"] = _utf8("candidate")
    for field, value in (
        ("architecture", fits["J4"].architecture_sha256),
        ("weights", fits["J4"].weights_sha256),
    ):
        candidate_arrays[f"metadata.model.J4.{field}_sha256_raw32"] = _raw32(
            value
        )

    diagnostic_ids = tuple(
        name for name in ("J0", "J2") if name in fits
    )
    comparator_model_ids = (*required_comparators, *diagnostic_ids)
    comparator_arrays = _bundle_arrays(
        {name: fits[name] for name in comparator_model_ids}
    )
    comparator_arrays.update(
        _self_test_expected_arrays(
            fits,
            inputs,
            ("J1", "J2N", "J2+", "J3"),
            geometry=geometry,
            codec_id=codec_id,
            campaign=campaign,
        )
    )
    comparator_arrays["J1.analytic_c_base_passthrough"] = np.asarray(
        [1], dtype=np.uint8
    )
    comparator_arrays["metadata.frozen_baseline_code"] = np.asarray(
        [("J1", "J2N", "J2+", "J3").index(baseline_id)], dtype=np.int64
    )
    comparator_arrays.update(
        {name: value.copy() for name, value in shared_metadata.items()}
    )
    comparator_arrays["metadata.bundle_role_utf8"] = _utf8("comparator")
    comparator_arrays["metadata.baseline_id_utf8"] = _utf8(baseline_id)
    for model_id in ("J1", *comparator_model_ids):
        if model_id == "J1" and model_id not in fits:
            analytic = make_frozen_completion_bank(
                "J1",
                reference_bank.chart_ids,
                reference_bank.head_count,
                reference_bank.coefficient_dimension,
                initialization_seed=0,
            )
            architecture_sha256 = _architecture_sha256(analytic, "J1")
            weights_sha256 = _state_sha256(
                analytic.state_dict(),
                domain=b"fisher-cmc/synthetic-J1-state/v1",
            )
        else:
            fit = fits[model_id]
            architecture_sha256 = fit.architecture_sha256
            weights_sha256 = fit.weights_sha256
        for field, value in (
            ("architecture", architecture_sha256),
            ("weights", weights_sha256),
        ):
            comparator_arrays[
                f"metadata.model.{model_id}.{field}_sha256_raw32"
            ] = _raw32(value)

    candidate = encode_safetensors(candidate_arrays)
    comparator = encode_safetensors(comparator_arrays)
    return candidate, comparator


def _required_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype[object] | None = None,
) -> np.ndarray:
    value = arrays.get(name)
    if value is None:
        raise QualificationError(f"bundle tensor {name!r} is missing")
    if shape is not None and value.shape != shape:
        raise QualificationError(f"bundle tensor {name!r} has an invalid shape")
    if dtype is not None and value.dtype != dtype:
        raise QualificationError(f"bundle tensor {name!r} has an invalid dtype")
    return value


def _identity_hex(arrays: Mapping[str, np.ndarray], name: str) -> str:
    value = _required_array(
        arrays, name, shape=(32,), dtype=np.dtype("uint8")
    )
    return value.tobytes().hex()


def _load_bundle_bank(
    arrays: Mapping[str, np.ndarray],
    model_id: str,
    *,
    chart_ids: tuple[int, ...],
    head_count: int,
    coefficient_dimension: int,
) -> ChartHeadCompletionBank:
    bank = make_frozen_completion_bank(
        model_id,
        chart_ids,
        head_count,
        coefficient_dimension,
        initialization_seed=0,
    )
    if model_id == "J1":
        marker = _required_array(
            arrays,
            "J1.analytic_c_base_passthrough",
            shape=(1,),
            dtype=np.dtype("uint8"),
        )
        if int(marker[0]) != 1:
            raise QualificationError("analytic J1 marker is invalid")
    else:
        state_arrays = _model_identity_arrays(arrays, model_id)
        expected_state = bank.state_dict()
        if set(state_arrays) != set(expected_state):
            raise QualificationError(
                f"bundle state for {model_id} is incomplete or contains extras"
            )
        state: dict[str, Tensor] = {}
        for name, expected in expected_state.items():
            value = state_arrays[name]
            expected_dtype = expected.detach().cpu().numpy().dtype
            if value.shape != tuple(expected.shape) or value.dtype != expected_dtype:
                raise QualificationError(
                    f"bundle state tensor {model_id}.{name} has an invalid shape or dtype"
                )
            state[name] = torch.from_numpy(np.ascontiguousarray(value).copy())
        try:
            bank.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise QualificationError(
                f"bundle state for {model_id} cannot be loaded"
            ) from error
    expected_architecture = _identity_hex(
        arrays, f"metadata.model.{model_id}.architecture_sha256_raw32"
    )
    if _architecture_sha256(bank, model_id) != expected_architecture:
        raise QualificationError(f"bundle architecture identity for {model_id} differs")
    expected_weights = _identity_hex(
        arrays, f"metadata.model.{model_id}.weights_sha256_raw32"
    )
    weight_domain = (
        b"fisher-cmc/synthetic-J1-state/v1"
        if model_id == "J1"
        else f"fisher-cmc/synthetic-{model_id}-weights/v1".encode()
    )
    if _state_sha256(bank.state_dict(), domain=weight_domain) != expected_weights:
        raise QualificationError(f"bundle weights identity for {model_id} differs")
    bank.requires_grad_(False)
    bank.eval()
    return bank


def _validate_candidate_execution_state(
    arrays: Mapping[str, np.ndarray], dimensions: tuple[int, ...]
) -> None:
    (
        chart_count,
        head_count,
        hidden_dimension,
        message_dimension,
        activation_dimension,
        message_rank,
        q_rank,
        v_rank,
        interaction_rank,
        vocabulary_size,
    ) = dimensions
    if (
        activation_dimension != 32
        or message_rank != 32
        or q_rank not in {32, 64}
        or v_rank not in {32, 64}
        or interaction_rank != 8
    ):
        raise QualificationError("bundle dimensions violate the frozen architecture")
    floating_shapes = {
        "source.attention_output.weight": (
            head_count,
            hidden_dimension,
            message_dimension,
        ),
        "source.carrier.vectors": (hidden_dimension, hidden_dimension),
        "source.h4_fisher_diagonal": (hidden_dimension,),
        "source.lm_head.weight": (vocabulary_size, hidden_dimension),
        "source.post_attention_norm.weight": (hidden_dimension,),
        "source.post_mlp_norm.weight": (hidden_dimension,),
        "source.rmsnorm_epsilon": (1,),
    }
    for name, shape in floating_shapes.items():
        value = _required_array(arrays, name, shape=shape)
        if not np.issubdtype(value.dtype, np.floating) or not bool(
            np.isfinite(value).all()
        ):
            raise QualificationError(f"bundle tensor {name!r} must be finite floating")
    gate = _required_array(arrays, "source.mlp.gate.weight")
    up = _required_array(arrays, "source.mlp.up.weight")
    down = _required_array(arrays, "source.mlp.down.weight")
    if (
        gate.ndim != 2
        or gate.shape[1] != hidden_dimension
        or up.shape != gate.shape
        or down.shape != (hidden_dimension, gate.shape[0])
        or any(
            not np.issubdtype(value.dtype, np.floating)
            or not bool(np.isfinite(value).all())
            for value in (gate, up, down)
        )
    ):
        raise QualificationError("bundle source MLP tensors are malformed")
    epsilon = float(arrays["source.rmsnorm_epsilon"][0])
    if epsilon != 1e-6:
        raise QualificationError("bundle RMSNorm epsilon differs from protocol")
    fisher = arrays["source.h4_fisher_diagonal"]
    if bool((fisher <= 0.0).any()):
        raise QualificationError("bundle Fisher diagonal must be positive")

    chart_ids = _required_array(
        arrays,
        "execution.chart_ids",
        shape=(chart_count,),
        dtype=np.dtype("int64"),
    )
    if not np.array_equal(chart_ids, np.arange(chart_count, dtype=np.int64)):
        raise QualificationError("bundle chart catalog must be contiguous and ordered")
    ranks = _required_array(
        arrays,
        "execution.codec_ranks",
        shape=(2,),
        dtype=np.dtype("int64"),
    )
    if not np.array_equal(ranks, np.asarray([q_rank, v_rank], dtype=np.int64)):
        raise QualificationError("bundle codec ranks differ from dimensions")
    for chart in range(chart_count):
        prefix = f"representation.A_in.chart{chart}"
        for suffix, shape in (
            ("mean", (hidden_dimension,)),
            ("basis", (hidden_dimension, activation_dimension)),
            ("coordinate_mean", (activation_dimension,)),
            ("coordinate_scale", (activation_dimension,)),
        ):
            value = _required_array(arrays, f"{prefix}.{suffix}", shape=shape)
            if not np.issubdtype(value.dtype, np.floating) or not bool(
                np.isfinite(value).all()
            ):
                raise QualificationError("bundle A_in tensors must be finite floating")
        if bool((arrays[f"{prefix}.coordinate_scale"] <= 0.0).any()):
            raise QualificationError("bundle A_in coordinate scales must be positive")
        support_prefix = f"support.A_in.chart{chart}"
        _required_array(
            arrays,
            f"{support_prefix}.pseudoinverse",
            shape=(activation_dimension, activation_dimension),
        )
        for suffix in (
            "sigma_perp_squared",
            "parallel_limit",
            "perpendicular_limit",
        ):
            scalar = _required_array(
                arrays, f"{support_prefix}.{suffix}", shape=(1,)
            )
            if not np.issubdtype(scalar.dtype, np.floating) or not bool(
                np.isfinite(scalar).all()
            ):
                raise QualificationError("bundle support scalars must be finite")
        if float(arrays[f"{support_prefix}.sigma_perp_squared"][0]) <= 0.0:
            raise QualificationError("bundle normal support scale must be positive")
        for head in range(head_count):
            for suffix, shape in (
                ("encoder", (q_rank, hidden_dimension)),
                ("decoder", (hidden_dimension, q_rank)),
                ("projector", (hidden_dimension, hidden_dimension)),
            ):
                value = _required_array(
                    arrays, f"codec.q.chart{chart}.{suffix}", shape=shape
                )
                if not np.issubdtype(value.dtype, np.floating) or not bool(
                    np.isfinite(value).all()
                ):
                    raise QualificationError("bundle q codec tensors must be finite")
    for head in range(head_count):
        prefix = f"message_sketch.head{head}"
        for suffix, shape in (
            ("mean", (message_dimension,)),
            ("basis", (message_dimension, message_rank)),
            ("coordinate_mean", (message_rank,)),
            ("coordinate_scale", (message_rank,)),
        ):
            value = _required_array(arrays, f"{prefix}.{suffix}", shape=shape)
            if not np.issubdtype(value.dtype, np.floating) or not bool(
                np.isfinite(value).all()
            ):
                raise QualificationError("bundle message sketch tensors must be finite")
        if bool((arrays[f"{prefix}.coordinate_scale"] <= 0.0).any()):
            raise QualificationError("bundle message coordinate scales must be positive")


@dataclass(frozen=True, slots=True)
class LoadedWeightsOnlyBundles:
    """Strictly decoded frozen execution state with no source-object fallback."""

    candidate_sha256: str
    comparator_sha256: str
    baseline_id: str
    codec_id: str
    chart_ids: tuple[int, ...]
    head_count: int
    hidden_dimension: int
    message_dimension: int
    coefficient_dimension: int
    interaction_rank: int
    source_execution_identity_sha256: str
    execution_protocol_sha256: str
    banks: Mapping[str, ChartHeadCompletionBank]
    candidate_tensors: Mapping[str, np.ndarray]
    comparator_tensors: Mapping[str, np.ndarray]

    def bank(self, model_id: str) -> ChartHeadCompletionBank:
        try:
            return self.banks[model_id]
        except KeyError as error:
            raise QualificationError(
                f"model {model_id!r} is absent from the frozen bundles"
            ) from error

    def execute_raw(
        self,
        model_id: str,
        h: np.ndarray | Tensor,
        message: np.ndarray | Tensor,
        *,
        require_support: bool = False,
    ) -> LoadedRawExecution:
        """Execute raw ``h``/messages through reconstruction and exact O7."""

        h_tensor = (
            torch.from_numpy(np.asarray(h).copy())
            if isinstance(h, np.ndarray)
            else h.detach().cpu().clone()
        ).to(torch.float64)
        message_tensor = (
            torch.from_numpy(np.asarray(message).copy())
            if isinstance(message, np.ndarray)
            else message.detach().cpu().clone()
        ).to(torch.float64)
        if h_tensor.ndim != 2 or h_tensor.shape[1] != self.hidden_dimension:
            raise QualificationError("raw h must have shape [rows, hidden_dimension]")
        if message_tensor.shape != (
            h_tensor.shape[0],
            self.head_count,
            self.message_dimension,
        ):
            raise QualificationError(
                "raw message must have shape [rows, heads, message_dimension]"
            )
        if not bool(torch.isfinite(h_tensor).all()) or not bool(
            torch.isfinite(message_tensor).all()
        ):
            raise QualificationError("raw bundle inputs must be finite")

        arrays = self.candidate_tensors
        parallel: list[Tensor] = []
        perpendicular: list[Tensor] = []
        for chart in self.chart_ids:
            prefix = f"representation.A_in.chart{chart}"
            mean = torch.from_numpy(arrays[f"{prefix}.mean"].copy()).to(torch.float64)
            basis = torch.from_numpy(arrays[f"{prefix}.basis"].copy()).to(torch.float64)
            centered = h_tensor - mean
            raw = centered @ basis
            pseudoinverse = torch.from_numpy(
                arrays[f"support.A_in.chart{chart}.pseudoinverse"].copy()
            ).to(torch.float64)
            parallel.append(
                torch.einsum("ni,ij,nj->n", raw, pseudoinverse, raw)
            )
            residual = centered - raw @ basis.T
            sigma = float(
                arrays[f"support.A_in.chart{chart}.sigma_perp_squared"][0]
            )
            perpendicular.append(residual.square().sum(-1) / sigma)
        parallel_tensor = torch.stack(parallel, dim=1)
        perpendicular_tensor = torch.stack(perpendicular, dim=1)
        chart_indexes = torch.argmin(
            parallel_tensor + perpendicular_tensor, dim=1
        )
        catalog = torch.tensor(self.chart_ids, dtype=torch.int64)
        chart_ids = catalog[chart_indexes]
        row_indexes = torch.arange(h_tensor.shape[0], dtype=torch.int64)
        parallel_limits = torch.tensor(
            [
                float(arrays[f"support.A_in.chart{chart}.parallel_limit"][0])
                for chart in self.chart_ids
            ],
            dtype=torch.float64,
        )
        perpendicular_limits = torch.tensor(
            [
                float(
                    arrays[
                        f"support.A_in.chart{chart}.perpendicular_limit"
                    ][0]
                )
                for chart in self.chart_ids
            ],
            dtype=torch.float64,
        )
        support_mask = (
            parallel_tensor[row_indexes, chart_indexes]
            <= parallel_limits[chart_indexes]
        ) & (
            perpendicular_tensor[row_indexes, chart_indexes]
            <= perpendicular_limits[chart_indexes]
        )
        if require_support and not bool(support_mask.all()):
            raise QualificationError("raw bundle input is outside frozen A_in support")

        z = torch.empty(
            (h_tensor.shape[0], 32), dtype=torch.float64
        )
        for chart in self.chart_ids:
            mask = chart_ids == chart
            if not bool(mask.any()):
                continue
            prefix = f"representation.A_in.chart{chart}"
            mean = torch.from_numpy(arrays[f"{prefix}.mean"].copy()).to(torch.float64)
            basis = torch.from_numpy(arrays[f"{prefix}.basis"].copy()).to(torch.float64)
            coordinate_mean = torch.from_numpy(
                arrays[f"{prefix}.coordinate_mean"].copy()
            ).to(torch.float64)
            coordinate_scale = torch.from_numpy(
                arrays[f"{prefix}.coordinate_scale"].copy()
            ).to(torch.float64)
            z[mask] = (
                (h_tensor[mask] - mean) @ basis - coordinate_mean
            ) / coordinate_scale

        r_heads: list[Tensor] = []
        reconstructed_heads: list[Tensor] = []
        for head in range(self.head_count):
            prefix = f"message_sketch.head{head}"
            mean = torch.from_numpy(arrays[f"{prefix}.mean"].copy()).to(torch.float64)
            basis = torch.from_numpy(arrays[f"{prefix}.basis"].copy()).to(torch.float64)
            coordinate_mean = torch.from_numpy(
                arrays[f"{prefix}.coordinate_mean"].copy()
            ).to(torch.float64)
            coordinate_scale = torch.from_numpy(
                arrays[f"{prefix}.coordinate_scale"].copy()
            ).to(torch.float64)
            coordinates = (message_tensor[:, head, :] - mean) @ basis
            standardized = (coordinates - coordinate_mean) / coordinate_scale
            r_heads.append(standardized)
            reconstructed_heads.append(
                mean + (coordinate_mean + standardized * coordinate_scale) @ basis.T
            )
        r = torch.stack(r_heads, dim=1)
        reconstructed_message = torch.stack(reconstructed_heads, dim=1)
        output_weights = torch.from_numpy(
            arrays["source.attention_output.weight"].copy()
        ).to(torch.float64)
        q_base_heads = torch.einsum(
            "ahm,nam->nah", output_weights, reconstructed_message
        )
        c_base = torch.empty(
            (
                h_tensor.shape[0],
                self.head_count,
                self.coefficient_dimension,
            ),
            dtype=torch.float64,
        )
        for chart in self.chart_ids:
            mask = chart_ids == chart
            if not bool(mask.any()):
                continue
            encoder = torch.from_numpy(
                arrays[f"codec.q.chart{chart}.encoder"].copy()
            ).to(torch.float64)
            c_base[mask] = q_base_heads[mask] @ encoder.T

        bank = self.bank(model_id)
        parameter = next(bank.parameters(), None)
        dtype = torch.float32 if parameter is None else parameter.dtype
        with torch.no_grad():
            coefficients = bank(
                c_base.to(dtype),
                z.to(dtype),
                r.to(dtype),
                chart_ids,
            ).to(torch.float64)
        q_heads = torch.empty(
            (h_tensor.shape[0], self.head_count, self.hidden_dimension),
            dtype=torch.float64,
        )
        for chart in self.chart_ids:
            mask = chart_ids == chart
            if not bool(mask.any()):
                continue
            decoder = torch.from_numpy(
                arrays[f"codec.q.chart{chart}.decoder"].copy()
            ).to(torch.float64)
            q_heads[mask] = coefficients[mask] @ decoder.T
        q_total = q_heads.sum(dim=1)
        epsilon = float(arrays["source.rmsnorm_epsilon"][0])
        v_attention = torch.from_numpy(
            synthetic_rmsnorm(
                q_total.detach().cpu().numpy(),
                arrays["source.post_attention_norm.weight"],
                epsilon=epsilon,
            ).copy()
        ).to(torch.float64)
        h_a = h_tensor + v_attention

        # Preserve the fixture's exact contiguous-floor four-block teacher MLP
        # path, including NumPy operation ordering used by its capture seam.
        h_a_array = h_a.detach().cpu().numpy()
        gate = h_a_array @ arrays["source.mlp.gate.weight"].T
        up = h_a_array @ arrays["source.mlp.up.weight"].T
        clipped = np.clip(gate, -60.0, 60.0)
        intermediate = gate / (1.0 + np.exp(-clipped)) * up
        down = arrays["source.mlp.down.weight"]
        width = intermediate.shape[-1]
        boundaries = tuple((block * width) // 4 for block in range(5))
        blocks = np.stack(
            [
                intermediate[:, boundaries[block] : boundaries[block + 1]]
                @ down[:, boundaries[block] : boundaries[block + 1]].T
                for block in range(4)
            ],
            axis=1,
        )
        q_mlp_array = blocks.sum(axis=1)
        v_mlp_array = synthetic_rmsnorm(
            q_mlp_array,
            arrays["source.post_mlp_norm.weight"],
            epsilon=epsilon,
        )
        q_mlp = torch.from_numpy(q_mlp_array.copy()).to(torch.float64)
        v_mlp = torch.from_numpy(v_mlp_array.copy()).to(torch.float64)
        h_plus = h_a + v_mlp
        lm_head = torch.from_numpy(arrays["source.lm_head.weight"].copy()).to(
            torch.float64
        )
        logits = h_plus @ lm_head.T
        return LoadedRawExecution(
            model_id=model_id,
            chart_ids=chart_ids,
            support_mask=support_mask,
            z=z,
            r=r,
            reconstructed_message=reconstructed_message,
            q_base_heads=q_base_heads,
            c_base=c_base,
            coefficients=coefficients,
            q_heads=q_heads,
            q_total=q_total,
            v_attention=v_attention,
            h_a=h_a,
            q_mlp=q_mlp,
            v_mlp=v_mlp,
            h_plus=h_plus,
            logits=logits,
        )


def load_weights_only_bundles(
    candidate: bytes,
    comparator: bytes,
) -> LoadedWeightsOnlyBundles:
    """Strictly decode and reconstruct the complete frozen bundle pair."""

    candidate_arrays = decode_safetensors(candidate)
    comparator_arrays = decode_safetensors(comparator)
    if _decode_utf8(candidate_arrays, "metadata.bundle_role_utf8") != "candidate":
        raise QualificationError("candidate bundle role marker differs")
    if _decode_utf8(comparator_arrays, "metadata.bundle_role_utf8") != "comparator":
        raise QualificationError("comparator bundle role marker differs")
    for arrays in (candidate_arrays, comparator_arrays):
        version = _required_array(
            arrays,
            "metadata.bundle_format_version",
            shape=(1,),
            dtype=np.dtype("int64"),
        )
        if int(version[0]) != 2:
            raise QualificationError("unsupported weights-only bundle format")
    shared_names = (
        "metadata.bundle_format_version",
        "metadata.codec_id_utf8",
        "metadata.dimensions",
        "metadata.execution_protocol_sha256_raw32",
        "metadata.row_derivation_protocol_sha256_raw32",
        "metadata.representation_identity_raw32",
        "metadata.locator_A_in_identity_raw32",
        "metadata.message_sketch_identity_raw32",
        "metadata.selected_codec_identity_raw32",
        "metadata.interaction_mean_identity_raw32",
        "metadata.selection_execution_identity_raw32",
        "metadata.source_execution_identity_raw32",
        "metadata.source_model_identity_raw32",
        "metadata.carrier_identity_raw32",
        "metadata.carrier_artifact_identity_raw32",
        "metadata.tokenizer_identity_raw32",
    )
    for name in shared_names:
        left = _required_array(candidate_arrays, name)
        right = _required_array(comparator_arrays, name)
        if not np.array_equal(left, right):
            raise QualificationError(
                f"candidate/comparator shared binding {name!r} differs"
            )
    dimensions_array = _required_array(
        candidate_arrays,
        "metadata.dimensions",
        shape=(10,),
        dtype=np.dtype("int64"),
    )
    dimensions = tuple(int(value) for value in dimensions_array.tolist())
    if any(value <= 0 for value in dimensions):
        raise QualificationError("bundle dimensions must be positive")
    _validate_candidate_execution_state(candidate_arrays, dimensions)
    codec_id = _decode_utf8(candidate_arrays, "metadata.codec_id_utf8")
    baseline_id = _decode_utf8(comparator_arrays, "metadata.baseline_id_utf8")
    if baseline_id not in {"J1", "J2N", "J2+", "J3"}:
        raise QualificationError("frozen baseline ID is invalid")
    baseline_code = _required_array(
        comparator_arrays,
        "metadata.frozen_baseline_code",
        shape=(1,),
        dtype=np.dtype("int64"),
    )
    if int(baseline_code[0]) != ("J1", "J2N", "J2+", "J3").index(
        baseline_id
    ):
        raise QualificationError("frozen baseline code and ID differ")

    protocol_identity = _identity_hex(
        candidate_arrays, "metadata.execution_protocol_sha256_raw32"
    )
    if protocol_identity != RAW_EXECUTION_PROTOCOL_SHA256:
        raise QualificationError("bundle raw execution protocol differs")
    if _identity_hex(
        candidate_arrays, "metadata.row_derivation_protocol_sha256_raw32"
    ) != synthetic_row_derivation_protocol_sha256():
        raise QualificationError("bundle row-derivation protocol differs")
    if _identity_hex(
        candidate_arrays, "metadata.tokenizer_identity_raw32"
    ) != hashlib.sha256(
        b"fisher-cmc/synthetic-tokenizer/no-tokenizer/v1"
    ).hexdigest():
        raise QualificationError("bundle tokenizer identity differs")
    source_arrays = {
        name: _required_array(candidate_arrays, name)
        for name in _SOURCE_EXECUTION_NAMES
    }
    source_identity = _numpy_arrays_sha256(
        source_arrays,
        domain=b"fisher-cmc/synthetic-source-execution-state/v1",
    )
    if source_identity != _identity_hex(
        candidate_arrays, "metadata.source_execution_identity_raw32"
    ):
        raise QualificationError("bundle source execution identity differs")
    carrier_identity = _numpy_arrays_sha256(
        {
            "source.carrier.vectors": source_arrays["source.carrier.vectors"],
            "source.h4_fisher_diagonal": source_arrays[
                "source.h4_fisher_diagonal"
            ],
        },
        domain=b"fisher-cmc/synthetic-source-carrier/v1",
    )
    if carrier_identity != _identity_hex(
        candidate_arrays, "metadata.carrier_identity_raw32"
    ):
        raise QualificationError("bundle carrier identity differs")
    carrier_artifact_identity = hashlib.sha256(
        encode_safetensors(
            {
                "h4_fisher_diagonal": source_arrays[
                    "source.h4_fisher_diagonal"
                ],
                "h4_fisher_eigenvectors": source_arrays[
                    "source.carrier.vectors"
                ],
            }
        )
    ).hexdigest()
    if carrier_artifact_identity != _identity_hex(
        candidate_arrays, "metadata.carrier_artifact_identity_raw32"
    ):
        raise QualificationError("bundle carrier artifact identity differs")
    selection_identity = _numpy_arrays_sha256(
        _selection_execution_identity_arrays(candidate_arrays),
        domain=b"fisher-cmc/synthetic-selection-execution-state/v1",
    )
    if selection_identity != _identity_hex(
        candidate_arrays, "metadata.selection_execution_identity_raw32"
    ):
        raise QualificationError("bundle selection execution identity differs")

    chart_count, head_count, hidden, message, _, _, q_rank, _, interaction, _ = (
        dimensions
    )
    chart_ids = tuple(range(chart_count))
    banks: dict[str, ChartHeadCompletionBank] = {
        "J4": _load_bundle_bank(
            candidate_arrays,
            "J4",
            chart_ids=chart_ids,
            head_count=head_count,
            coefficient_dimension=q_rank,
        )
    }
    for model_id in ("J1", "J2N", "J2+", "J3"):
        banks[model_id] = _load_bundle_bank(
            comparator_arrays,
            model_id,
            chart_ids=chart_ids,
            head_count=head_count,
            coefficient_dimension=q_rank,
        )
    for model_id in ("J0", "J2"):
        if any(name.startswith(f"{model_id}.") for name in comparator_arrays):
            banks[model_id] = _load_bundle_bank(
                comparator_arrays,
                model_id,
                chart_ids=chart_ids,
                head_count=head_count,
                coefficient_dimension=q_rank,
            )
    return LoadedWeightsOnlyBundles(
        candidate_sha256=hashlib.sha256(candidate).hexdigest(),
        comparator_sha256=hashlib.sha256(comparator).hexdigest(),
        baseline_id=baseline_id,
        codec_id=codec_id,
        chart_ids=chart_ids,
        head_count=head_count,
        hidden_dimension=hidden,
        message_dimension=message,
        coefficient_dimension=q_rank,
        interaction_rank=interaction,
        source_execution_identity_sha256=source_identity,
        execution_protocol_sha256=protocol_identity,
        banks=MappingProxyType(banks),
        candidate_tensors=MappingProxyType(candidate_arrays),
        comparator_tensors=MappingProxyType(comparator_arrays),
    )


def self_test_weights_only_bundles(
    candidate: bytes,
    comparator: bytes,
) -> WeightsOnlyBundleSelfTest:
    """Reload bytes and execute raw-message reconstruction through O7."""

    loaded = load_weights_only_bundles(candidate, comparator)
    candidate_arrays = loaded.candidate_tensors
    comparator_arrays = loaded.comparator_tensors
    input_names = (
        "execution.self_test.raw_h",
        "execution.self_test.raw_message",
        "execution.self_test.c_base",
        "execution.self_test.z",
        "execution.self_test.r",
        "execution.self_test.chart_ids",
    )
    for name in input_names:
        if name not in candidate_arrays or name not in comparator_arrays:
            raise QualificationError("bundle self-test execution tensor is missing")
        if not np.array_equal(candidate_arrays[name], comparator_arrays[name]):
            raise QualificationError("candidate/comparator self-test inputs differ")
    outputs: dict[str, np.ndarray] = {}
    errors: list[tuple[str, float]] = []
    passed = True
    stage_attributes = {
        "z": "z",
        "r": "r",
        "reconstructed_message": "reconstructed_message",
        "q_base_heads": "q_base_heads",
        "c_base": "c_base",
        "coefficients": "coefficients",
        "q_heads": "q_heads",
        "q_total": "q_total",
        "h_A": "h_a",
        "q_M": "q_mlp",
        "h_plus": "h_plus",
        "logits": "logits",
    }
    for model_id, arrays in (
        ("J4", candidate_arrays),
        ("J1", comparator_arrays),
        ("J2N", comparator_arrays),
        ("J2+", comparator_arrays),
        ("J3", comparator_arrays),
    ):
        execution = loaded.execute_raw(
            model_id,
            candidate_arrays["execution.self_test.raw_h"],
            candidate_arrays["execution.self_test.raw_message"],
        )
        if not torch.equal(
            execution.chart_ids,
            torch.from_numpy(
                candidate_arrays["execution.self_test.chart_ids"].copy()
            ),
        ):
            passed = False
        maximum = 0.0
        for stage, attribute in stage_attributes.items():
            actual = getattr(execution, attribute).detach().cpu().numpy()
            expected_name = f"execution.self_test.expected.{stage}.{model_id}"
            expected = arrays.get(expected_name)
            if expected is None or expected.shape != actual.shape:
                raise QualificationError(
                    f"bundle self-test expected {stage} for {model_id} is missing or malformed"
                )
            difference = np.abs(
                actual.astype(np.float64) - expected.astype(np.float64)
            )
            stage_maximum = (
                float(np.max(difference)) if difference.size else 0.0
            )
            maximum = max(maximum, stage_maximum)
            if not np.array_equal(actual, expected):
                passed = False
            outputs[f"actual.{stage}.{model_id}"] = np.ascontiguousarray(actual)
        errors.append((model_id, maximum))
        outputs[f"actual.{model_id}"] = np.ascontiguousarray(
            execution.coefficients.detach().cpu().numpy()
        )
    output_bytes = encode_safetensors(outputs)
    return WeightsOnlyBundleSelfTest(
        output_sha256=hashlib.sha256(output_bytes).hexdigest(),
        model_max_abs_errors=tuple(errors),
        row_count=int(candidate_arrays["execution.self_test.raw_h"].shape[0]),
        passed=passed,
    )
