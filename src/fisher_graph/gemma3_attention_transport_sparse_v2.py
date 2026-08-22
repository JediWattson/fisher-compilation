"""Sparse-deployment-aware fitting for Gemma attention V/O transport.

The sealed v1 campaign fitted a shared generator bank with every generator
active and introduced its exact sparse router teacher and held-out oracle gate
only after freezing that bank.  This development-only module aligns the bank
with that globally sparse oracle target.  It alternates an exact,
positive-ridge A-side sparse Fisher oracle with gradient updates of the
low-rank generators and static head coefficients::

    beta* = argmin_{||beta||_0 <= s} L_F(sum_k beta_k z_k, y)
    G, rho = argmin L_F(sum_k stopgrad(beta*_k) z_k(G, rho), y)

The chart-conditioned runtime applies TopK independently inside each chart and
can therefore union more than ``s`` generators after membership mixing.  This
module deliberately does not claim to optimize that larger router-in-the-loop
equation; it tests the necessary oracle ceiling used by v1's teacher and gate.
The final fit candidate is selected on fit rows only.  Evaluation is a
separate function call after fitting; the fitter API cannot receive evaluation
rows.  No router, B panel, C panel, or behavioral claim is part of this
diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .gemma3_attention_transport_compiler import (
    AttentionTransportBankFit,
    AttentionTransportBatch,
    AttentionTransportFitConfig,
    AttentionTransportMetrics,
    FisherAttentionTransportGeneratorBank,
    _canonical_bank,
    _capture_catalog,
    _metrics,
    fit_fisher_attention_transport_generator_bank,
    samplewise_joint_fisher_loss,
    solve_fisher_sparse_transport_oracle,
)


@dataclass(frozen=True, slots=True)
class SparseAwareAttentionTransportFitConfig:
    generator_count: int
    generator_rank: int
    active_count: int
    dense_warm_start_steps: int = 96
    alternating_rounds: int = 4
    steps_per_round: int = 32
    learning_rate: float = 1.0e-2
    euclidean_floor: float = 1.0e-3
    oracle_coefficient_ridge: float = 1.0e-8
    dense_head_coefficient_ridge: float = 1.0e-8
    gradient_clip_norm: float = 10.0
    gauge_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        for label, value, minimum in (
            ("generator_count", self.generator_count, 1),
            ("generator_rank", self.generator_rank, 1),
            ("active_count", self.active_count, 1),
            ("dense_warm_start_steps", self.dense_warm_start_steps, 0),
            ("alternating_rounds", self.alternating_rounds, 1),
            ("steps_per_round", self.steps_per_round, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{label} is outside its valid range")
        if self.active_count > self.generator_count:
            raise ValueError("active_count exceeds the generator inventory")
        for label, value, positive in (
            ("learning_rate", self.learning_rate, True),
            ("euclidean_floor", self.euclidean_floor, True),
            ("oracle_coefficient_ridge", self.oracle_coefficient_ridge, False),
            (
                "dense_head_coefficient_ridge",
                self.dense_head_coefficient_ridge,
                False,
            ),
            ("gradient_clip_norm", self.gradient_clip_norm, True),
            ("gauge_floor", self.gauge_floor, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (positive and float(value) <= 0.0)
                or (not positive and float(value) < 0.0)
            ):
                raise ValueError(f"{label} is invalid")

    @property
    def dense_warm_start_config(self) -> AttentionTransportFitConfig:
        return AttentionTransportFitConfig(
            generator_count=self.generator_count,
            generator_rank=self.generator_rank,
            steps=self.dense_warm_start_steps,
            learning_rate=self.learning_rate,
            euclidean_floor=self.euclidean_floor,
            coefficient_ridge=self.dense_head_coefficient_ridge,
            gradient_clip_norm=self.gradient_clip_norm,
        )


@dataclass(frozen=True, slots=True)
class SparseAttentionTransportOracleEvaluation:
    output_metrics: AttentionTransportMetrics
    zero_predictor_fisher_loss: float
    relative_fisher_loss: float
    active_count_limit: int
    coefficient_ridge: float
    per_invocation_relative_fisher_loss: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output_metrics, AttentionTransportMetrics)
            or not math.isfinite(float(self.zero_predictor_fisher_loss))
            or float(self.zero_predictor_fisher_loss) <= 0.0
            or not math.isfinite(float(self.relative_fisher_loss))
            or float(self.relative_fisher_loss) < 0.0
            or type(self.active_count_limit) is not int
            or self.active_count_limit <= 0
            or isinstance(self.coefficient_ridge, bool)
            or not isinstance(self.coefficient_ridge, (int, float))
            or not math.isfinite(float(self.coefficient_ridge))
            or float(self.coefficient_ridge) < 0.0
            or not self.per_invocation_relative_fisher_loss
            or len({key for key, _ in self.per_invocation_relative_fisher_loss})
            != len(self.per_invocation_relative_fisher_loss)
            or any(
                not isinstance(key, str)
                or not key
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for key, value in self.per_invocation_relative_fisher_loss
            )
        ):
            raise ValueError("sparse oracle evaluation is invalid")


@dataclass(frozen=True, slots=True)
class SparseAwareAttentionTransportBankFit:
    bank: FisherAttentionTransportGeneratorBank
    dense_warm_start: AttentionTransportBankFit
    initial_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_fit_oracle: SparseAttentionTransportOracleEvaluation
    fit_round_relative_fisher_losses: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bank, FisherAttentionTransportGeneratorBank)
            or not isinstance(self.dense_warm_start, AttentionTransportBankFit)
            or self.bank.artifact_sha256
            == ""
            or self.dense_warm_start.bank.generator_count
            != self.bank.generator_count
            or self.dense_warm_start.bank.generator_rank
            != self.bank.generator_rank
            or not self.fit_round_relative_fisher_losses
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in self.fit_round_relative_fisher_losses
            )
            or self.final_fit_oracle.relative_fisher_loss
            > self.initial_fit_oracle.relative_fisher_loss + 1.0e-12
        ):
            raise ValueError("sparse-aware bank fit is invalid")


def _group_capture_indices(
    captures: Sequence[AttentionTransportBatch],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    groups: dict[str, list[int]] = {}
    for index, capture in enumerate(captures):
        groups.setdefault(capture.invocation_id, []).append(index)
    return tuple(
        (invocation_id, tuple(groups[invocation_id]))
        for invocation_id in sorted(groups)
    )


def _sparse_oracle_coefficients(
    bank: FisherAttentionTransportGeneratorBank,
    captures: Sequence[AttentionTransportBatch],
    *,
    active_count: int,
    euclidean_floor: float,
    coefficient_ridge: float,
) -> tuple[Tensor, ...]:
    values, _, _, _ = _capture_catalog(captures)
    coefficients: list[Tensor | None] = [None] * len(values)
    for _, indices in _group_capture_indices(values):
        candidates_by_capture = tuple(
            bank.candidate_outputs(values[index]) for index in indices
        )
        row_counts = tuple(int(value.shape[0]) for value in candidates_by_capture)
        oracle = solve_fisher_sparse_transport_oracle(
            torch.cat(candidates_by_capture),
            torch.cat(tuple(values[index].valid_operator_outputs for index in indices)),
            torch.cat(tuple(values[index].valid_score_gradients for index in indices)),
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        )
        split = oracle.coefficients.split(row_counts)
        for index, value in zip(indices, split, strict=True):
            coefficients[index] = value.detach().clone().contiguous()
    if any(value is None for value in coefficients):
        raise RuntimeError("sparse oracle omitted one capture")
    return tuple(value for value in coefficients if value is not None)


def evaluate_sparse_attention_transport_bank(
    bank: FisherAttentionTransportGeneratorBank,
    captures: Sequence[AttentionTransportBatch],
    *,
    active_count: int,
    euclidean_floor: float,
    coefficient_ridge: float = 0.0,
) -> SparseAttentionTransportOracleEvaluation:
    """Evaluate the exact sparse Fisher oracle without retaining coefficients."""

    values, _, _, _ = _capture_catalog(captures)
    if type(active_count) is not int or not 0 < active_count <= bank.generator_count:
        raise ValueError("active_count is outside the generator inventory")
    oracle_coefficients = _sparse_oracle_coefficients(
        bank,
        values,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    predictions = tuple(
        bank.predict_operator_outputs(capture, coefficients)
        for capture, coefficients in zip(
            values,
            oracle_coefficients,
            strict=True,
        )
    )
    targets = tuple(value.valid_operator_outputs for value in values)
    gradients = tuple(value.valid_score_gradients for value in values)
    joined_prediction = torch.cat(predictions)
    joined_target = torch.cat(targets)
    joined_gradient = torch.cat(gradients)
    metrics = _metrics(
        joined_prediction,
        joined_target,
        joined_gradient,
        euclidean_floor,
    )
    zero_metrics = _metrics(
        torch.zeros_like(joined_target),
        joined_target,
        joined_gradient,
        euclidean_floor,
    )
    per_invocation = []
    for invocation_id, indices in _group_capture_indices(values):
        prediction = torch.cat(tuple(predictions[index] for index in indices))
        target = torch.cat(tuple(targets[index] for index in indices))
        gradient = torch.cat(tuple(gradients[index] for index in indices))
        invocation_metrics = _metrics(
            prediction,
            target,
            gradient,
            euclidean_floor,
        )
        invocation_zero = _metrics(
            torch.zeros_like(target),
            target,
            gradient,
            euclidean_floor,
        )
        per_invocation.append(
            (
                invocation_id,
                float(
                    invocation_metrics.fisher_loss
                    / invocation_zero.fisher_loss
                ),
            )
        )
    return SparseAttentionTransportOracleEvaluation(
        output_metrics=metrics,
        zero_predictor_fisher_loss=float(zero_metrics.fisher_loss),
        relative_fisher_loss=float(
            metrics.fisher_loss / zero_metrics.fisher_loss
        ),
        active_count_limit=active_count,
        coefficient_ridge=float(coefficient_ridge),
        per_invocation_relative_fisher_loss=tuple(per_invocation),
    )


def _raw_factors(
    bank: FisherAttentionTransportGeneratorBank,
) -> tuple[Tensor, Tensor, Tensor]:
    roots = bank.singular_scales.clamp_min(0.0).sqrt()
    return (
        bank.left_factors * roots[:, None, :],
        bank.right_factors * roots[:, None, :],
        bank.head_coefficients.clone(),
    )


def _gauge_normalized_parameters(
    left: Tensor,
    right: Tensor,
    head_coefficients: Tensor,
    *,
    gauge_floor: float,
) -> tuple[Tensor, Tensor, Tensor]:
    left_gram = torch.einsum("khr,khs->krs", left, left)
    right_gram = torch.einsum("khr,khs->krs", right, right)
    generator_norms = torch.einsum(
        "krs,krs->k",
        left_gram,
        right_gram,
    ).clamp_min(0.0).sqrt()
    head_norms = torch.linalg.vector_norm(head_coefficients, dim=0)
    if bool((generator_norms <= gauge_floor).any()) or bool(
        (head_norms <= gauge_floor).any()
    ):
        raise RuntimeError("sparse-aware gauge encountered a zero atom")
    return (
        left / generator_norms[:, None, None],
        right,
        head_coefficients / head_norms[None, :],
    )


def _gauge_normalized_bank(
    bank: FisherAttentionTransportGeneratorBank,
    *,
    gauge_floor: float,
) -> FisherAttentionTransportGeneratorBank:
    left, right, heads = _raw_factors(bank)
    normalized = _gauge_normalized_parameters(
        left,
        right,
        heads,
        gauge_floor=gauge_floor,
    )
    return _canonical_bank(*normalized, bank.head_ids)


def _sparse_parameter_prediction(
    capture: AttentionTransportBatch,
    left: Tensor,
    right: Tensor,
    head_coefficients: Tensor,
    sparse_coefficients: Tensor,
) -> Tensor:
    contextual = torch.einsum(
        "nah,ak->nkh",
        capture.head_contexts,
        head_coefficients,
    )
    latent = torch.einsum("nkh,khr->nkr", contextual, right)
    candidates = torch.einsum("nkr,khr->nkh", latent, left)
    return torch.einsum("nk,nkh->nh", sparse_coefficients, candidates)


def fit_sparse_aware_attention_transport_generator_bank(
    fit_captures: Sequence[AttentionTransportBatch],
    *,
    config: SparseAwareAttentionTransportFitConfig,
    dense_warm_start: AttentionTransportBankFit | None = None,
    round_observer: Callable[
        [int, int, SparseAttentionTransportOracleEvaluation], None
    ]
    | None = None,
) -> SparseAwareAttentionTransportBankFit:
    """Alternate exact sparse A-side supports with low-rank bank updates."""

    if not isinstance(config, SparseAwareAttentionTransportFitConfig):
        raise TypeError("config must be a sparse-aware fit config")
    if round_observer is not None and not callable(round_observer):
        raise TypeError("round_observer must be callable")
    fit, fit_invocations, head_ids, _ = _capture_catalog(fit_captures)
    warm = (
        fit_fisher_attention_transport_generator_bank(
            fit,
            fit,
            config=config.dense_warm_start_config,
        )
        if dense_warm_start is None
        else dense_warm_start
    )
    if (
        warm.bank.generator_count != config.generator_count
        or warm.bank.generator_rank != config.generator_rank
        or warm.bank.head_ids != head_ids
    ):
        raise ValueError("dense warm start does not match sparse-aware config")

    current_bank = _gauge_normalized_bank(
        warm.bank,
        gauge_floor=config.gauge_floor,
    )
    initial = evaluate_sparse_attention_transport_bank(
        current_bank,
        fit,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.oracle_coefficient_ridge,
    )
    best_bank = current_bank
    best_fit = initial
    history = [initial.relative_fisher_loss]
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for capture in fit_invocations:
        stop = start + capture.query_head_count
        offsets[capture.invocation_id] = (start, stop)
        start = stop

    for round_index in range(1, config.alternating_rounds + 1):
        sparse_coefficients = _sparse_oracle_coefficients(
            current_bank,
            fit,
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.oracle_coefficient_ridge,
        )
        initial_left, initial_right, initial_heads = _raw_factors(current_bank)
        left = nn.Parameter(initial_left)
        right = nn.Parameter(initial_right)
        heads = nn.Parameter(initial_heads)
        optimizer = torch.optim.Adam(
            (left, right, heads),
            lr=float(config.learning_rate),
        )
        row_count = float(sum(value.row_count for value in fit))

        def fixed_support_objective(
            normalized_left: Tensor,
            normalized_right: Tensor,
            normalized_heads: Tensor,
        ) -> Tensor:
            objective = torch.zeros((), dtype=torch.float64)
            for capture, beta in zip(fit, sparse_coefficients, strict=True):
                begin, end = offsets[capture.invocation_id]
                prediction = _sparse_parameter_prediction(
                    capture,
                    normalized_left,
                    normalized_right,
                    normalized_heads[begin:end],
                    beta,
                )
                objective = objective + (
                    samplewise_joint_fisher_loss(
                        prediction,
                        capture.valid_operator_outputs,
                        capture.valid_score_gradients,
                        euclidean_floor=config.euclidean_floor,
                    )
                    * capture.row_count
                    / row_count
                )
            return objective

        best_fixed_loss = math.inf
        best_fixed_parameters: tuple[Tensor, Tensor, Tensor] | None = None
        for _ in range(config.steps_per_round):
            optimizer.zero_grad(set_to_none=True)
            normalized_left, normalized_right, normalized_heads = (
                _gauge_normalized_parameters(
                    left,
                    right,
                    heads,
                    gauge_floor=config.gauge_floor,
                )
            )
            objective = fixed_support_objective(
                normalized_left,
                normalized_right,
                normalized_heads,
            )
            if not bool(torch.isfinite(objective)):
                raise RuntimeError("sparse-aware bank fit became nonfinite")
            observed_loss = float(objective.detach())
            if observed_loss < best_fixed_loss:
                best_fixed_loss = observed_loss
                best_fixed_parameters = (
                    left.detach().clone(),
                    right.detach().clone(),
                    heads.detach().clone(),
                )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                (left, right, heads),
                max_norm=float(config.gradient_clip_norm),
                error_if_nonfinite=True,
            )
            optimizer.step()

        terminal_left, terminal_right, terminal_heads = (
            _gauge_normalized_parameters(
                left.detach(),
                right.detach(),
                heads.detach(),
                gauge_floor=config.gauge_floor,
            )
        )
        terminal_objective = fixed_support_objective(
            terminal_left,
            terminal_right,
            terminal_heads,
        )
        if not bool(torch.isfinite(terminal_objective)):
            raise RuntimeError("sparse-aware terminal fit became nonfinite")
        if float(terminal_objective) < best_fixed_loss:
            best_fixed_parameters = (
                left.detach().clone(),
                right.detach().clone(),
                heads.detach().clone(),
            )
        if best_fixed_parameters is None:
            raise RuntimeError("sparse-aware fit did not retain a checkpoint")
        normalized_left, normalized_right, normalized_heads = (
            _gauge_normalized_parameters(
                *best_fixed_parameters,
                gauge_floor=config.gauge_floor,
            )
        )
        current_bank = _canonical_bank(
            normalized_left,
            normalized_right,
            normalized_heads,
            head_ids,
        )
        current_fit = evaluate_sparse_attention_transport_bank(
            current_bank,
            fit,
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.oracle_coefficient_ridge,
        )
        if current_fit.relative_fisher_loss < best_fit.relative_fisher_loss:
            best_bank = current_bank
            best_fit = current_fit
        else:
            current_bank = best_bank
        history.append(best_fit.relative_fisher_loss)
        if round_observer is not None:
            round_observer(round_index, config.alternating_rounds, best_fit)

    return SparseAwareAttentionTransportBankFit(
        bank=best_bank,
        dense_warm_start=warm,
        initial_fit_oracle=initial,
        final_fit_oracle=best_fit,
        fit_round_relative_fisher_losses=tuple(float(value) for value in history),
    )


__all__ = [
    "SparseAttentionTransportOracleEvaluation",
    "SparseAwareAttentionTransportBankFit",
    "SparseAwareAttentionTransportFitConfig",
    "evaluate_sparse_attention_transport_bank",
    "fit_sparse_aware_attention_transport_generator_bank",
]
