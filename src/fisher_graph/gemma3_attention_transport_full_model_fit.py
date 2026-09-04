"""A-only fitting and Fisher evaluation for the cross-layer V/O artifact.

Only endpoint macroblocks (layers 0--5 and 12--17) are accepted by the fitter.
The middle six-layer block cannot enter any fitted tensor.  A principal
K=8/r=8 dictionary is initialized from endpoint native head operators, then
only the structural terms ``q + depth * beta + gamma[site]`` and repeated-site
head mixing are refined against fixed exact Top-2 Fisher oracles.  Finally one
shared router is trained to imitate those endpoint oracles from live-source
features and fixed structural coordinates.

The held evaluator is a separate function.  It does not mutate the fitted
artifact and reports deployable-router and exact-oracle losses independently.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .gemma3_attention_transport_compiler import (
    AttentionTransportBatch,
    AttentionTransportFitConfig,
    FisherAttentionTransportGeneratorBank,
    fit_fisher_attention_transport_generator_bank,
    samplewise_joint_fisher_loss,
    solve_fisher_sparse_transport_oracle,
)
from .gemma3_attention_transport_full_model_cross_layer import (
    FIXED_INVOCATION_FEATURE_NAMES,
    INVOCATION_IDS,
    ROUTER_FEATURE_NAMES,
    FullModelCrossLayerAttentionTransport,
    SharedCrossLayerGeneratorBank,
    SharedTop2CrossLayerRouter,
    fixed_invocation_coordinates,
    fixed_invocation_encoding,
)
from .gemma3_attention_transport_full_model_partition import (
    ENDPOINT_TRAIN_INVOCATION_IDS,
    MIDDLE_HOLDOUT_INVOCATION_IDS,
)


ProgressCallback = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class FullModelCrossLayerFitConfig:
    generator_count: int = 8
    generator_rank: int = 8
    active_count: int = 2
    dictionary_steps: int = 0
    structural_rounds: int = 2
    structural_epochs_per_round: int = 2
    structural_learning_rate: float = 1.0e-2
    router_steps: int = 240
    router_learning_rate: float = 2.0e-2
    euclidean_floor: float = 1.0e-3
    oracle_coefficient_ridge: float = 1.0e-8
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        if (
            self.generator_count != 8
            or self.generator_rank != 8
            or self.active_count != 2
            or type(self.dictionary_steps) is not int
            or self.dictionary_steps < 0
            or type(self.structural_rounds) is not int
            or self.structural_rounds <= 0
            or type(self.structural_epochs_per_round) is not int
            or self.structural_epochs_per_round <= 0
            or type(self.router_steps) is not int
            or self.router_steps <= 0
        ):
            raise ValueError("full-model cross-layer fixed fit config drifted")
        for value in (
            self.structural_learning_rate,
            self.router_learning_rate,
            self.euclidean_floor,
            self.gradient_clip_norm,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError("full-model fit scalar must be positive")
        if (
            isinstance(self.oracle_coefficient_ridge, bool)
            or not isinstance(self.oracle_coefficient_ridge, (int, float))
            or not math.isfinite(float(self.oracle_coefficient_ridge))
            or float(self.oracle_coefficient_ridge) < 0.0
        ):
            raise ValueError("oracle coefficient ridge is invalid")


@dataclass(frozen=True, slots=True)
class FullModelCrossLayerFitResult:
    artifact: FullModelCrossLayerAttentionTransport
    base_bank_artifact_sha256: str
    fit_row_count: int
    fit_capture_count: int
    endpoint_invocation_ids: tuple[str, ...]
    structural_round_relative_fisher_losses: tuple[float, ...]
    router_training_losses: tuple[float, ...]
    endpoint_evaluation: "FullModelFisherEvaluation"


@dataclass(frozen=True, slots=True)
class FullModelFisherEvaluation:
    row_count: int
    deployable_fisher_loss: float
    oracle_fisher_loss: float
    zero_fisher_loss: float
    deployable_relative_fisher_loss: float
    oracle_relative_fisher_loss: float
    per_invocation_deployable_relative_fisher_loss: tuple[tuple[str, float], ...]
    per_invocation_oracle_relative_fisher_loss: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if (
            type(self.row_count) is not int
            or self.row_count <= 0
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in (
                    self.deployable_fisher_loss,
                    self.oracle_fisher_loss,
                    self.zero_fisher_loss,
                    self.deployable_relative_fisher_loss,
                    self.oracle_relative_fisher_loss,
                )
            )
            or self.zero_fisher_loss <= 0.0
            or self.deployable_fisher_loss + 1.0e-10 < self.oracle_fisher_loss
        ):
            raise ValueError("full-model Fisher evaluation is invalid")


def _progress(
    callback: ProgressCallback | None,
    phase: str,
    completed: int,
    total: int,
    **extra: object,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": phase,
                "completed": completed,
                "total": total,
                **extra,
            }
        )


def _capture_groups(
    captures: Sequence[AttentionTransportBatch],
) -> tuple[tuple[str, tuple[AttentionTransportBatch, ...]], ...]:
    groups: dict[str, list[AttentionTransportBatch]] = {}
    for capture in captures:
        groups.setdefault(capture.invocation_id, []).append(capture)
    return tuple(
        (invocation_id, tuple(groups[invocation_id]))
        for invocation_id in sorted(groups, key=INVOCATION_IDS.index)
    )


def _validate_fit_captures(
    captures: Sequence[AttentionTransportBatch],
) -> tuple[AttentionTransportBatch, ...]:
    values = tuple(captures)
    if not values:
        raise ValueError("full-model fit captures cannot be empty")
    observed = {value.invocation_id for value in values}
    if observed != set(ENDPOINT_TRAIN_INVOCATION_IDS):
        raise ValueError("fit captures must cover exactly the endpoint invocations")
    counts = {key: len(group) for key, group in _capture_groups(values)}
    if len(set(counts.values())) != 1:
        raise ValueError("endpoint invocations must have balanced capture counts")
    if len({value.hidden_width for value in values}) != 1 or len(
        {value.query_head_count for value in values}
    ) != 1:
        raise ValueError("endpoint capture geometry drifted")
    return values


def _site_head_initialization(
    base: FisherAttentionTransportGeneratorBank,
) -> Tensor:
    lookup = {head_id: index for index, head_id in enumerate(base.head_ids)}
    query_heads = len(
        tuple(
            head_id
            for head_id in base.head_ids
            if head_id.startswith(f"{ENDPOINT_TRAIN_INVOCATION_IDS[0]}/")
        )
    )
    sites = []
    for site in range(6):
        endpoint_ids = (INVOCATION_IDS[site], INVOCATION_IDS[12 + site])
        heads = []
        for head in range(query_heads):
            rows = [
                base.head_coefficients[
                    lookup[f"{invocation_id}/query-head.{head}"]
                ]
                for invocation_id in endpoint_ids
            ]
            heads.append(torch.stack(rows).mean(dim=0))
        sites.append(torch.stack(heads))
    return torch.stack(sites).contiguous()


def _raw_candidates(
    *,
    invocation_id: str,
    head_contexts: Tensor,
    left: Tensor,
    base_diagonal: Tensor,
    depth_slopes: Tensor,
    raw_site_offsets: Tensor,
    right: Tensor,
    site_heads: Tensor,
) -> Tensor:
    coordinates = fixed_invocation_coordinates(invocation_id)
    centered_offsets = raw_site_offsets - raw_site_offsets.mean(
        dim=0,
        keepdim=True,
    )
    diagonal = (
        base_diagonal
        + float(coordinates.normalized_depth) * depth_slopes
        + centered_offsets[coordinates.site_index]
    )
    contextual = torch.einsum(
        "nah,ak->nkh",
        head_contexts,
        site_heads[coordinates.site_index],
    )
    latent = torch.einsum("nkh,khr->nkr", contextual, right)
    return torch.einsum(
        "nkr,khr->nkh",
        latent * diagonal.unsqueeze(0),
        left,
    )


def _group_tensors(
    group: Sequence[AttentionTransportBatch],
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.cat(tuple(item.head_contexts for item in group)),
        torch.cat(tuple(item.valid_operator_outputs for item in group)),
        torch.cat(tuple(item.valid_score_gradients for item in group)),
    )


def _oracle_for_group(
    candidates: Tensor,
    target: Tensor,
    gradient: Tensor,
    *,
    config: FullModelCrossLayerFitConfig,
    ridge: float | None = None,
) -> Tensor:
    return solve_fisher_sparse_transport_oracle(
        candidates,
        target,
        gradient,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=(
            config.oracle_coefficient_ridge if ridge is None else ridge
        ),
    ).coefficients


def _relative_fisher_from_predictions(
    predictions: Sequence[Tensor],
    targets: Sequence[Tensor],
    gradients: Sequence[Tensor],
    *,
    euclidean_floor: float,
) -> tuple[float, float, float]:
    prediction = torch.cat(tuple(predictions))
    target = torch.cat(tuple(targets))
    gradient = torch.cat(tuple(gradients))
    fitted = float(
        samplewise_joint_fisher_loss(
            prediction,
            target,
            gradient,
            euclidean_floor=euclidean_floor,
        )
    )
    zero = float(
        samplewise_joint_fisher_loss(
            torch.zeros_like(target),
            target,
            gradient,
            euclidean_floor=euclidean_floor,
        )
    )
    return fitted, zero, fitted / zero


def fit_full_model_cross_layer_attention_transport(
    fit_captures: Sequence[AttentionTransportBatch],
    *,
    config: FullModelCrossLayerFitConfig | None = None,
    progress: ProgressCallback | None = None,
) -> FullModelCrossLayerFitResult:
    """Fit one endpoint-only artifact; held middle captures are unrepresentable."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    resolved = FullModelCrossLayerFitConfig() if config is None else config
    if not isinstance(resolved, FullModelCrossLayerFitConfig):
        raise TypeError("config must be a full-model cross-layer fit config")
    captures = _validate_fit_captures(fit_captures)
    groups = _capture_groups(captures)
    _progress(progress, "dictionary_fit", 0, 1)
    dense = fit_fisher_attention_transport_generator_bank(
        captures,
        captures,
        config=AttentionTransportFitConfig(
            generator_count=resolved.generator_count,
            generator_rank=resolved.generator_rank,
            steps=resolved.dictionary_steps,
            learning_rate=1.0e-2,
            euclidean_floor=resolved.euclidean_floor,
            coefficient_ridge=resolved.oracle_coefficient_ridge,
            gradient_clip_norm=resolved.gradient_clip_norm,
        ),
    )
    base = dense.bank
    _progress(progress, "dictionary_fit", 1, 1)

    left = base.left_factors.detach().clone()
    right = base.right_factors.detach().clone()
    base_diagonal = nn.Parameter(base.singular_scales.detach().clone())
    depth_slopes = nn.Parameter(torch.zeros_like(base_diagonal))
    raw_offsets = nn.Parameter(
        torch.zeros(
            6,
            resolved.generator_count,
            resolved.generator_rank,
            dtype=torch.float64,
        )
    )
    site_heads = nn.Parameter(_site_head_initialization(base))
    optimizer = torch.optim.Adam(
        (base_diagonal, depth_slopes, raw_offsets, site_heads),
        lr=resolved.structural_learning_rate,
    )
    structural_history: list[float] = []
    for round_index in range(resolved.structural_rounds):
        fixed_oracles: dict[str, Tensor] = {}
        joined: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        with torch.no_grad():
            for invocation_id, group in groups:
                contexts, target, gradient = _group_tensors(group)
                candidates = _raw_candidates(
                    invocation_id=invocation_id,
                    head_contexts=contexts,
                    left=left,
                    base_diagonal=base_diagonal,
                    depth_slopes=depth_slopes,
                    raw_site_offsets=raw_offsets,
                    right=right,
                    site_heads=site_heads,
                )
                fixed_oracles[invocation_id] = _oracle_for_group(
                    candidates,
                    target,
                    gradient,
                    config=resolved,
                )
                joined[invocation_id] = (contexts, target, gradient)
        for _ in range(resolved.structural_epochs_per_round):
            for invocation_id, _group in groups:
                contexts, target, gradient = joined[invocation_id]
                optimizer.zero_grad(set_to_none=True)
                candidates = _raw_candidates(
                    invocation_id=invocation_id,
                    head_contexts=contexts,
                    left=left,
                    base_diagonal=base_diagonal,
                    depth_slopes=depth_slopes,
                    raw_site_offsets=raw_offsets,
                    right=right,
                    site_heads=site_heads,
                )
                prediction = torch.einsum(
                    "nk,nkh->nh",
                    fixed_oracles[invocation_id],
                    candidates,
                )
                objective = samplewise_joint_fisher_loss(
                    prediction,
                    target,
                    gradient,
                    euclidean_floor=resolved.euclidean_floor,
                )
                if not bool(torch.isfinite(objective)):
                    raise RuntimeError("structural bank fit became nonfinite")
                objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    (base_diagonal, depth_slopes, raw_offsets, site_heads),
                    max_norm=resolved.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                with torch.no_grad():
                    base_diagonal.clamp_(min=0.0)

        predictions = []
        targets = []
        gradients = []
        with torch.no_grad():
            for invocation_id, _group in groups:
                contexts, target, gradient = joined[invocation_id]
                candidates = _raw_candidates(
                    invocation_id=invocation_id,
                    head_contexts=contexts,
                    left=left,
                    base_diagonal=base_diagonal,
                    depth_slopes=depth_slopes,
                    raw_site_offsets=raw_offsets,
                    right=right,
                    site_heads=site_heads,
                )
                predictions.append(
                    torch.einsum(
                        "nk,nkh->nh",
                        fixed_oracles[invocation_id],
                        candidates,
                    )
                )
                targets.append(target)
                gradients.append(gradient)
        _, _, relative = _relative_fisher_from_predictions(
            predictions,
            targets,
            gradients,
            euclidean_floor=resolved.euclidean_floor,
        )
        structural_history.append(relative)
        _progress(
            progress,
            "structural_fit",
            round_index + 1,
            resolved.structural_rounds,
            relative_fisher_loss=relative,
        )

    bank = SharedCrossLayerGeneratorBank(
        left_factors=left,
        base_singular_diagonal=base_diagonal.detach(),
        depth_singular_slopes=depth_slopes.detach(),
        site_singular_offsets=raw_offsets.detach(),
        right_factors=right,
        site_head_coefficients=site_heads.detach(),
    )

    feature_rows = []
    oracle_rows = []
    support_rows = []
    for invocation_id, group in groups:
        contexts, target, gradient = _group_tensors(group)
        execution = bank.execute(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        oracle = solve_fisher_sparse_transport_oracle(
            execution.candidate_outputs,
            target,
            gradient,
            active_count=resolved.active_count,
            euclidean_floor=resolved.euclidean_floor,
            coefficient_ridge=resolved.oracle_coefficient_ridge,
        )
        fixed = fixed_invocation_encoding(invocation_id).view(1, 1, -1)
        fixed = fixed.expand(contexts.shape[0], resolved.generator_count, -1)
        feature_rows.append(
            torch.cat((fixed, execution.dynamic_router_features), dim=-1)
        )
        oracle_rows.append(oracle.coefficients)
        support_rows.append(oracle.support)
    features = torch.cat(feature_rows)
    oracle_coefficients = torch.cat(oracle_rows)
    oracle_support = torch.cat(support_rows)
    if features.shape[-1] != len(ROUTER_FEATURE_NAMES):
        raise RuntimeError("shared router feature inventory drifted")

    selection_weight = nn.Parameter(
        torch.zeros(
            resolved.generator_count,
            len(ROUTER_FEATURE_NAMES),
            dtype=torch.float64,
        )
    )
    support_frequency = oracle_support.to(torch.float64).mean(dim=0).clamp_min(1.0e-8)
    selection_bias = nn.Parameter(support_frequency.log())
    amplitude_weight = nn.Parameter(torch.zeros_like(selection_weight))
    amplitude_bias = nn.Parameter(
        torch.stack(
            tuple(
                (
                    oracle_coefficients[:, index][oracle_support[:, index]].mean()
                    * resolved.active_count
                    if bool(oracle_support[:, index].any())
                    else torch.zeros((), dtype=torch.float64)
                )
                for index in range(resolved.generator_count)
            )
        )
    )
    router_optimizer = torch.optim.Adam(
        (selection_weight, selection_bias, amplitude_weight, amplitude_bias),
        lr=resolved.router_learning_rate,
    )
    router_history: list[float] = []
    coefficient_power = float(oracle_coefficients.square().mean().clamp_min(1.0e-12))
    for step in range(resolved.router_steps):
        router_optimizer.zero_grad(set_to_none=True)
        logits = (
            torch.einsum("nkf,kf->nk", features, selection_weight)
            + selection_bias.unsqueeze(0)
        )
        support_distribution = oracle_support.to(torch.float64)
        support_distribution = support_distribution / support_distribution.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        support_loss = -(
            support_distribution * torch.log_softmax(logits, dim=-1)
        ).sum(dim=-1).mean()
        amplitudes = (
            torch.einsum("nkf,kf->nk", features, amplitude_weight)
            + amplitude_bias.unsqueeze(0)
        )
        oracle_sparse_probabilities = torch.softmax(
            logits.masked_fill(~oracle_support, torch.finfo(logits.dtype).min),
            dim=-1,
        )
        predicted_coefficients = oracle_sparse_probabilities * amplitudes
        coefficient_loss = (
            predicted_coefficients - oracle_coefficients
        ).square().mean() / coefficient_power
        loss = support_loss + coefficient_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("shared router fit became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (selection_weight, selection_bias, amplitude_weight, amplitude_bias),
            max_norm=resolved.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        router_optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == resolved.router_steps:
            router_history.append(float(loss.detach()))
            _progress(
                progress,
                "router_fit",
                step + 1,
                resolved.router_steps,
                training_loss=float(loss.detach()),
            )

    router = SharedTop2CrossLayerRouter(
        generator_bank_sha256=bank.artifact_sha256,
        selection_weight=selection_weight.detach(),
        selection_bias=selection_bias.detach(),
        amplitude_weight=amplitude_weight.detach(),
        amplitude_bias=amplitude_bias.detach(),
        active_count=resolved.active_count,
    )
    artifact = FullModelCrossLayerAttentionTransport(bank=bank, router=router)
    endpoint_evaluation = evaluate_full_model_cross_layer_fisher(
        artifact,
        captures,
        euclidean_floor=resolved.euclidean_floor,
    )
    return FullModelCrossLayerFitResult(
        artifact=artifact,
        base_bank_artifact_sha256=base.artifact_sha256,
        fit_row_count=sum(value.row_count for value in captures),
        fit_capture_count=len(captures),
        endpoint_invocation_ids=ENDPOINT_TRAIN_INVOCATION_IDS,
        structural_round_relative_fisher_losses=tuple(structural_history),
        router_training_losses=tuple(router_history),
        endpoint_evaluation=endpoint_evaluation,
    )


def evaluate_full_model_cross_layer_fisher(
    artifact: FullModelCrossLayerAttentionTransport,
    captures: Sequence[AttentionTransportBatch],
    *,
    euclidean_floor: float = 1.0e-3,
) -> FullModelFisherEvaluation:
    """Read captures once and compare deployable Top-2 with the exact oracle."""

    if not isinstance(artifact, FullModelCrossLayerAttentionTransport):
        raise TypeError("artifact must be a full-model cross-layer transport")
    values = tuple(captures)
    if not values:
        raise ValueError("Fisher evaluation captures cannot be empty")
    deployable_predictions: list[Tensor] = []
    oracle_predictions: list[Tensor] = []
    targets: list[Tensor] = []
    gradients: list[Tensor] = []
    per_deployable = []
    per_oracle = []
    for invocation_id, group in _capture_groups(values):
        contexts, target, gradient = _group_tensors(group)
        forward = artifact(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        oracle = solve_fisher_sparse_transport_oracle(
            forward.bank_execution.candidate_outputs,
            target,
            gradient,
            active_count=artifact.router.active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=0.0,
        )
        oracle_prediction = torch.einsum(
            "nk,nkh->nh",
            oracle.coefficients,
            forward.bank_execution.candidate_outputs,
        )
        deployable_predictions.append(forward.operator_outputs)
        oracle_predictions.append(oracle_prediction)
        targets.append(target)
        gradients.append(gradient)
        _, zero, deployable_relative = _relative_fisher_from_predictions(
            (forward.operator_outputs,),
            (target,),
            (gradient,),
            euclidean_floor=euclidean_floor,
        )
        oracle_loss, _, oracle_relative = _relative_fisher_from_predictions(
            (oracle_prediction,),
            (target,),
            (gradient,),
            euclidean_floor=euclidean_floor,
        )
        if zero <= 0.0 or oracle_loss < 0.0:
            raise RuntimeError("per-invocation Fisher denominator drifted")
        per_deployable.append((invocation_id, deployable_relative))
        per_oracle.append((invocation_id, oracle_relative))
    deployable_loss, zero_loss, deployable_relative = (
        _relative_fisher_from_predictions(
            deployable_predictions,
            targets,
            gradients,
            euclidean_floor=euclidean_floor,
        )
    )
    oracle_loss, oracle_zero, oracle_relative = _relative_fisher_from_predictions(
        oracle_predictions,
        targets,
        gradients,
        euclidean_floor=euclidean_floor,
    )
    if not math.isclose(zero_loss, oracle_zero, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise RuntimeError("deployable/oracle Fisher baselines drifted")
    return FullModelFisherEvaluation(
        row_count=sum(value.row_count for value in values),
        deployable_fisher_loss=deployable_loss,
        oracle_fisher_loss=oracle_loss,
        zero_fisher_loss=zero_loss,
        deployable_relative_fisher_loss=deployable_relative,
        oracle_relative_fisher_loss=oracle_relative,
        per_invocation_deployable_relative_fisher_loss=tuple(per_deployable),
        per_invocation_oracle_relative_fisher_loss=tuple(per_oracle),
    )


__all__ = [
    "FullModelCrossLayerFitConfig",
    "FullModelCrossLayerFitResult",
    "FullModelFisherEvaluation",
    "evaluate_full_model_cross_layer_fisher",
    "fit_full_model_cross_layer_attention_transport",
]
