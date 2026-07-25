"""Split-safe activation-aware modal selection and bounded Jacobian probing.

Calibration A is the only fit split.  It supplies complete pooled activation
Fisher eigensystems, exact pooled activation covariances, and three
predeclared codec families:

* the native Fisher ordering (scientific control);
* Fisher modes reordered by eigenvalue times activation variance; and
* generalized Fisher codecs for explicit ``(alpha, beta)`` floor pairs.

Calibration B evaluates the predeclared joint rank/family schedule and locks
the first candidate in canonical ``(rank, family)`` order that passes both
quality gates.  A native full-width candidate is an unconditional fallback,
and calibration B also evaluates a full-width identity for every codec
family.  Every family identity must pass before reduced candidates are
eligible for selection.  Validation evaluates the locked rank, its family-specific
full-width identity, and the native full-width identity; the two identity
executions are deduplicated when the locked family is native.  Reserved test
prompts are parsed and hashed but never tokenized or model-evaluated.

An optional, deliberately small post-selection pilot uses true forward JVPs
through the selected frozen block.  It is a diagnostic edge map, not a fitted
executor or a compression/speed claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter, LayerBlockBoundaryPlan
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .gemma3_ablation_experiment import (
    DEFAULT_END_LAYER,
    DEFAULT_FULL_RANK_NLL_ATOL,
    DEFAULT_START_LAYER,
    _FrozenModelTensorGuard,
    _is_sha256,
    _update_payload_digest,
    _validate_ablation_metadata,
    _validate_full_rank_identity,
    _validate_model_metadata,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    make_causal_lm_calibration_batches,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_stability_experiment import (
    DEFAULT_PROMPT_SPLITS,
    _CalibrationStreamProvenance,
    _library_versions,
    _ordered_prompt_hash_digest,
    _tokenizer_provenance,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .jacobian_probe import (
    CausalLagJacobianStatistics,
    collect_block_causal_lag_jacobian,
)
from .linear_codec import (
    ActivationCovarianceResult,
    LinearActivationCodec,
    StreamingActivationCovariance,
    build_generalized_fisher_codec,
    build_native_fisher_codec,
    build_variance_weighted_fisher_codec,
)
from .modal_ablation import (
    ModalAblationCondition,
    ModalAblationConditionResult,
    ModalAblationAggregate,
    ModalAblationExample,
    ModalAblationResult,
    _aggregate,
    _causal_lm_batch_scores,
    _example_ids,
)
from .streaming_analysis import (
    StreamingActivationFisherBasis,
    StreamingFisherCollection,
    iter_activation_score_gradient_rows,
)
from .streaming_fisher import StreamingActivationFisherEstimator
from .weighted_jacobian import (
    CausalWeightedJacobianResult,
    factor_causal_weighted_jacobian,
)


DEFAULT_RANKS = (512, 576, 608, 624, 632, 636, 638, 639, 640)
DEFAULT_GENERALIZED_REGULARIZATION = (
    (1e-3, 1e-6),
    (1e-2, 1e-5),
)
DEFAULT_SELECTION_NLL_ATOL = 0.05
DEFAULT_SELECTION_TOP1_MIN = 0.95
DEFAULT_JACOBIAN_MODES = 4
DEFAULT_JACOBIAN_MAX_LAG = 1
DEFAULT_JACOBIAN_MAX_SEQUENCES = 1
DEFAULT_JACOBIAN_FACTOR_RANK = 2

_ARTIFACT_SCHEMA = "fisher_graph.gemma3_weighted_jacobian"
_ARTIFACT_FORMAT_VERSION = 2
_PAYLOAD_DOMAIN = b"fisher_graph.gemma3_weighted_jacobian_payload.v1\0"
_REPORT_DOMAIN = b"fisher_graph.gemma3_weighted_jacobian_report.v1\0"


def default_gemma3_weighted_jacobian_output(
    model_id: str = DEFAULT_MODEL_ID,
    start_layer: int = DEFAULT_START_LAYER,
    end_layer: int = DEFAULT_END_LAYER,
) -> Path:
    """Return an ignored model/block-specific output path."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if (
        type(start_layer) is not int
        or type(end_layer) is not int
        or start_layer < 0
        or end_layer < start_layer
    ):
        raise ValueError("layer range must be nonnegative and ascending")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip("._-")
    return (
        Path(".local-runs")
        / (slug or "gemma3-model")
        / f"layers-{start_layer}-{end_layer}-weighted-jacobian.pt"
    )


def _scientific_payload_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, payload)
    return digest.hexdigest()


def _report_sha256(report: Mapping[str, object]) -> str:
    serialized = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_REPORT_DOMAIN)
    digest.update(serialized)
    return digest.hexdigest()


def _codec_state_sha256(
    codec: LinearActivationCodec | Mapping[str, object],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.linear_activation_codec_binding.v1\0")
    state = codec.state_dict() if isinstance(
        codec,
        LinearActivationCodec,
    ) else codec
    _update_payload_digest(digest, state)
    return digest.hexdigest()


def _validated_ranks(
    ranks: Iterable[int],
    *,
    width: int,
) -> tuple[int, ...]:
    if isinstance(ranks, (str, bytes)):
        raise TypeError("ranks must be an iterable of positive integers")
    try:
        requested = tuple(ranks)
    except TypeError as error:
        raise TypeError(
            "ranks must be an iterable of positive integers"
        ) from error
    if not requested:
        raise ValueError("ranks cannot be empty")
    if any(
        type(rank) is not int or not 1 <= rank <= width
        for rank in requested
    ):
        raise ValueError(
            f"ranks must contain integers between 1 and width {width}"
        )
    return tuple(sorted(set((*requested, width))))


def _validated_regularization_pairs(
    pairs: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    if isinstance(pairs, (str, bytes)):
        raise TypeError(
            "generalized_regularization_pairs must contain pairs"
        )
    try:
        requested = tuple(pairs)
    except TypeError as error:
        raise TypeError(
            "generalized_regularization_pairs must be iterable"
        ) from error
    if not requested:
        raise ValueError(
            "generalized_regularization_pairs cannot be empty"
        )
    normalized: list[tuple[float, float]] = []
    for pair in requested:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                "each generalized regularization value must be "
                "an (alpha, beta) pair"
            )
        alpha, beta = pair
        if (
            isinstance(alpha, bool)
            or isinstance(beta, bool)
            or not isinstance(alpha, (int, float))
            or not isinstance(beta, (int, float))
            or not math.isfinite(float(alpha))
            or not math.isfinite(float(beta))
            or float(alpha) <= 0
            or float(beta) <= 0
        ):
            raise ValueError(
                "generalized alpha/beta floors must be finite and positive"
            )
        value = (float(alpha), float(beta))
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class _CodecVariant:
    variant_id: str
    method: str
    alpha: float | None = None
    beta: float | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "method": self.method,
            "alpha": self.alpha,
            "beta": self.beta,
        }


@dataclass(frozen=True, slots=True)
class _CodecCandidate:
    candidate_id: str
    variant: _CodecVariant
    retained_rank: int
    codecs: Mapping[str, LinearActivationCodec]
    sites: tuple[str, ...]
    fallback: bool = False

    def condition(self, *, name: str | None = None) -> ModalAblationCondition:
        return ModalAblationCondition(
            name=self.candidate_id if name is None else name,
            retained_modes={
                site: self.retained_rank for site in self.sites
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "variant_id": self.variant.variant_id,
            "method": self.variant.method,
            "alpha": self.variant.alpha,
            "beta": self.variant.beta,
            "retained_rank": self.retained_rank,
            "sites": self.sites,
            "fallback": self.fallback,
        }


def _variants(
    regularization_pairs: Sequence[tuple[float, float]],
) -> tuple[_CodecVariant, ...]:
    values = [
        _CodecVariant("native_fisher", "native_fisher"),
        _CodecVariant(
            "variance_weighted_fisher",
            "variance_weighted_fisher",
        ),
    ]
    values.extend(
        _CodecVariant(
            f"generalized_fisher.reg_{index:02d}",
            "generalized_fisher",
            alpha,
            beta,
        )
        for index, (alpha, beta) in enumerate(regularization_pairs)
    )
    return tuple(values)


def _fit_calibration_a(
    adapter: Gemma3CausalLMAdapter,
    batches: Iterable[CalibrationBatch],
    *,
    plan: LayerBlockBoundaryPlan,
    width: int,
    sketch_rows: int,
) -> tuple[
    StreamingFisherCollection,
    dict[str, ActivationCovarianceResult],
]:
    fisher_estimators = {
        site: StreamingActivationFisherEstimator(
            activation_name=site,
            rank=width,
            width=width,
            sketch_rows=sketch_rows,
            accumulation_dtype=torch.float64,
        )
        for site in plan.activation_sites
    }
    covariance_estimators = {
        site: StreamingActivationCovariance(
            activation_name=site,
            width=width,
        )
        for site in plan.activation_sites
    }
    loss_sum = 0.0
    sequences = 0
    rows = iter_activation_score_gradient_rows(
        adapter,
        batches,
        activation_names=plan.activation_sites,
        score_objective=CausalLanguageModelNLL(),
        leaf_activation_name=plan.leaf_activation_name,
        accumulation_dtype=torch.float64,
    )
    try:
        for sequence_rows in rows:
            for site in plan.activation_sites:
                fisher_estimators[site].update(
                    sequence_rows.score_gradients[site]
                )
                covariance_estimators[site].update(
                    sequence_rows.activations[site]
                )
            loss_sum += sequence_rows.loss
            sequences += 1
    finally:
        close = getattr(rows, "close", None)
        if callable(close):
            close()
    if sequences == 0:
        raise ValueError("calibration A cannot be empty")

    covariances = {
        site: estimator.finalize()
        for site, estimator in covariance_estimators.items()
    }
    bases = {}
    for site, estimator in fisher_estimators.items():
        fisher = estimator.finalize()
        covariance = covariances[site]
        if (
            fisher.observations != covariance.observations
            or fisher.rows_seen != covariance.rows_seen
            or fisher.width != width
            or fisher.modes != width
        ):
            raise RuntimeError(
                f"{site!r} Fisher/covariance row accounting diverged"
            )
        bases[site] = StreamingActivationFisherBasis(
            activation_name=site,
            mean=covariance.mean,
            fisher=fisher,
            sequences=sequences,
        )
    return (
        StreamingFisherCollection(
            bases=bases,
            mean_loss=loss_sum / sequences,
            sequences=sequences,
        ),
        covariances,
    )


def _build_codec_families(
    *,
    fisher: StreamingFisherCollection,
    covariances: Mapping[str, ActivationCovarianceResult],
    variants: Sequence[_CodecVariant],
) -> dict[str, dict[str, LinearActivationCodec]]:
    result: dict[str, dict[str, LinearActivationCodec]] = {}
    for variant in variants:
        by_site = {}
        for site, basis in fisher.bases.items():
            covariance = covariances[site]
            if variant.method == "native_fisher":
                codec = build_native_fisher_codec(
                    covariance=covariance,
                    fisher_eigenvalues=basis.fisher.eigenvalues,
                    fisher_vectors=basis.fisher.vectors,
                )
            elif variant.method == "variance_weighted_fisher":
                codec = build_variance_weighted_fisher_codec(
                    covariance=covariance,
                    fisher_eigenvalues=basis.fisher.eigenvalues,
                    fisher_vectors=basis.fisher.vectors,
                )
            elif variant.method == "generalized_fisher":
                assert variant.alpha is not None
                assert variant.beta is not None
                codec = build_generalized_fisher_codec(
                    covariance=covariance,
                    fisher_matrix=basis.fisher.approximate_matrix(),
                    alpha=variant.alpha,
                    beta=variant.beta,
                )
            else:  # pragma: no cover - variants are constructed locally.
                raise RuntimeError(
                    f"unsupported codec variant {variant.method!r}"
                )
            if codec.width != basis.fisher.width:
                raise RuntimeError("codec width does not match Fisher basis")
            by_site[site] = codec
        result[variant.variant_id] = by_site
    return result


def _candidate_schedule(
    *,
    ranks: Sequence[int],
    width: int,
    variants: Sequence[_CodecVariant],
    codecs: Mapping[str, Mapping[str, LinearActivationCodec]],
    sites: tuple[str, ...],
) -> tuple[_CodecCandidate, ...]:
    candidates = []
    for rank in ranks:
        for variant in variants:
            candidates.append(
                _CodecCandidate(
                    candidate_id=(
                        f"{variant.variant_id}.joint.rank_{rank}"
                    ),
                    variant=variant,
                    retained_rank=rank,
                    codecs={
                        site: codecs[variant.variant_id][site]
                        for site in sites
                    },
                    sites=sites,
                    # Every family must execute its own full-width behavioral
                    # identity path.  Only reduced-rank candidates are
                    # eligible for a nonfallback selection.
                    fallback=rank == width,
                )
            )
    if not variants or variants[0].method != "native_fisher":
        raise RuntimeError("native Fisher must be the first codec variant")
    return tuple(candidates)


def _full_rank_candidate(
    schedule: Sequence[_CodecCandidate],
    *,
    variant_id: str,
) -> _CodecCandidate:
    matches = [
        candidate
        for candidate in schedule
        if candidate.variant.variant_id == variant_id
        and candidate.fallback
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"candidate schedule must contain one full-rank "
            f"{variant_id!r} identity"
        )
    return matches[0]


def _project_codec_valid_positions(
    activation: Tensor,
    *,
    codec: LinearActivationCodec,
    rank: int,
    valid_positions: Tensor,
) -> Tensor:
    if (
        not isinstance(activation, Tensor)
        or not activation.is_floating_point()
        or activation.ndim != 3
        or activation.shape[-1] != codec.width
    ):
        raise ValueError(
            "codec intervention requires a floating "
            "[batch, sequence, width] activation"
        )
    if (
        not isinstance(valid_positions, Tensor)
        or valid_positions.dtype is not torch.bool
        or valid_positions.shape != activation.shape[:2]
    ):
        raise ValueError(
            "valid_positions must match the activation sequence grid"
        )
    reconstructed = codec.reconstruct(activation, rank=rank)
    mask = valid_positions.to(device=activation.device).unsqueeze(-1)
    return torch.where(mask, reconstructed, activation)


def _evaluate_candidates(
    adapter: Gemma3CausalLMAdapter,
    batches: Iterable[CalibrationBatch],
    *,
    candidates: Sequence[_CodecCandidate],
) -> ModalAblationResult:
    if not candidates:
        raise ValueError("candidate schedule cannot be empty")
    module = adapter.module
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError("candidate evaluation requires frozen weights")
    names = tuple(candidate.candidate_id for candidate in candidates)
    if len(set(names)) != len(names):
        raise ValueError("candidate IDs must be unique")
    objective = CausalLanguageModelNLL()
    baseline_nll_total = 0.0
    supervised_total = 0
    sequence_count = 0
    seen_ids: set[str] = set()
    candidate_nll = {name: 0.0 for name in names}
    candidate_matches = {name: 0 for name in names}
    examples: dict[str, list[ModalAblationExample]] = {
        name: [] for name in names
    }
    was_training = module.training
    module.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                if not isinstance(batch, CalibrationBatch):
                    raise TypeError(
                        "evaluation stream must contain CalibrationBatch"
                    )
                example_ids = _example_ids(
                    batch,
                    sequence_offset=sequence_count,
                )
                duplicates = seen_ids.intersection(example_ids)
                if duplicates:
                    raise ValueError(
                        f"duplicate evaluation example IDs: "
                        f"{sorted(duplicates)}"
                    )
                seen_ids.update(example_ids)
                baseline_run = adapter.forward(batch.model_inputs)
                baseline = _causal_lm_batch_scores(
                    baseline_run.logits,
                    batch,
                    objective=objective,
                )
                del baseline_run
                baseline_nll_total += float(
                    baseline.summed_nll.sum().item()
                )
                supervised_total += int(
                    baseline.supervised_tokens.sum().item()
                )
                sequence_count += batch.batch_size

                for candidate in candidates:
                    interventions = {}
                    for site in candidate.sites:
                        codec = candidate.codecs[site]

                        def project(
                            value: Tensor,
                            *,
                            selected_codec: LinearActivationCodec = codec,
                            selected_rank: int = candidate.retained_rank,
                            valid: Tensor = batch.valid_positions,
                        ) -> Tensor:
                            return _project_codec_valid_positions(
                                value,
                                codec=selected_codec,
                                rank=selected_rank,
                                valid_positions=valid,
                            )

                        interventions[site] = project
                    run = adapter.forward(
                        batch.model_inputs,
                        interventions=interventions,
                    )
                    ablated = _causal_lm_batch_scores(
                        run.logits,
                        batch,
                        objective=objective,
                    )
                    del run
                    if (
                        not torch.equal(
                            baseline.supervised_tokens,
                            ablated.supervised_tokens,
                        )
                        or not torch.equal(
                            baseline.supervised_mask,
                            ablated.supervised_mask,
                        )
                    ):
                        raise RuntimeError(
                            "codec changed supervised-token accounting"
                        )
                    matches = (
                        (baseline.predictions == ablated.predictions)
                        & baseline.supervised_mask
                    ).sum(dim=1)
                    name = candidate.candidate_id
                    candidate_nll[name] += float(
                        ablated.summed_nll.sum().item()
                    )
                    candidate_matches[name] += int(matches.sum().item())
                    for index, example_id in enumerate(example_ids):
                        tokens = int(
                            baseline.supervised_tokens[index].item()
                        )
                        baseline_sum = float(
                            baseline.summed_nll[index].item()
                        )
                        ablated_sum = float(
                            ablated.summed_nll[index].item()
                        )
                        matched = int(matches[index].item())
                        examples[name].append(
                            ModalAblationExample(
                                example_id=example_id,
                                supervised_tokens=tokens,
                                baseline_summed_nll=baseline_sum,
                                baseline_nll_per_token=baseline_sum / tokens,
                                ablated_summed_nll=ablated_sum,
                                ablated_nll_per_token=ablated_sum / tokens,
                                delta_summed_nll=(
                                    ablated_sum - baseline_sum
                                ),
                                delta_nll_per_token=(
                                    ablated_sum - baseline_sum
                                )
                                / tokens,
                                top1_matches=matched,
                                top1_agreement_to_baseline=matched / tokens,
                            )
                        )
    finally:
        module.train(was_training)
    if sequence_count == 0 or supervised_total == 0:
        raise ValueError("evaluation stream cannot be empty")

    baseline_aggregate = _aggregate(
        sequences=sequence_count,
        supervised_tokens=supervised_total,
        summed_nll=baseline_nll_total,
        baseline_summed_nll=baseline_nll_total,
        top1_matches=supervised_total,
    )
    results = []
    for candidate in candidates:
        name = candidate.candidate_id
        results.append(
            ModalAblationConditionResult(
                condition=candidate.condition(),
                aggregate=_aggregate(
                    sequences=sequence_count,
                    supervised_tokens=supervised_total,
                    summed_nll=candidate_nll[name],
                    baseline_summed_nll=baseline_nll_total,
                    top1_matches=candidate_matches[name],
                ),
                examples=tuple(examples[name]),
            )
        )
    return ModalAblationResult(
        baseline=baseline_aggregate,
        conditions=tuple(results),
    )


def _identity_control(
    result: ModalAblationResult,
    *,
    candidate: _CodecCandidate,
    tolerance: float,
) -> dict[str, object]:
    selected = result.condition(candidate.candidate_id)
    aggregate = selected.aggregate
    max_example = max(
        abs(example.delta_nll_per_token)
        for example in selected.examples
    )
    max_example_sum = max(
        abs(example.delta_summed_nll)
        for example in selected.examples
    )
    top1_identity = aggregate.top1_agreement_to_baseline == 1.0
    passed = (
        abs(aggregate.delta_nll_per_token) <= tolerance
        and max_example <= tolerance
        and top1_identity
    )
    return {
        "condition": candidate.condition().metadata(),
        "baseline_nll_per_token": result.baseline.nll_per_token,
        "projected_nll_per_token": aggregate.nll_per_token,
        "delta_nll_per_token": aggregate.delta_nll_per_token,
        "absolute_delta_nll_per_token": abs(
            aggregate.delta_nll_per_token
        ),
        "maximum_absolute_example_delta_nll_per_token": max_example,
        "maximum_absolute_example_delta_summed_nll": max_example_sum,
        "top1_agreement_to_baseline": (
            aggregate.top1_agreement_to_baseline
        ),
        "top1_identity": top1_identity,
        "tolerance": tolerance,
        "passed": passed,
    }


def _require_family_full_rank_identities(
    *,
    schedule: Sequence[_CodecCandidate],
    variants: Sequence[_CodecVariant],
    calibration_b: ModalAblationResult,
    tolerance: float,
) -> dict[str, dict[str, object]]:
    """Require behavioral full-width identity for every codec family.

    This gate intentionally runs before candidate selection.  A reduced
    candidate is never eligible merely because another family's full-width
    codec happens to be an identity.
    """

    controls: dict[str, dict[str, object]] = {}
    failed = []
    for variant in variants:
        candidate = _full_rank_candidate(
            schedule,
            variant_id=variant.variant_id,
        )
        control = _identity_control(
            calibration_b,
            candidate=candidate,
            tolerance=tolerance,
        )
        controls[variant.variant_id] = control
        if control["passed"] is not True:
            failed.append(variant.variant_id)
    if failed:
        raise RuntimeError(
            "calibration-B full-rank identity control failed for codec "
            f"families: {', '.join(failed)}"
        )
    return controls


def _lock_candidate(
    *,
    schedule: Sequence[_CodecCandidate],
    calibration_b: ModalAblationResult,
    nll_atol: float,
    top1_min: float,
) -> tuple[_CodecCandidate, dict[str, object]]:
    ledger = []
    locked: _CodecCandidate | None = None
    for candidate in schedule:
        result = calibration_b.condition(candidate.candidate_id).aggregate
        nll_passed = (
            abs(result.delta_nll_per_token) <= nll_atol
        )
        top1_passed = (
            result.top1_agreement_to_baseline >= top1_min
        )
        passed = nll_passed and top1_passed
        eligible = not candidate.fallback
        ledger.append(
            {
                "candidate": candidate.metadata(),
                "delta_nll_per_token": result.delta_nll_per_token,
                "top1_agreement_to_baseline": (
                    result.top1_agreement_to_baseline
                ),
                "nll_gate_passed": nll_passed,
                "top1_gate_passed": top1_passed,
                "passed": passed,
                "eligible_for_nonfallback_lock": eligible,
            }
        )
        if locked is None and eligible and passed:
            locked = candidate
    reason = "lowest_rank_passing_gates"
    if locked is None:
        locked = _full_rank_candidate(
            schedule,
            variant_id="native_fisher",
        )
        reason = "native_full_rank_fallback"
    return locked, {
        "ordering": "rank_ascending_then_predeclared_family_order",
        "nll_absolute_delta_atol": nll_atol,
        "minimum_top1_agreement": top1_min,
        "ledger": ledger,
        "locked_candidate": locked.metadata(),
        "reason": reason,
        "calibration_b_only": True,
    }


def _renamed_candidate(
    candidate: _CodecCandidate,
    *,
    name: str,
) -> _CodecCandidate:
    return _CodecCandidate(
        candidate_id=name,
        variant=candidate.variant,
        retained_rank=candidate.retained_rank,
        codecs=candidate.codecs,
        sites=candidate.sites,
        fallback=candidate.fallback,
    )


def _validation_candidates(
    *,
    locked: _CodecCandidate,
    locked_family_full_rank: _CodecCandidate,
    native_full_rank: _CodecCandidate,
) -> tuple[
    tuple[_CodecCandidate, ...],
    _CodecCandidate,
    _CodecCandidate,
]:
    locked_condition = _renamed_candidate(
        locked,
        name="locked_candidate",
    )
    locked_identity = _renamed_candidate(
        locked_family_full_rank,
        name="locked_family_full_rank_identity",
    )
    values = [locked_condition, locked_identity]
    if locked.variant.variant_id == "native_fisher":
        native_identity = locked_identity
    else:
        native_identity = _renamed_candidate(
            native_full_rank,
            name="native_full_rank_identity",
        )
        values.append(native_identity)
    return tuple(values), locked_identity, native_identity


def _finite_gate(
    value: float,
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
        or (maximum is not None and float(value) > maximum)
    ):
        raise ValueError(f"{label} is outside its valid range")
    return float(value)


def _factor_metadata(
    result: CausalWeightedJacobianResult,
) -> dict[str, object]:
    total = result.total_weighted_energy
    retained = result.retained_weighted_energy
    return {
        "algorithm": result.algorithm,
        "algorithm_version": result.algorithm_version,
        "sequence_length": result.sequence_length,
        "input_width": result.input_width,
        "output_width": result.output_width,
        "retained_ranks": result.retained_ranks,
        "total_weighted_energy": total,
        "retained_weighted_energy": retained,
        "discarded_weighted_energy": result.discarded_weighted_energy,
        "retained_weighted_energy_fraction": (
            1.0 if total == 0 else retained / total
        ),
        "dense_causal_coefficient_count": (
            result.dense_causal_coefficient_count
        ),
        "factor_coefficient_count": result.factor_coefficient_count,
        "factor_to_dense_coefficient_ratio": (
            result.factor_to_dense_coefficient_ratio
        ),
        "dense_causal_mac_count": result.dense_causal_mac_count,
        "factor_mac_count": result.factor_mac_count,
        "factor_to_dense_mac_ratio": result.factor_to_dense_mac_ratio,
        "affine_mean_coefficient_count": (
            result.affine_mean_coefficient_count
        ),
        "weighted_reference_energy_by_lag": tuple(
            float(value)
            for value in result.weighted_energy_by_lag.tolist()
        ),
        "weighted_reference_energy_fraction_by_lag": tuple(
            float(value)
            for value in result.weighted_energy_fraction_by_lag.tolist()
        ),
        "retained_approximation_energy_by_lag": tuple(
            float(value)
            for value in result.retained_weighted_energy_by_lag.tolist()
        ),
        "retained_approximation_energy_fraction_by_lag": tuple(
            float(value)
            for value in (
                result.retained_weighted_energy_fraction_by_lag.tolist()
            )
        ),
        "coordinate_scope": "projected_modal_slice",
        "projected_modal_slice_shape": (
            result.input_width,
            result.output_width,
        ),
        "full_residual_jacobian_energy_claim": False,
        "reference_structure": "causal_toeplitz_from_pooled_signed_lags",
        "reference_sequence_length_semantics": "max_lag_plus_one",
        "factor_energy_semantics": (
            "weighted_synthetic_reference_not_probe_captured_energy"
        ),
        "moment_semantics": (
            "pooled_C_and_F_replicated_by_position_no_cross_position_blocks"
        ),
        "dense_ratio_denominator": (
            "unshared_dense_causal_operator_not_lag_shared_storage"
        ),
        "behavioral_validation": False,
        "variable_length_executor_claim": False,
        "compression_claim": False,
        "runtime_acceptance_claim": False,
    }


def _factor_jacobian_pilot(
    *,
    statistics: CausalLagJacobianStatistics,
    input_codec: LinearActivationCodec,
    output_codec: LinearActivationCodec,
    input_covariance: ActivationCovarianceResult,
    output_fisher: Tensor,
    retained_rank: int,
) -> CausalWeightedJacobianResult:
    """Merge a bounded signed lag map using activation/Fisher weighting."""

    input_modes = statistics.input_modes
    output_modes = statistics.output_modes
    if (
        input_codec.activation_name != statistics.input_activation
        or output_codec.activation_name != statistics.output_activation
    ):
        raise ValueError("Jacobian statistics do not match selected codecs")
    if type(retained_rank) is not int or not (
        0 <= retained_rank <= output_modes
    ):
        raise ValueError(
            "Jacobian factor rank must be within the output pilot width"
        )
    sequence_length = statistics.max_lag + 1
    jacobian = torch.zeros(
        sequence_length,
        output_modes,
        sequence_length,
        input_modes,
        dtype=torch.float64,
    )
    lag_maps = statistics.lag_matrices
    for target in range(sequence_length):
        for source in range(target + 1):
            lag = target - source
            jacobian[target, :, source, :] = lag_maps[lag].T

    input_encoder = input_codec.encoder[:, :input_modes]
    modal_covariance = (
        input_encoder.T
        @ input_covariance.covariance
        @ input_encoder
    )
    modal_covariance = (modal_covariance + modal_covariance.T) * 0.5
    output_decoder = output_codec.decoder[:, :output_modes]
    raw_output_fisher = output_fisher.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    modal_fisher = (
        output_decoder.T
        @ raw_output_fisher
        @ output_decoder
    )
    modal_fisher = (modal_fisher + modal_fisher.T) * 0.5
    input_blocks = modal_covariance.unsqueeze(0).repeat(
        sequence_length,
        1,
        1,
    )
    output_blocks = modal_fisher.unsqueeze(0).repeat(
        sequence_length,
        1,
        1,
    )
    return factor_causal_weighted_jacobian(
        jacobian,
        input_blocks,
        output_blocks,
        retained_ranks=retained_rank,
    )


def _build_report(
    *,
    payload: Mapping[str, object],
    fisher: StreamingFisherCollection,
    covariances: Mapping[str, ActivationCovarianceResult],
    codecs: Mapping[str, Mapping[str, LinearActivationCodec]],
    jacobian: CausalLagJacobianStatistics | None,
    merged_factor: CausalWeightedJacobianResult | None,
    output: Path,
    scientific_digest: str,
) -> dict[str, object]:
    calibration_b = payload["calibration_b"]
    validation = payload["validation"]
    jacobian_payload = payload["jacobian"]
    calibration_a_payload = payload["calibration_a"]
    assert isinstance(calibration_b, Mapping)
    assert isinstance(validation, Mapping)
    assert isinstance(jacobian_payload, Mapping)
    assert isinstance(calibration_a_payload, Mapping)
    raw_codec_states = calibration_a_payload["codecs"]
    assert isinstance(raw_codec_states, Mapping)
    codec_tensor_fields = {
        "mean",
        "encoder",
        "decoder",
        "importance_scores",
        "eigenvalues",
    }
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "scientific_status": {
            "scope": "split_safe_activation_aware_modal_selection",
            "fit_split": "calibration_a_only",
            "selection_split": "calibration_b_only",
            "validation_locked_before_evaluation": True,
            "all_family_full_rank_identities_passed": True,
            "locked_family_full_rank_identity_passed": True,
            "native_full_rank_identity_passed": True,
            "test_split_evaluated": False,
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "prompt_text_in_artifact": False,
            "tokenizer_state_in_artifact": False,
            "compilation_claim": False,
            "compression_claim": False,
            "speed_claim": False,
        },
        "model": copy.deepcopy(dict(payload["model"])),  # type: ignore[arg-type]
        "protocol": copy.deepcopy(
            dict(payload["protocol"])  # type: ignore[arg-type]
        ),
        "analysis": {
            "calibration_a": {
                "fisher": fisher.metadata(),
                "activation_covariance": {
                    site: result.metadata()
                    for site, result in covariances.items()
                },
                "codecs": {
                    variant: {
                        site: {
                            key: copy.deepcopy(value)
                            for key, value in state.items()
                            if key not in codec_tensor_fields
                        }
                        for site, state in by_site.items()
                    }
                    for variant, by_site in raw_codec_states.items()
                },
            },
            "calibration_b": {
                "candidate_evaluation": copy.deepcopy(
                    calibration_b["candidate_evaluation"]
                ),
                "selection": copy.deepcopy(calibration_b["selection"]),
                "family_full_rank_identities": copy.deepcopy(
                    calibration_b["family_full_rank_identities"]
                ),
                "locked_family_full_rank_identity": copy.deepcopy(
                    calibration_b[
                        "locked_family_full_rank_identity"
                    ]
                ),
                "native_full_rank_identity": copy.deepcopy(
                    calibration_b["native_full_rank_identity"]
                ),
            },
            "jacobian": {
                "enabled": jacobian_payload["enabled"],
                "codec_binding": copy.deepcopy(
                    jacobian_payload["codec_binding"]
                ),
                "statistics": (
                    None if jacobian is None else jacobian.metadata()
                ),
                "merged_factor": (
                    copy.deepcopy(
                        jacobian_payload["merged_factor_metadata"]
                    )
                ),
                "tokenized_stream": copy.deepcopy(
                    jacobian_payload["tokenized_stream"]
                ),
            },
            "validation": {
                "locked_evaluation": copy.deepcopy(
                    validation["locked_evaluation"]
                ),
                "locked_family_full_rank_identity": copy.deepcopy(
                    validation[
                        "locked_family_full_rank_identity"
                    ]
                ),
                "native_full_rank_identity": copy.deepcopy(
                    validation["native_full_rank_identity"]
                ),
            },
        },
        "artifact": {
            "tensor_output": output.name,
            "contains_model_state_dict": False,
            "contains_tokenizer": False,
            "contains_prompt_text": False,
            "scientific_payload_sha256": scientific_digest,
        },
    }


def run_gemma3_weighted_jacobian(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    prompt_splits_path: Path | str = DEFAULT_PROMPT_SPLITS,
    start_layer: int = DEFAULT_START_LAYER,
    end_layer: int = DEFAULT_END_LAYER,
    max_length: int = 128,
    tokenization_batch_size: int = 4,
    ranks: Iterable[int] = DEFAULT_RANKS,
    generalized_regularization_pairs: Iterable[
        tuple[float, float]
    ] = DEFAULT_GENERALIZED_REGULARIZATION,
    sketch_rows: int | None = None,
    selection_nll_atol: float = DEFAULT_SELECTION_NLL_ATOL,
    selection_top1_min: float = DEFAULT_SELECTION_TOP1_MIN,
    full_rank_nll_atol: float = DEFAULT_FULL_RANK_NLL_ATOL,
    jacobian_max_sequences: int = DEFAULT_JACOBIAN_MAX_SEQUENCES,
    jacobian_modes: int = DEFAULT_JACOBIAN_MODES,
    jacobian_max_lag: int = DEFAULT_JACOBIAN_MAX_LAG,
    jacobian_factor_rank: int = DEFAULT_JACOBIAN_FACTOR_RANK,
    device_name: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Fit on A, select on B, and evaluate the locked codec on validation."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if (
        type(start_layer) is not int
        or type(end_layer) is not int
        or start_layer < 0
        or end_layer < start_layer
    ):
        raise ValueError("layer range must be nonnegative and ascending")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    selection_nll = _finite_gate(
        selection_nll_atol,
        label="selection_nll_atol",
        minimum=0.0,
    )
    selection_top1 = _finite_gate(
        selection_top1_min,
        label="selection_top1_min",
        minimum=0.0,
        maximum=1.0,
    )
    identity_tolerance = _finite_gate(
        full_rank_nll_atol,
        label="full_rank_nll_atol",
        minimum=0.0,
    )
    if (
        type(jacobian_max_sequences) is not int
        or jacobian_max_sequences < 0
    ):
        raise ValueError("jacobian_max_sequences must be nonnegative")
    if type(jacobian_modes) is not int or jacobian_modes <= 0:
        raise ValueError("jacobian_modes must be positive")
    if type(jacobian_max_lag) is not int or jacobian_max_lag < 0:
        raise ValueError("jacobian_max_lag must be nonnegative")
    if type(jacobian_factor_rank) is not int or jacobian_factor_rank <= 0:
        raise ValueError("jacobian_factor_rank must be positive")
    regularization_pairs = _validated_regularization_pairs(
        generalized_regularization_pairs
    )
    resolved_output = (
        default_gemma3_weighted_jacobian_output(
            model_id,
            start_layer,
            end_layer,
        )
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")

    # Loading validates all four prompt splits.  Only hashes from the returned
    # metadata survive into the artifact.
    prompt_splits = load_gemma3_prompt_splits(prompt_splits_path)
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    model.eval()
    model.requires_grad_(False)
    guard = _FrozenModelTensorGuard(model)
    adapter = Gemma3CausalLMAdapter(model)
    plan = adapter.plan_layer_block(start_layer, end_layer)
    if len(set(plan.widths)) != 1:
        raise ValueError(
            "weighted-Jacobian selection requires a shared residual width"
        )
    width = plan.widths[0]
    resolved_ranks = _validated_ranks(ranks, width=width)
    resolved_sketch_rows = (
        width + 1 if sketch_rows is None else sketch_rows
    )
    if (
        type(resolved_sketch_rows) is not int
        or resolved_sketch_rows < width + 1
    ):
        raise ValueError(
            "sketch_rows must be at least residual width + 1"
        )

    calibration_a_provenance = _CalibrationStreamProvenance(
        "calibration_a",
        prompt_splits.calibration_a,
    )
    calibration_a_batches = make_causal_lm_calibration_batches(
        tokenizer,
        prompt_splits.calibration_a,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    fisher, covariances = _fit_calibration_a(
        adapter,
        calibration_a_provenance.wrap(calibration_a_batches),
        plan=plan,
        width=width,
        sketch_rows=resolved_sketch_rows,
    )
    calibration_a_stream = calibration_a_provenance.metadata()
    guard.assert_unchanged()

    codec_variants = _variants(regularization_pairs)
    codecs = _build_codec_families(
        fisher=fisher,
        covariances=covariances,
        variants=codec_variants,
    )
    gated_sites = plan.activation_sites[1:]
    schedule = _candidate_schedule(
        ranks=resolved_ranks,
        width=width,
        variants=codec_variants,
        codecs=codecs,
        sites=gated_sites,
    )

    calibration_b_provenance = _CalibrationStreamProvenance(
        "calibration_b",
        prompt_splits.calibration_b,
    )
    calibration_b_batches = make_causal_lm_calibration_batches(
        tokenizer,
        prompt_splits.calibration_b,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    calibration_b = _evaluate_candidates(
        adapter,
        calibration_b_provenance.wrap(calibration_b_batches),
        candidates=schedule,
    )
    calibration_b_stream = calibration_b_provenance.metadata()
    guard.assert_unchanged()

    family_calibration_b_identities = (
        _require_family_full_rank_identities(
            schedule=schedule,
            variants=codec_variants,
            calibration_b=calibration_b,
            tolerance=identity_tolerance,
        )
    )
    native_full_rank = _full_rank_candidate(
        schedule,
        variant_id="native_fisher",
    )
    native_calibration_b_identity = family_calibration_b_identities[
        "native_fisher"
    ]
    locked, selection = _lock_candidate(
        schedule=schedule,
        calibration_b=calibration_b,
        nll_atol=selection_nll,
        top1_min=selection_top1,
    )
    locked_family_full_rank = _full_rank_candidate(
        schedule,
        variant_id=locked.variant.variant_id,
    )
    locked_calibration_b_identity = family_calibration_b_identities[
        locked.variant.variant_id
    ]

    # This intentionally happens only after calibration B has locked the
    # family/rank.  It replays a bounded prefix of calibration A and cannot
    # influence selection or validation.
    jacobian: CausalLagJacobianStatistics | None = None
    merged_factor: CausalWeightedJacobianResult | None = None
    jacobian_stream: dict[str, object] | None = None
    jacobian_codec_binding: dict[str, object] | None = None
    if jacobian_max_sequences > 0:
        jacobian_prompts = prompt_splits.calibration_a[
            : min(
                jacobian_max_sequences,
                len(prompt_splits.calibration_a),
            )
        ]
        jacobian_provenance = _CalibrationStreamProvenance(
            "calibration_a_jacobian",
            jacobian_prompts,
        )
        jacobian_batches = make_causal_lm_calibration_batches(
            tokenizer,
            jacobian_prompts,
            max_length=max_length,
            # A batch size of one makes the exact consumed-sequence budget
            # identical to its stream provenance.
            tokenization_batch_size=1,
            device=device,
        )
        selected_site_codecs = codecs[locked.variant.variant_id]
        bounded_modes = min(
            jacobian_modes,
            locked.retained_rank,
            width,
        )
        jacobian = collect_block_causal_lag_jacobian(
            adapter,
            plan,
            jacobian_provenance.wrap(jacobian_batches),
            input_codec=selected_site_codecs[
                plan.activation_sites[0]
            ],
            output_codec=selected_site_codecs[
                plan.activation_sites[-1]
            ],
            input_modes=bounded_modes,
            output_modes=bounded_modes,
            max_lag=jacobian_max_lag,
            max_sequences=len(jacobian_prompts),
        )
        merged_factor = _factor_jacobian_pilot(
            statistics=jacobian,
            input_codec=selected_site_codecs[
                plan.activation_sites[0]
            ],
            output_codec=selected_site_codecs[
                plan.activation_sites[-1]
            ],
            input_covariance=covariances[
                plan.activation_sites[0]
            ],
            output_fisher=fisher.bases[
                plan.activation_sites[-1]
            ].fisher.approximate_matrix(),
            retained_rank=min(
                jacobian_factor_rank,
                bounded_modes,
            ),
        )
        jacobian_codec_binding = {
            "locked_candidate_id": locked.candidate_id,
            "variant_id": locked.variant.variant_id,
            "input_activation": plan.activation_sites[0],
            "output_activation": plan.activation_sites[-1],
            "input_codec_sha256": _codec_state_sha256(
                selected_site_codecs[plan.activation_sites[0]]
            ),
            "output_codec_sha256": _codec_state_sha256(
                selected_site_codecs[plan.activation_sites[-1]]
            ),
            "coordinate_gauge_policy": (
                "energy_comparisons_only_within_locked_codec_family"
            ),
        }
        jacobian_stream = jacobian_provenance.metadata()
        guard.assert_unchanged()

    (
        validation_candidates,
        locked_family_identity_validation,
        native_identity_validation,
    ) = _validation_candidates(
        locked=locked,
        locked_family_full_rank=locked_family_full_rank,
        native_full_rank=native_full_rank,
    )
    validation_provenance = _CalibrationStreamProvenance(
        "validation",
        prompt_splits.validation,
    )
    validation_batches = make_causal_lm_calibration_batches(
        tokenizer,
        prompt_splits.validation,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    validation = _evaluate_candidates(
        adapter,
        validation_provenance.wrap(validation_batches),
        candidates=validation_candidates,
    )
    validation_stream = validation_provenance.metadata()
    guard.assert_unchanged()
    locked_validation_identity = _identity_control(
        validation,
        candidate=locked_family_identity_validation,
        tolerance=identity_tolerance,
    )
    if not locked_validation_identity["passed"]:
        raise RuntimeError(
            "validation locked-family full-rank identity control failed"
        )
    native_validation_identity = _identity_control(
        validation,
        candidate=native_identity_validation,
        tolerance=identity_tolerance,
    )
    if not native_validation_identity["passed"]:
        raise RuntimeError(
            "validation native full-rank identity control failed"
        )

    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )
    protocol = {
        "start_layer": start_layer,
        "end_layer_inclusive": end_layer,
        "layer_ids": plan.layer_ids,
        "canonical_boundaries": plan.activation_sites,
        "boundary_widths": plan.widths,
        "leaf_boundary": plan.leaf_activation_name,
        "gated_output_sites": gated_sites,
        "residual_width": width,
        "ranks": resolved_ranks,
        "codec_variants": tuple(
            variant.metadata() for variant in codec_variants
        ),
        "candidate_schedule": tuple(
            candidate.metadata() for candidate in schedule
        ),
        "generalized_regularization_pairs": regularization_pairs,
        "extraction_rank": width,
        "sketch_rows": resolved_sketch_rows,
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "gradient_batching": "one_sequence_at_a_time",
        "selection_batching": "one_forward_per_batch_per_candidate",
        "validation_batching": (
            "baseline_plus_locked_candidate_plus_locked_family_and_native_"
            "full_rank_identities_with_native_deduplication"
        ),
        "score": "summed_hard_target_next_token_nll",
        "score_compute_dtype": (
            "float32_for_float16_or_bfloat16_logits"
        ),
        "fisher_scope": "width_pooled",
        "activation_covariance_scope": "width_pooled_population",
        "normalizer": "valid_activation_positions",
        "fit_policy": "calibration_a_only",
        "selection_policy": (
            "calibration_b_lowest_rank_then_family_passing_both_gates"
        ),
        "selection_nll_atol": selection_nll,
        "selection_top1_min": selection_top1,
        "full_rank_nll_atol": identity_tolerance,
        "validation_policy": (
            "locked_candidate_plus_locked_family_and_native_full_rank_"
            "identities_deduplicated_when_same_family"
        ),
        "calibration_b_identity_policy": (
            "predeclared_full_rank_identity_for_every_codec_family"
        ),
        "jacobian_policy": (
            "post_selection_calibration_a_prefix_true_forward_jvp"
        ),
        "jacobian_max_sequences": jacobian_max_sequences,
        "jacobian_modes": jacobian_modes,
        "jacobian_max_lag": jacobian_max_lag,
        "jacobian_factor_rank": jacobian_factor_rank,
        "test_policy": "parse_validate_hash_only",
        "claim": "diagnostic_node_and_edge_selection_only",
        "cache_policy": "external_to_git_worktree",
        "model_state_guard": guard.metadata(),
        "library_versions": _library_versions(),
        "tokenizer": _tokenizer_provenance(tokenizer),
        "tokenized_splits": {
            "calibration_a": calibration_a_stream,
            "calibration_b": calibration_b_stream,
            "validation": validation_stream,
            "calibration_a_jacobian": jacobian_stream,
        },
        "prompt_splits": prompt_splits.metadata(),
    }
    payload = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "model": model_metadata,
        "protocol": protocol,
        "calibration_a": {
            "fisher": fisher.state_dict(),
            "activation_covariance": {
                site: covariance.state_dict()
                for site, covariance in covariances.items()
            },
            "codecs": {
                variant: {
                    site: codec.state_dict()
                    for site, codec in by_site.items()
                }
                for variant, by_site in codecs.items()
            },
            "tokenized_stream": calibration_a_stream,
        },
        "calibration_b": {
            "candidate_evaluation": calibration_b.metadata(),
            "selection": selection,
            "family_full_rank_identities": (
                family_calibration_b_identities
            ),
            "locked_family_full_rank_identity": (
                locked_calibration_b_identity
            ),
            "native_full_rank_identity": native_calibration_b_identity,
            "tokenized_stream": calibration_b_stream,
        },
        "jacobian": {
            "enabled": jacobian is not None,
            "statistics": (
                None if jacobian is None else jacobian.state_dict()
            ),
            "merged_factor": (
                None
                if merged_factor is None
                else merged_factor.state_dict()
            ),
            "merged_factor_metadata": (
                None
                if merged_factor is None
                else _factor_metadata(merged_factor)
            ),
            "codec_binding": jacobian_codec_binding,
            "tokenized_stream": jacobian_stream,
        },
        "validation": {
            "locked_evaluation": validation.metadata(),
            "locked_candidate": locked.metadata(),
            "locked_family_full_rank_identity": (
                locked_validation_identity
            ),
            "native_full_rank_identity": native_validation_identity,
            "tokenized_stream": validation_stream,
        },
    }
    scientific_digest = _scientific_payload_sha256(payload)
    report = _build_report(
        payload=payload,
        fisher=fisher,
        covariances=covariances,
        codecs=codecs,
        jacobian=jacobian,
        merged_factor=merged_factor,
        output=resolved_output,
        scientific_digest=scientific_digest,
    )
    report_digest = _report_sha256(report)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **payload,
            "scientific_payload_sha256": scientific_digest,
            "report_sha256": report_digest,
        },
        resolved_output,
    )
    resolved_output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def _modal_result_from_metadata(
    value: Mapping[str, object],
) -> ModalAblationResult:
    baseline_raw = value["baseline"]
    conditions_raw = value["conditions"]
    assert isinstance(baseline_raw, Mapping)
    assert isinstance(conditions_raw, list)
    baseline = ModalAblationAggregate(**dict(baseline_raw))
    conditions = []
    for raw in conditions_raw:
        assert isinstance(raw, Mapping)
        condition_raw = raw["condition"]
        aggregate_raw = raw["aggregate"]
        examples_raw = raw["examples"]
        assert isinstance(condition_raw, Mapping)
        assert isinstance(aggregate_raw, Mapping)
        assert isinstance(examples_raw, list)
        conditions.append(
            ModalAblationConditionResult(
                condition=ModalAblationCondition(
                    name=str(condition_raw["name"]),
                    retained_modes=dict(
                        condition_raw["retained_modes"]  # type: ignore[arg-type]
                    ),
                ),
                aggregate=ModalAblationAggregate(
                    **dict(aggregate_raw)
                ),
                examples=tuple(
                    ModalAblationExample(**dict(example))
                    for example in examples_raw
                ),
            )
        )
    return ModalAblationResult(
        baseline=baseline,
        conditions=tuple(conditions),
    )


def _state_values_close(actual: object, expected: object) -> bool:
    if isinstance(actual, Tensor) and isinstance(expected, Tensor):
        return (
            actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and torch.allclose(
                actual,
                expected,
                rtol=2e-11,
                atol=2e-12,
            )
        )
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return (
            set(actual) == set(expected)
            and all(
                _state_values_close(actual[key], expected[key])
                for key in actual
            )
        )
    if isinstance(actual, (tuple, list)) and isinstance(
        expected,
        (tuple, list),
    ):
        return (
            len(actual) == len(expected)
            and all(
                _state_values_close(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(
            actual,
            expected,
            rel_tol=2e-11,
            abs_tol=2e-12,
        )
    return actual == expected


def _validate_prompt_provenance(
    value: object,
    *,
    streams: Mapping[str, Mapping[str, object] | None],
    jacobian_enabled: bool,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "scientific_status",
        "counts",
        "normalized_sha256",
        "per_prompt_sha256",
    }:
        raise ValueError("prompt split provenance fields are invalid")
    split_names = {
        "calibration_a",
        "calibration_b",
        "validation",
        "test",
    }
    counts = value["counts"]
    normalized = value["normalized_sha256"]
    per_prompt = value["per_prompt_sha256"]
    if (
        not isinstance(value["scientific_status"], str)
        or not value["scientific_status"]
        or not isinstance(counts, Mapping)
        or set(counts) != split_names
        or not isinstance(normalized, Mapping)
        or set(normalized) != split_names
        or not isinstance(per_prompt, Mapping)
        or set(per_prompt) != split_names
    ):
        raise ValueError("prompt split provenance mappings are invalid")
    all_hashes = []
    for split in split_names:
        hashes = per_prompt[split]
        if (
            type(counts[split]) is not int
            or counts[split] <= 0
            or not isinstance(hashes, list)
            or len(hashes) != counts[split]
            or any(not _is_sha256(item) for item in hashes)
            or normalized[split]
            != _ordered_prompt_hash_digest(hashes)
        ):
            raise ValueError("prompt split hashes are invalid")
        all_hashes.extend(hashes)
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("prompt split hashes must be disjoint")
    for split in ("calibration_a", "calibration_b", "validation"):
        stream = streams[split]
        if (
            not isinstance(stream, Mapping)
            or stream["source_prompt_sha256"] != per_prompt[split]
            or stream["sequences"] != counts[split]
        ):
            raise ValueError(
                f"{split} tokenized stream does not match prompt hashes"
            )
    jacobian_stream = streams["calibration_a_jacobian"]
    if jacobian_enabled:
        if not isinstance(jacobian_stream, Mapping):
            raise ValueError("enabled Jacobian lacks tokenized provenance")
        source = jacobian_stream["source_prompt_sha256"]
        if (
            not isinstance(source, list)
            or source != per_prompt["calibration_a"][: len(source)]
        ):
            raise ValueError(
                "Jacobian stream is not a calibration-A prefix"
            )
    elif jacobian_stream is not None:
        raise ValueError("disabled Jacobian cannot have tokenized provenance")
    streamed = {
        item
        for stream in streams.values()
        if isinstance(stream, Mapping)
        for item in stream["source_prompt_sha256"]  # type: ignore[index]
    }
    if streamed & set(per_prompt["test"]):
        raise ValueError("reserved test prompts appear in tokenized streams")
    return copy.deepcopy(dict(value))


def load_gemma3_weighted_jacobian_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strictly load and cross-check a weighted-Jacobian artifact."""

    artifact_path = Path(path)
    raw = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    required = {
        "schema",
        "format_version",
        "contains_model_weights",
        "contains_prompt_text",
        "contains_tokenizer_state",
        "model",
        "protocol",
        "calibration_a",
        "calibration_b",
        "jacobian",
        "validation",
        "scientific_payload_sha256",
        "report_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("weighted-Jacobian artifact fields are invalid")
    if (
        raw["schema"] != _ARTIFACT_SCHEMA
        or raw["format_version"] != _ARTIFACT_FORMAT_VERSION
    ):
        raise ValueError("unsupported weighted-Jacobian artifact")
    if (
        raw["contains_model_weights"] is not False
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
    ):
        raise ValueError("artifact contains forbidden source state")
    if (
        not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError("artifact digest fields are invalid")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    scientific_digest = _scientific_payload_sha256(payload)
    if scientific_digest != raw["scientific_payload_sha256"]:
        raise ValueError("weighted-Jacobian scientific payload digest mismatch")

    model = _validate_model_metadata(raw["model"])
    protocol_raw = raw["protocol"]
    protocol_fields = {
        "start_layer",
        "end_layer_inclusive",
        "layer_ids",
        "canonical_boundaries",
        "boundary_widths",
        "leaf_boundary",
        "gated_output_sites",
        "residual_width",
        "ranks",
        "codec_variants",
        "candidate_schedule",
        "generalized_regularization_pairs",
        "extraction_rank",
        "sketch_rows",
        "maximum_tokenized_length",
        "tokenization_batch_size",
        "gradient_batching",
        "selection_batching",
        "validation_batching",
        "score",
        "score_compute_dtype",
        "fisher_scope",
        "activation_covariance_scope",
        "normalizer",
        "fit_policy",
        "selection_policy",
        "selection_nll_atol",
        "selection_top1_min",
        "full_rank_nll_atol",
        "calibration_b_identity_policy",
        "validation_policy",
        "jacobian_policy",
        "jacobian_max_sequences",
        "jacobian_modes",
        "jacobian_max_lag",
        "jacobian_factor_rank",
        "test_policy",
        "claim",
        "cache_policy",
        "model_state_guard",
        "library_versions",
        "tokenizer",
        "tokenized_splits",
        "prompt_splits",
    }
    if (
        not isinstance(protocol_raw, Mapping)
        or set(protocol_raw) != protocol_fields
    ):
        raise ValueError("weighted-Jacobian protocol fields are invalid")
    protocol = copy.deepcopy(dict(protocol_raw))
    width = protocol["residual_width"]
    boundaries = protocol["canonical_boundaries"]
    widths = protocol["boundary_widths"]
    sites = protocol["gated_output_sites"]
    start = protocol["start_layer"]
    end = protocol["end_layer_inclusive"]
    if (
        type(width) is not int
        or width <= 0
        or type(boundaries) is not tuple
        or len(boundaries) < 2
        or type(widths) is not tuple
        or widths != (width,) * len(boundaries)
        or sites != boundaries[1:]
        or protocol["leaf_boundary"] != boundaries[0]
        or type(start) is not int
        or type(end) is not int
        or start < 0
        or end < start
        or len(protocol["layer_ids"]) != end - start + 1
    ):
        raise ValueError("weighted-Jacobian block geometry is invalid")
    if (
        model["hidden_size"] is not None
        and model["hidden_size"] != width
    ):
        raise ValueError("model hidden size does not match residual width")
    if (
        model["num_hidden_layers"] is not None
        and end >= model["num_hidden_layers"]
    ):
        raise ValueError("block range exceeds model layer metadata")
    ranks = _validated_ranks(protocol["ranks"], width=width)
    if ranks != protocol["ranks"]:
        raise ValueError("rank schedule is not canonical")
    pairs = _validated_regularization_pairs(
        protocol["generalized_regularization_pairs"]
    )
    if pairs != protocol["generalized_regularization_pairs"]:
        raise ValueError("regularization schedule is not canonical")
    expected_variants = _variants(pairs)
    if protocol["codec_variants"] != tuple(
        variant.metadata() for variant in expected_variants
    ):
        raise ValueError("codec variant schedule is invalid")
    if (
        protocol["extraction_rank"] != width
        or type(protocol["sketch_rows"]) is not int
        or protocol["sketch_rows"] < width + 1
        or protocol["gradient_batching"] != "one_sequence_at_a_time"
        or protocol["selection_batching"]
        != "one_forward_per_batch_per_candidate"
        or protocol["validation_batching"]
        != "baseline_plus_locked_candidate_plus_locked_family_and_native_"
        "full_rank_identities_with_native_deduplication"
        or protocol["score"]
        != "summed_hard_target_next_token_nll"
        or protocol["score_compute_dtype"]
        != "float32_for_float16_or_bfloat16_logits"
        or protocol["fisher_scope"] != "width_pooled"
        or protocol["activation_covariance_scope"]
        != "width_pooled_population"
        or protocol["normalizer"] != "valid_activation_positions"
        or protocol["fit_policy"] != "calibration_a_only"
        or protocol["selection_policy"]
        != "calibration_b_lowest_rank_then_family_passing_both_gates"
        or protocol["validation_policy"]
        != "locked_candidate_plus_locked_family_and_native_full_rank_"
        "identities_deduplicated_when_same_family"
        or protocol["calibration_b_identity_policy"]
        != "predeclared_full_rank_identity_for_every_codec_family"
        or protocol["jacobian_policy"]
        != "post_selection_calibration_a_prefix_true_forward_jvp"
        or protocol["test_policy"] != "parse_validate_hash_only"
        or protocol["claim"] != "diagnostic_node_and_edge_selection_only"
        or protocol["cache_policy"] != "external_to_git_worktree"
        or type(protocol["maximum_tokenized_length"]) is not int
        or protocol["maximum_tokenized_length"] < 2
        or type(protocol["tokenization_batch_size"]) is not int
        or protocol["tokenization_batch_size"] <= 0
        or type(protocol["jacobian_max_sequences"]) is not int
        or protocol["jacobian_max_sequences"] < 0
        or type(protocol["jacobian_modes"]) is not int
        or protocol["jacobian_modes"] <= 0
        or type(protocol["jacobian_max_lag"]) is not int
        or protocol["jacobian_max_lag"] < 0
        or type(protocol["jacobian_factor_rank"]) is not int
        or protocol["jacobian_factor_rank"] <= 0
    ):
        raise ValueError("weighted-Jacobian scientific semantics are invalid")
    guard = protocol["model_state_guard"]
    if (
        not isinstance(guard, Mapping)
        or set(guard)
        != {
            "verified",
            "training",
            "parameters_frozen",
            "parameter_tensors",
            "buffer_tensors",
            "checks",
        }
        or guard["verified"] is not True
        or guard["training"] is not False
        or guard["parameters_frozen"] is not True
        or type(guard["parameter_tensors"]) is not int
        or guard["parameter_tensors"] < 0
        or type(guard["buffer_tensors"]) is not int
        or guard["buffer_tensors"] < 0
        or guard["checks"]
        != (
            "tensor_object_identity",
            "tensor_version_counter",
            "tensor_storage_identity",
        )
    ):
        raise ValueError("model-state guard metadata is invalid")
    libraries = protocol["library_versions"]
    if (
        not isinstance(libraries, Mapping)
        or set(libraries)
        != {
            "python",
            "torch",
            "transformers",
            "tokenizers",
            "sentencepiece",
        }
        or any(
            item is not None and not isinstance(item, str)
            for item in libraries.values()
        )
    ):
        raise ValueError("library provenance is invalid")
    tokenizer = protocol["tokenizer"]
    if (
        not isinstance(tokenizer, Mapping)
        or set(tokenizer)
        != {
            "tokenizer_class",
            "name_or_path",
            "configuration_sha256",
        }
        or not isinstance(tokenizer["tokenizer_class"], str)
        or not tokenizer["tokenizer_class"]
        or (
            tokenizer["name_or_path"] is not None
            and not isinstance(tokenizer["name_or_path"], str)
        )
        or not _is_sha256(tokenizer["configuration_sha256"])
    ):
        raise ValueError("tokenizer provenance is invalid")
    selection_nll = _finite_gate(
        protocol["selection_nll_atol"],
        label="selection_nll_atol",
        minimum=0.0,
    )
    selection_top1 = _finite_gate(
        protocol["selection_top1_min"],
        label="selection_top1_min",
        minimum=0.0,
        maximum=1.0,
    )
    identity_tolerance = _finite_gate(
        protocol["full_rank_nll_atol"],
        label="full_rank_nll_atol",
        minimum=0.0,
    )

    calibration_a_raw = raw["calibration_a"]
    calibration_b_raw = raw["calibration_b"]
    validation_raw = raw["validation"]
    jacobian_raw = raw["jacobian"]
    if (
        not isinstance(calibration_a_raw, Mapping)
        or set(calibration_a_raw)
        != {
            "fisher",
            "activation_covariance",
            "codecs",
            "tokenized_stream",
        }
        or not isinstance(calibration_b_raw, Mapping)
        or set(calibration_b_raw)
        != {
            "candidate_evaluation",
            "selection",
            "family_full_rank_identities",
            "locked_family_full_rank_identity",
            "native_full_rank_identity",
            "tokenized_stream",
        }
        or not isinstance(validation_raw, Mapping)
        or set(validation_raw)
        != {
            "locked_evaluation",
            "locked_candidate",
            "locked_family_full_rank_identity",
            "native_full_rank_identity",
            "tokenized_stream",
        }
        or not isinstance(jacobian_raw, Mapping)
        or set(jacobian_raw)
        != {
            "enabled",
            "statistics",
            "merged_factor",
            "merged_factor_metadata",
            "codec_binding",
            "tokenized_stream",
        }
    ):
        raise ValueError("weighted-Jacobian payload structure is invalid")

    calibration_a_stream, calibration_a_tokens = (
        _validated_tokenized_stream(
            calibration_a_raw["tokenized_stream"],
            split_name="calibration_a",
        )
    )
    calibration_b_stream, _ = _validated_tokenized_stream(
        calibration_b_raw["tokenized_stream"],
        split_name="calibration_b",
    )
    validation_stream, _ = _validated_tokenized_stream(
        validation_raw["tokenized_stream"],
        split_name="validation",
    )
    enabled = jacobian_raw["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("Jacobian enabled flag is invalid")
    if enabled is not (protocol["jacobian_max_sequences"] > 0):
        raise ValueError("Jacobian enabled flag disagrees with its budget")
    jacobian_stream = None
    if enabled:
        jacobian_stream, _ = _validated_tokenized_stream(
            jacobian_raw["tokenized_stream"],
            split_name="calibration_a_jacobian",
        )
    elif jacobian_raw["tokenized_stream"] is not None:
        raise ValueError("disabled Jacobian has tokenized provenance")
    streams = {
        "calibration_a": calibration_a_stream,
        "calibration_b": calibration_b_stream,
        "validation": validation_stream,
        "calibration_a_jacobian": jacobian_stream,
    }
    _validate_prompt_provenance(
        protocol["prompt_splits"],
        streams=streams,
        jacobian_enabled=enabled,
    )
    if protocol["tokenized_splits"] != streams:
        raise ValueError("protocol tokenized split binding is invalid")

    fisher_state = calibration_a_raw["fisher"]
    if not isinstance(fisher_state, Mapping):
        raise ValueError("calibration-A Fisher state is invalid")
    fisher = StreamingFisherCollection.from_state_dict(fisher_state)
    if (
        tuple(fisher.bases) != boundaries
        or fisher.sequences != calibration_a_stream["sequences"]
    ):
        raise ValueError("Fisher collection does not match block/provenance")
    raw_covariances = calibration_a_raw["activation_covariance"]
    raw_codecs = calibration_a_raw["codecs"]
    if (
        not isinstance(raw_covariances, Mapping)
        or tuple(raw_covariances) != boundaries
        or not isinstance(raw_codecs, Mapping)
        or tuple(raw_codecs)
        != tuple(variant.variant_id for variant in expected_variants)
    ):
        raise ValueError("calibration-A covariance/codec sites are invalid")
    covariances = {}
    for site in boundaries:
        state = raw_covariances[site]
        if not isinstance(state, Mapping):
            raise TypeError("activation covariance state must be a mapping")
        covariance = ActivationCovarianceResult.from_state_dict(state)
        basis = fisher.bases[site]
        if (
            covariance.activation_name != site
            or covariance.width != width
            or covariance.observations != calibration_a_tokens
            or covariance.rows_seen != calibration_a_tokens
            or basis.fisher.observations != calibration_a_tokens
            or basis.fisher.rows_seen != calibration_a_tokens
            or basis.fisher.width != width
            or basis.fisher.modes != width
            or basis.fisher.requested_rank != width
            or basis.fisher.sketch_rows != protocol["sketch_rows"]
            or not torch.equal(basis.mean, covariance.mean)
        ):
            raise ValueError(
                f"{site!r} calibration-A row/width binding is invalid"
            )
        covariances[site] = covariance

    codecs: dict[str, dict[str, LinearActivationCodec]] = {}
    for variant in expected_variants:
        by_site_raw = raw_codecs[variant.variant_id]
        if (
            not isinstance(by_site_raw, Mapping)
            or tuple(by_site_raw) != boundaries
        ):
            raise ValueError("codec boundary mapping is invalid")
        by_site = {}
        for site in boundaries:
            state = by_site_raw[site]
            if not isinstance(state, Mapping):
                raise TypeError("codec state must be a mapping")
            codec = LinearActivationCodec.from_state_dict(state)
            if (
                codec.activation_name != site
                or codec.width != width
                or codec.method != variant.method
                or codec.alpha_floor != variant.alpha
                or codec.beta_floor != variant.beta
                or codec.activation_observations
                != calibration_a_tokens
            ):
                raise ValueError("codec does not match its variant/site")
            by_site[site] = codec
        codecs[variant.variant_id] = by_site
    recomputed_codecs = _build_codec_families(
        fisher=fisher,
        covariances=covariances,
        variants=expected_variants,
    )
    if not _state_values_close(
        {
            variant: {
                site: codec.state_dict()
                for site, codec in by_site.items()
            }
            for variant, by_site in codecs.items()
        },
        {
            variant: {
                site: codec.state_dict()
                for site, codec in by_site.items()
            }
            for variant, by_site in recomputed_codecs.items()
        },
    ):
        raise ValueError("serialized codecs do not match Fisher/covariance")

    schedule = _candidate_schedule(
        ranks=ranks,
        width=width,
        variants=expected_variants,
        codecs=codecs,
        sites=sites,
    )
    if protocol["candidate_schedule"] != tuple(
        candidate.metadata() for candidate in schedule
    ):
        raise ValueError("candidate schedule is invalid")
    calibration_b_expected = tuple(
        candidate.condition().metadata() for candidate in schedule
    )
    calibration_b_valid = _validate_ablation_metadata(
        calibration_b_raw["candidate_evaluation"],
        expected_conditions=calibration_b_expected,
        expected_example_ids=tuple(
            str(row["example_id"])
            for row in calibration_b_stream["examples"]
        ),
        expected_supervised_tokens_by_example=tuple(
            int(row["supervised_positions"])
            for row in calibration_b_stream["examples"]
        ),
        expected_supervised_tokens=sum(
            int(row["supervised_positions"])
            for row in calibration_b_stream["examples"]
        ),
    )
    calibration_b_result = _modal_result_from_metadata(
        calibration_b_valid
    )
    family_identities_raw = calibration_b_raw[
        "family_full_rank_identities"
    ]
    expected_family_ids = tuple(
        variant.variant_id for variant in expected_variants
    )
    if (
        not isinstance(family_identities_raw, Mapping)
        or tuple(family_identities_raw) != expected_family_ids
    ):
        raise ValueError(
            "calibration-B family full-rank identity mapping is invalid"
        )
    validated_family_identities: dict[str, dict[str, object]] = {}
    for variant in expected_variants:
        full_rank = _full_rank_candidate(
            schedule,
            variant_id=variant.variant_id,
        )
        validated_family_identities[variant.variant_id] = (
            _validate_full_rank_identity(
                family_identities_raw[variant.variant_id],
                expected_condition=full_rank.condition().metadata(),
                validation=calibration_b_valid,
                tolerance=identity_tolerance,
            )
        )

    # Selection is downstream of every family-specific identity prerequisite.
    recomputed_locked, recomputed_selection = _lock_candidate(
        schedule=schedule,
        calibration_b=calibration_b_result,
        nll_atol=selection_nll,
        top1_min=selection_top1,
    )
    if calibration_b_raw["selection"] != recomputed_selection:
        raise ValueError("calibration-B locked selection is invalid")
    native_full_rank = _full_rank_candidate(
        schedule,
        variant_id="native_fisher",
    )
    locked_family_full_rank = _full_rank_candidate(
        schedule,
        variant_id=recomputed_locked.variant.variant_id,
    )
    locked_identity = _validate_full_rank_identity(
        calibration_b_raw["locked_family_full_rank_identity"],
        expected_condition=locked_family_full_rank.condition().metadata(),
        validation=calibration_b_valid,
        tolerance=identity_tolerance,
    )
    native_identity = _validate_full_rank_identity(
        calibration_b_raw["native_full_rank_identity"],
        expected_condition=native_full_rank.condition().metadata(),
        validation=calibration_b_valid,
        tolerance=identity_tolerance,
    )
    if (
        locked_identity
        != validated_family_identities[
            recomputed_locked.variant.variant_id
        ]
        or native_identity
        != validated_family_identities["native_fisher"]
    ):
        raise ValueError(
            "calibration-B identity aliases do not match family entries"
        )

    if validation_raw["locked_candidate"] != recomputed_locked.metadata():
        raise ValueError("validation locked candidate binding is invalid")
    (
        validation_candidates,
        locked_family_identity_validation,
        native_identity_validation,
    ) = _validation_candidates(
        locked=recomputed_locked,
        locked_family_full_rank=locked_family_full_rank,
        native_full_rank=native_full_rank,
    )
    validation_valid = _validate_ablation_metadata(
        validation_raw["locked_evaluation"],
        expected_conditions=tuple(
            candidate.condition().metadata()
            for candidate in validation_candidates
        ),
        expected_example_ids=tuple(
            str(row["example_id"])
            for row in validation_stream["examples"]
        ),
        expected_supervised_tokens_by_example=tuple(
            int(row["supervised_positions"])
            for row in validation_stream["examples"]
        ),
        expected_supervised_tokens=sum(
            int(row["supervised_positions"])
            for row in validation_stream["examples"]
        ),
    )
    _validate_full_rank_identity(
        validation_raw["locked_family_full_rank_identity"],
        expected_condition=(
            locked_family_identity_validation.condition().metadata()
        ),
        validation=validation_valid,
        tolerance=identity_tolerance,
    )
    _validate_full_rank_identity(
        validation_raw["native_full_rank_identity"],
        expected_condition=native_identity_validation.condition().metadata(),
        validation=validation_valid,
        tolerance=identity_tolerance,
    )

    jacobian: CausalLagJacobianStatistics | None = None
    merged_factor: CausalWeightedJacobianResult | None = None
    if enabled:
        if (
            not isinstance(jacobian_raw["statistics"], Mapping)
            or not isinstance(jacobian_raw["merged_factor"], Mapping)
            or not isinstance(
                jacobian_raw["merged_factor_metadata"],
                Mapping,
            )
            or not isinstance(jacobian_raw["codec_binding"], Mapping)
            or jacobian_stream is None
        ):
            raise ValueError("enabled Jacobian payload is incomplete")
        jacobian = CausalLagJacobianStatistics.from_state_dict(
            jacobian_raw["statistics"]
        )
        merged_factor = CausalWeightedJacobianResult.from_state_dict(
            jacobian_raw["merged_factor"]
        )
        selected_codecs = codecs[
            recomputed_locked.variant.variant_id
        ]
        input_codec = selected_codecs[boundaries[0]]
        output_codec = selected_codecs[boundaries[-1]]
        bounded_modes = min(
            protocol["jacobian_modes"],
            recomputed_locked.retained_rank,
            width,
        )
        binding = {
            "locked_candidate_id": recomputed_locked.candidate_id,
            "variant_id": recomputed_locked.variant.variant_id,
            "input_activation": boundaries[0],
            "output_activation": boundaries[-1],
            "input_codec_sha256": _codec_state_sha256(
                raw_codecs[recomputed_locked.variant.variant_id][
                    boundaries[0]
                ]
            ),
            "output_codec_sha256": _codec_state_sha256(
                raw_codecs[recomputed_locked.variant.variant_id][
                    boundaries[-1]
                ]
            ),
            "coordinate_gauge_policy": (
                "energy_comparisons_only_within_locked_codec_family"
            ),
        }
        if jacobian_raw["codec_binding"] != binding:
            raise ValueError("Jacobian codec binding is invalid")
        if (
            jacobian.input_activation != boundaries[0]
            or jacobian.output_activation != boundaries[-1]
            or jacobian.input_modes != bounded_modes
            or jacobian.output_modes != bounded_modes
            or jacobian.max_lag != protocol["jacobian_max_lag"]
            or jacobian.sequences != jacobian_stream["sequences"]
            or jacobian.sequences
            != min(
                protocol["jacobian_max_sequences"],
                calibration_a_stream["sequences"],
            )
        ):
            raise ValueError("Jacobian endpoint/mode provenance is invalid")
        expected_factor = _factor_jacobian_pilot(
            statistics=jacobian,
            input_codec=input_codec,
            output_codec=output_codec,
            input_covariance=covariances[boundaries[0]],
            output_fisher=fisher.bases[
                boundaries[-1]
            ].fisher.approximate_matrix(),
            retained_rank=min(
                protocol["jacobian_factor_rank"],
                bounded_modes,
            ),
        )
        if not _state_values_close(
            merged_factor.state_dict(),
            expected_factor.state_dict(),
        ):
            raise ValueError(
                "merged weighted-Jacobian factor does not match inputs"
            )
        expected_factor_metadata = _factor_metadata(expected_factor)
        if not _state_values_close(
            jacobian_raw["merged_factor_metadata"],
            expected_factor_metadata,
        ):
            raise ValueError(
                "merged weighted-Jacobian lag metadata is invalid"
            )
    elif any(
        jacobian_raw[name] is not None
        for name in (
            "statistics",
            "merged_factor",
            "merged_factor_metadata",
            "codec_binding",
        )
    ):
        raise ValueError("disabled Jacobian payload must be empty")

    expected_report = _build_report(
        payload=payload,
        fisher=fisher,
        covariances=covariances,
        codecs=codecs,
        jacobian=jacobian,
        merged_factor=merged_factor,
        output=artifact_path,
        scientific_digest=scientific_digest,
    )
    report_path = artifact_path.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, Mapping)
        or _report_sha256(report) != raw["report_sha256"]
    ):
        raise ValueError("weighted-Jacobian JSON report digest mismatch")
    canonical_expected = json.loads(
        json.dumps(
            expected_report,
            sort_keys=True,
            allow_nan=False,
        )
    )
    if report != canonical_expected:
        raise ValueError(
            "weighted-Jacobian JSON report does not match payload"
        )
    return {
        "fisher": fisher,
        "covariances": covariances,
        "codecs": codecs,
        "jacobian": jacobian,
        "merged_factor": merged_factor,
        "selection": recomputed_selection,
        "validation": validation_valid,
        "metadata": {
            "schema": raw["schema"],
            "format_version": raw["format_version"],
            "model": model,
            "protocol": protocol,
            "scientific_payload_sha256": scientific_digest,
            "report_sha256": raw["report_sha256"],
        },
        "report": copy.deepcopy(dict(report)),
    }


def _parse_regularization_pair(value: str) -> tuple[float, float]:
    pieces = value.split(":")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(
            "regularization must use ALPHA:BETA"
        )
    try:
        alpha, beta = (float(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "regularization floors must be numbers"
        ) from error
    try:
        return _validated_regularization_pairs(((alpha, beta),))[0]
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit activation-aware Gemma codecs on calibration A, "
            "select rank/family and verify every full-width codec on "
            "calibration B, validate the locked rank plus locked-family and "
            "native identities, and optionally factor a bounded true-JVP "
            "edge pilot."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--check-paths-only",
        action="store_true",
        help="validate external Hugging Face paths without loading a model",
    )
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        default=DEFAULT_PROMPT_SPLITS,
    )
    parser.add_argument("--start-layer", type=int, default=DEFAULT_START_LAYER)
    parser.add_argument("--end-layer", type=int, default=DEFAULT_END_LAYER)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tokenization-batch-size", type=int, default=4)
    parser.add_argument(
        "--retained-ranks",
        "--ranks",
        dest="ranks",
        type=int,
        nargs="+",
        default=list(DEFAULT_RANKS),
    )
    parser.add_argument(
        "--generalized-regularization",
        type=_parse_regularization_pair,
        action="append",
        dest="regularization_pairs",
        help=(
            "predeclared absolute activation/Fisher floors ALPHA:BETA; "
            "repeat for multiple generalized codecs"
        ),
    )
    parser.add_argument("--sketch-rows", type=int)
    parser.add_argument(
        "--selection-nll-atol",
        type=float,
        default=DEFAULT_SELECTION_NLL_ATOL,
    )
    parser.add_argument(
        "--selection-top1-min",
        type=float,
        default=DEFAULT_SELECTION_TOP1_MIN,
    )
    parser.add_argument(
        "--full-rank-nll-atol",
        type=float,
        default=DEFAULT_FULL_RANK_NLL_ATOL,
    )
    parser.add_argument(
        "--jacobian-max-sequences",
        type=int,
        default=DEFAULT_JACOBIAN_MAX_SEQUENCES,
        help="set to zero to disable the true-forward-JVP pilot",
    )
    parser.add_argument(
        "--jacobian-modes",
        type=int,
        default=DEFAULT_JACOBIAN_MODES,
    )
    parser.add_argument(
        "--jacobian-max-lag",
        type=int,
        default=DEFAULT_JACOBIAN_MAX_LAG,
    )
    parser.add_argument(
        "--jacobian-factor-rank",
        type=int,
        default=DEFAULT_JACOBIAN_FACTOR_RANK,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.check_paths_only:
        paths = resolve_gemma3_huggingface_paths(arguments.cache_dir)
        print("Validated external Hugging Face paths; no model loaded:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return
    output = (
        arguments.output
        if arguments.output is not None
        else default_gemma3_weighted_jacobian_output(
            arguments.model,
            arguments.start_layer,
            arguments.end_layer,
        )
    )
    report = run_gemma3_weighted_jacobian(
        model_id=arguments.model,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        prompt_splits_path=arguments.prompt_splits,
        start_layer=arguments.start_layer,
        end_layer=arguments.end_layer,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        ranks=arguments.ranks,
        generalized_regularization_pairs=(
            DEFAULT_GENERALIZED_REGULARIZATION
            if arguments.regularization_pairs is None
            else arguments.regularization_pairs
        ),
        sketch_rows=arguments.sketch_rows,
        selection_nll_atol=arguments.selection_nll_atol,
        selection_top1_min=arguments.selection_top1_min,
        full_rank_nll_atol=arguments.full_rank_nll_atol,
        jacobian_max_sequences=arguments.jacobian_max_sequences,
        jacobian_modes=arguments.jacobian_modes,
        jacobian_max_lag=arguments.jacobian_max_lag,
        jacobian_factor_rank=arguments.jacobian_factor_rank,
        device_name=arguments.device,
        dtype=arguments.dtype,
        local_files_only=arguments.local_files_only,
        output=output,
    )
    selection = report["analysis"]["calibration_b"]["selection"]
    assert isinstance(selection, Mapping)
    locked = selection["locked_candidate"]
    assert isinstance(locked, Mapping)
    print(f"Wrote weighted-Jacobian analysis to {output}")
    print(
        "Locked calibration-B candidate: "
        f"{locked['candidate_id']} ({selection['reason']})"
    )
    print(f"Report: {output.with_suffix('.json')}")
    print("Reserved test prompts were never tokenized or evaluated.")
    print(
        "The bounded factor is a stationary reference diagnostic, "
        "not a runtime/compression acceptance result."
    )


if __name__ == "__main__":
    main()
