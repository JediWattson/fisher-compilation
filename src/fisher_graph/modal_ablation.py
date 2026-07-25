"""Full-width Fisher-modal projection ablations for causal language models.

This module deliberately evaluates projections rather than fitting an
executor.  At each selected activation site, a complete width-by-width Fisher
basis is required.  Keeping the leading ``k`` modes therefore has an
unambiguous complement: the lowest ``width - k`` modes are removed around the
pooled calibration mean.

The evaluator runs one baseline forward and one forward per ablation condition
for every :class:`~fisher_graph.compiler.calibration.CalibrationBatch`.  Losses
and top-1 agreement are still reduced per example, so conditions remain paired
without falling back to one forward per sequence.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .instrumentation import InstrumentedModel, validate_instrumented_model
from .modes import FisherModeBasis
from .streaming_analysis import StreamingActivationFisherBasis
from .streaming_block_validation import _validate_orthonormal_columns


FullWidthModalBasis = FisherModeBasis | StreamingActivationFisherBasis


@dataclass(frozen=True, slots=True)
class _ResolvedFullWidthBasis:
    activation_name: str
    width: int
    mean: Tensor
    vectors: Tensor


def _resolve_full_width_basis(
    basis: FullWidthModalBasis,
) -> _ResolvedFullWidthBasis:
    if isinstance(basis, StreamingActivationFisherBasis):
        activation_name = basis.activation_name
        width = basis.fisher.width
        mean = basis.mean
        eigenvalues = basis.fisher.eigenvalues
        vectors = basis.fisher.vectors
        modes = basis.fisher.modes
    elif isinstance(basis, FisherModeBasis):
        activation_name = basis.activation_name
        width = basis.width
        mean = basis.mean
        eigenvalues = basis.eigenvalues
        vectors = basis.vectors
        modes = basis.vectors.shape[1]
    else:
        raise TypeError(
            "basis must be a FisherModeBasis or "
            "StreamingActivationFisherBasis"
        )
    if modes != width or vectors.shape != (width, width):
        raise ValueError(
            f"{activation_name!r} must provide a complete "
            f"{width}x{width} Fisher basis"
        )
    if (
        mean.shape != (width,)
        or mean.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(mean).all()
    ):
        raise ValueError(
            f"{activation_name!r} pooled mean must be a finite "
            f"length-{width} float vector"
        )
    if (
        eigenvalues.shape != (width,)
        or not eigenvalues.is_floating_point()
        or not torch.isfinite(eigenvalues).all()
        or (eigenvalues < 0).any()
    ):
        raise ValueError(
            f"{activation_name!r} full Fisher eigenvalues are invalid"
        )
    if width > 1:
        scale = max(float(eigenvalues.abs().max().item()), 1.0)
        tolerance = (
            64
            * torch.finfo(eigenvalues.dtype).eps
            * scale
        )
        if (
            eigenvalues[1:] - eigenvalues[:-1] > tolerance
        ).any():
            raise ValueError(
                f"{activation_name!r} Fisher modes must be ordered "
                "from highest to lowest eigenvalue"
            )
    _validate_orthonormal_columns(
        vectors,
        label=f"{activation_name!r} full Fisher basis",
    )
    return _ResolvedFullWidthBasis(
        activation_name=activation_name,
        width=width,
        mean=mean.detach(),
        vectors=vectors.detach(),
    )


def _project_valid_positions(
    activation: Tensor,
    *,
    basis: _ResolvedFullWidthBasis,
    retained_modes: int,
    valid_positions: Tensor,
) -> Tensor:
    if not isinstance(activation, Tensor) or not activation.is_floating_point():
        raise TypeError("modal projection requires a floating activation Tensor")
    if activation.ndim != 3 or activation.shape[-1] != basis.width:
        raise ValueError(
            "activation must have shape "
            f"[batch, sequence, {basis.width}]"
        )
    if (
        not isinstance(valid_positions, Tensor)
        or valid_positions.dtype is not torch.bool
        or valid_positions.shape != activation.shape[:2]
    ):
        raise ValueError(
            "valid_positions must be a boolean [batch, sequence] mask "
            "matching the activation"
        )
    if type(retained_modes) is not int or not (
        0 <= retained_modes <= basis.width
    ):
        raise ValueError(
            f"retained_modes must be between 0 and {basis.width}"
        )

    compute_dtype = (
        torch.float32
        if activation.dtype in (torch.float16, torch.bfloat16)
        else activation.dtype
    )
    compute_activation = activation.to(dtype=compute_dtype)
    mean = basis.mean.to(
        device=activation.device,
        dtype=compute_dtype,
    )
    vectors = basis.vectors[:, :retained_modes].to(
        device=activation.device,
        dtype=compute_dtype,
    )
    centered = compute_activation - mean
    # Do not special-case retained_modes == width.  The full-rank condition is
    # an identity-through-the-projection-path control, including its numerical
    # dtype and backend behavior.
    projected = (
        (centered @ vectors) @ vectors.transpose(0, 1) + mean
    ).to(dtype=activation.dtype)
    mask = valid_positions.to(device=activation.device).unsqueeze(-1)
    return torch.where(mask, projected, activation)


@dataclass(frozen=True, slots=True)
class PooledModalProjection:
    """Keep leading Fisher modes around a pooled mean at valid positions."""

    basis: FullWidthModalBasis
    retained_modes: int
    valid_positions: Tensor
    _resolved_basis: _ResolvedFullWidthBasis = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        resolved = _resolve_full_width_basis(self.basis)
        if type(self.retained_modes) is not int or not (
            0 <= self.retained_modes <= resolved.width
        ):
            raise ValueError(
                f"retained_modes must be between 0 and {resolved.width}"
            )
        if (
            not isinstance(self.valid_positions, Tensor)
            or self.valid_positions.dtype is not torch.bool
            or self.valid_positions.ndim != 2
        ):
            raise ValueError(
                "valid_positions must be a boolean [batch, sequence] mask"
            )
        object.__setattr__(self, "_resolved_basis", resolved)

    def __call__(self, activation: Tensor) -> Tensor:
        return _project_valid_positions(
            activation,
            basis=self._resolved_basis,
            retained_modes=self.retained_modes,
            valid_positions=self.valid_positions,
        )


@dataclass(frozen=True, slots=True)
class ModalAblationCondition:
    """One named singleton or joint collection of retained-mode gates."""

    name: str
    retained_modes: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("condition name must be a nonempty string")
        if not isinstance(self.retained_modes, Mapping) or not (
            self.retained_modes
        ):
            raise ValueError("retained_modes must be a nonempty mapping")
        normalized: dict[str, int] = {}
        for site, rank in self.retained_modes.items():
            if not isinstance(site, str) or not site:
                raise ValueError(
                    "retained_modes keys must be nonempty site names"
                )
            if type(rank) is not int or rank < 0:
                raise ValueError(
                    "retained_modes values must be nonnegative integers"
                )
            normalized[site] = rank
        object.__setattr__(self, "retained_modes", normalized)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "retained_modes": dict(self.retained_modes),
        }


def build_modal_ablation_conditions(
    *,
    sites: Collection[str],
    ranks: Iterable[int],
    include_joint: bool = True,
    include_singletons: bool = False,
) -> tuple[ModalAblationCondition, ...]:
    """Build common-rank joint and optional singleton ablation curves."""

    if isinstance(sites, (str, bytes)):
        raise TypeError("sites must be a collection of activation names")
    raw_sites = tuple(sites)
    normalized_sites = tuple(dict.fromkeys(raw_sites))
    if not normalized_sites or any(
        not isinstance(site, str) or not site
        for site in normalized_sites
    ):
        raise ValueError("sites must contain nonempty strings")
    if len(normalized_sites) != len(raw_sites):
        raise ValueError("sites cannot contain duplicates")
    if isinstance(ranks, (str, bytes)):
        raise TypeError("ranks must be an iterable of positive integers")
    try:
        rank_values = tuple(ranks)
    except TypeError as error:
        raise TypeError(
            "ranks must be an iterable of positive integers"
        ) from error
    if not rank_values or any(
        type(rank) is not int or rank < 0 for rank in rank_values
    ):
        raise ValueError("ranks must contain nonnegative integers")
    normalized_ranks = tuple(dict.fromkeys(rank_values))
    if not include_joint and not include_singletons:
        raise ValueError(
            "at least one of include_joint or include_singletons is required"
        )

    conditions: list[ModalAblationCondition] = []
    for rank in normalized_ranks:
        if include_joint:
            conditions.append(
                ModalAblationCondition(
                    name=f"joint.rank_{rank}",
                    retained_modes={
                        site: rank for site in normalized_sites
                    },
                )
            )
        if include_singletons:
            conditions.extend(
                ModalAblationCondition(
                    name=f"singleton.{site}.rank_{rank}",
                    retained_modes={site: rank},
                )
                for site in normalized_sites
            )
    return tuple(conditions)


@dataclass(frozen=True, slots=True)
class ModalAblationExample:
    """Paired baseline/ablated metrics for one causal-LM example."""

    example_id: str
    supervised_tokens: int
    baseline_summed_nll: float
    baseline_nll_per_token: float
    ablated_summed_nll: float
    ablated_nll_per_token: float
    delta_summed_nll: float
    delta_nll_per_token: float
    top1_matches: int
    top1_agreement_to_baseline: float

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("example_id must be a nonempty string")
        if type(self.supervised_tokens) is not int or (
            self.supervised_tokens <= 0
        ):
            raise ValueError("supervised_tokens must be positive")
        if (
            type(self.top1_matches) is not int
            or not 0 <= self.top1_matches <= self.supervised_tokens
        ):
            raise ValueError("top1_matches is outside the supervised count")
        for label, value in (
            ("baseline_summed_nll", self.baseline_summed_nll),
            ("baseline_nll_per_token", self.baseline_nll_per_token),
            ("ablated_summed_nll", self.ablated_summed_nll),
            ("ablated_nll_per_token", self.ablated_nll_per_token),
            ("delta_summed_nll", self.delta_summed_nll),
            ("delta_nll_per_token", self.delta_nll_per_token),
            (
                "top1_agreement_to_baseline",
                self.top1_agreement_to_baseline,
            ),
        ):
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{label} must be a finite float")
        if (
            self.baseline_summed_nll < 0
            or self.baseline_nll_per_token < 0
            or self.ablated_summed_nll < 0
            or self.ablated_nll_per_token < 0
        ):
            raise ValueError("NLL values must be nonnegative")
        if not 0.0 <= self.top1_agreement_to_baseline <= 1.0:
            raise ValueError("top1 agreement must be in [0, 1]")

    def metadata(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "supervised_tokens": self.supervised_tokens,
            "baseline_summed_nll": self.baseline_summed_nll,
            "baseline_nll_per_token": self.baseline_nll_per_token,
            "ablated_summed_nll": self.ablated_summed_nll,
            "ablated_nll_per_token": self.ablated_nll_per_token,
            "delta_summed_nll": self.delta_summed_nll,
            "delta_nll_per_token": self.delta_nll_per_token,
            "top1_matches": self.top1_matches,
            "top1_agreement_to_baseline": (
                self.top1_agreement_to_baseline
            ),
        }


@dataclass(frozen=True, slots=True)
class ModalAblationAggregate:
    """Token- and sequence-level aggregate for one evaluation condition."""

    sequences: int
    supervised_tokens: int
    summed_nll: float
    mean_sequence_summed_nll: float
    nll_per_token: float
    top1_matches: int
    top1_agreement_to_baseline: float
    delta_summed_nll: float
    delta_mean_sequence_summed_nll: float
    delta_nll_per_token: float

    def __post_init__(self) -> None:
        if type(self.sequences) is not int or self.sequences <= 0:
            raise ValueError("sequences must be positive")
        if type(self.supervised_tokens) is not int or (
            self.supervised_tokens < self.sequences
        ):
            raise ValueError(
                "supervised_tokens must include at least one per sequence"
            )
        if (
            type(self.top1_matches) is not int
            or not 0 <= self.top1_matches <= self.supervised_tokens
        ):
            raise ValueError("top1_matches is outside the supervised count")
        for label, value in (
            ("summed_nll", self.summed_nll),
            ("mean_sequence_summed_nll", self.mean_sequence_summed_nll),
            ("nll_per_token", self.nll_per_token),
            (
                "top1_agreement_to_baseline",
                self.top1_agreement_to_baseline,
            ),
            ("delta_summed_nll", self.delta_summed_nll),
            (
                "delta_mean_sequence_summed_nll",
                self.delta_mean_sequence_summed_nll,
            ),
            ("delta_nll_per_token", self.delta_nll_per_token),
        ):
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{label} must be a finite float")
        if (
            self.summed_nll < 0
            or self.mean_sequence_summed_nll < 0
            or self.nll_per_token < 0
        ):
            raise ValueError("aggregate NLL values must be nonnegative")
        if not 0.0 <= self.top1_agreement_to_baseline <= 1.0:
            raise ValueError("top1 agreement must be in [0, 1]")

    def metadata(self) -> dict[str, object]:
        return {
            "sequences": self.sequences,
            "supervised_tokens": self.supervised_tokens,
            "summed_nll": self.summed_nll,
            "mean_sequence_summed_nll": (
                self.mean_sequence_summed_nll
            ),
            "nll_per_token": self.nll_per_token,
            "top1_matches": self.top1_matches,
            "top1_agreement_to_baseline": (
                self.top1_agreement_to_baseline
            ),
            "delta_summed_nll": self.delta_summed_nll,
            "delta_mean_sequence_summed_nll": (
                self.delta_mean_sequence_summed_nll
            ),
            "delta_nll_per_token": self.delta_nll_per_token,
        }


@dataclass(frozen=True, slots=True)
class ModalAblationConditionResult:
    """One ablation condition with its paired ledger and aggregate."""

    condition: ModalAblationCondition
    aggregate: ModalAblationAggregate
    examples: tuple[ModalAblationExample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.condition, ModalAblationCondition):
            raise TypeError("condition must be a ModalAblationCondition")
        if not isinstance(self.aggregate, ModalAblationAggregate):
            raise TypeError("aggregate must be a ModalAblationAggregate")
        if (
            not isinstance(self.examples, tuple)
            or len(self.examples) != self.aggregate.sequences
            or any(
                not isinstance(example, ModalAblationExample)
                for example in self.examples
            )
        ):
            raise ValueError(
                "examples must contain one paired row per sequence"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "condition": self.condition.metadata(),
            "aggregate": self.aggregate.metadata(),
            "examples": [example.metadata() for example in self.examples],
        }


@dataclass(frozen=True, slots=True)
class ModalAblationResult:
    """Baseline and paired singleton/joint modal-ablation results."""

    baseline: ModalAblationAggregate
    conditions: tuple[ModalAblationConditionResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, ModalAblationAggregate):
            raise TypeError("baseline must be a ModalAblationAggregate")
        if not isinstance(self.conditions, tuple) or not self.conditions:
            raise ValueError("conditions must be a nonempty tuple")
        if any(
            not isinstance(result, ModalAblationConditionResult)
            for result in self.conditions
        ):
            raise TypeError(
                "conditions must contain ModalAblationConditionResult values"
            )
        names = tuple(result.condition.name for result in self.conditions)
        if len(set(names)) != len(names):
            raise ValueError("condition result names must be unique")
        for result in self.conditions:
            aggregate = result.aggregate
            if (
                aggregate.sequences != self.baseline.sequences
                or aggregate.supervised_tokens
                != self.baseline.supervised_tokens
            ):
                raise ValueError(
                    "all conditions must share baseline row accounting"
                )

    def condition(self, name: str) -> ModalAblationConditionResult:
        for result in self.conditions:
            if result.condition.name == name:
                return result
        raise KeyError(f"unknown modal ablation condition: {name!r}")

    def metadata(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.metadata(),
            "conditions": [
                condition.metadata() for condition in self.conditions
            ],
        }


@dataclass(frozen=True, slots=True)
class _BatchScores:
    summed_nll: Tensor
    supervised_tokens: Tensor
    predictions: Tensor
    supervised_mask: Tensor


def _causal_lm_batch_scores(
    logits: Tensor,
    batch: CalibrationBatch,
    *,
    objective: CausalLanguageModelNLL,
) -> _BatchScores:
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError(
            "causal language-model logits must have shape "
            "[batch, sequence, vocabulary]"
        )
    if batch.targets.shape != logits.shape[:2]:
        raise ValueError(
            "causal language-model targets must match logits positions"
        )
    targets = batch.targets.to(device=logits.device)
    valid_positions = batch.valid_positions.to(device=logits.device)
    supervised = targets != objective.ignore_index
    if (supervised & ~valid_positions).any():
        raise ValueError(
            "supervised targets must be at valid calibration positions"
        )
    supervised_tokens = supervised.sum(dim=1)
    if (supervised_tokens <= 0).any():
        raise ValueError(
            "every calibration example must have a supervised target"
        )
    compute_logits = (
        logits.float()
        if logits.dtype in (torch.float16, torch.bfloat16)
        else logits
    )
    token_losses = F.cross_entropy(
        compute_logits.reshape(-1, compute_logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=objective.ignore_index,
        reduction="none",
    ).reshape(targets.shape)
    return _BatchScores(
        summed_nll=token_losses.sum(dim=1).detach().to(
            device="cpu",
            dtype=torch.float64,
        ),
        supervised_tokens=supervised_tokens.detach().to(
            device="cpu",
            dtype=torch.int64,
        ),
        predictions=compute_logits.argmax(dim=-1).detach().to(
            device="cpu",
            dtype=torch.int64,
        ),
        supervised_mask=supervised.detach().to(device="cpu"),
    )


def _condition_interventions(
    *,
    condition: ModalAblationCondition,
    bases: Mapping[str, _ResolvedFullWidthBasis],
    valid_positions: Tensor,
) -> dict[str, object]:
    interventions: dict[str, object] = {}
    for site, retained_modes in condition.retained_modes.items():
        resolved = bases[site]

        def project(
            activation: Tensor,
            *,
            basis: _ResolvedFullWidthBasis = resolved,
            rank: int = retained_modes,
            valid: Tensor = valid_positions,
        ) -> Tensor:
            return _project_valid_positions(
                activation,
                basis=basis,
                retained_modes=rank,
                valid_positions=valid,
            )

        interventions[site] = project
    return interventions


def _example_ids(
    batch: CalibrationBatch,
    *,
    sequence_offset: int,
) -> tuple[str, ...]:
    if batch.example_ids is not None:
        return batch.example_ids
    return tuple(
        f"sequence.{index:06d}"
        for index in range(
            sequence_offset,
            sequence_offset + batch.batch_size,
        )
    )


def _aggregate(
    *,
    sequences: int,
    supervised_tokens: int,
    summed_nll: float,
    baseline_summed_nll: float,
    top1_matches: int,
) -> ModalAblationAggregate:
    return ModalAblationAggregate(
        sequences=sequences,
        supervised_tokens=supervised_tokens,
        summed_nll=summed_nll,
        mean_sequence_summed_nll=summed_nll / sequences,
        nll_per_token=summed_nll / supervised_tokens,
        top1_matches=top1_matches,
        top1_agreement_to_baseline=top1_matches / supervised_tokens,
        delta_summed_nll=summed_nll - baseline_summed_nll,
        delta_mean_sequence_summed_nll=(
            summed_nll - baseline_summed_nll
        )
        / sequences,
        delta_nll_per_token=(
            summed_nll - baseline_summed_nll
        )
        / supervised_tokens,
    )


def evaluate_causal_lm_modal_ablation(
    model: InstrumentedModel,
    calibration_batches: Iterable[CalibrationBatch],
    *,
    bases: Mapping[str, FullWidthModalBasis],
    conditions: Collection[ModalAblationCondition],
    objective: CausalLanguageModelNLL | None = None,
) -> ModalAblationResult:
    """Evaluate paired full-width-to-low-rank activation projections.

    Source-model weights must already be frozen.  The no-intervention baseline
    and every condition are evaluated on each input batch under inference
    mode.  A full-rank condition is not optimized away: callers can use it as
    an explicit numerical identity control before interpreting lower ranks.
    """

    validate_instrumented_model(model)
    module = model.module
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError(
            "modal ablation requires all source-model weights to be frozen"
        )
    if not isinstance(bases, Mapping) or not bases:
        raise ValueError("bases must be a nonempty mapping")
    normalized_conditions = tuple(conditions)
    if not normalized_conditions or any(
        not isinstance(condition, ModalAblationCondition)
        for condition in normalized_conditions
    ):
        raise ValueError(
            "conditions must contain ModalAblationCondition values"
        )
    condition_names = tuple(
        condition.name for condition in normalized_conditions
    )
    if len(set(condition_names)) != len(condition_names):
        raise ValueError("condition names must be unique")
    resolved_objective = (
        CausalLanguageModelNLL() if objective is None else objective
    )
    if not isinstance(resolved_objective, CausalLanguageModelNLL):
        raise TypeError("objective must be a CausalLanguageModelNLL")

    sites_by_id = {site.id: site for site in model.activation_sites}
    known_sites = set(sites_by_id)
    used_sites = {
        site
        for condition in normalized_conditions
        for site in condition.retained_modes
    }
    if not used_sites.issubset(known_sites):
        raise KeyError(
            f"unknown activation sites: {sorted(used_sites - known_sites)}"
        )
    if not used_sites.issubset(bases):
        raise KeyError(
            f"missing Fisher bases: {sorted(used_sites - set(bases))}"
        )
    resolved_bases = {
        site: _resolve_full_width_basis(bases[site])
        for site in used_sites
    }
    for site, basis in resolved_bases.items():
        site_metadata = sites_by_id[site]
        if (
            not site_metadata.intervenable
            or not site_metadata.modal_eligible
            or site_metadata.width != basis.width
        ):
            raise ValueError(
                f"activation site {site!r} is not an intervenable "
                f"width-{basis.width} modal boundary"
            )
        if basis.activation_name != site:
            raise ValueError(
                f"basis key {site!r} does not match "
                f"activation name {basis.activation_name!r}"
            )
    for condition in normalized_conditions:
        for site, retained_modes in condition.retained_modes.items():
            width = resolved_bases[site].width
            if retained_modes > width:
                raise ValueError(
                    f"condition {condition.name!r} retains "
                    f"{retained_modes} modes at width-{width} site {site!r}"
                )

    baseline_nll_total = 0.0
    supervised_total = 0
    sequence_count = 0
    seen_example_ids: set[str] = set()
    condition_nll_totals = {
        condition.name: 0.0 for condition in normalized_conditions
    }
    condition_top1_matches = {
        condition.name: 0 for condition in normalized_conditions
    }
    condition_examples: dict[str, list[ModalAblationExample]] = {
        condition.name: [] for condition in normalized_conditions
    }

    was_training = module.training
    module.eval()
    try:
        with torch.inference_mode():
            for batch in calibration_batches:
                if not isinstance(batch, CalibrationBatch):
                    raise TypeError(
                        "calibration_batches must contain CalibrationBatch "
                        "values"
                    )
                example_ids = _example_ids(
                    batch,
                    sequence_offset=sequence_count,
                )
                duplicates = seen_example_ids.intersection(example_ids)
                if duplicates:
                    raise ValueError(
                        "example IDs must be unique across the replay: "
                        f"{sorted(duplicates)}"
                    )
                seen_example_ids.update(example_ids)

                baseline_run = model.forward(batch.model_inputs)
                baseline = _causal_lm_batch_scores(
                    baseline_run.logits,
                    batch,
                    objective=resolved_objective,
                )
                del baseline_run
                batch_baseline_nll = float(baseline.summed_nll.sum().item())
                batch_tokens = int(
                    baseline.supervised_tokens.sum().item()
                )
                baseline_nll_total += batch_baseline_nll
                supervised_total += batch_tokens
                sequence_count += batch.batch_size

                for condition in normalized_conditions:
                    interventions = _condition_interventions(
                        condition=condition,
                        bases=resolved_bases,
                        valid_positions=batch.valid_positions,
                    )
                    ablated_run = model.forward(
                        batch.model_inputs,
                        interventions=interventions,  # type: ignore[arg-type]
                    )
                    ablated = _causal_lm_batch_scores(
                        ablated_run.logits,
                        batch,
                        objective=resolved_objective,
                    )
                    del ablated_run
                    if not torch.equal(
                        ablated.supervised_tokens,
                        baseline.supervised_tokens,
                    ) or not torch.equal(
                        ablated.supervised_mask,
                        baseline.supervised_mask,
                    ):
                        raise ValueError(
                            "ablation changed supervised-token accounting"
                        )
                    matches = (
                        (ablated.predictions == baseline.predictions)
                        & baseline.supervised_mask
                    ).sum(dim=1)
                    condition_nll_totals[condition.name] += float(
                        ablated.summed_nll.sum().item()
                    )
                    condition_top1_matches[condition.name] += int(
                        matches.sum().item()
                    )
                    for index, example_id in enumerate(example_ids):
                        tokens = int(
                            baseline.supervised_tokens[index].item()
                        )
                        baseline_nll = float(
                            baseline.summed_nll[index].item()
                        )
                        ablated_nll = float(
                            ablated.summed_nll[index].item()
                        )
                        top1_matches = int(matches[index].item())
                        condition_examples[condition.name].append(
                            ModalAblationExample(
                                example_id=example_id,
                                supervised_tokens=tokens,
                                baseline_summed_nll=baseline_nll,
                                baseline_nll_per_token=(
                                    baseline_nll / tokens
                                ),
                                ablated_summed_nll=ablated_nll,
                                ablated_nll_per_token=(
                                    ablated_nll / tokens
                                ),
                                delta_summed_nll=(
                                    ablated_nll - baseline_nll
                                ),
                                delta_nll_per_token=(
                                    ablated_nll - baseline_nll
                                )
                                / tokens,
                                top1_matches=top1_matches,
                                top1_agreement_to_baseline=(
                                    top1_matches / tokens
                                ),
                            )
                        )
    finally:
        module.train(was_training)

    if sequence_count == 0:
        raise ValueError("calibration batch stream cannot be empty")
    baseline_aggregate = _aggregate(
        sequences=sequence_count,
        supervised_tokens=supervised_total,
        summed_nll=baseline_nll_total,
        baseline_summed_nll=baseline_nll_total,
        top1_matches=supervised_total,
    )
    condition_results = tuple(
        ModalAblationConditionResult(
            condition=condition,
            aggregate=_aggregate(
                sequences=sequence_count,
                supervised_tokens=supervised_total,
                summed_nll=condition_nll_totals[condition.name],
                baseline_summed_nll=baseline_nll_total,
                top1_matches=condition_top1_matches[condition.name],
            ),
            examples=tuple(condition_examples[condition.name]),
        )
        for condition in normalized_conditions
    )
    return ModalAblationResult(
        baseline=baseline_aggregate,
        conditions=condition_results,
    )


__all__ = [
    "FullWidthModalBasis",
    "ModalAblationAggregate",
    "ModalAblationCondition",
    "ModalAblationConditionResult",
    "ModalAblationExample",
    "ModalAblationResult",
    "PooledModalProjection",
    "build_modal_ablation_conditions",
    "evaluate_causal_lm_modal_ablation",
]
