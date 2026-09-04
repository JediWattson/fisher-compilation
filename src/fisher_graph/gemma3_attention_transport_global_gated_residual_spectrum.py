"""Global-gated allocation of invocation-local Gemma V/O residuals.

This rung preserves the residual proposal mechanism from
``gemma3_attention_transport_all_layer_residual_spectrum`` and changes only
the greedy allocation objective.  On the allocation-validation panel, define

    L_S(B) = sum_{i in S} sum_{n in D_i} ell_F(B; i, n)
             -------------------------------------------------
             sum_{i in S} sum_{n in D_i} ell_F(0; i, n).

While ``L_global(B) > tau`` only the three global-attention invocations are
eligible and the selected marginal maximizes the reduction in ``L_global``.
Once the gate is closed, the *next* round uses the all-layer reduction over
all 18 invocations.  The budget remains a maximum: if no eligible global
candidate exists above the gate, allocation stops rather than spending a
component at a local site.

Every round retains both global and all-layer normalized Fisher-mass gains for
all 18 candidates.  Local candidates have exact zero global gain because an
invocation-local residual is bit-for-bit inactive off site.  Only the selected
site's marginal is recomputed after acceptance; all off-site marginals remain
exactly cacheable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from .gemma3_attention_transport_compiler import AttentionTransportBatch
from .gemma3_attention_transport_global_rank1 import (
    GLOBAL_ATTENTION_INVOCATION_IDS,
)
from .gemma3_attention_transport_global_site_rank1 import (
    GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
)
from .gemma3_attention_transport_hierarchical_edge import _mutation_ordinal
from .gemma3_attention_transport_sparse_v2 import (
    SparseAttentionTransportOracleEvaluation,
    evaluate_sparse_attention_transport_bank,
)
from .gemma3_attention_transport_all_layer_residual_spectrum import (
    AllLayerFirstResidualSpectrum,
    AllLayerResidualSpectrumConfig,
    AllLayerResidualSpectrumGeneratorBank,
    ResidualSpectrumProgressCallback,
    ResidualSpectrumSiteMarginal,
    _all_layer_capture_groups,
    _apply_cached_marginal,
    _notify_progress,
    all_layer_residual_spectrum_generator_bank_from_parent,
    compute_all_layer_first_residual_spectrum,
    compute_site_residual_spectrum_marginal,
    residual_allocation_profile_sha256,
)


GLOBAL_GATED_RESIDUAL_OBJECTIVE_SEMANTICS = (
    "ratio_of_summed_fisher_mass_global_until_0_10_then_all_layer_v1"
)
GLOBAL_GATED_RESIDUAL_GREEDY_SEMANTICS = (
    "budget_two_global_only_above_gate_cached_off_site_numeric_tiebreak_v1"
)
GLOBAL_GATED_RESIDUAL_LEDGER_SEMANTICS = (
    "all_18_global_and_all_layer_normalized_marginals_v1"
)
DEFAULT_GLOBAL_RELATIVE_FISHER_THRESHOLD = 0.10

GLOBAL_OBJECTIVE_MODE = "global_attention"
ALL_LAYER_OBJECTIVE_MODE = "all_layers"
_OBJECTIVE_MODES = frozenset((GLOBAL_OBJECTIVE_MODE, ALL_LAYER_OBJECTIVE_MODE))
_GLOBAL_INVOCATION_SET = frozenset(GLOBAL_ATTENTION_INVOCATION_IDS)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-11, abs_tol=1.0e-12)


def _global_evaluation(
    bank: AllLayerResidualSpectrumGeneratorBank,
    captures: Sequence[AttentionTransportBatch],
    *,
    config: AllLayerResidualSpectrumConfig,
) -> SparseAttentionTransportOracleEvaluation:
    values = tuple(
        capture
        for capture in captures
        if capture.invocation_id in _GLOBAL_INVOCATION_SET
    )
    if {capture.invocation_id for capture in values} != _GLOBAL_INVOCATION_SET:
        raise ValueError("global-gated evaluation requires all three global sites")
    return evaluate_sparse_attention_transport_bank(
        bank,
        values,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.coefficient_ridge,
    )


@dataclass(frozen=True, slots=True)
class GlobalGatedResidualSpectrumAllocationRound:
    """An independently checkable all-18 cache snapshot before one decision."""

    allocation_index: int
    current_bank_sha256: str
    objective_mode: str
    global_relative_fisher_threshold: float
    global_relative_fisher_loss_before: float
    all_layer_relative_fisher_loss_before: float
    global_evaluation_row_count: int
    global_evaluation_zero_predictor_fisher_loss: float
    all_layer_evaluation_row_count: int
    all_layer_evaluation_zero_predictor_fisher_loss: float
    marginals: tuple[ResidualSpectrumSiteMarginal, ...]
    global_normalized_fisher_mass_gains: tuple[tuple[str, float], ...]
    all_layer_normalized_fisher_mass_gains: tuple[tuple[str, float], ...]
    selected_invocation_id: str | None
    minimum_relative_fisher_gain: float
    tie_tolerance: float

    def __post_init__(self) -> None:
        if (
            type(self.allocation_index) is not int
            or self.allocation_index < 0
            or not _is_sha256(self.current_bank_sha256)
            or self.objective_mode not in _OBJECTIVE_MODES
            or not _finite_number(self.global_relative_fisher_threshold)
            or not 0.0 < float(self.global_relative_fisher_threshold) < 1.0
            or not _finite_number(self.global_relative_fisher_loss_before)
            or float(self.global_relative_fisher_loss_before) < 0.0
            or not _finite_number(self.all_layer_relative_fisher_loss_before)
            or float(self.all_layer_relative_fisher_loss_before) < 0.0
            or type(self.global_evaluation_row_count) is not int
            or self.global_evaluation_row_count <= 0
            or type(self.all_layer_evaluation_row_count) is not int
            or self.all_layer_evaluation_row_count <= 0
            or self.global_evaluation_row_count
            >= self.all_layer_evaluation_row_count
            or not _finite_number(
                self.global_evaluation_zero_predictor_fisher_loss
            )
            or float(self.global_evaluation_zero_predictor_fisher_loss) <= 0.0
            or not _finite_number(
                self.all_layer_evaluation_zero_predictor_fisher_loss
            )
            or float(self.all_layer_evaluation_zero_predictor_fisher_loss) <= 0.0
            or len(self.marginals) != 18
            or any(
                not isinstance(value, ResidualSpectrumSiteMarginal)
                for value in self.marginals
            )
            or len({value.invocation_id for value in self.marginals}) != 18
            or tuple(value.invocation_id for value in self.marginals)
            != tuple(
                sorted(
                    (value.invocation_id for value in self.marginals),
                    key=_mutation_ordinal,
                )
            )
            or tuple(key for key, _gain in self.global_normalized_fisher_mass_gains)
            != tuple(value.invocation_id for value in self.marginals)
            or tuple(
                key for key, _gain in self.all_layer_normalized_fisher_mass_gains
            )
            != tuple(value.invocation_id for value in self.marginals)
            or any(
                not _finite_number(gain)
                for _key, gain in (
                    self.global_normalized_fisher_mass_gains
                    + self.all_layer_normalized_fisher_mass_gains
                )
            )
            or (
                self.selected_invocation_id is not None
                and self.selected_invocation_id
                not in {value.invocation_id for value in self.marginals}
            )
            or not _finite_number(self.minimum_relative_fisher_gain)
            or float(self.minimum_relative_fisher_gain) <= 0.0
            or not _finite_number(self.tie_tolerance)
            or float(self.tie_tolerance) < 0.0
            or any(
                float(value.minimum_relative_fisher_gain)
                != float(self.minimum_relative_fisher_gain)
                for value in self.marginals
            )
        ):
            raise ValueError("global-gated allocation round is invalid")

        expected_mode = (
            GLOBAL_OBJECTIVE_MODE
            if float(self.global_relative_fisher_loss_before)
            > float(self.global_relative_fisher_threshold)
            else ALL_LAYER_OBJECTIVE_MODE
        )
        if self.objective_mode != expected_mode:
            raise ValueError("global-gated objective mode does not match the gate")

        global_gains = dict(self.global_normalized_fisher_mass_gains)
        all_layer_gains = dict(self.all_layer_normalized_fisher_mass_gains)
        global_marginals = tuple(
            value
            for value in self.marginals
            if value.invocation_id in _GLOBAL_INVOCATION_SET
        )
        all_rows = sum(
            value.evaluation_current_oracle.output_metrics.row_count
            for value in self.marginals
        )
        global_rows = sum(
            value.evaluation_current_oracle.output_metrics.row_count
            for value in global_marginals
        )

        def weighted_current_fisher(
            values: tuple[ResidualSpectrumSiteMarginal, ...],
        ) -> float:
            return sum(
                value.evaluation_current_oracle.output_metrics.row_count
                * value.evaluation_current_oracle.output_metrics.fisher_loss
                for value in values
            )

        def weighted_current_zero(
            values: tuple[ResidualSpectrumSiteMarginal, ...],
        ) -> float:
            return sum(
                value.evaluation_current_oracle.output_metrics.row_count
                * value.evaluation_current_oracle.zero_predictor_fisher_loss
                for value in values
            )

        if (
            len(global_marginals) != 3
            or all_rows != self.all_layer_evaluation_row_count
            or global_rows != self.global_evaluation_row_count
            or not _close(
                weighted_current_zero(self.marginals) / all_rows,
                self.all_layer_evaluation_zero_predictor_fisher_loss,
            )
            or not _close(
                weighted_current_fisher(self.marginals) / all_rows,
                self.all_layer_relative_fisher_loss_before
                * self.all_layer_evaluation_zero_predictor_fisher_loss,
            )
            or not _close(
                weighted_current_zero(global_marginals) / global_rows,
                self.global_evaluation_zero_predictor_fisher_loss,
            )
            or not _close(
                weighted_current_fisher(global_marginals) / global_rows,
                self.global_relative_fisher_loss_before
                * self.global_evaluation_zero_predictor_fisher_loss,
            )
        ):
            raise ValueError("global-gated aggregate Fisher mass does not close")

        for marginal in self.marginals:
            invocation_id = marginal.invocation_id
            expected_all_layer = (
                marginal.evaluation_fisher_loss_sum_gain
                / self.all_layer_evaluation_row_count
                / float(self.all_layer_evaluation_zero_predictor_fisher_loss)
            )
            expected_global = (
                marginal.evaluation_fisher_loss_sum_gain
                / self.global_evaluation_row_count
                / float(self.global_evaluation_zero_predictor_fisher_loss)
                if invocation_id in _GLOBAL_INVOCATION_SET
                else 0.0
            )
            if (
                all_layer_gains[invocation_id] != expected_all_layer
                or global_gains[invocation_id] != expected_global
            ):
                raise ValueError("global-gated normalized Fisher gain does not close")

        eligible = self.eligible_marginals
        if not eligible:
            if self.selected_invocation_id is not None:
                raise ValueError("global-gated round selected an ineligible candidate")
            return
        if self.selected_invocation_id is None:
            raise ValueError("global-gated round omitted an eligible candidate")
        objective_gains = self.objective_normalized_fisher_mass_gain_by_site
        best_gain = max(objective_gains[value.invocation_id] for value in eligible)
        tied = tuple(
            value
            for value in eligible
            if objective_gains[value.invocation_id]
            >= best_gain - float(self.tie_tolerance)
        )
        expected_selected = min(
            tied,
            key=lambda value: _mutation_ordinal(value.invocation_id),
        ).invocation_id
        if self.selected_invocation_id != expected_selected:
            raise ValueError("global-gated round did not select the numeric argmax")

    @property
    def global_gate_active(self) -> bool:
        return self.objective_mode == GLOBAL_OBJECTIVE_MODE

    @property
    def objective_normalized_fisher_mass_gain_by_site(self) -> dict[str, float]:
        return dict(
            self.global_normalized_fisher_mass_gains
            if self.global_gate_active
            else self.all_layer_normalized_fisher_mass_gains
        )

    @property
    def eligible_marginals(self) -> tuple[ResidualSpectrumSiteMarginal, ...]:
        objective_gains = self.objective_normalized_fisher_mass_gain_by_site
        return tuple(
            value
            for value in self.marginals
            if value.strictly_improves
            and (
                not self.global_gate_active
                or value.invocation_id in _GLOBAL_INVOCATION_SET
            )
            and objective_gains[value.invocation_id]
            > float(self.minimum_relative_fisher_gain)
        )

    @property
    def eligible_invocation_ids(self) -> tuple[str, ...]:
        return tuple(value.invocation_id for value in self.eligible_marginals)

    @property
    def selected_objective_normalized_fisher_gain(self) -> float | None:
        if self.selected_invocation_id is None:
            return None
        return self.objective_normalized_fisher_mass_gain_by_site[
            self.selected_invocation_id
        ]

    @property
    def source_bank_sha256(self) -> str:
        return self.current_bank_sha256


@dataclass(frozen=True, slots=True)
class GlobalGatedResidualSpectrumAllocationStep:
    """One accepted global-gated residual allocation."""

    allocation_index: int
    invocation_id: str
    objective_mode: str
    source_bank_sha256: str
    result_bank_sha256: str
    marginal_source_bank_sha256: str
    used_cached_off_site_marginal: bool
    selection_objective_normalized_fisher_gain: float
    selection_global_normalized_fisher_gain: float
    selection_all_layer_normalized_fisher_gain: float
    marginal: ResidualSpectrumSiteMarginal
    proposal_fit_before: SparseAttentionTransportOracleEvaluation
    proposal_fit_after: SparseAttentionTransportOracleEvaluation
    evaluation_before: SparseAttentionTransportOracleEvaluation
    evaluation_after: SparseAttentionTransportOracleEvaluation
    global_evaluation_before: SparseAttentionTransportOracleEvaluation
    global_evaluation_after: SparseAttentionTransportOracleEvaluation

    def __post_init__(self) -> None:
        evaluations = (
            self.proposal_fit_before,
            self.proposal_fit_after,
            self.evaluation_before,
            self.evaluation_after,
            self.global_evaluation_before,
            self.global_evaluation_after,
        )
        if (
            type(self.allocation_index) is not int
            or self.allocation_index < 0
            or self.objective_mode not in _OBJECTIVE_MODES
            or self.invocation_id != self.marginal.invocation_id
            or not _is_sha256(self.source_bank_sha256)
            or not _is_sha256(self.result_bank_sha256)
            or self.source_bank_sha256 == self.result_bank_sha256
            or self.marginal_source_bank_sha256
            != self.marginal.source_bank_sha256
            or type(self.used_cached_off_site_marginal) is not bool
            or self.used_cached_off_site_marginal
            != (self.marginal_source_bank_sha256 != self.source_bank_sha256)
            or any(
                not _finite_number(value)
                for value in (
                    self.selection_objective_normalized_fisher_gain,
                    self.selection_global_normalized_fisher_gain,
                    self.selection_all_layer_normalized_fisher_gain,
                )
            )
            or float(self.selection_objective_normalized_fisher_gain)
            <= float(self.marginal.minimum_relative_fisher_gain)
            or not self.marginal.strictly_improves
            or any(
                not isinstance(value, SparseAttentionTransportOracleEvaluation)
                for value in evaluations
            )
        ):
            raise ValueError("global-gated allocation step is invalid")

        observed_all_layer = (
            self.evaluation_before.relative_fisher_loss
            - self.evaluation_after.relative_fisher_loss
        )
        observed_global = (
            self.global_evaluation_before.relative_fisher_loss
            - self.global_evaluation_after.relative_fisher_loss
        )
        observed_objective = (
            observed_global
            if self.objective_mode == GLOBAL_OBJECTIVE_MODE
            else observed_all_layer
        )
        if (
            not _close(
                self.selection_all_layer_normalized_fisher_gain,
                observed_all_layer,
            )
            or not _close(
                self.selection_global_normalized_fisher_gain,
                observed_global,
            )
            or not _close(
                self.selection_objective_normalized_fisher_gain,
                observed_objective,
            )
            or self.proposal_fit_after.relative_fisher_loss
            > self.proposal_fit_before.relative_fisher_loss + 1.0e-12
            or self.evaluation_after.relative_fisher_loss
            > self.evaluation_before.relative_fisher_loss + 1.0e-12
            or self.global_evaluation_after.relative_fisher_loss
            > self.global_evaluation_before.relative_fisher_loss + 1.0e-12
            or (
                self.objective_mode == GLOBAL_OBJECTIVE_MODE
                and self.invocation_id not in _GLOBAL_INVOCATION_SET
            )
        ):
            raise ValueError("global-gated allocation step does not close")


@dataclass(frozen=True, slots=True)
class GlobalGatedResidualSpectrumAllocation:
    """At-most-budget allocation with a global gate and all-18 audit ledger."""

    initial_bank_sha256: str
    final_bank: AllLayerResidualSpectrumGeneratorBank
    allocation_profile: tuple[str, ...]
    first_spectrum: AllLayerFirstResidualSpectrum
    steps: tuple[GlobalGatedResidualSpectrumAllocationStep, ...]
    candidate_rounds: tuple[GlobalGatedResidualSpectrumAllocationRound, ...]
    initial_proposal_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_proposal_fit_oracle: SparseAttentionTransportOracleEvaluation
    initial_evaluation_oracle: SparseAttentionTransportOracleEvaluation
    final_evaluation_oracle: SparseAttentionTransportOracleEvaluation
    initial_global_evaluation_oracle: SparseAttentionTransportOracleEvaluation
    final_global_evaluation_oracle: SparseAttentionTransportOracleEvaluation
    proposal_fit_relative_fisher_loss_history: tuple[float, ...]
    evaluation_relative_fisher_loss_history: tuple[float, ...]
    global_evaluation_relative_fisher_loss_history: tuple[float, ...]
    allocation_budget: int
    global_relative_fisher_threshold: float

    def __post_init__(self) -> None:
        histories = (
            self.proposal_fit_relative_fisher_loss_history,
            self.evaluation_relative_fisher_loss_history,
            self.global_evaluation_relative_fisher_loss_history,
        )
        if (
            not _is_sha256(self.initial_bank_sha256)
            or not isinstance(self.final_bank, AllLayerResidualSpectrumGeneratorBank)
            or type(self.allocation_budget) is not int
            or self.allocation_budget != 2
            or len(self.allocation_profile) > self.allocation_budget
            or len(self.steps) != len(self.allocation_profile)
            or tuple(step.allocation_index for step in self.steps)
            != tuple(range(len(self.steps)))
            or tuple(step.invocation_id for step in self.steps)
            != self.allocation_profile
            or tuple(round_value.allocation_index for round_value in self.candidate_rounds)
            != tuple(range(len(self.candidate_rounds)))
            or any(
                not isinstance(value, GlobalGatedResidualSpectrumAllocationRound)
                for value in self.candidate_rounds
            )
            or self.final_bank.allocated_invocation_ids != self.allocation_profile
            or not isinstance(self.first_spectrum, AllLayerFirstResidualSpectrum)
            or self.first_spectrum.source_bank_sha256 != self.initial_bank_sha256
            or any(len(history) != len(self.steps) + 1 for history in histories)
            or self.proposal_fit_relative_fisher_loss_history[0]
            != self.initial_proposal_fit_oracle.relative_fisher_loss
            or self.proposal_fit_relative_fisher_loss_history[-1]
            != self.final_proposal_fit_oracle.relative_fisher_loss
            or self.evaluation_relative_fisher_loss_history[0]
            != self.initial_evaluation_oracle.relative_fisher_loss
            or self.evaluation_relative_fisher_loss_history[-1]
            != self.final_evaluation_oracle.relative_fisher_loss
            or self.global_evaluation_relative_fisher_loss_history[0]
            != self.initial_global_evaluation_oracle.relative_fisher_loss
            or self.global_evaluation_relative_fisher_loss_history[-1]
            != self.final_global_evaluation_oracle.relative_fisher_loss
            or any(
                right > left + 1.0e-12
                for history in histories
                for left, right in zip(history, history[1:])
            )
            or not _finite_number(self.global_relative_fisher_threshold)
            or not 0.0 < float(self.global_relative_fisher_threshold) < 1.0
            or any(
                not isinstance(value, SparseAttentionTransportOracleEvaluation)
                for value in (
                    self.initial_proposal_fit_oracle,
                    self.final_proposal_fit_oracle,
                    self.initial_evaluation_oracle,
                    self.final_evaluation_oracle,
                    self.initial_global_evaluation_oracle,
                    self.final_global_evaluation_oracle,
                )
            )
        ):
            raise ValueError("global-gated allocation is invalid")

        exhausted = len(self.steps) == self.allocation_budget
        expected_round_count = len(self.steps) if exhausted else len(self.steps) + 1
        if (
            len(self.candidate_rounds) != expected_round_count
            or not self.candidate_rounds
            or self.candidate_rounds[0].current_bank_sha256
            != self.initial_bank_sha256
            or tuple(
                round_value.selected_invocation_id
                for round_value in self.candidate_rounds[: len(self.steps)]
            )
            != self.allocation_profile
            or any(
                round_value.current_bank_sha256 != step.source_bank_sha256
                or round_value.objective_mode != step.objective_mode
                for round_value, step in zip(
                    self.candidate_rounds,
                    self.steps,
                    strict=False,
                )
            )
            or any(
                not _close(
                    round_value.global_relative_fisher_loss_before,
                    self.global_evaluation_relative_fisher_loss_history[index],
                )
                or not _close(
                    round_value.all_layer_relative_fisher_loss_before,
                    self.evaluation_relative_fisher_loss_history[index],
                )
                for index, round_value in enumerate(self.candidate_rounds)
            )
            or any(
                float(round_value.global_relative_fisher_threshold)
                != float(self.global_relative_fisher_threshold)
                for round_value in self.candidate_rounds
            )
            or (
                not exhausted
                and (
                    self.candidate_rounds[-1].selected_invocation_id is not None
                    or self.candidate_rounds[-1].current_bank_sha256
                    != self.final_bank.artifact_sha256
                )
            )
            or (
                self.steps
                and self.steps[-1].result_bank_sha256
                != self.final_bank.artifact_sha256
            )
            or (
                not self.steps
                and self.final_bank.artifact_sha256 != self.initial_bank_sha256
            )
        ):
            raise ValueError("global-gated allocation lineage is invalid")

        first_round = self.candidate_rounds[0]
        if (
            self.first_spectrum.proposal_fit_current_oracle
            is not self.initial_proposal_fit_oracle
            or self.first_spectrum.evaluation_current_oracle
            is not self.initial_evaluation_oracle
            or tuple(value.invocation_id for value in first_round.marginals)
            != self.first_spectrum.invocation_ids
            or any(
                round_marginal is not spectrum_marginal
                for round_marginal, spectrum_marginal in zip(
                    first_round.marginals,
                    self.first_spectrum.per_site,
                    strict=True,
                )
            )
        ):
            raise ValueError("first global-gated round does not match first spectrum")

        state_sha256s = (self.initial_bank_sha256,) + tuple(
            step.result_bank_sha256 for step in self.steps
        )
        for index, round_value in enumerate(self.candidate_rounds):
            if round_value.current_bank_sha256 != state_sha256s[index]:
                raise ValueError("global-gated round state chain drifted")
            for marginal in round_value.marginals:
                source_indices = tuple(
                    state_index
                    for state_index, state_sha256 in enumerate(
                        state_sha256s[: index + 1]
                    )
                    if state_sha256 == marginal.source_bank_sha256
                )
                if len(source_indices) != 1:
                    raise ValueError("global-gated marginal source is not a prior state")
                source_index = source_indices[0]
                if marginal.invocation_id in self.allocation_profile[
                    source_index:index
                ]:
                    raise ValueError("global-gated marginal crossed a same-site update")

        for index, step in enumerate(self.steps):
            round_value = self.candidate_rounds[index]
            if (
                step.source_bank_sha256 != state_sha256s[index]
                or step.result_bank_sha256 != state_sha256s[index + 1]
            ):
                raise ValueError("global-gated step state chain drifted")
            round_marginal = next(
                value
                for value in round_value.marginals
                if value.invocation_id == step.invocation_id
            )
            source_index = state_sha256s[: index + 1].index(
                step.marginal_source_bank_sha256
            )
            expected_objective_gain = (
                dict(round_value.global_normalized_fisher_mass_gains)[
                    step.invocation_id
                ]
                if round_value.objective_mode == GLOBAL_OBJECTIVE_MODE
                else dict(round_value.all_layer_normalized_fisher_mass_gains)[
                    step.invocation_id
                ]
            )
            if (
                step.marginal is not round_marginal
                or step.used_cached_off_site_marginal != (source_index < index)
                or (
                    source_index == index
                    and step.result_bank_sha256
                    != step.marginal.proposal_bank.artifact_sha256
                )
                or not _close(
                    step.selection_global_normalized_fisher_gain,
                    dict(round_value.global_normalized_fisher_mass_gains)[
                        step.invocation_id
                    ],
                )
                or not _close(
                    step.selection_all_layer_normalized_fisher_gain,
                    dict(round_value.all_layer_normalized_fisher_mass_gains)[
                        step.invocation_id
                    ],
                )
                or not _close(
                    step.selection_objective_normalized_fisher_gain,
                    expected_objective_gain,
                )
                or not _close(
                    step.proposal_fit_before.relative_fisher_loss,
                    self.proposal_fit_relative_fisher_loss_history[index],
                )
                or not _close(
                    step.proposal_fit_after.relative_fisher_loss,
                    self.proposal_fit_relative_fisher_loss_history[index + 1],
                )
                or not _close(
                    step.evaluation_before.relative_fisher_loss,
                    self.evaluation_relative_fisher_loss_history[index],
                )
                or not _close(
                    step.evaluation_after.relative_fisher_loss,
                    self.evaluation_relative_fisher_loss_history[index + 1],
                )
                or not _close(
                    step.global_evaluation_before.relative_fisher_loss,
                    self.global_evaluation_relative_fisher_loss_history[index],
                )
                or not _close(
                    step.global_evaluation_after.relative_fisher_loss,
                    self.global_evaluation_relative_fisher_loss_history[index + 1],
                )
                or (
                    index + 1 < len(self.candidate_rounds)
                    and self.candidate_rounds[index + 1].current_bank_sha256
                    != step.result_bank_sha256
                )
                or (
                    index + 1 < len(self.steps)
                    and self.steps[index + 1].source_bank_sha256
                    != step.result_bank_sha256
                )
                or (
                    index == 0
                    and (
                        step.proposal_fit_before
                        is not self.initial_proposal_fit_oracle
                        or step.evaluation_before
                        is not self.initial_evaluation_oracle
                        or step.global_evaluation_before
                        is not self.initial_global_evaluation_oracle
                    )
                )
                or (
                    index > 0
                    and (
                        step.proposal_fit_before
                        is not self.steps[index - 1].proposal_fit_after
                        or step.evaluation_before
                        is not self.steps[index - 1].evaluation_after
                        or step.global_evaluation_before
                        is not self.steps[index - 1].global_evaluation_after
                    )
                )
            ):
                raise ValueError("global-gated step and round binding drifted")

            if index + 1 < len(self.candidate_rounds):
                next_round = self.candidate_rounds[index + 1]
                prior_by_id = {
                    value.invocation_id: value for value in round_value.marginals
                }
                next_by_id = {
                    value.invocation_id: value for value in next_round.marginals
                }
                if (
                    next_by_id[step.invocation_id].source_bank_sha256
                    != step.result_bank_sha256
                    or any(
                        next_by_id[invocation_id] is not prior_by_id[invocation_id]
                        for invocation_id in prior_by_id
                        if invocation_id != step.invocation_id
                    )
                ):
                    raise ValueError("global-gated marginal cache lineage drifted")

        if self.steps and (
            self.steps[-1].proposal_fit_after is not self.final_proposal_fit_oracle
            or self.steps[-1].evaluation_after is not self.final_evaluation_oracle
            or self.steps[-1].global_evaluation_after
            is not self.final_global_evaluation_oracle
        ):
            raise ValueError("final global-gated evaluation binding drifted")
        if not self.steps and (
            self.initial_proposal_fit_oracle is not self.final_proposal_fit_oracle
            or self.initial_evaluation_oracle is not self.final_evaluation_oracle
            or self.initial_global_evaluation_oracle
            is not self.final_global_evaluation_oracle
        ):
            raise ValueError("empty global-gated evaluation binding drifted")

        seen_all_layer_mode = False
        for round_value in self.candidate_rounds:
            if round_value.objective_mode == ALL_LAYER_OBJECTIVE_MODE:
                seen_all_layer_mode = True
            elif seen_all_layer_mode:
                raise ValueError("global gate reopened after all-layer fallback")

    @property
    def budget_exhausted(self) -> bool:
        return len(self.allocation_profile) == self.allocation_budget

    @property
    def global_gate_passed(self) -> bool:
        return (
            self.final_global_evaluation_oracle.relative_fisher_loss
            <= float(self.global_relative_fisher_threshold)
        )

    @property
    def allocation_profile_sha256(self) -> str:
        return residual_allocation_profile_sha256(self.allocation_profile)

    @property
    def termination_reason(self) -> str:
        if self.budget_exhausted:
            return "allocation_budget_exhausted"
        terminal = self.candidate_rounds[-1]
        return (
            "no_eligible_global_candidate_above_gate"
            if terminal.global_gate_active
            else "no_eligible_all_layer_candidate"
        )


def _allocation_round(
    *,
    allocation_index: int,
    current_bank: AllLayerResidualSpectrumGeneratorBank,
    current_evaluation: SparseAttentionTransportOracleEvaluation,
    current_global_evaluation: SparseAttentionTransportOracleEvaluation,
    cache: Mapping[str, ResidualSpectrumSiteMarginal],
    invocation_ids: tuple[str, ...],
    config: AllLayerResidualSpectrumConfig,
    global_relative_fisher_threshold: float,
) -> GlobalGatedResidualSpectrumAllocationRound:
    marginals = tuple(cache[invocation_id] for invocation_id in invocation_ids)
    all_rows = current_evaluation.output_metrics.row_count
    all_zero = current_evaluation.zero_predictor_fisher_loss
    global_rows = current_global_evaluation.output_metrics.row_count
    global_zero = current_global_evaluation.zero_predictor_fisher_loss
    all_gains = tuple(
        (
            marginal.invocation_id,
            marginal.evaluation_fisher_loss_sum_gain / all_rows / all_zero,
        )
        for marginal in marginals
    )
    global_gains = tuple(
        (
            marginal.invocation_id,
            (
                marginal.evaluation_fisher_loss_sum_gain
                / global_rows
                / global_zero
                if marginal.invocation_id in _GLOBAL_INVOCATION_SET
                else 0.0
            ),
        )
        for marginal in marginals
    )
    mode = (
        GLOBAL_OBJECTIVE_MODE
        if current_global_evaluation.relative_fisher_loss
        > float(global_relative_fisher_threshold)
        else ALL_LAYER_OBJECTIVE_MODE
    )
    objective = dict(global_gains if mode == GLOBAL_OBJECTIVE_MODE else all_gains)
    eligible = tuple(
        marginal
        for marginal in marginals
        if marginal.strictly_improves
        and (
            mode != GLOBAL_OBJECTIVE_MODE
            or marginal.invocation_id in _GLOBAL_INVOCATION_SET
        )
        and objective[marginal.invocation_id]
        > float(config.minimum_relative_fisher_gain)
    )
    selected: str | None = None
    if eligible:
        best_gain = max(objective[value.invocation_id] for value in eligible)
        tied = tuple(
            value
            for value in eligible
            if objective[value.invocation_id]
            >= best_gain - float(config.tie_tolerance)
        )
        selected = min(
            tied,
            key=lambda value: _mutation_ordinal(value.invocation_id),
        ).invocation_id
    return GlobalGatedResidualSpectrumAllocationRound(
        allocation_index=allocation_index,
        current_bank_sha256=current_bank.artifact_sha256,
        objective_mode=mode,
        global_relative_fisher_threshold=float(global_relative_fisher_threshold),
        global_relative_fisher_loss_before=float(
            current_global_evaluation.relative_fisher_loss
        ),
        all_layer_relative_fisher_loss_before=float(
            current_evaluation.relative_fisher_loss
        ),
        global_evaluation_row_count=global_rows,
        global_evaluation_zero_predictor_fisher_loss=global_zero,
        all_layer_evaluation_row_count=all_rows,
        all_layer_evaluation_zero_predictor_fisher_loss=all_zero,
        marginals=marginals,
        global_normalized_fisher_mass_gains=global_gains,
        all_layer_normalized_fisher_mass_gains=all_gains,
        selected_invocation_id=selected,
        minimum_relative_fisher_gain=config.minimum_relative_fisher_gain,
        tie_tolerance=config.tie_tolerance,
    )


def greedily_allocate_global_gated_residual_spectrum(
    parent_bank: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    proposal_captures: Sequence[AttentionTransportBatch],
    evaluation_captures: Sequence[AttentionTransportBatch],
    *,
    config: AllLayerResidualSpectrumConfig,
    global_relative_fisher_threshold: float = (
        DEFAULT_GLOBAL_RELATIVE_FISHER_THRESHOLD
    ),
    progress_callback: ResidualSpectrumProgressCallback | None = None,
) -> GlobalGatedResidualSpectrumAllocation:
    """Allocate at most two residuals under a global-first Fisher gate."""

    if not isinstance(
        parent_bank,
        GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    ):
        raise TypeError("parent_bank must be a global site rank-one bank")
    if not isinstance(config, AllLayerResidualSpectrumConfig):
        raise TypeError("config must be an all-layer residual spectrum config")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    if (
        not _finite_number(global_relative_fisher_threshold)
        or not 0.0 < float(global_relative_fisher_threshold) < 1.0
    ):
        raise ValueError("global relative Fisher threshold is invalid")
    if config.allocation_budget != 2:
        raise ValueError("global-gated residual allocation requires budget two")
    if config.coefficient_ridge != 0.0:
        raise ValueError("global-gated residual allocation requires zero ridge")
    config.validate_bank(parent_bank)

    current = all_layer_residual_spectrum_generator_bank_from_parent(parent_bank)
    invocation_ids = tuple(current.hierarchy.invocation_ids)
    fit_values, fit_groups = _all_layer_capture_groups(
        proposal_captures,
        invocation_ids,
    )
    evaluation_values, evaluation_groups = _all_layer_capture_groups(
        evaluation_captures,
        invocation_ids,
    )
    first_spectrum = compute_all_layer_first_residual_spectrum(
        current,
        fit_values,
        evaluation_values,
        config=config,
        progress_callback=progress_callback,
    )
    cache = {value.invocation_id: value for value in first_spectrum.per_site}
    initial_fit = first_spectrum.proposal_fit_current_oracle
    initial_evaluation = first_spectrum.evaluation_current_oracle
    initial_global = _global_evaluation(
        current,
        evaluation_values,
        config=config,
    )
    current_fit = initial_fit
    current_evaluation = initial_evaluation
    current_global = initial_global
    fit_history = [float(initial_fit.relative_fisher_loss)]
    evaluation_history = [float(initial_evaluation.relative_fisher_loss)]
    global_history = [float(initial_global.relative_fisher_loss)]
    steps: list[GlobalGatedResidualSpectrumAllocationStep] = []
    profile: list[str] = []
    rounds: list[GlobalGatedResidualSpectrumAllocationRound] = []

    for allocation_index in range(config.allocation_budget):
        round_value = _allocation_round(
            allocation_index=allocation_index,
            current_bank=current,
            current_evaluation=current_evaluation,
            current_global_evaluation=current_global,
            cache=cache,
            invocation_ids=invocation_ids,
            config=config,
            global_relative_fisher_threshold=float(
                global_relative_fisher_threshold
            ),
        )
        rounds.append(round_value)
        if round_value.selected_invocation_id is None:
            break

        selected = cache[round_value.selected_invocation_id]
        source_sha = current.artifact_sha256
        used_cached = selected.source_bank_sha256 != source_sha
        candidate = (
            _apply_cached_marginal(current, selected)
            if used_cached
            else selected.proposal_bank
        )

        site_fit_before = evaluate_sparse_attention_transport_bank(
            current,
            fit_groups[selected.invocation_id],
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        site_fit_after = evaluate_sparse_attention_transport_bank(
            candidate,
            fit_groups[selected.invocation_id],
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        site_evaluation_before = evaluate_sparse_attention_transport_bank(
            current,
            evaluation_groups[selected.invocation_id],
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        site_evaluation_after = evaluate_sparse_attention_transport_bank(
            candidate,
            evaluation_groups[selected.invocation_id],
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        observed_site_gain = (
            site_evaluation_before.relative_fisher_loss
            - site_evaluation_after.relative_fisher_loss
        )
        if (
            site_fit_after.relative_fisher_loss
            >= site_fit_before.relative_fisher_loss
            - float(config.minimum_relative_fisher_gain)
            or observed_site_gain <= float(config.minimum_relative_fisher_gain)
            or not _close(observed_site_gain, selected.evaluation_relative_fisher_gain)
        ):
            raise RuntimeError("cached global-gated marginal did not replay")

        candidate_fit = evaluate_sparse_attention_transport_bank(
            candidate,
            fit_values,
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        candidate_evaluation = evaluate_sparse_attention_transport_bank(
            candidate,
            evaluation_values,
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        candidate_global = _global_evaluation(
            candidate,
            evaluation_values,
            config=config,
        )
        observed_all_gain = (
            current_evaluation.relative_fisher_loss
            - candidate_evaluation.relative_fisher_loss
        )
        observed_global_gain = (
            current_global.relative_fisher_loss
            - candidate_global.relative_fisher_loss
        )
        selected_all_gain = dict(
            round_value.all_layer_normalized_fisher_mass_gains
        )[selected.invocation_id]
        selected_global_gain = dict(
            round_value.global_normalized_fisher_mass_gains
        )[selected.invocation_id]
        selected_objective_gain = (
            selected_global_gain
            if round_value.objective_mode == GLOBAL_OBJECTIVE_MODE
            else selected_all_gain
        )
        if (
            candidate_fit.relative_fisher_loss
            > current_fit.relative_fisher_loss + 1.0e-12
            or candidate_evaluation.relative_fisher_loss
            > current_evaluation.relative_fisher_loss + 1.0e-12
            or candidate_global.relative_fisher_loss
            > current_global.relative_fisher_loss + 1.0e-12
            or not _close(observed_all_gain, selected_all_gain)
            or not _close(observed_global_gain, selected_global_gain)
            or selected_objective_gain
            <= float(config.minimum_relative_fisher_gain)
        ):
            raise RuntimeError("accepted global-gated allocation did not close")

        steps.append(
            GlobalGatedResidualSpectrumAllocationStep(
                allocation_index=allocation_index,
                invocation_id=selected.invocation_id,
                objective_mode=round_value.objective_mode,
                source_bank_sha256=source_sha,
                result_bank_sha256=candidate.artifact_sha256,
                marginal_source_bank_sha256=selected.source_bank_sha256,
                used_cached_off_site_marginal=used_cached,
                selection_objective_normalized_fisher_gain=(
                    selected_objective_gain
                ),
                selection_global_normalized_fisher_gain=selected_global_gain,
                selection_all_layer_normalized_fisher_gain=selected_all_gain,
                marginal=selected,
                proposal_fit_before=current_fit,
                proposal_fit_after=candidate_fit,
                evaluation_before=current_evaluation,
                evaluation_after=candidate_evaluation,
                global_evaluation_before=current_global,
                global_evaluation_after=candidate_global,
            )
        )
        profile.append(selected.invocation_id)
        current = candidate
        current_fit = candidate_fit
        current_evaluation = candidate_evaluation
        current_global = candidate_global
        fit_history.append(float(current_fit.relative_fisher_loss))
        evaluation_history.append(float(current_evaluation.relative_fisher_loss))
        global_history.append(float(current_global.relative_fisher_loss))
        _notify_progress(
            progress_callback,
            "global_gated_greedy_allocation",
            allocation_index + 1,
            config.allocation_budget,
            selected.invocation_id,
        )

        if allocation_index + 1 < config.allocation_budget:
            cache[selected.invocation_id] = compute_site_residual_spectrum_marginal(
                current,
                fit_groups[selected.invocation_id],
                evaluation_groups[selected.invocation_id],
                config=config,
                progress_callback=progress_callback,
            )

    return GlobalGatedResidualSpectrumAllocation(
        initial_bank_sha256=first_spectrum.source_bank_sha256,
        final_bank=current,
        allocation_profile=tuple(profile),
        first_spectrum=first_spectrum,
        steps=tuple(steps),
        candidate_rounds=tuple(rounds),
        initial_proposal_fit_oracle=initial_fit,
        final_proposal_fit_oracle=current_fit,
        initial_evaluation_oracle=initial_evaluation,
        final_evaluation_oracle=current_evaluation,
        initial_global_evaluation_oracle=initial_global,
        final_global_evaluation_oracle=current_global,
        proposal_fit_relative_fisher_loss_history=tuple(fit_history),
        evaluation_relative_fisher_loss_history=tuple(evaluation_history),
        global_evaluation_relative_fisher_loss_history=tuple(global_history),
        allocation_budget=config.allocation_budget,
        global_relative_fisher_threshold=float(global_relative_fisher_threshold),
    )


__all__ = [
    "ALL_LAYER_OBJECTIVE_MODE",
    "DEFAULT_GLOBAL_RELATIVE_FISHER_THRESHOLD",
    "GLOBAL_GATED_RESIDUAL_GREEDY_SEMANTICS",
    "GLOBAL_GATED_RESIDUAL_LEDGER_SEMANTICS",
    "GLOBAL_GATED_RESIDUAL_OBJECTIVE_SEMANTICS",
    "GLOBAL_OBJECTIVE_MODE",
    "GlobalGatedResidualSpectrumAllocation",
    "GlobalGatedResidualSpectrumAllocationRound",
    "GlobalGatedResidualSpectrumAllocationStep",
    "greedily_allocate_global_gated_residual_spectrum",
]
