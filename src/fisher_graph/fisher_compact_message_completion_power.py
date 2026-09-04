"""Reporting-only development-to-guard power projection.

The projection in this module is deliberately empirical and deterministic.  It
uses development analysis units only; it has no eligibility or guard-decision
input and its result is permanently labelled ``reporting_only``.

Method
------

* The assumed guard size is exactly 12 independent analysis units.
* The frozen development strata are allocated to those 12 slots by exact
  Hamilton apportionment.  Within each stratum, unit archetypes are sampled
  with replacement by SHA-256 counter draws using seed 424242.
* There are 10,000 projection replicates.  Observed paired effects
  ``baseline - J4`` are centered within stratum.  These centered effects form
  the empirical null; the alternative adds ``0.05 * baseline`` per sampled
  unit.
* A two-sided 95% basic empirical interval uses the 2.5% and 97.5% linear
  quantiles of the projected null means.  A replicate confirms the frozen 5%
  alternative only when its lower bound is positive and its aggregate relative
  effect is at least 5%; a projected baseline at or below the frozen ``1e-12``
  denominator floor cannot confirm.
* One- and two-adverse-unit sensitivity values are absolute paired effects in
  the same units as the interval width.  They replace one or two typical 5%
  alternative units with the least-favourable distinct observed development
  effects, capped at zero so that an "adverse" substitution cannot improve the
  projection.
* Expected per-chart counts are replicate means rounded half up.  A chart is
  reported supported when its rounded expected independent-unit count is at
  least four, matching the development chart-support convention.

The seed, replicate count, interval level, alternative, and guard size are
constants rather than caller-selectable analysis choices.  The method estimates
what a future panel might look like; it must never permit, prevent, or modify a
guard claim.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import struct
from typing import Final

from fisher_graph.fisher_compact_message_completion_splits import stratum_order_key


GUARD_ANALYSIS_UNIT_COUNT: Final = 12
POWER_PROJECTION_REPLICATES: Final = 10_000
POWER_PROJECTION_SEED: Final = 424_242
POWER_PROJECTION_CONFIDENCE: Final = 0.95
POWER_PROJECTION_ALTERNATIVE: Final = 0.05
POWER_RELATIVE_DENOMINATOR_FLOOR: Final = 1e-12
POWER_PROJECTION_METHOD: Final = "deterministic_stratified_empirical_basic_v1"

_DRAW_DOMAIN: Final = b"fisher-cmc/development-guard-power/draw/v1"


class PowerProjectionError(ValueError):
    """Development inputs cannot define the frozen empirical projection."""


@dataclass(frozen=True, slots=True)
class SupportedChartMembership:
    """Supported rows from one aligned development unit in one active chart.

    Omit a record when the unit has no supported row in that chart.  Pass
    ``active_chart_ids`` to :func:`project_development_power` when an active
    chart has no supported membership anywhere and must still be reported.
    """

    chart_id: int
    sequence_count: int
    row_count: int


@dataclass(frozen=True, slots=True)
class ExpectedChartSupport:
    """Integer-valued expected support record required by the artifact schema."""

    chart_id: int
    analysis_unit_count: int
    sequence_count: int
    row_count: int
    supported: bool

    def as_artifact_record(self) -> dict[str, object]:
        return {
            "chart_id": self.chart_id,
            "analysis_unit_count": self.analysis_unit_count,
            "sequence_count": self.sequence_count,
            "row_count": self.row_count,
            "supported": self.supported,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentPowerProjection:
    """Frozen reporting payload for ``development_power_assessment.json``."""

    expected_paired_interval_width: float
    probability_confirming_five_percent_effect: float
    one_adverse_unit_sensitivity: float
    two_adverse_unit_sensitivity: float
    expected_chart_support: tuple[ExpectedChartSupport, ...]
    guard_analysis_unit_count_assumed: int = GUARD_ANALYSIS_UNIT_COUNT
    reporting_only: bool = True
    eligibility_affecting: bool = False

    def as_artifact_body(self) -> dict[str, object]:
        """Return exactly the current strict development-power body schema."""

        return {
            "guard_analysis_unit_count_assumed": (
                self.guard_analysis_unit_count_assumed
            ),
            "expected_paired_interval_width": self.expected_paired_interval_width,
            "probability_confirming_five_percent_effect": (
                self.probability_confirming_five_percent_effect
            ),
            "one_adverse_unit_sensitivity": self.one_adverse_unit_sensitivity,
            "two_adverse_unit_sensitivity": self.two_adverse_unit_sensitivity,
            "expected_chart_support": [
                record.as_artifact_record()
                for record in self.expected_chart_support
            ],
            "reporting_only": self.reporting_only,
            "eligibility_affecting": self.eligibility_affecting,
        }


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise PowerProjectionError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PowerProjectionError(
            f"{label} must be a finite real number"
        ) from error
    if not math.isfinite(result):
        raise PowerProjectionError(f"{label} must be finite")
    return result


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise PowerProjectionError("projection quantiles require replicates")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (1.0 - fraction) * ordered[lower] + fraction * ordered[upper]


def _hamilton_guard_allocations(strata: Sequence[str]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = defaultdict(int)
    for stratum in strata:
        counts[stratum] += 1
    total = len(strata)
    allocation = {
        stratum: (count * GUARD_ANALYSIS_UNIT_COUNT) // total
        for stratum, count in counts.items()
    }
    remaining = GUARD_ANALYSIS_UNIT_COUNT - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda stratum: (
            -((counts[stratum] * GUARD_ANALYSIS_UNIT_COUNT) % total),
            *stratum_order_key(stratum),
        ),
    )
    for stratum in order[:remaining]:
        allocation[stratum] += 1
    return tuple(
        (stratum, allocation[stratum])
        for stratum in sorted(counts, key=stratum_order_key)
    )


def _counter_draw(
    *,
    replicate: int,
    stratum: str,
    draw: int,
    population_size: int,
) -> int:
    label = stratum.encode("utf-8")
    digest = hashlib.sha256(_DRAW_DOMAIN + b"\0")
    digest.update(struct.pack(">Q", POWER_PROJECTION_SEED))
    digest.update(struct.pack(">Q", replicate))
    digest.update(struct.pack(">I", len(label)))
    digest.update(label)
    digest.update(struct.pack(">I", draw))
    return int.from_bytes(digest.digest(), "big") % population_size


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def project_development_power(
    baseline_unit_losses: Sequence[float],
    j4_unit_losses: Sequence[float],
    frozen_strata: Sequence[str],
    supported_chart_memberships: Sequence[
        Sequence[SupportedChartMembership]
    ],
    *,
    unit_ids: Sequence[bytes] | None = None,
    active_chart_ids: Sequence[int] | None = None,
) -> DevelopmentPowerProjection:
    """Project a 12-unit guard panel from development observations only.

    Inputs are aligned by development analysis unit.  ``unit_ids`` are optional
    for convenience, but supplying unique raw 32-byte commitments also makes
    the projection invariant to caller input order.  There is intentionally no
    eligibility/result argument: this diagnostic cannot influence readiness.
    """

    baseline = tuple(
        _finite_float(value, label="baseline unit loss")
        for value in baseline_unit_losses
    )
    j4 = tuple(_finite_float(value, label="J4 unit loss") for value in j4_unit_losses)
    strata = tuple(frozen_strata)
    memberships = tuple(tuple(records) for records in supported_chart_memberships)
    unit_count = len(baseline)
    if unit_count < 2:
        raise PowerProjectionError("at least two development units are required")
    if not (
        len(j4) == unit_count
        and len(strata) == unit_count
        and len(memberships) == unit_count
    ):
        raise PowerProjectionError(
            "losses, frozen strata, and chart memberships must be aligned"
        )
    if any(value < 0.0 for value in baseline + j4):
        raise PowerProjectionError("unit losses must be nonnegative")
    if any(not isinstance(value, str) or not value for value in strata):
        raise PowerProjectionError("frozen strata must be nonempty strings")

    if unit_ids is None:
        identifiers = tuple(
            index.to_bytes(8, "big", signed=False) for index in range(unit_count)
        )
    else:
        identifiers = tuple(unit_ids)
        if len(identifiers) != unit_count:
            raise PowerProjectionError("unit_ids must align with unit losses")
        if any(type(value) is not bytes or len(value) != 32 for value in identifiers):
            raise PowerProjectionError("unit_ids must be raw 32-byte commitments")
        if len(set(identifiers)) != unit_count:
            raise PowerProjectionError("unit_ids must be unique")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        grouped[stratum].append(index)
    if any(len(indexes) < 2 for indexes in grouped.values()):
        raise PowerProjectionError(
            "every frozen projection stratum must contain at least two units"
        )
    for indexes in grouped.values():
        indexes.sort(key=lambda index: identifiers[index])

    derived_charts: set[int] = set()
    membership_maps: list[dict[int, SupportedChartMembership]] = []
    for unit_records in memberships:
        by_chart: dict[int, SupportedChartMembership] = {}
        for record in unit_records:
            if not isinstance(record, SupportedChartMembership):
                raise PowerProjectionError(
                    "chart memberships must be SupportedChartMembership values"
                )
            if type(record.chart_id) is not int or record.chart_id < 0:
                raise PowerProjectionError("chart_id must be a nonnegative integer")
            if (
                type(record.sequence_count) is not int
                or record.sequence_count <= 0
                or type(record.row_count) is not int
                or record.row_count <= 0
                or record.sequence_count > record.row_count
            ):
                raise PowerProjectionError(
                    "supported chart sequence/row counts must be positive and ordered"
                )
            if record.chart_id in by_chart:
                raise PowerProjectionError(
                    "a development unit may have only one record per chart"
                )
            by_chart[record.chart_id] = record
            derived_charts.add(record.chart_id)
        membership_maps.append(by_chart)

    if active_chart_ids is None:
        charts = tuple(sorted(derived_charts))
    else:
        if any(type(chart) is not int or chart < 0 for chart in active_chart_ids):
            raise PowerProjectionError(
                "active_chart_ids must be nonnegative integers"
            )
        charts = tuple(sorted(set(active_chart_ids)))
        if len(charts) != len(tuple(active_chart_ids)):
            raise PowerProjectionError("active_chart_ids must be unique")
        if not derived_charts.issubset(charts):
            raise PowerProjectionError(
                "supported membership refers to an inactive chart"
            )
    if not charts:
        raise PowerProjectionError("at least one active chart is required")

    effects = tuple(base - candidate for base, candidate in zip(baseline, j4, strict=True))
    stratum_effect_means = {
        stratum: math.fsum(effects[index] for index in indexes) / len(indexes)
        for stratum, indexes in grouped.items()
    }
    residuals = tuple(
        effect - stratum_effect_means[stratum]
        for effect, stratum in zip(effects, strata, strict=True)
    )
    allocations = _hamilton_guard_allocations(strata)

    null_means: list[float] = []
    alternative_means: list[float] = []
    alternative_baseline_means: list[float] = []
    support_unit_totals = {chart: 0 for chart in charts}
    support_sequence_totals = {chart: 0 for chart in charts}
    support_row_totals = {chart: 0 for chart in charts}
    for replicate in range(POWER_PROJECTION_REPLICATES):
        selected: list[int] = []
        for stratum, allocation in allocations:
            group = grouped[stratum]
            selected.extend(
                group[
                    _counter_draw(
                        replicate=replicate,
                        stratum=stratum,
                        draw=draw,
                        population_size=len(group),
                    )
                ]
                for draw in range(allocation)
            )
        if len(selected) != GUARD_ANALYSIS_UNIT_COUNT:
            raise AssertionError("Hamilton projection did not allocate 12 guard units")

        null_mean = math.fsum(residuals[index] for index in selected) / len(selected)
        alternative_mean = math.fsum(
            POWER_PROJECTION_ALTERNATIVE * baseline[index] + residuals[index]
            for index in selected
        ) / len(selected)
        baseline_mean = math.fsum(baseline[index] for index in selected) / len(selected)
        null_means.append(null_mean)
        alternative_means.append(alternative_mean)
        alternative_baseline_means.append(baseline_mean)

        for index in selected:
            for chart, record in membership_maps[index].items():
                support_unit_totals[chart] += 1
                support_sequence_totals[chart] += record.sequence_count
                support_row_totals[chart] += record.row_count

    tail = (1.0 - POWER_PROJECTION_CONFIDENCE) / 2.0
    null_lower = _linear_quantile(null_means, tail)
    null_upper = _linear_quantile(null_means, 1.0 - tail)
    interval_width = max(0.0, null_upper - null_lower)
    confirmed = 0
    for effect, base in zip(
        alternative_means, alternative_baseline_means, strict=True
    ):
        lower_bound = effect - null_upper
        relative_effect = (
            effect / base if base > POWER_RELATIVE_DENOMINATOR_FLOOR else None
        )
        if (
            lower_bound > 0.0
            and relative_effect is not None
            and relative_effect + 1e-15 >= POWER_PROJECTION_ALTERNATIVE
        ):
            confirmed += 1
    confirmation_probability = confirmed / POWER_PROJECTION_REPLICATES

    expected_baseline = math.fsum(
        Fraction(allocation, GUARD_ANALYSIS_UNIT_COUNT)
        * Fraction.from_float(
            math.fsum(baseline[index] for index in grouped[stratum])
            / len(grouped[stratum])
        )
        for stratum, allocation in allocations
    )
    expected_alternative_effect = POWER_PROJECTION_ALTERNATIVE * float(
        expected_baseline
    )
    least_favourable = sorted(
        range(unit_count), key=lambda index: (effects[index], identifiers[index])
    )[:2]
    adverse_effects = tuple(min(effects[index], 0.0) for index in least_favourable)

    def adverse_sensitivity(count: int) -> float:
        total = (
            (GUARD_ANALYSIS_UNIT_COUNT - count) * expected_alternative_effect
            + math.fsum(adverse_effects[:count])
        )
        return total / GUARD_ANALYSIS_UNIT_COUNT

    chart_support = tuple(
        ExpectedChartSupport(
            chart_id=chart,
            analysis_unit_count=_round_half_up(
                support_unit_totals[chart] / POWER_PROJECTION_REPLICATES
            ),
            sequence_count=_round_half_up(
                support_sequence_totals[chart] / POWER_PROJECTION_REPLICATES
            ),
            row_count=_round_half_up(
                support_row_totals[chart] / POWER_PROJECTION_REPLICATES
            ),
            supported=(
                _round_half_up(
                    support_unit_totals[chart] / POWER_PROJECTION_REPLICATES
                )
                >= 4
            ),
        )
        for chart in charts
    )
    scalar_values = (
        interval_width,
        confirmation_probability,
        adverse_sensitivity(1),
        adverse_sensitivity(2),
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise PowerProjectionError("projection produced a non-finite result")
    return DevelopmentPowerProjection(
        expected_paired_interval_width=interval_width,
        probability_confirming_five_percent_effect=confirmation_probability,
        one_adverse_unit_sensitivity=adverse_sensitivity(1),
        two_adverse_unit_sensitivity=adverse_sensitivity(2),
        expected_chart_support=chart_support,
    )


__all__ = [
    "GUARD_ANALYSIS_UNIT_COUNT",
    "POWER_PROJECTION_ALTERNATIVE",
    "POWER_PROJECTION_CONFIDENCE",
    "POWER_PROJECTION_METHOD",
    "POWER_PROJECTION_REPLICATES",
    "POWER_PROJECTION_SEED",
    "POWER_RELATIVE_DENOMINATOR_FLOOR",
    "DevelopmentPowerProjection",
    "ExpectedChartSupport",
    "PowerProjectionError",
    "SupportedChartMembership",
    "project_development_power",
]
