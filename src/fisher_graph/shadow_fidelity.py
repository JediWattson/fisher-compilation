"""Source-authoritative shadow fidelity metrics.

This module deliberately knows nothing about a particular model or executor.
Callers provide the supervised source and candidate logits for each example.
The source path remains authoritative; candidate logits are consumed only to
measure how closely a shadow execution path reproduces it.

An exact example-to-family manifest is required.  That makes family-disjoint
evaluation auditable: undeclared examples, family relabeling, duplicates, and
incomplete evaluation all fail closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True, slots=True)
class ShadowFidelityGates:
    """Thresholds for the established source-shadow fidelity gate."""

    absolute_delta_nll_per_token_max: float = 0.05
    top1_agreement_to_source_min: float = 0.95
    source_to_candidate_kl_per_token_max: float = 0.05
    per_prompt_p90_absolute_delta_nll_per_token_max: float = 0.10
    per_prompt_p10_top1_agreement_to_source_min: float = 0.90

    def __post_init__(self) -> None:
        upper_bounds = (
            self.absolute_delta_nll_per_token_max,
            self.source_to_candidate_kl_per_token_max,
            self.per_prompt_p90_absolute_delta_nll_per_token_max,
        )
        unit_interval = (
            self.top1_agreement_to_source_min,
            self.per_prompt_p10_top1_agreement_to_source_min,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in upper_bounds):
            raise ValueError("NLL and KL gate maxima must be finite and nonnegative")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in unit_interval
        ):
            raise ValueError("agreement gate minima must lie in [0, 1]")

    def metadata(self) -> dict[str, float]:
        return {
            "absolute_delta_nll_per_token_max": (
                self.absolute_delta_nll_per_token_max
            ),
            "top1_agreement_to_source_min": (
                self.top1_agreement_to_source_min
            ),
            "source_to_candidate_kl_per_token_max": (
                self.source_to_candidate_kl_per_token_max
            ),
            "per_prompt_p90_absolute_delta_nll_per_token_max": (
                self.per_prompt_p90_absolute_delta_nll_per_token_max
            ),
            "per_prompt_p10_top1_agreement_to_source_min": (
                self.per_prompt_p10_top1_agreement_to_source_min
            ),
        }

    def evaluate(
        self,
        *,
        delta_nll_per_token: float,
        top1_agreement_to_source: float,
        source_to_candidate_kl_per_token: float,
        per_prompt_p90_absolute_delta_nll_per_token: float,
        per_prompt_p10_top1_agreement_to_source: float,
    ) -> dict[str, bool]:
        values = (
            delta_nll_per_token,
            top1_agreement_to_source,
            source_to_candidate_kl_per_token,
            per_prompt_p90_absolute_delta_nll_per_token,
            per_prompt_p10_top1_agreement_to_source,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("shadow fidelity gate inputs must be finite")
        result = {
            "absolute_delta_nll_per_token": (
                abs(delta_nll_per_token)
                <= self.absolute_delta_nll_per_token_max
            ),
            "aggregate_top1_agreement": (
                top1_agreement_to_source
                >= self.top1_agreement_to_source_min
            ),
            "source_to_candidate_kl_per_token": (
                source_to_candidate_kl_per_token
                <= self.source_to_candidate_kl_per_token_max
            ),
            "per_prompt_p90_absolute_delta_nll": (
                per_prompt_p90_absolute_delta_nll_per_token
                <= self.per_prompt_p90_absolute_delta_nll_per_token_max
            ),
            "per_prompt_p10_top1_agreement": (
                per_prompt_p10_top1_agreement_to_source
                >= self.per_prompt_p10_top1_agreement_to_source_min
            ),
        }
        result["passed"] = all(result.values())
        return result


ESTABLISHED_SHADOW_FIDELITY_GATES = ShadowFidelityGates()


@dataclass(frozen=True, slots=True)
class ShadowFidelityExample:
    """One prompt's supervised source/candidate comparison."""

    example_id: str
    family_id: str
    source_logits: Tensor
    candidate_logits: Tensor
    targets: Tensor


ShadowFidelityInput = ShadowFidelityExample | Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ExampleStatistics:
    example_id: str
    family_id: str
    supervised_tokens: int
    source_summed_nll: float
    candidate_summed_nll: float
    source_to_candidate_summed_kl: float
    top1_matches: int

    @property
    def source_nll_per_token(self) -> float:
        return self.source_summed_nll / self.supervised_tokens

    @property
    def candidate_nll_per_token(self) -> float:
        return self.candidate_summed_nll / self.supervised_tokens

    @property
    def delta_nll_per_token(self) -> float:
        return (
            self.candidate_summed_nll - self.source_summed_nll
        ) / self.supervised_tokens

    @property
    def source_to_candidate_kl_per_token(self) -> float:
        return self.source_to_candidate_summed_kl / self.supervised_tokens

    @property
    def top1_agreement_to_source(self) -> float:
        return self.top1_matches / self.supervised_tokens


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    if value != value.strip():
        raise ValueError(f"{name} cannot contain surrounding whitespace")
    return value


def _nearest_rank(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile values cannot be empty")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must lie in [0, 1]")
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(math.ceil(fraction * len(ordered))) - 1,
        ),
    )
    return ordered[index]


def _lower_nearest_rank(values: Iterable[float], fraction: float) -> float:
    """Return a lower-tail rank using the project's adverse-tail convention."""

    materialized = [float(value) for value in values]
    return -_nearest_rank((-value for value in materialized), 1.0 - fraction)


def _coerce_example(example: ShadowFidelityInput) -> ShadowFidelityExample:
    if isinstance(example, ShadowFidelityExample):
        return example
    if not isinstance(example, Mapping):
        raise TypeError(
            "shadow examples must be ShadowFidelityExample instances or mappings"
        )
    required = (
        "example_id",
        "family_id",
        "source_logits",
        "candidate_logits",
        "targets",
    )
    missing = [key for key in required if key not in example]
    if missing:
        raise ValueError(
            "shadow example mapping is missing required keys: "
            + ", ".join(missing)
        )
    source_logits = example["source_logits"]
    candidate_logits = example["candidate_logits"]
    targets = example["targets"]
    if not isinstance(source_logits, Tensor):
        raise TypeError("source_logits must be a torch.Tensor")
    if not isinstance(candidate_logits, Tensor):
        raise TypeError("candidate_logits must be a torch.Tensor")
    if not isinstance(targets, Tensor):
        raise TypeError("targets must be a torch.Tensor")
    return ShadowFidelityExample(
        example_id=_identifier(example["example_id"], name="example_id"),
        family_id=_identifier(example["family_id"], name="family_id"),
        source_logits=source_logits,
        candidate_logits=candidate_logits,
        targets=targets,
    )


def _validate_tensors(example: ShadowFidelityExample) -> None:
    source = example.source_logits
    candidate = example.candidate_logits
    targets = example.targets
    if not isinstance(source, Tensor):
        raise TypeError("source_logits must be a torch.Tensor")
    if not isinstance(candidate, Tensor):
        raise TypeError("candidate_logits must be a torch.Tensor")
    if not isinstance(targets, Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if source.ndim != 2:
        raise ValueError("source_logits must have shape [supervised_tokens, vocab]")
    if source.shape != candidate.shape:
        raise ValueError("source_logits and candidate_logits must have equal shape")
    if source.shape[0] == 0:
        raise ValueError("each shadow example must contain supervised tokens")
    if source.shape[1] < 2:
        raise ValueError("shadow logits must contain at least two vocabulary items")
    if not source.dtype.is_floating_point:
        raise TypeError("source_logits must use a floating dtype")
    if not candidate.dtype.is_floating_point:
        raise TypeError("candidate_logits must use a floating dtype")
    if targets.ndim != 1 or targets.shape[0] != source.shape[0]:
        raise ValueError("targets must have shape [supervised_tokens]")
    if targets.dtype not in _INTEGER_DTYPES:
        raise TypeError("targets must use an integer dtype")
    if not bool(torch.isfinite(source).all().item()):
        raise ValueError("source_logits must be finite")
    if not bool(torch.isfinite(candidate).all().item()):
        raise ValueError("candidate_logits must be finite")
    if bool((targets < 0).any().item()) or bool(
        (targets >= source.shape[1]).any().item()
    ):
        raise ValueError("targets must index the logits vocabulary")


def _measure_example(
    example: ShadowFidelityExample,
    *,
    vocab_chunk_size: int,
) -> _ExampleStatistics:
    _validate_tensors(example)
    with torch.inference_mode():
        source = example.source_logits.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        candidate = example.candidate_logits.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        targets = example.targets.detach().to(device="cpu", dtype=torch.int64)

        source_log_normalizer = torch.logsumexp(source, dim=-1)
        candidate_log_normalizer = torch.logsumexp(candidate, dim=-1)
        token_indices = torch.arange(targets.shape[0])
        source_summed_nll = float(
            (
                source_log_normalizer - source[token_indices, targets]
            ).sum().item()
        )
        candidate_summed_nll = float(
            (
                candidate_log_normalizer - candidate[token_indices, targets]
            ).sum().item()
        )

        summed_kl = 0.0
        for start in range(0, source.shape[1], vocab_chunk_size):
            stop = min(source.shape[1], start + vocab_chunk_size)
            source_log_probability = (
                source[:, start:stop] - source_log_normalizer[:, None]
            )
            candidate_log_probability = (
                candidate[:, start:stop] - candidate_log_normalizer[:, None]
            )
            summed_kl += float(
                (
                    source_log_probability.exp()
                    * (source_log_probability - candidate_log_probability)
                )
                .sum()
                .item()
            )
        # Floating-point cancellation can make mathematically nonnegative KL
        # infinitesimally negative.
        source_to_candidate_summed_kl = max(0.0, summed_kl)
        top1_matches = int(
            (
                source.argmax(dim=-1) == candidate.argmax(dim=-1)
            ).sum().item()
        )

    return _ExampleStatistics(
        example_id=example.example_id,
        family_id=example.family_id,
        supervised_tokens=int(targets.shape[0]),
        source_summed_nll=source_summed_nll,
        candidate_summed_nll=candidate_summed_nll,
        source_to_candidate_summed_kl=source_to_candidate_summed_kl,
        top1_matches=top1_matches,
    )


def _aggregate_rows(
    rows: Iterable[_ExampleStatistics],
) -> dict[str, int | float]:
    materialized = sorted(rows, key=lambda row: row.example_id)
    if not materialized:
        raise ValueError("shadow fidelity rows cannot be empty")
    supervised_tokens = sum(row.supervised_tokens for row in materialized)
    source_summed_nll = math.fsum(
        row.source_summed_nll for row in materialized
    )
    candidate_summed_nll = math.fsum(
        row.candidate_summed_nll for row in materialized
    )
    source_to_candidate_summed_kl = max(
        0.0,
        math.fsum(
            row.source_to_candidate_summed_kl for row in materialized
        ),
    )
    top1_matches = sum(row.top1_matches for row in materialized)
    return {
        "example_count": len(materialized),
        "supervised_tokens": supervised_tokens,
        "source_summed_nll": source_summed_nll,
        "candidate_summed_nll": candidate_summed_nll,
        "source_nll_per_token": source_summed_nll / supervised_tokens,
        "candidate_nll_per_token": (
            candidate_summed_nll / supervised_tokens
        ),
        "delta_nll_per_token": (
            candidate_summed_nll - source_summed_nll
        )
        / supervised_tokens,
        "source_to_candidate_summed_kl": source_to_candidate_summed_kl,
        "source_to_candidate_kl_per_token": (
            source_to_candidate_summed_kl / supervised_tokens
        ),
        "top1_matches": top1_matches,
        "top1_agreement_to_source": top1_matches / supervised_tokens,
    }


def _prompt_tail(rows: Iterable[_ExampleStatistics]) -> dict[str, object]:
    materialized = sorted(rows, key=lambda row: row.example_id)
    absolute_delta = [abs(row.delta_nll_per_token) for row in materialized]
    top1 = [row.top1_agreement_to_source for row in materialized]
    return {
        "absolute_delta_nll_per_token": {
            "p90": _nearest_rank(absolute_delta, 0.90),
            "worst": max(absolute_delta),
        },
        "top1_agreement_to_source": {
            "p10": _lower_nearest_rank(top1, 0.10),
            "worst": min(top1),
        },
    }


def _family_summary(
    rows: Iterable[_ExampleStatistics],
) -> dict[str, object]:
    grouped: dict[str, list[_ExampleStatistics]] = {}
    for row in rows:
        grouped.setdefault(row.family_id, []).append(row)

    families: list[dict[str, object]] = []
    for family_id in sorted(grouped):
        family_rows = grouped[family_id]
        aggregate = _aggregate_rows(family_rows)
        prompt_tail = _prompt_tail(family_rows)
        families.append(
            {
                "family_id": family_id,
                **aggregate,
                "absolute_delta_nll_per_token": abs(
                    float(aggregate["delta_nll_per_token"])
                ),
                "per_prompt_p90_absolute_delta_nll_per_token": float(
                    prompt_tail["absolute_delta_nll_per_token"]["p90"]  # type: ignore[index]
                ),
                "per_prompt_p10_top1_agreement_to_source": float(
                    prompt_tail["top1_agreement_to_source"]["p10"]  # type: ignore[index]
                ),
            }
        )

    macro_metric_names = (
        "source_nll_per_token",
        "candidate_nll_per_token",
        "delta_nll_per_token",
        "absolute_delta_nll_per_token",
        "source_to_candidate_kl_per_token",
        "top1_agreement_to_source",
        "per_prompt_p90_absolute_delta_nll_per_token",
        "per_prompt_p10_top1_agreement_to_source",
    )
    macro = {
        name: math.fsum(float(row[name]) for row in families) / len(families)
        for name in macro_metric_names
    }

    adverse = {
        "absolute_delta_nll_per_token": max(
            float(row["absolute_delta_nll_per_token"]) for row in families
        ),
        "source_to_candidate_kl_per_token": max(
            float(row["source_to_candidate_kl_per_token"]) for row in families
        ),
        "top1_agreement_to_source": min(
            float(row["top1_agreement_to_source"]) for row in families
        ),
        "per_prompt_p90_absolute_delta_nll_per_token": max(
            float(row["per_prompt_p90_absolute_delta_nll_per_token"])
            for row in families
        ),
        "per_prompt_p10_top1_agreement_to_source": min(
            float(row["per_prompt_p10_top1_agreement_to_source"])
            for row in families
        ),
    }
    adverse_ids = {
        f"{name}_family_ids": [
            str(row["family_id"])
            for row in families
            if float(row[name]) == value
        ]
        for name, value in adverse.items()
    }
    return {
        "family_count": len(families),
        "weighting": {
            "families": "unweighted_macro",
            "examples_within_family": "supervised_token_weighted",
        },
        "families": families,
        "macro": macro,
        "worst": {**adverse, **adverse_ids},
    }


class SourceAuthoritativeShadowFidelityAccumulator:
    """Accumulate strict, family-aware shadow comparisons.

    The accumulator retains only scalar statistics.  It never returns, caches,
    or selects candidate logits for inference.
    """

    def __init__(
        self,
        expected_family_by_example: Mapping[str, str],
        *,
        gates: ShadowFidelityGates = ESTABLISHED_SHADOW_FIDELITY_GATES,
        vocab_chunk_size: int = 16_384,
    ) -> None:
        if not isinstance(expected_family_by_example, Mapping):
            raise TypeError("expected_family_by_example must be a mapping")
        if not expected_family_by_example:
            raise ValueError("expected_family_by_example cannot be empty")
        if not isinstance(gates, ShadowFidelityGates):
            raise TypeError("gates must be a ShadowFidelityGates instance")
        if (
            isinstance(vocab_chunk_size, bool)
            or not isinstance(vocab_chunk_size, int)
            or vocab_chunk_size <= 0
        ):
            raise ValueError("vocab_chunk_size must be a positive integer")

        manifest: dict[str, str] = {}
        for raw_example_id, raw_family_id in expected_family_by_example.items():
            example_id = _identifier(raw_example_id, name="manifest example_id")
            family_id = _identifier(raw_family_id, name="manifest family_id")
            manifest[example_id] = family_id
        self._expected_family_by_example = manifest
        self._gates = gates
        self._vocab_chunk_size = vocab_chunk_size
        self._rows: dict[str, _ExampleStatistics] = {}

    def add(self, example: ShadowFidelityInput) -> None:
        normalized = _coerce_example(example)
        example_id = _identifier(normalized.example_id, name="example_id")
        family_id = _identifier(normalized.family_id, name="family_id")
        if example_id not in self._expected_family_by_example:
            raise ValueError(f"undeclared shadow example: {example_id!r}")
        expected_family = self._expected_family_by_example[example_id]
        if family_id != expected_family:
            raise ValueError(
                f"shadow example {example_id!r} belongs to family "
                f"{expected_family!r}, not {family_id!r}"
            )
        if example_id in self._rows:
            raise ValueError(f"duplicate shadow example: {example_id!r}")
        self._rows[example_id] = _measure_example(
            normalized,
            vocab_chunk_size=self._vocab_chunk_size,
        )

    def finalize(self) -> dict[str, object]:
        missing = sorted(
            set(self._expected_family_by_example).difference(self._rows)
        )
        if missing:
            raise ValueError(
                "shadow fidelity evaluation is incomplete; missing examples: "
                + ", ".join(missing)
            )
        ordered_rows = [
            self._rows[example_id] for example_id in sorted(self._rows)
        ]
        aggregate = _aggregate_rows(ordered_rows)
        per_prompt = _prompt_tail(ordered_rows)
        family_summary = _family_summary(ordered_rows)
        gate_results = self._gates.evaluate(
            delta_nll_per_token=float(aggregate["delta_nll_per_token"]),
            top1_agreement_to_source=float(
                aggregate["top1_agreement_to_source"]
            ),
            source_to_candidate_kl_per_token=float(
                aggregate["source_to_candidate_kl_per_token"]
            ),
            per_prompt_p90_absolute_delta_nll_per_token=float(
                per_prompt["absolute_delta_nll_per_token"]["p90"]  # type: ignore[index]
            ),
            per_prompt_p10_top1_agreement_to_source=float(
                per_prompt["top1_agreement_to_source"]["p10"]  # type: ignore[index]
            ),
        )
        return {
            "schema": "fisher_graph.source_authoritative_shadow_fidelity",
            "format_version": 1,
            "semantics": {
                "execution_mode": "shadow",
                "authoritative_path": "source",
                "source_outputs_authoritative": True,
                "candidate_outputs_authoritative": False,
                "candidate_logits_used_for_metrics_only": True,
                "candidate_outputs_must_not_be_served": True,
            },
            "manifest": {
                "strict_example_membership": True,
                "strict_family_membership": True,
                "expected_examples": len(self._expected_family_by_example),
                "observed_examples": len(ordered_rows),
                "complete": True,
                "family_count": len(
                    set(self._expected_family_by_example.values())
                ),
            },
            "thresholds": self._gates.metadata(),
            "aggregate": aggregate,
            "per_prompt": per_prompt,
            "family_summary": family_summary,
            "gates": gate_results,
        }


def evaluate_source_authoritative_shadow(
    rows_or_examples: Iterable[ShadowFidelityInput],
    *,
    expected_family_by_example: Mapping[str, str],
    gates: ShadowFidelityGates = ESTABLISHED_SHADOW_FIDELITY_GATES,
    vocab_chunk_size: int = 16_384,
) -> dict[str, object]:
    """Evaluate a complete source-authoritative shadow manifest."""

    accumulator = SourceAuthoritativeShadowFidelityAccumulator(
        expected_family_by_example,
        gates=gates,
        vocab_chunk_size=vocab_chunk_size,
    )
    for example in rows_or_examples:
        accumulator.add(example)
    return accumulator.finalize()


__all__ = [
    "ESTABLISHED_SHADOW_FIDELITY_GATES",
    "ShadowFidelityExample",
    "ShadowFidelityGates",
    "SourceAuthoritativeShadowFidelityAccumulator",
    "evaluate_source_authoritative_shadow",
]
