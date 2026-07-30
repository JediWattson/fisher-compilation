"""Source-authoritative finite-NLL gate for the frozen H4 damping recipe.

This module deliberately does not load Gemma or know how an execution arm is
implemented.  A live runner may implement :class:`GemmaH4DampingArmProvider`;
tests and offline checks can supply already-collected logits directly.  Raw
logits are consumed transiently by the established shadow-fidelity
accumulator.  The returned report contains only JSON scalars, identifiers, and
cryptographic hashes.

The three arm identifiers are semantic ABI, not display labels:

``accepted_x4_only``
    The accepted X4 parent with no candidate H4 correction.
``matched_alpha0_with_B``
    The accepted X4 parent plus the matched lag-only H4 baseline ``B``.
``challenger_alpha0_5``
    The accepted X4 parent plus the frozen alpha=0.5 H4 correction.

The paired causal test is challenger versus matched alpha0.  The accepted-X4
arm remains a deployment-context control and must not be substituted for the
matched alpha0 baseline.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, Protocol

import torch
from torch import Tensor

from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    ShadowFidelityExample,
    SourceAuthoritativeShadowFidelityAccumulator,
)


DampingFiniteNLLArmId = Literal[
    "accepted_x4_only",
    "matched_alpha0_with_B",
    "challenger_alpha0_5",
]

ACCEPTED_X4_ONLY_ARM: DampingFiniteNLLArmId = "accepted_x4_only"
MATCHED_ALPHA0_ARM: DampingFiniteNLLArmId = "matched_alpha0_with_B"
CHALLENGER_ALPHA0_5_ARM: DampingFiniteNLLArmId = "challenger_alpha0_5"
DAMPING_FINITE_NLL_ARM_IDS: tuple[DampingFiniteNLLArmId, ...] = (
    ACCEPTED_X4_ONLY_ARM,
    MATCHED_ALPHA0_ARM,
    CHALLENGER_ALPHA0_5_ARM,
)

DAMPING_FINITE_NLL_ARM_SEMANTICS: Mapping[
    DampingFiniteNLLArmId,
    str,
] = MappingProxyType(
    {
        ACCEPTED_X4_ONLY_ARM: (
            "accepted_x4_parent_without_candidate_h4_correction"
        ),
        MATCHED_ALPHA0_ARM: (
            "accepted_x4_plus_matched_lag_only_h4_baseline_B"
        ),
        CHALLENGER_ALPHA0_5_ARM: (
            "accepted_x4_plus_frozen_alpha0_5_independent_h4_correction"
        ),
    }
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_h4_damping_finite_nll"
_FORMAT_VERSION = 1
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-finite-nll-report:v1\0"
)
_GRID_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-source-grid:v1\0"
)
_TENSOR_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-grid-tensor:v1\0"
)
_OBSERVATION_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-arm-observations:v1\0"
)
_SINGLE_OBSERVATION_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-scalar-observation:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_EXPECTED_EXAMPLE_COUNT = 16
_EXPECTED_FAMILY_COUNT = 8
_EXPECTED_EXAMPLES_PER_FAMILY = 2
_MINIMUM_FAMILY_WIN_COUNT = 6
_MACRO_IMPROVEMENT_MIN = 0.02
_WORST_FAMILY_IMPROVEMENT_MIN = -0.02
_SECONDARY_IMPROVEMENT_MIN = -0.02


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _canonical_arm_id(value: object) -> DampingFiniteNLLArmId:
    if value not in DAMPING_FINITE_NLL_ARM_IDS:
        raise ValueError(
            "finite-NLL arm_id must be exactly one of "
            f"{DAMPING_FINITE_NLL_ARM_IDS!r}"
        )
    return value  # type: ignore[return-value]


def _tensor_sha256(value: Tensor) -> str:
    """Hash exact tensor geometry and bytes without retaining the tensor."""

    detached = value.detach().to(device="cpu").contiguous()
    byte_view = detached.view(torch.uint8)
    header = _canonical_json_bytes(
        {
            "dtype": str(detached.dtype),
            "shape": tuple(detached.shape),
        }
    )
    return hashlib.sha256(
        _TENSOR_DOMAIN + header + byte_view.numpy().tobytes()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class GemmaH4DampingFiniteNLLObservation:
    """One immutable scalar measurement with no retained tensor payload."""

    example_id: str
    family_id: str
    supervised_tokens: int
    source_summed_nll: float
    candidate_summed_nll: float
    source_to_candidate_summed_kl: float
    top1_matches: int
    source_logits_sha256: str
    candidate_logits_sha256: str
    targets_sha256: str
    observation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.example_id, label="observation example_id")
        _identifier(self.family_id, label="observation family_id")
        if (
            type(self.supervised_tokens) is not int
            or self.supervised_tokens <= 0
        ):
            raise ValueError(
                "observation supervised_tokens must be a positive integer"
            )
        if (
            type(self.top1_matches) is not int
            or not 0 <= self.top1_matches <= self.supervised_tokens
        ):
            raise ValueError(
                "observation top1_matches must lie within supervised tokens"
            )
        for name in (
            "source_summed_nll",
            "candidate_summed_nll",
            "source_to_candidate_summed_kl",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"observation {name} must be finite and nonnegative"
                )
            object.__setattr__(self, name, float(value))
        for name in (
            "source_logits_sha256",
            "candidate_logits_sha256",
            "targets_sha256",
        ):
            _require_sha256(
                getattr(self, name),
                label=f"observation {name}",
            )
        object.__setattr__(
            self,
            "observation_sha256",
            _sha256(_SINGLE_OBSERVATION_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "supervised_tokens": self.supervised_tokens,
            "source_summed_nll": self.source_summed_nll,
            "candidate_summed_nll": self.candidate_summed_nll,
            "source_to_candidate_summed_kl": (
                self.source_to_candidate_summed_kl
            ),
            "top1_matches": self.top1_matches,
            "source_logits_sha256": self.source_logits_sha256,
            "candidate_logits_sha256": self.candidate_logits_sha256,
            "targets_sha256": self.targets_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "observation_sha256": self.observation_sha256,
        }


@dataclass(frozen=True, slots=True)
class GemmaH4DampingFiniteNLLArmInput:
    """One semantic arm backed by tensors or premeasured scalar observations."""

    arm_id: DampingFiniteNLLArmId
    semantic: str
    execution_receipt_sha256: str
    examples: tuple[ShadowFidelityExample, ...] = ()
    observations: tuple[GemmaH4DampingFiniteNLLObservation, ...] = ()

    def __post_init__(self) -> None:
        arm_id = _canonical_arm_id(self.arm_id)
        object.__setattr__(self, "arm_id", arm_id)
        expected_semantic = DAMPING_FINITE_NLL_ARM_SEMANTICS[arm_id]
        if self.semantic != expected_semantic:
            raise ValueError(
                f"{arm_id} semantic must be {expected_semantic!r}"
            )
        _require_sha256(
            self.execution_receipt_sha256,
            label=f"{arm_id} execution receipt",
        )
        if type(self.examples) is not tuple or any(
                not isinstance(example, ShadowFidelityExample)
                for example in self.examples
        ):
            raise ValueError(
                f"{arm_id} examples must be a tuple of "
                "ShadowFidelityExample values"
            )
        if type(self.observations) is not tuple or any(
            not isinstance(
                observation,
                GemmaH4DampingFiniteNLLObservation,
            )
            for observation in self.observations
        ):
            raise ValueError(
                f"{arm_id} observations must be a tuple of "
                "GemmaH4DampingFiniteNLLObservation values"
            )
        if bool(self.examples) is bool(self.observations):
            raise ValueError(
                f"{arm_id} must provide exactly one of examples or "
                "observations"
            )
        selected = self.examples if self.examples else self.observations
        if len(selected) != _EXPECTED_EXAMPLE_COUNT:
            raise ValueError(
                f"{arm_id} must provide exactly sixteen examples or "
                "observations"
            )
        if self.observations:
            example_ids = tuple(
                observation.example_id for observation in self.observations
            )
            family_counts = Counter(
                observation.family_id for observation in self.observations
            )
            if (
                len(set(example_ids)) != _EXPECTED_EXAMPLE_COUNT
                or len(family_counts) != _EXPECTED_FAMILY_COUNT
                or set(family_counts.values())
                != {_EXPECTED_EXAMPLES_PER_FAMILY}
            ):
                raise ValueError(
                    f"{arm_id} observations must contain sixteen unique "
                    "examples and two examples in each of eight families"
                )


class GemmaH4DampingArmProvider(Protocol):
    """Live-model boundary: collect one strictly named arm at a time."""

    def collect(
        self,
        arm_id: DampingFiniteNLLArmId,
    ) -> GemmaH4DampingFiniteNLLArmInput: ...


def _canonical_manifest(
    value: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("expected_family_by_example must be nonempty")
    result: dict[str, str] = {}
    for example_id, family_id in value.items():
        parsed_example = _identifier(example_id, label="manifest example_id")
        parsed_family = _identifier(family_id, label="manifest family_id")
        result[parsed_example] = parsed_family
    if len(result) != _EXPECTED_EXAMPLE_COUNT:
        raise ValueError(
            "finite-NLL gate requires exactly sixteen source examples"
        )
    family_ids = set(result.values())
    if len(family_ids) != _EXPECTED_FAMILY_COUNT:
        raise ValueError(
            "finite-NLL gate requires exactly eight source families"
        )
    if Counter(result.values()) != Counter(
        {
            family_id: _EXPECTED_EXAMPLES_PER_FAMILY
            for family_id in family_ids
        }
    ):
        raise ValueError(
            "finite-NLL gate requires exactly two examples per source family"
        )
    return result


def _source_grid(
    examples: tuple[ShadowFidelityExample, ...],
) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for example in examples:
        source = example.source_logits.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        targets = example.targets.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        token_indices = torch.arange(targets.shape[0])
        source_summed_nll = float(
            (
                torch.logsumexp(source, dim=-1)
                - source[token_indices, targets]
            )
            .sum()
            .item()
        )
        rows.append(
            (
                example.example_id,
                example.family_id,
                int(targets.numel()),
                source_summed_nll,
                _tensor_sha256(example.source_logits),
                _tensor_sha256(example.targets),
            )
        )
    return tuple(sorted(rows, key=lambda row: str(row[0])))


_OBSERVATION_FIELDS = frozenset(
    {
        "example_id",
        "family_id",
        "supervised_tokens",
        "source_summed_nll",
        "candidate_summed_nll",
        "source_to_candidate_summed_kl",
        "top1_matches",
        "source_logits_sha256",
        "candidate_logits_sha256",
        "targets_sha256",
        "observation_sha256",
    }
)


def _arm_observations(
    examples: tuple[ShadowFidelityExample, ...],
    measured_rows: Mapping[str, object],
) -> tuple[GemmaH4DampingFiniteNLLObservation, ...]:
    """Bind exact tensor identities to the scalar rows already measured."""

    observations: list[GemmaH4DampingFiniteNLLObservation] = []
    for example in sorted(examples, key=lambda value: value.example_id):
        statistics = measured_rows.get(example.example_id)
        if statistics is None:
            raise RuntimeError("shadow accumulator omitted a measured example")
        observations.append(
            GemmaH4DampingFiniteNLLObservation(
                example_id=example.example_id,
                family_id=example.family_id,
                supervised_tokens=int(
                    getattr(statistics, "supervised_tokens")
                ),
                source_summed_nll=float(
                    getattr(statistics, "source_summed_nll")
                ),
                candidate_summed_nll=float(
                    getattr(statistics, "candidate_summed_nll")
                ),
                source_to_candidate_summed_kl=float(
                    getattr(
                        statistics,
                        "source_to_candidate_summed_kl",
                    )
                ),
                top1_matches=int(getattr(statistics, "top1_matches")),
                source_logits_sha256=_tensor_sha256(
                    example.source_logits
                ),
                candidate_logits_sha256=_tensor_sha256(
                    example.candidate_logits
                ),
                targets_sha256=_tensor_sha256(example.targets),
            )
        )
    return tuple(observations)


def measure_gemma_h4_damping_finite_nll_observation(
    example: ShadowFidelityExample,
    *,
    vocab_chunk_size: int = 16_384,
) -> GemmaH4DampingFiniteNLLObservation:
    """Measure one example now and return only immutable scalar/hash state."""

    if not isinstance(example, ShadowFidelityExample):
        raise TypeError("example must be a ShadowFidelityExample")
    accumulator = SourceAuthoritativeShadowFidelityAccumulator(
        {example.example_id: example.family_id},
        gates=ESTABLISHED_SHADOW_FIDELITY_GATES,
        vocab_chunk_size=vocab_chunk_size,
    )
    accumulator.add(example)
    accumulator.finalize()
    measured_rows = getattr(accumulator, "_rows", None)
    if not isinstance(measured_rows, Mapping):
        raise RuntimeError(
            "shadow accumulator did not expose measured scalar rows"
        )
    return _arm_observations((example,), measured_rows)[0]


def _canonical_observation_payloads(
    observations: tuple[GemmaH4DampingFiniteNLLObservation, ...],
    *,
    manifest: Mapping[str, str],
    label: str,
) -> list[Mapping[str, object]]:
    observed_family_by_example = {
        observation.example_id: observation.family_id
        for observation in observations
    }
    if (
        len(observed_family_by_example) != len(observations)
        or observed_family_by_example != dict(manifest)
    ):
        raise ValueError(
            f"{label} observation membership differs from the source manifest"
        )
    return [
        observation.to_dict()
        for observation in sorted(
            observations,
            key=lambda value: value.example_id,
        )
    ]


def _nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("finite-NLL percentile values cannot be empty")
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(math.ceil(fraction * len(ordered))) - 1,
        ),
    )
    return ordered[index]


def _lower_nearest_rank(values: list[float], fraction: float) -> float:
    return -_nearest_rank([-value for value in values], 1.0 - fraction)


def _observation_delta_nll_per_token(
    observation: Mapping[str, object],
) -> float:
    return (
        float(observation["candidate_summed_nll"])
        - float(observation["source_summed_nll"])
    ) / int(observation["supervised_tokens"])


def _aggregate_observations(
    observations: list[Mapping[str, object]],
) -> dict[str, int | float]:
    ordered = sorted(
        observations,
        key=lambda observation: str(observation["example_id"]),
    )
    if not ordered:
        raise ValueError("finite-NLL observations cannot be empty")
    supervised_tokens = sum(
        int(observation["supervised_tokens"]) for observation in ordered
    )
    source_summed_nll = math.fsum(
        float(observation["source_summed_nll"]) for observation in ordered
    )
    candidate_summed_nll = math.fsum(
        float(observation["candidate_summed_nll"]) for observation in ordered
    )
    source_to_candidate_summed_kl = max(
        0.0,
        math.fsum(
            float(observation["source_to_candidate_summed_kl"])
            for observation in ordered
        ),
    )
    top1_matches = sum(
        int(observation["top1_matches"]) for observation in ordered
    )
    return {
        "example_count": len(ordered),
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


def _prompt_tail_from_observations(
    observations: list[Mapping[str, object]],
) -> dict[str, object]:
    absolute_delta = [
        abs(_observation_delta_nll_per_token(observation))
        for observation in observations
    ]
    top1 = [
        int(observation["top1_matches"])
        / int(observation["supervised_tokens"])
        for observation in observations
    ]
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


def _family_summary_from_observations(
    observations: list[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for observation in observations:
        grouped.setdefault(str(observation["family_id"]), []).append(
            observation
        )

    families: list[dict[str, object]] = []
    for family_id in sorted(grouped):
        family_observations = grouped[family_id]
        aggregate = _aggregate_observations(family_observations)
        prompt_tail = _prompt_tail_from_observations(family_observations)
        prompt_absolute_nll = _mapping(
            prompt_tail["absolute_delta_nll_per_token"],
            label=f"{family_id} prompt absolute NLL",
        )
        prompt_top1 = _mapping(
            prompt_tail["top1_agreement_to_source"],
            label=f"{family_id} prompt top1",
        )
        families.append(
            {
                "family_id": family_id,
                **aggregate,
                "absolute_delta_nll_per_token": abs(
                    float(aggregate["delta_nll_per_token"])
                ),
                "per_prompt_p90_absolute_delta_nll_per_token": float(
                    prompt_absolute_nll["p90"]
                ),
                "per_prompt_p10_top1_agreement_to_source": float(
                    prompt_top1["p10"]
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


def _fidelity_from_observations(
    observations: list[Mapping[str, object]],
) -> dict[str, object]:
    aggregate = _aggregate_observations(observations)
    per_prompt = _prompt_tail_from_observations(observations)
    family_summary = _family_summary_from_observations(observations)
    gates = ESTABLISHED_SHADOW_FIDELITY_GATES.evaluate(
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
            "expected_examples": len(observations),
            "observed_examples": len(observations),
            "complete": True,
            "family_count": len(
                {
                    str(observation["family_id"])
                    for observation in observations
                }
            ),
        },
        "thresholds": ESTABLISHED_SHADOW_FIDELITY_GATES.metadata(),
        "aggregate": aggregate,
        "per_prompt": per_prompt,
        "family_summary": family_summary,
        "gates": gates,
    }


def _source_grid_from_observations(
    observations: list[Mapping[str, object]],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            str(observation["example_id"]),
            str(observation["family_id"]),
            int(observation["supervised_tokens"]),
            float(observation["source_summed_nll"]),
            str(observation["source_logits_sha256"]),
            str(observation["targets_sha256"]),
        )
        for observation in sorted(
            observations,
            key=lambda value: str(value["example_id"]),
        )
    )


def _observation_receipt_sha256(
    *,
    arm_id: DampingFiniteNLLArmId,
    execution_receipt_sha256: str,
    observations: list[Mapping[str, object]],
) -> str:
    return _sha256(
        _OBSERVATION_DOMAIN,
        {
            "arm_id": arm_id,
            "semantic": DAMPING_FINITE_NLL_ARM_SEMANTICS[arm_id],
            "execution_receipt_sha256": execution_receipt_sha256,
            "observations": observations,
        },
    )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _relative_improvement(new: float, baseline: float) -> float:
    if baseline < 0.0 or new < 0.0:
        raise ValueError("paired error metrics must be nonnegative")
    if baseline == 0.0:
        return 0.0 if new == 0.0 else -1.0
    result = 1.0 - new / baseline
    if not math.isfinite(result):
        raise ValueError("relative improvement became nonfinite")
    return result


def _family_rows(
    report: Mapping[str, object],
    *,
    label: str,
) -> dict[str, Mapping[str, object]]:
    summary = _mapping(report.get("family_summary"), label=f"{label} family")
    raw = summary.get("families")
    if not isinstance(raw, list):
        raise TypeError(f"{label} family rows must be a list")
    result: dict[str, Mapping[str, object]] = {}
    for value in raw:
        row = _mapping(value, label=f"{label} family row")
        family_id = _identifier(
            row.get("family_id"),
            label=f"{label} family_id",
        )
        if family_id in result:
            raise ValueError(f"{label} contains duplicate family rows")
        result[family_id] = row
    if len(result) != _EXPECTED_FAMILY_COUNT:
        raise ValueError(f"{label} must contain exactly eight families")
    return result


def _metric(
    value: Mapping[str, object],
    name: str,
    *,
    label: str,
) -> float:
    return _finite_float(value.get(name), label=f"{label} {name}")


def _family_mean_prompt_absolute_delta_nll(
    observations: list[Mapping[str, object]],
    *,
    label: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for observation in observations:
        family_id = _identifier(
            observation.get("family_id"),
            label=f"{label} observation family_id",
        )
        grouped.setdefault(family_id, []).append(
            abs(_observation_delta_nll_per_token(observation))
        )
    if (
        len(grouped) != _EXPECTED_FAMILY_COUNT
        or any(
            len(values) != _EXPECTED_EXAMPLES_PER_FAMILY
            for values in grouped.values()
        )
    ):
        raise ValueError(
            f"{label} observations must contain two prompts in each of "
            "eight families"
        )
    return {
        family_id: math.fsum(values) / len(values)
        for family_id, values in grouped.items()
    }


def _paired_comparison(
    baseline: Mapping[str, object],
    challenger: Mapping[str, object],
    *,
    baseline_observations: list[Mapping[str, object]],
    challenger_observations: list[Mapping[str, object]],
) -> dict[str, object]:
    baseline_families = _family_rows(baseline, label="matched alpha0")
    challenger_families = _family_rows(challenger, label="challenger")
    if set(baseline_families) != set(challenger_families):
        raise ValueError("paired finite-NLL family grids differ")
    baseline_prompt_errors = _family_mean_prompt_absolute_delta_nll(
        baseline_observations,
        label="matched alpha0",
    )
    challenger_prompt_errors = _family_mean_prompt_absolute_delta_nll(
        challenger_observations,
        label="challenger",
    )
    if (
        set(baseline_prompt_errors) != set(baseline_families)
        or set(challenger_prompt_errors) != set(challenger_families)
    ):
        raise ValueError("paired finite-NLL observation families differ")

    family_rows: list[dict[str, object]] = []
    for family_id in sorted(baseline_families):
        baseline_error = baseline_prompt_errors[family_id]
        challenger_error = challenger_prompt_errors[family_id]
        improvement = _relative_improvement(
            challenger_error,
            baseline_error,
        )
        family_rows.append(
            {
                "family_id": family_id,
                "error_metric": (
                    "mean_per_prompt_absolute_delta_nll_per_token"
                ),
                "matched_alpha0_mean_prompt_absolute_delta_nll_per_token": (
                    baseline_error
                ),
                "challenger_mean_prompt_absolute_delta_nll_per_token": (
                    challenger_error
                ),
                "relative_improvement": improvement,
                "strict_win": challenger_error < baseline_error,
            }
        )

    baseline_summary = _mapping(
        baseline["family_summary"],
        label="matched alpha0 family summary",
    )
    challenger_summary = _mapping(
        challenger["family_summary"],
        label="challenger family summary",
    )
    baseline_macro = _mapping(
        baseline_summary.get("macro"),
        label="matched alpha0 family macro",
    )
    challenger_macro = _mapping(
        challenger_summary.get("macro"),
        label="challenger family macro",
    )
    baseline_prompt = _mapping(
        baseline.get("per_prompt"),
        label="matched alpha0 prompt summary",
    )
    challenger_prompt = _mapping(
        challenger.get("per_prompt"),
        label="challenger prompt summary",
    )

    baseline_macro_error = math.fsum(
        baseline_prompt_errors.values()
    ) / _EXPECTED_FAMILY_COUNT
    challenger_macro_error = math.fsum(
        challenger_prompt_errors.values()
    ) / _EXPECTED_FAMILY_COUNT
    macro_improvement = _relative_improvement(
        challenger_macro_error,
        baseline_macro_error,
    )
    family_win_count = sum(bool(row["strict_win"]) for row in family_rows)
    worst_family_improvement = min(
        float(row["relative_improvement"]) for row in family_rows
    )

    def secondary(
        name: str,
        baseline_value: float,
        challenger_value: float,
    ) -> dict[str, object]:
        improvement = _relative_improvement(
            challenger_value,
            baseline_value,
        )
        return {
            "metric": name,
            "matched_alpha0": baseline_value,
            "challenger": challenger_value,
            "relative_improvement": improvement,
            "regression_at_most_2pct": (
                improvement >= _SECONDARY_IMPROVEMENT_MIN
            ),
        }

    baseline_p90 = _mapping(
        baseline_prompt.get("absolute_delta_nll_per_token"),
        label="matched alpha0 prompt absolute NLL",
    )
    challenger_p90 = _mapping(
        challenger_prompt.get("absolute_delta_nll_per_token"),
        label="challenger prompt absolute NLL",
    )
    baseline_p10 = _mapping(
        baseline_prompt.get("top1_agreement_to_source"),
        label="matched alpha0 prompt top1",
    )
    challenger_p10 = _mapping(
        challenger_prompt.get("top1_agreement_to_source"),
        label="challenger prompt top1",
    )
    secondary_rows = (
        secondary(
            "family_macro_source_to_candidate_kl_per_token",
            _metric(
                baseline_macro,
                "source_to_candidate_kl_per_token",
                label="matched alpha0 family macro",
            ),
            _metric(
                challenger_macro,
                "source_to_candidate_kl_per_token",
                label="challenger family macro",
            ),
        ),
        secondary(
            "family_macro_top1_disagreement_to_source",
            1.0
            - _metric(
                baseline_macro,
                "top1_agreement_to_source",
                label="matched alpha0 family macro",
            ),
            1.0
            - _metric(
                challenger_macro,
                "top1_agreement_to_source",
                label="challenger family macro",
            ),
        ),
        secondary(
            "per_prompt_p90_absolute_delta_nll_per_token",
            _metric(baseline_p90, "p90", label="matched alpha0 prompt"),
            _metric(challenger_p90, "p90", label="challenger prompt"),
        ),
        secondary(
            "per_prompt_p10_top1_disagreement_to_source",
            1.0 - _metric(baseline_p10, "p10", label="matched alpha0 prompt"),
            1.0 - _metric(challenger_p10, "p10", label="challenger prompt"),
        ),
    )
    gates = {
        "family_macro_mean_prompt_absolute_delta_nll_improvement_at_least_2pct": (
            macro_improvement >= _MACRO_IMPROVEMENT_MIN
        ),
        "strict_family_win_count_at_least_6_of_8": (
            family_win_count >= _MINIMUM_FAMILY_WIN_COUNT
        ),
        "worst_family_improvement_at_least_minus_2pct": (
            worst_family_improvement
            >= _WORST_FAMILY_IMPROVEMENT_MIN
        ),
        "family_macro_kl_regression_at_most_2pct": bool(
            secondary_rows[0]["regression_at_most_2pct"]
        ),
        "family_macro_top1_disagreement_regression_at_most_2pct": bool(
            secondary_rows[1]["regression_at_most_2pct"]
        ),
        "prompt_p90_absolute_delta_nll_regression_at_most_2pct": bool(
            secondary_rows[2]["regression_at_most_2pct"]
        ),
        "prompt_p10_top1_disagreement_regression_at_most_2pct": bool(
            secondary_rows[3]["regression_at_most_2pct"]
        ),
    }
    gates["passed"] = all(gates.values())
    return {
        "baseline_arm_id": MATCHED_ALPHA0_ARM,
        "challenger_arm_id": CHALLENGER_ALPHA0_5_ARM,
        "family_count": len(family_rows),
        "family_rows": family_rows,
        "family_macro_mean_prompt_absolute_delta_nll_per_token": {
            "matched_alpha0": baseline_macro_error,
            "challenger": challenger_macro_error,
            "relative_improvement": macro_improvement,
        },
        "strict_family_win_count": family_win_count,
        "minimum_strict_family_win_count": (
            _MINIMUM_FAMILY_WIN_COUNT
        ),
        "worst_family_relative_improvement": (
            worst_family_improvement
        ),
        "secondary_metrics": list(secondary_rows),
        "gates": gates,
    }


def _assert_scalar_hash_only(value: object, *, path: str = "report") -> None:
    if isinstance(value, Tensor):
        raise TypeError(f"{path} exposes a tensor payload")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite scalar")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            _assert_scalar_hash_only(item, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_scalar_hash_only(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported payload {type(value)!r}")


def _strict_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _parse_report_observations(
    value: object,
    *,
    label: str,
) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or len(value) != _EXPECTED_EXAMPLE_COUNT:
        raise ValueError(
            f"{label} must contain exactly sixteen scalar observations"
        )
    observations: list[Mapping[str, object]] = []
    example_ids: list[str] = []
    for index, raw in enumerate(value):
        observation = _mapping(raw, label=f"{label}[{index}]")
        if set(observation) != _OBSERVATION_FIELDS:
            raise ValueError(f"{label}[{index}] fields differ")
        example_id = _identifier(
            observation["example_id"],
            label=f"{label}[{index}] example_id",
        )
        family_id = _identifier(
            observation["family_id"],
            label=f"{label}[{index}] family_id",
        )
        supervised_tokens = _strict_int(
            observation["supervised_tokens"],
            label=f"{label}[{index}] supervised_tokens",
            minimum=1,
        )
        source_summed_nll = _finite_float(
            observation["source_summed_nll"],
            label=f"{label}[{index}] source_summed_nll",
        )
        candidate_summed_nll = _finite_float(
            observation["candidate_summed_nll"],
            label=f"{label}[{index}] candidate_summed_nll",
        )
        source_to_candidate_summed_kl = _finite_float(
            observation["source_to_candidate_summed_kl"],
            label=f"{label}[{index}] source_to_candidate_summed_kl",
        )
        if (
            source_summed_nll < 0.0
            or candidate_summed_nll < 0.0
            or source_to_candidate_summed_kl < 0.0
        ):
            raise ValueError(f"{label}[{index}] losses must be nonnegative")
        top1_matches = _strict_int(
            observation["top1_matches"],
            label=f"{label}[{index}] top1_matches",
        )
        if top1_matches > supervised_tokens:
            raise ValueError(
                f"{label}[{index}] top1_matches exceeds supervised tokens"
            )
        measured = GemmaH4DampingFiniteNLLObservation(
            example_id=example_id,
            family_id=family_id,
            supervised_tokens=supervised_tokens,
            source_summed_nll=source_summed_nll,
            candidate_summed_nll=candidate_summed_nll,
            source_to_candidate_summed_kl=(
                source_to_candidate_summed_kl
            ),
            top1_matches=top1_matches,
            source_logits_sha256=_require_sha256(
                observation["source_logits_sha256"],
                label=f"{label}[{index}] source logits",
            ),
            candidate_logits_sha256=_require_sha256(
                observation["candidate_logits_sha256"],
                label=f"{label}[{index}] candidate logits",
            ),
            targets_sha256=_require_sha256(
                observation["targets_sha256"],
                label=f"{label}[{index}] targets",
            ),
        )
        if _require_sha256(
            observation["observation_sha256"],
            label=f"{label}[{index}] observation",
        ) != measured.observation_sha256:
            raise ValueError(f"{label}[{index}] observation hash differs")
        normalized = measured.to_dict()
        observations.append(normalized)
        example_ids.append(example_id)
    if example_ids != sorted(set(example_ids)):
        raise ValueError(f"{label} must be sorted by unique example_id")
    family_counts = Counter(
        str(observation["family_id"]) for observation in observations
    )
    if (
        len(family_counts) != _EXPECTED_FAMILY_COUNT
        or set(family_counts.values()) != {_EXPECTED_EXAMPLES_PER_FAMILY}
    ):
        raise ValueError(
            f"{label} must contain exactly two observations in each of "
            "eight families"
        )
    return observations


def validate_gemma_h4_damping_finite_nll_report(
    report: Mapping[str, object],
) -> None:
    """Reauthenticate and replay every scalar-derived report claim."""

    root = _mapping(report, label="finite-NLL report")
    expected_keys = {
        "schema",
        "format_version",
        "semantics",
        "thresholds",
        "source_grid",
        "arms",
        "paired_comparison",
        "qualification",
        "safety",
        "report_sha256",
    }
    if set(root) != expected_keys:
        raise ValueError("finite-NLL report keys differ")
    _assert_scalar_hash_only(root)
    if root["schema"] != _SCHEMA or root["format_version"] != _FORMAT_VERSION:
        raise ValueError("finite-NLL report schema or version differs")
    observed_sha256 = _require_sha256(
        root["report_sha256"],
        label="finite-NLL report",
    )
    payload = dict(root)
    payload.pop("report_sha256")
    if _sha256(_REPORT_DOMAIN, payload) != observed_sha256:
        raise ValueError("finite-NLL report hash differs")

    semantics = _mapping(root["semantics"], label="finite-NLL semantics")
    expected_semantics = {
        "execution_mode": "matched_source_authoritative_shadow_forwards",
        "source_outputs_authoritative": True,
        "candidate_outputs_metrics_only": True,
        "arm_ids": DAMPING_FINITE_NLL_ARM_IDS,
        "arm_semantics": dict(DAMPING_FINITE_NLL_ARM_SEMANTICS),
        "deployment_context_arm_id": ACCEPTED_X4_ONLY_ARM,
        "paired_baseline_arm_id": MATCHED_ALPHA0_ARM,
        "paired_challenger_arm_id": CHALLENGER_ALPHA0_5_ARM,
        "accepted_x4_is_not_the_alpha0_baseline": True,
    }
    if _canonical_json_bytes(semantics) != _canonical_json_bytes(
        expected_semantics
    ):
        raise ValueError("finite-NLL semantic ABI differs")

    thresholds = _mapping(
        root["thresholds"],
        label="finite-NLL thresholds",
    )
    expected_thresholds = {
        "absolute": ESTABLISHED_SHADOW_FIDELITY_GATES.metadata(),
        "paired": {
            "family_count": _EXPECTED_FAMILY_COUNT,
            (
                "family_macro_mean_prompt_absolute_delta_nll_"
                "relative_improvement_min"
            ): _MACRO_IMPROVEMENT_MIN,
            "minimum_strict_family_win_count": (
                _MINIMUM_FAMILY_WIN_COUNT
            ),
            "worst_family_relative_improvement_min": (
                _WORST_FAMILY_IMPROVEMENT_MIN
            ),
            "secondary_metric_relative_improvement_min": (
                _SECONDARY_IMPROVEMENT_MIN
            ),
        },
    }
    if thresholds != expected_thresholds:
        raise ValueError("finite-NLL gate thresholds differ")

    expected_safety = {
        "source_authoritative": True,
        "candidate_logits_metrics_only": True,
        "raw_logits_in_report": False,
        "targets_in_report": False,
        "tensor_payload_exposed": False,
        "model_load_required": False,
        "per_example_scalar_observations_in_report": True,
        "candidate_observations_hash_bound": True,
    }
    if _mapping(root["safety"], label="finite-NLL safety") != expected_safety:
        raise ValueError("finite-NLL safety claims differ")

    arms = _mapping(root["arms"], label="finite-NLL arms")
    if set(arms) != set(DAMPING_FINITE_NLL_ARM_IDS):
        raise ValueError("finite-NLL report arm IDs differ")
    execution_receipts: set[str] = set()
    observation_receipts: set[str] = set()
    observations_by_arm: dict[
        DampingFiniteNLLArmId,
        list[Mapping[str, object]],
    ] = {}
    fidelity_by_arm: dict[
        DampingFiniteNLLArmId,
        Mapping[str, object],
    ] = {}
    source_grids: dict[
        DampingFiniteNLLArmId,
        tuple[tuple[object, ...], ...],
    ] = {}
    expected_arm_keys = {
        "arm_id",
        "semantic",
        "execution_receipt_sha256",
        "observation_receipt_sha256",
        "observations",
        "fidelity",
    }
    for arm_id in DAMPING_FINITE_NLL_ARM_IDS:
        arm = _mapping(arms[arm_id], label=f"{arm_id} report")
        if set(arm) != expected_arm_keys:
            raise ValueError(f"{arm_id} report fields differ")
        if (
            arm["arm_id"] != arm_id
            or arm["semantic"] != DAMPING_FINITE_NLL_ARM_SEMANTICS[arm_id]
        ):
            raise ValueError(f"{arm_id} report semantic differs")
        execution_receipt = _require_sha256(
            arm["execution_receipt_sha256"],
            label=f"{arm_id} execution receipt",
        )
        if execution_receipt in execution_receipts:
            raise ValueError("finite-NLL arm execution receipts must differ")
        execution_receipts.add(execution_receipt)
        observations = _parse_report_observations(
            arm["observations"],
            label=f"{arm_id} observations",
        )
        observations_by_arm[arm_id] = observations
        expected_observation_receipt = _observation_receipt_sha256(
            arm_id=arm_id,
            execution_receipt_sha256=execution_receipt,
            observations=observations,
        )
        observation_receipt = _require_sha256(
            arm["observation_receipt_sha256"],
            label=f"{arm_id} observation receipt",
        )
        if observation_receipt != expected_observation_receipt:
            raise ValueError(f"{arm_id} observation receipt differs")
        if observation_receipt in observation_receipts:
            raise ValueError("finite-NLL observation receipts must differ")
        observation_receipts.add(observation_receipt)
        fidelity = _mapping(arm["fidelity"], label=f"{arm_id} fidelity")
        expected_fidelity = _fidelity_from_observations(observations)
        if fidelity != expected_fidelity:
            raise ValueError(f"{arm_id} fidelity differs from observations")
        fidelity_by_arm[arm_id] = fidelity
        source_grids[arm_id] = _source_grid_from_observations(observations)

    reference_grid = source_grids[ACCEPTED_X4_ONLY_ARM]
    if any(
        source_grids[arm_id] != reference_grid
        for arm_id in DAMPING_FINITE_NLL_ARM_IDS[1:]
    ):
        raise ValueError("finite-NLL source grids differ across arms")
    reference_aggregate = _aggregate_observations(
        observations_by_arm[ACCEPTED_X4_ONLY_ARM]
    )
    expected_source_grid = {
        "source_grid_sha256": _sha256(_GRID_DOMAIN, reference_grid),
        "identical_across_arms": True,
        "example_count": _EXPECTED_EXAMPLE_COUNT,
        "family_count": _EXPECTED_FAMILY_COUNT,
        "examples_per_family": _EXPECTED_EXAMPLES_PER_FAMILY,
        "supervised_tokens": int(reference_aggregate["supervised_tokens"]),
        "source_summed_nll": float(
            reference_aggregate["source_summed_nll"]
        ),
        "source_nll_per_token": float(
            reference_aggregate["source_nll_per_token"]
        ),
    }
    if _mapping(
        root["source_grid"],
        label="finite-NLL source grid",
    ) != expected_source_grid:
        raise ValueError("finite-NLL source-grid metadata differs")

    expected_paired = _paired_comparison(
        fidelity_by_arm[MATCHED_ALPHA0_ARM],
        fidelity_by_arm[CHALLENGER_ALPHA0_5_ARM],
        baseline_observations=observations_by_arm[MATCHED_ALPHA0_ARM],
        challenger_observations=observations_by_arm[
            CHALLENGER_ALPHA0_5_ARM
        ],
    )
    if _mapping(
        root["paired_comparison"],
        label="paired finite-NLL comparison",
    ) != expected_paired:
        raise ValueError("paired finite-NLL comparison differs")
    paired_passed = (
        _mapping(
            expected_paired["gates"],
            label="replayed paired finite-NLL gates",
        )["passed"]
        is True
    )
    challenger_absolute_passed = (
        _mapping(
            fidelity_by_arm[CHALLENGER_ALPHA0_5_ARM]["gates"],
            label="replayed challenger fidelity gates",
        )["passed"]
        is True
    )
    expected_qualification = {
        "paired_gate_passed": paired_passed,
        "challenger_absolute_gate_passed": challenger_absolute_passed,
        "qualified": paired_passed and challenger_absolute_passed,
        "relative_pass_absolute_fail_qualifies": False,
    }
    if _mapping(
        root["qualification"],
        label="finite-NLL qualification",
    ) != expected_qualification:
        raise ValueError("finite-NLL qualification logic differs")


def evaluate_gemma_h4_damping_finite_nll(
    arms: Mapping[
        DampingFiniteNLLArmId,
        GemmaH4DampingFiniteNLLArmInput,
    ],
    *,
    expected_family_by_example: Mapping[str, str],
    vocab_chunk_size: int = 16_384,
) -> dict[str, object]:
    """Evaluate the strict three-arm finite-NLL experiment.

    The function consumes the supplied logits immediately.  It returns the
    complete scalar reports from ``SourceAuthoritativeShadowFidelityAccumulator``
    and a hash-bound paired qualification decision.
    """

    if not isinstance(arms, Mapping) or set(arms) != set(
        DAMPING_FINITE_NLL_ARM_IDS
    ):
        raise ValueError(
            "arms must contain exactly the three frozen finite-NLL arm IDs"
        )
    manifest = _canonical_manifest(expected_family_by_example)
    materialized: dict[
        DampingFiniteNLLArmId,
        GemmaH4DampingFiniteNLLArmInput,
    ] = {}
    receipts: set[str] = set()
    reports: dict[DampingFiniteNLLArmId, Mapping[str, object]] = {}
    observations_by_arm: dict[
        DampingFiniteNLLArmId,
        list[Mapping[str, object]],
    ] = {}
    source_grids: dict[
        DampingFiniteNLLArmId,
        tuple[tuple[object, ...], ...],
    ] = {}

    for arm_id in DAMPING_FINITE_NLL_ARM_IDS:
        value = arms[arm_id]
        if (
            not isinstance(value, GemmaH4DampingFiniteNLLArmInput)
            or value.arm_id != arm_id
        ):
            raise ValueError(f"{arm_id} input is mislabeled")
        if value.execution_receipt_sha256 in receipts:
            raise ValueError("finite-NLL arm execution receipts must differ")
        receipts.add(value.execution_receipt_sha256)
        materialized[arm_id] = value
        if value.examples:
            accumulator = SourceAuthoritativeShadowFidelityAccumulator(
                manifest,
                gates=ESTABLISHED_SHADOW_FIDELITY_GATES,
                vocab_chunk_size=vocab_chunk_size,
            )
            for example in value.examples:
                accumulator.add(example)
            reports[arm_id] = accumulator.finalize()
            measured_rows = getattr(accumulator, "_rows", None)
            if not isinstance(measured_rows, Mapping):
                raise RuntimeError(
                    "shadow accumulator did not expose measured scalar rows"
                )
            measured = _arm_observations(value.examples, measured_rows)
            observations = _canonical_observation_payloads(
                measured,
                manifest=manifest,
                label=arm_id,
            )
            direct_source_grid = _source_grid(value.examples)
        else:
            observations = _canonical_observation_payloads(
                value.observations,
                manifest=manifest,
                label=arm_id,
            )
            reports[arm_id] = _fidelity_from_observations(observations)
            direct_source_grid = _source_grid_from_observations(observations)
        derived_fidelity = _fidelity_from_observations(observations)
        if reports[arm_id] != derived_fidelity:
            raise RuntimeError(
                f"{arm_id} scalar observation replay differs from fidelity"
            )
        observations_by_arm[arm_id] = observations
        source_grids[arm_id] = direct_source_grid
        if (
            _source_grid_from_observations(observations)
            != source_grids[arm_id]
        ):
            raise RuntimeError(
                f"{arm_id} scalar observations differ from source grid"
            )

    reference_grid = source_grids[ACCEPTED_X4_ONLY_ARM]
    for arm_id in DAMPING_FINITE_NLL_ARM_IDS[1:]:
        if source_grids[arm_id] != reference_grid:
            raise ValueError(
                f"{arm_id} source NLL/token/example/family grid differs"
            )

    reference_report = reports[ACCEPTED_X4_ONLY_ARM]
    reference_aggregate = _mapping(
        reference_report["aggregate"],
        label="accepted X4 aggregate",
    )
    for arm_id in DAMPING_FINITE_NLL_ARM_IDS[1:]:
        aggregate = _mapping(
            reports[arm_id]["aggregate"],
            label=f"{arm_id} aggregate",
        )
        for name in (
            "example_count",
            "supervised_tokens",
            "source_summed_nll",
            "source_nll_per_token",
        ):
            if aggregate.get(name) != reference_aggregate.get(name):
                raise ValueError(f"{arm_id} aggregate source grid differs")

    paired = _paired_comparison(
        reports[MATCHED_ALPHA0_ARM],
        reports[CHALLENGER_ALPHA0_5_ARM],
        baseline_observations=observations_by_arm[MATCHED_ALPHA0_ARM],
        challenger_observations=observations_by_arm[
            CHALLENGER_ALPHA0_5_ARM
        ],
    )
    paired_gates = _mapping(
        paired["gates"],
        label="paired finite-NLL gates",
    )
    challenger_gates = _mapping(
        reports[CHALLENGER_ALPHA0_5_ARM]["gates"],
        label="challenger absolute gates",
    )
    paired_passed = paired_gates.get("passed") is True
    absolute_passed = challenger_gates.get("passed") is True

    arm_payload = {
        arm_id: {
            "arm_id": arm_id,
            "semantic": DAMPING_FINITE_NLL_ARM_SEMANTICS[arm_id],
            "execution_receipt_sha256": (
                materialized[arm_id].execution_receipt_sha256
            ),
            "observation_receipt_sha256": _observation_receipt_sha256(
                arm_id=arm_id,
                execution_receipt_sha256=(
                    materialized[arm_id].execution_receipt_sha256
                ),
                observations=observations_by_arm[arm_id],
            ),
            "observations": observations_by_arm[arm_id],
            "fidelity": reports[arm_id],
        }
        for arm_id in DAMPING_FINITE_NLL_ARM_IDS
    }
    payload: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "semantics": {
            "execution_mode": "matched_source_authoritative_shadow_forwards",
            "source_outputs_authoritative": True,
            "candidate_outputs_metrics_only": True,
            "arm_ids": DAMPING_FINITE_NLL_ARM_IDS,
            "arm_semantics": dict(DAMPING_FINITE_NLL_ARM_SEMANTICS),
            "deployment_context_arm_id": ACCEPTED_X4_ONLY_ARM,
            "paired_baseline_arm_id": MATCHED_ALPHA0_ARM,
            "paired_challenger_arm_id": CHALLENGER_ALPHA0_5_ARM,
            "accepted_x4_is_not_the_alpha0_baseline": True,
        },
        "thresholds": {
            "absolute": (
                ESTABLISHED_SHADOW_FIDELITY_GATES.metadata()
            ),
            "paired": {
                "family_count": _EXPECTED_FAMILY_COUNT,
                (
                    "family_macro_mean_prompt_absolute_delta_nll_"
                    "relative_improvement_min"
                ): _MACRO_IMPROVEMENT_MIN,
                "minimum_strict_family_win_count": (
                    _MINIMUM_FAMILY_WIN_COUNT
                ),
                "worst_family_relative_improvement_min": (
                    _WORST_FAMILY_IMPROVEMENT_MIN
                ),
                "secondary_metric_relative_improvement_min": (
                    _SECONDARY_IMPROVEMENT_MIN
                ),
            },
        },
        "source_grid": {
            "source_grid_sha256": _sha256(_GRID_DOMAIN, reference_grid),
            "identical_across_arms": True,
            "example_count": int(reference_aggregate["example_count"]),
            "family_count": _EXPECTED_FAMILY_COUNT,
            "examples_per_family": _EXPECTED_EXAMPLES_PER_FAMILY,
            "supervised_tokens": int(
                reference_aggregate["supervised_tokens"]
            ),
            "source_summed_nll": float(
                reference_aggregate["source_summed_nll"]
            ),
            "source_nll_per_token": float(
                reference_aggregate["source_nll_per_token"]
            ),
        },
        "arms": arm_payload,
        "paired_comparison": paired,
        "qualification": {
            "paired_gate_passed": paired_passed,
            "challenger_absolute_gate_passed": absolute_passed,
            "qualified": paired_passed and absolute_passed,
            "relative_pass_absolute_fail_qualifies": False,
        },
        "safety": {
            "source_authoritative": True,
            "candidate_logits_metrics_only": True,
            "raw_logits_in_report": False,
            "targets_in_report": False,
            "tensor_payload_exposed": False,
            "model_load_required": False,
            "per_example_scalar_observations_in_report": True,
            "candidate_observations_hash_bound": True,
        },
    }
    _assert_scalar_hash_only(payload)
    payload["report_sha256"] = _sha256(_REPORT_DOMAIN, payload)
    validate_gemma_h4_damping_finite_nll_report(payload)
    return payload


def evaluate_gemma_h4_damping_finite_nll_from_provider(
    provider: GemmaH4DampingArmProvider,
    *,
    expected_family_by_example: Mapping[str, str],
    vocab_chunk_size: int = 16_384,
) -> dict[str, object]:
    """Collect live arms through the narrow callback boundary, then score."""

    collect = getattr(provider, "collect", None)
    if not callable(collect):
        raise TypeError("provider must implement collect(arm_id)")
    arms = {
        arm_id: collect(arm_id)
        for arm_id in DAMPING_FINITE_NLL_ARM_IDS
    }
    return evaluate_gemma_h4_damping_finite_nll(
        arms,
        expected_family_by_example=expected_family_by_example,
        vocab_chunk_size=vocab_chunk_size,
    )


__all__ = [
    "ACCEPTED_X4_ONLY_ARM",
    "CHALLENGER_ALPHA0_5_ARM",
    "DAMPING_FINITE_NLL_ARM_IDS",
    "DAMPING_FINITE_NLL_ARM_SEMANTICS",
    "DampingFiniteNLLArmId",
    "GemmaH4DampingArmProvider",
    "GemmaH4DampingFiniteNLLArmInput",
    "GemmaH4DampingFiniteNLLObservation",
    "MATCHED_ALPHA0_ARM",
    "evaluate_gemma_h4_damping_finite_nll",
    "evaluate_gemma_h4_damping_finite_nll_from_provider",
    "measure_gemma_h4_damping_finite_nll_observation",
    "validate_gemma_h4_damping_finite_nll_report",
]
