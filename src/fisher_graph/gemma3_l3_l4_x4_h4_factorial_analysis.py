"""Tensor-free replay analysis for the development-only X4/H4 factorial.

The live Gemma runner owns tensors and forwards.  This module accepts only the
scalar sufficient statistics left after each forward and builds a canonical
six-arm report.  Its validator reconstructs every aggregate and contrast from
those scalar rows, so changing a derived result and merely re-signing the JSON
does not make the report valid.

All contrasts use ``new_error - reference_error``.  Negative values therefore
mean that the newly added component improved source fidelity.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Literal


FactorialArmId = Literal[
    "base_none",
    "base_lag_b",
    "base_independent_state",
    "accepted_none",
    "accepted_lag_b",
    "accepted_independent_state",
]

BASE_NONE_ARM: FactorialArmId = "base_none"
BASE_LAG_B_ARM: FactorialArmId = "base_lag_b"
BASE_INDEPENDENT_STATE_ARM: FactorialArmId = "base_independent_state"
ACCEPTED_NONE_ARM: FactorialArmId = "accepted_none"
ACCEPTED_LAG_B_ARM: FactorialArmId = "accepted_lag_b"
ACCEPTED_INDEPENDENT_STATE_ARM: FactorialArmId = (
    "accepted_independent_state"
)
FACTORIAL_ARM_IDS: tuple[FactorialArmId, ...] = (
    BASE_NONE_ARM,
    BASE_LAG_B_ARM,
    BASE_INDEPENDENT_STATE_ARM,
    ACCEPTED_NONE_ARM,
    ACCEPTED_LAG_B_ARM,
    ACCEPTED_INDEPENDENT_STATE_ARM,
)

FACTORIAL_ARM_FACTORS: Mapping[FactorialArmId, Mapping[str, str]] = (
    MappingProxyType(
        {
            BASE_NONE_ARM: MappingProxyType(
                {"x4": "base", "h4": "none"}
            ),
            BASE_LAG_B_ARM: MappingProxyType(
                {"x4": "base", "h4": "lag_b"}
            ),
            BASE_INDEPENDENT_STATE_ARM: MappingProxyType(
                {"x4": "base", "h4": "independent_state"}
            ),
            ACCEPTED_NONE_ARM: MappingProxyType(
                {"x4": "accepted", "h4": "none"}
            ),
            ACCEPTED_LAG_B_ARM: MappingProxyType(
                {"x4": "accepted", "h4": "lag_b"}
            ),
            ACCEPTED_INDEPENDENT_STATE_ARM: MappingProxyType(
                {"x4": "accepted", "h4": "independent_state"}
            ),
        }
    )
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_x4_h4_factorial_analysis"
_FORMAT_VERSION = 1
_REPORT_DOMAIN = b"fisher-graph:gemma-x4-h4-factorial-report:v1\0"
_MANIFEST_DOMAIN = b"fisher-graph:gemma-x4-h4-factorial-manifest:v1\0"
_LINEAGE_DOMAIN = b"fisher-graph:gemma-x4-h4-factorial-lineage:v1\0"
_FINAL_OBSERVATION_DOMAIN = (
    b"fisher-graph:gemma-x4-h4-factorial-final-observation:v1\0"
)
_BOUNDARY_OBSERVATION_DOMAIN = (
    b"fisher-graph:gemma-x4-h4-factorial-boundary-observation:v1\0"
)
_CELL_DOMAIN = b"fisher-graph:gemma-x4-h4-factorial-cell:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_FINAL_FIELDS = frozenset(
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
_STATS_FIELDS = frozenset(
    {
        "scalar_count",
        "squared_error_sum",
        "source_squared_sum",
        "candidate_squared_sum",
        "source_candidate_dot",
        "max_absolute_error",
    }
)
_BOUNDARY_FIELDS = frozenset(
    {
        "arm_id",
        "example_id",
        "family_id",
        "x4",
        "h4",
        "boundary_observation_sha256",
    }
)
_CELL_FIELDS = frozenset(
    {
        "arm_id",
        "example_id",
        "family_id",
        "final_output",
        "boundary",
        "cell_sha256",
    }
)

_ERROR_METRICS = (
    "final_absolute_delta_nll_per_token",
    "final_kl_per_token",
    "final_top1_disagreement",
    "x4_rmse",
    "x4_nrmse",
    "x4_cosine_error",
    "h4_rmse",
    "h4_nrmse",
    "h4_cosine_error",
)
_SIGNED_METRICS = ("final_signed_delta_nll_per_token",)

_REQUIRED_SAFETY: Mapping[str, object] = MappingProxyType(
    {
        "development_only": True,
        "reusable_development_inputs_only": True,
        "source_outputs_authoritative": True,
        "candidate_outputs_metrics_only": True,
        "tensors_retained": False,
        "raw_prompts_retained": False,
        "raw_token_ids_retained": False,
        "raw_logits_retained": False,
        "raw_activations_retained": False,
        "model_weights_retained": False,
        "deployment_claim": False,
        "generalization_claim": False,
        "compression_qualification_claim": False,
    }
)

_SEMANTICS = {
    "analysis_role": "reusable_development_factorial_attribution",
    "arm_ids": FACTORIAL_ARM_IDS,
    "arm_factors": {
        arm_id: dict(FACTORIAL_ARM_FACTORS[arm_id])
        for arm_id in FACTORIAL_ARM_IDS
    },
    "factor_design": "two_x4_levels_by_three_h4_levels",
    "contrast_sign": "new_error_minus_reference_error",
    "negative_contrast_means": "fidelity_improvement",
    "source_authority": "direct_factorized_model",
    "no_promotion_or_deployment_decision": True,
}


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


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_float(
    value: object,
    *,
    label: str,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _strict_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _arm_id(value: object) -> FactorialArmId:
    if value not in FACTORIAL_ARM_IDS:
        raise ValueError(
            f"arm_id must be exactly one of {FACTORIAL_ARM_IDS!r}"
        )
    return value  # type: ignore[return-value]


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sensitive_key_allowed(key: str, value: object) -> bool:
    lowered = key.lower()
    if lowered.endswith("_sha256"):
        return True
    if lowered.endswith("_retained") and value is False:
        return True
    return not (
        "prompt" in lowered
        or lowered in {"text", "input", "inputs", "targets"}
        or "input_ids" in lowered
        or "token_ids" in lowered
        or "logits" in lowered
        or "activation" in lowered
        or "tensor" in lowered
        or "hidden_state" in lowered
        or "model_weight" in lowered
    )


def _canonical_scalar_tree(value: object, *, path: str) -> object:
    """Copy a JSON scalar tree while rejecting raw/tensor-shaped payloads."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite scalar")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            item = value[key]
            if not _sensitive_key_allowed(key, item):
                raise ValueError(f"{path}.{key} is a raw-sensitive key")
            result[key] = _canonical_scalar_tree(
                item,
                path=f"{path}.{key}",
            )
        return result
    if isinstance(value, (tuple, list)):
        return [
            _canonical_scalar_tree(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported payload {type(value)!r}; "
        "only scalar/hash JSON is allowed"
    )


@dataclass(frozen=True, slots=True)
class GemmaX4H4FactorialFinalObservation:
    """Immutable final-output fidelity scalars for one arm/example cell."""

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
        _identifier(self.example_id, label="final example_id")
        _identifier(self.family_id, label="final family_id")
        _strict_int(
            self.supervised_tokens,
            label="final supervised_tokens",
            minimum=1,
        )
        _strict_int(self.top1_matches, label="final top1_matches")
        if self.top1_matches > self.supervised_tokens:
            raise ValueError(
                "final top1_matches cannot exceed supervised_tokens"
            )
        for name in (
            "source_summed_nll",
            "candidate_summed_nll",
            "source_to_candidate_summed_kl",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    label=f"final {name}",
                    nonnegative=True,
                ),
            )
        for name in (
            "source_logits_sha256",
            "candidate_logits_sha256",
            "targets_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"final {name}")
        object.__setattr__(
            self,
            "observation_sha256",
            _sha256(_FINAL_OBSERVATION_DOMAIN, self._payload()),
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
        return {**self._payload(), "observation_sha256": self.observation_sha256}


@dataclass(frozen=True, slots=True)
class GemmaX4H4BoundarySufficientStats:
    """Additive activation-error statistics with no activation payload."""

    scalar_count: int
    squared_error_sum: float
    source_squared_sum: float
    candidate_squared_sum: float
    source_candidate_dot: float
    max_absolute_error: float

    def __post_init__(self) -> None:
        _strict_int(
            self.scalar_count,
            label="boundary scalar_count",
            minimum=1,
        )
        for name in (
            "squared_error_sum",
            "source_squared_sum",
            "candidate_squared_sum",
            "max_absolute_error",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    label=f"boundary {name}",
                    nonnegative=True,
                ),
            )
        object.__setattr__(
            self,
            "source_candidate_dot",
            _finite_float(
                self.source_candidate_dot,
                label="boundary source_candidate_dot",
            ),
        )
        if self.source_squared_sum <= 0.0:
            raise ValueError(
                "boundary source_squared_sum must be positive for NRMSE"
            )
        expected_sse = (
            self.source_squared_sum
            + self.candidate_squared_sum
            - 2.0 * self.source_candidate_dot
        )
        if not math.isclose(
            self.squared_error_sum,
            max(0.0, expected_sse),
            rel_tol=1e-5,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "boundary squared_error_sum is inconsistent with "
                "source/candidate sufficient statistics"
            )
        norm_product = math.sqrt(
            self.source_squared_sum * self.candidate_squared_sum
        )
        if abs(self.source_candidate_dot) > norm_product + max(
            1e-8,
            1e-5 * norm_product,
        ):
            raise ValueError(
                "boundary source_candidate_dot violates Cauchy-Schwarz"
            )
        if (
            self.max_absolute_error * self.max_absolute_error
            > self.squared_error_sum
            + max(1e-8, 1e-5 * self.squared_error_sum)
        ):
            raise ValueError(
                "boundary max_absolute_error exceeds squared-error energy"
            )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "scalar_count": self.scalar_count,
            "squared_error_sum": self.squared_error_sum,
            "source_squared_sum": self.source_squared_sum,
            "candidate_squared_sum": self.candidate_squared_sum,
            "source_candidate_dot": self.source_candidate_dot,
            "max_absolute_error": self.max_absolute_error,
        }


@dataclass(frozen=True, slots=True)
class GemmaX4H4FactorialBoundaryObservation:
    """The X4 and H4 boundary statistics paired to one executed cell."""

    arm_id: FactorialArmId
    example_id: str
    family_id: str
    x4: GemmaX4H4BoundarySufficientStats
    h4: GemmaX4H4BoundarySufficientStats
    boundary_observation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _arm_id(self.arm_id))
        _identifier(self.example_id, label="boundary example_id")
        _identifier(self.family_id, label="boundary family_id")
        if not isinstance(self.x4, GemmaX4H4BoundarySufficientStats):
            raise TypeError("boundary x4 must be sufficient statistics")
        if not isinstance(self.h4, GemmaX4H4BoundarySufficientStats):
            raise TypeError("boundary h4 must be sufficient statistics")
        object.__setattr__(
            self,
            "boundary_observation_sha256",
            _sha256(_BOUNDARY_OBSERVATION_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "example_id": self.example_id,
            "family_id": self.family_id,
            "x4": self.x4.to_dict(),
            "h4": self.h4.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "boundary_observation_sha256": (
                self.boundary_observation_sha256
            ),
        }


def _compatible_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"final observation omits {name}")
        return value[name]
    if not hasattr(value, name):
        raise TypeError(
            "final observation must be compatible with "
            "GemmaH4DampingFiniteNLLObservation"
        )
    return getattr(value, name)


def _coerce_final_observation(
    value: object,
) -> GemmaX4H4FactorialFinalObservation:
    if isinstance(value, GemmaX4H4FactorialFinalObservation):
        return value
    return GemmaX4H4FactorialFinalObservation(
        example_id=_compatible_field(value, "example_id"),  # type: ignore[arg-type]
        family_id=_compatible_field(value, "family_id"),  # type: ignore[arg-type]
        supervised_tokens=_compatible_field(  # type: ignore[arg-type]
            value, "supervised_tokens"
        ),
        source_summed_nll=_compatible_field(  # type: ignore[arg-type]
            value, "source_summed_nll"
        ),
        candidate_summed_nll=_compatible_field(  # type: ignore[arg-type]
            value, "candidate_summed_nll"
        ),
        source_to_candidate_summed_kl=_compatible_field(  # type: ignore[arg-type]
            value, "source_to_candidate_summed_kl"
        ),
        top1_matches=_compatible_field(  # type: ignore[arg-type]
            value, "top1_matches"
        ),
        source_logits_sha256=_compatible_field(  # type: ignore[arg-type]
            value, "source_logits_sha256"
        ),
        candidate_logits_sha256=_compatible_field(  # type: ignore[arg-type]
            value, "candidate_logits_sha256"
        ),
        targets_sha256=_compatible_field(  # type: ignore[arg-type]
            value, "targets_sha256"
        ),
    )


def _canonical_manifest(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("manifest must be a nonempty example-to-family map")
    result: dict[str, str] = {}
    for example_id, family_id in value.items():
        example = _identifier(example_id, label="manifest example_id")
        family = _identifier(family_id, label="manifest family_id")
        if example in result:
            raise ValueError("manifest contains duplicate example_id")
        result[example] = family
    return dict(sorted(result.items()))


def _canonical_lineage(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("lineage must contain at least one SHA-256 identity")
    result: dict[str, str] = {}
    for name, digest in value.items():
        key = _identifier(name, label="lineage key")
        if not _sensitive_key_allowed(key, digest):
            raise ValueError(f"lineage.{key} is a raw-sensitive key")
        result[key] = _require_sha256(digest, label=f"lineage {key}")
    return dict(sorted(result.items()))


def _stats_summary(
    rows: Sequence[GemmaX4H4BoundarySufficientStats],
) -> dict[str, int | float]:
    if not rows:
        raise ValueError("boundary summary requires observations")
    scalar_count = sum(row.scalar_count for row in rows)
    squared_error_sum = max(
        0.0, math.fsum(row.squared_error_sum for row in rows)
    )
    source_squared_sum = math.fsum(
        row.source_squared_sum for row in rows
    )
    candidate_squared_sum = math.fsum(
        row.candidate_squared_sum for row in rows
    )
    source_candidate_dot = math.fsum(
        row.source_candidate_dot for row in rows
    )
    max_absolute_error = max(row.max_absolute_error for row in rows)
    rmse = math.sqrt(squared_error_sum / scalar_count)
    nrmse = math.sqrt(squared_error_sum / source_squared_sum)
    norm_product = math.sqrt(
        source_squared_sum * candidate_squared_sum
    )
    if norm_product == 0.0:
        cosine = 0.0
    else:
        cosine = max(
            -1.0,
            min(1.0, source_candidate_dot / norm_product),
        )
    return {
        "scalar_count": scalar_count,
        "squared_error_sum": squared_error_sum,
        "source_squared_sum": source_squared_sum,
        "candidate_squared_sum": candidate_squared_sum,
        "source_candidate_dot": source_candidate_dot,
        "max_absolute_error": max_absolute_error,
        "rmse": rmse,
        "nrmse": nrmse,
        "cosine": cosine,
        "cosine_error": 1.0 - cosine,
    }


def _final_summary(
    rows: Sequence[GemmaX4H4FactorialFinalObservation],
) -> dict[str, int | float]:
    if not rows:
        raise ValueError("final-output summary requires observations")
    supervised_tokens = sum(row.supervised_tokens for row in rows)
    source_summed_nll = math.fsum(row.source_summed_nll for row in rows)
    candidate_summed_nll = math.fsum(
        row.candidate_summed_nll for row in rows
    )
    absolute_delta_nll = math.fsum(
        abs(row.candidate_summed_nll - row.source_summed_nll)
        for row in rows
    )
    source_to_candidate_summed_kl = max(
        0.0,
        math.fsum(row.source_to_candidate_summed_kl for row in rows),
    )
    top1_matches = sum(row.top1_matches for row in rows)
    return {
        "example_count": len(rows),
        "supervised_tokens": supervised_tokens,
        "source_summed_nll": source_summed_nll,
        "candidate_summed_nll": candidate_summed_nll,
        "delta_nll_per_token": (
            candidate_summed_nll - source_summed_nll
        )
        / supervised_tokens,
        "absolute_delta_nll_per_token": (
            absolute_delta_nll / supervised_tokens
        ),
        "source_to_candidate_summed_kl": (
            source_to_candidate_summed_kl
        ),
        "kl_per_token": (
            source_to_candidate_summed_kl / supervised_tokens
        ),
        "top1_matches": top1_matches,
        "top1_agreement": top1_matches / supervised_tokens,
        "top1_disagreement": (
            supervised_tokens - top1_matches
        )
        / supervised_tokens,
    }


def _error_metrics(
    *,
    final: Mapping[str, int | float],
    x4: Mapping[str, int | float],
    h4: Mapping[str, int | float],
) -> dict[str, float]:
    return {
        "final_absolute_delta_nll_per_token": float(
            final["absolute_delta_nll_per_token"]
        ),
        "final_kl_per_token": float(final["kl_per_token"]),
        "final_top1_disagreement": float(final["top1_disagreement"]),
        "x4_rmse": float(x4["rmse"]),
        "x4_nrmse": float(x4["nrmse"]),
        "x4_cosine_error": float(x4["cosine_error"]),
        "h4_rmse": float(h4["rmse"]),
        "h4_nrmse": float(h4["nrmse"]),
        "h4_cosine_error": float(h4["cosine_error"]),
    }


def _signed_metrics(
    final: Mapping[str, int | float],
) -> dict[str, float]:
    return {
        "final_signed_delta_nll_per_token": float(
            final["delta_nll_per_token"]
        )
    }


def _summarize_cells(
    cells: Sequence[
        tuple[
            GemmaX4H4FactorialFinalObservation,
            GemmaX4H4FactorialBoundaryObservation,
        ]
    ],
) -> dict[str, object]:
    final = _final_summary([row[0] for row in cells])
    x4 = _stats_summary([row[1].x4 for row in cells])
    h4 = _stats_summary([row[1].h4 for row in cells])
    return {
        "final_output": final,
        "x4_boundary": x4,
        "h4_boundary": h4,
        "error_metrics": _error_metrics(final=final, x4=x4, h4=h4),
        "signed_metrics": _signed_metrics(final),
    }


def _arm_summary(
    cells: Sequence[
        tuple[
            GemmaX4H4FactorialFinalObservation,
            GemmaX4H4FactorialBoundaryObservation,
        ]
    ],
) -> dict[str, object]:
    overall = _summarize_cells(cells)
    grouped: dict[
        str,
        list[
            tuple[
                GemmaX4H4FactorialFinalObservation,
                GemmaX4H4FactorialBoundaryObservation,
            ]
        ],
    ] = {}
    for cell in cells:
        grouped.setdefault(cell[0].family_id, []).append(cell)
    families = [
        {
            "family_id": family_id,
            **_summarize_cells(grouped[family_id]),
        }
        for family_id in sorted(grouped)
    ]
    family_macro = {
        metric: math.fsum(
            float(family["error_metrics"][metric])  # type: ignore[index]
            for family in families
        )
        / len(families)
        for metric in _ERROR_METRICS
    }
    family_macro_signed = {
        metric: math.fsum(
            float(family["signed_metrics"][metric])  # type: ignore[index]
            for family in families
        )
        / len(families)
        for metric in _SIGNED_METRICS
    }
    family_worst: dict[str, object] = {}
    for metric in _ERROR_METRICS:
        worst_value = max(
            float(family["error_metrics"][metric])  # type: ignore[index]
            for family in families
        )
        family_worst[metric] = {
            "value": worst_value,
            "family_ids": [
                str(family["family_id"])
                for family in families
                if float(family["error_metrics"][metric])  # type: ignore[index]
                == worst_value
            ],
        }
    return {
        "overall": overall,
        "families": families,
        "family_macro": family_macro,
        "family_macro_signed": family_macro_signed,
        "family_worst": family_worst,
    }


def _arm_scope_metrics(
    arms: Mapping[str, Mapping[str, object]],
    arm_id: str,
    scope: str,
) -> Mapping[str, object]:
    arm = _mapping(arms[arm_id], label=f"{arm_id} summary")
    if scope == "overall":
        overall = _mapping(arm["overall"], label=f"{arm_id} overall")
        return _mapping(
            overall["error_metrics"],
            label=f"{arm_id} overall errors",
        )
    return _mapping(arm[scope], label=f"{arm_id} {scope}")


def _family_metric_rows(
    arms: Mapping[str, Mapping[str, object]],
    arm_id: str,
) -> dict[str, Mapping[str, object]]:
    arm = _mapping(arms[arm_id], label=f"{arm_id} summary")
    rows = arm.get("families")
    if not isinstance(rows, list):
        raise TypeError(f"{arm_id} family summaries must be a list")
    result: dict[str, Mapping[str, object]] = {}
    for raw in rows:
        row = _mapping(raw, label=f"{arm_id} family row")
        family_id = _identifier(
            row.get("family_id"),
            label=f"{arm_id} family_id",
        )
        result[family_id] = _mapping(
            row.get("error_metrics"),
            label=f"{arm_id} family errors",
        )
    return result


def _arm_signed_metrics(
    arms: Mapping[str, Mapping[str, object]],
    arm_id: str,
    scope: str,
) -> Mapping[str, object]:
    arm = _mapping(arms[arm_id], label=f"{arm_id} summary")
    if scope == "overall":
        overall = _mapping(arm["overall"], label=f"{arm_id} overall")
        return _mapping(
            overall["signed_metrics"],
            label=f"{arm_id} overall signed metrics",
        )
    if scope == "family_macro":
        return _mapping(
            arm["family_macro_signed"],
            label=f"{arm_id} family macro signed metrics",
        )
    raise ValueError(f"unknown signed metric scope {scope!r}")


def _family_signed_rows(
    arms: Mapping[str, Mapping[str, object]],
    arm_id: str,
) -> dict[str, Mapping[str, object]]:
    arm = _mapping(arms[arm_id], label=f"{arm_id} summary")
    rows = arm.get("families")
    if not isinstance(rows, list):
        raise TypeError(f"{arm_id} family summaries must be a list")
    return {
        _identifier(row["family_id"], label=f"{arm_id} family_id"): _mapping(
            row["signed_metrics"],
            label=f"{arm_id} family signed metrics",
        )
        for raw in rows
        for row in [_mapping(raw, label=f"{arm_id} family row")]
    }


def _metric_delta(
    new: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict[str, float]:
    return {
        metric: float(new[metric]) - float(reference[metric])
        for metric in _ERROR_METRICS
    }


def _signed_delta(
    new: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict[str, float]:
    return {
        metric: float(new[metric]) - float(reference[metric])
        for metric in _SIGNED_METRICS
    }


_PAIR_DEFINITIONS = (
    (
        "accepted_x4_effect_h4_none",
        "accepted_x4_effect",
        BASE_NONE_ARM,
        ACCEPTED_NONE_ARM,
    ),
    (
        "accepted_x4_effect_h4_lag_b",
        "accepted_x4_effect",
        BASE_LAG_B_ARM,
        ACCEPTED_LAG_B_ARM,
    ),
    (
        "accepted_x4_effect_h4_independent_state",
        "accepted_x4_effect",
        BASE_INDEPENDENT_STATE_ARM,
        ACCEPTED_INDEPENDENT_STATE_ARM,
    ),
    (
        "lag_b_effect_x4_base",
        "lag_b_effect",
        BASE_NONE_ARM,
        BASE_LAG_B_ARM,
    ),
    (
        "lag_b_effect_x4_accepted",
        "lag_b_effect",
        ACCEPTED_NONE_ARM,
        ACCEPTED_LAG_B_ARM,
    ),
    (
        "independent_state_effect_x4_base",
        "independent_state_effect",
        BASE_NONE_ARM,
        BASE_INDEPENDENT_STATE_ARM,
    ),
    (
        "independent_state_effect_x4_accepted",
        "independent_state_effect",
        ACCEPTED_NONE_ARM,
        ACCEPTED_INDEPENDENT_STATE_ARM,
    ),
    (
        "independent_state_vs_lag_b_x4_base",
        "independent_state_vs_lag_b",
        BASE_LAG_B_ARM,
        BASE_INDEPENDENT_STATE_ARM,
    ),
    (
        "independent_state_vs_lag_b_x4_accepted",
        "independent_state_vs_lag_b",
        ACCEPTED_LAG_B_ARM,
        ACCEPTED_INDEPENDENT_STATE_ARM,
    ),
)

_INTERACTION_DEFINITIONS = (
    (
        "x4_by_lag_b",
        {
            ACCEPTED_LAG_B_ARM: 1,
            BASE_LAG_B_ARM: -1,
            ACCEPTED_NONE_ARM: -1,
            BASE_NONE_ARM: 1,
        },
    ),
    (
        "x4_by_independent_state",
        {
            ACCEPTED_INDEPENDENT_STATE_ARM: 1,
            BASE_INDEPENDENT_STATE_ARM: -1,
            ACCEPTED_NONE_ARM: -1,
            BASE_NONE_ARM: 1,
        },
    ),
    (
        "x4_by_independent_state_vs_lag_b",
        {
            ACCEPTED_INDEPENDENT_STATE_ARM: 1,
            ACCEPTED_LAG_B_ARM: -1,
            BASE_INDEPENDENT_STATE_ARM: -1,
            BASE_LAG_B_ARM: 1,
        },
    ),
)


def _pair_contrasts(
    arms: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for contrast_id, kind, reference_arm, new_arm in _PAIR_DEFINITIONS:
        reference_families = _family_metric_rows(arms, reference_arm)
        new_families = _family_metric_rows(arms, new_arm)
        reference_signed_families = _family_signed_rows(
            arms, reference_arm
        )
        new_signed_families = _family_signed_rows(arms, new_arm)
        rows.append(
            {
                "contrast_id": contrast_id,
                "kind": kind,
                "reference_arm_id": reference_arm,
                "new_arm_id": new_arm,
                "sign": "new_error_minus_reference_error",
                "overall_delta": _metric_delta(
                    _arm_scope_metrics(arms, new_arm, "overall"),
                    _arm_scope_metrics(arms, reference_arm, "overall"),
                ),
                "family_macro_delta": _metric_delta(
                    _arm_scope_metrics(arms, new_arm, "family_macro"),
                    _arm_scope_metrics(
                        arms, reference_arm, "family_macro"
                    ),
                ),
                "overall_signed_delta": _signed_delta(
                    _arm_signed_metrics(arms, new_arm, "overall"),
                    _arm_signed_metrics(arms, reference_arm, "overall"),
                ),
                "family_macro_signed_delta": _signed_delta(
                    _arm_signed_metrics(
                        arms, new_arm, "family_macro"
                    ),
                    _arm_signed_metrics(
                        arms, reference_arm, "family_macro"
                    ),
                ),
                "family_worst_delta": {
                    metric: float(
                        _mapping(
                            _arm_scope_metrics(
                                arms, new_arm, "family_worst"
                            )[metric],
                            label="new family worst",
                        )["value"]
                    )
                    - float(
                        _mapping(
                            _arm_scope_metrics(
                                arms, reference_arm, "family_worst"
                            )[metric],
                            label="reference family worst",
                        )["value"]
                    )
                    for metric in _ERROR_METRICS
                },
                "families": [
                    {
                        "family_id": family_id,
                        "delta": _metric_delta(
                            new_families[family_id],
                            reference_families[family_id],
                        ),
                        "signed_delta": _signed_delta(
                            new_signed_families[family_id],
                            reference_signed_families[family_id],
                        ),
                    }
                    for family_id in sorted(reference_families)
                ],
            }
        )
    return rows


def _linear_metric_combination(
    arms: Mapping[str, Mapping[str, object]],
    coefficients: Mapping[str, int],
    *,
    scope: str,
) -> dict[str, float]:
    return {
        metric: math.fsum(
            coefficient
            * (
                float(
                    _mapping(
                        _arm_scope_metrics(arms, arm_id, "family_worst")[
                            metric
                        ],
                        label="family worst metric",
                    )["value"]
                )
                if scope == "family_worst"
                else float(
                    _arm_scope_metrics(arms, arm_id, scope)[metric]
                )
            )
            for arm_id, coefficient in coefficients.items()
        )
        for metric in _ERROR_METRICS
    }


def _linear_signed_combination(
    arms: Mapping[str, Mapping[str, object]],
    coefficients: Mapping[str, int],
    *,
    scope: str,
) -> dict[str, float]:
    return {
        metric: math.fsum(
            coefficient
            * float(_arm_signed_metrics(arms, arm_id, scope)[metric])
            for arm_id, coefficient in coefficients.items()
        )
        for metric in _SIGNED_METRICS
    }


def _interaction_contrasts(
    arms: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    family_maps = {
        arm_id: _family_metric_rows(arms, arm_id)
        for arm_id in FACTORIAL_ARM_IDS
    }
    signed_family_maps = {
        arm_id: _family_signed_rows(arms, arm_id)
        for arm_id in FACTORIAL_ARM_IDS
    }
    family_ids = sorted(family_maps[BASE_NONE_ARM])
    for contrast_id, coefficients in _INTERACTION_DEFINITIONS:
        rows.append(
            {
                "contrast_id": contrast_id,
                "kind": "difference_of_differences",
                "coefficients": dict(coefficients),
                "sign": "new_error_minus_reference_error_interaction",
                "overall": _linear_metric_combination(
                    arms, coefficients, scope="overall"
                ),
                "family_macro": _linear_metric_combination(
                    arms, coefficients, scope="family_macro"
                ),
                "family_worst": _linear_metric_combination(
                    arms, coefficients, scope="family_worst"
                ),
                "overall_signed": _linear_signed_combination(
                    arms, coefficients, scope="overall"
                ),
                "family_macro_signed": _linear_signed_combination(
                    arms, coefficients, scope="family_macro"
                ),
                "families": [
                    {
                        "family_id": family_id,
                        "interaction": {
                            metric: math.fsum(
                                coefficient
                                * float(
                                    family_maps[arm_id][family_id][
                                        metric
                                    ]
                                )
                                for arm_id, coefficient in (
                                    coefficients.items()
                                )
                            )
                            for metric in _ERROR_METRICS
                        },
                        "signed_interaction": {
                            metric: math.fsum(
                                coefficient
                                * float(
                                    signed_family_maps[arm_id][family_id][
                                        metric
                                    ]
                                )
                                for arm_id, coefficient in (
                                    coefficients.items()
                                )
                            )
                            for metric in _SIGNED_METRICS
                        },
                    }
                    for family_id in family_ids
                ],
            }
        )
    return rows


def _source_identity(
    observation: GemmaX4H4FactorialFinalObservation,
) -> tuple[object, ...]:
    return (
        observation.family_id,
        observation.supervised_tokens,
        observation.source_summed_nll,
        observation.source_logits_sha256,
        observation.targets_sha256,
    )


def _validate_x4_invariance(
    boundary_by_cell: Mapping[
        tuple[str, str], GemmaX4H4FactorialBoundaryObservation
    ],
    manifest: Mapping[str, str],
) -> None:
    for example_id in manifest:
        for prefix, arm_ids in (
            (
                "base",
                (
                    BASE_NONE_ARM,
                    BASE_LAG_B_ARM,
                    BASE_INDEPENDENT_STATE_ARM,
                ),
            ),
            (
                "accepted",
                (
                    ACCEPTED_NONE_ARM,
                    ACCEPTED_LAG_B_ARM,
                    ACCEPTED_INDEPENDENT_STATE_ARM,
                ),
            ),
        ):
            x4_rows = [
                boundary_by_cell[(arm_id, example_id)].x4.to_dict()
                for arm_id in arm_ids
            ]
            if any(row != x4_rows[0] for row in x4_rows[1:]):
                raise ValueError(
                    f"{prefix} X4 boundary differs across H4 levels for "
                    f"{example_id}"
                )


def build_gemma_x4_h4_factorial_report(
    *,
    observations: Mapping[str, Sequence[object]],
    boundary_observations: Sequence[
        GemmaX4H4FactorialBoundaryObservation
    ],
    manifest: Mapping[str, str],
    lineage: Mapping[str, str],
    execution: Mapping[str, object],
    resources: Mapping[str, object],
    safety: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the canonical scalar-only report from one complete 2x3 grid."""

    canonical_manifest = _canonical_manifest(manifest)
    canonical_lineage = _canonical_lineage(lineage)
    if not isinstance(observations, Mapping) or set(observations) != set(
        FACTORIAL_ARM_IDS
    ):
        raise ValueError(
            "observations must contain exactly the six factorial arm IDs"
        )
    final_by_cell: dict[
        tuple[str, str], GemmaX4H4FactorialFinalObservation
    ] = {}
    for arm_id in FACTORIAL_ARM_IDS:
        raw_rows = observations[arm_id]
        if not isinstance(raw_rows, Sequence) or isinstance(
            raw_rows, (str, bytes)
        ):
            raise TypeError(f"{arm_id} observations must be a sequence")
        for raw in raw_rows:
            row = _coerce_final_observation(raw)
            key = (arm_id, row.example_id)
            if key in final_by_cell:
                raise ValueError(
                    f"duplicate arm-example final observation {key!r}"
                )
            if canonical_manifest.get(row.example_id) != row.family_id:
                raise ValueError(
                    f"{arm_id}/{row.example_id} differs from manifest"
                )
            final_by_cell[key] = row

    expected_cells = {
        (arm_id, example_id)
        for arm_id in FACTORIAL_ARM_IDS
        for example_id in canonical_manifest
    }
    if set(final_by_cell) != expected_cells:
        raise ValueError(
            "final observations contain missing or extra arm-example cells"
        )

    if not isinstance(boundary_observations, Sequence) or isinstance(
        boundary_observations, (str, bytes)
    ):
        raise TypeError("boundary_observations must be a sequence")
    boundary_by_cell: dict[
        tuple[str, str], GemmaX4H4FactorialBoundaryObservation
    ] = {}
    for row in boundary_observations:
        if not isinstance(row, GemmaX4H4FactorialBoundaryObservation):
            raise TypeError(
                "boundary_observations must contain immutable boundary rows"
            )
        key = (row.arm_id, row.example_id)
        if key in boundary_by_cell:
            raise ValueError(
                f"duplicate arm-example boundary observation {key!r}"
            )
        if canonical_manifest.get(row.example_id) != row.family_id:
            raise ValueError(
                f"{row.arm_id}/{row.example_id} boundary differs from "
                "manifest"
            )
        boundary_by_cell[key] = row
    if set(boundary_by_cell) != expected_cells:
        raise ValueError(
            "boundary observations contain missing or extra arm-example cells"
        )

    for key in expected_cells:
        if (
            final_by_cell[key].family_id
            != boundary_by_cell[key].family_id
        ):
            raise ValueError(f"{key!r} final/boundary family mismatch")
    for example_id in canonical_manifest:
        source_identities = {
            _source_identity(final_by_cell[(arm_id, example_id)])
            for arm_id in FACTORIAL_ARM_IDS
        }
        if len(source_identities) != 1:
            raise ValueError(
                f"source output identity differs across arms for {example_id}"
            )
        for site in ("x4", "h4"):
            source_stats = {
                (
                    getattr(boundary_by_cell[(arm_id, example_id)], site)
                    .scalar_count,
                    getattr(boundary_by_cell[(arm_id, example_id)], site)
                    .source_squared_sum,
                )
                for arm_id in FACTORIAL_ARM_IDS
            }
            if len(source_stats) != 1:
                raise ValueError(
                    f"source {site} boundary differs across arms for "
                    f"{example_id}"
                )
    _validate_x4_invariance(boundary_by_cell, canonical_manifest)

    execution_payload = _canonical_scalar_tree(
        execution, path="execution"
    )
    resources_payload = _canonical_scalar_tree(
        resources, path="resources"
    )
    supplied_safety = (
        dict(_REQUIRED_SAFETY) if safety is None else dict(safety)
    )
    for key, expected in _REQUIRED_SAFETY.items():
        if supplied_safety.get(key) is not expected:
            raise ValueError(
                f"safety.{key} must be exactly {expected!r}"
            )
    safety_payload = _canonical_scalar_tree(supplied_safety, path="safety")

    manifest_rows = [
        {"example_id": example_id, "family_id": family_id}
        for example_id, family_id in canonical_manifest.items()
    ]
    manifest_payload = {
        "example_count": len(canonical_manifest),
        "family_count": len(set(canonical_manifest.values())),
        "examples_per_family": dict(
            sorted(Counter(canonical_manifest.values()).items())
        ),
        "family_by_example": manifest_rows,
        "manifest_sha256": _sha256(_MANIFEST_DOMAIN, manifest_rows),
    }
    lineage_payload = {
        "sha256s": canonical_lineage,
        "lineage_sha256": _sha256(
            _LINEAGE_DOMAIN, canonical_lineage
        ),
    }

    report_observations: dict[str, list[dict[str, object]]] = {}
    arm_summaries: dict[str, dict[str, object]] = {}
    for arm_id in FACTORIAL_ARM_IDS:
        cells: list[
            tuple[
                GemmaX4H4FactorialFinalObservation,
                GemmaX4H4FactorialBoundaryObservation,
            ]
        ] = []
        report_rows: list[dict[str, object]] = []
        for example_id in canonical_manifest:
            final = final_by_cell[(arm_id, example_id)]
            boundary = boundary_by_cell[(arm_id, example_id)]
            cell_payload = {
                "arm_id": arm_id,
                "example_id": example_id,
                "family_id": final.family_id,
                "final_output": final.to_dict(),
                "boundary": boundary.to_dict(),
            }
            report_rows.append(
                {
                    **cell_payload,
                    "cell_sha256": _sha256(_CELL_DOMAIN, cell_payload),
                }
            )
            cells.append((final, boundary))
        report_observations[arm_id] = report_rows
        arm_summaries[arm_id] = {
            "arm_id": arm_id,
            "factors": dict(FACTORIAL_ARM_FACTORS[arm_id]),
            **_arm_summary(cells),
        }

    contrasts = {
        "error_metric_names": _ERROR_METRICS,
        "signed_metric_names": _SIGNED_METRICS,
        "sign": "new_error_minus_reference_error",
        "negative_means": "fidelity_improvement",
        "signed_nll_note": (
            "signed deltas preserve over-versus-under-NLL direction and "
            "are reported separately from positive fidelity errors"
        ),
        "pair_contrasts": _pair_contrasts(arm_summaries),
        "difference_of_differences": _interaction_contrasts(
            arm_summaries
        ),
    }
    payload: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "semantics": _SEMANTICS,
        "manifest": manifest_payload,
        "lineage": lineage_payload,
        "execution": execution_payload,
        "resources": resources_payload,
        "observations": report_observations,
        "arms": arm_summaries,
        "contrasts": contrasts,
        "safety": safety_payload,
    }
    _canonical_scalar_tree(payload, path="report")
    return {
        **payload,
        "report_sha256": _sha256(_REPORT_DOMAIN, payload),
    }


def _parse_stats(
    value: object,
    *,
    label: str,
) -> GemmaX4H4BoundarySufficientStats:
    row = _mapping(value, label=label)
    if set(row) != _STATS_FIELDS:
        raise ValueError(f"{label} fields differ")
    return GemmaX4H4BoundarySufficientStats(
        scalar_count=_strict_int(
            row["scalar_count"], label=f"{label} scalar_count", minimum=1
        ),
        squared_error_sum=_finite_float(
            row["squared_error_sum"],
            label=f"{label} squared_error_sum",
            nonnegative=True,
        ),
        source_squared_sum=_finite_float(
            row["source_squared_sum"],
            label=f"{label} source_squared_sum",
            nonnegative=True,
        ),
        candidate_squared_sum=_finite_float(
            row["candidate_squared_sum"],
            label=f"{label} candidate_squared_sum",
            nonnegative=True,
        ),
        source_candidate_dot=_finite_float(
            row["source_candidate_dot"],
            label=f"{label} source_candidate_dot",
        ),
        max_absolute_error=_finite_float(
            row["max_absolute_error"],
            label=f"{label} max_absolute_error",
            nonnegative=True,
        ),
    )


def _parse_final(
    value: object,
    *,
    label: str,
) -> GemmaX4H4FactorialFinalObservation:
    row = _mapping(value, label=label)
    if set(row) != _FINAL_FIELDS:
        raise ValueError(f"{label} fields differ")
    parsed = GemmaX4H4FactorialFinalObservation(
        example_id=_identifier(
            row["example_id"], label=f"{label} example_id"
        ),
        family_id=_identifier(
            row["family_id"], label=f"{label} family_id"
        ),
        supervised_tokens=_strict_int(
            row["supervised_tokens"],
            label=f"{label} supervised_tokens",
            minimum=1,
        ),
        source_summed_nll=_finite_float(
            row["source_summed_nll"],
            label=f"{label} source_summed_nll",
            nonnegative=True,
        ),
        candidate_summed_nll=_finite_float(
            row["candidate_summed_nll"],
            label=f"{label} candidate_summed_nll",
            nonnegative=True,
        ),
        source_to_candidate_summed_kl=_finite_float(
            row["source_to_candidate_summed_kl"],
            label=f"{label} source_to_candidate_summed_kl",
            nonnegative=True,
        ),
        top1_matches=_strict_int(
            row["top1_matches"], label=f"{label} top1_matches"
        ),
        source_logits_sha256=_require_sha256(
            row["source_logits_sha256"],
            label=f"{label} source logits",
        ),
        candidate_logits_sha256=_require_sha256(
            row["candidate_logits_sha256"],
            label=f"{label} candidate logits",
        ),
        targets_sha256=_require_sha256(
            row["targets_sha256"], label=f"{label} targets"
        ),
    )
    if (
        _require_sha256(
            row["observation_sha256"],
            label=f"{label} observation",
        )
        != parsed.observation_sha256
    ):
        raise ValueError(f"{label} observation hash differs")
    return parsed


def _parse_boundary(
    value: object,
    *,
    label: str,
) -> GemmaX4H4FactorialBoundaryObservation:
    row = _mapping(value, label=label)
    if set(row) != _BOUNDARY_FIELDS:
        raise ValueError(f"{label} fields differ")
    parsed = GemmaX4H4FactorialBoundaryObservation(
        arm_id=_arm_id(row["arm_id"]),
        example_id=_identifier(
            row["example_id"], label=f"{label} example_id"
        ),
        family_id=_identifier(
            row["family_id"], label=f"{label} family_id"
        ),
        x4=_parse_stats(row["x4"], label=f"{label} x4"),
        h4=_parse_stats(row["h4"], label=f"{label} h4"),
    )
    if (
        _require_sha256(
            row["boundary_observation_sha256"],
            label=f"{label} observation",
        )
        != parsed.boundary_observation_sha256
    ):
        raise ValueError(f"{label} observation hash differs")
    return parsed


def validate_gemma_x4_h4_factorial_report(
    report: Mapping[str, object],
    *,
    expected_manifest: Mapping[str, str] | None = None,
    expected_lineage: Mapping[str, str] | None = None,
) -> None:
    """Authenticate and replay every scalar-derived factorial result."""

    root = _mapping(report, label="factorial report")
    expected_keys = {
        "schema",
        "format_version",
        "semantics",
        "manifest",
        "lineage",
        "execution",
        "resources",
        "observations",
        "arms",
        "contrasts",
        "safety",
        "report_sha256",
    }
    if set(root) != expected_keys:
        raise ValueError("factorial report keys differ")
    _canonical_scalar_tree(root, path="report")
    if root["schema"] != _SCHEMA or root["format_version"] != _FORMAT_VERSION:
        raise ValueError("factorial report schema or version differs")
    supplied_sha = _require_sha256(
        root["report_sha256"], label="factorial report"
    )
    payload = dict(root)
    payload.pop("report_sha256")
    if _sha256(_REPORT_DOMAIN, payload) != supplied_sha:
        raise ValueError("factorial report hash differs")
    if _canonical_json_bytes(root["semantics"]) != _canonical_json_bytes(
        _SEMANTICS
    ):
        raise ValueError("factorial semantic ABI differs")

    manifest_payload = _mapping(
        root["manifest"], label="factorial manifest"
    )
    if set(manifest_payload) != {
        "example_count",
        "family_count",
        "examples_per_family",
        "family_by_example",
        "manifest_sha256",
    }:
        raise ValueError("factorial manifest fields differ")
    manifest_rows = manifest_payload["family_by_example"]
    if not isinstance(manifest_rows, list) or not manifest_rows:
        raise ValueError("manifest family_by_example must be nonempty")
    parsed_manifest: dict[str, str] = {}
    for index, raw in enumerate(manifest_rows):
        row = _mapping(raw, label=f"manifest row {index}")
        if set(row) != {"example_id", "family_id"}:
            raise ValueError("manifest row fields differ")
        example_id = _identifier(
            row["example_id"], label="manifest example_id"
        )
        family_id = _identifier(
            row["family_id"], label="manifest family_id"
        )
        if example_id in parsed_manifest:
            raise ValueError("manifest contains duplicate example_id")
        parsed_manifest[example_id] = family_id
    parsed_manifest = dict(sorted(parsed_manifest.items()))
    canonical_rows = [
        {"example_id": key, "family_id": value}
        for key, value in parsed_manifest.items()
    ]
    if manifest_rows != canonical_rows:
        raise ValueError("manifest rows are not canonical")
    if (
        _strict_int(
            manifest_payload["example_count"],
            label="manifest example_count",
            minimum=1,
        )
        != len(parsed_manifest)
        or _strict_int(
            manifest_payload["family_count"],
            label="manifest family_count",
            minimum=1,
        )
        != len(set(parsed_manifest.values()))
        or manifest_payload["examples_per_family"]
        != dict(sorted(Counter(parsed_manifest.values()).items()))
        or _require_sha256(
            manifest_payload["manifest_sha256"],
            label="manifest",
        )
        != _sha256(_MANIFEST_DOMAIN, canonical_rows)
    ):
        raise ValueError("factorial manifest summary differs")
    if (
        expected_manifest is not None
        and _canonical_manifest(expected_manifest) != parsed_manifest
    ):
        raise ValueError("factorial report differs from expected manifest")

    lineage_payload = _mapping(
        root["lineage"], label="factorial lineage"
    )
    if set(lineage_payload) != {"sha256s", "lineage_sha256"}:
        raise ValueError("factorial lineage fields differ")
    lineage_values = _mapping(
        lineage_payload["sha256s"], label="lineage sha256s"
    )
    parsed_lineage = _canonical_lineage(
        {
            str(name): _require_sha256(
                digest, label=f"lineage {name}"
            )
            for name, digest in lineage_values.items()
        }
    )
    if (
        _require_sha256(
            lineage_payload["lineage_sha256"], label="lineage"
        )
        != _sha256(_LINEAGE_DOMAIN, parsed_lineage)
    ):
        raise ValueError("factorial lineage hash differs")
    if (
        expected_lineage is not None
        and _canonical_lineage(expected_lineage) != parsed_lineage
    ):
        raise ValueError("factorial report differs from expected lineage")

    observations_payload = _mapping(
        root["observations"], label="factorial observations"
    )
    if set(observations_payload) != set(FACTORIAL_ARM_IDS):
        raise ValueError("factorial observation arms differ")
    parsed_observations: dict[
        str, list[GemmaX4H4FactorialFinalObservation]
    ] = {arm_id: [] for arm_id in FACTORIAL_ARM_IDS}
    parsed_boundaries: list[
        GemmaX4H4FactorialBoundaryObservation
    ] = []
    seen_cells: set[tuple[str, str]] = set()
    for arm_id in FACTORIAL_ARM_IDS:
        raw_cells = observations_payload[arm_id]
        if not isinstance(raw_cells, list):
            raise TypeError(f"{arm_id} cells must be a list")
        for index, raw in enumerate(raw_cells):
            cell = _mapping(raw, label=f"{arm_id} cell {index}")
            if set(cell) != _CELL_FIELDS:
                raise ValueError(f"{arm_id} cell fields differ")
            if _arm_id(cell["arm_id"]) != arm_id:
                raise ValueError(f"{arm_id} cell arm differs")
            final = _parse_final(
                cell["final_output"], label=f"{arm_id} final {index}"
            )
            boundary = _parse_boundary(
                cell["boundary"], label=f"{arm_id} boundary {index}"
            )
            if (
                final.example_id != cell["example_id"]
                or final.family_id != cell["family_id"]
                or boundary.arm_id != arm_id
                or boundary.example_id != final.example_id
                or boundary.family_id != final.family_id
            ):
                raise ValueError(f"{arm_id} cell identities differ")
            key = (arm_id, final.example_id)
            if key in seen_cells:
                raise ValueError("factorial report contains duplicate cell")
            seen_cells.add(key)
            cell_payload = dict(cell)
            supplied_cell_sha = cell_payload.pop("cell_sha256")
            if (
                _require_sha256(
                    supplied_cell_sha, label=f"{arm_id} cell"
                )
                != _sha256(_CELL_DOMAIN, cell_payload)
            ):
                raise ValueError(f"{arm_id} cell hash differs")
            parsed_observations[arm_id].append(final)
            parsed_boundaries.append(boundary)

    execution = _mapping(root["execution"], label="execution")
    resources = _mapping(root["resources"], label="resources")
    safety = _mapping(root["safety"], label="safety")
    rebuilt = build_gemma_x4_h4_factorial_report(
        observations=parsed_observations,
        boundary_observations=parsed_boundaries,
        manifest=parsed_manifest,
        lineage=parsed_lineage,
        execution=execution,
        resources=resources,
        safety=safety,
    )
    if _canonical_json_bytes(rebuilt) != _canonical_json_bytes(root):
        raise ValueError(
            "factorial report does not replay from scalar observations"
        )


__all__ = [
    "ACCEPTED_INDEPENDENT_STATE_ARM",
    "ACCEPTED_LAG_B_ARM",
    "ACCEPTED_NONE_ARM",
    "BASE_INDEPENDENT_STATE_ARM",
    "BASE_LAG_B_ARM",
    "BASE_NONE_ARM",
    "FACTORIAL_ARM_FACTORS",
    "FACTORIAL_ARM_IDS",
    "FactorialArmId",
    "GemmaX4H4BoundarySufficientStats",
    "GemmaX4H4FactorialBoundaryObservation",
    "GemmaX4H4FactorialFinalObservation",
    "build_gemma_x4_h4_factorial_report",
    "validate_gemma_x4_h4_factorial_report",
]
