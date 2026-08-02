"""Immutable protocol artifacts for adaptive generator-innovation v2.

This module deliberately contains no model execution.  It defines two
replayable artifacts:

``scale characterization receipt``
    Reduces per-example, target-blind raw-feature summaries into robust
    prompt-balanced channel scales and binds pre-target candidate-health
    summaries.  Raw modal or feature rows are forbidden from the artifact.

``candidate plan``
    Authenticates the v1 plan, failed development report, and public panel
    receipt; derives the exact v2 candidate bank and feature eligibility from
    the scale receipt; and freezes the nested evaluator and claim boundary.

The already-open 16-by-8 panel is permanently development-only.  A plan built
here may nominate a recipe for a later fresh confirmation panel, but it cannot
authorize finite displacement, provider compilation, runtime, fidelity, or
compression claims.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

from .gemma3_l3_l4_iterative_generator_innovation_development import (
    validate_gemma_iterative_generator_innovation_development_report,
)
from .gemma3_l3_l4_iterative_generator_innovation_panel import (
    Gemma3L3L4GeneratorInnovationPanelReceipt,
)
from .gemma3_l3_l4_iterative_generator_innovation_plan import (
    validate_gemma_iterative_generator_innovation_plan,
)
from .token_loss_fisher_generator_innovation import (
    GENERATOR_INNOVATION_GATE_CONFIG,
)
from .token_loss_fisher_generator_innovation_adaptive_v2 import (
    AdaptiveGeneratorInnovationCandidateSpec,
    AdaptiveGeneratorInnovationEligibilityReceipt,
    AdaptiveGeneratorInnovationPortfolioSpec,
    AdaptiveGeneratorInnovationV2Protocol,
)


__all__ = [
    "GENERATOR_INNOVATION_V2_AGE_BUCKET_ORDER",
    "GENERATOR_INNOVATION_V2_CANDIDATE_ORDER",
    "GENERATOR_INNOVATION_V2_CANDIDATE_SIMPLICITY_ORDER",
    "GENERATOR_INNOVATION_V2_CHANNEL_ORDER",
    "GENERATOR_INNOVATION_V2_HALF_LIVES",
    "GENERATOR_INNOVATION_V2_PLAN_SCHEMA",
    "GENERATOR_INNOVATION_V2_RAW_SOURCE_ORDER",
    "GENERATOR_INNOVATION_V2_SCALE_SCHEMA",
    "GENERATOR_INNOVATION_V2_TEMPERATURE_MULTIPLIERS",
    "build_gemma_iterative_generator_innovation_v2_candidate_plan",
    "build_gemma_iterative_generator_innovation_v2_scale_receipt",
    "generator_innovation_v2_candidate_specs",
    "replay_gemma_iterative_generator_innovation_v2_candidate_plan",
    "replay_gemma_iterative_generator_innovation_v2_scale_receipt",
    "validate_gemma_iterative_generator_innovation_v2_candidate_plan",
    "validate_gemma_iterative_generator_innovation_v2_scale_receipt",
]


GENERATOR_INNOVATION_V2_SCALE_SCHEMA = (
    "fisher_graph.gemma3_l3_l4."
    "iterative_generator_innovation_v2_scale_characterization.v1"
)
GENERATOR_INNOVATION_V2_PLAN_SCHEMA = (
    "fisher_graph.gemma3_l3_l4."
    "iterative_generator_innovation_v2_candidate_plan.v1"
)
GENERATOR_INNOVATION_V2_CHANNEL_ORDER = ("real", "imag")
GENERATOR_INNOVATION_V2_HALF_LIVES = (4, 16, 64)
GENERATOR_INNOVATION_V2_TEMPERATURE_MULTIPLIERS = (0.5, 1.0, 2.0)
GENERATOR_INNOVATION_V2_RAW_SOURCE_ORDER = (
    "current_only",
    "ew04",
    "ew16",
    "ew64",
)
GENERATOR_INNOVATION_V2_AGE_BUCKET_ORDER = (
    "1",
    "2_4",
    "5_16",
    "17_plus",
)
GENERATOR_INNOVATION_V2_CANDIDATE_ORDER = (
    "exact_v1_ew16_tau1",
    "current_only_scale_x0p5",
    "current_only_scale_x1",
    "current_only_scale_x2",
    "ew04_scale_x0p5",
    "ew04_scale_x1",
    "ew04_scale_x2",
    "ew16_scale_x0p5",
    "ew16_scale_x1",
    "ew16_scale_x2",
    "ew64_scale_x0p5",
    "ew64_scale_x1",
    "ew64_scale_x2",
)
GENERATOR_INNOVATION_V2_CANDIDATE_SIMPLICITY_ORDER = (
    "exact_v1_ew16_tau1",
    "ew16_scale_x1",
    "ew16_scale_x2",
    "ew16_scale_x0p5",
    "current_only_scale_x1",
    "current_only_scale_x2",
    "current_only_scale_x0p5",
    "ew04_scale_x1",
    "ew04_scale_x2",
    "ew04_scale_x0p5",
    "ew64_scale_x1",
    "ew64_scale_x2",
    "ew64_scale_x0p5",
)

_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_EXAMPLES_PER_FAMILY = 2
_QUANTILE_ORDER = ("q50", "q90", "q99")
_SCALE_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-v2-scale-receipt:v1\0"
)
_PLAN_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-v2-candidate-plan:v1\0"
)
_HEALTH_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-v2-feature-health:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_PRIOR_LINEAGE_FIELDS = {
    "accepted_x4_head_sha256",
    "adapter_execution_sha256",
    "basis_sha256",
    "bridge_binding_sha256",
    "collection_manifest_sha256",
    "collection_membership_receipt_sha256",
    "collection_role_input_file_sha256",
    "factorial_report_file_sha256",
    "factorial_report_sha256",
    "fit_manifest_sha256",
    "model_sha256",
    "parent_artifact_sha256",
    "parent_h4_head_sha256",
    "v1_development_report_file_sha256",
    "v1_development_report_sha256",
    "v1_panel_receipt_file_sha256",
    "v1_panel_receipt_sha256",
    "v1_plan_file_sha256",
    "v1_plan_sha256",
}
_EXAMPLE_FIELDS = {
    "active_count",
    "audit",
    "candidate_health_by_id",
    "example_id",
    "family_id",
    "parent_modal_trace_sha256",
    "raw_by_source",
    "top_mode_indices",
    "top_mode_norms",
}
_RAW_SOURCE_FIELDS = {
    "abs_raw_quantiles_by_channel",
    "active_count",
    "age_buckets",
    "half_life_active_positions",
    "padding_updates_state",
    "prior_kind",
    "raw_trace_sha256",
    "sign_counts_by_channel",
    "source_id",
    "whole_sequence_equals_two_chunks",
}
_AGE_BUCKET_FIELDS = {
    "abs_raw_quantiles_by_channel",
    "active_count",
    "sign_counts_by_channel",
}
_HEALTH_FIELDS = {
    "active_count",
    "bounded_trace_sha256",
    "candidate_feature_receipt_sha256",
    "candidate_id",
    "central_fraction_by_channel",
    "q90_absolute_bounded_by_channel",
}
_EXAMPLE_AUDIT = {
    "accepted_parent_only": True,
    "candidate_output_read": False,
    "compensation_target_read": False,
    "prompt_or_family_outcome_read": False,
    "raw_feature_rows_retained": False,
    "raw_modal_rows_retained": False,
    "target_blind": True,
    "token_gradient_read": False,
    "token_loss_read": False,
}
_FEATURE_HEALTH_POLICY = {
    "aggregation": {
        "q90_absolute_bounded_by_channel": (
            "median_of_16_per_prompt_values"
        ),
        "central_fraction_by_channel": (
            "mean_of_16_per_prompt_values"
        ),
    },
    "central_interval": {
        "absolute_lower_exclusive": 0.1,
        "absolute_upper_exclusive": 0.9,
    },
    "maximum_prompt_balanced_q90_absolute_bounded": 0.95,
    "minimum_prompt_balanced_central_fraction": 0.50,
    "both_channels_must_pass": True,
    "failed_candidate_rows_retained_as_receipts": True,
    "failed_candidate_excluded_from_inner_selection": True,
    "exact_v1_is_a_control_not_an_adaptive_candidate": True,
    "static_generator_is_always_available": True,
}
_ATTRIBUTION_GATES = {
    "minimum_scale_rescue_macro_improvement_vs_v1": 0.005,
    "minimum_scale_rescue_macro_improvement_vs_static": 0.005,
    "minimum_memory_rescue_macro_improvement_vs_scaled_l16": 0.005,
    "minimum_temporal_value_macro_improvement_vs_current_only": 0.005,
    "minimum_material_family_relative_improvement": 0.001,
    "minimum_material_family_win_count": 5,
    "minimum_active_conditional_fold_count": 5,
    "minimum_non16_selected_fold_count_for_memory_rescue": 5,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_equal(left: object, right: object) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _finite(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _pair(
    value: object,
    *,
    label: str,
    positive: bool = False,
    unit_interval: bool = False,
) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a two-value sequence")
    items = tuple(value)
    if len(items) != 2:
        raise ValueError(f"{label} must contain two values")
    result = tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(items)
    )
    if positive and any(item <= 0.0 for item in result):
        raise ValueError(f"{label} must be strictly positive")
    if unit_interval and any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError(f"{label} must lie in [0, 1]")
    return result  # type: ignore[return-value]


def _count_pair(value: object, *, label: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a two-value sequence")
    items = tuple(value)
    if (
        len(items) != 2
        or any(type(item) is not int or item < 0 for item in items)
    ):
        raise ValueError(f"{label} must contain two nonnegative integers")
    return items  # type: ignore[return-value]


def _median(values: Sequence[float]) -> float:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered:
        raise ValueError("cannot take the median of an empty sequence")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def _source_half_life(source_id: str) -> int | None:
    if source_id == "current_only":
        return None
    return int(source_id.removeprefix("ew"))


def _validate_quantiles(
    value: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> dict[str, dict[str, float | None]]:
    channels = _mapping(value, label=label)
    if set(channels) != set(GENERATOR_INNOVATION_V2_CHANNEL_ORDER):
        raise ValueError(f"{label} channel fields differ")
    result: dict[str, dict[str, float | None]] = {}
    for channel in GENERATOR_INNOVATION_V2_CHANNEL_ORDER:
        row = _mapping(channels[channel], label=f"{label}.{channel}")
        if set(row) != set(_QUANTILE_ORDER):
            raise ValueError(f"{label}.{channel} quantile fields differ")
        if allow_empty:
            if any(row[quantile] is not None for quantile in _QUANTILE_ORDER):
                raise ValueError(
                    f"{label}.{channel} empty quantiles must be null"
                )
            parsed: dict[str, float | None] = {
                quantile: None for quantile in _QUANTILE_ORDER
            }
            result[channel] = parsed
            continue
        parsed = {
            quantile: _finite(
                row[quantile],
                label=f"{label}.{channel}.{quantile}",
                minimum=0.0,
            )
            for quantile in _QUANTILE_ORDER
        }
        if not (
            parsed["q50"] <= parsed["q90"] <= parsed["q99"]
        ):
            raise ValueError(f"{label}.{channel} quantiles are not ordered")
        result[channel] = parsed
    return result


def _validate_sign_counts(
    value: object,
    *,
    active_count: int,
    label: str,
) -> dict[str, dict[str, int]]:
    channels = _mapping(value, label=label)
    if set(channels) != set(GENERATOR_INNOVATION_V2_CHANNEL_ORDER):
        raise ValueError(f"{label} channel fields differ")
    result: dict[str, dict[str, int]] = {}
    for channel in GENERATOR_INNOVATION_V2_CHANNEL_ORDER:
        row = _mapping(channels[channel], label=f"{label}.{channel}")
        if set(row) != {"negative", "positive", "zero"}:
            raise ValueError(f"{label}.{channel} sign fields differ")
        parsed: dict[str, int] = {}
        for key in ("negative", "zero", "positive"):
            item = row[key]
            if type(item) is not int or item < 0:
                raise ValueError(f"{label}.{channel}.{key} is invalid")
            parsed[key] = item
        if sum(parsed.values()) != active_count:
            raise ValueError(f"{label}.{channel} counts do not sum to active")
        result[channel] = parsed
    return result


def _normalize_raw_source(
    value: object,
    *,
    source_id: str,
    example_active_count: int,
) -> dict[str, object]:
    row = _mapping(value, label=f"raw source {source_id}")
    if set(row) != _RAW_SOURCE_FIELDS:
        raise ValueError(f"raw source {source_id} fields differ")
    if row.get("source_id") != source_id:
        raise ValueError(f"raw source {source_id} identity differs")
    active_count = row.get("active_count")
    if type(active_count) is not int or active_count != example_active_count:
        raise ValueError(f"raw source {source_id} active count differs")
    expected_half_life = _source_half_life(source_id)
    if row.get("half_life_active_positions") != expected_half_life:
        raise ValueError(f"raw source {source_id} half-life differs")
    expected_prior = (
        "none_current_only"
        if source_id == "current_only"
        else "ew_prior_before_current_update"
    )
    if row.get("prior_kind") != expected_prior:
        raise ValueError(f"raw source {source_id} prior kind differs")
    if (
        row.get("whole_sequence_equals_two_chunks") is not True
        or row.get("padding_updates_state") is not False
    ):
        raise ValueError(f"raw source {source_id} causal audit failed")
    quantiles = _validate_quantiles(
        row.get("abs_raw_quantiles_by_channel"),
        label=f"raw source {source_id} quantiles",
    )
    signs = _validate_sign_counts(
        row.get("sign_counts_by_channel"),
        active_count=active_count,
        label=f"raw source {source_id} signs",
    )
    buckets = _mapping(
        row.get("age_buckets"),
        label=f"raw source {source_id} age buckets",
    )
    if set(buckets) != set(GENERATOR_INNOVATION_V2_AGE_BUCKET_ORDER):
        raise ValueError(f"raw source {source_id} age bucket identities differ")
    normalized_buckets: dict[str, object] = {}
    bucket_total = 0
    for bucket_id in GENERATOR_INNOVATION_V2_AGE_BUCKET_ORDER:
        bucket = _mapping(
            buckets[bucket_id],
            label=f"raw source {source_id} age bucket {bucket_id}",
        )
        if set(bucket) != _AGE_BUCKET_FIELDS:
            raise ValueError(
                f"raw source {source_id} age bucket fields differ"
            )
        count = bucket.get("active_count")
        if type(count) is not int or count < 0:
            raise ValueError(
                f"raw source {source_id} age bucket count is invalid"
            )
        bucket_total += count
        normalized_buckets[bucket_id] = {
            "active_count": count,
            "abs_raw_quantiles_by_channel": _validate_quantiles(
                bucket.get("abs_raw_quantiles_by_channel"),
                label=(
                    f"raw source {source_id} age bucket "
                    f"{bucket_id} quantiles"
                ),
                allow_empty=count == 0,
            ),
            "sign_counts_by_channel": _validate_sign_counts(
                bucket.get("sign_counts_by_channel"),
                active_count=count,
                label=(
                    f"raw source {source_id} age bucket "
                    f"{bucket_id} signs"
                ),
            ),
        }
    if bucket_total != active_count:
        raise ValueError(f"raw source {source_id} age counts differ")
    return {
        "source_id": source_id,
        "half_life_active_positions": expected_half_life,
        "active_count": active_count,
        "abs_raw_quantiles_by_channel": quantiles,
        "sign_counts_by_channel": signs,
        "age_buckets": normalized_buckets,
        "raw_trace_sha256": _sha(
            row.get("raw_trace_sha256"),
            label=f"raw source {source_id} trace",
        ),
        "prior_kind": expected_prior,
        "whole_sequence_equals_two_chunks": True,
        "padding_updates_state": False,
    }


def _normalize_health(
    value: object,
    *,
    candidate_id: str,
    active_count: int,
) -> dict[str, object]:
    row = _mapping(value, label=f"candidate health {candidate_id}")
    if set(row) != _HEALTH_FIELDS:
        raise ValueError(f"candidate health {candidate_id} fields differ")
    if (
        row.get("candidate_id") != candidate_id
        or row.get("active_count") != active_count
    ):
        raise ValueError(f"candidate health {candidate_id} identity differs")
    return {
        "candidate_id": candidate_id,
        "active_count": active_count,
        "q90_absolute_bounded_by_channel": _pair(
            row.get("q90_absolute_bounded_by_channel"),
            label=f"candidate health {candidate_id} q90",
            unit_interval=True,
        ),
        "central_fraction_by_channel": _pair(
            row.get("central_fraction_by_channel"),
            label=f"candidate health {candidate_id} central fraction",
            unit_interval=True,
        ),
        "bounded_trace_sha256": _sha(
            row.get("bounded_trace_sha256"),
            label=f"candidate health {candidate_id} trace",
        ),
        "candidate_feature_receipt_sha256": _sha(
            row.get("candidate_feature_receipt_sha256"),
            label=f"candidate health {candidate_id} feature receipt",
        ),
    }


def _normalize_example(value: object, *, key: str) -> dict[str, object]:
    row = _mapping(value, label=f"scale example {key}")
    if set(row) != _EXAMPLE_FIELDS:
        raise ValueError(f"scale example {key} fields differ")
    example_id = _identifier(row.get("example_id"), label="example_id")
    if example_id != key:
        raise ValueError("scale example map key differs from example_id")
    _sha(example_id, label="example_id")
    family_id = _identifier(row.get("family_id"), label="family_id")
    active_count = row.get("active_count")
    if type(active_count) is not int or active_count <= 0:
        raise ValueError("scale example active_count must be positive")
    indices = row.get("top_mode_indices")
    if (
        not isinstance(indices, Sequence)
        or isinstance(indices, (str, bytes))
        or tuple(indices) != (0, 1)
    ):
        raise ValueError("scale example top mode indices differ")
    norms = _pair(
        row.get("top_mode_norms"),
        label="scale example top mode norms",
        positive=True,
    )
    audit = _mapping(row.get("audit"), label="scale example audit")
    if not _canonical_equal(audit, _EXAMPLE_AUDIT):
        raise ValueError("scale characterization is not target blind")
    sources = _mapping(row.get("raw_by_source"), label="raw_by_source")
    if set(sources) != set(GENERATOR_INNOVATION_V2_RAW_SOURCE_ORDER):
        raise ValueError("raw source identities differ")
    normalized_sources = {
        source_id: _normalize_raw_source(
            sources[source_id],
            source_id=source_id,
            example_active_count=active_count,
        )
        for source_id in GENERATOR_INNOVATION_V2_RAW_SOURCE_ORDER
    }
    health = _mapping(
        row.get("candidate_health_by_id"),
        label="candidate_health_by_id",
    )
    if set(health) != set(GENERATOR_INNOVATION_V2_CANDIDATE_ORDER):
        raise ValueError("candidate health identities differ")
    normalized_health = {
        candidate_id: _normalize_health(
            health[candidate_id],
            candidate_id=candidate_id,
            active_count=active_count,
        )
        for candidate_id in GENERATOR_INNOVATION_V2_CANDIDATE_ORDER
    }
    return {
        "example_id": example_id,
        "family_id": family_id,
        "active_count": active_count,
        "top_mode_indices": (0, 1),
        "top_mode_norms": norms,
        "parent_modal_trace_sha256": _sha(
            row.get("parent_modal_trace_sha256"),
            label="parent modal trace",
        ),
        "raw_by_source": normalized_sources,
        "candidate_health_by_id": normalized_health,
        "audit": dict(_EXAMPLE_AUDIT),
    }


def _normalize_lineage(value: object) -> dict[str, str]:
    lineage = _mapping(value, label="v2 prior lineage")
    if set(lineage) != _PRIOR_LINEAGE_FIELDS:
        raise ValueError("v2 prior lineage fields differ")
    return {
        key: _sha(lineage[key], label=f"v2 prior lineage {key}")
        for key in sorted(_PRIOR_LINEAGE_FIELDS)
    }


def _scale_rows(
    examples: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for source_id in GENERATOR_INNOVATION_V2_RAW_SOURCE_ORDER:
        medians: list[list[float]] = [[], []]
        for example in examples.values():
            source = _mapping(
                _mapping(
                    example["raw_by_source"],
                    label="example raw sources",
                )[source_id],
                label=f"source {source_id}",
            )
            quantiles = _mapping(
                source["abs_raw_quantiles_by_channel"],
                label=f"source {source_id} quantiles",
            )
            for channel_index, channel in enumerate(
                GENERATOR_INNOVATION_V2_CHANNEL_ORDER
            ):
                medians[channel_index].append(
                    float(_mapping(quantiles[channel], label=channel)["q50"])
                )
        scale = tuple(_median(values) for values in medians)
        if any(value <= 0.0 for value in scale):
            raise ValueError(f"source {source_id} robust scale is not positive")
        result[source_id] = {
            "source_id": source_id,
            "half_life_active_positions": _source_half_life(source_id),
            "prompt_balanced_median_absolute_raw_by_channel": scale,
            "derivation": (
                "median_of_16_per_prompt_q50_absolute_raw_"
                "linear_even_median"
            ),
        }
    return result


def _health_rows(
    examples: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for candidate_id in GENERATOR_INNOVATION_V2_CANDIDATE_ORDER:
        q90s: list[list[float]] = [[], []]
        central: list[list[float]] = [[], []]
        prompt_receipts: dict[str, str] = {}
        candidate_feature_receipts: dict[str, str] = {}
        for example_id, example in examples.items():
            row = _mapping(
                _mapping(
                    example["candidate_health_by_id"],
                    label="candidate health rows",
                )[candidate_id],
                label=f"candidate health {candidate_id}",
            )
            q90 = _pair(
                row["q90_absolute_bounded_by_channel"],
                label=f"{candidate_id} q90",
            )
            fraction = _pair(
                row["central_fraction_by_channel"],
                label=f"{candidate_id} central",
            )
            for index in range(2):
                q90s[index].append(q90[index])
                central[index].append(fraction[index])
            prompt_receipts[example_id] = str(row["bounded_trace_sha256"])
            candidate_feature_receipts[example_id] = str(
                row["candidate_feature_receipt_sha256"]
            )
        q90_aggregate = tuple(_median(values) for values in q90s)
        central_aggregate = tuple(
            math.fsum(values) / len(values) for values in central
        )
        payload = {
            "candidate_id": candidate_id,
            "prompt_balanced_q90_absolute_bounded_by_channel": q90_aggregate,
            "prompt_balanced_central_fraction_by_channel": central_aggregate,
            "bounded_trace_sha256_by_example_id": prompt_receipts,
            "candidate_feature_receipt_sha256_by_example_id": (
                candidate_feature_receipts
            ),
        }
        result[candidate_id] = {
            **payload,
            "feature_health_receipt_sha256": _sha256(
                _HEALTH_DOMAIN,
                payload,
            ),
        }
    return result


def build_gemma_iterative_generator_innovation_v2_scale_receipt(
    *,
    per_example_raw_summaries: Mapping[str, object],
    prior_lineage: Mapping[str, object],
) -> dict[str, object]:
    """Build the target-blind scale/health receipt from aggregate summaries."""

    if tuple(per_example_raw_summaries) != tuple(
        sorted(per_example_raw_summaries)
    ):
        raise ValueError("scale example order must be lexicographically sorted")
    examples = {
        key: _normalize_example(value, key=key)
        for key, value in per_example_raw_summaries.items()
    }
    if len(examples) != _EXPECTED_EXAMPLES:
        raise ValueError("scale characterization needs exactly 16 examples")
    family_counts = Counter(
        str(example["family_id"]) for example in examples.values()
    )
    if (
        len(family_counts) != _EXPECTED_FAMILIES
        or set(family_counts.values()) != {_EXPECTED_EXAMPLES_PER_FAMILY}
    ):
        raise ValueError("scale characterization needs 8 families of 2")
    modes = {
        (
            tuple(example["top_mode_indices"]),
            tuple(example["top_mode_norms"]),
        )
        for example in examples.values()
    }
    if len(modes) != 1:
        raise ValueError("scale characterization top modes differ")
    lineage = _normalize_lineage(prior_lineage)
    payload = {
        "schema": GENERATOR_INNOVATION_V2_SCALE_SCHEMA,
        "lineage": lineage,
        "recipe": {
            "role": "adaptive_development_target_blind_scale_characterization",
            "channel_order": GENERATOR_INNOVATION_V2_CHANNEL_ORDER,
            "raw_source_order": GENERATOR_INNOVATION_V2_RAW_SOURCE_ORDER,
            "half_lives_active_positions": (
                GENERATOR_INNOVATION_V2_HALF_LIVES
            ),
            "temperature_multipliers": (
                GENERATOR_INNOVATION_V2_TEMPERATURE_MULTIPLIERS
            ),
            "quantiles": _QUANTILE_ORDER,
            "quantile_method": "linear",
            "age_bucket_order": GENERATOR_INNOVATION_V2_AGE_BUCKET_ORDER,
            "scale_estimator": (
                "median_of_per_prompt_q50_absolute_raw_by_channel"
            ),
            "target_or_outcome_allowed": False,
        },
        "example_ids": tuple(examples),
        "family_ids": tuple(sorted(family_counts)),
        "family_id_by_example_id": {
            key: str(value["family_id"]) for key, value in examples.items()
        },
        "per_example_raw_summaries": examples,
        "scale_by_source": _scale_rows(examples),
        "candidate_health_by_id": _health_rows(examples),
        "audit": {
            "development_only": True,
            "target_blind": True,
            "raw_rows_retained": False,
            "prompt_text_retained": False,
            "token_ids_retained": False,
            "loss_or_compensation_target_read": False,
            "token_gradient_read": False,
            "candidate_output_read": False,
            "family_or_prompt_outcome_read": False,
            "model_forward_count_claimed": False,
            "finite_displacements_opened": False,
            "provider_compiled": False,
        },
    }
    return {
        **payload,
        "receipt_sha256": _sha256(_SCALE_DOMAIN, payload),
    }


def validate_gemma_iterative_generator_innovation_v2_scale_receipt(
    receipt: object,
) -> None:
    """Validate and self-rebuild a target-blind scale receipt."""

    value = _mapping(receipt, label="v2 scale receipt")
    expected = {
        "audit",
        "candidate_health_by_id",
        "example_ids",
        "family_id_by_example_id",
        "family_ids",
        "lineage",
        "per_example_raw_summaries",
        "receipt_sha256",
        "recipe",
        "scale_by_source",
        "schema",
    }
    if set(value) != expected:
        raise ValueError("v2 scale receipt fields differ")
    if value.get("schema") != GENERATOR_INNOVATION_V2_SCALE_SCHEMA:
        raise ValueError("v2 scale receipt schema differs")
    rebuilt = build_gemma_iterative_generator_innovation_v2_scale_receipt(
        per_example_raw_summaries=_mapping(
            value["per_example_raw_summaries"],
            label="v2 scale examples",
        ),
        prior_lineage=_mapping(value["lineage"], label="v2 scale lineage"),
    )
    if not _canonical_equal(rebuilt, value):
        raise ValueError("v2 scale receipt replay differs")


def replay_gemma_iterative_generator_innovation_v2_scale_receipt(
    *,
    per_example_raw_summaries: Mapping[str, object],
    prior_lineage: Mapping[str, object],
    expected_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild the receipt from its original aggregate inputs."""

    validate_gemma_iterative_generator_innovation_v2_scale_receipt(
        expected_receipt
    )
    rebuilt = build_gemma_iterative_generator_innovation_v2_scale_receipt(
        per_example_raw_summaries=per_example_raw_summaries,
        prior_lineage=prior_lineage,
    )
    if not _canonical_equal(rebuilt, expected_receipt):
        raise ValueError("v2 scale receipt differs from original inputs")
    return rebuilt


def _multiplier_label(value: float) -> str:
    return {0.5: "0p5", 1.0: "1", 2.0: "2"}[value]


def generator_innovation_v2_candidate_specs(
    scale_receipt: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Derive the exact 13-candidate bank from a validated scale receipt."""

    validate_gemma_iterative_generator_innovation_v2_scale_receipt(
        scale_receipt
    )
    scales = _mapping(
        scale_receipt["scale_by_source"],
        label="v2 scale rows",
    )
    specs: list[dict[str, object]] = [
        {
            "candidate_id": "exact_v1_ew16_tau1",
            "feature_kind": "v1",
            "half_life_active_positions": 16,
            "temperatures": (1.0, 1.0),
            "temperature_source": "exact_v1_unit_temperature",
            "temperature_multiplier": None,
            "state_floats_per_sequence": 3,
        }
    ]
    for multiplier in GENERATOR_INNOVATION_V2_TEMPERATURE_MULTIPLIERS:
        base = _pair(
            _mapping(
                scales["current_only"],
                label="current-only scale",
            )["prompt_balanced_median_absolute_raw_by_channel"],
            label="current-only robust scale",
            positive=True,
        )
        specs.append(
            {
                "candidate_id": (
                    "current_only_scale_x"
                    f"{_multiplier_label(multiplier)}"
                ),
                "feature_kind": "current_only",
                "half_life_active_positions": None,
                "temperatures": tuple(
                    multiplier * value for value in base
                ),
                "temperature_source": "current_only",
                "temperature_multiplier": multiplier,
                "state_floats_per_sequence": 0,
            }
        )
    for half_life in GENERATOR_INNOVATION_V2_HALF_LIVES:
        source_id = f"ew{half_life:02d}"
        base = _pair(
            _mapping(scales[source_id], label=f"{source_id} scale")[
                "prompt_balanced_median_absolute_raw_by_channel"
            ],
            label=f"{source_id} robust scale",
            positive=True,
        )
        for multiplier in GENERATOR_INNOVATION_V2_TEMPERATURE_MULTIPLIERS:
            specs.append(
                {
                    "candidate_id": (
                        f"{source_id}_scale_x"
                        f"{_multiplier_label(multiplier)}"
                    ),
                    "feature_kind": "temporal",
                    "half_life_active_positions": half_life,
                    "temperatures": tuple(
                        multiplier * value for value in base
                    ),
                    "temperature_source": source_id,
                    "temperature_multiplier": multiplier,
                    "state_floats_per_sequence": 3,
                }
            )
    if tuple(row["candidate_id"] for row in specs) != (
        GENERATOR_INNOVATION_V2_CANDIDATE_ORDER
    ):
        raise RuntimeError("v2 candidate construction order drifted")
    return tuple(specs)


def _candidate_health_decisions(
    scale_receipt: Mapping[str, object],
) -> dict[str, object]:
    health = _mapping(
        scale_receipt["candidate_health_by_id"],
        label="scale candidate health",
    )
    result: dict[str, object] = {}
    for candidate_id in GENERATOR_INNOVATION_V2_CANDIDATE_ORDER:
        row = _mapping(health[candidate_id], label=candidate_id)
        q90 = _pair(
            row["prompt_balanced_q90_absolute_bounded_by_channel"],
            label=f"{candidate_id} prompt-balanced q90",
        )
        central = _pair(
            row["prompt_balanced_central_fraction_by_channel"],
            label=f"{candidate_id} prompt-balanced central",
        )
        health_passed = (
            all(value < 0.95 for value in q90)
            and all(value >= 0.50 for value in central)
        )
        result[candidate_id] = {
            "candidate_id": candidate_id,
            "prompt_balanced_q90_absolute_bounded_by_channel": q90,
            "prompt_balanced_central_fraction_by_channel": central,
            "health_policy_passed": health_passed,
            "eligible_for_adaptive_inner_selection": (
                candidate_id != "exact_v1_ew16_tau1" and health_passed
            ),
            "role": (
                "exact_v1_control"
                if candidate_id == "exact_v1_ew16_tau1"
                else "adaptive_candidate"
            ),
        }
    return result


def _adaptive_analyzer_bindings(
    *,
    specs: Sequence[Mapping[str, object]],
    health_decisions: Mapping[str, object],
    scale_receipt: Mapping[str, object],
) -> tuple[
    AdaptiveGeneratorInnovationV2Protocol,
    AdaptiveGeneratorInnovationEligibilityReceipt,
]:
    """Build the exact injectable analyzer protocol and eligibility receipt."""

    spec_by_id = {
        str(row["candidate_id"]): row
        for row in specs
    }
    analyzer_specs = tuple(
        AdaptiveGeneratorInnovationCandidateSpec(
            candidate_id=candidate_id,
            family=str(spec_by_id[candidate_id]["feature_kind"]),
            metadata=(
                (
                    "half_life_active_positions",
                    spec_by_id[candidate_id]["half_life_active_positions"],
                ),
                (
                    "state_floats_per_sequence",
                    spec_by_id[candidate_id]["state_floats_per_sequence"],
                ),
                (
                    "temperature_imag",
                    float(spec_by_id[candidate_id]["temperatures"][1]),  # type: ignore[index]
                ),
                (
                    "temperature_multiplier",
                    spec_by_id[candidate_id]["temperature_multiplier"],
                ),
                (
                    "temperature_real",
                    float(spec_by_id[candidate_id]["temperatures"][0]),  # type: ignore[index]
                ),
                (
                    "temperature_source",
                    spec_by_id[candidate_id]["temperature_source"],
                ),
            ),
        )
        for candidate_id in GENERATOR_INNOVATION_V2_CANDIDATE_ORDER
    )
    simplicity = GENERATOR_INNOVATION_V2_CANDIDATE_SIMPLICITY_ORDER

    def ordered_prefix(prefixes: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            candidate_id
            for candidate_id in simplicity
            if candidate_id.startswith(prefixes)
        )

    protocol = AdaptiveGeneratorInnovationV2Protocol(
        candidate_specs=analyzer_specs,
        candidate_simplicity_order=simplicity,
        portfolio_specs=(
            AdaptiveGeneratorInnovationPortfolioSpec(
                portfolio_id="scaled_l16",
                candidate_ids=ordered_prefix(("ew16_",)),
            ),
            AdaptiveGeneratorInnovationPortfolioSpec(
                portfolio_id="current_only",
                candidate_ids=ordered_prefix(("current_only_",)),
            ),
            AdaptiveGeneratorInnovationPortfolioSpec(
                portfolio_id="full_temporal_grid",
                candidate_ids=ordered_prefix(("ew04_", "ew16_", "ew64_")),
            ),
        ),
        static_reference_candidate_id="exact_v1_ew16_tau1",
        v1_candidate_id="exact_v1_ew16_tau1",
        conditional_ridge_labels=("0.1", "1", "10", "inf"),
        ridge_simplicity_order=("10", "1", "0.1"),
        static_candidate_id="static_u",
        minimum_family_count=8,
        required_prompts_per_family=2,
    )
    aggregate_health = _mapping(
        scale_receipt["candidate_health_by_id"],
        label="aggregate candidate health",
    )
    eligible = tuple(
        candidate_id
        for candidate_id in simplicity
        if bool(
            _mapping(
                health_decisions[candidate_id],
                label=candidate_id,
            )["eligible_for_adaptive_inner_selection"]
        )
    )
    eligibility = AdaptiveGeneratorInnovationEligibilityReceipt(
        protocol_sha256=protocol.protocol_sha256,
        scale_receipt_sha256=str(scale_receipt["receipt_sha256"]),
        eligible_candidate_ids=eligible,
        feature_health_receipt_sha256_by_candidate=tuple(
            (
                candidate_id,
                str(
                    _mapping(
                        aggregate_health[candidate_id],
                        label=candidate_id,
                    )["feature_health_receipt_sha256"]
                ),
            )
            for candidate_id in GENERATOR_INNOVATION_V2_CANDIDATE_ORDER
        ),
    )
    return protocol, eligibility


def _panel_membership(
    panel_receipt: Mapping[str, object],
) -> tuple[dict[str, str], Gemma3L3L4GeneratorInnovationPanelReceipt]:
    parsed = Gemma3L3L4GeneratorInnovationPanelReceipt.from_dict(
        panel_receipt
    )
    return (
        dict(
            zip(
                parsed.ordered_prompt_sha256s,
                parsed.ordered_family_ids,
                strict=True,
            )
        ),
        parsed,
    )


def _authenticate_candidate_sources(
    *,
    scale_receipt: Mapping[str, object],
    scale_receipt_file_sha256: str,
    v1_plan: Mapping[str, object],
    v1_plan_file_sha256: str,
    v1_development_report: Mapping[str, object],
    v1_development_report_file_sha256: str,
    v1_panel_receipt: Mapping[str, object],
    v1_panel_receipt_file_sha256: str,
) -> dict[str, object]:
    validate_gemma_iterative_generator_innovation_v2_scale_receipt(
        scale_receipt
    )
    validate_gemma_iterative_generator_innovation_plan(v1_plan)
    validate_gemma_iterative_generator_innovation_development_report(
        v1_development_report
    )
    membership, parsed_panel = _panel_membership(v1_panel_receipt)
    scale_file = _sha(
        scale_receipt_file_sha256,
        label="v2 scale receipt file",
    )
    plan_file = _sha(v1_plan_file_sha256, label="v1 plan file")
    report_file = _sha(
        v1_development_report_file_sha256,
        label="v1 development report file",
    )
    panel_file = _sha(
        v1_panel_receipt_file_sha256,
        label="v1 panel receipt file",
    )
    plan_sha = _sha(v1_plan.get("plan_sha256"), label="v1 plan")
    report_sha = _sha(
        v1_development_report.get("report_sha256"),
        label="v1 development report",
    )
    panel_sha = _sha(parsed_panel.receipt_sha256, label="v1 panel receipt")
    scale_sha = _sha(scale_receipt.get("receipt_sha256"), label="v2 scale")
    scale_lineage = _mapping(
        scale_receipt["lineage"],
        label="v2 scale lineage",
    )
    expected_direct = {
        "v1_plan_sha256": plan_sha,
        "v1_plan_file_sha256": plan_file,
        "v1_development_report_sha256": report_sha,
        "v1_development_report_file_sha256": report_file,
        "v1_panel_receipt_sha256": panel_sha,
        "v1_panel_receipt_file_sha256": panel_file,
    }
    for key, expected in expected_direct.items():
        if scale_lineage.get(key) != expected:
            raise ValueError(f"v2 scale lineage {key} differs")
    report_lineage = _mapping(
        v1_development_report["lineage"],
        label="v1 development lineage",
    )
    live = _mapping(report_lineage["live_lineage"], label="v1 live lineage")
    collection = _mapping(
        report_lineage["collection"],
        label="v1 collection lineage",
    )
    for key in (
        "accepted_x4_head_sha256",
        "adapter_execution_sha256",
        "basis_sha256",
        "bridge_binding_sha256",
        "factorial_report_file_sha256",
        "factorial_report_sha256",
        "fit_manifest_sha256",
        "model_sha256",
        "parent_artifact_sha256",
        "parent_h4_head_sha256",
    ):
        if scale_lineage.get(key) != live.get(key):
            raise ValueError(f"v2 scale live lineage {key} differs")
    collection_keys = {
        "collection_manifest_sha256": "collection_manifest_sha256",
        "collection_membership_receipt_sha256": (
            "collection_membership_receipt_sha256"
        ),
        "collection_role_input_file_sha256": (
            "collection_role_input_file_sha256"
        ),
    }
    for scale_key, collection_key in collection_keys.items():
        if scale_lineage.get(scale_key) != collection.get(collection_key):
            raise ValueError(f"v2 scale collection lineage {scale_key} differs")
    if (
        parsed_panel.plan_sha256 != plan_sha
        or parsed_panel.plan_file_sha256 != plan_file
        or collection.get("prompt_free_panel_artifact_receipt_sha256")
        != panel_sha
        or parsed_panel.manifest_sha256
        != collection.get("collection_manifest_sha256")
        or parsed_panel.membership_receipt_sha256
        != collection.get("collection_membership_receipt_sha256")
        or parsed_panel.role_input_file_sha256
        != collection.get("collection_role_input_file_sha256")
    ):
        raise ValueError("v2 panel, plan, and report lineage differ")
    if not _canonical_equal(
        scale_receipt["family_id_by_example_id"],
        membership,
    ):
        raise ValueError("v2 scale examples differ from authenticated panel")
    decision = _mapping(
        v1_development_report["decision"],
        label="v1 decision",
    )
    if (
        decision.get("nested_family_derivative_screen_passed") is not False
        or decision.get("finite_displacement_opened") is not False
        or decision.get("provider_compiled") is not False
        or decision.get("runtime_or_compression_claim_authorized") is not False
    ):
        raise ValueError("v1 failed-development boundary differs")
    feature_examples = _mapping(
        _mapping(
            v1_development_report["feature_audit"],
            label="v1 feature audit",
        )["by_example_id"],
        label="v1 feature examples",
    )
    scale_examples = _mapping(
        scale_receipt["per_example_raw_summaries"],
        label="v2 scale examples",
    )
    for example_id, scale_example_value in scale_examples.items():
        scale_example = _mapping(scale_example_value, label=example_id)
        prior = _mapping(feature_examples[example_id], label=example_id)
        top = _mapping(prior["top_mode_receipt"], label=example_id)
        if not _canonical_equal(
            {
                "top_mode_indices": scale_example["top_mode_indices"],
                "top_mode_norms": scale_example["top_mode_norms"],
            },
            top,
        ):
            raise ValueError("v2 scale top modes differ from v1 parent")
    return {
        "v2_scale_receipt_sha256": scale_sha,
        "v2_scale_receipt_file_sha256": scale_file,
        "v1_plan_sha256": plan_sha,
        "v1_plan_file_sha256": plan_file,
        "v1_development_report_sha256": report_sha,
        "v1_development_report_file_sha256": report_file,
        "v1_panel_receipt_sha256": panel_sha,
        "v1_panel_receipt_file_sha256": panel_file,
        "basis_sha256": str(scale_lineage["basis_sha256"]),
        "collection_manifest_sha256": parsed_panel.manifest_sha256,
        "collection_membership_receipt_sha256": (
            parsed_panel.membership_receipt_sha256
        ),
        "collection_role_input_file_sha256": (
            parsed_panel.role_input_file_sha256
        ),
    }


def build_gemma_iterative_generator_innovation_v2_candidate_plan(
    *,
    scale_receipt: Mapping[str, object],
    scale_receipt_file_sha256: str,
    v1_plan: Mapping[str, object],
    v1_plan_file_sha256: str,
    v1_development_report: Mapping[str, object],
    v1_development_report_file_sha256: str,
    v1_panel_receipt: Mapping[str, object],
    v1_panel_receipt_file_sha256: str,
) -> dict[str, object]:
    """Authenticate v1 and freeze the adaptive-development candidate plan."""

    lineage = _authenticate_candidate_sources(
        scale_receipt=scale_receipt,
        scale_receipt_file_sha256=scale_receipt_file_sha256,
        v1_plan=v1_plan,
        v1_plan_file_sha256=v1_plan_file_sha256,
        v1_development_report=v1_development_report,
        v1_development_report_file_sha256=(
            v1_development_report_file_sha256
        ),
        v1_panel_receipt=v1_panel_receipt,
        v1_panel_receipt_file_sha256=v1_panel_receipt_file_sha256,
    )
    specs = generator_innovation_v2_candidate_specs(scale_receipt)
    health = _candidate_health_decisions(scale_receipt)
    analyzer_protocol, analyzer_eligibility = _adaptive_analyzer_bindings(
        specs=specs,
        health_decisions=health,
        scale_receipt=scale_receipt,
    )
    eligible = analyzer_eligibility.eligible_candidate_ids
    coordinate_order = [
        "generator_real_shared",
        "generator_imag_shared",
    ]
    for candidate_id in GENERATOR_INNOVATION_V2_CANDIDATE_ORDER:
        coordinate_order.extend(
            (
                f"{candidate_id}__generator_real_conditioned",
                f"{candidate_id}__generator_imag_conditioned",
            )
        )
    payload = {
        "schema": GENERATOR_INNOVATION_V2_PLAN_SCHEMA,
        "lineage": lineage,
        "candidate_bank": {
            "candidate_order": GENERATOR_INNOVATION_V2_CANDIDATE_ORDER,
            "candidate_simplicity_order": (
                GENERATOR_INNOVATION_V2_CANDIDATE_SIMPLICITY_ORDER
            ),
            "candidate_specs": specs,
            "coordinate_order": tuple(coordinate_order),
            "shared_coordinate_count": 2,
            "conditioned_coordinates_per_candidate": 2,
            "total_coordinate_count": len(coordinate_order),
            "static_generator_representation": (
                "evaluator_level_infinite_conditional_ridge_exact_h_zero"
            ),
            "static_reference_candidate_id": "exact_v1_ew16_tau1",
            "v1_candidate_id": "exact_v1_ew16_tau1",
            "candidate_features_multiplied_before_gradient_contraction": True,
            "one_source_execution_and_token_vjp_bank_shared_by_candidates": True,
        },
        "candidate_health_policy": dict(_FEATURE_HEALTH_POLICY),
        "candidate_health_decisions": health,
        "eligible_adaptive_candidate_order": eligible,
        "nested_evaluator": {
            "role": "adaptive_development_only",
            "outer_split": "leave_one_family_out",
            "inner_split": "leave_one_outer_training_family_out",
            "selection_unit": "whole_family",
            "weighting": (
                "equal_family_then_equal_prompt_then_equal_token"
            ),
            "normalization_scope": "inner_or_outer_training_families_only",
            "conditional_ridge_order": ("0.1", "1", "10", "inf"),
            "ridge_simplicity_order": ("10", "1", "0.1"),
            "selection_rule": (
                "one_standard_error_then_prefer_static_then_larger_ridge_"
                "then_l16_then_multiplier_1_then_2_then_0p5"
            ),
            "outer_held_family_used_for_selection": False,
            "arms": {
                "parent": (),
                "legacy_shared": (),
                "static_generator": (),
                "exact_v1": ("exact_v1_ew16_tau1",),
                "scaled_l16": tuple(
                    candidate_id
                    for candidate_id in eligible
                    if candidate_id.startswith("ew16_")
                ),
                "current_only": tuple(
                    candidate_id
                    for candidate_id in eligible
                    if candidate_id.startswith("current_only_")
                ),
                "full_temporal_grid": tuple(
                    candidate_id
                    for candidate_id in eligible
                    if candidate_id.startswith(("ew04_", "ew16_", "ew64_"))
                ),
            },
            "trust_projection": {
                "operator_norm_bound": 0.25,
                "corner_count": 16,
                "projection": "one_global_nonnegative_radial_scale",
                "coordinatewise_clipping_allowed": False,
            },
            "base_gate_config": dict(GENERATOR_INNOVATION_GATE_CONFIG),
            "attribution_gate_config": dict(_ATTRIBUTION_GATES),
            "adaptive_analyzer_protocol": analyzer_protocol.to_dict(),
            "activation_only_eligibility_receipt": (
                analyzer_eligibility.to_dict()
            ),
        },
        "claim_boundary": {
            "panel_role": "adaptive_development",
            "already_open_panel_permanently_development_only": True,
            "outer_lofo_is_internal_development_estimate": True,
            "may_diagnose_scale_vs_memory": True,
            "may_nominate_and_freeze_one_v2_recipe": True,
            "fresh_family_disjoint_confirmation_required": True,
            "finite_displacement_authorized": False,
            "provider_compilation_authorized": False,
            "runtime_claim_authorized": False,
            "fidelity_claim_authorized": False,
            "compression_claim_authorized": False,
            "fresh_confirmation_result_may_refit_or_select": False,
        },
        "audit": {
            "model_execution_performed": False,
            "target_or_loss_outcome_used_to_build_candidate_bank": False,
            "basis_refit": False,
            "v1_result_reclassified_as_confirmation": False,
        },
    }
    return {**payload, "plan_sha256": _sha256(_PLAN_DOMAIN, payload)}


def validate_gemma_iterative_generator_innovation_v2_candidate_plan(
    plan: object,
) -> None:
    """Validate the self-contained fixed recipe and hash."""

    value = _mapping(plan, label="v2 candidate plan")
    expected_fields = {
        "audit",
        "candidate_bank",
        "candidate_health_decisions",
        "candidate_health_policy",
        "claim_boundary",
        "eligible_adaptive_candidate_order",
        "lineage",
        "nested_evaluator",
        "plan_sha256",
        "schema",
    }
    if set(value) != expected_fields:
        raise ValueError("v2 candidate plan fields differ")
    if value.get("schema") != GENERATOR_INNOVATION_V2_PLAN_SCHEMA:
        raise ValueError("v2 candidate plan schema differs")
    bank = _mapping(value["candidate_bank"], label="v2 candidate bank")
    if (
        tuple(bank.get("candidate_order", ()))
        != GENERATOR_INNOVATION_V2_CANDIDATE_ORDER
        or tuple(bank.get("candidate_simplicity_order", ()))
        != GENERATOR_INNOVATION_V2_CANDIDATE_SIMPLICITY_ORDER
        or tuple(
            _mapping(row, label="candidate spec").get("candidate_id")
            for row in bank.get("candidate_specs", ())  # type: ignore[union-attr]
        )
        != GENERATOR_INNOVATION_V2_CANDIDATE_ORDER
        or bank.get("total_coordinate_count")
        != 2 + 2 * len(GENERATOR_INNOVATION_V2_CANDIDATE_ORDER)
    ):
        raise ValueError("v2 candidate bank geometry differs")
    if not _canonical_equal(
        value["candidate_health_policy"],
        _FEATURE_HEALTH_POLICY,
    ):
        raise ValueError("v2 feature-health policy differs")
    nested = _mapping(value["nested_evaluator"], label="v2 evaluator")
    if (
        not _canonical_equal(
            nested.get("base_gate_config"),
            GENERATOR_INNOVATION_GATE_CONFIG,
        )
        or not _canonical_equal(
            nested.get("attribution_gate_config"),
            _ATTRIBUTION_GATES,
        )
    ):
        raise ValueError("v2 evaluator gates differ")
    analyzer_protocol = AdaptiveGeneratorInnovationV2Protocol.from_dict(
        _mapping(
            nested.get("adaptive_analyzer_protocol"),
            label="adaptive analyzer protocol",
        )
    )
    analyzer_eligibility = (
        AdaptiveGeneratorInnovationEligibilityReceipt.from_dict(
            _mapping(
                nested.get("activation_only_eligibility_receipt"),
                label="adaptive analyzer eligibility",
            )
        )
    )
    analyzer_candidate_specs: list[dict[str, object]] = []
    expected_metadata_fields = {
        "half_life_active_positions",
        "state_floats_per_sequence",
        "temperature_imag",
        "temperature_multiplier",
        "temperature_real",
        "temperature_source",
    }
    for spec in analyzer_protocol.candidate_specs:
        metadata = dict(spec.metadata)
        if set(metadata) != expected_metadata_fields:
            raise ValueError("v2 analyzer candidate metadata differs")
        analyzer_candidate_specs.append(
            {
                "candidate_id": spec.candidate_id,
                "feature_kind": spec.family,
                "half_life_active_positions": metadata[
                    "half_life_active_positions"
                ],
                "temperatures": (
                    metadata["temperature_real"],
                    metadata["temperature_imag"],
                ),
                "temperature_source": metadata["temperature_source"],
                "temperature_multiplier": metadata[
                    "temperature_multiplier"
                ],
                "state_floats_per_sequence": metadata[
                    "state_floats_per_sequence"
                ],
            }
        )
    if (
        tuple(
            spec.candidate_id
            for spec in analyzer_protocol.candidate_specs
        )
        != GENERATOR_INNOVATION_V2_CANDIDATE_ORDER
        or not _canonical_equal(
            bank.get("candidate_specs"),
            tuple(analyzer_candidate_specs),
        )
        or analyzer_protocol.candidate_simplicity_order
        != GENERATOR_INNOVATION_V2_CANDIDATE_SIMPLICITY_ORDER
        or analyzer_protocol.static_reference_candidate_id
        != "exact_v1_ew16_tau1"
        or analyzer_protocol.v1_candidate_id != "exact_v1_ew16_tau1"
        or analyzer_eligibility.protocol_sha256
        != analyzer_protocol.protocol_sha256
        or tuple(value["eligible_adaptive_candidate_order"])
        != analyzer_eligibility.eligible_candidate_ids
    ):
        raise ValueError("v2 adaptive analyzer binding differs")
    boundary = _mapping(value["claim_boundary"], label="v2 claim boundary")
    for key in (
        "finite_displacement_authorized",
        "provider_compilation_authorized",
        "runtime_claim_authorized",
        "fidelity_claim_authorized",
        "compression_claim_authorized",
    ):
        if boundary.get(key) is not False:
            raise ValueError("v2 claim boundary was widened")
    receipt = _sha(value.get("plan_sha256"), label="v2 candidate plan")
    payload = {key: item for key, item in value.items() if key != "plan_sha256"}
    if receipt != _sha256(_PLAN_DOMAIN, payload):
        raise ValueError("v2 candidate plan hash mismatch")


def replay_gemma_iterative_generator_innovation_v2_candidate_plan(
    *,
    scale_receipt: Mapping[str, object],
    scale_receipt_file_sha256: str,
    v1_plan: Mapping[str, object],
    v1_plan_file_sha256: str,
    v1_development_report: Mapping[str, object],
    v1_development_report_file_sha256: str,
    v1_panel_receipt: Mapping[str, object],
    v1_panel_receipt_file_sha256: str,
    expected_plan: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild the candidate plan from all authenticated immutable inputs."""

    validate_gemma_iterative_generator_innovation_v2_candidate_plan(
        expected_plan
    )
    rebuilt = build_gemma_iterative_generator_innovation_v2_candidate_plan(
        scale_receipt=scale_receipt,
        scale_receipt_file_sha256=scale_receipt_file_sha256,
        v1_plan=v1_plan,
        v1_plan_file_sha256=v1_plan_file_sha256,
        v1_development_report=v1_development_report,
        v1_development_report_file_sha256=(
            v1_development_report_file_sha256
        ),
        v1_panel_receipt=v1_panel_receipt,
        v1_panel_receipt_file_sha256=v1_panel_receipt_file_sha256,
    )
    if not _canonical_equal(rebuilt, expected_plan):
        raise ValueError("v2 candidate plan differs from authenticated inputs")
    return rebuilt
