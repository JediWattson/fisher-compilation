"""Pure V20i training-only reflection fit for the four-term soft router.

The input is one authenticated V20g outer natural-direction receipt or one
V20i six-family direction derived from it.  This module constructs the
identity direction and each one-coordinate sign reflection, normalizes every
variant against the exact four corners of the bilinear box, and ranks only
variants that remain robust first-order descent directions on the direction
receipt's training families.

No API in this module accepts held objectives, prompt text, logits,
Calibration-B, validation, or test evidence.  The V20i runner is responsible
for freezing the selected direction before it obtains any held score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

from fisher_graph.complete_h4_fisher_soft_polarity_trust_region_fit import (
    SOFT_POLARITY_FIT_PROTOCOL_SHA256,
    validate_soft_polarity_direction_receipt,
)

__all__ = [
    "SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256",
    "SOFT_POLARITY_REFLECTION_FEATURES",
    "SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256",
    "SOFT_POLARITY_REFLECTION_VARIANTS",
    "build_soft_polarity_masked_direction_receipt",
    "build_soft_polarity_reflection_fit_receipt",
    "validate_soft_polarity_masked_direction_receipt",
    "validate_soft_polarity_reflection_fit_receipt",
    "validate_soft_polarity_reflection_variant_receipt",
]


SOFT_POLARITY_REFLECTION_FEATURES = (
    "one",
    "c1",
    "c2",
    "c1_times_c2",
)
SOFT_POLARITY_REFLECTION_VARIANTS = (
    "identity",
    "flip_coordinate_0",
    "flip_coordinate_1",
    "flip_coordinate_2",
    "flip_coordinate_3",
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_DAMPING_FRACTION = 1.0e-3
_DAMPING_FLOOR = 1.0e-12
_BOX_CORNERS = (
    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),
)
_MASKED_PROTOCOL_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-masked-direction:protocol:v20i\0"
)
_MASKED_DIRECTION_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-masked-direction:receipt:v20i\0"
)
_PROTOCOL_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-reflection-fit:protocol:v20i\0"
)
_VARIANT_DOMAIN = (
    b"fisher-graph:complete-h4-soft-polarity-reflection-fit:variant:v20i\0"
)
_FIT_DOMAIN = b"fisher-graph:complete-h4-soft-polarity-reflection-fit:fit:v20i\0"

_V20G_DIRECTION_KEYS = frozenset(
    {
        "schema",
        "protocol_sha256",
        "source_artifact_sha256s",
        "all_development_family_ids",
        "held_family_id",
        "training_family_ids",
        "gradient_evidence_sha256",
        "training_example_ids_by_family",
        "training_gradient_sha256s_by_family",
        "family_mean_gradients",
        "family_mean_opg_fishers",
        "family_equal_mean_gradient",
        "family_equal_opg_fisher",
        "damping",
        "damped_fisher",
        "raw_natural_direction",
        "raw_directional_derivative",
        "raw_direction_l2_norm",
        "normalization",
        "normalization_corner_coordinates",
        "raw_normalization_corner_logits",
        "normalization_scale",
        "natural_direction",
        "directional_derivative",
        "direction_l2_norm",
        "normalized_corner_logits",
        "normalized_box_logit_max_abs",
        "finite",
        "strict_descent",
        "data_boundary",
        "artifact_sha256",
    }
)

_DATA_BOUNDARY = {
    "role": "post_v20h_reused_A16_development_hypothesis_only",
    "selection_inputs": (
        "authenticated_v20g_or_v20i_masked_training_family_mean_gradients_only"
    ),
    "held_objectives_consumed": False,
    "prompt_text_consumed": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "fresh_family_disjoint_scores_consumed": False,
}
_MASKED_DIRECTION_PROTOCOL = {
    "protocol": "v20i_six_family_masked_natural_direction",
    "scientific_status": (
        "posthoc_after_v20h_reused_A16_development_hypothesis_only"
    ),
    "source_protocol_sha256": SOFT_POLARITY_FIT_PROTOCOL_SHA256,
    "source": "authenticated_v20g_seven_training_family_outer_direction",
    "mask": "exactly_one_v20g_training_family_excluded",
    "mask_semantics": (
        "excluded_family_has_zero_numerical_influence_on_direction_and_"
        "reflection_selection_while_full_source_is_embedded_for_provenance"
    ),
    "accepted_v20g_direction_keys": tuple(sorted(_V20G_DIRECTION_KEYS)),
    "retained_outer_held_family": True,
    "retained_training_family_count": 6,
    "fisher": "family_equal_mean_of_retained_v20g_family_opg_fishers",
    "gradient": "family_equal_mean_of_retained_v20g_family_mean_gradients",
    "damping": "max_1e-12_1e-3_times_trace_over_four",
    "direction": "negative_damped_fisher_inverse_times_mean_gradient",
    "normalization": (
        "divide_by_exact_max_abs_bilinear_logit_over_all_four_box_corners"
    ),
    "authorization": "training_only_inner_mask_no_held_or_serving_authority",
    "data_boundary": _DATA_BOUNDARY,
}
def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical(value)).hexdigest()


SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256 = _hash(
    _MASKED_PROTOCOL_DOMAIN, _MASKED_DIRECTION_PROTOCOL
)
_PROTOCOL = {
    "protocol": "v20i_training_only_one_coordinate_reflection_fit",
    "scientific_status": (
        "posthoc_after_v20h_reused_A16_development_hypothesis_only"
    ),
    "accepted_direction_protocol_sha256s": (
        SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256,
    ),
    "accepted_v20g_direction_keys": tuple(sorted(_V20G_DIRECTION_KEYS)),
    "features": SOFT_POLARITY_REFLECTION_FEATURES,
    "variant_order": SOFT_POLARITY_REFLECTION_VARIANTS,
    "variants": "identity_and_exactly_one_coordinate_sign_reflection",
    "normalization": (
        "divide_by_exact_max_abs_bilinear_logit_over_all_four_box_corners"
    ),
    "family_score": "family_mean_gradient_dot_normalized_direction",
    "cvar2": "mean_of_two_largest_family_directional_derivatives",
    "admissibility": (
        "strict_negative_family_equal_mean_and_at_least_n_minus_one_"
        "strictly_negative_family_derivatives"
    ),
    "selection": (
        "minimum_cvar2_then_minimum_mean_then_fixed_variant_order_then_"
        "artifact_sha256_among_admissible_variants"
    ),
    "authorization": (
        "training_only_direction_proposal_no_held_or_serving_authority"
    ),
    "data_boundary": _DATA_BOUNDARY,
}
SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256 = _hash(_PROTOCOL_DOMAIN, _PROTOCOL)


def _finish(
    schema: str,
    domain: bytes,
    payload: Mapping[str, object],
) -> dict[str, object]:
    result = {"schema": schema, **dict(payload)}
    result["artifact_sha256"] = _hash(domain, result)
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a canonical nonempty string")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _vector4(value: object, label: str) -> tuple[float, float, float, float]:
    result = tuple(_number(item, label) for item in _sequence(value, label))
    if len(result) != 4:
        raise ValueError(f"{label} must have four values")
    return result  # type: ignore[return-value]


def _matrix4(
    value: object,
    label: str,
) -> tuple[tuple[float, float, float, float], ...]:
    result = tuple(_vector4(row, label) for row in _sequence(value, label))
    if len(result) != 4:
        raise ValueError(f"{label} must be four by four")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _mean_vectors(
    rows: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    return tuple(
        _mean([row[index] for row in rows]) for index in range(4)
    )  # type: ignore[return-value]


def _mean_matrices(
    matrices: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(
            _mean([matrix[row][column] for matrix in matrices])
            for column in range(4)
        )
        for row in range(4)
    )  # type: ignore[return-value]


def _solve_spd(
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
) -> tuple[float, float, float, float]:
    lower = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i + 1):
            residual = matrix[i][j] - math.fsum(
                lower[i][k] * lower[j][k] for k in range(j)
            )
            if i == j:
                if not math.isfinite(residual) or residual <= 0.0:
                    raise ValueError("masked damped Fisher is not positive definite")
                lower[i][j] = math.sqrt(residual)
            else:
                lower[i][j] = residual / lower[j][j]
    y = [0.0] * 4
    for i in range(4):
        y[i] = (
            rhs[i] - math.fsum(lower[i][k] * y[k] for k in range(i))
        ) / lower[i][i]
    x = [0.0] * 4
    for i in range(3, -1, -1):
        x[i] = (
            y[i] - math.fsum(lower[k][i] * x[k] for k in range(i + 1, 4))
        ) / lower[i][i]
    return tuple(0.0 if value == 0.0 else value for value in x)  # type: ignore[return-value]


def _box_corner_logits(
    coefficients: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return tuple(
        math.fsum(
            (
                coefficients[0],
                coefficients[1] * c1,
                coefficients[2] * c2,
                coefficients[3] * c1 * c2,
            )
        )
        for c1, c2 in _BOX_CORNERS
    )  # type: ignore[return-value]


def _expected_masked_direction_receipt(
    *,
    source_direction_receipt: Mapping[str, object],
    excluded_training_family_id: str,
) -> dict[str, object]:
    if set(source_direction_receipt) != _V20G_DIRECTION_KEYS:
        raise ValueError("V20g reflection source receipt shape drifted")
    validate_soft_polarity_direction_receipt(source_direction_receipt)
    held = source_direction_receipt.get("held_family_id")
    if held is None:
        raise ValueError("masked direction requires a V20g outer direction")
    source_training = tuple(
        _identifier(item, "source training family id")
        for item in _sequence(
            source_direction_receipt.get("training_family_ids"),
            "source training family ids",
        )
    )
    if len(source_training) != 7:
        raise ValueError("masked direction requires seven V20g training families")
    excluded = _identifier(
        excluded_training_family_id,
        "excluded training family id",
    )
    if excluded not in source_training:
        raise ValueError("excluded family is not a V20g training family")
    training = tuple(family for family in source_training if family != excluded)
    if len(training) != 6:
        raise ValueError("masked direction must retain exactly six training families")

    source_ids = _mapping(
        source_direction_receipt.get("training_example_ids_by_family"),
        "source training example ids",
    )
    source_hashes = _mapping(
        source_direction_receipt.get("training_gradient_sha256s_by_family"),
        "source training gradient hashes",
    )
    source_means = _mapping(
        source_direction_receipt.get("family_mean_gradients"),
        "source family mean gradients",
    )
    source_opgs = _mapping(
        source_direction_receipt.get("family_mean_opg_fishers"),
        "source family mean OPG Fishers",
    )
    example_ids = {
        family: tuple(
            _identifier(item, "training example id")
            for item in _sequence(source_ids[family], "training example ids")
        )
        for family in training
    }
    gradient_hashes = {
        family: tuple(
            _sha(item, "training gradient hash")
            for item in _sequence(source_hashes[family], "training gradient hashes")
        )
        for family in training
    }
    family_means = {
        family: _vector4(source_means[family], "family mean gradient")
        for family in training
    }
    family_opgs = {
        family: _matrix4(source_opgs[family], "family mean OPG Fisher")
        for family in training
    }
    mean_gradient = _mean_vectors([family_means[family] for family in training])
    fisher = _mean_matrices([family_opgs[family] for family in training])
    damping = max(
        _DAMPING_FLOOR,
        _DAMPING_FRACTION
        * math.fsum(fisher[index][index] for index in range(4))
        / 4.0,
    )
    damped = tuple(
        tuple(
            fisher[row][column] + (damping if row == column else 0.0)
            for column in range(4)
        )
        for row in range(4)
    )
    raw_direction = _solve_spd(damped, tuple(-value for value in mean_gradient))
    raw_derivative = math.fsum(
        mean_gradient[index] * raw_direction[index] for index in range(4)
    )
    raw_norm = math.sqrt(math.fsum(value * value for value in raw_direction))
    raw_corner_logits = _box_corner_logits(raw_direction)
    normalization_scale = max(abs(value) for value in raw_corner_logits)
    if not math.isfinite(normalization_scale) or normalization_scale <= 0.0:
        raise ValueError("masked direction has no finite box-logit scale")
    direction = tuple(
        0.0 if value == 0.0 else value / normalization_scale
        for value in raw_direction
    )
    derivative = raw_derivative / normalization_scale
    norm = raw_norm / normalization_scale
    normalized_corner_logits = tuple(
        0.0 if value == 0.0 else value / normalization_scale
        for value in raw_corner_logits
    )
    payload = {
        "protocol_sha256": SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256,
        "source_direction_protocol_sha256": SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "source_direction_artifact_sha256": _sha(
            source_direction_receipt.get("artifact_sha256"),
            "source direction artifact",
        ),
        "source_direction_receipt": dict(source_direction_receipt),
        "mask_semantics": (
            "computational_noninterference_with_full_source_provenance_binding"
        ),
        "excluded_family_numerical_summaries_retained": False,
        "full_source_receipt_embedded_for_provenance_only": True,
        "all_development_family_ids": tuple(
            _identifier(item, "development family id")
            for item in _sequence(
                source_direction_receipt.get("all_development_family_ids"),
                "development family ids",
            )
        ),
        "held_family_id": _identifier(held, "outer held family id"),
        "source_training_family_ids": source_training,
        "excluded_training_family_id": excluded,
        "training_family_ids": training,
        "gradient_evidence_sha256": _sha(
            source_direction_receipt.get("gradient_evidence_sha256"),
            "gradient evidence",
        ),
        "training_example_ids_by_family": example_ids,
        "training_gradient_sha256s_by_family": gradient_hashes,
        "family_mean_gradients": family_means,
        "family_mean_opg_fishers": family_opgs,
        "family_equal_mean_gradient": mean_gradient,
        "family_equal_opg_fisher": fisher,
        "damping": damping,
        "damped_fisher": damped,
        "raw_natural_direction": raw_direction,
        "raw_directional_derivative": raw_derivative,
        "raw_direction_l2_norm": raw_norm,
        "normalization": "exact_bilinear_box_corner_logit_max_abs",
        "normalization_corner_coordinates": _BOX_CORNERS,
        "raw_normalization_corner_logits": raw_corner_logits,
        "normalization_scale": normalization_scale,
        "natural_direction": direction,
        "directional_derivative": derivative,
        "direction_l2_norm": norm,
        "normalized_corner_logits": normalized_corner_logits,
        "normalized_box_logit_max_abs": max(
            abs(value) for value in normalized_corner_logits
        ),
        "finite": all(
            math.isfinite(value)
            for value in (
                *mean_gradient,
                *(value for row in fisher for value in row),
                *raw_direction,
                *direction,
                *normalized_corner_logits,
            )
        ),
        "strict_descent": (
            raw_derivative < 0.0
            and raw_norm > 0.0
            and derivative < 0.0
            and norm > 0.0
        ),
        "data_boundary": dict(_DATA_BOUNDARY),
    }
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_masked_direction.v20i",
        _MASKED_DIRECTION_DOMAIN,
        payload,
    )


def build_soft_polarity_masked_direction_receipt(
    source_direction_receipt: Mapping[str, object],
    excluded_training_family_id: str,
) -> dict[str, object]:
    """Recompute a six-family natural direction after one inner mask."""

    result = _expected_masked_direction_receipt(
        source_direction_receipt=source_direction_receipt,
        excluded_training_family_id=excluded_training_family_id,
    )
    validate_soft_polarity_masked_direction_receipt(
        result,
        source_direction_receipt=source_direction_receipt,
        expected_excluded_training_family_id=excluded_training_family_id,
    )
    return result


def validate_soft_polarity_masked_direction_receipt(
    value: Mapping[str, object],
    *,
    source_direction_receipt: Mapping[str, object] | None = None,
    expected_excluded_training_family_id: str | None = None,
) -> None:
    """Authenticate and replay a standalone six-family V20i direction."""

    receipt = _mapping(value, "masked direction receipt")
    embedded_source = _mapping(
        receipt.get("source_direction_receipt"),
        "embedded V20g source direction receipt",
    )
    if (
        source_direction_receipt is not None
        and _canonical(source_direction_receipt) != _canonical(embedded_source)
    ):
        raise ValueError("masked direction source receipt drifted")
    excluded = _identifier(
        receipt.get("excluded_training_family_id"),
        "excluded training family id",
    )
    if (
        expected_excluded_training_family_id is not None
        and excluded
        != _identifier(
            expected_excluded_training_family_id,
            "expected excluded training family id",
        )
    ):
        raise ValueError("masked direction excluded family differs")
    expected = _expected_masked_direction_receipt(
        source_direction_receipt=embedded_source,
        excluded_training_family_id=excluded,
    )
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("masked direction receipt content drifted")


def _validate_supported_direction_receipt(
    direction_receipt: Mapping[str, object],
) -> str:
    schema = direction_receipt.get("schema")
    if schema == "fisher_graph.complete_h4_soft_polarity_trust_region_direction.v20g":
        # V20g's historical validator authenticates known fields but permits a
        # newly re-hashed unknown top-level field.  V20i has a stronger data
        # boundary: the imported receipt may not carry unmodelled information
        # (in particular, held objectives) even when that information is
        # ignored by the reflection mathematics.
        if set(direction_receipt) != _V20G_DIRECTION_KEYS:
            raise ValueError("V20g reflection source receipt shape drifted")
        validate_soft_polarity_direction_receipt(direction_receipt)
        if (
            direction_receipt.get("held_family_id") is None
            or len(tuple(direction_receipt.get("training_family_ids", ()))) != 7
        ):
            raise ValueError("reflection fit requires a V20g outer direction")
        return SOFT_POLARITY_FIT_PROTOCOL_SHA256
    if schema == "fisher_graph.complete_h4_soft_polarity_masked_direction.v20i":
        validate_soft_polarity_masked_direction_receipt(direction_receipt)
        return SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
    raise ValueError("unsupported reflection direction receipt schema")


def _variant_spec(variant_id: str) -> tuple[int, int | None]:
    try:
        variant_index = SOFT_POLARITY_REFLECTION_VARIANTS.index(variant_id)
    except ValueError as error:
        raise ValueError("unknown reflection variant") from error
    reflected_coordinate_index = None if variant_index == 0 else variant_index - 1
    return variant_index, reflected_coordinate_index


def _direction_context(
    direction_receipt: Mapping[str, object],
) -> tuple[
    tuple[float, float, float, float],
    tuple[str, ...],
    dict[str, tuple[float, float, float, float]],
    str,
]:
    source_protocol_sha256 = _validate_supported_direction_receipt(
        direction_receipt
    )
    direction = _vector4(
        direction_receipt.get("natural_direction"), "source natural direction"
    )
    training = tuple(
        _identifier(item, "training family id")
        for item in _sequence(
            direction_receipt.get("training_family_ids"), "training family ids"
        )
    )
    if len(training) < 2 or len(set(training)) != len(training):
        raise ValueError("reflection fit requires distinct training families")
    raw_gradients = _mapping(
        direction_receipt.get("family_mean_gradients"), "family mean gradients"
    )
    if set(raw_gradients) != set(training):
        raise ValueError("reflection family-gradient geometry drifted")
    gradients = {
        family: _vector4(raw_gradients[family], "family mean gradient")
        for family in training
    }
    return direction, training, gradients, source_protocol_sha256


def _expected_variant_receipt(
    *,
    direction_receipt: Mapping[str, object],
    variant_id: str,
) -> dict[str, object]:
    (
        source_direction,
        training,
        gradients,
        source_protocol_sha256,
    ) = _direction_context(direction_receipt)
    variant_index, reflected_index = _variant_spec(variant_id)
    reflected = tuple(
        0.0
        if value == 0.0
        else (-value if index == reflected_index else value)
        for index, value in enumerate(source_direction)
    )
    raw_corner_logits = _box_corner_logits(reflected)  # type: ignore[arg-type]
    normalization_scale = max(abs(value) for value in raw_corner_logits)
    if not math.isfinite(normalization_scale) or normalization_scale <= 0.0:
        raise ValueError("reflection variant has no finite box-logit scale")
    normalized = tuple(
        0.0 if value == 0.0 else value / normalization_scale
        for value in reflected
    )
    normalized_corner_logits = tuple(
        0.0 if value == 0.0 else value / normalization_scale
        for value in raw_corner_logits
    )
    box_max = max(abs(value) for value in normalized_corner_logits)
    if box_max > 1.0 + 1.0e-12:
        raise ValueError("normalized reflection exceeds its unit box")

    derivatives = {
        family: math.fsum(
            gradients[family][index] * normalized[index] for index in range(4)
        )
        for family in training
    }
    ordered_worst = tuple(
        family
        for family, _ in sorted(
            derivatives.items(),
            key=lambda item: (-item[1], item[0]),
        )[:2]
    )
    cvar2 = _mean([derivatives[family] for family in ordered_worst])
    mean_derivative = _mean([derivatives[family] for family in training])
    negative_count = sum(value < 0.0 for value in derivatives.values())
    strict_mean_descent = mean_derivative < 0.0
    n_minus_one_family_descent = negative_count >= len(training) - 1
    admissible = strict_mean_descent and n_minus_one_family_descent
    payload = {
        "protocol_sha256": SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256,
        "source_direction_protocol_sha256": source_protocol_sha256,
        "source_direction_artifact_sha256": _sha(
            direction_receipt.get("artifact_sha256"), "source direction artifact"
        ),
        "held_family_id": direction_receipt.get("held_family_id"),
        "excluded_training_family_id": direction_receipt.get(
            "excluded_training_family_id"
        ),
        "training_family_ids": training,
        "feature_order": SOFT_POLARITY_REFLECTION_FEATURES,
        "variant_id": variant_id,
        "variant_index": variant_index,
        "reflected_coordinate_index": reflected_index,
        "source_direction": source_direction,
        "reflected_direction_before_normalization": reflected,
        "normalization": "exact_four_bilinear_box_corners_max_abs",
        "normalization_corner_coordinates": _BOX_CORNERS,
        "normalization_corner_logits_before_normalization": raw_corner_logits,
        "normalization_scale": normalization_scale,
        "normalized_direction": normalized,
        "normalized_corner_logits": normalized_corner_logits,
        "normalized_box_logit_max_abs": box_max,
        "family_mean_gradients": gradients,
        "family_directional_derivatives": derivatives,
        "cvar2_worst_family_ids": ordered_worst,
        "cvar2_directional_derivative": cvar2,
        "family_equal_mean_directional_derivative": mean_derivative,
        "negative_family_derivative_count": negative_count,
        "required_negative_family_derivative_count": len(training) - 1,
        "strict_mean_descent": strict_mean_descent,
        "n_minus_one_family_descent": n_minus_one_family_descent,
        "admissible": admissible,
        "finite": all(
            math.isfinite(value)
            for value in (
                *reflected,
                *raw_corner_logits,
                *normalized,
                *normalized_corner_logits,
                *derivatives.values(),
                cvar2,
                mean_derivative,
            )
        ),
        "data_boundary": dict(_DATA_BOUNDARY),
    }
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_reflection_variant.v20i",
        _VARIANT_DOMAIN,
        payload,
    )


def validate_soft_polarity_reflection_variant_receipt(
    value: Mapping[str, object],
    *,
    direction_receipt: Mapping[str, object],
) -> None:
    """Authenticate and replay one V20i reflection variant."""

    receipt = _mapping(value, "reflection variant receipt")
    variant_id = _identifier(receipt.get("variant_id"), "reflection variant id")
    expected = _expected_variant_receipt(
        direction_receipt=direction_receipt,
        variant_id=variant_id,
    )
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("reflection variant receipt content drifted")


def _selection_key(variant: Mapping[str, object]) -> tuple[float, float, int, str]:
    return (
        _number(
            variant.get("cvar2_directional_derivative"),
            "variant CVaR-2 derivative",
        ),
        _number(
            variant.get("family_equal_mean_directional_derivative"),
            "variant mean derivative",
        ),
        int(variant["variant_index"]),
        _sha(variant.get("artifact_sha256"), "variant artifact"),
    )


def _expected_fit_receipt(
    direction_receipt: Mapping[str, object],
) -> dict[str, object]:
    _, training, _, source_protocol_sha256 = _direction_context(direction_receipt)
    variants = tuple(
        _expected_variant_receipt(
            direction_receipt=direction_receipt,
            variant_id=variant_id,
        )
        for variant_id in SOFT_POLARITY_REFLECTION_VARIANTS
    )
    admissible = tuple(variant for variant in variants if variant["admissible"])
    ranking = tuple(sorted(admissible, key=_selection_key))
    selected = ranking[0] if ranking else None
    payload = {
        "protocol_sha256": SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256,
        "source_direction_protocol_sha256": source_protocol_sha256,
        "source_direction_artifact_sha256": _sha(
            direction_receipt.get("artifact_sha256"), "source direction artifact"
        ),
        "held_family_id": direction_receipt.get("held_family_id"),
        "excluded_training_family_id": direction_receipt.get(
            "excluded_training_family_id"
        ),
        "training_family_ids": training,
        "feature_order": SOFT_POLARITY_REFLECTION_FEATURES,
        "variant_order": SOFT_POLARITY_REFLECTION_VARIANTS,
        "variant_receipts": variants,
        "variant_artifact_sha256s_by_id": {
            str(variant["variant_id"]): variant["artifact_sha256"]
            for variant in variants
        },
        "admissible_variant_ids": tuple(
            str(variant["variant_id"])
            for variant in variants
            if variant["admissible"]
        ),
        "admissible_variant_ranking": tuple(
            str(variant["variant_id"]) for variant in ranking
        ),
        "selection_rule": (
            "minimum_cvar2_then_minimum_mean_then_fixed_variant_order_then_"
            "artifact_sha256_among_admissible_variants"
        ),
        "selected_variant_available": selected is not None,
        "selected_variant_id": (
            None if selected is None else selected["variant_id"]
        ),
        "selected_variant_index": (
            None if selected is None else selected["variant_index"]
        ),
        "selected_variant_artifact_sha256": (
            None if selected is None else selected["artifact_sha256"]
        ),
        "selected_normalized_direction": (
            None if selected is None else selected["normalized_direction"]
        ),
        "selected_cvar2_directional_derivative": (
            None if selected is None else selected["cvar2_directional_derivative"]
        ),
        "selected_family_equal_mean_directional_derivative": (
            None
            if selected is None
            else selected["family_equal_mean_directional_derivative"]
        ),
        "selected_negative_family_derivative_count": (
            None
            if selected is None
            else selected["negative_family_derivative_count"]
        ),
        "selection_frozen_from_training_gradients_only": True,
        "held_scores_consumed_before_selection": False,
        "serving_authorized": False,
        "data_boundary": dict(_DATA_BOUNDARY),
    }
    return _finish(
        "fisher_graph.complete_h4_soft_polarity_reflection_fit.v20i",
        _FIT_DOMAIN,
        payload,
    )


def build_soft_polarity_reflection_fit_receipt(
    *,
    direction_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Build the deterministic training-only V20i reflection selection."""

    result = _expected_fit_receipt(direction_receipt)
    validate_soft_polarity_reflection_fit_receipt(
        result,
        direction_receipt=direction_receipt,
    )
    return result


def validate_soft_polarity_reflection_fit_receipt(
    value: Mapping[str, object],
    *,
    direction_receipt: Mapping[str, object],
) -> None:
    """Authenticate and replay a complete V20i reflection-fit receipt."""

    receipt = _mapping(value, "reflection fit receipt")
    raw_variants = _sequence(receipt.get("variant_receipts"), "variant receipts")
    if len(raw_variants) != len(SOFT_POLARITY_REFLECTION_VARIANTS):
        raise ValueError("reflection fit variant ladder drifted")
    for raw in raw_variants:
        validate_soft_polarity_reflection_variant_receipt(
            _mapping(raw, "reflection variant receipt"),
            direction_receipt=direction_receipt,
        )
    expected = _expected_fit_receipt(direction_receipt)
    if _canonical(receipt) != _canonical(expected):
        raise ValueError("reflection fit receipt content drifted")
