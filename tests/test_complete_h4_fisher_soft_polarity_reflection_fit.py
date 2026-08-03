from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math

import pytest

from fisher_graph.complete_h4_fisher_soft_polarity_reflection_fit import (
    SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256,
    SOFT_POLARITY_REFLECTION_FEATURES,
    SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256,
    SOFT_POLARITY_REFLECTION_VARIANTS,
    build_soft_polarity_masked_direction_receipt,
    build_soft_polarity_reflection_fit_receipt,
    validate_soft_polarity_masked_direction_receipt,
    validate_soft_polarity_reflection_fit_receipt,
    validate_soft_polarity_reflection_variant_receipt,
)
from fisher_graph.complete_h4_fisher_soft_polarity_trust_region_fit import (
    SOFT_POLARITY_FIT_PROTOCOL_SHA256,
    build_soft_polarity_direction_receipt,
)
from fisher_graph import complete_h4_fisher_soft_polarity_trust_region_fit as v20g


FAMILIES = tuple(f"development_family_{index}" for index in range(8))
BOX_CORNERS = ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
EXPECTED_MASKED_DIRECTION_PROTOCOL_SHA256 = (
    "4205e43f62b8736c70162e184d7e6b8e191c9fe748ff94de7a22ce752cc333ae"
)
EXPECTED_REFLECTION_FIT_PROTOCOL_SHA256 = (
    "0b6af60841f2fad2c471c13f875ea438778383c42a06127fd735baaf7ab6c245"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _direction(
    held: str = FAMILIES[0],
    *,
    perturb_family: str | None = None,
):
    rows = {}
    training = tuple(family for family in FAMILIES if family != held)
    for index, family in enumerate(training):
        rows[family] = {
            f"{family}:0": (
                1.0 + 0.02 * index,
                0.5 - 0.01 * index,
                -0.3 + 0.02 * index,
                0.2 + 0.01 * index,
            ),
            f"{family}:1": (
                0.8 + 0.01 * index,
                0.3 + 0.02 * index,
                -0.15 - 0.01 * index,
                0.1 + 0.03 * index,
            ),
        }
        if family == perturb_family:
            rows[family][f"{family}:0"] = tuple(
                value * 7.0 + coordinate
                for coordinate, value in enumerate(rows[family][f"{family}:0"])
            )
    return build_soft_polarity_direction_receipt(
        source_artifact_sha256s={"source": _sha(f"source:{held}")},
        all_development_family_ids=FAMILIES,
        held_family_id=held,
        gradient_rows_by_family=rows,
        gradient_evidence_sha256=_sha(f"gradient-evidence:{held}"),
    )


def _direction_from_vectors(vectors):
    training = FAMILIES[1:]
    return build_soft_polarity_direction_receipt(
        source_artifact_sha256s={"source": _sha(f"vectors:{vectors!r}")},
        all_development_family_ids=FAMILIES,
        held_family_id=FAMILIES[0],
        gradient_rows_by_family={
            family: {f"{family}:0": vector}
            for family, vector in zip(training, vectors, strict=True)
        },
        gradient_evidence_sha256=_sha(f"vector-evidence:{vectors!r}"),
    )


def _manual_corners(coefficients):
    return tuple(
        math.fsum(
            (
                coefficients[0],
                coefficients[1] * c1,
                coefficients[2] * c2,
                coefficients[3] * c1 * c2,
            )
        )
        for c1, c2 in BOX_CORNERS
    )


def test_reflection_fit_has_fixed_five_variant_four_feature_geometry() -> None:
    direction = _direction()
    fit = build_soft_polarity_reflection_fit_receipt(
        direction_receipt=direction
    )

    validate_soft_polarity_reflection_fit_receipt(
        fit,
        direction_receipt=direction,
    )
    assert fit["protocol_sha256"] == (
        SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
    )
    assert SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256 == (
        EXPECTED_REFLECTION_FIT_PROTOCOL_SHA256
    )
    assert SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256 == (
        EXPECTED_MASKED_DIRECTION_PROTOCOL_SHA256
    )
    assert fit["source_direction_protocol_sha256"] == (
        SOFT_POLARITY_FIT_PROTOCOL_SHA256
    )
    assert tuple(fit["feature_order"]) == SOFT_POLARITY_REFLECTION_FEATURES
    assert tuple(fit["variant_order"]) == SOFT_POLARITY_REFLECTION_VARIANTS
    assert tuple(
        variant["variant_id"] for variant in fit["variant_receipts"]
    ) == SOFT_POLARITY_REFLECTION_VARIANTS
    assert fit["selection_frozen_from_training_gradients_only"] is True
    assert fit["held_scores_consumed_before_selection"] is False
    assert fit["serving_authorized"] is False


def test_each_variant_is_one_flip_with_exact_four_corner_normalization() -> None:
    direction = _direction()
    fit = build_soft_polarity_reflection_fit_receipt(
        direction_receipt=direction
    )
    source = tuple(direction["natural_direction"])

    for variant_index, variant in enumerate(fit["variant_receipts"]):
        validate_soft_polarity_reflection_variant_receipt(
            variant,
            direction_receipt=direction,
        )
        reflected_coordinate = None if variant_index == 0 else variant_index - 1
        expected_reflected = tuple(
            -value if index == reflected_coordinate else value
            for index, value in enumerate(source)
        )
        assert tuple(variant["reflected_direction_before_normalization"]) == (
            pytest.approx(expected_reflected)
        )
        assert tuple(variant["normalization_corner_coordinates"]) == BOX_CORNERS
        assert tuple(
            variant["normalization_corner_logits_before_normalization"]
        ) == pytest.approx(_manual_corners(expected_reflected))
        normalized = tuple(variant["normalized_direction"])
        assert tuple(variant["normalized_corner_logits"]) == pytest.approx(
            _manual_corners(normalized)
        )
        assert variant["normalized_box_logit_max_abs"] == pytest.approx(1.0)


def test_family_derivatives_cvar2_admissibility_and_selection_replay() -> None:
    direction = _direction()
    fit = build_soft_polarity_reflection_fit_receipt(
        direction_receipt=direction
    )
    training = tuple(direction["training_family_ids"])
    gradients = direction["family_mean_gradients"]

    for variant in fit["variant_receipts"]:
        normalized = tuple(variant["normalized_direction"])
        expected_derivatives = {
            family: math.fsum(
                gradients[family][index] * normalized[index]
                for index in range(4)
            )
            for family in training
        }
        assert variant["family_directional_derivatives"] == pytest.approx(
            expected_derivatives
        )
        ordered_worst = sorted(
            expected_derivatives,
            key=lambda family: (-expected_derivatives[family], family),
        )[:2]
        expected_cvar2 = math.fsum(
            expected_derivatives[family] for family in ordered_worst
        ) / 2.0
        expected_mean = math.fsum(expected_derivatives.values()) / len(training)
        expected_negative_count = sum(
            value < 0.0 for value in expected_derivatives.values()
        )
        assert tuple(variant["cvar2_worst_family_ids"]) == tuple(ordered_worst)
        assert variant["cvar2_directional_derivative"] == pytest.approx(
            expected_cvar2
        )
        assert variant["family_equal_mean_directional_derivative"] == (
            pytest.approx(expected_mean)
        )
        assert variant["negative_family_derivative_count"] == (
            expected_negative_count
        )
        assert variant["admissible"] is (
            expected_mean < 0.0
            and expected_negative_count >= len(training) - 1
        )

    admissible = [
        variant for variant in fit["variant_receipts"] if variant["admissible"]
    ]
    expected_ranking = sorted(
        admissible,
        key=lambda variant: (
            variant["cvar2_directional_derivative"],
            variant["family_equal_mean_directional_derivative"],
            variant["variant_index"],
            variant["artifact_sha256"],
        ),
    )
    assert tuple(fit["admissible_variant_ranking"]) == tuple(
        variant["variant_id"] for variant in expected_ranking
    )
    assert fit["selected_variant_id"] == expected_ranking[0]["variant_id"]
    assert fit["selected_variant_id"] == "flip_coordinate_1"
    assert tuple(fit["selected_normalized_direction"]) == pytest.approx(
        expected_ranking[0]["normalized_direction"]
    )


def test_receipt_is_canonical_across_json_round_trip_and_rejects_tamper() -> None:
    direction = _direction()
    fit = build_soft_polarity_reflection_fit_receipt(
        direction_receipt=direction
    )
    round_tripped_direction = json.loads(json.dumps(direction))
    round_tripped_fit = json.loads(json.dumps(fit))

    validate_soft_polarity_reflection_fit_receipt(
        round_tripped_fit,
        direction_receipt=round_tripped_direction,
    )
    assert round_tripped_fit["artifact_sha256"] == fit["artifact_sha256"]

    tampered = copy.deepcopy(fit)
    tampered["variant_receipts"][0]["cvar2_directional_derivative"] += 1.0e-6
    with pytest.raises(ValueError, match="variant receipt content drifted"):
        validate_soft_polarity_reflection_fit_receipt(
            tampered,
            direction_receipt=direction,
        )


def test_fixed_variant_order_breaks_exact_metric_ties() -> None:
    direction = _direction_from_vectors(((1.0, 0.0, 0.0, 0.0),) * 7)
    fit = build_soft_polarity_reflection_fit_receipt(
        direction_receipt=direction
    )

    by_id = {
        variant["variant_id"]: variant for variant in fit["variant_receipts"]
    }
    tied = ("identity", "flip_coordinate_1", "flip_coordinate_2", "flip_coordinate_3")
    assert len(
        {
            (
                by_id[variant]["cvar2_directional_derivative"],
                by_id[variant]["family_equal_mean_directional_derivative"],
            )
            for variant in tied
        }
    ) == 1
    assert fit["selected_variant_id"] == "identity"
    assert tuple(fit["admissible_variant_ranking"])[:4] == tied
    assert all(
        not (value == 0.0 and math.copysign(1.0, value) < 0.0)
        for variant in fit["variant_receipts"]
        for value in variant["reflected_direction_before_normalization"]
    )


def test_no_admissible_variant_is_a_hashed_non_authorizing_result() -> None:
    direction = _direction_from_vectors(
        (
            (2.0, 0.0, 0.0, 2.0),
            (-2.0, 1.0, -1.0, -2.0),
            (-1.0, -2.0, 0.0, 1.0),
            (-1.0, 1.0, 2.0, -2.0),
            (2.0, -1.0, -2.0, -1.0),
            (1.0, 0.0, -1.0, 1.0),
            (-1.0, -2.0, -1.0, 2.0),
        )
    )
    fit = build_soft_polarity_reflection_fit_receipt(
        direction_receipt=direction
    )

    validate_soft_polarity_reflection_fit_receipt(
        fit,
        direction_receipt=direction,
    )
    assert fit["admissible_variant_ids"] == ()
    assert fit["selected_variant_available"] is False
    assert fit["selected_variant_id"] is None
    assert fit["selected_normalized_direction"] is None
    assert fit["serving_authorized"] is False


def test_receipt_rejects_cross_direction_replay() -> None:
    direction = _direction(FAMILIES[0])
    other_direction = _direction(FAMILIES[1])
    fit = build_soft_polarity_reflection_fit_receipt(
        direction_receipt=direction
    )

    with pytest.raises(ValueError, match="variant receipt content drifted"):
        validate_soft_polarity_reflection_fit_receipt(
            fit,
            direction_receipt=other_direction,
        )


def test_v20i_rejects_rehashed_unknown_v20g_source_fields() -> None:
    direction = copy.deepcopy(_direction())
    direction["held_objectives_by_family"] = {FAMILIES[0]: 123.0}
    direction["artifact_sha256"] = v20g._hash(
        v20g._DIRECTION_DOMAIN,
        {
            key: value
            for key, value in direction.items()
            if key != "artifact_sha256"
        },
    )

    # This documents the permissive historical boundary that V20i hardens.
    v20g.validate_soft_polarity_direction_receipt(direction)
    with pytest.raises(ValueError, match="source receipt shape drifted"):
        build_soft_polarity_reflection_fit_receipt(
            direction_receipt=direction
        )
    with pytest.raises(ValueError, match="source receipt shape drifted"):
        build_soft_polarity_masked_direction_receipt(
            direction,
            direction["training_family_ids"][0],
        )


def test_masked_direction_recomputes_exact_six_family_natural_system() -> None:
    source = _direction()
    excluded = source["training_family_ids"][2]
    masked = build_soft_polarity_masked_direction_receipt(source, excluded)

    validate_soft_polarity_masked_direction_receipt(
        masked,
        source_direction_receipt=source,
    )
    retained = tuple(
        family
        for family in source["training_family_ids"]
        if family != excluded
    )
    assert masked["protocol_sha256"] == (
        SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
    )
    assert masked["source_direction_artifact_sha256"] == source["artifact_sha256"]
    assert masked["held_family_id"] == source["held_family_id"]
    assert masked["excluded_training_family_id"] == excluded
    assert tuple(masked["training_family_ids"]) == retained
    assert len(retained) == 6
    assert excluded not in masked["family_mean_gradients"]

    expected_mean = tuple(
        math.fsum(
            source["family_mean_gradients"][family][coordinate]
            for family in retained
        )
        / len(retained)
        for coordinate in range(4)
    )
    expected_fisher = tuple(
        tuple(
            math.fsum(
                source["family_mean_opg_fishers"][family][row][column]
                for family in retained
            )
            / len(retained)
            for column in range(4)
        )
        for row in range(4)
    )
    expected_damping = max(
        1.0e-12,
        1.0e-3
        * math.fsum(expected_fisher[index][index] for index in range(4))
        / 4.0,
    )
    assert tuple(masked["family_equal_mean_gradient"]) == pytest.approx(
        expected_mean
    )
    for observed, expected in zip(
        masked["family_equal_opg_fisher"],
        expected_fisher,
        strict=True,
    ):
        assert tuple(observed) == pytest.approx(expected)
    assert masked["damping"] == pytest.approx(expected_damping)

    raw_direction = tuple(masked["raw_natural_direction"])
    damped = masked["damped_fisher"]
    for row in range(4):
        assert math.fsum(
            damped[row][column] * raw_direction[column]
            for column in range(4)
        ) == pytest.approx(-expected_mean[row], abs=1.0e-11)
    normalized = tuple(masked["natural_direction"])
    assert tuple(masked["normalized_corner_logits"]) == pytest.approx(
        _manual_corners(normalized)
    )
    assert masked["normalized_box_logit_max_abs"] == pytest.approx(1.0)
    assert masked["strict_descent"] is True


def test_masked_direction_is_standalone_tamper_evident_and_source_bound() -> None:
    source = _direction(FAMILIES[0])
    excluded = source["training_family_ids"][0]
    masked = build_soft_polarity_masked_direction_receipt(source, excluded)
    round_tripped = json.loads(json.dumps(masked))

    validate_soft_polarity_masked_direction_receipt(round_tripped)
    assert round_tripped["artifact_sha256"] == masked["artifact_sha256"]

    tampered = copy.deepcopy(masked)
    retained = tampered["training_family_ids"][0]
    changed_gradient = list(tampered["family_mean_gradients"][retained])
    changed_gradient[0] += 1.0e-6
    tampered["family_mean_gradients"][retained] = changed_gradient
    with pytest.raises(ValueError, match="masked direction receipt content drifted"):
        validate_soft_polarity_masked_direction_receipt(tampered)

    with pytest.raises(ValueError, match="source receipt drifted"):
        validate_soft_polarity_masked_direction_receipt(
            masked,
            source_direction_receipt=_direction(FAMILIES[1]),
        )
    with pytest.raises(ValueError, match="not a V20g training family"):
        build_soft_polarity_masked_direction_receipt(
            source,
            source["held_family_id"],
        )


def test_masked_direction_rejects_cross_slot_replay() -> None:
    source = _direction(FAMILIES[0])
    first, second = source["training_family_ids"][:2]
    masked = build_soft_polarity_masked_direction_receipt(source, first)

    validate_soft_polarity_masked_direction_receipt(
        masked,
        source_direction_receipt=source,
        expected_excluded_training_family_id=first,
    )
    with pytest.raises(ValueError, match="excluded family differs"):
        validate_soft_polarity_masked_direction_receipt(
            masked,
            source_direction_receipt=source,
            expected_excluded_training_family_id=second,
        )


def test_mask_has_numerical_noninterference_with_provenance_binding() -> None:
    source = _direction(FAMILIES[0])
    excluded = source["training_family_ids"][2]
    changed_source = _direction(FAMILIES[0], perturb_family=excluded)
    masked = build_soft_polarity_masked_direction_receipt(source, excluded)
    changed_masked = build_soft_polarity_masked_direction_receipt(
        changed_source,
        excluded,
    )
    fit = build_soft_polarity_reflection_fit_receipt(
        direction_receipt=masked
    )
    changed_fit = build_soft_polarity_reflection_fit_receipt(
        direction_receipt=changed_masked
    )

    assert masked["artifact_sha256"] != changed_masked["artifact_sha256"]
    assert fit["artifact_sha256"] != changed_fit["artifact_sha256"]
    for field in (
        "family_equal_mean_gradient",
        "family_equal_opg_fisher",
        "damping",
        "damped_fisher",
        "raw_natural_direction",
        "raw_directional_derivative",
        "raw_direction_l2_norm",
        "natural_direction",
        "directional_derivative",
        "direction_l2_norm",
        "normalized_corner_logits",
    ):
        assert masked[field] == changed_masked[field]
    assert masked["mask_semantics"] == (
        "computational_noninterference_with_full_source_provenance_binding"
    )
    assert masked["excluded_family_numerical_summaries_retained"] is False
    assert masked["full_source_receipt_embedded_for_provenance_only"] is True
    assert fit["selected_variant_id"] == changed_fit["selected_variant_id"]
    assert fit["selected_normalized_direction"] == changed_fit[
        "selected_normalized_direction"
    ]
    for left, right in zip(
        fit["variant_receipts"],
        changed_fit["variant_receipts"],
        strict=True,
    ):
        for field in (
            "normalized_direction",
            "family_directional_derivatives",
            "cvar2_directional_derivative",
            "family_equal_mean_directional_derivative",
            "negative_family_derivative_count",
            "admissible",
        ):
            assert left[field] == right[field]


def test_reflection_fit_schema_dispatch_accepts_masked_direction() -> None:
    source = _direction()
    masked = build_soft_polarity_masked_direction_receipt(
        source,
        source["training_family_ids"][1],
    )
    fit = build_soft_polarity_reflection_fit_receipt(
        direction_receipt=masked
    )

    validate_soft_polarity_reflection_fit_receipt(
        fit,
        direction_receipt=masked,
    )
    assert fit["source_direction_protocol_sha256"] == (
        SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
    )
    assert fit["source_direction_artifact_sha256"] == masked["artifact_sha256"]
    assert tuple(fit["training_family_ids"]) == tuple(
        masked["training_family_ids"]
    )
    assert all(
        variant["required_negative_family_derivative_count"] == 5
        for variant in fit["variant_receipts"]
    )


def test_builder_api_cannot_accept_held_scores_or_prompt_data() -> None:
    signature = inspect.signature(build_soft_polarity_reflection_fit_receipt)

    assert tuple(signature.parameters) == ("direction_receipt",)
    with pytest.raises(TypeError, match="unexpected keyword"):
        build_soft_polarity_reflection_fit_receipt(
            direction_receipt=_direction(),
            held_objectives_by_family={},  # type: ignore[call-arg]
        )
