"""Pure V20c continuous-response law and development-sentinel receipts.

V20c is deliberately a *development-only* experiment on the reused A16
panel.  It cannot produce a fresh family-disjoint validation claim.  The
module contains scalar mathematics and authenticated provenance only: no
model, tensor, prompt, objective fitting, or runtime routing code lives here.

The selected law is a smooth, zero-safe signed-log response

``lambda * asinh(9 z) / asinh(9)``

where ``z`` is the selected raw bounded Fisher coordinate in ``[-1, 1]``.
Fit-only center and scale are recorded as diagnostics but are never applied,
so the existing router origin and domain remain exact.  A linear law, the
constant ``+1`` V20b endpoint, and the exact negative of the selected law are
compulsory controls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
import sys

__all__ = [
    "CONTINUOUS_RESPONSE_ARMS",
    "CONTINUOUS_RESPONSE_KAPPA",
    "CONTINUOUS_RESPONSE_PROTOCOL_SHA256",
    "build_continuous_response_arm_score",
    "build_continuous_response_law_receipt",
    "build_continuous_response_role_receipt",
    "build_continuous_response_sentinel_receipt",
    "continuous_response_value",
    "evaluate_continuous_response_qualification",
    "fit_coordinate_statistics",
    "select_strongest_fisher_coordinate",
    "signed_log",
    "validate_continuous_response_arm_score",
    "validate_continuous_response_law_receipt",
    "validate_continuous_response_role_receipt",
    "validate_continuous_response_sentinel_receipt",
]


CONTINUOUS_RESPONSE_KAPPA = 9.0
CONTINUOUS_RESPONSE_ARMS = (
    "base",
    "signed_log",
    "constant_plus_one",
    "signed_log_sign_flip",
    "linear",
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_FAMILY_COUNT = 8
_ROLE_COUNT = 7
_EXCLUDED_COUNT = 2
_MATERIALITY = 0.01
_BASE_WINS = 6
_CONSTANT_WINS = 5
_MIRROR_WINS = 6
_WORST_REGRESSION = 0.02
_SCALE_FLOOR = 1.0e-12
_ABSOLUTE_IMPROVEMENT_FLOOR = 1.0e-12
_ROUNDOFF_MULTIPLIER = 128.0

_PROTOCOL_DOMAIN = b"fisher-graph:complete-h4-continuous-response:protocol:v20c\0"
_COORDINATE_DOMAIN = b"fisher-graph:complete-h4-continuous-response:coordinates:v20c\0"
_LAW_DOMAIN = b"fisher-graph:complete-h4-continuous-response:law:v20c\0"
_ARM_DEFINITION_DOMAIN = b"fisher-graph:complete-h4-continuous-response:arm-definition:v20c\0"
_ARM_SCORE_DOMAIN = b"fisher-graph:complete-h4-continuous-response:arm-score:v20c\0"
_ROLE_DOMAIN = b"fisher-graph:complete-h4-continuous-response:role:v20c\0"
_QUALIFICATION_DOMAIN = b"fisher-graph:complete-h4-continuous-response:qualification:v20c\0"
_SENTINEL_DOMAIN = b"fisher-graph:complete-h4-continuous-response:sentinel:v20c\0"

_FIXED_PROTOCOL = {
    "protocol": "fit_only_fisher_continuous_response_sentinel_v20c",
    "scientific_status": "development_only_reused_a16",
    "fresh_family_disjoint_claim_authorized": False,
    "panel": "reused_A16",
    "response_family": "lambda_times_asinh_normalized",
    "signed_log_lambda": 1.0,
    "linear_lambda": 1.0,
    "lambda_status": "fixed_not_fitted_in_sentinel",
    "kappa": CONTINUOUS_RESPONSE_KAPPA,
    "coordinate_selection": "largest_fit_only_family_equal_population_variance_lowest_index_tie",
    "coordinate_statistics": "fit_only_mean_and_population_std_diagnostic_only",
    "response_coordinate_application": "raw_selected_bounded_coordinate_no_center_scale_or_clamp",
    "selection_may_use_objectives": False,
    "family_ids_define_equal_weight_groups": True,
    "family_labels_enter_the_numeric_response": False,
    "held_data_may_define_statistics": False,
    "controls": CONTINUOUS_RESPONSE_ARMS,
    "mirror": "exact_negative_of_signed_log_response",
    "aggregation": "family_equal_seven_roles",
    "base_materiality": _MATERIALITY,
    "base_wins": _BASE_WINS,
    "constant_wins": _CONSTANT_WINS,
    "mirror_wins": _MIRROR_WINS,
    "worst_regression_maximum": _WORST_REGRESSION,
    "numerical_floor": "max_1e-12_128_float64_eps_times_reference",
    "score_source": "exact_finite_execution_only",
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


CONTINUOUS_RESPONSE_PROTOCOL_SHA256 = _hash(_PROTOCOL_DOMAIN, _FIXED_PROTOCOL)


def _finish(schema: str, domain: bytes, payload: Mapping[str, object]) -> dict[str, object]:
    result = {"schema": schema, **dict(payload)}
    result["artifact_sha256"] = _hash(domain, result)
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _exact(value: Mapping[str, object], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields differ")


def _same(left: object, right: object, *, label: str) -> None:
    if _canonical(left) != _canonical(right):
        raise ValueError(f"{label} receipt drifted")


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty canonical string")
    return value


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _number(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _unit_lambda(value: object, *, label: str) -> float:
    selected = _number(value, label=label)
    if selected != 1.0:
        raise ValueError(f"{label} is fixed at exactly 1 for the V20c sentinel")
    return selected


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be boolean")
    return value


def _families(value: object, *, label: str, count: int | None = None) -> tuple[str, ...]:
    result = tuple(sorted(_identifier(item, label=label) for item in _sequence(value, label=label)))
    if len(set(result)) != len(result) or (count is not None and len(result) != count):
        raise ValueError(f"{label} geometry differs")
    return result


def _sources(value: object) -> dict[str, str]:
    source = _mapping(value, label="V20b source hashes")
    if not source:
        raise ValueError("V20b source hashes must not be empty")
    return dict(
        sorted(
            (
                _identifier(key, label="V20b source name"),
                _sha(item, label=f"V20b source {key}"),
            )
            for key, item in source.items()
        )
    )


def _matrix(value: object, *, minimum_rows: int = 1) -> tuple[tuple[float, ...], ...]:
    rows = _sequence(value, label="fit coordinates")
    if len(rows) < minimum_rows:
        raise ValueError(f"fit coordinates require at least {minimum_rows} rows")
    result = tuple(
        tuple(_number(item, label="fit coordinate") for item in _sequence(row, label="fit coordinate row"))
        for row in rows
    )
    width = len(result[0])
    if width == 0 or any(len(row) != width for row in result):
        raise ValueError("fit coordinate matrix geometry differs")
    if any(abs(item) > 1.0 for row in result for item in row):
        raise ValueError("fit Fisher coordinates must remain in the raw bounded [-1, 1] domain")
    return result


def _family_matrices(value: object) -> tuple[tuple[tuple[float, ...], ...], ...]:
    families = tuple(
        _matrix(item)
        for item in _sequence(value, label="fit-family coordinate groups")
    )
    if len(families) < 2:
        raise ValueError("family-equal statistics require at least two fit families")
    width = len(families[0][0])
    if any(len(row) != width for family in families for row in family):
        raise ValueError("fit-family coordinate widths differ")
    return families


def _family_equal_moments(
    families: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    family_count = len(families)
    width = len(families[0][0])
    family_means = tuple(
        tuple(
            math.fsum(row[index] for row in family) / len(family)
            for index in range(width)
        )
        for family in families
    )
    means = tuple(
        math.fsum(value[index] for value in family_means) / family_count
        for index in range(width)
    )
    variances = tuple(
        math.fsum(
            math.fsum((row[index] - means[index]) ** 2 for row in family)
            / len(family)
            for family in families
        )
        / family_count
        for index in range(width)
    )
    return means, variances


def signed_log(value: float) -> float:
    """Return the fixed smooth odd V20c signed-log transform."""

    selected = _number(value, label="signed-log input")
    return math.asinh(CONTINUOUS_RESPONSE_KAPPA * selected) / math.asinh(
        CONTINUOUS_RESPONSE_KAPPA
    )


def select_strongest_fisher_coordinate(
    fit_family_coordinates: Sequence[Sequence[Sequence[float]]],
) -> int:
    """Select by fit-only coordinate variance; exact ties use lowest index.

    The API accepts coordinates only.  Objectives, family identifiers, and
    held-role metadata therefore cannot influence this selector.
    """

    families = _family_matrices(fit_family_coordinates)
    _means, variances = _family_equal_moments(families)
    return max(range(len(variances)), key=lambda index: (variances[index], -index))


def fit_coordinate_statistics(
    fit_family_coordinates: Sequence[Sequence[Sequence[float]]],
) -> dict[str, object]:
    """Compute deterministic family-equal fit-only diagnostic statistics."""

    families = _family_matrices(fit_family_coordinates)
    means, variances = _family_equal_moments(families)
    selected = max(range(len(variances)), key=lambda index: (variances[index], -index))
    center = means[selected]
    raw_scale = math.sqrt(variances[selected])
    scale = max(raw_scale, _SCALE_FLOOR)
    return {
        "coordinate_family_count": len(families),
        "coordinate_family_row_counts": tuple(len(family) for family in families),
        "coordinate_row_count": sum(len(family) for family in families),
        "coordinate_width": len(families[0][0]),
        "coordinate_means": means,
        "coordinate_variances": variances,
        "selected_coordinate_index": selected,
        "selected_coordinate_variance": variances[selected],
        "selected_coordinate_center": center,
        "selected_coordinate_scale": scale,
        "coordinate_scale_was_floored": raw_scale < _SCALE_FLOOR,
        "fit_coordinate_rows_sha256": _hash(_COORDINATE_DOMAIN, families),
    }


def _validated_statistics(value: Mapping[str, object]) -> dict[str, object]:
    _exact(
        value,
        {
            "coordinate_row_count",
            "coordinate_family_count",
            "coordinate_family_row_counts",
            "coordinate_width",
            "coordinate_means",
            "coordinate_variances",
            "selected_coordinate_index",
            "selected_coordinate_variance",
            "selected_coordinate_center",
            "selected_coordinate_scale",
            "coordinate_scale_was_floored",
            "fit_coordinate_rows_sha256",
        },
        label="coordinate statistics",
    )
    family_count = _integer(value["coordinate_family_count"], label="coordinate family count", minimum=2)
    family_row_counts = tuple(
        _integer(item, label="coordinate family row count", minimum=1)
        for item in _sequence(value["coordinate_family_row_counts"], label="coordinate family row counts")
    )
    if len(family_row_counts) != family_count:
        raise ValueError("coordinate family row counts differ")
    row_count = _integer(value["coordinate_row_count"], label="coordinate row count", minimum=2)
    if row_count != sum(family_row_counts):
        raise ValueError("coordinate row count differs from family rows")
    width = _integer(value["coordinate_width"], label="coordinate width", minimum=1)
    means = tuple(
        _number(item, label="coordinate mean")
        for item in _sequence(value["coordinate_means"], label="coordinate means")
    )
    variances = tuple(
        _number(item, label="coordinate variance")
        for item in _sequence(value["coordinate_variances"], label="coordinate variances")
    )
    if len(means) != width or len(variances) != width or any(item < 0.0 for item in variances):
        raise ValueError("coordinate variances differ")
    selected = _integer(value["selected_coordinate_index"], label="selected coordinate index")
    if selected >= width or selected != max(range(width), key=lambda index: (variances[index], -index)):
        raise ValueError("selected coordinate is not the strongest fit-only variance")
    selected_variance = _number(value["selected_coordinate_variance"], label="selected coordinate variance")
    if selected_variance != variances[selected]:
        raise ValueError("selected coordinate variance differs")
    center = _number(value["selected_coordinate_center"], label="coordinate center")
    if center != means[selected]:
        raise ValueError("selected coordinate center differs")
    scale = _number(value["selected_coordinate_scale"], label="coordinate scale", positive=True)
    expected_scale = max(math.sqrt(selected_variance), _SCALE_FLOOR)
    if scale != expected_scale:
        raise ValueError("coordinate scale differs from fit-only variance")
    floored = _boolean(value["coordinate_scale_was_floored"], label="coordinate scale floor marker")
    if floored != (math.sqrt(selected_variance) < _SCALE_FLOOR):
        raise ValueError("coordinate scale floor marker differs")
    return {
        "coordinate_family_count": family_count,
        "coordinate_family_row_counts": family_row_counts,
        "coordinate_row_count": row_count,
        "coordinate_width": width,
        "coordinate_means": means,
        "coordinate_variances": variances,
        "selected_coordinate_index": selected,
        "selected_coordinate_variance": selected_variance,
        "selected_coordinate_center": center,
        "selected_coordinate_scale": scale,
        "coordinate_scale_was_floored": floored,
        "fit_coordinate_rows_sha256": _sha(value["fit_coordinate_rows_sha256"], label="fit coordinate rows"),
    }


_LAW_KEYS = {
    "schema",
    "protocol_sha256",
    "scientific_status",
    "fresh_family_disjoint_claim_authorized",
    "v20b_source_sha256s",
    "family_ids",
    "outer_held_family_id",
    "held_family_id",
    "excluded_family_ids",
    "fit_family_ids",
    "coordinate_source_family_ids",
    "coordinate_statistics_scope",
    "held_data_used_for_statistics_or_fit",
    "objectives_used_for_coordinate_selection",
    "family_labels_used_for_coordinate_selection",
    "base_provider_artifact_sha256",
    "proposal_provider_artifact_sha256",
    "lambda_fit_evidence_sha256",
    "response_family",
    "response_coordinate_application",
    "coordinate_statistics_usage",
    "kappa",
    "signed_log_lambda",
    "linear_lambda",
    "coordinate_statistics",
    "raw_coordinates_or_objectives_serialized",
    "artifact_sha256",
}


def _build_law_from_statistics(
    *,
    v20b_source_sha256s: Mapping[str, str],
    family_ids: Sequence[str],
    outer_held_family_id: str,
    held_family_id: str,
    coordinate_source_family_ids: Sequence[str],
    base_provider_artifact_sha256: str,
    proposal_provider_artifact_sha256: str,
    lambda_fit_evidence_sha256: str,
    signed_log_lambda: float,
    linear_lambda: float,
    coordinate_statistics: Mapping[str, object],
    held_data_used_for_statistics_or_fit: bool,
    objectives_used_for_coordinate_selection: bool,
    family_labels_used_for_coordinate_selection: bool,
) -> dict[str, object]:
    sources = _sources(v20b_source_sha256s)
    families = _families(family_ids, label="A16 family IDs", count=_FAMILY_COUNT)
    outer = _identifier(outer_held_family_id, label="outer held family")
    held = _identifier(held_family_id, label="held family")
    if outer == held or outer not in families or held not in families:
        raise ValueError("outer and held family geometry differs")
    excluded = tuple(sorted((outer, held)))
    fit_families = tuple(item for item in families if item not in excluded)
    coordinate_families = _families(
        coordinate_source_family_ids,
        label="coordinate source families",
        count=_FAMILY_COUNT - _EXCLUDED_COUNT,
    )
    if coordinate_families != fit_families:
        raise ValueError("coordinate statistics must use exactly the six fit families")
    if _boolean(held_data_used_for_statistics_or_fit, label="held-data marker"):
        raise ValueError("held data may not define continuous-response statistics or fit")
    if _boolean(objectives_used_for_coordinate_selection, label="objective-selection marker"):
        raise ValueError("objectives may not select the Fisher coordinate")
    if _boolean(family_labels_used_for_coordinate_selection, label="family-selection marker"):
        raise ValueError("family labels may not select the Fisher coordinate")
    statistics = _validated_statistics(_mapping(coordinate_statistics, label="coordinate statistics"))
    return _finish(
        "fisher_graph.complete_h4_fisher_continuous_response_law.v1",
        _LAW_DOMAIN,
        {
            "protocol_sha256": CONTINUOUS_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "v20b_source_sha256s": sources,
            "family_ids": families,
            "outer_held_family_id": outer,
            "held_family_id": held,
            "excluded_family_ids": excluded,
            "fit_family_ids": fit_families,
            "coordinate_source_family_ids": coordinate_families,
            "coordinate_statistics_scope": "fit_only",
            "held_data_used_for_statistics_or_fit": False,
            "objectives_used_for_coordinate_selection": False,
            "family_labels_used_for_coordinate_selection": False,
            "base_provider_artifact_sha256": _sha(base_provider_artifact_sha256, label="base provider"),
            "proposal_provider_artifact_sha256": _sha(proposal_provider_artifact_sha256, label="proposal provider"),
            "lambda_fit_evidence_sha256": _sha(lambda_fit_evidence_sha256, label="lambda fit evidence"),
            "response_family": "lambda_times_asinh_normalized",
            "response_coordinate_application": "raw_selected_bounded_coordinate_no_center_scale_or_clamp",
            "coordinate_statistics_usage": "diagnostic_only_not_applied",
            "kappa": CONTINUOUS_RESPONSE_KAPPA,
            "signed_log_lambda": _unit_lambda(signed_log_lambda, label="signed-log lambda"),
            "linear_lambda": _unit_lambda(linear_lambda, label="linear lambda"),
            "coordinate_statistics": statistics,
            "raw_coordinates_or_objectives_serialized": False,
        },
    )


def build_continuous_response_law_receipt(
    *,
    v20b_source_sha256s: Mapping[str, str],
    family_ids: Sequence[str],
    outer_held_family_id: str,
    held_family_id: str,
    coordinate_source_family_ids: Sequence[str],
    fit_coordinates_by_family: Mapping[str, Sequence[Sequence[float]]],
    base_provider_artifact_sha256: str,
    proposal_provider_artifact_sha256: str,
    lambda_fit_evidence_sha256: str,
    signed_log_lambda: float,
    linear_lambda: float,
    held_data_used_for_statistics_or_fit: bool = False,
    objectives_used_for_coordinate_selection: bool = False,
    family_labels_used_for_coordinate_selection: bool = False,
) -> dict[str, object]:
    """Build a scalar-only law from explicitly fit-only coordinates."""

    coordinate_families = _families(
        coordinate_source_family_ids,
        label="coordinate source families",
        count=_FAMILY_COUNT - _EXCLUDED_COUNT,
    )
    grouped = _mapping(fit_coordinates_by_family, label="fit coordinates by family")
    if set(grouped) != set(coordinate_families):
        raise ValueError("fit coordinate groups must match the six coordinate source families")
    numeric_groups = tuple(grouped[family] for family in coordinate_families)
    return _build_law_from_statistics(
        v20b_source_sha256s=v20b_source_sha256s,
        family_ids=family_ids,
        outer_held_family_id=outer_held_family_id,
        held_family_id=held_family_id,
        coordinate_source_family_ids=coordinate_source_family_ids,
        base_provider_artifact_sha256=base_provider_artifact_sha256,
        proposal_provider_artifact_sha256=proposal_provider_artifact_sha256,
        lambda_fit_evidence_sha256=lambda_fit_evidence_sha256,
        signed_log_lambda=signed_log_lambda,
        linear_lambda=linear_lambda,
        coordinate_statistics=fit_coordinate_statistics(numeric_groups),
        held_data_used_for_statistics_or_fit=held_data_used_for_statistics_or_fit,
        objectives_used_for_coordinate_selection=objectives_used_for_coordinate_selection,
        family_labels_used_for_coordinate_selection=family_labels_used_for_coordinate_selection,
    )


def validate_continuous_response_law_receipt(
    value: Mapping[str, object],
    *,
    expected_v20b_source_sha256s: Mapping[str, str] | None = None,
    expected_base_provider_artifact_sha256: str | None = None,
    expected_proposal_provider_artifact_sha256: str | None = None,
) -> dict[str, object]:
    selected = _mapping(value, label="continuous-response law")
    _exact(selected, _LAW_KEYS, label="continuous-response law")
    if selected["protocol_sha256"] != CONTINUOUS_RESPONSE_PROTOCOL_SHA256:
        raise ValueError("continuous-response protocol differs")
    if selected["scientific_status"] != "development_only_reused_a16" or selected["fresh_family_disjoint_claim_authorized"] is not False:
        raise ValueError("continuous-response scientific status differs")
    if selected["coordinate_statistics_scope"] != "fit_only" or selected["raw_coordinates_or_objectives_serialized"] is not False:
        raise ValueError("continuous-response fit-only/scalar-only boundary differs")
    if selected["response_family"] != "lambda_times_asinh_normalized" or selected["kappa"] != CONTINUOUS_RESPONSE_KAPPA:
        raise ValueError("continuous-response family differs")
    if selected["response_coordinate_application"] != "raw_selected_bounded_coordinate_no_center_scale_or_clamp" or selected["coordinate_statistics_usage"] != "diagnostic_only_not_applied":
        raise ValueError("continuous-response raw-coordinate application differs")
    rebuilt = _build_law_from_statistics(
        v20b_source_sha256s=_mapping(selected["v20b_source_sha256s"], label="V20b sources"),
        family_ids=tuple(_sequence(selected["family_ids"], label="family IDs")),
        outer_held_family_id=selected["outer_held_family_id"],
        held_family_id=selected["held_family_id"],
        coordinate_source_family_ids=tuple(_sequence(selected["coordinate_source_family_ids"], label="coordinate families")),
        base_provider_artifact_sha256=selected["base_provider_artifact_sha256"],
        proposal_provider_artifact_sha256=selected["proposal_provider_artifact_sha256"],
        lambda_fit_evidence_sha256=selected["lambda_fit_evidence_sha256"],
        signed_log_lambda=selected["signed_log_lambda"],
        linear_lambda=selected["linear_lambda"],
        coordinate_statistics=_mapping(selected["coordinate_statistics"], label="coordinate statistics"),
        held_data_used_for_statistics_or_fit=selected["held_data_used_for_statistics_or_fit"],
        objectives_used_for_coordinate_selection=selected["objectives_used_for_coordinate_selection"],
        family_labels_used_for_coordinate_selection=selected["family_labels_used_for_coordinate_selection"],
    )
    _same(selected, rebuilt, label="continuous-response law")
    if expected_v20b_source_sha256s is not None and rebuilt["v20b_source_sha256s"] != _sources(expected_v20b_source_sha256s):
        raise ValueError("continuous-response V20b source lineage differs")
    if expected_base_provider_artifact_sha256 is not None and rebuilt["base_provider_artifact_sha256"] != _sha(expected_base_provider_artifact_sha256, label="expected base provider"):
        raise ValueError("continuous-response base endpoint differs")
    if expected_proposal_provider_artifact_sha256 is not None and rebuilt["proposal_provider_artifact_sha256"] != _sha(expected_proposal_provider_artifact_sha256, label="expected proposal provider"):
        raise ValueError("continuous-response proposal endpoint differs")
    return rebuilt


def continuous_response_value(
    coordinate: float,
    law_receipt: Mapping[str, object],
    *,
    arm: str = "signed_log",
) -> float:
    """Evaluate one frozen scalar law or compulsory control."""

    law = validate_continuous_response_law_receipt(law_receipt)
    if arm not in CONTINUOUS_RESPONSE_ARMS:
        raise ValueError("continuous-response arm differs")
    raw_coordinate = _number(coordinate, label="response coordinate")
    if abs(raw_coordinate) > 1.0:
        raise ValueError("response coordinate must remain in the raw bounded [-1, 1] domain")
    signed = float(law["signed_log_lambda"]) * signed_log(raw_coordinate)
    if arm == "base":
        return 0.0
    if arm == "signed_log":
        return signed
    if arm == "constant_plus_one":
        return 1.0
    if arm == "signed_log_sign_flip":
        return -signed
    return float(law["linear_lambda"]) * raw_coordinate


def _arm_definition(law: Mapping[str, object], arm: str) -> dict[str, object]:
    if arm not in CONTINUOUS_RESPONSE_ARMS:
        raise ValueError("continuous-response arm differs")
    if arm == "base":
        return {"kind": "zero_base", "intercept": 0.0, "lambda": 0.0, "response_sign": 1.0, "exact_sign_flip_of": None}
    if arm == "constant_plus_one":
        return {"kind": "constant", "intercept": 1.0, "lambda": 0.0, "response_sign": 1.0, "exact_sign_flip_of": None}
    if arm == "linear":
        return {"kind": "linear", "intercept": 0.0, "lambda": law["linear_lambda"], "response_sign": 1.0, "exact_sign_flip_of": None}
    return {
        "kind": "signed_log_asinh_kappa_9",
        "intercept": 0.0,
        "lambda": law["signed_log_lambda"],
        "response_sign": -1.0 if arm == "signed_log_sign_flip" else 1.0,
        "exact_sign_flip_of": "signed_log" if arm == "signed_log_sign_flip" else None,
    }


_ARM_SCORE_KEYS = {
    "schema",
    "law_artifact_sha256",
    "outer_held_family_id",
    "held_family_id",
    "arm",
    "arm_definition",
    "arm_definition_sha256",
    "objective",
    "execution_receipt_sha256",
    "response_trace_sha256",
    "score_source",
    "predicted_only",
    "finite",
    "pointwise_trust_passed",
    "rank_is_16",
    "execution_changed_from_base",
    "artifact_sha256",
}


def build_continuous_response_arm_score(
    *,
    law_receipt: Mapping[str, object],
    arm: str,
    objective: float,
    execution_receipt_sha256: str,
    response_trace_sha256: str,
    finite: bool,
    pointwise_trust_passed: bool,
    rank_is_16: bool,
    execution_changed_from_base: bool,
    score_source: str = "exact_finite_execution",
) -> dict[str, object]:
    law = validate_continuous_response_law_receipt(law_receipt)
    selected_arm = _identifier(arm, label="continuous-response arm")
    definition = _arm_definition(law, selected_arm)
    if score_source != "exact_finite_execution":
        raise ValueError("continuous-response scores require exact finite execution")
    changed = _boolean(execution_changed_from_base, label="execution change")
    if selected_arm == "base" and changed:
        raise ValueError("base execution cannot be changed from itself")
    return _finish(
        "fisher_graph.complete_h4_fisher_continuous_response_arm_score.v1",
        _ARM_SCORE_DOMAIN,
        {
            "law_artifact_sha256": law["artifact_sha256"],
            "outer_held_family_id": law["outer_held_family_id"],
            "held_family_id": law["held_family_id"],
            "arm": selected_arm,
            "arm_definition": definition,
            "arm_definition_sha256": _hash(_ARM_DEFINITION_DOMAIN, {"law_artifact_sha256": law["artifact_sha256"], "arm": selected_arm, **definition}),
            "objective": _number(objective, label="arm objective", positive=True),
            "execution_receipt_sha256": _sha(execution_receipt_sha256, label="arm execution"),
            "response_trace_sha256": _sha(response_trace_sha256, label="response trace"),
            "score_source": "exact_finite_execution",
            "predicted_only": False,
            "finite": _boolean(finite, label="arm finite"),
            "pointwise_trust_passed": _boolean(pointwise_trust_passed, label="arm trust"),
            "rank_is_16": _boolean(rank_is_16, label="arm rank"),
            "execution_changed_from_base": changed,
        },
    )


def validate_continuous_response_arm_score(
    value: Mapping[str, object], *, law_receipt: Mapping[str, object]
) -> dict[str, object]:
    selected = _mapping(value, label="continuous-response arm score")
    _exact(selected, _ARM_SCORE_KEYS, label="continuous-response arm score")
    if selected["score_source"] != "exact_finite_execution" or selected["predicted_only"] is not False:
        raise ValueError("predicted-only continuous-response score is forbidden")
    rebuilt = build_continuous_response_arm_score(
        law_receipt=law_receipt,
        arm=selected["arm"],
        objective=selected["objective"],
        execution_receipt_sha256=selected["execution_receipt_sha256"],
        response_trace_sha256=selected["response_trace_sha256"],
        finite=selected["finite"],
        pointwise_trust_passed=selected["pointwise_trust_passed"],
        rank_is_16=selected["rank_is_16"],
        execution_changed_from_base=selected["execution_changed_from_base"],
        score_source=selected["score_source"],
    )
    _same(selected, rebuilt, label="continuous-response arm score")
    return rebuilt


_ROLE_KEYS = {
    "schema",
    "law_receipt",
    "law_artifact_sha256",
    "outer_held_family_id",
    "held_family_id",
    "arm_scores",
    "arm_objectives",
    "arm_execution_receipt_sha256s",
    "arm_response_trace_sha256s",
    "all_arms_exactly_executed",
    "artifact_sha256",
}


def build_continuous_response_role_receipt(
    *, law_receipt: Mapping[str, object], arm_scores: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    law = validate_continuous_response_law_receipt(law_receipt)
    scores = tuple(
        validate_continuous_response_arm_score(value, law_receipt=law)
        for value in _sequence(arm_scores, label="arm scores")
    )
    by_arm = {str(value["arm"]): value for value in scores}
    if len(scores) != len(CONTINUOUS_RESPONSE_ARMS) or len(by_arm) != len(scores) or set(by_arm) != set(CONTINUOUS_RESPONSE_ARMS):
        raise ValueError("continuous-response role requires all five unique arms")
    ordered = tuple(by_arm[arm] for arm in CONTINUOUS_RESPONSE_ARMS)
    execution_hashes = tuple(str(value["execution_receipt_sha256"]) for value in ordered)
    if len(set(execution_hashes)) != len(execution_hashes):
        raise ValueError("continuous-response arms require distinct exact executions")
    return _finish(
        "fisher_graph.complete_h4_fisher_continuous_response_role.v1",
        _ROLE_DOMAIN,
        {
            "law_receipt": law,
            "law_artifact_sha256": law["artifact_sha256"],
            "outer_held_family_id": law["outer_held_family_id"],
            "held_family_id": law["held_family_id"],
            "arm_scores": ordered,
            "arm_objectives": {arm: by_arm[arm]["objective"] for arm in CONTINUOUS_RESPONSE_ARMS},
            "arm_execution_receipt_sha256s": {arm: by_arm[arm]["execution_receipt_sha256"] for arm in CONTINUOUS_RESPONSE_ARMS},
            "arm_response_trace_sha256s": {arm: by_arm[arm]["response_trace_sha256"] for arm in CONTINUOUS_RESPONSE_ARMS},
            "all_arms_exactly_executed": True,
        },
    )


def validate_continuous_response_role_receipt(value: Mapping[str, object]) -> dict[str, object]:
    selected = _mapping(value, label="continuous-response role")
    _exact(selected, _ROLE_KEYS, label="continuous-response role")
    if selected["all_arms_exactly_executed"] is not True:
        raise ValueError("continuous-response role lacks exact arm executions")
    rebuilt = build_continuous_response_role_receipt(
        law_receipt=_mapping(selected["law_receipt"], label="law receipt"),
        arm_scores=tuple(_mapping(item, label="arm score") for item in _sequence(selected["arm_scores"], label="arm scores")),
    )
    _same(selected, rebuilt, label="continuous-response role")
    return rebuilt


def _numerical_floor(reference: float) -> float:
    return max(
        _ABSOLUTE_IMPROVEMENT_FLOOR,
        _ROUNDOFF_MULTIPLIER * sys.float_info.epsilon * abs(reference),
    )


def evaluate_continuous_response_qualification(
    roles: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected_roles = tuple(
        validate_continuous_response_role_receipt(value)
        for value in _sequence(roles, label="continuous-response roles")
    )
    if len(selected_roles) != _ROLE_COUNT or len({value["held_family_id"] for value in selected_roles}) != _ROLE_COUNT:
        raise ValueError("continuous-response qualification requires seven unique held roles")
    objectives = {
        arm: tuple(float(_mapping(role["arm_objectives"], label="arm objectives")[arm]) for role in selected_roles)
        for arm in CONTINUOUS_RESPONSE_ARMS
    }
    macros = {arm: math.fsum(values) / _ROLE_COUNT for arm, values in objectives.items()}
    signed = objectives["signed_log"]
    base = objectives["base"]
    constant = objectives["constant_plus_one"]
    mirror = objectives["signed_log_sign_flip"]
    linear = objectives["linear"]
    relative = tuple((base_value - selected_value) / base_value for base_value, selected_value in zip(base, signed))
    base_wins = sum(base_value - selected_value > _numerical_floor(base_value) for base_value, selected_value in zip(base, signed))
    constant_wins = sum(control - selected_value > _numerical_floor(control) for control, selected_value in zip(constant, signed))
    mirror_wins = sum(control - selected_value > _numerical_floor(control) for control, selected_value in zip(mirror, signed))
    base_relative = (macros["base"] - macros["signed_log"]) / macros["base"]
    beats_constant_macro = macros["constant_plus_one"] - macros["signed_log"] > _numerical_floor(macros["constant_plus_one"])
    beats_mirror_macro = macros["signed_log_sign_flip"] - macros["signed_log"] > _numerical_floor(macros["signed_log_sign_flip"])
    beats_linear_macro = macros["linear"] - macros["signed_log"] > _numerical_floor(macros["linear"])
    health = all(
        score["finite"] is True
        and score["pointwise_trust_passed"] is True
        and score["rank_is_16"] is True
        and score["score_source"] == "exact_finite_execution"
        and score["predicted_only"] is False
        and (score["arm"] == "base" or score["execution_changed_from_base"] is True)
        for role in selected_roles
        for score in role["arm_scores"]
    )
    worst = min(relative)
    passed = bool(
        base_relative >= _MATERIALITY
        and base_wins >= _BASE_WINS
        and worst >= -_WORST_REGRESSION
        and beats_constant_macro
        and constant_wins >= _CONSTANT_WINS
        and beats_mirror_macro
        and mirror_wins >= _MIRROR_WINS
        and beats_linear_macro
        and health
    )
    return _finish(
        "fisher_graph.complete_h4_fisher_continuous_response_qualification.v1",
        _QUALIFICATION_DOMAIN,
        {
            "aggregation": "family_equal",
            "role_artifact_sha256s": tuple(value["artifact_sha256"] for value in selected_roles),
            "arm_macro_objectives": macros,
            "signed_log_base_relative_improvement": base_relative,
            "required_base_relative_improvement": _MATERIALITY,
            "base_win_count": base_wins,
            "required_base_win_count": _BASE_WINS,
            "worst_family_relative_improvement": worst,
            "worst_family_regression_maximum": _WORST_REGRESSION,
            "signed_log_beats_constant_macro_by_numerical_floor": beats_constant_macro,
            "constant_win_count": constant_wins,
            "required_constant_win_count": _CONSTANT_WINS,
            "signed_log_beats_mirror_macro_by_numerical_floor": beats_mirror_macro,
            "mirror_win_count": mirror_wins,
            "required_mirror_win_count": _MIRROR_WINS,
            "signed_log_beats_linear_macro_by_numerical_floor": beats_linear_macro,
            "all_finite_trusted_rank16_changed_exact": health,
            "passed": passed,
        },
    )


_SENTINEL_KEYS = {
    "schema",
    "protocol_sha256",
    "scientific_status",
    "fresh_family_disjoint_claim_authorized",
    "family_ids",
    "outer_held_family_id",
    "held_family_ids",
    "v20b_source_sha256s",
    "law_artifact_sha256s_by_held_family",
    "endpoint_artifact_sha256s_by_held_family",
    "coordinate_statistics_by_held_family",
    "roles",
    "qualification",
    "sentinel_passed",
    "next_full_reused_panel_screen_authorized",
    "artifact_sha256",
}


def build_continuous_response_sentinel_receipt(
    *,
    family_ids: Sequence[str],
    outer_held_family_id: str,
    v20b_source_sha256s: Mapping[str, str],
    roles: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    families = _families(family_ids, label="A16 family IDs", count=_FAMILY_COUNT)
    outer = _identifier(outer_held_family_id, label="outer held family")
    if outer not in families:
        raise ValueError("outer held family differs from A16 panel")
    sources = _sources(v20b_source_sha256s)
    selected_roles = tuple(
        sorted(
            (validate_continuous_response_role_receipt(value) for value in _sequence(roles, label="sentinel roles")),
            key=lambda value: str(value["held_family_id"]),
        )
    )
    expected_held = tuple(item for item in families if item != outer)
    if len(selected_roles) != _ROLE_COUNT or tuple(value["held_family_id"] for value in selected_roles) != expected_held:
        raise ValueError("sentinel requires exactly the seven non-outer held roles")
    for role in selected_roles:
        law = _mapping(role["law_receipt"], label="role law")
        if law["family_ids"] != families or law["outer_held_family_id"] != outer or law["v20b_source_sha256s"] != sources:
            raise ValueError("sentinel role source, panel, or outer lineage differs")
        if outer in law["fit_family_ids"] or role["held_family_id"] in law["fit_family_ids"]:
            raise ValueError("sentinel role exposed an excluded family to fit")
    qualification = evaluate_continuous_response_qualification(selected_roles)
    passed = qualification["passed"] is True
    return _finish(
        "fisher_graph.complete_h4_fisher_continuous_response_sentinel.v1",
        _SENTINEL_DOMAIN,
        {
            "protocol_sha256": CONTINUOUS_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "family_ids": families,
            "outer_held_family_id": outer,
            "held_family_ids": expected_held,
            "v20b_source_sha256s": sources,
            "law_artifact_sha256s_by_held_family": {str(role["held_family_id"]): role["law_artifact_sha256"] for role in selected_roles},
            "endpoint_artifact_sha256s_by_held_family": {
                str(role["held_family_id"]): {
                    "base": role["law_receipt"]["base_provider_artifact_sha256"],
                    "proposal": role["law_receipt"]["proposal_provider_artifact_sha256"],
                }
                for role in selected_roles
            },
            "coordinate_statistics_by_held_family": {
                str(role["held_family_id"]): role["law_receipt"]["coordinate_statistics"]
                for role in selected_roles
            },
            "roles": selected_roles,
            "qualification": qualification,
            "sentinel_passed": passed,
            "next_full_reused_panel_screen_authorized": passed,
        },
    )


def validate_continuous_response_sentinel_receipt(
    value: Mapping[str, object],
    *,
    expected_v20b_source_sha256s: Mapping[str, str] | None = None,
    expected_endpoint_artifact_sha256s_by_held_family: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, object]:
    selected = _mapping(value, label="continuous-response sentinel")
    _exact(selected, _SENTINEL_KEYS, label="continuous-response sentinel")
    if selected["protocol_sha256"] != CONTINUOUS_RESPONSE_PROTOCOL_SHA256:
        raise ValueError("continuous-response sentinel protocol differs")
    if selected["scientific_status"] != "development_only_reused_a16" or selected["fresh_family_disjoint_claim_authorized"] is not False:
        raise ValueError("continuous-response sentinel scientific status differs")
    rebuilt = build_continuous_response_sentinel_receipt(
        family_ids=tuple(_sequence(selected["family_ids"], label="sentinel family IDs")),
        outer_held_family_id=selected["outer_held_family_id"],
        v20b_source_sha256s=_mapping(selected["v20b_source_sha256s"], label="sentinel V20b sources"),
        roles=tuple(_mapping(item, label="sentinel role") for item in _sequence(selected["roles"], label="sentinel roles")),
    )
    _same(selected, rebuilt, label="continuous-response sentinel")
    if expected_v20b_source_sha256s is not None and rebuilt["v20b_source_sha256s"] != _sources(expected_v20b_source_sha256s):
        raise ValueError("sentinel V20b source lineage differs")
    if expected_endpoint_artifact_sha256s_by_held_family is not None:
        expected = {
            _identifier(family, label="expected endpoint family"): {
                "base": _sha(_mapping(pair, label="expected endpoint pair").get("base"), label="expected base endpoint"),
                "proposal": _sha(_mapping(pair, label="expected endpoint pair").get("proposal"), label="expected proposal endpoint"),
            }
            for family, pair in expected_endpoint_artifact_sha256s_by_held_family.items()
        }
        if rebuilt["endpoint_artifact_sha256s_by_held_family"] != dict(sorted(expected.items())):
            raise ValueError("sentinel endpoint lineage differs")
    return rebuilt
