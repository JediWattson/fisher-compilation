"""Pure V20d fit-only natural-response protocol.

V20d learns one three-parameter response direction from six fit families and
chooses one finite step before either held-family capability may exist.  The
module owns scalar mathematics and hash receipts only; tensors, prompts,
logits, targets, provider sidecars, and held objectives are never serialized.

For bounded Fisher coordinates ``c=(c1,c2)`` the response projection is

``z = w1*c1 + w2*c2 + w12*c1*c2``

and the nonlinear law is ``asinh(9*z)/asinh(9)``.  Feasibility is certified
exactly at the four corners of ``[-1,1]^2`` and infeasible proposals are
projected radially.  The fixed starting weight is ``(0,1,0)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
import sys

__all__ = [
    "NATURAL_RESPONSE_ALPHAS",
    "NATURAL_RESPONSE_ARMS",
    "NATURAL_RESPONSE_INITIAL_WEIGHTS",
    "NATURAL_RESPONSE_KAPPA",
    "NATURAL_RESPONSE_LAWS",
    "NATURAL_RESPONSE_PROTOCOL_SHA256",
    "bilinear_corner_values",
    "bilinear_box_certificate",
    "build_natural_response_alpha_candidate",
    "build_natural_response_direction_receipt",
    "build_natural_response_fit_receipt",
    "build_natural_response_two_fit_bundle_receipt",
    "build_natural_response_held_arm_score",
    "build_natural_response_held_role_receipt",
    "build_natural_response_pair_qualification",
    "natural_response_features",
    "natural_response_gain",
    "natural_response_work_accounting",
    "radially_project_bilinear_weights",
    "signed_log_response",
    "validate_natural_response_alpha_candidate",
    "validate_natural_response_direction_receipt",
    "validate_natural_response_fit_receipt",
    "validate_natural_response_two_fit_bundle_receipt",
    "validate_natural_response_held_arm_score",
    "validate_natural_response_held_role_receipt",
    "validate_natural_response_pair_qualification",
]


NATURAL_RESPONSE_KAPPA = 9.0
NATURAL_RESPONSE_LAWS = ("signed_log", "linear")
NATURAL_RESPONSE_INITIAL_WEIGHTS = (0.0, 1.0, 0.0)
NATURAL_RESPONSE_ALPHAS = (0.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0)
NATURAL_RESPONSE_ARMS = (
    "base",
    "constant_plus_one",
    "fixed_signed_log",
    "fixed_linear",
    "learned_signed_log",
    "learned_linear",
    "learned_signed_log_sign_flip",
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_FAMILY_COUNT = 8
_FIT_FAMILY_COUNT = 6
_HELD_ROLE_COUNT = 2
_FEATURE_COUNT = 3
_BASE_MATERIALITY = 0.01
_FIXED_LOG_MATERIALITY = 0.001
_ABSOLUTE_FLOOR = 1.0e-12
_ROUNDOFF_MULTIPLIER = 128.0

_PROTOCOL_DOMAIN = b"fisher-graph:complete-h4-fisher-natural-response:protocol:v20d\0"
_WEIGHT_DOMAIN = b"fisher-graph:complete-h4-fisher-natural-response:weights:v20d\0"
_GRADIENT_DOMAIN = b"fisher-graph:complete-h4-fisher-natural-response:gradient:v20d\0"
_DIRECTION_DOMAIN = b"fisher-graph:complete-h4-fisher-natural-response:direction:v20d\0"
_CANDIDATE_DOMAIN = b"fisher-graph:complete-h4-fisher-natural-response:candidate:v20d\0"
_FIT_DOMAIN = b"fisher-graph:complete-h4-fisher-natural-response:fit:v20d\0"
_BUNDLE_DOMAIN = b"fisher-graph:complete-h4-fisher-natural-response:two-fit-bundle:v20d\0"
_SCORE_DOMAIN = b"fisher-graph:complete-h4-fisher-natural-response:score:v20d\0"
_ROLE_DOMAIN = b"fisher-graph:complete-h4-fisher-natural-response:role:v20d\0"
_QUALIFICATION_DOMAIN = b"fisher-graph:complete-h4-fisher-natural-response:qualification:v20d\0"

_FIXED_PROTOCOL = {
    "protocol": "fit_only_three_feature_natural_response_v20d",
    "scientific_status": "development_only_reused_a16",
    "fresh_family_disjoint_claim_authorized": False,
    "serving_claim_authorized": False,
    "compression_claim_authorized": False,
    "features": ("c1", "c2", "c1_times_c2"),
    "intercept": False,
    "initial_weights": NATURAL_RESPONSE_INITIAL_WEIGHTS,
    "response_laws": NATURAL_RESPONSE_LAWS,
    "independent_law_specific_gradients_directions_and_fits": True,
    "signed_log_kappa": NATURAL_RESPONSE_KAPPA,
    "feasibility": "exact_four_corner_bilinear_box_certificate_at_most_one",
    "projection": "radial_to_box_certificate_one",
    "gradient_aggregation": "family_equal_then_example_equal",
    "fisher": "family_equal_response_parameter_empirical_gradient_Fisher_OPG",
    "damping": "max_1e-12_1e-3_times_trace_over_three",
    "natural_direction_count": 1,
    "alpha_ladder": NATURAL_RESPONSE_ALPHAS,
    "line_search_objective": "exact_fit_objectives_only",
    "line_search_tie": "objective_then_smaller_alpha_then_canonical_weight_hash",
    "line_search_selection": "filter_valid_improving_candidates_before_argmin",
    "rollback": "alpha_zero_unless_improvement_exceeds_float64_floor",
    "degenerate_gradient": "valid_direction_receipt_then_deterministic_rollback",
    "two_fit_bundle_authorization": "conjunction_of_signed_log_and_linear_fit_authorization",
    "held_arms": NATURAL_RESPONSE_ARMS,
    "held_base_materiality": _BASE_MATERIALITY,
    "held_fixed_log_materiality": _FIXED_LOG_MATERIALITY,
    "held_learned_signed_log_nonconstant_both_roles": True,
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


NATURAL_RESPONSE_PROTOCOL_SHA256 = _hash(_PROTOCOL_DOMAIN, _FIXED_PROTOCOL)


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


def _number(value: object, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return 0.0 if result == 0.0 else result


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
    source = _mapping(value, label="V20c source hashes")
    if not source:
        raise ValueError("V20c source hashes must not be empty")
    return dict(
        sorted(
            (
                _identifier(key, label="V20c source name"),
                _sha(item, label=f"V20c source {key}"),
            )
            for key, item in source.items()
        )
    )


def _vector(value: object, *, label: str) -> tuple[float, float, float]:
    selected = tuple(_number(item, label=label) for item in _sequence(value, label=label))
    if len(selected) != _FEATURE_COUNT:
        raise ValueError(f"{label} must have exactly three values")
    return selected  # type: ignore[return-value]


def _matrix3(value: object, *, label: str) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(_vector(row, label=label) for row in _sequence(value, label=label))
    if len(rows) != _FEATURE_COUNT:
        raise ValueError(f"{label} must be 3 by 3")
    return rows


def _numerical_floor(reference: float) -> float:
    selected = _number(reference, label="numerical-floor reference", nonnegative=True)
    return max(
        _ABSOLUTE_FLOOR,
        _ROUNDOFF_MULTIPLIER * sys.float_info.epsilon * abs(selected),
    )


def natural_response_features(c1: float, c2: float) -> tuple[float, float, float]:
    """Return the fixed no-intercept V20d response features."""

    left = _number(c1, label="c1")
    right = _number(c2, label="c2")
    if abs(left) > 1.0 or abs(right) > 1.0:
        raise ValueError("natural-response coordinates must lie in [-1, 1]")
    return left, right, left * right


def signed_log_response(value: float) -> float:
    """Return ``asinh(9*x)/asinh(9)`` without a branch at zero."""

    selected = _number(value, label="signed-log response input")
    if abs(selected) > 1.0 + 1.0e-15:
        raise ValueError("signed-log response input escaped the certified box")
    return math.asinh(NATURAL_RESPONSE_KAPPA * selected) / math.asinh(
        NATURAL_RESPONSE_KAPPA
    )


def bilinear_corner_values(weights: Sequence[float]) -> tuple[float, float, float, float]:
    """Return exact values in frozen corner order ``--,-+,+-,++``."""

    w1, w2, w12 = _vector(weights, label="bilinear weights")
    values = tuple(
        w1 * c1 + w2 * c2 + w12 * c1 * c2
        for c1, c2 in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bilinear corner value became nonfinite")
    return values


def bilinear_box_certificate(weights: Sequence[float]) -> float:
    """Return the exact ``B(w)`` maximum absolute four-corner value."""

    return max(abs(value) for value in bilinear_corner_values(weights))


def radially_project_bilinear_weights(weights: Sequence[float]) -> dict[str, object]:
    """Project toward the origin until the exact box certificate is at most one."""

    source = _vector(weights, label="unprojected bilinear weights")
    corners_before = bilinear_corner_values(source)
    before = bilinear_box_certificate(source)
    scale = 1.0 if before <= 1.0 else 1.0 / before
    projected = tuple(0.0 if item * scale == 0.0 else item * scale for item in source)
    corners_after = bilinear_corner_values(projected)
    after = bilinear_box_certificate(projected)
    # Division may round outward by one ulp.  A certificate is executable only
    # when its independently recomputed value is <= 1 exactly, so contract once
    # more toward zero instead of accepting a tolerance that a provider cannot.
    if after > 1.0:
        correction = math.nextafter(1.0 / after, 0.0)
        scale *= correction
        projected = tuple(
            0.0 if item * correction == 0.0 else item * correction
            for item in projected
        )
        corners_after = bilinear_corner_values(projected)
        after = bilinear_box_certificate(projected)
    if after > 1.0:
        raise RuntimeError("radial box projection failed exact certificate")
    return {
        "unprojected_weights": source,
        "unprojected_corner_values": corners_before,
        "unprojected_box_certificate": before,
        "radial_projection_scale": scale,
        "projection_semantics": "radial_scaling_not_euclidean_projection",
        "weights": projected,
        "corner_values": corners_after,
        "box_certificate": after,
        "feasible": after <= 1.0,
    }


def natural_response_gain(
    weights: Sequence[float], c1: float, c2: float, *, law: str = "signed_log"
) -> float:
    """Evaluate a certified learned linear or signed-log response."""

    selected = _vector(weights, label="natural-response weights")
    if bilinear_box_certificate(selected) > 1.0:
        raise ValueError("natural-response weights lack a box certificate")
    features = natural_response_features(c1, c2)
    projection = math.fsum(weight * feature for weight, feature in zip(selected, features))
    if law == "linear":
        return projection
    if law == "signed_log":
        return signed_log_response(projection)
    raise ValueError("natural-response law differs")


def _outer(left: Sequence[float], right: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(a * b for b in right) for a in left)


def _natural_direction(
    gradient: Sequence[float], fisher: Sequence[Sequence[float]], damping: float
) -> tuple[float, float, float]:
    """Solve ``-(F+dI)^-1 g`` using deterministic Cholesky arithmetic."""

    g = _vector(gradient, label="gradient mean")
    f = _matrix3(fisher, label="empirical Fisher")
    damp = _number(damping, label="natural damping", nonnegative=True)
    a = [[f[row][column] + (damp if row == column else 0.0) for column in range(3)] for row in range(3)]
    lower = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(row + 1):
            subtotal = math.fsum(lower[row][index] * lower[column][index] for index in range(column))
            if row == column:
                pivot = a[row][row] - subtotal
                if pivot <= 0.0 or not math.isfinite(pivot):
                    raise ValueError("damped empirical Fisher is not positive definite")
                lower[row][column] = math.sqrt(pivot)
            else:
                lower[row][column] = (a[row][column] - subtotal) / lower[column][column]
    forward = [0.0] * 3
    for row in range(3):
        forward[row] = (-g[row] - math.fsum(lower[row][index] * forward[index] for index in range(row))) / lower[row][row]
    result = [0.0] * 3
    for row in range(2, -1, -1):
        result[row] = (forward[row] - math.fsum(lower[index][row] * result[index] for index in range(row + 1, 3))) / lower[row][row]
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError("natural direction became nonfinite")
    return tuple(0.0 if item == 0.0 else item for item in result)  # type: ignore[return-value]


_DIRECTION_KEYS = {
    "schema",
    "protocol_sha256",
    "scientific_status",
    "fresh_family_disjoint_claim_authorized",
    "serving_claim_authorized",
    "compression_claim_authorized",
    "v20c_source_sha256s",
    "family_ids",
    "excluded_family_ids",
    "fit_family_ids",
    "base_provider_artifact_sha256",
    "proposal_provider_artifact_sha256",
    "gradient_evidence_sha256",
    "response_law",
    "fit_example_ids_by_family",
    "fit_example_counts_by_family",
    "example_gradient_sha256s_by_family",
    "fit_gradient_rows_sha256",
    "gradient_mean",
    "empirical_fisher",
    "fisher_semantics",
    "empirical_fisher_trace",
    "damping",
    "natural_direction",
    "natural_direction_norm",
    "gradient_dot_natural_direction",
    "strict_descent_direction",
    "initial_weights",
    "feature_names",
    "family_equal_per_example_aggregation",
    "held_objectives_or_gradients_used",
    "raw_gradients_or_tensors_serialized",
    "artifact_sha256",
}


def _build_direction_from_statistics(
    *,
    v20c_source_sha256s: Mapping[str, str],
    family_ids: Sequence[str],
    excluded_family_ids: Sequence[str],
    base_provider_artifact_sha256: str,
    proposal_provider_artifact_sha256: str,
    gradient_evidence_sha256: str,
    response_law: str,
    fit_example_ids_by_family: Mapping[str, Sequence[str]],
    fit_example_counts_by_family: Mapping[str, int],
    example_gradient_sha256s_by_family: Mapping[str, Mapping[str, str]],
    fit_gradient_rows_sha256: str,
    gradient_mean: Sequence[float],
    empirical_fisher: Sequence[Sequence[float]],
    held_objectives_or_gradients_used: bool,
) -> dict[str, object]:
    sources = _sources(v20c_source_sha256s)
    families = _families(family_ids, label="A16 family IDs", count=_FAMILY_COUNT)
    excluded = _families(excluded_family_ids, label="excluded families", count=2)
    if not set(excluded) <= set(families):
        raise ValueError("excluded families differ from the A16 panel")
    fit_families = tuple(family for family in families if family not in excluded)
    if len(fit_families) != _FIT_FAMILY_COUNT:
        raise ValueError("natural-response fit requires exactly six families")
    if _boolean(held_objectives_or_gradients_used, label="held evidence marker"):
        raise ValueError("held objectives or gradients may not enter natural-response fitting")
    ids_source = _mapping(fit_example_ids_by_family, label="fit example IDs")
    count_source = _mapping(fit_example_counts_by_family, label="fit example counts")
    hash_source = _mapping(example_gradient_sha256s_by_family, label="example gradient hashes")
    if set(ids_source) != set(fit_families) or set(count_source) != set(fit_families) or set(hash_source) != set(fit_families):
        raise ValueError("direction fit-family evidence geometry differs")
    normalized_ids: dict[str, tuple[str, ...]] = {}
    normalized_counts: dict[str, int] = {}
    normalized_hashes: dict[str, dict[str, str]] = {}
    for family in fit_families:
        ids = tuple(sorted(_identifier(item, label="fit example ID") for item in _sequence(ids_source[family], label="fit example IDs")))
        count = _integer(count_source[family], label="fit example count", minimum=1)
        hashes = _mapping(hash_source[family], label="family gradient hashes")
        if len(ids) != count or len(set(ids)) != count or set(hashes) != set(ids):
            raise ValueError("per-family gradient example geometry differs")
        normalized_ids[family] = ids
        normalized_counts[family] = count
        normalized_hashes[family] = {
            example: _sha(hashes[example], label="example gradient") for example in ids
        }
    mean = _vector(gradient_mean, label="gradient mean")
    fisher = _matrix3(empirical_fisher, label="empirical Fisher")
    for row in range(3):
        for column in range(3):
            if fisher[row][column] != fisher[column][row]:
                raise ValueError("empirical Fisher must be exactly symmetric")
    trace = math.fsum(fisher[index][index] for index in range(3))
    if trace < 0.0:
        raise ValueError("empirical Fisher trace must be nonnegative")
    damping = max(1.0e-12, 1.0e-3 * trace / 3.0)
    law = _identifier(response_law, label="response law")
    if law not in NATURAL_RESPONSE_LAWS:
        raise ValueError("natural-response fit law differs")
    direction = _natural_direction(mean, fisher, damping)
    direction_dot = math.fsum(left * right for left, right in zip(mean, direction))
    return _finish(
        "fisher_graph.complete_h4_fisher_natural_response_direction.v1",
        _DIRECTION_DOMAIN,
        {
            "protocol_sha256": NATURAL_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "serving_claim_authorized": False,
            "compression_claim_authorized": False,
            "v20c_source_sha256s": sources,
            "family_ids": families,
            "excluded_family_ids": excluded,
            "fit_family_ids": fit_families,
            "base_provider_artifact_sha256": _sha(base_provider_artifact_sha256, label="base provider"),
            "proposal_provider_artifact_sha256": _sha(proposal_provider_artifact_sha256, label="proposal provider"),
            "gradient_evidence_sha256": _sha(gradient_evidence_sha256, label="gradient evidence"),
            "response_law": law,
            "fit_example_ids_by_family": normalized_ids,
            "fit_example_counts_by_family": normalized_counts,
            "example_gradient_sha256s_by_family": normalized_hashes,
            "fit_gradient_rows_sha256": _sha(fit_gradient_rows_sha256, label="fit gradient rows"),
            "gradient_mean": mean,
            "empirical_fisher": fisher,
            "fisher_semantics": "family_equal_response_parameter_empirical_gradient_Fisher_OPG",
            "empirical_fisher_trace": trace,
            "damping": damping,
            "natural_direction": direction,
            "natural_direction_norm": math.sqrt(math.fsum(item * item for item in direction)),
            "gradient_dot_natural_direction": direction_dot,
            "strict_descent_direction": direction_dot < 0.0,
            "initial_weights": NATURAL_RESPONSE_INITIAL_WEIGHTS,
            "feature_names": ("c1", "c2", "c1_times_c2"),
            "family_equal_per_example_aggregation": True,
            "held_objectives_or_gradients_used": False,
            "raw_gradients_or_tensors_serialized": False,
        },
    )


def build_natural_response_direction_receipt(
    *,
    v20c_source_sha256s: Mapping[str, str],
    family_ids: Sequence[str],
    excluded_family_ids: Sequence[str],
    fit_gradients_by_family: Mapping[str, Mapping[str, Sequence[float]]],
    base_provider_artifact_sha256: str,
    proposal_provider_artifact_sha256: str,
    gradient_evidence_sha256: str,
    response_law: str,
    held_objectives_or_gradients_used: bool = False,
) -> dict[str, object]:
    """Build one family-equal empirical-Fisher natural direction."""

    families = _families(family_ids, label="A16 family IDs", count=_FAMILY_COUNT)
    excluded = _families(excluded_family_ids, label="excluded families", count=2)
    fit_families = tuple(family for family in families if family not in excluded)
    source = _mapping(fit_gradients_by_family, label="fit gradients by family")
    if len(fit_families) != _FIT_FAMILY_COUNT or set(source) != set(fit_families):
        raise ValueError("fit gradients must contain exactly the six non-held families")
    ids: dict[str, tuple[str, ...]] = {}
    counts: dict[str, int] = {}
    hashes: dict[str, dict[str, str]] = {}
    family_means: list[tuple[float, float, float]] = []
    family_fishers: list[tuple[tuple[float, ...], ...]] = []
    canonical_rows: dict[str, tuple[tuple[str, tuple[float, ...]], ...]] = {}
    for family in fit_families:
        examples = _mapping(source[family], label="family fit gradients")
        if not examples:
            raise ValueError("every fit family requires at least one example gradient")
        rows = tuple(
            (
                _identifier(example, label="fit example ID"),
                _vector(gradient, label="per-example gradient"),
            )
            for example, gradient in sorted(examples.items())
        )
        if len({example for example, _gradient in rows}) != len(rows):
            raise ValueError("fit example IDs must be unique within each family")
        ids[family] = tuple(example for example, _gradient in rows)
        counts[family] = len(rows)
        hashes[family] = {
            example: _hash(
                _GRADIENT_DOMAIN,
                {"family_id": family, "example_id": example, "gradient": gradient},
            )
            for example, gradient in rows
        }
        gradients = tuple(gradient for _example, gradient in rows)
        family_means.append(
            tuple(math.fsum(row[index] for row in gradients) / len(gradients) for index in range(3))  # type: ignore[arg-type]
        )
        outers = tuple(_outer(row, row) for row in gradients)
        family_fishers.append(
            tuple(
                tuple(math.fsum(item[row][column] for item in outers) / len(outers) for column in range(3))
                for row in range(3)
            )
        )
        canonical_rows[family] = rows
    gradient_mean = tuple(
        math.fsum(value[index] for value in family_means) / len(family_means)
        for index in range(3)
    )
    fisher = tuple(
        tuple(
            math.fsum(value[row][column] for value in family_fishers) / len(family_fishers)
            for column in range(3)
        )
        for row in range(3)
    )
    return _build_direction_from_statistics(
        v20c_source_sha256s=v20c_source_sha256s,
        family_ids=families,
        excluded_family_ids=excluded,
        base_provider_artifact_sha256=base_provider_artifact_sha256,
        proposal_provider_artifact_sha256=proposal_provider_artifact_sha256,
        gradient_evidence_sha256=gradient_evidence_sha256,
        response_law=response_law,
        fit_example_ids_by_family=ids,
        fit_example_counts_by_family=counts,
        example_gradient_sha256s_by_family=hashes,
        fit_gradient_rows_sha256=_hash(_GRADIENT_DOMAIN, canonical_rows),
        gradient_mean=gradient_mean,
        empirical_fisher=fisher,
        held_objectives_or_gradients_used=held_objectives_or_gradients_used,
    )


def validate_natural_response_direction_receipt(
    value: Mapping[str, object],
    *,
    expected_v20c_source_sha256s: Mapping[str, str] | None = None,
    expected_base_provider_artifact_sha256: str | None = None,
    expected_proposal_provider_artifact_sha256: str | None = None,
) -> dict[str, object]:
    selected = _mapping(value, label="natural-response direction")
    _exact(selected, _DIRECTION_KEYS, label="natural-response direction")
    if selected["protocol_sha256"] != NATURAL_RESPONSE_PROTOCOL_SHA256:
        raise ValueError("natural-response direction protocol differs")
    if (
        selected["scientific_status"] != "development_only_reused_a16"
        or selected["fresh_family_disjoint_claim_authorized"] is not False
        or selected["serving_claim_authorized"] is not False
        or selected["compression_claim_authorized"] is not False
        or selected["family_equal_per_example_aggregation"] is not True
        or selected["raw_gradients_or_tensors_serialized"] is not False
        or selected["fisher_semantics"]
        != "family_equal_response_parameter_empirical_gradient_Fisher_OPG"
    ):
        raise ValueError("natural-response direction scientific boundary differs")
    rebuilt = _build_direction_from_statistics(
        v20c_source_sha256s=_mapping(selected["v20c_source_sha256s"], label="V20c sources"),
        family_ids=tuple(_sequence(selected["family_ids"], label="family IDs")),
        excluded_family_ids=tuple(_sequence(selected["excluded_family_ids"], label="excluded families")),
        base_provider_artifact_sha256=selected["base_provider_artifact_sha256"],
        proposal_provider_artifact_sha256=selected["proposal_provider_artifact_sha256"],
        gradient_evidence_sha256=selected["gradient_evidence_sha256"],
        response_law=selected["response_law"],
        fit_example_ids_by_family=_mapping(selected["fit_example_ids_by_family"], label="fit example IDs"),
        fit_example_counts_by_family=_mapping(selected["fit_example_counts_by_family"], label="fit example counts"),
        example_gradient_sha256s_by_family=_mapping(selected["example_gradient_sha256s_by_family"], label="gradient hashes"),
        fit_gradient_rows_sha256=selected["fit_gradient_rows_sha256"],
        gradient_mean=tuple(_sequence(selected["gradient_mean"], label="gradient mean")),
        empirical_fisher=tuple(_sequence(selected["empirical_fisher"], label="empirical Fisher")),
        held_objectives_or_gradients_used=selected["held_objectives_or_gradients_used"],
    )
    _same(selected, rebuilt, label="natural-response direction")
    if expected_v20c_source_sha256s is not None and rebuilt["v20c_source_sha256s"] != _sources(expected_v20c_source_sha256s):
        raise ValueError("natural-response V20c source lineage differs")
    if expected_base_provider_artifact_sha256 is not None and rebuilt["base_provider_artifact_sha256"] != _sha(expected_base_provider_artifact_sha256, label="expected base provider"):
        raise ValueError("natural-response base endpoint differs")
    if expected_proposal_provider_artifact_sha256 is not None and rebuilt["proposal_provider_artifact_sha256"] != _sha(expected_proposal_provider_artifact_sha256, label="expected proposal provider"):
        raise ValueError("natural-response proposal endpoint differs")
    return rebuilt


_CANDIDATE_KEYS = {
    "schema",
    "protocol_sha256",
    "direction_artifact_sha256",
    "response_law",
    "alpha",
    "initial_weights",
    "natural_direction",
    "unprojected_weights",
    "unprojected_corner_values",
    "unprojected_box_certificate",
    "radial_projection_scale",
    "projection_semantics",
    "weights",
    "corner_values",
    "box_certificate",
    "feasible",
    "weight_sha256",
    "projected_displacement",
    "weight_displacement",
    "projected_displacement_dot_gradient",
    "provider_artifact_sha256",
    "fit_example_ids_by_family",
    "exact_fit_objectives_by_family",
    "fit_execution_receipt_sha256s_by_family",
    "family_objectives",
    "family_equal_objective",
    "objective_source",
    "held_objectives_used",
    "raw_tensors_logits_or_targets_serialized",
    "artifact_sha256",
}


def build_natural_response_alpha_candidate(
    *,
    direction_receipt: Mapping[str, object],
    alpha: float,
    provider_artifact_sha256: str,
    exact_fit_objectives_by_family: Mapping[str, Mapping[str, float]],
    fit_execution_receipt_sha256s_by_family: Mapping[str, Mapping[str, str]],
    objective_source: str = "exact_finite_fit_execution",
    held_objectives_used: bool = False,
) -> dict[str, object]:
    direction = validate_natural_response_direction_receipt(direction_receipt)
    selected_alpha = _number(alpha, label="natural-response alpha", nonnegative=True)
    if selected_alpha not in NATURAL_RESPONSE_ALPHAS:
        raise ValueError("natural-response alpha is outside the fixed ladder")
    if objective_source != "exact_finite_fit_execution":
        raise ValueError("natural-response line search requires exact fit objectives")
    if _boolean(held_objectives_used, label="held-objective marker"):
        raise ValueError("held objectives may not enter natural-response line search")
    fit_families = tuple(direction["fit_family_ids"])
    objective_source_map = _mapping(exact_fit_objectives_by_family, label="exact fit objectives")
    execution_source = _mapping(fit_execution_receipt_sha256s_by_family, label="fit executions")
    if set(objective_source_map) != set(fit_families) or set(execution_source) != set(fit_families):
        raise ValueError("alpha candidate fit-family geometry differs")
    expected_ids = _mapping(direction["fit_example_ids_by_family"], label="direction fit examples")
    objectives: dict[str, dict[str, float]] = {}
    executions: dict[str, dict[str, str]] = {}
    family_objectives: dict[str, float] = {}
    for family in fit_families:
        raw_values = _mapping(objective_source_map[family], label="family fit objectives")
        raw_executions = _mapping(execution_source[family], label="family fit executions")
        ids = tuple(expected_ids[family])
        if set(raw_values) != set(ids) or set(raw_executions) != set(ids):
            raise ValueError("alpha candidate fit-example geometry differs")
        objectives[family] = {
            example: _number(raw_values[example], label="exact fit objective", nonnegative=True)
            for example in ids
        }
        executions[family] = {
            example: _sha(raw_executions[example], label="fit execution") for example in ids
        }
        family_objectives[family] = math.fsum(objectives[family].values()) / len(ids)
    macro = math.fsum(family_objectives.values()) / len(fit_families)
    natural_direction = tuple(direction["natural_direction"])
    proposal = tuple(
        initial + selected_alpha * step
        for initial, step in zip(NATURAL_RESPONSE_INITIAL_WEIGHTS, natural_direction)
    )
    projected = radially_project_bilinear_weights(proposal)
    weights = tuple(projected["weights"])
    displacement_vector = tuple(
        value - initial for value, initial in zip(weights, NATURAL_RESPONSE_INITIAL_WEIGHTS)
    )
    displacement = math.sqrt(math.fsum(value * value for value in displacement_vector))
    displacement_dot = math.fsum(
        value * gradient
        for value, gradient in zip(displacement_vector, direction["gradient_mean"])
    )
    return _finish(
        "fisher_graph.complete_h4_fisher_natural_response_alpha_candidate.v1",
        _CANDIDATE_DOMAIN,
        {
            "protocol_sha256": NATURAL_RESPONSE_PROTOCOL_SHA256,
            "direction_artifact_sha256": direction["artifact_sha256"],
            "response_law": direction["response_law"],
            "alpha": selected_alpha,
            "initial_weights": NATURAL_RESPONSE_INITIAL_WEIGHTS,
            "natural_direction": natural_direction,
            **projected,
            "weight_sha256": _hash(_WEIGHT_DOMAIN, weights),
            "projected_displacement": displacement_vector,
            "weight_displacement": displacement,
            "projected_displacement_dot_gradient": displacement_dot,
            "provider_artifact_sha256": _sha(provider_artifact_sha256, label="alpha provider"),
            "fit_example_ids_by_family": expected_ids,
            "exact_fit_objectives_by_family": objectives,
            "fit_execution_receipt_sha256s_by_family": executions,
            "family_objectives": family_objectives,
            "family_equal_objective": macro,
            "objective_source": "exact_finite_fit_execution",
            "held_objectives_used": False,
            "raw_tensors_logits_or_targets_serialized": False,
        },
    )


def validate_natural_response_alpha_candidate(
    value: Mapping[str, object], *, direction_receipt: Mapping[str, object]
) -> dict[str, object]:
    selected = _mapping(value, label="natural-response alpha candidate")
    _exact(selected, _CANDIDATE_KEYS, label="natural-response alpha candidate")
    if selected["protocol_sha256"] != NATURAL_RESPONSE_PROTOCOL_SHA256 or selected["raw_tensors_logits_or_targets_serialized"] is not False:
        raise ValueError("natural-response alpha protocol/scalar boundary differs")
    rebuilt = build_natural_response_alpha_candidate(
        direction_receipt=direction_receipt,
        alpha=selected["alpha"],
        provider_artifact_sha256=selected["provider_artifact_sha256"],
        exact_fit_objectives_by_family=_mapping(selected["exact_fit_objectives_by_family"], label="exact fit objectives"),
        fit_execution_receipt_sha256s_by_family=_mapping(selected["fit_execution_receipt_sha256s_by_family"], label="fit executions"),
        objective_source=selected["objective_source"],
        held_objectives_used=selected["held_objectives_used"],
    )
    _same(selected, rebuilt, label="natural-response alpha candidate")
    return rebuilt


_FIT_KEYS = {
    "schema",
    "protocol_sha256",
    "scientific_status",
    "fresh_family_disjoint_claim_authorized",
    "serving_claim_authorized",
    "compression_claim_authorized",
    "direction_receipt",
    "direction_artifact_sha256",
    "response_law",
    "candidate_receipts",
    "candidate_artifact_sha256s",
    "alpha_ladder",
    "alpha_zero_objective",
    "selected_alpha",
    "selected_weights",
    "initial_weight_sha256",
    "selected_weight_sha256",
    "selected_corner_values",
    "selected_box_certificate",
    "selected_projected_displacement_dot_gradient",
    "selected_provider_artifact_sha256",
    "selected_candidate_artifact_sha256",
    "selected_objective",
    "objective_absolute_improvement",
    "objective_numerical_improvement_floor",
    "line_search_tie_order",
    "selected_alpha_is_positive",
    "selected_weight_hash_changed",
    "selected_box_certificate_passed",
    "selected_response_is_nonconstant",
    "selected_projected_displacement_is_descent",
    "exact_fit_objective_improved",
    "learned_candidate_authorized",
    "rollback_to_initial_weights",
    "held_score_authorized",
    "exact_fit_objectives_only",
    "raw_tensors_logits_targets_or_gradients_serialized",
    "artifact_sha256",
}


def build_natural_response_fit_receipt(
    *,
    direction_receipt: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    direction = validate_natural_response_direction_receipt(direction_receipt)
    values = tuple(
        validate_natural_response_alpha_candidate(value, direction_receipt=direction)
        for value in _sequence(candidates, label="alpha candidates")
    )
    by_alpha = {float(value["alpha"]): value for value in values}
    if len(values) != len(NATURAL_RESPONSE_ALPHAS) or len(by_alpha) != len(values) or set(by_alpha) != set(NATURAL_RESPONSE_ALPHAS):
        raise ValueError("fit receipt requires the complete six-alpha ladder")
    ordered = tuple(by_alpha[alpha] for alpha in NATURAL_RESPONSE_ALPHAS)
    baseline = by_alpha[0.0]
    baseline_objective = float(baseline["family_equal_objective"])
    floor = _numerical_floor(baseline_objective)
    initial_weight_sha = _hash(_WEIGHT_DOMAIN, NATURAL_RESPONSE_INITIAL_WEIGHTS)
    eligible = tuple(
        value
        for value in ordered
        if float(value["alpha"]) > 0.0
        and str(value["weight_sha256"]) != initial_weight_sha
        and float(value["box_certificate"]) <= 1.0
        and any(float(weight) != 0.0 for weight in value["weights"])
        and float(value["projected_displacement_dot_gradient"]) < 0.0
        and baseline_objective - float(value["family_equal_objective"]) > floor
    )
    authorized = bool(eligible)
    selected = (
        min(
            eligible,
            key=lambda value: (
                float(value["family_equal_objective"]),
                float(value["alpha"]),
                str(value["weight_sha256"]),
            ),
        )
        if authorized
        else baseline
    )
    selected_objective = float(selected["family_equal_objective"])
    improvement = baseline_objective - selected_objective if authorized else 0.0
    alpha_positive = float(selected["alpha"]) > 0.0
    hash_changed = str(selected["weight_sha256"]) != initial_weight_sha
    box_passed = float(selected["box_certificate"]) <= 1.0
    nonconstant = any(float(value) != 0.0 for value in selected["weights"])
    descent = float(selected["projected_displacement_dot_gradient"]) < 0.0
    objective_improved = improvement > floor
    return _finish(
        "fisher_graph.complete_h4_fisher_natural_response_fit.v1",
        _FIT_DOMAIN,
        {
            "protocol_sha256": NATURAL_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "serving_claim_authorized": False,
            "compression_claim_authorized": False,
            "direction_receipt": direction,
            "direction_artifact_sha256": direction["artifact_sha256"],
            "response_law": direction["response_law"],
            "candidate_receipts": ordered,
            "candidate_artifact_sha256s": tuple(value["artifact_sha256"] for value in ordered),
            "alpha_ladder": NATURAL_RESPONSE_ALPHAS,
            "alpha_zero_objective": baseline_objective,
            "selected_alpha": float(selected["alpha"]),
            "selected_weights": selected["weights"],
            "initial_weight_sha256": initial_weight_sha,
            "selected_weight_sha256": selected["weight_sha256"],
            "selected_corner_values": selected["corner_values"],
            "selected_box_certificate": selected["box_certificate"],
            "selected_projected_displacement_dot_gradient": selected[
                "projected_displacement_dot_gradient"
            ],
            "selected_provider_artifact_sha256": selected["provider_artifact_sha256"] if authorized else None,
            "selected_candidate_artifact_sha256": selected["artifact_sha256"] if authorized else None,
            "selected_objective": selected_objective,
            "objective_absolute_improvement": improvement,
            "objective_numerical_improvement_floor": floor,
            "line_search_tie_order": "objective_then_smaller_alpha_then_canonical_weight_hash",
            "selected_alpha_is_positive": alpha_positive,
            "selected_weight_hash_changed": hash_changed,
            "selected_box_certificate_passed": box_passed,
            "selected_response_is_nonconstant": nonconstant,
            "selected_projected_displacement_is_descent": descent,
            "exact_fit_objective_improved": objective_improved,
            "learned_candidate_authorized": authorized,
            "rollback_to_initial_weights": not authorized,
            "held_score_authorized": authorized,
            "exact_fit_objectives_only": True,
            "raw_tensors_logits_targets_or_gradients_serialized": False,
        },
    )


def validate_natural_response_fit_receipt(
    value: Mapping[str, object],
    *,
    expected_v20c_source_sha256s: Mapping[str, str] | None = None,
    expected_base_provider_artifact_sha256: str | None = None,
    expected_proposal_provider_artifact_sha256: str | None = None,
) -> dict[str, object]:
    selected = _mapping(value, label="natural-response fit")
    _exact(selected, _FIT_KEYS, label="natural-response fit")
    if (
        selected["protocol_sha256"] != NATURAL_RESPONSE_PROTOCOL_SHA256
        or selected["scientific_status"] != "development_only_reused_a16"
        or selected["fresh_family_disjoint_claim_authorized"] is not False
        or selected["serving_claim_authorized"] is not False
        or selected["compression_claim_authorized"] is not False
        or selected["exact_fit_objectives_only"] is not True
        or selected["raw_tensors_logits_targets_or_gradients_serialized"] is not False
    ):
        raise ValueError("natural-response fit scientific boundary differs")
    direction = validate_natural_response_direction_receipt(
        _mapping(selected["direction_receipt"], label="direction receipt"),
        expected_v20c_source_sha256s=expected_v20c_source_sha256s,
        expected_base_provider_artifact_sha256=expected_base_provider_artifact_sha256,
        expected_proposal_provider_artifact_sha256=expected_proposal_provider_artifact_sha256,
    )
    rebuilt = build_natural_response_fit_receipt(
        direction_receipt=direction,
        candidates=tuple(
            _mapping(item, label="alpha candidate")
            for item in _sequence(selected["candidate_receipts"], label="alpha candidates")
        ),
    )
    _same(selected, rebuilt, label="natural-response fit")
    return rebuilt


_BUNDLE_KEYS = {
    "schema",
    "protocol_sha256",
    "scientific_status",
    "fresh_family_disjoint_claim_authorized",
    "serving_claim_authorized",
    "compression_claim_authorized",
    "law_order",
    "fit_receipts_by_law",
    "fit_artifact_sha256s_by_law",
    "selected_provider_artifact_sha256s_by_law",
    "selected_weight_sha256s_by_law",
    "selected_weights_by_law",
    "shared_v20c_source_sha256s",
    "family_ids",
    "excluded_family_ids",
    "fit_family_ids",
    "base_provider_artifact_sha256",
    "proposal_provider_artifact_sha256",
    "alpha_ladder",
    "both_fits_authorized",
    "held_score_authorized",
    "raw_tensors_logits_targets_or_gradients_serialized",
    "artifact_sha256",
}


def build_natural_response_two_fit_bundle_receipt(
    *,
    signed_log_fit_receipt: Mapping[str, object],
    linear_fit_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Bind two independently differentiated and selected response-law fits."""

    fits = {
        "signed_log": validate_natural_response_fit_receipt(signed_log_fit_receipt),
        "linear": validate_natural_response_fit_receipt(linear_fit_receipt),
    }
    for law in NATURAL_RESPONSE_LAWS:
        if fits[law]["response_law"] != law:
            raise ValueError(f"{law} fit receipt carries the wrong response law")
    directions = {law: fits[law]["direction_receipt"] for law in NATURAL_RESPONSE_LAWS}
    shared_fields = (
        "v20c_source_sha256s",
        "family_ids",
        "excluded_family_ids",
        "fit_family_ids",
        "base_provider_artifact_sha256",
        "proposal_provider_artifact_sha256",
        "protocol_sha256",
    )
    for field in shared_fields:
        if _canonical(directions["signed_log"][field]) != _canonical(directions["linear"][field]):
            raise ValueError(f"two-fit bundle {field} lineage differs")
    if (
        directions["signed_log"]["artifact_sha256"]
        == directions["linear"]["artifact_sha256"]
        or directions["signed_log"]["gradient_evidence_sha256"]
        == directions["linear"]["gradient_evidence_sha256"]
        or fits["signed_log"]["artifact_sha256"] == fits["linear"]["artifact_sha256"]
    ):
        raise ValueError("two-fit bundle requires independent law-specific fit evidence")
    if fits["signed_log"]["alpha_ladder"] != fits["linear"]["alpha_ladder"]:
        raise ValueError("two-fit bundle alpha ladders differ")
    both_authorized = all(
        fits[law]["learned_candidate_authorized"] is True
        and fits[law]["held_score_authorized"] is True
        for law in NATURAL_RESPONSE_LAWS
    )
    provider_hashes: dict[str, str | None] = {}
    for law in NATURAL_RESPONSE_LAWS:
        value = fits[law]["selected_provider_artifact_sha256"]
        provider_hashes[law] = None if value is None else _sha(value, label=f"{law} selected provider")
    if both_authorized and (
        any(value is None for value in provider_hashes.values())
        or len(set(provider_hashes.values())) != len(provider_hashes)
    ):
        raise ValueError("learned linear and signed-log providers must be distinct")
    return _finish(
        "fisher_graph.complete_h4_fisher_natural_response_two_fit_bundle.v1",
        _BUNDLE_DOMAIN,
        {
            "protocol_sha256": NATURAL_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "serving_claim_authorized": False,
            "compression_claim_authorized": False,
            "law_order": NATURAL_RESPONSE_LAWS,
            "fit_receipts_by_law": fits,
            "fit_artifact_sha256s_by_law": {
                law: fits[law]["artifact_sha256"] for law in NATURAL_RESPONSE_LAWS
            },
            "selected_provider_artifact_sha256s_by_law": provider_hashes,
            "selected_weight_sha256s_by_law": {
                law: fits[law]["selected_weight_sha256"] for law in NATURAL_RESPONSE_LAWS
            },
            "selected_weights_by_law": {
                law: fits[law]["selected_weights"] for law in NATURAL_RESPONSE_LAWS
            },
            "shared_v20c_source_sha256s": directions["signed_log"]["v20c_source_sha256s"],
            "family_ids": directions["signed_log"]["family_ids"],
            "excluded_family_ids": directions["signed_log"]["excluded_family_ids"],
            "fit_family_ids": directions["signed_log"]["fit_family_ids"],
            "base_provider_artifact_sha256": directions["signed_log"]["base_provider_artifact_sha256"],
            "proposal_provider_artifact_sha256": directions["signed_log"]["proposal_provider_artifact_sha256"],
            "alpha_ladder": fits["signed_log"]["alpha_ladder"],
            "both_fits_authorized": both_authorized,
            "held_score_authorized": both_authorized,
            "raw_tensors_logits_targets_or_gradients_serialized": False,
        },
    )


def validate_natural_response_two_fit_bundle_receipt(
    value: Mapping[str, object],
    *,
    expected_v20c_source_sha256s: Mapping[str, str] | None = None,
    expected_base_provider_artifact_sha256: str | None = None,
    expected_proposal_provider_artifact_sha256: str | None = None,
) -> dict[str, object]:
    selected = _mapping(value, label="natural-response two-fit bundle")
    _exact(selected, _BUNDLE_KEYS, label="natural-response two-fit bundle")
    if (
        selected["protocol_sha256"] != NATURAL_RESPONSE_PROTOCOL_SHA256
        or selected["scientific_status"] != "development_only_reused_a16"
        or selected["fresh_family_disjoint_claim_authorized"] is not False
        or selected["serving_claim_authorized"] is not False
        or selected["compression_claim_authorized"] is not False
        or tuple(_sequence(selected["law_order"], label="two-fit law order")) != NATURAL_RESPONSE_LAWS
        or type(selected["both_fits_authorized"]) is not bool
        or type(selected["held_score_authorized"]) is not bool
        or selected["raw_tensors_logits_targets_or_gradients_serialized"] is not False
    ):
        raise ValueError("natural-response two-fit bundle scientific boundary differs")
    fit_source = _mapping(selected["fit_receipts_by_law"], label="two-fit receipts")
    if set(fit_source) != set(NATURAL_RESPONSE_LAWS):
        raise ValueError("two-fit bundle laws differ")
    rebuilt = build_natural_response_two_fit_bundle_receipt(
        signed_log_fit_receipt=_mapping(fit_source["signed_log"], label="signed-log fit"),
        linear_fit_receipt=_mapping(fit_source["linear"], label="linear fit"),
    )
    _same(selected, rebuilt, label="natural-response two-fit bundle")
    if expected_v20c_source_sha256s is not None and rebuilt["shared_v20c_source_sha256s"] != _sources(expected_v20c_source_sha256s):
        raise ValueError("two-fit bundle V20c source lineage differs")
    if expected_base_provider_artifact_sha256 is not None and rebuilt["base_provider_artifact_sha256"] != _sha(expected_base_provider_artifact_sha256, label="expected base provider"):
        raise ValueError("two-fit bundle base endpoint differs")
    if expected_proposal_provider_artifact_sha256 is not None and rebuilt["proposal_provider_artifact_sha256"] != _sha(expected_proposal_provider_artifact_sha256, label="expected proposal provider"):
        raise ValueError("two-fit bundle proposal endpoint differs")
    return rebuilt


_SCORE_KEYS = {
    "schema",
    "fit_bundle_artifact_sha256",
    "outer_held_family_id",
    "held_family_id",
    "arm",
    "objective",
    "provider_artifact_sha256",
    "execution_receipt_sha256",
    "score_source",
    "predicted_only",
    "finite",
    "pointwise_trust_passed",
    "rank_is_16",
    "execution_changed_from_base",
    "response_nonconstant",
    "artifact_sha256",
}


def build_natural_response_held_arm_score(
    *,
    fit_bundle_receipt: Mapping[str, object],
    outer_held_family_id: str,
    held_family_id: str,
    arm: str,
    objective: float,
    provider_artifact_sha256: str,
    execution_receipt_sha256: str,
    finite: bool,
    pointwise_trust_passed: bool,
    rank_is_16: bool,
    execution_changed_from_base: bool,
    response_nonconstant: bool,
    score_source: str = "exact_finite_held_execution",
) -> dict[str, object]:
    bundle = validate_natural_response_two_fit_bundle_receipt(fit_bundle_receipt)
    if bundle["held_score_authorized"] is not True:
        raise ValueError("two-fit line search did not authorize a learned held score")
    outer = _identifier(outer_held_family_id, label="outer held family")
    held = _identifier(held_family_id, label="held family")
    if outer == held or tuple(sorted((outer, held))) != tuple(bundle["excluded_family_ids"]):
        raise ValueError("held score family geometry differs")
    selected_arm = _identifier(arm, label="natural-response arm")
    if selected_arm not in NATURAL_RESPONSE_ARMS:
        raise ValueError("natural-response arm differs")
    if score_source != "exact_finite_held_execution":
        raise ValueError("natural-response held scores require exact finite execution")
    selected_objective = _number(objective, label="held objective", nonnegative=True)
    selected_provider = _sha(provider_artifact_sha256, label="held provider")
    learned_providers = bundle["selected_provider_artifact_sha256s_by_law"]
    if selected_arm == "learned_signed_log" and selected_provider != learned_providers["signed_log"]:
        raise ValueError("learned signed-log arm provider differs from its fit")
    if selected_arm == "learned_linear" and selected_provider != learned_providers["linear"]:
        raise ValueError("learned linear arm provider differs from its fit")
    changed = _boolean(execution_changed_from_base, label="execution change")
    if selected_arm == "base" and changed:
        raise ValueError("base execution cannot differ from itself")
    return _finish(
        "fisher_graph.complete_h4_fisher_natural_response_held_score.v1",
        _SCORE_DOMAIN,
        {
            "fit_bundle_artifact_sha256": bundle["artifact_sha256"],
            "outer_held_family_id": outer,
            "held_family_id": held,
            "arm": selected_arm,
            "objective": selected_objective,
            "provider_artifact_sha256": selected_provider,
            "execution_receipt_sha256": _sha(execution_receipt_sha256, label="held execution"),
            "score_source": "exact_finite_held_execution",
            "predicted_only": False,
            "finite": _boolean(finite, label="held finite"),
            "pointwise_trust_passed": _boolean(pointwise_trust_passed, label="held trust"),
            "rank_is_16": _boolean(rank_is_16, label="held rank"),
            "execution_changed_from_base": changed,
            "response_nonconstant": _boolean(
                response_nonconstant, label="held response nonconstant"
            ),
        },
    )


def validate_natural_response_held_arm_score(
    value: Mapping[str, object], *, fit_bundle_receipt: Mapping[str, object]
) -> dict[str, object]:
    selected = _mapping(value, label="natural-response held score")
    _exact(selected, _SCORE_KEYS, label="natural-response held score")
    if selected["predicted_only"] is not False:
        raise ValueError("predicted-only held score is forbidden")
    rebuilt = build_natural_response_held_arm_score(
        fit_bundle_receipt=fit_bundle_receipt,
        outer_held_family_id=selected["outer_held_family_id"],
        held_family_id=selected["held_family_id"],
        arm=selected["arm"],
        objective=selected["objective"],
        provider_artifact_sha256=selected["provider_artifact_sha256"],
        execution_receipt_sha256=selected["execution_receipt_sha256"],
        finite=selected["finite"],
        pointwise_trust_passed=selected["pointwise_trust_passed"],
        rank_is_16=selected["rank_is_16"],
        execution_changed_from_base=selected["execution_changed_from_base"],
        response_nonconstant=selected["response_nonconstant"],
        score_source=selected["score_source"],
    )
    _same(selected, rebuilt, label="natural-response held score")
    return rebuilt


_ROLE_KEYS = {
    "schema",
    "fit_bundle_artifact_sha256",
    "outer_held_family_id",
    "held_family_id",
    "arm_scores",
    "arm_objectives",
    "arm_provider_artifact_sha256s",
    "arm_execution_receipt_sha256s",
    "all_arms_exactly_executed",
    "artifact_sha256",
}


def build_natural_response_held_role_receipt(
    *, fit_bundle_receipt: Mapping[str, object], arm_scores: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    bundle = validate_natural_response_two_fit_bundle_receipt(fit_bundle_receipt)
    scores = tuple(
        validate_natural_response_held_arm_score(value, fit_bundle_receipt=bundle)
        for value in _sequence(arm_scores, label="held arm scores")
    )
    by_arm = {str(value["arm"]): value for value in scores}
    if len(scores) != len(NATURAL_RESPONSE_ARMS) or len(by_arm) != len(scores) or set(by_arm) != set(NATURAL_RESPONSE_ARMS):
        raise ValueError("held role requires all seven unique arms")
    ordered = tuple(by_arm[arm] for arm in NATURAL_RESPONSE_ARMS)
    outer_ids = {value["outer_held_family_id"] for value in ordered}
    held_ids = {value["held_family_id"] for value in ordered}
    if len(outer_ids) != 1 or len(held_ids) != 1:
        raise ValueError("held role family bindings differ")
    execution_hashes = tuple(str(value["execution_receipt_sha256"]) for value in ordered)
    provider_hashes = tuple(str(value["provider_artifact_sha256"]) for value in ordered)
    if len(set(execution_hashes)) != len(execution_hashes) or len(set(provider_hashes)) != len(provider_hashes):
        raise ValueError("held role arms require distinct providers and executions")
    return _finish(
        "fisher_graph.complete_h4_fisher_natural_response_held_role.v1",
        _ROLE_DOMAIN,
        {
            "fit_bundle_artifact_sha256": bundle["artifact_sha256"],
            "outer_held_family_id": next(iter(outer_ids)),
            "held_family_id": next(iter(held_ids)),
            "arm_scores": ordered,
            "arm_objectives": {arm: by_arm[arm]["objective"] for arm in NATURAL_RESPONSE_ARMS},
            "arm_provider_artifact_sha256s": {arm: by_arm[arm]["provider_artifact_sha256"] for arm in NATURAL_RESPONSE_ARMS},
            "arm_execution_receipt_sha256s": {arm: by_arm[arm]["execution_receipt_sha256"] for arm in NATURAL_RESPONSE_ARMS},
            "all_arms_exactly_executed": True,
        },
    )


def validate_natural_response_held_role_receipt(
    value: Mapping[str, object], *, fit_bundle_receipt: Mapping[str, object]
) -> dict[str, object]:
    selected = _mapping(value, label="natural-response held role")
    _exact(selected, _ROLE_KEYS, label="natural-response held role")
    if selected["all_arms_exactly_executed"] is not True:
        raise ValueError("held role lacks exact arm executions")
    rebuilt = build_natural_response_held_role_receipt(
        fit_bundle_receipt=fit_bundle_receipt,
        arm_scores=tuple(
            _mapping(item, label="held arm score")
            for item in _sequence(selected["arm_scores"], label="held arm scores")
        ),
    )
    _same(selected, rebuilt, label="natural-response held role")
    return rebuilt


_QUALIFICATION_KEYS = {
    "schema",
    "protocol_sha256",
    "scientific_status",
    "fresh_family_disjoint_claim_authorized",
    "serving_claim_authorized",
    "compression_claim_authorized",
    "fit_bundle_artifact_sha256",
    "role_artifact_sha256s",
    "aggregation",
    "arm_macro_objectives",
    "base_macro_denominator_valid",
    "learned_signed_log_base_macro_relative_improvement",
    "required_base_macro_relative_improvement",
    "learned_signed_log_improves_both_roles",
    "learned_signed_log_fixed_log_macro_relative_improvement",
    "fixed_log_macro_denominator_valid",
    "required_fixed_log_macro_relative_improvement",
    "learned_signed_log_beats_constant_macro",
    "learned_signed_log_beats_learned_linear_macro",
    "learned_signed_log_beats_mirror_both_roles",
    "learned_signed_log_nonconstant_both_roles",
    "all_arms_finite_trusted_rank16_changed_exact",
    "passed",
    "artifact_sha256",
}


def build_natural_response_pair_qualification(
    *, fit_bundle_receipt: Mapping[str, object], roles: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    bundle = validate_natural_response_two_fit_bundle_receipt(fit_bundle_receipt)
    if bundle["held_score_authorized"] is not True:
        raise ValueError("natural-response two-fit bundle did not authorize held qualification")
    selected_roles = tuple(
        validate_natural_response_held_role_receipt(value, fit_bundle_receipt=bundle)
        for value in _sequence(roles, label="held roles")
    )
    if len(selected_roles) != _HELD_ROLE_COUNT or len({value["artifact_sha256"] for value in selected_roles}) != _HELD_ROLE_COUNT:
        raise ValueError("pair qualification requires two unique reciprocal roles")
    pairs = {(value["outer_held_family_id"], value["held_family_id"]) for value in selected_roles}
    excluded = tuple(bundle["excluded_family_ids"])
    expected_pairs = {(excluded[0], excluded[1]), (excluded[1], excluded[0])}
    if pairs != expected_pairs:
        raise ValueError("pair qualification roles are not reciprocal")
    ordered = tuple(sorted(selected_roles, key=lambda value: str(value["outer_held_family_id"])))
    objectives = {
        arm: tuple(float(_mapping(role["arm_objectives"], label="arm objectives")[arm]) for role in ordered)
        for arm in NATURAL_RESPONSE_ARMS
    }
    macros = {arm: math.fsum(values) / _HELD_ROLE_COUNT for arm, values in objectives.items()}
    learned = objectives["learned_signed_log"]
    base = objectives["base"]
    fixed = objectives["fixed_signed_log"]
    constant = objectives["constant_plus_one"]
    linear = objectives["learned_linear"]
    mirror = objectives["learned_signed_log_sign_flip"]
    base_denominator_valid = macros["base"] > _numerical_floor(macros["base"])
    fixed_denominator_valid = macros["fixed_signed_log"] > _numerical_floor(macros["fixed_signed_log"])
    base_relative = (
        (macros["base"] - macros["learned_signed_log"]) / macros["base"]
        if base_denominator_valid
        else 0.0
    )
    fixed_relative = (
        (macros["fixed_signed_log"] - macros["learned_signed_log"]) / macros["fixed_signed_log"]
        if fixed_denominator_valid
        else 0.0
    )
    improves_both = all(base_value - learned_value > _numerical_floor(base_value) for base_value, learned_value in zip(base, learned))
    beats_constant = macros["constant_plus_one"] - macros["learned_signed_log"] > _numerical_floor(macros["constant_plus_one"])
    beats_linear = macros["learned_linear"] - macros["learned_signed_log"] > _numerical_floor(macros["learned_linear"])
    beats_mirror_both = all(mirror_value - learned_value > _numerical_floor(mirror_value) for mirror_value, learned_value in zip(mirror, learned))
    learned_nonconstant_both = all(
        next(
            score
            for score in role["arm_scores"]
            if score["arm"] == "learned_signed_log"
        )["response_nonconstant"]
        is True
        for role in ordered
    )
    health = all(
        score["finite"] is True
        and score["pointwise_trust_passed"] is True
        and score["rank_is_16"] is True
        and score["score_source"] == "exact_finite_held_execution"
        and score["predicted_only"] is False
        and (score["arm"] == "base" or score["execution_changed_from_base"] is True)
        for role in ordered
        for score in role["arm_scores"]
    )
    passed = bool(
        base_denominator_valid
        and base_relative >= _BASE_MATERIALITY
        and improves_both
        and fixed_denominator_valid
        and fixed_relative >= _FIXED_LOG_MATERIALITY
        and beats_constant
        and beats_linear
        and beats_mirror_both
        and learned_nonconstant_both
        and health
    )
    return _finish(
        "fisher_graph.complete_h4_fisher_natural_response_pair_qualification.v1",
        _QUALIFICATION_DOMAIN,
        {
            "protocol_sha256": NATURAL_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "serving_claim_authorized": False,
            "compression_claim_authorized": False,
            "fit_bundle_artifact_sha256": bundle["artifact_sha256"],
            "role_artifact_sha256s": tuple(value["artifact_sha256"] for value in ordered),
            "aggregation": "family_equal_two_reciprocal_roles",
            "arm_macro_objectives": macros,
            "base_macro_denominator_valid": base_denominator_valid,
            "learned_signed_log_base_macro_relative_improvement": base_relative,
            "required_base_macro_relative_improvement": _BASE_MATERIALITY,
            "learned_signed_log_improves_both_roles": improves_both,
            "learned_signed_log_fixed_log_macro_relative_improvement": fixed_relative,
            "fixed_log_macro_denominator_valid": fixed_denominator_valid,
            "required_fixed_log_macro_relative_improvement": _FIXED_LOG_MATERIALITY,
            "learned_signed_log_beats_constant_macro": beats_constant,
            "learned_signed_log_beats_learned_linear_macro": beats_linear,
            "learned_signed_log_beats_mirror_both_roles": beats_mirror_both,
            "learned_signed_log_nonconstant_both_roles": learned_nonconstant_both,
            "all_arms_finite_trusted_rank16_changed_exact": health,
            "passed": passed,
        },
    )


def validate_natural_response_pair_qualification(
    value: Mapping[str, object], *, fit_bundle_receipt: Mapping[str, object], roles: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    selected = _mapping(value, label="natural-response pair qualification")
    _exact(selected, _QUALIFICATION_KEYS, label="natural-response pair qualification")
    rebuilt = build_natural_response_pair_qualification(fit_bundle_receipt=fit_bundle_receipt, roles=roles)
    _same(selected, rebuilt, label="natural-response pair qualification")
    return rebuilt


def natural_response_work_accounting(
    *,
    collection_forward_count: int,
    collection_backward_count: int,
    endpoint_forward_count: int,
    endpoint_backward_count: int,
    endpoint_local_contraction_count: int,
    law_count: int,
    fit_prompt_count: int,
    alpha_count: int,
    alpha_zero_vjp_reused: bool,
    held_role_count: int,
    held_arm_count: int,
    held_prompts_per_role: int,
    held_scoring_executed: bool,
) -> dict[str, int | bool]:
    """Return exact V20d work for either the fit terminal or held path."""

    collection_forward = _integer(collection_forward_count, label="collection forward count")
    collection_backward = _integer(collection_backward_count, label="collection backward count")
    endpoint_forward = _integer(endpoint_forward_count, label="endpoint forward count")
    endpoint_backward = _integer(endpoint_backward_count, label="endpoint backward count")
    endpoint_contractions = _integer(
        endpoint_local_contraction_count, label="endpoint local contraction count"
    )
    laws = _integer(law_count, label="law count", minimum=1)
    fit_prompts = _integer(fit_prompt_count, label="fit prompt count", minimum=1)
    alphas = _integer(alpha_count, label="alpha count", minimum=1)
    reuse_alpha_zero = _boolean(alpha_zero_vjp_reused, label="alpha-zero VJP reuse")
    held_roles = _integer(held_role_count, label="held role count", minimum=1)
    held_arms = _integer(held_arm_count, label="held arm count", minimum=1)
    held_prompts = _integer(
        held_prompts_per_role, label="held prompts per role", minimum=1
    )
    held_executed = _boolean(held_scoring_executed, label="held scoring executed")
    if laws != len(NATURAL_RESPONSE_LAWS):
        raise ValueError("V20d requires exactly two independently fit response laws")
    if alphas != len(NATURAL_RESPONSE_ALPHAS):
        raise ValueError("V20d alpha-count differs from the frozen ladder")
    if held_roles != _HELD_ROLE_COUNT or held_arms != len(NATURAL_RESPONSE_ARMS):
        raise ValueError("V20d held geometry differs")

    law_fit_gradient_forward = laws * fit_prompts
    law_fit_gradient_backward = laws * fit_prompts
    law_fit_gradient_contractions = laws * fit_prompts
    evaluated_alphas = alphas - (1 if reuse_alpha_zero else 0)
    alpha_fit_forward = laws * evaluated_alphas * fit_prompts
    held_forward = held_roles * held_arms * held_prompts if held_executed else 0
    total_forward = (
        collection_forward
        + endpoint_forward
        + law_fit_gradient_forward
        + alpha_fit_forward
        + held_forward
    )
    total_backward = collection_backward + endpoint_backward + law_fit_gradient_backward
    total_contractions = endpoint_contractions + law_fit_gradient_contractions
    gradient_calls = total_backward + total_contractions
    teacher_accesses = (
        endpoint_forward + law_fit_gradient_forward + alpha_fit_forward + held_forward
    )
    return {
        "collection_forward_count": collection_forward,
        "collection_backward_count": collection_backward,
        "endpoint_forward_count": endpoint_forward,
        "endpoint_backward_count": endpoint_backward,
        "endpoint_local_contraction_count": endpoint_contractions,
        "law_count": laws,
        "fit_prompt_count": fit_prompts,
        "alpha_count": alphas,
        "alpha_zero_vjp_reused": reuse_alpha_zero,
        "held_role_count": held_roles,
        "held_arm_count": held_arms,
        "held_prompts_per_role": held_prompts,
        "held_scoring_executed": held_executed,
        "law_fit_gradient_forward_count": law_fit_gradient_forward,
        "law_fit_gradient_backward_count": law_fit_gradient_backward,
        "law_fit_gradient_local_contraction_count": law_fit_gradient_contractions,
        "fit_example_gradient_count": laws * fit_prompts,
        "empirical_fisher_outer_product_count": laws * fit_prompts,
        "natural_direction_solve_count": laws,
        "fit_candidate_count": laws * alphas,
        "fit_candidate_prompt_score_count": laws * alphas * fit_prompts,
        "fit_candidate_example_execution_count": laws * alphas * fit_prompts,
        "alpha_zero_reused_fit_score_count": laws * fit_prompts if reuse_alpha_zero else 0,
        "alpha_fit_forward_count": alpha_fit_forward,
        "held_forward_count": held_forward,
        "held_role_arm_score_count": held_roles * held_arms if held_executed else 0,
        "held_arm_prompt_score_count": held_forward,
        "held_arm_example_execution_count": held_forward,
        "total_model_forward_count": total_forward,
        "total_model_backward_count": total_backward,
        "total_local_contraction_count": total_contractions,
        "total_backward_or_local_contraction_count": gradient_calls,
        "non_collection_forward_count": teacher_accesses,
        "teacher_h4_logit_access_count": teacher_accesses,
        "teacher_h4_access_count": teacher_accesses,
        "teacher_logit_check_count": teacher_accesses,
    }
