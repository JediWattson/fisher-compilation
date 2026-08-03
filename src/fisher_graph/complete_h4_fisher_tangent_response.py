"""Pure V20e tangent-constrained response fitting protocol.

V20e fixes the boundary-collapse failure observed in V20d.  The initial
bilinear response ``w0=(0,1,0)`` already lies on ``B(w)=1``.  Rather than take
an unconstrained natural step and radially project it back to that boundary,
V20e solves the three-dimensional empirical-Fisher quadratic inside the exact
tangent cone, computes the complete feasible ray, and selects a convex
fraction by six-fold leave-one-family-out validation.

This module owns deterministic scalar mathematics and hash receipts only.
Raw gradients are accepted by builders but are represented in receipts only by
per-example hashes and family-equal scalar statistics.  Tensors, logits,
targets, prompts, and raw gradients are never serialized.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
import hashlib
import json
import math
import re
import sys

__all__ = [
    "TANGENT_RESPONSE_ARMS",
    "TANGENT_RESPONSE_FRACTIONS",
    "TANGENT_RESPONSE_INITIAL_WEIGHTS",
    "TANGENT_RESPONSE_KAPPA",
    "TANGENT_RESPONSE_LAWS",
    "TANGENT_RESPONSE_PROTOCOL_SHA256",
    "bilinear_box_certificate",
    "bilinear_corner_values",
    "build_tangent_response_cv_candidate",
    "build_tangent_response_cv_receipt",
    "build_tangent_response_gradient_bank_receipt",
    "build_tangent_response_direction_from_gradient_bank_receipt",
    "build_tangent_response_direction_receipt",
    "build_tangent_response_final_candidate_receipt",
    "build_tangent_response_fit_receipt",
    "build_tangent_response_held_arm_score",
    "build_tangent_response_held_role_receipt",
    "build_tangent_response_pair_qualification",
    "build_tangent_response_ray_receipt",
    "build_tangent_response_two_fit_bundle_receipt",
    "signed_log_response",
    "tangent_response_features",
    "tangent_response_gain",
    "tangent_response_fraction_proposal",
    "tangent_response_work_accounting",
    "validate_tangent_response_cv_candidate",
    "validate_tangent_response_cv_receipt",
    "validate_tangent_response_gradient_bank_receipt",
    "validate_tangent_response_direction_receipt",
    "validate_tangent_response_final_candidate_receipt",
    "validate_tangent_response_fit_receipt",
    "validate_tangent_response_held_arm_score",
    "validate_tangent_response_held_role_receipt",
    "validate_tangent_response_pair_qualification",
    "validate_tangent_response_ray_receipt",
    "validate_tangent_response_two_fit_bundle_receipt",
]


TANGENT_RESPONSE_KAPPA = 9.0
TANGENT_RESPONSE_LAWS = ("signed_log", "linear")
TANGENT_RESPONSE_INITIAL_WEIGHTS = (0.0, 1.0, 0.0)
TANGENT_RESPONSE_FRACTIONS = (
    0.0,
    1.0 / 256.0,
    1.0 / 128.0,
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
)
TANGENT_RESPONSE_ARMS = (
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
_EXCLUDED_COUNT = 2
_FIT_FAMILY_COUNT = 6
_CV_TRAIN_FAMILY_COUNT = 5
_HELD_ROLE_COUNT = 2
_FEATURE_COUNT = 3
_BASE_MATERIALITY = 0.01
_FIXED_LOG_MATERIALITY = 0.001
_ABSOLUTE_FLOOR = 1.0e-12
_ROUNDOFF_MULTIPLIER = 128.0

_PROTOCOL_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:protocol:v20e\0"
_WEIGHT_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:weights:v20e\0"
_GRADIENT_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:gradient:v20e\0"
_GRADIENT_FAMILY_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:gradient-family:v20e\0"
_GRADIENT_BANK_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:gradient-bank:v20e\0"
_GRADIENT_SUBSET_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:gradient-subset:v20e\0"
_DIRECTION_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:direction:v20e\0"
_RAY_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:ray:v20e\0"
_CV_CANDIDATE_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:cv-candidate:v20e\0"
_CV_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:cv:v20e\0"
_FINAL_CANDIDATE_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:final-candidate:v20e\0"
_FIT_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:fit:v20e\0"
_BUNDLE_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:two-fit-bundle:v20e\0"
_SCORE_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:score:v20e\0"
_ROLE_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:role:v20e\0"
_QUALIFICATION_DOMAIN = b"fisher-graph:complete-h4-fisher-tangent-response:qualification:v20e\0"

_CORNER_FEATURES = (
    (-1.0, -1.0, 1.0),
    (-1.0, 1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, 1.0),
)
_CORNER_LABELS = ("--", "-+", "+-", "++")
# sign(a.w0) * a for each active box face at w0.  These four rows are
# equivalent to d2 + |d1| + |d12| <= 0.
_TANGENT_NORMALS = (
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
)
_INWARD_DIRECTION = (0.0, -1.0, 0.0)

_FIXED_PROTOCOL = {
    "protocol": "six_fold_tangent_constrained_natural_response_v20e",
    "scientific_status": "development_only_reused_a16",
    "fresh_family_disjoint_claim_authorized": False,
    "serving_claim_authorized": False,
    "compression_claim_authorized": False,
    "features": ("c1", "c2", "c1_times_c2"),
    "intercept": False,
    "initial_weights": TANGENT_RESPONSE_INITIAL_WEIGHTS,
    "response_laws": TANGENT_RESPONSE_LAWS,
    "signed_log_kappa": TANGENT_RESPONSE_KAPPA,
    "fisher": "family_equal_response_parameter_empirical_gradient_Fisher_OPG",
    "gradient_bank": "one_law_specific_six_family_commitment_with_two_rows_per_family",
    "gradient_bank_reuse": "all_lofo_and_all_six_directions_aggregate_the_same_hash_only_family_summaries",
    "damping": "max_1e-12_1e-3_times_trace_over_three",
    "tangent_geometry": "four_exact_active_signed_corner_inequalities_at_w0",
    "tangent_equivalent": "d2_plus_abs_d1_plus_abs_d12_at_most_zero",
    "solver": "deterministic_exhaustive_active_set_convex_qp",
    "roundoff": "exact_inward_correction_along_negative_w2",
    "ray": "exact_opposite_box_face_maximum_then_inward_float_correction",
    "fraction_ladder": TANGENT_RESPONSE_FRACTIONS,
    "cv": "six_fold_leave_one_fit_family_out",
    "cv_selection": "macro_improvement_and_at_least_four_of_six_folds_improved",
    "cv_tie": "objective_then_smaller_fraction_then_candidate_hashes",
    "final_direction": "recomputed_from_all_six_fit_families",
    "rollback": "no_held_capability_unless_oof_cv_and_structural_all_six_fit_support_authorize",
    "held_arms": TANGENT_RESPONSE_ARMS,
    "held_base_materiality": _BASE_MATERIALITY,
    "held_fixed_log_materiality": _FIXED_LOG_MATERIALITY,
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


TANGENT_RESPONSE_PROTOCOL_SHA256 = _hash(_PROTOCOL_DOMAIN, _FIXED_PROTOCOL)


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


def _families(value: object, *, label: str, count: int) -> tuple[str, ...]:
    result = tuple(sorted(_identifier(item, label=label) for item in _sequence(value, label=label)))
    if len(result) != count or len(set(result)) != count:
        raise ValueError(f"{label} geometry differs")
    return result


def _sources(value: object) -> dict[str, str]:
    source = _mapping(value, label="source artifact hashes")
    if not source:
        raise ValueError("source artifact hashes must not be empty")
    return dict(
        sorted(
            (
                _identifier(key, label="source artifact name"),
                _sha(item, label=f"source artifact {key}"),
            )
            for key, item in source.items()
        )
    )


def _numerical_floor(reference: float) -> float:
    selected = _number(reference, label="numerical-floor reference", nonnegative=True)
    return max(_ABSOLUTE_FLOOR, _ROUNDOFF_MULTIPLIER * sys.float_info.epsilon * abs(selected))


def _law(value: object) -> str:
    selected = _identifier(value, label="response law")
    if selected not in TANGENT_RESPONSE_LAWS:
        raise ValueError("tangent-response law differs")
    return selected


def tangent_response_features(c1: float, c2: float) -> tuple[float, float, float]:
    left = _number(c1, label="c1")
    right = _number(c2, label="c2")
    if abs(left) > 1.0 or abs(right) > 1.0:
        raise ValueError("tangent-response coordinates must lie in [-1, 1]")
    return left, right, left * right


def signed_log_response(value: float) -> float:
    selected = _number(value, label="signed-log response input")
    if abs(selected) > 1.0 + 1.0e-15:
        raise ValueError("signed-log response input escaped the certified box")
    return math.asinh(TANGENT_RESPONSE_KAPPA * selected) / math.asinh(TANGENT_RESPONSE_KAPPA)


def bilinear_corner_values(weights: Sequence[float]) -> tuple[float, float, float, float]:
    selected = _vector(weights, label="bilinear weights")
    values = tuple(
        math.fsum(weight * feature for weight, feature in zip(selected, corner))
        for corner in _CORNER_FEATURES
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bilinear corner value became nonfinite")
    return values  # type: ignore[return-value]


def bilinear_box_certificate(weights: Sequence[float]) -> float:
    # For this no-intercept bilinear form the four-corner maximum is also the
    # exact L1 norm.  Recomputing corners keeps the provider-facing certificate
    # independent of that simplification.
    return max(abs(value) for value in bilinear_corner_values(weights))


def tangent_response_gain(
    weights: Sequence[float], c1: float, c2: float, *, law: str = "signed_log"
) -> float:
    selected = _vector(weights, label="tangent-response weights")
    if bilinear_box_certificate(selected) > 1.0:
        raise ValueError("tangent-response weights lack an exact box certificate")
    projection = math.fsum(
        weight * feature
        for weight, feature in zip(selected, tangent_response_features(c1, c2))
    )
    selected_law = _law(law)
    return projection if selected_law == "linear" else signed_log_response(projection)


def _outer(vector: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(left * right for right in vector) for left in vector)


def _require_psd3(
    matrix: Sequence[Sequence[float]], *, label: str
) -> tuple[tuple[float, float, float], ...]:
    """Validate a symmetric 3x3 PSD matrix with scale-aware roundoff slack."""

    selected = _matrix3(matrix, label=label)
    for row in range(_FEATURE_COUNT):
        for column in range(_FEATURE_COUNT):
            if selected[row][column] != selected[column][row]:
                raise ValueError(f"{label} must be exactly symmetric")
    scale = max(1.0, *(abs(item) for row in selected for item in row))
    base_tolerance = 65536.0 * sys.float_info.epsilon
    one_tolerance = base_tolerance * scale
    two_tolerance = base_tolerance * scale * scale
    three_tolerance = base_tolerance * scale * scale * scale
    if any(selected[index][index] < -one_tolerance for index in range(3)):
        raise ValueError(f"{label} must be positive semidefinite")
    for left, right in combinations(range(3), 2):
        minor = (
            selected[left][left] * selected[right][right]
            - selected[left][right] * selected[right][left]
        )
        if minor < -two_tolerance:
            raise ValueError(f"{label} must be positive semidefinite")
    a, b, c = selected[0]
    d, e, f = selected[1]
    g, h, i = selected[2]
    determinant = math.fsum((a * (e * i - f * h), -b * (d * i - f * g), c * (d * h - e * g)))
    if determinant < -three_tolerance:
        raise ValueError(f"{label} must be positive semidefinite")
    return selected


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, ...]:
    return tuple(_dot(row, vector) for row in matrix)


def _solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> tuple[float, ...]:
    """Deterministic partial-pivot Gaussian solve; singular systems are rejected."""

    size = len(rhs)
    work = [[float(matrix[row][column]) for column in range(size)] + [float(rhs[row])] for row in range(size)]
    # Singularity is a property of the coefficient matrix, not the RHS.  A
    # large gradient must not make an otherwise well-conditioned Hessian look
    # singular.
    scale = max(
        1.0,
        *(abs(float(matrix[row][column])) for row in range(size) for column in range(size)),
    )
    singular_floor = 4096.0 * sys.float_info.epsilon * scale
    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: (abs(work[row][column]), -row))
        if abs(work[pivot_row][column]) <= singular_floor:
            raise ValueError("active-set KKT system is numerically singular")
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
        pivot = work[column][column]
        for row in range(column + 1, size):
            multiplier = work[row][column] / pivot
            if multiplier == 0.0:
                continue
            work[row][column] = 0.0
            for index in range(column + 1, size + 1):
                work[row][index] -= multiplier * work[column][index]
    result = [0.0] * size
    for row in range(size - 1, -1, -1):
        result[row] = (
            work[row][size]
            - math.fsum(work[row][column] * result[column] for column in range(row + 1, size))
        ) / work[row][row]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("active-set solution became nonfinite")
    return tuple(result)


def _tangent_qp(
    gradient: Sequence[float], fisher: Sequence[Sequence[float]], damping: float
) -> dict[str, object]:
    """Solve the strictly-convex tangent QP by exhaustive active sets."""

    g = _vector(gradient, label="gradient mean")
    f = _matrix3(fisher, label="empirical Fisher")
    damp = _number(damping, label="natural damping", nonnegative=True)
    hessian = tuple(
        tuple(f[row][column] + (damp if row == column else 0.0) for column in range(3))
        for row in range(3)
    )
    numeric_scale = max(
        1.0,
        *(abs(item) for item in g),
        *(abs(item) for row in hessian for item in row),
    )
    tolerance = 16384.0 * sys.float_info.epsilon * numeric_scale
    feasible: list[tuple[float, tuple[int, ...], tuple[float, float, float], tuple[float, ...]]] = []
    for active_count in range(0, 4):
        for active in combinations(range(4), active_count):
            size = 3 + active_count
            kkt = [[0.0] * size for _ in range(size)]
            rhs = [-g[index] for index in range(3)] + [0.0] * active_count
            for row in range(3):
                for column in range(3):
                    kkt[row][column] = hessian[row][column]
            for offset, constraint_index in enumerate(active):
                normal = _TANGENT_NORMALS[constraint_index]
                for feature in range(3):
                    kkt[feature][3 + offset] = normal[feature]
                    kkt[3 + offset][feature] = normal[feature]
            try:
                solution = _solve_linear(kkt, rhs)
            except ValueError:
                continue
            direction = _vector(solution[:3], label="active-set direction")
            multipliers = tuple(solution[3:])
            inequalities = tuple(_dot(normal, direction) for normal in _TANGENT_NORMALS)
            if max(inequalities) > tolerance or any(value < -tolerance for value in multipliers):
                continue
            h_direction = _matvec(hessian, direction)
            objective = _dot(g, direction) + 0.5 * _dot(direction, h_direction)
            feasible.append((objective, active, direction, multipliers))
    if not feasible:
        raise RuntimeError("no feasible tangent-QP active set was found")
    objective, active, raw_direction, multipliers = min(
        feasible,
        key=lambda item: (
            item[0],
            len(item[1]),
            sum(1 << index for index in item[1]),
            _canonical(item[2]),
        ),
    )
    raw_inequalities = tuple(_dot(normal, raw_direction) for normal in _TANGENT_NORMALS)
    raw_stationarity = tuple(
        _matvec(hessian, raw_direction)[index]
        + g[index]
        + math.fsum(
            multipliers[offset] * _TANGENT_NORMALS[constraint][index]
            for offset, constraint in enumerate(active)
        )
        for index in range(3)
    )
    active_equalities = tuple(raw_inequalities[index] for index in active)
    stationarity_max_abs = max(abs(value) for value in raw_stationarity)
    active_equality_max_abs = max((abs(value) for value in active_equalities), default=0.0)
    dual_minimum = min(multipliers, default=0.0)
    primal_violation_max = max(0.0, max(raw_inequalities))
    dual_violation_max = max(0.0, -dual_minimum)
    raw_h_direction = _matvec(hessian, raw_direction)
    gradient_curvature_identity_residual = _dot(g, raw_direction) + _dot(raw_direction, raw_h_direction)
    if stationarity_max_abs > tolerance or active_equality_max_abs > tolerance or dual_violation_max > tolerance or primal_violation_max > tolerance or abs(gradient_curvature_identity_residual) > tolerance:
        raise RuntimeError("selected tangent-QP active set lacks a valid KKT certificate")
    correction = 0.0
    corrected = raw_direction
    for _attempt in range(64):
        inequalities = tuple(_dot(normal, corrected) for normal in _TANGENT_NORMALS)
        violation = max(inequalities)
        if violation <= 0.0:
            break
        # A sub-ULP violation may be too small for ``d2 - violation`` to
        # change its representation.  Step the resulting coordinate one ULP
        # inward and account for the *actual* representable correction.
        new_d2 = math.nextafter(corrected[1] - violation, -math.inf)
        increment = corrected[1] - new_d2
        if increment <= 0.0 or not math.isfinite(increment):
            raise RuntimeError("tangent roundoff correction was not representable")
        correction += increment
        corrected = (corrected[0], new_d2, corrected[2])
    else:
        raise RuntimeError("inward tangent roundoff correction did not converge")
    if correction > tolerance:
        raise RuntimeError("tangent roundoff correction exceeds the solver tolerance")
    inequalities = tuple(_dot(normal, corrected) for normal in _TANGENT_NORMALS)
    if max(inequalities) > 0.0:
        raise RuntimeError("corrected tangent direction violates an exact inequality")
    h_corrected = _matvec(hessian, corrected)
    corrected_objective = _dot(g, corrected) + 0.5 * _dot(corrected, h_corrected)
    return {
        "damped_hessian": hessian,
        "active_constraint_indices": active,
        "active_constraint_bitmask": sum(1 << index for index in active),
        "active_corner_labels": tuple(_CORNER_LABELS[index] for index in active),
        "active_multipliers": multipliers,
        "kkt_stationarity_residual": raw_stationarity,
        "kkt_stationarity_max_abs": stationarity_max_abs,
        "kkt_active_equality_residual": active_equalities,
        "kkt_active_equality_max_abs": active_equality_max_abs,
        "kkt_dual_minimum": dual_minimum,
        "kkt_primal_violation_max": primal_violation_max,
        "kkt_dual_violation_max": dual_violation_max,
        "gradient_curvature_identity_residual": gradient_curvature_identity_residual,
        "kkt_certificate_passed": True,
        "solver_feasibility_tolerance": tolerance,
        "raw_tangent_direction": raw_direction,
        "raw_tangent_inequality_values": raw_inequalities,
        "raw_qp_objective": objective,
        "inward_roundoff_correction": correction,
        "inward_roundoff_direction": _INWARD_DIRECTION,
        "tangent_direction": corrected,
        "tangent_inequality_values": inequalities,
        "tangent_constraints_exactly_feasible": max(inequalities) <= 0.0,
        "qp_objective": corrected_objective,
    }


_DIRECTION_KEYS = {
    "schema", "protocol_sha256", "scientific_status",
    "fresh_family_disjoint_claim_authorized", "serving_claim_authorized",
    "compression_claim_authorized", "source_artifact_sha256s", "family_ids",
    "excluded_family_ids", "fit_family_ids", "validation_family_id",
    "direction_family_ids", "base_provider_artifact_sha256",
    "proposal_provider_artifact_sha256", "gradient_evidence_sha256", "response_law",
    "gradient_bank_artifact_sha256", "family_gradient_summaries_by_family",
    "family_gradient_summary_artifact_sha256s_by_family",
    "fit_example_ids_by_family", "fit_example_counts_by_family",
    "example_gradient_sha256s_by_family", "fit_gradient_rows_sha256", "gradient_mean",
    "empirical_fisher", "fisher_semantics", "empirical_fisher_trace", "damping",
    "damped_hessian", "tangent_normals", "tangent_constraint_semantics",
    "active_constraint_indices", "active_constraint_bitmask", "active_corner_labels", "active_multipliers",
    "kkt_stationarity_residual", "kkt_stationarity_max_abs",
    "kkt_active_equality_residual", "kkt_active_equality_max_abs", "kkt_dual_minimum",
    "kkt_primal_violation_max", "kkt_dual_violation_max",
    "gradient_curvature_identity_residual", "kkt_certificate_passed",
    "solver_feasibility_tolerance", "raw_tangent_direction",
    "raw_tangent_inequality_values", "raw_qp_objective", "inward_roundoff_correction",
    "inward_roundoff_direction", "tangent_direction", "tangent_direction_norm",
    "tangent_inequality_values", "tangent_constraints_exactly_feasible", "qp_objective",
    "gradient_dot_tangent_direction", "gradient_descent_numerical_floor",
    "strict_descent_direction", "direction_l1_norm", "initial_weights",
    "feature_names", "family_equal_per_example_aggregation",
    "held_objectives_or_gradients_used", "raw_gradients_or_tensors_serialized",
    "artifact_sha256",
}


def _normalize_panel(
    family_ids: Sequence[str], excluded_family_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    families = _families(family_ids, label="A16 family IDs", count=_FAMILY_COUNT)
    excluded = _families(excluded_family_ids, label="excluded families", count=_EXCLUDED_COUNT)
    if not set(excluded) < set(families):
        raise ValueError("excluded families differ from the A16 panel")
    fit_families = tuple(family for family in families if family not in excluded)
    if len(fit_families) != _FIT_FAMILY_COUNT:
        raise ValueError("tangent-response fitting requires exactly six fit families")
    return families, excluded, fit_families


_FAMILY_GRADIENT_SUMMARY_KEYS = {
    "schema", "protocol_sha256", "response_law", "family_id",
    "example_ids", "example_count", "example_gradient_sha256s",
    "gradient_mean", "empirical_fisher", "fisher_semantics",
    "raw_gradients_or_tensors_serialized", "artifact_sha256",
}


def _build_family_gradient_summary_from_statistics(
    *, response_law: str, family_id: str, example_ids: Sequence[str],
    example_gradient_sha256s: Mapping[str, str], gradient_mean: Sequence[float],
    empirical_fisher: Sequence[Sequence[float]],
) -> dict[str, object]:
    law = _law(response_law)
    family = _identifier(family_id, label="gradient family")
    ids = tuple(
        sorted(
            _identifier(item, label="gradient-bank example ID")
            for item in _sequence(example_ids, label="gradient-bank example IDs")
        )
    )
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("every V20e gradient-bank family requires exactly two unique examples")
    hashes_source = _mapping(
        example_gradient_sha256s, label="gradient-bank example hashes"
    )
    if set(hashes_source) != set(ids):
        raise ValueError("gradient-bank example hash geometry differs")
    hashes = {
        example: _sha(hashes_source[example], label="gradient-bank example gradient")
        for example in ids
    }
    mean = _vector(gradient_mean, label="family gradient mean")
    fisher = _require_psd3(empirical_fisher, label="family empirical Fisher")
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_gradient_family.v1",
        _GRADIENT_FAMILY_DOMAIN,
        {
            "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
            "response_law": law,
            "family_id": family,
            "example_ids": ids,
            "example_count": 2,
            "example_gradient_sha256s": hashes,
            "gradient_mean": mean,
            "empirical_fisher": fisher,
            "fisher_semantics": "per_example_response_parameter_gradient_outer_product_mean",
            "raw_gradients_or_tensors_serialized": False,
        },
    )


def _validate_family_gradient_summary(
    value: Mapping[str, object], *, expected_response_law: str | None = None,
    expected_family_id: str | None = None,
) -> dict[str, object]:
    selected = _mapping(value, label="gradient-family summary")
    _exact(selected, _FAMILY_GRADIENT_SUMMARY_KEYS, label="gradient-family summary")
    if (
        selected["protocol_sha256"] != TANGENT_RESPONSE_PROTOCOL_SHA256
        or selected["raw_gradients_or_tensors_serialized"] is not False
        or selected["fisher_semantics"]
        != "per_example_response_parameter_gradient_outer_product_mean"
        or selected["example_count"] != 2
    ):
        raise ValueError("gradient-family summary protocol boundary differs")
    rebuilt = _build_family_gradient_summary_from_statistics(
        response_law=selected["response_law"],
        family_id=selected["family_id"],
        example_ids=tuple(selected["example_ids"]),
        example_gradient_sha256s=_mapping(
            selected["example_gradient_sha256s"],
            label="gradient-family example hashes",
        ),
        gradient_mean=tuple(selected["gradient_mean"]),
        empirical_fisher=tuple(selected["empirical_fisher"]),
    )
    _same(selected, rebuilt, label="gradient-family summary")
    if expected_response_law is not None and rebuilt["response_law"] != _law(
        expected_response_law
    ):
        raise ValueError("gradient-family response law differs")
    if expected_family_id is not None and rebuilt["family_id"] != _identifier(
        expected_family_id, label="expected gradient family"
    ):
        raise ValueError("gradient-family identity differs")
    return rebuilt


_GRADIENT_BANK_KEYS = {
    "schema", "protocol_sha256", "scientific_status",
    "fresh_family_disjoint_claim_authorized", "serving_claim_authorized",
    "compression_claim_authorized", "source_artifact_sha256s", "family_ids",
    "excluded_family_ids", "fit_family_ids", "base_provider_artifact_sha256",
    "proposal_provider_artifact_sha256", "response_law",
    "fit_example_ids_by_family", "fit_example_counts_by_family",
    "example_gradient_sha256s_by_family", "family_gradient_summaries_by_family",
    "family_gradient_summary_artifact_sha256s_by_family",
    "gradient_example_ids_globally_unique", "unique_gradient_row_count",
    "empirical_fisher_outer_product_evaluation_count",
    "family_equal_per_example_aggregation", "held_objectives_or_gradients_used",
    "raw_gradients_or_tensors_serialized", "artifact_sha256",
}


def _build_gradient_bank_from_summaries(
    *, source_artifact_sha256s: Mapping[str, str], family_ids: Sequence[str],
    excluded_family_ids: Sequence[str], base_provider_artifact_sha256: str,
    proposal_provider_artifact_sha256: str, response_law: str,
    family_gradient_summaries_by_family: Mapping[str, Mapping[str, object]],
    held_objectives_or_gradients_used: bool,
) -> dict[str, object]:
    families, excluded, fit_families = _normalize_panel(
        family_ids, excluded_family_ids
    )
    if _boolean(
        held_objectives_or_gradients_used, label="gradient-bank held evidence marker"
    ):
        raise ValueError("held objectives or gradients may not enter the gradient bank")
    law = _law(response_law)
    summaries_source = _mapping(
        family_gradient_summaries_by_family,
        label="gradient-family summaries",
    )
    if set(summaries_source) != set(fit_families):
        raise ValueError("gradient-bank family summary geometry differs")
    summaries = {
        family: _validate_family_gradient_summary(
            _mapping(summaries_source[family], label="gradient-family summary"),
            expected_response_law=law,
            expected_family_id=family,
        )
        for family in fit_families
    }
    ids = {
        family: tuple(summaries[family]["example_ids"])
        for family in fit_families
    }
    flattened_ids = [example for family in fit_families for example in ids[family]]
    globally_unique = len(flattened_ids) == len(set(flattened_ids))
    if not globally_unique:
        raise ValueError("gradient-bank example IDs must be globally family-disjoint")
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_gradient_bank.v1",
        _GRADIENT_BANK_DOMAIN,
        {
            "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "serving_claim_authorized": False,
            "compression_claim_authorized": False,
            "source_artifact_sha256s": _sources(source_artifact_sha256s),
            "family_ids": families,
            "excluded_family_ids": excluded,
            "fit_family_ids": fit_families,
            "base_provider_artifact_sha256": _sha(
                base_provider_artifact_sha256, label="base provider"
            ),
            "proposal_provider_artifact_sha256": _sha(
                proposal_provider_artifact_sha256, label="proposal provider"
            ),
            "response_law": law,
            "fit_example_ids_by_family": ids,
            "fit_example_counts_by_family": {
                family: 2 for family in fit_families
            },
            "example_gradient_sha256s_by_family": {
                family: summaries[family]["example_gradient_sha256s"]
                for family in fit_families
            },
            "family_gradient_summaries_by_family": summaries,
            "family_gradient_summary_artifact_sha256s_by_family": {
                family: summaries[family]["artifact_sha256"]
                for family in fit_families
            },
            "gradient_example_ids_globally_unique": globally_unique,
            "unique_gradient_row_count": _FIT_FAMILY_COUNT * 2,
            "empirical_fisher_outer_product_evaluation_count": _FIT_FAMILY_COUNT * 2,
            "family_equal_per_example_aggregation": True,
            "held_objectives_or_gradients_used": False,
            "raw_gradients_or_tensors_serialized": False,
        },
    )


def build_tangent_response_gradient_bank_receipt(
    *, source_artifact_sha256s: Mapping[str, str], family_ids: Sequence[str],
    excluded_family_ids: Sequence[str],
    fit_gradients_by_family: Mapping[str, Mapping[str, Sequence[float]]],
    base_provider_artifact_sha256: str, proposal_provider_artifact_sha256: str,
    response_law: str,
    held_objectives_or_gradients_used: bool = False,
) -> dict[str, object]:
    """Commit one law-specific six-family gradient bank without raw rows.

    Each family contributes exactly two response-parameter gradients.  The
    receipt stores row hashes and the sufficient first/second moments needed
    by every LOFO and all-six tangent solve, so those solves never reconstruct
    per-example outer products.
    """

    _families, _excluded, fit_families = _normalize_panel(
        family_ids, excluded_family_ids
    )
    law = _law(response_law)
    source = _mapping(fit_gradients_by_family, label="fit gradients by family")
    if set(source) != set(fit_families):
        raise ValueError("fit gradients differ from the six-family gradient bank")
    summaries: dict[str, dict[str, object]] = {}
    for family in fit_families:
        examples = _mapping(source[family], label="family fit gradients")
        rows = tuple(
            (
                _identifier(example, label="fit example ID"),
                _vector(gradient, label="per-example gradient"),
            )
            for example, gradient in sorted(examples.items())
        )
        if len(rows) != 2 or len({example for example, _gradient in rows}) != 2:
            raise ValueError(
                "every V20e gradient-bank family requires exactly two unique examples"
            )
        gradients = tuple(gradient for _example, gradient in rows)
        example_hashes = {
            example: _hash(
                _GRADIENT_DOMAIN,
                {
                    "response_law": law,
                    "family_id": family,
                    "example_id": example,
                    "gradient": gradient,
                },
            )
            for example, gradient in rows
        }
        mean = tuple(
            math.fsum(row[index] for row in gradients) / 2.0
            for index in range(_FEATURE_COUNT)
        )
        outers = tuple(_outer(row) for row in gradients)
        fisher = tuple(
            tuple(
                math.fsum(item[row][column] for item in outers) / 2.0
                for column in range(_FEATURE_COUNT)
            )
            for row in range(_FEATURE_COUNT)
        )
        summaries[family] = _build_family_gradient_summary_from_statistics(
            response_law=law,
            family_id=family,
            example_ids=tuple(example for example, _gradient in rows),
            example_gradient_sha256s=example_hashes,
            gradient_mean=mean,
            empirical_fisher=fisher,
        )
    return _build_gradient_bank_from_summaries(
        source_artifact_sha256s=source_artifact_sha256s,
        family_ids=family_ids,
        excluded_family_ids=excluded_family_ids,
        base_provider_artifact_sha256=base_provider_artifact_sha256,
        proposal_provider_artifact_sha256=proposal_provider_artifact_sha256,
        response_law=law,
        family_gradient_summaries_by_family=summaries,
        held_objectives_or_gradients_used=held_objectives_or_gradients_used,
    )


def validate_tangent_response_gradient_bank_receipt(
    value: Mapping[str, object], *,
    expected_source_artifact_sha256s: Mapping[str, str] | None = None,
) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response gradient bank")
    _exact(selected, _GRADIENT_BANK_KEYS, label="tangent-response gradient bank")
    if (
        selected["protocol_sha256"] != TANGENT_RESPONSE_PROTOCOL_SHA256
        or selected["scientific_status"] != "development_only_reused_a16"
        or selected["fresh_family_disjoint_claim_authorized"] is not False
        or selected["serving_claim_authorized"] is not False
        or selected["compression_claim_authorized"] is not False
        or selected["gradient_example_ids_globally_unique"] is not True
        or selected["unique_gradient_row_count"] != 12
        or selected["empirical_fisher_outer_product_evaluation_count"] != 12
        or selected["family_equal_per_example_aggregation"] is not True
        or selected["held_objectives_or_gradients_used"] is not False
        or selected["raw_gradients_or_tensors_serialized"] is not False
    ):
        raise ValueError("tangent-response gradient-bank protocol boundary differs")
    rebuilt = _build_gradient_bank_from_summaries(
        source_artifact_sha256s=_mapping(
            selected["source_artifact_sha256s"], label="source hashes"
        ),
        family_ids=tuple(selected["family_ids"]),
        excluded_family_ids=tuple(selected["excluded_family_ids"]),
        base_provider_artifact_sha256=selected["base_provider_artifact_sha256"],
        proposal_provider_artifact_sha256=selected[
            "proposal_provider_artifact_sha256"
        ],
        response_law=selected["response_law"],
        family_gradient_summaries_by_family=_mapping(
            selected["family_gradient_summaries_by_family"],
            label="gradient-family summaries",
        ),
        held_objectives_or_gradients_used=selected[
            "held_objectives_or_gradients_used"
        ],
    )
    _same(selected, rebuilt, label="tangent-response gradient bank")
    if (
        expected_source_artifact_sha256s is not None
        and rebuilt["source_artifact_sha256s"]
        != _sources(expected_source_artifact_sha256s)
    ):
        raise ValueError("tangent-response gradient-bank source lineage differs")
    return rebuilt


def _build_direction_from_statistics(
    *, source_artifact_sha256s: Mapping[str, str], family_ids: Sequence[str],
    excluded_family_ids: Sequence[str], validation_family_id: str | None,
    base_provider_artifact_sha256: str, proposal_provider_artifact_sha256: str,
    gradient_evidence_sha256: str, response_law: str,
    gradient_bank_artifact_sha256: str,
    family_gradient_summaries_by_family: Mapping[str, Mapping[str, object]],
    held_objectives_or_gradients_used: bool,
) -> dict[str, object]:
    families, excluded, fit_families = _normalize_panel(family_ids, excluded_family_ids)
    validation = None if validation_family_id is None else _identifier(validation_family_id, label="validation family")
    if validation is not None and validation not in fit_families:
        raise ValueError("validation family is not one of the six fit families")
    direction_families = tuple(family for family in fit_families if family != validation)
    expected_count = _FIT_FAMILY_COUNT if validation is None else _CV_TRAIN_FAMILY_COUNT
    if len(direction_families) != expected_count:
        raise ValueError("direction family geometry differs")
    if _boolean(held_objectives_or_gradients_used, label="held evidence marker"):
        raise ValueError("held objectives or gradients may not enter tangent-response fitting")
    law = _law(response_law)
    summary_source = _mapping(
        family_gradient_summaries_by_family,
        label="direction gradient-family summaries",
    )
    if set(summary_source) != set(direction_families):
        raise ValueError("direction family summary geometry differs")
    summaries = {
        family: _validate_family_gradient_summary(
            _mapping(summary_source[family], label="direction gradient-family summary"),
            expected_response_law=law,
            expected_family_id=family,
        )
        for family in direction_families
    }
    ids = {
        family: tuple(summaries[family]["example_ids"])
        for family in direction_families
    }
    counts = {family: 2 for family in direction_families}
    hashes = {
        family: summaries[family]["example_gradient_sha256s"]
        for family in direction_families
    }
    summary_hashes = {
        family: summaries[family]["artifact_sha256"]
        for family in direction_families
    }
    mean = tuple(
        math.fsum(
            float(summaries[family]["gradient_mean"][index])
            for family in direction_families
        )
        / len(direction_families)
        for index in range(_FEATURE_COUNT)
    )
    fisher = tuple(
        tuple(
            math.fsum(
                float(summaries[family]["empirical_fisher"][row][column])
                for family in direction_families
            )
            / len(direction_families)
            for column in range(_FEATURE_COUNT)
        )
        for row in range(_FEATURE_COUNT)
    )
    fisher = _require_psd3(fisher, label="empirical Fisher")
    trace = math.fsum(fisher[index][index] for index in range(3))
    if trace < 0.0:
        raise ValueError("empirical Fisher trace must be nonnegative")
    damping = max(1.0e-12, 1.0e-3 * trace / 3.0)
    qp = _tangent_qp(mean, fisher, damping)
    direction = tuple(qp["tangent_direction"])
    direction_dot = _dot(mean, direction)
    direction_l1 = math.fsum(abs(item) for item in direction)
    descent_floor = max(
        _ABSOLUTE_FLOOR,
        _ROUNDOFF_MULTIPLIER
        * sys.float_info.epsilon
        * math.sqrt(_dot(mean, mean))
        * math.sqrt(_dot(direction, direction)),
    )
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_direction.v1",
        _DIRECTION_DOMAIN,
        {
            "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "serving_claim_authorized": False,
            "compression_claim_authorized": False,
            "source_artifact_sha256s": _sources(source_artifact_sha256s),
            "family_ids": families,
            "excluded_family_ids": excluded,
            "fit_family_ids": fit_families,
            "validation_family_id": validation,
            "direction_family_ids": direction_families,
            "base_provider_artifact_sha256": _sha(base_provider_artifact_sha256, label="base provider"),
            "proposal_provider_artifact_sha256": _sha(proposal_provider_artifact_sha256, label="proposal provider"),
            "gradient_evidence_sha256": _sha(gradient_evidence_sha256, label="gradient evidence"),
            "response_law": law,
            "gradient_bank_artifact_sha256": _sha(
                gradient_bank_artifact_sha256, label="gradient bank"
            ),
            "family_gradient_summaries_by_family": summaries,
            "family_gradient_summary_artifact_sha256s_by_family": summary_hashes,
            "fit_example_ids_by_family": ids,
            "fit_example_counts_by_family": counts,
            "example_gradient_sha256s_by_family": hashes,
            "fit_gradient_rows_sha256": _hash(
                _GRADIENT_SUBSET_DOMAIN,
                {
                    "response_law": law,
                    "gradient_bank_artifact_sha256": _sha(
                        gradient_bank_artifact_sha256, label="gradient bank"
                    ),
                    "validation_family_id": validation,
                    "family_gradient_summary_artifact_sha256s_by_family": summary_hashes,
                },
            ),
            "gradient_mean": mean,
            "empirical_fisher": fisher,
            "fisher_semantics": "family_equal_response_parameter_empirical_gradient_Fisher_OPG",
            "empirical_fisher_trace": trace,
            "damping": damping,
            **qp,
            "tangent_normals": _TANGENT_NORMALS,
            "tangent_constraint_semantics": "sign_corner_value_times_corner_feature_dot_direction_at_most_zero",
            "tangent_direction_norm": math.sqrt(_dot(direction, direction)),
            "gradient_dot_tangent_direction": direction_dot,
            "gradient_descent_numerical_floor": descent_floor,
            "strict_descent_direction": direction_dot < -descent_floor,
            "direction_l1_norm": direction_l1,
            "initial_weights": TANGENT_RESPONSE_INITIAL_WEIGHTS,
            "feature_names": ("c1", "c2", "c1_times_c2"),
            "family_equal_per_example_aggregation": True,
            "held_objectives_or_gradients_used": False,
            "raw_gradients_or_tensors_serialized": False,
        },
    )


def build_tangent_response_direction_receipt(
    *, source_artifact_sha256s: Mapping[str, str], family_ids: Sequence[str],
    excluded_family_ids: Sequence[str], validation_family_id: str | None,
    fit_gradients_by_family: Mapping[str, Mapping[str, Sequence[float]]],
    base_provider_artifact_sha256: str, proposal_provider_artifact_sha256: str,
    gradient_evidence_sha256: str, response_law: str,
    held_objectives_or_gradients_used: bool = False,
) -> dict[str, object]:
    """Build a direction through the same committed bank used by live runs."""

    bank = build_tangent_response_gradient_bank_receipt(
        source_artifact_sha256s=source_artifact_sha256s,
        family_ids=family_ids,
        excluded_family_ids=excluded_family_ids,
        fit_gradients_by_family=fit_gradients_by_family,
        base_provider_artifact_sha256=base_provider_artifact_sha256,
        proposal_provider_artifact_sha256=proposal_provider_artifact_sha256,
        response_law=response_law,
        held_objectives_or_gradients_used=held_objectives_or_gradients_used,
    )
    return build_tangent_response_direction_from_gradient_bank_receipt(
        gradient_bank_receipt=bank,
        gradient_evidence_sha256=gradient_evidence_sha256,
        validation_family_id=validation_family_id,
    )


def build_tangent_response_direction_from_gradient_bank_receipt(
    *, gradient_bank_receipt: Mapping[str, object],
    gradient_evidence_sha256: str,
    validation_family_id: str | None,
) -> dict[str, object]:
    """Aggregate one committed bank and bind its outer initial evidence."""

    bank = validate_tangent_response_gradient_bank_receipt(gradient_bank_receipt)
    validation = (
        None
        if validation_family_id is None
        else _identifier(validation_family_id, label="validation family")
    )
    direction_families = tuple(
        family for family in bank["fit_family_ids"] if family != validation
    )
    summaries = _mapping(
        bank["family_gradient_summaries_by_family"],
        label="gradient-bank family summaries",
    )
    return _build_direction_from_statistics(
        source_artifact_sha256s=_mapping(
            bank["source_artifact_sha256s"], label="gradient-bank source hashes"
        ),
        family_ids=tuple(bank["family_ids"]),
        excluded_family_ids=tuple(bank["excluded_family_ids"]),
        validation_family_id=validation,
        base_provider_artifact_sha256=bank["base_provider_artifact_sha256"],
        proposal_provider_artifact_sha256=bank[
            "proposal_provider_artifact_sha256"
        ],
        gradient_evidence_sha256=gradient_evidence_sha256,
        response_law=bank["response_law"],
        gradient_bank_artifact_sha256=bank["artifact_sha256"],
        family_gradient_summaries_by_family={
            family: _mapping(
                summaries[family], label="gradient-bank family summary"
            )
            for family in direction_families
        },
        held_objectives_or_gradients_used=bank[
            "held_objectives_or_gradients_used"
        ],
    )


def validate_tangent_response_direction_receipt(
    value: Mapping[str, object], *, expected_source_artifact_sha256s: Mapping[str, str] | None = None
) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response direction")
    _exact(selected, _DIRECTION_KEYS, label="tangent-response direction")
    if selected["protocol_sha256"] != TANGENT_RESPONSE_PROTOCOL_SHA256:
        raise ValueError("tangent-response direction protocol differs")
    if selected["scientific_status"] != "development_only_reused_a16" or selected["fresh_family_disjoint_claim_authorized"] is not False or selected["serving_claim_authorized"] is not False or selected["compression_claim_authorized"] is not False or selected["family_equal_per_example_aggregation"] is not True or selected["held_objectives_or_gradients_used"] is not False or selected["raw_gradients_or_tensors_serialized"] is not False:
        raise ValueError("tangent-response direction scientific boundary differs")
    rebuilt = _build_direction_from_statistics(
        source_artifact_sha256s=_mapping(selected["source_artifact_sha256s"], label="source hashes"),
        family_ids=tuple(selected["family_ids"]), excluded_family_ids=tuple(selected["excluded_family_ids"]),
        validation_family_id=selected["validation_family_id"],
        base_provider_artifact_sha256=selected["base_provider_artifact_sha256"],
        proposal_provider_artifact_sha256=selected["proposal_provider_artifact_sha256"],
        gradient_evidence_sha256=selected["gradient_evidence_sha256"], response_law=selected["response_law"],
        gradient_bank_artifact_sha256=selected["gradient_bank_artifact_sha256"],
        family_gradient_summaries_by_family=_mapping(
            selected["family_gradient_summaries_by_family"],
            label="direction gradient-family summaries",
        ),
        held_objectives_or_gradients_used=selected["held_objectives_or_gradients_used"],
    )
    _same(selected, rebuilt, label="tangent-response direction")
    if expected_source_artifact_sha256s is not None and rebuilt["source_artifact_sha256s"] != _sources(expected_source_artifact_sha256s):
        raise ValueError("tangent-response source lineage differs")
    return rebuilt


_RAY_KEYS = {
    "schema", "protocol_sha256", "direction_artifact_sha256", "response_law",
    "validation_family_id", "initial_weights", "tangent_direction",
    "tangent_inequality_values", "ray_denominator", "analytical_ray_max_step",
    "feasible_ray_max_step", "ray_step_roundoff_contractions", "endpoint_weights",
    "endpoint_corner_values", "endpoint_box_certificate", "endpoint_exactly_feasible",
    "direction_l1_norm", "endpoint_displacement_l1", "endpoint_displacement_l1_error_from_two",
    "ray_comparability_tolerance", "ray_comparability_invariant_applicable",
    "ray_comparability_invariant_passed",
    "radial_projection_used", "direction_degenerate", "ray_sha256", "artifact_sha256",
}


def build_tangent_response_ray_receipt(*, direction_receipt: Mapping[str, object]) -> dict[str, object]:
    direction = validate_tangent_response_direction_receipt(direction_receipt)
    vector = _vector(direction["tangent_direction"], label="tangent direction")
    direction_l1 = math.fsum(abs(item) for item in vector)
    denominator = direction_l1
    degenerate = denominator <= 0.0 or not bool(direction["strict_descent_direction"])
    analytical = 0.0 if degenerate else 2.0 / denominator
    safe_step = analytical
    contractions = 0
    if not degenerate:
        for _attempt in range(128):
            endpoint = tuple(initial + safe_step * delta for initial, delta in zip(TANGENT_RESPONSE_INITIAL_WEIGHTS, vector))
            if bilinear_box_certificate(endpoint) <= 1.0:
                break
            safe_step = math.nextafter(safe_step, 0.0)
            contractions += 1
        else:
            raise RuntimeError("feasible tangent ray endpoint did not converge")
    endpoint = tuple(initial + safe_step * delta for initial, delta in zip(TANGENT_RESPONSE_INITIAL_WEIGHTS, vector))
    corners = bilinear_corner_values(endpoint)
    certificate = bilinear_box_certificate(endpoint)
    if certificate > 1.0:
        raise RuntimeError("tangent ray lacks an exact endpoint certificate")
    endpoint_displacement_l1 = math.fsum(
        abs(value - initial) for value, initial in zip(endpoint, TANGENT_RESPONSE_INITIAL_WEIGHTS)
    )
    comparability_tolerance = max(
        _ABSOLUTE_FLOOR,
        1024.0 * sys.float_info.epsilon * max(1.0, endpoint_displacement_l1),
    )
    comparability_applicable = not degenerate
    comparability_passed = comparability_applicable and abs(endpoint_displacement_l1 - 2.0) <= comparability_tolerance
    if comparability_applicable and not comparability_passed:
        raise RuntimeError("tangent ray endpoint lacks the frozen L1 comparability invariant")
    ray_payload = {
        "direction_artifact_sha256": direction["artifact_sha256"],
        "response_law": direction["response_law"],
        "validation_family_id": direction["validation_family_id"],
        "tangent_direction": vector,
        "analytical_ray_max_step": analytical,
        "feasible_ray_max_step": safe_step,
        "endpoint_weights": endpoint,
    }
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_ray.v1", _RAY_DOMAIN,
        {
            "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
            "direction_artifact_sha256": direction["artifact_sha256"],
            "response_law": direction["response_law"],
            "validation_family_id": direction["validation_family_id"],
            "initial_weights": TANGENT_RESPONSE_INITIAL_WEIGHTS,
            "tangent_direction": vector,
            "tangent_inequality_values": direction["tangent_inequality_values"],
            "ray_denominator": denominator,
            "analytical_ray_max_step": analytical,
            "feasible_ray_max_step": safe_step,
            "ray_step_roundoff_contractions": contractions,
            "endpoint_weights": endpoint,
            "endpoint_corner_values": corners,
            "endpoint_box_certificate": certificate,
            "endpoint_exactly_feasible": certificate <= 1.0,
            "direction_l1_norm": direction_l1,
            "endpoint_displacement_l1": endpoint_displacement_l1,
            "endpoint_displacement_l1_error_from_two": endpoint_displacement_l1 - 2.0,
            "ray_comparability_tolerance": comparability_tolerance,
            "ray_comparability_invariant_applicable": comparability_applicable,
            "ray_comparability_invariant_passed": comparability_passed,
            "radial_projection_used": False,
            "direction_degenerate": degenerate,
            "ray_sha256": _hash(_RAY_DOMAIN, ray_payload),
        },
    )


def validate_tangent_response_ray_receipt(
    value: Mapping[str, object], *, direction_receipt: Mapping[str, object]
) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response ray")
    _exact(selected, _RAY_KEYS, label="tangent-response ray")
    rebuilt = build_tangent_response_ray_receipt(direction_receipt=direction_receipt)
    _same(selected, rebuilt, label="tangent-response ray")
    return rebuilt


def _fraction_weights(ray: Mapping[str, object], fraction: float) -> tuple[tuple[float, float, float], float, int]:
    selected = _number(fraction, label="convex fraction", nonnegative=True)
    if selected not in TANGENT_RESPONSE_FRACTIONS:
        raise ValueError("convex fraction is outside the frozen ladder")
    endpoint = _vector(ray["endpoint_weights"], label="ray endpoint")
    effective = selected
    contractions = 0
    for _attempt in range(128):
        weights = tuple((1.0 - effective) * initial + effective * final for initial, final in zip(TANGENT_RESPONSE_INITIAL_WEIGHTS, endpoint))
        if bilinear_box_certificate(weights) <= 1.0:
            return weights, effective, contractions  # type: ignore[return-value]
        effective = math.nextafter(effective, 0.0)
        contractions += 1
    raise RuntimeError("convex fraction roundoff correction did not converge")


def tangent_response_fraction_proposal(
    *, direction_receipt: Mapping[str, object], ray_receipt: Mapping[str, object], fraction: float
) -> dict[str, object]:
    """Return provider-instantiation scalars before any objective is observed."""

    direction = validate_tangent_response_direction_receipt(direction_receipt)
    ray = validate_tangent_response_ray_receipt(ray_receipt, direction_receipt=direction)
    selected = _number(fraction, label="convex fraction", nonnegative=True)
    if ray["direction_degenerate"] is True and selected > 0.0:
        raise ValueError("degenerate tangent ray cannot instantiate a positive fraction")
    weights, effective, contractions = _fraction_weights(ray, selected)
    corners = bilinear_corner_values(weights)
    certificate = bilinear_box_certificate(weights)
    displacement_l1 = math.fsum(
        abs(value - initial) for value, initial in zip(weights, TANGENT_RESPONSE_INITIAL_WEIGHTS)
    )
    expected_l1 = 2.0 * effective
    tolerance = max(_ABSOLUTE_FLOOR, 1024.0 * sys.float_info.epsilon * max(1.0, expected_l1))
    if abs(displacement_l1 - expected_l1) > tolerance:
        raise RuntimeError("convex fraction lacks the frozen L1 comparability invariant")
    return {
        "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
        "direction_artifact_sha256": direction["artifact_sha256"],
        "ray_artifact_sha256": ray["artifact_sha256"],
        "ray_sha256": ray["ray_sha256"],
        "response_law": direction["response_law"],
        "validation_family_id": direction["validation_family_id"],
        "fraction": selected,
        "effective_fraction": effective,
        "fraction_roundoff_contractions": contractions,
        "weights": weights,
        "corner_values": corners,
        "box_certificate": certificate,
        "exactly_feasible": certificate <= 1.0,
        "displacement_l1": displacement_l1,
        "expected_displacement_l1": expected_l1,
        "displacement_l1_error": displacement_l1 - expected_l1,
        "comparability_tolerance": tolerance,
        "comparability_invariant_passed": True,
        "radial_projection_used": False,
        "weight_sha256": _hash(
            _WEIGHT_DOMAIN,
            {
                "law": direction["response_law"],
                "validation_family_id": direction["validation_family_id"],
                "fraction": selected,
                "weights": weights,
            },
        ),
    }


def _normalize_example_objectives(
    *, expected_ids: Sequence[str], objectives: Mapping[str, float], executions: Mapping[str, str],
    objective_label: str,
) -> tuple[dict[str, float], dict[str, str], float]:
    ids = tuple(expected_ids)
    objective_source = _mapping(objectives, label=objective_label)
    execution_source = _mapping(executions, label=f"{objective_label} executions")
    if set(objective_source) != set(ids) or set(execution_source) != set(ids):
        raise ValueError(f"{objective_label} example geometry differs")
    normalized_objectives = {example: _number(objective_source[example], label=objective_label, nonnegative=True) for example in ids}
    normalized_executions = {example: _sha(execution_source[example], label=f"{objective_label} execution") for example in ids}
    return normalized_objectives, normalized_executions, math.fsum(normalized_objectives.values()) / len(ids)


_CV_CANDIDATE_KEYS = {
    "schema", "protocol_sha256", "direction_artifact_sha256", "ray_artifact_sha256",
    "ray_sha256", "response_law", "validation_family_id", "fraction",
    "effective_fraction", "fraction_roundoff_contractions", "initial_weights",
    "endpoint_weights", "weights", "corner_values", "box_certificate", "exactly_feasible",
    "weight_sha256", "displacement", "displacement_l1", "expected_displacement_l1",
    "displacement_l1_error", "comparability_tolerance", "comparability_invariant_passed",
    "radial_projection_used", "displacement_dot_gradient",
    "provider_artifact_sha256", "validation_example_ids", "validation_objectives_by_example",
    "validation_execution_receipt_sha256s_by_example", "validation_family_objective",
    "execution_evidence_sha256", "finite", "pointwise_trust_passed", "rank_is_16",
    "execution_exact", "execution_changed_from_baseline", "candidate_health_passed",
    "objective_source", "held_objectives_used", "raw_tensors_logits_or_targets_serialized",
    "artifact_sha256",
}


def build_tangent_response_cv_candidate(
    *, direction_receipt: Mapping[str, object], ray_receipt: Mapping[str, object], fraction: float,
    provider_artifact_sha256: str, validation_example_ids: Sequence[str],
    validation_objectives_by_example: Mapping[str, float],
    validation_execution_receipt_sha256s_by_example: Mapping[str, str],
    execution_evidence_sha256: str, finite: bool, pointwise_trust_passed: bool,
    rank_is_16: bool, execution_exact: bool, execution_changed_from_baseline: bool,
    objective_source: str = "exact_finite_leave_one_family_out_execution",
    held_objectives_used: bool = False,
) -> dict[str, object]:
    direction = validate_tangent_response_direction_receipt(direction_receipt)
    ray = validate_tangent_response_ray_receipt(ray_receipt, direction_receipt=direction)
    validation = direction["validation_family_id"]
    if validation is None:
        raise ValueError("CV candidate requires a leave-one-family-out direction")
    if objective_source != "exact_finite_leave_one_family_out_execution":
        raise ValueError("CV candidate objective source differs")
    if _boolean(held_objectives_used, label="held-objective marker"):
        raise ValueError("outer held objectives may not enter cross-validation")
    ids = tuple(sorted(_identifier(item, label="validation example ID") for item in _sequence(validation_example_ids, label="validation example IDs")))
    if len(ids) != 2 or len(set(ids)) != len(ids):
        raise ValueError("validation fold requires exactly two unique example IDs")
    objectives, executions, family_objective = _normalize_example_objectives(
        expected_ids=ids, objectives=validation_objectives_by_example,
        executions=validation_execution_receipt_sha256s_by_example,
        objective_label="validation objective",
    )
    selected_fraction = _number(fraction, label="convex fraction", nonnegative=True)
    proposal = tangent_response_fraction_proposal(
        direction_receipt=direction, ray_receipt=ray, fraction=selected_fraction
    )
    weights = tuple(proposal["weights"])
    effective = float(proposal["effective_fraction"])
    contractions = int(proposal["fraction_roundoff_contractions"])
    displacement = tuple(weight - initial for weight, initial in zip(weights, TANGENT_RESPONSE_INITIAL_WEIGHTS))
    corners = bilinear_corner_values(weights)
    certificate = bilinear_box_certificate(weights)
    selected_finite = _boolean(finite, label="CV execution finite")
    selected_trust = _boolean(pointwise_trust_passed, label="CV pointwise trust")
    selected_rank = _boolean(rank_is_16, label="CV rank")
    selected_exact = _boolean(execution_exact, label="CV exact execution")
    selected_changed = _boolean(execution_changed_from_baseline, label="CV execution change")
    if selected_fraction == 0.0 and selected_changed:
        raise ValueError("zero-fraction execution cannot differ from its baseline")
    health = selected_finite and selected_trust and selected_rank and selected_exact
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_cv_candidate.v1",
        _CV_CANDIDATE_DOMAIN,
        {
            "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
            "direction_artifact_sha256": direction["artifact_sha256"],
            "ray_artifact_sha256": ray["artifact_sha256"],
            "ray_sha256": ray["ray_sha256"],
            "response_law": direction["response_law"],
            "validation_family_id": validation,
            "fraction": selected_fraction,
            "effective_fraction": effective,
            "fraction_roundoff_contractions": contractions,
            "initial_weights": TANGENT_RESPONSE_INITIAL_WEIGHTS,
            "endpoint_weights": ray["endpoint_weights"],
            "weights": weights,
            "corner_values": corners,
            "box_certificate": certificate,
            "exactly_feasible": certificate <= 1.0,
            "weight_sha256": proposal["weight_sha256"],
            "displacement": displacement,
            "displacement_l1": proposal["displacement_l1"],
            "expected_displacement_l1": proposal["expected_displacement_l1"],
            "displacement_l1_error": proposal["displacement_l1_error"],
            "comparability_tolerance": proposal["comparability_tolerance"],
            "comparability_invariant_passed": proposal["comparability_invariant_passed"],
            "radial_projection_used": False,
            "displacement_dot_gradient": _dot(displacement, direction["gradient_mean"]),
            "provider_artifact_sha256": _sha(provider_artifact_sha256, label="CV provider"),
            "validation_example_ids": ids,
            "validation_objectives_by_example": objectives,
            "validation_execution_receipt_sha256s_by_example": executions,
            "validation_family_objective": family_objective,
            "execution_evidence_sha256": _sha(execution_evidence_sha256, label="CV execution evidence"),
            "finite": selected_finite,
            "pointwise_trust_passed": selected_trust,
            "rank_is_16": selected_rank,
            "execution_exact": selected_exact,
            "execution_changed_from_baseline": selected_changed,
            "candidate_health_passed": health,
            "objective_source": objective_source,
            "held_objectives_used": False,
            "raw_tensors_logits_or_targets_serialized": False,
        },
    )


def validate_tangent_response_cv_candidate(
    value: Mapping[str, object], *, direction_receipt: Mapping[str, object], ray_receipt: Mapping[str, object]
) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response CV candidate")
    _exact(selected, _CV_CANDIDATE_KEYS, label="tangent-response CV candidate")
    if selected["protocol_sha256"] != TANGENT_RESPONSE_PROTOCOL_SHA256 or selected["held_objectives_used"] is not False or selected["raw_tensors_logits_or_targets_serialized"] is not False:
        raise ValueError("CV candidate scalar/scientific boundary differs")
    rebuilt = build_tangent_response_cv_candidate(
        direction_receipt=direction_receipt, ray_receipt=ray_receipt, fraction=selected["fraction"],
        provider_artifact_sha256=selected["provider_artifact_sha256"],
        validation_example_ids=tuple(selected["validation_example_ids"]),
        validation_objectives_by_example=_mapping(selected["validation_objectives_by_example"], label="validation objectives"),
        validation_execution_receipt_sha256s_by_example=_mapping(selected["validation_execution_receipt_sha256s_by_example"], label="validation executions"),
        execution_evidence_sha256=selected["execution_evidence_sha256"],
        finite=selected["finite"], pointwise_trust_passed=selected["pointwise_trust_passed"],
        rank_is_16=selected["rank_is_16"], execution_exact=selected["execution_exact"],
        execution_changed_from_baseline=selected["execution_changed_from_baseline"],
        objective_source=selected["objective_source"], held_objectives_used=selected["held_objectives_used"],
    )
    _same(selected, rebuilt, label="tangent-response CV candidate")
    return rebuilt


_CV_KEYS = {
    "schema", "protocol_sha256", "scientific_status",
    "fresh_family_disjoint_claim_authorized", "serving_claim_authorized",
    "compression_claim_authorized", "response_law", "source_artifact_sha256s",
    "base_provider_artifact_sha256", "proposal_provider_artifact_sha256",
    "gradient_evidence_sha256", "gradient_bank_artifact_sha256",
    "family_ids", "excluded_family_ids", "fit_family_ids", "fold_order",
    "shared_gradient_example_ids_by_family", "shared_gradient_sha256s_by_family",
    "shared_gradient_family_summaries_by_family",
    "shared_gradient_family_summary_artifact_sha256s_by_family",
    "shared_gradient_example_ids_globally_unique",
    "direction_receipts_by_validation_family", "direction_artifact_sha256s_by_validation_family",
    "ray_receipts_by_validation_family", "ray_artifact_sha256s_by_validation_family",
    "candidate_receipts_by_validation_family", "candidate_artifact_sha256s_by_validation_family",
    "fraction_ladder", "validation_family_objectives_by_fraction",
    "macro_objectives_by_fraction", "fold_improved_by_fraction",
    "fold_improvement_counts_by_fraction", "baseline_macro_objective",
    "objective_numerical_improvement_floor", "eligible_positive_fractions",
    "selected_fraction", "selected_macro_objective", "selected_fold_improvement_count",
    "selected_candidate_artifact_sha256s_by_validation_family",
    "selection_tie_order", "required_improved_fold_count", "macro_improved",
    "at_least_four_of_six_folds_improved", "cv_selection_authorized",
    "rollback_to_zero_fraction", "held_score_authorized", "all_candidates_healthy",
    "selected_candidates_healthy", "positive_provider_hashes_unique_across_fold_fraction",
    "zero_fraction_provider_reused_across_folds", "execution_hashes_unique_across_fold_fraction_example",
    "outer_held_objectives_used", "raw_tensors_logits_targets_or_gradients_serialized",
    "artifact_sha256",
}


def build_tangent_response_cv_receipt(
    *, direction_receipts: Sequence[Mapping[str, object]],
    ray_receipts: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select one shared convex fraction from six out-of-fold objectives."""

    directions_raw = tuple(_mapping(item, label="CV direction") for item in _sequence(direction_receipts, label="CV directions"))
    rays_raw = tuple(_mapping(item, label="CV ray") for item in _sequence(ray_receipts, label="CV rays"))
    if len(directions_raw) != _FIT_FAMILY_COUNT or len(rays_raw) != _FIT_FAMILY_COUNT:
        raise ValueError("CV requires six leave-one-family-out directions and rays")
    directions_values = tuple(validate_tangent_response_direction_receipt(item) for item in directions_raw)
    directions = {str(item["validation_family_id"]): item for item in directions_values}
    if None in (item["validation_family_id"] for item in directions_values) or len(directions) != _FIT_FAMILY_COUNT:
        raise ValueError("CV directions require six unique validation families")
    first = directions_values[0]
    fold_order = tuple(first["fit_family_ids"])
    if set(directions) != set(fold_order):
        raise ValueError("CV validation folds differ from the six fit families")
    shared_fields = (
        "protocol_sha256", "source_artifact_sha256s", "family_ids", "excluded_family_ids",
        "fit_family_ids", "base_provider_artifact_sha256", "proposal_provider_artifact_sha256",
        "response_law", "gradient_evidence_sha256", "gradient_bank_artifact_sha256",
    )
    for direction in directions_values[1:]:
        for field in shared_fields:
            if _canonical(direction[field]) != _canonical(first[field]):
                raise ValueError(f"CV direction {field} lineage differs")
    for validation, direction in directions.items():
        expected_train = tuple(family for family in fold_order if family != validation)
        if tuple(direction["direction_family_ids"]) != expected_train:
            raise ValueError("CV direction does not leave out exactly its validation family")
    shared_gradient_ids: dict[str, tuple[str, ...]] = {}
    shared_gradient_hashes: dict[str, dict[str, str]] = {}
    shared_gradient_summaries: dict[str, dict[str, object]] = {}
    shared_gradient_summary_hashes: dict[str, str] = {}
    for family in fold_order:
        appearances = tuple(
            direction
            for validation, direction in directions.items()
            if validation != family
        )
        if len(appearances) != _CV_TRAIN_FAMILY_COUNT:
            raise ValueError("shared gradient bank fold geometry differs")
        first_ids = tuple(_mapping(appearances[0]["fit_example_ids_by_family"], label="fold gradient IDs")[family])
        first_hashes = dict(_mapping(_mapping(appearances[0]["example_gradient_sha256s_by_family"], label="fold gradient hashes")[family], label="family gradient hashes"))
        first_summary = _mapping(
            _mapping(
                appearances[0]["family_gradient_summaries_by_family"],
                label="fold gradient-family summaries",
            )[family],
            label="fold gradient-family summary",
        )
        first_summary_hash = _mapping(
            appearances[0]["family_gradient_summary_artifact_sha256s_by_family"],
            label="fold gradient-family summary hashes",
        )[family]
        for appearance in appearances[1:]:
            appearance_ids = tuple(_mapping(appearance["fit_example_ids_by_family"], label="fold gradient IDs")[family])
            appearance_hashes = dict(_mapping(_mapping(appearance["example_gradient_sha256s_by_family"], label="fold gradient hashes")[family], label="family gradient hashes"))
            appearance_summary = _mapping(
                _mapping(
                    appearance["family_gradient_summaries_by_family"],
                    label="fold gradient-family summaries",
                )[family],
                label="fold gradient-family summary",
            )
            appearance_summary_hash = _mapping(
                appearance["family_gradient_summary_artifact_sha256s_by_family"],
                label="fold gradient-family summary hashes",
            )[family]
            if (
                _canonical(appearance_ids) != _canonical(first_ids)
                or _canonical(appearance_hashes) != _canonical(first_hashes)
                or _canonical(appearance_summary) != _canonical(first_summary)
                or appearance_summary_hash != first_summary_hash
            ):
                raise ValueError("overlapping CV folds do not derive from one shared gradient bank")
        if len(first_ids) != 2:
            raise ValueError("shared V20e gradient bank requires exactly two examples per fit family")
        shared_gradient_ids[family] = first_ids
        shared_gradient_hashes[family] = {example: _sha(first_hashes[example], label="shared gradient hash") for example in first_ids}
        shared_gradient_summaries[family] = _validate_family_gradient_summary(
            first_summary,
            expected_response_law=first["response_law"],
            expected_family_id=family,
        )
        shared_gradient_summary_hashes[family] = _sha(
            first_summary_hash, label="shared gradient-family summary"
        )
    flattened_gradient_ids = [example for family in fold_order for example in shared_gradient_ids[family]]
    globally_unique_gradient_ids = len(set(flattened_gradient_ids)) == len(flattened_gradient_ids)
    if not globally_unique_gradient_ids:
        raise ValueError("shared gradient-bank example IDs must be globally family-disjoint")
    rebuilt_gradient_bank = _build_gradient_bank_from_summaries(
        source_artifact_sha256s=_mapping(
            first["source_artifact_sha256s"], label="CV source hashes"
        ),
        family_ids=tuple(first["family_ids"]),
        excluded_family_ids=tuple(first["excluded_family_ids"]),
        base_provider_artifact_sha256=first["base_provider_artifact_sha256"],
        proposal_provider_artifact_sha256=first["proposal_provider_artifact_sha256"],
        response_law=first["response_law"],
        family_gradient_summaries_by_family=shared_gradient_summaries,
        held_objectives_or_gradients_used=False,
    )
    if rebuilt_gradient_bank["artifact_sha256"] != first["gradient_bank_artifact_sha256"]:
        raise ValueError("CV directions do not authenticate their shared gradient bank")
    rays_values: dict[str, dict[str, object]] = {}
    ray_by_direction = {str(item["direction_artifact_sha256"]): item for item in rays_raw}
    for validation in fold_order:
        direction = directions[validation]
        raw_ray = ray_by_direction.get(str(direction["artifact_sha256"]))
        if raw_ray is None:
            raise ValueError("CV ray lacks its direction lineage")
        rays_values[validation] = validate_tangent_response_ray_receipt(raw_ray, direction_receipt=direction)
    if len(ray_by_direction) != _FIT_FAMILY_COUNT:
        raise ValueError("CV ray directions are duplicated")
    candidates_raw = tuple(_mapping(item, label="CV candidate") for item in _sequence(candidates, label="CV candidates"))
    expected_candidate_count = _FIT_FAMILY_COUNT * len(TANGENT_RESPONSE_FRACTIONS)
    if len(candidates_raw) != expected_candidate_count:
        raise ValueError("CV requires the complete six-fold fraction grid")
    by_fold_fraction: dict[tuple[str, float], dict[str, object]] = {}
    for raw in candidates_raw:
        validation = _identifier(raw.get("validation_family_id"), label="candidate validation family")
        if validation not in directions:
            raise ValueError("CV candidate validation family differs")
        candidate = validate_tangent_response_cv_candidate(
            raw, direction_receipt=directions[validation], ray_receipt=rays_values[validation]
        )
        if tuple(candidate["validation_example_ids"]) != shared_gradient_ids[validation]:
            raise ValueError("CV candidate validation rows differ from the shared gradient bank")
        key = (validation, float(candidate["fraction"]))
        if key in by_fold_fraction:
            raise ValueError("CV candidate fold/fraction is duplicated")
        by_fold_fraction[key] = candidate
    expected_keys = {(family, fraction) for family in fold_order for fraction in TANGENT_RESPONSE_FRACTIONS}
    if set(by_fold_fraction) != expected_keys:
        raise ValueError("CV candidate grid differs from the frozen ladder")
    positive_provider_hashes = [
        str(item["provider_artifact_sha256"])
        for (family, fraction), item in by_fold_fraction.items()
        if fraction > 0.0
    ]
    positive_provider_unique = len(set(positive_provider_hashes)) == len(positive_provider_hashes)
    if not positive_provider_unique:
        raise ValueError("positive CV providers must bind a unique law/fold/fraction lineage")
    zero_provider_hashes = {
        str(by_fold_fraction[(family, 0.0)]["provider_artifact_sha256"])
        for family in fold_order
    }
    zero_provider_reused = len(zero_provider_hashes) == 1
    if not zero_provider_reused:
        raise ValueError("zero-fraction CV folds must reuse the one actual law-specific initial provider")
    execution_hashes = [
        str(execution)
        for item in by_fold_fraction.values()
        for execution in _mapping(item["validation_execution_receipt_sha256s_by_example"], label="CV executions").values()
    ]
    executions_unique = len(set(execution_hashes)) == len(execution_hashes)
    if not executions_unique:
        raise ValueError("CV executions must bind a unique law/fold/fraction/example lineage")
    family_objectives_by_fraction: dict[str, dict[str, float]] = {}
    macros: dict[str, float] = {}
    fold_improved: dict[str, dict[str, bool]] = {}
    improvement_counts: dict[str, int] = {}
    for fraction in TANGENT_RESPONSE_FRACTIONS:
        fraction_key = repr(fraction)
        family_values = {
            family: float(by_fold_fraction[(family, fraction)]["validation_family_objective"])
            for family in fold_order
        }
        family_objectives_by_fraction[fraction_key] = family_values
        macros[fraction_key] = math.fsum(family_values.values()) / _FIT_FAMILY_COUNT
        improvements = {
            family: (
                float(by_fold_fraction[(family, 0.0)]["validation_family_objective"])
                - family_values[family]
                > _numerical_floor(float(by_fold_fraction[(family, 0.0)]["validation_family_objective"]))
            )
            for family in fold_order
        }
        fold_improved[fraction_key] = improvements
        improvement_counts[fraction_key] = sum(improvements.values())
    baseline = macros[repr(0.0)]
    floor = _numerical_floor(baseline)
    eligible = tuple(
        fraction
        for fraction in TANGENT_RESPONSE_FRACTIONS
        if fraction > 0.0
        and baseline - macros[repr(fraction)] > floor
        and improvement_counts[repr(fraction)] >= 4
        and all(bool(by_fold_fraction[(family, fraction)]["exactly_feasible"]) for family in fold_order)
        and all(bool(by_fold_fraction[(family, fraction)]["candidate_health_passed"]) for family in fold_order)
        and all(bool(by_fold_fraction[(family, fraction)]["execution_changed_from_baseline"]) for family in fold_order)
        and all(bool(directions[family]["strict_descent_direction"]) for family in fold_order)
    )
    authorized = bool(eligible)
    selected_fraction = (
        min(
            eligible,
            key=lambda fraction: (
                macros[repr(fraction)],
                fraction,
                tuple(by_fold_fraction[(family, fraction)]["artifact_sha256"] for family in fold_order),
            ),
        )
        if authorized
        else 0.0
    )
    selected_key = repr(selected_fraction)
    ordered_directions = {family: directions[family] for family in fold_order}
    ordered_rays = {family: rays_values[family] for family in fold_order}
    ordered_candidates = {
        family: tuple(by_fold_fraction[(family, fraction)] for fraction in TANGENT_RESPONSE_FRACTIONS)
        for family in fold_order
    }
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_cv.v1", _CV_DOMAIN,
        {
            "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "serving_claim_authorized": False,
            "compression_claim_authorized": False,
            "response_law": first["response_law"],
            "source_artifact_sha256s": first["source_artifact_sha256s"],
            "base_provider_artifact_sha256": first["base_provider_artifact_sha256"],
            "proposal_provider_artifact_sha256": first["proposal_provider_artifact_sha256"],
            "gradient_evidence_sha256": first["gradient_evidence_sha256"],
            "gradient_bank_artifact_sha256": first["gradient_bank_artifact_sha256"],
            "family_ids": first["family_ids"],
            "excluded_family_ids": first["excluded_family_ids"],
            "fit_family_ids": fold_order,
            "fold_order": fold_order,
            "shared_gradient_example_ids_by_family": shared_gradient_ids,
            "shared_gradient_sha256s_by_family": shared_gradient_hashes,
            "shared_gradient_family_summaries_by_family": shared_gradient_summaries,
            "shared_gradient_family_summary_artifact_sha256s_by_family": shared_gradient_summary_hashes,
            "shared_gradient_example_ids_globally_unique": globally_unique_gradient_ids,
            "direction_receipts_by_validation_family": ordered_directions,
            "direction_artifact_sha256s_by_validation_family": {family: directions[family]["artifact_sha256"] for family in fold_order},
            "ray_receipts_by_validation_family": ordered_rays,
            "ray_artifact_sha256s_by_validation_family": {family: rays_values[family]["artifact_sha256"] for family in fold_order},
            "candidate_receipts_by_validation_family": ordered_candidates,
            "candidate_artifact_sha256s_by_validation_family": {family: tuple(item["artifact_sha256"] for item in ordered_candidates[family]) for family in fold_order},
            "fraction_ladder": TANGENT_RESPONSE_FRACTIONS,
            "validation_family_objectives_by_fraction": family_objectives_by_fraction,
            "macro_objectives_by_fraction": macros,
            "fold_improved_by_fraction": fold_improved,
            "fold_improvement_counts_by_fraction": improvement_counts,
            "baseline_macro_objective": baseline,
            "objective_numerical_improvement_floor": floor,
            "eligible_positive_fractions": eligible,
            "selected_fraction": selected_fraction,
            "selected_macro_objective": macros[selected_key],
            "selected_fold_improvement_count": improvement_counts[selected_key],
            "selected_candidate_artifact_sha256s_by_validation_family": {family: by_fold_fraction[(family, selected_fraction)]["artifact_sha256"] for family in fold_order},
            "selection_tie_order": "macro_objective_then_smaller_fraction_then_ordered_candidate_hashes",
            "required_improved_fold_count": 4,
            "macro_improved": baseline - macros[selected_key] > floor,
            "at_least_four_of_six_folds_improved": improvement_counts[selected_key] >= 4,
            "cv_selection_authorized": authorized,
            "rollback_to_zero_fraction": not authorized,
            "held_score_authorized": False,
            "all_candidates_healthy": all(bool(item["candidate_health_passed"]) for item in by_fold_fraction.values()),
            "selected_candidates_healthy": all(bool(by_fold_fraction[(family, selected_fraction)]["candidate_health_passed"]) for family in fold_order),
            "positive_provider_hashes_unique_across_fold_fraction": positive_provider_unique,
            "zero_fraction_provider_reused_across_folds": zero_provider_reused,
            "execution_hashes_unique_across_fold_fraction_example": executions_unique,
            "outer_held_objectives_used": False,
            "raw_tensors_logits_targets_or_gradients_serialized": False,
        },
    )


def validate_tangent_response_cv_receipt(value: Mapping[str, object]) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response CV receipt")
    _exact(selected, _CV_KEYS, label="tangent-response CV receipt")
    if selected["protocol_sha256"] != TANGENT_RESPONSE_PROTOCOL_SHA256 or selected["held_score_authorized"] is not False or selected["outer_held_objectives_used"] is not False or selected["raw_tensors_logits_targets_or_gradients_serialized"] is not False:
        raise ValueError("CV receipt scientific boundary differs")
    directions = tuple(_mapping(item, label="CV direction") for item in _mapping(selected["direction_receipts_by_validation_family"], label="CV directions").values())
    rays = tuple(_mapping(item, label="CV ray") for item in _mapping(selected["ray_receipts_by_validation_family"], label="CV rays").values())
    candidates = tuple(
        _mapping(item, label="CV candidate")
        for family_candidates in _mapping(selected["candidate_receipts_by_validation_family"], label="CV candidates").values()
        for item in _sequence(family_candidates, label="family CV candidates")
    )
    rebuilt = build_tangent_response_cv_receipt(direction_receipts=directions, ray_receipts=rays, candidates=candidates)
    _same(selected, rebuilt, label="tangent-response CV receipt")
    return rebuilt


_FINAL_CANDIDATE_KEYS = {
    "schema", "protocol_sha256", "cv_artifact_sha256", "direction_artifact_sha256",
    "ray_artifact_sha256", "ray_sha256", "response_law", "selected_cv_fraction",
    "effective_fraction", "fraction_roundoff_contractions", "initial_weights",
    "endpoint_weights", "weights", "corner_values", "box_certificate", "exactly_feasible",
    "weight_sha256", "weight_hash_changed", "displacement", "displacement_dot_gradient",
    "displacement_l1", "expected_displacement_l1", "comparability_tolerance",
    "comparability_invariant_passed", "radial_projection_used",
    "selected_provider_artifact_sha256", "fit_support_family_ids",
    "fit_support_example_ids_by_family", "fit_support_provider_trace_receipt_sha256s_by_family",
    "fit_support_gain_trace_sha256s_by_family", "fit_support_gain_min_by_family",
    "fit_support_gain_max_by_family", "fit_support_gain_distinct_count_by_family",
    "fit_support_global_gain_min", "fit_support_global_gain_max", "fit_support_global_gain_range",
    "fit_support_global_gain_distinct_count", "response_nonconstant_on_fit_support",
    "provider_trace_evidence_sha256", "provider_trace_finite", "pointwise_trust_passed",
    "rank_is_16", "provider_trace_exact", "provider_trace_changed_from_initial",
    "cv_selection_authorized", "strict_descent_direction", "final_candidate_authorized",
    "rollback_to_initial_weights", "final_exact_fit_objectives_used",
    "outer_held_objectives_used", "raw_tensors_logits_targets_or_gradients_serialized",
    "artifact_sha256",
}


def build_tangent_response_final_candidate_receipt(
    *, cv_receipt: Mapping[str, object], direction_receipt: Mapping[str, object],
    ray_receipt: Mapping[str, object], selected_provider_artifact_sha256: str,
    fit_support_example_ids_by_family: Mapping[str, Sequence[str]],
    fit_support_provider_trace_receipt_sha256s_by_family: Mapping[str, Mapping[str, str]],
    fit_support_gain_trace_sha256s_by_family: Mapping[str, str],
    fit_support_gain_min_by_family: Mapping[str, float],
    fit_support_gain_max_by_family: Mapping[str, float],
    fit_support_gain_distinct_count_by_family: Mapping[str, int],
    provider_trace_evidence_sha256: str, provider_trace_finite: bool,
    pointwise_trust_passed: bool, rank_is_16: bool, provider_trace_exact: bool,
    provider_trace_changed_from_initial: bool,
    final_exact_fit_objectives_used: bool = False,
    outer_held_objectives_used: bool = False,
) -> dict[str, object]:
    """Instantiate the OOF-selected fraction on the all-six direction.

    No all-six objective is accepted here: rescoring and reselecting on the
    training panel after OOF selection would weaken the cross-validation
    boundary.  Only structural and nonconstant fit-support evidence is bound.
    """

    cv = validate_tangent_response_cv_receipt(cv_receipt)
    direction = validate_tangent_response_direction_receipt(direction_receipt)
    ray = validate_tangent_response_ray_receipt(ray_receipt, direction_receipt=direction)
    if direction["validation_family_id"] is not None or tuple(direction["direction_family_ids"]) != tuple(direction["fit_family_ids"]):
        raise ValueError("final candidate requires an all-six direction")
    for field in (
        "response_law", "source_artifact_sha256s", "family_ids", "excluded_family_ids",
        "fit_family_ids", "base_provider_artifact_sha256",
        "proposal_provider_artifact_sha256", "gradient_evidence_sha256",
        "gradient_bank_artifact_sha256",
    ):
        if _canonical(direction[field]) != _canonical(cv[field]):
            raise ValueError(f"final candidate {field} lineage differs from CV")
    final_gradient_ids = _mapping(direction["fit_example_ids_by_family"], label="final gradient IDs")
    final_gradient_hashes = _mapping(direction["example_gradient_sha256s_by_family"], label="final gradient hashes")
    cv_gradient_ids = _mapping(cv["shared_gradient_example_ids_by_family"], label="CV gradient IDs")
    cv_gradient_hashes = _mapping(cv["shared_gradient_sha256s_by_family"], label="CV gradient hashes")
    final_gradient_summaries = _mapping(
        direction["family_gradient_summaries_by_family"],
        label="final gradient-family summaries",
    )
    final_gradient_summary_hashes = _mapping(
        direction["family_gradient_summary_artifact_sha256s_by_family"],
        label="final gradient-family summary hashes",
    )
    cv_gradient_summaries = _mapping(
        cv["shared_gradient_family_summaries_by_family"],
        label="CV gradient-family summaries",
    )
    cv_gradient_summary_hashes = _mapping(
        cv["shared_gradient_family_summary_artifact_sha256s_by_family"],
        label="CV gradient-family summary hashes",
    )
    if (
        _canonical(final_gradient_ids) != _canonical(cv_gradient_ids)
        or _canonical(final_gradient_hashes) != _canonical(cv_gradient_hashes)
        or _canonical(final_gradient_summaries) != _canonical(cv_gradient_summaries)
        or _canonical(final_gradient_summary_hashes)
        != _canonical(cv_gradient_summary_hashes)
    ):
        raise ValueError("final all-six direction differs from the canonical CV gradient bank")
    if _boolean(final_exact_fit_objectives_used, label="final fit objective marker"):
        raise ValueError("V20e forbids an all-six exact rescore after CV selection")
    if _boolean(outer_held_objectives_used, label="outer held objective marker"):
        raise ValueError("outer held objectives may not authorize the final candidate")
    fraction = float(cv["selected_fraction"])
    proposal = tangent_response_fraction_proposal(
        direction_receipt=direction, ray_receipt=ray, fraction=fraction
    )
    weights = tuple(proposal["weights"])
    effective = float(proposal["effective_fraction"])
    contractions = int(proposal["fraction_roundoff_contractions"])
    displacement = tuple(weight - initial for weight, initial in zip(weights, TANGENT_RESPONSE_INITIAL_WEIGHTS))
    corners = bilinear_corner_values(weights)
    certificate = bilinear_box_certificate(weights)
    families = tuple(direction["fit_family_ids"])
    ids_source = _mapping(fit_support_example_ids_by_family, label="fit-support example IDs")
    execution_source = _mapping(fit_support_provider_trace_receipt_sha256s_by_family, label="fit-support provider traces")
    trace_source = _mapping(fit_support_gain_trace_sha256s_by_family, label="fit-support gain traces")
    minimum_source = _mapping(fit_support_gain_min_by_family, label="fit-support gain minima")
    maximum_source = _mapping(fit_support_gain_max_by_family, label="fit-support gain maxima")
    count_source = _mapping(fit_support_gain_distinct_count_by_family, label="fit-support gain distinct counts")
    if any(set(source) != set(families) for source in (ids_source, execution_source, trace_source, minimum_source, maximum_source, count_source)):
        raise ValueError("fit-support family geometry differs")
    ids: dict[str, tuple[str, ...]] = {}
    executions: dict[str, dict[str, str]] = {}
    traces: dict[str, str] = {}
    minima: dict[str, float] = {}
    maxima: dict[str, float] = {}
    counts: dict[str, int] = {}
    for family in families:
        family_ids = tuple(sorted(_identifier(item, label="fit-support example ID") for item in _sequence(ids_source[family], label="fit-support example IDs")))
        if family_ids != tuple(final_gradient_ids[family]):
            raise ValueError("fit-support IDs differ from the final all-six gradient bank")
        family_executions = _mapping(execution_source[family], label="fit-support family provider traces")
        if not family_ids or len(set(family_ids)) != len(family_ids) or set(family_executions) != set(family_ids):
            raise ValueError("fit-support example geometry differs")
        minimum = _number(minimum_source[family], label="fit-support gain minimum")
        maximum = _number(maximum_source[family], label="fit-support gain maximum")
        count = _integer(count_source[family], label="fit-support gain distinct count", minimum=1)
        if maximum < minimum or (count == 1 and maximum != minimum) or (count > 1 and maximum <= minimum):
            raise ValueError("fit-support gain summary is inconsistent")
        ids[family] = family_ids
        executions[family] = {example: _sha(family_executions[example], label="fit-support provider trace") for example in family_ids}
        traces[family] = _sha(trace_source[family], label="fit-support gain trace")
        minima[family] = minimum
        maxima[family] = maximum
        counts[family] = count
    all_execution_hashes = [value for family in families for value in executions[family].values()]
    if len(set(all_execution_hashes)) != len(all_execution_hashes):
        raise ValueError("fit-support provider traces must bind unique law/family/example lineage")
    global_min = min(minima.values())
    global_max = max(maxima.values())
    global_count = sum(counts.values())
    nonconstant = global_max > global_min and global_count >= 2
    weight_hash = _hash(_WEIGHT_DOMAIN, {"law": direction["response_law"], "scope": "all_six", "fraction": fraction, "weights": weights})
    initial_hash = _hash(_WEIGHT_DOMAIN, {"law": direction["response_law"], "scope": "all_six", "fraction": 0.0, "weights": TANGENT_RESPONSE_INITIAL_WEIGHTS})
    changed = weight_hash != initial_hash and fraction > 0.0 and weights != TANGENT_RESPONSE_INITIAL_WEIGHTS
    descent = _dot(displacement, direction["gradient_mean"])
    selected_finite = _boolean(provider_trace_finite, label="provider trace finite")
    selected_trust = _boolean(pointwise_trust_passed, label="provider trace trust")
    selected_rank = _boolean(rank_is_16, label="provider trace rank")
    selected_exact = _boolean(provider_trace_exact, label="provider trace exact")
    selected_runtime_changed = _boolean(provider_trace_changed_from_initial, label="provider trace change")
    authorized = bool(
        cv["cv_selection_authorized"] is True
        and fraction > 0.0
        and direction["strict_descent_direction"] is True
        and changed
        and certificate <= 1.0
        and descent < 0.0
        and nonconstant
        and selected_finite
        and selected_trust
        and selected_rank
        and selected_exact
        and selected_runtime_changed
    )
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_final_candidate.v1",
        _FINAL_CANDIDATE_DOMAIN,
        {
            "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
            "cv_artifact_sha256": cv["artifact_sha256"],
            "direction_artifact_sha256": direction["artifact_sha256"],
            "ray_artifact_sha256": ray["artifact_sha256"],
            "ray_sha256": ray["ray_sha256"],
            "response_law": direction["response_law"],
            "selected_cv_fraction": fraction,
            "effective_fraction": effective,
            "fraction_roundoff_contractions": contractions,
            "initial_weights": TANGENT_RESPONSE_INITIAL_WEIGHTS,
            "endpoint_weights": ray["endpoint_weights"],
            "weights": weights,
            "corner_values": corners,
            "box_certificate": certificate,
            "exactly_feasible": certificate <= 1.0,
            "weight_sha256": weight_hash,
            "weight_hash_changed": changed,
            "displacement": displacement,
            "displacement_l1": proposal["displacement_l1"],
            "expected_displacement_l1": proposal["expected_displacement_l1"],
            "comparability_tolerance": proposal["comparability_tolerance"],
            "comparability_invariant_passed": proposal["comparability_invariant_passed"],
            "radial_projection_used": False,
            "displacement_dot_gradient": descent,
            "selected_provider_artifact_sha256": _sha(selected_provider_artifact_sha256, label="selected all-six provider"),
            "fit_support_family_ids": families,
            "fit_support_example_ids_by_family": ids,
            "fit_support_provider_trace_receipt_sha256s_by_family": executions,
            "fit_support_gain_trace_sha256s_by_family": traces,
            "fit_support_gain_min_by_family": minima,
            "fit_support_gain_max_by_family": maxima,
            "fit_support_gain_distinct_count_by_family": counts,
            "fit_support_global_gain_min": global_min,
            "fit_support_global_gain_max": global_max,
            "fit_support_global_gain_range": global_max - global_min,
            "fit_support_global_gain_distinct_count": global_count,
            "response_nonconstant_on_fit_support": nonconstant,
            "provider_trace_evidence_sha256": _sha(provider_trace_evidence_sha256, label="provider trace evidence"),
            "provider_trace_finite": selected_finite,
            "pointwise_trust_passed": selected_trust,
            "rank_is_16": selected_rank,
            "provider_trace_exact": selected_exact,
            "provider_trace_changed_from_initial": selected_runtime_changed,
            "cv_selection_authorized": cv["cv_selection_authorized"],
            "strict_descent_direction": direction["strict_descent_direction"],
            "final_candidate_authorized": authorized,
            "rollback_to_initial_weights": not authorized,
            "final_exact_fit_objectives_used": False,
            "outer_held_objectives_used": False,
            "raw_tensors_logits_targets_or_gradients_serialized": False,
        },
    )


def validate_tangent_response_final_candidate_receipt(
    value: Mapping[str, object], *, cv_receipt: Mapping[str, object],
    direction_receipt: Mapping[str, object], ray_receipt: Mapping[str, object],
) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response final candidate")
    _exact(selected, _FINAL_CANDIDATE_KEYS, label="tangent-response final candidate")
    rebuilt = build_tangent_response_final_candidate_receipt(
        cv_receipt=cv_receipt, direction_receipt=direction_receipt, ray_receipt=ray_receipt,
        selected_provider_artifact_sha256=selected["selected_provider_artifact_sha256"],
        fit_support_example_ids_by_family=_mapping(selected["fit_support_example_ids_by_family"], label="fit-support IDs"),
        fit_support_provider_trace_receipt_sha256s_by_family=_mapping(selected["fit_support_provider_trace_receipt_sha256s_by_family"], label="fit-support provider traces"),
        fit_support_gain_trace_sha256s_by_family=_mapping(selected["fit_support_gain_trace_sha256s_by_family"], label="fit-support traces"),
        fit_support_gain_min_by_family=_mapping(selected["fit_support_gain_min_by_family"], label="fit-support minima"),
        fit_support_gain_max_by_family=_mapping(selected["fit_support_gain_max_by_family"], label="fit-support maxima"),
        fit_support_gain_distinct_count_by_family=_mapping(selected["fit_support_gain_distinct_count_by_family"], label="fit-support counts"),
        provider_trace_evidence_sha256=selected["provider_trace_evidence_sha256"],
        provider_trace_finite=selected["provider_trace_finite"],
        pointwise_trust_passed=selected["pointwise_trust_passed"],
        rank_is_16=selected["rank_is_16"], provider_trace_exact=selected["provider_trace_exact"],
        provider_trace_changed_from_initial=selected["provider_trace_changed_from_initial"],
        final_exact_fit_objectives_used=selected["final_exact_fit_objectives_used"],
        outer_held_objectives_used=selected["outer_held_objectives_used"],
    )
    _same(selected, rebuilt, label="tangent-response final candidate")
    return rebuilt


_FIT_KEYS = {
    "schema", "protocol_sha256", "scientific_status",
    "fresh_family_disjoint_claim_authorized", "serving_claim_authorized",
    "compression_claim_authorized", "response_law", "cv_receipt", "cv_artifact_sha256",
    "final_direction_receipt", "final_direction_artifact_sha256", "final_ray_receipt",
    "final_ray_artifact_sha256", "final_candidate_receipt", "final_candidate_artifact_sha256",
    "selected_fraction", "selected_weights", "selected_weight_sha256",
    "selected_provider_artifact_sha256", "selected_box_certificate",
    "response_nonconstant_on_fit_support", "cv_selection_authorized",
    "learned_candidate_authorized", "rollback_to_initial_weights", "held_score_authorized",
    "post_cv_all_six_exact_rescore_performed", "raw_tensors_logits_targets_or_gradients_serialized",
    "artifact_sha256",
}


def build_tangent_response_fit_receipt(
    *, cv_receipt: Mapping[str, object], final_direction_receipt: Mapping[str, object],
    final_ray_receipt: Mapping[str, object], final_candidate_receipt: Mapping[str, object],
) -> dict[str, object]:
    cv = validate_tangent_response_cv_receipt(cv_receipt)
    direction = validate_tangent_response_direction_receipt(final_direction_receipt)
    ray = validate_tangent_response_ray_receipt(final_ray_receipt, direction_receipt=direction)
    candidate = validate_tangent_response_final_candidate_receipt(
        final_candidate_receipt, cv_receipt=cv, direction_receipt=direction, ray_receipt=ray
    )
    authorized = candidate["final_candidate_authorized"] is True
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_fit.v1", _FIT_DOMAIN,
        {
            "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "serving_claim_authorized": False,
            "compression_claim_authorized": False,
            "response_law": cv["response_law"],
            "cv_receipt": cv,
            "cv_artifact_sha256": cv["artifact_sha256"],
            "final_direction_receipt": direction,
            "final_direction_artifact_sha256": direction["artifact_sha256"],
            "final_ray_receipt": ray,
            "final_ray_artifact_sha256": ray["artifact_sha256"],
            "final_candidate_receipt": candidate,
            "final_candidate_artifact_sha256": candidate["artifact_sha256"],
            "selected_fraction": candidate["selected_cv_fraction"],
            "selected_weights": candidate["weights"],
            "selected_weight_sha256": candidate["weight_sha256"],
            "selected_provider_artifact_sha256": candidate["selected_provider_artifact_sha256"] if authorized else None,
            "selected_box_certificate": candidate["box_certificate"],
            "response_nonconstant_on_fit_support": candidate["response_nonconstant_on_fit_support"],
            "cv_selection_authorized": cv["cv_selection_authorized"],
            "learned_candidate_authorized": authorized,
            "rollback_to_initial_weights": not authorized,
            "held_score_authorized": authorized,
            "post_cv_all_six_exact_rescore_performed": False,
            "raw_tensors_logits_targets_or_gradients_serialized": False,
        },
    )


def validate_tangent_response_fit_receipt(value: Mapping[str, object]) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response fit")
    _exact(selected, _FIT_KEYS, label="tangent-response fit")
    if selected["post_cv_all_six_exact_rescore_performed"] is not False or selected["raw_tensors_logits_targets_or_gradients_serialized"] is not False:
        raise ValueError("tangent-response fit scientific boundary differs")
    rebuilt = build_tangent_response_fit_receipt(
        cv_receipt=_mapping(selected["cv_receipt"], label="CV receipt"),
        final_direction_receipt=_mapping(selected["final_direction_receipt"], label="final direction"),
        final_ray_receipt=_mapping(selected["final_ray_receipt"], label="final ray"),
        final_candidate_receipt=_mapping(selected["final_candidate_receipt"], label="final candidate"),
    )
    _same(selected, rebuilt, label="tangent-response fit")
    return rebuilt


_BUNDLE_KEYS = {
    "schema", "protocol_sha256", "scientific_status",
    "fresh_family_disjoint_claim_authorized", "serving_claim_authorized",
    "compression_claim_authorized", "law_order", "fit_receipts_by_law",
    "fit_artifact_sha256s_by_law", "selected_provider_artifact_sha256s_by_law",
    "selected_weight_sha256s_by_law", "selected_weights_by_law",
    "selected_fractions_by_law", "initial_provider_artifact_sha256s_by_law",
    "gradient_evidence_sha256s_by_law", "shared_source_artifact_sha256s",
    "family_ids", "excluded_family_ids", "fit_family_ids", "fraction_ladder",
    "cv_provider_hash_sets_disjoint_by_law", "cv_execution_hash_sets_disjoint_by_law",
    "fit_support_trace_hash_sets_disjoint_by_law",
    "both_cv_selections_authorized", "both_final_fit_support_responses_nonconstant",
    "both_fits_authorized", "held_score_authorized",
    "post_cv_all_six_exact_rescore_performed", "raw_tensors_logits_targets_or_gradients_serialized",
    "artifact_sha256",
}


def build_tangent_response_two_fit_bundle_receipt(
    *, signed_log_fit_receipt: Mapping[str, object], linear_fit_receipt: Mapping[str, object]
) -> dict[str, object]:
    """Bind two independent law-specific OOF fits without permitting swaps."""

    fits = {
        "signed_log": validate_tangent_response_fit_receipt(signed_log_fit_receipt),
        "linear": validate_tangent_response_fit_receipt(linear_fit_receipt),
    }
    for law in TANGENT_RESPONSE_LAWS:
        if fits[law]["response_law"] != law:
            raise ValueError(f"{law} fit receipt carries the wrong response law")
    cv_by_law = {law: fits[law]["cv_receipt"] for law in TANGENT_RESPONSE_LAWS}
    final_by_law = {law: fits[law]["final_direction_receipt"] for law in TANGENT_RESPONSE_LAWS}
    for field in ("source_artifact_sha256s", "family_ids", "excluded_family_ids", "fit_family_ids"):
        if _canonical(cv_by_law["signed_log"][field]) != _canonical(cv_by_law["linear"][field]):
            raise ValueError(f"two-fit bundle {field} lineage differs")
    gradient_evidence = {
        law: _sha(final_by_law[law]["gradient_evidence_sha256"], label=f"{law} gradient evidence")
        for law in TANGENT_RESPONSE_LAWS
    }
    if len(set(gradient_evidence.values())) != len(TANGENT_RESPONSE_LAWS):
        raise ValueError("two-fit bundle requires independent law-specific gradient evidence")
    initial_providers = {}
    for law in TANGENT_RESPONSE_LAWS:
        candidates_by_fold = _mapping(
            cv_by_law[law]["candidate_receipts_by_validation_family"],
            label=f"{law} CV candidates",
        )
        first_fold = tuple(cv_by_law[law]["fold_order"])[0]
        fold_candidates = tuple(candidates_by_fold[first_fold])
        zero_candidate = next(
            item for item in fold_candidates if float(item["fraction"]) == 0.0
        )
        initial_providers[law] = _sha(
            zero_candidate["provider_artifact_sha256"],
            label=f"{law} initial continuous provider",
        )
    if len(set(initial_providers.values())) != len(TANGENT_RESPONSE_LAWS):
        raise ValueError("signed-log and linear initial providers must be law-distinct")
    cv_provider_sets: dict[str, set[str]] = {}
    cv_execution_sets: dict[str, set[str]] = {}
    fit_support_trace_sets: dict[str, set[str]] = {}
    for law in TANGENT_RESPONSE_LAWS:
        candidate_folds = _mapping(cv_by_law[law]["candidate_receipts_by_validation_family"], label=f"{law} CV candidates")
        candidate_values = tuple(item for fold in candidate_folds.values() for item in tuple(fold))
        cv_provider_sets[law] = {str(item["provider_artifact_sha256"]) for item in candidate_values}
        cv_execution_sets[law] = {
            str(receipt)
            for item in candidate_values
            for receipt in _mapping(item["validation_execution_receipt_sha256s_by_example"], label=f"{law} CV executions").values()
        }
        final_candidate = _mapping(fits[law]["final_candidate_receipt"], label=f"{law} final candidate")
        fit_support_trace_sets[law] = {
            str(receipt)
            for family_traces in _mapping(final_candidate["fit_support_provider_trace_receipt_sha256s_by_family"], label=f"{law} fit-support traces").values()
            for receipt in _mapping(family_traces, label=f"{law} family fit-support traces").values()
        }
    providers_disjoint = cv_provider_sets["signed_log"].isdisjoint(cv_provider_sets["linear"])
    executions_disjoint = cv_execution_sets["signed_log"].isdisjoint(cv_execution_sets["linear"])
    traces_disjoint = fit_support_trace_sets["signed_log"].isdisjoint(fit_support_trace_sets["linear"])
    if not providers_disjoint or not executions_disjoint or not traces_disjoint:
        raise ValueError("two-fit bundle requires law-disjoint provider, execution, and trace evidence")
    selected_providers: dict[str, str | None] = {}
    for law in TANGENT_RESPONSE_LAWS:
        raw = fits[law]["selected_provider_artifact_sha256"]
        selected_providers[law] = None if raw is None else _sha(raw, label=f"{law} selected provider")
    both_cv = all(fits[law]["cv_selection_authorized"] is True for law in TANGENT_RESPONSE_LAWS)
    both_nonconstant = all(fits[law]["response_nonconstant_on_fit_support"] is True for law in TANGENT_RESPONSE_LAWS)
    both_authorized = all(
        fits[law]["learned_candidate_authorized"] is True
        and fits[law]["held_score_authorized"] is True
        for law in TANGENT_RESPONSE_LAWS
    )
    if both_authorized and (
        any(value is None for value in selected_providers.values())
        or len(set(selected_providers.values())) != len(TANGENT_RESPONSE_LAWS)
    ):
        raise ValueError("learned law providers must be present and distinct")
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_two_fit_bundle.v1",
        _BUNDLE_DOMAIN,
        {
            "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "serving_claim_authorized": False,
            "compression_claim_authorized": False,
            "law_order": TANGENT_RESPONSE_LAWS,
            "fit_receipts_by_law": fits,
            "fit_artifact_sha256s_by_law": {law: fits[law]["artifact_sha256"] for law in TANGENT_RESPONSE_LAWS},
            "selected_provider_artifact_sha256s_by_law": selected_providers,
            "selected_weight_sha256s_by_law": {law: fits[law]["selected_weight_sha256"] for law in TANGENT_RESPONSE_LAWS},
            "selected_weights_by_law": {law: fits[law]["selected_weights"] for law in TANGENT_RESPONSE_LAWS},
            "selected_fractions_by_law": {law: fits[law]["selected_fraction"] for law in TANGENT_RESPONSE_LAWS},
            "initial_provider_artifact_sha256s_by_law": initial_providers,
            "gradient_evidence_sha256s_by_law": gradient_evidence,
            "shared_source_artifact_sha256s": cv_by_law["signed_log"]["source_artifact_sha256s"],
            "family_ids": cv_by_law["signed_log"]["family_ids"],
            "excluded_family_ids": cv_by_law["signed_log"]["excluded_family_ids"],
            "fit_family_ids": cv_by_law["signed_log"]["fit_family_ids"],
            "fraction_ladder": TANGENT_RESPONSE_FRACTIONS,
            "cv_provider_hash_sets_disjoint_by_law": providers_disjoint,
            "cv_execution_hash_sets_disjoint_by_law": executions_disjoint,
            "fit_support_trace_hash_sets_disjoint_by_law": traces_disjoint,
            "both_cv_selections_authorized": both_cv,
            "both_final_fit_support_responses_nonconstant": both_nonconstant,
            "both_fits_authorized": both_authorized,
            "held_score_authorized": both_authorized and both_cv and both_nonconstant,
            "post_cv_all_six_exact_rescore_performed": False,
            "raw_tensors_logits_targets_or_gradients_serialized": False,
        },
    )


def validate_tangent_response_two_fit_bundle_receipt(value: Mapping[str, object]) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response two-fit bundle")
    _exact(selected, _BUNDLE_KEYS, label="tangent-response two-fit bundle")
    if selected["protocol_sha256"] != TANGENT_RESPONSE_PROTOCOL_SHA256 or selected["post_cv_all_six_exact_rescore_performed"] is not False or selected["raw_tensors_logits_targets_or_gradients_serialized"] is not False:
        raise ValueError("two-fit bundle protocol/scientific boundary differs")
    fits = _mapping(selected["fit_receipts_by_law"], label="fit receipts by law")
    if set(fits) != set(TANGENT_RESPONSE_LAWS):
        raise ValueError("two-fit bundle law geometry differs")
    rebuilt = build_tangent_response_two_fit_bundle_receipt(
        signed_log_fit_receipt=_mapping(fits["signed_log"], label="signed-log fit"),
        linear_fit_receipt=_mapping(fits["linear"], label="linear fit"),
    )
    _same(selected, rebuilt, label="tangent-response two-fit bundle")
    return rebuilt


_SCORE_KEYS = {
    "schema", "fit_bundle_artifact_sha256", "outer_held_family_id", "held_family_id",
    "arm", "response_law", "response_polarity", "objective",
    "response_weight_sha256", "provider_artifact_sha256", "execution_receipt_sha256", "score_source",
    "predicted_only", "finite", "pointwise_trust_passed", "rank_is_16",
    "execution_changed_from_base", "response_nonconstant", "artifact_sha256",
}

_ARM_SEMANTICS = {
    "base": ("base", 0),
    "constant_plus_one": ("constant", 1),
    "fixed_signed_log": ("signed_log", 1),
    "fixed_linear": ("linear", 1),
    "learned_signed_log": ("signed_log", 1),
    "learned_linear": ("linear", 1),
    "learned_signed_log_sign_flip": ("signed_log", -1),
}


def build_tangent_response_held_arm_score(
    *, fit_bundle_receipt: Mapping[str, object], outer_held_family_id: str,
    held_family_id: str, arm: str, response_law: str, response_polarity: int,
    response_weight_sha256: str, objective: float, provider_artifact_sha256: str,
    execution_receipt_sha256: str,
    finite: bool, pointwise_trust_passed: bool, rank_is_16: bool,
    execution_changed_from_base: bool, response_nonconstant: bool,
    score_source: str = "exact_finite_held_execution",
) -> dict[str, object]:
    bundle = validate_tangent_response_two_fit_bundle_receipt(fit_bundle_receipt)
    if bundle["held_score_authorized"] is not True:
        raise ValueError("two-fit tangent fit did not authorize held scoring")
    outer = _identifier(outer_held_family_id, label="outer held family")
    held = _identifier(held_family_id, label="held family")
    if outer == held or tuple(sorted((outer, held))) != tuple(bundle["excluded_family_ids"]):
        raise ValueError("held score family geometry differs")
    selected_arm = _identifier(arm, label="tangent-response arm")
    if selected_arm not in TANGENT_RESPONSE_ARMS:
        raise ValueError("tangent-response arm differs")
    expected_law, expected_polarity = _ARM_SEMANTICS[selected_arm]
    selected_law = _identifier(response_law, label="held response law")
    if selected_law != expected_law or type(response_polarity) is not int or response_polarity != expected_polarity:
        raise ValueError("held arm response law or polarity differs")
    if score_source != "exact_finite_held_execution":
        raise ValueError("held scores require exact finite execution")
    provider = _sha(provider_artifact_sha256, label="held provider")
    learned = bundle["selected_provider_artifact_sha256s_by_law"]
    learned_weights = bundle["selected_weight_sha256s_by_law"]
    if selected_arm == "learned_signed_log" and provider != learned["signed_log"]:
        raise ValueError("learned signed-log provider differs from its fit")
    if selected_arm == "learned_linear" and provider != learned["linear"]:
        raise ValueError("learned linear provider differs from its fit")
    weight_hash = _sha(response_weight_sha256, label="held response weights")
    if selected_arm in ("learned_signed_log", "learned_signed_log_sign_flip") and weight_hash != learned_weights["signed_log"]:
        raise ValueError("learned signed-log or mirror weights differ from the fitted weights")
    if selected_arm == "learned_linear" and weight_hash != learned_weights["linear"]:
        raise ValueError("learned linear weights differ from the fitted weights")
    changed = _boolean(execution_changed_from_base, label="execution change")
    nonconstant = _boolean(response_nonconstant, label="response nonconstant")
    if selected_arm == "base" and changed:
        raise ValueError("base execution cannot differ from itself")
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_held_score.v1", _SCORE_DOMAIN,
        {
            "fit_bundle_artifact_sha256": bundle["artifact_sha256"],
            "outer_held_family_id": outer,
            "held_family_id": held,
            "arm": selected_arm,
            "response_law": selected_law,
            "response_polarity": response_polarity,
            "response_weight_sha256": weight_hash,
            "objective": _number(objective, label="held objective", nonnegative=True),
            "provider_artifact_sha256": provider,
            "execution_receipt_sha256": _sha(execution_receipt_sha256, label="held execution"),
            "score_source": score_source,
            "predicted_only": False,
            "finite": _boolean(finite, label="held finite"),
            "pointwise_trust_passed": _boolean(pointwise_trust_passed, label="held trust"),
            "rank_is_16": _boolean(rank_is_16, label="held rank"),
            "execution_changed_from_base": changed,
            "response_nonconstant": nonconstant,
        },
    )


def validate_tangent_response_held_arm_score(
    value: Mapping[str, object], *, fit_bundle_receipt: Mapping[str, object]
) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response held score")
    _exact(selected, _SCORE_KEYS, label="tangent-response held score")
    if selected["predicted_only"] is not False:
        raise ValueError("predicted-only held score is forbidden")
    rebuilt = build_tangent_response_held_arm_score(
        fit_bundle_receipt=fit_bundle_receipt,
        outer_held_family_id=selected["outer_held_family_id"], held_family_id=selected["held_family_id"],
        arm=selected["arm"], response_law=selected["response_law"], response_polarity=selected["response_polarity"],
        response_weight_sha256=selected["response_weight_sha256"], objective=selected["objective"],
        provider_artifact_sha256=selected["provider_artifact_sha256"],
        execution_receipt_sha256=selected["execution_receipt_sha256"], finite=selected["finite"],
        pointwise_trust_passed=selected["pointwise_trust_passed"], rank_is_16=selected["rank_is_16"],
        execution_changed_from_base=selected["execution_changed_from_base"],
        response_nonconstant=selected["response_nonconstant"], score_source=selected["score_source"],
    )
    _same(selected, rebuilt, label="tangent-response held score")
    return rebuilt


_ROLE_KEYS = {
    "schema", "fit_bundle_artifact_sha256", "outer_held_family_id", "held_family_id",
    "arm_scores", "arm_objectives", "arm_provider_artifact_sha256s",
    "arm_execution_receipt_sha256s", "all_arms_exactly_executed", "artifact_sha256",
}


def build_tangent_response_held_role_receipt(
    *, fit_bundle_receipt: Mapping[str, object], arm_scores: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    bundle = validate_tangent_response_two_fit_bundle_receipt(fit_bundle_receipt)
    scores = tuple(
        validate_tangent_response_held_arm_score(item, fit_bundle_receipt=bundle)
        for item in _sequence(arm_scores, label="held arm scores")
    )
    by_arm = {str(item["arm"]): item for item in scores}
    if len(scores) != len(TANGENT_RESPONSE_ARMS) or len(by_arm) != len(scores) or set(by_arm) != set(TANGENT_RESPONSE_ARMS):
        raise ValueError("held role requires all seven unique arms")
    ordered = tuple(by_arm[arm] for arm in TANGENT_RESPONSE_ARMS)
    outer_ids = {item["outer_held_family_id"] for item in ordered}
    held_ids = {item["held_family_id"] for item in ordered}
    if len(outer_ids) != 1 or len(held_ids) != 1:
        raise ValueError("held role family bindings differ")
    providers = tuple(str(item["provider_artifact_sha256"]) for item in ordered)
    executions = tuple(str(item["execution_receipt_sha256"]) for item in ordered)
    if len(set(providers)) != len(providers) or len(set(executions)) != len(executions):
        raise ValueError("held role arms require distinct providers and executions")
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_held_role.v1", _ROLE_DOMAIN,
        {
            "fit_bundle_artifact_sha256": bundle["artifact_sha256"],
            "outer_held_family_id": next(iter(outer_ids)),
            "held_family_id": next(iter(held_ids)),
            "arm_scores": ordered,
            "arm_objectives": {arm: by_arm[arm]["objective"] for arm in TANGENT_RESPONSE_ARMS},
            "arm_provider_artifact_sha256s": {arm: by_arm[arm]["provider_artifact_sha256"] for arm in TANGENT_RESPONSE_ARMS},
            "arm_execution_receipt_sha256s": {arm: by_arm[arm]["execution_receipt_sha256"] for arm in TANGENT_RESPONSE_ARMS},
            "all_arms_exactly_executed": True,
        },
    )


def validate_tangent_response_held_role_receipt(
    value: Mapping[str, object], *, fit_bundle_receipt: Mapping[str, object]
) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response held role")
    _exact(selected, _ROLE_KEYS, label="tangent-response held role")
    if selected["all_arms_exactly_executed"] is not True:
        raise ValueError("held role lacks exact executions")
    rebuilt = build_tangent_response_held_role_receipt(
        fit_bundle_receipt=fit_bundle_receipt,
        arm_scores=tuple(_mapping(item, label="held arm score") for item in _sequence(selected["arm_scores"], label="held arm scores")),
    )
    _same(selected, rebuilt, label="tangent-response held role")
    return rebuilt


_QUALIFICATION_KEYS = {
    "schema", "protocol_sha256", "scientific_status",
    "fresh_family_disjoint_claim_authorized", "serving_claim_authorized",
    "compression_claim_authorized", "fit_bundle_artifact_sha256", "role_artifact_sha256s",
    "aggregation", "arm_macro_objectives", "base_macro_denominator_valid",
    "learned_signed_log_base_macro_relative_improvement", "required_base_macro_relative_improvement",
    "learned_signed_log_improves_both_roles", "fixed_log_macro_denominator_valid",
    "learned_signed_log_fixed_log_macro_relative_improvement",
    "required_fixed_log_macro_relative_improvement", "learned_signed_log_beats_constant_macro",
    "learned_signed_log_beats_learned_linear_macro", "learned_signed_log_beats_mirror_both_roles",
    "learned_signed_log_nonconstant_both_roles", "all_arms_finite_trusted_rank16_changed_exact",
    "passed", "artifact_sha256",
}


def build_tangent_response_pair_qualification(
    *, fit_bundle_receipt: Mapping[str, object], roles: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    bundle = validate_tangent_response_two_fit_bundle_receipt(fit_bundle_receipt)
    if bundle["held_score_authorized"] is not True:
        raise ValueError("tangent-response bundle did not authorize held qualification")
    selected_roles = tuple(
        validate_tangent_response_held_role_receipt(item, fit_bundle_receipt=bundle)
        for item in _sequence(roles, label="held roles")
    )
    if len(selected_roles) != _HELD_ROLE_COUNT or len({item["artifact_sha256"] for item in selected_roles}) != _HELD_ROLE_COUNT:
        raise ValueError("pair qualification requires two unique reciprocal roles")
    excluded = tuple(bundle["excluded_family_ids"])
    pairs = {(item["outer_held_family_id"], item["held_family_id"]) for item in selected_roles}
    if pairs != {(excluded[0], excluded[1]), (excluded[1], excluded[0])}:
        raise ValueError("pair qualification roles are not reciprocal")
    ordered = tuple(sorted(selected_roles, key=lambda item: str(item["outer_held_family_id"])))
    objectives = {
        arm: tuple(float(_mapping(role["arm_objectives"], label="arm objectives")[arm]) for role in ordered)
        for arm in TANGENT_RESPONSE_ARMS
    }
    macros = {arm: math.fsum(values) / _HELD_ROLE_COUNT for arm, values in objectives.items()}
    learned = objectives["learned_signed_log"]
    base = objectives["base"]
    fixed = objectives["fixed_signed_log"]
    mirror = objectives["learned_signed_log_sign_flip"]
    base_valid = macros["base"] > _numerical_floor(macros["base"])
    fixed_valid = macros["fixed_signed_log"] > _numerical_floor(macros["fixed_signed_log"])
    base_relative = (macros["base"] - macros["learned_signed_log"]) / macros["base"] if base_valid else 0.0
    fixed_relative = (macros["fixed_signed_log"] - macros["learned_signed_log"]) / macros["fixed_signed_log"] if fixed_valid else 0.0
    improves_both = all(base_value - learned_value > _numerical_floor(base_value) for base_value, learned_value in zip(base, learned))
    beats_constant = macros["constant_plus_one"] - macros["learned_signed_log"] > _numerical_floor(macros["constant_plus_one"])
    beats_linear = macros["learned_linear"] - macros["learned_signed_log"] > _numerical_floor(macros["learned_linear"])
    beats_mirror = all(mirror_value - learned_value > _numerical_floor(mirror_value) for mirror_value, learned_value in zip(mirror, learned))
    nonconstant_both = all(
        next(score for score in role["arm_scores"] if score["arm"] == "learned_signed_log")["response_nonconstant"] is True
        for role in ordered
    )
    health = all(
        score["finite"] is True
        and score["pointwise_trust_passed"] is True
        and score["rank_is_16"] is True
        and score["score_source"] == "exact_finite_held_execution"
        and score["predicted_only"] is False
        and (score["arm"] == "base" or score["execution_changed_from_base"] is True)
        for role in ordered for score in role["arm_scores"]
    )
    passed = bool(
        base_valid and base_relative >= _BASE_MATERIALITY and improves_both
        and fixed_valid and fixed_relative >= _FIXED_LOG_MATERIALITY
        and beats_constant and beats_linear and beats_mirror and nonconstant_both and health
    )
    return _finish(
        "fisher_graph.complete_h4_fisher_tangent_response_pair_qualification.v1",
        _QUALIFICATION_DOMAIN,
        {
            "protocol_sha256": TANGENT_RESPONSE_PROTOCOL_SHA256,
            "scientific_status": "development_only_reused_a16",
            "fresh_family_disjoint_claim_authorized": False,
            "serving_claim_authorized": False,
            "compression_claim_authorized": False,
            "fit_bundle_artifact_sha256": bundle["artifact_sha256"],
            "role_artifact_sha256s": tuple(item["artifact_sha256"] for item in ordered),
            "aggregation": "family_equal_two_reciprocal_roles",
            "arm_macro_objectives": macros,
            "base_macro_denominator_valid": base_valid,
            "learned_signed_log_base_macro_relative_improvement": base_relative,
            "required_base_macro_relative_improvement": _BASE_MATERIALITY,
            "learned_signed_log_improves_both_roles": improves_both,
            "fixed_log_macro_denominator_valid": fixed_valid,
            "learned_signed_log_fixed_log_macro_relative_improvement": fixed_relative,
            "required_fixed_log_macro_relative_improvement": _FIXED_LOG_MATERIALITY,
            "learned_signed_log_beats_constant_macro": beats_constant,
            "learned_signed_log_beats_learned_linear_macro": beats_linear,
            "learned_signed_log_beats_mirror_both_roles": beats_mirror,
            "learned_signed_log_nonconstant_both_roles": nonconstant_both,
            "all_arms_finite_trusted_rank16_changed_exact": health,
            "passed": passed,
        },
    )


def validate_tangent_response_pair_qualification(
    value: Mapping[str, object], *, fit_bundle_receipt: Mapping[str, object],
    roles: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected = _mapping(value, label="tangent-response pair qualification")
    _exact(selected, _QUALIFICATION_KEYS, label="tangent-response pair qualification")
    rebuilt = build_tangent_response_pair_qualification(
        fit_bundle_receipt=fit_bundle_receipt, roles=roles
    )
    _same(selected, rebuilt, label="tangent-response pair qualification")
    return rebuilt


def tangent_response_work_accounting(
    *, collection_forward_count: int, collection_backward_count: int,
    endpoint_forward_count: int, endpoint_backward_count: int,
    endpoint_local_contraction_count: int, endpoint_teacher_access_count: int,
    law_count: int, fit_prompt_count: int,
    cv_fold_count: int, validation_prompts_per_fold: int, fraction_count: int,
    fraction_zero_vjp_reused: bool, held_role_count: int, held_arm_count: int,
    held_prompts_per_role: int, held_scoring_executed: bool,
) -> dict[str, int | bool]:
    """Return exact V20e work for CV-terminal or full reused-diagnostic paths."""

    collection_forward = _integer(collection_forward_count, label="collection forward count")
    collection_backward = _integer(collection_backward_count, label="collection backward count")
    endpoint_forward = _integer(endpoint_forward_count, label="endpoint forward count")
    endpoint_backward = _integer(endpoint_backward_count, label="endpoint backward count")
    endpoint_local = _integer(endpoint_local_contraction_count, label="endpoint local contraction count")
    endpoint_teacher = _integer(endpoint_teacher_access_count, label="endpoint teacher access count")
    if endpoint_teacher > endpoint_forward:
        raise ValueError("endpoint teacher accesses cannot exceed endpoint forwards")
    laws = _integer(law_count, label="law count", minimum=1)
    prompts = _integer(fit_prompt_count, label="fit prompt count", minimum=1)
    folds = _integer(cv_fold_count, label="CV fold count", minimum=1)
    validation_prompts = _integer(validation_prompts_per_fold, label="validation prompts per fold", minimum=1)
    fractions = _integer(fraction_count, label="fraction count", minimum=1)
    reuse_zero = _boolean(fraction_zero_vjp_reused, label="zero-fraction VJP reuse")
    held_roles = _integer(held_role_count, label="held role count", minimum=1)
    held_arms = _integer(held_arm_count, label="held arm count", minimum=1)
    held_prompts = _integer(held_prompts_per_role, label="held prompts per role", minimum=1)
    held_executed = _boolean(held_scoring_executed, label="held scoring executed")
    frozen_geometry = {
        "collection_forward_count": 32,
        "collection_backward_count": 16,
        "endpoint_forward_count": 12,
        "endpoint_backward_count": 12,
        "endpoint_local_contraction_count": 12,
        "endpoint_teacher_access_count": 12,
        "law_count": len(TANGENT_RESPONSE_LAWS),
        "fit_prompt_count": 12,
        "cv_fold_count": _FIT_FAMILY_COUNT,
        "validation_prompts_per_fold": 2,
        "fraction_count": len(TANGENT_RESPONSE_FRACTIONS),
        "held_role_count": _HELD_ROLE_COUNT,
        "held_arm_count": len(TANGENT_RESPONSE_ARMS),
        "held_prompts_per_role": 2,
    }
    observed_geometry = {
        "collection_forward_count": collection_forward,
        "collection_backward_count": collection_backward,
        "endpoint_forward_count": endpoint_forward,
        "endpoint_backward_count": endpoint_backward,
        "endpoint_local_contraction_count": endpoint_local,
        "endpoint_teacher_access_count": endpoint_teacher,
        "law_count": laws,
        "fit_prompt_count": prompts,
        "cv_fold_count": folds,
        "validation_prompts_per_fold": validation_prompts,
        "fraction_count": fractions,
        "held_role_count": held_roles,
        "held_arm_count": held_arms,
        "held_prompts_per_role": held_prompts,
    }
    if observed_geometry != frozen_geometry:
        raise ValueError("V20e work phase allocation differs from the frozen protocol")
    if not reuse_zero:
        raise ValueError("V20e requires exact beta-zero VJP reuse")
    law_gradient_forward = laws * prompts
    law_gradient_backward = laws * prompts
    law_gradient_local = laws * prompts
    scored_fractions = fractions - (1 if reuse_zero else 0)
    cv_forward = laws * folds * validation_prompts * scored_fractions
    held_forward = held_roles * held_arms * held_prompts if held_executed else 0
    total_forward = collection_forward + endpoint_forward + law_gradient_forward + cv_forward + held_forward
    total_backward = collection_backward + endpoint_backward + law_gradient_backward
    total_local = endpoint_local + law_gradient_local
    teacher_accesses = endpoint_teacher + law_gradient_forward + cv_forward + held_forward
    return {
        "collection_forward_count": collection_forward,
        "collection_backward_count": collection_backward,
        "endpoint_forward_count": endpoint_forward,
        "endpoint_backward_count": endpoint_backward,
        "endpoint_local_contraction_count": endpoint_local,
        "endpoint_teacher_access_count": endpoint_teacher,
        "law_count": laws,
        "fit_prompt_count": prompts,
        "cv_fold_count": folds,
        "validation_prompts_per_fold": validation_prompts,
        "fraction_count": fractions,
        "fraction_zero_vjp_reused": reuse_zero,
        "held_role_count": held_roles,
        "held_arm_count": held_arms,
        "held_prompts_per_role": held_prompts,
        "held_scoring_executed": held_executed,
        "frozen_phase_geometry_passed": True,
        "law_fit_gradient_forward_count": law_gradient_forward,
        "law_fit_gradient_backward_count": law_gradient_backward,
        "law_fit_gradient_local_contraction_count": law_gradient_local,
        # Each law-specific gradient bank evaluates every row outer product
        # exactly once.  LOFO and all-six directions subsequently aggregate
        # the committed family summaries, so they perform no further outers.
        "unique_empirical_fisher_gradient_row_count": laws * prompts,
        "empirical_fisher_outer_product_evaluation_count": laws * prompts,
        "tangent_qp_solve_count": laws * (folds + 1),
        "cv_positive_fraction_score_count": laws * folds * scored_fractions,
        "cv_validation_forward_count": cv_forward,
        "final_all_six_provider_teacher_forward_count": 0,
        "held_forward_count": held_forward,
        "total_forward_count": total_forward,
        "total_backward_count": total_backward,
        "total_local_contraction_count": total_local,
        "total_backward_or_local_gradient_call_count": total_backward + total_local,
        "teacher_access_count": teacher_accesses,
        "held_capability_count": held_roles if held_executed else 0,
        "held_score_count": held_roles * held_arms if held_executed else 0,
    }
