"""Pure V20g box-normalized soft-polarity trust-region protocol.

The response is continuous: ``gain = signed_log(c2) * tanh(eta @ phi)`` with
``phi = [1, c1, c2, c1*c2]``.  This module performs scalar mathematics and
builds authenticated receipts only.  It has no API that accepts Calibration-B,
validation, test, prompt text, tensors, logits, or held-out model data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
import sys

__all__ = [
    "SOFT_POLARITY_ALPHAS",
    "SOFT_POLARITY_FIT_ALPHAS",
    "SOFT_POLARITY_FIT_ETA_MAX_ABS",
    "SOFT_POLARITY_FIT_PROTOCOL_SHA256",
    "SOFT_POLARITY_PROTOCOL_SHA256",
    "build_soft_polarity_candidate_receipt",
    "build_soft_polarity_direction_receipt",
    "build_soft_polarity_fold_receipt",
    "build_soft_polarity_oof_qualification",
    "soft_polarity_work_accounting",
    "validate_soft_polarity_candidate_receipt",
    "validate_soft_polarity_direction_receipt",
    "validate_soft_polarity_fold_receipt",
    "validate_soft_polarity_oof_qualification",
]


SOFT_POLARITY_FIT_ALPHAS = (
    0.0,
    1.0 / 128.0,
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
    2.0,
)
SOFT_POLARITY_ALPHAS = SOFT_POLARITY_FIT_ALPHAS
SOFT_POLARITY_FIT_ETA_MAX_ABS = sys.float_info.max / 8.0
_ARMS = ("base", "fixed_plus", "fixed_minus", "soft_router")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FAMILY_COUNT = 8
_FEATURE_COUNT = 4
_DAMPING_FRACTION = 1.0e-3
_DAMPING_FLOOR = 1.0e-12
_BOX_CORNERS = ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))

_PROTOCOL_DOMAIN = b"fisher-graph:complete-h4-soft-polarity-trust-region-fit:protocol:v20g\0"
_DIRECTION_DOMAIN = b"fisher-graph:complete-h4-soft-polarity-trust-region-fit:direction:v20g\0"
_CANDIDATE_DOMAIN = b"fisher-graph:complete-h4-soft-polarity-trust-region-fit:candidate:v20g\0"
_FOLD_DOMAIN = b"fisher-graph:complete-h4-soft-polarity-trust-region-fit:fold:v20g\0"
_OOF_DOMAIN = b"fisher-graph:complete-h4-soft-polarity-trust-region-fit:oof:v20g\0"
_GRADIENT_DOMAIN = b"fisher-graph:complete-h4-soft-polarity-trust-region-fit:gradient:v20g\0"
_OBJECTIVE_DOMAIN = b"fisher-graph:complete-h4-soft-polarity-trust-region-fit:objective:v20g\0"

_DATA_BOUNDARY = {
    "role": "historically_reused_calibration_a_fit_A16_development_only",
    "development_family_count": 8,
    "outer_validation": "leave_one_whole_development_family_out",
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "fresh_family_disjoint_scores_consumed": False,
}
_PROTOCOL = {
    "protocol": "v20g_continuous_soft_polarity_box_normalized_trust_region",
    "scientific_status": "adaptive_after_pinned_v20f_reused_A16_development_only",
    "features": ("one", "c1", "c2", "c1_times_c2"),
    "envelope": "signed_log_c2_kappa_9",
    "polarity": "tanh_eta_dot_phi",
    "gain": "envelope_times_polarity",
    "fisher": "family_equal_empirical_gradient_opg",
    "damping": "max_1e-12_1e-3_times_trace_over_four",
    "alphas": SOFT_POLARITY_FIT_ALPHAS,
    "direction_normalization": (
        "divide_by_exact_max_abs_bilinear_router_logit_over_four_box_corners"
    ),
    "trust_region": "normalized_router_logit_radius_tau_2^-7_through_2",
    "schedule_basis": (
        "pinned_v20f_all_eight_folds_selected_alpha_zero_after_"
        "the_first_2^-8_candidate_overshot"
    ),
    "box_guarantee": "abs_eta_dot_phi_at_most_tau_for_all_c1_c2_in_minus1_plus1",
    "alpha_field_semantics": "dimensionless_box_logit_trust_radius_tau",
    "eta_max_abs": SOFT_POLARITY_FIT_ETA_MAX_ABS,
    "selection": "minimum_exact_training_objective_only",
    "tie": "objective_then_smaller_alpha_then_artifact_sha256",
    "outer_validation": "eight_leave_one_whole_development_family_out_folds",
    "outer_arms": _ARMS,
    "authorization": "soft_beats_base_and_fixed_plus_macros_and_wins_six_of_eight_vs_each",
    "rollback": "no_full_refit_or_calibration_b_eligibility_on_any_failed_gate",
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


SOFT_POLARITY_FIT_PROTOCOL_SHA256 = _hash(_PROTOCOL_DOMAIN, _PROTOCOL)
SOFT_POLARITY_PROTOCOL_SHA256 = SOFT_POLARITY_FIT_PROTOCOL_SHA256


def _finish(schema: str, domain: bytes, payload: Mapping[str, object]) -> dict[str, object]:
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


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a canonical nonempty string")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be boolean")
    return value


def _vector4(value: object, label: str) -> tuple[float, float, float, float]:
    result = tuple(_number(item, label) for item in _sequence(value, label))
    if len(result) != _FEATURE_COUNT:
        raise ValueError(f"{label} must have four values")
    return result  # type: ignore[return-value]


def _matrix4(value: object, label: str) -> tuple[tuple[float, float, float, float], ...]:
    result = tuple(_vector4(row, label) for row in _sequence(value, label))
    if len(result) != _FEATURE_COUNT:
        raise ValueError(f"{label} must be four by four")
    return result


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


def _families(value: object) -> tuple[str, ...]:
    result = tuple(sorted(_identifier(item, "family id") for item in _sequence(value, "families")))
    if len(result) != _FAMILY_COUNT or len(set(result)) != _FAMILY_COUNT:
        raise ValueError("exactly eight distinct development families are required")
    return result


def _sources(value: object) -> dict[str, str]:
    mapping = _mapping(value, "source artifacts")
    if not mapping:
        raise ValueError("source artifacts must not be empty")
    return dict(sorted((_identifier(key, "source name"), _sha(item, "source hash")) for key, item in mapping.items()))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _mean_vectors(rows: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    return tuple(_mean([row[index] for row in rows]) for index in range(4))  # type: ignore[return-value]


def _mean_matrices(matrices: Sequence[Sequence[Sequence[float]]]) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(_mean([matrix[row][column] for matrix in matrices]) for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def _opg(rows: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(_mean([row[i] * row[j] for row in rows]) for j in range(4))
        for i in range(4)
    )  # type: ignore[return-value]


def _solve_spd(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> tuple[float, float, float, float]:
    lower = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i + 1):
            residual = matrix[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if not math.isfinite(residual) or residual <= 0.0:
                    raise ValueError("damped Fisher is not positive definite")
                lower[i][j] = math.sqrt(residual)
            else:
                lower[i][j] = residual / lower[j][j]
    y = [0.0] * 4
    for i in range(4):
        y[i] = (rhs[i] - math.fsum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
    x = [0.0] * 4
    for i in range(3, -1, -1):
        x[i] = (y[i] - math.fsum(lower[k][i] * x[k] for k in range(i + 1, 4))) / lower[i][i]
    return tuple(0.0 if item == 0.0 else item for item in x)  # type: ignore[return-value]


def _direction_payload_from_summaries(
    *,
    sources: Mapping[str, str],
    families: tuple[str, ...],
    held: str | None,
    evidence: str,
    example_ids: Mapping[str, tuple[str, ...]],
    gradient_hashes: Mapping[str, tuple[str, ...]],
    family_means: Mapping[str, tuple[float, float, float, float]],
    family_opgs: Mapping[str, tuple[tuple[float, float, float, float], ...]],
) -> dict[str, object]:
    training = (
        families
        if held is None
        else tuple(family for family in families if family != held)
    )
    mean_gradient = _mean_vectors([family_means[family] for family in training])
    fisher = _mean_matrices([family_opgs[family] for family in training])
    damping = max(_DAMPING_FLOOR, _DAMPING_FRACTION * math.fsum(fisher[i][i] for i in range(4)) / 4.0)
    damped = tuple(tuple(fisher[i][j] + (damping if i == j else 0.0) for j in range(4)) for i in range(4))
    raw_direction = _solve_spd(damped, tuple(-item for item in mean_gradient))
    raw_derivative = math.fsum(
        mean_gradient[i] * raw_direction[i] for i in range(4)
    )
    raw_norm = math.sqrt(math.fsum(item * item for item in raw_direction))
    raw_corner_logits = _box_corner_logits(raw_direction)
    normalization_scale = max(abs(item) for item in raw_corner_logits)
    if not math.isfinite(normalization_scale) or normalization_scale <= 0.0:
        raise ValueError("natural direction has no finite box-logit scale")
    direction = tuple(item / normalization_scale for item in raw_direction)
    derivative = raw_derivative / normalization_scale
    norm = raw_norm / normalization_scale
    normalized_corner_logits = tuple(
        item / normalization_scale for item in raw_corner_logits
    )
    return {
        "protocol_sha256": SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "source_artifact_sha256s": dict(sources),
        "all_development_family_ids": families,
        "held_family_id": held,
        "training_family_ids": training,
        "gradient_evidence_sha256": evidence,
        "training_example_ids_by_family": dict(example_ids),
        "training_gradient_sha256s_by_family": dict(gradient_hashes),
        "family_mean_gradients": dict(family_means),
        "family_mean_opg_fishers": dict(family_opgs),
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
            abs(item) for item in normalized_corner_logits
        ),
        "finite": all(
            math.isfinite(item)
            for item in (*raw_direction, *direction, *normalized_corner_logits)
        ),
        "strict_descent": (
            raw_derivative < 0.0
            and raw_norm > 0.0
            and derivative < 0.0
            and norm > 0.0
        ),
        "data_boundary": dict(_DATA_BOUNDARY),
    }


def build_soft_polarity_direction_receipt(
    *,
    source_artifact_sha256s: Mapping[str, str],
    all_development_family_ids: Sequence[str],
    held_family_id: str | None,
    gradient_rows_by_family: Mapping[str, Mapping[str, Sequence[float]]],
    gradient_evidence_sha256: str,
) -> dict[str, object]:
    """Build one seven-family natural direction for an outer fold."""

    sources = _sources(source_artifact_sha256s)
    families = _families(all_development_family_ids)
    held = (
        None
        if held_family_id is None
        else _identifier(held_family_id, "held development family")
    )
    if held is not None and held not in families:
        raise ValueError("held development family is outside the panel")
    training = (
        families
        if held is None
        else tuple(family for family in families if family != held)
    )
    if set(gradient_rows_by_family) != set(training):
        raise ValueError("gradient rows must contain exactly the seven training families")
    example_ids: dict[str, tuple[str, ...]] = {}
    gradient_hashes: dict[str, tuple[str, ...]] = {}
    family_means = {}
    family_opgs = {}
    seen: set[str] = set()
    for family in training:
        raw = _mapping(gradient_rows_by_family[family], f"gradient rows for {family}")
        if not raw:
            raise ValueError("every training family needs gradient rows")
        ids = tuple(sorted(_identifier(key, "gradient example id") for key in raw))
        if seen.intersection(ids):
            raise ValueError("gradient example ids must be globally distinct")
        seen.update(ids)
        rows = tuple(_vector4(raw[example], "gradient row") for example in ids)
        example_ids[family] = ids
        gradient_hashes[family] = tuple(_hash(_GRADIENT_DOMAIN, {"example_id": example, "gradient": row}) for example, row in zip(ids, rows, strict=True))
        family_means[family] = _mean_vectors(rows)
        family_opgs[family] = _opg(rows)
    payload = _direction_payload_from_summaries(
        sources=sources,
        families=families,
        held=held,
        evidence=_sha(gradient_evidence_sha256, "gradient evidence"),
        example_ids=example_ids,
        gradient_hashes=gradient_hashes,
        family_means=family_means,
        family_opgs=family_opgs,
    )
    result = _finish("fisher_graph.complete_h4_soft_polarity_trust_region_direction.v20g", _DIRECTION_DOMAIN, payload)
    validate_soft_polarity_direction_receipt(result)
    return result


def validate_soft_polarity_direction_receipt(value: Mapping[str, object]) -> None:
    receipt = _mapping(value, "direction receipt")
    families = _families(receipt.get("all_development_family_ids"))
    raw_held = receipt.get("held_family_id")
    held = (
        None
        if raw_held is None
        else _identifier(raw_held, "held development family")
    )
    if held is not None and held not in families:
        raise ValueError("held development family is outside the panel")
    training = (
        families
        if held is None
        else tuple(family for family in families if family != held)
    )
    if tuple(receipt.get("training_family_ids", ())) != training:
        raise ValueError("direction training families drifted")
    ids_raw = _mapping(receipt.get("training_example_ids_by_family"), "training example ids")
    hashes_raw = _mapping(receipt.get("training_gradient_sha256s_by_family"), "gradient hashes")
    means_raw = _mapping(receipt.get("family_mean_gradients"), "family gradients")
    opgs_raw = _mapping(receipt.get("family_mean_opg_fishers"), "family Fishers")
    if set(ids_raw) != set(training) or set(hashes_raw) != set(training) or set(means_raw) != set(training) or set(opgs_raw) != set(training):
        raise ValueError("direction family summaries drifted")
    ids = {family: tuple(_identifier(item, "example id") for item in _sequence(ids_raw[family], "example ids")) for family in training}
    hashes = {family: tuple(_sha(item, "gradient hash") for item in _sequence(hashes_raw[family], "gradient hashes")) for family in training}
    if any(not ids[family] or len(ids[family]) != len(hashes[family]) for family in training):
        raise ValueError("direction row commitment geometry drifted")
    flattened_ids = tuple(item for family in training for item in ids[family])
    if (
        any(ids[family] != tuple(sorted(ids[family])) for family in training)
        or len(set(flattened_ids)) != len(flattened_ids)
    ):
        raise ValueError("direction example ids are not canonical and globally distinct")
    means = {family: _vector4(means_raw[family], "family mean gradient") for family in training}
    opgs = {family: _matrix4(opgs_raw[family], "family Fisher") for family in training}
    expected = _direction_payload_from_summaries(
        sources=_sources(receipt.get("source_artifact_sha256s")),
        families=families,
        held=held,
        evidence=_sha(receipt.get("gradient_evidence_sha256"), "gradient evidence"),
        example_ids=ids,
        gradient_hashes=hashes,
        family_means=means,
        family_opgs=opgs,
    )
    if receipt.get("schema") != "fisher_graph.complete_h4_soft_polarity_trust_region_direction.v20g" or receipt.get("protocol_sha256") != SOFT_POLARITY_FIT_PROTOCOL_SHA256 or _canonical({key: receipt.get(key) for key in expected}) != _canonical(expected):
        raise ValueError("direction receipt content drifted")
    artifact = _sha(receipt.get("artifact_sha256"), "direction artifact")
    if artifact != _hash(_DIRECTION_DOMAIN, {key: item for key, item in receipt.items() if key != "artifact_sha256"}):
        raise ValueError("direction receipt hash drifted")


def build_soft_polarity_candidate_receipt(
    *,
    direction_receipt: Mapping[str, object],
    alpha: float,
    exact_train_objectives_by_family: Mapping[str, Mapping[str, float]],
    execution_receipt_sha256: str,
    exact_execution: bool,
) -> dict[str, object]:
    """Bind one alpha proposal to exact seven-family training objectives."""

    validate_soft_polarity_direction_receipt(direction_receipt)
    selected_alpha = _number(alpha, "alpha")
    if selected_alpha not in SOFT_POLARITY_FIT_ALPHAS:
        raise ValueError("alpha is outside the frozen ladder")
    training = tuple(direction_receipt["training_family_ids"])
    if set(exact_train_objectives_by_family) != set(training):
        raise ValueError("candidate objectives must contain seven training families")
    family_means = {}
    example_ids = {}
    objective_hashes = {}
    for family in training:
        raw = _mapping(exact_train_objectives_by_family[family], f"objectives for {family}")
        if not raw:
            raise ValueError("every family needs exact training objectives")
        ids = tuple(sorted(_identifier(key, "objective example id") for key in raw))
        objectives = tuple(_number(raw[example], "training objective") for example in ids)
        example_ids[family] = ids
        objective_hashes[family] = tuple(_hash(_OBJECTIVE_DOMAIN, {"example_id": example, "objective": objective}) for example, objective in zip(ids, objectives, strict=True))
        family_means[family] = _mean(objectives)
    direction = _vector4(direction_receipt["natural_direction"], "natural direction")
    eta = tuple(selected_alpha * item for item in direction)
    if any(abs(item) > SOFT_POLARITY_FIT_ETA_MAX_ABS for item in eta):
        raise ValueError("candidate eta exceeds the runtime numerical domain")
    corner_logits = _box_corner_logits(eta)
    box_logit_max_abs = max(abs(item) for item in corner_logits)
    if box_logit_max_abs > selected_alpha + max(1.0e-15, selected_alpha * 1.0e-12):
        raise ValueError("candidate exceeds the normalized box-logit trust radius")
    payload = {
        "protocol_sha256": SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "direction_artifact_sha256": direction_receipt["artifact_sha256"],
        "held_family_id": direction_receipt["held_family_id"],
        "training_family_ids": training,
        "alpha": selected_alpha,
        "eta": eta,
        "box_corner_logits": corner_logits,
        "box_logit_max_abs": box_logit_max_abs,
        "box_logit_bound": selected_alpha,
        "training_example_ids_by_family": example_ids,
        "training_objective_sha256s_by_family": objective_hashes,
        "family_mean_train_objectives": family_means,
        "family_equal_train_objective": _mean([family_means[family] for family in training]),
        "execution_receipt_sha256": _sha(execution_receipt_sha256, "candidate execution"),
        "exact_execution": _boolean(exact_execution, "exact execution"),
        "finite": all(math.isfinite(item) for item in eta),
        "execution_changed_from_base": selected_alpha > 0.0 and bool(direction_receipt["strict_descent"]),
        "data_boundary": dict(_DATA_BOUNDARY),
    }
    result = _finish("fisher_graph.complete_h4_soft_polarity_trust_region_candidate.v20g", _CANDIDATE_DOMAIN, payload)
    validate_soft_polarity_candidate_receipt(result, direction_receipt=direction_receipt)
    return result


def validate_soft_polarity_candidate_receipt(value: Mapping[str, object], *, direction_receipt: Mapping[str, object]) -> None:
    validate_soft_polarity_direction_receipt(direction_receipt)
    receipt = _mapping(value, "candidate receipt")
    alpha = _number(receipt.get("alpha"), "alpha")
    if alpha not in SOFT_POLARITY_FIT_ALPHAS:
        raise ValueError("candidate alpha drifted")
    direction = _vector4(direction_receipt["natural_direction"], "direction")
    eta = _vector4(receipt.get("eta"), "eta")
    if any(abs(item) > SOFT_POLARITY_FIT_ETA_MAX_ABS for item in eta):
        raise ValueError("candidate eta exceeds the runtime numerical domain")
    training = tuple(direction_receipt["training_family_ids"])
    family_means_raw = _mapping(receipt.get("family_mean_train_objectives"), "family means")
    ids_raw = _mapping(receipt.get("training_example_ids_by_family"), "objective ids")
    hashes_raw = _mapping(receipt.get("training_objective_sha256s_by_family"), "objective hashes")
    if set(family_means_raw) != set(training) or set(ids_raw) != set(training) or set(hashes_raw) != set(training):
        raise ValueError("candidate objective families drifted")
    family_means = {family: _number(family_means_raw[family], "family objective") for family in training}
    expected_corner_logits = _box_corner_logits(eta)
    expected_box_logit_max_abs = max(abs(item) for item in expected_corner_logits)
    seen_ids: set[str] = set()
    for family in training:
        ids = tuple(
            _identifier(item, "objective id")
            for item in _sequence(ids_raw[family], "objective ids")
        )
        hashes = _sequence(hashes_raw[family], "objective hashes")
        if not ids or len(ids) != len(hashes):
            raise ValueError("candidate objective commitment drifted")
        if ids != tuple(sorted(ids)) or seen_ids.intersection(ids):
            raise ValueError("candidate objective ids are not canonical and globally distinct")
        seen_ids.update(ids)
        for item in hashes:
            _sha(item, "objective hash")
    expected_pairs = {
        "schema": "fisher_graph.complete_h4_soft_polarity_trust_region_candidate.v20g",
        "protocol_sha256": SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "direction_artifact_sha256": direction_receipt["artifact_sha256"],
        "held_family_id": direction_receipt["held_family_id"],
        "training_family_ids": training,
        "eta": tuple(alpha * item for item in direction),
        "box_corner_logits": expected_corner_logits,
        "box_logit_max_abs": expected_box_logit_max_abs,
        "box_logit_bound": alpha,
        "family_equal_train_objective": _mean([family_means[family] for family in training]),
        "exact_execution": True,
        "finite": True,
        "execution_changed_from_base": alpha > 0.0 and bool(direction_receipt["strict_descent"]),
        "data_boundary": _DATA_BOUNDARY,
    }
    for key, expected in expected_pairs.items():
        if _canonical(receipt.get(key)) != _canonical(expected):
            raise ValueError(f"candidate {key} drifted")
    _sha(receipt.get("execution_receipt_sha256"), "candidate execution")
    artifact = _sha(receipt.get("artifact_sha256"), "candidate artifact")
    if artifact != _hash(_CANDIDATE_DOMAIN, {key: item for key, item in receipt.items() if key != "artifact_sha256"}):
        raise ValueError("candidate receipt hash drifted")


def _validated_candidate_ladder(
    candidate_receipts: Sequence[Mapping[str, object]],
    *,
    direction_receipt: Mapping[str, object],
) -> dict[float, Mapping[str, object]]:
    candidates = tuple(candidate_receipts)
    if len(candidates) != len(SOFT_POLARITY_FIT_ALPHAS):
        raise ValueError("fold requires one candidate for every alpha")
    by_alpha: dict[float, Mapping[str, object]] = {}
    example_geometry: bytes | None = None
    for candidate in candidates:
        validate_soft_polarity_candidate_receipt(
            candidate, direction_receipt=direction_receipt
        )
        alpha = _number(candidate.get("alpha"), "candidate alpha")
        if alpha in by_alpha:
            raise ValueError("duplicate fold alpha")
        geometry = _canonical(
            _mapping(
                candidate.get("training_example_ids_by_family"),
                "candidate training example ids",
            )
        )
        if example_geometry is not None and geometry != example_geometry:
            raise ValueError("fold candidate scoring geometry differs across alphas")
        example_geometry = geometry
        by_alpha[alpha] = candidate
    if set(by_alpha) != set(SOFT_POLARITY_FIT_ALPHAS):
        raise ValueError("fold alpha ladder is incomplete")
    return by_alpha


def build_soft_polarity_fold_receipt(
    *,
    direction_receipt: Mapping[str, object],
    candidate_receipts: Sequence[Mapping[str, object]],
    held_objectives_by_arm: Mapping[str, Mapping[str, float]],
    held_execution_receipt_sha256s_by_arm: Mapping[str, str],
    held_trace_evidence_sha256: str,
    response_gain_min: float,
    response_gain_max: float,
    response_gain_distinct_count: int,
    finite: bool,
    pointwise_trust_passed: bool,
    exact_execution: bool,
) -> dict[str, object]:
    """Select on seven-family training only, then bind four outer-fold arms."""

    validate_soft_polarity_direction_receipt(direction_receipt)
    if direction_receipt.get("held_family_id") is None:
        raise ValueError("outer-fold receipt requires one held development family")
    candidates = tuple(candidate_receipts)
    by_alpha = _validated_candidate_ladder(
        candidates, direction_receipt=direction_receipt
    )
    selected = min(candidates, key=lambda item: (float(item["family_equal_train_objective"]), float(item["alpha"]), str(item["artifact_sha256"])))
    if set(held_objectives_by_arm) != set(_ARMS) or set(held_execution_receipt_sha256s_by_arm) != set(_ARMS):
        raise ValueError("fold requires exact base/fixed+/fixed-/soft arms")
    held_ids: tuple[str, ...] | None = None
    means = {}
    hashes = {}
    for arm in _ARMS:
        raw = _mapping(held_objectives_by_arm[arm], f"held {arm} objectives")
        ids = tuple(sorted(_identifier(key, "held example id") for key in raw))
        if not ids or (held_ids is not None and ids != held_ids):
            raise ValueError("held arm example geometry differs")
        held_ids = ids
        objectives = tuple(_number(raw[example], f"held {arm} objective") for example in ids)
        means[arm] = _mean(objectives)
        hashes[arm] = tuple(_hash(_OBJECTIVE_DOMAIN, {"arm": arm, "example_id": example, "objective": objective}) for example, objective in zip(ids, objectives, strict=True))
    training_ids = {item for ids in _mapping(direction_receipt["training_example_ids_by_family"], "training ids").values() for item in _sequence(ids, "training ids")}
    if training_ids.intersection(held_ids or ()):
        raise ValueError("outer held examples leaked into direction training")
    gain_min = _number(response_gain_min, "gain minimum")
    gain_max = _number(response_gain_max, "gain maximum")
    if type(response_gain_distinct_count) is not int or response_gain_distinct_count < 0:
        raise TypeError("response gain distinct count must be nonnegative integer")
    bounded = -1.0 <= gain_min <= gain_max <= 1.0
    nonconstant = response_gain_distinct_count >= 2 and gain_min < gain_max
    health = bool(selected["execution_changed_from_base"]) and _boolean(finite, "finite") and _boolean(pointwise_trust_passed, "trust") and _boolean(exact_execution, "exact execution") and bounded and nonconstant
    payload = {
        "protocol_sha256": SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "source_artifact_sha256s": direction_receipt["source_artifact_sha256s"],
        "all_development_family_ids": direction_receipt["all_development_family_ids"],
        "held_family_id": direction_receipt["held_family_id"],
        "direction_artifact_sha256": direction_receipt["artifact_sha256"],
        "candidate_artifact_sha256s_by_alpha": {str(alpha): by_alpha[alpha]["artifact_sha256"] for alpha in SOFT_POLARITY_FIT_ALPHAS},
        "selected_alpha": selected["alpha"],
        "selected_eta": selected["eta"],
        "selected_train_objective": selected["family_equal_train_objective"],
        "selected_candidate_artifact_sha256": selected["artifact_sha256"],
        "selection_frozen_before_held_scores": True,
        "held_example_ids": held_ids,
        "held_objective_sha256s_by_arm": hashes,
        "held_family_mean_objectives": means,
        "held_execution_receipt_sha256s_by_arm": {arm: _sha(held_execution_receipt_sha256s_by_arm[arm], f"{arm} execution") for arm in _ARMS},
        "held_trace_evidence_sha256": _sha(held_trace_evidence_sha256, "held trace evidence"),
        "response_gain_min": gain_min,
        "response_gain_max": gain_max,
        "response_gain_distinct_count": response_gain_distinct_count,
        "response_finite": finite,
        "response_bounded": bounded,
        "response_nonconstant": nonconstant,
        "pointwise_trust_passed": pointwise_trust_passed,
        "exact_execution": exact_execution,
        "soft_response_health_passed": health,
        "data_boundary": dict(_DATA_BOUNDARY),
    }
    result = _finish("fisher_graph.complete_h4_soft_polarity_trust_region_fold.v20g", _FOLD_DOMAIN, payload)
    validate_soft_polarity_fold_receipt(result, direction_receipt=direction_receipt, candidate_receipts=candidates)
    return result


def validate_soft_polarity_fold_receipt(value: Mapping[str, object], *, direction_receipt: Mapping[str, object], candidate_receipts: Sequence[Mapping[str, object]]) -> None:
    validate_soft_polarity_direction_receipt(direction_receipt)
    receipt = _mapping(value, "fold receipt")
    candidates = tuple(candidate_receipts)
    by_alpha = _validated_candidate_ladder(
        candidates, direction_receipt=direction_receipt
    )
    selected = min(by_alpha.values(), key=lambda item: (float(item["family_equal_train_objective"]), float(item["alpha"]), str(item["artifact_sha256"])))
    expected_candidate_artifacts = {
        str(alpha): by_alpha[alpha]["artifact_sha256"]
        for alpha in SOFT_POLARITY_FIT_ALPHAS
    }
    expected = {
        "schema": "fisher_graph.complete_h4_soft_polarity_trust_region_fold.v20g",
        "protocol_sha256": SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "source_artifact_sha256s": direction_receipt["source_artifact_sha256s"],
        "all_development_family_ids": direction_receipt["all_development_family_ids"],
        "held_family_id": direction_receipt["held_family_id"],
        "direction_artifact_sha256": direction_receipt["artifact_sha256"],
        "candidate_artifact_sha256s_by_alpha": expected_candidate_artifacts,
        "selected_alpha": selected["alpha"],
        "selected_eta": selected["eta"],
        "selected_train_objective": selected["family_equal_train_objective"],
        "selected_candidate_artifact_sha256": selected["artifact_sha256"],
        "selection_frozen_before_held_scores": True,
        "data_boundary": _DATA_BOUNDARY,
    }
    for key, item in expected.items():
        if _canonical(receipt.get(key)) != _canonical(item):
            raise ValueError(f"fold {key} drifted")
    means = _mapping(receipt.get("held_family_mean_objectives"), "held means")
    hashes = _mapping(receipt.get("held_objective_sha256s_by_arm"), "held objective hashes")
    executions = _mapping(receipt.get("held_execution_receipt_sha256s_by_arm"), "held executions")
    if set(means) != set(_ARMS) or set(hashes) != set(_ARMS) or set(executions) != set(_ARMS):
        raise ValueError("fold arm geometry drifted")
    held_ids = tuple(
        _identifier(item, "held example id")
        for item in _sequence(receipt.get("held_example_ids"), "held example ids")
    )
    if not held_ids or len(set(held_ids)) != len(held_ids) or held_ids != tuple(sorted(held_ids)):
        raise ValueError("fold held example geometry drifted")
    training_ids = {
        _identifier(item, "training example id")
        for raw in _mapping(
            direction_receipt.get("training_example_ids_by_family"),
            "direction training ids",
        ).values()
        for item in _sequence(raw, "direction training ids")
    }
    if training_ids.intersection(held_ids):
        raise ValueError("outer held examples leaked into direction training")
    for arm in _ARMS:
        _number(means[arm], f"{arm} mean")
        arm_hashes = _sequence(hashes[arm], f"{arm} hashes")
        if len(arm_hashes) != len(held_ids):
            raise ValueError("fold held objective commitment geometry drifted")
        for item in arm_hashes:
            _sha(item, f"{arm} objective hash")
        _sha(executions[arm], f"{arm} execution")
    gain_min = _number(receipt.get("response_gain_min"), "gain minimum")
    gain_max = _number(receipt.get("response_gain_max"), "gain maximum")
    distinct = receipt.get("response_gain_distinct_count")
    if type(distinct) is not int or distinct < 0:
        raise ValueError("fold gain distinct count drifted")
    bounded = -1.0 <= gain_min <= gain_max <= 1.0
    nonconstant = distinct >= 2 and gain_min < gain_max
    finite = _boolean(receipt.get("response_finite"), "response finite")
    trust = _boolean(receipt.get("pointwise_trust_passed"), "trust")
    exact = _boolean(receipt.get("exact_execution"), "exact execution")
    health = bool(selected["execution_changed_from_base"]) and finite and trust and exact and bounded and nonconstant
    if receipt.get("response_bounded") is not bounded or receipt.get("response_nonconstant") is not nonconstant or receipt.get("soft_response_health_passed") is not health:
        raise ValueError("fold response health drifted")
    _sha(receipt.get("held_trace_evidence_sha256"), "held trace evidence")
    artifact = _sha(receipt.get("artifact_sha256"), "fold artifact")
    if artifact != _hash(_FOLD_DOMAIN, {key: item for key, item in receipt.items() if key != "artifact_sha256"}):
        raise ValueError("fold receipt hash drifted")


def _authenticated_oof_fold_lineage(
    folds: Sequence[Mapping[str, object]],
    *,
    families: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    sources_by_family: dict[str, dict[str, str]] = {}
    source_names: frozenset[str] | None = None
    for fold in folds:
        held = _identifier(fold.get("held_family_id"), "OOF held family")
        if (
            fold.get("schema")
            != "fisher_graph.complete_h4_soft_polarity_trust_region_fold.v20g"
            or fold.get("protocol_sha256")
            != SOFT_POLARITY_FIT_PROTOCOL_SHA256
            or tuple(fold.get("all_development_family_ids", ())) != families
        ):
            raise ValueError("OOF fold lineage differs")
        sources = _sources(fold.get("source_artifact_sha256s"))
        observed_names = frozenset(sources)
        if source_names is None:
            source_names = observed_names
        elif observed_names != source_names:
            raise ValueError("OOF fold source artifact names differ")
        sources_by_family[held] = sources
        artifact = _sha(fold.get("artifact_sha256"), "fold artifact")
        if artifact != _hash(
            _FOLD_DOMAIN,
            {key: item for key, item in fold.items() if key != "artifact_sha256"},
        ):
            raise ValueError("OOF fold receipt hash drifted")
    return {family: sources_by_family[family] for family in families}


def build_soft_polarity_oof_qualification(*, fold_receipts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate eight development folds and authorize only the next boundary."""

    folds = tuple(fold_receipts)
    if len(folds) != _FAMILY_COUNT:
        raise ValueError("OOF qualification requires exactly eight folds")
    ordered = tuple(sorted(folds, key=lambda item: str(item.get("held_family_id"))))
    families = _families(ordered[0].get("all_development_family_ids"))
    if tuple(str(item.get("held_family_id")) for item in ordered) != families:
        raise ValueError("OOF folds must hold each development family exactly once")
    sources_by_family = _authenticated_oof_fold_lineage(
        ordered, families=families
    )
    per_family = {str(fold["held_family_id"]): dict(_mapping(fold["held_family_mean_objectives"], "held means")) for fold in ordered}
    macros = {arm: _mean([_number(per_family[family][arm], f"{arm} objective") for family in families]) for arm in _ARMS}
    base_wins = sum(per_family[family]["soft_router"] < per_family[family]["base"] for family in families)
    plus_wins = sum(per_family[family]["soft_router"] < per_family[family]["fixed_plus"] for family in families)
    minus_wins = sum(per_family[family]["soft_router"] < per_family[family]["fixed_minus"] for family in families)
    gates = {
        "all_eight_fold_health_passed": all(fold.get("soft_response_health_passed") is True for fold in ordered),
        "soft_macro_beats_base": macros["soft_router"] < macros["base"],
        "soft_macro_beats_fixed_plus": macros["soft_router"] < macros["fixed_plus"],
        "soft_beats_base_in_at_least_six_families": base_wins >= 6,
        "soft_beats_fixed_plus_in_at_least_six_families": plus_wins >= 6,
    }
    authorized = all(gates.values())
    payload = {
        "protocol_sha256": SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "source_artifact_sha256s_by_family": sources_by_family,
        "development_family_ids": families,
        "fold_artifact_sha256s_by_family": {str(fold["held_family_id"]): fold["artifact_sha256"] for fold in ordered},
        "selected_alphas_by_family": {str(fold["held_family_id"]): fold["selected_alpha"] for fold in ordered},
        "family_mean_objectives_by_family": per_family,
        "family_equal_macro_objectives": macros,
        "soft_vs_base_family_win_count": base_wins,
        "soft_vs_fixed_plus_family_win_count": plus_wins,
        "soft_vs_fixed_minus_family_win_count": minus_wins,
        "gates": gates,
        "passed": authorized,
        "full_refit_authorized": authorized,
        "calibration_b_eligibility_gate_passed": authorized,
        "calibration_b_eligible": False,
        "rollback_to_base": not authorized,
        "calibration_b_opened": False,
        "fresh_family_disjoint_scoring_performed": False,
        "data_boundary": dict(_DATA_BOUNDARY),
    }
    result = _finish("fisher_graph.complete_h4_soft_polarity_trust_region_oof.v20g", _OOF_DOMAIN, payload)
    validate_soft_polarity_oof_qualification(result, fold_receipts=ordered)
    return result


def validate_soft_polarity_oof_qualification(value: Mapping[str, object], *, fold_receipts: Sequence[Mapping[str, object]]) -> None:
    receipt = _mapping(value, "OOF receipt")
    folds = tuple(sorted(fold_receipts, key=lambda item: str(item.get("held_family_id"))))
    # Rebuild from already-hashed folds without calling the public builder.
    families = _families(receipt.get("development_family_ids"))
    if len(folds) != 8 or tuple(str(item.get("held_family_id")) for item in folds) != families:
        raise ValueError("OOF validator fold geometry differs")
    sources_by_family = _authenticated_oof_fold_lineage(
        folds, families=families
    )
    per_family = {str(fold["held_family_id"]): dict(_mapping(fold["held_family_mean_objectives"], "held means")) for fold in folds}
    macros = {arm: _mean([_number(per_family[family][arm], f"{arm} objective") for family in families]) for arm in _ARMS}
    base = sum(per_family[f]["soft_router"] < per_family[f]["base"] for f in families)
    plus = sum(per_family[f]["soft_router"] < per_family[f]["fixed_plus"] for f in families)
    minus = sum(per_family[f]["soft_router"] < per_family[f]["fixed_minus"] for f in families)
    gates = {
        "all_eight_fold_health_passed": all(fold.get("soft_response_health_passed") is True for fold in folds),
        "soft_macro_beats_base": macros["soft_router"] < macros["base"],
        "soft_macro_beats_fixed_plus": macros["soft_router"] < macros["fixed_plus"],
        "soft_beats_base_in_at_least_six_families": base >= 6,
        "soft_beats_fixed_plus_in_at_least_six_families": plus >= 6,
    }
    authorized = all(gates.values())
    expected = {
        "schema": "fisher_graph.complete_h4_soft_polarity_trust_region_oof.v20g",
        "protocol_sha256": SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "source_artifact_sha256s_by_family": sources_by_family,
        "development_family_ids": families,
        "fold_artifact_sha256s_by_family": {
            str(fold["held_family_id"]): fold["artifact_sha256"]
            for fold in folds
        },
        "selected_alphas_by_family": {
            str(fold["held_family_id"]): fold["selected_alpha"]
            for fold in folds
        },
        "family_mean_objectives_by_family": per_family,
        "family_equal_macro_objectives": macros,
        "soft_vs_base_family_win_count": base,
        "soft_vs_fixed_plus_family_win_count": plus,
        "soft_vs_fixed_minus_family_win_count": minus,
        "gates": gates,
        "passed": authorized,
        "full_refit_authorized": authorized,
        "calibration_b_eligibility_gate_passed": authorized,
        "calibration_b_eligible": False,
        "rollback_to_base": not authorized,
        "calibration_b_opened": False,
        "fresh_family_disjoint_scoring_performed": False,
        "data_boundary": _DATA_BOUNDARY,
    }
    for key, item in expected.items():
        if _canonical(receipt.get(key)) != _canonical(item):
            raise ValueError(f"OOF {key} drifted")
    artifact = _sha(receipt.get("artifact_sha256"), "OOF artifact")
    if artifact != _hash(_OOF_DOMAIN, {key: item for key, item in receipt.items() if key != "artifact_sha256"}):
        raise ValueError("OOF receipt hash drifted")


def soft_polarity_work_accounting(
    *,
    direction_receipts: Sequence[Mapping[str, object]],
    candidate_receipts: Sequence[Mapping[str, object]],
    fold_receipts: Sequence[Mapping[str, object]],
    full_refit_performed: bool = False,
) -> dict[str, object]:
    """Return exact logical development work; protected-role work is zero."""

    directions = tuple(direction_receipts)
    candidates = tuple(candidate_receipts)
    folds = tuple(fold_receipts)
    full_refit = _boolean(full_refit_performed, "full refit performed")
    if len(directions) != 8 or len(candidates) != 8 * len(SOFT_POLARITY_FIT_ALPHAS) or len(folds) != 8:
        raise ValueError("work accounting requires the complete eight-fold campaign")
    direction_by_artifact: dict[str, Mapping[str, object]] = {}
    direction_by_held: dict[str, Mapping[str, object]] = {}
    campaign_families: tuple[str, ...] | None = None
    gradient_geometry_by_family: dict[str, bytes] = {}
    unique_rows: set[tuple[str, str]] = set()
    logical_rows = 0
    for direction in directions:
        validate_soft_polarity_direction_receipt(direction)
        artifact = _sha(direction.get("artifact_sha256"), "direction artifact")
        held = _identifier(direction.get("held_family_id"), "held family")
        families = _families(direction.get("all_development_family_ids"))
        if campaign_families is None:
            campaign_families = families
        elif families != campaign_families:
            raise ValueError("work accounting direction panels differ")
        if artifact in direction_by_artifact or held in direction_by_held:
            raise ValueError("work accounting directions are duplicated")
        direction_by_artifact[artifact] = direction
        direction_by_held[held] = direction
        ids = _mapping(direction["training_example_ids_by_family"], "training ids")
        for family, raw in ids.items():
            examples = tuple(_sequence(raw, "training ids"))
            geometry = _canonical(examples)
            previous = gradient_geometry_by_family.setdefault(str(family), geometry)
            if previous != geometry:
                raise ValueError("work accounting gradient row geometry differs by fold")
            logical_rows += len(examples)
            unique_rows.update((str(family), str(example)) for example in examples)
    if campaign_families is None or set(direction_by_held) != set(campaign_families):
        raise ValueError("work accounting must hold every development family once")

    candidates_by_direction: dict[str, list[Mapping[str, object]]] = {
        artifact: [] for artifact in direction_by_artifact
    }
    objective_geometry_by_family: dict[str, bytes] = {}
    unique_objective_rows: set[tuple[str, str]] = set()
    train_scores = 0
    for candidate in candidates:
        direction_artifact = _sha(
            candidate.get("direction_artifact_sha256"),
            "candidate direction artifact",
        )
        if direction_artifact not in direction_by_artifact:
            raise ValueError("work accounting candidate direction is unknown")
        direction = direction_by_artifact[direction_artifact]
        validate_soft_polarity_candidate_receipt(
            candidate, direction_receipt=direction
        )
        candidates_by_direction[direction_artifact].append(candidate)
        ids = _mapping(candidate.get("training_example_ids_by_family"), "candidate ids")
        for family, raw in ids.items():
            examples = tuple(_sequence(raw, "candidate ids"))
            geometry = _canonical(examples)
            previous = objective_geometry_by_family.setdefault(str(family), geometry)
            if previous != geometry:
                raise ValueError("work accounting candidate row geometry differs by fold")
            train_scores += len(examples)
            unique_objective_rows.update(
                (str(family), str(example)) for example in examples
            )
    for artifact, direction in direction_by_artifact.items():
        _validated_candidate_ladder(
            candidates_by_direction[artifact], direction_receipt=direction
        )

    seen_fold_families: set[str] = set()
    held_scores = 0
    for fold in folds:
        held = _identifier(fold.get("held_family_id"), "fold held family")
        if held in seen_fold_families or held not in direction_by_held:
            raise ValueError("work accounting fold families are duplicated or unknown")
        seen_fold_families.add(held)
        direction = direction_by_held[held]
        direction_artifact = str(direction["artifact_sha256"])
        _core_candidates = candidates_by_direction[direction_artifact]
        validate_soft_polarity_fold_receipt(
            fold,
            direction_receipt=direction,
            candidate_receipts=_core_candidates,
        )
        held_scores += len(
            _sequence(fold.get("held_example_ids"), "held ids")
        ) * len(_ARMS)
    if seen_fold_families != set(campaign_families):
        raise ValueError("work accounting must score every held family once")
    full_refit_rows = len(unique_rows) if full_refit else 0
    full_refit_train_scores = (
        len(unique_objective_rows) * len(SOFT_POLARITY_FIT_ALPHAS)
        if full_refit
        else 0
    )
    return {
        "schema": "fisher_graph.complete_h4_soft_polarity_trust_region_work.v20g",
        "development_family_count": 8,
        "outer_fold_count": 8,
        "alpha_count": len(SOFT_POLARITY_FIT_ALPHAS),
        "trust_radius_count": len(SOFT_POLARITY_FIT_ALPHAS),
        "natural_direction_solve_count": 8 + int(full_refit),
        "box_direction_normalization_count": 8 + int(full_refit),
        "box_direction_corner_evaluation_count": 4 * (8 + int(full_refit)),
        "outer_candidate_trust_certificate_count": len(candidates),
        "full_refit_candidate_trust_certificate_count": (
            len(SOFT_POLARITY_FIT_ALPHAS) if full_refit else 0
        ),
        "candidate_trust_certificate_count": (
            len(candidates)
            + (len(SOFT_POLARITY_FIT_ALPHAS) if full_refit else 0)
        ),
        "positive_candidate_trust_certificate_count": (
            (len(SOFT_POLARITY_FIT_ALPHAS) - 1) * (8 + int(full_refit))
        ),
        "outer_fold_direction_solve_count": 8,
        "full_refit_direction_solve_count": int(full_refit),
        "unique_development_gradient_row_count": len(unique_rows),
        "unique_development_candidate_objective_row_count": len(
            unique_objective_rows
        ),
        "logical_fold_gradient_row_use_count": logical_rows,
        "logical_full_refit_gradient_row_use_count": full_refit_rows,
        "logical_total_gradient_row_use_count": logical_rows + full_refit_rows,
        "family_opg_outer_product_evaluation_count": logical_rows + full_refit_rows,
        "outer_fold_training_candidate_objective_score_count": train_scores,
        "full_refit_training_candidate_objective_score_count": full_refit_train_scores,
        "exact_training_candidate_objective_score_count": (
            train_scores + full_refit_train_scores
        ),
        "exact_outer_development_arm_score_count": held_scores,
        "calibration_b_example_access_count": 0,
        "validation_example_access_count": 0,
        "test_example_access_count": 0,
        "fresh_family_disjoint_score_count": 0,
        "data_boundary": dict(_DATA_BOUNDARY),
    }
