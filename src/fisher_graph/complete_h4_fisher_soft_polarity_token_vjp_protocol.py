"""Frozen tensor-free policy for the V20q token-VJP compiler rung.

The numerical compiler is intentionally outside this module.  This layer only
accepts scalar/hash metadata produced by the post-cast H4 secant and token-VJP
fit authorities, freezes the finite candidate library, enforces six/one
family-disjoint selection, and emits an all-seven final-refit specification.

Neither model tensors nor an outer-family objective are accepted by any
public function.  The selected coefficients still execute through the
unchanged V20p local signed-field provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
import struct


__all__ = [
    "SOFT_POLARITY_TOKEN_VJP_ALPHA_LADDER",
    "SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP",
    "SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS",
    "SOFT_POLARITY_TOKEN_VJP_CANDIDATE_LIBRARY",
    "SOFT_POLARITY_TOKEN_VJP_FEATURE_IDS",
    "SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS",
    "SOFT_POLARITY_TOKEN_VJP_FIT_LIBRARY",
    "SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID",
    "SOFT_POLARITY_TOKEN_VJP_MAXIMUM_NORM_RATIO",
    "SOFT_POLARITY_TOKEN_VJP_MINIMUM_COSINE",
    "SOFT_POLARITY_TOKEN_VJP_MINIMUM_NORM_RATIO",
    "SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP",
    "SOFT_POLARITY_TOKEN_VJP_PROTOCOL_SHA256",
    "SOFT_POLARITY_TOKEN_VJP_RIDGE_LADDER",
    "SOFT_POLARITY_TOKEN_VJP_SEED_A",
    "SOFT_POLARITY_TOKEN_VJP_SEED_SIGNS",
    "build_soft_polarity_token_vjp_all_seven_refit_receipt",
    "build_soft_polarity_token_vjp_candidate_receipt",
    "build_soft_polarity_token_vjp_inner_oof_selection_receipt",
    "build_soft_polarity_token_vjp_natural_direction_output",
    "build_soft_polarity_token_vjp_protocol_receipt",
    "build_soft_polarity_token_vjp_scalar_fit_output",
    "validate_soft_polarity_token_vjp_all_seven_refit_receipt",
    "validate_soft_polarity_token_vjp_candidate_receipt",
    "validate_soft_polarity_token_vjp_inner_oof_selection_receipt",
    "validate_soft_polarity_token_vjp_protocol_receipt",
]


SOFT_POLARITY_TOKEN_VJP_FEATURE_IDS = (
    "c1",
    "c2",
    "c1_times_c2",
    "source_z",
)
SOFT_POLARITY_TOKEN_VJP_SEED_SIGNS = (-1, 1)
SOFT_POLARITY_TOKEN_VJP_SEED_A = 0.0
SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP = 2.0**-6
SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP = 2.0**-7
SOFT_POLARITY_TOKEN_VJP_RIDGE_LADDER = (0.1, 1.0, 10.0)
SOFT_POLARITY_TOKEN_VJP_ALPHA_LADDER = (
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
)
SOFT_POLARITY_TOKEN_VJP_MINIMUM_COSINE = 0.99
SOFT_POLARITY_TOKEN_VJP_MINIMUM_NORM_RATIO = 0.80
SOFT_POLARITY_TOKEN_VJP_MAXIMUM_NORM_RATIO = 1.25

SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID = "v20q_v20p_incumbent"
_ANCHOR_IDS = (
    "v20q_anchor_zero",
    "v20q_anchor_minus",
    "v20q_anchor_plus",
)
_SEED_IDS = ("v20q_seed_minus", "v20q_seed_plus")


def _fit_candidate_id(
    feature_index: int,
    sign_index: int,
    ridge_index: int,
    alpha_index: int,
) -> str:
    return (
        f"v20q_fit_f{feature_index}_s{sign_index}_"
        f"r{ridge_index}_a{alpha_index}"
    )


SOFT_POLARITY_TOKEN_VJP_FIT_LIBRARY = tuple(
    (
        _fit_candidate_id(
            feature_index, sign_index, ridge_index, alpha_index
        ),
        feature_id,
        sign,
        float(sign) * 0.5,
        SOFT_POLARITY_TOKEN_VJP_SEED_A,
        ridge,
        alpha,
    )
    for feature_index, feature_id in enumerate(
        SOFT_POLARITY_TOKEN_VJP_FEATURE_IDS
    )
    for sign_index, sign in enumerate(SOFT_POLARITY_TOKEN_VJP_SEED_SIGNS)
    for ridge_index, ridge in enumerate(SOFT_POLARITY_TOKEN_VJP_RIDGE_LADDER)
    for alpha_index, alpha in enumerate(SOFT_POLARITY_TOKEN_VJP_ALPHA_LADDER)
)
SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS = tuple(
    item[0] for item in SOFT_POLARITY_TOKEN_VJP_FIT_LIBRARY
)
SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS = (
    SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID,
    *_ANCHOR_IDS,
    *_SEED_IDS,
    *SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS,
)


def _library_rows() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = [
        {
            "candidate_id": SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID,
            "role": "v20p_incumbent",
            "feature_id": None,
            "seed_sign": None,
            "seed_b": None,
            "seed_a": None,
            "ridge": None,
            "alpha": None,
            "fixed_b": None,
            "fixed_a": None,
        },
        {
            "candidate_id": "v20q_anchor_zero",
            "role": "exact_anchor",
            "feature_id": "source_z",
            "seed_sign": None,
            "seed_b": None,
            "seed_a": None,
            "ridge": None,
            "alpha": None,
            "fixed_b": 0.0,
            "fixed_a": 0.0,
        },
        {
            "candidate_id": "v20q_anchor_minus",
            "role": "exact_anchor",
            "feature_id": "source_z",
            "seed_sign": None,
            "seed_b": None,
            "seed_a": None,
            "ridge": None,
            "alpha": None,
            "fixed_b": -1.0,
            "fixed_a": 0.0,
        },
        {
            "candidate_id": "v20q_anchor_plus",
            "role": "exact_anchor",
            "feature_id": "source_z",
            "seed_sign": None,
            "seed_b": None,
            "seed_a": None,
            "ridge": None,
            "alpha": None,
            "fixed_b": 1.0,
            "fixed_a": 0.0,
        },
        {
            "candidate_id": "v20q_seed_minus",
            "role": "smooth_seed",
            "feature_id": "source_z",
            "seed_sign": -1,
            "seed_b": -0.5,
            "seed_a": 0.0,
            "ridge": None,
            "alpha": 0.0,
            "fixed_b": -0.5,
            "fixed_a": 0.0,
        },
        {
            "candidate_id": "v20q_seed_plus",
            "role": "smooth_seed",
            "feature_id": "source_z",
            "seed_sign": 1,
            "seed_b": 0.5,
            "seed_a": 0.0,
            "ridge": None,
            "alpha": 0.0,
            "fixed_b": 0.5,
            "fixed_a": 0.0,
        },
    ]
    rows.extend(
        {
            "candidate_id": candidate_id,
            "role": "token_vjp_fit",
            "feature_id": feature_id,
            "seed_sign": sign,
            "seed_b": seed_b,
            "seed_a": seed_a,
            "ridge": ridge,
            "alpha": alpha,
            "fixed_b": None,
            "fixed_a": None,
        }
        for (
            candidate_id,
            feature_id,
            sign,
            seed_b,
            seed_a,
            ridge,
            alpha,
        ) in SOFT_POLARITY_TOKEN_VJP_FIT_LIBRARY
    )
    return tuple(rows)


SOFT_POLARITY_TOKEN_VJP_CANDIDATE_LIBRARY = tuple(
    (
        row["candidate_id"],
        row["role"],
        row["feature_id"],
        row["seed_sign"],
        row["seed_b"],
        row["seed_a"],
        row["ridge"],
        row["alpha"],
        row["fixed_b"],
        row["fixed_a"],
    )
    for row in _library_rows()
)
_SPEC_BY_ID = {row["candidate_id"]: row for row in _library_rows()}

_SHA = re.compile(r"^[0-9a-f]{64}$")
_PROTOCOL_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-protocol:v20q\0"
_CANDIDATE_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-candidate:v20q\0"
_SELECTION_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-selection:v20q\0"
_FINAL_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-final-refit:v20q\0"
_SCALAR_FIT_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-scalar-fit:v20q\0"
_FIT_TENSOR_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-tensor:v1\0"

_DATA_BOUNDARY = {
    "role": "v20q_training_only_token_vjp_policy",
    "development_family_count": 8,
    "inner_oof_training_family_count": 6,
    "inner_oof_held_family_count": 1,
    "all_seven_final_refit_family_count": 7,
    "candidate_library_frozen_before_any_exact_kl_objective": True,
    "inner_candidate_frozen_before_inner_held_objective": True,
    "outer_held_family_absent_from_fit_and_selection": True,
    "outer_held_objectives_consumed": False,
    "prompt_text_consumed": False,
    "raw_logits_consumed": False,
    "raw_h4_consumed": False,
    "raw_gradients_consumed": False,
    "raw_model_tensors_serialized": False,
    "token_teacher_kl_vjps_are_compiler_only": True,
    "runtime_provider": "unchanged_v20p_local_signed_field_provider",
    "calibration_b_opened": False,
    "compression_claim_authorized": False,
    "speed_claim_authorized": False,
    "serving_authorized": False,
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
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
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


def _float(value: object, label: str, *, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    result = 0.0 if value == 0.0 else value
    if nonnegative and result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be bool")
    return value


def _family_ids(value: object, label: str, *, count: int) -> tuple[str, ...]:
    result = tuple(_identifier(item, f"{label} entry") for item in _sequence(value, label))
    if len(result) != count or result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must contain {count} sorted unique IDs")
    return result


def _check_hash(receipt: Mapping[str, object], domain: bytes, label: str) -> None:
    artifact = _sha(receipt.get("artifact_sha256"), f"{label} artifact")
    payload = dict(receipt)
    payload.pop("artifact_sha256", None)
    if _hash(domain, payload) != artifact:
        raise ValueError(f"{label} artifact hash differs")


def _same(actual: object, expected: object, label: str) -> None:
    if _canonical(actual) != _canonical(expected):
        raise ValueError(f"{label} content differs")


def _scalar_tree(value: object, label: str) -> None:
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _identifier(key, f"{label} key")
            _scalar_tree(item, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _scalar_tree(item, f"{label}[{index}]")
        return
    raise TypeError(f"{label} may contain only scalar/hash metadata")


_PROTOCOL = {
    "protocol": "v20q_token_vjp_continuous_local_field_nested_oof",
    "scientific_status": "development_only_after_v20p",
    "feature_order": SOFT_POLARITY_TOKEN_VJP_FEATURE_IDS,
    "seed_sign_order": SOFT_POLARITY_TOKEN_VJP_SEED_SIGNS,
    "seed_b_by_sign": (-0.5, 0.5),
    "seed_a": SOFT_POLARITY_TOKEN_VJP_SEED_A,
    "primary_secant_half_step": SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
    "audit_secant_half_step": SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
    "secant_method": "central_finite_secant_through_post_cast_h4",
    "secants_not_claimed_as": "analytic_jacobian_at_abs_or_clamp_kink",
    "ridge_order": SOFT_POLARITY_TOKEN_VJP_RIDGE_LADDER,
    "alpha_order": SOFT_POLARITY_TOKEN_VJP_ALPHA_LADDER,
    "minimum_secant_cosine": SOFT_POLARITY_TOKEN_VJP_MINIMUM_COSINE,
    "minimum_secant_norm_ratio": SOFT_POLARITY_TOKEN_VJP_MINIMUM_NORM_RATIO,
    "maximum_secant_norm_ratio": SOFT_POLARITY_TOKEN_VJP_MAXIMUM_NORM_RATIO,
    "fit_candidate_count": len(SOFT_POLARITY_TOKEN_VJP_FIT_CANDIDATE_IDS),
    "control_candidate_count": 6,
    "candidate_count": len(SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS),
    "candidate_order": SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS,
    "candidate_library": _library_rows(),
    "inner_fit": "six_families_fit_one_family_exact_kl_score",
    "inner_fold_count": 7,
    "selection": "paired_to_incumbent_one_standard_error",
    "one_standard_error_reference": "exact_mean_kl_best_candidate_paired_delta_to_v20p_incumbent",
    "conservative_order": (
        "v20p_incumbent",
        "exact_anchor_before_smooth_seed_before_fit",
        "smaller_alpha",
        "larger_ridge",
        "smaller_inner_mean_abs_a",
        "smaller_inner_mean_abs_b",
        "fixed_feature_order",
        "negative_seed_before_positive_seed",
        "fixed_candidate_index",
        "candidate_receipt_set_sha256",
    ),
    "final_refit": "selected_index_refit_on_all_seven_outer_training_families",
    "runtime": "unchanged_v20p_local_signed_field_provider",
    "data_boundary": _DATA_BOUNDARY,
}
SOFT_POLARITY_TOKEN_VJP_PROTOCOL_SHA256 = _hash(_PROTOCOL_DOMAIN, _PROTOCOL)


def build_soft_polarity_token_vjp_protocol_receipt() -> dict[str, object]:
    """Return the objective-free frozen V20q policy receipt."""

    return _finish(
        "fisher_graph.soft_polarity_token_vjp_protocol.v20q",
        _PROTOCOL_DOMAIN,
        {**_PROTOCOL, "protocol_sha256": SOFT_POLARITY_TOKEN_VJP_PROTOCOL_SHA256},
    )


def validate_soft_polarity_token_vjp_protocol_receipt(
    receipt: Mapping[str, object],
) -> None:
    selected = _mapping(receipt, "V20q protocol receipt")
    _check_hash(selected, _PROTOCOL_DOMAIN, "V20q protocol receipt")
    _same(selected, build_soft_polarity_token_vjp_protocol_receipt(), "V20q protocol receipt")


def _float64_tensor_sha256(
    values: Sequence[float], shape: tuple[int, ...]
) -> str:
    payload = b"".join(struct.pack("<d", value) for value in values)
    return hashlib.sha256(
        _FIT_TENSOR_DOMAIN
        + _canonical(
            {
                "dtype": "float64-little-endian",
                "shape": shape,
            }
        )
        + payload
    ).hexdigest()


def build_soft_polarity_token_vjp_natural_direction_output(
    *,
    aggregate_metadata: Mapping[str, object],
    mean_gradient: Sequence[float],
    gradient_gram: Sequence[Sequence[float]],
    ridge_multiplier: float,
) -> dict[str, object]:
    """Compute V20q's frozen 2-D natural-OPG direction without tensors.

    Callers obtain the two scalar sufficient statistics from the typed
    ``SoftPolarityTokenVJPAggregate`` and pass its ``metadata()`` alongside
    them.  Their canonical float64 hashes must match that aggregate receipt.
    """

    aggregate = _mapping(aggregate_metadata, "token-VJP aggregate metadata")
    _scalar_tree(aggregate, "token-VJP aggregate metadata")
    g = tuple(
        _float(item, "mean-gradient coordinate")
        for item in _sequence(mean_gradient, "mean gradient")
    )
    gram_rows = tuple(
        tuple(
            _float(item, "gradient-Gram coordinate")
            for item in _sequence(row, "gradient-Gram row")
        )
        for row in _sequence(gradient_gram, "gradient Gram")
    )
    if len(g) != 2 or len(gram_rows) != 2 or any(
        len(row) != 2 for row in gram_rows
    ):
        raise ValueError("V20q natural direction requires 2-D g and 2x2 F")
    if gram_rows[0][1].hex() != gram_rows[1][0].hex():
        raise ValueError("gradient Gram must be exactly symmetric")
    if (
        _float64_tensor_sha256(g, (2,))
        != _sha(
            aggregate.get("mean_parameter_gradient_sha256"),
            "aggregate mean gradient",
        )
        or _float64_tensor_sha256(
            (
                gram_rows[0][0],
                gram_rows[0][1],
                gram_rows[1][0],
                gram_rows[1][1],
            ),
            (2, 2),
        )
        != _sha(
            aggregate.get("gradient_gram_sha256"),
            "aggregate gradient Gram",
        )
    ):
        raise ValueError("scalar g/F do not replay from the aggregate receipt")
    f00, f01 = gram_rows[0]
    _, f11 = gram_rows[1]
    scale = max(1.0, abs(f00), abs(f01), abs(f11))
    psd_tolerance = math.ulp(scale) * 64.0
    if f00 < -psd_tolerance or f11 < -psd_tolerance or (
        f00 * f11 - f01 * f01
    ) < -psd_tolerance:
        raise ValueError("gradient Gram must be positive semidefinite")

    ridge = _float(
        ridge_multiplier, "ridge multiplier", nonnegative=True
    )
    if not any(
        ridge.hex() == item.hex()
        for item in SOFT_POLARITY_TOKEN_VJP_RIDGE_LADDER
    ):
        raise ValueError("ridge multiplier is outside the frozen ladder")
    trace = f00 + f11
    tau = max(trace / 2.0, 2.0**-24)
    damping = ridge * tau
    a00 = f00 + damping
    a11 = f11 + damping
    determinant = a00 * a11 - f01 * f01
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("damped natural-OPG system is not positive definite")
    raw_b = -(a11 * g[0] - f01 * g[1]) / determinant
    raw_a = -(-f01 * g[0] + a00 * g[1]) / determinant
    raw_linf = max(abs(raw_b), abs(raw_a))
    if not math.isfinite(raw_linf) or raw_linf <= 2.0**-24:
        raise ValueError("natural-OPG direction is degenerate")
    direction_b = raw_b / raw_linf
    direction_a = raw_a / raw_linf
    predicted = g[0] * direction_b + g[1] * direction_a
    if not math.isfinite(predicted) or predicted >= 0.0:
        raise ValueError("natural-OPG direction is not descending")

    payload = {
        "method": "mean_kl_natural_opg_trace_scaled_ridge_linf_direction",
        "parameter_order": ("field_bias", "field_slope"),
        "aggregate_artifact_sha256": _sha(
            aggregate.get("artifact_sha256"), "aggregate artifact"
        ),
        "feature_id": _identifier(
            aggregate.get("feature_id"), "aggregate feature_id"
        ),
        "held_family_id": _identifier(
            aggregate.get("held_family_id"), "aggregate held family"
        ),
        "reference_b": _float(
            aggregate.get("reference_b"), "aggregate reference_b"
        ),
        "reference_a": _float(
            aggregate.get("reference_a"), "aggregate reference_a"
        ),
        "ridge_multiplier": ridge,
        "gradient_gram_trace": trace,
        "tau": tau,
        "damping": damping,
        "raw_direction_b": raw_b,
        "raw_direction_a": raw_a,
        "raw_direction_linf": raw_linf,
        "direction_b": direction_b,
        "direction_a": direction_a,
        "direction_linf": max(abs(direction_b), abs(direction_a)),
        "predicted_derivative": predicted,
        "no_op": False,
        "mean_gradient_sha256": aggregate["mean_parameter_gradient_sha256"],
        "gradient_gram_sha256": aggregate["gradient_gram_sha256"],
        "raw_model_tensors_serialized": False,
    }
    return _finish(
        "fisher_graph.soft_polarity_token_vjp_natural_direction.v20q",
        _SCALAR_FIT_DOMAIN,
        payload,
    )


def build_soft_polarity_token_vjp_scalar_fit_output(
    *,
    direction_metadata: Mapping[str, object],
    aggregate_metadata: Mapping[str, object],
    primary_secant_receipt_sha256: str,
    audit_secant_receipt_sha256: str,
    secant_stability: Mapping[str, object],
) -> dict[str, object]:
    """Normalize a natural-OPG direction into a tensor-free policy input.

    The direction authority is deliberately distinct from the lower-level
    residual-GN/L2-trust fit API.  V20q requires trace-scaled ridge damping and
    one L-infinity-normalized direction that is shared by seven alpha children.
    """

    fit = _mapping(direction_metadata, "token-VJP direction metadata")
    aggregate = _mapping(aggregate_metadata, "token-VJP aggregate metadata")
    stability = _mapping(secant_stability, "post-cast secant stability")
    _scalar_tree(fit, "token-VJP direction metadata")
    _scalar_tree(aggregate, "token-VJP aggregate metadata")
    _scalar_tree(stability, "post-cast secant stability")

    if fit.get("method") != "mean_kl_natural_opg_trace_scaled_ridge_linf_direction":
        raise ValueError("V20q requires the frozen natural-OPG direction method")
    if tuple(fit.get("parameter_order", ())) != (
        "field_bias",
        "field_slope",
    ):
        raise ValueError("token-VJP direction parameter order differs")
    feature_id = _identifier(fit.get("feature_id"), "direction feature_id")
    held_family_id = _identifier(
        fit.get("held_family_id"), "direction held_family_id"
    )
    aggregate_artifact = _sha(
        aggregate.get("artifact_sha256"), "aggregate artifact"
    )
    if _sha(fit.get("aggregate_artifact_sha256"), "fit aggregate artifact") != aggregate_artifact:
        raise ValueError("direction and aggregate artifacts differ")
    if _identifier(aggregate.get("feature_id"), "aggregate feature_id") != feature_id:
        raise ValueError("direction and aggregate features differ")
    if _identifier(aggregate.get("held_family_id"), "aggregate held family") != held_family_id:
        raise ValueError("direction and aggregate held families differ")
    training_ids = tuple(
        _identifier(item, "aggregate training family")
        for item in _sequence(
            aggregate.get("training_family_ids"), "aggregate training families"
        )
    )
    if training_ids != tuple(sorted(set(training_ids))) or not training_ids:
        raise ValueError("aggregate training families must be sorted and unique")

    ridge = _float(
        fit.get("ridge_multiplier"),
        "direction ridge multiplier",
        nonnegative=True,
    )
    if not any(ridge.hex() == item.hex() for item in SOFT_POLARITY_TOKEN_VJP_RIDGE_LADDER):
        raise ValueError("direction ridge multiplier is outside the frozen ladder")
    gram_trace = _float(
        fit.get("gradient_gram_trace"),
        "direction gradient Gram trace",
        nonnegative=True,
    )
    tau = _float(fit.get("tau"), "direction tau", nonnegative=True)
    expected_tau = max(gram_trace / 2.0, 2.0**-24)
    if tau.hex() != expected_tau.hex():
        raise ValueError("direction tau differs from the frozen trace scale")
    damping = _float(fit.get("damping"), "direction damping", nonnegative=True)
    if damping.hex() != (ridge * tau).hex():
        raise ValueError("direction damping differs from ridge times tau")
    direction_b = _float(fit.get("direction_b"), "bias direction")
    direction_a = _float(fit.get("direction_a"), "slope direction")
    direction_linf = _float(
        fit.get("direction_linf"), "direction L-infinity norm", nonnegative=True
    )
    if (
        direction_linf.hex() != 1.0.hex()
        or max(abs(direction_b), abs(direction_a)).hex() != 1.0.hex()
    ):
        raise ValueError("V20q direction must be L-infinity normalized")
    predicted = _float(
        fit.get("predicted_derivative"), "direction predicted derivative"
    )
    if predicted >= 0.0 or _bool(fit.get("no_op"), "direction no_op"):
        raise ValueError("V20q direction must be non-no-op and descending")

    cosines = tuple(
        _float(item, "secant cosine")
        for item in _sequence(
            stability.get("cosine_by_parameter"), "secant cosines"
        )
    )
    ratios = tuple(
        _float(item, "secant norm ratio", nonnegative=True)
        for item in _sequence(
            stability.get("audit_to_primary_norm_ratio_by_parameter"),
            "secant norm ratios",
        )
    )
    if len(cosines) != 2 or len(ratios) != 2:
        raise ValueError("secant stability must contain bias and slope values")
    passed = all(
        value >= SOFT_POLARITY_TOKEN_VJP_MINIMUM_COSINE
        for value in cosines
    ) and all(
        SOFT_POLARITY_TOKEN_VJP_MINIMUM_NORM_RATIO
        <= value
        <= SOFT_POLARITY_TOKEN_VJP_MAXIMUM_NORM_RATIO
        for value in ratios
    )
    if _bool(stability.get("passed"), "secant stability passed") != passed:
        raise ValueError("secant stability decision differs from frozen gates")

    payload = {
        "feature_id": feature_id,
        "held_family_id": held_family_id,
        "training_family_ids": training_ids,
        "reference_b": _float(
            fit.get("reference_b"), "direction reference_b"
        ),
        "reference_a": _float(
            fit.get("reference_a"), "direction reference_a"
        ),
        "ridge": ridge,
        "gradient_gram_trace": gram_trace,
        "tau": tau,
        "damping": damping,
        "direction_b": direction_b,
        "direction_a": direction_a,
        "direction_linf": direction_linf,
        "predicted_derivative": predicted,
        "no_op": False,
        "fit_method": fit["method"],
        "parameter_order": ("field_bias", "field_slope"),
        "secant_parameter_order": ("bias", "slope"),
        "parameter_axis_mapping": (
            ("field_bias", "bias"),
            ("field_slope", "slope"),
        ),
        "mean_gradient_sha256": _sha(
            aggregate.get("mean_parameter_gradient_sha256"),
            "aggregate mean gradient",
        ),
        "gradient_gram_sha256": _sha(
            aggregate.get("gradient_gram_sha256"), "aggregate gradient Gram"
        ),
        "fit_artifact_sha256": _sha(
            fit.get("artifact_sha256"), "direction artifact"
        ),
        "aggregate_artifact_sha256": aggregate_artifact,
        "primary_secant_receipt_sha256": _sha(
            primary_secant_receipt_sha256, "primary secant receipt"
        ),
        "audit_secant_receipt_sha256": _sha(
            audit_secant_receipt_sha256, "audit secant receipt"
        ),
        "cosine_by_parameter": cosines,
        "audit_to_primary_norm_ratio_by_parameter": ratios,
        "secant_stability_passed": passed,
        "raw_fit_or_secant_tensors_serialized": False,
    }
    return _finish(
        "fisher_graph.soft_polarity_token_vjp_scalar_fit_output.v20q",
        _SCALAR_FIT_DOMAIN,
        payload,
    )


def _normalize_scalar_fit_output(value: object) -> dict[str, object]:
    selected = dict(_mapping(value, "scalar token-VJP fit output"))
    _scalar_tree(selected, "scalar token-VJP fit output")
    _check_hash(selected, _SCALAR_FIT_DOMAIN, "scalar token-VJP fit output")
    expected_keys = {
        "schema",
        "feature_id",
        "held_family_id",
        "training_family_ids",
        "reference_b",
        "reference_a",
        "ridge",
        "gradient_gram_trace",
        "tau",
        "damping",
        "direction_b",
        "direction_a",
        "direction_linf",
        "predicted_derivative",
        "no_op",
        "fit_method",
        "parameter_order",
        "secant_parameter_order",
        "parameter_axis_mapping",
        "mean_gradient_sha256",
        "gradient_gram_sha256",
        "fit_artifact_sha256",
        "aggregate_artifact_sha256",
        "primary_secant_receipt_sha256",
        "audit_secant_receipt_sha256",
        "cosine_by_parameter",
        "audit_to_primary_norm_ratio_by_parameter",
        "secant_stability_passed",
        "raw_fit_or_secant_tensors_serialized",
        "artifact_sha256",
    }
    if set(selected) != expected_keys:
        raise ValueError("scalar token-VJP fit output fields differ")
    for key in (
        "fit_artifact_sha256",
        "aggregate_artifact_sha256",
        "mean_gradient_sha256",
        "gradient_gram_sha256",
        "primary_secant_receipt_sha256",
        "audit_secant_receipt_sha256",
        "artifact_sha256",
    ):
        _sha(selected[key], key)
    if selected["schema"] != "fisher_graph.soft_polarity_token_vjp_scalar_fit_output.v20q":
        raise ValueError("scalar token-VJP fit output schema differs")
    if selected["raw_fit_or_secant_tensors_serialized"] is not False:
        raise ValueError("scalar fit output may not serialize tensors")
    return selected


def _family_split(
    all_development_family_ids: object,
    outer_held_family_id: object,
    inner_held_family_id: object | None,
) -> tuple[tuple[str, ...], str, str | None, tuple[str, ...]]:
    all_ids = _family_ids(
        all_development_family_ids, "development family IDs", count=8
    )
    outer = _identifier(outer_held_family_id, "outer held family")
    if outer not in all_ids:
        raise ValueError("outer held family is outside development families")
    if inner_held_family_id is None:
        return all_ids, outer, None, tuple(item for item in all_ids if item != outer)
    inner = _identifier(inner_held_family_id, "inner held family")
    if inner not in all_ids or inner == outer:
        raise ValueError("inner held family must be an outer-training family")
    training = tuple(item for item in all_ids if item not in (outer, inner))
    return all_ids, outer, inner, training


def _spec(candidate_id: object) -> dict[str, object]:
    selected = _identifier(candidate_id, "candidate_id")
    if selected not in _SPEC_BY_ID:
        raise ValueError("candidate_id is outside the frozen V20q library")
    return dict(_SPEC_BY_ID[selected])


def build_soft_polarity_token_vjp_candidate_receipt(
    *,
    protocol_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    inner_held_family_id: str,
    candidate_id: str,
    candidate_provider_sha256: str,
    scalar_fit_output: Mapping[str, object] | None = None,
    incumbent_feature_id: str | None = None,
    incumbent_b: float | None = None,
    incumbent_a: float | None = None,
    incumbent_fit_receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Freeze one candidate before its inner-held exact KL is available."""

    validate_soft_polarity_token_vjp_protocol_receipt(protocol_receipt)
    all_ids, outer, inner, training_ids = _family_split(
        all_development_family_ids,
        outer_held_family_id,
        inner_held_family_id,
    )
    assert inner is not None
    spec = _spec(candidate_id)
    provider_hash = _sha(candidate_provider_sha256, "candidate provider")
    role = spec["role"]
    fit_output: dict[str, object] | None = None
    incumbent_source: str | None = None

    if role == "token_vjp_fit":
        if any(
            value is not None
            for value in (
                incumbent_feature_id,
                incumbent_b,
                incumbent_a,
                incumbent_fit_receipt_sha256,
            )
        ):
            raise ValueError("fitted candidates may not consume incumbent fields")
        fit_output = _normalize_scalar_fit_output(scalar_fit_output)
        if (
            fit_output["feature_id"] != spec["feature_id"]
            or fit_output["held_family_id"] != inner
            or tuple(fit_output["training_family_ids"]) != training_ids
            or float(fit_output["reference_b"]).hex()
            != float(spec["seed_b"]).hex()
            or float(fit_output["reference_a"]).hex()
            != float(spec["seed_a"]).hex()
            or float(fit_output["ridge"]).hex()
            != float(spec["ridge"]).hex()
        ):
            raise ValueError("scalar fit output differs from candidate index")
        if (
            fit_output["no_op"] is not False
            or fit_output["secant_stability_passed"] is not True
            or float(fit_output["predicted_derivative"]) >= 0.0
        ):
            raise ValueError("fitted candidate is not stable and descending")
        feature_id = str(fit_output["feature_id"])
        alpha = float(spec["alpha"])
        b = float(fit_output["reference_b"]) + alpha * float(
            fit_output["direction_b"]
        )
        a = float(fit_output["reference_a"]) + alpha * float(
            fit_output["direction_a"]
        )
    elif role == "v20p_incumbent":
        if scalar_fit_output is not None:
            raise ValueError("incumbent may not consume a token-VJP fit")
        feature_id = _identifier(incumbent_feature_id, "incumbent feature")
        if feature_id not in SOFT_POLARITY_TOKEN_VJP_FEATURE_IDS:
            raise ValueError("incumbent feature is outside the V20p field library")
        b = _float(incumbent_b, "incumbent b")
        a = _float(incumbent_a, "incumbent a")
        incumbent_source = _sha(
            incumbent_fit_receipt_sha256, "V20p incumbent fit receipt"
        )
    else:
        if scalar_fit_output is not None or any(
            value is not None
            for value in (
                incumbent_feature_id,
                incumbent_b,
                incumbent_a,
                incumbent_fit_receipt_sha256,
            )
        ):
            raise ValueError("fixed controls may not consume fit fields")
        feature_id = str(spec["feature_id"])
        b = float(spec["fixed_b"])
        a = float(spec["fixed_a"])

    payload = {
        "protocol_sha256": protocol_receipt["artifact_sha256"],
        "candidate_id": spec["candidate_id"],
        "candidate_index": SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS.index(
            str(spec["candidate_id"])
        ),
        "role": role,
        "candidate_spec": spec,
        "all_development_family_ids": all_ids,
        "outer_held_family_id": outer,
        "inner_held_family_id": inner,
        "training_family_ids": training_ids,
        "training_family_count": 6,
        "feature_id": feature_id,
        "b": b,
        "b_hex": b.hex(),
        "a": a,
        "a_hex": a.hex(),
        "candidate_provider_sha256": provider_hash,
        "scalar_fit_output": fit_output,
        "incumbent_fit_receipt_sha256": incumbent_source,
        "candidate_frozen_before_inner_held_objective": True,
        "inner_held_objective_consumed": False,
        "outer_held_objective_consumed": False,
        "raw_model_tensors_serialized": False,
        "runtime_provider": "unchanged_v20p_local_signed_field_provider",
        "data_boundary": dict(_DATA_BOUNDARY),
    }
    return _finish(
        "fisher_graph.soft_polarity_token_vjp_candidate.v20q",
        _CANDIDATE_DOMAIN,
        payload,
    )


def validate_soft_polarity_token_vjp_candidate_receipt(
    receipt: Mapping[str, object],
    *,
    protocol_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    inner_held_family_id: str,
) -> None:
    selected = _mapping(receipt, "V20q candidate receipt")
    _check_hash(selected, _CANDIDATE_DOMAIN, "V20q candidate receipt")
    role = selected.get("role")
    kwargs: dict[str, object] = {}
    if role == "token_vjp_fit":
        kwargs["scalar_fit_output"] = selected.get("scalar_fit_output")
    elif role == "v20p_incumbent":
        kwargs.update(
            incumbent_feature_id=selected.get("feature_id"),
            incumbent_b=selected.get("b"),
            incumbent_a=selected.get("a"),
            incumbent_fit_receipt_sha256=selected.get(
                "incumbent_fit_receipt_sha256"
            ),
        )
    rebuilt = build_soft_polarity_token_vjp_candidate_receipt(
        protocol_receipt=protocol_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        inner_held_family_id=inner_held_family_id,
        candidate_id=selected.get("candidate_id"),
        candidate_provider_sha256=selected.get("candidate_provider_sha256"),
        **kwargs,
    )
    _same(selected, rebuilt, "V20q candidate receipt")


def _candidate_receipts_for_family(
    value: object,
    *,
    protocol_receipt: Mapping[str, object],
    all_ids: tuple[str, ...],
    outer: str,
    inner: str,
) -> dict[str, Mapping[str, object]]:
    mapping = _mapping(value, f"candidate receipts for {inner}")
    if set(mapping) != set(SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS):
        raise ValueError("inner candidate receipt geometry differs")
    result: dict[str, Mapping[str, object]] = {}
    for candidate_id in SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS:
        receipt = _mapping(mapping[candidate_id], f"candidate receipt {candidate_id}")
        validate_soft_polarity_token_vjp_candidate_receipt(
            receipt,
            protocol_receipt=protocol_receipt,
            all_development_family_ids=all_ids,
            outer_held_family_id=outer,
            inner_held_family_id=inner,
        )
        if receipt["candidate_id"] != candidate_id:
            raise ValueError("candidate receipt key and identity differ")
        result[candidate_id] = receipt
    return result


def _objective_row(value: object, family_id: str) -> dict[str, float]:
    mapping = _mapping(value, f"exact objectives for {family_id}")
    if set(mapping) != set(SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS):
        raise ValueError("inner exact objective candidate geometry differs")
    return {
        candidate_id: _float(
            mapping[candidate_id],
            f"{family_id} {candidate_id} exact KL",
            nonnegative=True,
        )
        for candidate_id in SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS
    }


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _standard_error(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = math.fsum((item - mean) ** 2 for item in values) / (
        len(values) - 1
    )
    return math.sqrt(variance / len(values))


def _candidate_receipt_set_sha256(
    receipts: Sequence[Mapping[str, object]],
) -> str:
    return _hash(
        _CANDIDATE_DOMAIN,
        tuple(receipt["artifact_sha256"] for receipt in receipts),
    )


def _conservative_key(
    candidate_id: str,
    receipts: Sequence[Mapping[str, object]],
) -> tuple[object, ...]:
    spec = _SPEC_BY_ID[candidate_id]
    role = spec["role"]
    role_order = {
        "v20p_incumbent": 0,
        "exact_anchor": 1,
        "smooth_seed": 2,
        "token_vjp_fit": 3,
    }[str(role)]
    alpha = 0.0 if spec["alpha"] is None else float(spec["alpha"])
    ridge_order = (
        0.0 if spec["ridge"] is None else -float(spec["ridge"])
    )
    mean_abs_a = _mean([abs(float(receipt["a"])) for receipt in receipts])
    mean_abs_b = _mean([abs(float(receipt["b"])) for receipt in receipts])
    feature = spec["feature_id"]
    feature_order = (
        -1
        if feature is None
        else SOFT_POLARITY_TOKEN_VJP_FEATURE_IDS.index(str(feature))
    )
    sign = 0 if spec["seed_sign"] is None else int(spec["seed_sign"])
    return (
        role_order,
        alpha,
        ridge_order,
        mean_abs_a,
        mean_abs_b,
        feature_order,
        sign,
        SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS.index(candidate_id),
        _candidate_receipt_set_sha256(receipts),
    )


def build_soft_polarity_token_vjp_inner_oof_selection_receipt(
    *,
    protocol_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    candidate_receipts_by_inner_family: Mapping[
        str, Mapping[str, Mapping[str, object]]
    ],
    exact_objectives_by_inner_family_and_candidate: Mapping[
        str, Mapping[str, float]
    ],
) -> dict[str, object]:
    """Select one frozen candidate index from seven six/one OOF folds."""

    validate_soft_polarity_token_vjp_protocol_receipt(protocol_receipt)
    all_ids, outer, _, inner_ids = _family_split(
        all_development_family_ids, outer_held_family_id, None
    )
    receipt_rows = _mapping(
        candidate_receipts_by_inner_family, "candidate receipts by inner family"
    )
    objective_rows = _mapping(
        exact_objectives_by_inner_family_and_candidate,
        "exact objectives by inner family",
    )
    if set(receipt_rows) != set(inner_ids) or set(objective_rows) != set(inner_ids):
        raise ValueError("inner OOF family geometry differs")

    candidates_by_family: dict[str, dict[str, Mapping[str, object]]] = {}
    objectives_by_family: dict[str, dict[str, float]] = {}
    for inner in inner_ids:
        candidates_by_family[inner] = _candidate_receipts_for_family(
            receipt_rows[inner],
            protocol_receipt=protocol_receipt,
            all_ids=all_ids,
            outer=outer,
            inner=inner,
        )
        objectives_by_family[inner] = _objective_row(
            objective_rows[inner], inner
        )

    incumbent_id = SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID
    aggregates: dict[str, dict[str, object]] = {}
    receipts_by_candidate: dict[str, tuple[Mapping[str, object], ...]] = {}
    for candidate_id in SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS:
        objectives = tuple(
            objectives_by_family[inner][candidate_id] for inner in inner_ids
        )
        incumbent = tuple(
            objectives_by_family[inner][incumbent_id] for inner in inner_ids
        )
        deltas = tuple(
            candidate - reference
            for candidate, reference in zip(objectives, incumbent, strict=True)
        )
        candidate_receipts = tuple(
            candidates_by_family[inner][candidate_id] for inner in inner_ids
        )
        receipts_by_candidate[candidate_id] = candidate_receipts
        aggregates[candidate_id] = {
            "candidate_id": candidate_id,
            "family_equal_exact_kl": _mean(objectives),
            "family_equal_exact_kl_hex": _mean(objectives).hex(),
            "paired_delta_to_incumbent": _mean(deltas),
            "paired_delta_to_incumbent_hex": _mean(deltas).hex(),
            "paired_delta_standard_error": _standard_error(deltas),
            "paired_delta_standard_error_hex": _standard_error(deltas).hex(),
            "strict_inner_family_wins_over_incumbent": sum(
                delta < 0.0 for delta in deltas
            ),
            "candidate_receipt_set_sha256": _candidate_receipt_set_sha256(
                candidate_receipts
            ),
        }

    exact_best_id = min(
        SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS,
        key=lambda candidate_id: (
            float(aggregates[candidate_id]["paired_delta_to_incumbent"]),
            _conservative_key(
                candidate_id, receipts_by_candidate[candidate_id]
            ),
        ),
    )
    exact_best = aggregates[exact_best_id]
    one_se_threshold = (
        float(exact_best["paired_delta_to_incumbent"])
        + float(exact_best["paired_delta_standard_error"])
    )
    eligible = tuple(
        candidate_id
        for candidate_id in SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS
        if float(aggregates[candidate_id]["paired_delta_to_incumbent"])
        <= one_se_threshold
    )
    conservative_ranking = tuple(
        sorted(
            eligible,
            key=lambda candidate_id: _conservative_key(
                candidate_id, receipts_by_candidate[candidate_id]
            ),
        )
    )
    selected_id = conservative_ranking[0]

    family_receipts = {
        inner: _finish(
            "fisher_graph.soft_polarity_token_vjp_inner_exact_kl.v20q",
            _SELECTION_DOMAIN,
            {
                "inner_held_family_id": inner,
                "candidate_receipt_sha256_by_candidate": {
                    candidate_id: candidates_by_family[inner][candidate_id][
                        "artifact_sha256"
                    ]
                    for candidate_id in SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS
                },
                "exact_kl_by_candidate": objectives_by_family[inner],
                "exact_kl_hex_by_candidate": {
                    candidate_id: objectives_by_family[inner][
                        candidate_id
                    ].hex()
                    for candidate_id in SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS
                },
                "candidate_frozen_before_objective": True,
                "outer_held_objective_consumed": False,
            },
        )
        for inner in inner_ids
    }
    payload = {
        "protocol_sha256": protocol_receipt["artifact_sha256"],
        "all_development_family_ids": all_ids,
        "outer_held_family_id": outer,
        "inner_held_family_ids": inner_ids,
        "inner_fold_count": 7,
        "training_families_per_inner_fold": 6,
        "candidate_count": len(SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS),
        "family_oof_receipts": family_receipts,
        "aggregate_by_candidate": aggregates,
        "exact_best_candidate_id": exact_best_id,
        "one_standard_error_reference_candidate_id": exact_best_id,
        "one_standard_error_threshold_delta_to_incumbent": one_se_threshold,
        "one_standard_error_threshold_delta_to_incumbent_hex": one_se_threshold.hex(),
        "eligible_candidate_ids": eligible,
        "conservative_ranking": conservative_ranking,
        "selected_candidate_id": selected_id,
        "selected_candidate_spec": dict(_SPEC_BY_ID[selected_id]),
        "selected_inner_candidate_receipt_sha256s": tuple(
            receipt["artifact_sha256"]
            for receipt in receipts_by_candidate[selected_id]
        ),
        "selection_used_only_inner_held_exact_kl": True,
        "outer_held_objective_consumed": False,
        "all_seven_refit_performed": False,
        "data_boundary": dict(_DATA_BOUNDARY),
    }
    return _finish(
        "fisher_graph.soft_polarity_token_vjp_inner_oof_selection.v20q",
        _SELECTION_DOMAIN,
        payload,
    )


def validate_soft_polarity_token_vjp_inner_oof_selection_receipt(
    receipt: Mapping[str, object],
    *,
    protocol_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    candidate_receipts_by_inner_family: Mapping[
        str, Mapping[str, Mapping[str, object]]
    ],
    exact_objectives_by_inner_family_and_candidate: Mapping[
        str, Mapping[str, float]
    ],
) -> None:
    selected = _mapping(receipt, "V20q inner OOF selection receipt")
    _check_hash(selected, _SELECTION_DOMAIN, "V20q inner OOF selection receipt")
    rebuilt = build_soft_polarity_token_vjp_inner_oof_selection_receipt(
        protocol_receipt=protocol_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        candidate_receipts_by_inner_family=candidate_receipts_by_inner_family,
        exact_objectives_by_inner_family_and_candidate=(
            exact_objectives_by_inner_family_and_candidate
        ),
    )
    _same(selected, rebuilt, "V20q inner OOF selection receipt")


def build_soft_polarity_token_vjp_all_seven_refit_receipt(
    *,
    protocol_receipt: Mapping[str, object],
    selection_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
    final_candidate_provider_sha256: str,
    scalar_fit_output: Mapping[str, object] | None = None,
    incumbent_feature_id: str | None = None,
    incumbent_b: float | None = None,
    incumbent_a: float | None = None,
    incumbent_fit_receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Freeze the selected index after a seven-family refit, before outer KL."""

    validate_soft_polarity_token_vjp_protocol_receipt(protocol_receipt)
    selection = _mapping(selection_receipt, "V20q selection receipt")
    _check_hash(selection, _SELECTION_DOMAIN, "V20q selection receipt")
    all_ids, outer, _, training_ids = _family_split(
        all_development_family_ids, outer_held_family_id, None
    )
    if (
        selection.get("protocol_sha256") != protocol_receipt["artifact_sha256"]
        or tuple(selection.get("all_development_family_ids", ())) != all_ids
        or selection.get("outer_held_family_id") != outer
    ):
        raise ValueError("selection and all-seven refit lineage differs")
    selected_id = _identifier(
        selection.get("selected_candidate_id"), "selected candidate"
    )
    spec = _spec(selected_id)
    provider_hash = _sha(
        final_candidate_provider_sha256, "final candidate provider"
    )
    fit_output: dict[str, object] | None = None
    incumbent_source: str | None = None

    if spec["role"] == "token_vjp_fit":
        if any(
            value is not None
            for value in (
                incumbent_feature_id,
                incumbent_b,
                incumbent_a,
                incumbent_fit_receipt_sha256,
            )
        ):
            raise ValueError("final fitted candidate may not consume incumbent fields")
        fit_output = _normalize_scalar_fit_output(scalar_fit_output)
        if (
            fit_output["feature_id"] != spec["feature_id"]
            or fit_output["held_family_id"] != outer
            or tuple(fit_output["training_family_ids"]) != training_ids
            or float(fit_output["reference_b"]).hex()
            != float(spec["seed_b"]).hex()
            or float(fit_output["reference_a"]).hex()
            != float(spec["seed_a"]).hex()
            or float(fit_output["ridge"]).hex() != float(spec["ridge"]).hex()
            or fit_output["no_op"] is not False
            or fit_output["secant_stability_passed"] is not True
            or float(fit_output["predicted_derivative"]) >= 0.0
        ):
            raise ValueError("all-seven scalar fit differs from selected index")
        feature_id = str(fit_output["feature_id"])
        alpha = float(spec["alpha"])
        b = float(fit_output["reference_b"]) + alpha * float(
            fit_output["direction_b"]
        )
        a = float(fit_output["reference_a"]) + alpha * float(
            fit_output["direction_a"]
        )
    elif spec["role"] == "v20p_incumbent":
        if scalar_fit_output is not None:
            raise ValueError("final incumbent may not consume a token-VJP fit")
        feature_id = _identifier(incumbent_feature_id, "incumbent feature")
        if feature_id not in SOFT_POLARITY_TOKEN_VJP_FEATURE_IDS:
            raise ValueError("incumbent feature is outside the V20p field library")
        b = _float(incumbent_b, "incumbent b")
        a = _float(incumbent_a, "incumbent a")
        incumbent_source = _sha(
            incumbent_fit_receipt_sha256, "V20p incumbent fit receipt"
        )
    else:
        if scalar_fit_output is not None or any(
            value is not None
            for value in (
                incumbent_feature_id,
                incumbent_b,
                incumbent_a,
                incumbent_fit_receipt_sha256,
            )
        ):
            raise ValueError("final fixed control may not consume fit fields")
        feature_id = str(spec["feature_id"])
        b = float(spec["fixed_b"])
        a = float(spec["fixed_a"])

    payload = {
        "protocol_sha256": protocol_receipt["artifact_sha256"],
        "selection_receipt_sha256": selection["artifact_sha256"],
        "all_development_family_ids": all_ids,
        "outer_held_family_id": outer,
        "training_family_ids": training_ids,
        "training_family_count": 7,
        "selected_candidate_id": selected_id,
        "selected_candidate_spec": spec,
        "feature_id": feature_id,
        "b": b,
        "b_hex": b.hex(),
        "a": a,
        "a_hex": a.hex(),
        "final_candidate_provider_sha256": provider_hash,
        "scalar_fit_output": fit_output,
        "incumbent_fit_receipt_sha256": incumbent_source,
        "all_seven_refit_completed": spec["role"] == "token_vjp_fit",
        "selected_fixed_control_replayed": spec["role"] != "token_vjp_fit",
        "provider_frozen_before_outer_held_objective": True,
        "outer_held_objective_consumed": False,
        "runtime_provider": "unchanged_v20p_local_signed_field_provider",
        "data_boundary": dict(_DATA_BOUNDARY),
    }
    return _finish(
        "fisher_graph.soft_polarity_token_vjp_all_seven_refit.v20q",
        _FINAL_DOMAIN,
        payload,
    )


def validate_soft_polarity_token_vjp_all_seven_refit_receipt(
    receipt: Mapping[str, object],
    *,
    protocol_receipt: Mapping[str, object],
    selection_receipt: Mapping[str, object],
    all_development_family_ids: Sequence[str],
    outer_held_family_id: str,
) -> None:
    selected = _mapping(receipt, "V20q all-seven refit receipt")
    _check_hash(selected, _FINAL_DOMAIN, "V20q all-seven refit receipt")
    kwargs: dict[str, object] = {}
    if selected.get("scalar_fit_output") is not None:
        kwargs["scalar_fit_output"] = selected.get("scalar_fit_output")
    elif selected.get("selected_candidate_id") == SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID:
        kwargs.update(
            incumbent_feature_id=selected.get("feature_id"),
            incumbent_b=selected.get("b"),
            incumbent_a=selected.get("a"),
            incumbent_fit_receipt_sha256=selected.get(
                "incumbent_fit_receipt_sha256"
            ),
        )
    rebuilt = build_soft_polarity_token_vjp_all_seven_refit_receipt(
        protocol_receipt=protocol_receipt,
        selection_receipt=selection_receipt,
        all_development_family_ids=all_development_family_ids,
        outer_held_family_id=outer_held_family_id,
        final_candidate_provider_sha256=selected.get(
            "final_candidate_provider_sha256"
        ),
        **kwargs,
    )
    _same(selected, rebuilt, "V20q all-seven refit receipt")
