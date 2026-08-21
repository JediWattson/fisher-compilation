"""Traced, model-independent streaming shared-chart atlas selection.

This module is a versioned extension of
``streaming_shared_chart_energy_transport``.  It consumes only authenticated
raw row spools and never receives prompts, token IDs, model weights, gradients,
or model objects.  The fixed-order CPU-float64 center solver is copied from the
v1 production implementation and augmented with public, tensor-free receipts.

The standalone diagnostic evaluates exactly the frozen ``K=8`` chart count.
The selector evaluates the frozen descending capacity order ``8, ..., 1`` and
returns the largest admissible capacity together with the fully constructed
atlas.  Atlas radius, covariance, tangent, and metric construction continues
from the selected traced center state using the unchanged v1 numerical path.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import math
import struct
from typing import Mapping, Sequence

import torch
from torch import Tensor

from . import shared_chart_energy_transport as _reference
from . import streaming_shared_chart_energy_transport as _streaming
from .shared_chart_energy_transport import (
    FIT_ROLE,
    SharedChartAtlas,
    SharedChartNumericsConfig,
)
from .streaming_shared_chart_energy_transport import (
    RawModeRowSpool,
    StreamingChartResourceConfig,
    StreamingChartResourceEstimate,
)


STREAMING_SHARED_CHART_ATLAS_SEED_SCHEMA_V1 = (
    "fisher_graph.streaming_shared_chart_atlas_seed_receipt.v1"
)
STREAMING_SHARED_CHART_ATLAS_PASS_SCHEMA_V1 = (
    "fisher_graph.streaming_shared_chart_atlas_pass_receipt.v1"
)
STREAMING_SHARED_CHART_ATLAS_CANDIDATE_SCHEMA_V1 = (
    "fisher_graph.streaming_shared_chart_atlas_candidate_outcome.v1"
)
STREAMING_SHARED_CHART_ATLAS_OCCUPANCY_DIAGNOSIS_SCHEMA_V1 = (
    "fisher_graph.streaming_shared_chart_atlas_occupancy_diagnosis.v1"
)
STREAMING_SHARED_CHART_ATLAS_SELECTION_SCHEMA_V1 = (
    "fisher_graph.streaming_shared_chart_atlas_selection_receipt.v1"
)
STREAMING_SHARED_CHART_ATLAS_SELECTION_KERNEL_SCHEMA_V1 = (
    "fisher_graph.streaming_shared_chart_atlas_selection_kernel.v1"
)
STREAMING_SHARED_CHART_ATLAS_GATE_EVIDENCE_SCHEMA_V1 = (
    "fisher_graph.streaming_shared_chart_atlas_gate_evidence.v1"
)

STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1 = tuple(range(8, 0, -1))
STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1 = 16
STREAMING_SHARED_CHART_ATLAS_LLOYD_PASS_COUNT_V1 = 12
STREAMING_SHARED_CHART_ATLAS_MINIMUM_HARD_MASS_V1 = 0.005
STREAMING_SHARED_CHART_ATLAS_COMPUTE_DTYPE_V1 = "torch.float64"
STREAMING_SHARED_CHART_ATLAS_COMPUTE_DEVICE_V1 = "cpu"

ADMISSIBLE_OUTCOME = "ADMISSIBLE"
CENTER_INITIALIZATION_DEGENERATE = "CENTER_INITIALIZATION_DEGENERATE"
LLOYD_EMPTY_CHART = "LLOYD_EMPTY_CHART"
HARD_COUNT_BELOW_RANK_PLUS_ONE = "HARD_COUNT_BELOW_RANK_PLUS_ONE"
BALANCED_HARD_MASS_BELOW_MINIMUM = "BALANCED_HARD_MASS_BELOW_MINIMUM"
RADIUS_CONTRACT_FAILED = "RADIUS_CONTRACT_FAILED"
SOFT_OCCUPANCY_CONTRACT_FAILED = "SOFT_OCCUPANCY_CONTRACT_FAILED"
COVARIANCE_CONTRACT_FAILED = "COVARIANCE_CONTRACT_FAILED"
EIGENGAP_ORIENTATION_UNDEFINED = "EIGENGAP_ORIENTATION_UNDEFINED"
SPD_CONTRACT_FAILED = "SPD_CONTRACT_FAILED"
A_ATLAS_SELECTED = "A_ATLAS_SELECTED"
SELECTED_ATLAS_OUTCOME = A_ATLAS_SELECTED
A_ATLAS_NO_ADMISSIBLE_K = "A_ATLAS_NO_ADMISSIBLE_K"

# Compatibility aliases name the three center-stage rejection classes without
# weakening their exact public codes.
EMPTY_CHART_OUTCOME = LLOYD_EMPTY_CHART
INSUFFICIENT_HARD_COUNT_OUTCOME = HARD_COUNT_BELOW_RANK_PLUS_ONE
INSUFFICIENT_HARD_MASS_OUTCOME = BALANCED_HARD_MASS_BELOW_MINIMUM
_OUTCOMES = frozenset(
    {
        ADMISSIBLE_OUTCOME,
        CENTER_INITIALIZATION_DEGENERATE,
        LLOYD_EMPTY_CHART,
        HARD_COUNT_BELOW_RANK_PLUS_ONE,
        BALANCED_HARD_MASS_BELOW_MINIMUM,
        RADIUS_CONTRACT_FAILED,
        SOFT_OCCUPANCY_CONTRACT_FAILED,
        COVARIANCE_CONTRACT_FAILED,
        EIGENGAP_ORIENTATION_UNDEFINED,
        SPD_CONTRACT_FAILED,
    }
)

_SEED_RECEIPT_DOMAIN = b"fisher_graph:streaming-chart-atlas-seed-receipt:v1\0"
_PASS_RECEIPT_DOMAIN = b"fisher_graph:streaming-chart-atlas-pass-receipt:v1\0"
_CANDIDATE_RECEIPT_DOMAIN = (
    b"fisher_graph:streaming-chart-atlas-candidate-receipt:v1\0"
)
_DIAGNOSIS_RECEIPT_DOMAIN = (
    b"fisher_graph:streaming-chart-atlas-occupancy-diagnosis:v1\0"
)
_SELECTION_RECEIPT_DOMAIN = (
    b"fisher_graph:streaming-chart-atlas-selection-receipt:v1\0"
)
_SELECTION_KERNEL_DOMAIN = (
    b"fisher_graph:streaming-chart-atlas-selection-kernel:v1\0"
)
_GATE_EVIDENCE_DOMAIN = (
    b"fisher_graph:streaming-chart-atlas-gate-evidence:v1\0"
)
_CONFIG_DOMAIN = b"fisher_graph:streaming-chart-atlas-selection-config:v1\0"
_MODE_IDENTITY_DOMAIN = b"fisher_graph:streaming-chart-mode-identity:v1\0"
_ROW_IDENTITY_DOMAIN = b"fisher_graph:streaming-chart-seed-row-identity:v1\0"
_COMPOSITION_CATALOG_DOMAIN = (
    b"fisher_graph:streaming-chart-atlas-composition-catalog:v1\0"
)


def _canonical_json_bytes(value: object) -> bytes:
    return _streaming._canonical_json_bytes(value)


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    return _streaming._require_sha256(value, label=label)


def _require_exact_fields(
    value: object, expected: set[str], *, label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return value


def _binary64(value: float | Tensor) -> str:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("receipt binary64 values must be finite")
    return struct.pack(">d", resolved).hex()


def _binary64_bits(value: float | Tensor) -> str:
    """Return the exact binary64 bits, including non-finite gate evidence."""

    return struct.pack(">d", float(value)).hex()


def _parse_binary64(value: object, *, label: str, nonnegative: bool = False) -> float:
    if (
        not isinstance(value, str)
        or len(value) != 16
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase binary64 bit pattern")
    resolved = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(resolved) or (nonnegative and resolved < 0.0):
        raise ValueError(f"{label} must encode a finite valid value")
    return resolved


def _parse_binary64_bits(value: object, *, label: str) -> float:
    if (
        not isinstance(value, str)
        or len(value) != 16
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase binary64 bit pattern")
    return struct.unpack(">d", bytes.fromhex(value))[0]


def _config_sha256(config: SharedChartNumericsConfig) -> str:
    return _digest(_CONFIG_DOMAIN, config.payload())


def streaming_shared_chart_atlas_mode_identity_sha256_v1(
    mode_id: str,
    mode_ordinal: int,
) -> str:
    """Return the public prompt-free mode/ordinal identity digest."""

    if not isinstance(mode_id, str) or not mode_id:
        raise ValueError("mode_id must be a nonempty string")
    if type(mode_ordinal) is not int or mode_ordinal < 0:
        raise ValueError("mode_ordinal must be a nonnegative integer")
    return _digest(
        _MODE_IDENTITY_DOMAIN,
        {"mode_id": mode_id, "mode_ordinal": mode_ordinal},
    )


def _composition_catalog_sha256(
    *,
    chart_count: int,
    mode_count: int,
    family_count: int,
    mode_ids: tuple[str, ...],
    family_ids: tuple[str, ...],
    fit_row_count_by_family: tuple[int, ...],
    count_by_chart_and_mode: tuple[tuple[int, ...], ...],
    mass_by_chart_and_mode: tuple[tuple[str, ...], ...],
    count_by_chart_and_family: tuple[tuple[int, ...], ...],
    mass_by_chart_and_family: tuple[tuple[str, ...], ...],
) -> str:
    return _digest(
        _COMPOSITION_CATALOG_DOMAIN,
        {
            "chart_count": chart_count,
            "count_by_chart_and_family": [
                list(row) for row in count_by_chart_and_family
            ],
            "count_by_chart_and_mode": [
                list(row) for row in count_by_chart_and_mode
            ],
            "family_count": family_count,
            "family_ids": list(family_ids),
            "fit_row_count_by_family": list(fit_row_count_by_family),
            "mass_binary64_by_chart_and_family": [
                list(row) for row in mass_by_chart_and_family
            ],
            "mass_binary64_by_chart_and_mode": [
                list(row) for row in mass_by_chart_and_mode
            ],
            "mode_count": mode_count,
            "mode_ids": list(mode_ids),
        },
    )


def _config_from_payload(value: object) -> SharedChartNumericsConfig:
    expected = set(SharedChartNumericsConfig().payload())
    row = _require_exact_fields(value, expected, label="selection numerics config")
    try:
        return SharedChartNumericsConfig(**dict(row))
    except (TypeError, ValueError) as error:
        raise ValueError("selection numerics config is invalid") from error


def _resources_from_payload(value: object) -> StreamingChartResourceConfig:
    expected = set(StreamingChartResourceConfig().payload())
    row = _require_exact_fields(value, expected, label="selection resources")
    try:
        return StreamingChartResourceConfig(**dict(row))
    except (TypeError, ValueError) as error:
        raise ValueError("selection resources are invalid") from error


def _validate_numeric_policy(config: SharedChartNumericsConfig) -> None:
    if not isinstance(config, SharedChartNumericsConfig):
        raise TypeError("config must be SharedChartNumericsConfig")
    if (
        type(config.chart_count) is not int
        or config.chart_count
        not in STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1
    ):
        raise ValueError("chart_count must be in the frozen range 1..8")
    if config.tangent_rank != STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1:
        raise ValueError("streaming atlas selection requires tangent_rank=16")
    if config.lloyd_iteration_count != STREAMING_SHARED_CHART_ATLAS_LLOYD_PASS_COUNT_V1:
        raise ValueError("streaming atlas selection requires exactly 12 Lloyd passes")
    if config.minimum_hard_occupancy_fraction != (
        STREAMING_SHARED_CHART_ATLAS_MINIMUM_HARD_MASS_V1
    ):
        raise ValueError("streaming atlas selection requires minimum hard mass 0.005")
    default = SharedChartNumericsConfig()
    for label in (
        "membership_temperature",
        "radius_floor_fraction",
        "variance_floor_fraction",
        "eigenvalue_tie_relative_tolerance",
        "transport_relative_ridge",
        "ridge_scale_floor",
        "transport_fisher_energy_floor",
    ):
        if getattr(config, label) != getattr(default, label):
            raise ValueError(
                f"streaming atlas selection requires frozen v1 {label}"
            )


def _validate_resources(resources: StreamingChartResourceConfig) -> None:
    if not isinstance(resources, StreamingChartResourceConfig):
        raise TypeError("resources must be StreamingChartResourceConfig")


def streaming_shared_chart_atlas_selection_kernel_receipt_v1(
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> dict[str, object]:
    """Authenticate the traced selection formulas, gate order, and backend."""

    _validate_resources(resources)
    underlying = _streaming.streaming_chart_kernel_receipt(resources)
    default = SharedChartNumericsConfig()
    payload: dict[str, object] = {
        "candidate_order": list(
            STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1
        ),
        "center_trace_algorithm": (
            "weighted_norm_identity_farthest_first_occurrence_then_"
            "twelve_fixed_order_weighted_lloyd_passes_with_before_after_hashes_v1"
        ),
        "compute_device": STREAMING_SHARED_CHART_ATLAS_COMPUTE_DEVICE_V1,
        "compute_dtype": STREAMING_SHARED_CHART_ATLAS_COMPUTE_DTYPE_V1,
        "covariance_symmetry_contract": {
            "bound_formula": "1e-12_times_max_1_abs_cij_abs_cji",
            "predicate": "pairwise_absolute_residual_lte_scale_bound",
            "relative_tolerance": 1.0e-12,
            "scale_floor": 1.0,
        },
        "full_intrinsic_continuation": (
            "radius_soft_occupancy_covariance_eigengap_spd_then_atlas_integrity_v1"
        ),
        "gate_evidence_order": [
            "radius_finite_positive_by_chart",
            "minimum_membership_nonnegative_by_chart",
            "worst_membership_simplex_residual_global",
            "soft_occupancy_finite_positive_by_chart",
            "covariance_scale_symmetric_worst_pair_by_chart",
            "all_16_strict_relative_eigengaps_by_chart_then_ordinal",
            "basis_orthonormality_worst_pair_by_chart",
            "minimum_tangent_variance_by_chart",
            "orthogonal_alpha_by_chart",
        ],
        "gate_evidence_policy": (
            "cumulative_canonical_pass_prefix_then_terminal_fail_or_full_pass_v1"
        ),
        "eigengap_trace_contract": (
            "exact_binary64_adjacent_pair_chain_nonincreasing_order_and_"
            "complete_17_value_tensor_sha256_only_v1"
        ),
        "lloyd_pass_count": (
            STREAMING_SHARED_CHART_ATLAS_LLOYD_PASS_COUNT_V1
        ),
        "minimum_hard_occupancy_mass_binary64": _binary64(
            STREAMING_SHARED_CHART_ATLAS_MINIMUM_HARD_MASS_V1
        ),
        "intrinsic_gate_order": [
            CENTER_INITIALIZATION_DEGENERATE,
            LLOYD_EMPTY_CHART,
            HARD_COUNT_BELOW_RANK_PLUS_ONE,
            BALANCED_HARD_MASS_BELOW_MINIMUM,
            RADIUS_CONTRACT_FAILED,
            SOFT_OCCUPANCY_CONTRACT_FAILED,
            COVARIANCE_CONTRACT_FAILED,
            EIGENGAP_ORIENTATION_UNDEFINED,
            SPD_CONTRACT_FAILED,
        ],
        "k_dependent_row_feasibility_policy": (
            "no_early_rank_plus_one_abort_run_center_stage_and_descend_on_"
            "typed_initialization_lloyd_or_final_hard_count_outcome_v1"
        ),
        "resources": resources.payload(),
        "schema": STREAMING_SHARED_CHART_ATLAS_SELECTION_KERNEL_SCHEMA_V1,
        "selection_outcomes": [A_ATLAS_SELECTED, A_ATLAS_NO_ADMISSIBLE_K],
        "standalone_occupancy_diagnosis_chart_count": 8,
        "scalar_tensor_trace_bindings": [
            "radius_evidence_to_radius_squared_sha256",
            "soft_mass_evidence_to_soft_occupancy_mass_binary64",
            "complete_eigengap_chain_to_leading_eigenvalues_sha256s",
            "alpha_evidence_to_orthogonal_alpha_sha256",
        ],
        "trace_closure_contract": (
            "ordered_per_center_hash_chain_family_count_margins_"
            "positive_weight_support_and_soft_mass_closure_v1"
        ),
        "tangent_rank": STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1,
        "torch_build_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "torch_version": str(torch.__version__),
        "underlying_streaming_kernel_receipt_sha256": underlying[
            "receipt_sha256"
        ],
        "unchanged_v1_scalar_numerics": {
            label: getattr(default, label)
            for label in (
                "membership_temperature",
                "radius_floor_fraction",
                "variance_floor_fraction",
                "eigenvalue_tie_relative_tolerance",
                "transport_relative_ridge",
                "ridge_scale_floor",
                "transport_fisher_energy_floor",
            )
        },
    }
    return {
        **payload,
        "receipt_sha256": _digest(_SELECTION_KERNEL_DOMAIN, payload),
    }


def _center_sha256s(centers: Tensor) -> tuple[str, ...]:
    return tuple(
        _reference._tensor_sha256(centers[index].contiguous())
        for index in range(len(centers))
    )


def _covariance_symmetry_summary(
    covariance: Tensor,
) -> tuple[bool, int, float, float, float, float, float, float]:
    """Return the canonical worst/first-failure symmetry scalar summary."""

    if (
        covariance.dtype != torch.float64
        or covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not bool(torch.isfinite(covariance).all())
    ):
        raise ValueError("covariance symmetry predicate requires a finite f64 square")
    residuals = torch.abs(covariance - covariance.T)
    scales = torch.maximum(
        torch.ones_like(covariance),
        torch.maximum(torch.abs(covariance), torch.abs(covariance.T)),
    )
    bounds = 1.0e-12 * scales
    failing_pairs = torch.nonzero(residuals > bounds, as_tuple=False)
    margins = bounds - residuals
    if len(failing_pairs):
        row, column = (int(value) for value in failing_pairs[0])
        passed = False
    else:
        flat = int(torch.argmin(margins))
        row, column = divmod(flat, covariance.shape[1])
        passed = True
    residual = float(residuals[row, column])
    scale = float(scales[row, column])
    bound = float(bounds[row, column])
    return (
        passed,
        row * covariance.shape[1] + column,
        float(covariance[row, column]),
        float(covariance[column, row]),
        residual,
        scale,
        bound,
        bound - residual,
    )


def _expected_fit_solver_receipt_sha256(
    *,
    config: SharedChartNumericsConfig,
    resources: StreamingChartResourceConfig,
    row_count_per_mode: int,
    mode_count: int,
    fit_context_catalog_sha256: str,
    fit_balanced_row_weight_catalog_sha256: str,
) -> str:
    estimate = _streaming.estimate_streaming_chart_resources(
        row_count_per_mode=row_count_per_mode,
        mode_count=mode_count,
        config=config,
        resources=resources,
    )
    kernel = _streaming.streaming_chart_kernel_receipt(resources)
    return _reference._digest(
        _reference._ATLAS_SOLVER_RECEIPT_DOMAIN,
        {
            "config": config.payload(),
            "fit_balanced_row_weight_catalog_sha256": (
                fit_balanced_row_weight_catalog_sha256
            ),
            "fit_context_catalog_sha256": fit_context_catalog_sha256,
            "kernel_receipt_sha256": kernel["receipt_sha256"],
            "resource_estimate_receipt_sha256": estimate.receipt_sha256,
            "solver_id": (
                "streamed_balanced_norm_identity_farthest_"
                "weighted_lloyd12_soft_covariance_eigh_v1"
            ),
        },
    )
_GATE_EVIDENCE_PREDICATES = {
    "finite_and_positive_radius_squared": "radius_contract",
    "finite_membership": "soft_occupancy_contract",
    "nonnegative_membership": "soft_occupancy_contract",
    "membership_simplex_absolute_residual": "soft_occupancy_contract",
    "finite_and_positive_soft_occupancy": "soft_occupancy_contract",
    "finite_symmetric_covariance": "covariance_contract",
    "strict_relative_eigengap": "eigengap_orientation",
    "basis_orthonormality": "spd_contract",
    "finite_and_positive_tangent_variance": "spd_contract",
    "finite_and_positive_orthogonal_alpha": "spd_contract",
}


@dataclass(frozen=True, slots=True)
class StreamingSharedChartAtlasGateEvidenceV1:
    gate_id: str
    predicate_id: str
    status: str
    chart_ordinal: int | None
    component_ordinal: int | None
    global_row_ordinal: int | None
    observed_binary64: tuple[str, ...]
    bound_binary64: tuple[str, ...]
    signed_margin_binary64: str | None
    receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            self.predicate_id not in _GATE_EVIDENCE_PREDICATES
            or self.gate_id != _GATE_EVIDENCE_PREDICATES[self.predicate_id]
            or not isinstance(self.status, str)
            or not self.status
        ):
            raise ValueError("intrinsic gate evidence identity differs")
        for label in (
            "chart_ordinal",
            "component_ordinal",
            "global_row_ordinal",
        ):
            value = getattr(self, label)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"gate evidence {label} must be a nonnegative int")
        if (
            not isinstance(self.observed_binary64, tuple)
            or not self.observed_binary64
            or not isinstance(self.bound_binary64, tuple)
        ):
            raise ValueError("intrinsic gate evidence scalar inventory differs")
        for label, values in (
            ("observed_binary64", self.observed_binary64),
            ("bound_binary64", self.bound_binary64),
        ):
            for value in values:
                _parse_binary64_bits(value, label=label)
        observed = tuple(
            _parse_binary64_bits(value, label="observed_binary64")
            for value in self.observed_binary64
        )
        bounds = tuple(
            _parse_binary64_bits(value, label="bound_binary64")
            for value in self.bound_binary64
        )
        margin = (
            None
            if self.signed_margin_binary64 is None
            else _parse_binary64_bits(
                self.signed_margin_binary64,
                label="signed_margin_binary64",
            )
        )
        if self.signed_margin_binary64 is not None:
            _parse_binary64_bits(
                self.signed_margin_binary64,
                label="signed_margin_binary64",
            )
        if self.predicate_id == "membership_simplex_absolute_residual":
            if (
                self.chart_ordinal is not None
                or self.global_row_ordinal is None
                or len(self.observed_binary64) != 2
                or len(self.bound_binary64) != 1
                or self.signed_margin_binary64 is None
            ):
                raise ValueError("simplex gate evidence shape differs")
        elif self.predicate_id in {"finite_membership", "nonnegative_membership"}:
            if (
                self.chart_ordinal is None
                or self.global_row_ordinal is None
                or len(self.observed_binary64) != 1
            ):
                raise ValueError("membership gate evidence shape differs")
        elif (
            self.chart_ordinal is None
            or self.global_row_ordinal is not None
            or len(self.observed_binary64) < 1
        ):
            raise ValueError("chart gate evidence shape differs")
        if self.predicate_id == "strict_relative_eigengap" and (
            self.component_ordinal is None
            or len(self.observed_binary64) != 4
            or len(self.bound_binary64) != 1
            or self.signed_margin_binary64 is None
        ):
            raise ValueError("eigengap evidence shape differs")
        positive_predicates = {
            "finite_and_positive_radius_squared",
            "finite_and_positive_soft_occupancy",
            "finite_and_positive_tangent_variance",
            "finite_and_positive_orthogonal_alpha",
        }
        if self.predicate_id in positive_predicates:
            expected_status = (
                "NONFINITE"
                if not math.isfinite(observed[0])
                else "PASS"
                if observed[0] > 0.0
                else "NONPOSITIVE"
            )
            if (
                len(observed) != 1
                or self.bound_binary64 != (_binary64_bits(0.0),)
                or self.signed_margin_binary64
                != self.observed_binary64[0]
                or self.status != expected_status
            ):
                raise ValueError("positive scalar gate evidence formula differs")
        elif self.predicate_id == "finite_membership":
            if (
                self.status != "NONFINITE"
                or math.isfinite(observed[0])
                or bounds
                or margin is not None
            ):
                raise ValueError("finite-membership evidence formula differs")
        elif self.predicate_id == "nonnegative_membership":
            expected_status = (
                "PASS" if math.isfinite(observed[0]) and observed[0] >= 0.0 else "NEGATIVE"
            )
            if (
                self.status != expected_status
                or not math.isfinite(observed[0])
                or self.bound_binary64 != (_binary64_bits(0.0),)
                or self.signed_margin_binary64
                != self.observed_binary64[0]
            ):
                raise ValueError("nonnegative-membership evidence formula differs")
        elif self.predicate_id == "membership_simplex_absolute_residual":
            expected_residual = abs(observed[0] - 1.0)
            expected_bound = 2.0e-12
            if (
                self.status
                != (
                    "PASS"
                    if expected_residual <= expected_bound
                    else "RESIDUAL_EXCEEDS_BOUND"
                )
                or not all(math.isfinite(value) for value in observed + bounds)
                or self.observed_binary64[1]
                != _binary64_bits(expected_residual)
                or self.bound_binary64
                != (_binary64_bits(expected_bound),)
                or self.signed_margin_binary64
                != _binary64_bits(expected_bound - expected_residual)
            ):
                raise ValueError("simplex evidence formula differs")
        elif self.predicate_id == "finite_symmetric_covariance":
            if self.status == "NONFINITE":
                if (
                    len(observed) != 4
                    or all(math.isfinite(value) for value in observed[:2])
                    or any(math.isfinite(value) for value in observed[2:])
                    or any(math.isfinite(value) for value in bounds)
                    or margin is None
                    or math.isfinite(margin)
                    or self.component_ordinal is None
                ):
                    raise ValueError("nonfinite covariance evidence differs")
            elif self.status in {"PASS", "RESIDUAL_EXCEEDS_BOUND"}:
                expected_residual = abs(observed[0] - observed[1])
                expected_scale = max(1.0, abs(observed[0]), abs(observed[1]))
                expected_bound = 1.0e-12 * expected_scale
                if (
                    len(observed) != 4
                    or self.component_ordinal is None
                    or not all(math.isfinite(value) for value in observed + bounds)
                    or self.observed_binary64[2]
                    != _binary64_bits(expected_residual)
                    or self.observed_binary64[3]
                    != _binary64_bits(expected_scale)
                    or self.bound_binary64
                    != (_binary64_bits(expected_bound),)
                    or self.signed_margin_binary64
                    != _binary64_bits(expected_bound - expected_residual)
                    or self.status
                    != (
                        "PASS"
                        if expected_residual <= expected_bound
                        else "RESIDUAL_EXCEEDS_BOUND"
                    )
                ):
                    raise ValueError("asymmetric covariance evidence differs")
            else:
                raise ValueError("covariance evidence status differs")
        elif self.predicate_id == "strict_relative_eigengap":
            expected_gap = abs(observed[0] - observed[1])
            expected_scale = max(1.0, abs(observed[0]), abs(observed[1]))
            expected_bound = 1.0e-12 * max(
                1.0, abs(observed[0]), abs(observed[1])
            )
            expected_margin = expected_gap - expected_bound
            if (
                self.status
                != ("PASS" if expected_margin > 0.0 else "NONPOSITIVE_MARGIN")
                or not all(math.isfinite(value) for value in observed + bounds)
                or self.observed_binary64[2] != _binary64_bits(expected_gap)
                or self.observed_binary64[3] != _binary64_bits(expected_scale)
                or self.bound_binary64 != (_binary64_bits(expected_bound),)
                or self.signed_margin_binary64
                != _binary64_bits(expected_margin)
            ):
                raise ValueError("eigengap evidence formula differs")
        elif self.predicate_id == "basis_orthonormality":
            if self.component_ordinal is None or len(observed) != 3:
                raise ValueError("basis evidence shape differs")
            row, column = divmod(
                self.component_ordinal,
                STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1,
            )
            expected_identity = 1.0 if row == column else 0.0
            expected_residual = abs(observed[0] - expected_identity)
            expected_bound = 1.0e-9 + 1.0e-9 * abs(expected_identity)
            if self.status == "NONFINITE":
                valid = (
                    not math.isfinite(observed[0])
                    and observed[1] == expected_identity
                    and not math.isfinite(observed[2])
                    and self.bound_binary64 == (_binary64_bits(expected_bound),)
                    and margin is not None
                    and not math.isfinite(margin)
                )
            else:
                valid = (
                    self.status
                    == (
                        "PASS"
                        if expected_residual <= expected_bound
                        else "RESIDUAL_EXCEEDS_BOUND"
                    )
                    and all(math.isfinite(value) for value in observed + bounds)
                    and observed[1] == expected_identity
                    and self.observed_binary64[2]
                    == _binary64_bits(expected_residual)
                    and self.bound_binary64
                    == (_binary64_bits(expected_bound),)
                    and self.signed_margin_binary64
                    == _binary64_bits(expected_bound - expected_residual)
                )
            if not valid:
                raise ValueError("basis evidence formula differs")
        _require_sha256(self.receipt_sha256, label="receipt_sha256")
        if self.receipt_sha256 != _digest(
            _GATE_EVIDENCE_DOMAIN, self._payload_without_hash()
        ):
            raise ValueError("intrinsic gate evidence hash differs")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "bound_binary64": list(self.bound_binary64),
            "chart_ordinal": self.chart_ordinal,
            "component_ordinal": self.component_ordinal,
            "gate_id": self.gate_id,
            "global_row_ordinal": self.global_row_ordinal,
            "observed_binary64": list(self.observed_binary64),
            "predicate_id": self.predicate_id,
            "schema": STREAMING_SHARED_CHART_ATLAS_GATE_EVIDENCE_SCHEMA_V1,
            "signed_margin_binary64": self.signed_margin_binary64,
            "status": self.status,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "receipt_sha256": self.receipt_sha256}


def _make_gate_evidence(**values: object) -> StreamingSharedChartAtlasGateEvidenceV1:
    provisional = object.__new__(StreamingSharedChartAtlasGateEvidenceV1)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    return StreamingSharedChartAtlasGateEvidenceV1(
        **values,  # type: ignore[arg-type]
        receipt_sha256=_digest(
            _GATE_EVIDENCE_DOMAIN, provisional._payload_without_hash()
        ),
    )


@dataclass(frozen=True, slots=True)
class StreamingSharedChartAtlasSeedReceiptV1:
    seed_ordinal: int
    selected_global_row_ordinal: int
    selected_mode_ordinal: int
    selected_row_ordinal: int
    mode_id: str
    family_id: str
    example_id: str
    prediction_query_position: int
    target_label_position: int
    mode_identity_sha256: str
    row_identity_sha256: str
    weighted_farthest_score_binary64: str
    selected_center_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for label in (
            "seed_ordinal",
            "selected_global_row_ordinal",
            "selected_mode_ordinal",
            "selected_row_ordinal",
        ):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        if (
            not isinstance(self.mode_id, str)
            or not self.mode_id
            or not isinstance(self.family_id, str)
            or not self.family_id
            or not isinstance(self.example_id, str)
            or not self.example_id
        ):
            raise ValueError("seed mode/family/example identities must be nonempty")
        if (
            type(self.prediction_query_position) is not int
            or self.prediction_query_position < 0
            or type(self.target_label_position) is not int
            or self.target_label_position != self.prediction_query_position + 1
        ):
            raise ValueError("seed row positions must satisfy t=q+1")
        for label in (
            "mode_identity_sha256",
            "row_identity_sha256",
            "selected_center_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        farthest_score = _parse_binary64(
            self.weighted_farthest_score_binary64,
            label="weighted_farthest_score_binary64",
            nonnegative=True,
        )
        if self.seed_ordinal >= 1 and farthest_score <= 0.0:
            raise ValueError("later farthest-first seed score must be positive")
        expected_mode_identity = streaming_shared_chart_atlas_mode_identity_sha256_v1(
            self.mode_id,
            self.selected_mode_ordinal,
        )
        expected_row_identity = _digest(
            _ROW_IDENTITY_DOMAIN,
            {
                "global_row_ordinal": self.selected_global_row_ordinal,
                "mode_identity_sha256": expected_mode_identity,
                "row_key": [
                    self.family_id,
                    self.example_id,
                    self.prediction_query_position,
                    self.target_label_position,
                ],
                "row_ordinal": self.selected_row_ordinal,
            },
        )
        if (
            self.mode_identity_sha256 != expected_mode_identity
            or self.row_identity_sha256 != expected_row_identity
        ):
            raise ValueError("seed explicit identity/hash binding differs")
        if self.receipt_sha256 != _digest(
            _SEED_RECEIPT_DOMAIN, self._payload_without_hash()
        ):
            raise ValueError("seed receipt hash differs")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "mode_identity_sha256": self.mode_identity_sha256,
            "mode_id": self.mode_id,
            "family_id": self.family_id,
            "example_id": self.example_id,
            "prediction_query_position": self.prediction_query_position,
            "target_label_position": self.target_label_position,
            "row_identity_sha256": self.row_identity_sha256,
            "schema": STREAMING_SHARED_CHART_ATLAS_SEED_SCHEMA_V1,
            "seed_ordinal": self.seed_ordinal,
            "selected_center_sha256": self.selected_center_sha256,
            "selected_global_row_ordinal": self.selected_global_row_ordinal,
            "selected_mode_ordinal": self.selected_mode_ordinal,
            "selected_row_ordinal": self.selected_row_ordinal,
            "weighted_farthest_score_binary64": (
                self.weighted_farthest_score_binary64
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "receipt_sha256": self.receipt_sha256}


def _make_seed_receipt(
    *,
    seed_ordinal: int,
    global_row_ordinal: int,
    rows_per_mode: int,
    aligned: tuple[RawModeRowSpool, ...],
    score: float,
    center: Tensor,
) -> StreamingSharedChartAtlasSeedReceiptV1:
    mode_ordinal, row_ordinal = divmod(global_row_ordinal, rows_per_mode)
    spool = aligned[mode_ordinal]
    row_key = spool.row_keys[row_ordinal]
    mode_identity_sha256 = streaming_shared_chart_atlas_mode_identity_sha256_v1(
        spool.mode_id,
        mode_ordinal,
    )
    row_identity_sha256 = _digest(
        _ROW_IDENTITY_DOMAIN,
        {
            "global_row_ordinal": global_row_ordinal,
            "mode_identity_sha256": mode_identity_sha256,
            "row_key": row_key.payload(),
            "row_ordinal": row_ordinal,
        },
    )
    values = {
        "seed_ordinal": seed_ordinal,
        "selected_global_row_ordinal": global_row_ordinal,
        "selected_mode_ordinal": mode_ordinal,
        "selected_row_ordinal": row_ordinal,
        "mode_id": spool.mode_id,
        "family_id": row_key.family_id,
        "example_id": row_key.example_id,
        "prediction_query_position": row_key.prediction_query_position,
        "target_label_position": row_key.target_label_position,
        "mode_identity_sha256": mode_identity_sha256,
        "row_identity_sha256": row_identity_sha256,
        "weighted_farthest_score_binary64": _binary64(score),
        "selected_center_sha256": _reference._tensor_sha256(center),
    }
    provisional = object.__new__(StreamingSharedChartAtlasSeedReceiptV1)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    return StreamingSharedChartAtlasSeedReceiptV1(
        **values,
        receipt_sha256=_digest(
            _SEED_RECEIPT_DOMAIN, provisional._payload_without_hash()
        ),
    )


@dataclass(frozen=True, slots=True)
class StreamingSharedChartAtlasPassReceiptV1:
    pass_ordinal: int
    centers_before_assignment_sha256: str
    centers_before_assignment_sha256s: tuple[str, ...]
    hard_occupancy_count: tuple[int, ...]
    hard_occupancy_mass_binary64: tuple[str, ...]
    empty_chart_ordinals: tuple[int, ...]
    updated_centers_sha256: str | None
    updated_centers_sha256s: tuple[str, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.pass_ordinal) is not int or not (
            1 <= self.pass_ordinal <= STREAMING_SHARED_CHART_ATLAS_LLOYD_PASS_COUNT_V1
        ):
            raise ValueError("pass_ordinal must be in 1..12")
        _require_sha256(
            self.centers_before_assignment_sha256,
            label="centers_before_assignment_sha256",
        )
        if (
            not isinstance(self.centers_before_assignment_sha256s, tuple)
            or not self.centers_before_assignment_sha256s
            or len(self.centers_before_assignment_sha256s)
            != len(self.hard_occupancy_count)
        ):
            raise ValueError("pass before-center hash inventory differs")
        for value in self.centers_before_assignment_sha256s:
            _require_sha256(value, label="centers_before_assignment_sha256s")
        if (
            not isinstance(self.hard_occupancy_count, tuple)
            or not self.hard_occupancy_count
            or any(type(value) is not int or value < 0 for value in self.hard_occupancy_count)
        ):
            raise ValueError("pass hard counts must be a nonempty integer tuple")
        if (
            not isinstance(self.hard_occupancy_mass_binary64, tuple)
            or len(self.hard_occupancy_mass_binary64)
            != len(self.hard_occupancy_count)
        ):
            raise ValueError("pass hard masses must align with chart counts")
        if not isinstance(self.updated_centers_sha256s, tuple):
            raise ValueError("pass updated per-center hashes must be a tuple")
        decoded_masses = tuple(
            _parse_binary64(value, label="pass hard mass", nonnegative=True)
            for value in self.hard_occupancy_mass_binary64
        )
        if any(
            (count == 0) != (mass == 0.0)
            for count, mass in zip(self.hard_occupancy_count, decoded_masses)
        ):
            raise ValueError("pass hard count/mass support differs")
        expected_empty = tuple(
            index
            for index, value in enumerate(decoded_masses)
            if value == 0.0
        )
        if (
            not isinstance(self.empty_chart_ordinals, tuple)
            or any(type(value) is not int for value in self.empty_chart_ordinals)
            or self.empty_chart_ordinals != expected_empty
        ):
            raise ValueError("pass empty-chart ordinals differ from hard masses")
        if self.updated_centers_sha256 is None:
            if not self.empty_chart_ordinals or self.updated_centers_sha256s:
                raise ValueError("nonempty pass must authenticate updated centers")
        else:
            _require_sha256(self.updated_centers_sha256, label="updated_centers_sha256")
            if (
                self.empty_chart_ordinals
                or not isinstance(self.updated_centers_sha256s, tuple)
                or len(self.updated_centers_sha256s)
                != len(self.centers_before_assignment_sha256s)
            ):
                raise ValueError("empty pass cannot publish updated centers")
            for value in self.updated_centers_sha256s:
                _require_sha256(value, label="updated_centers_sha256s")
        _require_sha256(self.receipt_sha256, label="receipt_sha256")
        if self.receipt_sha256 != _digest(
            _PASS_RECEIPT_DOMAIN, self._payload_without_hash()
        ):
            raise ValueError("pass receipt hash differs")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "empty_chart_ordinals": list(self.empty_chart_ordinals),
            "centers_before_assignment_sha256": (
                self.centers_before_assignment_sha256
            ),
            "centers_before_assignment_sha256s": list(
                self.centers_before_assignment_sha256s
            ),
            "hard_occupancy_count": list(self.hard_occupancy_count),
            "hard_occupancy_mass_binary64": list(
                self.hard_occupancy_mass_binary64
            ),
            "pass_ordinal": self.pass_ordinal,
            "schema": STREAMING_SHARED_CHART_ATLAS_PASS_SCHEMA_V1,
            "updated_centers_sha256": self.updated_centers_sha256,
            "updated_centers_sha256s": list(self.updated_centers_sha256s),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "receipt_sha256": self.receipt_sha256}


def _make_pass_receipt(
    *,
    pass_ordinal: int,
    centers_before_assignment: Tensor,
    counts: Tensor,
    masses: Tensor,
    updated_centers: Tensor | None,
) -> StreamingSharedChartAtlasPassReceiptV1:
    mass_bits = tuple(_binary64(value) for value in masses)
    values = {
        "pass_ordinal": pass_ordinal,
        "centers_before_assignment_sha256": _reference._tensor_sha256(
            centers_before_assignment
        ),
        "centers_before_assignment_sha256s": _center_sha256s(
            centers_before_assignment
        ),
        "hard_occupancy_count": tuple(int(value) for value in counts.tolist()),
        "hard_occupancy_mass_binary64": mass_bits,
        "empty_chart_ordinals": tuple(
            index for index, value in enumerate(masses) if float(value) <= 0.0
        ),
        "updated_centers_sha256": (
            None
            if updated_centers is None
            else _reference._tensor_sha256(updated_centers)
        ),
        "updated_centers_sha256s": (
            () if updated_centers is None else _center_sha256s(updated_centers)
        ),
    }
    provisional = object.__new__(StreamingSharedChartAtlasPassReceiptV1)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    return StreamingSharedChartAtlasPassReceiptV1(
        **values,
        receipt_sha256=_digest(
            _PASS_RECEIPT_DOMAIN, provisional._payload_without_hash()
        ),
    )


@dataclass(frozen=True, slots=True)
class StreamingSharedChartAtlasCandidateOutcomeV1:
    chart_count: int
    source_feature_width: int
    total_row_count: int
    mode_count: int
    family_count: int
    mode_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    fit_row_count_by_family: tuple[int, ...]
    total_balanced_mass_binary64: str
    config_sha256: str
    seed_receipts: tuple[StreamingSharedChartAtlasSeedReceiptV1, ...]
    initialized_centers_sha256: str
    initialized_center_sha256s: tuple[str, ...]
    lloyd_pass_receipts: tuple[StreamingSharedChartAtlasPassReceiptV1, ...]
    final_sort_permutation: tuple[int, ...]
    final_pre_sort_center_sha256s: tuple[str, ...]
    sorted_centers_sha256: str | None
    sorted_center_sha256s: tuple[str, ...]
    final_hard_occupancy_count: tuple[int, ...]
    final_hard_occupancy_mass_binary64: tuple[str, ...]
    final_hard_count_by_chart_and_mode: tuple[tuple[int, ...], ...]
    final_hard_mass_binary64_by_chart_and_mode: tuple[tuple[str, ...], ...]
    final_hard_count_by_chart_and_family: tuple[tuple[int, ...], ...]
    final_hard_mass_binary64_by_chart_and_family: tuple[tuple[str, ...], ...]
    composition_catalog_sha256: str | None
    radius_squared_sha256: str | None
    soft_occupancy_mass_binary64: tuple[str, ...]
    covariance_sha256s: tuple[str, ...]
    leading_eigenvalues_sha256s: tuple[str, ...]
    tangent_bases_sha256: str | None
    tangent_variance_sha256: str | None
    orthogonal_alpha_sha256: str | None
    atlas_sha256: str | None
    fit_solver_receipt_sha256: str | None
    gate_evidence: tuple[StreamingSharedChartAtlasGateEvidenceV1, ...]
    minimum_required_hard_count: int
    minimum_hard_occupancy_mass_binary64: str
    minimum_observed_hard_mass_binary64: str | None
    minimum_hard_mass_margin_binary64: str | None
    minimum_hard_mass_chart_ordinals: tuple[int, ...]
    outcome: str
    terminal_phase: str
    failing_chart_ordinals: tuple[int, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.chart_count) is not int or self.chart_count not in (
            STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1
        ):
            raise ValueError("candidate chart_count must be in 1..8")
        if type(self.source_feature_width) is not int or self.source_feature_width != self.chart_count * (
            STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1 + 1
        ):
            raise ValueError("candidate source feature width differs")
        if type(self.total_row_count) is not int or self.total_row_count <= 0:
            raise ValueError("candidate total_row_count must be positive")
        if (
            type(self.mode_count) is not int
            or self.mode_count <= 0
            or type(self.family_count) is not int
            or self.family_count <= 0
            or self.total_row_count % self.mode_count
        ):
            raise ValueError("candidate mode/family inventory is invalid")
        if (
            not isinstance(self.mode_ids, tuple)
            or len(self.mode_ids) != self.mode_count
            or len(set(self.mode_ids)) != self.mode_count
            or any(not isinstance(value, str) or not value for value in self.mode_ids)
            or tuple(sorted(self.mode_ids)) != self.mode_ids
            or not isinstance(self.family_ids, tuple)
            or len(self.family_ids) != self.family_count
            or len(set(self.family_ids)) != self.family_count
            or any(not isinstance(value, str) or not value for value in self.family_ids)
            or tuple(sorted(self.family_ids)) != self.family_ids
            or not isinstance(self.fit_row_count_by_family, tuple)
            or len(self.fit_row_count_by_family) != self.family_count
            or any(
                type(value) is not int or value <= 0
                for value in self.fit_row_count_by_family
            )
        ):
            raise ValueError("candidate composition axes are invalid")
        rows_per_mode = self.total_row_count // self.mode_count
        if sum(self.fit_row_count_by_family) != rows_per_mode:
            raise ValueError("candidate family row inventory differs")
        total_mass = _parse_binary64(
            self.total_balanced_mass_binary64,
            label="total_balanced_mass_binary64",
            nonnegative=True,
        )
        if not math.isclose(total_mass, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("candidate total balanced mass must close to one")
        _require_sha256(self.config_sha256, label="config_sha256")
        _require_sha256(self.initialized_centers_sha256, label="initialized_centers_sha256")
        if not isinstance(self.seed_receipts, tuple) or not self.seed_receipts:
            raise ValueError("candidate seed receipts must be nonempty")
        expected_seed_count = (
            len(self.seed_receipts)
            if self.outcome == CENTER_INITIALIZATION_DEGENERATE
            else self.chart_count
        )
        if (
            len(self.seed_receipts) != expected_seed_count
            or expected_seed_count > self.chart_count
            or tuple(seed.seed_ordinal for seed in self.seed_receipts)
            != tuple(range(expected_seed_count))
            or len({seed.selected_global_row_ordinal for seed in self.seed_receipts})
            != expected_seed_count
            or any(
                seed.selected_global_row_ordinal >= self.total_row_count
                or seed.selected_mode_ordinal >= self.mode_count
                or seed.mode_id != self.mode_ids[seed.selected_mode_ordinal]
                or seed.selected_row_ordinal >= rows_per_mode
                or seed.selected_global_row_ordinal
                != seed.selected_mode_ordinal * rows_per_mode
                + seed.selected_row_ordinal
                for seed in self.seed_receipts
            )
        ):
            raise ValueError("candidate seed receipts differ from reached initialization")
        if (
            not isinstance(self.initialized_center_sha256s, tuple)
            or self.initialized_center_sha256s
            != tuple(seed.selected_center_sha256 for seed in self.seed_receipts)
        ):
            raise ValueError("candidate initialized per-center hashes differ")
        if not isinstance(self.lloyd_pass_receipts, tuple) or len(
            self.lloyd_pass_receipts
        ) > STREAMING_SHARED_CHART_ATLAS_LLOYD_PASS_COUNT_V1:
            raise ValueError("candidate Lloyd pass trace is invalid")
        if tuple(item.pass_ordinal for item in self.lloyd_pass_receipts) != tuple(
            range(1, len(self.lloyd_pass_receipts) + 1)
        ):
            raise ValueError("candidate Lloyd pass ordinals differ")
        expected_before_sha256 = self.initialized_centers_sha256
        expected_before_sha256s = self.initialized_center_sha256s
        for item in self.lloyd_pass_receipts:
            if (
                item.centers_before_assignment_sha256 != expected_before_sha256
                or item.centers_before_assignment_sha256s
                != expected_before_sha256s
            ):
                raise ValueError("candidate Lloyd center-hash chain differs")
            if (
                len(item.hard_occupancy_count) != self.chart_count
                or sum(item.hard_occupancy_count) != self.total_row_count
                or len(item.hard_occupancy_mass_binary64) != self.chart_count
            ):
                raise ValueError("candidate Lloyd pass chart inventory differs")
            pass_mass = sum(
                _parse_binary64(value, label="pass hard mass", nonnegative=True)
                for value in item.hard_occupancy_mass_binary64
            )
            if not math.isclose(
                pass_mass, total_mass, rel_tol=1.0e-12, abs_tol=1.0e-12
            ):
                raise ValueError("candidate Lloyd pass masses do not close")
            if item.updated_centers_sha256 is not None:
                expected_before_sha256 = item.updated_centers_sha256
                expected_before_sha256s = item.updated_centers_sha256s
        if type(self.minimum_required_hard_count) is not int or self.minimum_required_hard_count != (
            STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1 + 1
        ):
            raise ValueError("candidate minimum hard count differs")
        threshold = _parse_binary64(
            self.minimum_hard_occupancy_mass_binary64,
            label="minimum_hard_occupancy_mass_binary64",
            nonnegative=True,
        )
        if threshold != STREAMING_SHARED_CHART_ATLAS_MINIMUM_HARD_MASS_V1:
            raise ValueError("candidate minimum hard mass differs")
        if self.outcome not in _OUTCOMES:
            raise ValueError("candidate outcome is invalid")
        if (
            not isinstance(self.gate_evidence, tuple)
            or any(
                not isinstance(item, StreamingSharedChartAtlasGateEvidenceV1)
                for item in self.gate_evidence
            )
        ):
            raise ValueError("candidate gate evidence inventory differs")
        for item in self.gate_evidence:
            if (
                item.chart_ordinal is not None
                and item.chart_ordinal >= self.chart_count
            ) or (
                item.global_row_ordinal is not None
                and item.global_row_ordinal >= self.total_row_count
            ):
                raise ValueError("candidate gate evidence ordinal is out of range")
            component_limit = {
                "strict_relative_eigengap": self.minimum_required_hard_count - 1,
                "basis_orthonormality": (
                    STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1**2
                ),
                "finite_and_positive_tangent_variance": (
                    STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1
                ),
            }.get(item.predicate_id)
            if (
                item.component_ordinal is not None
                and item.predicate_id != "finite_symmetric_covariance"
                and (
                    component_limit is None
                    or item.component_ordinal >= component_limit
                )
            ):
                raise ValueError("candidate gate evidence component is out of range")
        if (
            not isinstance(self.failing_chart_ordinals, tuple)
            or tuple(sorted(set(self.failing_chart_ordinals)))
            != self.failing_chart_ordinals
            or any(
                type(index) is not int or not (0 <= index < self.chart_count)
                for index in self.failing_chart_ordinals
            )
        ):
            raise ValueError("candidate failing chart ordinals are invalid")

        geometry_hashes = (
            self.radius_squared_sha256,
            self.tangent_bases_sha256,
            self.tangent_variance_sha256,
            self.orthogonal_alpha_sha256,
            self.atlas_sha256,
        )
        for value in geometry_hashes:
            if value is not None:
                _require_sha256(value, label="candidate geometry sha256")
        if self.fit_solver_receipt_sha256 is not None:
            _require_sha256(
                self.fit_solver_receipt_sha256,
                label="fit_solver_receipt_sha256",
            )
        if (
            self.terminal_phase != "atlas_admissible"
            and self.fit_solver_receipt_sha256 is not None
        ):
            raise ValueError(
                "candidate solver receipt is only valid for an admissible atlas"
            )
        decoded_soft_masses: tuple[float, ...] = ()
        for label, values in (
            ("soft occupancy masses", self.soft_occupancy_mass_binary64),
        ):
            if not isinstance(values, tuple):
                raise ValueError(f"candidate {label} must be a tuple")
            for value in values:
                _parse_binary64(value, label=label, nonnegative=True)
            decoded_soft_masses = tuple(
                _parse_binary64(value, label=label, nonnegative=True)
                for value in values
            )
        if self.soft_occupancy_mass_binary64 and (
            len(decoded_soft_masses) != self.chart_count
            or any(value <= 0.0 for value in decoded_soft_masses)
            or not math.isclose(
                sum(decoded_soft_masses),
                total_mass,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("candidate soft occupancy masses do not close")
        for label, values in (
            ("covariance_sha256s", self.covariance_sha256s),
            ("leading_eigenvalues_sha256s", self.leading_eigenvalues_sha256s),
        ):
            if not isinstance(values, tuple) or len(values) > self.chart_count:
                raise ValueError(f"candidate {label} inventory differs")
            for value in values:
                _require_sha256(value, label=label)

        no_final = (
            not self.final_sort_permutation
            and not self.final_pre_sort_center_sha256s
            and self.sorted_centers_sha256 is None
            and not self.sorted_center_sha256s
            and not self.final_hard_occupancy_count
            and not self.final_hard_occupancy_mass_binary64
            and not self.final_hard_count_by_chart_and_mode
            and not self.final_hard_mass_binary64_by_chart_and_mode
            and not self.final_hard_count_by_chart_and_family
            and not self.final_hard_mass_binary64_by_chart_and_family
            and self.composition_catalog_sha256 is None
            and self.minimum_observed_hard_mass_binary64 is None
            and self.minimum_hard_mass_margin_binary64 is None
            and not self.minimum_hard_mass_chart_ordinals
        )
        no_geometry = (
            all(value is None for value in geometry_hashes)
            and not self.soft_occupancy_mass_binary64
            and not self.covariance_sha256s
            and not self.leading_eigenvalues_sha256s
        )
        radius_only = (
            self.radius_squared_sha256 is not None
            and not self.soft_occupancy_mass_binary64
            and not self.covariance_sha256s
            and not self.leading_eigenvalues_sha256s
            and self.tangent_bases_sha256 is None
            and self.tangent_variance_sha256 is None
            and self.orthogonal_alpha_sha256 is None
            and self.atlas_sha256 is None
            and self.fit_solver_receipt_sha256 is None
        )
        radius_soft_only = (
            self.radius_squared_sha256 is not None
            and len(self.soft_occupancy_mass_binary64) == self.chart_count
            and not self.covariance_sha256s
            and not self.leading_eigenvalues_sha256s
            and self.tangent_bases_sha256 is None
            and self.tangent_variance_sha256 is None
            and self.orthogonal_alpha_sha256 is None
            and self.atlas_sha256 is None
        )
        if self.outcome == CENTER_INITIALIZATION_DEGENERATE:
            if (
                self.terminal_phase != "center_initialization"
                or len(self.seed_receipts) >= self.chart_count
                or self.lloyd_pass_receipts
                or not no_final
                or not no_geometry
                or self.failing_chart_ordinals
                != (len(self.seed_receipts),)
                or self.gate_evidence
            ):
                raise ValueError("degenerate initialization evidence differs")
        elif self.outcome == LLOYD_EMPTY_CHART:
            if not self.lloyd_pass_receipts:
                raise ValueError("empty-chart candidate omitted its Lloyd pass")
            last = self.lloyd_pass_receipts[-1]
            if (
                self.terminal_phase != f"lloyd_pass_{last.pass_ordinal}"
                or not last.empty_chart_ordinals
                or self.failing_chart_ordinals != last.empty_chart_ordinals
                or not no_final
                or not no_geometry
                or self.gate_evidence
            ):
                raise ValueError("empty-chart candidate terminal evidence differs")
        else:
            if len(self.lloyd_pass_receipts) != (
                STREAMING_SHARED_CHART_ATLAS_LLOYD_PASS_COUNT_V1
            ) or any(item.empty_chart_ordinals for item in self.lloyd_pass_receipts):
                raise ValueError("completed candidate Lloyd trace differs")
            if (
                not isinstance(self.final_sort_permutation, tuple)
                or len(self.final_sort_permutation) != self.chart_count
                or any(type(value) is not int for value in self.final_sort_permutation)
                or tuple(sorted(self.final_sort_permutation))
                != tuple(range(self.chart_count))
            ):
                raise ValueError("candidate final sort permutation is invalid")
            if (
                not isinstance(self.final_pre_sort_center_sha256s, tuple)
                or self.final_pre_sort_center_sha256s
                != expected_before_sha256s
                or not isinstance(self.sorted_center_sha256s, tuple)
                or self.sorted_center_sha256s
                != tuple(
                    self.final_pre_sort_center_sha256s[index]
                    for index in self.final_sort_permutation
                )
            ):
                raise ValueError("candidate final per-center permutation differs")
            _require_sha256(self.sorted_centers_sha256, label="sorted_centers_sha256")
            if (
                len(self.final_hard_occupancy_count) != self.chart_count
                or any(
                    type(value) is not int or value < 0
                    for value in self.final_hard_occupancy_count
                )
                or sum(self.final_hard_occupancy_count) != self.total_row_count
                or len(self.final_hard_occupancy_mass_binary64) != self.chart_count
            ):
                raise ValueError("candidate final reassignment inventory differs")
            masses = tuple(
                _parse_binary64(value, label="final hard mass", nonnegative=True)
                for value in self.final_hard_occupancy_mass_binary64
            )
            if not math.isclose(
                sum(masses), total_mass, rel_tol=1.0e-12, abs_tol=1.0e-12
            ):
                raise ValueError("candidate final masses do not close")
            if any(
                (count == 0) != (mass == 0.0)
                for count, mass in zip(self.final_hard_occupancy_count, masses)
            ):
                raise ValueError("candidate final count/mass support differs")
            count_mode = self.final_hard_count_by_chart_and_mode
            mass_mode = self.final_hard_mass_binary64_by_chart_and_mode
            count_family = self.final_hard_count_by_chart_and_family
            mass_family = self.final_hard_mass_binary64_by_chart_and_family
            for label, matrix, width in (
                ("count by mode", count_mode, self.mode_count),
                ("mass by mode", mass_mode, self.mode_count),
                ("count by family", count_family, self.family_count),
                ("mass by family", mass_family, self.family_count),
            ):
                if (
                    not isinstance(matrix, tuple)
                    or len(matrix) != self.chart_count
                    or any(
                        not isinstance(row, tuple) or len(row) != width
                        for row in matrix
                    )
                ):
                    raise ValueError(f"candidate composition {label} shape differs")
            if any(
                type(value) is not int or value < 0
                for matrix in (count_mode, count_family)
                for row in matrix
                for value in row
            ):
                raise ValueError("candidate composition counts are invalid")
            decoded_mode = tuple(
                tuple(
                    _parse_binary64(value, label="composition mode mass", nonnegative=True)
                    for value in row
                )
                for row in mass_mode
            )
            decoded_family = tuple(
                tuple(
                    _parse_binary64(value, label="composition family mass", nonnegative=True)
                    for value in row
                )
                for row in mass_family
            )
            for counts_matrix, masses_matrix, label in (
                (count_mode, decoded_mode, "mode"),
                (count_family, decoded_family, "family"),
            ):
                if any(
                    (count == 0) != (mass == 0.0)
                    for count_row, mass_row in zip(counts_matrix, masses_matrix)
                    for count, mass in zip(count_row, mass_row)
                ):
                    raise ValueError(
                        f"candidate composition {label} count/mass support differs"
                    )
            for chart in range(self.chart_count):
                if (
                    sum(count_mode[chart]) != self.final_hard_occupancy_count[chart]
                    or sum(count_family[chart])
                    != self.final_hard_occupancy_count[chart]
                    or not math.isclose(
                        sum(decoded_mode[chart]),
                        masses[chart],
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                    or not math.isclose(
                        sum(decoded_family[chart]),
                        masses[chart],
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                ):
                    raise ValueError("candidate composition chart margins differ")
            for mode in range(self.mode_count):
                if sum(row[mode] for row in count_mode) != rows_per_mode or not math.isclose(
                    sum(row[mode] for row in decoded_mode),
                    1.0 / self.mode_count,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError("candidate composition mode margins differ")
            for family in range(self.family_count):
                if (
                    sum(row[family] for row in count_family)
                    != self.mode_count * self.fit_row_count_by_family[family]
                    or not math.isclose(
                    sum(row[family] for row in decoded_family),
                    1.0 / self.family_count,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                    )
                ):
                    raise ValueError("candidate composition family margins differ")
            _require_sha256(
                self.composition_catalog_sha256,
                label="composition_catalog_sha256",
            )
            if self.composition_catalog_sha256 != _composition_catalog_sha256(
                chart_count=self.chart_count,
                mode_count=self.mode_count,
                family_count=self.family_count,
                mode_ids=self.mode_ids,
                family_ids=self.family_ids,
                fit_row_count_by_family=self.fit_row_count_by_family,
                count_by_chart_and_mode=count_mode,
                mass_by_chart_and_mode=mass_mode,
                count_by_chart_and_family=count_family,
                mass_by_chart_and_family=mass_family,
            ):
                raise ValueError("candidate composition catalog hash differs")
            observed_minimum = _parse_binary64(
                self.minimum_observed_hard_mass_binary64,
                label="minimum_observed_hard_mass_binary64",
                nonnegative=True,
            )
            margin = _parse_binary64(
                self.minimum_hard_mass_margin_binary64,
                label="minimum_hard_mass_margin_binary64",
            )
            minimum_ordinals = tuple(
                index for index, mass in enumerate(masses) if mass == min(masses)
            )
            if (
                observed_minimum != min(masses)
                or margin != observed_minimum - threshold
                or not isinstance(self.minimum_hard_mass_chart_ordinals, tuple)
                or any(
                    type(value) is not int
                    for value in self.minimum_hard_mass_chart_ordinals
                )
                or self.minimum_hard_mass_chart_ordinals != minimum_ordinals
            ):
                raise ValueError("candidate minimum hard-mass summary differs")
            count_failures = tuple(
                index
                for index, count in enumerate(self.final_hard_occupancy_count)
                if count < self.minimum_required_hard_count
            )
            mass_failures = tuple(
                index for index, mass in enumerate(masses) if mass < threshold
            )
            if self.outcome == HARD_COUNT_BELOW_RANK_PLUS_ONE:
                expected = (
                    self.terminal_phase == "final_hard_count_gate"
                    and bool(count_failures)
                    and self.failing_chart_ordinals == count_failures
                    and no_geometry
                    and not self.gate_evidence
                )
            elif self.outcome == BALANCED_HARD_MASS_BELOW_MINIMUM:
                expected = (
                    self.terminal_phase == "final_hard_mass_gate"
                    and not count_failures
                    and bool(mass_failures)
                    and self.failing_chart_ordinals == mass_failures
                    and no_geometry
                    and not self.gate_evidence
                )
            else:
                expected = not count_failures and not mass_failures
                if self.outcome == ADMISSIBLE_OUTCOME:
                    if self.terminal_phase == "occupancy_admissible":
                        expected = (
                            expected
                            and not self.failing_chart_ordinals
                            and no_geometry
                            and not self.gate_evidence
                            and self.fit_solver_receipt_sha256 is None
                        )
                    elif self.terminal_phase == "atlas_admissible":
                        expected = (
                            expected
                            and not self.failing_chart_ordinals
                            and self.radius_squared_sha256 is not None
                            and len(self.soft_occupancy_mass_binary64) == self.chart_count
                            and len(self.covariance_sha256s) == self.chart_count
                            and len(self.leading_eigenvalues_sha256s) == self.chart_count
                            and self.tangent_bases_sha256 is not None
                            and self.tangent_variance_sha256 is not None
                            and self.orthogonal_alpha_sha256 is not None
                            and self.atlas_sha256 is not None
                            and self.fit_solver_receipt_sha256 is not None
                            and self._gate_evidence_inventory_valid()
                        )
                    else:
                        expected = False
                elif self.outcome == RADIUS_CONTRACT_FAILED:
                    expected = (
                        expected
                        and self.terminal_phase == "radius_contract"
                        and bool(self.failing_chart_ordinals)
                        and no_geometry
                        and self.fit_solver_receipt_sha256 is None
                        and self._gate_evidence_inventory_valid()
                    )
                elif self.outcome == SOFT_OCCUPANCY_CONTRACT_FAILED:
                    expected = (
                        expected
                        and self.terminal_phase == "soft_occupancy_contract"
                        and bool(self.failing_chart_ordinals)
                        and radius_only
                        and self.fit_solver_receipt_sha256 is None
                        and self._gate_evidence_inventory_valid()
                    )
                elif self.outcome == COVARIANCE_CONTRACT_FAILED:
                    expected = (
                        expected
                        and self.terminal_phase == "covariance_contract"
                        and bool(self.failing_chart_ordinals)
                        and radius_soft_only
                        and self.fit_solver_receipt_sha256 is None
                        and self._gate_evidence_inventory_valid()
                    )
                elif self.outcome == EIGENGAP_ORIENTATION_UNDEFINED:
                    terminal_eigengap = (
                        self.gate_evidence[-1]
                        if self.gate_evidence
                        and self.gate_evidence[-1].predicate_id
                        == "strict_relative_eigengap"
                        else None
                    )
                    expected = (
                        expected
                        and self.terminal_phase == "eigengap_orientation"
                        and len(self.failing_chart_ordinals) == 1
                        and len(self.covariance_sha256s) == self.chart_count
                        and len(self.soft_occupancy_mass_binary64)
                        == self.chart_count
                        and self.radius_squared_sha256 is not None
                        and len(self.leading_eigenvalues_sha256s)
                        == self.failing_chart_ordinals[0]
                        + int(
                            terminal_eigengap is not None
                            and terminal_eigengap.component_ordinal
                            == STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1 - 1
                        )
                        and self.tangent_bases_sha256 is None
                        and self.tangent_variance_sha256 is None
                        and self.orthogonal_alpha_sha256 is None
                        and self.atlas_sha256 is None
                        and self.fit_solver_receipt_sha256 is None
                        and self._gate_evidence_inventory_valid()
                    )
                elif self.outcome == SPD_CONTRACT_FAILED:
                    expected = (
                        expected
                        and self.terminal_phase == "spd_contract"
                        and bool(self.failing_chart_ordinals)
                        and len(self.covariance_sha256s) == self.chart_count
                        and len(self.soft_occupancy_mass_binary64)
                        == self.chart_count
                        and self.radius_squared_sha256 is not None
                        and len(self.leading_eigenvalues_sha256s)
                        == self.chart_count
                        and self.tangent_variance_sha256 is None
                        and self.orthogonal_alpha_sha256 is None
                        and self.atlas_sha256 is None
                        and self.fit_solver_receipt_sha256 is None
                        and (
                            (
                                any(
                                    item.predicate_id == "basis_orthonormality"
                                    and item.status != "PASS"
                                    for item in self.gate_evidence
                                )
                                and self.tangent_bases_sha256 is None
                            )
                            or (
                                not any(
                                    item.predicate_id == "basis_orthonormality"
                                    and item.status != "PASS"
                                    for item in self.gate_evidence
                                )
                                and self.tangent_bases_sha256 is not None
                            )
                        )
                        and self._gate_evidence_inventory_valid()
                    )
                else:
                    expected = False
            if not expected:
                raise ValueError("candidate typed outcome differs from final gates")
        _require_sha256(self.receipt_sha256, label="receipt_sha256")
        if self.receipt_sha256 != _digest(
            _CANDIDATE_RECEIPT_DOMAIN, self._payload_without_hash()
        ):
            raise ValueError("candidate receipt hash differs")

    def _gate_evidence_inventory_valid(self) -> bool:
        rows = self.gate_evidence
        cursor = 0

        def take_chart_rows(
            predicate_id: str,
            count: int,
        ) -> tuple[StreamingSharedChartAtlasGateEvidenceV1, ...] | None:
            nonlocal cursor
            selected = rows[cursor : cursor + count]
            if (
                len(selected) != count
                or tuple(item.predicate_id for item in selected)
                != (predicate_id,) * count
                or tuple(item.chart_ordinal for item in selected)
                != tuple(range(count))
            ):
                return None
            cursor += count
            return selected

        radius = take_chart_rows(
            "finite_and_positive_radius_squared", self.chart_count
        )
        if radius is None:
            return False
        radius_failures = tuple(
            item.chart_ordinal for item in radius if item.status != "PASS"
        )
        if self.radius_squared_sha256 is not None and (
            _reference._tensor_sha256(
                torch.tensor(
                    [
                        _parse_binary64_bits(
                            item.observed_binary64[0],
                            label="radius gate evidence",
                        )
                        for item in radius
                    ],
                    dtype=torch.float64,
                )
            )
            != self.radius_squared_sha256
        ):
            return False
        if self.outcome == RADIUS_CONTRACT_FAILED:
            return (
                cursor == len(rows)
                and radius_failures == self.failing_chart_ordinals
            )
        if radius_failures:
            return False

        if self.outcome == SOFT_OCCUPANCY_CONTRACT_FAILED and (
            len(rows) == cursor + 1
        ):
            terminal = rows[cursor]
            if terminal.status == "PASS" or terminal.predicate_id not in {
                "finite_membership",
                "nonnegative_membership",
                "membership_simplex_absolute_residual",
            }:
                return False
            expected_failures = (
                tuple(range(self.chart_count))
                if terminal.chart_ordinal is None
                else (terminal.chart_ordinal,)
            )
            return expected_failures == self.failing_chart_ordinals

        membership = take_chart_rows(
            "nonnegative_membership", self.chart_count
        )
        if membership is None or any(item.status != "PASS" for item in membership):
            return False
        if cursor >= len(rows):
            return False
        simplex = rows[cursor]
        cursor += 1
        if (
            simplex.predicate_id != "membership_simplex_absolute_residual"
            or simplex.status != "PASS"
            or simplex.chart_ordinal is not None
        ):
            return False
        soft = take_chart_rows(
            "finite_and_positive_soft_occupancy", self.chart_count
        )
        if soft is None:
            return False
        soft_failures = tuple(
            item.chart_ordinal for item in soft if item.status != "PASS"
        )
        if self.outcome == SOFT_OCCUPANCY_CONTRACT_FAILED:
            return (
                cursor == len(rows)
                and soft_failures == self.failing_chart_ordinals
            )
        if soft_failures:
            return False
        if tuple(item.observed_binary64[0] for item in soft) != (
            self.soft_occupancy_mass_binary64
        ):
            return False

        covariance = take_chart_rows(
            "finite_symmetric_covariance", self.chart_count
        )
        if covariance is None:
            return False
        covariance_failures = tuple(
            item.chart_ordinal
            for item in covariance
            if item.status != "PASS"
        )
        if self.outcome == COVARIANCE_CONTRACT_FAILED:
            return (
                cursor == len(rows)
                and covariance_failures == self.failing_chart_ordinals
            )
        if covariance_failures:
            return False

        if self.outcome == EIGENGAP_ORIENTATION_UNDEFINED:
            failing_chart = self.failing_chart_ordinals[0]
            eigengaps = rows[cursor:]
            if not eigengaps:
                return False
            terminal_component = eigengaps[-1].component_ordinal
            if terminal_component is None:
                return False
            expected_pairs = tuple(
                (chart, component)
                for chart in range(failing_chart)
                for component in range(
                    STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1
                )
            ) + tuple(
                (failing_chart, component)
                for component in range(terminal_component + 1)
            )
            if not self._eigengap_rows_are_chained(eigengaps):
                return False
            complete_charts = failing_chart + int(terminal_component == 15)
            if any(
                not self._eigengap_spectrum_hash_matches(
                    chart,
                    eigengaps[
                        chart
                        * STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1 :
                        (chart + 1)
                        * STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1
                    ],
                )
                for chart in range(complete_charts)
            ):
                return False
            return (
                tuple(
                    (item.chart_ordinal, item.component_ordinal)
                    for item in eigengaps
                )
                == expected_pairs
                and all(
                    item.predicate_id == "strict_relative_eigengap"
                    and item.status == "PASS"
                    for item in eigengaps[:-1]
                )
                and eigengaps[-1].predicate_id
                == "strict_relative_eigengap"
                and eigengaps[-1].status == "NONPOSITIVE_MARGIN"
            )
        eigengap_count = (
            self.chart_count
            * STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1
        )
        eigengaps = rows[cursor : cursor + eigengap_count]
        expected_pairs = tuple(
            (chart, component)
            for chart in range(self.chart_count)
            for component in range(
                STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1
            )
        )
        if (
            len(eigengaps) != eigengap_count
            or tuple(
                (item.chart_ordinal, item.component_ordinal)
                for item in eigengaps
            )
            != expected_pairs
            or any(
                item.predicate_id != "strict_relative_eigengap"
                or item.status != "PASS"
                for item in eigengaps
            )
        ):
            return False
        if not self._eigengap_rows_are_chained(eigengaps) or any(
            not self._eigengap_spectrum_hash_matches(
                chart,
                eigengaps[
                    chart * STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1 :
                    (chart + 1)
                    * STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1
                ],
            )
            for chart in range(self.chart_count)
        ):
            return False
        cursor += eigengap_count

        basis = take_chart_rows("basis_orthonormality", self.chart_count)
        if basis is None:
            return False
        basis_failures = tuple(
            item.chart_ordinal for item in basis if item.status != "PASS"
        )
        if self.outcome == SPD_CONTRACT_FAILED and basis_failures:
            return (
                cursor == len(rows)
                and basis_failures == self.failing_chart_ordinals
            )
        if basis_failures:
            return False
        tangent = take_chart_rows(
            "finite_and_positive_tangent_variance", self.chart_count
        )
        alpha = take_chart_rows(
            "finite_and_positive_orthogonal_alpha", self.chart_count
        )
        if tangent is None or alpha is None:
            return False
        if self.orthogonal_alpha_sha256 is not None and (
            _reference._tensor_sha256(
                torch.tensor(
                    [
                        _parse_binary64_bits(
                            item.observed_binary64[0],
                            label="orthogonal-alpha gate evidence",
                        )
                        for item in alpha
                    ],
                    dtype=torch.float64,
                )
            )
            != self.orthogonal_alpha_sha256
        ):
            return False
        spd_failures = tuple(
            sorted(
                {
                    item.chart_ordinal
                    for item in (*tangent, *alpha)
                    if item.status != "PASS"
                }
            )
        )
        if self.outcome == SPD_CONTRACT_FAILED:
            return (
                cursor == len(rows)
                and spd_failures == self.failing_chart_ordinals
            )
        return (
            self.outcome == ADMISSIBLE_OUTCOME
            and self.terminal_phase == "atlas_admissible"
            and cursor == len(rows)
            and not spd_failures
        )

    @staticmethod
    def _eigengap_rows_are_chained(
        rows: tuple[StreamingSharedChartAtlasGateEvidenceV1, ...],
    ) -> bool:
        for item in rows:
            selected = _parse_binary64_bits(
                item.observed_binary64[0], label="eigengap selected eigenvalue"
            )
            following = _parse_binary64_bits(
                item.observed_binary64[1], label="eigengap following eigenvalue"
            )
            if selected < following:
                return False
        for previous, current in zip(rows, rows[1:]):
            if (
                previous.chart_ordinal == current.chart_ordinal
                and previous.observed_binary64[1]
                != current.observed_binary64[0]
            ):
                return False
        return True

    def _eigengap_spectrum_hash_matches(
        self,
        chart: int,
        rows: tuple[StreamingSharedChartAtlasGateEvidenceV1, ...],
    ) -> bool:
        if (
            len(rows) != STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1
            or chart >= len(self.leading_eigenvalues_sha256s)
        ):
            return False
        values = [
            _parse_binary64_bits(
                rows[0].observed_binary64[0],
                label="eigengap leading eigenvalue",
            )
        ]
        values.extend(
            _parse_binary64_bits(
                item.observed_binary64[1],
                label="eigengap following eigenvalue",
            )
            for item in rows
        )
        return (
            _reference._tensor_sha256(
                torch.tensor(values, dtype=torch.float64)
            )
            == self.leading_eigenvalues_sha256s[chart]
        )

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "chart_count": self.chart_count,
            "mode_count": self.mode_count,
            "family_count": self.family_count,
            "mode_ids": list(self.mode_ids),
            "family_ids": list(self.family_ids),
            "fit_row_count_by_family": list(self.fit_row_count_by_family),
            "config_sha256": self.config_sha256,
            "failing_chart_ordinals": list(self.failing_chart_ordinals),
            "final_hard_occupancy_count": list(self.final_hard_occupancy_count),
            "final_hard_occupancy_mass_binary64": list(
                self.final_hard_occupancy_mass_binary64
            ),
            "final_hard_count_by_chart_and_mode": [
                list(row) for row in self.final_hard_count_by_chart_and_mode
            ],
            "final_hard_mass_binary64_by_chart_and_mode": [
                list(row)
                for row in self.final_hard_mass_binary64_by_chart_and_mode
            ],
            "final_hard_count_by_chart_and_family": [
                list(row) for row in self.final_hard_count_by_chart_and_family
            ],
            "final_hard_mass_binary64_by_chart_and_family": [
                list(row)
                for row in self.final_hard_mass_binary64_by_chart_and_family
            ],
            "composition_catalog_sha256": self.composition_catalog_sha256,
            "final_sort_permutation": list(self.final_sort_permutation),
            "final_pre_sort_center_sha256s": list(
                self.final_pre_sort_center_sha256s
            ),
            "radius_squared_sha256": self.radius_squared_sha256,
            "soft_occupancy_mass_binary64": list(
                self.soft_occupancy_mass_binary64
            ),
            "covariance_sha256s": list(self.covariance_sha256s),
            "leading_eigenvalues_sha256s": list(
                self.leading_eigenvalues_sha256s
            ),
            "tangent_bases_sha256": self.tangent_bases_sha256,
            "tangent_variance_sha256": self.tangent_variance_sha256,
            "orthogonal_alpha_sha256": self.orthogonal_alpha_sha256,
            "atlas_sha256": self.atlas_sha256,
            "fit_solver_receipt_sha256": self.fit_solver_receipt_sha256,
            "gate_evidence": [item.state_dict() for item in self.gate_evidence],
            "initialized_centers_sha256": self.initialized_centers_sha256,
            "initialized_center_sha256s": list(
                self.initialized_center_sha256s
            ),
            "lloyd_pass_receipts": [
                item.state_dict() for item in self.lloyd_pass_receipts
            ],
            "minimum_hard_occupancy_mass_binary64": (
                self.minimum_hard_occupancy_mass_binary64
            ),
            "minimum_required_hard_count": self.minimum_required_hard_count,
            "minimum_observed_hard_mass_binary64": (
                self.minimum_observed_hard_mass_binary64
            ),
            "minimum_hard_mass_margin_binary64": (
                self.minimum_hard_mass_margin_binary64
            ),
            "minimum_hard_mass_chart_ordinals": list(
                self.minimum_hard_mass_chart_ordinals
            ),
            "outcome": self.outcome,
            "schema": STREAMING_SHARED_CHART_ATLAS_CANDIDATE_SCHEMA_V1,
            "seed_receipts": [item.state_dict() for item in self.seed_receipts],
            "sorted_centers_sha256": self.sorted_centers_sha256,
            "sorted_center_sha256s": list(self.sorted_center_sha256s),
            "source_feature_width": self.source_feature_width,
            "terminal_phase": self.terminal_phase,
            "total_balanced_mass_binary64": self.total_balanced_mass_binary64,
            "total_row_count": self.total_row_count,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "receipt_sha256": self.receipt_sha256}


def _make_candidate_outcome(**values: object) -> StreamingSharedChartAtlasCandidateOutcomeV1:
    values = {
        "final_pre_sort_center_sha256s": (),
        "sorted_center_sha256s": (),
        "final_hard_count_by_chart_and_mode": (),
        "final_hard_mass_binary64_by_chart_and_mode": (),
        "final_hard_count_by_chart_and_family": (),
        "final_hard_mass_binary64_by_chart_and_family": (),
        "composition_catalog_sha256": None,
        "radius_squared_sha256": None,
        "soft_occupancy_mass_binary64": (),
        "covariance_sha256s": (),
        "leading_eigenvalues_sha256s": (),
        "tangent_bases_sha256": None,
        "tangent_variance_sha256": None,
        "orthogonal_alpha_sha256": None,
        "atlas_sha256": None,
        "fit_solver_receipt_sha256": None,
        "gate_evidence": (),
        "minimum_observed_hard_mass_binary64": None,
        "minimum_hard_mass_margin_binary64": None,
        "minimum_hard_mass_chart_ordinals": (),
        **values,
    }
    provisional = object.__new__(StreamingSharedChartAtlasCandidateOutcomeV1)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    return StreamingSharedChartAtlasCandidateOutcomeV1(
        **values,  # type: ignore[arg-type]
        receipt_sha256=_digest(
            _CANDIDATE_RECEIPT_DOMAIN, provisional._payload_without_hash()
        ),
    )


def _revise_candidate(
    candidate: StreamingSharedChartAtlasCandidateOutcomeV1,
    **updates: object,
) -> StreamingSharedChartAtlasCandidateOutcomeV1:
    values = {
        field.name: getattr(candidate, field.name)
        for field in fields(StreamingSharedChartAtlasCandidateOutcomeV1)
        if field.name != "receipt_sha256"
    }
    values.update(updates)
    return _make_candidate_outcome(**values)


@dataclass(frozen=True, slots=True)
class StreamingSharedChartAtlasOccupancyDiagnosisV1:
    config: SharedChartNumericsConfig
    resources: StreamingChartResourceConfig
    config_sha256: str
    resource_config_receipt_sha256: str
    kernel_receipt_sha256: str
    selection_kernel_receipt_sha256: str
    fit_role: str
    fit_panel_binding_sha256: str
    fit_row_catalog_sha256: str
    fit_context_catalog_sha256: str
    fit_balanced_row_weight_catalog_sha256: str
    fit_mode_ids: tuple[str, ...]
    fit_family_ids: tuple[str, ...]
    fit_row_count_by_family: tuple[int, ...]
    spool_artifact_sha256s: tuple[str, ...]
    spool_mode_identity_sha256s: tuple[str, ...]
    spool_tensor_catalog_sha256s: tuple[str, ...]
    mode_count: int
    row_count_per_mode: int
    total_row_count: int
    total_balanced_mass_binary64: str
    compute_device: str
    compute_dtype: str
    candidate: StreamingSharedChartAtlasCandidateOutcomeV1
    receipt_sha256: str

    def __post_init__(self) -> None:
        _validate_numeric_policy(self.config)
        _validate_resources(self.resources)
        if self.config_sha256 != _config_sha256(self.config):
            raise ValueError("diagnosis config hash differs")
        if self.resource_config_receipt_sha256 != self.resources.receipt_sha256:
            raise ValueError("diagnosis resource config hash differs")
        for label in (
            "kernel_receipt_sha256",
            "selection_kernel_receipt_sha256",
            "fit_panel_binding_sha256",
            "fit_row_catalog_sha256",
            "fit_context_catalog_sha256",
            "fit_balanced_row_weight_catalog_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        expected_kernel = _streaming.streaming_chart_kernel_receipt(self.resources)[
            "receipt_sha256"
        ]
        if self.kernel_receipt_sha256 != expected_kernel:
            raise ValueError("diagnosis kernel receipt differs")
        expected_selection_kernel = (
            streaming_shared_chart_atlas_selection_kernel_receipt_v1(
                self.resources
            )["receipt_sha256"]
        )
        if self.selection_kernel_receipt_sha256 != expected_selection_kernel:
            raise ValueError("diagnosis selection-kernel receipt differs")
        if self.fit_role != FIT_ROLE:
            raise ValueError(f"diagnosis fit_role must be {FIT_ROLE}")
        if type(self.mode_count) is not int or self.mode_count <= 0:
            raise ValueError("diagnosis mode_count must be positive")
        if type(self.row_count_per_mode) is not int or self.row_count_per_mode <= 0:
            raise ValueError("diagnosis row_count_per_mode must be positive")
        if (
            not isinstance(self.fit_mode_ids, tuple)
            or any(
                not isinstance(value, str) or not value
                for value in self.fit_mode_ids
            )
            or self.fit_mode_ids != tuple(sorted(self.fit_mode_ids))
            or len(self.fit_mode_ids) != self.mode_count
            or len(set(self.fit_mode_ids)) != self.mode_count
            or not isinstance(self.fit_family_ids, tuple)
            or any(
                not isinstance(value, str) or not value
                for value in self.fit_family_ids
            )
            or self.fit_family_ids != tuple(sorted(self.fit_family_ids))
            or not self.fit_family_ids
            or len(set(self.fit_family_ids)) != len(self.fit_family_ids)
            or not isinstance(self.fit_row_count_by_family, tuple)
            or len(self.fit_row_count_by_family) != len(self.fit_family_ids)
            or any(
                type(value) is not int or value <= 0
                for value in self.fit_row_count_by_family
            )
            or sum(self.fit_row_count_by_family) != self.row_count_per_mode
        ):
            raise ValueError("diagnosis mode/family axes are invalid")
        if (
            type(self.total_row_count) is not int
            or self.total_row_count
            != self.mode_count * self.row_count_per_mode
        ):
            raise ValueError("diagnosis total row inventory differs")
        for label, values in (
            ("spool_artifact_sha256s", self.spool_artifact_sha256s),
            ("spool_mode_identity_sha256s", self.spool_mode_identity_sha256s),
            ("spool_tensor_catalog_sha256s", self.spool_tensor_catalog_sha256s),
        ):
            if not isinstance(values, tuple) or len(values) != self.mode_count:
                raise ValueError(f"{label} must bind every mode")
            for value in values:
                _require_sha256(value, label=label)
        if len(set(self.spool_mode_identity_sha256s)) != self.mode_count:
            raise ValueError("diagnosis mode identity hashes must be unique")
        expected_mode_identities = tuple(
            streaming_shared_chart_atlas_mode_identity_sha256_v1(
                mode_id, ordinal
            )
            for ordinal, mode_id in enumerate(self.fit_mode_ids)
        )
        if self.spool_mode_identity_sha256s != expected_mode_identities:
            raise ValueError("diagnosis mode identity/hash binding differs")
        total_mass = _parse_binary64(
            self.total_balanced_mass_binary64,
            label="total_balanced_mass_binary64",
            nonnegative=True,
        )
        if not math.isclose(total_mass, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("diagnosis total balanced mass must close to one")
        if (
            self.compute_device != STREAMING_SHARED_CHART_ATLAS_COMPUTE_DEVICE_V1
            or self.compute_dtype != STREAMING_SHARED_CHART_ATLAS_COMPUTE_DTYPE_V1
        ):
            raise ValueError("diagnosis compute policy differs")
        if (
            not isinstance(self.candidate, StreamingSharedChartAtlasCandidateOutcomeV1)
            or self.candidate.chart_count != self.config.chart_count
            or self.candidate.mode_count != self.mode_count
            or self.candidate.mode_ids != self.fit_mode_ids
            or self.candidate.family_ids != self.fit_family_ids
            or self.candidate.fit_row_count_by_family
            != self.fit_row_count_by_family
            or self.candidate.config_sha256 != self.config_sha256
            or self.candidate.total_row_count != self.total_row_count
            or self.candidate.total_balanced_mass_binary64
            != self.total_balanced_mass_binary64
        ):
            raise ValueError("diagnosis candidate binding differs")
        if any(
            item.predicate_id == "finite_symmetric_covariance"
            and item.component_ordinal is not None
            and item.component_ordinal >= self.config.hidden_width**2
            for item in self.candidate.gate_evidence
        ):
            raise ValueError("diagnosis covariance evidence component is out of range")
        if self.candidate.terminal_phase == "atlas_admissible":
            expected_solver = _expected_fit_solver_receipt_sha256(
                config=self.config,
                resources=self.resources,
                row_count_per_mode=self.row_count_per_mode,
                mode_count=self.mode_count,
                fit_context_catalog_sha256=self.fit_context_catalog_sha256,
                fit_balanced_row_weight_catalog_sha256=(
                    self.fit_balanced_row_weight_catalog_sha256
                ),
            )
            if self.candidate.fit_solver_receipt_sha256 != expected_solver:
                raise ValueError("diagnosis selected solver receipt differs")
        if self.receipt_sha256 != _digest(
            _DIAGNOSIS_RECEIPT_DOMAIN, self._payload_without_hash()
        ):
            raise ValueError("occupancy diagnosis receipt hash differs")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.state_dict(),
            "compute_device": self.compute_device,
            "compute_dtype": self.compute_dtype,
            "config": self.config.payload(),
            "config_sha256": self.config_sha256,
            "fit_balanced_row_weight_catalog_sha256": (
                self.fit_balanced_row_weight_catalog_sha256
            ),
            "fit_context_catalog_sha256": self.fit_context_catalog_sha256,
            "fit_mode_ids": list(self.fit_mode_ids),
            "fit_family_ids": list(self.fit_family_ids),
            "fit_row_count_by_family": list(self.fit_row_count_by_family),
            "fit_panel_binding_sha256": self.fit_panel_binding_sha256,
            "fit_role": self.fit_role,
            "fit_row_catalog_sha256": self.fit_row_catalog_sha256,
            "kernel_receipt_sha256": self.kernel_receipt_sha256,
            "selection_kernel_receipt_sha256": (
                self.selection_kernel_receipt_sha256
            ),
            "mode_count": self.mode_count,
            "resource_config_receipt_sha256": (
                self.resource_config_receipt_sha256
            ),
            "resources": self.resources.payload(),
            "row_count_per_mode": self.row_count_per_mode,
            "schema": (
                STREAMING_SHARED_CHART_ATLAS_OCCUPANCY_DIAGNOSIS_SCHEMA_V1
            ),
            "spool_artifact_sha256s": list(self.spool_artifact_sha256s),
            "spool_mode_identity_sha256s": list(
                self.spool_mode_identity_sha256s
            ),
            "spool_tensor_catalog_sha256s": list(
                self.spool_tensor_catalog_sha256s
            ),
            "total_balanced_mass_binary64": self.total_balanced_mass_binary64,
            "total_row_count": self.total_row_count,
        }

    def state_dict(self) -> dict[str, object]:
        """Return the full canonical, prompt/token/tensor-free public receipt."""

        return {**self._payload_without_hash(), "receipt_sha256": self.receipt_sha256}


def _revise_diagnosis_candidate(
    diagnosis: StreamingSharedChartAtlasOccupancyDiagnosisV1,
    candidate: StreamingSharedChartAtlasCandidateOutcomeV1,
) -> StreamingSharedChartAtlasOccupancyDiagnosisV1:
    values = {
        field.name: getattr(diagnosis, field.name)
        for field in fields(StreamingSharedChartAtlasOccupancyDiagnosisV1)
        if field.name != "receipt_sha256"
    }
    values["candidate"] = candidate
    provisional = object.__new__(StreamingSharedChartAtlasOccupancyDiagnosisV1)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    return StreamingSharedChartAtlasOccupancyDiagnosisV1(
        **values,  # type: ignore[arg-type]
        receipt_sha256=_digest(
            _DIAGNOSIS_RECEIPT_DOMAIN, provisional._payload_without_hash()
        ),
    )


@dataclass(frozen=True, slots=True)
class StreamingSharedChartAtlasSelectionReceiptV1:
    selection_outcome: str
    selection_kernel_receipt_sha256: str
    candidate_order: tuple[int, ...]
    attempted_diagnoses: tuple[StreamingSharedChartAtlasOccupancyDiagnosisV1, ...]
    attempted_chart_counts: tuple[int, ...]
    unattempted_chart_counts: tuple[int, ...]
    selected_chart_count: int | None
    selected_source_feature_width: int | None
    selected_atlas_sha256: str | None
    selected_fit_solver_receipt_sha256: str | None
    selection_policy_id: str
    claim_tier: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_order, tuple)
            or any(type(value) is not int for value in self.candidate_order)
            or self.candidate_order
            != STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1
        ):
            raise ValueError("selection candidate order differs")
        if (
            not isinstance(self.attempted_diagnoses, tuple)
            or not self.attempted_diagnoses
            or any(
                not isinstance(
                    item, StreamingSharedChartAtlasOccupancyDiagnosisV1
                )
                for item in self.attempted_diagnoses
            )
        ):
            raise ValueError("selection must contain attempted diagnoses")
        expected_attempted = self.candidate_order[: len(self.attempted_diagnoses)]
        if (
            not isinstance(self.attempted_chart_counts, tuple)
            or any(
                type(value) is not int for value in self.attempted_chart_counts
            )
            or self.attempted_chart_counts != expected_attempted
            or tuple(
                item.candidate.chart_count for item in self.attempted_diagnoses
            )
            != expected_attempted
        ):
            raise ValueError("selection attempted prefix differs")
        if (
            not isinstance(self.unattempted_chart_counts, tuple)
            or any(
                type(value) is not int for value in self.unattempted_chart_counts
            )
            or self.unattempted_chart_counts
            != self.candidate_order[len(self.attempted_diagnoses) :]
        ):
            raise ValueError("selection unattempted suffix differs")
        first = self.attempted_diagnoses[0]
        if first.config.chart_count != 8:
            raise ValueError("selection first diagnosis is not the K8 template")
        _validate_numeric_policy(first.config)
        expected_selection_kernel = (
            streaming_shared_chart_atlas_selection_kernel_receipt_v1(
                first.resources
            )["receipt_sha256"]
        )
        _require_sha256(
            self.selection_kernel_receipt_sha256,
            label="selection_kernel_receipt_sha256",
        )
        if self.selection_kernel_receipt_sha256 != expected_selection_kernel:
            raise ValueError("selection kernel receipt differs")
        lineage_fields = (
            "resource_config_receipt_sha256",
            "kernel_receipt_sha256",
            "fit_role",
            "fit_panel_binding_sha256",
            "fit_row_catalog_sha256",
            "fit_context_catalog_sha256",
            "fit_balanced_row_weight_catalog_sha256",
            "fit_mode_ids",
            "fit_family_ids",
            "fit_row_count_by_family",
            "spool_artifact_sha256s",
            "spool_mode_identity_sha256s",
            "spool_tensor_catalog_sha256s",
            "mode_count",
            "row_count_per_mode",
            "total_row_count",
            "total_balanced_mass_binary64",
            "compute_device",
            "compute_dtype",
        )
        for chart_count, item in zip(expected_attempted, self.attempted_diagnoses):
            if (
                any(
                    getattr(item, field) != getattr(first, field)
                    for field in lineage_fields
                )
                or item.resources != first.resources
                or item.config != replace(first.config, chart_count=chart_count)
                or item.selection_kernel_receipt_sha256
                != self.selection_kernel_receipt_sha256
            ):
                raise ValueError("selection diagnoses do not share one input lineage")
        _require_sha256(self.receipt_sha256, label="receipt_sha256")
        if self.selection_policy_id != (
            "largest_intrinsic_atlas_admissible_k_descending_8_to_1_v1"
        ):
            raise ValueError("selection policy identity differs")
        if self.selection_outcome == SELECTED_ATLAS_OUTCOME:
            if (
                type(self.selected_chart_count) is not int
                or self.selected_chart_count != expected_attempted[-1]
                or type(self.selected_source_feature_width) is not int
                or self.selected_source_feature_width
                != self.selected_chart_count
                * (STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1 + 1)
                or any(
                    item.candidate.outcome == ADMISSIBLE_OUTCOME
                    for item in self.attempted_diagnoses[:-1]
                )
                or self.attempted_diagnoses[-1].candidate.outcome
                != ADMISSIBLE_OUTCOME
                or self.attempted_diagnoses[-1].candidate.terminal_phase
                != "atlas_admissible"
            ):
                raise ValueError("selection did not stop at first full atlas pass")
            _require_sha256(self.selected_atlas_sha256, label="selected_atlas_sha256")
            _require_sha256(
                self.selected_fit_solver_receipt_sha256,
                label="selected_fit_solver_receipt_sha256",
            )
            selected_candidate = self.attempted_diagnoses[-1].candidate
            expected_solver = _expected_fit_solver_receipt_sha256(
                config=self.attempted_diagnoses[-1].config,
                resources=self.attempted_diagnoses[-1].resources,
                row_count_per_mode=self.attempted_diagnoses[-1].row_count_per_mode,
                mode_count=self.attempted_diagnoses[-1].mode_count,
                fit_context_catalog_sha256=(
                    self.attempted_diagnoses[-1].fit_context_catalog_sha256
                ),
                fit_balanced_row_weight_catalog_sha256=(
                    self.attempted_diagnoses[-1]
                    .fit_balanced_row_weight_catalog_sha256
                ),
            )
            if (
                selected_candidate.atlas_sha256 != self.selected_atlas_sha256
                or selected_candidate.fit_solver_receipt_sha256
                != self.selected_fit_solver_receipt_sha256
                or self.selected_fit_solver_receipt_sha256 != expected_solver
            ):
                raise ValueError("selection candidate/atlas solver digest differs")
            expected_tier = (
                "single_global_rank16_chart_transport_geometry"
                if self.selected_chart_count == 1
                else "selected_shared_rank16_chart_transport_geometry"
            )
            if self.claim_tier != expected_tier:
                raise ValueError("selection claim tier differs")
        elif self.selection_outcome == A_ATLAS_NO_ADMISSIBLE_K:
            if (
                len(self.attempted_diagnoses) != len(self.candidate_order)
                or self.unattempted_chart_counts
                or any(
                    item.candidate.outcome == ADMISSIBLE_OUTCOME
                    for item in self.attempted_diagnoses
                )
                or any(
                    value is not None
                    for value in (
                        self.selected_chart_count,
                        self.selected_source_feature_width,
                        self.selected_atlas_sha256,
                        self.selected_fit_solver_receipt_sha256,
                    )
                )
                or self.claim_tier != "no_admissible_shared_chart_atlas"
            ):
                raise ValueError("no-admissible-K selection evidence differs")
        else:
            raise ValueError("selection outcome is invalid")
        if self.receipt_sha256 != _digest(
            _SELECTION_RECEIPT_DOMAIN, self._payload_without_hash()
        ):
            raise ValueError("selection receipt hash differs")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "selection_outcome": self.selection_outcome,
            "selection_kernel_receipt_sha256": (
                self.selection_kernel_receipt_sha256
            ),
            "attempted_chart_counts": list(self.attempted_chart_counts),
            "attempted_diagnoses": [
                item.state_dict() for item in self.attempted_diagnoses
            ],
            "candidate_order": list(self.candidate_order),
            "claim_tier": self.claim_tier,
            "schema": STREAMING_SHARED_CHART_ATLAS_SELECTION_SCHEMA_V1,
            "selected_atlas_sha256": self.selected_atlas_sha256,
            "selected_chart_count": self.selected_chart_count,
            "selected_fit_solver_receipt_sha256": (
                self.selected_fit_solver_receipt_sha256
            ),
            "selected_source_feature_width": self.selected_source_feature_width,
            "selection_policy_id": self.selection_policy_id,
            "unattempted_chart_counts": list(self.unattempted_chart_counts),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class StreamingSharedChartAtlasSelectionResultV1:
    atlas: SharedChartAtlas | None
    receipt: StreamingSharedChartAtlasSelectionReceiptV1

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, StreamingSharedChartAtlasSelectionReceiptV1):
            raise TypeError("selection result receipt has the wrong type")
        if self.receipt.selection_outcome == A_ATLAS_NO_ADMISSIBLE_K:
            if self.atlas is not None:
                raise ValueError("no-admissible-K result cannot contain an atlas")
        else:
            if not isinstance(self.atlas, SharedChartAtlas):
                raise TypeError("selected result atlas must be SharedChartAtlas")
            if (
                self.atlas.atlas_sha256 != self.receipt.selected_atlas_sha256
                or self.atlas.fit_solver_receipt_sha256
                != self.receipt.selected_fit_solver_receipt_sha256
                or self.atlas.config.chart_count != self.receipt.selected_chart_count
                or self.atlas.config.source_feature_width
                != self.receipt.selected_source_feature_width
            ):
                raise ValueError("selection result atlas/receipt binding differs")


@dataclass(slots=True)
class _CenterStageResult:
    diagnosis: StreamingSharedChartAtlasOccupancyDiagnosisV1
    aligned: tuple[RawModeRowSpool, ...]
    config: SharedChartNumericsConfig
    resources: StreamingChartResourceConfig
    estimate: StreamingChartResourceEstimate
    base_weights: Tensor
    global_mean: Tensor
    centers: Tensor | None
    counts: Tensor | None
    occupancy_weights: Tensor | None


def _fit_catalogs(
    aligned: tuple[RawModeRowSpool, ...], base_weights: Tensor
) -> tuple[str, str]:
    context_catalog = _reference._digest(
        _reference._ATLAS_CONTEXT_CATALOG_DOMAIN,
        {
            "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
            "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
            "modes": [
                {
                    "context_values_sha256": (
                        item.field_semantic_sha256s["context_values"]
                    ),
                    "mode_id": item.mode_id,
                }
                for item in aligned
            ],
        },
    )
    weight_sha256 = _reference._tensor_sha256(base_weights)
    weight_catalog = _reference._digest(
        _reference._ATLAS_WEIGHT_CATALOG_DOMAIN,
        {
            "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
            "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
            "mode_weights": [
                {"mode_id": item.mode_id, "weight_sha256": weight_sha256}
                for item in aligned
            ],
        },
    )
    return context_catalog, weight_catalog


def _make_diagnosis(
    *,
    config: SharedChartNumericsConfig,
    resources: StreamingChartResourceConfig,
    aligned: tuple[RawModeRowSpool, ...],
    context_catalog: str,
    weight_catalog: str,
    total_mass: Tensor,
    candidate: StreamingSharedChartAtlasCandidateOutcomeV1,
) -> StreamingSharedChartAtlasOccupancyDiagnosisV1:
    kernel_sha256 = _streaming.streaming_chart_kernel_receipt(resources)[
        "receipt_sha256"
    ]
    family_ids = tuple(
        sorted({key.family_id for key in aligned[0].row_keys})
    )
    fit_row_count_by_family = tuple(
        sum(key.family_id == family_id for key in aligned[0].row_keys)
        for family_id in family_ids
    )
    values = {
        "config": config,
        "resources": resources,
        "config_sha256": _config_sha256(config),
        "resource_config_receipt_sha256": resources.receipt_sha256,
        "kernel_receipt_sha256": kernel_sha256,
        "selection_kernel_receipt_sha256": (
            streaming_shared_chart_atlas_selection_kernel_receipt_v1(
                resources
            )["receipt_sha256"]
        ),
        "fit_role": FIT_ROLE,
        "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
        "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
        "fit_context_catalog_sha256": context_catalog,
        "fit_balanced_row_weight_catalog_sha256": weight_catalog,
        "fit_mode_ids": tuple(item.mode_id for item in aligned),
        "fit_family_ids": family_ids,
        "fit_row_count_by_family": fit_row_count_by_family,
        "spool_artifact_sha256s": tuple(item.artifact_sha256 for item in aligned),
        "spool_mode_identity_sha256s": tuple(
            streaming_shared_chart_atlas_mode_identity_sha256_v1(
                item.mode_id, ordinal
            )
            for ordinal, item in enumerate(aligned)
        ),
        "spool_tensor_catalog_sha256s": tuple(
            item.tensor_catalog_sha256 for item in aligned
        ),
        "mode_count": len(aligned),
        "row_count_per_mode": len(aligned[0].row_keys),
        "total_row_count": len(aligned) * len(aligned[0].row_keys),
        "total_balanced_mass_binary64": _binary64(total_mass),
        "compute_device": STREAMING_SHARED_CHART_ATLAS_COMPUTE_DEVICE_V1,
        "compute_dtype": STREAMING_SHARED_CHART_ATLAS_COMPUTE_DTYPE_V1,
        "candidate": candidate,
    }
    provisional = object.__new__(StreamingSharedChartAtlasOccupancyDiagnosisV1)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    return StreamingSharedChartAtlasOccupancyDiagnosisV1(
        **values,
        receipt_sha256=_digest(
            _DIAGNOSIS_RECEIPT_DOMAIN, provisional._payload_without_hash()
        ),
    )


def _run_center_stage(
    raw_spools: Sequence[RawModeRowSpool],
    *,
    config: SharedChartNumericsConfig,
    resources: StreamingChartResourceConfig,
) -> _CenterStageResult:
    _validate_numeric_policy(config)
    _validate_resources(resources)
    aligned = _streaming._aligned_raw_spools(
        raw_spools, resources=resources, expected_role=FIT_ROLE
    )
    if aligned[0].hidden_width != config.hidden_width:
        raise ValueError("raw spool width disagrees with atlas config")
    estimate = _streaming.estimate_streaming_chart_resources(
        row_count_per_mode=len(aligned[0].row_keys),
        mode_count=len(aligned),
        config=config,
        resources=resources,
    )
    _streaming._enforce_resident_budget(
        resources,
        required_bytes=estimate.atlas_peak_resident_bytes,
        operation="streaming atlas selection",
    )
    rows_per_mode = len(aligned[0].row_keys)
    total_rows = len(aligned) * rows_per_mode
    family_ids = tuple(
        sorted({key.family_id for key in aligned[0].row_keys})
    )
    family_count = len(family_ids)
    fit_row_count_by_family = tuple(
        sum(key.family_id == family_id for key in aligned[0].row_keys)
        for family_id in family_ids
    )
    base_weights = _streaming._balanced_row_weights(
        aligned[0].row_keys, mode_count=len(aligned)
    )
    context_catalog, weight_catalog = _fit_catalogs(aligned, base_weights)

    weighted_sum = torch.zeros(config.hidden_width, dtype=torch.float64)
    weight_sum = torch.zeros((), dtype=torch.float64)
    for _, contexts, weights in _streaming._iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        weighted_sum += (weights[:, None] * contexts).sum(dim=0)
        weight_sum += weights.sum()
    global_mean = weighted_sum / weight_sum

    selected_indices: list[int] = []
    selected_centers: list[Tensor] = []
    seed_receipts: list[StreamingSharedChartAtlasSeedReceiptV1] = []
    best_score = -math.inf
    best_index = -1
    best_center: Tensor | None = None
    mean_center = global_mean[None, :]
    for global_start, contexts, weights in _streaming._iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        scores = weights * _streaming._norm_identity_distances(
            contexts, mean_center
        )[:, 0]
        local = int(torch.argmax(scores).item())
        score = float(scores[local])
        if score > best_score:
            best_score = score
            best_index = global_start + local
            best_center = contexts[local].clone()
    if best_center is None:
        raise RuntimeError("streaming farthest initialization found no row")
    selected_indices.append(best_index)
    selected_centers.append(best_center)
    seed_receipts.append(
        _make_seed_receipt(
            seed_ordinal=0,
            global_row_ordinal=best_index,
            rows_per_mode=rows_per_mode,
            aligned=aligned,
            score=best_score,
            center=best_center,
        )
    )
    while len(selected_centers) < config.chart_count:
        centers_now = torch.stack(selected_centers)
        best_score = -math.inf
        best_index = -1
        best_center = None
        selected_set = set(selected_indices)
        for global_start, contexts, weights in _streaming._iter_context_chunks(
            aligned,
            base_weights=base_weights,
            chunk_size=resources.atlas_chunk_size,
        ):
            distances = _streaming._norm_identity_distances(
                contexts, centers_now
            )
            scores = weights * distances.min(dim=1).values
            for selected_index in selected_set:
                local = selected_index - global_start
                if 0 <= local < len(scores):
                    scores[local] = -1.0
            local = int(torch.argmax(scores).item())
            score = float(scores[local])
            if score > best_score:
                best_score = score
                best_index = global_start + local
                best_center = contexts[local].clone()
        if best_center is None or best_score <= 0.0:
            partial_centers = torch.stack(selected_centers)
            candidate = _make_candidate_outcome(
                chart_count=config.chart_count,
                source_feature_width=config.source_feature_width,
                total_row_count=total_rows,
                mode_count=len(aligned),
                family_count=family_count,
                mode_ids=tuple(item.mode_id for item in aligned),
                family_ids=family_ids,
                fit_row_count_by_family=fit_row_count_by_family,
                total_balanced_mass_binary64=_binary64(weight_sum),
                config_sha256=_config_sha256(config),
                seed_receipts=tuple(seed_receipts),
                initialized_centers_sha256=_reference._tensor_sha256(
                    partial_centers
                ),
                initialized_center_sha256s=_center_sha256s(partial_centers),
                lloyd_pass_receipts=(),
                final_sort_permutation=(),
                sorted_centers_sha256=None,
                final_hard_occupancy_count=(),
                final_hard_occupancy_mass_binary64=(),
                minimum_required_hard_count=config.tangent_rank + 1,
                minimum_hard_occupancy_mass_binary64=_binary64(
                    config.minimum_hard_occupancy_fraction
                ),
                outcome=CENTER_INITIALIZATION_DEGENERATE,
                terminal_phase="center_initialization",
                failing_chart_ordinals=(len(selected_centers),),
            )
            diagnosis = _make_diagnosis(
                config=config,
                resources=resources,
                aligned=aligned,
                context_catalog=context_catalog,
                weight_catalog=weight_catalog,
                total_mass=weight_sum,
                candidate=candidate,
            )
            return _CenterStageResult(
                diagnosis,
                aligned,
                config,
                resources,
                estimate,
                base_weights,
                global_mean,
                None,
                None,
                None,
            )
        seed_ordinal = len(selected_centers)
        selected_indices.append(best_index)
        selected_centers.append(best_center)
        seed_receipts.append(
            _make_seed_receipt(
                seed_ordinal=seed_ordinal,
                global_row_ordinal=best_index,
                rows_per_mode=rows_per_mode,
                aligned=aligned,
                score=best_score,
                center=best_center,
            )
        )
    centers = torch.stack(selected_centers)
    initialized_centers_sha256 = _reference._tensor_sha256(centers)

    pass_receipts: list[StreamingSharedChartAtlasPassReceiptV1] = []
    for pass_index in range(config.lloyd_iteration_count):
        centers_before_assignment = centers
        sums = torch.zeros_like(centers)
        occupancy = torch.zeros(config.chart_count, dtype=torch.float64)
        counts = torch.zeros(config.chart_count, dtype=torch.int64)
        for _, contexts, weights in _streaming._iter_context_chunks(
            aligned,
            base_weights=base_weights,
            chunk_size=resources.atlas_chunk_size,
        ):
            assignments = torch.argmin(
                _streaming._norm_identity_distances(contexts, centers), dim=1
            )
            counts += torch.bincount(assignments, minlength=config.chart_count)
            for chart in range(config.chart_count):
                mask = assignments == chart
                if bool(mask.any()):
                    chart_weights = weights[mask]
                    sums[chart] += (
                        chart_weights[:, None] * contexts[mask]
                    ).sum(dim=0)
                    occupancy[chart] += chart_weights.sum()
        if bool((occupancy <= 0.0).any()):
            pass_receipts.append(
                _make_pass_receipt(
                    pass_ordinal=pass_index + 1,
                    centers_before_assignment=centers_before_assignment,
                    counts=counts,
                    masses=occupancy,
                    updated_centers=None,
                )
            )
            candidate = _make_candidate_outcome(
                chart_count=config.chart_count,
                source_feature_width=config.source_feature_width,
                total_row_count=total_rows,
                mode_count=len(aligned),
                family_count=family_count,
                mode_ids=tuple(item.mode_id for item in aligned),
                family_ids=family_ids,
                fit_row_count_by_family=fit_row_count_by_family,
                total_balanced_mass_binary64=_binary64(weight_sum),
                config_sha256=_config_sha256(config),
                seed_receipts=tuple(seed_receipts),
                initialized_centers_sha256=initialized_centers_sha256,
                initialized_center_sha256s=tuple(
                    seed.selected_center_sha256 for seed in seed_receipts
                ),
                lloyd_pass_receipts=tuple(pass_receipts),
                final_sort_permutation=(),
                sorted_centers_sha256=None,
                final_hard_occupancy_count=(),
                final_hard_occupancy_mass_binary64=(),
                minimum_required_hard_count=config.tangent_rank + 1,
                minimum_hard_occupancy_mass_binary64=_binary64(
                    config.minimum_hard_occupancy_fraction
                ),
                outcome=EMPTY_CHART_OUTCOME,
                terminal_phase=f"lloyd_pass_{pass_index + 1}",
                failing_chart_ordinals=tuple(
                    int(index)
                    for index in torch.nonzero(
                        occupancy <= 0.0, as_tuple=False
                    ).flatten()
                ),
            )
            diagnosis = _make_diagnosis(
                config=config,
                resources=resources,
                aligned=aligned,
                context_catalog=context_catalog,
                weight_catalog=weight_catalog,
                total_mass=weight_sum,
                candidate=candidate,
            )
            return _CenterStageResult(
                diagnosis,
                aligned,
                config,
                resources,
                estimate,
                base_weights,
                global_mean,
                None,
                None,
                None,
            )
        centers = sums / occupancy[:, None]
        pass_receipts.append(
            _make_pass_receipt(
                pass_ordinal=pass_index + 1,
                centers_before_assignment=centers_before_assignment,
                counts=counts,
                masses=occupancy,
                updated_centers=centers,
            )
        )

    order = tuple(
        sorted(
            range(config.chart_count),
            key=lambda index: tuple(float(value) for value in centers[index]),
        )
    )
    centers = centers[list(order)].contiguous()
    sorted_centers_sha256 = _reference._tensor_sha256(centers)
    counts = torch.zeros(config.chart_count, dtype=torch.int64)
    occupancy_weights = torch.zeros(config.chart_count, dtype=torch.float64)
    count_by_chart_and_mode = torch.zeros(
        (config.chart_count, len(aligned)), dtype=torch.int64
    )
    mass_by_chart_and_mode = torch.zeros(
        (config.chart_count, len(aligned)), dtype=torch.float64
    )
    count_by_chart_and_family = torch.zeros(
        (config.chart_count, family_count), dtype=torch.int64
    )
    mass_by_chart_and_family = torch.zeros(
        (config.chart_count, family_count), dtype=torch.float64
    )
    family_ordinal = {family: index for index, family in enumerate(family_ids)}
    family_by_row = torch.tensor(
        [family_ordinal[key.family_id] for key in aligned[0].row_keys],
        dtype=torch.int64,
    )
    for global_start, contexts, weights in _streaming._iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        mode_ordinal, row_start = divmod(global_start, rows_per_mode)
        family_chunk = family_by_row[row_start : row_start + len(contexts)]
        distances = _streaming._norm_identity_distances(contexts, centers)
        assignments = torch.argmin(distances, dim=1)
        counts += torch.bincount(assignments, minlength=config.chart_count)
        for chart in range(config.chart_count):
            mask = assignments == chart
            if bool(mask.any()):
                chart_mass = weights[mask].sum()
                occupancy_weights[chart] += chart_mass
                count_by_chart_and_mode[chart, mode_ordinal] += int(mask.sum())
                mass_by_chart_and_mode[chart, mode_ordinal] += chart_mass
                for family in range(family_count):
                    family_mask = mask & (family_chunk == family)
                    if bool(family_mask.any()):
                        count_by_chart_and_family[chart, family] += int(
                            family_mask.sum()
                        )
                        mass_by_chart_and_family[chart, family] += weights[
                            family_mask
                        ].sum()
    count_mode_tuple = tuple(
        tuple(int(value) for value in row)
        for row in count_by_chart_and_mode.tolist()
    )
    mass_mode_tuple = tuple(
        tuple(_binary64(value) for value in row)
        for row in mass_by_chart_and_mode
    )
    count_family_tuple = tuple(
        tuple(int(value) for value in row)
        for row in count_by_chart_and_family.tolist()
    )
    mass_family_tuple = tuple(
        tuple(_binary64(value) for value in row)
        for row in mass_by_chart_and_family
    )
    composition_catalog_sha256 = _composition_catalog_sha256(
        chart_count=config.chart_count,
        mode_count=len(aligned),
        family_count=family_count,
        mode_ids=tuple(item.mode_id for item in aligned),
        family_ids=family_ids,
        fit_row_count_by_family=fit_row_count_by_family,
        count_by_chart_and_mode=count_mode_tuple,
        mass_by_chart_and_mode=mass_mode_tuple,
        count_by_chart_and_family=count_family_tuple,
        mass_by_chart_and_family=mass_family_tuple,
    )
    minimum_observed_mass = float(occupancy_weights.min())
    minimum_mass_ordinals = tuple(
        int(index)
        for index in torch.nonzero(
            occupancy_weights == minimum_observed_mass, as_tuple=False
        ).flatten()
    )
    count_failures = tuple(
        int(index)
        for index in torch.nonzero(
            counts < (config.tangent_rank + 1), as_tuple=False
        ).flatten()
    )
    mass_failures = tuple(
        int(index)
        for index in torch.nonzero(
            occupancy_weights < config.minimum_hard_occupancy_fraction,
            as_tuple=False,
        ).flatten()
    )
    if count_failures:
        outcome = INSUFFICIENT_HARD_COUNT_OUTCOME
        terminal_phase = "final_hard_count_gate"
        failures = count_failures
    elif mass_failures:
        outcome = INSUFFICIENT_HARD_MASS_OUTCOME
        terminal_phase = "final_hard_mass_gate"
        failures = mass_failures
    else:
        outcome = ADMISSIBLE_OUTCOME
        terminal_phase = "occupancy_admissible"
        failures = ()
    candidate = _make_candidate_outcome(
        chart_count=config.chart_count,
        source_feature_width=config.source_feature_width,
        total_row_count=total_rows,
        mode_count=len(aligned),
        family_count=family_count,
        mode_ids=tuple(item.mode_id for item in aligned),
        family_ids=family_ids,
        fit_row_count_by_family=fit_row_count_by_family,
        total_balanced_mass_binary64=_binary64(weight_sum),
        config_sha256=_config_sha256(config),
        seed_receipts=tuple(seed_receipts),
        initialized_centers_sha256=initialized_centers_sha256,
        initialized_center_sha256s=tuple(
            seed.selected_center_sha256 for seed in seed_receipts
        ),
        lloyd_pass_receipts=tuple(pass_receipts),
        final_sort_permutation=order,
        final_pre_sort_center_sha256s=(
            pass_receipts[-1].updated_centers_sha256s
        ),
        sorted_centers_sha256=sorted_centers_sha256,
        sorted_center_sha256s=_center_sha256s(centers),
        final_hard_occupancy_count=tuple(int(value) for value in counts.tolist()),
        final_hard_occupancy_mass_binary64=tuple(
            _binary64(value) for value in occupancy_weights
        ),
        final_hard_count_by_chart_and_mode=count_mode_tuple,
        final_hard_mass_binary64_by_chart_and_mode=mass_mode_tuple,
        final_hard_count_by_chart_and_family=count_family_tuple,
        final_hard_mass_binary64_by_chart_and_family=mass_family_tuple,
        composition_catalog_sha256=composition_catalog_sha256,
        minimum_required_hard_count=config.tangent_rank + 1,
        minimum_hard_occupancy_mass_binary64=_binary64(
            config.minimum_hard_occupancy_fraction
        ),
        minimum_observed_hard_mass_binary64=_binary64(
            minimum_observed_mass
        ),
        minimum_hard_mass_margin_binary64=_binary64(
            minimum_observed_mass
            - config.minimum_hard_occupancy_fraction
        ),
        minimum_hard_mass_chart_ordinals=minimum_mass_ordinals,
        outcome=outcome,
        terminal_phase=terminal_phase,
        failing_chart_ordinals=failures,
    )
    diagnosis = _make_diagnosis(
        config=config,
        resources=resources,
        aligned=aligned,
        context_catalog=context_catalog,
        weight_catalog=weight_catalog,
        total_mass=weight_sum,
        candidate=candidate,
    )
    return _CenterStageResult(
        diagnosis,
        aligned,
        config,
        resources,
        estimate,
        base_weights,
        global_mean,
        centers,
        counts,
        occupancy_weights,
    )


def diagnose_shared_chart_atlas_occupancy_streaming(
    raw_spools: Sequence[RawModeRowSpool],
    *,
    config: SharedChartNumericsConfig = SharedChartNumericsConfig(),
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> StreamingSharedChartAtlasOccupancyDiagnosisV1:
    """Run the standalone exact K=8 center-stage diagnostic."""

    _validate_numeric_policy(config)
    if config.chart_count != STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1[0]:
        raise ValueError("standalone occupancy diagnosis requires the K8 template")

    return _run_center_stage(
        raw_spools, config=config, resources=resources
    ).diagnosis


def validate_streaming_shared_chart_atlas_diagnosis_against_spools_v1(
    diagnosis: StreamingSharedChartAtlasOccupancyDiagnosisV1,
    raw_spools: Sequence[RawModeRowSpool],
    *,
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> None:
    """Validate receipt preimages against authenticated spools before cleanup.

    This read-only boundary proves the public input catalogs and every selected
    seed-center hash while the raw context rows still exist.  It retains and
    returns no tensor data.
    """

    if not isinstance(
        diagnosis, StreamingSharedChartAtlasOccupancyDiagnosisV1
    ):
        raise TypeError("diagnosis has the wrong type")
    _validate_resources(resources)
    if diagnosis.resources != resources:
        raise ValueError("diagnosis/spool validation resources differ")
    aligned = _streaming._aligned_raw_spools(
        raw_spools, resources=resources, expected_role=FIT_ROLE
    )
    if diagnosis.config.hidden_width != aligned[0].hidden_width:
        raise ValueError("diagnosis hidden width differs from authenticated spools")
    base_weights = _streaming._balanced_row_weights(
        aligned[0].row_keys, mode_count=len(aligned)
    )
    context_catalog, weight_catalog = _fit_catalogs(aligned, base_weights)
    family_ids = tuple(
        sorted({key.family_id for key in aligned[0].row_keys})
    )
    fit_row_count_by_family = tuple(
        sum(key.family_id == family_id for key in aligned[0].row_keys)
        for family_id in family_ids
    )
    total_mass = torch.zeros((), dtype=torch.float64)
    for _ in aligned:
        for start in range(0, len(base_weights), resources.atlas_chunk_size):
            total_mass += base_weights[
                start : start + resources.atlas_chunk_size
            ].sum()
    expected = {
        "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
        "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
        "fit_context_catalog_sha256": context_catalog,
        "fit_balanced_row_weight_catalog_sha256": weight_catalog,
        "fit_mode_ids": tuple(item.mode_id for item in aligned),
        "fit_family_ids": family_ids,
        "fit_row_count_by_family": fit_row_count_by_family,
        "spool_artifact_sha256s": tuple(
            item.artifact_sha256 for item in aligned
        ),
        "spool_mode_identity_sha256s": tuple(
            streaming_shared_chart_atlas_mode_identity_sha256_v1(
                item.mode_id, ordinal
            )
            for ordinal, item in enumerate(aligned)
        ),
        "spool_tensor_catalog_sha256s": tuple(
            item.tensor_catalog_sha256 for item in aligned
        ),
        "mode_count": len(aligned),
        "row_count_per_mode": len(aligned[0].row_keys),
        "total_row_count": len(aligned) * len(aligned[0].row_keys),
        "total_balanced_mass_binary64": _binary64(total_mass),
        "kernel_receipt_sha256": _streaming.streaming_chart_kernel_receipt(
            resources
        )["receipt_sha256"],
        "selection_kernel_receipt_sha256": (
            streaming_shared_chart_atlas_selection_kernel_receipt_v1(
                resources
            )["receipt_sha256"]
        ),
    }
    for label, value in expected.items():
        if getattr(diagnosis, label) != value:
            raise ValueError(f"diagnosis {label} differs from authenticated spools")
    rows_per_mode = len(aligned[0].row_keys)
    selected_contexts: list[Tensor] = []
    try:
        for seed in diagnosis.candidate.seed_receipts:
            mode_ordinal, row_ordinal = divmod(
                seed.selected_global_row_ordinal, rows_per_mode
            )
            spool = aligned[mode_ordinal]
            key = spool.row_keys[row_ordinal]
            if (
                seed.selected_mode_ordinal != mode_ordinal
                or seed.selected_row_ordinal != row_ordinal
                or seed.mode_id != spool.mode_id
                or seed.family_id != key.family_id
                or seed.example_id != key.example_id
                or seed.prediction_query_position
                != key.prediction_query_position
                or seed.target_label_position != key.target_label_position
            ):
                raise ValueError(
                    "diagnosis seed identity differs from authenticated spool"
                )
            context = spool._store.read_slice(
                "context_values", row_ordinal, row_ordinal + 1
            ).to(device="cpu", dtype=torch.float64)[0].contiguous()
            if _reference._tensor_sha256(context) != seed.selected_center_sha256:
                raise ValueError(
                    "diagnosis seed center hash differs from authenticated spool"
                )
            selected_contexts.append(context)
        initialized = torch.stack(selected_contexts).contiguous()
        if (
            _reference._tensor_sha256(initialized)
            != diagnosis.candidate.initialized_centers_sha256
        ):
            raise ValueError(
                "diagnosis initialized centers differ from authenticated seeds"
            )
    finally:
        selected_contexts.clear()


def validate_streaming_shared_chart_atlas_occupancy_diagnosis_against_raw_spools_v1(
    diagnosis: StreamingSharedChartAtlasOccupancyDiagnosisV1,
    raw_spools: Sequence[RawModeRowSpool],
    *,
    config: SharedChartNumericsConfig,
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> None:
    """Named A-diagnostic wrapper with an explicit caller config binding."""

    if not isinstance(config, SharedChartNumericsConfig):
        raise TypeError("config must be SharedChartNumericsConfig")
    if diagnosis.config != config:
        raise ValueError("diagnosis/caller config differs")
    validate_streaming_shared_chart_atlas_diagnosis_against_spools_v1(
        diagnosis, raw_spools, resources=resources
    )


def _construct_selected_atlas(state: _CenterStageResult) -> SharedChartAtlas | None:
    diagnosis = state.diagnosis
    if (
        diagnosis.candidate.outcome != ADMISSIBLE_OUTCOME
        or diagnosis.candidate.terminal_phase != "occupancy_admissible"
    ):
        raise ValueError("only an admissible center state can construct an atlas")
    if state.centers is None or state.counts is None or state.occupancy_weights is None:
        raise RuntimeError("admissible center state omitted private tensors")
    aligned = state.aligned
    config = state.config
    resources = state.resources
    base_weights = state.base_weights
    centers = state.centers
    counts = state.counts
    occupancy_weights = state.occupancy_weights

    def reject(
        outcome: str,
        terminal_phase: str,
        failing_chart_ordinals: tuple[int, ...],
        **trace_updates: object,
    ) -> None:
        revised = _revise_candidate(
            state.diagnosis.candidate,
            outcome=outcome,
            terminal_phase=terminal_phase,
            failing_chart_ordinals=failing_chart_ordinals,
            **trace_updates,
        )
        state.diagnosis = _revise_diagnosis_candidate(
            state.diagnosis, revised
        )

    gate_evidence_rows: list[StreamingSharedChartAtlasGateEvidenceV1] = []

    radius_numerators = torch.zeros(config.chart_count, dtype=torch.float64)
    context_reference_sum = torch.zeros((), dtype=torch.float64)
    global_variance_sum = torch.zeros((), dtype=torch.float64)
    for _, contexts, weights in _streaming._iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        distances = _streaming._norm_identity_distances(contexts, centers)
        assignments = torch.argmin(distances, dim=1)
        for chart in range(config.chart_count):
            mask = assignments == chart
            if bool(mask.any()):
                chart_weights = weights[mask]
                radius_numerators[chart] += (
                    chart_weights * distances[mask, chart]
                ).sum()
        context_reference_sum += (
            weights * contexts.square().sum(dim=1)
        ).sum()
        global_variance_sum += (
            weights
            * _streaming._norm_identity_distances(
                contexts, state.global_mean[None, :]
            )[:, 0]
        ).sum()
    context_reference = max(float(context_reference_sum), 1.0e-30)
    radius_floor = config.radius_floor_fraction * context_reference
    radii = torch.maximum(
        radius_numerators / occupancy_weights,
        torch.full(
            (config.chart_count,), radius_floor, dtype=torch.float64
        ),
    )
    invalid_radius = tuple(
        int(index)
        for index in torch.nonzero(
            (~torch.isfinite(radii)) | (radii <= 0.0), as_tuple=False
        ).flatten()
    )
    radius_evidence = tuple(
            _make_gate_evidence(
                gate_id="radius_contract",
                predicate_id="finite_and_positive_radius_squared",
                status=(
                    "NONFINITE"
                    if not math.isfinite(float(radii[index]))
                    else "PASS"
                    if float(radii[index]) > 0.0
                    else "NONPOSITIVE"
                ),
                chart_ordinal=index,
                component_ordinal=None,
                global_row_ordinal=None,
                observed_binary64=(_binary64_bits(radii[index]),),
                bound_binary64=(_binary64_bits(0.0),),
                signed_margin_binary64=_binary64_bits(radii[index]),
            )
            for index in range(config.chart_count)
        )
    gate_evidence_rows.extend(radius_evidence)
    if invalid_radius:
        reject(
            RADIUS_CONTRACT_FAILED,
            "radius_contract",
            invalid_radius,
            gate_evidence=tuple(gate_evidence_rows),
        )
        return None
    radius_squared_sha256 = _reference._tensor_sha256(radii)

    covariance_numerators = torch.zeros(
        (config.chart_count, config.hidden_width, config.hidden_width),
        dtype=torch.float64,
    )
    soft_occupancy = torch.zeros(config.chart_count, dtype=torch.float64)
    minimum_membership = torch.full(
        (config.chart_count,), math.inf, dtype=torch.float64
    )
    minimum_membership_global_rows = torch.full(
        (config.chart_count,), -1, dtype=torch.int64
    )
    worst_simplex_residual = -1.0
    worst_simplex_row_sum = 1.0
    worst_simplex_global_row = -1
    for global_start, contexts, weights in _streaming._iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        memberships = _streaming._streaming_memberships(
            contexts=contexts,
            centers=centers,
            radius_squared=radii,
            temperature=config.membership_temperature,
        )
        nonfinite = torch.nonzero(
            ~torch.isfinite(memberships), as_tuple=False
        )
        negative = torch.nonzero(memberships < 0.0, as_tuple=False)
        row_sums = memberships.sum(dim=1)
        simplex_ok = torch.isclose(
            row_sums,
            torch.ones(len(memberships), dtype=torch.float64),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        simplex_failure = torch.nonzero(~simplex_ok, as_tuple=False).flatten()
        if len(nonfinite) or len(negative) or len(simplex_failure):
            if len(nonfinite):
                row_ordinal, chart_ordinal = (
                    int(value) for value in nonfinite[0]
                )
                observed = memberships[row_ordinal, chart_ordinal]
                failures = (chart_ordinal,)
                evidence = _make_gate_evidence(
                    gate_id="soft_occupancy_contract",
                    predicate_id="finite_membership",
                    status="NONFINITE",
                    chart_ordinal=chart_ordinal,
                    component_ordinal=None,
                    global_row_ordinal=global_start + row_ordinal,
                    observed_binary64=(_binary64_bits(observed),),
                    bound_binary64=(),
                    signed_margin_binary64=None,
                )
            elif len(negative):
                row_ordinal, chart_ordinal = (
                    int(value) for value in negative[0]
                )
                observed = memberships[row_ordinal, chart_ordinal]
                failures = (chart_ordinal,)
                evidence = _make_gate_evidence(
                    gate_id="soft_occupancy_contract",
                    predicate_id="nonnegative_membership",
                    status="NEGATIVE",
                    chart_ordinal=chart_ordinal,
                    component_ordinal=None,
                    global_row_ordinal=global_start + row_ordinal,
                    observed_binary64=(_binary64_bits(observed),),
                    bound_binary64=(_binary64_bits(0.0),),
                    signed_margin_binary64=_binary64_bits(observed),
                )
            else:
                row_ordinal = int(simplex_failure[0])
                residual = abs(float(row_sums[row_ordinal]) - 1.0)
                bound = 2.0e-12
                failures = tuple(range(config.chart_count))
                evidence = _make_gate_evidence(
                    gate_id="soft_occupancy_contract",
                    predicate_id="membership_simplex_absolute_residual",
                    status="RESIDUAL_EXCEEDS_BOUND",
                    chart_ordinal=None,
                    component_ordinal=None,
                    global_row_ordinal=global_start + row_ordinal,
                    observed_binary64=(
                        _binary64_bits(row_sums[row_ordinal]),
                        _binary64_bits(residual),
                    ),
                    bound_binary64=(_binary64_bits(bound),),
                    signed_margin_binary64=_binary64_bits(bound - residual),
                )
            reject(
                SOFT_OCCUPANCY_CONTRACT_FAILED,
                "soft_occupancy_contract",
                failures,
                radius_squared_sha256=radius_squared_sha256,
                gate_evidence=tuple(gate_evidence_rows + [evidence]),
            )
            return None
        for chart in range(config.chart_count):
            local = int(torch.argmin(memberships[:, chart]))
            value = float(memberships[local, chart])
            if value < float(minimum_membership[chart]):
                minimum_membership[chart] = value
                minimum_membership_global_rows[chart] = global_start + local
        simplex_residuals = torch.abs(row_sums - 1.0)
        local_worst = int(torch.argmax(simplex_residuals))
        local_residual = float(simplex_residuals[local_worst])
        if local_residual > worst_simplex_residual:
            worst_simplex_residual = local_residual
            worst_simplex_row_sum = float(row_sums[local_worst])
            worst_simplex_global_row = global_start + local_worst
        for chart in range(config.chart_count):
            chart_weights = weights * memberships[:, chart]
            centered = contexts - centers[chart]
            covariance_numerators[chart] += centered.T @ (
                chart_weights[:, None] * centered
            )
            soft_occupancy[chart] += chart_weights.sum()
    gate_evidence_rows.extend(
        _make_gate_evidence(
            gate_id="soft_occupancy_contract",
            predicate_id="nonnegative_membership",
            status="PASS",
            chart_ordinal=chart,
            component_ordinal=None,
            global_row_ordinal=int(minimum_membership_global_rows[chart]),
            observed_binary64=(_binary64_bits(minimum_membership[chart]),),
            bound_binary64=(_binary64_bits(0.0),),
            signed_margin_binary64=_binary64_bits(minimum_membership[chart]),
        )
        for chart in range(config.chart_count)
    )
    simplex_bound = 2.0e-12
    gate_evidence_rows.append(
        _make_gate_evidence(
            gate_id="soft_occupancy_contract",
            predicate_id="membership_simplex_absolute_residual",
            status="PASS",
            chart_ordinal=None,
            component_ordinal=None,
            global_row_ordinal=worst_simplex_global_row,
            observed_binary64=(
                _binary64_bits(worst_simplex_row_sum),
                _binary64_bits(worst_simplex_residual),
            ),
            bound_binary64=(_binary64_bits(simplex_bound),),
            signed_margin_binary64=_binary64_bits(
                simplex_bound - worst_simplex_residual
            ),
        )
    )
    invalid_soft_occupancy = tuple(
        int(index)
        for index in torch.nonzero(
            (~torch.isfinite(soft_occupancy)) | (soft_occupancy <= 0.0),
            as_tuple=False,
        ).flatten()
    )
    soft_occupancy_evidence = tuple(
            _make_gate_evidence(
                gate_id="soft_occupancy_contract",
                predicate_id="finite_and_positive_soft_occupancy",
                status=(
                    "NONFINITE"
                    if not math.isfinite(float(soft_occupancy[index]))
                    else "PASS"
                    if float(soft_occupancy[index]) > 0.0
                    else "NONPOSITIVE"
                ),
                chart_ordinal=index,
                component_ordinal=None,
                global_row_ordinal=None,
                observed_binary64=(_binary64_bits(soft_occupancy[index]),),
                bound_binary64=(_binary64_bits(0.0),),
                signed_margin_binary64=_binary64_bits(soft_occupancy[index]),
            )
            for index in range(config.chart_count)
        )
    gate_evidence_rows.extend(soft_occupancy_evidence)
    if invalid_soft_occupancy:
        reject(
            SOFT_OCCUPANCY_CONTRACT_FAILED,
            "soft_occupancy_contract",
            invalid_soft_occupancy,
            radius_squared_sha256=radius_squared_sha256,
            gate_evidence=tuple(gate_evidence_rows),
        )
        return None
    soft_occupancy_mass_binary64 = tuple(
        _binary64(value) for value in soft_occupancy
    )
    covariances = covariance_numerators / soft_occupancy[:, None, None]
    covariance_finite = tuple(
        bool(torch.isfinite(covariances[chart]).all())
        for chart in range(config.chart_count)
    )
    covariance_symmetry_summaries = []
    for chart, finite in enumerate(covariance_finite):
        if not finite:
            covariance_symmetry_summaries.append(None)
            continue
        covariance_symmetry_summaries.append(
            _covariance_symmetry_summary(covariances[chart])
        )
    covariance_symmetry_summaries = tuple(covariance_symmetry_summaries)
    invalid_covariance = tuple(
        chart
        for chart in range(config.chart_count)
        if not covariance_finite[chart]
        or not covariance_symmetry_summaries[chart][0]
    )
    covariance_evidence = []
    for chart in range(config.chart_count):
        if covariance_finite[chart]:
            summary = covariance_symmetry_summaries[chart]
            if summary is None:
                raise RuntimeError("finite covariance omitted symmetry summary")
            (
                passed,
                component,
                cij,
                cji,
                residual,
                scale,
                bound_value,
                margin_value,
            ) = summary
            observed = (
                _binary64_bits(cij),
                _binary64_bits(cji),
                _binary64_bits(residual),
                _binary64_bits(scale),
            )
            bound = (_binary64_bits(bound_value),)
            margin = _binary64_bits(margin_value)
            status = "PASS" if passed else "RESIDUAL_EXCEEDS_BOUND"
        else:
            covariance = covariances[chart]
            first_nonfinite = torch.nonzero(
                ~torch.isfinite(covariance), as_tuple=False
            )[0]
            row, column = (int(value) for value in first_nonfinite)
            component = row * config.hidden_width + column
            observed = (
                _binary64_bits(covariance[row, column]),
                _binary64_bits(covariance[column, row]),
                _binary64_bits(float("nan")),
                _binary64_bits(float("nan")),
            )
            bound = (_binary64_bits(float("nan")),)
            margin = _binary64_bits(float("nan"))
            status = "NONFINITE"
        covariance_evidence.append(
            _make_gate_evidence(
                gate_id="covariance_contract",
                predicate_id="finite_symmetric_covariance",
                status=status,
                chart_ordinal=chart,
                component_ordinal=component,
                global_row_ordinal=None,
                observed_binary64=observed,
                bound_binary64=bound,
                signed_margin_binary64=margin,
            )
        )
    gate_evidence_rows.extend(covariance_evidence)
    if invalid_covariance:
        reject(
            COVARIANCE_CONTRACT_FAILED,
            "covariance_contract",
            invalid_covariance,
            radius_squared_sha256=radius_squared_sha256,
            soft_occupancy_mass_binary64=soft_occupancy_mass_binary64,
            gate_evidence=tuple(gate_evidence_rows),
        )
        return None
    covariance_sha256s = tuple(
        _reference._tensor_sha256(covariances[chart])
        for chart in range(config.chart_count)
    )
    context_variance_reference = max(
        float(global_variance_sum) / config.hidden_width,
        1.0e-30,
    )
    variance_floor = config.variance_floor_fraction * context_variance_reference
    bases = torch.empty(
        (config.chart_count, config.hidden_width, config.tangent_rank),
        dtype=torch.float64,
    )
    tangent_variance = torch.empty(
        (config.chart_count, config.tangent_rank), dtype=torch.float64
    )
    leading_eigenvalues_sha256s: list[str] = []
    for chart in range(config.chart_count):
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances[chart])
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        if not bool(torch.isfinite(eigenvalues).all()) or not bool(
            torch.isfinite(eigenvectors).all()
        ):
            raise RuntimeError(
                "finite selected covariance produced a non-finite eigendecomposition"
            )
        leading_eigenvalues_sha256 = _reference._tensor_sha256(
            eigenvalues[: config.tangent_rank + 1]
        )
        for ordinal in range(config.tangent_rank):
            following = eigenvalues[ordinal + 1]
            scale = max(
                1.0,
                abs(float(eigenvalues[ordinal])),
                abs(float(following)),
            )
            bound = config.eigenvalue_tie_relative_tolerance * scale
            gap = abs(float(eigenvalues[ordinal] - following))
            margin = gap - bound
            evidence = _make_gate_evidence(
                gate_id="eigengap_orientation",
                predicate_id="strict_relative_eigengap",
                status=("PASS" if margin > 0.0 else "NONPOSITIVE_MARGIN"),
                chart_ordinal=chart,
                component_ordinal=ordinal,
                global_row_ordinal=None,
                observed_binary64=(
                    _binary64_bits(eigenvalues[ordinal]),
                    _binary64_bits(following),
                    _binary64_bits(gap),
                    _binary64_bits(scale),
                ),
                bound_binary64=(_binary64_bits(bound),),
                signed_margin_binary64=_binary64_bits(margin),
            )
            gate_evidence_rows.append(evidence)
            if gap <= bound:
                reached_spectrum_sha256s = tuple(
                    leading_eigenvalues_sha256s
                    + (
                        [leading_eigenvalues_sha256]
                        if ordinal == config.tangent_rank - 1
                        else []
                    )
                )
                reject(
                    EIGENGAP_ORIENTATION_UNDEFINED,
                    "eigengap_orientation",
                    (chart,),
                    radius_squared_sha256=radius_squared_sha256,
                    soft_occupancy_mass_binary64=soft_occupancy_mass_binary64,
                    covariance_sha256s=covariance_sha256s,
                    leading_eigenvalues_sha256s=reached_spectrum_sha256s,
                    gate_evidence=tuple(gate_evidence_rows),
                )
                return None
        leading_eigenvalues_sha256s.append(leading_eigenvalues_sha256)
        basis = _reference._canonicalize_vector_signs(
            eigenvectors[:, : config.tangent_rank]
        )
        bases[chart] = basis
        tangent_variance[chart] = torch.clamp(
            eigenvalues[: config.tangent_rank], min=variance_floor
        )

    identity = torch.eye(config.tangent_rank, dtype=torch.float64)
    basis_grams = tuple(
        bases[chart].T @ bases[chart]
        for chart in range(config.chart_count)
    )
    basis_evidence = []
    invalid_basis_list = []
    for chart in range(config.chart_count):
        gram = basis_grams[chart]
        allowed = 1.0e-9 + 1.0e-9 * torch.abs(identity)
        residual = torch.abs(gram - identity)
        if not bool(torch.isfinite(gram).all()):
            first_nonfinite = torch.nonzero(
                ~torch.isfinite(gram), as_tuple=False
            )[0]
            row, column = (int(value) for value in first_nonfinite)
            flat = row * config.tangent_rank + column
            margin_value = float("nan")
            passed = False
            status = "NONFINITE"
        else:
            margins = allowed - residual
            flat = int(torch.argmin(margins))
            row, column = divmod(flat, config.tangent_rank)
            margin_value = float(margins[row, column])
            passed = bool((margins >= 0.0).all())
            status = "PASS" if passed else "RESIDUAL_EXCEEDS_BOUND"
        if not passed:
            invalid_basis_list.append(chart)
        basis_evidence.append(
            _make_gate_evidence(
                gate_id="spd_contract",
                predicate_id="basis_orthonormality",
                status=status,
                chart_ordinal=chart,
                component_ordinal=flat,
                global_row_ordinal=None,
                observed_binary64=(
                    _binary64_bits(gram[row, column]),
                    _binary64_bits(identity[row, column]),
                    _binary64_bits(residual[row, column]),
                ),
                bound_binary64=(_binary64_bits(allowed[row, column]),),
                signed_margin_binary64=_binary64_bits(margin_value),
            )
        )
    invalid_basis = tuple(invalid_basis_list)
    gate_evidence_rows.extend(basis_evidence)
    if invalid_basis:
        reject(
            SPD_CONTRACT_FAILED,
            "spd_contract",
            invalid_basis,
            radius_squared_sha256=radius_squared_sha256,
            soft_occupancy_mass_binary64=soft_occupancy_mass_binary64,
            covariance_sha256s=covariance_sha256s,
            leading_eigenvalues_sha256s=tuple(
                leading_eigenvalues_sha256s
            ),
            gate_evidence=tuple(gate_evidence_rows),
        )
        return None

    orthogonal_numerators = torch.zeros(
        config.chart_count, dtype=torch.float64
    )
    for _, contexts, weights in _streaming._iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        memberships = _streaming._streaming_memberships(
            contexts=contexts,
            centers=centers,
            radius_squared=radii,
            temperature=config.membership_temperature,
        )
        for chart in range(config.chart_count):
            centered = contexts - centers[chart]
            projection = centered @ bases[chart]
            residual_square = (
                centered.square().sum(dim=1)
                - projection.square().sum(dim=1)
            ).clamp_min(0.0)
            orthogonal_numerators[chart] += (
                weights * memberships[:, chart] * residual_square
            ).sum()
    alpha = torch.empty(config.chart_count, dtype=torch.float64)
    for chart in range(config.chart_count):
        orthogonal_variance = float(
            orthogonal_numerators[chart]
            / (
                soft_occupancy[chart]
                * (config.hidden_width - config.tangent_rank)
            )
        )
        alpha[chart] = 1.0 / max(orthogonal_variance, variance_floor)
    invalid_spd = tuple(
        chart
        for chart in range(config.chart_count)
        if (
            not bool(torch.isfinite(tangent_variance[chart]).all())
            or bool((tangent_variance[chart] <= 0.0).any())
            or not math.isfinite(float(alpha[chart]))
            or float(alpha[chart]) <= 0.0
        )
    )
    tangent_evidence = []
    alpha_evidence = []
    for chart in range(config.chart_count):
        finite_tangent = torch.where(
            torch.isfinite(tangent_variance[chart]),
            tangent_variance[chart],
            torch.full_like(tangent_variance[chart], -math.inf),
        )
        component = int(torch.argmin(finite_tangent))
        tangent_observed = tangent_variance[chart, component]
        tangent_evidence.append(
            _make_gate_evidence(
                gate_id="spd_contract",
                predicate_id="finite_and_positive_tangent_variance",
                status=(
                    "NONFINITE"
                    if not math.isfinite(float(tangent_observed))
                    else "PASS"
                    if float(tangent_observed) > 0.0
                    else "NONPOSITIVE"
                ),
                chart_ordinal=chart,
                component_ordinal=component,
                global_row_ordinal=None,
                observed_binary64=(_binary64_bits(tangent_observed),),
                bound_binary64=(_binary64_bits(0.0),),
                signed_margin_binary64=_binary64_bits(tangent_observed),
            )
        )
        alpha_evidence.append(
            _make_gate_evidence(
                gate_id="spd_contract",
                predicate_id="finite_and_positive_orthogonal_alpha",
                status=(
                    "NONFINITE"
                    if not math.isfinite(float(alpha[chart]))
                    else "PASS"
                    if float(alpha[chart]) > 0.0
                    else "NONPOSITIVE"
                ),
                chart_ordinal=chart,
                component_ordinal=None,
                global_row_ordinal=None,
                observed_binary64=(_binary64_bits(alpha[chart]),),
                bound_binary64=(_binary64_bits(0.0),),
                signed_margin_binary64=_binary64_bits(alpha[chart]),
            )
        )
    gate_evidence_rows.extend(tangent_evidence)
    gate_evidence_rows.extend(alpha_evidence)
    if invalid_spd:
        reject(
            SPD_CONTRACT_FAILED,
            "spd_contract",
            invalid_spd,
            radius_squared_sha256=radius_squared_sha256,
            soft_occupancy_mass_binary64=soft_occupancy_mass_binary64,
            covariance_sha256s=covariance_sha256s,
            leading_eigenvalues_sha256s=tuple(
                leading_eigenvalues_sha256s
            ),
            tangent_bases_sha256=_reference._tensor_sha256(bases),
            gate_evidence=tuple(gate_evidence_rows),
        )
        return None

    fit_solver_receipt_sha256 = _expected_fit_solver_receipt_sha256(
        config=config,
        resources=resources,
        row_count_per_mode=len(aligned[0].row_keys),
        mode_count=len(aligned),
        fit_context_catalog_sha256=state.diagnosis.fit_context_catalog_sha256,
        fit_balanced_row_weight_catalog_sha256=(
            state.diagnosis.fit_balanced_row_weight_catalog_sha256
        ),
    )
    values = {
        "config": config,
        "fit_role": FIT_ROLE,
        "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
        "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
        "fit_mode_ids": tuple(item.mode_id for item in aligned),
        "fit_context_catalog_sha256": state.diagnosis.fit_context_catalog_sha256,
        "fit_balanced_row_weight_catalog_sha256": (
            state.diagnosis.fit_balanced_row_weight_catalog_sha256
        ),
        "fit_solver_receipt_sha256": fit_solver_receipt_sha256,
        "centers": centers,
        "radius_squared": radii,
        "tangent_bases": bases,
        "tangent_variance": tangent_variance,
        "orthogonal_alpha": alpha,
        "hard_occupancy_count": counts,
        "hard_occupancy_weight": occupancy_weights,
    }
    provisional = object.__new__(SharedChartAtlas)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "atlas_sha256", "0" * 64)
    atlas = SharedChartAtlas(
        **values,
        atlas_sha256=_reference._atlas_sha256(provisional),
    )
    revised = _revise_candidate(
        state.diagnosis.candidate,
        outcome=ADMISSIBLE_OUTCOME,
        terminal_phase="atlas_admissible",
        failing_chart_ordinals=(),
        radius_squared_sha256=radius_squared_sha256,
        soft_occupancy_mass_binary64=soft_occupancy_mass_binary64,
        covariance_sha256s=covariance_sha256s,
        leading_eigenvalues_sha256s=tuple(leading_eigenvalues_sha256s),
        tangent_bases_sha256=_reference._tensor_sha256(bases),
        tangent_variance_sha256=_reference._tensor_sha256(tangent_variance),
        orthogonal_alpha_sha256=_reference._tensor_sha256(alpha),
        atlas_sha256=atlas.atlas_sha256,
        fit_solver_receipt_sha256=atlas.fit_solver_receipt_sha256,
        gate_evidence=tuple(gate_evidence_rows),
    )
    state.diagnosis = _revise_diagnosis_candidate(
        state.diagnosis, revised
    )
    return atlas


def select_and_fit_shared_chart_atlas_streaming(
    raw_spools: Sequence[RawModeRowSpool],
    *,
    config: SharedChartNumericsConfig = SharedChartNumericsConfig(),
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> StreamingSharedChartAtlasSelectionResultV1:
    """Select the largest fully intrinsic-admissible K in ``8..1``.

    The input config is the immutable K=8 template.  Only ``chart_count`` is
    replaced while descending the candidate order.  The first admissible
    full context-only atlas is frozen before any transport, application, or
    graph outcome is observed.  Only exact typed intrinsic rejections permit
    descent; engineering and freeze-integrity failures abort by propagation.
    """

    _validate_numeric_policy(config)
    _validate_resources(resources)
    if config.chart_count != STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1[0]:
        raise ValueError("selection config must be the frozen K=8 template")
    attempts: list[_CenterStageResult] = []
    selected: _CenterStageResult | None = None
    atlas: SharedChartAtlas | None = None
    for chart_count in STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1:
        candidate_config = replace(config, chart_count=chart_count)
        state = _run_center_stage(
            raw_spools, config=candidate_config, resources=resources
        )
        attempts.append(state)
        if state.diagnosis.candidate.outcome == ADMISSIBLE_OUTCOME:
            candidate_atlas = _construct_selected_atlas(state)
            if candidate_atlas is not None:
                selected = state
                atlas = candidate_atlas
                break
    attempted_diagnoses = tuple(item.diagnosis for item in attempts)
    attempted_chart_counts = tuple(
        item.candidate.chart_count for item in attempted_diagnoses
    )
    common_values = {
        "selection_kernel_receipt_sha256": (
            streaming_shared_chart_atlas_selection_kernel_receipt_v1(
                resources
            )["receipt_sha256"]
        ),
        "candidate_order": STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1,
        "attempted_diagnoses": attempted_diagnoses,
        "attempted_chart_counts": attempted_chart_counts,
        "unattempted_chart_counts": (
            STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1[
                len(attempted_chart_counts) :
            ]
        ),
        "selection_policy_id": (
            "largest_intrinsic_atlas_admissible_k_descending_8_to_1_v1"
        ),
    }
    if selected is None or atlas is None:
        values = {
            **common_values,
            "selection_outcome": A_ATLAS_NO_ADMISSIBLE_K,
            "selected_chart_count": None,
            "selected_source_feature_width": None,
            "selected_atlas_sha256": None,
            "selected_fit_solver_receipt_sha256": None,
            "claim_tier": "no_admissible_shared_chart_atlas",
        }
    else:
        values = {
            **common_values,
            "selection_outcome": SELECTED_ATLAS_OUTCOME,
            "selected_chart_count": atlas.config.chart_count,
            "selected_source_feature_width": atlas.config.source_feature_width,
            "selected_atlas_sha256": atlas.atlas_sha256,
            "selected_fit_solver_receipt_sha256": atlas.fit_solver_receipt_sha256,
            "claim_tier": (
                "single_global_rank16_chart_transport_geometry"
                if atlas.config.chart_count == 1
                else "selected_shared_rank16_chart_transport_geometry"
            ),
        }
    provisional = object.__new__(StreamingSharedChartAtlasSelectionReceiptV1)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    receipt = StreamingSharedChartAtlasSelectionReceiptV1(
        **values,
        receipt_sha256=_digest(
            _SELECTION_RECEIPT_DOMAIN, provisional._payload_without_hash()
        ),
    )
    return StreamingSharedChartAtlasSelectionResultV1(atlas=atlas, receipt=receipt)


def _parse_seed(value: object) -> StreamingSharedChartAtlasSeedReceiptV1:
    expected = {
        "example_id",
        "family_id",
        "mode_id",
        "mode_identity_sha256",
        "prediction_query_position",
        "receipt_sha256",
        "row_identity_sha256",
        "schema",
        "seed_ordinal",
        "selected_center_sha256",
        "selected_global_row_ordinal",
        "selected_mode_ordinal",
        "selected_row_ordinal",
        "target_label_position",
        "weighted_farthest_score_binary64",
    }
    row = _require_exact_fields(value, expected, label="seed receipt")
    if row["schema"] != STREAMING_SHARED_CHART_ATLAS_SEED_SCHEMA_V1:
        raise ValueError("seed receipt schema differs")
    return StreamingSharedChartAtlasSeedReceiptV1(
        seed_ordinal=row["seed_ordinal"],  # type: ignore[arg-type]
        selected_global_row_ordinal=row["selected_global_row_ordinal"],  # type: ignore[arg-type]
        selected_mode_ordinal=row["selected_mode_ordinal"],  # type: ignore[arg-type]
        selected_row_ordinal=row["selected_row_ordinal"],  # type: ignore[arg-type]
        mode_id=row["mode_id"],  # type: ignore[arg-type]
        family_id=row["family_id"],  # type: ignore[arg-type]
        example_id=row["example_id"],  # type: ignore[arg-type]
        prediction_query_position=row["prediction_query_position"],  # type: ignore[arg-type]
        target_label_position=row["target_label_position"],  # type: ignore[arg-type]
        mode_identity_sha256=row["mode_identity_sha256"],  # type: ignore[arg-type]
        row_identity_sha256=row["row_identity_sha256"],  # type: ignore[arg-type]
        weighted_farthest_score_binary64=row[
            "weighted_farthest_score_binary64"
        ],  # type: ignore[arg-type]
        selected_center_sha256=row["selected_center_sha256"],  # type: ignore[arg-type]
        receipt_sha256=row["receipt_sha256"],  # type: ignore[arg-type]
    )


def _parse_gate_evidence(
    value: object,
) -> StreamingSharedChartAtlasGateEvidenceV1:
    expected = {
        "bound_binary64",
        "chart_ordinal",
        "component_ordinal",
        "gate_id",
        "global_row_ordinal",
        "observed_binary64",
        "predicate_id",
        "receipt_sha256",
        "schema",
        "signed_margin_binary64",
        "status",
    }
    row = _require_exact_fields(value, expected, label="gate evidence")
    if row["schema"] != STREAMING_SHARED_CHART_ATLAS_GATE_EVIDENCE_SCHEMA_V1:
        raise ValueError("gate evidence schema differs")
    for label in ("bound_binary64", "observed_binary64"):
        if not isinstance(row[label], list):
            raise ValueError(f"gate evidence {label} must be a list")
    return StreamingSharedChartAtlasGateEvidenceV1(
        gate_id=row["gate_id"],  # type: ignore[arg-type]
        predicate_id=row["predicate_id"],  # type: ignore[arg-type]
        status=row["status"],  # type: ignore[arg-type]
        chart_ordinal=row["chart_ordinal"],  # type: ignore[arg-type]
        component_ordinal=row["component_ordinal"],  # type: ignore[arg-type]
        global_row_ordinal=row["global_row_ordinal"],  # type: ignore[arg-type]
        observed_binary64=tuple(row["observed_binary64"]),  # type: ignore[arg-type]
        bound_binary64=tuple(row["bound_binary64"]),  # type: ignore[arg-type]
        signed_margin_binary64=row["signed_margin_binary64"],  # type: ignore[arg-type]
        receipt_sha256=row["receipt_sha256"],  # type: ignore[arg-type]
    )


def _parse_pass(value: object) -> StreamingSharedChartAtlasPassReceiptV1:
    expected = {
        "centers_before_assignment_sha256",
        "centers_before_assignment_sha256s",
        "empty_chart_ordinals",
        "hard_occupancy_count",
        "hard_occupancy_mass_binary64",
        "pass_ordinal",
        "receipt_sha256",
        "schema",
        "updated_centers_sha256",
        "updated_centers_sha256s",
    }
    row = _require_exact_fields(value, expected, label="pass receipt")
    if row["schema"] != STREAMING_SHARED_CHART_ATLAS_PASS_SCHEMA_V1:
        raise ValueError("pass receipt schema differs")
    for label in (
        "empty_chart_ordinals",
        "hard_occupancy_count",
        "hard_occupancy_mass_binary64",
        "centers_before_assignment_sha256s",
        "updated_centers_sha256s",
    ):
        if not isinstance(row[label], list):
            raise ValueError(f"pass receipt {label} must be a list")
    return StreamingSharedChartAtlasPassReceiptV1(
        pass_ordinal=row["pass_ordinal"],  # type: ignore[arg-type]
        centers_before_assignment_sha256=row[
            "centers_before_assignment_sha256"
        ],  # type: ignore[arg-type]
        centers_before_assignment_sha256s=tuple(
            row["centers_before_assignment_sha256s"]
        ),  # type: ignore[arg-type]
        hard_occupancy_count=tuple(row["hard_occupancy_count"]),  # type: ignore[arg-type]
        hard_occupancy_mass_binary64=tuple(
            row["hard_occupancy_mass_binary64"]
        ),  # type: ignore[arg-type]
        empty_chart_ordinals=tuple(row["empty_chart_ordinals"]),  # type: ignore[arg-type]
        updated_centers_sha256=row["updated_centers_sha256"],  # type: ignore[arg-type]
        updated_centers_sha256s=tuple(
            row["updated_centers_sha256s"]
        ),  # type: ignore[arg-type]
        receipt_sha256=row["receipt_sha256"],  # type: ignore[arg-type]
    )


def _parse_candidate(
    value: object,
) -> StreamingSharedChartAtlasCandidateOutcomeV1:
    expected = {
        "chart_count",
        "mode_count",
        "family_count",
        "mode_ids",
        "family_ids",
        "fit_row_count_by_family",
        "config_sha256",
        "failing_chart_ordinals",
        "final_hard_occupancy_count",
        "final_hard_occupancy_mass_binary64",
        "final_hard_count_by_chart_and_mode",
        "final_hard_mass_binary64_by_chart_and_mode",
        "final_hard_count_by_chart_and_family",
        "final_hard_mass_binary64_by_chart_and_family",
        "composition_catalog_sha256",
        "final_sort_permutation",
        "final_pre_sort_center_sha256s",
        "radius_squared_sha256",
        "soft_occupancy_mass_binary64",
        "covariance_sha256s",
        "leading_eigenvalues_sha256s",
        "tangent_bases_sha256",
        "tangent_variance_sha256",
        "orthogonal_alpha_sha256",
        "atlas_sha256",
        "fit_solver_receipt_sha256",
        "gate_evidence",
        "initialized_centers_sha256",
        "initialized_center_sha256s",
        "lloyd_pass_receipts",
        "minimum_hard_occupancy_mass_binary64",
        "minimum_required_hard_count",
        "minimum_observed_hard_mass_binary64",
        "minimum_hard_mass_margin_binary64",
        "minimum_hard_mass_chart_ordinals",
        "outcome",
        "receipt_sha256",
        "schema",
        "seed_receipts",
        "sorted_centers_sha256",
        "sorted_center_sha256s",
        "source_feature_width",
        "terminal_phase",
        "total_balanced_mass_binary64",
        "total_row_count",
    }
    row = _require_exact_fields(value, expected, label="candidate outcome")
    if row["schema"] != STREAMING_SHARED_CHART_ATLAS_CANDIDATE_SCHEMA_V1:
        raise ValueError("candidate outcome schema differs")
    for label in (
        "failing_chart_ordinals",
        "mode_ids",
        "family_ids",
        "fit_row_count_by_family",
        "final_hard_occupancy_count",
        "final_hard_occupancy_mass_binary64",
        "final_sort_permutation",
        "final_pre_sort_center_sha256s",
        "sorted_center_sha256s",
        "initialized_center_sha256s",
        "gate_evidence",
        "lloyd_pass_receipts",
        "seed_receipts",
        "soft_occupancy_mass_binary64",
        "covariance_sha256s",
        "leading_eigenvalues_sha256s",
        "minimum_hard_mass_chart_ordinals",
        "final_hard_count_by_chart_and_mode",
        "final_hard_mass_binary64_by_chart_and_mode",
        "final_hard_count_by_chart_and_family",
        "final_hard_mass_binary64_by_chart_and_family",
    ):
        if not isinstance(row[label], list):
            raise ValueError(f"candidate outcome {label} must be a list")
    for label in (
        "final_hard_count_by_chart_and_mode",
        "final_hard_mass_binary64_by_chart_and_mode",
        "final_hard_count_by_chart_and_family",
        "final_hard_mass_binary64_by_chart_and_family",
    ):
        if any(not isinstance(inner, list) for inner in row[label]):  # type: ignore[union-attr]
            raise ValueError(f"candidate outcome {label} rows must be lists")
    return StreamingSharedChartAtlasCandidateOutcomeV1(
        chart_count=row["chart_count"],  # type: ignore[arg-type]
        source_feature_width=row["source_feature_width"],  # type: ignore[arg-type]
        total_row_count=row["total_row_count"],  # type: ignore[arg-type]
        mode_count=row["mode_count"],  # type: ignore[arg-type]
        family_count=row["family_count"],  # type: ignore[arg-type]
        mode_ids=tuple(row["mode_ids"]),  # type: ignore[arg-type]
        family_ids=tuple(row["family_ids"]),  # type: ignore[arg-type]
        fit_row_count_by_family=tuple(
            row["fit_row_count_by_family"]
        ),  # type: ignore[arg-type]
        total_balanced_mass_binary64=row["total_balanced_mass_binary64"],  # type: ignore[arg-type]
        config_sha256=row["config_sha256"],  # type: ignore[arg-type]
        seed_receipts=tuple(_parse_seed(item) for item in row["seed_receipts"]),  # type: ignore[union-attr]
        initialized_centers_sha256=row["initialized_centers_sha256"],  # type: ignore[arg-type]
        initialized_center_sha256s=tuple(
            row["initialized_center_sha256s"]
        ),  # type: ignore[arg-type]
        lloyd_pass_receipts=tuple(
            _parse_pass(item) for item in row["lloyd_pass_receipts"]  # type: ignore[union-attr]
        ),
        final_sort_permutation=tuple(row["final_sort_permutation"]),  # type: ignore[arg-type]
        final_pre_sort_center_sha256s=tuple(
            row["final_pre_sort_center_sha256s"]
        ),  # type: ignore[arg-type]
        sorted_centers_sha256=row["sorted_centers_sha256"],  # type: ignore[arg-type]
        sorted_center_sha256s=tuple(
            row["sorted_center_sha256s"]
        ),  # type: ignore[arg-type]
        final_hard_occupancy_count=tuple(
            row["final_hard_occupancy_count"]
        ),  # type: ignore[arg-type]
        final_hard_occupancy_mass_binary64=tuple(
            row["final_hard_occupancy_mass_binary64"]
        ),  # type: ignore[arg-type]
        final_hard_count_by_chart_and_mode=tuple(
            tuple(inner) for inner in row["final_hard_count_by_chart_and_mode"]
        ),  # type: ignore[arg-type]
        final_hard_mass_binary64_by_chart_and_mode=tuple(
            tuple(inner)
            for inner in row["final_hard_mass_binary64_by_chart_and_mode"]
        ),  # type: ignore[arg-type]
        final_hard_count_by_chart_and_family=tuple(
            tuple(inner)
            for inner in row["final_hard_count_by_chart_and_family"]
        ),  # type: ignore[arg-type]
        final_hard_mass_binary64_by_chart_and_family=tuple(
            tuple(inner)
            for inner in row["final_hard_mass_binary64_by_chart_and_family"]
        ),  # type: ignore[arg-type]
        composition_catalog_sha256=row["composition_catalog_sha256"],  # type: ignore[arg-type]
        radius_squared_sha256=row["radius_squared_sha256"],  # type: ignore[arg-type]
        soft_occupancy_mass_binary64=tuple(
            row["soft_occupancy_mass_binary64"]
        ),  # type: ignore[arg-type]
        covariance_sha256s=tuple(row["covariance_sha256s"]),  # type: ignore[arg-type]
        leading_eigenvalues_sha256s=tuple(
            row["leading_eigenvalues_sha256s"]
        ),  # type: ignore[arg-type]
        tangent_bases_sha256=row["tangent_bases_sha256"],  # type: ignore[arg-type]
        tangent_variance_sha256=row["tangent_variance_sha256"],  # type: ignore[arg-type]
        orthogonal_alpha_sha256=row["orthogonal_alpha_sha256"],  # type: ignore[arg-type]
        atlas_sha256=row["atlas_sha256"],  # type: ignore[arg-type]
        fit_solver_receipt_sha256=row["fit_solver_receipt_sha256"],  # type: ignore[arg-type]
        gate_evidence=tuple(
            _parse_gate_evidence(item) for item in row["gate_evidence"]  # type: ignore[union-attr]
        ),
        minimum_required_hard_count=row["minimum_required_hard_count"],  # type: ignore[arg-type]
        minimum_observed_hard_mass_binary64=row[
            "minimum_observed_hard_mass_binary64"
        ],  # type: ignore[arg-type]
        minimum_hard_mass_margin_binary64=row[
            "minimum_hard_mass_margin_binary64"
        ],  # type: ignore[arg-type]
        minimum_hard_mass_chart_ordinals=tuple(
            row["minimum_hard_mass_chart_ordinals"]
        ),  # type: ignore[arg-type]
        minimum_hard_occupancy_mass_binary64=row[
            "minimum_hard_occupancy_mass_binary64"
        ],  # type: ignore[arg-type]
        outcome=row["outcome"],  # type: ignore[arg-type]
        terminal_phase=row["terminal_phase"],  # type: ignore[arg-type]
        failing_chart_ordinals=tuple(row["failing_chart_ordinals"]),  # type: ignore[arg-type]
        receipt_sha256=row["receipt_sha256"],  # type: ignore[arg-type]
    )


def parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
    value: Mapping[str, object],
) -> StreamingSharedChartAtlasOccupancyDiagnosisV1:
    """Strictly parse and authenticate one public occupancy diagnosis."""

    expected = {
        "candidate",
        "compute_device",
        "compute_dtype",
        "config",
        "config_sha256",
        "fit_balanced_row_weight_catalog_sha256",
        "fit_context_catalog_sha256",
        "fit_mode_ids",
        "fit_family_ids",
        "fit_row_count_by_family",
        "fit_panel_binding_sha256",
        "fit_role",
        "fit_row_catalog_sha256",
        "kernel_receipt_sha256",
        "selection_kernel_receipt_sha256",
        "mode_count",
        "receipt_sha256",
        "resource_config_receipt_sha256",
        "resources",
        "row_count_per_mode",
        "schema",
        "spool_artifact_sha256s",
        "spool_mode_identity_sha256s",
        "spool_tensor_catalog_sha256s",
        "total_balanced_mass_binary64",
        "total_row_count",
    }
    row = _require_exact_fields(value, expected, label="occupancy diagnosis")
    if row["schema"] != (
        STREAMING_SHARED_CHART_ATLAS_OCCUPANCY_DIAGNOSIS_SCHEMA_V1
    ):
        raise ValueError("occupancy diagnosis schema differs")
    for label in (
        "spool_artifact_sha256s",
        "spool_mode_identity_sha256s",
        "spool_tensor_catalog_sha256s",
        "fit_mode_ids",
        "fit_family_ids",
        "fit_row_count_by_family",
    ):
        if not isinstance(row[label], list):
            raise ValueError(f"occupancy diagnosis {label} must be a list")
    return StreamingSharedChartAtlasOccupancyDiagnosisV1(
        config=_config_from_payload(row["config"]),
        resources=_resources_from_payload(row["resources"]),
        config_sha256=row["config_sha256"],  # type: ignore[arg-type]
        resource_config_receipt_sha256=row[
            "resource_config_receipt_sha256"
        ],  # type: ignore[arg-type]
        kernel_receipt_sha256=row["kernel_receipt_sha256"],  # type: ignore[arg-type]
        selection_kernel_receipt_sha256=row[
            "selection_kernel_receipt_sha256"
        ],  # type: ignore[arg-type]
        fit_role=row["fit_role"],  # type: ignore[arg-type]
        fit_panel_binding_sha256=row["fit_panel_binding_sha256"],  # type: ignore[arg-type]
        fit_row_catalog_sha256=row["fit_row_catalog_sha256"],  # type: ignore[arg-type]
        fit_context_catalog_sha256=row["fit_context_catalog_sha256"],  # type: ignore[arg-type]
        fit_balanced_row_weight_catalog_sha256=row[
            "fit_balanced_row_weight_catalog_sha256"
        ],  # type: ignore[arg-type]
        fit_mode_ids=tuple(row["fit_mode_ids"]),  # type: ignore[arg-type]
        fit_family_ids=tuple(row["fit_family_ids"]),  # type: ignore[arg-type]
        fit_row_count_by_family=tuple(
            row["fit_row_count_by_family"]
        ),  # type: ignore[arg-type]
        spool_artifact_sha256s=tuple(row["spool_artifact_sha256s"]),  # type: ignore[arg-type]
        spool_mode_identity_sha256s=tuple(
            row["spool_mode_identity_sha256s"]
        ),  # type: ignore[arg-type]
        spool_tensor_catalog_sha256s=tuple(
            row["spool_tensor_catalog_sha256s"]
        ),  # type: ignore[arg-type]
        mode_count=row["mode_count"],  # type: ignore[arg-type]
        row_count_per_mode=row["row_count_per_mode"],  # type: ignore[arg-type]
        total_row_count=row["total_row_count"],  # type: ignore[arg-type]
        total_balanced_mass_binary64=row["total_balanced_mass_binary64"],  # type: ignore[arg-type]
        compute_device=row["compute_device"],  # type: ignore[arg-type]
        compute_dtype=row["compute_dtype"],  # type: ignore[arg-type]
        candidate=_parse_candidate(row["candidate"]),
        receipt_sha256=row["receipt_sha256"],  # type: ignore[arg-type]
    )


def parse_streaming_shared_chart_atlas_selection_receipt_v1(
    value: Mapping[str, object],
) -> StreamingSharedChartAtlasSelectionReceiptV1:
    """Strictly parse and authenticate one descending-K selection receipt."""

    expected = {
        "attempted_chart_counts",
        "attempted_diagnoses",
        "candidate_order",
        "claim_tier",
        "receipt_sha256",
        "schema",
        "selected_atlas_sha256",
        "selected_chart_count",
        "selected_fit_solver_receipt_sha256",
        "selected_source_feature_width",
        "selection_outcome",
        "selection_kernel_receipt_sha256",
        "selection_policy_id",
        "unattempted_chart_counts",
    }
    row = _require_exact_fields(value, expected, label="selection receipt")
    if row["schema"] != STREAMING_SHARED_CHART_ATLAS_SELECTION_SCHEMA_V1:
        raise ValueError("selection receipt schema differs")
    for label in (
        "attempted_chart_counts",
        "attempted_diagnoses",
        "candidate_order",
        "unattempted_chart_counts",
    ):
        if not isinstance(row[label], list):
            raise ValueError(f"selection receipt {label} must be a list")
    return StreamingSharedChartAtlasSelectionReceiptV1(
        selection_outcome=row["selection_outcome"],  # type: ignore[arg-type]
        selection_kernel_receipt_sha256=row[
            "selection_kernel_receipt_sha256"
        ],  # type: ignore[arg-type]
        candidate_order=tuple(row["candidate_order"]),  # type: ignore[arg-type]
        attempted_diagnoses=tuple(
            parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(item)
            for item in row["attempted_diagnoses"]  # type: ignore[union-attr]
        ),
        attempted_chart_counts=tuple(row["attempted_chart_counts"]),  # type: ignore[arg-type]
        unattempted_chart_counts=tuple(row["unattempted_chart_counts"]),  # type: ignore[arg-type]
        selected_chart_count=row["selected_chart_count"],  # type: ignore[arg-type]
        selected_source_feature_width=row["selected_source_feature_width"],  # type: ignore[arg-type]
        selected_atlas_sha256=row["selected_atlas_sha256"],  # type: ignore[arg-type]
        selected_fit_solver_receipt_sha256=row[
            "selected_fit_solver_receipt_sha256"
        ],  # type: ignore[arg-type]
        selection_policy_id=row["selection_policy_id"],  # type: ignore[arg-type]
        claim_tier=row["claim_tier"],  # type: ignore[arg-type]
        receipt_sha256=row["receipt_sha256"],  # type: ignore[arg-type]
    )


__all__ = [
    "ADMISSIBLE_OUTCOME",
    "A_ATLAS_NO_ADMISSIBLE_K",
    "A_ATLAS_SELECTED",
    "BALANCED_HARD_MASS_BELOW_MINIMUM",
    "CENTER_INITIALIZATION_DEGENERATE",
    "COVARIANCE_CONTRACT_FAILED",
    "EIGENGAP_ORIENTATION_UNDEFINED",
    "EMPTY_CHART_OUTCOME",
    "HARD_COUNT_BELOW_RANK_PLUS_ONE",
    "INSUFFICIENT_HARD_COUNT_OUTCOME",
    "INSUFFICIENT_HARD_MASS_OUTCOME",
    "LLOYD_EMPTY_CHART",
    "RADIUS_CONTRACT_FAILED",
    "SOFT_OCCUPANCY_CONTRACT_FAILED",
    "SPD_CONTRACT_FAILED",
    "STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1",
    "STREAMING_SHARED_CHART_ATLAS_CANDIDATE_SCHEMA_V1",
    "STREAMING_SHARED_CHART_ATLAS_GATE_EVIDENCE_SCHEMA_V1",
    "STREAMING_SHARED_CHART_ATLAS_LLOYD_PASS_COUNT_V1",
    "STREAMING_SHARED_CHART_ATLAS_MINIMUM_HARD_MASS_V1",
    "STREAMING_SHARED_CHART_ATLAS_OCCUPANCY_DIAGNOSIS_SCHEMA_V1",
    "STREAMING_SHARED_CHART_ATLAS_PASS_SCHEMA_V1",
    "STREAMING_SHARED_CHART_ATLAS_SEED_SCHEMA_V1",
    "STREAMING_SHARED_CHART_ATLAS_SELECTION_SCHEMA_V1",
    "STREAMING_SHARED_CHART_ATLAS_SELECTION_KERNEL_SCHEMA_V1",
    "STREAMING_SHARED_CHART_ATLAS_TANGENT_RANK_V1",
    "StreamingSharedChartAtlasCandidateOutcomeV1",
    "StreamingSharedChartAtlasGateEvidenceV1",
    "StreamingSharedChartAtlasOccupancyDiagnosisV1",
    "StreamingSharedChartAtlasPassReceiptV1",
    "StreamingSharedChartAtlasSeedReceiptV1",
    "StreamingSharedChartAtlasSelectionReceiptV1",
    "StreamingSharedChartAtlasSelectionResultV1",
    "diagnose_shared_chart_atlas_occupancy_streaming",
    "parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1",
    "parse_streaming_shared_chart_atlas_selection_receipt_v1",
    "select_and_fit_shared_chart_atlas_streaming",
    "streaming_shared_chart_atlas_selection_kernel_receipt_v1",
    "streaming_shared_chart_atlas_mode_identity_sha256_v1",
    "validate_streaming_shared_chart_atlas_diagnosis_against_spools_v1",
    "validate_streaming_shared_chart_atlas_occupancy_diagnosis_against_raw_spools_v1",
]
