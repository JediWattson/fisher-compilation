"""A-only architecture selection for Gemma-3 layer-local multimode flows.

The selector consumes authenticated *scalar* candidate-evaluation receipts.  It
does not fit candidates and has no API for discovery/confirmation panels,
graphs, or transports.  Every one of the 18 token-local MLP residual maps is
selected independently from the frozen chart x dynamic-mode x tangent-rank
grid.  Selection is compression first: choose the smallest serialized
candidate that passes the A-validation gates, then break ties by validation
NRMSE and the numeric ``(C, M, R)`` tuple.

``M=1`` is an ordinary (non-mixture) candidate.  ``M>1`` must additionally
beat its exact same-layer, same-chart, same-rank ``M=1`` control and pass the
noncollapse, distinctness, and split-refit stability gates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from weakref import WeakValueDictionary


__all__ = [
    "AGGREGATE_COSINE_MINIMUM",
    "CHART_COUNT_LADDER",
    "DYNAMIC_MODE_COUNT_LADDER",
    "FAMILY_NRMSE_MAXIMUM",
    "HARD_ROUTE_ARI_MINIMUM",
    "HIDDEN_WIDTH",
    "LAYER_COUNT",
    "MINIMUM_MEAN_MODE_MASS",
    "MINIMUM_CHART_SOFT_MASS",
    "MINIMUM_MODE_SSE_IMPROVEMENT",
    "MODE_VELOCITY_SEPARATION_MINIMUM",
    "NATIVE_MLP_MATRIX_PARAMETER_COUNT",
    "NATIVE_MLP_MATRIX_MACS_PER_TOKEN",
    "PRODUCTION_FULL_A_CAPTURE_SCOPE",
    "PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256",
    "NoAdmissibleLayerMultimodeCandidateError",
    "LayerMultimodeCandidateEvaluationReceipt",
    "LayerMultimodeCandidateGateDecision",
    "LayerMultimodeFamilyFidelityReceipt",
    "LayerMultimodeLayerSelectionReceipt",
    "LayerMultimodeSelectionReceipt",
    "SPLIT_HARD_ROUTE_ARI_MINIMUM",
    "SPLIT_PREDICTION_COSINE_MINIMUM",
    "STATE_MODE_MI_MINIMUM_NATS",
    "TANGENT_RANK_LADDER",
    "VALIDATION_VELOCITY_NRMSE_MAXIMUM",
    "candidate_matrix_macs_per_token",
    "candidate_serialized_parameter_count",
    "layer_multimode_selection_policy",
    "layer_multimode_selection_policy_sha256",
    "select_layer_multimode_candidates",
]


HIDDEN_WIDTH = 640
MLP_CHANNEL_COUNT = 2048
LAYER_COUNT = 18
CHART_COUNT_LADDER = (2, 1)
DYNAMIC_MODE_COUNT_LADDER = (4, 2, 1)
TANGENT_RANK_LADDER = (64, 32, 16)
CANDIDATES_PER_LAYER = 18
FAMILY_COUNT = 6

VALIDATION_VELOCITY_NRMSE_MAXIMUM = 0.25
AGGREGATE_COSINE_MINIMUM = 0.95
FAMILY_NRMSE_MAXIMUM = 0.35
MINIMUM_MODE_SSE_IMPROVEMENT = 0.05
MINIMUM_CHART_SOFT_MASS = 0.05
MINIMUM_MEAN_MODE_MASS = 0.05
STATE_MODE_MI_MINIMUM_NATS = 0.05
MODE_VELOCITY_SEPARATION_MINIMUM = 0.05
SPLIT_PREDICTION_COSINE_MINIMUM = 0.80
SPLIT_HARD_ROUTE_ARI_MINIMUM = 0.60
# Public spelling retained for setup/report consumers.
HARD_ROUTE_ARI_MINIMUM = SPLIT_HARD_ROUTE_ARI_MINIMUM

NATIVE_MLP_MATRIX_PARAMETER_COUNT = MLP_CHANNEL_COUNT * 3 * HIDDEN_WIDTH
NATIVE_MLP_MATRIX_MACS_PER_TOKEN = NATIVE_MLP_MATRIX_PARAMETER_COUNT
RETAINED_INPUT_RMSNORM_PARAMETER_COUNT = HIDDEN_WIDTH
NATIVE_MLP_RMSNORM_PARAMETER_COUNT = 2 * HIDDEN_WIDTH
NATIVE_MLP_BOUNDARY_PARAMETER_COUNT = (
    NATIVE_MLP_RMSNORM_PARAMETER_COUNT + NATIVE_MLP_MATRIX_PARAMETER_COUNT
)
# Scalar-only copies of the production fit boundary.  The fitter asserts at
# import time that its canonical policy hashes to this exact value and that
# the capture module exports this exact scope string.  Keeping these constants
# here avoids making the receipt-only selector import torch/capture code.
PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256 = (
    "be4908d50da4a0924484b192baac82e453e39ed316854e749e9a867252eae636"
)
PRODUCTION_FULL_A_CAPTURE_SCOPE = (
    "production_full_A_RoleCaptureReceipt_spools"
)
_DEVELOPMENT_SINGLE_BATCH_CAPTURE_SCOPE = (
    "development_single_CapturedBatch_only"
)

_FORMAT_VERSION = 1
_FAMILY_KIND = "fisher_graph.gemma3_layer_multimode_family_fidelity"
_CANDIDATE_KIND = "fisher_graph.gemma3_layer_multimode_candidate_evaluation"
_DECISION_KIND = "fisher_graph.gemma3_layer_multimode_candidate_gate_decision"
_LAYER_KIND = "fisher_graph.gemma3_layer_multimode_layer_selection"
_SELECTION_KIND = "fisher_graph.gemma3_layer_multimode_selection"
_FAMILY_DOMAIN = b"fisher_graph.gemma3_layer_multimode_family_fidelity.v1\0"
_CANDIDATE_DOMAIN = b"fisher_graph.gemma3_layer_multimode_candidate.v1\0"
_DECISION_DOMAIN = b"fisher_graph.gemma3_layer_multimode_decision.v1\0"
_LAYER_DOMAIN = b"fisher_graph.gemma3_layer_multimode_layer_selection.v1\0"
_SELECTION_DOMAIN = b"fisher_graph.gemma3_layer_multimode_selection.v1\0"
_POLICY_DOMAIN = b"fisher_graph.gemma3_layer_multimode_policy.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_REL_TOL = 1e-12
_FLOAT_ABS_TOL = 1e-12
_LIVE_CANDIDATE_EVALUATION_AUTHORITY = object()
_LIVE_SELECTOR_AUTHORITY = object()
_LIVE_CANDIDATE_EVALUATIONS: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_LIVE_SELECTION_RECEIPTS: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)

_SAFETY_METADATA: dict[str, bool] = {
    "contains_raw_tensors": False,
    "contains_model_weights": False,
    "contains_prompt_text": False,
    "contains_activation_rows": False,
    "contains_gradient_rows": False,
    "uses_fit_a_only": True,
    "uses_b_c_d_or_graph_inputs": False,
}

_GATE_ORDER = (
    "resource_receipt_failed",
    "not_smaller_than_native_mlp_matrices",
    "validation_velocity_nrmse",
    "aggregate_velocity_cosine",
    "family_velocity_nrmse",
    "matched_m1_sse_improvement",
    "chart_soft_mass",
    "mean_mode_mass",
    "state_mode_mutual_information",
    "mode_velocity_separation",
    "split_refit_prediction_cosine",
    "split_refit_hard_route_ari",
)


class NoAdmissibleLayerMultimodeCandidateError(RuntimeError):
    """Raised before fresh-panel work when any layer has no A-valid candidate."""

    def __init__(
        self,
        layer_ordinal: int,
        decisions: tuple[LayerMultimodeCandidateGateDecision, ...],
    ) -> None:
        self.layer_ordinal = layer_ordinal
        self.decisions = decisions
        super().__init__(
            f"layer {layer_ordinal} has no admissible A-only multimode candidate"
        )


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be a finite float")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_FLOAT_REL_TOL,
        abs_tol=_FLOAT_ABS_TOL,
    )


def _strict_fields(
    state: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


def _json_safe_state(value: object) -> object:
    """Recursively emit JSON-native containers; tuples are receipt internals."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe_state(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_json_safe_state(item) for item in value]
    if type(value) is list:
        return [_json_safe_state(item) for item in value]
    return value


def _require_json_list(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON list")
    return value


def candidate_serialized_parameter_count(
    chart_count: int,
    dynamic_mode_count: int,
    tangent_rank: int,
) -> int:
    """Stored-scalar ABI, retaining only the input H640 RMSNorm vector."""

    for label, value in (
        ("chart_count", chart_count),
        ("dynamic_mode_count", dynamic_mode_count),
        ("tangent_rank", tangent_rank),
    ):
        _require_int(value, label=label, minimum=1)
    c = chart_count
    m = dynamic_mode_count
    r = tangent_rank
    return (
        HIDDEN_WIDTH
        + c * HIDDEN_WIDTH
        + c * HIDDEN_WIDTH * r
        + 1
        + c * m * (r + 1)
        + c * m * (r * r + r)
    )


def candidate_matrix_macs_per_token(
    chart_count: int,
    dynamic_mode_count: int,
    tangent_rank: int,
) -> int:
    """Frozen matrix-MAC ABI; this is not a wall-clock latency claim."""

    for label, value in (
        ("chart_count", chart_count),
        ("dynamic_mode_count", dynamic_mode_count),
        ("tangent_rank", tangent_rank),
    ):
        _require_int(value, label=label, minimum=1)
    c = chart_count
    m = dynamic_mode_count
    r = tangent_rank
    return 2 * c * HIDDEN_WIDTH * r + c * m * r + c * m * r * r


def layer_multimode_selection_policy() -> dict[str, object]:
    """Return the JSON-safe frozen production selection contract."""

    return {
        "format_version": _FORMAT_VERSION,
        "scope": "replace_token_local_mlp_residual_only_attention_remains_native",
        "data_boundary": "fit_and_validate_on_A_only_before_fresh_B",
        "required_fit_policy_sha256": (
            PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256
        ),
        "required_source_capture_scope": PRODUCTION_FULL_A_CAPTURE_SCOPE,
        "layer_count": LAYER_COUNT,
        "hidden_width": HIDDEN_WIDTH,
        "native_mlp_channel_count": MLP_CHANNEL_COUNT,
        "chart_count_ladder": list(CHART_COUNT_LADDER),
        "dynamic_mode_count_ladder": list(DYNAMIC_MODE_COUNT_LADDER),
        "tangent_rank_ladder": list(TANGENT_RANK_LADDER),
        "candidate_count_per_layer": CANDIDATES_PER_LAYER,
        "candidate_grid_complete": True,
        "production_candidate_authority": (
            "all_324_original_live_fit_adapter_evaluations"
        ),
        "production_selection_authority": (
            "original_live_selector_receipt_over_the_full_grid"
        ),
        "family_count": FAMILY_COUNT,
        "chart_router": "softmax_of_negative_squared_normal_residual_over_tau",
        "c1_temperature": "fixed_exactly_1_and_serialized_accounted",
        "c2_temperature": "A_fitted_finite_positive_scalar",
        "parameter_formula": "H+C*H+C*H*R+1+C*M*(R+1)+C*M*(R*R+R)",
        "candidate_retained_norm": "native_mlp_input_rmsnorm_only",
        "native_output_rmsnorm": (
            "folded_into_teacher_velocity_target_and_not_called_at_deploy"
        ),
        "matrix_macs_per_token_formula": (
            "2*C*H*R+C*M*R+C*M*R*R"
        ),
        "native_mlp_matrix_parameter_count": (
            NATIVE_MLP_MATRIX_PARAMETER_COUNT
        ),
        "native_mlp_matrix_macs_per_token": NATIVE_MLP_MATRIX_MACS_PER_TOKEN,
        "native_mlp_boundary_parameter_count": (
            NATIVE_MLP_BOUNDARY_PARAMETER_COUNT
        ),
        "resource_pass_formula": (
            "candidate_parameters_lt_native_matrix_parameters_and_"
            "candidate_matrix_macs_lt_native_matrix_macs"
        ),
        "validation_velocity_nrmse_maximum": (
            VALIDATION_VELOCITY_NRMSE_MAXIMUM
        ),
        "aggregate_velocity_cosine_minimum": AGGREGATE_COSINE_MINIMUM,
        "family_velocity_nrmse_maximum": FAMILY_NRMSE_MAXIMUM,
        "minimum_m_gt_1_sse_improvement_over_matched_m1": (
            MINIMUM_MODE_SSE_IMPROVEMENT
        ),
        "minimum_chart_soft_mass": MINIMUM_CHART_SOFT_MASS,
        "minimum_mean_mode_mass": MINIMUM_MEAN_MODE_MASS,
        "minimum_state_mode_mutual_information_nats": (
            STATE_MODE_MI_MINIMUM_NATS
        ),
        "minimum_pairwise_mode_velocity_separation_ratio": (
            MODE_VELOCITY_SEPARATION_MINIMUM
        ),
        "minimum_split_refit_matched_prediction_cosine": (
            SPLIT_PREDICTION_COSINE_MINIMUM
        ),
        "minimum_split_refit_hard_route_ari": (
            SPLIT_HARD_ROUTE_ARI_MINIMUM
        ),
        "selection_tie_key": [
            "serialized_parameter_count",
            "validation_velocity_nrmse",
            "chart_count",
            "dynamic_mode_count",
            "tangent_rank",
        ],
        "latency_claimed": False,
        "serialized_byte_reduction_claimed_before_d3": False,
    }


def layer_multimode_selection_policy_sha256() -> str:
    return _json_sha256(
        layer_multimode_selection_policy(),
        domain=_POLICY_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class LayerMultimodeFamilyFidelityReceipt:
    family_id: str
    scalar_count: int
    prediction_sse: float
    teacher_energy: float
    velocity_nrmse: float
    prediction_energy: float
    prediction_teacher_dot: float
    velocity_cosine: float
    artifact_sha256: str = ""
    artifact_kind: str = _FAMILY_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_string(self.family_id, label="family_id")
        _require_int(self.scalar_count, label="scalar_count", minimum=1)
        _require_float(self.prediction_sse, label="prediction_sse", minimum=0.0)
        _require_float(self.teacher_energy, label="teacher_energy", minimum=0.0)
        if self.teacher_energy == 0.0:
            raise ValueError("teacher_energy must be positive")
        _require_float(self.velocity_nrmse, label="velocity_nrmse", minimum=0.0)
        expected = math.sqrt(self.prediction_sse / self.teacher_energy)
        if not _close(self.velocity_nrmse, expected):
            raise ValueError("family velocity_nrmse does not recompute")
        _require_float(
            self.prediction_energy,
            label="prediction_energy",
            minimum=0.0,
        )
        if self.prediction_energy == 0.0:
            raise ValueError("prediction_energy must be positive")
        _require_float(
            self.prediction_teacher_dot,
            label="prediction_teacher_dot",
        )
        _require_float(
            self.velocity_cosine,
            label="velocity_cosine",
            minimum=-1.0,
            maximum=1.0,
        )
        expected_cosine = self.prediction_teacher_dot / math.sqrt(
            self.prediction_energy * self.teacher_energy
        )
        if not _close(self.velocity_cosine, expected_cosine):
            raise ValueError("family velocity_cosine does not recompute")
        if (
            self.artifact_kind != _FAMILY_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("family fidelity receipt header is invalid")
        computed = _json_sha256(self._payload(), domain=_FAMILY_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="artifact_sha256"
        ) != computed:
            raise ValueError("family fidelity receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "family_id": self.family_id,
            "scalar_count": self.scalar_count,
            "prediction_sse": self.prediction_sse,
            "teacher_energy": self.teacher_energy,
            "velocity_nrmse": self.velocity_nrmse,
            "prediction_energy": self.prediction_energy,
            "prediction_teacher_dot": self.prediction_teacher_dot,
            "velocity_cosine": self.velocity_cosine,
        }

    def validate_integrity(self) -> None:
        if _json_sha256(self._payload(), domain=_FAMILY_DOMAIN) != self.artifact_sha256:
            raise ValueError("family fidelity receipt hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return _json_safe_state(
            {**self._payload(), "artifact_sha256": self.artifact_sha256}
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerMultimodeFamilyFidelityReceipt:
        expected = {
            "artifact_kind",
            "format_version",
            "family_id",
            "scalar_count",
            "prediction_sse",
            "teacher_energy",
            "velocity_nrmse",
            "prediction_energy",
            "prediction_teacher_dot",
            "velocity_cosine",
            "artifact_sha256",
        }
        _strict_fields(state, expected=expected, label="family fidelity receipt")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LayerMultimodeCandidateEvaluationReceipt:
    """Authenticated scalar-only A evaluation for one layer/C/M/R candidate."""

    layer_ordinal: int
    chart_count: int
    dynamic_mode_count: int
    tangent_rank: int
    chart_temperature: float
    chart_temperature_fitted_on_a: bool
    source_model_sha256: str
    source_rows_receipt_sha256: str
    fit_policy_sha256: str
    fit_policy_test_only: bool
    source_capture_scope: str
    production_fit_authorized: bool
    fit_source_sha256: str
    validation_source_sha256: str
    atlas_sha256: str
    fit_a_sha256: str
    validation_a_sha256: str
    training_receipt_sha256: str
    evaluation_protocol_sha256: str
    resource_receipt_sha256: str
    serialized_parameter_count: int
    matrix_macs_per_token: int
    validation_scalar_count: int
    validation_prediction_sse: float
    validation_teacher_energy: float
    validation_velocity_nrmse: float
    validation_prediction_energy: float
    validation_prediction_teacher_dot: float
    aggregate_velocity_cosine: float
    family_fidelity: tuple[LayerMultimodeFamilyFidelityReceipt, ...]
    chart_soft_masses: tuple[float, ...]
    conditional_mode_masses: tuple[tuple[float, ...], ...]
    per_chart_mean_row_router_entropy_nats: tuple[float, ...]
    per_chart_state_mode_mutual_information_nats: tuple[float, ...]
    per_chart_minimum_pairwise_velocity_separation_rms: tuple[float, ...]
    per_chart_minimum_pairwise_velocity_separation_ratio: tuple[float, ...]
    minimum_chart_soft_mass: float
    minimum_conditional_mode_mass: float
    minimum_per_chart_state_mode_mutual_information_nats: float
    minimum_per_chart_velocity_separation_ratio: float
    split_refit_matched_prediction_cosine: float
    split_refit_hard_route_ari: float
    resource_pass: bool
    artifact_sha256: str = ""
    artifact_kind: str = _CANDIDATE_KIND
    format_version: int = _FORMAT_VERSION
    contains_raw_tensors: bool = False
    contains_model_weights: bool = False
    contains_prompt_text: bool = False
    contains_activation_rows: bool = False
    contains_gradient_rows: bool = False
    uses_fit_a_only: bool = True
    uses_b_c_d_or_graph_inputs: bool = False
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_int(self.layer_ordinal, label="layer_ordinal")
        if self.layer_ordinal >= LAYER_COUNT:
            raise ValueError("layer_ordinal is outside the 18-layer model")
        if (
            self.chart_count not in CHART_COUNT_LADDER
            or type(self.chart_count) is not int
        ):
            raise ValueError("chart_count is outside the frozen ladder")
        if (
            self.dynamic_mode_count not in DYNAMIC_MODE_COUNT_LADDER
            or type(self.dynamic_mode_count) is not int
        ):
            raise ValueError("dynamic_mode_count is outside the frozen ladder")
        if (
            self.tangent_rank not in TANGENT_RANK_LADDER
            or type(self.tangent_rank) is not int
        ):
            raise ValueError("tangent_rank is outside the frozen ladder")
        _require_float(
            self.chart_temperature,
            label="chart_temperature",
            minimum=0.0,
        )
        if self.chart_temperature == 0.0:
            raise ValueError("chart_temperature must be positive")
        if type(self.chart_temperature_fitted_on_a) is not bool:
            raise TypeError("chart_temperature_fitted_on_a must be a bool")
        if self.chart_count == 1 and self.chart_temperature != 1.0:
            raise ValueError("C=1 chart_temperature must be exactly 1.0")
        if self.chart_temperature_fitted_on_a is not (self.chart_count > 1):
            raise ValueError(
                "chart_temperature A-fit status does not match C1/C2 policy"
            )
        for name in (
            "source_model_sha256",
            "source_rows_receipt_sha256",
            "fit_policy_sha256",
            "fit_source_sha256",
            "validation_source_sha256",
            "atlas_sha256",
            "fit_a_sha256",
            "validation_a_sha256",
            "training_receipt_sha256",
            "evaluation_protocol_sha256",
            "resource_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if type(self.fit_policy_test_only) is not bool:
            raise TypeError("fit_policy_test_only must be a bool")
        if self.source_capture_scope not in (
            PRODUCTION_FULL_A_CAPTURE_SCOPE,
            _DEVELOPMENT_SINGLE_BATCH_CAPTURE_SCOPE,
        ) or type(self.source_capture_scope) is not str:
            raise ValueError("source_capture_scope is outside the frozen boundary")
        if type(self.production_fit_authorized) is not bool:
            raise TypeError("production_fit_authorized must be a bool")
        if self.fit_policy_test_only is (
            self.fit_policy_sha256
            == PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256
        ):
            raise ValueError(
                "fit policy test-only status conflicts with the frozen production hash"
            )
        expected_production_authority = (
            not self.fit_policy_test_only
            and self.fit_policy_sha256
            == PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256
            and self.source_capture_scope == PRODUCTION_FULL_A_CAPTURE_SCOPE
        )
        if self.production_fit_authorized is not expected_production_authority:
            raise ValueError(
                "production_fit_authorized does not recompute from policy/scope"
            )
        if self.fit_a_sha256 == self.validation_a_sha256:
            raise ValueError("A fit and validation partitions must be distinct")
        if len(
            {
                self.source_rows_receipt_sha256,
                self.fit_source_sha256,
                self.validation_source_sha256,
            }
        ) != 3:
            raise ValueError(
                "full A rows and layer fit/validation subset digests must be distinct"
            )
        expected_parameters = candidate_serialized_parameter_count(
            self.chart_count,
            self.dynamic_mode_count,
            self.tangent_rank,
        )
        expected_macs = candidate_matrix_macs_per_token(
            self.chart_count,
            self.dynamic_mode_count,
            self.tangent_rank,
        )
        if (
            type(self.serialized_parameter_count) is not int
            or self.serialized_parameter_count != expected_parameters
        ):
            raise ValueError(
                "serialized_parameter_count does not match the frozen ABI"
            )
        if (
            type(self.matrix_macs_per_token) is not int
            or self.matrix_macs_per_token != expected_macs
        ):
            raise ValueError("matrix_macs_per_token does not match the frozen ABI")
        _require_int(
            self.validation_scalar_count,
            label="validation_scalar_count",
            minimum=1,
        )
        _require_float(
            self.validation_prediction_sse,
            label="validation_prediction_sse",
            minimum=0.0,
        )
        _require_float(
            self.validation_teacher_energy,
            label="validation_teacher_energy",
            minimum=0.0,
        )
        if self.validation_teacher_energy == 0.0:
            raise ValueError("validation_teacher_energy must be positive")
        _require_float(
            self.validation_velocity_nrmse,
            label="validation_velocity_nrmse",
            minimum=0.0,
        )
        expected_nrmse = math.sqrt(
            self.validation_prediction_sse / self.validation_teacher_energy
        )
        if not _close(self.validation_velocity_nrmse, expected_nrmse):
            raise ValueError("validation_velocity_nrmse does not recompute")
        _require_float(
            self.validation_prediction_energy,
            label="validation_prediction_energy",
            minimum=0.0,
        )
        if self.validation_prediction_energy == 0.0:
            raise ValueError("validation_prediction_energy must be positive")
        _require_float(
            self.validation_prediction_teacher_dot,
            label="validation_prediction_teacher_dot",
        )
        _require_float(
            self.aggregate_velocity_cosine,
            label="aggregate_velocity_cosine",
            minimum=-1.0,
            maximum=1.0,
        )
        expected_cosine = self.validation_prediction_teacher_dot / math.sqrt(
            self.validation_prediction_energy * self.validation_teacher_energy
        )
        if not _close(self.aggregate_velocity_cosine, expected_cosine):
            raise ValueError("aggregate_velocity_cosine does not recompute")
        if (
            type(self.family_fidelity) is not tuple
            or len(self.family_fidelity) != FAMILY_COUNT
            or any(
                not isinstance(value, LayerMultimodeFamilyFidelityReceipt)
                for value in self.family_fidelity
            )
        ):
            raise ValueError("family_fidelity must contain exactly six receipts")
        family_ids = tuple(value.family_id for value in self.family_fidelity)
        if (
            family_ids != tuple(sorted(family_ids))
            or len(set(family_ids)) != FAMILY_COUNT
        ):
            raise ValueError("family_fidelity must use unique canonical family order")
        for value in self.family_fidelity:
            value.validate_integrity()
        if (
            sum(value.scalar_count for value in self.family_fidelity)
            != self.validation_scalar_count
        ):
            raise ValueError(
                "family scalar counts do not sum to validation_scalar_count"
            )
        if not _close(
            sum(value.prediction_sse for value in self.family_fidelity),
            self.validation_prediction_sse,
        ):
            raise ValueError("family prediction SSE does not sum to validation SSE")
        if not _close(
            sum(value.teacher_energy for value in self.family_fidelity),
            self.validation_teacher_energy,
        ):
            raise ValueError("family teacher energy does not sum to validation energy")
        if not _close(
            sum(value.prediction_energy for value in self.family_fidelity),
            self.validation_prediction_energy,
        ):
            raise ValueError(
                "family prediction energy does not sum to validation energy"
            )
        if not _close(
            sum(
                value.prediction_teacher_dot for value in self.family_fidelity
            ),
            self.validation_prediction_teacher_dot,
        ):
            raise ValueError(
                "family prediction-teacher dot does not sum to validation dot"
            )
        self._validate_mode_diagnostics()
        if type(self.resource_pass) is not bool:
            raise TypeError("resource_pass must be a bool")
        expected_resource_pass = (
            self.serialized_parameter_count
            < NATIVE_MLP_MATRIX_PARAMETER_COUNT
            and self.matrix_macs_per_token < NATIVE_MLP_MATRIX_MACS_PER_TOKEN
        )
        if self.resource_pass is not expected_resource_pass:
            raise ValueError("resource_pass does not recompute from typed counts")
        for name, expected in _SAFETY_METADATA.items():
            if getattr(self, name) is not expected:
                raise ValueError("candidate safety metadata is invalid")
        if (
            self.artifact_kind != _CANDIDATE_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("candidate evaluation receipt header is invalid")
        computed = _json_sha256(self._payload(), domain=_CANDIDATE_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="artifact_sha256"
        ) != computed:
            raise ValueError("candidate evaluation receipt hash mismatch")

    @property
    def candidate_key(self) -> tuple[int, int, int, int]:
        return (
            self.layer_ordinal,
            self.chart_count,
            self.dynamic_mode_count,
            self.tangent_rank,
        )

    @property
    def teacher_rms(self) -> float:
        return math.sqrt(
            self.validation_teacher_energy / self.validation_scalar_count
        )

    def _validate_mode_diagnostics(self) -> None:
        if (
            type(self.chart_soft_masses) is not tuple
            or len(self.chart_soft_masses) != self.chart_count
        ):
            raise ValueError("chart_soft_masses length must equal chart_count")
        for index, value in enumerate(self.chart_soft_masses):
            _require_float(
                value,
                label=f"chart_soft_masses[{index}]",
                minimum=0.0,
                maximum=1.0,
            )
        if not _close(sum(self.chart_soft_masses), 1.0):
            raise ValueError("chart_soft_masses must sum to one")
        if self.chart_count == 1 and self.chart_soft_masses != (1.0,):
            raise ValueError("C=1 chart_soft_masses must be exactly (1.0,)")
        chart_vector_fields = (
            self.conditional_mode_masses,
            self.per_chart_mean_row_router_entropy_nats,
            self.per_chart_state_mode_mutual_information_nats,
            self.per_chart_minimum_pairwise_velocity_separation_rms,
            self.per_chart_minimum_pairwise_velocity_separation_ratio,
        )
        if any(
            type(values) is not tuple or len(values) != self.chart_count
            for values in chart_vector_fields
        ):
            raise ValueError("per-chart mode diagnostics must have chart_count rows")
        for chart_index, masses in enumerate(self.conditional_mode_masses):
            if type(masses) is not tuple or len(masses) != self.dynamic_mode_count:
                raise ValueError(
                    "conditional_mode_masses rows must have dynamic_mode_count entries"
                )
            for mode_index, value in enumerate(masses):
                _require_float(
                    value,
                    label=(
                        f"conditional_mode_masses[{chart_index}]"
                        f"[{mode_index}]"
                    ),
                    minimum=0.0,
                    maximum=1.0,
                )
            if not _close(sum(masses), 1.0):
                raise ValueError("every conditional_mode_masses row must sum to one")
            mean_entropy = self.per_chart_mean_row_router_entropy_nats[
                chart_index
            ]
            mutual_information = (
                self.per_chart_state_mode_mutual_information_nats[chart_index]
            )
            separation_rms = (
                self.per_chart_minimum_pairwise_velocity_separation_rms[
                    chart_index
                ]
            )
            separation_ratio = (
                self.per_chart_minimum_pairwise_velocity_separation_ratio[
                    chart_index
                ]
            )
            _require_float(
                mean_entropy,
                label=(
                    "per_chart_mean_row_router_entropy_nats"
                    f"[{chart_index}]"
                ),
                minimum=0.0,
                maximum=math.log(self.dynamic_mode_count),
            )
            _require_float(
                mutual_information,
                label=(
                    "per_chart_state_mode_mutual_information_nats"
                    f"[{chart_index}]"
                ),
                minimum=0.0,
                maximum=math.log(self.dynamic_mode_count),
            )
            mass_entropy = -sum(
                value * math.log(value) for value in masses if value > 0.0
            )
            if not _close(mutual_information, mass_entropy - mean_entropy):
                raise ValueError(
                    "per-chart state-mode mutual information does not recompute"
                )
            _require_float(
                separation_rms,
                label=(
                    "per_chart_minimum_pairwise_velocity_separation_rms"
                    f"[{chart_index}]"
                ),
                minimum=0.0,
            )
            _require_float(
                separation_ratio,
                label=(
                    "per_chart_minimum_pairwise_velocity_separation_ratio"
                    f"[{chart_index}]"
                ),
                minimum=0.0,
            )
            if not _close(separation_ratio, separation_rms / self.teacher_rms):
                raise ValueError(
                    "per-chart mode velocity separation ratio does not recompute"
                )
        minima = (
            (
                "minimum_chart_soft_mass",
                self.minimum_chart_soft_mass,
                min(self.chart_soft_masses),
            ),
            (
                "minimum_conditional_mode_mass",
                self.minimum_conditional_mode_mass,
                min(min(values) for values in self.conditional_mode_masses),
            ),
            (
                "minimum_per_chart_state_mode_mutual_information_nats",
                self.minimum_per_chart_state_mode_mutual_information_nats,
                min(self.per_chart_state_mode_mutual_information_nats),
            ),
            (
                "minimum_per_chart_velocity_separation_ratio",
                self.minimum_per_chart_velocity_separation_ratio,
                min(self.per_chart_minimum_pairwise_velocity_separation_ratio),
            ),
        )
        for label, stored, expected in minima:
            _require_float(stored, label=label, minimum=0.0)
            if not _close(stored, expected):
                raise ValueError(f"{label} does not recompute")
        _require_float(
            self.split_refit_matched_prediction_cosine,
            label="split_refit_matched_prediction_cosine",
            minimum=-1.0,
            maximum=1.0,
        )
        _require_float(
            self.split_refit_hard_route_ari,
            label="split_refit_hard_route_ari",
            minimum=-1.0,
            maximum=1.0,
        )
        if self.dynamic_mode_count == 1 and (
            any(value != (1.0,) for value in self.conditional_mode_masses)
            or any(
                value != 0.0
                for value in self.per_chart_mean_row_router_entropy_nats
            )
            or any(
                value != 0.0
                for value in self.per_chart_state_mode_mutual_information_nats
            )
            or any(
                value != 0.0
                for value in self.per_chart_minimum_pairwise_velocity_separation_rms
            )
            or any(
                value != 0.0
                for value in self.per_chart_minimum_pairwise_velocity_separation_ratio
            )
            or self.minimum_conditional_mode_mass != 1.0
            or self.minimum_per_chart_state_mode_mutual_information_nats != 0.0
            or self.minimum_per_chart_velocity_separation_ratio != 0.0
            or self.split_refit_matched_prediction_cosine != 1.0
            or self.split_refit_hard_route_ari != 1.0
        ):
            raise ValueError("M=1 diagnostics must use the frozen degenerate sentinels")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "layer_ordinal": self.layer_ordinal,
            "chart_count": self.chart_count,
            "dynamic_mode_count": self.dynamic_mode_count,
            "tangent_rank": self.tangent_rank,
            "chart_temperature": self.chart_temperature,
            "chart_temperature_fitted_on_a": (
                self.chart_temperature_fitted_on_a
            ),
            "source_model_sha256": self.source_model_sha256,
            "source_rows_receipt_sha256": (
                self.source_rows_receipt_sha256
            ),
            "fit_policy_sha256": self.fit_policy_sha256,
            "fit_policy_test_only": self.fit_policy_test_only,
            "source_capture_scope": self.source_capture_scope,
            "production_fit_authorized": self.production_fit_authorized,
            "fit_source_sha256": self.fit_source_sha256,
            "validation_source_sha256": self.validation_source_sha256,
            "atlas_sha256": self.atlas_sha256,
            "fit_a_sha256": self.fit_a_sha256,
            "validation_a_sha256": self.validation_a_sha256,
            "training_receipt_sha256": self.training_receipt_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "resource_receipt_sha256": self.resource_receipt_sha256,
            "serialized_parameter_count": self.serialized_parameter_count,
            "matrix_macs_per_token": self.matrix_macs_per_token,
            "validation_scalar_count": self.validation_scalar_count,
            "validation_prediction_sse": self.validation_prediction_sse,
            "validation_teacher_energy": self.validation_teacher_energy,
            "validation_velocity_nrmse": self.validation_velocity_nrmse,
            "validation_prediction_energy": self.validation_prediction_energy,
            "validation_prediction_teacher_dot": (
                self.validation_prediction_teacher_dot
            ),
            "aggregate_velocity_cosine": self.aggregate_velocity_cosine,
            "family_fidelity_sha256s": tuple(
                value.artifact_sha256 for value in self.family_fidelity
            ),
            "chart_soft_masses": self.chart_soft_masses,
            "conditional_mode_masses": self.conditional_mode_masses,
            "per_chart_mean_row_router_entropy_nats": (
                self.per_chart_mean_row_router_entropy_nats
            ),
            "per_chart_state_mode_mutual_information_nats": (
                self.per_chart_state_mode_mutual_information_nats
            ),
            "per_chart_minimum_pairwise_velocity_separation_rms": (
                self.per_chart_minimum_pairwise_velocity_separation_rms
            ),
            "per_chart_minimum_pairwise_velocity_separation_ratio": (
                self.per_chart_minimum_pairwise_velocity_separation_ratio
            ),
            "minimum_chart_soft_mass": self.minimum_chart_soft_mass,
            "minimum_conditional_mode_mass": (
                self.minimum_conditional_mode_mass
            ),
            "minimum_per_chart_state_mode_mutual_information_nats": (
                self.minimum_per_chart_state_mode_mutual_information_nats
            ),
            "minimum_per_chart_velocity_separation_ratio": (
                self.minimum_per_chart_velocity_separation_ratio
            ),
            "split_refit_matched_prediction_cosine": (
                self.split_refit_matched_prediction_cosine
            ),
            "split_refit_hard_route_ari": self.split_refit_hard_route_ari,
            "resource_pass": self.resource_pass,
            **_SAFETY_METADATA,
        }

    def validate_integrity(self) -> None:
        for value in self.family_fidelity:
            value.validate_integrity()
        if (
            _json_sha256(self._payload(), domain=_CANDIDATE_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("candidate evaluation receipt hash mismatch")

    @property
    def live_candidate_evaluation_authorized(self) -> bool:
        """True only for the original scalar receipt returned by the adapter."""

        return (
            self._construction_authority
            is _LIVE_CANDIDATE_EVALUATION_AUTHORITY
            and _LIVE_CANDIDATE_EVALUATIONS.get(id(self)) is self
        )

    def state_dict(self) -> dict[str, object]:
        return _json_safe_state(
            {
                **self._payload(),
                "family_fidelity": [
                    value.state_dict() for value in self.family_fidelity
                ],
                "artifact_sha256": self.artifact_sha256,
            }
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerMultimodeCandidateEvaluationReceipt:
        expected = {
            "artifact_kind",
            "format_version",
            "layer_ordinal",
            "chart_count",
            "dynamic_mode_count",
            "tangent_rank",
            "chart_temperature",
            "chart_temperature_fitted_on_a",
            "source_model_sha256",
            "source_rows_receipt_sha256",
            "fit_policy_sha256",
            "fit_policy_test_only",
            "source_capture_scope",
            "production_fit_authorized",
            "fit_source_sha256",
            "validation_source_sha256",
            "atlas_sha256",
            "fit_a_sha256",
            "validation_a_sha256",
            "training_receipt_sha256",
            "evaluation_protocol_sha256",
            "resource_receipt_sha256",
            "serialized_parameter_count",
            "matrix_macs_per_token",
            "validation_scalar_count",
            "validation_prediction_sse",
            "validation_teacher_energy",
            "validation_velocity_nrmse",
            "validation_prediction_energy",
            "validation_prediction_teacher_dot",
            "aggregate_velocity_cosine",
            "family_fidelity_sha256s",
            "family_fidelity",
            "chart_soft_masses",
            "conditional_mode_masses",
            "per_chart_mean_row_router_entropy_nats",
            "per_chart_state_mode_mutual_information_nats",
            "per_chart_minimum_pairwise_velocity_separation_rms",
            "per_chart_minimum_pairwise_velocity_separation_ratio",
            "minimum_chart_soft_mass",
            "minimum_conditional_mode_mass",
            "minimum_per_chart_state_mode_mutual_information_nats",
            "minimum_per_chart_velocity_separation_ratio",
            "split_refit_matched_prediction_cosine",
            "split_refit_hard_route_ari",
            "resource_pass",
            *_SAFETY_METADATA,
            "artifact_sha256",
        }
        _strict_fields(state, expected=expected, label="candidate evaluation receipt")
        family_states = _require_json_list(
            state["family_fidelity"],
            label="family_fidelity",
        )
        family_hashes = _require_json_list(
            state["family_fidelity_sha256s"],
            label="family_fidelity_sha256s",
        )
        families = tuple(
            LayerMultimodeFamilyFidelityReceipt.from_state_dict(value)
            for value in family_states
        )
        if family_hashes != [
            value.artifact_sha256 for value in families
        ]:
            raise ValueError("family fidelity hash inventory is invalid")
        arguments = dict(state)
        arguments.pop("family_fidelity_sha256s")
        arguments["family_fidelity"] = families
        for name in (
            "chart_soft_masses",
            "per_chart_mean_row_router_entropy_nats",
            "per_chart_state_mode_mutual_information_nats",
            "per_chart_minimum_pairwise_velocity_separation_rms",
            "per_chart_minimum_pairwise_velocity_separation_ratio",
        ):
            arguments[name] = tuple(
                _require_json_list(state[name], label=name)
            )
        conditional_rows = _require_json_list(
            state["conditional_mode_masses"],
            label="conditional_mode_masses",
        )
        arguments["conditional_mode_masses"] = tuple(
            tuple(
                _require_json_list(
                    row,
                    label=f"conditional_mode_masses[{index}]",
                )
            )
            for index, row in enumerate(conditional_rows)
        )
        return cls(**arguments)  # type: ignore[arg-type]


def _grant_live_candidate_evaluation_authority(
    candidate: LayerMultimodeCandidateEvaluationReceipt,
) -> LayerMultimodeCandidateEvaluationReceipt:
    if type(candidate) is not LayerMultimodeCandidateEvaluationReceipt:
        raise TypeError("live candidate grant requires the exact receipt type")
    candidate.validate_integrity()
    object.__setattr__(
        candidate,
        "_construction_authority",
        _LIVE_CANDIDATE_EVALUATION_AUTHORITY,
    )
    _LIVE_CANDIDATE_EVALUATIONS[id(candidate)] = candidate
    return candidate


@dataclass(frozen=True, slots=True)
class LayerMultimodeCandidateGateDecision:
    candidate_sha256: str
    layer_ordinal: int
    chart_count: int
    dynamic_mode_count: int
    tangent_rank: int
    matched_m1_candidate_sha256: str
    relative_sse_improvement_over_matched_m1: float
    failed_gates: tuple[str, ...]
    passed: bool
    artifact_sha256: str = ""
    artifact_kind: str = _DECISION_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.candidate_sha256, label="candidate_sha256")
        _require_sha256(
            self.matched_m1_candidate_sha256,
            label="matched_m1_candidate_sha256",
        )
        _require_int(self.layer_ordinal, label="layer_ordinal")
        for label, value, allowed in (
            ("chart_count", self.chart_count, CHART_COUNT_LADDER),
            (
                "dynamic_mode_count",
                self.dynamic_mode_count,
                DYNAMIC_MODE_COUNT_LADDER,
            ),
            ("tangent_rank", self.tangent_rank, TANGENT_RANK_LADDER),
        ):
            if type(value) is not int or value not in allowed:
                raise ValueError(f"{label} is outside the frozen ladder")
        _require_float(
            self.relative_sse_improvement_over_matched_m1,
            label="relative_sse_improvement_over_matched_m1",
        )
        if (
            type(self.failed_gates) is not tuple
            or any(type(value) is not str for value in self.failed_gates)
            or tuple(value for value in _GATE_ORDER if value in self.failed_gates)
            != self.failed_gates
            or len(set(self.failed_gates)) != len(self.failed_gates)
        ):
            raise ValueError("failed_gates is not a canonical gate inventory")
        if any(value not in _GATE_ORDER for value in self.failed_gates):
            raise ValueError("failed_gates contains an unknown gate")
        if type(self.passed) is not bool or self.passed is not (not self.failed_gates):
            raise ValueError("passed does not match failed_gates")
        if (
            self.artifact_kind != _DECISION_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("candidate decision header is invalid")
        computed = _json_sha256(self._payload(), domain=_DECISION_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="artifact_sha256"
        ) != computed:
            raise ValueError("candidate decision hash mismatch")

    @property
    def candidate_key(self) -> tuple[int, int, int, int]:
        return (
            self.layer_ordinal,
            self.chart_count,
            self.dynamic_mode_count,
            self.tangent_rank,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "candidate_sha256": self.candidate_sha256,
            "layer_ordinal": self.layer_ordinal,
            "chart_count": self.chart_count,
            "dynamic_mode_count": self.dynamic_mode_count,
            "tangent_rank": self.tangent_rank,
            "matched_m1_candidate_sha256": self.matched_m1_candidate_sha256,
            "relative_sse_improvement_over_matched_m1": (
                self.relative_sse_improvement_over_matched_m1
            ),
            "failed_gates": self.failed_gates,
            "passed": self.passed,
        }

    def validate_integrity(self) -> None:
        if (
            _json_sha256(self._payload(), domain=_DECISION_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("candidate decision hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return _json_safe_state(
            {**self._payload(), "artifact_sha256": self.artifact_sha256}
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerMultimodeCandidateGateDecision:
        expected = {
            "artifact_kind",
            "format_version",
            "candidate_sha256",
            "layer_ordinal",
            "chart_count",
            "dynamic_mode_count",
            "tangent_rank",
            "matched_m1_candidate_sha256",
            "relative_sse_improvement_over_matched_m1",
            "failed_gates",
            "passed",
            "artifact_sha256",
        }
        _strict_fields(state, expected=expected, label="candidate gate decision")
        arguments = dict(state)
        arguments["failed_gates"] = tuple(
            _require_json_list(state["failed_gates"], label="failed_gates")
        )
        return cls(**arguments)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LayerMultimodeLayerSelectionReceipt:
    layer_ordinal: int
    decisions: tuple[LayerMultimodeCandidateGateDecision, ...]
    selected_candidate_sha256: str
    selected_chart_count: int
    selected_dynamic_mode_count: int
    selected_tangent_rank: int
    selected_serialized_parameter_count: int
    selected_matrix_macs_per_token: int
    selected_validation_velocity_nrmse: float
    artifact_sha256: str = ""
    artifact_kind: str = _LAYER_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_int(self.layer_ordinal, label="layer_ordinal")
        if (
            type(self.decisions) is not tuple
            or len(self.decisions) != CANDIDATES_PER_LAYER
            or any(
                not isinstance(value, LayerMultimodeCandidateGateDecision)
                for value in self.decisions
            )
        ):
            raise ValueError(
                "decisions must contain the complete 18-candidate grid"
            )
        keys = tuple(value.candidate_key for value in self.decisions)
        if keys != tuple(sorted(keys)) or any(
            key[0] != self.layer_ordinal for key in keys
        ):
            raise ValueError(
                "candidate decisions are not in canonical layer grid order"
            )
        if {(key[1], key[2], key[3]) for key in keys} != _candidate_grid():
            raise ValueError("candidate decisions do not cover the frozen grid")
        for value in self.decisions:
            value.validate_integrity()
        _require_sha256(
            self.selected_candidate_sha256,
            label="selected_candidate_sha256",
        )
        selected = tuple(
            value
            for value in self.decisions
            if value.candidate_sha256 == self.selected_candidate_sha256
        )
        if len(selected) != 1 or not selected[0].passed:
            raise ValueError("selected candidate must identify one passing decision")
        decision = selected[0]
        if (
            self.selected_chart_count != decision.chart_count
            or self.selected_dynamic_mode_count != decision.dynamic_mode_count
            or self.selected_tangent_rank != decision.tangent_rank
        ):
            raise ValueError("selected candidate coordinates do not match decision")
        expected_parameters = candidate_serialized_parameter_count(
            self.selected_chart_count,
            self.selected_dynamic_mode_count,
            self.selected_tangent_rank,
        )
        expected_macs = candidate_matrix_macs_per_token(
            self.selected_chart_count,
            self.selected_dynamic_mode_count,
            self.selected_tangent_rank,
        )
        if (
            type(self.selected_serialized_parameter_count) is not int
            or self.selected_serialized_parameter_count != expected_parameters
        ):
            raise ValueError("selected parameter count does not recompute")
        if (
            type(self.selected_matrix_macs_per_token) is not int
            or self.selected_matrix_macs_per_token != expected_macs
        ):
            raise ValueError("selected MAC count does not recompute")
        _require_float(
            self.selected_validation_velocity_nrmse,
            label="selected_validation_velocity_nrmse",
            minimum=0.0,
        )
        if (
            self.artifact_kind != _LAYER_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("layer selection receipt header is invalid")
        computed = _json_sha256(self._payload(), domain=_LAYER_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="artifact_sha256"
        ) != computed:
            raise ValueError("layer selection receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "layer_ordinal": self.layer_ordinal,
            "decision_sha256s": tuple(
                value.artifact_sha256 for value in self.decisions
            ),
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "selected_chart_count": self.selected_chart_count,
            "selected_dynamic_mode_count": self.selected_dynamic_mode_count,
            "selected_tangent_rank": self.selected_tangent_rank,
            "selected_serialized_parameter_count": (
                self.selected_serialized_parameter_count
            ),
            "selected_matrix_macs_per_token": self.selected_matrix_macs_per_token,
            "selected_validation_velocity_nrmse": (
                self.selected_validation_velocity_nrmse
            ),
        }

    def validate_integrity(self) -> None:
        for value in self.decisions:
            value.validate_integrity()
        if _json_sha256(self._payload(), domain=_LAYER_DOMAIN) != self.artifact_sha256:
            raise ValueError("layer selection receipt hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return _json_safe_state(
            {
                **self._payload(),
                "decisions": [value.state_dict() for value in self.decisions],
                "artifact_sha256": self.artifact_sha256,
            }
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerMultimodeLayerSelectionReceipt:
        expected = {
            "artifact_kind",
            "format_version",
            "layer_ordinal",
            "decision_sha256s",
            "decisions",
            "selected_candidate_sha256",
            "selected_chart_count",
            "selected_dynamic_mode_count",
            "selected_tangent_rank",
            "selected_serialized_parameter_count",
            "selected_matrix_macs_per_token",
            "selected_validation_velocity_nrmse",
            "artifact_sha256",
        }
        _strict_fields(state, expected=expected, label="layer selection receipt")
        decision_states = _require_json_list(
            state["decisions"],
            label="decisions",
        )
        decision_hashes = _require_json_list(
            state["decision_sha256s"],
            label="decision_sha256s",
        )
        decisions = tuple(
            LayerMultimodeCandidateGateDecision.from_state_dict(value)
            for value in decision_states
        )
        if decision_hashes != [
            value.artifact_sha256 for value in decisions
        ]:
            raise ValueError("decision hash inventory is invalid")
        arguments = dict(state)
        arguments.pop("decision_sha256s")
        arguments["decisions"] = decisions
        return cls(**arguments)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LayerMultimodeSelectionReceipt:
    policy_sha256: str
    source_model_sha256: str
    fit_policy_sha256: str
    fit_policy_test_only: bool
    source_capture_scope: str
    production_fit_authorized: bool
    fit_a_sha256: str
    validation_a_sha256: str
    evaluation_protocol_sha256: str
    family_ids: tuple[str, ...]
    candidates: tuple[LayerMultimodeCandidateEvaluationReceipt, ...]
    layers: tuple[LayerMultimodeLayerSelectionReceipt, ...]
    selected_serialized_parameter_count: int
    selected_matrix_macs_per_token: int
    native_boundary_parameter_count: int
    native_matrix_macs_per_token: int
    serialized_parameter_savings: int
    matrix_mac_savings_per_token: int
    all_layers_selected: bool = True
    latency_claimed: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _SELECTION_KIND
    format_version: int = _FORMAT_VERSION
    contains_raw_tensors: bool = False
    contains_model_weights: bool = False
    contains_prompt_text: bool = False
    contains_activation_rows: bool = False
    contains_gradient_rows: bool = False
    uses_fit_a_only: bool = True
    uses_b_c_d_or_graph_inputs: bool = False
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.policy_sha256 != layer_multimode_selection_policy_sha256():
            raise ValueError("selection policy hash is invalid")
        for name in (
            "source_model_sha256",
            "fit_policy_sha256",
            "fit_a_sha256",
            "validation_a_sha256",
            "evaluation_protocol_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if type(self.fit_policy_test_only) is not bool:
            raise TypeError("fit_policy_test_only must be a bool")
        if self.source_capture_scope not in (
            PRODUCTION_FULL_A_CAPTURE_SCOPE,
            _DEVELOPMENT_SINGLE_BATCH_CAPTURE_SCOPE,
        ) or type(self.source_capture_scope) is not str:
            raise ValueError("source_capture_scope is outside the frozen boundary")
        if type(self.production_fit_authorized) is not bool:
            raise TypeError("production_fit_authorized must be a bool")
        expected_production_authority = (
            not self.fit_policy_test_only
            and self.fit_policy_sha256
            == PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256
            and self.source_capture_scope == PRODUCTION_FULL_A_CAPTURE_SCOPE
        )
        if self.production_fit_authorized is not expected_production_authority:
            raise ValueError(
                "selection production authority does not recompute from policy/scope"
            )
        if (
            type(self.family_ids) is not tuple
            or len(self.family_ids) != FAMILY_COUNT
            or self.family_ids != tuple(sorted(self.family_ids))
            or len(set(self.family_ids)) != FAMILY_COUNT
            or any(type(value) is not str or not value for value in self.family_ids)
        ):
            raise ValueError("family_ids must be six unique canonical strings")
        if (
            type(self.candidates) is not tuple
            or len(self.candidates) != LAYER_COUNT * CANDIDATES_PER_LAYER
            or any(
                type(value) is not LayerMultimodeCandidateEvaluationReceipt
                for value in self.candidates
            )
        ):
            raise ValueError("candidates must contain the complete 18-layer grid")
        candidate_keys = tuple(value.candidate_key for value in self.candidates)
        if candidate_keys != tuple(sorted(candidate_keys)) or len(
            set(candidate_keys)
        ) != len(candidate_keys):
            raise ValueError("candidate inventory is not canonical and unique")
        for candidate in self.candidates:
            candidate.validate_integrity()
            if (
                candidate.source_model_sha256 != self.source_model_sha256
                or candidate.fit_policy_sha256 != self.fit_policy_sha256
                or candidate.fit_policy_test_only is not self.fit_policy_test_only
                or candidate.source_capture_scope != self.source_capture_scope
                or candidate.production_fit_authorized
                is not self.production_fit_authorized
                or candidate.fit_a_sha256 != self.fit_a_sha256
                or candidate.validation_a_sha256 != self.validation_a_sha256
                or candidate.evaluation_protocol_sha256
                != self.evaluation_protocol_sha256
                or tuple(value.family_id for value in candidate.family_fidelity)
                != self.family_ids
            ):
                raise ValueError("candidate provenance does not match selection")
        if (
            type(self.layers) is not tuple
            or len(self.layers) != LAYER_COUNT
            or tuple(value.layer_ordinal for value in self.layers)
            != tuple(range(LAYER_COUNT))
        ):
            raise ValueError("layers must contain all 18 canonical selections")
        candidates_by_hash = {
            value.artifact_sha256: value for value in self.candidates
        }
        candidates_by_key = {
            value.candidate_key: value for value in self.candidates
        }
        for layer in self.layers:
            layer.validate_integrity()
            layer_candidates = tuple(
                value
                for value in self.candidates
                if value.layer_ordinal == layer.layer_ordinal
            )
            expected_decisions = tuple(
                _decision_for_candidate(
                    candidate,
                    candidates_by_key[
                        (
                            candidate.layer_ordinal,
                            candidate.chart_count,
                            1,
                            candidate.tangent_rank,
                        )
                    ],
                )
                for candidate in layer_candidates
            )
            if tuple(
                value.artifact_sha256 for value in layer.decisions
            ) != tuple(value.artifact_sha256 for value in expected_decisions):
                raise ValueError("layer gate decisions do not recompute")
            passing_candidates = tuple(
                candidates_by_hash[value.candidate_sha256]
                for value in layer.decisions
                if value.passed
            )
            expected_selected = min(
                passing_candidates,
                key=lambda value: (
                    value.serialized_parameter_count,
                    value.validation_velocity_nrmse,
                    value.chart_count,
                    value.dynamic_mode_count,
                    value.tangent_rank,
                ),
            )
            if (
                layer.selected_candidate_sha256
                != expected_selected.artifact_sha256
                or not _close(
                    layer.selected_validation_velocity_nrmse,
                    expected_selected.validation_velocity_nrmse,
                )
            ):
                raise ValueError("layer selected candidate does not recompute")
        if type(self.all_layers_selected) is not bool or not self.all_layers_selected:
            raise ValueError("all_layers_selected must be true")
        if type(self.latency_claimed) is not bool or self.latency_claimed:
            raise ValueError("latency_claimed must be false")
        selected_parameters = sum(
            value.selected_serialized_parameter_count for value in self.layers
        )
        selected_macs = sum(
            value.selected_matrix_macs_per_token for value in self.layers
        )
        native_parameters = LAYER_COUNT * NATIVE_MLP_BOUNDARY_PARAMETER_COUNT
        native_macs = LAYER_COUNT * NATIVE_MLP_MATRIX_MACS_PER_TOKEN
        expected_counts = {
            "selected_serialized_parameter_count": selected_parameters,
            "selected_matrix_macs_per_token": selected_macs,
            "native_boundary_parameter_count": native_parameters,
            "native_matrix_macs_per_token": native_macs,
            "serialized_parameter_savings": native_parameters - selected_parameters,
            "matrix_mac_savings_per_token": native_macs - selected_macs,
        }
        for name, expected in expected_counts.items():
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError(f"{name} does not recompute")
        for name, expected in _SAFETY_METADATA.items():
            if getattr(self, name) is not expected:
                raise ValueError("selection safety metadata is invalid")
        if (
            self.artifact_kind != _SELECTION_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("selection receipt header is invalid")
        computed = _json_sha256(self._payload(), domain=_SELECTION_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="artifact_sha256"
        ) != computed:
            raise ValueError("selection receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "policy_sha256": self.policy_sha256,
            "source_model_sha256": self.source_model_sha256,
            "fit_policy_sha256": self.fit_policy_sha256,
            "fit_policy_test_only": self.fit_policy_test_only,
            "source_capture_scope": self.source_capture_scope,
            "production_fit_authorized": self.production_fit_authorized,
            "fit_a_sha256": self.fit_a_sha256,
            "validation_a_sha256": self.validation_a_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "family_ids": self.family_ids,
            "candidate_sha256s": tuple(
                value.artifact_sha256 for value in self.candidates
            ),
            "layer_selection_sha256s": tuple(
                value.artifact_sha256 for value in self.layers
            ),
            "selected_serialized_parameter_count": (
                self.selected_serialized_parameter_count
            ),
            "selected_matrix_macs_per_token": self.selected_matrix_macs_per_token,
            "native_boundary_parameter_count": self.native_boundary_parameter_count,
            "native_matrix_macs_per_token": self.native_matrix_macs_per_token,
            "serialized_parameter_savings": self.serialized_parameter_savings,
            "matrix_mac_savings_per_token": self.matrix_mac_savings_per_token,
            "all_layers_selected": self.all_layers_selected,
            "latency_claimed": self.latency_claimed,
            **_SAFETY_METADATA,
        }

    def validate_integrity(self) -> None:
        for value in self.candidates:
            value.validate_integrity()
        for value in self.layers:
            value.validate_integrity()
        if (
            _json_sha256(self._payload(), domain=_SELECTION_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("selection receipt hash mismatch")

    @property
    def live_selector_authorized(self) -> bool:
        """True only for the original receipt returned by the selector."""

        return (
            self._construction_authority is _LIVE_SELECTOR_AUTHORITY
            and _LIVE_SELECTION_RECEIPTS.get(id(self)) is self
        )

    @property
    def live_full_grid_selection_authorized(self) -> bool:
        """True only for an original selection over 324 original evaluations."""

        return self.live_selector_authorized and all(
            candidate.live_candidate_evaluation_authorized
            for candidate in self.candidates
        )

    def state_dict(self) -> dict[str, object]:
        return _json_safe_state(
            {
                **self._payload(),
                "candidates": [
                    value.state_dict() for value in self.candidates
                ],
                "layers": [value.state_dict() for value in self.layers],
                "artifact_sha256": self.artifact_sha256,
            }
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerMultimodeSelectionReceipt:
        expected = {
            "artifact_kind",
            "format_version",
            "policy_sha256",
            "source_model_sha256",
            "fit_policy_sha256",
            "fit_policy_test_only",
            "source_capture_scope",
            "production_fit_authorized",
            "fit_a_sha256",
            "validation_a_sha256",
            "evaluation_protocol_sha256",
            "family_ids",
            "candidate_sha256s",
            "candidates",
            "layer_selection_sha256s",
            "layers",
            "selected_serialized_parameter_count",
            "selected_matrix_macs_per_token",
            "native_boundary_parameter_count",
            "native_matrix_macs_per_token",
            "serialized_parameter_savings",
            "matrix_mac_savings_per_token",
            "all_layers_selected",
            "latency_claimed",
            *_SAFETY_METADATA,
            "artifact_sha256",
        }
        _strict_fields(state, expected=expected, label="multimode selection receipt")
        candidate_states = _require_json_list(
            state["candidates"],
            label="candidates",
        )
        layer_states = _require_json_list(state["layers"], label="layers")
        candidate_hashes = _require_json_list(
            state["candidate_sha256s"],
            label="candidate_sha256s",
        )
        layer_hashes = _require_json_list(
            state["layer_selection_sha256s"],
            label="layer_selection_sha256s",
        )
        candidates = tuple(
            LayerMultimodeCandidateEvaluationReceipt.from_state_dict(value)
            for value in candidate_states
        )
        layers = tuple(
            LayerMultimodeLayerSelectionReceipt.from_state_dict(value)
            for value in layer_states
        )
        if candidate_hashes != [
            value.artifact_sha256 for value in candidates
        ]:
            raise ValueError("candidate hash inventory is invalid")
        if layer_hashes != [
            value.artifact_sha256 for value in layers
        ]:
            raise ValueError("layer selection hash inventory is invalid")
        arguments = dict(state)
        arguments.pop("candidate_sha256s")
        arguments.pop("layer_selection_sha256s")
        arguments["family_ids"] = tuple(
            _require_json_list(state["family_ids"], label="family_ids")
        )
        arguments["candidates"] = candidates
        arguments["layers"] = layers
        return cls(**arguments)  # type: ignore[arg-type]


def _candidate_grid() -> set[tuple[int, int, int]]:
    return {
        (chart_count, dynamic_mode_count, tangent_rank)
        for chart_count in CHART_COUNT_LADDER
        for dynamic_mode_count in DYNAMIC_MODE_COUNT_LADDER
        for tangent_rank in TANGENT_RANK_LADDER
    }


def _decision_for_candidate(
    candidate: LayerMultimodeCandidateEvaluationReceipt,
    matched_m1: LayerMultimodeCandidateEvaluationReceipt,
) -> LayerMultimodeCandidateGateDecision:
    failed: list[str] = []
    if not candidate.resource_pass:
        failed.append("resource_receipt_failed")
    if candidate.serialized_parameter_count >= NATIVE_MLP_MATRIX_PARAMETER_COUNT:
        failed.append("not_smaller_than_native_mlp_matrices")
    if candidate.validation_velocity_nrmse > VALIDATION_VELOCITY_NRMSE_MAXIMUM:
        failed.append("validation_velocity_nrmse")
    if candidate.aggregate_velocity_cosine < AGGREGATE_COSINE_MINIMUM:
        failed.append("aggregate_velocity_cosine")
    if (
        max(value.velocity_nrmse for value in candidate.family_fidelity)
        > FAMILY_NRMSE_MAXIMUM
    ):
        failed.append("family_velocity_nrmse")

    if candidate.dynamic_mode_count == 1:
        improvement = 0.0
    elif matched_m1.validation_prediction_sse <= 0.0:
        improvement = float("-inf")
        failed.append("matched_m1_sse_improvement")
    else:
        improvement = (
            matched_m1.validation_prediction_sse
            - candidate.validation_prediction_sse
        ) / matched_m1.validation_prediction_sse
        if improvement < MINIMUM_MODE_SSE_IMPROVEMENT:
            failed.append("matched_m1_sse_improvement")
        if candidate.minimum_conditional_mode_mass < MINIMUM_MEAN_MODE_MASS:
            failed.append("mean_mode_mass")
        if (
            candidate.minimum_per_chart_state_mode_mutual_information_nats
            < STATE_MODE_MI_MINIMUM_NATS
        ):
            failed.append("state_mode_mutual_information")
        if (
            candidate.minimum_per_chart_velocity_separation_ratio
            < MODE_VELOCITY_SEPARATION_MINIMUM
        ):
            failed.append("mode_velocity_separation")
        if (
            candidate.split_refit_matched_prediction_cosine
            < SPLIT_PREDICTION_COSINE_MINIMUM
        ):
            failed.append("split_refit_prediction_cosine")
        if (
            candidate.split_refit_hard_route_ari
            < SPLIT_HARD_ROUTE_ARI_MINIMUM
        ):
            failed.append("split_refit_hard_route_ari")

    if (
        candidate.chart_count > 1
        and candidate.minimum_chart_soft_mass < MINIMUM_CHART_SOFT_MASS
    ):
        failed.append("chart_soft_mass")

    # A strict receipt cannot serialize infinity.  The zero-SSE branch is a
    # terminal failed comparison and uses a finite sentinel below any gate.
    if not math.isfinite(improvement):
        improvement = -1.0
    canonical_failed = tuple(value for value in _GATE_ORDER if value in failed)
    return LayerMultimodeCandidateGateDecision(
        candidate_sha256=candidate.artifact_sha256,
        layer_ordinal=candidate.layer_ordinal,
        chart_count=candidate.chart_count,
        dynamic_mode_count=candidate.dynamic_mode_count,
        tangent_rank=candidate.tangent_rank,
        matched_m1_candidate_sha256=matched_m1.artifact_sha256,
        relative_sse_improvement_over_matched_m1=float(improvement),
        failed_gates=canonical_failed,
        passed=not canonical_failed,
    )


def select_layer_multimode_candidates(
    candidates: tuple[LayerMultimodeCandidateEvaluationReceipt, ...],
    *,
    allow_test_only_nonproduction: bool = False,
    allow_test_only_nonlive_authority: bool = False,
) -> LayerMultimodeSelectionReceipt:
    """Select all 18 layer architectures from a complete A-only grid.

    Production selection is fail-closed on the exact fit-policy digest and
    authenticated full-A capture scope.  The keyword opt-in exists only for
    model-free tests and emits a receipt whose production authority is false.
    """

    if type(candidates) is not tuple:
        raise TypeError("candidates must be a tuple")
    if type(allow_test_only_nonproduction) is not bool:
        raise TypeError("allow_test_only_nonproduction must be a bool")
    if type(allow_test_only_nonlive_authority) is not bool:
        raise TypeError("allow_test_only_nonlive_authority must be a bool")
    if len(candidates) != LAYER_COUNT * CANDIDATES_PER_LAYER:
        raise ValueError("candidate inventory must contain exactly 18 x 18 receipts")
    if any(
        type(value) is not LayerMultimodeCandidateEvaluationReceipt
        for value in candidates
    ):
        raise TypeError("candidate inventory contains a non-receipt value")
    for candidate in candidates:
        candidate.validate_integrity()
    canonical = tuple(sorted(candidates, key=lambda value: value.candidate_key))
    keys = tuple(value.candidate_key for value in canonical)
    if len(set(keys)) != len(keys):
        raise ValueError("candidate inventory contains duplicate coordinates")
    expected_keys = {
        (layer, chart_count, mode_count, rank)
        for layer in range(LAYER_COUNT)
        for chart_count, mode_count, rank in _candidate_grid()
    }
    if set(keys) != expected_keys:
        raise ValueError("candidate inventory does not cover the complete frozen grid")

    first = canonical[0]
    provenance = (
        first.source_model_sha256,
        first.fit_policy_sha256,
        first.fit_policy_test_only,
        first.source_capture_scope,
        first.production_fit_authorized,
        first.fit_a_sha256,
        first.validation_a_sha256,
        first.evaluation_protocol_sha256,
    )
    family_ids = tuple(value.family_id for value in first.family_fidelity)
    for candidate in canonical:
        if (
            (
                candidate.source_model_sha256,
                candidate.fit_policy_sha256,
                candidate.fit_policy_test_only,
                candidate.source_capture_scope,
                candidate.production_fit_authorized,
                candidate.fit_a_sha256,
                candidate.validation_a_sha256,
                candidate.evaluation_protocol_sha256,
            )
            != provenance
            or tuple(value.family_id for value in candidate.family_fidelity)
            != family_ids
        ):
            raise ValueError("candidate grid provenance or family catalog drifted")
    if not first.production_fit_authorized and not allow_test_only_nonproduction:
        raise ValueError(
            "production selection requires the exact production fit policy "
            "and authenticated full-A capture scope"
        )
    if (
        not first.production_fit_authorized
        and allow_test_only_nonproduction
        and not first.fit_policy_test_only
    ):
        raise ValueError(
            "nonproduction selector opt-in requires an explicitly test-only fit policy"
        )
    if allow_test_only_nonlive_authority and (
        first.production_fit_authorized or not first.fit_policy_test_only
    ):
        raise ValueError(
            "nonlive selector opt-in requires explicitly test-only "
            "nonproduction provenance"
        )
    all_candidate_evaluations_live = all(
        candidate.live_candidate_evaluation_authorized
        for candidate in canonical
    )
    if (
        not all_candidate_evaluations_live
        and not allow_test_only_nonlive_authority
    ):
        raise ValueError(
            "selection requires all 324 original live adapter evaluations; "
            "the nonlive path requires explicit test-only opt-in"
        )

    layers: list[LayerMultimodeLayerSelectionReceipt] = []
    for layer_ordinal in range(LAYER_COUNT):
        layer_candidates = tuple(
            value for value in canonical if value.layer_ordinal == layer_ordinal
        )
        layer_source_lineage = {
            (
                value.source_rows_receipt_sha256,
                value.fit_source_sha256,
                value.validation_source_sha256,
            )
            for value in layer_candidates
        }
        if len(layer_source_lineage) != 1:
            raise ValueError(
                f"layer {layer_ordinal} candidates do not share one A row/source lineage"
            )
        for chart_count in CHART_COUNT_LADDER:
            for tangent_rank in TANGENT_RANK_LADDER:
                atlases = {
                    value.atlas_sha256
                    for value in layer_candidates
                    if value.chart_count == chart_count
                    and value.tangent_rank == tangent_rank
                }
                if len(atlases) != 1:
                    raise ValueError(
                        f"layer {layer_ordinal} C={chart_count} R={tangent_rank} "
                        "candidates do not share one state atlas across M"
                    )
        reference_targets = {
            (
                value.validation_scalar_count,
                value.validation_teacher_energy,
                tuple(
                    (family.scalar_count, family.teacher_energy)
                    for family in value.family_fidelity
                ),
            )
            for value in layer_candidates
        }
        if len(reference_targets) != 1:
            raise ValueError(
                f"layer {layer_ordinal} candidates do not share one A-validation target"
            )
        by_coordinates = {
            (value.chart_count, value.dynamic_mode_count, value.tangent_rank): value
            for value in layer_candidates
        }
        decisions = tuple(
            _decision_for_candidate(
                candidate,
                by_coordinates[(candidate.chart_count, 1, candidate.tangent_rank)],
            )
            for candidate in layer_candidates
        )
        passing = tuple(
            (candidate, decision)
            for candidate, decision in zip(layer_candidates, decisions, strict=True)
            if decision.passed
        )
        if not passing:
            raise NoAdmissibleLayerMultimodeCandidateError(
                layer_ordinal,
                decisions,
            )
        selected, _ = min(
            passing,
            key=lambda item: (
                item[0].serialized_parameter_count,
                item[0].validation_velocity_nrmse,
                item[0].chart_count,
                item[0].dynamic_mode_count,
                item[0].tangent_rank,
            ),
        )
        layers.append(
            LayerMultimodeLayerSelectionReceipt(
                layer_ordinal=layer_ordinal,
                decisions=decisions,
                selected_candidate_sha256=selected.artifact_sha256,
                selected_chart_count=selected.chart_count,
                selected_dynamic_mode_count=selected.dynamic_mode_count,
                selected_tangent_rank=selected.tangent_rank,
                selected_serialized_parameter_count=(
                    selected.serialized_parameter_count
                ),
                selected_matrix_macs_per_token=selected.matrix_macs_per_token,
                selected_validation_velocity_nrmse=(
                    selected.validation_velocity_nrmse
                ),
            )
        )

    selected_parameters = sum(
        value.selected_serialized_parameter_count for value in layers
    )
    selected_macs = sum(value.selected_matrix_macs_per_token for value in layers)
    native_parameters = LAYER_COUNT * NATIVE_MLP_BOUNDARY_PARAMETER_COUNT
    native_macs = LAYER_COUNT * NATIVE_MLP_MATRIX_MACS_PER_TOKEN
    receipt = LayerMultimodeSelectionReceipt(
        policy_sha256=layer_multimode_selection_policy_sha256(),
        source_model_sha256=provenance[0],
        fit_policy_sha256=provenance[1],
        fit_policy_test_only=provenance[2],
        source_capture_scope=provenance[3],
        production_fit_authorized=provenance[4],
        fit_a_sha256=provenance[5],
        validation_a_sha256=provenance[6],
        evaluation_protocol_sha256=provenance[7],
        family_ids=family_ids,
        candidates=canonical,
        layers=tuple(layers),
        selected_serialized_parameter_count=selected_parameters,
        selected_matrix_macs_per_token=selected_macs,
        native_boundary_parameter_count=native_parameters,
        native_matrix_macs_per_token=native_macs,
        serialized_parameter_savings=native_parameters - selected_parameters,
        matrix_mac_savings_per_token=native_macs - selected_macs,
    )
    receipt.validate_integrity()
    object.__setattr__(
        receipt,
        "_construction_authority",
        _LIVE_SELECTOR_AUTHORITY,
    )
    _LIVE_SELECTION_RECEIPTS[id(receipt)] = receipt
    return receipt
