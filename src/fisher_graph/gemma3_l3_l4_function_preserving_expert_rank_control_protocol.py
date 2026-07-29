"""Pure declaration for the paired Gemma L3/L4 expert-rank control.

The experiment keeps the outer modal width fixed at 64 and compares the
authenticated width-control expert-rank-16 cold start with an expert-rank-64
function-preserving lift.  Both arms have the same observable function,
provider-chart JVP, objective, fit rows, optimizer, and training budget at
step zero.  The 48 added coordinates in each expert are initialized behind an
exactly-zero output factor and must be demonstrably gradient-open.

The authenticated width-control final provider is validation-only.  It is
never used to initialize either arm.  This module intentionally imports only
the Python standard library and does not load PyTorch, a model, or an
experiment artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re


__all__ = [
    "DEFAULT_FUNCTION_PRESERVING_EXPERT_RANK_CONTROL_PROTOCOL_SHA256",
    "EXPERT_RANK_CONTROL_EXECUTION_DEVICE",
    "EXPERT_RANK_CONTROL_EXECUTION_DTYPE",
    "EXPERT_RANK_CONTROL_OUTER_RANK",
    "EXPERT_RANK_CONTROL_PRIMARY_SEED",
    "EXPERT_RANK_CONTROL_REPLICATION_SEED",
    "EXPERT_RANK_CONTROL_SOURCE_RANK",
    "EXPERT_RANK_CONTROL_TARGET_RANK",
    "ExpertRankExecutorSpec",
    "ExpertRankPreflightBindings",
    "FunctionPreservingExpertRankControlProtocol",
    "NestedExpertRankLiftSpec",
    "PairedExpertRankTrainingSpec",
    "SourceWidthResultBindings",
    "default_function_preserving_expert_rank_control_protocol",
    "fit_replay_sequence_sha256",
]


EXPERT_RANK_CONTROL_OUTER_RANK = 64
EXPERT_RANK_CONTROL_SOURCE_RANK = 16
EXPERT_RANK_CONTROL_TARGET_RANK = 64
EXPERT_RANK_CONTROL_EXECUTION_DEVICE = "cpu"
EXPERT_RANK_CONTROL_EXECUTION_DTYPE = "float32"
EXPERT_RANK_CONTROL_PRIMARY_SEED = 20_260_728_402
EXPERT_RANK_CONTROL_REPLICATION_SEED = 20_260_729_402

_FORMAT_VERSION = 1
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_function_preserving_expert_rank_control."
    "protocol.v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXECUTOR_DOMAIN = (
    b"fisher-graph:function-preserving-expert-rank:executor:v1\0"
)
_TRAINING_DOMAIN = (
    b"fisher-graph:function-preserving-expert-rank:training:v1\0"
)
_LIFT_DOMAIN = b"fisher-graph:function-preserving-expert-rank:lift:v1\0"
_PREFLIGHT_DOMAIN = (
    b"fisher-graph:function-preserving-expert-rank:preflight:v1\0"
)
_SOURCE_DOMAIN = (
    b"fisher-graph:function-preserving-expert-rank:sources:v1\0"
)
_PROTOCOL_DOMAIN = (
    b"fisher-graph:function-preserving-expert-rank:protocol:v1\0"
)
_FIT_REPLAY_SEQUENCE_DOMAIN = (
    b"fisher-graph:function-preserving-expert-rank:"
    b"fit-replay-sequence:v1\0"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def fit_replay_sequence_sha256(
    sequence_name: str,
    values: tuple[str, ...],
) -> str:
    """Seal one exact ordered sequence of fit tensor-artifact identities."""

    if not isinstance(sequence_name, str) or not sequence_name:
        raise ValueError("fit replay sequence name must be nonempty")
    if (
        type(values) is not tuple
        or not values
        or any(
            not isinstance(value, str)
            or _SHA256.fullmatch(value) is None
            for value in values
        )
    ):
        raise ValueError("fit replay sequence must contain SHA-256 values")
    return _digest(
        {"sequence_name": sequence_name, "values": values},
        domain=_FIT_REPLAY_SEQUENCE_DOMAIN,
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return value


def _strict_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"{label} keys differ: missing={missing}, extra={extra}"
        )


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _exact_float(
    value: object,
    *,
    label: str,
    minimum: float = 0.0,
) -> float:
    if type(value) is not float or value < minimum:
        raise ValueError(f"{label} must be a float >= {minimum}")
    return value


def _bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be boolean")
    return value


def _digest_or_validate(
    *,
    supplied: str,
    payload: object,
    domain: bytes,
    label: str,
) -> str:
    computed = _digest(payload, domain=domain)
    if supplied:
        if _require_sha256(supplied, label=label) != computed:
            raise ValueError(f"{label} hash mismatch")
        return supplied
    return computed


@dataclass(frozen=True, slots=True)
class ExpertRankExecutorSpec:
    """One exact fixed-outer-width gated causal executor instance."""

    input_modes: int = 66
    output_modes: int = 64
    expert_count: int = 2
    expert_rank: int = 16
    router_width: int = 16
    same_position_skip: bool = False
    max_positive_lag: int | None = None
    router_activation: str = "tanh"
    source_normalized_routing: bool = True
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "input_modes": 66,
            "output_modes": 64,
            "expert_count": 2,
            "router_width": 16,
            "same_position_skip": False,
            "max_positive_lag": None,
            "router_activation": "tanh",
            "source_normalized_routing": True,
        }
        for name, frozen in expected.items():
            value = getattr(self, name)
            if value != frozen or type(value) is not type(frozen):
                raise ValueError(f"expert-rank executor {name} is not frozen")
        if type(self.expert_rank) is not int or self.expert_rank not in {
            EXPERT_RANK_CONTROL_SOURCE_RANK,
            EXPERT_RANK_CONTROL_TARGET_RANK,
        }:
            raise ValueError("executor expert rank must be 16 or 64")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_EXECUTOR_DOMAIN,
                label="expert-rank executor",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.expert_rank_control_executor",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "outer_rank": EXPERT_RANK_CONTROL_OUTER_RANK,
            "causality_semantics": (
                "unlimited_strict_positive_logical_lag_no_same_position_skip"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "ExpertRankExecutorSpec":
        state = _mapping(raw, label="expert-rank executor")
        expected = set(cls().state_dict())
        _strict_keys(state, expected=expected, label="expert-rank executor")
        defaults = cls()._payload()
        for name in (
            "artifact_kind",
            "format_version",
            "outer_rank",
            "causality_semantics",
        ):
            if state[name] != defaults[name]:
                raise ValueError("expert-rank executor semantics drifted")
        lag = state["max_positive_lag"]
        if lag is not None:
            _exact_int(lag, label="max positive lag", minimum=1)
        return cls(
            input_modes=_exact_int(
                state["input_modes"], label="input modes", minimum=1
            ),
            output_modes=_exact_int(
                state["output_modes"], label="output modes", minimum=1
            ),
            expert_count=_exact_int(
                state["expert_count"], label="expert count", minimum=1
            ),
            expert_rank=_exact_int(
                state["expert_rank"], label="expert rank", minimum=1
            ),
            router_width=_exact_int(
                state["router_width"], label="router width", minimum=1
            ),
            same_position_skip=_bool(
                state["same_position_skip"], label="same-position skip"
            ),
            max_positive_lag=lag,  # type: ignore[arg-type]
            router_activation=str(state["router_activation"]),
            source_normalized_routing=_bool(
                state["source_normalized_routing"],
                label="source-normalized routing",
            ),
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class PairedExpertRankTrainingSpec:
    """Exact D3 objective, data, gauge, gates, and paired schedule."""

    recipe_id: str = "d3_unit_rms_family_balanced_direction"
    training_metric: str = "fit_teacher_weighted_rms"
    signed_pair_multiplicity: int = 2
    pointwise_weight: float = 1.0
    relative_delta_weight: float = 2.0
    direction_weight: float = 2.0
    midpoint_jvp_weight: float = 1.0
    intended_null_weight: float = 1.0
    sensitivity_relative_floor: float = 1e-6
    direction_norm_floor: float = 1e-8
    jvp_relative_floor: float = 1e-6
    steps: int = 600
    learning_rate: float = 1e-3
    primary_seed: int = EXPERT_RANK_CONTROL_PRIMARY_SEED
    replication_seed: int = EXPERT_RANK_CONTROL_REPLICATION_SEED
    fit_data_binding_sha256: str = (
        "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
    )
    fit_batch_sha256s_sequence_sha256: str = (
        "517b165912a5f9c67cd9de77befbf1cdcaf4a11009695b73f7897a2ded6b3d45"
    )
    fit_batch_content_sha256s_sequence_sha256: str = (
        "fe26b9c8c5cc72e6c1e93a64f781f0be27eac938365f08e597ae94c480357d7b"
    )
    fit_indexed_batch_sha256s_sequence_sha256: str = (
        "58eea7d3c775ecf52ba19aad130c64ca86c46d165fb6be3384d50867e6da0f45"
    )
    fit_pair_sha256s_sequence_sha256: str = (
        "ebd5c2fcb8f0273f2ff67398fd3a8c3e3f0b7728bfd741d6e734453c762afb43"
    )
    ordinary_gates_sha256: str = (
        "0ef08515366888ff11f83269c68fc154477202b64bf2308e6b68985b05e91cd5"
    )
    contrast_gates_sha256: str = (
        "bedf561b190b04f880aabad6020ddb680d187734d572c9ec1abba7573cad0de1"
    )
    measurement_evidence_sha256: str = (
        "54b27fdbda5d471b71ac507e0b54fc90ecc892d8a546ba5c3871d3f182a3942c"
    )
    source_model_sha256: str = (
        "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
    )
    basis_package_payload_sha256: str = (
        "b2217153911436673f2ff7475c658c928112e802f5999619393287d2b0803c01"
    )
    basis_package_file_sha256: str = (
        "359c9659358cbaf97232848a10bdf0e2261d95820ad5effda9bdafeead6a7605"
    )
    pre_feedforward_norm_sha256: str = (
        "53a898a10dd76575f0d603f06ef33a9af73b91335e0f1a3c4fec1d475e2706f9"
    )
    canonical_metric_weight_sha256: str = (
        "e9d643a297c583a4dfe4a264ef49bd525e9d33fd4786e1ad1f4272a8d8ccf5ac"
    )
    standardized_gauge_sha256: str = (
        "e1bd1659b762476aee4622d3473fa12d6efcca70f7e72bd4b4b45ee86a8413b7"
    )
    unit_rms_gauge_sha256: str = (
        "4a553347335815c56643fdde56c32247e32153ed20e8c713626e35a9a072c312"
    )
    controls_sha256: str = (
        "7c150b9d051abea9a3f9dbbc934465932aacd4a62b73558b22b1bf89075c964d"
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "recipe_id": "d3_unit_rms_family_balanced_direction",
            "training_metric": "fit_teacher_weighted_rms",
            "signed_pair_multiplicity": 2,
            "pointwise_weight": 1.0,
            "relative_delta_weight": 2.0,
            "direction_weight": 2.0,
            "midpoint_jvp_weight": 1.0,
            "intended_null_weight": 1.0,
            "sensitivity_relative_floor": 1e-6,
            "direction_norm_floor": 1e-8,
            "jvp_relative_floor": 1e-6,
            "steps": 600,
            "learning_rate": 1e-3,
            "primary_seed": EXPERT_RANK_CONTROL_PRIMARY_SEED,
            "replication_seed": EXPERT_RANK_CONTROL_REPLICATION_SEED,
            "fit_data_binding_sha256": (
                "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
            ),
            "fit_batch_sha256s_sequence_sha256": (
                "517b165912a5f9c67cd9de77befbf1cdcaf4a11009695b73f"
                "7897a2ded6b3d45"
            ),
            "fit_batch_content_sha256s_sequence_sha256": (
                "fe26b9c8c5cc72e6c1e93a64f781f0be27eac938365f08e5"
                "97ae94c480357d7b"
            ),
            "fit_indexed_batch_sha256s_sequence_sha256": (
                "58eea7d3c775ecf52ba19aad130c64ca86c46d165fb6be338"
                "4d50867e6da0f45"
            ),
            "fit_pair_sha256s_sequence_sha256": (
                "ebd5c2fcb8f0273f2ff67398fd3a8c3e3f0b7728bfd741d6"
                "e734453c762afb43"
            ),
            "ordinary_gates_sha256": (
                "0ef08515366888ff11f83269c68fc154477202b64bf2308e6b68985b05e91cd5"
            ),
            "contrast_gates_sha256": (
                "bedf561b190b04f880aabad6020ddb680d187734d572c9ec1"
                "abba7573cad0de1"
            ),
            "measurement_evidence_sha256": (
                "54b27fdbda5d471b71ac507e0b54fc90ecc892d8a546ba5c"
                "3871d3f182a3942c"
            ),
            "source_model_sha256": (
                "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9"
                "cc38260ca945d4b9"
            ),
            "basis_package_payload_sha256": (
                "b2217153911436673f2ff7475c658c928112e802f599961939"
                "3287d2b0803c01"
            ),
            "basis_package_file_sha256": (
                "359c9659358cbaf97232848a10bdf0e2261d95820ad5effda"
                "9bdafeead6a7605"
            ),
            "pre_feedforward_norm_sha256": (
                "53a898a10dd76575f0d603f06ef33a9af73b91335e0f1a3c4"
                "fec1d475e2706f9"
            ),
            "canonical_metric_weight_sha256": (
                "e9d643a297c583a4dfe4a264ef49bd525e9d33fd4786e1ad1"
                "f4272a8d8ccf5ac"
            ),
            "standardized_gauge_sha256": (
                "e1bd1659b762476aee4622d3473fa12d6efcca70f7e72bd4b"
                "4b45ee86a8413b7"
            ),
            "unit_rms_gauge_sha256": (
                "4a553347335815c56643fdde56c32247e32153ed20e8c71362"
                "6e35a9a072c312"
            ),
            "controls_sha256": (
                "7c150b9d051abea9a3f9dbbc934465932aacd4a62b73558b2"
                "2b1bf89075c964d"
            ),
        }
        for name, frozen in expected.items():
            value = getattr(self, name)
            if value != frozen or type(value) is not type(frozen):
                raise ValueError(f"paired expert-rank {name} is not frozen")
        for name in self.__dataclass_fields__:
            if name.endswith("_sha256") and name != "artifact_sha256":
                _require_sha256(getattr(self, name), label=name)
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_TRAINING_DOMAIN,
                label="paired expert-rank training",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.paired_expert_rank_training",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "optimizer": "fresh_adam_per_arm",
            "batch_semantics": "exact_authenticated_d3_order",
            "initialization_schedule": (
                "regenerate_width_control_initial_outer64_lift_per_seed"
            ),
            "source_final_provider_initialization_allowed": False,
            "primary_schedule": (
                "expert16_width_primary_replay_then_expert64_nested_lift"
            ),
            "replication_schedule": (
                "conditional_fresh_cold_paired_seed_only_after_valid_primary_"
                "expert16_fail_expert64_pass"
            ),
            "fit_materialization_roles": ["pilot", "fit"],
            "canonical_scoring_semantics": (
                "raw_canonical_fisher_metric_for_all_fit_scoring"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "PairedExpertRankTrainingSpec":
        state = _mapping(raw, label="paired expert-rank training")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="paired expert-rank training",
        )
        defaults = cls()._payload()
        semantic_names = set(defaults) - {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError("paired expert-rank semantics drifted")
        fields = {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        return cls(
            **{name: state[name] for name in fields},  # type: ignore[arg-type]
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class NestedExpertRankLiftSpec:
    """Function-preserving balanced signed expert-factor expansion."""

    source_rank: int = EXPERT_RANK_CONTROL_SOURCE_RANK
    target_rank: int = EXPERT_RANK_CONTROL_TARGET_RANK
    added_rank: int = 48
    equivalence_absolute_tolerance: float = 1e-12
    equivalence_relative_tolerance: float = 1e-12
    gradient_norm_floor: float = 1e-12
    coordinate_indexing: str = "zero_based"
    added_coordinate_count: int = 48
    added_coordinate_rule: str = (
        "for_j_0_through_47_u_row_17_plus_j_column_16_plus_j"
    )
    expert0_added_u_value: float = 1.0
    expert1_added_u_value: float = -1.0
    added_v_initialization: str = "exact_zero"
    copied_parameter_rule: str = (
        "copy_all_shared_parameters_and_first16_u_columns_and_v_rows_exactly"
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "source_rank": 16,
            "target_rank": 64,
            "added_rank": 48,
            "equivalence_absolute_tolerance": 1e-12,
            "equivalence_relative_tolerance": 1e-12,
            "gradient_norm_floor": 1e-12,
            "coordinate_indexing": "zero_based",
            "added_coordinate_count": 48,
            "added_coordinate_rule": (
                "for_j_0_through_47_u_row_17_plus_j_column_16_plus_j"
            ),
            "expert0_added_u_value": 1.0,
            "expert1_added_u_value": -1.0,
            "added_v_initialization": "exact_zero",
            "copied_parameter_rule": (
                "copy_all_shared_parameters_and_first16_u_columns_and_"
                "v_rows_exactly"
            ),
        }
        for name, frozen in expected.items():
            value = getattr(self, name)
            if value != frozen or type(value) is not type(frozen):
                raise ValueError(f"nested expert-rank lift {name} is not frozen")
        if self.target_rank - self.source_rank != self.added_rank:
            raise ValueError("nested expert-rank arithmetic drifted")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_LIFT_DOMAIN,
                label="nested expert-rank lift",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.nested_expert_rank_lift",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "u_tensor_semantics": (
                "expert_input_weight_expert_input_mode_expert_rank"
            ),
            "v_tensor_semantics": (
                "expert_output_weight_expert_rank_output_mode"
            ),
            "balanced_signed_identity_semantics": (
                "expert0_positive_expert1_negative_same_added_coordinates"
            ),
            "observable_initialization": "exact_expert16_function",
            "jvp_initialization": "exact_expert16_provider_chart_jvp",
            "gradient_open_rule": (
                "step1_added_v_gradient_above_floor_added_u_gradient_exact_"
                "zero_then_step2_added_u_gradient_and_u_v_parameter_deltas_"
                "above_floor"
            ),
            "split_wrapper_parity_rule": (
                "split_and_concatenated_outputs_and_jvps_match_within_"
                "equivalence_tolerances"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "NestedExpertRankLiftSpec":
        state = _mapping(raw, label="nested expert-rank lift")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="nested expert-rank lift",
        )
        defaults = cls()._payload()
        semantic_names = set(defaults) - {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError("nested expert-rank lift semantics drifted")
        return cls(
            source_rank=_exact_int(
                state["source_rank"], label="source rank", minimum=1
            ),
            target_rank=_exact_int(
                state["target_rank"], label="target rank", minimum=1
            ),
            added_rank=_exact_int(
                state["added_rank"], label="added rank", minimum=1
            ),
            equivalence_absolute_tolerance=_exact_float(
                state["equivalence_absolute_tolerance"],
                label="absolute tolerance",
            ),
            equivalence_relative_tolerance=_exact_float(
                state["equivalence_relative_tolerance"],
                label="relative tolerance",
            ),
            gradient_norm_floor=_exact_float(
                state["gradient_norm_floor"], label="gradient floor"
            ),
            coordinate_indexing=str(state["coordinate_indexing"]),
            added_coordinate_count=_exact_int(
                state["added_coordinate_count"],
                label="added coordinate count",
                minimum=1,
            ),
            added_coordinate_rule=str(state["added_coordinate_rule"]),
            expert0_added_u_value=_exact_float(
                state["expert0_added_u_value"],
                label="expert zero added U value",
            ),
            expert1_added_u_value=float(state["expert1_added_u_value"]),
            added_v_initialization=str(state["added_v_initialization"]),
            copied_parameter_rule=str(state["copied_parameter_rule"]),
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ExpertRankPreflightBindings:
    """Exact two-step implementation audit for both frozen seeds."""

    preflight_steps: int = 2
    expected_jvp_pair_count: int = 32
    extra_u_initial_sha256: str = (
        "c5ed3867d712bb2ffd567204229137ece7e977f19cdbe69dff28d3b78a23d911"
    )
    extra_v_initial_sha256: str = (
        "f649e254fd3858b5f8e42e37cd64870509638ab39e672ed44589f00042fc0bbe"
    )
    primary_initialization_audit_sha256: str = (
        "1e0581479ad9b78339c616fa60ef882381698426647bb7e15878d15e8668cae4"
    )
    primary_control_initial_executor_sha256: str = (
        "916a22f45e0f6a34213261e361eb060a0eed094cce56bd2a795283379f59434d"
    )
    primary_treatment_initial_executor_sha256: str = (
        "eb0493e6f7c20cae8c6a8caa3bc80d30709d04e9a7c4da6380f4c97f973a6e63"
    )
    primary_treatment_base_executor_sha256: str = (
        "717bdc26c43e7d8d78e93a5577d7b68e7656a5c98b990b9f1144d64ab59aabc2"
    )
    primary_initial_observable_absolute_error: float = 0.0
    primary_initial_observable_relative_error: float = 0.0
    primary_initial_jvp_absolute_error: float = 0.0
    primary_initial_jvp_relative_error: float = 0.0
    primary_initial_wrapper_output_absolute_error: float = (
        8.881784197001252e-16
    )
    primary_initial_wrapper_output_relative_error: float = (
        5.4739078996963417e-17
    )
    primary_initial_wrapper_jvp_absolute_error: float = (
        1.7763568394002505e-15
    )
    primary_initial_wrapper_jvp_relative_error: float = (
        3.856726769329641e-17
    )
    primary_gradient_audit_sha256: str = (
        "50300a4317c680543fa080b3766688656ad5d384366b171628b908af8c73fa73"
    )
    primary_step1_v_gradient_norm: float = 1.9633094586148923
    primary_step1_u_gradient_norm: float = 0.0
    primary_step1_v_parameter_delta_norm: float = 0.03918682697223519
    primary_step1_u_parameter_delta_norm: float = 0.0
    primary_step2_v_gradient_norm: float = 1.6076096838506702
    primary_step2_u_gradient_norm: float = 0.014977110302041203
    primary_step2_v_parameter_delta_norm: float = 0.09194759732980043
    primary_step2_u_parameter_delta_norm: float = 0.05920228747007784
    primary_postfit_parity_sha256: str = (
        "8d97d6af566afdd8d3910fa9d1b98c4ae35c2576e81503c1de35c84d842bceb9"
    )
    primary_postfit_output_absolute_error: float = (
        3.552713678800501e-15
    )
    primary_postfit_output_relative_error: float = (
        9.252682324830749e-17
    )
    primary_postfit_jvp_absolute_error: float = 7.105427357601002e-15
    primary_postfit_jvp_relative_error: float = (
        1.0280229133187093e-16
    )
    primary_postfit_weighted_total_absolute_error: float = 0.0
    primary_control_postfit_executor_sha256: str = (
        "693e989bbc44b69fbb1c2a4bdc3795e6b0f55efc147d338534ddd59161d56d2b"
    )
    primary_control_postfit_metrics_sha256: str = (
        "c33c956512f103f38789afd6b5118121adeb44a531f55ba52475238a7aef41a0"
    )
    primary_postfit_concatenated_executor_sha256: str = (
        "49e86f324cd5368f15d6d26e7fae851cf6b1332188fced706d28e90f8e94c362"
    )
    primary_postfit_metrics_sha256: str = (
        "d110675f411b01a8c7114afe51e05002a4663155dd786cc318cf22e904f274bb"
    )
    replication_initialization_audit_sha256: str = (
        "b738c7c98120817a6dc796ad6ed7e7d3473c551234d7881b6bcd504cf8fe9206"
    )
    replication_control_initial_executor_sha256: str = (
        "44dc28b6f11556dfcc7afc1d6d8362a26f9f63bfd34320552fd3074e67f0e646"
    )
    replication_treatment_initial_executor_sha256: str = (
        "24cdb9f1cb1a7b55fe52d14953a4e4a884228c76fed19633ffd8d7673cd3caae"
    )
    replication_treatment_base_executor_sha256: str = (
        "94d03be042711fd277cf46ab82fee16c2353dcc01d41550a753d5b7ee71ff572"
    )
    replication_initial_observable_absolute_error: float = 0.0
    replication_initial_observable_relative_error: float = 0.0
    replication_initial_jvp_absolute_error: float = 0.0
    replication_initial_jvp_relative_error: float = 0.0
    replication_initial_wrapper_output_absolute_error: float = (
        8.881784197001252e-16
    )
    replication_initial_wrapper_output_relative_error: float = (
        5.20174161758012e-17
    )
    replication_initial_wrapper_jvp_absolute_error: float = (
        1.7763568394002505e-15
    )
    replication_initial_wrapper_jvp_relative_error: float = (
        4.484019282409784e-17
    )
    replication_gradient_audit_sha256: str = (
        "c8cc7d21815d06ceb6a5ec13c477663d54e291521bad8e5653d81185b131d3c8"
    )
    replication_step1_v_gradient_norm: float = 2.520408133261302
    replication_step1_u_gradient_norm: float = 0.0
    replication_step1_v_parameter_delta_norm: float = 0.03919088590697034
    replication_step1_u_parameter_delta_norm: float = 0.0
    replication_step2_v_gradient_norm: float = 2.052136171037132
    replication_step2_u_gradient_norm: float = 0.018177312436317483
    replication_step2_v_parameter_delta_norm: float = 0.09173359550260829
    replication_step2_u_parameter_delta_norm: float = (
        0.05921275340740287
    )
    replication_postfit_parity_sha256: str = (
        "3cd6fe3db3ad3c3443fe0c3867b04f7ebcc3ea87de2e7352010cda22be8d50ec"
    )
    replication_postfit_output_absolute_error: float = (
        1.7763568394002505e-15
    )
    replication_postfit_output_relative_error: float = (
        9.795356052744377e-17
    )
    replication_postfit_jvp_absolute_error: float = (
        7.105427357601002e-15
    )
    replication_postfit_jvp_relative_error: float = (
        1.3179666040752108e-16
    )
    replication_postfit_weighted_total_absolute_error: float = 0.0
    replication_control_postfit_executor_sha256: str = (
        "84df9102b6aa7a5c80a5321ad57eca8731fe56610a798f754cd1004722f5d468"
    )
    replication_control_postfit_metrics_sha256: str = (
        "cec05fc379a63d38d19a1d12979a6264b665594d92a4c97eb29b68d96390da5e"
    )
    replication_postfit_concatenated_executor_sha256: str = (
        "aa6340de201ef4cd9c26da8646058c16c072ade852409f5d5de1dcad770495ca"
    )
    replication_postfit_metrics_sha256: str = (
        "ec5bc7dd3af449b3ebb2ebcb0bcc8d54b0bb54c811bc2f5c55c9de4545f5c646"
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        frozen = {
            name: field.default
            for name, field in self.__dataclass_fields__.items()
            if name != "artifact_sha256"
        }
        for name, expected in frozen.items():
            value = getattr(self, name)
            if value != expected or type(value) is not type(expected):
                raise ValueError(f"expert-rank preflight {name} is not frozen")
            if name.endswith("_sha256"):
                _require_sha256(value, label=name)
        tolerance = 1e-12
        for role in ("primary", "replication"):
            if getattr(self, f"{role}_step1_v_gradient_norm") <= tolerance:
                raise ValueError("step-one V gradient is not open")
            if getattr(self, f"{role}_step1_v_parameter_delta_norm") <= (
                tolerance
            ):
                raise ValueError("step-one V parameter is not open")
            if getattr(self, f"{role}_step1_u_gradient_norm") != 0.0:
                raise ValueError("step-one U gradient is not exactly zero")
            if getattr(self, f"{role}_step1_u_parameter_delta_norm") != 0.0:
                raise ValueError("step-one U parameter changed")
            for quantity in ("u", "v"):
                if (
                    getattr(
                        self,
                        f"{role}_step2_{quantity}_gradient_norm",
                    )
                    <= tolerance
                    or getattr(
                        self,
                        f"{role}_step2_{quantity}_parameter_delta_norm",
                    )
                    <= tolerance
                ):
                    raise ValueError("step-two added factor is not open")
            for stage in ("initial_wrapper", "postfit"):
                for kind in ("output", "jvp"):
                    if (
                        getattr(
                            self,
                            f"{role}_{stage}_{kind}_absolute_error",
                        )
                        > tolerance
                        or getattr(
                            self,
                            f"{role}_{stage}_{kind}_relative_error",
                        )
                        > tolerance
                    ):
                        raise ValueError("split-wrapper parity failed")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_PREFLIGHT_DOMAIN,
                label="expert-rank preflight bindings",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.expert_rank_preflight_bindings",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "execution_scope": (
                "fit_side_two_step_only_no_scoring_or_publication"
            ),
            "initial_equivalence_semantics": (
                "control_treatment_and_split_concatenated_output_jvp"
            ),
            "postfit_parity_semantics": (
                "split_training_wrapper_equals_concatenated_executor_on_all_"
                "fit_batches_32_provider_chart_jvps_and_weighted_total"
            ),
            "top_level_unsealed_preflight_sha256_frozen": False,
            "top_level_hash_exclusion_reason": (
                "contains_protocol_unsealed_sha256_and_would_be_circular"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def for_role(self, pair_role: str) -> dict[str, object]:
        """Return the exact nested audit values expected for one seed role."""

        if pair_role not in {"primary", "replication"}:
            raise ValueError("preflight pair role is invalid")
        prefix = pair_role
        seed = (
            EXPERT_RANK_CONTROL_PRIMARY_SEED
            if pair_role == "primary"
            else EXPERT_RANK_CONTROL_REPLICATION_SEED
        )
        return {
            "pair_role": pair_role,
            "seed": seed,
            "preflight_steps": self.preflight_steps,
            "expected_jvp_pair_count": self.expected_jvp_pair_count,
            "initialization_audit_sha256": getattr(
                self,
                f"{prefix}_initialization_audit_sha256",
            ),
            "control_initial_executor_sha256": getattr(
                self,
                f"{prefix}_control_initial_executor_sha256",
            ),
            "treatment_initial_executor_sha256": getattr(
                self,
                f"{prefix}_treatment_initial_executor_sha256",
            ),
            "treatment_base_executor_sha256": getattr(
                self,
                f"{prefix}_treatment_base_executor_sha256",
            ),
            "extra_u_initial_sha256": self.extra_u_initial_sha256,
            "extra_v_initial_sha256": self.extra_v_initial_sha256,
            "initial_equivalence": {
                "maximum_observable_absolute_error": getattr(
                    self,
                    f"{prefix}_initial_observable_absolute_error",
                ),
                "maximum_observable_relative_error": getattr(
                    self,
                    f"{prefix}_initial_observable_relative_error",
                ),
                "maximum_jvp_absolute_error": getattr(
                    self,
                    f"{prefix}_initial_jvp_absolute_error",
                ),
                "maximum_jvp_relative_error": getattr(
                    self,
                    f"{prefix}_initial_jvp_relative_error",
                ),
                "maximum_wrapper_concat_observable_absolute_error": getattr(
                    self,
                    f"{prefix}_initial_wrapper_output_absolute_error",
                ),
                "maximum_wrapper_concat_observable_relative_error": getattr(
                    self,
                    f"{prefix}_initial_wrapper_output_relative_error",
                ),
                "maximum_wrapper_concat_jvp_absolute_error": getattr(
                    self,
                    f"{prefix}_initial_wrapper_jvp_absolute_error",
                ),
                "maximum_wrapper_concat_jvp_relative_error": getattr(
                    self,
                    f"{prefix}_initial_wrapper_jvp_relative_error",
                ),
            },
            "treatment_gradient_audit_sha256": getattr(
                self,
                f"{prefix}_gradient_audit_sha256",
            ),
            "treatment_gradient": {
                "step1_extra_output_gradient_norm": getattr(
                    self,
                    f"{prefix}_step1_v_gradient_norm",
                ),
                "step1_extra_input_gradient_norm": getattr(
                    self,
                    f"{prefix}_step1_u_gradient_norm",
                ),
                "step1_extra_output_parameter_delta_norm": getattr(
                    self,
                    f"{prefix}_step1_v_parameter_delta_norm",
                ),
                "step1_extra_input_parameter_delta_norm": getattr(
                    self,
                    f"{prefix}_step1_u_parameter_delta_norm",
                ),
                "step2_extra_output_gradient_norm": getattr(
                    self,
                    f"{prefix}_step2_v_gradient_norm",
                ),
                "step2_extra_input_gradient_norm": getattr(
                    self,
                    f"{prefix}_step2_u_gradient_norm",
                ),
                "step2_extra_output_parameter_delta_norm": getattr(
                    self,
                    f"{prefix}_step2_v_parameter_delta_norm",
                ),
                "step2_extra_input_parameter_delta_norm": getattr(
                    self,
                    f"{prefix}_step2_u_parameter_delta_norm",
                ),
            },
            "two_step_postfit_parity_sha256": getattr(
                self,
                f"{prefix}_postfit_parity_sha256",
            ),
            "control_two_step": {
                "metrics_sha256": getattr(
                    self,
                    f"{prefix}_control_postfit_metrics_sha256",
                ),
                "executor_sha256": getattr(
                    self,
                    f"{prefix}_control_postfit_executor_sha256",
                ),
            },
            "two_step_postfit_parity": {
                "maximum_output_absolute_error": getattr(
                    self,
                    f"{prefix}_postfit_output_absolute_error",
                ),
                "maximum_output_relative_error": getattr(
                    self,
                    f"{prefix}_postfit_output_relative_error",
                ),
                "maximum_jvp_absolute_error": getattr(
                    self,
                    f"{prefix}_postfit_jvp_absolute_error",
                ),
                "maximum_jvp_relative_error": getattr(
                    self,
                    f"{prefix}_postfit_jvp_relative_error",
                ),
                "weighted_total_absolute_error": getattr(
                    self,
                    f"{prefix}_postfit_weighted_total_absolute_error",
                ),
                "concatenated_executor_sha256": getattr(
                    self,
                    f"{prefix}_postfit_concatenated_executor_sha256",
                ),
                "metrics_sha256": getattr(
                    self,
                    f"{prefix}_postfit_metrics_sha256",
                ),
            },
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "ExpertRankPreflightBindings":
        state = _mapping(raw, label="expert-rank preflight bindings")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="expert-rank preflight bindings",
        )
        defaults = cls()._payload()
        semantic_names = set(defaults) - {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError("expert-rank preflight semantics drifted")
        fields = {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        return cls(
            **{name: state[name] for name in fields},  # type: ignore[arg-type]
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class SourceWidthResultBindings:
    """Authenticated width result and exact outer-64 primary identity."""

    width_protocol_sha256: str = (
        "c3ad81c84d41108839b5fcab13e3b5d47d99a55ae9a9223c3f116edb6b457597"
    )
    width_code_bundle_sha256: str = (
        "5c314fff7959f659257911ca0190605ea4ef41c556bd18a27108acb48d2545a4"
    )
    width_logical_artifact_sha256: str = (
        "9e07c7208b3b690a8024bd809a0d80c2842145cfa73e655bb737e5497913ce47"
    )
    width_tensor_file_sha256: str = (
        "5a3c8de7bd6731a78904a14c488648f6641d6b3cbe96167438f633b65f9104c5"
    )
    width_report_sha256: str = (
        "6aacad6f05e3b43bbeba62b6ce7ae35897af6af60d53f9dfa96eec951ad6965f"
    )
    width_outcome: str = "primary_both_fail"
    width_primary_comparison_status: str = "both_fail"
    width_primary_treatment_valid: bool = True
    width_expert_core_control_authorized: bool = True
    width_primary_rank64_plan_sha256: str = (
        "b5de47f2fb89e0198a38d851649e566eaa65a884e2be0eb5201528078d58383b"
    )
    width_primary_rank64_result_sha256: str = (
        "7e3b675cdd031d5c09b8cbbd0b72506cf2ac79254cb3c57f21e56147c203f59a"
    )
    width_primary_rank64_initial_metrics_sha256: str = (
        "56ee6759438f98b1ba9628c055358d29622af48899d49a51e23297ad9779aa20"
    )
    width_primary_rank64_final_metrics_sha256: str = (
        "a2d1581a38f0f5c4cc1b66c363642e93fce5ba205b88d55670bb351d0e35d1e1"
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        frozen = {
            name: field.default
            for name, field in self.__dataclass_fields__.items()
            if name != "artifact_sha256"
        }
        for name, expected in frozen.items():
            value = getattr(self, name)
            if value != expected or type(value) is not type(expected):
                raise ValueError(f"source width result {name} is not frozen")
            if name.endswith("_sha256"):
                _require_sha256(value, label=name)
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_SOURCE_DOMAIN,
                label="source width result bindings",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.expert_rank_source_width_result",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "source_replication_executed": False,
            "source_outer_rank": 64,
            "source_expert_rank": 16,
            "source_final_parameter_use": "validation_only",
            "source_final_parameter_initialization_allowed": False,
            "source_primary_replay_requirement": (
                "expert16_primary_must_exactly_reproduce_bound_plan_and_"
                "initial_and_final_metrics"
            ),
            "external_receipt_semantics": (
                "durable_logical_tensor_report_sha256_triple"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "SourceWidthResultBindings":
        state = _mapping(raw, label="source width result bindings")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="source width result bindings",
        )
        defaults = cls()._payload()
        semantic_names = set(defaults) - {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError("source width result semantics drifted")
        fields = {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        return cls(
            **{name: state[name] for name in fields},  # type: ignore[arg-type]
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class FunctionPreservingExpertRankControlProtocol:
    """Complete immutable declaration for the expert-rank control."""

    execution_device: str
    execution_dtype: str
    expert16_executor: ExpertRankExecutorSpec
    expert64_executor: ExpertRankExecutorSpec
    training: PairedExpertRankTrainingSpec
    lift: NestedExpertRankLiftSpec
    preflight: ExpertRankPreflightBindings
    source: SourceWidthResultBindings
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.execution_device != EXPERT_RANK_CONTROL_EXECUTION_DEVICE:
            raise ValueError("execution device must remain cpu")
        if self.execution_dtype != EXPERT_RANK_CONTROL_EXECUTION_DTYPE:
            raise ValueError("execution dtype must remain float32")
        if self.expert16_executor != ExpertRankExecutorSpec(expert_rank=16):
            raise ValueError("expert-rank-16 executor drifted")
        if self.expert64_executor != ExpertRankExecutorSpec(expert_rank=64):
            raise ValueError("expert-rank-64 executor drifted")
        if self.training != PairedExpertRankTrainingSpec():
            raise ValueError("paired expert-rank training drifted")
        if self.lift != NestedExpertRankLiftSpec():
            raise ValueError("nested expert-rank lift drifted")
        if self.preflight != ExpertRankPreflightBindings():
            raise ValueError("expert-rank preflight bindings drifted")
        if self.source != SourceWidthResultBindings():
            raise ValueError("source width result bindings drifted")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_PROTOCOL_DOMAIN,
                label="function-preserving expert-rank protocol",
            ),
        )

    @property
    def protocol_sha256(self) -> str:
        return self.artifact_sha256

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "scientific_scope": (
                "fit_only_paired_function_preserving_expert_rank_control"
            ),
            "execution_device": self.execution_device,
            "execution_dtype": self.execution_dtype,
            "outer_rank": EXPERT_RANK_CONTROL_OUTER_RANK,
            "expert16_executor": self.expert16_executor.state_dict(),
            "expert64_executor": self.expert64_executor.state_dict(),
            "training": self.training.state_dict(),
            "lift": self.lift.state_dict(),
            "preflight": self.preflight.state_dict(),
            "source": self.source.state_dict(),
            "controlled_change": (
                "gradient_open_nested_expert_rank_16_to_64_at_fixed_outer64_"
                "and_exact_initial_observable_and_jvp"
            ),
            "unchanged_contract": (
                "outer_rank_encoder_decoder_same_position_router_routing_"
                "causality_d3_objective_fit_data_order_optimizer_gauge_gates_"
                "seeds_device_dtype"
            ),
            "primary_arms": [
                "expert16_exact_width_primary_replay",
                "expert64_function_preserving_lift",
            ],
            "primary_decision": (
                "expert16_fail_expert64_pass_authorizes_paired_replication_only"
            ),
            "replication_decision": (
                "expert16_fail_expert64_pass_supports_inner_expert_rank_"
                "attribution"
            ),
            "both_fail_authority": (
                "expert_rank_alone_insufficient_under_matched_fit_budget"
            ),
            "valid_primary_both_fail_next_rung": (
                "separately_preregister_fixed_outer64_matched_expert_count_"
                "control_only"
            ),
            "both_pass_authority": (
                "optimization_or_seed_effect_no_unique_expert_rank_attribution"
            ),
            "expert16_pass_expert64_fail_authority": (
                "no_expert_rank_bottleneck_support_and_no_replication"
            ),
            "invalid_authority": "no_capacity_or_optimization_conclusion",
            "two_seed_expert_rank_support_next_rung": (
                "separately_preregister_descending_expert_rank_ladder_only"
            ),
            "fresh_c3_authority": (
                "not_authorized_by_this_control_only_after_successful_"
                "descending_expert_rank_ladder"
            ),
            "result_authentication": (
                "mandatory_external_logical_tensor_report_sha256_triple"
            ),
            "run_generated_receipt_scope": (
                "immediate_publication_integrity_check_only_not_an_external_"
                "scientific_trust_root"
            ),
            "durable_scientific_authority": (
                "begins_only_after_the_exact_receipt_triple_is_recorded_"
                "outside_the_artifact"
            ),
            "self_hashes_authoritative_without_external_receipt": False,
            "allowed_materialization_roles": ["pilot", "fit"],
            "c2_selection_allowed": False,
            "c2_provider_artifact_loading_allowed": False,
            "c3_allowed": False,
            "compression_claim_allowed": False,
            "held_out_generalization_claim_allowed": False,
            "natural_prompt_fidelity_claim_allowed": False,
            "full_model_replacement_claim_allowed": False,
            "wall_clock_speed_claim_allowed": False,
            "expert_count_change_mixed_into_primary": False,
            "outer_width_change_mixed_into_primary": False,
            "longer_optimization_mixed_into_primary": False,
            "conditional_residual_mixed_into_primary": False,
            "source_final_parameters_used_for_initialization": False,
            "publication_root": "ignored_local_runs_only",
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "FunctionPreservingExpertRankControlProtocol":
        state = _mapping(raw, label="function-preserving expert-rank protocol")
        expected = set(_default_unchecked().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="function-preserving expert-rank protocol",
        )
        defaults = _default_unchecked()._payload()
        semantic_names = set(defaults) - {
            "execution_device",
            "execution_dtype",
            "expert16_executor",
            "expert64_executor",
            "training",
            "lift",
            "preflight",
            "source",
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError(
                    "function-preserving expert-rank semantics drifted"
                )
        return cls(
            execution_device=str(state["execution_device"]),
            execution_dtype=str(state["execution_dtype"]),
            expert16_executor=ExpertRankExecutorSpec.from_state_dict(
                state["expert16_executor"]
            ),
            expert64_executor=ExpertRankExecutorSpec.from_state_dict(
                state["expert64_executor"]
            ),
            training=PairedExpertRankTrainingSpec.from_state_dict(
                state["training"]
            ),
            lift=NestedExpertRankLiftSpec.from_state_dict(state["lift"]),
            preflight=ExpertRankPreflightBindings.from_state_dict(
                state["preflight"]
            ),
            source=SourceWidthResultBindings.from_state_dict(
                state["source"]
            ),
            artifact_sha256=str(state["artifact_sha256"]),
        )


def _default_unchecked() -> FunctionPreservingExpertRankControlProtocol:
    return FunctionPreservingExpertRankControlProtocol(
        execution_device=EXPERT_RANK_CONTROL_EXECUTION_DEVICE,
        execution_dtype=EXPERT_RANK_CONTROL_EXECUTION_DTYPE,
        expert16_executor=ExpertRankExecutorSpec(expert_rank=16),
        expert64_executor=ExpertRankExecutorSpec(expert_rank=64),
        training=PairedExpertRankTrainingSpec(),
        lift=NestedExpertRankLiftSpec(),
        preflight=ExpertRankPreflightBindings(),
        source=SourceWidthResultBindings(),
    )


def default_function_preserving_expert_rank_control_protocol(
) -> FunctionPreservingExpertRankControlProtocol:
    """Return the declaration and verify its literal trust anchor."""

    protocol = _default_unchecked()
    if (
        protocol.protocol_sha256
        != DEFAULT_FUNCTION_PRESERVING_EXPERT_RANK_CONTROL_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            "function-preserving expert-rank protocol trust anchor drifted"
        )
    return protocol


# Literal trust anchor for the declaration; never computed during import.
DEFAULT_FUNCTION_PRESERVING_EXPERT_RANK_CONTROL_PROTOCOL_SHA256 = (
    "94b24068fa583c627faa7d06838c6cd80065f6180c3047ee2923ed95b587014c"
)
