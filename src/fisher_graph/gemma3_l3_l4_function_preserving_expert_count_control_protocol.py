"""Pure declaration for the paired Gemma L3/L4 expert-count control.

The experiment keeps outer modal width 64, expert rank 64, and router hidden
width 16 fixed while comparing two routed experts with four.  Each parent is
lifted into an active child and a dormant child: their logits and biases are
duplicated, the active child receives ``U`` and ``2V``, and the dormant child
receives ``U`` and exactly-zero ``V``.  The child pair therefore has the same
observable function and provider-chart JVP as its parent at step zero while
leaving the dormant path gradient-open.

The authenticated expert-rank result is validation-only.  Its final provider
is never loaded to initialize either arm.  This module intentionally imports
only the Python standard library and does not load PyTorch, a model, or an
experiment artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re


__all__ = [
    "DEFAULT_FUNCTION_PRESERVING_EXPERT_COUNT_CONTROL_PROTOCOL_SHA256",
    "EXPERT_COUNT_CONTROL_EXECUTION_DEVICE",
    "EXPERT_COUNT_CONTROL_EXECUTION_DTYPE",
    "EXPERT_COUNT_CONTROL_OUTER_RANK",
    "EXPERT_COUNT_CONTROL_PRIMARY_SEED",
    "EXPERT_COUNT_CONTROL_REPLICATION_SEED",
    "EXPERT_COUNT_CONTROL_EXPERT_RANK",
    "EXPERT_COUNT_CONTROL_SOURCE_EXPERTS",
    "EXPERT_COUNT_CONTROL_TARGET_EXPERTS",
    "ExpertCountExecutorSpec",
    "ExpertCountPreflightBindings",
    "FunctionPreservingExpertCountControlProtocol",
    "DormantChildExpertCountLiftSpec",
    "PairedExpertCountTrainingSpec",
    "SourceExpertRankResultBindings",
    "default_function_preserving_expert_count_control_protocol",
    "fit_replay_sequence_sha256",
]


EXPERT_COUNT_CONTROL_OUTER_RANK = 64
EXPERT_COUNT_CONTROL_EXPERT_RANK = 64
EXPERT_COUNT_CONTROL_SOURCE_EXPERTS = 2
EXPERT_COUNT_CONTROL_TARGET_EXPERTS = 4
EXPERT_COUNT_CONTROL_EXECUTION_DEVICE = "cpu"
EXPERT_COUNT_CONTROL_EXECUTION_DTYPE = "float32"
EXPERT_COUNT_CONTROL_PRIMARY_SEED = 20_260_728_402
EXPERT_COUNT_CONTROL_REPLICATION_SEED = 20_260_729_402

_FORMAT_VERSION = 1
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_function_preserving_expert_count_control."
    "protocol.v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXECUTOR_DOMAIN = (
    b"fisher-graph:function-preserving-expert-count:executor:v1\0"
)
_TRAINING_DOMAIN = (
    b"fisher-graph:function-preserving-expert-count:training:v1\0"
)
_LIFT_DOMAIN = b"fisher-graph:function-preserving-expert-count:lift:v1\0"
_PREFLIGHT_DOMAIN = (
    b"fisher-graph:function-preserving-expert-count:preflight:v1\0"
)
_SOURCE_DOMAIN = (
    b"fisher-graph:function-preserving-expert-count:sources:v1\0"
)
_PROTOCOL_DOMAIN = (
    b"fisher-graph:function-preserving-expert-count:protocol:v1\0"
)
_FIT_REPLAY_SEQUENCE_DOMAIN = (
    b"fisher-graph:function-preserving-expert-count:"
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
class ExpertCountExecutorSpec:
    """One exact fixed-outer-width gated causal executor instance."""

    input_modes: int = 66
    output_modes: int = 64
    expert_count: int = 2
    expert_rank: int = 64
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
            "expert_rank": 64,
            "router_width": 16,
            "same_position_skip": False,
            "max_positive_lag": None,
            "router_activation": "tanh",
            "source_normalized_routing": True,
        }
        for name, frozen in expected.items():
            value = getattr(self, name)
            if value != frozen or type(value) is not type(frozen):
                raise ValueError(f"expert-count executor {name} is not frozen")
        if type(self.expert_count) is not int or self.expert_count not in {
            EXPERT_COUNT_CONTROL_SOURCE_EXPERTS,
            EXPERT_COUNT_CONTROL_TARGET_EXPERTS,
        }:
            raise ValueError("executor expert count must be 2 or 4")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_EXECUTOR_DOMAIN,
                label="expert-count executor",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.expert_count_control_executor",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "outer_rank": EXPERT_COUNT_CONTROL_OUTER_RANK,
            "causality_semantics": (
                "unlimited_strict_positive_logical_lag_no_same_position_skip"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "ExpertCountExecutorSpec":
        state = _mapping(raw, label="expert-count executor")
        expected = set(cls().state_dict())
        _strict_keys(state, expected=expected, label="expert-count executor")
        defaults = cls()._payload()
        for name in (
            "artifact_kind",
            "format_version",
            "outer_rank",
            "causality_semantics",
        ):
            if state[name] != defaults[name]:
                raise ValueError("expert-count executor semantics drifted")
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
class PairedExpertCountTrainingSpec:
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
    primary_seed: int = EXPERT_COUNT_CONTROL_PRIMARY_SEED
    replication_seed: int = EXPERT_COUNT_CONTROL_REPLICATION_SEED
    fit_data_binding_sha256: str = (
        "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
    )
    fit_batch_sha256s_sequence_sha256: str = (
        "6bd6d6fc172fe6bf11b8ea3095646d6d73c559e5e213fd7d3c8767b125371a61"
    )
    fit_batch_content_sha256s_sequence_sha256: str = (
        "c9f42c9d035809c54be93d342a52c34c1b25836c8c42b4cc083b15d4484f6901"
    )
    fit_indexed_batch_sha256s_sequence_sha256: str = (
        "c4583b2fbff6f82026bd47e14373d5e2442ca56f70ff17356eed5097e84b8400"
    )
    fit_pair_sha256s_sequence_sha256: str = (
        "55b132bd2b47db3f4b30343f22c6d941d7142dd44f7b23c4bac3f54cd1eddd2c"
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
            "primary_seed": EXPERT_COUNT_CONTROL_PRIMARY_SEED,
            "replication_seed": EXPERT_COUNT_CONTROL_REPLICATION_SEED,
            "fit_data_binding_sha256": (
                "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
            ),
            "fit_batch_sha256s_sequence_sha256": (
                "6bd6d6fc172fe6bf11b8ea3095646d6d73c559e5e213fd7d"
                "3c8767b125371a61"
            ),
            "fit_batch_content_sha256s_sequence_sha256": (
                "c9f42c9d035809c54be93d342a52c34c1b25836c8c42b4cc"
                "083b15d4484f6901"
            ),
            "fit_indexed_batch_sha256s_sequence_sha256": (
                "c4583b2fbff6f82026bd47e14373d5e2442ca56f70ff17356"
                "eed5097e84b8400"
            ),
            "fit_pair_sha256s_sequence_sha256": (
                "55b132bd2b47db3f4b30343f22c6d941d7142dd44f7b23c4"
                "bac3f54cd1eddd2c"
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
                raise ValueError(f"paired expert-count {name} is not frozen")
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
                label="paired expert-count training",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.paired_expert_count_training",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "optimizer": "fresh_adam_per_arm",
            "batch_semantics": "exact_authenticated_d3_order",
            "initialization_schedule": (
                "regenerate_d3_outer64_expert_rank64_split_cold_start_per_seed"
            ),
            "source_final_provider_initialization_allowed": False,
            "primary_schedule": (
                "e2_r64_exact_expert_rank_primary_replay_then_e4_dormant_"
                "child_lift"
            ),
            "replication_schedule": (
                "conditional_fresh_cold_paired_seed_only_after_valid_primary_"
                "e2_fail_e4_pass"
            ),
            "fit_materialization_roles": ["pilot", "fit"],
            "canonical_scoring_semantics": (
                "raw_canonical_fisher_metric_for_all_fit_scoring"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "PairedExpertCountTrainingSpec":
        state = _mapping(raw, label="paired expert-count training")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="paired expert-count training",
        )
        defaults = cls()._payload()
        semantic_names = set(defaults) - {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError("paired expert-count semantics drifted")
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
class DormantChildExpertCountLiftSpec:
    """Exact function-preserving E2-to-E4 child split."""

    source_expert_count: int = EXPERT_COUNT_CONTROL_SOURCE_EXPERTS
    target_expert_count: int = EXPERT_COUNT_CONTROL_TARGET_EXPERTS
    parent_to_child_rule: str = (
        "parent_e_maps_to_active_child_e_and_dormant_child_e_plus_2"
    )
    child_index_groups: str = "active_children_0_1_dormant_children_2_3"
    router_logit_rule: str = (
        "copy_parent_router_column_and_bias_to_child_e_and_child_e_plus_2"
    )
    router_bias_log2_adjustment: float = 0.0
    active_child_u_scale: float = 1.0
    active_child_v_scale: float = 2.0
    dormant_child_u_scale: float = 1.0
    dormant_child_v_initialization: str = "exact_zero"
    base_rank: int = 16
    extra_rank: int = 48
    concatenated_rank: int = 64
    equivalence_absolute_tolerance: float = 1e-12
    equivalence_relative_tolerance: float = 1e-12
    gradient_norm_floor: float = 1e-12
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
                raise ValueError(
                    f"dormant-child expert-count lift {name} is not frozen"
                )
        if self.target_expert_count != 2 * self.source_expert_count:
            raise ValueError("each source expert must produce two children")
        if self.base_rank + self.extra_rank != self.concatenated_rank:
            raise ValueError("split expert-rank arithmetic drifted")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_LIFT_DOMAIN,
                label="dormant-child expert-count lift",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.dormant_child_expert_count_lift",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "softmax_identity": (
                "duplicating_every_parent_logit_halves_each_child_probability_"
                "without_any_log2_bias_adjustment"
            ),
            "observable_identity": (
                "p_over_2_times_u_times_2v_plus_p_over_2_times_u_times_"
                "zero_equals_p_times_u_times_v"
            ),
            "observable_initialization": "exact_e2_r64_function",
            "jvp_initialization": "exact_e2_r64_provider_chart_jvp",
            "split_rank_preservation": (
                "apply_active_2v_and_dormant_zero_v_separately_to_base16_"
                "and_extra48_then_concatenate_for_scoring"
            ),
            "gradient_open_rule": (
                "step1_dormant_v_and_child_router_split_gradient_and_delta_"
                "above_floor_with_dormant_u_exact_zero_then_step2_dormant_u_"
                "gradient_and_delta_above_floor"
            ),
            "split_wrapper_parity_rule": (
                "split_and_concatenated_outputs_jvps_and_weighted_loss_match_"
                "within_equivalence_tolerances_initially_and_postfit"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "DormantChildExpertCountLiftSpec":
        state = _mapping(raw, label="dormant-child expert-count lift")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="dormant-child expert-count lift",
        )
        defaults = cls()._payload()
        semantic_names = set(defaults) - {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError("dormant-child expert-count semantics drifted")
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
class ExpertCountPreflightBindings:
    """Frozen two-step implementation audit for both seed roles.

    Every value was measured in a fit-only run before any 600-step outcome fit,
    ordinary scoring, contrast scoring, or publication.  Both frozen seed roles
    must replay this audit exactly before the primary outcome run may start.
    """

    bindings_finalized: bool = True
    preflight_steps: int = 2
    expected_jvp_pair_count: int = 32
    extra_u_initial_sha256: str = (
        "7155a40e51f6a293d27125d3a708799027731dd7c4fc8cf2341ea2e40fa34975"
    )
    extra_v_initial_sha256: str = (
        "1b749f3533752550e7a4e52023731adb254d9cb2f9deafdb3c83e7d715ed8b76"
    )
    primary_initialization_audit_sha256: str = (
        "305af575c3fa3579e1573971b949b412e0f8f4626ec6477c794ef146641bc21b"
    )
    primary_control_initial_executor_sha256: str = (
        "eb0493e6f7c20cae8c6a8caa3bc80d30709d04e9a7c4da6380f4c97f973a6e63"
    )
    primary_treatment_initial_executor_sha256: str = (
        "3cb6d3c2fa1a2804cd187922ca0cf3ca571d8d1900d87ba85bfa6d503efdaf70"
    )
    primary_treatment_base_executor_sha256: str = (
        "47fad109d604bc4bbfc21f2a60357f6ff397f2ce7e83c1d063d2c0560aff9c51"
    )
    primary_initial_observable_absolute_error: float = (
        1.7763568394002505e-15
    )
    primary_initial_observable_relative_error: float = (
        1.4378553700958835e-17
    )
    primary_initial_jvp_absolute_error: float = 1.7763568394002505e-15
    primary_initial_jvp_relative_error: float = 5.4144611405591306e-17
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
        3.834675111389868e-17
    )
    primary_initial_weighted_total_absolute_error: float = 0.0
    primary_parent_route_mass_absolute_error: float = (
        1.3877787807814457e-16
    )
    primary_parent_route_mass_relative_error: float = (
        1.4510231935028425e-17
    )
    primary_sibling_route_absolute_error: float = 0.0
    primary_source_probability_absolute_error: float = 0.0
    primary_allowed_edge_route_sum_absolute_error: float = (
        2.220446049250313e-16
    )
    primary_causal_masks_exact: bool = True
    primary_outside_edge_routes_zero: bool = True
    primary_gradient_audit_sha256: str = (
        "9b9ad88620d86f1062ab28dec15a213e60dda73fefdba0a93f48a12868008c45"
    )
    primary_step1_active_base_input_gradient_norm: float = 0.7376307361831621
    primary_step1_active_base_input_delta_norm: float = 0.04595587444373063
    primary_step1_active_base_output_gradient_norm: float = (
        0.16774201674461414
    )
    primary_step1_active_base_output_delta_norm: float = (
        0.022625791695366368
    )
    primary_step1_dormant_base_input_gradient_norm: float = 0.0
    primary_step1_dormant_base_input_delta_norm: float = 0.0
    primary_step1_dormant_base_output_gradient_norm: float = (
        0.16774201674461414
    )
    primary_step1_dormant_base_output_delta_norm: float = (
        0.022625791695366358
    )
    primary_step1_active_extra_input_gradient_norm: float = 0.0
    primary_step1_active_extra_input_delta_norm: float = 0.0
    primary_step1_active_extra_output_gradient_norm: float = (
        0.9816547293074461
    )
    primary_step1_active_extra_output_delta_norm: float = (
        0.039183069909569664
    )
    primary_step1_dormant_extra_input_gradient_norm: float = 0.0
    primary_step1_dormant_extra_input_delta_norm: float = 0.0
    primary_step1_dormant_extra_output_gradient_norm: float = (
        0.9816547293074461
    )
    primary_step1_dormant_extra_output_delta_norm: float = (
        0.039183069909569664
    )
    primary_step1_router_sibling_gradient_norm: float = 0.06637324229861168
    primary_step1_router_sibling_delta_norm: float = 0.009797917171561364
    primary_step2_dormant_base_input_gradient_norm: float = (
        0.004194408581233414
    )
    primary_step2_dormant_base_input_delta_norm: float = 0.03415928270018813
    primary_step2_active_extra_input_gradient_norm: float = (
        0.007472341178611839
    )
    primary_step2_active_extra_input_delta_norm: float = 0.05918069995530761
    primary_step2_dormant_extra_input_gradient_norm: float = (
        0.007502102819059041
    )
    primary_step2_dormant_extra_input_delta_norm: float = 0.05917074421390244
    primary_postfit_parity_sha256: str = (
        "490c1c7aee61c9340aeef85ef9bc61f1a4b40b765e4b7d263a570da108883c1f"
    )
    primary_postfit_output_absolute_error: float = (
        3.552713678800501e-15
    )
    primary_postfit_output_relative_error: float = (
        1.0731193706505196e-16
    )
    primary_postfit_jvp_absolute_error: float = 7.105427357601002e-15
    primary_postfit_jvp_relative_error: float = (
        1.034979480035721e-16
    )
    primary_postfit_weighted_total_absolute_error: float = 0.0
    primary_control_postfit_executor_sha256: str = (
        "49e86f324cd5368f15d6d26e7fae851cf6b1332188fced706d28e90f8e94c362"
    )
    primary_control_postfit_metrics_sha256: str = (
        "1ec5504323fa14a250af8de56c9eb5c575a90cb9ad008e8819ad6b7f24315dbb"
    )
    primary_postfit_concatenated_executor_sha256: str = (
        "55c1d61c7b61361a318253c34c500114f8af4c1876e301fa9688476603823b09"
    )
    primary_postfit_metrics_sha256: str = (
        "c99624d8f30ea7be3f0bc31db8294a999f3b43c10bc315c980fc5ec4b0aa6ab6"
    )
    replication_initialization_audit_sha256: str = (
        "698819152030ff0b33c3a6dab146c9851178691e828edb5976d89d865c374275"
    )
    replication_control_initial_executor_sha256: str = (
        "24cdb9f1cb1a7b55fe52d14953a4e4a884228c76fed19633ffd8d7673cd3caae"
    )
    replication_treatment_initial_executor_sha256: str = (
        "33944f86a76fec23778cf3a938d878677d47cc01a85575bc040ec2723bfd06ae"
    )
    replication_treatment_base_executor_sha256: str = (
        "98260efc9c4b72c0fa29af952739345a9e0686e2d648883dd29814330a0c192a"
    )
    replication_initial_observable_absolute_error: float = (
        8.881784197001252e-16
    )
    replication_initial_observable_relative_error: float = (
        8.444488456880822e-18
    )
    replication_initial_jvp_absolute_error: float = (
        3.552713678800501e-15
    )
    replication_initial_jvp_relative_error: float = (
        4.083480588235153e-17
    )
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
        3.676619482318369e-17
    )
    replication_initial_weighted_total_absolute_error: float = 0.0
    replication_parent_route_mass_absolute_error: float = (
        1.1102230246251565e-16
    )
    replication_parent_route_mass_relative_error: float = (
        7.847065376894128e-18
    )
    replication_sibling_route_absolute_error: float = 0.0
    replication_source_probability_absolute_error: float = 0.0
    replication_allowed_edge_route_sum_absolute_error: float = (
        2.220446049250313e-16
    )
    replication_causal_masks_exact: bool = True
    replication_outside_edge_routes_zero: bool = True
    replication_gradient_audit_sha256: str = (
        "9b1816fc91a33b2e182e49a7df8caf459dcb7c2d560607946944780233360547"
    )
    replication_step1_active_base_input_gradient_norm: float = (
        0.8721020448088423
    )
    replication_step1_active_base_input_delta_norm: float = (
        0.04595617126738638
    )
    replication_step1_active_base_output_gradient_norm: float = (
        0.2253240731091315
    )
    replication_step1_active_base_output_delta_norm: float = (
        0.022623150402947344
    )
    replication_step1_dormant_base_input_gradient_norm: float = 0.0
    replication_step1_dormant_base_input_delta_norm: float = 0.0
    replication_step1_dormant_base_output_gradient_norm: float = (
        0.2253240731091315
    )
    replication_step1_dormant_base_output_delta_norm: float = (
        0.022623150402947344
    )
    replication_step1_active_extra_input_gradient_norm: float = 0.0
    replication_step1_active_extra_input_delta_norm: float = 0.0
    replication_step1_active_extra_output_gradient_norm: float = (
        1.260204066630651
    )
    replication_step1_active_extra_output_delta_norm: float = (
        0.039189940389855474
    )
    replication_step1_dormant_extra_input_gradient_norm: float = 0.0
    replication_step1_dormant_extra_input_delta_norm: float = 0.0
    replication_step1_dormant_extra_output_gradient_norm: float = (
        1.260204066630651
    )
    replication_step1_dormant_extra_output_delta_norm: float = (
        0.039189940389855474
    )
    replication_step1_router_sibling_gradient_norm: float = (
        0.10800865788827035
    )
    replication_step1_router_sibling_delta_norm: float = (
        0.01113865731122708
    )
    replication_step2_dormant_base_input_gradient_norm: float = (
        0.005111500576015453
    )
    replication_step2_dormant_base_input_delta_norm: float = (
        0.034162828412323674
    )
    replication_step2_active_extra_input_gradient_norm: float = (
        0.009069318706432902
    )
    replication_step2_active_extra_input_delta_norm: float = (
        0.059194319932118586
    )
    replication_step2_dormant_extra_input_gradient_norm: float = (
        0.009109959489848364
    )
    replication_step2_dormant_extra_input_delta_norm: float = (
        0.059192167576655186
    )
    replication_postfit_parity_sha256: str = (
        "9134c050e07de59ff0a182801c4e6135ddb4976ce1700a0a4a5f92bd1bc97e6e"
    )
    replication_postfit_output_absolute_error: float = (
        1.7763568394002505e-15
    )
    replication_postfit_output_relative_error: float = (
        1.0859384987897777e-16
    )
    replication_postfit_jvp_absolute_error: float = (
        3.552713678800501e-15
    )
    replication_postfit_jvp_relative_error: float = (
        1.0310651872597173e-16
    )
    replication_postfit_weighted_total_absolute_error: float = (
        8.881784197001252e-16
    )
    replication_control_postfit_executor_sha256: str = (
        "aa6340de201ef4cd9c26da8646058c16c072ade852409f5d5de1dcad770495ca"
    )
    replication_control_postfit_metrics_sha256: str = (
        "73cce51e57a29bea19c984d5b8373993d5b9e74dec475170b257f79291a08971"
    )
    replication_postfit_concatenated_executor_sha256: str = (
        "5a506673387d71a7fea1b23589b6c0706aa40efa63e0ca59aa4fd9b6866b52a8"
    )
    replication_postfit_metrics_sha256: str = (
        "b9bb37bde7f126a54556ffc085158811e595b55d70add3a35fcb708b1b7006fc"
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
                raise ValueError(f"expert-count preflight {name} is not frozen")
            if name.endswith("_sha256"):
                _require_sha256(value, label=name)
        tolerance = 1e-12
        for role in ("primary", "replication"):
            zero_fragments = (
                "step1_dormant_base_input_",
                "step1_active_extra_input_",
                "step1_dormant_extra_input_",
            )
            gradient_names = (
                name
                for name in self.__dataclass_fields__
                if name.startswith(f"{role}_step")
            )
            for name in gradient_names:
                value = getattr(self, name)
                if any(fragment in name for fragment in zero_fragments):
                    if value != 0.0:
                        raise ValueError(
                            "step-one dormant/input negative control opened"
                        )
                elif value <= tolerance:
                    raise ValueError(
                        "bank-specific expert-count direction is not open"
                    )
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
            for name in (
                "causal_masks_exact",
                "outside_edge_routes_zero",
            ):
                if getattr(self, f"{role}_{name}") is not True:
                    raise ValueError("initial route semantics failed")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_PREFLIGHT_DOMAIN,
                label="expert-count preflight bindings",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.expert_count_preflight_bindings",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "execution_scope": (
                "fit_side_two_step_only_no_scoring_or_publication"
            ),
            "freeze_policy": (
                "exact_two_seed_fit_only_preflight_must_replay_before_any_"
                "outcome_fit_or_scoring"
            ),
            "initial_equivalence_semantics": (
                "e2_control_e4_treatment_and_split_concatenated_output_jvp_"
                "weighted_loss_and_parent_aggregated_child_probability"
            ),
            "postfit_parity_semantics": (
                "split_training_wrapper_equals_concatenated_executor_on_all_"
                "fit_batches_32_provider_chart_jvps_and_weighted_total"
            ),
            "top_level_live_preflight_artifact_sha256_frozen": False,
            "top_level_hash_exclusion_reason": (
                "live_preflight_contains_final_protocol_sha256_so_embedding_"
                "its_artifact_hash_in_the_protocol_would_be_circular"
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
            EXPERT_COUNT_CONTROL_PRIMARY_SEED
            if pair_role == "primary"
            else EXPERT_COUNT_CONTROL_REPLICATION_SEED
        )
        return {
            "bindings_finalized": self.bindings_finalized,
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
                "initial_weighted_total_absolute_error": getattr(
                    self,
                    f"{prefix}_initial_weighted_total_absolute_error",
                ),
                "maximum_parent_route_mass_absolute_error": getattr(
                    self,
                    f"{prefix}_parent_route_mass_absolute_error",
                ),
                "maximum_parent_route_mass_relative_error": getattr(
                    self,
                    f"{prefix}_parent_route_mass_relative_error",
                ),
                "maximum_sibling_route_absolute_error": getattr(
                    self,
                    f"{prefix}_sibling_route_absolute_error",
                ),
                "maximum_source_probability_absolute_error": getattr(
                    self,
                    f"{prefix}_source_probability_absolute_error",
                ),
                "causal_masks_exact": getattr(
                    self,
                    f"{prefix}_causal_masks_exact",
                ),
                "outside_edge_routes_zero": getattr(
                    self,
                    f"{prefix}_outside_edge_routes_zero",
                ),
                "maximum_allowed_edge_route_sum_absolute_error": getattr(
                    self,
                    f"{prefix}_allowed_edge_route_sum_absolute_error",
                ),
                "lift_parameter_flags": {
                    name: True
                    for name in (
                        "base_input_active_copy",
                        "base_input_dormant_copy",
                        "base_output_active_double",
                        "base_output_dormant_zero",
                        "extra_input_active_copy",
                        "extra_input_dormant_copy",
                        "extra_output_active_double",
                        "extra_output_dormant_zero",
                        "router_output_active_copy",
                        "router_output_dormant_copy",
                        "router_bias_active_copy",
                        "router_bias_dormant_copy",
                        "same_position_weight_copy",
                        "same_position_bias_copy",
                        "router_query_weight_copy",
                        "router_key_weight_copy",
                        "router_lag_weight_copy",
                        "source_score_weight_copy",
                    )
                },
            },
            "treatment_gradient_audit_sha256": getattr(
                self,
                f"{prefix}_gradient_audit_sha256",
            ),
            "treatment_gradient": {
                name.removeprefix(f"{prefix}_"): getattr(self, name)
                for name in self.__dataclass_fields__
                if name.startswith(f"{prefix}_step")
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
    def from_state_dict(cls, raw: object) -> "ExpertCountPreflightBindings":
        state = _mapping(raw, label="expert-count preflight bindings")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="expert-count preflight bindings",
        )
        defaults = cls()._payload()
        semantic_names = set(defaults) - {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError("expert-count preflight semantics drifted")
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
class SourceExpertRankResultBindings:
    """Authenticated expert-rank result and exact E2/R64 identity."""

    expert_rank_protocol_sha256: str = (
        "94b24068fa583c627faa7d06838c6cd80065f6180c3047ee2923ed95b587014c"
    )
    expert_rank_code_bundle_sha256: str = (
        "bcdd356aea62fedbdffd57bca39e1287f6da1374bb7477a2b28de374c9afebc3"
    )
    expert_rank_logical_artifact_sha256: str = (
        "9759407bf2f2c0a1deb1d29aba7fdbf453bdda8a727aa1e672452a00299a48a9"
    )
    expert_rank_tensor_file_sha256: str = (
        "2139696efebcee68dd379f8226e04cb5edce57c10571f7db5b697700854a2a61"
    )
    expert_rank_report_sha256: str = (
        "f2a8cb19f5aeebf9e5a1b46ac880655611db45138ab9943216ed9d2acea78c7c"
    )
    expert_rank_outcome: str = "primary_both_fail"
    expert_rank_primary_comparison_status: str = "both_fail"
    expert_rank_primary_treatment_valid: bool = True
    expert_rank_replication_executed: bool = False
    expert_rank_expert_count_control_authorized: bool = True
    expert_rank_compressed_rank_ladder_authorized: bool = False
    expert_rank_fresh_c3_authorized: bool = False
    expert_rank_two_seed_support: bool = False
    expert_rank_primary_e2r64_plan_sha256: str = (
        "57268369d21a66e464f7155a5f9b99868b1240e5f1ca0e3d59b2d40f5d5373de"
    )
    expert_rank_primary_e2r64_result_sha256: str = (
        "2d7492248d279df96b6c6f60049fbc1ed4bd2907f37cd160700ad8d1532a3395"
    )
    expert_rank_primary_e2r64_initial_metrics_sha256: str = (
        "56ee6759438f98b1ba9628c055358d29622af48899d49a51e23297ad9779aa20"
    )
    expert_rank_primary_e2r64_final_metrics_sha256: str = (
        "92e16d27f160144fc24cab76dbcfee58e5c88df143fbb1d7916dc8092a4ac882"
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
                raise ValueError(
                    f"source expert-rank result {name} is not frozen"
                )
            if name.endswith("_sha256"):
                _require_sha256(value, label=name)
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_SOURCE_DOMAIN,
                label="source expert-rank result bindings",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": (
                "fisher_graph.expert_count_source_expert_rank_result"
            ),
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "source_outer_rank": 64,
            "source_expert_count": 2,
            "source_expert_rank": 64,
            "source_router_width": 16,
            "source_final_parameter_use": "validation_only",
            "source_final_parameter_initialization_allowed": False,
            "source_primary_replay_requirement": (
                "e2_r64_primary_must_exactly_reproduce_bound_plan_result_"
                "initial_and_final_metrics"
            ),
            "external_receipt_semantics": (
                "durable_logical_tensor_report_sha256_triple"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "SourceExpertRankResultBindings":
        state = _mapping(raw, label="source expert-rank result bindings")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="source expert-rank result bindings",
        )
        defaults = cls()._payload()
        semantic_names = set(defaults) - {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError("source expert-rank result semantics drifted")
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
class FunctionPreservingExpertCountControlProtocol:
    """Complete immutable declaration for the expert-count control."""

    execution_device: str
    execution_dtype: str
    e2_executor: ExpertCountExecutorSpec
    e4_executor: ExpertCountExecutorSpec
    training: PairedExpertCountTrainingSpec
    lift: DormantChildExpertCountLiftSpec
    preflight: ExpertCountPreflightBindings
    source: SourceExpertRankResultBindings
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.execution_device != EXPERT_COUNT_CONTROL_EXECUTION_DEVICE:
            raise ValueError("execution device must remain cpu")
        if self.execution_dtype != EXPERT_COUNT_CONTROL_EXECUTION_DTYPE:
            raise ValueError("execution dtype must remain float32")
        if self.e2_executor != ExpertCountExecutorSpec(expert_count=2):
            raise ValueError("E2/R64 executor drifted")
        if self.e4_executor != ExpertCountExecutorSpec(expert_count=4):
            raise ValueError("E4/R64 executor drifted")
        if self.training != PairedExpertCountTrainingSpec():
            raise ValueError("paired expert-count training drifted")
        if self.lift != DormantChildExpertCountLiftSpec():
            raise ValueError("dormant-child expert-count lift drifted")
        if self.preflight != ExpertCountPreflightBindings():
            raise ValueError("expert-count preflight bindings drifted")
        if self.source != SourceExpertRankResultBindings():
            raise ValueError("source expert-rank result bindings drifted")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_PROTOCOL_DOMAIN,
                label="function-preserving expert-count protocol",
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
                "fit_only_paired_function_preserving_expert_count_control"
            ),
            "execution_device": self.execution_device,
            "execution_dtype": self.execution_dtype,
            "outer_rank": EXPERT_COUNT_CONTROL_OUTER_RANK,
            "expert_rank": EXPERT_COUNT_CONTROL_EXPERT_RANK,
            "router_width": 16,
            "e2_executor": self.e2_executor.state_dict(),
            "e4_executor": self.e4_executor.state_dict(),
            "training": self.training.state_dict(),
            "lift": self.lift.state_dict(),
            "preflight": self.preflight.state_dict(),
            "source": self.source.state_dict(),
            "controlled_change": (
                "gradient_open_dormant_child_expert_count_2_to_4_at_fixed_"
                "outer64_expert_rank64_router_width16_and_exact_initial_"
                "observable_and_jvp"
            ),
            "unchanged_contract": (
                "outer_rank_expert_rank_encoder_decoder_same_position_router_"
                "hidden_features_activation_source_normalized_routing_"
                "causality_d3_objective_fit_data_order_optimizer_gauge_gates_"
                "seeds_device_dtype"
            ),
            "primary_arms": [
                "e2_r64_exact_expert_rank_primary_replay",
                "e4_r64_dormant_child_function_preserving_lift",
            ],
            "primary_decision": (
                "e2_fail_e4_pass_authorizes_paired_replication_only"
            ),
            "replication_decision": (
                "e2_fail_e4_pass_supports_additional_routed_expert_partitions"
            ),
            "both_fail_authority": (
                "e4_insufficient_and_e8_full_count_oracle_only"
            ),
            "valid_primary_both_fail_next_rung": (
                "separately_preregister_e8_full_count_oracle_only"
            ),
            "both_pass_authority": (
                "optimization_or_seed_effect_no_unique_expert_count_attribution"
            ),
            "e2_pass_e4_fail_authority": (
                "adverse_lift_or_optimization_effect_no_count_support"
            ),
            "invalid_authority": "no_capacity_or_optimization_conclusion",
            "two_seed_expert_count_support_next_rung": (
                "separately_preregister_e3_threshold_or_descending_count_"
                "ladder_only"
            ),
            "fresh_c3_authority": (
                "not_authorized_by_this_control_only_after_successful_"
                "descending_expert_count_ladder"
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
            "outcome_run_allowed": self.preflight.bindings_finalized,
            "c2_selection_allowed": False,
            "c2_provider_artifact_loading_allowed": False,
            "c3_allowed": False,
            "compression_claim_allowed": False,
            "held_out_generalization_claim_allowed": False,
            "natural_prompt_fidelity_claim_allowed": False,
            "full_model_replacement_claim_allowed": False,
            "wall_clock_speed_claim_allowed": False,
            "expert_count_is_only_controlled_change": True,
            "router_output_cardinality_change_unavoidable": True,
            "router_cardinality_confound_scope": (
                "success_supports_additional_routed_expert_partitions_not_"
                "expert_map_count_independent_of_routing_cardinality"
            ),
            "outer_width_change_mixed_into_primary": False,
            "expert_rank_change_mixed_into_primary": False,
            "router_hidden_width_change_mixed_into_primary": False,
            "longer_optimization_mixed_into_primary": False,
            "conditional_residual_mixed_into_primary": False,
            "source_final_parameters_used_for_initialization": False,
            "capacity_oracle_not_compression_or_speed_evidence": True,
            "exact_accounting": {
                "e2_executor_parameters": 23106,
                "e4_executor_parameters": 39780,
                "e2_total_stored_scalars": 31492,
                "e4_total_stored_scalars": 48166,
                "e2_canonical_core_macs_length128": 4499008,
                "e4_canonical_core_macs_length128": 7929024,
                "e2_canonical_total_macs_length128": 5555776,
                "e4_canonical_total_macs_length128": 8985792,
                "e2_fit_panel_core_macs": 506306160,
                "e4_fit_panel_core_macs": 897039840,
                "e2_fit_panel_total_macs": 610001520,
                "e4_fit_panel_total_macs": 1000735200,
            },
            "publication_root": "ignored_local_runs_only",
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "FunctionPreservingExpertCountControlProtocol":
        state = _mapping(raw, label="function-preserving expert-count protocol")
        expected = set(_default_unchecked().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="function-preserving expert-count protocol",
        )
        defaults = _default_unchecked()._payload()
        semantic_names = set(defaults) - {
            "execution_device",
            "execution_dtype",
            "e2_executor",
            "e4_executor",
            "training",
            "lift",
            "preflight",
            "source",
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError(
                    "function-preserving expert-count semantics drifted"
                )
        return cls(
            execution_device=str(state["execution_device"]),
            execution_dtype=str(state["execution_dtype"]),
            e2_executor=ExpertCountExecutorSpec.from_state_dict(
                state["e2_executor"]
            ),
            e4_executor=ExpertCountExecutorSpec.from_state_dict(
                state["e4_executor"]
            ),
            training=PairedExpertCountTrainingSpec.from_state_dict(
                state["training"]
            ),
            lift=DormantChildExpertCountLiftSpec.from_state_dict(
                state["lift"]
            ),
            preflight=ExpertCountPreflightBindings.from_state_dict(
                state["preflight"]
            ),
            source=SourceExpertRankResultBindings.from_state_dict(
                state["source"]
            ),
            artifact_sha256=str(state["artifact_sha256"]),
        )


def _default_unchecked() -> FunctionPreservingExpertCountControlProtocol:
    return FunctionPreservingExpertCountControlProtocol(
        execution_device=EXPERT_COUNT_CONTROL_EXECUTION_DEVICE,
        execution_dtype=EXPERT_COUNT_CONTROL_EXECUTION_DTYPE,
        e2_executor=ExpertCountExecutorSpec(expert_count=2),
        e4_executor=ExpertCountExecutorSpec(expert_count=4),
        training=PairedExpertCountTrainingSpec(),
        lift=DormantChildExpertCountLiftSpec(),
        preflight=ExpertCountPreflightBindings(),
        source=SourceExpertRankResultBindings(),
    )


def default_function_preserving_expert_count_control_protocol(
) -> FunctionPreservingExpertCountControlProtocol:
    """Return the declaration and verify its literal trust anchor."""

    protocol = _default_unchecked()
    if (
        protocol.protocol_sha256
        != DEFAULT_FUNCTION_PRESERVING_EXPERT_COUNT_CONTROL_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            "function-preserving expert-count protocol trust anchor drifted"
        )
    return protocol


# Literal trust anchor for the declaration; never computed during import.
DEFAULT_FUNCTION_PRESERVING_EXPERT_COUNT_CONTROL_PROTOCOL_SHA256 = (
    "3d4cfbc2e69434e5cfb5845ad59ae3087457b175faa02afefa7edad5935acc27"
)
