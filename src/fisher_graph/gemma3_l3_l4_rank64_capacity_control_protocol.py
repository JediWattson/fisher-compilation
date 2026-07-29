"""Pure declaration for the matched Gemma L3/L4 rank-64 capacity control.

The control replays the fit-only D3 training problem from the authenticated
objective-balance diagnostic while changing the packed latent rank from 16 to
64.  All objective, fit-data, batch-order, optimizer, executor-family, gate,
seed, device, and dtype choices remain frozen.

This is a capacity diagnosis, not a compression or generalization experiment.
A primary fit pass authorizes only the frozen replication seed.  Passing both
fit seeds supports opening a separately preregistered compressed-width ladder;
it does not authorize C3.  C2 selection remains forbidden.

The declaration intentionally imports only the Python standard library and
does not load either the model or an experiment artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re


__all__ = [
    "CAPACITY_CONTROL_BASELINE_LATENT_RANK",
    "CAPACITY_CONTROL_EXECUTION_DEVICE",
    "CAPACITY_CONTROL_EXECUTION_DTYPE",
    "CAPACITY_CONTROL_LATENT_RANK",
    "CAPACITY_CONTROL_PRIMARY_SEED",
    "CAPACITY_CONTROL_REPLICATION_SEED",
    "DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256",
    "MatchedD3TrainingSpec",
    "ObjectiveBalanceResultBinding",
    "Rank64CapacityControlProtocol",
    "Rank64ExecutorSpec",
    "default_rank64_capacity_control_protocol",
    "source_replay_binding_sha256",
    "source_sequence_binding_sha256",
]


CAPACITY_CONTROL_BASELINE_LATENT_RANK = 16
CAPACITY_CONTROL_LATENT_RANK = 64
CAPACITY_CONTROL_EXECUTION_DEVICE = "cpu"
CAPACITY_CONTROL_EXECUTION_DTYPE = "float32"
CAPACITY_CONTROL_PRIMARY_SEED = 20_260_728_402
CAPACITY_CONTROL_REPLICATION_SEED = 20_260_729_402

_FORMAT_VERSION = 1
_SCHEMA = "fisher_graph.gemma3_l3_l4_rank64_capacity_control.protocol.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXECUTOR_DOMAIN = b"fisher-graph:rank64-capacity-control:executor:v1\0"
_TRAINING_DOMAIN = b"fisher-graph:rank64-capacity-control:training:v1\0"
_RESULT_BINDING_DOMAIN = (
    b"fisher-graph:rank64-capacity-control:objective-result:v1\0"
)
_SOURCE_SEQUENCE_DOMAIN = (
    b"fisher-graph:rank64-capacity-control:source-sequence:v1\0"
)
_SOURCE_REPLAY_DOMAIN = (
    b"fisher-graph:rank64-capacity-control:source-replay:v1\0"
)
_PROTOCOL_DOMAIN = b"fisher-graph:rank64-capacity-control:protocol:v1\0"


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


def source_sequence_binding_sha256(
    name: str,
    sha256s: object,
) -> str:
    """Bind one exact ordered source-D3 identity sequence."""

    if not isinstance(name, str) or not name:
        raise ValueError("source sequence name must be nonempty")
    if not isinstance(sha256s, (tuple, list)) or not sha256s:
        raise ValueError("source sequence hashes must be nonempty")
    values = tuple(
        _require_sha256(value, label=f"source sequence {name}")
        for value in sha256s
    )
    return _digest(
        {"name": name, "sha256s": values},
        domain=_SOURCE_SEQUENCE_DOMAIN,
    )


def source_replay_binding_sha256(
    *,
    sequence_sha256s: object,
    natural_pair_sha256s: object,
    balanced_pair_sha256s: object,
    shared_manifest: object,
) -> str:
    """Bind the complete ordered D3 replay problem without tensors."""

    if not isinstance(sequence_sha256s, Mapping) or not isinstance(
        shared_manifest,
        Mapping,
    ):
        raise TypeError("source replay tables must be mappings")
    sequences = {
        str(name): tuple(
            _require_sha256(value, label=f"source replay {name}")
            for value in values
        )
        for name, values in sequence_sha256s.items()
        if isinstance(values, (tuple, list)) and values
    }
    if set(sequences) != set(sequence_sha256s):
        raise ValueError("source replay sequence table is invalid")
    natural = tuple(
        _require_sha256(value, label="source replay natural pair")
        for value in natural_pair_sha256s  # type: ignore[union-attr]
    )
    balanced = tuple(
        _require_sha256(value, label="source replay balanced pair")
        for value in balanced_pair_sha256s  # type: ignore[union-attr]
    )
    if not natural or not balanced:
        raise ValueError("source replay pair sequences must be nonempty")
    return _digest(
        {
            "sequence_sha256s": sequences,
            "natural_pair_sha256s": natural,
            "balanced_pair_sha256s": balanced,
            "shared_manifest": dict(shared_manifest),
        },
        domain=_SOURCE_REPLAY_DOMAIN,
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
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


def _bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")
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
class Rank64ExecutorSpec:
    """Exact full-rank instance of the existing gated executor family."""

    input_modes: int = 66
    output_modes: int = 64
    expert_count: int = 2
    expert_rank: int = 16
    router_width: int = 16
    same_position_skip: bool = False
    max_positive_lag: None = None
    router_activation: str = "tanh"
    source_normalized_routing: bool = True
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        expected = {
            "input_modes": 66,
            "output_modes": 64,
            "expert_count": 2,
            "expert_rank": 16,
            "router_width": 16,
            "same_position_skip": False,
            "max_positive_lag": None,
            "router_activation": "tanh",
            "source_normalized_routing": True,
        }
        for name, frozen in expected.items():
            value = getattr(self, name)
            if value != frozen or type(value) is not type(frozen):
                raise ValueError(f"rank-64 executor {name} is not frozen")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_EXECUTOR_DOMAIN,
                label="rank-64 executor artifact",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.rank64_capacity_executor",
            "format_version": _FORMAT_VERSION,
            "input_modes": self.input_modes,
            "output_modes": self.output_modes,
            "expert_count": self.expert_count,
            "expert_rank": self.expert_rank,
            "router_width": self.router_width,
            "same_position_skip": self.same_position_skip,
            "max_positive_lag": self.max_positive_lag,
            "router_activation": self.router_activation,
            "source_normalized_routing": self.source_normalized_routing,
            "causality_semantics": (
                "unlimited_positive_lag_with_causal_mask_no_same_position_skip"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "Rank64ExecutorSpec":
        state = _mapping(raw, label="rank-64 executor")
        expected = set(cls().state_dict())
        _strict_keys(state, expected=expected, label="rank-64 executor")
        defaults = cls()._payload()
        for name in (
            "artifact_kind",
            "format_version",
            "causality_semantics",
        ):
            if state[name] != defaults[name]:
                raise ValueError("rank-64 executor semantics drifted")
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
            max_positive_lag=state["max_positive_lag"],  # type: ignore[arg-type]
            router_activation=str(state["router_activation"]),
            source_normalized_routing=_bool(
                state["source_normalized_routing"],
                label="source-normalized routing",
            ),
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class MatchedD3TrainingSpec:
    """The exact D3 fit objective, data order, optimizer, and fit gates."""

    recipe_id: str = "d3_unit_rms_family_balanced_direction"
    training_metric: str = "fit_teacher_weighted_rms"
    signed_pair_multiplicity: int = 2
    pointwise_weight: float = 1.0
    sensitivity_relative_delta_weight: float = 2.0
    sensitivity_direction_weight: float = 2.0
    midpoint_jvp_weight: float = 1.0
    intended_null_weight: float = 1.0
    sensitivity_relative_floor: float = 1e-6
    direction_norm_floor: float = 1e-8
    jvp_relative_floor: float = 1e-6
    steps: int = 600
    learning_rate: float = 1e-3
    fit_data_binding_sha256: str = (
        "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
    )
    ordinary_gates_sha256: str = (
        "0ef08515366888ff11f83269c68fc154477202b64bf2308e6b68985b05e91cd5"
    )
    contrast_gates_sha256: str = (
        "bedf561b190b04f880aabad6020ddb680d187734d572c9ec1abba7573cad0de1"
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "recipe_id": "d3_unit_rms_family_balanced_direction",
            "training_metric": "fit_teacher_weighted_rms",
            "signed_pair_multiplicity": 2,
            "pointwise_weight": 1.0,
            "sensitivity_relative_delta_weight": 2.0,
            "sensitivity_direction_weight": 2.0,
            "midpoint_jvp_weight": 1.0,
            "intended_null_weight": 1.0,
            "sensitivity_relative_floor": 1e-6,
            "direction_norm_floor": 1e-8,
            "jvp_relative_floor": 1e-6,
            "steps": 600,
            "learning_rate": 1e-3,
            "fit_data_binding_sha256": (
                "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
            ),
            "ordinary_gates_sha256": (
                "0ef08515366888ff11f83269c68fc154477202b64bf2308e6b68985b05e91cd5"
            ),
            "contrast_gates_sha256": (
                "bedf561b190b04f880aabad6020ddb680d187734d572c9ec1abba7573cad0de1"
            ),
        }
        for name, frozen in expected.items():
            value = getattr(self, name)
            if value != frozen or type(value) is not type(frozen):
                raise ValueError(f"rank-64 matched D3 {name} is not frozen")
        for name in (
            "fit_data_binding_sha256",
            "ordinary_gates_sha256",
            "contrast_gates_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_TRAINING_DOMAIN,
                label="matched D3 training artifact",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.rank64_matched_d3_training",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "fit_materialization_roles": ["pilot", "fit"],
            "batch_order_semantics": (
                "exact_recipe_independent_c2_fit_data_binding_and_index_order"
            ),
            "canonical_scoring_semantics": (
                "raw_canonical_fisher_metric_for_all_fit_scoring"
            ),
            "objective_balance_gate_semantics": (
                "same_d3_initial_contribution_balance_gate"
            ),
            "ordinary_gate_semantics": "all_12_existing_gate_flags_must_pass",
            "contrast_gate_semantics": (
                "all_null_radial_and_signed_fit_contrasts_must_pass"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "MatchedD3TrainingSpec":
        state = _mapping(raw, label="matched D3 training")
        expected = set(cls().state_dict())
        _strict_keys(state, expected=expected, label="matched D3 training")
        defaults = cls()._payload()
        semantic_names = (
            "artifact_kind",
            "format_version",
            "fit_materialization_roles",
            "batch_order_semantics",
            "canonical_scoring_semantics",
            "objective_balance_gate_semantics",
            "ordinary_gate_semantics",
            "contrast_gate_semantics",
        )
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError("matched D3 training semantics drifted")
        return cls(
            **{
                name: state[name]
                for name in cls.__dataclass_fields__
            }
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ObjectiveBalanceResultBinding:
    """Authenticated source result and exact failed D3 primary identity."""

    protocol_sha256: str = (
        "d502d003fd86f6ef7322e35854d0bb738fdc4cfa6fa5089c2812e366a142d2eb"
    )
    logical_artifact_sha256: str = (
        "e171361c2a2f43083f9e591d27c1d7b4555302c89d9d6e6e2f54e6be7cfd9cb0"
    )
    tensor_sha256: str = (
        "a76519a832519103030cafe6646ecd1912212a4f9154a0f4a246277c516c9d5a"
    )
    report_sha256: str = (
        "88394ae24648eca541a7b83ad48afec0681772bb231ff0fa5866ab75d74510ed"
    )
    code_bundle_sha256: str = (
        "c88cd41db520e953c08b389df47e5befb6e3207c336e2dea79231e76ac4bed31"
    )
    d3_recipe_sha256: str = (
        "853f09e644b2020cb90b056a06b594e7cc8af2f72c89a447e0a4988ebabb6c3b"
    )
    d3_primary_plan_sha256: str = (
        "4591214359dd37a39c95f79419870d871814690e5dad546696d93da045813142"
    )
    d3_primary_result_sha256: str = (
        "601bfbdddd91fe37364947a3703810bfb88db4043ac5ed0b1817e9b35d4950f1"
    )
    fit_data_binding_sha256: str = (
        "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
    )
    d3_fit_batch_sequence_sha256: str = (
        "775da8a3506c7f7574aba6ee651cf472f45cbb35cf390d925f891bcb40bdcde5"
    )
    d3_fit_batch_content_sequence_sha256: str = (
        "72e634d78bab6002b5bb88b58ca05cab03094ea97658679608fa06c96818cde0"
    )
    d3_fit_indexed_batch_sequence_sha256: str = (
        "f80a057a292db7a15e3a5673e5c7d002ac3e5ccd74942c9a915f9700b004cdde"
    )
    d3_fit_endpoint_sequence_sha256: str = (
        "ea3dcc77851607f3267343164fb9f4690f4a1c9e610a29099ed1668823a5cf04"
    )
    d3_fit_pair_sequence_sha256: str = (
        "a3823fd453cc13770132cab9e57cc2b9159af16fa3d9bb913f043475361d62df"
    )
    d3_natural_pair_sequence_sha256: str = (
        "26cadfb562be90f5719a95733d6cbdd23c64865605d864b122a1144e1e21282b"
    )
    d3_balanced_pair_sequence_sha256: str = (
        "fb2234ff3342585dd1e0098241e6351fe5e30a2aa9edb2646ffa59a0b37fb379"
    )
    d3_source_replay_binding_sha256: str = (
        "cb033006bbbeb6e2692bb9957d4b21b9fedeeefe03fc7c39f9fbe7fb10cc211b"
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        frozen = _default_result_binding_values()
        for name, expected in frozen.items():
            value = getattr(self, name)
            if value != expected:
                raise ValueError(
                    f"objective-balance result {name} is not frozen"
                )
            _require_sha256(value, label=name)
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_RESULT_BINDING_DOMAIN,
                label="objective-balance result binding",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": (
                "fisher_graph.rank64_capacity_objective_result_binding"
            ),
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "source_outcome": "no_primary_treatment_passed_fit_gates",
            "source_d3_seed_role": "primary",
            "source_d3_passed_all_fit_gates": False,
            "source_authorized_fresh_c3_recipe_id": None,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "ObjectiveBalanceResultBinding":
        state = _mapping(raw, label="objective-balance result binding")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="objective-balance result binding",
        )
        defaults = cls()._payload()
        for name in (
            "artifact_kind",
            "format_version",
            "source_outcome",
            "source_d3_seed_role",
            "source_d3_passed_all_fit_gates",
            "source_authorized_fresh_c3_recipe_id",
        ):
            if state[name] != defaults[name]:
                raise ValueError(
                    "objective-balance result binding semantics drifted"
                )
        return cls(
            **{
                name: state[name]
                for name in cls.__dataclass_fields__
            }
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Rank64CapacityControlProtocol:
    """Complete fit-only declaration for the matched rank-64 control."""

    baseline_latent_rank: int
    latent_rank: int
    visible_source_modes: int
    visible_target_modes: int
    execution_device: str
    execution_dtype: str
    primary_seed: int
    replication_seed: int
    executor: Rank64ExecutorSpec
    training: MatchedD3TrainingSpec
    source_result: ObjectiveBalanceResultBinding
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        exact_values: dict[str, object] = {
            "baseline_latent_rank": CAPACITY_CONTROL_BASELINE_LATENT_RANK,
            "latent_rank": CAPACITY_CONTROL_LATENT_RANK,
            "visible_source_modes": 64,
            "visible_target_modes": 64,
            "execution_device": CAPACITY_CONTROL_EXECUTION_DEVICE,
            "execution_dtype": CAPACITY_CONTROL_EXECUTION_DTYPE,
            "primary_seed": CAPACITY_CONTROL_PRIMARY_SEED,
            "replication_seed": CAPACITY_CONTROL_REPLICATION_SEED,
        }
        for name, expected in exact_values.items():
            value = getattr(self, name)
            if value != expected or type(value) is not type(expected):
                raise ValueError(f"rank-64 protocol {name} is not frozen")
        if not isinstance(self.executor, Rank64ExecutorSpec):
            raise TypeError("rank-64 executor has the wrong type")
        if self.executor != Rank64ExecutorSpec():
            raise ValueError("rank-64 executor drifted")
        if not isinstance(self.training, MatchedD3TrainingSpec):
            raise TypeError("matched D3 training has the wrong type")
        if self.training != MatchedD3TrainingSpec():
            raise ValueError("matched D3 training drifted")
        if not isinstance(self.source_result, ObjectiveBalanceResultBinding):
            raise TypeError("source result binding has the wrong type")
        if self.source_result != ObjectiveBalanceResultBinding():
            raise ValueError("source result binding drifted")
        if (
            self.training.fit_data_binding_sha256
            != self.source_result.fit_data_binding_sha256
        ):
            raise ValueError("fit-data binding differs from source D3")
        if self.executor.input_modes != self.latent_rank + 2:
            raise ValueError("rank-64 executor input width is not r+2")
        if self.executor.output_modes != self.latent_rank:
            raise ValueError("rank-64 executor output width is not r")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_PROTOCOL_DOMAIN,
                label="rank-64 capacity-control protocol",
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
                "fit_only_rank64_capacity_control_not_compression_or_"
                "generalization"
            ),
            "baseline_latent_rank": self.baseline_latent_rank,
            "latent_rank": self.latent_rank,
            "visible_source_modes": self.visible_source_modes,
            "visible_target_modes": self.visible_target_modes,
            "execution_device": self.execution_device,
            "execution_dtype": self.execution_dtype,
            "primary_seed": self.primary_seed,
            "replication_seed": self.replication_seed,
            "executor": self.executor.state_dict(),
            "training": self.training.state_dict(),
            "source_result": self.source_result.state_dict(),
            "controlled_change": (
                "packed_latent_rank_16_to_64_only_with_derived_core_"
                "input_18_to_66_and_output_16_to_64"
            ),
            "unchanged_contract": (
                "d3_objective_fit_data_batch_order_optimizer_executor_family_"
                "expert_count_expert_rank_router_width_routing_causality_"
                "gates_seeds_device_dtype"
            ),
            "packing_semantics": "learned_64_to_r_to_64_modal_packing",
            "materialization_roles_allowed": ["pilot", "fit"],
            "c2_selection_role_allowed": False,
            "c2_provider_artifact_loading_allowed": False,
            "authenticated_source_result_artifact_loading_allowed": True,
            "source_result_loading_semantics": (
                "strict_hash_authenticated_source_safe_result_only"
            ),
            "primary_pass_authority": "authorize_replication_seed_only",
            "valid_full_rank_fit_failure_authority": (
                "investigate_executor_objective_or_optimization_budget_"
                "at_full_outer_rank"
            ),
            "treatment_validity_requirements": (
                "authenticated_source_and_fit_data_gauge_teacher_integrity_"
                "objective_balance_and_structural_checks"
            ),
            "treatment_validity_failure_authority": (
                "invalidate_rank_comparison_no_capacity_conclusion"
            ),
            "two_seed_pass_authority": (
                "supports_capacity_conclusion_and_separate_compressed_width_"
                "ladder_preregistration_only"
            ),
            "fresh_c3_authorized": False,
            "compression_claim_authorized": False,
            "full_rank_parameter_reduction_claim_authorized": False,
            "next_rung_if_two_seed_pass": (
                "separately_preregister_compressed_width_ladder"
            ),
            "next_rung_if_primary_or_replication_fails": (
                "investigate_executor_objective_or_optimization_before_"
                "width_ladder"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "Rank64CapacityControlProtocol":
        state = _mapping(raw, label="rank-64 capacity-control protocol")
        expected = set(_default_unchecked().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="rank-64 capacity-control protocol",
        )
        defaults = _default_unchecked()._payload()
        semantic_names = (
            "schema",
            "format_version",
            "scientific_scope",
            "controlled_change",
            "unchanged_contract",
            "packing_semantics",
            "materialization_roles_allowed",
            "c2_selection_role_allowed",
            "c2_provider_artifact_loading_allowed",
            "authenticated_source_result_artifact_loading_allowed",
            "source_result_loading_semantics",
            "primary_pass_authority",
            "valid_full_rank_fit_failure_authority",
            "treatment_validity_requirements",
            "treatment_validity_failure_authority",
            "two_seed_pass_authority",
            "fresh_c3_authorized",
            "compression_claim_authorized",
            "full_rank_parameter_reduction_claim_authorized",
            "next_rung_if_two_seed_pass",
            "next_rung_if_primary_or_replication_fails",
        )
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError(
                    "rank-64 capacity-control protocol semantics drifted"
                )
        return cls(
            baseline_latent_rank=_exact_int(
                state["baseline_latent_rank"],
                label="baseline latent rank",
                minimum=1,
            ),
            latent_rank=_exact_int(
                state["latent_rank"], label="latent rank", minimum=1
            ),
            visible_source_modes=_exact_int(
                state["visible_source_modes"],
                label="visible source modes",
                minimum=1,
            ),
            visible_target_modes=_exact_int(
                state["visible_target_modes"],
                label="visible target modes",
                minimum=1,
            ),
            execution_device=str(state["execution_device"]),
            execution_dtype=str(state["execution_dtype"]),
            primary_seed=_exact_int(
                state["primary_seed"], label="primary seed", minimum=1
            ),
            replication_seed=_exact_int(
                state["replication_seed"],
                label="replication seed",
                minimum=1,
            ),
            executor=Rank64ExecutorSpec.from_state_dict(state["executor"]),
            training=MatchedD3TrainingSpec.from_state_dict(state["training"]),
            source_result=ObjectiveBalanceResultBinding.from_state_dict(
                state["source_result"]
            ),
            artifact_sha256=str(state["artifact_sha256"]),
        )


def _default_result_binding_values() -> dict[str, str]:
    return {
        "protocol_sha256": (
            "d502d003fd86f6ef7322e35854d0bb738fdc4cfa6fa5089c2812e366a142d2eb"
        ),
        "logical_artifact_sha256": (
            "e171361c2a2f43083f9e591d27c1d7b4555302c89d9d6e6e2f54e6be7cfd9cb0"
        ),
        "tensor_sha256": (
            "a76519a832519103030cafe6646ecd1912212a4f9154a0f4a246277c516c9d5a"
        ),
        "report_sha256": (
            "88394ae24648eca541a7b83ad48afec0681772bb231ff0fa5866ab75d74510ed"
        ),
        "code_bundle_sha256": (
            "c88cd41db520e953c08b389df47e5befb6e3207c336e2dea79231e76ac4bed31"
        ),
        "d3_recipe_sha256": (
            "853f09e644b2020cb90b056a06b594e7cc8af2f72c89a447e0a4988ebabb6c3b"
        ),
        "d3_primary_plan_sha256": (
            "4591214359dd37a39c95f79419870d871814690e5dad546696d93da045813142"
        ),
        "d3_primary_result_sha256": (
            "601bfbdddd91fe37364947a3703810bfb88db4043ac5ed0b1817e9b35d4950f1"
        ),
        "fit_data_binding_sha256": (
            "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
        ),
        "d3_fit_batch_sequence_sha256": (
            "775da8a3506c7f7574aba6ee651cf472f45cbb35cf390d925f891bcb40bdcde5"
        ),
        "d3_fit_batch_content_sequence_sha256": (
            "72e634d78bab6002b5bb88b58ca05cab03094ea97658679608fa06c96818cde0"
        ),
        "d3_fit_indexed_batch_sequence_sha256": (
            "f80a057a292db7a15e3a5673e5c7d002ac3e5ccd74942c9a915f9700b004cdde"
        ),
        "d3_fit_endpoint_sequence_sha256": (
            "ea3dcc77851607f3267343164fb9f4690f4a1c9e610a29099ed1668823a5cf04"
        ),
        "d3_fit_pair_sequence_sha256": (
            "a3823fd453cc13770132cab9e57cc2b9159af16fa3d9bb913f043475361d62df"
        ),
        "d3_natural_pair_sequence_sha256": (
            "26cadfb562be90f5719a95733d6cbdd23c64865605d864b122a1144e1e21282b"
        ),
        "d3_balanced_pair_sequence_sha256": (
            "fb2234ff3342585dd1e0098241e6351fe5e30a2aa9edb2646ffa59a0b37fb379"
        ),
        "d3_source_replay_binding_sha256": (
            "cb033006bbbeb6e2692bb9957d4b21b9fedeeefe03fc7c39f9fbe7fb10cc211b"
        ),
    }


def _default_unchecked() -> Rank64CapacityControlProtocol:
    return Rank64CapacityControlProtocol(
        baseline_latent_rank=CAPACITY_CONTROL_BASELINE_LATENT_RANK,
        latent_rank=CAPACITY_CONTROL_LATENT_RANK,
        visible_source_modes=64,
        visible_target_modes=64,
        execution_device=CAPACITY_CONTROL_EXECUTION_DEVICE,
        execution_dtype=CAPACITY_CONTROL_EXECUTION_DTYPE,
        primary_seed=CAPACITY_CONTROL_PRIMARY_SEED,
        replication_seed=CAPACITY_CONTROL_REPLICATION_SEED,
        executor=Rank64ExecutorSpec(),
        training=MatchedD3TrainingSpec(),
        source_result=ObjectiveBalanceResultBinding(),
    )


def default_rank64_capacity_control_protocol(
) -> Rank64CapacityControlProtocol:
    """Return the complete immutable matched rank-64 declaration."""

    protocol = _default_unchecked()
    if (
        protocol.protocol_sha256
        != DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            "rank-64 capacity-control protocol trust anchor drifted"
        )
    return protocol


# Literal trust anchor for the complete declaration.  It is intentionally not
# computed at import time; tests bind this value to the canonical state.
DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256 = (
    "03b1e595836ee325b83f5c2fc7355b31f7e5e6deceba92f9ad98ae27c29e6cf5"
)
