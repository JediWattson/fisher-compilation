"""Pure declaration for the paired Gemma L3/L4 nested-width control.

The experiment replays the authenticated D3 rank-16 cold start beside a
function-preserving rank-64 lift.  Both arms have the same observable
function, provider-chart JVP, objective, fit rows, optimizer, and training
budget at step zero.  The lifted arm exposes 48 additional internal channels
behind an initially-zero decoder gate; those channels must be demonstrably
gradient-open before the comparison is valid.

This module intentionally imports only the Python standard library.  It does
not load PyTorch, a model, or either source artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re


__all__ = [
    "DEFAULT_FUNCTION_PRESERVING_WIDTH_CONTROL_PROTOCOL_SHA256",
    "DeterministicInitializationBindings",
    "FunctionPreservingWidthControlProtocol",
    "NestedLiftSpec",
    "PairedTrainingSpec",
    "SourceArtifactBindings",
    "WidthExecutorSpec",
    "default_function_preserving_width_control_protocol",
]


_FORMAT_VERSION = 1
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_function_preserving_width_control."
    "protocol.v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXECUTOR_DOMAIN = b"fisher-graph:function-preserving-width:executor:v1\0"
_TRAINING_DOMAIN = b"fisher-graph:function-preserving-width:training:v1\0"
_LIFT_DOMAIN = b"fisher-graph:function-preserving-width:lift:v1\0"
_INITIAL_BINDINGS_DOMAIN = (
    b"fisher-graph:function-preserving-width:initial-bindings:v1\0"
)
_SOURCE_DOMAIN = b"fisher-graph:function-preserving-width:sources:v1\0"
_PROTOCOL_DOMAIN = b"fisher-graph:function-preserving-width:protocol:v1\0"


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


def _exact_float(value: object, *, label: str, minimum: float = 0.0) -> float:
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
class WidthExecutorSpec:
    """One exact instance of the shared gated causal executor family."""

    input_modes: int
    output_modes: int
    expert_count: int = 2
    expert_rank: int = 16
    router_width: int = 16
    same_position_skip: bool = False
    max_positive_lag: int | None = None
    router_activation: str = "tanh"
    source_normalized_routing: bool = True
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "input_modes",
            "output_modes",
            "expert_count",
            "expert_rank",
            "router_width",
        ):
            _exact_int(getattr(self, name), label=name, minimum=1)
        if self.input_modes != self.output_modes + 2:
            raise ValueError("executor input width must equal output width + 2")
        if self.expert_count != 2 or self.expert_rank != 16:
            raise ValueError("executor expert geometry drifted")
        if self.router_width != 16:
            raise ValueError("executor router width drifted")
        if self.same_position_skip:
            raise ValueError("same-position skip must remain disabled")
        if self.max_positive_lag is not None:
            raise ValueError("positive-lag support must remain unlimited")
        if self.router_activation != "tanh":
            raise ValueError("router activation drifted")
        if self.source_normalized_routing is not True:
            raise ValueError("source-normalized routing must remain enabled")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_EXECUTOR_DOMAIN,
                label="width executor",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.width_control_executor",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "causality_semantics": (
                "unlimited_strict_positive_logical_lag_no_same_position_skip"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "WidthExecutorSpec":
        state = _mapping(raw, label="width executor")
        expected = set(cls(input_modes=18, output_modes=16).state_dict())
        _strict_keys(state, expected=expected, label="width executor")
        if (
            state["artifact_kind"]
            != "fisher_graph.width_control_executor"
            or state["format_version"] != _FORMAT_VERSION
            or state["causality_semantics"]
            != (
                "unlimited_strict_positive_logical_lag_"
                "no_same_position_skip"
            )
        ):
            raise ValueError("width executor semantics drifted")
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
class PairedTrainingSpec:
    """Exact D3 objective, optimizer, data, and paired schedule."""

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
    primary_seed: int = 20_260_728_402
    replication_seed: int = 20_260_729_402
    fit_data_binding_sha256: str = (
        "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
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
    primary_extra_decoder_gradient_norm_step1: float = (
        18.511274540334576
    )
    primary_extra_encoder_gradient_norm_step2: float = (
        0.2653025501110589
    )
    primary_extra_executor_gradient_norm_step2: float = (
        0.1185988699547137
    )
    primary_gradient_audit_sha256: str = (
        "76e5be7533aadc59a93d0abf30b705493a700662aa89a424e1143b23bcbf1b1a"
    )
    replication_extra_decoder_gradient_norm_step1: float = (
        30.595469515281412
    )
    replication_extra_encoder_gradient_norm_step2: float = (
        0.41916259966843405
    )
    replication_extra_executor_gradient_norm_step2: float = (
        0.18454393743555148
    )
    replication_gradient_audit_sha256: str = (
        "b16b2908cad51571bfbdccbecd74c48942b13679549376bd2a566fccf14d1af8"
    )
    primary_initial_metrics_sha256: str = (
        "56ee6759438f98b1ba9628c055358d29622af48899d49a51e23297ad9779aa20"
    )
    primary_initial_pointwise_share: float = 0.12768012527011063
    primary_initial_relative_delta_share: float = 0.4671375512527843
    primary_initial_direction_share: float = 0.16582785380780718
    primary_initial_jvp_share: float = 0.23935446293284365
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        defaults: dict[str, object] = {
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
            "primary_seed": 20_260_728_402,
            "replication_seed": 20_260_729_402,
            "fit_data_binding_sha256": (
                "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
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
            "primary_extra_decoder_gradient_norm_step1": (
                18.511274540334576
            ),
            "primary_extra_encoder_gradient_norm_step2": (
                0.2653025501110589
            ),
            "primary_extra_executor_gradient_norm_step2": (
                0.1185988699547137
            ),
            "primary_gradient_audit_sha256": (
                "76e5be7533aadc59a93d0abf30b705493a700662aa89a424"
                "e1143b23bcbf1b1a"
            ),
            "replication_extra_decoder_gradient_norm_step1": (
                30.595469515281412
            ),
            "replication_extra_encoder_gradient_norm_step2": (
                0.41916259966843405
            ),
            "replication_extra_executor_gradient_norm_step2": (
                0.18454393743555148
            ),
            "replication_gradient_audit_sha256": (
                "b16b2908cad51571bfbdccbecd74c48942b13679549376bd2"
                "a566fccf14d1af8"
            ),
            "primary_initial_metrics_sha256": (
                "56ee6759438f98b1ba9628c055358d29622af48899d49a51e23297ad9779aa20"
            ),
            "primary_initial_pointwise_share": 0.12768012527011063,
            "primary_initial_relative_delta_share": 0.4671375512527843,
            "primary_initial_direction_share": 0.16582785380780718,
            "primary_initial_jvp_share": 0.23935446293284365,
        }
        for name, expected in defaults.items():
            value = getattr(self, name)
            if value != expected or type(value) is not type(expected):
                raise ValueError(f"paired training {name} is not frozen")
        for name in (
            "fit_data_binding_sha256",
            "ordinary_gates_sha256",
            "contrast_gates_sha256",
            "measurement_evidence_sha256",
            "primary_gradient_audit_sha256",
            "replication_gradient_audit_sha256",
            "primary_initial_metrics_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_TRAINING_DOMAIN,
                label="paired training",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.paired_width_training",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "optimizer": "fresh_adam_per_arm",
            "batch_semantics": "exact_authenticated_d3_order",
            "primary_schedule": "rank16_replay_then_rank64_lift",
            "replication_schedule": (
                "conditional_paired_seed_only_after_valid_primary_"
                "rank16_fail_rank64_pass"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "PairedTrainingSpec":
        state = _mapping(raw, label="paired training")
        expected = set(cls().state_dict())
        _strict_keys(state, expected=expected, label="paired training")
        defaults = cls()._payload()
        for name in (
            "artifact_kind",
            "format_version",
            "optimizer",
            "batch_semantics",
            "primary_schedule",
            "replication_schedule",
        ):
            if state[name] != defaults[name]:
                raise ValueError("paired training semantics drifted")
        field_names = {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        return cls(
            **{name: state[name] for name in field_names},  # type: ignore[arg-type]
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class NestedLiftSpec:
    """Function-preserving, decoder-gated expansion contract."""

    source_rank: int = 16
    target_rank: int = 64
    added_rank: int = 48
    equivalence_absolute_tolerance: float = 1e-12
    equivalence_relative_tolerance: float = 1e-12
    gradient_norm_floor: float = 1e-12
    extra_encoder_initialization: str = "identity_remaining_modal_modes"
    extra_output_initialization: str = "same_position_identity_bank"
    extra_decoder_initialization: str = "exact_zero_rows"
    nested_path_rule: str = (
        "first16_outputs_copy_rank16_and_ignore_added_inputs"
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
            "extra_encoder_initialization": (
                "identity_remaining_modal_modes"
            ),
            "extra_output_initialization": "same_position_identity_bank",
            "extra_decoder_initialization": "exact_zero_rows",
            "nested_path_rule": (
                "first16_outputs_copy_rank16_and_ignore_added_inputs"
            ),
        }
        for name, frozen in expected.items():
            value = getattr(self, name)
            if value != frozen or type(value) is not type(frozen):
                raise ValueError(f"nested lift {name} is not frozen")
        if self.target_rank - self.source_rank != self.added_rank:
            raise ValueError("nested lift rank arithmetic drifted")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_LIFT_DOMAIN,
                label="nested lift",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.nested_width_lift",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "observable_initialization": "exact_rank16_function",
            "jvp_initialization": "exact_rank16_provider_chart_jvp",
            "gradient_open_rule": (
                "extra_decoder_nonzero_gradient_step1_and_added_"
                "encoder_executor_nonzero_gradient_step2"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "NestedLiftSpec":
        state = _mapping(raw, label="nested lift")
        expected = set(cls().state_dict())
        _strict_keys(state, expected=expected, label="nested lift")
        defaults = cls()._payload()
        for name in (
            "artifact_kind",
            "format_version",
            "observable_initialization",
            "jvp_initialization",
            "gradient_open_rule",
        ):
            if state[name] != defaults[name]:
                raise ValueError("nested lift semantics drifted")
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
            extra_encoder_initialization=str(
                state["extra_encoder_initialization"]
            ),
            extra_output_initialization=str(
                state["extra_output_initialization"]
            ),
            extra_decoder_initialization=str(
                state["extra_decoder_initialization"]
            ),
            nested_path_rule=str(state["nested_path_rule"]),
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class DeterministicInitializationBindings:
    """Frozen parameter hashes for every preregistered cold/lifted arm."""

    primary_rank16_encoder_sha256: str = (
        "11178cc4065df1200d17ffa2661a99caced0ddf54fbcbf78c7b2dbbdb46d0fef"
    )
    primary_rank16_executor_sha256: str = (
        "94a0d18e8437ec86e64f07ab6f1c87de1bf76b7b8b0c919ec0264599a410762a"
    )
    primary_rank16_decoder_sha256: str = (
        "50db66f3cca564c40f458ae63a0c9794b1f389d2b680a69cec7866f7d0d36133"
    )
    primary_rank64_encoder_sha256: str = (
        "b5f0f50daa62e32a8a728cf943fe2335b5e40a720b465d0a582dae4c22b8e111"
    )
    primary_rank64_executor_sha256: str = (
        "717bdc26c43e7d8d78e93a5577d7b68e7656a5c98b990b9f1144d64ab59aabc2"
    )
    primary_rank64_decoder_sha256: str = (
        "78463f0856f6bd43d7eb9b2bbae0b5f82e7a10f9aefae16d16801a6cc15b5df1"
    )
    replication_rank16_encoder_sha256: str = (
        "11178cc4065df1200d17ffa2661a99caced0ddf54fbcbf78c7b2dbbdb46d0fef"
    )
    replication_rank16_executor_sha256: str = (
        "9e7498e662e3c883eb9d7ae96f6ef8c27441424d0ab5189588ea96cd0f0dd6c5"
    )
    replication_rank16_decoder_sha256: str = (
        "50db66f3cca564c40f458ae63a0c9794b1f389d2b680a69cec7866f7d0d36133"
    )
    replication_rank64_encoder_sha256: str = (
        "b5f0f50daa62e32a8a728cf943fe2335b5e40a720b465d0a582dae4c22b8e111"
    )
    replication_rank64_executor_sha256: str = (
        "94d03be042711fd277cf46ab82fee16c2353dcc01d41550a753d5b7ee71ff572"
    )
    replication_rank64_decoder_sha256: str = (
        "78463f0856f6bd43d7eb9b2bbae0b5f82e7a10f9aefae16d16801a6cc15b5df1"
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "artifact_sha256":
                _require_sha256(getattr(self, name), label=name)
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_INITIAL_BINDINGS_DOMAIN,
                label="deterministic initialization bindings",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": (
                "fisher_graph.deterministic_width_initialization_bindings"
            ),
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "reconstruction": (
                "fresh_seeded_rank16_then_exact_nested_rank64_lift"
            ),
            "tensor_hash_semantics": (
                "canonical_tensor_sha256_and_module_state_fingerprint"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def hashes_for(self, *, seed_role: str, arm: str) -> dict[str, str]:
        if seed_role not in {"primary", "replication"}:
            raise ValueError("initialization seed role is invalid")
        if arm not in {"rank16", "rank64"}:
            raise ValueError("initialization arm is invalid")
        prefix = f"{seed_role}_{arm}"
        return {
            "encoder_sha256": getattr(self, f"{prefix}_encoder_sha256"),
            "executor_sha256": getattr(self, f"{prefix}_executor_sha256"),
            "decoder_sha256": getattr(self, f"{prefix}_decoder_sha256"),
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "DeterministicInitializationBindings":
        state = _mapping(raw, label="deterministic initialization bindings")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="deterministic initialization bindings",
        )
        defaults = cls()._payload()
        for name in (
            "artifact_kind",
            "format_version",
            "reconstruction",
            "tensor_hash_semantics",
        ):
            if state[name] != defaults[name]:
                raise ValueError(
                    "deterministic initialization semantics drifted"
                )
        field_names = {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        return cls(
            **{name: state[name] for name in field_names},  # type: ignore[arg-type]
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class SourceArtifactBindings:
    """Exact identities for both authenticated predecessor artifacts."""

    d3_protocol_sha256: str = (
        "d502d003fd86f6ef7322e35854d0bb738fdc4cfa6fa5089c2812e366a142d2eb"
    )
    d3_logical_artifact_sha256: str = (
        "e171361c2a2f43083f9e591d27c1d7b4555302c89d9d6e6e2f54e6be7cfd9cb0"
    )
    d3_tensor_file_sha256: str = (
        "a76519a832519103030cafe6646ecd1912212a4f9154a0f4a246277c516c9d5a"
    )
    d3_report_sha256: str = (
        "88394ae24648eca541a7b83ad48afec0681772bb231ff0fa5866ab75d74510ed"
    )
    d3_code_bundle_sha256: str = (
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
    rank64_protocol_sha256: str = (
        "03b1e595836ee325b83f5c2fc7355b31f7e5e6deceba92f9ad98ae27c29e6cf5"
    )
    rank64_logical_artifact_sha256: str = (
        "cccc3fc51ce123a74cea9238bd359e8c09872f2432300d684c02a84d6f8a84fa"
    )
    rank64_tensor_file_sha256: str = (
        "3d44215ae206f85cf778ee85944b36bd58e0255299685b05bf689c27e5ba0f07"
    )
    rank64_report_sha256: str = (
        "8d49ed2b8591645d0c4e0a641a1d5253c8171b0129fd3d9be0cefcaf4a2e13f7"
    )
    rank64_code_bundle_sha256: str = (
        "e139fe6e7b7a55ffc325e0ddf49f23acce8aa9cbeacac312f64fbb0328f7ca9e"
    )
    rank64_primary_plan_sha256: str = (
        "78a6395869d5913d8187c571ea1beeb9382bc3d0dd42bb3ef6c010668eb54da6"
    )
    rank64_primary_result_sha256: str = (
        "b9f2f6fc7352278921e7ab8a435f1c2a7b5fa250fa5bf2062fc13f539d33112f"
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "artifact_sha256":
                _require_sha256(getattr(self, name), label=name)
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_SOURCE_DOMAIN,
                label="source bindings",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.width_control_source_bindings",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "d3_parameter_use": "cold_start_reconstruction_only",
            "rank64_parameter_use": "forbidden_provenance_only",
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "SourceArtifactBindings":
        state = _mapping(raw, label="source bindings")
        expected = set(cls().state_dict())
        _strict_keys(state, expected=expected, label="source bindings")
        defaults = cls()._payload()
        for name in (
            "artifact_kind",
            "format_version",
            "d3_parameter_use",
            "rank64_parameter_use",
        ):
            if state[name] != defaults[name]:
                raise ValueError("source binding semantics drifted")
        field_names = {
            name
            for name in cls.__dataclass_fields__
            if name != "artifact_sha256"
        }
        return cls(
            **{name: state[name] for name in field_names},  # type: ignore[arg-type]
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class FunctionPreservingWidthControlProtocol:
    """Complete immutable declaration for the paired control."""

    execution_device: str
    execution_dtype: str
    rank16_executor: WidthExecutorSpec
    rank64_executor: WidthExecutorSpec
    training: PairedTrainingSpec
    lift: NestedLiftSpec
    initialization: DeterministicInitializationBindings
    sources: SourceArtifactBindings
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.execution_device != "cpu":
            raise ValueError("execution device must remain cpu")
        if self.execution_dtype != "float32":
            raise ValueError("execution dtype must remain float32")
        if self.rank16_executor != WidthExecutorSpec(
            input_modes=18,
            output_modes=16,
        ):
            raise ValueError("rank-16 executor drifted")
        if self.rank64_executor != WidthExecutorSpec(
            input_modes=66,
            output_modes=64,
        ):
            raise ValueError("rank-64 executor drifted")
        if self.training != PairedTrainingSpec():
            raise ValueError("paired training drifted")
        if self.lift != NestedLiftSpec():
            raise ValueError("nested lift drifted")
        if self.initialization != DeterministicInitializationBindings():
            raise ValueError("deterministic initialization drifted")
        if self.sources != SourceArtifactBindings():
            raise ValueError("source bindings drifted")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                supplied=self.artifact_sha256,
                payload=self._payload(),
                domain=_PROTOCOL_DOMAIN,
                label="function-preserving width protocol",
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
                "fit_only_paired_function_preserving_width_control"
            ),
            "execution_device": self.execution_device,
            "execution_dtype": self.execution_dtype,
            "rank16_executor": self.rank16_executor.state_dict(),
            "rank64_executor": self.rank64_executor.state_dict(),
            "training": self.training.state_dict(),
            "lift": self.lift.state_dict(),
            "initialization": self.initialization.state_dict(),
            "sources": self.sources.state_dict(),
            "controlled_change": (
                "gradient_open_nested_latent_width_16_to_64_at_exact_"
                "initial_observable_and_jvp"
            ),
            "primary_arms": [
                "rank16_exact_d3_replay",
                "rank64_function_preserving_lift",
            ],
            "primary_decision": (
                "rank16_fail_rank64_pass_authorizes_paired_replication_only"
            ),
            "replication_decision": (
                "rank16_fail_rank64_pass_supports_outer_width_attribution"
            ),
            "both_fail_authority": (
                "initial_balance_confound_removed_outer_width_alone_"
                "insufficient_under_matched_budget_next_expert_core_control"
            ),
            "both_pass_authority": (
                "seed_or_optimization_effect_no_unique_width_attribution"
            ),
            "invalid_authority": "no_capacity_or_optimization_conclusion",
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
            "longer_optimization_mixed_into_primary": False,
            "expert_rank_change_mixed_into_primary": False,
            "conditional_residual_mixed_into_primary": False,
            "publication_root": "ignored_local_runs_only",
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "FunctionPreservingWidthControlProtocol":
        state = _mapping(raw, label="function-preserving width protocol")
        expected = set(_default_unchecked().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="function-preserving width protocol",
        )
        defaults = _default_unchecked()._payload()
        semantic_names = set(defaults) - {
            "execution_device",
            "execution_dtype",
            "rank16_executor",
            "rank64_executor",
            "training",
            "lift",
            "initialization",
            "sources",
        }
        for name in semantic_names:
            if state[name] != defaults[name]:
                raise ValueError(
                    "function-preserving width semantics drifted"
                )
        return cls(
            execution_device=str(state["execution_device"]),
            execution_dtype=str(state["execution_dtype"]),
            rank16_executor=WidthExecutorSpec.from_state_dict(
                state["rank16_executor"]
            ),
            rank64_executor=WidthExecutorSpec.from_state_dict(
                state["rank64_executor"]
            ),
            training=PairedTrainingSpec.from_state_dict(state["training"]),
            lift=NestedLiftSpec.from_state_dict(state["lift"]),
            initialization=(
                DeterministicInitializationBindings.from_state_dict(
                    state["initialization"]
                )
            ),
            sources=SourceArtifactBindings.from_state_dict(state["sources"]),
            artifact_sha256=str(state["artifact_sha256"]),
        )


def _default_unchecked() -> FunctionPreservingWidthControlProtocol:
    return FunctionPreservingWidthControlProtocol(
        execution_device="cpu",
        execution_dtype="float32",
        rank16_executor=WidthExecutorSpec(
            input_modes=18,
            output_modes=16,
        ),
        rank64_executor=WidthExecutorSpec(
            input_modes=66,
            output_modes=64,
        ),
        training=PairedTrainingSpec(),
        lift=NestedLiftSpec(),
        initialization=DeterministicInitializationBindings(),
        sources=SourceArtifactBindings(),
    )


def default_function_preserving_width_control_protocol(
) -> FunctionPreservingWidthControlProtocol:
    """Return the complete declaration and verify its literal trust anchor."""

    protocol = _default_unchecked()
    if (
        protocol.protocol_sha256
        != DEFAULT_FUNCTION_PRESERVING_WIDTH_CONTROL_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            "function-preserving width protocol trust anchor drifted"
        )
    return protocol


# Literal trust anchor for the declaration; never computed during import.
DEFAULT_FUNCTION_PRESERVING_WIDTH_CONTROL_PROTOCOL_SHA256 = (
    "c3ad81c84d41108839b5fcab13e3b5d47d99a55ae9a9223c3f116edb6b457597"
)
