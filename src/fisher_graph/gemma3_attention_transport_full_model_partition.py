"""Stable protocol helpers for the full-model Gemma attention-transport test.

The experiment has two orthogonal holdouts.  Generator and router fitting may
read teacher rows only at the first and third six-layer macroblocks; the middle
macroblock is a zero-shot structural holdout.  Within the already-designated A
role, a stable 30/6/9 nested sample split is derived from prompt-free manifest
hashes.  Runtime execution receipts authenticate later executions, but never
choose or alter either split.

This module is intentionally standard-library-only.  It freezes structure,
feature encoding, budgets, gates, and firewall/decision receipts without
loading a model or a tensor runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math

from . import (
    gemma3_attention_transport_global_gated_residual_spectrum_partition
    as _stable_A,
)


FULL_MODEL_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_attention_transport.full_model_protocol.v1"
)
FULL_MODEL_PROTOCOL_FORMAT_VERSION = 1

LAYER_COUNT = 18
MACROBLOCK_COUNT = 3
SITES_PER_MACROBLOCK = 6
INVOCATION_IDS = tuple(f"mutation.{2 * index + 1}" for index in range(LAYER_COUNT))
GLOBAL_ATTENTION_INVOCATION_IDS = ("mutation.11", "mutation.23", "mutation.35")
ENDPOINT_MACROBLOCK_ORDINALS = (0, 2)
MIDDLE_HOLDOUT_MACROBLOCK_ORDINAL = 1
ENDPOINT_TRAIN_INVOCATION_IDS = INVOCATION_IDS[:6] + INVOCATION_IDS[12:]
MIDDLE_HOLDOUT_INVOCATION_IDS = INVOCATION_IDS[6:12]
HELD_GLOBAL_INVOCATION_ID = "mutation.23"

A_DIRECTION_TRAIN_BATCH_ORDINALS = _stable_A.DIRECTION_TRAIN_BATCH_ORDINALS
A_ALLOCATION_VALIDATION_BATCH_ORDINALS = (
    _stable_A.ALLOCATION_VALIDATION_BATCH_ORDINALS
)
A_OUTER_FIT_BATCH_ORDINALS = _stable_A.OUTER_FIT_BATCH_ORDINALS
A_OUTER_DEV_BATCH_ORDINALS = _stable_A.OUTER_DEV_BATCH_ORDINALS
A_BATCH_COUNT = _stable_A.BATCH_COUNT
A_TARGET_TOKEN_COUNTS = _stable_A.TARGET_TOKEN_COUNTS

GENERATOR_COUNT = 8
GENERATOR_RANK = 8
ACTIVE_GENERATOR_COUNT = 2
PARENT_BANK_PARAMETERS = 113_728
RESIDUAL_COMPONENT_PARAMETERS = 10_240
MAXIMUM_RESIDUAL_COMPONENT_COUNT = 2
MAXIMUM_GENERATOR_BANK_PARAMETERS = 134_208
ATLAS_PARAMETERS = 23_041
ROUTER_PARAMETERS = 28_800
MAXIMUM_CANDIDATE_PARAMETERS = 186_049
NATIVE_VALUE_OUTPUT_PARAMETERS = 14_745_600
INDEPENDENT_BASELINE_PARAMETER_CAP = MAXIMUM_CANDIDATE_PARAMETERS
REQUIRED_FULL_MODEL_ARMS = (
    "native",
    "shared_zero_shot_top2",
    "middle_block_zeroed_control",
    "middle_block_structural_feature_permutation_control",
    "budget_capped_independent_per_layer_ceiling",
)

if (
    MAXIMUM_GENERATOR_BANK_PARAMETERS
    != PARENT_BANK_PARAMETERS
    + MAXIMUM_RESIDUAL_COMPONENT_COUNT * RESIDUAL_COMPONENT_PARAMETERS
    or MAXIMUM_CANDIDATE_PARAMETERS
    != MAXIMUM_GENERATOR_BANK_PARAMETERS + ATLAS_PARAMETERS + ROUTER_PARAMETERS
):
    raise RuntimeError("full-model frozen parameter budget does not close")

NO_FREE_LAYER_FEATURE_NAMES = (
    "normalized_depth",
    "repeated_site_0",
    "repeated_site_1",
    "repeated_site_2",
    "repeated_site_3",
    "repeated_site_4",
    "repeated_site_5",
    "local_attention",
    "global_attention",
)
NO_FREE_LAYER_FEATURE_SEMANTICS = (
    "fixed_depth_plus_repeated_site_plus_local_global_one_hot_no_layer_id_v1"
)
FULL_MODEL_EQUATION_SEMANTICS = (
    "shared_k8_r8_depth_linear_repeated_site_top2_all_18_value_output_v1"
)
FULL_MODEL_SPLIT_SEMANTICS = (
    "endpoint_macroblocks_train_middle_macroblock_zero_shot_nested_A_30_6_9_v1"
)

GATE_ORDER = (
    "firewall",
    "held_block_relative_fisher_loss",
    "mutation_23_relative_fisher_loss",
    "router_oracle_relative_fisher_loss_gap",
    "delta_nll_per_token",
    "kl_per_token",
    "top1_agreement",
    "real_gain_over_zeroed_control",
    "permuted_control_gain_fraction",
)

_LAYER_CATALOG_DOMAIN = b"fisher_graph:attention-full-model-layer-catalog:v1\0"
_FEATURE_CONTRACT_DOMAIN = b"fisher_graph:attention-full-model-features:v1\0"
_STABLE_A_SPLIT_DOMAIN = b"fisher_graph:attention-full-model-stable-A-split:v1\0"
_GATE_THRESHOLDS_DOMAIN = b"fisher_graph:attention-full-model-gates:v1\0"
_PROTOCOL_DOMAIN = b"fisher_graph:attention-full-model-protocol:v1\0"
_FIREWALL_DOMAIN = b"fisher_graph:attention-full-model-firewall:v1\0"
_MEASUREMENTS_DOMAIN = b"fisher_graph:attention-full-model-measurements:v1\0"
_DECISION_DOMAIN = b"fisher_graph:attention-full-model-decision:v1\0"
_GATE_COMPARISON_ABS_TOLERANCE = 1.0e-12


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")
    return value


def _require_finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _exceeds_maximum(value: float, maximum: float) -> bool:
    return value > maximum and not math.isclose(
        value,
        maximum,
        rel_tol=0.0,
        abs_tol=_GATE_COMPARISON_ABS_TOLERANCE,
    )


def _falls_below_minimum(value: float, minimum: float) -> bool:
    return value < minimum and not math.isclose(
        value,
        minimum,
        rel_tol=0.0,
        abs_tol=_GATE_COMPARISON_ABS_TOLERANCE,
    )


@dataclass(frozen=True, slots=True)
class AttentionLayerSite:
    """One fixed Gemma attention layer in the crossed three-by-six layout."""

    layer_ordinal: int
    invocation_id: str
    macroblock_ordinal: int
    repeated_site_ordinal: int
    attention_kind: str

    def __post_init__(self) -> None:
        if type(self.layer_ordinal) is not int or not 0 <= self.layer_ordinal < 18:
            raise ValueError("attention layer ordinal is invalid")
        expected_invocation = f"mutation.{2 * self.layer_ordinal + 1}"
        expected_block, expected_site = divmod(self.layer_ordinal, 6)
        expected_kind = "global" if expected_site == 5 else "local"
        if (
            self.invocation_id != expected_invocation
            or self.macroblock_ordinal != expected_block
            or self.repeated_site_ordinal != expected_site
            or self.attention_kind != expected_kind
        ):
            raise ValueError("attention layer catalog entry drifted")

    @property
    def is_global_attention(self) -> bool:
        return self.attention_kind == "global"

    @property
    def normalized_depth(self) -> float:
        # Exact endpoints -1 and +1, with the middle block interpolated.
        return float(2 * self.layer_ordinal - (LAYER_COUNT - 1)) / float(
            LAYER_COUNT - 1
        )

    def payload(self) -> dict[str, object]:
        return {
            "attention_kind": self.attention_kind,
            "invocation_id": self.invocation_id,
            "layer_ordinal": self.layer_ordinal,
            "macroblock_ordinal": self.macroblock_ordinal,
            "normalized_depth_denominator": LAYER_COUNT - 1,
            "normalized_depth_numerator": (
                2 * self.layer_ordinal - (LAYER_COUNT - 1)
            ),
            "repeated_site_ordinal": self.repeated_site_ordinal,
        }


ATTENTION_LAYER_CATALOG = tuple(
    AttentionLayerSite(
        layer_ordinal=layer_ordinal,
        invocation_id=invocation_id,
        macroblock_ordinal=layer_ordinal // SITES_PER_MACROBLOCK,
        repeated_site_ordinal=layer_ordinal % SITES_PER_MACROBLOCK,
        attention_kind=(
            "global"
            if layer_ordinal % SITES_PER_MACROBLOCK == SITES_PER_MACROBLOCK - 1
            else "local"
        ),
    )
    for layer_ordinal, invocation_id in enumerate(INVOCATION_IDS)
)
ATTENTION_LAYER_CATALOG_SHA256 = _digest(
    _LAYER_CATALOG_DOMAIN,
    [site.payload() for site in ATTENTION_LAYER_CATALOG],
)


def attention_layer_site(invocation_id: str) -> AttentionLayerSite:
    """Return the frozen structural entry for ``invocation_id``."""

    if not isinstance(invocation_id, str):
        raise TypeError("invocation_id must be a string")
    for site in ATTENTION_LAYER_CATALOG:
        if site.invocation_id == invocation_id:
            return site
    raise ValueError("invocation_id is outside the fixed 18-layer catalog")


@dataclass(frozen=True, slots=True)
class NoFreeLayerFeatureEncodingContract:
    """Fixed structural router features with no learned layer identifier."""

    feature_names: tuple[str, ...] = NO_FREE_LAYER_FEATURE_NAMES
    feature_semantics: str = NO_FREE_LAYER_FEATURE_SEMANTICS
    learned_layer_embedding_count: int = 0
    free_per_layer_parameter_count: int = 0
    includes_layer_one_hot: bool = False
    includes_macroblock_one_hot: bool = False
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            self.feature_names != NO_FREE_LAYER_FEATURE_NAMES
            or self.feature_semantics != NO_FREE_LAYER_FEATURE_SEMANTICS
            or self.learned_layer_embedding_count != 0
            or self.free_per_layer_parameter_count != 0
            or self.includes_layer_one_hot is not False
            or self.includes_macroblock_one_hot is not False
        ):
            raise ValueError("no-free-layer feature contract drifted")
        computed = _digest(_FEATURE_CONTRACT_DOMAIN, self._payload())
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="feature contract artifact"
        ) != computed:
            raise ValueError("feature contract artifact hash mismatch")

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def encode(self, invocation_id: str) -> tuple[float, ...]:
        """Encode depth, repeated site, and attention type deterministically."""

        site = attention_layer_site(invocation_id)
        repeated = tuple(
            1.0 if index == site.repeated_site_ordinal else 0.0
            for index in range(SITES_PER_MACROBLOCK)
        )
        return (
            site.normalized_depth,
            *repeated,
            0.0 if site.is_global_attention else 1.0,
            1.0 if site.is_global_attention else 0.0,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "encoded_catalog": [
                {
                    "features": self.encode(site.invocation_id),
                    "invocation_id": site.invocation_id,
                }
                for site in ATTENTION_LAYER_CATALOG
            ],
            "feature_names": self.feature_names,
            "feature_semantics": self.feature_semantics,
            "free_per_layer_parameter_count": self.free_per_layer_parameter_count,
            "includes_layer_one_hot": self.includes_layer_one_hot,
            "includes_macroblock_one_hot": self.includes_macroblock_one_hot,
            "layer_catalog_sha256": ATTENTION_LAYER_CATALOG_SHA256,
            "learned_layer_embedding_count": self.learned_layer_embedding_count,
        }

    def validate_integrity(self) -> None:
        if _digest(_FEATURE_CONTRACT_DOMAIN, self._payload()) != self.artifact_sha256:
            raise ValueError("feature contract artifact hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


DEFAULT_NO_FREE_LAYER_FEATURE_ENCODING_CONTRACT = (
    NoFreeLayerFeatureEncodingContract()
)


def _catalog_payload(catalog: _stable_A.StableABatchCatalog) -> dict[str, object]:
    return {
        "batch_plan_artifact_sha256": catalog.batch_plan_artifact_sha256,
        "batches": [
            {
                "batch_identity_sha256": batch.batch_identity_sha256,
                "batch_ordinal": batch.batch_ordinal,
                "ordered_context_fingerprint_sha256s": (
                    batch.ordered_context_fingerprint_sha256s
                ),
                "ordered_objective_row_stream_sha256s": (
                    batch.ordered_objective_row_stream_sha256s
                ),
            }
            for batch in catalog.batches
        ],
        "role_panel_binding_sha256": catalog.role_panel_binding_sha256,
    }


def _selected_batch_identities(
    catalog: _stable_A.StableABatchCatalog,
    ordinals: Sequence[int],
) -> tuple[str, ...]:
    by_ordinal = {batch.batch_ordinal: batch for batch in catalog.batches}
    return tuple(by_ordinal[ordinal].batch_identity_sha256 for ordinal in ordinals)


@dataclass(frozen=True, slots=True)
class FullModelStableASplit:
    """Prompt-free A split whose identity is independent of runtime receipts."""

    catalog: _stable_A.StableABatchCatalog
    outer_split_artifact_sha256: str
    inner_split_artifact_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, _stable_A.StableABatchCatalog):
            raise TypeError("full-model stable A split requires a stable catalog")
        expected_outer = _stable_A.outer_split_artifact_sha256(self.catalog)
        expected_inner = _stable_A.inner_split_artifact_sha256(self.catalog)
        if (
            self.outer_split_artifact_sha256 != expected_outer
            or self.inner_split_artifact_sha256 != expected_inner
        ):
            raise ValueError("full-model stable A nested split drifted")
        groups = (
            self.direction_train_batch_identity_sha256s,
            self.allocation_validation_batch_identity_sha256s,
            self.outer_dev_batch_identity_sha256s,
        )
        if (
            tuple(map(len, groups)) != (30, 6, 9)
            or len(set().union(*(set(group) for group in groups))) != A_BATCH_COUNT
            or any(
                set(groups[left_index]).intersection(groups[right_index])
                for left_index in range(len(groups))
                for right_index in range(left_index + 1, len(groups))
            )
        ):
            raise ValueError("full-model stable A sample roles are not disjoint")
        computed = _digest(_STABLE_A_SPLIT_DOMAIN, self._payload())
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="stable A split artifact"
        ) != computed:
            raise ValueError("stable A split artifact hash mismatch")

    @property
    def direction_train_batch_identity_sha256s(self) -> tuple[str, ...]:
        return _selected_batch_identities(
            self.catalog, A_DIRECTION_TRAIN_BATCH_ORDINALS
        )

    @property
    def allocation_validation_batch_identity_sha256s(self) -> tuple[str, ...]:
        return _selected_batch_identities(
            self.catalog, A_ALLOCATION_VALIDATION_BATCH_ORDINALS
        )

    @property
    def outer_fit_batch_identity_sha256s(self) -> tuple[str, ...]:
        return _selected_batch_identities(self.catalog, A_OUTER_FIT_BATCH_ORDINALS)

    @property
    def outer_dev_batch_identity_sha256s(self) -> tuple[str, ...]:
        return _selected_batch_identities(self.catalog, A_OUTER_DEV_BATCH_ORDINALS)

    def _payload(self) -> dict[str, object]:
        return {
            "allocation_validation_batch_identity_sha256s": (
                self.allocation_validation_batch_identity_sha256s
            ),
            "allocation_validation_batch_ordinals": (
                A_ALLOCATION_VALIDATION_BATCH_ORDINALS
            ),
            "batch_catalog": _catalog_payload(self.catalog),
            "contains_runtime_execution_receipts": False,
            "direction_train_batch_identity_sha256s": (
                self.direction_train_batch_identity_sha256s
            ),
            "direction_train_batch_ordinals": A_DIRECTION_TRAIN_BATCH_ORDINALS,
            "inner_split_artifact_sha256": self.inner_split_artifact_sha256,
            "outer_dev_batch_identity_sha256s": (
                self.outer_dev_batch_identity_sha256s
            ),
            "outer_dev_batch_ordinals": A_OUTER_DEV_BATCH_ORDINALS,
            "outer_fit_batch_identity_sha256s": (
                self.outer_fit_batch_identity_sha256s
            ),
            "outer_fit_batch_ordinals": A_OUTER_FIT_BATCH_ORDINALS,
            "outer_split_artifact_sha256": self.outer_split_artifact_sha256,
            "split_semantics": FULL_MODEL_SPLIT_SEMANTICS,
        }

    def validate_integrity(self) -> None:
        if _digest(_STABLE_A_SPLIT_DOMAIN, self._payload()) != self.artifact_sha256:
            raise ValueError("stable A split artifact hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def build_full_model_stable_A_split(
    *,
    role_panel_binding_sha256: str,
    ordered_members: Sequence[Mapping[str, object]],
) -> FullModelStableASplit:
    """Build the pinned 30/6/9 A split without accepting runtime receipts."""

    catalog = _stable_A.build_stable_A_batch_catalog(
        role_panel_binding_sha256=role_panel_binding_sha256,
        ordered_members=ordered_members,
    )
    return FullModelStableASplit(
        catalog=catalog,
        outer_split_artifact_sha256=_stable_A.outer_split_artifact_sha256(catalog),
        inner_split_artifact_sha256=_stable_A.inner_split_artifact_sha256(catalog),
    )


@dataclass(frozen=True, slots=True)
class FullModelGateThresholds:
    """Frozen fail-closed A-screen thresholds."""

    held_block_relative_fisher_loss_max: float = 0.10
    mutation_23_relative_fisher_loss_max: float = 0.10
    router_oracle_relative_fisher_loss_gap_max: float = 0.02
    delta_nll_per_token_max: float = 0.10
    kl_per_token_max: float = 0.10
    top1_agreement_min: float = 0.90
    real_gain_over_zeroed_control_min_exclusive: float = 0.0
    permuted_control_gain_fraction_max: float = 0.25
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        expected = (0.10, 0.10, 0.02, 0.10, 0.10, 0.90, 0.0, 0.25)
        observed = (
            self.held_block_relative_fisher_loss_max,
            self.mutation_23_relative_fisher_loss_max,
            self.router_oracle_relative_fisher_loss_gap_max,
            self.delta_nll_per_token_max,
            self.kl_per_token_max,
            self.top1_agreement_min,
            self.real_gain_over_zeroed_control_min_exclusive,
            self.permuted_control_gain_fraction_max,
        )
        if any(
            _require_finite(value, label="gate threshold") != expected_value
            for value, expected_value in zip(observed, expected)
        ):
            raise ValueError("full-model gate thresholds drifted")
        computed = _digest(_GATE_THRESHOLDS_DOMAIN, self._payload())
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="gate thresholds artifact"
        ) != computed:
            raise ValueError("gate thresholds artifact hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "delta_nll_per_token_max": self.delta_nll_per_token_max,
            "held_block_relative_fisher_loss_max": (
                self.held_block_relative_fisher_loss_max
            ),
            "kl_per_token_max": self.kl_per_token_max,
            "mutation_23_relative_fisher_loss_max": (
                self.mutation_23_relative_fisher_loss_max
            ),
            "permuted_control_gain_fraction_max": (
                self.permuted_control_gain_fraction_max
            ),
            "real_gain_over_zeroed_control_min_exclusive": (
                self.real_gain_over_zeroed_control_min_exclusive
            ),
            "router_oracle_relative_fisher_loss_gap_max": (
                self.router_oracle_relative_fisher_loss_gap_max
            ),
            "top1_agreement_min": self.top1_agreement_min,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


DEFAULT_FULL_MODEL_GATE_THRESHOLDS = FullModelGateThresholds()


@dataclass(frozen=True, slots=True)
class AttentionTransportFullModelProtocol:
    """Hash-bound protocol for the zero-shot middle-block full-model screen."""

    stable_A_split_artifact_sha256: str
    A_batch_plan_artifact_sha256: str
    A_outer_split_artifact_sha256: str
    A_inner_split_artifact_sha256: str
    layer_catalog_sha256: str = ATTENTION_LAYER_CATALOG_SHA256
    feature_contract_sha256: str = (
        DEFAULT_NO_FREE_LAYER_FEATURE_ENCODING_CONTRACT.artifact_sha256
    )
    gate_thresholds_sha256: str = DEFAULT_FULL_MODEL_GATE_THRESHOLDS.artifact_sha256
    protocol_sha256: str = ""

    def __post_init__(self) -> None:
        for label, value in (
            ("stable A split", self.stable_A_split_artifact_sha256),
            ("A batch plan", self.A_batch_plan_artifact_sha256),
            ("A outer split", self.A_outer_split_artifact_sha256),
            ("A inner split", self.A_inner_split_artifact_sha256),
            ("layer catalog", self.layer_catalog_sha256),
            ("feature contract", self.feature_contract_sha256),
            ("gate thresholds", self.gate_thresholds_sha256),
        ):
            _require_sha256(value, label=label)
        if (
            self.layer_catalog_sha256 != ATTENTION_LAYER_CATALOG_SHA256
            or self.feature_contract_sha256
            != DEFAULT_NO_FREE_LAYER_FEATURE_ENCODING_CONTRACT.artifact_sha256
            or self.gate_thresholds_sha256
            != DEFAULT_FULL_MODEL_GATE_THRESHOLDS.artifact_sha256
        ):
            raise ValueError("full-model protocol fixed helper binding drifted")
        computed = _digest(_PROTOCOL_DOMAIN, self._payload())
        if self.protocol_sha256 == "":
            object.__setattr__(self, "protocol_sha256", computed)
        elif _require_sha256(
            self.protocol_sha256, label="full-model protocol"
        ) != computed:
            raise ValueError("full-model protocol hash mismatch")

    @property
    def maximum_parameter_reduction_fraction(self) -> float:
        return 1.0 - MAXIMUM_CANDIDATE_PARAMETERS / NATIVE_VALUE_OUTPUT_PARAMETERS

    def _payload(self) -> dict[str, object]:
        return {
            "A_batch_plan_artifact_sha256": self.A_batch_plan_artifact_sha256,
            "A_inner_split_artifact_sha256": self.A_inner_split_artifact_sha256,
            "A_outer_split_artifact_sha256": self.A_outer_split_artifact_sha256,
            "active_generator_count": ACTIVE_GENERATOR_COUNT,
            "all_18_value_output_sites_replaced": True,
            "atlas_parameters": ATLAS_PARAMETERS,
            "candidate_trajectory_execution": True,
            "endpoint_macroblock_ordinals": ENDPOINT_MACROBLOCK_ORDINALS,
            "endpoint_train_invocation_ids": ENDPOINT_TRAIN_INVOCATION_IDS,
            "equation_semantics": FULL_MODEL_EQUATION_SEMANTICS,
            "feature_contract_sha256": self.feature_contract_sha256,
            "format_version": FULL_MODEL_PROTOCOL_FORMAT_VERSION,
            "free_per_layer_parameter_count": 0,
            "gate_thresholds_sha256": self.gate_thresholds_sha256,
            "generator_count": GENERATOR_COUNT,
            "generator_rank": GENERATOR_RANK,
            "independent_baseline_parameter_cap": (
                INDEPENDENT_BASELINE_PARAMETER_CAP
            ),
            "layer_catalog_sha256": self.layer_catalog_sha256,
            "maximum_candidate_parameters": MAXIMUM_CANDIDATE_PARAMETERS,
            "maximum_generator_bank_parameters": (
                MAXIMUM_GENERATOR_BANK_PARAMETERS
            ),
            "maximum_parameter_reduction_fraction": (
                self.maximum_parameter_reduction_fraction
            ),
            "maximum_residual_component_count": (
                MAXIMUM_RESIDUAL_COMPONENT_COUNT
            ),
            "middle_holdout_invocation_ids": MIDDLE_HOLDOUT_INVOCATION_IDS,
            "middle_holdout_macroblock_ordinal": (
                MIDDLE_HOLDOUT_MACROBLOCK_ORDINAL
            ),
            "native_value_output_parameters": NATIVE_VALUE_OUTPUT_PARAMETERS,
            "native_qk_attention_retained": True,
            "oracle_full_model_trajectory_forbidden": True,
            "parent_bank_parameters": PARENT_BANK_PARAMETERS,
            "residual_component_parameters": RESIDUAL_COMPONENT_PARAMETERS,
            "router_parameters": ROUTER_PARAMETERS,
            "required_full_model_arms": REQUIRED_FULL_MODEL_ARMS,
            "schema": FULL_MODEL_PROTOCOL_SCHEMA,
            "split_semantics": FULL_MODEL_SPLIT_SEMANTICS,
            "stable_A_split_artifact_sha256": (
                self.stable_A_split_artifact_sha256
            ),
        }

    def validate_integrity(self) -> None:
        if _digest(_PROTOCOL_DOMAIN, self._payload()) != self.protocol_sha256:
            raise ValueError("full-model protocol hash mismatch")

    def validate_against_split(self, split: FullModelStableASplit) -> None:
        if not isinstance(split, FullModelStableASplit):
            raise TypeError("protocol split validation requires a stable A split")
        self.validate_integrity()
        split.validate_integrity()
        if (
            self.stable_A_split_artifact_sha256 != split.artifact_sha256
            or self.A_batch_plan_artifact_sha256
            != split.catalog.batch_plan_artifact_sha256
            or self.A_outer_split_artifact_sha256
            != split.outer_split_artifact_sha256
            or self.A_inner_split_artifact_sha256
            != split.inner_split_artifact_sha256
        ):
            raise ValueError("full-model protocol stable A split binding drifted")

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "protocol_sha256": self.protocol_sha256}


def build_attention_transport_full_model_protocol(
    split: FullModelStableASplit,
) -> AttentionTransportFullModelProtocol:
    if not isinstance(split, FullModelStableASplit):
        raise TypeError("full-model protocol requires a stable A split")
    split.validate_integrity()
    return AttentionTransportFullModelProtocol(
        stable_A_split_artifact_sha256=split.artifact_sha256,
        A_batch_plan_artifact_sha256=split.catalog.batch_plan_artifact_sha256,
        A_outer_split_artifact_sha256=split.outer_split_artifact_sha256,
        A_inner_split_artifact_sha256=split.inner_split_artifact_sha256,
    )


@dataclass(frozen=True, slots=True)
class FullModelFirewallReceipt:
    """Immutable pre-A-decision read/firewall and executable-content receipt."""

    protocol_sha256: str
    candidate_artifact_sha256: str
    candidate_frozen_before_outer_A_dev: bool
    heldout_teacher_in_candidate_lineage: bool
    heldout_teacher_reads_before_freeze: int
    outer_A_dev_reads_before_freeze: int
    outer_A_dev_opened_after_freeze: bool
    B_guard_opened_before_A_pass: bool
    B_guard_reads_before_A_pass: int
    C_assessment_opened_before_B_pass: bool
    C_assessment_reads_before_B_pass: int
    refit_after_sealed_read: bool
    reselection_after_sealed_read: bool
    oracle_used_in_full_model_trajectory: bool
    native_internal_states_used_in_candidate_trajectory: bool
    free_layer_parameter_count: int
    replaced_value_output_site_count: int
    native_value_output_parameters_in_executor: int
    teacher_payload_values_in_executor: int
    control_permutation_frozen_before_outer_A_dev: bool
    candidate_published_before_decision: bool
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_sha256, label="firewall protocol")
        _require_sha256(self.candidate_artifact_sha256, label="firewall candidate")
        for label, value in (
            (
                "candidate_frozen_before_outer_A_dev",
                self.candidate_frozen_before_outer_A_dev,
            ),
            (
                "heldout_teacher_in_candidate_lineage",
                self.heldout_teacher_in_candidate_lineage,
            ),
            (
                "outer_A_dev_opened_after_freeze",
                self.outer_A_dev_opened_after_freeze,
            ),
            ("B_guard_opened_before_A_pass", self.B_guard_opened_before_A_pass),
            (
                "C_assessment_opened_before_B_pass",
                self.C_assessment_opened_before_B_pass,
            ),
            ("refit_after_sealed_read", self.refit_after_sealed_read),
            ("reselection_after_sealed_read", self.reselection_after_sealed_read),
            (
                "oracle_used_in_full_model_trajectory",
                self.oracle_used_in_full_model_trajectory,
            ),
            (
                "native_internal_states_used_in_candidate_trajectory",
                self.native_internal_states_used_in_candidate_trajectory,
            ),
            (
                "control_permutation_frozen_before_outer_A_dev",
                self.control_permutation_frozen_before_outer_A_dev,
            ),
            (
                "candidate_published_before_decision",
                self.candidate_published_before_decision,
            ),
        ):
            _require_bool(value, label=label)
        for label, value in (
            (
                "heldout_teacher_reads_before_freeze",
                self.heldout_teacher_reads_before_freeze,
            ),
            ("outer_A_dev_reads_before_freeze", self.outer_A_dev_reads_before_freeze),
            ("B_guard_reads_before_A_pass", self.B_guard_reads_before_A_pass),
            (
                "C_assessment_reads_before_B_pass",
                self.C_assessment_reads_before_B_pass,
            ),
            ("free_layer_parameter_count", self.free_layer_parameter_count),
            (
                "replaced_value_output_site_count",
                self.replaced_value_output_site_count,
            ),
            (
                "native_value_output_parameters_in_executor",
                self.native_value_output_parameters_in_executor,
            ),
            (
                "teacher_payload_values_in_executor",
                self.teacher_payload_values_in_executor,
            ),
        ):
            _require_nonnegative_int(value, label=label)
        computed = _digest(_FIREWALL_DOMAIN, self._payload())
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="firewall receipt artifact"
        ) != computed:
            raise ValueError("firewall receipt artifact hash mismatch")

    @property
    def passed(self) -> bool:
        return (
            self.candidate_frozen_before_outer_A_dev
            and not self.heldout_teacher_in_candidate_lineage
            and self.heldout_teacher_reads_before_freeze == 0
            and self.outer_A_dev_reads_before_freeze == 0
            and self.outer_A_dev_opened_after_freeze
            and not self.B_guard_opened_before_A_pass
            and self.B_guard_reads_before_A_pass == 0
            and not self.C_assessment_opened_before_B_pass
            and self.C_assessment_reads_before_B_pass == 0
            and not self.refit_after_sealed_read
            and not self.reselection_after_sealed_read
            and not self.oracle_used_in_full_model_trajectory
            and not self.native_internal_states_used_in_candidate_trajectory
            and self.free_layer_parameter_count == 0
            and self.replaced_value_output_site_count == LAYER_COUNT
            and self.native_value_output_parameters_in_executor == 0
            and self.teacher_payload_values_in_executor == 0
            and self.control_permutation_frozen_before_outer_A_dev
            and not self.candidate_published_before_decision
        )

    def _payload(self) -> dict[str, object]:
        return {
            "B_guard_opened_before_A_pass": self.B_guard_opened_before_A_pass,
            "B_guard_reads_before_A_pass": self.B_guard_reads_before_A_pass,
            "C_assessment_opened_before_B_pass": (
                self.C_assessment_opened_before_B_pass
            ),
            "C_assessment_reads_before_B_pass": (
                self.C_assessment_reads_before_B_pass
            ),
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "candidate_frozen_before_outer_A_dev": (
                self.candidate_frozen_before_outer_A_dev
            ),
            "candidate_published_before_decision": (
                self.candidate_published_before_decision
            ),
            "control_permutation_frozen_before_outer_A_dev": (
                self.control_permutation_frozen_before_outer_A_dev
            ),
            "free_layer_parameter_count": self.free_layer_parameter_count,
            "heldout_teacher_in_candidate_lineage": (
                self.heldout_teacher_in_candidate_lineage
            ),
            "heldout_teacher_reads_before_freeze": (
                self.heldout_teacher_reads_before_freeze
            ),
            "native_internal_states_used_in_candidate_trajectory": (
                self.native_internal_states_used_in_candidate_trajectory
            ),
            "native_value_output_parameters_in_executor": (
                self.native_value_output_parameters_in_executor
            ),
            "oracle_used_in_full_model_trajectory": (
                self.oracle_used_in_full_model_trajectory
            ),
            "outer_A_dev_reads_before_freeze": (
                self.outer_A_dev_reads_before_freeze
            ),
            "outer_A_dev_opened_after_freeze": (
                self.outer_A_dev_opened_after_freeze
            ),
            "protocol_sha256": self.protocol_sha256,
            "refit_after_sealed_read": self.refit_after_sealed_read,
            "reselection_after_sealed_read": self.reselection_after_sealed_read,
            "replaced_value_output_site_count": (
                self.replaced_value_output_site_count
            ),
            "teacher_payload_values_in_executor": (
                self.teacher_payload_values_in_executor
            ),
        }

    def validate_integrity(self) -> None:
        if _digest(_FIREWALL_DOMAIN, self._payload()) != self.artifact_sha256:
            raise ValueError("firewall receipt artifact hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class FullModelGateMeasurements:
    """Raw finite measurements used to recompute every A-screen gate.

    ``held_block_relative_fisher_loss`` and the mutation-23 metric are from
    the deployable router.  The separately recorded exact-oracle held-block
    loss is diagnostic-only and closes the router-oracle gap.
    """

    held_block_relative_fisher_loss: float
    mutation_23_relative_fisher_loss: float
    held_block_oracle_relative_fisher_loss: float
    delta_nll_per_token: float
    kl_per_token: float
    top1_agreement: float
    real_gain_over_zeroed_control: float
    permuted_control_gain: float
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        nonnegative = (
            "held_block_relative_fisher_loss",
            "mutation_23_relative_fisher_loss",
            "held_block_oracle_relative_fisher_loss",
            "kl_per_token",
        )
        for name in nonnegative:
            if _require_finite(getattr(self, name), label=name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "delta_nll_per_token",
            "real_gain_over_zeroed_control",
            "permuted_control_gain",
        ):
            _require_finite(getattr(self, name), label=name)
        top1 = _require_finite(self.top1_agreement, label="top1_agreement")
        if not 0.0 <= top1 <= 1.0:
            raise ValueError("top1_agreement must lie in [0,1]")
        if (
            self.held_block_relative_fisher_loss + 1.0e-12
            < self.held_block_oracle_relative_fisher_loss
        ):
            raise ValueError("deployable router loss is below the exact oracle")
        computed = _digest(_MEASUREMENTS_DOMAIN, self._payload())
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="gate measurements artifact"
        ) != computed:
            raise ValueError("gate measurements artifact hash mismatch")

    @property
    def router_oracle_relative_fisher_loss_gap(self) -> float:
        return (
            self.held_block_relative_fisher_loss
            - self.held_block_oracle_relative_fisher_loss
        )

    @property
    def permuted_control_gain_fraction(self) -> float | None:
        if self.real_gain_over_zeroed_control <= 0.0:
            return None
        return self.permuted_control_gain / self.real_gain_over_zeroed_control

    def _payload(self) -> dict[str, object]:
        return {
            "delta_nll_per_token": self.delta_nll_per_token,
            "held_block_relative_fisher_loss": (
                self.held_block_relative_fisher_loss
            ),
            "held_block_oracle_relative_fisher_loss": (
                self.held_block_oracle_relative_fisher_loss
            ),
            "kl_per_token": self.kl_per_token,
            "mutation_23_relative_fisher_loss": (
                self.mutation_23_relative_fisher_loss
            ),
            "permuted_control_gain": self.permuted_control_gain,
            "permuted_control_gain_fraction": (
                self.permuted_control_gain_fraction
            ),
            "real_gain_over_zeroed_control": self.real_gain_over_zeroed_control,
            "router_oracle_relative_fisher_loss_gap": (
                self.router_oracle_relative_fisher_loss_gap
            ),
            "top1_agreement": self.top1_agreement,
        }

    def validate_integrity(self) -> None:
        if _digest(_MEASUREMENTS_DOMAIN, self._payload()) != self.artifact_sha256:
            raise ValueError("gate measurements artifact hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def _failed_gates(
    measurements: FullModelGateMeasurements,
    firewall: FullModelFirewallReceipt,
) -> tuple[str, ...]:
    thresholds = DEFAULT_FULL_MODEL_GATE_THRESHOLDS
    failed: list[str] = []
    if not firewall.passed:
        failed.append("firewall")
    if (
        _exceeds_maximum(
            measurements.held_block_relative_fisher_loss,
            thresholds.held_block_relative_fisher_loss_max,
        )
    ):
        failed.append("held_block_relative_fisher_loss")
    if (
        _exceeds_maximum(
            measurements.mutation_23_relative_fisher_loss,
            thresholds.mutation_23_relative_fisher_loss_max,
        )
    ):
        failed.append("mutation_23_relative_fisher_loss")
    if (
        _exceeds_maximum(
            measurements.router_oracle_relative_fisher_loss_gap,
            thresholds.router_oracle_relative_fisher_loss_gap_max,
        )
    ):
        failed.append("router_oracle_relative_fisher_loss_gap")
    if _exceeds_maximum(
        measurements.delta_nll_per_token,
        thresholds.delta_nll_per_token_max,
    ):
        failed.append("delta_nll_per_token")
    if _exceeds_maximum(
        measurements.kl_per_token,
        thresholds.kl_per_token_max,
    ):
        failed.append("kl_per_token")
    if _falls_below_minimum(
        measurements.top1_agreement,
        thresholds.top1_agreement_min,
    ):
        failed.append("top1_agreement")
    real_gain = measurements.real_gain_over_zeroed_control
    if real_gain <= thresholds.real_gain_over_zeroed_control_min_exclusive:
        failed.append("real_gain_over_zeroed_control")
    if (
        real_gain <= 0.0
        or _exceeds_maximum(
            measurements.permuted_control_gain,
            thresholds.permuted_control_gain_fraction_max * real_gain,
        )
    ):
        failed.append("permuted_control_gain_fraction")
    return tuple(failed)


@dataclass(frozen=True, slots=True)
class FullModelGateDecision:
    """Hash-bound fail-closed decision with recomputed gate inventory."""

    protocol_sha256: str
    candidate_artifact_sha256: str
    gate_thresholds_sha256: str
    firewall: FullModelFirewallReceipt
    measurements: FullModelGateMeasurements
    failed_gates: tuple[str, ...]
    passed: bool
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_sha256, label="decision protocol")
        _require_sha256(self.candidate_artifact_sha256, label="decision candidate")
        if (
            self.gate_thresholds_sha256
            != DEFAULT_FULL_MODEL_GATE_THRESHOLDS.artifact_sha256
        ):
            raise ValueError("decision gate thresholds binding drifted")
        if (
            not isinstance(self.firewall, FullModelFirewallReceipt)
            or not isinstance(self.measurements, FullModelGateMeasurements)
        ):
            raise TypeError("decision inputs have the wrong type")
        self.firewall.validate_integrity()
        self.measurements.validate_integrity()
        if (
            self.firewall.protocol_sha256 != self.protocol_sha256
            or self.firewall.candidate_artifact_sha256
            != self.candidate_artifact_sha256
        ):
            raise ValueError("decision firewall lineage drifted")
        expected_failed = _failed_gates(self.measurements, self.firewall)
        if (
            type(self.failed_gates) is not tuple
            or self.failed_gates != expected_failed
            or tuple(name for name in GATE_ORDER if name in self.failed_gates)
            != self.failed_gates
        ):
            raise ValueError("decision failed-gate inventory drifted")
        if type(self.passed) is not bool or self.passed is not (not expected_failed):
            raise ValueError("decision pass bit drifted")
        computed = _digest(_DECISION_DOMAIN, self._payload())
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="gate decision artifact"
        ) != computed:
            raise ValueError("gate decision artifact hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "failed_gates": self.failed_gates,
            "firewall_receipt_sha256": self.firewall.artifact_sha256,
            "gate_measurements_sha256": self.measurements.artifact_sha256,
            "gate_thresholds_sha256": self.gate_thresholds_sha256,
            "passed": self.passed,
            "protocol_sha256": self.protocol_sha256,
        }

    def validate_integrity(self) -> None:
        self.firewall.validate_integrity()
        self.measurements.validate_integrity()
        if (
            _failed_gates(self.measurements, self.firewall) != self.failed_gates
            or _digest(_DECISION_DOMAIN, self._payload()) != self.artifact_sha256
        ):
            raise ValueError("gate decision integrity drifted")

    def validate_against(
        self,
        protocol: AttentionTransportFullModelProtocol,
    ) -> None:
        if not isinstance(protocol, AttentionTransportFullModelProtocol):
            raise TypeError("decision validation requires a full-model protocol")
        protocol.validate_integrity()
        self.validate_integrity()
        if self.protocol_sha256 != protocol.protocol_sha256:
            raise ValueError("gate decision protocol binding drifted")

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "firewall": self.firewall.state_dict(),
            "measurements": self.measurements.state_dict(),
        }


def evaluate_full_model_gate_decision(
    *,
    protocol: AttentionTransportFullModelProtocol,
    firewall: FullModelFirewallReceipt,
    measurements: FullModelGateMeasurements,
) -> FullModelGateDecision:
    """Recompute every gate and return one immutable fail-closed decision."""

    if not isinstance(protocol, AttentionTransportFullModelProtocol):
        raise TypeError("gate evaluation requires a full-model protocol")
    if not isinstance(firewall, FullModelFirewallReceipt):
        raise TypeError("gate evaluation requires a firewall receipt")
    if not isinstance(measurements, FullModelGateMeasurements):
        raise TypeError("gate evaluation requires gate measurements")
    protocol.validate_integrity()
    firewall.validate_integrity()
    measurements.validate_integrity()
    if (
        firewall.protocol_sha256 != protocol.protocol_sha256
        or protocol.gate_thresholds_sha256
        != DEFAULT_FULL_MODEL_GATE_THRESHOLDS.artifact_sha256
    ):
        raise ValueError("gate evaluation lineage drifted")
    failed = _failed_gates(measurements, firewall)
    return FullModelGateDecision(
        protocol_sha256=protocol.protocol_sha256,
        candidate_artifact_sha256=firewall.candidate_artifact_sha256,
        gate_thresholds_sha256=protocol.gate_thresholds_sha256,
        firewall=firewall,
        measurements=measurements,
        failed_gates=failed,
        passed=not failed,
    )


__all__ = [
    "ACTIVE_GENERATOR_COUNT",
    "A_ALLOCATION_VALIDATION_BATCH_ORDINALS",
    "A_BATCH_COUNT",
    "A_DIRECTION_TRAIN_BATCH_ORDINALS",
    "A_OUTER_DEV_BATCH_ORDINALS",
    "A_OUTER_FIT_BATCH_ORDINALS",
    "A_TARGET_TOKEN_COUNTS",
    "ATLAS_PARAMETERS",
    "ATTENTION_LAYER_CATALOG",
    "ATTENTION_LAYER_CATALOG_SHA256",
    "AttentionLayerSite",
    "AttentionTransportFullModelProtocol",
    "DEFAULT_FULL_MODEL_GATE_THRESHOLDS",
    "DEFAULT_NO_FREE_LAYER_FEATURE_ENCODING_CONTRACT",
    "ENDPOINT_MACROBLOCK_ORDINALS",
    "ENDPOINT_TRAIN_INVOCATION_IDS",
    "FULL_MODEL_EQUATION_SEMANTICS",
    "FULL_MODEL_PROTOCOL_FORMAT_VERSION",
    "FULL_MODEL_PROTOCOL_SCHEMA",
    "FULL_MODEL_SPLIT_SEMANTICS",
    "FullModelFirewallReceipt",
    "FullModelGateDecision",
    "FullModelGateMeasurements",
    "FullModelGateThresholds",
    "FullModelStableASplit",
    "GATE_ORDER",
    "GENERATOR_COUNT",
    "GENERATOR_RANK",
    "GLOBAL_ATTENTION_INVOCATION_IDS",
    "HELD_GLOBAL_INVOCATION_ID",
    "INDEPENDENT_BASELINE_PARAMETER_CAP",
    "INVOCATION_IDS",
    "LAYER_COUNT",
    "MACROBLOCK_COUNT",
    "MAXIMUM_CANDIDATE_PARAMETERS",
    "MAXIMUM_GENERATOR_BANK_PARAMETERS",
    "MIDDLE_HOLDOUT_INVOCATION_IDS",
    "MIDDLE_HOLDOUT_MACROBLOCK_ORDINAL",
    "NATIVE_VALUE_OUTPUT_PARAMETERS",
    "NO_FREE_LAYER_FEATURE_NAMES",
    "NO_FREE_LAYER_FEATURE_SEMANTICS",
    "NoFreeLayerFeatureEncodingContract",
    "PARENT_BANK_PARAMETERS",
    "RESIDUAL_COMPONENT_PARAMETERS",
    "REQUIRED_FULL_MODEL_ARMS",
    "ROUTER_PARAMETERS",
    "SITES_PER_MACROBLOCK",
    "attention_layer_site",
    "build_attention_transport_full_model_protocol",
    "build_full_model_stable_A_split",
    "evaluate_full_model_gate_decision",
]
