"""Typed native-ABI partitions plus one common downstream-response atlas.

Native operator interchangeability and downstream response similarity answer
different questions.  A native atlas may compare modes only when their exact
execution ABI matches.  A downstream-response atlas may compare heterogeneous
operators only after each operator has been observed in the same authenticated
response space on the same aligned rows with the same weighting contract.

This module keeps those two evidence planes separate while reusing the frozen
single-ABI implementation in :mod:`fisher_graph.mode_similarity_atlas`.  It
does not load a model, manipulate tensors, or use source ordinals as numerical
features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Final

from .mode_similarity_atlas import (
    AtlasConfig,
    DirectedFamilyExampleStats,
    DirectedFamilyStats,
    FrozenFamilyResample,
    ModeABI,
    ModeSimilarityAtlasReceipt,
    ModeSpec,
    WithinFamilyBootstrapConfig,
    build_mode_similarity_atlas_receipt,
    parse_mode_similarity_atlas_receipt_payload,
)


MULTI_ABI_MODE_ATLAS_SCHEMA: Final = (
    "fisher_graph.multi_abi_mode_atlas"
)
MULTI_ABI_MODE_ATLAS_FORMAT_VERSION: Final = 1

_RECEIPT_DOMAIN = b"fisher_graph.multi_abi_mode_atlas.receipt.v1\0"
_RESPONSE_ABI_DOMAIN = (
    b"fisher_graph.multi_abi_mode_atlas.downstream_response_abi.v1\0"
)
_RESPONSE_CLASS_DOMAIN = (
    b"fisher_graph.multi_abi_mode_atlas.response_equivalence_class.v1\0"
)
_MAX_RECEIPT_BYTES = 128 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


__all__ = [
    "MULTI_ABI_MODE_ATLAS_FORMAT_VERSION",
    "MULTI_ABI_MODE_ATLAS_SCHEMA",
    "DownstreamResponseABI",
    "DownstreamResponseAtlasInput",
    "DownstreamResponseAtlasReceipt",
    "DownstreamResponseExamplePairStats",
    "DownstreamResponsePairStats",
    "DownstreamResponseEquivalenceClass",
    "ModeOperatorKind",
    "MultiABIAtlasInventory",
    "MultiABIModeAtlasReceipt",
    "NativeABIPartitionReceipt",
    "NativeAtlasInput",
    "NativePartitionReceiptOrigin",
    "NativePartitionStatus",
    "TypedModeSpec",
    "build_multi_abi_mode_atlas_receipt",
    "load_multi_abi_mode_atlas_receipt",
    "parse_multi_abi_mode_atlas_receipt_payload",
    "save_multi_abi_mode_atlas_receipt",
]


_SCIENTIFIC_STATUS: dict[str, object] = {
    "analysis_only": True,
    "native_operator_interchangeability_partitioned_by_exact_abi": True,
    "downstream_response_geometry_is_not_native_interchangeability": True,
    "downstream_response_geometry_descriptive_only": True,
    "source_ordinal_features_used": False,
    "operator_kind_features_used": False,
    "operator_kind_used_only_for_exact_equivalence_member_order": True,
    "mode_id_used_only_as_final_exact_equivalence_member_order_tie_break": True,
    "authorizes_neighbor_claims": False,
    "authorizes_cluster_claims": False,
    "authorizes_generator_family_claims": False,
    "authorizes_merge": False,
    "authorizes_compression": False,
    "authorizes_deployment": False,
}

_SAFETY: dict[str, object] = {
    "contains_model_weights": False,
    "contains_raw_activations": False,
    "contains_raw_gradients": False,
    "contains_raw_downstream_responses": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_only_aggregate_sufficient_statistics": True,
}

_TERMINAL_NORM_POLICY: dict[str, object] = {
    "operator_kind": "terminal_norm",
    "role": "exact_downstream_response_baseline_infrastructure",
    "included_in_downstream_response_mode_ids": False,
    "native_generator_atlas_built": False,
    "authorizes_generator_family_membership": False,
    "future_sensitivity_study_requires_distinct_intervention_abi": True,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable canonical name")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_finite(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields must be exactly {sorted(expected)}")
    return value


def _json_exact_equal(left: object, right: object) -> bool:
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


class ModeOperatorKind(str, Enum):
    """Typed runtime roles represented by the successor atlas."""

    ATTENTION = "attention"
    MLP = "mlp"
    LEXICAL_INGRESS = "lexical_ingress"
    LEXICAL_EGRESS = "lexical_egress"
    TERMINAL_NORM = "terminal_norm"


class NativePartitionStatus(str, Enum):
    """Whether one exact native ABI has enough evidence for seriation."""

    ATLAS_BUILT = "atlas_built"
    INSUFFICIENT_MODES_FOR_SERIATION = (
        "insufficient_modes_for_seriation"
    )
    EXACT_BASELINE_INFRASTRUCTURE = "exact_baseline_infrastructure"


class NativePartitionReceiptOrigin(str, Enum):
    """How an authenticated native partition entered this receipt."""

    AUTHENTICATED_EXISTING_RECEIPT = "authenticated_existing_receipt"
    BUILT_FROM_SUPPLIED_STATISTICS = "built_from_supplied_statistics"


@dataclass(frozen=True, slots=True)
class MultiABIAtlasInventory:
    """Authenticated model-size-independent inventory for this exact rung.

    The real Gemma rung declares eighteen attention and eighteen MLP modes.
    Tests may use smaller inventories while exercising the identical contract.
    Lexical ingress, lexical egress, and terminal norm are exact singletons.
    """

    attention_mode_count: int
    mlp_mode_count: int

    def __post_init__(self) -> None:
        if _require_positive_int(
            self.attention_mode_count,
            label="attention_mode_count",
        ) < 3:
            raise ValueError("attention_mode_count must be at least three")
        if _require_positive_int(
            self.mlp_mode_count,
            label="mlp_mode_count",
        ) < 3:
            raise ValueError("mlp_mode_count must be at least three")

    @property
    def downstream_response_mode_count(self) -> int:
        return self.attention_mode_count + self.mlp_mode_count + 2

    @property
    def total_typed_mode_count(self) -> int:
        return self.downstream_response_mode_count + 1

    def canonical_payload(self) -> dict[str, object]:
        return {
            "downstream_response_mode_count": (
                self.downstream_response_mode_count
            ),
            "native_partition_requirements": {
                "attention": {
                    "mode_count": self.attention_mode_count,
                    "partition_count": 1,
                    "receipt_origin": (
                        NativePartitionReceiptOrigin
                        .BUILT_FROM_SUPPLIED_STATISTICS.value
                    ),
                },
                "lexical_egress": {
                    "mode_count": 1,
                    "partition_count": 1,
                    "status": (
                        NativePartitionStatus
                        .INSUFFICIENT_MODES_FOR_SERIATION.value
                    ),
                },
                "lexical_ingress": {
                    "mode_count": 1,
                    "partition_count": 1,
                    "status": (
                        NativePartitionStatus
                        .INSUFFICIENT_MODES_FOR_SERIATION.value
                    ),
                },
                "mlp": {
                    "mode_count": self.mlp_mode_count,
                    "partition_count": 1,
                    "receipt_origin": (
                        NativePartitionReceiptOrigin
                        .AUTHENTICATED_EXISTING_RECEIPT.value
                    ),
                },
                "terminal_norm": {
                    "mode_count": 1,
                    "partition_count": 1,
                    "status": (
                        NativePartitionStatus
                        .EXACT_BASELINE_INFRASTRUCTURE.value
                    ),
                },
            },
            "operator_mode_counts": {
                "attention": self.attention_mode_count,
                "lexical_egress": 1,
                "lexical_ingress": 1,
                "mlp": self.mlp_mode_count,
                "terminal_norm": 1,
            },
            "total_typed_mode_count": self.total_typed_mode_count,
        }


@dataclass(frozen=True, slots=True)
class TypedModeSpec:
    """One source mode with a native ABI and an explicit runtime role."""

    mode_id: str
    operator_kind: ModeOperatorKind
    native_mode_kind: str
    state_fingerprint_sha256: str
    native_abi: ModeABI

    def __post_init__(self) -> None:
        _require_name(self.mode_id, label="mode_id")
        if not isinstance(self.operator_kind, ModeOperatorKind):
            raise TypeError("operator_kind must be a ModeOperatorKind")
        _require_name(self.native_mode_kind, label="native_mode_kind")
        _require_sha256(
            self.state_fingerprint_sha256,
            label="state_fingerprint_sha256",
        )
        if not isinstance(self.native_abi, ModeABI):
            raise TypeError("native_abi must be a ModeABI")

    def native_mode_spec(self) -> ModeSpec:
        return ModeSpec(
            mode_id=self.mode_id,
            mode_kind=self.native_mode_kind,
            state_fingerprint_sha256=self.state_fingerprint_sha256,
            compatibility_abi=self.native_abi,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "mode_id": self.mode_id,
            "native_abi": {
                **self.native_abi.canonical_payload(),
                "sha256": self.native_abi.sha256,
            },
            "native_mode_kind": self.native_mode_kind,
            "operator_kind": self.operator_kind.value,
            "state_fingerprint_sha256": self.state_fingerprint_sha256,
        }


def _typed_mode_sort_key(mode: TypedModeSpec) -> tuple[str, str, str]:
    """Canonicalize logical modes without assuming unique physical state."""

    return (
        mode.state_fingerprint_sha256,
        mode.operator_kind.value,
        mode.mode_id,
    )


def _validate_typed_mode_state_reuse(
    modes: Sequence[TypedModeSpec],
) -> None:
    """Permit only the declared tied lexical ingress/egress state reuse."""

    modes_by_fingerprint: dict[str, list[TypedModeSpec]] = {}
    for mode in modes:
        modes_by_fingerprint.setdefault(
            mode.state_fingerprint_sha256,
            [],
        ).append(mode)
    for shared_modes in modes_by_fingerprint.values():
        if len(shared_modes) == 1:
            continue
        shared_kinds = {mode.operator_kind for mode in shared_modes}
        if len(shared_modes) != 2 or shared_kinds != {
            ModeOperatorKind.LEXICAL_INGRESS,
            ModeOperatorKind.LEXICAL_EGRESS,
        }:
            raise ValueError(
                "typed mode state may be shared only by the tied lexical "
                "ingress/egress pair"
            )


@dataclass(frozen=True, slots=True)
class DownstreamResponseABI:
    """Exact shared readout, intervention, row, and weighting contract."""

    abi_name: str
    abi_version: int
    response_coordinate_signature: str
    response_width: int
    baseline_signature: str
    intervention_signature: str
    row_catalog_sha256: str
    weighting_catalog_sha256: str

    def __post_init__(self) -> None:
        name = _require_name(self.abi_name, label="abi_name")
        if not name.startswith("downstream_response."):
            raise ValueError("response ABI name must start with downstream_response.")
        if re.search(r"\.v[0-9]+$", name):
            raise ValueError(
                "response ABI version belongs in abi_version, not abi_name"
            )
        _require_positive_int(self.abi_version, label="abi_version")
        _require_name(
            self.response_coordinate_signature,
            label="response_coordinate_signature",
        )
        _require_positive_int(self.response_width, label="response_width")
        _require_name(self.baseline_signature, label="baseline_signature")
        _require_name(
            self.intervention_signature,
            label="intervention_signature",
        )
        _require_sha256(self.row_catalog_sha256, label="row_catalog_sha256")
        _require_sha256(
            self.weighting_catalog_sha256,
            label="weighting_catalog_sha256",
        )

    @property
    def abi_id(self) -> str:
        return f"{self.abi_name}.v{self.abi_version}"

    def _payload_without_sha256(self) -> dict[str, object]:
        return {
            "abi_id": self.abi_id,
            "abi_name": self.abi_name,
            "abi_version": self.abi_version,
            "baseline_signature": self.baseline_signature,
            "comparison_definition": (
                "aligned_weighted_downstream_response_delta_v1"
            ),
            "intervention_signature": self.intervention_signature,
            "native_operator_interchangeability_implied": False,
            "response_coordinate_signature": (
                self.response_coordinate_signature
            ),
            "response_width": self.response_width,
            "row_catalog_sha256": self.row_catalog_sha256,
            "weighting_catalog_sha256": self.weighting_catalog_sha256,
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(
            _RESPONSE_ABI_DOMAIN,
            self._payload_without_sha256(),
        )

    def canonical_payload(self) -> dict[str, object]:
        return {**self._payload_without_sha256(), "sha256": self.sha256}

    def compatibility_mode_abi(self) -> ModeABI:
        """Return the exact ABI shared by the derived response wrappers."""

        return ModeABI(
            abi_name=self.abi_name,
            abi_version=self.abi_version,
            input_signature=self.intervention_signature,
            output_signature=self.response_coordinate_signature,
            context_signature=f"binding.{self.sha256}",
        )


@dataclass(frozen=True, slots=True)
class DownstreamResponsePairStats:
    """Aligned sufficient statistics for one unordered mode pair/family."""

    family_id: str
    mode_a_id: str
    mode_b_id: str
    response_abi_sha256: str
    weighted_sse: float
    weighted_mode_a_energy: float
    weighted_dot: float
    weighted_mode_b_energy: float
    row_count: int

    def __post_init__(self) -> None:
        _require_name(self.family_id, label="family_id")
        _require_name(self.mode_a_id, label="mode_a_id")
        _require_name(self.mode_b_id, label="mode_b_id")
        if self.mode_a_id == self.mode_b_id:
            raise ValueError("response pair must contain two distinct modes")
        _require_sha256(
            self.response_abi_sha256,
            label="response_abi_sha256",
        )
        _require_finite(self.weighted_sse, label="weighted_sse", minimum=0.0)
        _require_finite(
            self.weighted_mode_a_energy,
            label="weighted_mode_a_energy",
            minimum=0.0,
        )
        _require_finite(self.weighted_dot, label="weighted_dot")
        _require_finite(
            self.weighted_mode_b_energy,
            label="weighted_mode_b_energy",
            minimum=0.0,
        )
        _require_positive_int(self.row_count, label="row_count")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "mode_a_id": self.mode_a_id,
            "mode_b_id": self.mode_b_id,
            "response_abi_sha256": self.response_abi_sha256,
            "row_count": self.row_count,
            "weighted_dot": float(self.weighted_dot),
            "weighted_mode_a_energy": float(self.weighted_mode_a_energy),
            "weighted_mode_b_energy": float(self.weighted_mode_b_energy),
            "weighted_sse": float(self.weighted_sse),
        }


@dataclass(frozen=True, slots=True)
class DownstreamResponseExamplePairStats:
    """One example-plus-response-coordinates cluster for an unordered pair."""

    family_id: str
    example_cluster_id: str
    mode_a_id: str
    mode_b_id: str
    response_abi_sha256: str
    weighted_sse: float
    weighted_mode_a_energy: float
    weighted_dot: float
    weighted_mode_b_energy: float
    row_count: int

    def __post_init__(self) -> None:
        _require_name(self.example_cluster_id, label="example_cluster_id")
        self.as_pair_stats()

    def as_pair_stats(self) -> DownstreamResponsePairStats:
        return DownstreamResponsePairStats(
            family_id=self.family_id,
            mode_a_id=self.mode_a_id,
            mode_b_id=self.mode_b_id,
            response_abi_sha256=self.response_abi_sha256,
            weighted_sse=self.weighted_sse,
            weighted_mode_a_energy=self.weighted_mode_a_energy,
            weighted_dot=self.weighted_dot,
            weighted_mode_b_energy=self.weighted_mode_b_energy,
            row_count=self.row_count,
        )

    def canonical_payload(self) -> dict[str, object]:
        payload = self.as_pair_stats().canonical_payload()
        payload["example_cluster_id"] = self.example_cluster_id
        return payload


@dataclass(frozen=True, slots=True)
class DownstreamResponseEquivalenceClass:
    """Modes proven identical in the exact shared response coordinates.

    This is functional equality under one response ABI and frozen row catalog;
    it is explicitly not native-operator ABI equality.  Members are kept as
    distinct logical modes and are only collapsed for spectral fitting.
    """

    class_id: str
    class_state_fingerprint_sha256: str
    member_mode_ids: tuple[str, ...]
    representative_mode_id: str

    def __post_init__(self) -> None:
        _require_name(self.class_id, label="class_id")
        _require_sha256(
            self.class_state_fingerprint_sha256,
            label="class_state_fingerprint_sha256",
        )
        if not isinstance(self.member_mode_ids, tuple) or not self.member_mode_ids:
            raise ValueError("response equivalence class members must be nonempty")
        if len(set(self.member_mode_ids)) != len(self.member_mode_ids):
            raise ValueError("response equivalence class members must be unique")
        for mode_id in self.member_mode_ids:
            _require_name(mode_id, label="member_mode_ids")
        if self.representative_mode_id != self.member_mode_ids[0]:
            raise ValueError(
                "response equivalence representative must be the first member"
            )

    @property
    def is_nontrivial(self) -> bool:
        return len(self.member_mode_ids) > 1

    def canonical_payload(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "class_state_fingerprint_sha256": (
                self.class_state_fingerprint_sha256
            ),
            "equivalence_definition": (
                "exact_zero_sse_and_identical_quadratics_on_every_row_v1"
            ),
            "member_count": len(self.member_mode_ids),
            "member_mode_ids": list(self.member_mode_ids),
            "native_abi_equivalence_implied": False,
            "representative_mode_id": self.representative_mode_id,
        }


@dataclass(frozen=True, slots=True)
class NativeAtlasInput:
    """One exact native ABI supplied either as evidence or fresh statistics."""

    compatibility_abi_sha256: str
    prebuilt_receipt: ModeSimilarityAtlasReceipt | None = None
    directed_stats: tuple[DirectedFamilyStats, ...] = ()
    config: AtlasConfig | None = None
    frozen_resamples: tuple[FrozenFamilyResample, ...] = ()
    directed_example_stats: tuple[DirectedFamilyExampleStats, ...] = ()
    within_family_bootstrap_config: WithinFamilyBootstrapConfig | None = None

    def __post_init__(self) -> None:
        _require_sha256(
            self.compatibility_abi_sha256,
            label="compatibility_abi_sha256",
        )
        for name in (
            "directed_stats",
            "frozen_resamples",
            "directed_example_stats",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be a tuple")
        if any(
            not isinstance(item, DirectedFamilyStats)
            for item in self.directed_stats
        ):
            raise TypeError(
                "directed_stats must contain DirectedFamilyStats values"
            )
        if any(
            not isinstance(item, FrozenFamilyResample)
            for item in self.frozen_resamples
        ):
            raise TypeError(
                "frozen_resamples must contain FrozenFamilyResample values"
            )
        if any(
            not isinstance(item, DirectedFamilyExampleStats)
            for item in self.directed_example_stats
        ):
            raise TypeError(
                "directed_example_stats must contain "
                "DirectedFamilyExampleStats values"
            )
        if self.prebuilt_receipt is not None:
            if not isinstance(
                self.prebuilt_receipt,
                ModeSimilarityAtlasReceipt,
            ):
                raise TypeError(
                    "prebuilt_receipt must be a ModeSimilarityAtlasReceipt"
                )
            if (
                self.directed_stats
                or self.frozen_resamples
                or self.directed_example_stats
                or self.config is not None
                or self.within_family_bootstrap_config is not None
            ):
                raise ValueError(
                    "a prebuilt native receipt cannot be mixed with build inputs"
                )
        else:
            if not self.directed_stats:
                raise ValueError(
                    "fresh native input requires directed statistics"
                )
            if self.config is not None and not isinstance(
                self.config,
                AtlasConfig,
            ):
                raise TypeError("config must be an AtlasConfig or None")
            if (
                self.within_family_bootstrap_config is not None
                and not isinstance(
                    self.within_family_bootstrap_config,
                    WithinFamilyBootstrapConfig,
                )
            ):
                raise TypeError(
                    "within_family_bootstrap_config must be a "
                    "WithinFamilyBootstrapConfig or None"
                )

    @classmethod
    def from_receipt(
        cls,
        receipt: ModeSimilarityAtlasReceipt,
    ) -> "NativeAtlasInput":
        if not isinstance(receipt, ModeSimilarityAtlasReceipt):
            raise TypeError("receipt must be a ModeSimilarityAtlasReceipt")
        authenticated = parse_mode_similarity_atlas_receipt_payload(
            receipt.canonical_payload()
        )
        return cls(
            compatibility_abi_sha256=(
                authenticated.compatibility_abi_sha256
            ),
            prebuilt_receipt=authenticated,
        )


@dataclass(frozen=True, slots=True)
class DownstreamResponseAtlasInput:
    """Typed sufficient-statistic input for the heterogeneous response atlas."""

    response_abi: DownstreamResponseABI
    measurement_input_sha256: str
    mode_ids: tuple[str, ...]
    family_pair_stats: tuple[DownstreamResponsePairStats, ...]
    config: AtlasConfig | None = None
    frozen_resamples: tuple[FrozenFamilyResample, ...] = ()
    example_pair_stats: tuple[DownstreamResponseExamplePairStats, ...] = ()
    within_family_bootstrap_config: WithinFamilyBootstrapConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.response_abi, DownstreamResponseABI):
            raise TypeError("response_abi must be a DownstreamResponseABI")
        _require_sha256(
            self.measurement_input_sha256,
            label="measurement_input_sha256",
        )
        if not isinstance(self.mode_ids, tuple):
            raise TypeError("mode_ids must be a tuple")
        for mode_id in self.mode_ids:
            _require_name(mode_id, label="mode_ids")
        if len(set(self.mode_ids)) != len(self.mode_ids):
            raise ValueError("response mode_ids must be unique")
        if len(self.mode_ids) < 3:
            raise ValueError("response atlas requires at least three modes")
        for name in (
            "family_pair_stats",
            "frozen_resamples",
            "example_pair_stats",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be a tuple")
        if any(
            not isinstance(item, DownstreamResponsePairStats)
            for item in self.family_pair_stats
        ):
            raise TypeError(
                "family_pair_stats must contain DownstreamResponsePairStats"
            )
        if any(
            not isinstance(item, FrozenFamilyResample)
            for item in self.frozen_resamples
        ):
            raise TypeError(
                "frozen_resamples must contain FrozenFamilyResample values"
            )
        if any(
            not isinstance(item, DownstreamResponseExamplePairStats)
            for item in self.example_pair_stats
        ):
            raise TypeError(
                "example_pair_stats must contain "
                "DownstreamResponseExamplePairStats"
            )
        if not self.family_pair_stats:
            raise ValueError("family_pair_stats must not be empty")
        if self.config is not None and not isinstance(self.config, AtlasConfig):
            raise TypeError("config must be an AtlasConfig or None")
        if (
            self.within_family_bootstrap_config is not None
            and not isinstance(
                self.within_family_bootstrap_config,
                WithinFamilyBootstrapConfig,
            )
        ):
            raise TypeError(
                "within_family_bootstrap_config must be a "
                "WithinFamilyBootstrapConfig or None"
            )


@dataclass(frozen=True, slots=True)
class NativeABIPartitionReceipt:
    """One exact native ABI partition; never a mixed-ABI atlas."""

    operator_kind: ModeOperatorKind
    compatibility_abi_sha256: str
    mode_ids: tuple[str, ...]
    status: NativePartitionStatus
    receipt_origin: NativePartitionReceiptOrigin | None
    atlas: ModeSimilarityAtlasReceipt | None

    def __post_init__(self) -> None:
        if not isinstance(self.operator_kind, ModeOperatorKind):
            raise TypeError("operator_kind must be a ModeOperatorKind")
        _require_sha256(
            self.compatibility_abi_sha256,
            label="compatibility_abi_sha256",
        )
        if not isinstance(self.mode_ids, tuple) or not self.mode_ids:
            raise ValueError("native partition mode_ids must be nonempty")
        if len(set(self.mode_ids)) != len(self.mode_ids):
            raise ValueError("native partition mode_ids must be unique")
        if not isinstance(self.status, NativePartitionStatus):
            raise TypeError("status must be a NativePartitionStatus")
        if self.receipt_origin is not None and not isinstance(
            self.receipt_origin,
            NativePartitionReceiptOrigin,
        ):
            raise TypeError(
                "receipt_origin must be a NativePartitionReceiptOrigin or None"
            )
        if self.status is NativePartitionStatus.ATLAS_BUILT:
            if self.receipt_origin is None or not isinstance(
                self.atlas,
                ModeSimilarityAtlasReceipt,
            ):
                raise ValueError(
                    "built native partition requires an origin and atlas"
                )
        elif self.receipt_origin is not None or self.atlas is not None:
            raise ValueError(
                "non-atlas native partition cannot carry an origin or atlas"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "atlas": (
                None if self.atlas is None else self.atlas.canonical_payload()
            ),
            "compatibility_abi_sha256": self.compatibility_abi_sha256,
            "mode_count": len(self.mode_ids),
            "mode_ids": list(self.mode_ids),
            "native_operator_interchangeability_only": True,
            "operator_kind": self.operator_kind.value,
            "receipt_origin": (
                None
                if self.receipt_origin is None
                else self.receipt_origin.value
            ),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class DownstreamResponseAtlasReceipt:
    """One logical response atlas fitted on exact equivalence classes.

    ``equivalence_class_atlas`` contains the spectral fit, LOFO, and bootstrap
    calculations.  ``symmetric_distance`` and ``seriation_order`` expand that
    fit back to every logical mode.  No nearest-neighbor or cluster claim is
    made at either level.
    """

    response_abi: DownstreamResponseABI
    measurement_input_sha256: str
    mode_ids: tuple[str, ...]
    equivalence_classes: tuple[DownstreamResponseEquivalenceClass, ...]
    family_pair_stats: tuple[DownstreamResponsePairStats, ...]
    example_pair_stats: tuple[DownstreamResponseExamplePairStats, ...]
    equivalence_class_atlas: ModeSimilarityAtlasReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.response_abi, DownstreamResponseABI):
            raise TypeError("response_abi must be a DownstreamResponseABI")
        _require_sha256(
            self.measurement_input_sha256,
            label="measurement_input_sha256",
        )
        if not isinstance(self.mode_ids, tuple) or len(self.mode_ids) < 3:
            raise ValueError("response receipt requires at least three modes")
        if len(set(self.mode_ids)) != len(self.mode_ids):
            raise ValueError("response receipt mode_ids must be unique")
        if not isinstance(self.equivalence_classes, tuple) or not all(
            isinstance(item, DownstreamResponseEquivalenceClass)
            for item in self.equivalence_classes
        ):
            raise TypeError(
                "equivalence_classes must contain response equivalence classes"
            )
        if not isinstance(self.family_pair_stats, tuple) or not all(
            isinstance(item, DownstreamResponsePairStats)
            for item in self.family_pair_stats
        ):
            raise TypeError("family_pair_stats must contain response pair stats")
        if not isinstance(self.example_pair_stats, tuple) or not all(
            isinstance(item, DownstreamResponseExamplePairStats)
            for item in self.example_pair_stats
        ):
            raise TypeError(
                "example_pair_stats must contain response example pair stats"
            )
        if not isinstance(
            self.equivalence_class_atlas,
            ModeSimilarityAtlasReceipt,
        ):
            raise TypeError(
                "equivalence_class_atlas must be a ModeSimilarityAtlasReceipt"
            )
        if not self.equivalence_classes:
            raise ValueError("response equivalence classes must be nonempty")
        class_ids = [item.class_id for item in self.equivalence_classes]
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("response equivalence class IDs must be unique")
        class_fingerprints = [
            item.class_state_fingerprint_sha256
            for item in self.equivalence_classes
        ]
        if len(set(class_fingerprints)) != len(class_fingerprints):
            raise ValueError(
                "response equivalence class fingerprints must be unique"
            )
        members = [
            mode_id
            for item in self.equivalence_classes
            for mode_id in item.member_mode_ids
        ]
        if len(set(members)) != len(members) or set(members) != set(self.mode_ids):
            raise ValueError(
                "response equivalence classes must partition logical modes"
            )
        if set(class_ids) != {
            mode.mode_id for mode in self.equivalence_class_atlas.modes
        }:
            raise ValueError(
                "response equivalence classes must match fitted class modes"
            )
        if not self.family_pair_stats:
            raise ValueError("response family pair statistics must be nonempty")

    @property
    def atlas(self) -> ModeSimilarityAtlasReceipt:
        """Compatibility alias for the fitted equivalence-class atlas."""

        return self.equivalence_class_atlas

    @property
    def seriation_order(self) -> tuple[str, ...]:
        classes = {
            item.class_id: item for item in self.equivalence_classes
        }
        return tuple(
            mode_id
            for class_id in self.equivalence_class_atlas.seriation_order
            for mode_id in classes[class_id].member_mode_ids
        )

    @property
    def symmetric_distance(self) -> tuple[tuple[float, ...], ...]:
        class_axis = self.equivalence_class_atlas.analysis.mode_axis
        class_index = {
            class_id: index for index, class_id in enumerate(class_axis)
        }
        class_by_member = {
            mode_id: item.class_id
            for item in self.equivalence_classes
            for mode_id in item.member_mode_ids
        }
        return tuple(
            tuple(
                float(
                    self.equivalence_class_atlas.symmetric_distance[
                        class_index[class_by_member[left]]
                    ][class_index[class_by_member[right]]]
                )
                for right in self.mode_ids
            )
            for left in self.mode_ids
        )

    def canonical_payload(self) -> dict[str, object]:
        mode_count = len(self.mode_ids)
        return {
            "analysis": {
                "equivalence_class_axis": list(
                    self.equivalence_class_atlas.analysis.mode_axis
                ),
                "equivalence_class_seriation_order": list(
                    self.equivalence_class_atlas.seriation_order
                ),
                "logical_mode_axis": list(self.mode_ids),
                "logical_seriation_order": list(self.seriation_order),
                "logical_symmetric_distance": [
                    list(row) for row in self.symmetric_distance
                ],
                "nearest_neighbor_claims_emitted": False,
                "spectral_fit_unit": "exact_response_equivalence_class",
                "stability_unit": "exact_response_equivalence_class",
            },
            "comparison_role": (
                "common_downstream_response_geometry_descriptive_only"
            ),
            "compatibility_mode_abi": {
                **self.response_abi.compatibility_mode_abi().canonical_payload(),
                "sha256": self.response_abi.compatibility_mode_abi().sha256,
            },
            "equivalence_class_atlas": (
                self.equivalence_class_atlas.canonical_payload()
            ),
            "equivalence_classes": [
                item.canonical_payload() for item in self.equivalence_classes
            ],
            "equivalence_policy": {
                "collapse_before_spectral_lofo_and_bootstrap": True,
                "exact_zero_required_without_numerical_jitter": True,
                "expanded_members_are_logically_distinct": True,
                "member_order_tie_break": (
                    "operator_role_then_state_fingerprint_then_mode_id_v1"
                ),
                "native_abi_equivalence_implied": False,
                "rejects_non_equivalence_class_degeneracy": True,
            },
            "family_pair_stats": [
                item.canonical_payload() for item in self.family_pair_stats
            ],
            "family_pair_stats_count": len(self.family_pair_stats),
            "example_pair_stats": [
                item.canonical_payload() for item in self.example_pair_stats
            ],
            "example_pair_stats_count": len(self.example_pair_stats),
            "mode_count": mode_count,
            "mode_ids": list(self.mode_ids),
            "measurement_input_sha256": self.measurement_input_sha256,
            "native_operator_interchangeability_implied": False,
            "response_abi": self.response_abi.canonical_payload(),
            "unordered_mode_pair_count": mode_count * (mode_count - 1) // 2,
        }


@dataclass(frozen=True, slots=True)
class MultiABIModeAtlasReceipt:
    """Authenticated native partitions and heterogeneous response geometry."""

    inventory: MultiABIAtlasInventory
    modes: tuple[TypedModeSpec, ...]
    native_partitions: tuple[NativeABIPartitionReceipt, ...]
    downstream_response: DownstreamResponseAtlasReceipt
    receipt_sha256: str

    def _payload_without_receipt_sha256(self) -> dict[str, object]:
        return {
            "downstream_response": self.downstream_response.canonical_payload(),
            "format_version": MULTI_ABI_MODE_ATLAS_FORMAT_VERSION,
            "inventory": self.inventory.canonical_payload(),
            "mode_specs": [mode.canonical_payload() for mode in self.modes],
            "native_partitions": [
                partition.canonical_payload()
                for partition in self.native_partitions
            ],
            "safety": dict(_SAFETY),
            "schema": MULTI_ABI_MODE_ATLAS_SCHEMA,
            "scientific_status": dict(_SCIENTIFIC_STATUS),
            "terminal_norm_policy": dict(_TERMINAL_NORM_POLICY),
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_receipt_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


def _close(left: float, right: float, *, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _validate_pair_quadratic(
    row: DownstreamResponsePairStats,
    *,
    tolerance: float,
) -> None:
    expected_sse = (
        float(row.weighted_mode_a_energy)
        + float(row.weighted_mode_b_energy)
        - 2.0 * float(row.weighted_dot)
    )
    scale = max(
        1.0,
        float(row.weighted_sse),
        float(row.weighted_mode_a_energy),
        float(row.weighted_mode_b_energy),
        abs(float(row.weighted_dot)),
    )
    if abs(float(row.weighted_sse) - expected_sse) > tolerance * scale:
        raise ValueError(
            "response weighted_sse is inconsistent with pair quadratics"
        )
    bound = math.sqrt(
        float(row.weighted_mode_a_energy)
        * float(row.weighted_mode_b_energy)
    )
    if abs(float(row.weighted_dot)) > bound + tolerance * scale:
        raise ValueError("response weighted_dot violates the Cauchy bound")


def _canonical_pair(
    row: DownstreamResponsePairStats,
    *,
    mode_order_keys: Mapping[str, tuple[str, str, str]],
) -> DownstreamResponsePairStats:
    if mode_order_keys[row.mode_a_id] < mode_order_keys[row.mode_b_id]:
        return row
    return DownstreamResponsePairStats(
        family_id=row.family_id,
        mode_a_id=row.mode_b_id,
        mode_b_id=row.mode_a_id,
        response_abi_sha256=row.response_abi_sha256,
        weighted_sse=row.weighted_sse,
        weighted_mode_a_energy=row.weighted_mode_b_energy,
        weighted_dot=row.weighted_dot,
        weighted_mode_b_energy=row.weighted_mode_a_energy,
        row_count=row.row_count,
    )


def _canonical_example_pair(
    row: DownstreamResponseExamplePairStats,
    *,
    mode_order_keys: Mapping[str, tuple[str, str, str]],
) -> DownstreamResponseExamplePairStats:
    pair = _canonical_pair(
        row.as_pair_stats(),
        mode_order_keys=mode_order_keys,
    )
    return DownstreamResponseExamplePairStats(
        family_id=pair.family_id,
        example_cluster_id=row.example_cluster_id,
        mode_a_id=pair.mode_a_id,
        mode_b_id=pair.mode_b_id,
        response_abi_sha256=pair.response_abi_sha256,
        weighted_sse=pair.weighted_sse,
        weighted_mode_a_energy=pair.weighted_mode_a_energy,
        weighted_dot=pair.weighted_dot,
        weighted_mode_b_energy=pair.weighted_mode_b_energy,
        row_count=pair.row_count,
    )


def _expand_pair_stats(
    rows: Sequence[DownstreamResponsePairStats],
    *,
    compatibility_abi_sha256: str,
) -> tuple[DirectedFamilyStats, ...]:
    directed: list[DirectedFamilyStats] = []
    for row in rows:
        directed.extend(
            (
                DirectedFamilyStats(
                    family_id=row.family_id,
                    reference_mode_id=row.mode_a_id,
                    candidate_mode_id=row.mode_b_id,
                    compatibility_abi_sha256=compatibility_abi_sha256,
                    weighted_sse=row.weighted_sse,
                    weighted_reference_energy=row.weighted_mode_a_energy,
                    weighted_dot=row.weighted_dot,
                    weighted_candidate_energy=row.weighted_mode_b_energy,
                    row_count=row.row_count,
                ),
                DirectedFamilyStats(
                    family_id=row.family_id,
                    reference_mode_id=row.mode_b_id,
                    candidate_mode_id=row.mode_a_id,
                    compatibility_abi_sha256=compatibility_abi_sha256,
                    weighted_sse=row.weighted_sse,
                    weighted_reference_energy=row.weighted_mode_b_energy,
                    weighted_dot=row.weighted_dot,
                    weighted_candidate_energy=row.weighted_mode_a_energy,
                    row_count=row.row_count,
                ),
            )
        )
    return tuple(directed)


def _expand_example_pair_stats(
    rows: Sequence[DownstreamResponseExamplePairStats],
    *,
    compatibility_abi_sha256: str,
) -> tuple[DirectedFamilyExampleStats, ...]:
    directed: list[DirectedFamilyExampleStats] = []
    for row in rows:
        directed.extend(
            (
                DirectedFamilyExampleStats(
                    family_id=row.family_id,
                    example_cluster_id=row.example_cluster_id,
                    reference_mode_id=row.mode_a_id,
                    candidate_mode_id=row.mode_b_id,
                    compatibility_abi_sha256=compatibility_abi_sha256,
                    weighted_sse=row.weighted_sse,
                    weighted_reference_energy=row.weighted_mode_a_energy,
                    weighted_dot=row.weighted_dot,
                    weighted_candidate_energy=row.weighted_mode_b_energy,
                    row_count=row.row_count,
                ),
                DirectedFamilyExampleStats(
                    family_id=row.family_id,
                    example_cluster_id=row.example_cluster_id,
                    reference_mode_id=row.mode_b_id,
                    candidate_mode_id=row.mode_a_id,
                    compatibility_abi_sha256=compatibility_abi_sha256,
                    weighted_sse=row.weighted_sse,
                    weighted_reference_energy=row.weighted_mode_b_energy,
                    weighted_dot=row.weighted_dot,
                    weighted_candidate_energy=row.weighted_mode_a_energy,
                    row_count=row.row_count,
                ),
            )
        )
    return tuple(directed)


def _validated_response_pairs(
    response_input: DownstreamResponseAtlasInput,
    *,
    modes_by_id: Mapping[str, TypedModeSpec],
    mode_order_keys: Mapping[str, tuple[str, str, str]],
    config: AtlasConfig,
) -> tuple[
    tuple[DownstreamResponsePairStats, ...],
    tuple[DownstreamResponseExamplePairStats, ...],
]:
    mode_ids = set(response_input.mode_ids)
    expected_mode_ids = {
        mode.mode_id
        for mode in modes_by_id.values()
        if mode.operator_kind is not ModeOperatorKind.TERMINAL_NORM
    }
    if mode_ids != expected_mode_ids:
        raise ValueError(
            "response mode_ids must contain every non-terminal typed mode"
        )
    if any(
        modes_by_id[mode_id].operator_kind is ModeOperatorKind.TERMINAL_NORM
        for mode_id in mode_ids
    ):
        raise ValueError("terminal norm cannot enter the generator response atlas")

    pair_rows: list[DownstreamResponsePairStats] = []
    pair_by_key: dict[
        tuple[str, frozenset[str]], DownstreamResponsePairStats
    ] = {}
    families: set[str] = set()
    response_sha256 = response_input.response_abi.sha256
    energy_invariants: dict[tuple[str, str], tuple[int, float]] = {}
    for raw_row in response_input.family_pair_stats:
        if not isinstance(raw_row, DownstreamResponsePairStats):
            raise TypeError(
                "family_pair_stats must contain DownstreamResponsePairStats"
            )
        if raw_row.mode_a_id not in mode_ids or raw_row.mode_b_id not in mode_ids:
            raise ValueError("response pair references an unknown mode")
        if raw_row.response_abi_sha256 != response_sha256:
            raise ValueError("response pair ABI does not match response_abi")
        _validate_pair_quadratic(
            raw_row,
            tolerance=config.sufficient_stat_relative_tolerance,
        )
        row = _canonical_pair(raw_row, mode_order_keys=mode_order_keys)
        key = (row.family_id, frozenset((row.mode_a_id, row.mode_b_id)))
        if key in pair_by_key:
            raise ValueError("duplicate downstream response family pair")
        pair_by_key[key] = row
        pair_rows.append(row)
        families.add(row.family_id)
        for mode_id, energy in (
            (row.mode_a_id, float(row.weighted_mode_a_energy)),
            (row.mode_b_id, float(row.weighted_mode_b_energy)),
        ):
            invariant_key = (row.family_id, mode_id)
            current = (row.row_count, energy)
            previous = energy_invariants.get(invariant_key)
            if previous is None:
                energy_invariants[invariant_key] = current
            elif previous[0] != current[0] or not _close(
                previous[1],
                current[1],
                tolerance=config.sufficient_stat_relative_tolerance,
            ):
                raise ValueError(
                    "response energy and row_count must be pair-invariant"
                )
    ordered_mode_ids = tuple(
        sorted(mode_ids, key=lambda mode_id: mode_order_keys[mode_id])
    )
    expected_pairs = {
        (family_id, frozenset((mode_a_id, mode_b_id)))
        for family_id in families
        for mode_a_index, mode_a_id in enumerate(ordered_mode_ids)
        for mode_b_id in ordered_mode_ids[mode_a_index + 1 :]
    }
    if set(pair_by_key) != expected_pairs:
        raise ValueError(
            "response family statistics must form a complete unordered matrix"
        )

    example_rows: list[DownstreamResponseExamplePairStats] = []
    example_by_key: dict[
        tuple[str, str, frozenset[str]],
        DownstreamResponseExamplePairStats,
    ] = {}
    clusters_by_family: dict[str, set[str]] = {
        family_id: set() for family_id in families
    }
    example_energy_invariants: dict[
        tuple[str, str, str], tuple[int, float]
    ] = {}
    for raw_row in response_input.example_pair_stats:
        if not isinstance(raw_row, DownstreamResponseExamplePairStats):
            raise TypeError(
                "example_pair_stats must contain "
                "DownstreamResponseExamplePairStats"
            )
        if raw_row.family_id not in families:
            raise ValueError("response example pair references an unknown family")
        if raw_row.mode_a_id not in mode_ids or raw_row.mode_b_id not in mode_ids:
            raise ValueError("response example pair references an unknown mode")
        if raw_row.response_abi_sha256 != response_sha256:
            raise ValueError("response example pair ABI does not match response_abi")
        _validate_pair_quadratic(
            raw_row.as_pair_stats(),
            tolerance=config.sufficient_stat_relative_tolerance,
        )
        row = _canonical_example_pair(
            raw_row,
            mode_order_keys=mode_order_keys,
        )
        key = (
            row.family_id,
            row.example_cluster_id,
            frozenset((row.mode_a_id, row.mode_b_id)),
        )
        if key in example_by_key:
            raise ValueError("duplicate downstream response example pair")
        example_by_key[key] = row
        example_rows.append(row)
        clusters_by_family[row.family_id].add(row.example_cluster_id)
        for mode_id, energy in (
            (row.mode_a_id, float(row.weighted_mode_a_energy)),
            (row.mode_b_id, float(row.weighted_mode_b_energy)),
        ):
            invariant_key = (
                row.family_id,
                row.example_cluster_id,
                mode_id,
            )
            current = (row.row_count, energy)
            previous = example_energy_invariants.get(invariant_key)
            if previous is None:
                example_energy_invariants[invariant_key] = current
            elif previous[0] != current[0] or not _close(
                previous[1],
                current[1],
                tolerance=config.sufficient_stat_relative_tolerance,
            ):
                raise ValueError(
                    "response example energy and row_count must be pair-invariant"
                )

    if example_rows:
        if any(not value for value in clusters_by_family.values()):
            raise ValueError("every response family must have example pairs")
        expected_example_keys = {
            (family_id, cluster_id, pair)
            for family_id, cluster_ids in clusters_by_family.items()
            for cluster_id in cluster_ids
            for candidate_family_id, pair in pair_by_key
            if candidate_family_id == family_id
        }
        if set(example_by_key) != expected_example_keys:
            raise ValueError(
                "response example statistics must form a complete matrix"
            )
        for aggregate_key, aggregate in pair_by_key.items():
            family_id, pair = aggregate_key
            components = [
                row
                for (candidate_family, _, candidate_pair), row
                in example_by_key.items()
                if candidate_family == family_id and candidate_pair == pair
            ]
            if sum(row.row_count for row in components) != aggregate.row_count:
                raise ValueError(
                    "response family row_count does not equal example sum"
                )
            for field in (
                "weighted_sse",
                "weighted_mode_a_energy",
                "weighted_dot",
                "weighted_mode_b_energy",
            ):
                component_sum = math.fsum(
                    float(getattr(row, field)) for row in components
                )
                if not _close(
                    component_sum,
                    float(getattr(aggregate, field)),
                    tolerance=config.sufficient_stat_relative_tolerance,
                ):
                    raise ValueError(
                        f"response family {field} does not equal example sum"
                    )
    elif response_input.within_family_bootstrap_config is not None:
        raise ValueError(
            "within-family response bootstrap requires example pair statistics"
        )

    pair_rows.sort(
        key=lambda row: (
            row.family_id,
            mode_order_keys[row.mode_a_id],
            mode_order_keys[row.mode_b_id],
        )
    )
    example_rows.sort(
        key=lambda row: (
            row.family_id,
            row.example_cluster_id,
            mode_order_keys[row.mode_a_id],
            mode_order_keys[row.mode_b_id],
        )
    )
    return tuple(pair_rows), tuple(example_rows)


def _oriented_pair_for_modes(
    row: DownstreamResponsePairStats,
    *,
    left_mode_id: str,
    right_mode_id: str,
) -> DownstreamResponsePairStats:
    if row.mode_a_id == left_mode_id and row.mode_b_id == right_mode_id:
        return row
    if row.mode_a_id == right_mode_id and row.mode_b_id == left_mode_id:
        return DownstreamResponsePairStats(
            family_id=row.family_id,
            mode_a_id=left_mode_id,
            mode_b_id=right_mode_id,
            response_abi_sha256=row.response_abi_sha256,
            weighted_sse=row.weighted_sse,
            weighted_mode_a_energy=row.weighted_mode_b_energy,
            weighted_dot=row.weighted_dot,
            weighted_mode_b_energy=row.weighted_mode_a_energy,
            row_count=row.row_count,
        )
    raise ValueError("response pair does not contain the requested modes")


def _oriented_example_pair_for_modes(
    row: DownstreamResponseExamplePairStats,
    *,
    left_mode_id: str,
    right_mode_id: str,
) -> DownstreamResponseExamplePairStats:
    pair = _oriented_pair_for_modes(
        row.as_pair_stats(),
        left_mode_id=left_mode_id,
        right_mode_id=right_mode_id,
    )
    return DownstreamResponseExamplePairStats(
        family_id=pair.family_id,
        example_cluster_id=row.example_cluster_id,
        mode_a_id=pair.mode_a_id,
        mode_b_id=pair.mode_b_id,
        response_abi_sha256=pair.response_abi_sha256,
        weighted_sse=pair.weighted_sse,
        weighted_mode_a_energy=pair.weighted_mode_a_energy,
        weighted_dot=pair.weighted_dot,
        weighted_mode_b_energy=pair.weighted_mode_b_energy,
        row_count=pair.row_count,
    )


def _pair_stats_close(
    left: DownstreamResponsePairStats,
    right: DownstreamResponsePairStats,
    *,
    tolerance: float,
) -> bool:
    return (
        left.row_count == right.row_count
        and all(
            _close(
                float(getattr(left, field)),
                float(getattr(right, field)),
                tolerance=tolerance,
            )
            for field in (
                "weighted_sse",
                "weighted_mode_a_energy",
                "weighted_dot",
                "weighted_mode_b_energy",
            )
        )
    )


def _is_exact_identity_pair(row: DownstreamResponsePairStats) -> bool:
    energy = float(row.weighted_mode_a_energy)
    return (
        float(row.weighted_sse) == 0.0
        and float(row.weighted_mode_b_energy) == energy
        and float(row.weighted_dot) == energy
    )


def _response_equivalence_and_collapsed_stats(
    *,
    response_abi: DownstreamResponseABI,
    response_mode_ids: tuple[str, ...],
    modes_by_id: Mapping[str, TypedModeSpec],
    pair_rows: tuple[DownstreamResponsePairStats, ...],
    example_pair_rows: tuple[DownstreamResponseExamplePairStats, ...],
    config: AtlasConfig,
) -> tuple[
    tuple[DownstreamResponseEquivalenceClass, ...],
    tuple[ModeSpec, ...],
    tuple[DownstreamResponsePairStats, ...],
    tuple[DownstreamResponseExamplePairStats, ...],
]:
    """Collapse only response pairs whose SSE is exactly zero everywhere."""

    family_ids = tuple(sorted({row.family_id for row in pair_rows}))
    pair_by_key = {
        (row.family_id, frozenset((row.mode_a_id, row.mode_b_id))): row
        for row in pair_rows
    }
    example_by_key = {
        (
            row.family_id,
            row.example_cluster_id,
            frozenset((row.mode_a_id, row.mode_b_id)),
        ): row
        for row in example_pair_rows
    }
    example_keys_by_family: dict[str, tuple[str, ...]] = {
        family_id: tuple(
            sorted(
                {
                    row.example_cluster_id
                    for row in example_pair_rows
                    if row.family_id == family_id
                }
            )
        )
        for family_id in family_ids
    }

    parent = {mode_id: mode_id for mode_id in response_mode_ids}

    def find(mode_id: str) -> str:
        while parent[mode_id] != mode_id:
            parent[mode_id] = parent[parent[mode_id]]
            mode_id = parent[mode_id]
        return mode_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        left_key = (
            modes_by_id[left_root].operator_kind.value,
            modes_by_id[left_root].state_fingerprint_sha256,
            left_root,
        )
        right_key = (
            modes_by_id[right_root].operator_kind.value,
            modes_by_id[right_root].state_fingerprint_sha256,
            right_root,
        )
        if right_key < left_key:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for left_index, left_mode_id in enumerate(response_mode_ids):
        for right_mode_id in response_mode_ids[left_index + 1 :]:
            mode_pair = frozenset((left_mode_id, right_mode_id))
            exact_zero = all(
                _is_exact_identity_pair(pair_by_key[(family_id, mode_pair)])
                for family_id in family_ids
            )
            if exact_zero and example_pair_rows:
                exact_zero = all(
                    _is_exact_identity_pair(
                        example_by_key[
                            (family_id, cluster_id, mode_pair)
                        ].as_pair_stats()
                    )
                    for family_id in family_ids
                    for cluster_id in example_keys_by_family[family_id]
                )
            if exact_zero:
                union(left_mode_id, right_mode_id)

    components: dict[str, list[str]] = {}
    for mode_id in response_mode_ids:
        components.setdefault(find(mode_id), []).append(mode_id)
    member_key = lambda mode_id: (
        modes_by_id[mode_id].operator_kind.value,
        modes_by_id[mode_id].state_fingerprint_sha256,
        mode_id,
    )
    classes: list[DownstreamResponseEquivalenceClass] = []
    for members in components.values():
        member_ids = tuple(sorted(members, key=member_key))
        seed = {
            "member_operator_roles_and_state_fingerprints": [
                {
                    "operator_kind": modes_by_id[mode_id].operator_kind.value,
                    "state_fingerprint_sha256": (
                        modes_by_id[mode_id].state_fingerprint_sha256
                    ),
                }
                for mode_id in member_ids
            ],
            "response_abi_sha256": response_abi.sha256,
        }
        class_fingerprint = _domain_sha256(_RESPONSE_CLASS_DOMAIN, seed)
        classes.append(
            DownstreamResponseEquivalenceClass(
                class_id=f"response_class.{class_fingerprint}",
                class_state_fingerprint_sha256=class_fingerprint,
                member_mode_ids=member_ids,
                representative_mode_id=member_ids[0],
            )
        )
    classes.sort(key=lambda item: item.class_state_fingerprint_sha256)
    class_tuple = tuple(classes)
    if len(class_tuple) < 3:
        raise ValueError(
            "fewer than three non-equivalent response classes; seriation is undefined"
        )

    # A connected zero graph is accepted only when every within-class pair is
    # itself exactly zero.  This fails closed on non-transitive numerical data.
    for item in class_tuple:
        for left_index, left_mode_id in enumerate(item.member_mode_ids):
            for right_mode_id in item.member_mode_ids[left_index + 1 :]:
                mode_pair = frozenset((left_mode_id, right_mode_id))
                if any(
                    not _is_exact_identity_pair(
                        pair_by_key[(family_id, mode_pair)]
                    )
                    for family_id in family_ids
                ):
                    raise ValueError(
                        "response exact-equivalence relation is not transitive"
                    )
                if example_pair_rows and any(
                    not _is_exact_identity_pair(
                        example_by_key[
                            (family_id, cluster_id, mode_pair)
                        ].as_pair_stats()
                    )
                    for family_id in family_ids
                    for cluster_id in example_keys_by_family[family_id]
                ):
                    raise ValueError(
                        "response example equivalence is not transitive"
                    )

    class_modes = tuple(
        ModeSpec(
            mode_id=item.class_id,
            mode_kind="downstream_response_equivalence_class",
            state_fingerprint_sha256=item.class_state_fingerprint_sha256,
            compatibility_abi=response_abi.compatibility_mode_abi(),
        )
        for item in class_tuple
    )
    collapsed_pairs: list[DownstreamResponsePairStats] = []
    collapsed_examples: list[DownstreamResponseExamplePairStats] = []
    for left_index, left_class in enumerate(class_tuple):
        for right_class in class_tuple[left_index + 1 :]:
            left_id = left_class.representative_mode_id
            right_id = right_class.representative_mode_id
            for family_id in family_ids:
                representative = _oriented_pair_for_modes(
                    pair_by_key[
                        (family_id, frozenset((left_id, right_id)))
                    ],
                    left_mode_id=left_id,
                    right_mode_id=right_id,
                )
                for candidate_left in left_class.member_mode_ids:
                    for candidate_right in right_class.member_mode_ids:
                        candidate = _oriented_pair_for_modes(
                            pair_by_key[
                                (
                                    family_id,
                                    frozenset(
                                        (candidate_left, candidate_right)
                                    ),
                                )
                            ],
                            left_mode_id=candidate_left,
                            right_mode_id=candidate_right,
                        )
                        if not _pair_stats_close(
                            representative,
                            candidate,
                            tolerance=(
                                config.sufficient_stat_relative_tolerance
                            ),
                        ):
                            raise ValueError(
                                "response-equivalent members disagree against "
                                "another response class"
                            )
                collapsed_pairs.append(
                    DownstreamResponsePairStats(
                        family_id=family_id,
                        mode_a_id=left_class.class_id,
                        mode_b_id=right_class.class_id,
                        response_abi_sha256=response_abi.sha256,
                        weighted_sse=representative.weighted_sse,
                        weighted_mode_a_energy=(
                            representative.weighted_mode_a_energy
                        ),
                        weighted_dot=representative.weighted_dot,
                        weighted_mode_b_energy=(
                            representative.weighted_mode_b_energy
                        ),
                        row_count=representative.row_count,
                    )
                )
                for cluster_id in example_keys_by_family[family_id]:
                    representative_example = _oriented_example_pair_for_modes(
                        example_by_key[
                            (
                                family_id,
                                cluster_id,
                                frozenset((left_id, right_id)),
                            )
                        ],
                        left_mode_id=left_id,
                        right_mode_id=right_id,
                    )
                    for candidate_left in left_class.member_mode_ids:
                        for candidate_right in right_class.member_mode_ids:
                            candidate_example = (
                                _oriented_example_pair_for_modes(
                                    example_by_key[
                                        (
                                            family_id,
                                            cluster_id,
                                            frozenset(
                                                (
                                                    candidate_left,
                                                    candidate_right,
                                                )
                                            ),
                                        )
                                    ],
                                    left_mode_id=candidate_left,
                                    right_mode_id=candidate_right,
                                )
                            )
                            if not _pair_stats_close(
                                representative_example.as_pair_stats(),
                                candidate_example.as_pair_stats(),
                                tolerance=(
                                    config.sufficient_stat_relative_tolerance
                                ),
                            ):
                                raise ValueError(
                                    "response-equivalent example members "
                                    "disagree against another response class"
                                )
                    collapsed_examples.append(
                        DownstreamResponseExamplePairStats(
                            family_id=family_id,
                            example_cluster_id=cluster_id,
                            mode_a_id=left_class.class_id,
                            mode_b_id=right_class.class_id,
                            response_abi_sha256=response_abi.sha256,
                            weighted_sse=representative_example.weighted_sse,
                            weighted_mode_a_energy=(
                                representative_example.weighted_mode_a_energy
                            ),
                            weighted_dot=representative_example.weighted_dot,
                            weighted_mode_b_energy=(
                                representative_example.weighted_mode_b_energy
                            ),
                            row_count=representative_example.row_count,
                        )
                    )
    return (
        class_tuple,
        class_modes,
        tuple(collapsed_pairs),
        tuple(collapsed_examples),
    )


def _authenticated_prebuilt_receipt(
    receipt: ModeSimilarityAtlasReceipt,
) -> ModeSimilarityAtlasReceipt:
    return parse_mode_similarity_atlas_receipt_payload(
        receipt.canonical_payload()
    )


def build_multi_abi_mode_atlas_receipt(
    inventory: MultiABIAtlasInventory,
    modes: Sequence[TypedModeSpec],
    *,
    native_atlas_inputs: Sequence[NativeAtlasInput],
    downstream_response_input: DownstreamResponseAtlasInput,
) -> MultiABIModeAtlasReceipt:
    """Build exact native partitions and one heterogeneous response atlas."""

    if not isinstance(inventory, MultiABIAtlasInventory):
        raise TypeError("inventory must be a MultiABIAtlasInventory")
    mode_tuple = tuple(modes)
    if any(not isinstance(mode, TypedModeSpec) for mode in mode_tuple):
        raise TypeError("modes must contain only TypedModeSpec values")
    if len(mode_tuple) != inventory.total_typed_mode_count:
        raise ValueError("typed mode count does not match inventory")
    mode_ids = [mode.mode_id for mode in mode_tuple]
    if len(set(mode_ids)) != len(mode_ids):
        raise ValueError("typed mode IDs must be unique")
    _validate_typed_mode_state_reuse(mode_tuple)
    mode_tuple = tuple(
        sorted(mode_tuple, key=_typed_mode_sort_key)
    )
    modes_by_id = {mode.mode_id: mode for mode in mode_tuple}
    mode_order_keys = {
        mode.mode_id: _typed_mode_sort_key(mode) for mode in mode_tuple
    }

    expected_counts = {
        ModeOperatorKind.ATTENTION: inventory.attention_mode_count,
        ModeOperatorKind.MLP: inventory.mlp_mode_count,
        ModeOperatorKind.LEXICAL_INGRESS: 1,
        ModeOperatorKind.LEXICAL_EGRESS: 1,
        ModeOperatorKind.TERMINAL_NORM: 1,
    }
    actual_counts = {
        kind: sum(mode.operator_kind is kind for mode in mode_tuple)
        for kind in ModeOperatorKind
    }
    if actual_counts != expected_counts:
        raise ValueError("typed operator inventory does not match contract")

    groups: dict[
        tuple[ModeOperatorKind, str], list[TypedModeSpec]
    ] = {}
    abi_owner: dict[str, ModeOperatorKind] = {}
    for mode in mode_tuple:
        abi_sha256 = mode.native_abi.sha256
        previous_owner = abi_owner.setdefault(abi_sha256, mode.operator_kind)
        if previous_owner is not mode.operator_kind:
            raise ValueError(
                "one native ABI cannot be shared across operator kinds"
            )
        groups.setdefault((mode.operator_kind, abi_sha256), []).append(mode)
    if any(
        sum(kind is candidate_kind for kind, _ in groups) != 1
        for candidate_kind in ModeOperatorKind
    ):
        raise ValueError("each operator kind requires exactly one native ABI")

    input_tuple = tuple(native_atlas_inputs)
    if any(not isinstance(item, NativeAtlasInput) for item in input_tuple):
        raise TypeError(
            "native_atlas_inputs must contain only NativeAtlasInput values"
        )
    if len({item.compatibility_abi_sha256 for item in input_tuple}) != len(
        input_tuple
    ):
        raise ValueError("native atlas input ABIs must be unique")
    input_by_abi = {
        item.compatibility_abi_sha256: item for item in input_tuple
    }
    expected_input_abis = {
        abi_sha256
        for (kind, abi_sha256), _ in groups.items()
        if kind in {ModeOperatorKind.ATTENTION, ModeOperatorKind.MLP}
    }
    if set(input_by_abi) != expected_input_abis:
        raise ValueError(
            "native inputs must cover exactly the attention and MLP ABIs"
        )

    native_partitions: list[NativeABIPartitionReceipt] = []
    for (operator_kind, abi_sha256), grouped_modes in sorted(
        groups.items(),
        key=lambda item: (item[0][0].value, item[0][1]),
    ):
        grouped_modes.sort(key=_typed_mode_sort_key)
        grouped_mode_ids = tuple(mode.mode_id for mode in grouped_modes)
        if operator_kind is ModeOperatorKind.TERMINAL_NORM:
            native_partitions.append(
                NativeABIPartitionReceipt(
                    operator_kind=operator_kind,
                    compatibility_abi_sha256=abi_sha256,
                    mode_ids=grouped_mode_ids,
                    status=(
                        NativePartitionStatus.EXACT_BASELINE_INFRASTRUCTURE
                    ),
                    receipt_origin=None,
                    atlas=None,
                )
            )
            continue
        if operator_kind in {
            ModeOperatorKind.LEXICAL_INGRESS,
            ModeOperatorKind.LEXICAL_EGRESS,
        }:
            native_partitions.append(
                NativeABIPartitionReceipt(
                    operator_kind=operator_kind,
                    compatibility_abi_sha256=abi_sha256,
                    mode_ids=grouped_mode_ids,
                    status=(
                        NativePartitionStatus
                        .INSUFFICIENT_MODES_FOR_SERIATION
                    ),
                    receipt_origin=None,
                    atlas=None,
                )
            )
            continue

        native_input = input_by_abi[abi_sha256]
        if operator_kind is ModeOperatorKind.MLP:
            if native_input.prebuilt_receipt is None:
                raise ValueError(
                    "MLP native partition requires an existing receipt"
                )
            native_atlas = _authenticated_prebuilt_receipt(
                native_input.prebuilt_receipt
            )
            origin = (
                NativePartitionReceiptOrigin
                .AUTHENTICATED_EXISTING_RECEIPT
            )
        else:
            if native_input.prebuilt_receipt is not None:
                raise ValueError(
                    "attention native partition requires newly supplied stats"
                )
            for row in native_input.directed_stats:
                if (
                    row.compatibility_abi_sha256 != abi_sha256
                    or row.reference_mode_id not in grouped_mode_ids
                    or row.candidate_mode_id not in grouped_mode_ids
                ):
                    raise ValueError(
                        "attention native statistics crossed an ABI partition"
                    )
            for row in native_input.directed_example_stats:
                if (
                    row.compatibility_abi_sha256 != abi_sha256
                    or row.reference_mode_id not in grouped_mode_ids
                    or row.candidate_mode_id not in grouped_mode_ids
                ):
                    raise ValueError(
                        "attention example statistics crossed an ABI partition"
                    )
            native_atlas = build_mode_similarity_atlas_receipt(
                tuple(mode.native_mode_spec() for mode in grouped_modes),
                native_input.directed_stats,
                config=native_input.config,
                frozen_resamples=native_input.frozen_resamples,
                directed_example_stats=native_input.directed_example_stats,
                within_family_bootstrap_config=(
                    native_input.within_family_bootstrap_config
                ),
            )
            origin = (
                NativePartitionReceiptOrigin.BUILT_FROM_SUPPLIED_STATISTICS
            )
        expected_native_modes = [
            mode.native_mode_spec().canonical_payload()
            for mode in grouped_modes
        ]
        actual_native_modes = [
            mode.canonical_payload() for mode in native_atlas.modes
        ]
        if not _json_exact_equal(expected_native_modes, actual_native_modes):
            raise ValueError(
                "native atlas modes do not match the typed ABI partition"
            )
        if native_atlas.compatibility_abi_sha256 != abi_sha256:
            raise ValueError("native atlas ABI does not match its partition")
        native_partitions.append(
            NativeABIPartitionReceipt(
                operator_kind=operator_kind,
                compatibility_abi_sha256=abi_sha256,
                mode_ids=grouped_mode_ids,
                status=NativePartitionStatus.ATLAS_BUILT,
                receipt_origin=origin,
                atlas=native_atlas,
            )
        )

    if not isinstance(
        downstream_response_input,
        DownstreamResponseAtlasInput,
    ):
        raise TypeError(
            "downstream_response_input must be a DownstreamResponseAtlasInput"
        )
    if (
        len(downstream_response_input.mode_ids)
        != inventory.downstream_response_mode_count
    ):
        raise ValueError("downstream response mode count does not match inventory")
    response_config = downstream_response_input.config or AtlasConfig()
    pair_rows, example_pair_rows = _validated_response_pairs(
        downstream_response_input,
        modes_by_id=modes_by_id,
        mode_order_keys=mode_order_keys,
        config=response_config,
    )
    response_mode_ids = tuple(
        sorted(
            downstream_response_input.mode_ids,
            key=lambda mode_id: mode_order_keys[mode_id],
        )
    )
    (
        equivalence_classes,
        equivalence_class_modes,
        collapsed_pair_rows,
        collapsed_example_rows,
    ) = _response_equivalence_and_collapsed_stats(
        response_abi=downstream_response_input.response_abi,
        response_mode_ids=response_mode_ids,
        modes_by_id=modes_by_id,
        pair_rows=pair_rows,
        example_pair_rows=example_pair_rows,
        config=response_config,
    )
    response_compatibility_abi = (
        downstream_response_input.response_abi.compatibility_mode_abi()
    )
    response_atlas = build_mode_similarity_atlas_receipt(
        equivalence_class_modes,
        _expand_pair_stats(
            collapsed_pair_rows,
            compatibility_abi_sha256=response_compatibility_abi.sha256,
        ),
        config=response_config,
        frozen_resamples=downstream_response_input.frozen_resamples,
        directed_example_stats=_expand_example_pair_stats(
            collapsed_example_rows,
            compatibility_abi_sha256=response_compatibility_abi.sha256,
        ),
        within_family_bootstrap_config=(
            downstream_response_input.within_family_bootstrap_config
        ),
    )
    downstream_receipt = DownstreamResponseAtlasReceipt(
        response_abi=downstream_response_input.response_abi,
        measurement_input_sha256=(
            downstream_response_input.measurement_input_sha256
        ),
        mode_ids=response_mode_ids,
        equivalence_classes=equivalence_classes,
        family_pair_stats=pair_rows,
        example_pair_stats=example_pair_rows,
        equivalence_class_atlas=response_atlas,
    )
    provisional = MultiABIModeAtlasReceipt(
        inventory=inventory,
        modes=mode_tuple,
        native_partitions=tuple(native_partitions),
        downstream_response=downstream_receipt,
        receipt_sha256="0" * 64,
    )
    _validate_multi_receipt(provisional, check_digest=False)
    digest = _domain_sha256(
        _RECEIPT_DOMAIN,
        provisional._payload_without_receipt_sha256(),
    )
    return MultiABIModeAtlasReceipt(
        inventory=inventory,
        modes=mode_tuple,
        native_partitions=tuple(native_partitions),
        downstream_response=downstream_receipt,
        receipt_sha256=digest,
    )


def _validate_mirrored_response_rows(
    atlas: ModeSimilarityAtlasReceipt,
) -> None:
    family_rows = {
        (row.family_id, row.reference_mode_id, row.candidate_mode_id): row
        for row in atlas.directed_stats
    }
    for row in atlas.directed_stats:
        reverse = family_rows[
            (row.family_id, row.candidate_mode_id, row.reference_mode_id)
        ]
        if (
            row.row_count != reverse.row_count
            or float(row.weighted_sse) != float(reverse.weighted_sse)
            or float(row.weighted_dot) != float(reverse.weighted_dot)
            or float(row.weighted_reference_energy)
            != float(reverse.weighted_candidate_energy)
            or float(row.weighted_candidate_energy)
            != float(reverse.weighted_reference_energy)
        ):
            raise ValueError(
                "downstream response family rows are not mirrored pair stats"
            )
    example_rows = {
        (
            row.family_id,
            row.example_cluster_id,
            row.reference_mode_id,
            row.candidate_mode_id,
        ): row
        for row in atlas.directed_example_stats
    }
    for row in atlas.directed_example_stats:
        reverse = example_rows[
            (
                row.family_id,
                row.example_cluster_id,
                row.candidate_mode_id,
                row.reference_mode_id,
            )
        ]
        if (
            row.row_count != reverse.row_count
            or float(row.weighted_sse) != float(reverse.weighted_sse)
            or float(row.weighted_dot) != float(reverse.weighted_dot)
            or float(row.weighted_reference_energy)
            != float(reverse.weighted_candidate_energy)
            or float(row.weighted_candidate_energy)
            != float(reverse.weighted_reference_energy)
        ):
            raise ValueError(
                "downstream response example rows are not mirrored pair stats"
            )


def _validate_multi_receipt(
    receipt: MultiABIModeAtlasReceipt,
    *,
    check_digest: bool,
) -> None:
    if not isinstance(receipt.inventory, MultiABIAtlasInventory):
        raise TypeError("receipt inventory has the wrong type")
    modes = receipt.modes
    if (
        not isinstance(modes, tuple)
        or len(modes) != receipt.inventory.total_typed_mode_count
        or any(not isinstance(mode, TypedModeSpec) for mode in modes)
        or tuple(sorted(modes, key=_typed_mode_sort_key))
        != modes
        or len({mode.mode_id for mode in modes}) != len(modes)
    ):
        raise ValueError("receipt typed mode inventory is not canonical")
    _validate_typed_mode_state_reuse(modes)
    modes_by_id = {mode.mode_id: mode for mode in modes}
    expected_counts = {
        ModeOperatorKind.ATTENTION: receipt.inventory.attention_mode_count,
        ModeOperatorKind.MLP: receipt.inventory.mlp_mode_count,
        ModeOperatorKind.LEXICAL_INGRESS: 1,
        ModeOperatorKind.LEXICAL_EGRESS: 1,
        ModeOperatorKind.TERMINAL_NORM: 1,
    }
    if {
        kind: sum(mode.operator_kind is kind for mode in modes)
        for kind in ModeOperatorKind
    } != expected_counts:
        raise ValueError("receipt operator counts diverged from inventory")
    native_abis_by_kind = {
        kind: {
            mode.native_abi.sha256
            for mode in modes
            if mode.operator_kind is kind
        }
        for kind in ModeOperatorKind
    }
    if any(len(values) != 1 for values in native_abis_by_kind.values()):
        raise ValueError("receipt operator kind crossed native ABI partitions")
    if len(
        {
            abi_sha256
            for values in native_abis_by_kind.values()
            for abi_sha256 in values
        }
    ) != len(ModeOperatorKind):
        raise ValueError("receipt native ABI is shared across operator kinds")
    partitions = receipt.native_partitions
    if (
        not isinstance(partitions, tuple)
        or len(partitions) != len(ModeOperatorKind)
        or any(
            not isinstance(partition, NativeABIPartitionReceipt)
            for partition in partitions
        )
        or tuple(
            sorted(
                partitions,
                key=lambda item: (
                    item.operator_kind.value,
                    item.compatibility_abi_sha256,
                ),
            )
        )
        != partitions
        or {partition.operator_kind for partition in partitions}
        != set(ModeOperatorKind)
    ):
        raise ValueError("receipt native partitions are not canonical")
    for partition in partitions:
        expected_modes = tuple(
            mode
            for mode in modes
            if mode.operator_kind is partition.operator_kind
        )
        if (
            partition.mode_ids
            != tuple(mode.mode_id for mode in expected_modes)
            or {mode.native_abi.sha256 for mode in expected_modes}
            != {partition.compatibility_abi_sha256}
        ):
            raise ValueError("native partition does not match typed modes")
        if partition.operator_kind is ModeOperatorKind.MLP:
            expected_status = NativePartitionStatus.ATLAS_BUILT
            expected_origin = (
                NativePartitionReceiptOrigin.AUTHENTICATED_EXISTING_RECEIPT
            )
        elif partition.operator_kind is ModeOperatorKind.ATTENTION:
            expected_status = NativePartitionStatus.ATLAS_BUILT
            expected_origin = (
                NativePartitionReceiptOrigin.BUILT_FROM_SUPPLIED_STATISTICS
            )
        elif partition.operator_kind is ModeOperatorKind.TERMINAL_NORM:
            expected_status = (
                NativePartitionStatus.EXACT_BASELINE_INFRASTRUCTURE
            )
            expected_origin = None
        else:
            expected_status = (
                NativePartitionStatus.INSUFFICIENT_MODES_FOR_SERIATION
            )
            expected_origin = None
        if (
            partition.status is not expected_status
            or partition.receipt_origin is not expected_origin
        ):
            raise ValueError("native partition status or origin drifted")
        if partition.atlas is not None:
            authenticated_native = parse_mode_similarity_atlas_receipt_payload(
                partition.atlas.canonical_payload()
            )
            if (
                partition.atlas.compatibility_abi_sha256
                != partition.compatibility_abi_sha256
                or not _json_exact_equal(
                    authenticated_native.canonical_payload(),
                    partition.atlas.canonical_payload(),
                )
                or not _json_exact_equal(
                    [
                        mode.native_mode_spec().canonical_payload()
                        for mode in expected_modes
                    ],
                    [
                        mode.canonical_payload()
                        for mode in partition.atlas.modes
                    ],
                )
            ):
                raise ValueError("nested native atlas cross-binding drifted")

    downstream = receipt.downstream_response
    if not isinstance(downstream, DownstreamResponseAtlasReceipt):
        raise TypeError("downstream_response has the wrong type")
    expected_response_modes = tuple(
        mode
        for mode in modes
        if mode.operator_kind is not ModeOperatorKind.TERMINAL_NORM
    )
    if (
        len(downstream.mode_ids)
        != receipt.inventory.downstream_response_mode_count
        or downstream.mode_ids
        != tuple(mode.mode_id for mode in expected_response_modes)
    ):
        raise ValueError("downstream response mode inventory drifted")
    response_mode_abi = downstream.response_abi.compatibility_mode_abi()
    authenticated_class_atlas = parse_mode_similarity_atlas_receipt_payload(
        downstream.equivalence_class_atlas.canonical_payload()
    )
    if not _json_exact_equal(
        authenticated_class_atlas.canonical_payload(),
        downstream.equivalence_class_atlas.canonical_payload(),
    ):
        raise ValueError("downstream equivalence-class atlas is not authentic")
    if (
        downstream.equivalence_class_atlas.compatibility_abi_sha256
        != response_mode_abi.sha256
    ):
        raise ValueError("downstream response compatibility ABI drifted")
    replay_input = DownstreamResponseAtlasInput(
        response_abi=downstream.response_abi,
        measurement_input_sha256=downstream.measurement_input_sha256,
        mode_ids=downstream.mode_ids,
        family_pair_stats=downstream.family_pair_stats,
        config=downstream.equivalence_class_atlas.config,
        frozen_resamples=downstream.equivalence_class_atlas.frozen_resamples,
        example_pair_stats=downstream.example_pair_stats,
        within_family_bootstrap_config=(
            downstream.equivalence_class_atlas
            .within_family_bootstrap_config
        ),
    )
    mode_order_keys = {
        mode.mode_id: _typed_mode_sort_key(mode) for mode in modes
    }
    replay_pairs, replay_examples = _validated_response_pairs(
        replay_input,
        modes_by_id=modes_by_id,
        mode_order_keys=mode_order_keys,
        config=downstream.equivalence_class_atlas.config,
    )
    if replay_pairs != downstream.family_pair_stats:
        raise ValueError("downstream family pair statistics are not canonical")
    if replay_examples != downstream.example_pair_stats:
        raise ValueError("downstream example pair statistics are not canonical")
    (
        expected_classes,
        expected_class_modes,
        expected_collapsed_pairs,
        expected_collapsed_examples,
    ) = _response_equivalence_and_collapsed_stats(
        response_abi=downstream.response_abi,
        response_mode_ids=downstream.mode_ids,
        modes_by_id=modes_by_id,
        pair_rows=replay_pairs,
        example_pair_rows=replay_examples,
        config=downstream.equivalence_class_atlas.config,
    )
    if expected_classes != downstream.equivalence_classes:
        raise ValueError("downstream response equivalence classes drifted")
    if expected_class_modes != downstream.equivalence_class_atlas.modes:
        raise ValueError("downstream response class modes drifted")
    expected_directed = _expand_pair_stats(
        expected_collapsed_pairs,
        compatibility_abi_sha256=response_mode_abi.sha256,
    )
    expected_example_directed = _expand_example_pair_stats(
        expected_collapsed_examples,
        compatibility_abi_sha256=response_mode_abi.sha256,
    )
    payload_sort = lambda item: _canonical_json_bytes(item.canonical_payload())
    if not _json_exact_equal(
        [
            item.canonical_payload()
            for item in sorted(expected_directed, key=payload_sort)
        ],
        [
            item.canonical_payload()
            for item in sorted(
                downstream.equivalence_class_atlas.directed_stats,
                key=payload_sort,
            )
        ],
    ):
        raise ValueError("downstream collapsed family statistics drifted")
    if not _json_exact_equal(
        [
            item.canonical_payload()
            for item in sorted(expected_example_directed, key=payload_sort)
        ],
        [
            item.canonical_payload()
            for item in sorted(
                downstream.equivalence_class_atlas.directed_example_stats,
                key=payload_sort,
            )
        ],
    ):
        raise ValueError("downstream collapsed example statistics drifted")
    if set(downstream.seriation_order) != set(downstream.mode_ids):
        raise ValueError("logical response seriation expansion drifted")
    _validate_mirrored_response_rows(downstream.equivalence_class_atlas)
    if check_digest:
        _require_sha256(receipt.receipt_sha256, label="receipt_sha256")
        expected_digest = _domain_sha256(
            _RECEIPT_DOMAIN,
            receipt._payload_without_receipt_sha256(),
        )
        if receipt.receipt_sha256 != expected_digest:
            raise ValueError("multi-ABI atlas receipt digest mismatch")


def _parse_mode_abi(value: object, *, label: str) -> ModeABI:
    raw = _require_exact_fields(
        value,
        {
            "abi_name",
            "abi_version",
            "context_signature",
            "input_signature",
            "output_signature",
            "sha256",
        },
        label=label,
    )
    abi = ModeABI(
        abi_name=_require_name(raw["abi_name"], label=f"{label}.abi_name"),
        abi_version=_require_positive_int(
            raw["abi_version"], label=f"{label}.abi_version"
        ),
        input_signature=_require_name(
            raw["input_signature"], label=f"{label}.input_signature"
        ),
        output_signature=_require_name(
            raw["output_signature"], label=f"{label}.output_signature"
        ),
        context_signature=_require_name(
            raw["context_signature"], label=f"{label}.context_signature"
        ),
    )
    if raw["sha256"] != abi.sha256:
        raise ValueError(f"{label}.sha256 does not authenticate the ABI")
    return abi


def _parse_typed_mode(value: object, *, index: int) -> TypedModeSpec:
    label = f"mode_specs[{index}]"
    raw = _require_exact_fields(
        value,
        {
            "mode_id",
            "native_abi",
            "native_mode_kind",
            "operator_kind",
            "state_fingerprint_sha256",
        },
        label=label,
    )
    try:
        operator_kind = ModeOperatorKind(raw["operator_kind"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.operator_kind is unsupported") from error
    return TypedModeSpec(
        mode_id=_require_name(raw["mode_id"], label=f"{label}.mode_id"),
        operator_kind=operator_kind,
        native_mode_kind=_require_name(
            raw["native_mode_kind"], label=f"{label}.native_mode_kind"
        ),
        state_fingerprint_sha256=_require_sha256(
            raw["state_fingerprint_sha256"],
            label=f"{label}.state_fingerprint_sha256",
        ),
        native_abi=_parse_mode_abi(
            raw["native_abi"], label=f"{label}.native_abi"
        ),
    )


def _parse_inventory(value: object) -> MultiABIAtlasInventory:
    raw = _require_exact_fields(
        value,
        {
            "downstream_response_mode_count",
            "native_partition_requirements",
            "operator_mode_counts",
            "total_typed_mode_count",
        },
        label="inventory",
    )
    counts = _require_exact_fields(
        raw["operator_mode_counts"],
        {
            "attention",
            "lexical_egress",
            "lexical_ingress",
            "mlp",
            "terminal_norm",
        },
        label="inventory.operator_mode_counts",
    )
    inventory = MultiABIAtlasInventory(
        attention_mode_count=_require_positive_int(
            counts["attention"],
            label="inventory.operator_mode_counts.attention",
        ),
        mlp_mode_count=_require_positive_int(
            counts["mlp"], label="inventory.operator_mode_counts.mlp"
        ),
    )
    if not _json_exact_equal(raw, inventory.canonical_payload()):
        raise ValueError("inventory is not the canonical exact contract")
    return inventory


def _parse_response_abi(value: object) -> DownstreamResponseABI:
    raw = _require_exact_fields(
        value,
        {
            "abi_id",
            "abi_name",
            "abi_version",
            "baseline_signature",
            "comparison_definition",
            "intervention_signature",
            "native_operator_interchangeability_implied",
            "response_coordinate_signature",
            "response_width",
            "row_catalog_sha256",
            "sha256",
            "weighting_catalog_sha256",
        },
        label="downstream_response.response_abi",
    )
    abi = DownstreamResponseABI(
        abi_name=_require_name(raw["abi_name"], label="response ABI name"),
        abi_version=_require_positive_int(
            raw["abi_version"], label="response ABI version"
        ),
        response_coordinate_signature=_require_name(
            raw["response_coordinate_signature"],
            label="response coordinate signature",
        ),
        response_width=_require_positive_int(
            raw["response_width"], label="response width"
        ),
        baseline_signature=_require_name(
            raw["baseline_signature"], label="response baseline signature"
        ),
        intervention_signature=_require_name(
            raw["intervention_signature"],
            label="response intervention signature",
        ),
        row_catalog_sha256=_require_sha256(
            raw["row_catalog_sha256"], label="response row catalog"
        ),
        weighting_catalog_sha256=_require_sha256(
            raw["weighting_catalog_sha256"],
            label="response weighting catalog",
        ),
    )
    if not _json_exact_equal(raw, abi.canonical_payload()):
        raise ValueError("downstream response ABI is not canonical")
    return abi


def _parse_response_pair(
    value: object,
    *,
    index: int,
) -> DownstreamResponsePairStats:
    label = f"downstream_response.family_pair_stats[{index}]"
    raw = _require_exact_fields(
        value,
        {
            "family_id",
            "mode_a_id",
            "mode_b_id",
            "response_abi_sha256",
            "row_count",
            "weighted_dot",
            "weighted_mode_a_energy",
            "weighted_mode_b_energy",
            "weighted_sse",
        },
        label=label,
    )
    return DownstreamResponsePairStats(
        family_id=_require_name(raw["family_id"], label=f"{label}.family_id"),
        mode_a_id=_require_name(raw["mode_a_id"], label=f"{label}.mode_a_id"),
        mode_b_id=_require_name(raw["mode_b_id"], label=f"{label}.mode_b_id"),
        response_abi_sha256=_require_sha256(
            raw["response_abi_sha256"],
            label=f"{label}.response_abi_sha256",
        ),
        weighted_sse=_require_finite(
            raw["weighted_sse"], label=f"{label}.weighted_sse", minimum=0.0
        ),
        weighted_mode_a_energy=_require_finite(
            raw["weighted_mode_a_energy"],
            label=f"{label}.weighted_mode_a_energy",
            minimum=0.0,
        ),
        weighted_dot=_require_finite(
            raw["weighted_dot"], label=f"{label}.weighted_dot"
        ),
        weighted_mode_b_energy=_require_finite(
            raw["weighted_mode_b_energy"],
            label=f"{label}.weighted_mode_b_energy",
            minimum=0.0,
        ),
        row_count=_require_positive_int(
            raw["row_count"], label=f"{label}.row_count"
        ),
    )


def _parse_response_example_pair(
    value: object,
    *,
    index: int,
) -> DownstreamResponseExamplePairStats:
    label = f"downstream_response.example_pair_stats[{index}]"
    raw = _require_exact_fields(
        value,
        {
            "example_cluster_id",
            "family_id",
            "mode_a_id",
            "mode_b_id",
            "response_abi_sha256",
            "row_count",
            "weighted_dot",
            "weighted_mode_a_energy",
            "weighted_mode_b_energy",
            "weighted_sse",
        },
        label=label,
    )
    pair = _parse_response_pair(
        {key: item for key, item in raw.items() if key != "example_cluster_id"},
        index=index,
    )
    return DownstreamResponseExamplePairStats(
        family_id=pair.family_id,
        example_cluster_id=_require_name(
            raw["example_cluster_id"],
            label=f"{label}.example_cluster_id",
        ),
        mode_a_id=pair.mode_a_id,
        mode_b_id=pair.mode_b_id,
        response_abi_sha256=pair.response_abi_sha256,
        weighted_sse=pair.weighted_sse,
        weighted_mode_a_energy=pair.weighted_mode_a_energy,
        weighted_dot=pair.weighted_dot,
        weighted_mode_b_energy=pair.weighted_mode_b_energy,
        row_count=pair.row_count,
    )


def _parse_equivalence_class(
    value: object,
    *,
    index: int,
) -> DownstreamResponseEquivalenceClass:
    label = f"downstream_response.equivalence_classes[{index}]"
    raw = _require_exact_fields(
        value,
        {
            "class_id",
            "class_state_fingerprint_sha256",
            "equivalence_definition",
            "member_count",
            "member_mode_ids",
            "native_abi_equivalence_implied",
            "representative_mode_id",
        },
        label=label,
    )
    members_raw = raw["member_mode_ids"]
    if not isinstance(members_raw, list):
        raise TypeError(f"{label}.member_mode_ids must be a list")
    item = DownstreamResponseEquivalenceClass(
        class_id=_require_name(raw["class_id"], label=f"{label}.class_id"),
        class_state_fingerprint_sha256=_require_sha256(
            raw["class_state_fingerprint_sha256"],
            label=f"{label}.class_state_fingerprint_sha256",
        ),
        member_mode_ids=tuple(
            _require_name(mode_id, label=f"{label}.member_mode_ids")
            for mode_id in members_raw
        ),
        representative_mode_id=_require_name(
            raw["representative_mode_id"],
            label=f"{label}.representative_mode_id",
        ),
    )
    if not _json_exact_equal(raw, item.canonical_payload()):
        raise ValueError(f"{label} is not canonical")
    return item


def _parse_native_partition(
    value: object,
    *,
    index: int,
) -> NativeABIPartitionReceipt:
    label = f"native_partitions[{index}]"
    raw = _require_exact_fields(
        value,
        {
            "atlas",
            "compatibility_abi_sha256",
            "mode_count",
            "mode_ids",
            "native_operator_interchangeability_only",
            "operator_kind",
            "receipt_origin",
            "status",
        },
        label=label,
    )
    mode_ids_raw = raw["mode_ids"]
    if not isinstance(mode_ids_raw, list):
        raise TypeError(f"{label}.mode_ids must be a list")
    try:
        kind = ModeOperatorKind(raw["operator_kind"])
        status_value = NativePartitionStatus(raw["status"])
        origin = (
            None
            if raw["receipt_origin"] is None
            else NativePartitionReceiptOrigin(raw["receipt_origin"])
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} enum value is unsupported") from error
    atlas_raw = raw["atlas"]
    atlas = (
        None
        if atlas_raw is None
        else parse_mode_similarity_atlas_receipt_payload(atlas_raw)
    )
    partition = NativeABIPartitionReceipt(
        operator_kind=kind,
        compatibility_abi_sha256=_require_sha256(
            raw["compatibility_abi_sha256"],
            label=f"{label}.compatibility_abi_sha256",
        ),
        mode_ids=tuple(
            _require_name(mode_id, label=f"{label}.mode_ids")
            for mode_id in mode_ids_raw
        ),
        status=status_value,
        receipt_origin=origin,
        atlas=atlas,
    )
    if not _json_exact_equal(raw, partition.canonical_payload()):
        raise ValueError(f"{label} is not canonical")
    return partition


def _parse_downstream_receipt(value: object) -> DownstreamResponseAtlasReceipt:
    raw = _require_exact_fields(
        value,
        {
            "analysis",
            "comparison_role",
            "compatibility_mode_abi",
            "equivalence_class_atlas",
            "equivalence_classes",
            "equivalence_policy",
            "example_pair_stats",
            "example_pair_stats_count",
            "family_pair_stats",
            "family_pair_stats_count",
            "measurement_input_sha256",
            "mode_count",
            "mode_ids",
            "native_operator_interchangeability_implied",
            "response_abi",
            "unordered_mode_pair_count",
        },
        label="downstream_response",
    )
    mode_ids_raw = raw["mode_ids"]
    classes_raw = raw["equivalence_classes"]
    family_raw = raw["family_pair_stats"]
    example_raw = raw["example_pair_stats"]
    for item, label in (
        (mode_ids_raw, "downstream_response.mode_ids"),
        (classes_raw, "downstream_response.equivalence_classes"),
        (family_raw, "downstream_response.family_pair_stats"),
        (example_raw, "downstream_response.example_pair_stats"),
    ):
        if not isinstance(item, list):
            raise TypeError(f"{label} must be a list")
    class_atlas_raw = raw["equivalence_class_atlas"]
    if not isinstance(class_atlas_raw, Mapping):
        raise TypeError("downstream equivalence_class_atlas must be a mapping")
    receipt = DownstreamResponseAtlasReceipt(
        response_abi=_parse_response_abi(raw["response_abi"]),
        measurement_input_sha256=_require_sha256(
            raw["measurement_input_sha256"],
            label="downstream_response.measurement_input_sha256",
        ),
        mode_ids=tuple(
            _require_name(mode_id, label="downstream_response.mode_ids")
            for mode_id in mode_ids_raw
        ),
        equivalence_classes=tuple(
            _parse_equivalence_class(item, index=index)
            for index, item in enumerate(classes_raw)
        ),
        family_pair_stats=tuple(
            _parse_response_pair(item, index=index)
            for index, item in enumerate(family_raw)
        ),
        example_pair_stats=tuple(
            _parse_response_example_pair(item, index=index)
            for index, item in enumerate(example_raw)
        ),
        equivalence_class_atlas=(
            parse_mode_similarity_atlas_receipt_payload(class_atlas_raw)
        ),
    )
    if not _json_exact_equal(raw, receipt.canonical_payload()):
        raise ValueError("downstream response receipt is not canonical")
    return receipt


def parse_multi_abi_mode_atlas_receipt_payload(
    value: Mapping[str, object],
) -> MultiABIModeAtlasReceipt:
    """Strictly authenticate a canonical multi-ABI receipt in memory."""

    raw = _require_exact_fields(
        value,
        {
            "downstream_response",
            "format_version",
            "inventory",
            "mode_specs",
            "native_partitions",
            "receipt_sha256",
            "safety",
            "schema",
            "scientific_status",
            "terminal_norm_policy",
        },
        label="receipt",
    )
    if raw["schema"] != MULTI_ABI_MODE_ATLAS_SCHEMA:
        raise ValueError("unsupported multi-ABI mode atlas schema")
    if (
        type(raw["format_version"]) is not int
        or raw["format_version"] != MULTI_ABI_MODE_ATLAS_FORMAT_VERSION
    ):
        raise ValueError("unsupported multi-ABI mode atlas format version")
    if not _json_exact_equal(raw["scientific_status"], _SCIENTIFIC_STATUS):
        raise ValueError("scientific_status is not the frozen contract")
    if not _json_exact_equal(raw["safety"], _SAFETY):
        raise ValueError("safety is not the frozen contract")
    if not _json_exact_equal(raw["terminal_norm_policy"], _TERMINAL_NORM_POLICY):
        raise ValueError("terminal_norm_policy is not the frozen contract")
    modes_raw = raw["mode_specs"]
    partitions_raw = raw["native_partitions"]
    if not isinstance(modes_raw, list):
        raise TypeError("mode_specs must be a list")
    if not isinstance(partitions_raw, list):
        raise TypeError("native_partitions must be a list")
    receipt = MultiABIModeAtlasReceipt(
        inventory=_parse_inventory(raw["inventory"]),
        modes=tuple(
            _parse_typed_mode(item, index=index)
            for index, item in enumerate(modes_raw)
        ),
        native_partitions=tuple(
            _parse_native_partition(item, index=index)
            for index, item in enumerate(partitions_raw)
        ),
        downstream_response=_parse_downstream_receipt(
            raw["downstream_response"]
        ),
        receipt_sha256=_require_sha256(
            raw["receipt_sha256"], label="receipt_sha256"
        ),
    )
    _validate_multi_receipt(receipt, check_digest=True)
    if not _json_exact_equal(raw, receipt.canonical_payload()):
        raise ValueError("multi-ABI receipt is not canonical")
    return receipt


def _strict_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def save_multi_abi_mode_atlas_receipt(
    path: Path | str,
    receipt: MultiABIModeAtlasReceipt,
) -> None:
    """Exclusively and atomically save one canonical JSON receipt."""

    if not isinstance(receipt, MultiABIModeAtlasReceipt):
        raise TypeError("receipt must be a MultiABIModeAtlasReceipt")
    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("multi-ABI atlas receipt must use .json")
    if output.is_symlink() or output.exists():
        raise FileExistsError("refusing to overwrite multi-ABI atlas receipt")
    _validate_multi_receipt(receipt, check_digest=True)
    payload = _canonical_json_bytes(receipt.canonical_payload()) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite multi-ABI atlas receipt"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def load_multi_abi_mode_atlas_receipt(
    path: Path | str,
) -> MultiABIModeAtlasReceipt:
    """Strict-load and authenticate one canonical multi-ABI receipt."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError("refusing to load a multi-ABI receipt through a symlink")
    try:
        metadata = source.stat()
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("multi-ABI receipt must be a regular file")
    if metadata.st_size <= 0 or metadata.st_size > _MAX_RECEIPT_BYTES:
        raise ValueError("multi-ABI receipt has an invalid byte size")
    raw_bytes = source.read_bytes()
    if len(raw_bytes) != metadata.st_size:
        raise ValueError("multi-ABI receipt has an invalid byte size")
    try:
        raw = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError("multi-ABI receipt must be UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise TypeError("multi-ABI receipt must be a JSON object")
    if raw_bytes != _canonical_json_bytes(raw) + b"\n":
        raise ValueError("multi-ABI receipt bytes are not canonical JSON")
    return parse_multi_abi_mode_atlas_receipt_payload(raw)
