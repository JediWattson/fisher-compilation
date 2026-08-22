"""Pure split-locked A/B/C core for attention transport experiments.

This module performs no model loading, disk I/O, or capture.  It consumes
authenticated split records produced by the runtime seam and enforces the
scientific protocol in its API:

* generator banks, chart atlases, and routers are fit from A rows only;
* frozen candidates are evaluated and selected on B;
* C is obtained lazily and only after one B candidate passes every gate;
* the selected candidate, gates, and B decision are hash-locked before C;
* C performs evaluation only and cannot change the compiled artifact.

The row object inside a split record is structural.  It may be a full
``AttentionTransportCapture`` or a bounded sufficient-statistic row batch as
long as it exposes the compiler-facing properties described below.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from typing import Protocol

import torch
from torch import Tensor

from .gemma3_attention_transport_compiler import (
    AttentionTransportBankFit,
    AttentionTransportFitConfig,
    AttentionTransportMetrics,
    AttentionTransportResourceAccounting,
    ChartConditionedAttentionTransportRouter,
    ChartConditionedRouterConfig,
    EXACT_SPARSE_ORACLE_SOLVER,
    FisherAttentionTransportGeneratorBank,
    FrozenAttentionChartAtlas,
    HELDOUT_ORACLE_COEFFICIENT_RIDGE,
    evaluate_chart_conditioned_attention_transport_router,
    fit_chart_conditioned_attention_transport_router,
    fit_fisher_attention_transport_generator_bank,
    fit_frozen_attention_chart_atlas,
)


_CONFIG_DOMAIN = b"fisher_graph:attention-transport-experiment-config:v1\0"
_CANDIDATE_DOMAIN = b"fisher_graph:attention-transport-fitted-candidate:v1\0"
_SPLIT_DOMAIN = b"fisher_graph:attention-transport-split-panel:v1\0"
_DEVELOPMENT_DOMAIN = b"fisher_graph:attention-transport-a-development:v1\0"
_EVALUATION_DOMAIN = b"fisher_graph:attention-transport-split-evaluation:v1\0"
_LOCK_DOMAIN = b"fisher_graph:attention-transport-selection-lock:v1\0"
_B_SELECTION_DOMAIN = b"fisher_graph:attention-transport-b-selection:v1\0"
_REPORT_DOMAIN = b"fisher_graph:attention-transport-confirmation-report:v1\0"
_SELECTION_RULE = (
    "minimum_B_relative_fisher_then_compiled_parameters_then_K_rank_"
    "active_count_then_candidate_id"
)


class AttentionTransportRows(Protocol):
    """Compiler-facing rows supplied by a full capture or compact row batch."""

    invocation_id: str
    head_contexts: Tensor
    head_operators: Tensor
    native_v_o_parameter_count: int
    query_head_count: int
    hidden_width: int
    row_count: int
    head_ids: tuple[str, ...]
    valid_operator_outputs: Tensor
    valid_score_gradients: Tensor
    valid_chart_inputs: Tensor


class AttentionTransportOutputNorm(Protocol):
    invocation_id: str
    artifact_sha256: str

    def apply(self, values: Tensor) -> Tensor: ...


class AttentionTransportSplitRecord(Protocol):
    """Runtime record consumed without importing the model-loading runtime."""

    role: str
    base_capture_receipt_sha256: str
    capture: AttentionTransportRows
    postnorm_targets: Tensor
    output_norm: AttentionTransportOutputNorm


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Tensor):
        return {
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
            "shape": list(value.shape),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__} is not hash-serializable")


def _artifact_sha256(domain: bytes, payload: object) -> str:
    return hashlib.sha256(
        domain
        + json.dumps(
            _jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_threshold(label: str, value: object, *, minimum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return float(value)


@dataclass(frozen=True, slots=True)
class AttentionTransportScientificStatus:
    claim_scope: str = "pre_output_rmsnorm_attention_operator_boundary_o_t"
    operator_boundary_only: bool = True
    post_rmsnorm_diagnostic_measured: bool = True
    behavioral_nll_measured: bool = False
    behavioral_quality_claimed: bool = False
    parameter_storage_reported: bool = True
    logical_speedup_claimed: bool = False


SCIENTIFIC_STATUS = AttentionTransportScientificStatus()


@dataclass(frozen=True, slots=True)
class AttentionTransportFrozenInvocationSignature:
    invocation_id: str
    query_head_count: int
    hidden_width: int
    head_ids: tuple[str, ...]
    head_operators_sha256: str
    native_v_o_parameter_count: int
    output_norm_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.invocation_id, str)
            or not self.invocation_id
            or type(self.query_head_count) is not int
            or self.query_head_count <= 0
            or type(self.hidden_width) is not int
            or self.hidden_width <= 0
            or len(self.head_ids) != self.query_head_count
            or len(set(self.head_ids)) != len(self.head_ids)
            or not _is_sha256(self.head_operators_sha256)
            or type(self.native_v_o_parameter_count) is not int
            or self.native_v_o_parameter_count <= 0
            or not _is_sha256(self.output_norm_sha256)
        ):
            raise ValueError("frozen invocation signature is invalid")


@dataclass(frozen=True, slots=True)
class AttentionTransportCandidateConfig:
    candidate_id: str
    bank: AttentionTransportFitConfig
    chart_count: int
    chart_rank: int
    router: ChartConditionedRouterConfig
    chart_temperature_scale: float = 1.0
    chart_lloyd_iterations: int = 12

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        if not isinstance(self.bank, AttentionTransportFitConfig):
            raise TypeError("candidate bank config is invalid")
        if not isinstance(self.router, ChartConditionedRouterConfig):
            raise TypeError("candidate router config is invalid")
        if type(self.chart_count) is not int or self.chart_count <= 0:
            raise ValueError("chart_count must be positive")
        if type(self.chart_rank) is not int or self.chart_rank <= 0:
            raise ValueError("chart_rank must be positive")
        if (
            type(self.chart_lloyd_iterations) is not int
            or self.chart_lloyd_iterations <= 0
        ):
            raise ValueError("chart_lloyd_iterations must be positive")
        _finite_threshold(
            "chart_temperature_scale",
            self.chart_temperature_scale,
            minimum=torch.finfo(torch.float64).tiny,
        )
        if self.router.active_count > self.bank.generator_count:
            raise ValueError("router active_count exceeds the generator inventory")
        if self.router.euclidean_floor != self.bank.euclidean_floor:
            raise ValueError("bank and router Fisher floors must match")

    @property
    def artifact_sha256(self) -> str:
        return _artifact_sha256(_CONFIG_DOMAIN, self)


@dataclass(frozen=True, slots=True)
class AttentionTransportSelectionGates:
    max_relative_fisher_loss: float
    max_oracle_router_fisher_gap: float
    max_postnorm_normalized_rmse: float
    min_postnorm_cosine: float
    oracle_gap_numerical_tolerance: float = 1.0e-10
    max_sparse_probability_renormalization_error: float = 1.0e-10
    max_effective_support_fraction: float = 1.0
    minimum_aggregate_rows: int = 12_960
    require_parameter_compression: bool = True

    def __post_init__(self) -> None:
        for label in (
            "max_relative_fisher_loss",
            "max_oracle_router_fisher_gap",
            "max_postnorm_normalized_rmse",
            "oracle_gap_numerical_tolerance",
            "max_sparse_probability_renormalization_error",
            "max_effective_support_fraction",
        ):
            _finite_threshold(label, getattr(self, label), minimum=0.0)
        if (
            isinstance(self.min_postnorm_cosine, bool)
            or not isinstance(self.min_postnorm_cosine, (int, float))
            or not math.isfinite(float(self.min_postnorm_cosine))
            or not -1.0 <= float(self.min_postnorm_cosine) <= 1.0
        ):
            raise ValueError("min_postnorm_cosine must lie in [-1, 1]")
        if float(self.max_effective_support_fraction) > 1.0:
            raise ValueError("max_effective_support_fraction cannot exceed one")
        if (
            type(self.minimum_aggregate_rows) is not int
            or self.minimum_aggregate_rows <= 0
        ):
            raise ValueError("minimum_aggregate_rows must be positive")
        if type(self.require_parameter_compression) is not bool:
            raise TypeError("require_parameter_compression must be boolean")

    @property
    def artifact_sha256(self) -> str:
        return _artifact_sha256(_CONFIG_DOMAIN, self)


@dataclass(frozen=True, slots=True)
class _AttentionTransportSplitPanel:
    role: str
    records: tuple[AttentionTransportSplitRecord, ...]
    invocation_ids: tuple[str, ...]
    receipt_sha256s: tuple[str, ...]
    signatures: tuple[AttentionTransportFrozenInvocationSignature, ...]
    artifact_sha256: str

    @property
    def rows(self) -> tuple[AttentionTransportRows, ...]:
        return tuple(record.capture for record in self.records)

    def records_for_invocation(
        self,
        invocation_id: str,
    ) -> tuple[AttentionTransportSplitRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.capture.invocation_id == invocation_id
        )


def _row_signature(
    record: AttentionTransportSplitRecord,
) -> AttentionTransportFrozenInvocationSignature:
    rows = record.capture
    operator_sha256 = getattr(rows, "head_operators_sha256", None)
    if not isinstance(operator_sha256, str) or not _is_sha256(operator_sha256):
        operator_sha256 = _tensor_sha256(rows.head_operators)
    return AttentionTransportFrozenInvocationSignature(
        invocation_id=rows.invocation_id,
        query_head_count=int(rows.query_head_count),
        hidden_width=int(rows.hidden_width),
        head_ids=tuple(rows.head_ids),
        head_operators_sha256=operator_sha256,
        native_v_o_parameter_count=int(rows.native_v_o_parameter_count),
        output_norm_sha256=record.output_norm.artifact_sha256,
    )


def _open_split_panel(
    records: Sequence[AttentionTransportSplitRecord],
    *,
    role: str,
) -> _AttentionTransportSplitPanel:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError(f"split {role} records must be a materialized sequence")
    values = tuple(records)
    if not values:
        raise ValueError(f"split {role} must contain capture records")
    signatures_by_invocation: dict[
        str,
        AttentionTransportFrozenInvocationSignature,
    ] = {}
    inventory_by_receipt: dict[str, set[str]] = {}
    validated_operator_catalogs: set[int] = set()
    for record in values:
        try:
            record_role = record.role
            receipt = record.base_capture_receipt_sha256
            rows = record.capture
            output_norm = record.output_norm
            postnorm = record.postnorm_targets
        except AttributeError as error:
            raise TypeError("split record does not expose transport fields") from error
        if record_role != role:
            raise ValueError(
                f"split {role} contains a record with role {record_role!r}"
            )
        if not _is_sha256(receipt):
            raise ValueError(f"split {role} contains an invalid capture receipt")
        if (
            not isinstance(rows.invocation_id, str)
            or not rows.invocation_id
            or output_norm.invocation_id != rows.invocation_id
            or type(rows.row_count) is not int
            or rows.row_count <= 0
            or not isinstance(postnorm, Tensor)
            or postnorm.shape != rows.valid_operator_outputs.shape
            or not postnorm.is_floating_point()
            or not bool(torch.isfinite(postnorm).all())
        ):
            raise ValueError(f"split {role} contains an invalid aligned row record")
        catalog_identity = getattr(
            rows,
            "head_operator_catalog_identity",
            id(rows),
        )
        if type(catalog_identity) is not int:
            raise ValueError("split row head-operator catalog identity is invalid")
        if catalog_identity not in validated_operator_catalogs:
            validator = getattr(rows, "validate_head_operator_catalog", None)
            if validator is not None:
                if not callable(validator):
                    raise TypeError("split head-operator catalog validator is invalid")
                validator(rehash=True)
            validated_operator_catalogs.add(catalog_identity)
        exact_postnorm = output_norm.apply(rows.valid_operator_outputs)
        if not torch.allclose(
            exact_postnorm.to(device="cpu", dtype=torch.float64),
            postnorm.detach().to(device="cpu", dtype=torch.float64),
            atol=2.0e-5,
            rtol=2.0e-5,
        ):
            raise ValueError("split postnorm target does not close over native rows")
        inventory = inventory_by_receipt.setdefault(receipt, set())
        if rows.invocation_id in inventory:
            raise ValueError("one split receipt repeats an invocation")
        inventory.add(rows.invocation_id)
        signature = _row_signature(record)
        prior = signatures_by_invocation.setdefault(rows.invocation_id, signature)
        if prior != signature:
            raise ValueError("one split invocation changed frozen operators or norm")

    expected_inventory = set(signatures_by_invocation)
    if any(value != expected_inventory for value in inventory_by_receipt.values()):
        raise ValueError("split receipts do not share one invocation inventory")
    invocation_ids = tuple(sorted(expected_inventory))
    receipts = tuple(sorted(inventory_by_receipt))
    signatures = tuple(signatures_by_invocation[key] for key in invocation_ids)
    ordered_values = tuple(
        sorted(
            values,
            key=lambda item: (
                item.capture.invocation_id,
                item.base_capture_receipt_sha256,
            ),
        )
    )
    record_digests = tuple(
        _artifact_sha256(
            _SPLIT_DOMAIN,
            {
                "base_capture_receipt_sha256": (
                    record.base_capture_receipt_sha256
                ),
                "chart_inputs": record.capture.valid_chart_inputs,
                "head_contexts": record.capture.head_contexts,
                "invocation_id": record.capture.invocation_id,
                "operator_outputs": record.capture.valid_operator_outputs,
                "postnorm_targets": record.postnorm_targets,
                "score_gradients": record.capture.valid_score_gradients,
            },
        )
        for record in sorted(
            ordered_values,
            key=lambda item: (
                item.base_capture_receipt_sha256,
                item.capture.invocation_id,
            ),
        )
    )
    split_hash = _artifact_sha256(
        _SPLIT_DOMAIN,
        {
            "invocation_signatures": signatures,
            "record_digests": record_digests,
            "receipt_sha256s": receipts,
            "role": role,
        },
    )
    return _AttentionTransportSplitPanel(
        role=role,
        records=ordered_values,
        invocation_ids=invocation_ids,
        receipt_sha256s=receipts,
        signatures=signatures,
        artifact_sha256=split_hash,
    )


def _validate_panel_lineage(
    panel: _AttentionTransportSplitPanel,
    *,
    signatures: tuple[AttentionTransportFrozenInvocationSignature, ...],
    prior_receipts: tuple[str, ...],
) -> None:
    if panel.signatures != signatures:
        raise ValueError(f"split {panel.role} frozen invocation catalog drifted")
    if set(panel.receipt_sha256s) & set(prior_receipts):
        raise ValueError("A/B/C capture receipts must be disjoint")


def _resource_accounting(
    bank: FisherAttentionTransportGeneratorBank,
    atlas: FrozenAttentionChartAtlas,
    routers: Sequence[ChartConditionedAttentionTransportRouter],
    signatures: Sequence[AttentionTransportFrozenInvocationSignature],
) -> AttentionTransportResourceAccounting:
    router_values = tuple(routers)
    signature_values = tuple(signatures)
    if (
        not router_values
        or len(router_values) != len(signature_values)
        or {value.invocation_id for value in router_values}
        != {value.invocation_id for value in signature_values}
        or any(
            value.generator_bank_sha256 != bank.artifact_sha256
            for value in router_values
        )
        or any(
            value.atlas.artifact_sha256 != atlas.artifact_sha256
            for value in router_values
        )
    ):
        raise ValueError("resource components do not share one frozen lineage")
    native = sum(value.native_v_o_parameter_count for value in signature_values)
    bank_count = bank.stored_parameter_count
    chart_count = atlas.stored_parameter_count
    router_count = sum(value.stored_parameter_count for value in router_values)
    compiled = bank_count + chart_count + router_count
    return AttentionTransportResourceAccounting(
        native_v_o_parameters=native,
        generator_bank_parameters=bank_count,
        chart_parameters=chart_count,
        router_parameters=router_count,
        compiled_parameters=compiled,
        parameter_reduction_fraction=float(1.0 - compiled / native),
        parameter_compression_eligible=compiled < native,
    )


@dataclass(frozen=True, slots=True)
class AttentionTransportPostNormMetrics:
    row_count: int
    normalized_rmse: float
    cosine: float


@dataclass(frozen=True, slots=True)
class AttentionTransportInvocationEvaluation:
    invocation_id: str
    output_metrics: AttentionTransportMetrics
    oracle_output_metrics: AttentionTransportMetrics
    oracle_solver_kind: str
    oracle_active_count_limit: int
    oracle_coefficient_ridge: float
    zero_predictor_fisher_loss: float
    relative_fisher_loss: float
    oracle_relative_fisher_loss: float
    oracle_router_fisher_gap: float
    postnorm_metrics: AttentionTransportPostNormMetrics
    per_chart_active_count_minimum: int
    per_chart_active_count_mean: float
    per_chart_active_count_maximum: int
    effective_support_count_minimum: int
    effective_support_count_mean: float
    effective_support_count_maximum: int
    effective_support_fraction: float
    sparse_probability_renormalization_max_error: float
    generator_usage: Tensor
    dead_generator_count: int

    def __post_init__(self) -> None:
        usage = self.generator_usage
        if (
            not isinstance(self.invocation_id, str)
            or not self.invocation_id
            or self.oracle_solver_kind != EXACT_SPARSE_ORACLE_SOLVER
            or type(self.oracle_active_count_limit) is not int
            or self.oracle_active_count_limit <= 0
            or type(self.oracle_coefficient_ridge) is not float
            or self.oracle_coefficient_ridge != HELDOUT_ORACLE_COEFFICIENT_RIDGE
            or not isinstance(usage, Tensor)
            or usage.ndim != 1
            or not usage.is_floating_point()
            or not bool(torch.isfinite(usage).all())
            or bool((usage < 0.0).any())
        ):
            raise ValueError("per-invocation transport evaluation is invalid")
        usage = usage.detach().to(device="cpu", dtype=torch.float64).clone()
        if not torch.allclose(
            usage.sum(),
            torch.ones((), dtype=torch.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        ):
            raise ValueError("per-invocation generator usage is not normalized")
        object.__setattr__(self, "generator_usage", usage)


@dataclass(frozen=True, slots=True)
class AttentionTransportAggregateEvaluation:
    role: str
    split_artifact_sha256: str
    row_count: int
    invocation_count: int
    fisher_loss: float
    zero_predictor_fisher_loss: float
    oracle_solver_kind: str
    oracle_active_count_limit: int
    oracle_coefficient_ridge: float
    relative_fisher_loss: float
    oracle_fisher_loss: float
    oracle_relative_fisher_loss: float
    oracle_router_fisher_gap: float
    operator_normalized_rmse: float
    operator_cosine: float
    postnorm_normalized_rmse: float
    postnorm_cosine: float
    per_chart_active_count_minimum: int
    per_chart_active_count_mean: float
    per_chart_active_count_maximum: int
    effective_support_count_minimum: int
    effective_support_count_mean: float
    effective_support_count_maximum: int
    effective_support_fraction: float
    sparse_probability_renormalization_max_error: float
    generator_usage: Tensor
    dead_generator_count: int
    per_invocation: tuple[AttentionTransportInvocationEvaluation, ...]
    resources: AttentionTransportResourceAccounting

    def __post_init__(self) -> None:
        if (
            self.role not in {"B", "C"}
            or not _is_sha256(self.split_artifact_sha256)
            or type(self.row_count) is not int
            or self.row_count <= 0
            or type(self.invocation_count) is not int
            or self.invocation_count != len(self.per_invocation)
            or self.invocation_count <= 0
            or self.oracle_solver_kind != EXACT_SPARSE_ORACLE_SOLVER
            or type(self.oracle_active_count_limit) is not int
            or self.oracle_active_count_limit <= 0
            or type(self.oracle_coefficient_ridge) is not float
            or self.oracle_coefficient_ridge != HELDOUT_ORACLE_COEFFICIENT_RIDGE
            or any(
                value.oracle_solver_kind != self.oracle_solver_kind
                or value.oracle_active_count_limit
                != self.oracle_active_count_limit
                or value.oracle_coefficient_ridge
                != self.oracle_coefficient_ridge
                for value in self.per_invocation
            )
            or not isinstance(self.resources, AttentionTransportResourceAccounting)
            or self.resources.logical_speedup_claimed
        ):
            raise ValueError("aggregate transport evaluation is invalid")
        usage = self.generator_usage
        if (
            not isinstance(usage, Tensor)
            or usage.ndim != 1
            or not usage.is_floating_point()
            or not bool(torch.isfinite(usage).all())
            or bool((usage < 0.0).any())
        ):
            raise ValueError("aggregate generator usage is invalid")
        usage = usage.detach().to(device="cpu", dtype=torch.float64).clone()
        if not torch.allclose(
            usage.sum(),
            torch.ones((), dtype=torch.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        ):
            raise ValueError("aggregate generator usage is not normalized")
        object.__setattr__(self, "generator_usage", usage)

    @property
    def artifact_sha256(self) -> str:
        return _artifact_sha256(_EVALUATION_DOMAIN, self)


@dataclass(frozen=True, slots=True)
class AttentionTransportFittedCandidate:
    config: AttentionTransportCandidateConfig
    bank_fit: AttentionTransportBankFit
    atlas: FrozenAttentionChartAtlas
    routers: tuple[ChartConditionedAttentionTransportRouter, ...]
    resources: AttentionTransportResourceAccounting
    a_split_artifact_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.config, AttentionTransportCandidateConfig)
            or not isinstance(self.bank_fit, AttentionTransportBankFit)
            or not isinstance(self.atlas, FrozenAttentionChartAtlas)
            or not isinstance(self.resources, AttentionTransportResourceAccounting)
            or not _is_sha256(self.a_split_artifact_sha256)
            or not self.routers
            or any(
                not isinstance(router, ChartConditionedAttentionTransportRouter)
                for router in self.routers
            )
        ):
            raise ValueError("fitted attention transport candidate is invalid")
        bank_hash = self.bank_fit.bank.artifact_sha256
        if (
            tuple(router.invocation_id for router in self.routers)
            != tuple(sorted(router.invocation_id for router in self.routers))
            or len({router.invocation_id for router in self.routers})
            != len(self.routers)
            or any(router.generator_bank_sha256 != bank_hash for router in self.routers)
            or any(
                router.atlas.artifact_sha256 != self.atlas.artifact_sha256
                for router in self.routers
            )
            or self.resources.logical_speedup_claimed
        ):
            raise ValueError("fitted candidate artifact lineage is invalid")

    @property
    def bank(self) -> FisherAttentionTransportGeneratorBank:
        return self.bank_fit.bank

    @property
    def artifact_sha256(self) -> str:
        return _artifact_sha256(
            _CANDIDATE_DOMAIN,
            {
                "a_split_artifact_sha256": self.a_split_artifact_sha256,
                "atlas_sha256": self.atlas.artifact_sha256,
                "bank_sha256": self.bank.artifact_sha256,
                "config_sha256": self.config.artifact_sha256,
                "resources": self.resources,
                "router_sha256s": [
                    router.artifact_sha256 for router in self.routers
                ],
            },
        )

    def router_for_invocation(
        self,
        invocation_id: str,
    ) -> ChartConditionedAttentionTransportRouter:
        for router in self.routers:
            if router.invocation_id == invocation_id:
                return router
        raise ValueError("candidate has no router for the requested invocation")


@dataclass(frozen=True, slots=True)
class AttentionTransportADevelopment:
    a_split_artifact_sha256: str
    a_receipt_sha256s: tuple[str, ...]
    frozen_invocation_signatures: tuple[
        AttentionTransportFrozenInvocationSignature,
        ...,
    ]
    candidates: tuple[AttentionTransportFittedCandidate, ...]
    candidate_artifact_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.a_split_artifact_sha256)
            or not self.a_receipt_sha256s
            or any(not _is_sha256(value) for value in self.a_receipt_sha256s)
            or not self.candidates
            or len(self.candidates) != len(self.candidate_artifact_sha256s)
            or tuple(value.artifact_sha256 for value in self.candidates)
            != self.candidate_artifact_sha256s
            or len({value.config.candidate_id for value in self.candidates})
            != len(self.candidates)
        ):
            raise ValueError("A-development artifact is invalid")

    @property
    def artifact_sha256(self) -> str:
        return _artifact_sha256(
            _DEVELOPMENT_DOMAIN,
            {
                "a_receipt_sha256s": self.a_receipt_sha256s,
                "a_split_artifact_sha256": self.a_split_artifact_sha256,
                "candidate_artifact_sha256s": self.candidate_artifact_sha256s,
                "frozen_invocation_signatures": self.frozen_invocation_signatures,
            },
        )


def fit_attention_transport_candidates_on_a(
    a_records: Sequence[AttentionTransportSplitRecord],
    candidate_configs: Sequence[AttentionTransportCandidateConfig],
) -> AttentionTransportADevelopment:
    """Fit every deployable artifact using A, and A, only."""

    panel = _open_split_panel(a_records, role="A")
    if (
        isinstance(candidate_configs, (str, bytes))
        or not isinstance(candidate_configs, Sequence)
    ):
        raise TypeError("candidate configs must be a materialized sequence")
    configs = tuple(candidate_configs)
    if (
        not configs
        or any(
            not isinstance(value, AttentionTransportCandidateConfig)
            for value in configs
        )
        or len({value.candidate_id for value in configs}) != len(configs)
    ):
        raise ValueError("candidate configs must be nonempty with unique ids")
    configs = tuple(sorted(configs, key=lambda value: value.candidate_id))

    # The compiler currently requests fit and evaluation rows together.  Both
    # arguments deliberately receive A.  B is unavailable in this function.
    rows = panel.rows
    bank_cache: dict[
        AttentionTransportFitConfig,
        tuple[AttentionTransportBankFit, str],
    ] = {}
    atlas_cache: dict[
        tuple[int, int, float, int],
        tuple[FrozenAttentionChartAtlas, str],
    ] = {}
    fitted = []
    for config in configs:
        cached_bank = bank_cache.get(config.bank)
        if cached_bank is None:
            bank_fit = fit_fisher_attention_transport_generator_bank(
                rows,
                rows,
                config=config.bank,
            )
            bank_sha256 = bank_fit.bank.artifact_sha256
            bank_cache[config.bank] = (bank_fit, bank_sha256)
        else:
            bank_fit, bank_sha256 = cached_bank
            if bank_fit.bank.artifact_sha256 != bank_sha256:
                raise RuntimeError("cached A-fitted generator bank changed")
        atlas_key = (
            config.chart_count,
            config.chart_rank,
            float(config.chart_temperature_scale),
            config.chart_lloyd_iterations,
        )
        cached_atlas = atlas_cache.get(atlas_key)
        if cached_atlas is None:
            atlas = fit_frozen_attention_chart_atlas(
                rows,
                chart_count=config.chart_count,
                chart_rank=config.chart_rank,
                temperature_scale=config.chart_temperature_scale,
                lloyd_iterations=config.chart_lloyd_iterations,
            )
            atlas_sha256 = atlas.artifact_sha256
            atlas_cache[atlas_key] = (atlas, atlas_sha256)
        else:
            atlas, atlas_sha256 = cached_atlas
            if atlas.artifact_sha256 != atlas_sha256:
                raise RuntimeError("cached A-fitted chart atlas changed")
        routers = tuple(
            fit_chart_conditioned_attention_transport_router(
                bank_fit.bank,
                tuple(
                    record.capture
                    for record in panel.records_for_invocation(invocation_id)
                ),
                atlas,
                config=config.router,
            ).router
            for invocation_id in panel.invocation_ids
        )
        if (
            bank_fit.bank.artifact_sha256 != bank_sha256
            or atlas.artifact_sha256 != atlas_sha256
        ):
            raise RuntimeError("router fit mutated a cached A-side artifact")
        resources = _resource_accounting(
            bank_fit.bank,
            atlas,
            routers,
            panel.signatures,
        )
        fitted.append(
            AttentionTransportFittedCandidate(
                config=config,
                bank_fit=bank_fit,
                atlas=atlas,
                routers=routers,
                resources=resources,
                a_split_artifact_sha256=panel.artifact_sha256,
            )
        )
    candidates = tuple(fitted)
    return AttentionTransportADevelopment(
        a_split_artifact_sha256=panel.artifact_sha256,
        a_receipt_sha256s=panel.receipt_sha256s,
        frozen_invocation_signatures=panel.signatures,
        candidates=candidates,
        candidate_artifact_sha256s=tuple(
            value.artifact_sha256 for value in candidates
        ),
    )


def _normalized_rmse_and_cosine(
    prediction: Tensor,
    target: Tensor,
) -> tuple[float, float]:
    prediction_flat = prediction.reshape(-1).to(dtype=torch.float64)
    target_flat = target.reshape(-1).to(dtype=torch.float64)
    error_energy = float((prediction_flat - target_flat).square().sum())
    target_energy = float(target_flat.square().sum())
    prediction_energy = float(prediction_flat.square().sum())
    normalized_rmse = (
        math.sqrt(error_energy / target_energy)
        if target_energy > 0.0
        else 0.0
        if error_energy == 0.0
        else math.inf
    )
    denominator = math.sqrt(prediction_energy * target_energy)
    cosine = (
        float(torch.dot(prediction_flat, target_flat)) / denominator
        if denominator > 0.0
        else 0.0
    )
    return float(normalized_rmse), float(cosine)


def _evaluate_candidate_on_panel(
    candidate: AttentionTransportFittedCandidate,
    panel: _AttentionTransportSplitPanel,
) -> AttentionTransportAggregateEvaluation:
    if panel.role not in {"B", "C"}:
        raise ValueError("frozen candidate evaluation is reserved for B or C")
    resources = _resource_accounting(
        candidate.bank,
        candidate.atlas,
        candidate.routers,
        panel.signatures,
    )
    if resources != candidate.resources:
        raise ValueError("held-out resource accounting drifted from locked A")

    per_invocation = []
    all_predictions = []
    all_targets = []
    all_postnorm_predictions = []
    all_postnorm_targets = []
    total_rows = 0
    fisher_sum = 0.0
    zero_sum = 0.0
    oracle_sum = 0.0
    active_weighted_sum = 0.0
    active_denominator = 0
    active_minimum = math.inf
    active_maximum = 0
    support_weighted_sum = 0.0
    support_denominator = 0
    support_minimum = math.inf
    support_maximum = 0
    renormalization_maximum = 0.0
    usage_weighted = torch.zeros(
        candidate.bank.generator_count,
        dtype=torch.float64,
    )

    for invocation_id in panel.invocation_ids:
        router = candidate.router_for_invocation(invocation_id)
        records = panel.records_for_invocation(invocation_id)
        rows = tuple(record.capture for record in records)
        evaluation = evaluate_chart_conditioned_attention_transport_router(
            candidate.bank,
            router,
            rows,
            euclidean_floor=candidate.config.router.euclidean_floor,
        )
        predictions = []
        targets = []
        postnorm_predictions = []
        postnorm_targets = []
        for record in records:
            row_batch = record.capture
            prediction = router.predict_operator_outputs(
                candidate.bank,
                head_contexts=row_batch.head_contexts,
                chart_inputs=row_batch.valid_chart_inputs,
            )
            predictions.append(prediction)
            targets.append(row_batch.valid_operator_outputs)
            postnorm_predictions.append(record.output_norm.apply(prediction))
            postnorm_targets.append(
                record.postnorm_targets.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
            )
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        postnorm_prediction = torch.cat(postnorm_predictions)
        postnorm_target = torch.cat(postnorm_targets)
        row_count = int(target.shape[0])
        postnorm_nrmse, postnorm_cosine = _normalized_rmse_and_cosine(
            postnorm_prediction,
            postnorm_target,
        )
        invocation_evaluation = AttentionTransportInvocationEvaluation(
            invocation_id=invocation_id,
            output_metrics=evaluation.output_metrics,
            oracle_output_metrics=evaluation.oracle_output_metrics,
            oracle_solver_kind=evaluation.oracle_solver_kind,
            oracle_active_count_limit=evaluation.oracle_active_count_limit,
            oracle_coefficient_ridge=evaluation.oracle_coefficient_ridge,
            zero_predictor_fisher_loss=evaluation.zero_predictor_fisher_loss,
            relative_fisher_loss=evaluation.relative_fisher_loss,
            oracle_relative_fisher_loss=evaluation.oracle_relative_fisher_loss,
            oracle_router_fisher_gap=evaluation.oracle_router_fisher_gap,
            postnorm_metrics=AttentionTransportPostNormMetrics(
                row_count=row_count,
                normalized_rmse=postnorm_nrmse,
                cosine=postnorm_cosine,
            ),
            per_chart_active_count_minimum=evaluation.active_count_minimum,
            per_chart_active_count_mean=evaluation.active_count_mean,
            per_chart_active_count_maximum=evaluation.active_count_maximum,
            effective_support_count_minimum=(
                evaluation.effective_support_count_minimum
            ),
            effective_support_count_mean=(
                evaluation.effective_support_count_mean
            ),
            effective_support_count_maximum=(
                evaluation.effective_support_count_maximum
            ),
            effective_support_fraction=float(
                evaluation.effective_support_count_mean
                / candidate.bank.generator_count
            ),
            sparse_probability_renormalization_max_error=(
                evaluation.sparse_probability_renormalization_max_error
            ),
            generator_usage=evaluation.generator_usage,
            dead_generator_count=evaluation.dead_generator_count,
        )
        per_invocation.append(invocation_evaluation)
        all_predictions.append(prediction)
        all_targets.append(target)
        all_postnorm_predictions.append(postnorm_prediction)
        all_postnorm_targets.append(postnorm_target)
        total_rows += row_count
        fisher_sum += evaluation.output_metrics.fisher_loss * row_count
        zero_sum += evaluation.zero_predictor_fisher_loss * row_count
        oracle_sum += evaluation.oracle_output_metrics.fisher_loss * row_count
        chart_routes = row_count * candidate.atlas.chart_count
        active_weighted_sum += evaluation.active_count_mean * chart_routes
        active_denominator += chart_routes
        active_minimum = min(active_minimum, evaluation.active_count_minimum)
        active_maximum = max(active_maximum, evaluation.active_count_maximum)
        support_weighted_sum += (
            evaluation.effective_support_count_mean * row_count
        )
        support_denominator += row_count
        support_minimum = min(
            support_minimum,
            evaluation.effective_support_count_minimum,
        )
        support_maximum = max(
            support_maximum,
            evaluation.effective_support_count_maximum,
        )
        renormalization_maximum = max(
            renormalization_maximum,
            evaluation.sparse_probability_renormalization_max_error,
        )
        usage_weighted += evaluation.generator_usage * row_count

    prediction = torch.cat(all_predictions)
    target = torch.cat(all_targets)
    postnorm_prediction = torch.cat(all_postnorm_predictions)
    postnorm_target = torch.cat(all_postnorm_targets)
    operator_nrmse, operator_cosine = _normalized_rmse_and_cosine(
        prediction,
        target,
    )
    postnorm_nrmse, postnorm_cosine = _normalized_rmse_and_cosine(
        postnorm_prediction,
        postnorm_target,
    )
    fisher_loss = fisher_sum / total_rows
    zero_loss = zero_sum / total_rows
    oracle_loss = oracle_sum / total_rows
    if zero_loss <= 0.0 or not math.isfinite(zero_loss):
        raise ValueError("aggregate zero-predictor Fisher loss is not evaluable")
    usage = usage_weighted / usage_weighted.sum().clamp_min(
        torch.finfo(torch.float64).tiny
    )
    effective_mean = support_weighted_sum / support_denominator
    return AttentionTransportAggregateEvaluation(
        role=panel.role,
        split_artifact_sha256=panel.artifact_sha256,
        row_count=total_rows,
        invocation_count=len(per_invocation),
        fisher_loss=float(fisher_loss),
        zero_predictor_fisher_loss=float(zero_loss),
        oracle_solver_kind=EXACT_SPARSE_ORACLE_SOLVER,
        oracle_active_count_limit=candidate.config.router.active_count,
        oracle_coefficient_ridge=HELDOUT_ORACLE_COEFFICIENT_RIDGE,
        relative_fisher_loss=float(fisher_loss / zero_loss),
        oracle_fisher_loss=float(oracle_loss),
        oracle_relative_fisher_loss=float(oracle_loss / zero_loss),
        oracle_router_fisher_gap=float((fisher_loss - oracle_loss) / zero_loss),
        operator_normalized_rmse=operator_nrmse,
        operator_cosine=operator_cosine,
        postnorm_normalized_rmse=postnorm_nrmse,
        postnorm_cosine=postnorm_cosine,
        per_chart_active_count_minimum=int(active_minimum),
        per_chart_active_count_mean=float(
            active_weighted_sum / active_denominator
        ),
        per_chart_active_count_maximum=int(active_maximum),
        effective_support_count_minimum=int(support_minimum),
        effective_support_count_mean=float(effective_mean),
        effective_support_count_maximum=int(support_maximum),
        effective_support_fraction=float(
            effective_mean / candidate.bank.generator_count
        ),
        sparse_probability_renormalization_max_error=float(
            renormalization_maximum
        ),
        generator_usage=usage,
        dead_generator_count=int((usage == 0.0).sum()),
        per_invocation=tuple(per_invocation),
        resources=resources,
    )


@dataclass(frozen=True, slots=True)
class AttentionTransportCandidateGateDecision:
    candidate_id: str
    candidate_artifact_sha256: str
    evaluation: AttentionTransportAggregateEvaluation
    relative_fisher_pass: bool
    oracle_gap_pass: bool
    postnorm_nrmse_pass: bool
    postnorm_cosine_pass: bool
    sparse_renormalization_pass: bool
    effective_support_pass: bool
    global_support_budget_pass: bool
    minimum_rows_pass: bool
    resource_pass: bool
    passed: bool

    def __post_init__(self) -> None:
        flags = (
            self.relative_fisher_pass,
            self.oracle_gap_pass,
            self.postnorm_nrmse_pass,
            self.postnorm_cosine_pass,
            self.sparse_renormalization_pass,
            self.effective_support_pass,
            self.global_support_budget_pass,
            self.minimum_rows_pass,
            self.resource_pass,
        )
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or not _is_sha256(self.candidate_artifact_sha256)
            or not isinstance(self.evaluation, AttentionTransportAggregateEvaluation)
            or self.evaluation.role not in {"B", "C"}
            or any(type(value) is not bool for value in (*flags, self.passed))
            or self.passed != all(flags)
        ):
            raise ValueError("candidate gate decision is inconsistent")

    @property
    def artifact_sha256(self) -> str:
        return _artifact_sha256(_EVALUATION_DOMAIN, self)


def _gate_candidate(
    candidate: AttentionTransportFittedCandidate,
    evaluation: AttentionTransportAggregateEvaluation,
    gates: AttentionTransportSelectionGates,
) -> AttentionTransportCandidateGateDecision:
    relative = (
        evaluation.relative_fisher_loss
        <= float(gates.max_relative_fisher_loss)
    )
    oracle_gap = (
        -float(gates.oracle_gap_numerical_tolerance)
        <= evaluation.oracle_router_fisher_gap
        <= float(gates.max_oracle_router_fisher_gap)
    )
    postnorm_nrmse = (
        evaluation.postnorm_normalized_rmse
        <= float(gates.max_postnorm_normalized_rmse)
    )
    postnorm_cosine = (
        evaluation.postnorm_cosine >= float(gates.min_postnorm_cosine)
    )
    sparse = (
        evaluation.sparse_probability_renormalization_max_error
        <= float(gates.max_sparse_probability_renormalization_error)
    )
    support = (
        evaluation.effective_support_fraction
        <= float(gates.max_effective_support_fraction)
    )
    global_support_budget = (
        evaluation.oracle_active_count_limit
        == candidate.config.router.active_count
        and evaluation.effective_support_count_maximum
        <= candidate.config.router.active_count
    )
    minimum_rows = evaluation.row_count >= gates.minimum_aggregate_rows
    resource = (
        evaluation.resources.parameter_compression_eligible
        if gates.require_parameter_compression
        else True
    )
    return AttentionTransportCandidateGateDecision(
        candidate_id=candidate.config.candidate_id,
        candidate_artifact_sha256=candidate.artifact_sha256,
        evaluation=evaluation,
        relative_fisher_pass=relative,
        oracle_gap_pass=oracle_gap,
        postnorm_nrmse_pass=postnorm_nrmse,
        postnorm_cosine_pass=postnorm_cosine,
        sparse_renormalization_pass=sparse,
        effective_support_pass=support,
        global_support_budget_pass=global_support_budget,
        minimum_rows_pass=minimum_rows,
        resource_pass=resource,
        passed=all(
            (
                relative,
                oracle_gap,
                postnorm_nrmse,
                postnorm_cosine,
                sparse,
                support,
                global_support_budget,
                minimum_rows,
                resource,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class AttentionTransportSelectionLock:
    selected_candidate: AttentionTransportFittedCandidate
    selected_candidate_artifact_sha256: str
    gates: AttentionTransportSelectionGates
    gates_artifact_sha256: str
    b_decision: AttentionTransportCandidateGateDecision
    a_development_artifact_sha256: str
    a_split_artifact_sha256: str
    b_split_artifact_sha256: str
    a_receipt_sha256s: tuple[str, ...]
    b_receipt_sha256s: tuple[str, ...]
    frozen_invocation_signatures: tuple[
        AttentionTransportFrozenInvocationSignature,
        ...,
    ]
    selection_rule: str = _SELECTION_RULE

    def __post_init__(self) -> None:
        rebuilt_decision_sha256 = (
            _gate_candidate(
                self.selected_candidate,
                self.b_decision.evaluation,
                self.gates,
            ).artifact_sha256
            if isinstance(
                self.selected_candidate,
                AttentionTransportFittedCandidate,
            )
            and isinstance(self.gates, AttentionTransportSelectionGates)
            and isinstance(
                self.b_decision,
                AttentionTransportCandidateGateDecision,
            )
            else None
        )
        if (
            not isinstance(
                self.selected_candidate,
                AttentionTransportFittedCandidate,
            )
            or self.selected_candidate.artifact_sha256
            != self.selected_candidate_artifact_sha256
            or not isinstance(self.gates, AttentionTransportSelectionGates)
            or self.gates.artifact_sha256 != self.gates_artifact_sha256
            or not isinstance(
                self.b_decision,
                AttentionTransportCandidateGateDecision,
            )
            or not self.b_decision.passed
            or self.b_decision.evaluation.role != "B"
            or self.b_decision.candidate_id
            != self.selected_candidate.config.candidate_id
            or self.b_decision.candidate_artifact_sha256
            != self.selected_candidate_artifact_sha256
            or rebuilt_decision_sha256 != self.b_decision.artifact_sha256
            or self.b_decision.evaluation.split_artifact_sha256
            != self.b_split_artifact_sha256
            or not _is_sha256(self.a_development_artifact_sha256)
            or not _is_sha256(self.a_split_artifact_sha256)
            or not _is_sha256(self.b_split_artifact_sha256)
            or set(self.a_receipt_sha256s) & set(self.b_receipt_sha256s)
            or not self.frozen_invocation_signatures
            or self.selection_rule != _SELECTION_RULE
        ):
            raise ValueError("attention transport selection lock is invalid")

    @property
    def artifact_sha256(self) -> str:
        return _artifact_sha256(
            _LOCK_DOMAIN,
            {
                "a_development_artifact_sha256": (
                    self.a_development_artifact_sha256
                ),
                "a_receipt_sha256s": self.a_receipt_sha256s,
                "a_split_artifact_sha256": self.a_split_artifact_sha256,
                "b_decision_sha256": self.b_decision.artifact_sha256,
                "b_receipt_sha256s": self.b_receipt_sha256s,
                "b_split_artifact_sha256": self.b_split_artifact_sha256,
                "frozen_invocation_signatures": (
                    self.frozen_invocation_signatures
                ),
                "gates_artifact_sha256": self.gates_artifact_sha256,
                "selected_candidate_artifact_sha256": (
                    self.selected_candidate_artifact_sha256
                ),
                "selection_rule": self.selection_rule,
            },
        )


@dataclass(frozen=True, slots=True)
class AttentionTransportBSelection:
    a_development_artifact_sha256: str
    b_split_artifact_sha256: str
    b_receipt_sha256s: tuple[str, ...]
    gates_artifact_sha256: str
    decisions: tuple[AttentionTransportCandidateGateDecision, ...]
    selected_lock: AttentionTransportSelectionLock | None
    c_open_allowed: bool

    def __post_init__(self) -> None:
        passing = tuple(value for value in self.decisions if value.passed)
        lock_matches = (
            self.selected_lock is not None
            and self.selected_lock.a_development_artifact_sha256
            == self.a_development_artifact_sha256
            and self.selected_lock.b_split_artifact_sha256
            == self.b_split_artifact_sha256
            and self.selected_lock.b_receipt_sha256s
            == self.b_receipt_sha256s
            and self.selected_lock.gates_artifact_sha256
            == self.gates_artifact_sha256
            and self.selected_lock.b_decision.artifact_sha256
            in {value.artifact_sha256 for value in self.decisions}
        )
        if (
            not _is_sha256(self.a_development_artifact_sha256)
            or not _is_sha256(self.b_split_artifact_sha256)
            or not _is_sha256(self.gates_artifact_sha256)
            or not self.decisions
            or len({value.candidate_id for value in self.decisions})
            != len(self.decisions)
            or type(self.c_open_allowed) is not bool
            or self.c_open_allowed != (self.selected_lock is not None)
            or bool(passing) != self.c_open_allowed
            or (self.selected_lock is not None and not lock_matches)
        ):
            raise ValueError("B selection result is invalid")

    @property
    def artifact_sha256(self) -> str:
        return _artifact_sha256(
            _B_SELECTION_DOMAIN,
            {
                "a_development_artifact_sha256": (
                    self.a_development_artifact_sha256
                ),
                "b_split_artifact_sha256": self.b_split_artifact_sha256,
                "b_receipt_sha256s": self.b_receipt_sha256s,
                "c_open_allowed": self.c_open_allowed,
                "decision_sha256s": tuple(
                    value.artifact_sha256 for value in self.decisions
                ),
                "gates_artifact_sha256": self.gates_artifact_sha256,
                "selected_lock_sha256": (
                    self.selected_lock.artifact_sha256
                    if self.selected_lock is not None
                    else None
                ),
            },
        )


def select_attention_transport_candidate_on_b(
    development: AttentionTransportADevelopment,
    b_records: Sequence[AttentionTransportSplitRecord],
    *,
    gates: AttentionTransportSelectionGates,
) -> AttentionTransportBSelection:
    """Evaluate frozen A artifacts on B and lock one passing candidate."""

    if not isinstance(development, AttentionTransportADevelopment):
        raise TypeError("development must be an A-development artifact")
    if not isinstance(gates, AttentionTransportSelectionGates):
        raise TypeError("gates must be attention transport selection gates")
    if tuple(value.artifact_sha256 for value in development.candidates) != (
        development.candidate_artifact_sha256s
    ):
        raise ValueError("one A-fitted candidate changed before B selection")
    panel = _open_split_panel(b_records, role="B")
    _validate_panel_lineage(
        panel,
        signatures=development.frozen_invocation_signatures,
        prior_receipts=development.a_receipt_sha256s,
    )
    decisions = tuple(
        _gate_candidate(
            candidate,
            _evaluate_candidate_on_panel(candidate, panel),
            gates,
        )
        for candidate in development.candidates
    )
    passing = [
        (candidate, decision)
        for candidate, decision in zip(
            development.candidates,
            decisions,
            strict=True,
        )
        if decision.passed
    ]
    lock = None
    if passing:
        selected_candidate, selected_decision = min(
            passing,
            key=lambda value: (
                value[1].evaluation.relative_fisher_loss,
                value[1].evaluation.resources.compiled_parameters,
                value[0].config.bank.generator_count,
                value[0].config.bank.generator_rank,
                value[0].config.router.active_count,
                value[0].config.candidate_id,
            ),
        )
        lock = AttentionTransportSelectionLock(
            selected_candidate=selected_candidate,
            selected_candidate_artifact_sha256=(
                selected_candidate.artifact_sha256
            ),
            gates=gates,
            gates_artifact_sha256=gates.artifact_sha256,
            b_decision=selected_decision,
            a_development_artifact_sha256=development.artifact_sha256,
            a_split_artifact_sha256=development.a_split_artifact_sha256,
            b_split_artifact_sha256=panel.artifact_sha256,
            a_receipt_sha256s=development.a_receipt_sha256s,
            b_receipt_sha256s=panel.receipt_sha256s,
            frozen_invocation_signatures=(
                development.frozen_invocation_signatures
            ),
        )
    return AttentionTransportBSelection(
        a_development_artifact_sha256=development.artifact_sha256,
        b_split_artifact_sha256=panel.artifact_sha256,
        b_receipt_sha256s=panel.receipt_sha256s,
        gates_artifact_sha256=gates.artifact_sha256,
        decisions=decisions,
        selected_lock=lock,
        c_open_allowed=lock is not None,
    )


@dataclass(frozen=True, slots=True)
class AttentionTransportConfirmationReport:
    selection_lock_artifact_sha256: str
    selected_candidate_id: str
    selected_candidate_artifact_sha256: str
    b_evaluation_artifact_sha256: str
    c_receipt_sha256s: tuple[str, ...]
    c_evaluation: AttentionTransportAggregateEvaluation
    c_gate_decision: AttentionTransportCandidateGateDecision
    scientific_status: AttentionTransportScientificStatus = SCIENTIFIC_STATUS

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.selection_lock_artifact_sha256)
            or not isinstance(self.selected_candidate_id, str)
            or not self.selected_candidate_id
            or not _is_sha256(self.selected_candidate_artifact_sha256)
            or not _is_sha256(self.b_evaluation_artifact_sha256)
            or not self.c_receipt_sha256s
            or any(not _is_sha256(value) for value in self.c_receipt_sha256s)
            or len(set(self.c_receipt_sha256s)) != len(self.c_receipt_sha256s)
            or self.c_evaluation.role != "C"
            or self.c_gate_decision.evaluation is not self.c_evaluation
            or self.c_gate_decision.evaluation.role != "C"
            or self.c_gate_decision.candidate_id != self.selected_candidate_id
            or self.c_gate_decision.candidate_artifact_sha256
            != self.selected_candidate_artifact_sha256
            or self.scientific_status != SCIENTIFIC_STATUS
        ):
            raise ValueError("C confirmation report is invalid")

    @property
    def artifact_sha256(self) -> str:
        return _artifact_sha256(_REPORT_DOMAIN, self)


def evaluate_attention_transport_selection_on_c(
    selection_lock: AttentionTransportSelectionLock,
    c_records: Sequence[AttentionTransportSplitRecord],
) -> AttentionTransportConfirmationReport:
    """Evaluate the already-selected artifact once on C; never refit it."""

    if not isinstance(selection_lock, AttentionTransportSelectionLock):
        raise TypeError("C evaluation requires a selection lock")
    if (
        selection_lock.selected_candidate.artifact_sha256
        != selection_lock.selected_candidate_artifact_sha256
        or selection_lock.gates.artifact_sha256
        != selection_lock.gates_artifact_sha256
    ):
        raise ValueError("selected candidate or gates changed before C")
    panel = _open_split_panel(c_records, role="C")
    _validate_panel_lineage(
        panel,
        signatures=selection_lock.frozen_invocation_signatures,
        prior_receipts=(
            *selection_lock.a_receipt_sha256s,
            *selection_lock.b_receipt_sha256s,
        ),
    )
    evaluation = _evaluate_candidate_on_panel(
        selection_lock.selected_candidate,
        panel,
    )
    c_decision = _gate_candidate(
        selection_lock.selected_candidate,
        evaluation,
        selection_lock.gates,
    )
    return AttentionTransportConfirmationReport(
        selection_lock_artifact_sha256=selection_lock.artifact_sha256,
        selected_candidate_id=(
            selection_lock.selected_candidate.config.candidate_id
        ),
        selected_candidate_artifact_sha256=(
            selection_lock.selected_candidate_artifact_sha256
        ),
        b_evaluation_artifact_sha256=(
            selection_lock.b_decision.evaluation.artifact_sha256
        ),
        c_receipt_sha256s=panel.receipt_sha256s,
        c_evaluation=evaluation,
        c_gate_decision=c_decision,
    )


@dataclass(frozen=True, slots=True)
class AttentionTransportExperimentOutcome:
    development: AttentionTransportADevelopment
    selection: AttentionTransportBSelection
    confirmation: AttentionTransportConfirmationReport | None
    c_opened: bool
    scientific_status: AttentionTransportScientificStatus = SCIENTIFIC_STATUS

    def __post_init__(self) -> None:
        confirmation_matches = (
            self.confirmation is not None
            and self.selection.selected_lock is not None
            and self.confirmation.selection_lock_artifact_sha256
            == self.selection.selected_lock.artifact_sha256
            and self.confirmation.selected_candidate_id
            == self.selection.selected_lock.selected_candidate.config.candidate_id
            and self.confirmation.selected_candidate_artifact_sha256
            == self.selection.selected_lock.selected_candidate_artifact_sha256
            and self.confirmation.b_evaluation_artifact_sha256
            == self.selection.selected_lock.b_decision.evaluation.artifact_sha256
        )
        if (
            type(self.c_opened) is not bool
            or self.c_opened != (self.confirmation is not None)
            or self.c_opened != self.selection.c_open_allowed
            or self.selection.a_development_artifact_sha256
            != self.development.artifact_sha256
            or (self.confirmation is not None and not confirmation_matches)
            or self.scientific_status != SCIENTIFIC_STATUS
        ):
            raise ValueError("split-locked experiment outcome is inconsistent")


def run_split_locked_attention_transport_experiment(
    a_records: Sequence[AttentionTransportSplitRecord],
    b_records: Sequence[AttentionTransportSplitRecord],
    c_supplier: Callable[[], Sequence[AttentionTransportSplitRecord]],
    *,
    candidate_configs: Sequence[AttentionTransportCandidateConfig],
    gates: AttentionTransportSelectionGates,
) -> AttentionTransportExperimentOutcome:
    """Run A->B->C while leaving C unopened after a no-pass B result."""

    if not callable(c_supplier):
        raise TypeError("c_supplier must be callable")
    development = fit_attention_transport_candidates_on_a(
        a_records,
        candidate_configs,
    )
    selection = select_attention_transport_candidate_on_b(
        development,
        b_records,
        gates=gates,
    )
    if selection.selected_lock is None:
        return AttentionTransportExperimentOutcome(
            development=development,
            selection=selection,
            confirmation=None,
            c_opened=False,
        )
    c_records = c_supplier()
    confirmation = evaluate_attention_transport_selection_on_c(
        selection.selected_lock,
        c_records,
    )
    return AttentionTransportExperimentOutcome(
        development=development,
        selection=selection,
        confirmation=confirmation,
        c_opened=True,
    )


__all__ = [
    "AttentionTransportADevelopment",
    "AttentionTransportAggregateEvaluation",
    "AttentionTransportBSelection",
    "AttentionTransportCandidateConfig",
    "AttentionTransportCandidateGateDecision",
    "AttentionTransportConfirmationReport",
    "AttentionTransportExperimentOutcome",
    "AttentionTransportFittedCandidate",
    "AttentionTransportFrozenInvocationSignature",
    "AttentionTransportInvocationEvaluation",
    "AttentionTransportOutputNorm",
    "AttentionTransportPostNormMetrics",
    "AttentionTransportRows",
    "AttentionTransportScientificStatus",
    "AttentionTransportSelectionGates",
    "AttentionTransportSelectionLock",
    "AttentionTransportSplitRecord",
    "SCIENTIFIC_STATUS",
    "evaluate_attention_transport_selection_on_c",
    "fit_attention_transport_candidates_on_a",
    "run_split_locked_attention_transport_experiment",
    "select_attention_transport_candidate_on_b",
]
