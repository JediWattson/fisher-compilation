"""Deterministic synthetic backend for compact-message qualification.

The fixture is intentionally not a scientific dataset and cannot support a
model claim.  It exists to exercise P1--P12 mechanics, physical tensor capture,
seam checks, compact representations, and paired completion models before an
independent D2 panel is acquired.  No repository corpus path or model cache is
read by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final

import numpy as np

from .fisher_compact_message_completion_protocol import (
    EvaluationRow,
    RowIndexPayload,
    ShardPlanEntry,
    build_row_index_payload,
)
from .fisher_compact_message_completion_splits import (
    SplitAssignment,
    SplitUnit,
    stratified_binary_split,
)


__all__ = [
    "SYNTHETIC_EVIDENCE_SCOPE",
    "SyntheticCampaignData",
    "SyntheticCampaignConfig",
    "SyntheticPartition",
    "SyntheticRowMetadata",
    "SyntheticSourceParameters",
    "build_synthetic_campaign",
    "exact_synthetic_mlp",
    "exact_synthetic_mlp_blocks",
    "synthetic_rmsnorm",
    "synthetic_row_derivation_protocol_body",
    "synthetic_row_derivation_protocol_sha256",
]


SYNTHETIC_EVIDENCE_SCOPE: Final = "synthetic_qualification_only"


def synthetic_row_derivation_protocol_body() -> dict[str, object]:
    """Return the exact public-fixture row derivation contract."""

    return {
        "tokenizer_identity_sha256": hashlib.sha256(
            b"fisher-cmc/synthetic-tokenizer/no-tokenizer/v1"
        ).hexdigest(),
        "causal_mask": "source positions 0..L-2; target=source+1",
        "canonical_sequence_identity": (
            "sha256 synthetic panel, unit, and sequence ordinal"
        ),
        "analysis_unit_commitment_format": "raw_sha256_32_bytes",
        "source_position_encoding": "uint64_be",
        "target_position_encoding": "uint64_be",
        "row_id_domain": "fisher-compact-message-completion/evaluation-row/v1",
        "row_ordering": "ascending_raw_row_id_bytes",
        "duplicate_policy": "reject",
        "row_set_domain": (
            "fisher-compact-message-completion/evaluation-row-set/v1"
        ),
        "ordered_sequence_domain": (
            "fisher-compact-message-completion/evaluation-row-sequence/v1"
        ),
    }


def synthetic_row_derivation_protocol_sha256() -> str:
    """Hash the semantic row protocol independently of artifact envelopes."""

    content = json.dumps(
        synthetic_row_derivation_protocol_body(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        b"fisher-cmc/synthetic-row-derivation-protocol/v1\0" + content
    ).hexdigest()


_ROW_PROTOCOL_HASH: Final = synthetic_row_derivation_protocol_sha256()


def _hash(domain: bytes, value: str) -> str:
    return hashlib.sha256(domain + b"\0" + value.encode("utf-8")).hexdigest()


def _orthonormal_columns(
    generator: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    matrix = generator.standard_normal((rows, columns), dtype=np.float64)
    q, _ = np.linalg.qr(matrix, mode="reduced")
    # Deterministically orient each column by its largest-magnitude loading.
    for index in range(q.shape[1]):
        column = q[:, index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            q[:, index] *= -1.0
    return q


def synthetic_rmsnorm(
    value: np.ndarray,
    weight: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Gemma-shaped RMS normalization over the final dimension."""

    array = np.asarray(value, dtype=np.float64)
    scale = np.asarray(weight, dtype=np.float64)
    if array.ndim < 1 or scale.shape != (array.shape[-1],):
        raise ValueError("RMSNorm value/weight shapes are incompatible")
    denominator = np.sqrt(np.mean(array * array, axis=-1, keepdims=True) + epsilon)
    return array / denominator * scale


def _silu(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -60.0, 60.0)
    return value / (1.0 + np.exp(-clipped))


def exact_synthetic_mlp(
    h_a: np.ndarray,
    parameters: "SyntheticSourceParameters",
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact synthetic pre-normalization MLP output and residual flow."""

    value = np.asarray(h_a, dtype=np.float64)
    blocks = exact_synthetic_mlp_blocks(value, parameters)
    q_m = blocks.sum(axis=-2)
    v_m = synthetic_rmsnorm(q_m, parameters.post_mlp_norm_weight)
    return q_m, v_m


def exact_synthetic_mlp_blocks(
    h_a: np.ndarray,
    parameters: "SyntheticSourceParameters",
) -> np.ndarray:
    """Return the exact contiguous-floor four-block MLP contributions."""

    value = np.asarray(h_a, dtype=np.float64)
    gate = value @ parameters.mlp_gate.T
    up = value @ parameters.mlp_up.T
    intermediate = _silu(gate) * up
    width = intermediate.shape[-1]
    boundaries = tuple((block * width) // 4 for block in range(5))
    blocks = []
    for block in range(4):
        start, end = boundaries[block], boundaries[block + 1]
        blocks.append(
            intermediate[..., start:end] @ parameters.mlp_down[:, start:end].T
        )
    return np.stack(blocks, axis=-2)


@dataclass(frozen=True, slots=True)
class SyntheticCampaignConfig:
    seed: int = 1701
    fit_analysis_units: int = 48
    d2_analysis_units: int = 48
    tune_analysis_units: int = 24
    fit_train_analysis_units: int = 38
    o1_calibration_units: int = 12
    sequences_per_unit: int = 1
    token_rows_per_sequence: int = 4
    chart_count: int = 4
    head_count: int = 4
    hidden_dimension: int = 64
    message_dimension: int = 64
    compact_dimension: int = 32
    interaction_rank: int = 8
    vocabulary_size: int = 32

    def __post_init__(self) -> None:
        integral = (
            self.fit_analysis_units,
            self.d2_analysis_units,
            self.tune_analysis_units,
            self.fit_train_analysis_units,
            self.o1_calibration_units,
            self.sequences_per_unit,
            self.token_rows_per_sequence,
            self.chart_count,
            self.head_count,
            self.hidden_dimension,
            self.message_dimension,
            self.compact_dimension,
            self.interaction_rank,
            self.vocabulary_size,
        )
        if any(type(value) is not int or value <= 0 for value in integral):
            raise ValueError("synthetic dimensions and counts must be positive integers")
        if self.fit_train_analysis_units >= self.fit_analysis_units:
            raise ValueError("fit-monitor must contain at least one analysis unit")
        if self.tune_analysis_units >= self.d2_analysis_units:
            raise ValueError("development-confirm must contain at least one unit")
        if self.o1_calibration_units >= self.fit_analysis_units:
            raise ValueError("O1 validation must contain at least one unit")
        if self.head_count != 4:
            raise ValueError("the frozen synthetic fixture exercises four heads")
        if self.chart_count != 4:
            raise ValueError("the frozen synthetic fixture exercises four charts")
        if self.message_dimension < 64:
            raise ValueError("message dimension must satisfy the p=32 architecture gate")
        if self.compact_dimension != 32 or self.interaction_rank != 8:
            raise ValueError("the primary synthetic representation is frozen at p=32, b=8")
        if self.hidden_dimension < self.compact_dimension:
            raise ValueError("hidden dimension must support the compact carrier")


@dataclass(frozen=True, slots=True)
class SyntheticSourceParameters:
    activation_basis: np.ndarray
    chart_offsets: np.ndarray
    message_base_basis: np.ndarray
    message_interaction_basis: np.ndarray
    output_weights: np.ndarray
    q_carrier: np.ndarray
    interaction_z: np.ndarray
    interaction_r: np.ndarray
    post_attention_norm_weight: np.ndarray
    mlp_gate: np.ndarray
    mlp_up: np.ndarray
    mlp_down: np.ndarray
    post_mlp_norm_weight: np.ndarray
    lm_head: np.ndarray
    h4_fisher_diagonal: np.ndarray


@dataclass(frozen=True, slots=True)
class SyntheticRowMetadata:
    row_id: str
    partition_role: str
    analysis_unit_commitment: str
    canonical_sequence_hash: str
    token_position: int
    family_id: str
    template_id: str
    chart_id: int


@dataclass(frozen=True, slots=True)
class SyntheticPartition:
    partition_role: str
    row_index: RowIndexPayload
    metadata: tuple[SyntheticRowMetadata, ...]
    tensors: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        row_ids = self.row_index.row_ids
        if tuple(item.row_id for item in self.metadata) != row_ids:
            raise ValueError("synthetic metadata is not aligned to canonical row order")
        for name, tensor in self.tensors.items():
            if not isinstance(tensor, np.ndarray) or tensor.shape[0] != len(row_ids):
                raise ValueError(f"synthetic tensor {name!r} is row-misaligned")

    @property
    def analysis_unit_commitments(self) -> tuple[str, ...]:
        return tuple(sorted({item.analysis_unit_commitment for item in self.metadata}))

    def tensors_for_shard(self, entry: ShardPlanEntry) -> dict[str, np.ndarray]:
        if entry.partition_role != self.partition_role:
            raise ValueError("shard plan belongs to a different partition")
        expected = self.row_index.row_ids[
            entry.start_row_ordinal:entry.end_row_ordinal_exclusive
        ]
        if expected != entry.row_ids:
            raise ValueError("shard entry does not slice the synthetic row index")
        return {
            name: np.ascontiguousarray(
                value[entry.start_row_ordinal:entry.end_row_ordinal_exclusive]
            )
            for name, value in self.tensors.items()
        }


@dataclass(frozen=True, slots=True)
class SyntheticCampaignData:
    config: SyntheticCampaignConfig
    source: SyntheticSourceParameters
    fit: SyntheticPartition
    tune: SyntheticPartition
    development_confirm: SyntheticPartition
    fit_split: SplitAssignment
    d2_split: SplitAssignment
    o1_split: SplitAssignment

    def partition(self, role: str) -> SyntheticPartition:
        if role == "fit":
            return self.fit
        if role == "tune":
            return self.tune
        if role == "development_confirm":
            return self.development_confirm
        raise KeyError(role)


@dataclass(frozen=True, slots=True)
class _UnitDescriptor:
    source_panel: str
    source_index: int
    split_unit: SplitUnit
    chart_id: int


def _unit_descriptors(panel: str, count: int) -> tuple[_UnitDescriptor, ...]:
    result: list[_UnitDescriptor] = []
    for index in range(count):
        family = f"family-{index % 4}"
        template = f"template-{index % 4}-{(index // 4) % 2}"
        commitment = _hash(
            b"fisher-cmc/synthetic-analysis-unit/v1", f"{panel}:{index}"
        )
        result.append(
            _UnitDescriptor(
                source_panel=panel,
                source_index=index,
                split_unit=SplitUnit(commitment, family, template),
                chart_id=index % 4,
            )
        )
    return tuple(result)


def _source_parameters(config: SyntheticCampaignConfig) -> SyntheticSourceParameters:
    generator = np.random.Generator(np.random.PCG64(config.seed))
    h = config.hidden_dimension
    p = config.compact_dimension
    m = config.message_dimension
    heads = config.head_count
    b = config.interaction_rank

    activation_basis = _orthonormal_columns(generator, h, p)
    # The frozen H4 carrier has full rank 64 in this fixture.  Synthetic q
    # contributions occupy its first 32 dense directions so both rungs of the
    # preregistered 32/64 codec ladder are executable without manufacturing
    # structural zeros that would distort the E11 median-scale tolerance.
    q_carrier = _orthonormal_columns(generator, h, h)
    chart_offsets = np.zeros((config.chart_count, p), dtype=np.float64)
    for chart in range(config.chart_count):
        chart_offsets[chart, chart] = 5.0

    base_basis = np.empty((heads, m, p), dtype=np.float64)
    interaction_basis = np.empty((heads, m, b), dtype=np.float64)
    output_weights = np.empty((heads, h, m), dtype=np.float64)
    interaction_z = np.empty((heads, b, p), dtype=np.float64)
    interaction_r = np.empty((heads, b, p), dtype=np.float64)
    for head in range(heads):
        full = _orthonormal_columns(generator, m, p + b)
        base_basis[head] = full[:, :p]
        interaction_basis[head] = full[:, p:]
        base_map = np.eye(p, dtype=np.float64)
        interaction_map = generator.standard_normal((p, b)) / math.sqrt(b)
        # Keep the synthetic float32 q capture inside the literal 1e-7 H7
        # full-rank hard-cap floor on disjoint fit-monitor rows.
        output_weights[head] = 0.8 * q_carrier[:, :p] @ (
            base_map @ base_basis[head].T
            + 0.55 * interaction_map @ interaction_basis[head].T
        )
        interaction_z[head] = generator.standard_normal((b, p)) / math.sqrt(p)
        interaction_r[head] = generator.standard_normal((b, p)) / math.sqrt(p)

    mlp_width = max(96, h)
    mlp_gate = generator.standard_normal((mlp_width, h)) / math.sqrt(h)
    mlp_up = generator.standard_normal((mlp_width, h)) / math.sqrt(h)
    mlp_down = generator.standard_normal((h, mlp_width)) / math.sqrt(mlp_width)
    lm_head = generator.standard_normal((config.vocabulary_size, h)) / math.sqrt(h)
    fisher = np.linspace(0.5, 1.5, h, dtype=np.float64)
    return SyntheticSourceParameters(
        activation_basis=activation_basis,
        chart_offsets=chart_offsets,
        message_base_basis=base_basis,
        message_interaction_basis=interaction_basis,
        output_weights=output_weights,
        q_carrier=q_carrier,
        interaction_z=interaction_z,
        interaction_r=interaction_r,
        post_attention_norm_weight=np.linspace(0.9, 1.1, h, dtype=np.float64),
        mlp_gate=mlp_gate,
        mlp_up=mlp_up,
        mlp_down=mlp_down,
        post_mlp_norm_weight=np.linspace(1.1, 0.9, h, dtype=np.float64),
        lm_head=lm_head,
        h4_fisher_diagonal=fisher,
    )


def _row_values(
    descriptor: _UnitDescriptor,
    sequence_index: int,
    token_position: int,
    config: SyntheticCampaignConfig,
    source: SyntheticSourceParameters,
) -> dict[str, np.ndarray]:
    # Use a small, panel-independent activation design grid so the frozen
    # 32-dimensional 99th-percentile support estimator can be qualified with
    # the protocol's fixed 38 fit-train units.  This deliberately repeated
    # public-fixture design is mechanical test data, never independence or
    # scientific evidence; row and unit identities remain distinct.
    activation_seed = int.from_bytes(
        hashlib.sha256(
            b"fisher-cmc/synthetic-activation-grid/v1\0"
            + descriptor.chart_id.to_bytes(4, "big")
            + ((descriptor.source_index // config.chart_count) % 4).to_bytes(
                4, "big"
            )
            + sequence_index.to_bytes(4, "big")
            + token_position.to_bytes(4, "big")
        ).digest()[:8],
        "big",
    )
    activation_generator = np.random.Generator(np.random.PCG64(activation_seed))
    p = config.compact_dimension
    heads = config.head_count

    z = activation_generator.standard_normal(p)
    z += source.chart_offsets[descriptor.chart_id]
    h = source.activation_basis @ z + 0.02 * activation_generator.standard_normal(
        config.hidden_dimension
    )
    message = np.empty((heads, config.message_dimension), dtype=np.float64)
    for head in range(heads):
        r = activation_generator.standard_normal(p)
        interaction = (
            source.interaction_z[head] @ z
        ) * (source.interaction_r[head] @ r)
        message[head] = (
            source.message_base_basis[head] @ r
            + 0.20 * source.message_interaction_basis[head] @ interaction
        )
    # Every named capture point is rounded before it feeds the next synthetic
    # operation.  This mirrors a real forward capture and, critically, makes
    # the stored message the exact source used by the mandatory full-rank
    # message -> W_O -> q identity smoke.
    h_capture = h.astype(np.float32)
    message_capture = message.astype(np.float32)
    q_heads = np.stack(
        [
            source.output_weights[head]
            @ message_capture[head].astype(np.float64)
            for head in range(heads)
        ],
        axis=0,
    )
    q_heads_capture = q_heads.astype(np.float32)
    q_a = np.sum(q_heads_capture.astype(np.float64), axis=0)
    q_a_capture = q_a.astype(np.float32)
    attention_denominator = math.sqrt(float(np.mean(q_a * q_a)) + 1e-6)
    v_a_heads = (
        q_heads_capture.astype(np.float64)
        / attention_denominator
        * source.post_attention_norm_weight
    )
    v_a_heads_capture = v_a_heads.astype(np.float32)
    v_a = np.sum(v_a_heads_capture.astype(np.float64), axis=0)
    v_a_capture = v_a.astype(np.float32)
    h_a = h_capture.astype(np.float64) + v_a_capture.astype(np.float64)
    h_a_capture = h_a.astype(np.float32)
    q_m_blocks = exact_synthetic_mlp_blocks(h_a_capture, source)
    q_m_blocks_capture = q_m_blocks.astype(np.float32)
    q_m = np.sum(q_m_blocks_capture.astype(np.float64), axis=0)
    q_m_capture = q_m.astype(np.float32)
    mlp_denominator = math.sqrt(float(np.mean(q_m * q_m)) + 1e-6)
    v_m_blocks = (
        q_m_blocks_capture.astype(np.float64)
        / mlp_denominator
        * source.post_mlp_norm_weight
    )
    v_m_blocks_capture = v_m_blocks.astype(np.float32)
    v_m = np.sum(v_m_blocks_capture.astype(np.float64), axis=0)
    v_m_capture = v_m.astype(np.float32)
    h_plus = h_a_capture.astype(np.float64) + v_m_capture.astype(np.float64)
    h_plus_capture = h_plus.astype(np.float32)
    logits = source.lm_head @ h_plus_capture.astype(np.float64)
    target = np.asarray(int(np.argmax(logits)), dtype=np.int64)
    return {
        "h": h_capture,
        "h_A": h_a_capture,
        "h_plus": h_plus_capture,
        "h4_fisher_diagonal": source.h4_fisher_diagonal.astype(np.float32),
        "message": message_capture,
        "q_A": q_a_capture,
        "q_A_head": q_heads_capture,
        "q_M": q_m_capture,
        "q_M_block": q_m_blocks_capture,
        "target_token_id": target,
        "v_A": v_a_capture,
        "v_A_head": v_a_heads_capture,
        "v_M": v_m_capture,
        "v_M_block": v_m_blocks_capture,
    }


def _partition(
    role: str,
    descriptors: tuple[_UnitDescriptor, ...],
    config: SyntheticCampaignConfig,
    source: SyntheticSourceParameters,
) -> SyntheticPartition:
    records: dict[str, tuple[EvaluationRow, SyntheticRowMetadata, dict[str, np.ndarray]]] = {}
    for descriptor in descriptors:
        unit = descriptor.split_unit
        for sequence_index in range(config.sequences_per_unit):
            sequence_hash = _hash(
                b"fisher-cmc/synthetic-sequence/v1",
                f"{descriptor.source_panel}:{descriptor.source_index}:{sequence_index}",
            )
            for token_position in range(config.token_rows_per_sequence):
                row = EvaluationRow(
                    analysis_unit_commitment=unit.analysis_unit_commitment,
                    canonical_sequence_hash=sequence_hash,
                    source_token_position=token_position,
                    target_token_position=token_position + 1,
                )
                metadata = SyntheticRowMetadata(
                    row_id=row.row_id,
                    partition_role=role,
                    analysis_unit_commitment=unit.analysis_unit_commitment,
                    canonical_sequence_hash=sequence_hash,
                    token_position=token_position,
                    family_id=unit.family_id,
                    template_id=unit.template_id or "",
                    chart_id=descriptor.chart_id,
                )
                if row.row_id in records:
                    raise AssertionError("synthetic row identity collision")
                records[row.row_id] = (
                    row,
                    metadata,
                    _row_values(
                        descriptor, sequence_index, token_position, config, source
                    ),
                )
    row_index = build_row_index_payload(
        role,
        _ROW_PROTOCOL_HASH,
        (record[0] for record in records.values()),
    )
    aligned = tuple(records[row_id] for row_id in row_index.row_ids)
    names = tuple(sorted(aligned[0][2]))
    tensors = {
        name: np.ascontiguousarray(
            np.stack([record[2][name] for record in aligned], axis=0)
        )
        for name in names
    }
    for value in tensors.values():
        value.setflags(write=False)
    return SyntheticPartition(
        partition_role=role,
        row_index=row_index,
        metadata=tuple(record[1] for record in aligned),
        tensors=tensors,
    )


def build_synthetic_campaign(
    config: SyntheticCampaignConfig | None = None,
) -> SyntheticCampaignData:
    """Build the complete deterministic, claim-ineligible qualification panel."""

    resolved = config or SyntheticCampaignConfig()
    source = _source_parameters(resolved)
    fit_descriptors = _unit_descriptors("fit", resolved.fit_analysis_units)
    d2_descriptors = _unit_descriptors("synthetic-d2", resolved.d2_analysis_units)

    fit_split = stratified_binary_split(
        (descriptor.split_unit for descriptor in fit_descriptors),
        purpose="A-fit-train-vs-fit-monitor",
        requested_left_count=(
            resolved.fit_analysis_units - resolved.fit_train_analysis_units
        ),
        left_name="fit_monitor",
        right_name="fit_train",
        independence_mode="mode_b_semantic_instance",
    )
    d2_split = stratified_binary_split(
        (descriptor.split_unit for descriptor in d2_descriptors),
        purpose="D-tune-vs-development-confirm",
        requested_left_count=resolved.tune_analysis_units,
        left_name="tune",
        right_name="development_confirm",
        independence_mode="mode_b_semantic_instance",
    )
    o1_split = stratified_binary_split(
        (descriptor.split_unit for descriptor in fit_descriptors),
        purpose="O1-calibration-vs-validation",
        requested_left_count=resolved.o1_calibration_units,
        left_name="calibration",
        right_name="validation",
        independence_mode="mode_b_semantic_instance",
        force_family_strata=True,
    )
    tune_set = set(d2_split.left_units)
    development_set = set(d2_split.right_units)
    tune_descriptors = tuple(
        descriptor
        for descriptor in d2_descriptors
        if descriptor.split_unit.analysis_unit_commitment in tune_set
    )
    development_descriptors = tuple(
        descriptor
        for descriptor in d2_descriptors
        if descriptor.split_unit.analysis_unit_commitment in development_set
    )
    return SyntheticCampaignData(
        config=resolved,
        source=source,
        fit=_partition("fit", fit_descriptors, resolved, source),
        tune=_partition("tune", tune_descriptors, resolved, source),
        development_confirm=_partition(
            "development_confirm", development_descriptors, resolved, source
        ),
        fit_split=fit_split,
        d2_split=d2_split,
        o1_split=o1_split,
    )
