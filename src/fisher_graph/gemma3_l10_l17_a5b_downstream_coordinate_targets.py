"""A5b bridge from downstream-selected coordinates to generator-fit rows.

A5a selects one joint coordinate vector per observed token inside the exact
four-node, rank-182 Layer-17 affine image.  Those oracle coordinates are not a
deployable generator target until they are split back into the frozen node
charts and paired with the compiled Layer-17 input and native Fisher weights.

This module performs only that deterministic, tensor-ephemeral translation.
It does not load a model, fit a generator, inspect an outer held family, or
serialize row identities.  The resulting :class:`AlignedFragmentRows` values
are directly consumable by ``fit_frozen_basis_coordinate_generators``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType

import torch
from torch import Tensor

from .computational_modes import ComputationalModeBasis
from .gemma3_l10_l17_a5_frozen_affine_capacity_oracle import (
    _tensor_sha256 as _a5_capacity_tensor_sha256,
    build_frozen_affine_image,
)
from .gemma3_layer17_family_lofo_protocol import (
    V8_FAMILY_LOFO_FAMILY_ALIASES,
)
from .gemma3_modal_generator_dev_experiment import LayerFragmentRows
from .gemma3_modal_generator_terminal_fanin import AlignedFragmentRows


__all__ = [
    "A5B_JOINT_COORDINATE_WIDTH",
    "GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_FORMAT_VERSION",
    "GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_SCHEMA",
    "A5bDownstreamCoordinateTargetBridge",
    "a5b_tensor_sha256",
    "build_a5b_downstream_coordinate_target_bridge",
]


GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5b_downstream_coordinate_targets"
)
GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_FORMAT_VERSION = 1
A5B_JOINT_COORDINATE_WIDTH = 182

_NODE_COUNT = 4
_TENSOR_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5b-tensor:v1\0"
_RECEIPT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5b-receipt:v1\0"
_EXAMPLE_ORDER_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5b-inner-example-order:v1\0"
)
_EXAMPLE_MEMBERSHIP_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5b-inner-membership:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROUNDTRIP_RTOL = 1.0e-11
_ROUNDTRIP_ATOL = 1.0e-11


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


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return any(_contains_tensor(child) for child in value)
    return False


def _canonical_matrix(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.layout is not torch.strided
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"{label} must be a finite contiguous CPU float64 matrix"
        )
    return value.detach()


def _canonical_weights(
    value: Tensor,
    *,
    rows: int,
    label: str,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.layout is not torch.strided
        or value.ndim != 1
        or value.shape != (rows,)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
        or bool((value < 0).any())
        or float(value.sum().item()) <= 0.0
    ):
        raise ValueError(
            f"{label} must be finite nonnegative CPU float64 rows with "
            "positive mass"
        )
    return value.detach()


def a5b_tensor_sha256(value: Tensor) -> str:
    """Return the canonical content identity for A5b-local row tensors."""

    if (
        not isinstance(value, Tensor)
        or value.layout is not torch.strided
        or not value.is_floating_point()
        or value.numel() <= 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("A5b tensor hash input must be finite and floating")
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": str(canonical.dtype),
                "shape": tuple(int(size) for size in canonical.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _ordered_node_catalog(
    bases_by_node: Mapping[str, ComputationalModeBasis],
    node_order: Sequence[str],
) -> tuple[str, ...]:
    if not isinstance(bases_by_node, Mapping):
        raise TypeError("bases_by_node must be a mapping")
    names = tuple(node_order)
    if (
        len(names) != _NODE_COUNT
        or len(set(names)) != _NODE_COUNT
        or set(names) != set(bases_by_node)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("A5b node order must exactly cover four unique bases")
    return names


def _validate_ownership(
    row_keys: tuple[tuple[str, int], ...],
    family_alias_by_example: Mapping[str, str],
    *,
    training_family_aliases: Sequence[str],
    held_family_alias: str,
    rows: int,
) -> tuple[tuple[str, ...], dict[str, str]]:
    if (
        type(row_keys) is not tuple
        or len(row_keys) != rows
        or len(set(row_keys)) != rows
        or any(
            type(key) is not tuple
            or len(key) != 2
            or not isinstance(key[0], str)
            or not key[0]
            or type(key[1]) is not int
            or key[1] < 0
            for key in row_keys
        )
    ):
        raise ValueError("row_keys must be unique valid keys for every row")
    if held_family_alias not in V8_FAMILY_LOFO_FAMILY_ALIASES:
        raise ValueError("held_family_alias is not a frozen V8 family alias")
    expected_training = tuple(
        alias
        for alias in V8_FAMILY_LOFO_FAMILY_ALIASES
        if alias != held_family_alias
    )
    training = tuple(training_family_aliases)
    if training != expected_training:
        raise ValueError("training families must be the exact outer complement")
    if not isinstance(family_alias_by_example, Mapping):
        raise TypeError("family_alias_by_example must be a mapping")
    ownership = dict(family_alias_by_example)
    examples = {example_id for example_id, _ in row_keys}
    if set(ownership) != examples or any(
        not isinstance(alias, str) or not alias for alias in ownership.values()
    ):
        raise ValueError("example ownership must exactly cover the row examples")
    if held_family_alias in ownership.values():
        raise ValueError("outer held-family data is forbidden in A5b targets")
    if set(ownership.values()) != set(training):
        raise ValueError("example ownership must cover exactly the training families")
    return training, ownership


def _example_split(
    row_keys: tuple[tuple[str, int], ...],
    ownership: Mapping[str, str],
    *,
    training_family_aliases: tuple[str, ...],
    inner_split_binding_sha256: str,
    inner_audit_examples_per_family: int,
) -> tuple[tuple[int, ...], tuple[int, ...], set[str], set[str]]:
    binding = _require_sha256(
        inner_split_binding_sha256,
        label="inner split binding",
    )
    if (
        type(inner_audit_examples_per_family) is not int
        or inner_audit_examples_per_family <= 0
    ):
        raise ValueError("inner_audit_examples_per_family must be positive")
    audit_examples: set[str] = set()
    for alias in training_family_aliases:
        examples = tuple(
            example_id
            for example_id, owner in ownership.items()
            if owner == alias
        )
        if len(examples) <= inner_audit_examples_per_family:
            raise ValueError(
                "every training family must retain fit examples after audit split"
            )
        ranked = tuple(
            sorted(
                examples,
                key=lambda example_id: (
                    _domain_sha256(
                        _EXAMPLE_ORDER_DOMAIN,
                        {
                            "inner_split_binding_sha256": binding,
                            "family_alias": alias,
                            "example_id": example_id,
                        },
                    ),
                    example_id,
                ),
            )
        )
        audit_examples.update(ranked[:inner_audit_examples_per_family])
    fit_examples = set(ownership) - audit_examples
    if (
        not fit_examples
        or not audit_examples
        or fit_examples & audit_examples
        or fit_examples | audit_examples != set(ownership)
    ):
        raise RuntimeError("inner example partition is not exact")
    fit_indices = tuple(
        index
        for index, (example_id, _) in enumerate(row_keys)
        if example_id in fit_examples
    )
    audit_indices = tuple(
        index
        for index, (example_id, _) in enumerate(row_keys)
        if example_id in audit_examples
    )
    if (
        not fit_indices
        or not audit_indices
        or set(fit_indices) & set(audit_indices)
        or set(fit_indices) | set(audit_indices) != set(range(len(row_keys)))
    ):
        raise RuntimeError("inner row partition is not exact")
    return fit_indices, audit_indices, fit_examples, audit_examples


def _index_and_normalize_aligned_rows(
    rows: AlignedFragmentRows,
    indices: tuple[int, ...],
    *,
    family_alias_by_example: Mapping[str, str],
    training_family_aliases: tuple[str, ...],
) -> AlignedFragmentRows:
    if not indices or len(indices) != len(set(indices)) or any(
        type(index) is not int or not 0 <= index < rows.observations
        for index in indices
    ):
        raise ValueError("aligned-row indices must be unique and in range")
    index = torch.tensor(indices, dtype=torch.long)
    fragment_ids = tuple(rows.rows_by_fragment)
    shared_inputs = rows.rows_by_fragment[fragment_ids[0]].inputs.index_select(
        0,
        index,
    )
    row_keys = tuple(rows.row_keys[value] for value in indices)
    sequences = len({example_id for example_id, _ in row_keys})
    family_count = len(training_family_aliases)
    normalized_fisher: dict[str, Tensor] = {}
    for fragment_id in fragment_ids:
        raw = rows.rows_by_fragment[fragment_id].fisher_weights.index_select(
            0,
            index,
        )
        normalized = torch.zeros_like(raw)
        for alias in training_family_aliases:
            family_index = torch.tensor(
                [
                    row_index
                    for row_index, (example_id, _) in enumerate(row_keys)
                    if family_alias_by_example[example_id] == alias
                ],
                dtype=torch.long,
            )
            if family_index.numel() <= 0:
                raise ValueError("inner role is missing a training family")
            current = raw.index_select(0, family_index)
            total = current.sum()
            if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
                raise ValueError("inner role family Fisher mass must be positive")
            normalized.index_copy_(
                0,
                family_index,
                current / total / family_count,
            )
        if not torch.allclose(
            normalized.sum(),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=0.0,
            atol=2.0e-12,
        ):
            raise RuntimeError("inner role Fisher normalization drifted")
        normalized_fisher[fragment_id] = normalized
    return AlignedFragmentRows(
        rows_by_fragment={
            fragment_id: LayerFragmentRows(
                inputs=shared_inputs,
                contributions=rows.rows_by_fragment[
                    fragment_id
                ].contributions.index_select(0, index),
                fisher_weights=normalized_fisher[fragment_id],
                sequences=sequences,
            )
            for fragment_id in fragment_ids
        },
        row_keys=row_keys,
    )


def _fisher_role_receipt(
    rows: AlignedFragmentRows,
    *,
    node_order: tuple[str, ...],
    fragment_id_by_node: Mapping[str, str],
    family_alias_by_example: Mapping[str, str],
    training_family_aliases: tuple[str, ...],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in node_order:
        fragment_id = fragment_id_by_node[name]
        weights = rows.rows_by_fragment[fragment_id].fisher_weights
        family_mass = {
            alias: float(
                weights.index_select(
                    0,
                    torch.tensor(
                        [
                            index
                            for index, (example_id, _) in enumerate(
                                rows.row_keys
                            )
                            if family_alias_by_example[example_id] == alias
                        ],
                        dtype=torch.long,
                    ),
                ).sum().item()
            )
            for alias in training_family_aliases
        }
        result[name] = {
            "fragment_id": fragment_id,
            "total_mass": float(weights.sum().item()),
            "family_mass_by_alias": family_mass,
            "unit_total_mass": math.isclose(
                float(weights.sum().item()),
                1.0,
                rel_tol=0.0,
                abs_tol=2.0e-12,
            ),
            "equal_family_mass": all(
                math.isclose(
                    value,
                    1.0 / len(training_family_aliases),
                    rel_tol=0.0,
                    abs_tol=2.0e-12,
                )
                for value in family_mass.values()
            ),
        }
    return result


def _validate_role_fisher_normalization(
    rows: AlignedFragmentRows,
    *,
    fragment_id_by_node: Mapping[str, str],
    node_order: tuple[str, ...],
    family_alias_by_example: Mapping[str, str],
    training_family_aliases: tuple[str, ...],
) -> None:
    receipt = _fisher_role_receipt(
        rows,
        node_order=node_order,
        fragment_id_by_node=fragment_id_by_node,
        family_alias_by_example=family_alias_by_example,
        training_family_aliases=training_family_aliases,
    )
    if any(
        value["unit_total_mass"] is not True
        or value["equal_family_mass"] is not True
        for value in receipt.values()
    ):
        raise ValueError("A5b inner role Fisher normalization is invalid")


def _row_tensor_hashes(rows: AlignedFragmentRows) -> dict[str, object]:
    return {
        "row_key_sha256": rows.row_key_sha256,
        "observations": rows.observations,
        "sequences": rows.sequences,
        "fragment_tensor_sha256s": {
            fragment_id: {
                "inputs_sha256": a5b_tensor_sha256(value.inputs),
                "contributions_sha256": a5b_tensor_sha256(
                    value.contributions
                ),
                "fisher_weights_sha256": a5b_tensor_sha256(
                    value.fisher_weights
                ),
            }
            for fragment_id, value in rows.rows_by_fragment.items()
        },
    }


def _membership_sha256(
    examples: set[str],
    ownership: Mapping[str, str],
    *,
    role: str,
    split_binding_sha256: str,
) -> str:
    return _domain_sha256(
        _EXAMPLE_MEMBERSHIP_DOMAIN,
        {
            "role": role,
            "inner_split_binding_sha256": split_binding_sha256,
            "ordered_members": tuple(
                sorted((example_id, ownership[example_id]) for example_id in examples)
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class A5bDownstreamCoordinateTargetBridge:
    """Ephemeral all/fit/audit rows plus a source-safe split receipt."""

    all_rows: AlignedFragmentRows
    fit_rows: AlignedFragmentRows
    audit_rows: AlignedFragmentRows
    node_order: tuple[str, ...]
    rank_by_node: tuple[int, ...]
    fragment_id_by_node: Mapping[str, str]
    family_alias_by_example: Mapping[str, str] = field(repr=False)
    training_family_aliases: tuple[str, ...]
    held_family_alias: str
    inner_split_binding_sha256: str
    inner_audit_examples_per_family: int
    authenticated_compiled_inputs_sha256: str
    authenticated_joint_coordinates_sha256: str
    basis_sha256_by_node: Mapping[str, str]
    mean_sha256_by_node: Mapping[str, str]
    encoder_sha256_by_node: Mapping[str, str]
    decoder_sha256_by_node: Mapping[str, str]
    joint_roundtrip_sha256: str
    summed_decoded_contribution_sha256: str
    roundtrip_max_abs_difference: float
    roundtrip_rms_difference: float
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.all_rows, "all_rows"),
            (self.fit_rows, "fit_rows"),
            (self.audit_rows, "audit_rows"),
        ):
            if not isinstance(value, AlignedFragmentRows):
                raise TypeError(f"{label} must be AlignedFragmentRows")
        if (
            type(self.node_order) is not tuple
            or len(self.node_order) != _NODE_COUNT
            or len(set(self.node_order)) != _NODE_COUNT
            or type(self.rank_by_node) is not tuple
            or len(self.rank_by_node) != _NODE_COUNT
            or any(type(rank) is not int or rank <= 0 for rank in self.rank_by_node)
            or sum(self.rank_by_node) != A5B_JOINT_COORDINATE_WIDTH
        ):
            raise ValueError("A5b node/rank catalog is invalid")
        fragments = dict(self.fragment_id_by_node)
        if (
            set(fragments) != set(self.node_order)
            or len(set(fragments.values())) != _NODE_COUNT
            or any(
                not isinstance(value, str) or not value
                for value in fragments.values()
            )
        ):
            raise ValueError("A5b fragment map must be one-to-one over nodes")
        fragment_order = tuple(fragments[name] for name in self.node_order)
        if any(
            tuple(rows.rows_by_fragment) != fragment_order
            for rows in (self.all_rows, self.fit_rows, self.audit_rows)
        ):
            raise ValueError("A5b aligned rows do not follow frozen node order")
        ownership = dict(self.family_alias_by_example)
        expected_examples = {example_id for example_id, _ in self.all_rows.row_keys}
        validated_training, validated_ownership = _validate_ownership(
            self.all_rows.row_keys,
            ownership,
            training_family_aliases=self.training_family_aliases,
            held_family_alias=self.held_family_alias,
            rows=self.all_rows.observations,
        )
        if (
            validated_training != self.training_family_aliases
            or validated_ownership != ownership
            or set(ownership) != expected_examples
        ):
            raise ValueError("A5b ownership differs from all rows")
        fit_examples = {example_id for example_id, _ in self.fit_rows.row_keys}
        audit_examples = {example_id for example_id, _ in self.audit_rows.row_keys}
        if (
            set(self.fit_rows.row_keys) & set(self.audit_rows.row_keys)
            or set(self.fit_rows.row_keys) | set(self.audit_rows.row_keys)
            != set(self.all_rows.row_keys)
            or len(self.fit_rows.row_keys) + len(self.audit_rows.row_keys)
            != len(self.all_rows.row_keys)
            or fit_examples & audit_examples
            or fit_examples | audit_examples != expected_examples
            or self.all_rows.sequences != len(expected_examples)
            or self.fit_rows.sequences != len(fit_examples)
            or self.audit_rows.sequences != len(audit_examples)
        ):
            raise ValueError("A5b inner fit/audit partition is not exact")
        for catalog, label in (
            (self.basis_sha256_by_node, "basis"),
            (self.mean_sha256_by_node, "mean"),
            (self.encoder_sha256_by_node, "encoder"),
            (self.decoder_sha256_by_node, "decoder"),
        ):
            copied = dict(catalog)
            if set(copied) != set(self.node_order):
                raise ValueError(f"A5b {label} hash catalog is incomplete")
            for digest in copied.values():
                _require_sha256(digest, label=f"A5b {label}")
            object.__setattr__(
                self,
                f"{label}_sha256_by_node",
                MappingProxyType(copied),
            )
        for digest, label in (
            (self.inner_split_binding_sha256, "inner split binding"),
            (self.authenticated_compiled_inputs_sha256, "compiled inputs"),
            (self.authenticated_joint_coordinates_sha256, "joint coordinates"),
            (self.joint_roundtrip_sha256, "joint roundtrip"),
            (
                self.summed_decoded_contribution_sha256,
                "summed decoded contribution",
            ),
        ):
            _require_sha256(digest, label=label)
        if (
            not math.isfinite(self.roundtrip_max_abs_difference)
            or self.roundtrip_max_abs_difference < 0.0
            or not math.isfinite(self.roundtrip_rms_difference)
            or self.roundtrip_rms_difference < 0.0
        ):
            raise ValueError("A5b roundtrip audit scalars are invalid")
        if self.held_family_alias in ownership.values():
            raise ValueError("outer held-family data is forbidden in A5b targets")
        if (
            type(self.inner_audit_examples_per_family) is not int
            or self.inner_audit_examples_per_family <= 0
            or any(
                sum(
                    ownership[example_id] == alias
                    for example_id in audit_examples
                )
                != self.inner_audit_examples_per_family
                for alias in self.training_family_aliases
            )
            or any(
                not any(
                    ownership[example_id] == alias
                    for example_id in fit_examples
                )
                for alias in self.training_family_aliases
            )
        ):
            raise ValueError("A5b audit split is not exact per training family")
        for role_rows in (self.fit_rows, self.audit_rows):
            _validate_role_fisher_normalization(
                role_rows,
                fragment_id_by_node=fragments,
                node_order=self.node_order,
                family_alias_by_example=ownership,
                training_family_aliases=self.training_family_aliases,
            )
        all_input_hashes = {
            a5b_tensor_sha256(value.inputs)
            for value in self.all_rows.rows_by_fragment.values()
        }
        if all_input_hashes != {self.authenticated_compiled_inputs_sha256}:
            raise ValueError("A5b all-row inputs differ from authentication")
        object.__setattr__(self, "fragment_id_by_node", MappingProxyType(fragments))
        object.__setattr__(
            self,
            "family_alias_by_example",
            MappingProxyType(ownership),
        )
        computed = _domain_sha256(_RECEIPT_DOMAIN, self._receipt_payload())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(self.receipt_sha256, label="A5b receipt") != computed:
            raise ValueError("A5b target bridge receipt hash mismatch")

    def _receipt_payload(self) -> dict[str, object]:
        fit_examples = {example_id for example_id, _ in self.fit_rows.row_keys}
        audit_examples = {example_id for example_id, _ in self.audit_rows.row_keys}
        return {
            "schema": GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_SCHEMA,
            "format_version": (
                GEMMA3_L10_L17_A5B_DOWNSTREAM_COORDINATE_TARGETS_FORMAT_VERSION
            ),
            "scientific_role": (
                "calibration_a_fit_outer_training_downstream_coordinate_targets"
            ),
            "source_safe": True,
            "contains_tensors": False,
            "contains_prompt_text": False,
            "contains_prompt_identities": False,
            "contains_token_ids": False,
            "heldout_confirmation": False,
            "outer_split": {
                "training_family_aliases": self.training_family_aliases,
                "held_family_alias": self.held_family_alias,
                "held_family_rows_accepted": False,
                "held_family_used_for_fit_or_audit": False,
            },
            "authentication": {
                "compiled_inputs_sha256": (
                    self.authenticated_compiled_inputs_sha256
                ),
                "selected_joint_coordinates_sha256": (
                    self.authenticated_joint_coordinates_sha256
                ),
                "joint_coordinate_width": A5B_JOINT_COORDINATE_WIDTH,
            },
            "frozen_affine_image": {
                "node_order": self.node_order,
                "rank_by_node": self.rank_by_node,
                "coordinate_slices": {
                    name: {"start": start, "stop": start + rank, "rank": rank}
                    for name, rank, start in zip(
                        self.node_order,
                        self.rank_by_node,
                        _coordinate_starts(self.rank_by_node),
                        strict=True,
                    )
                },
                "fragment_id_by_node": dict(self.fragment_id_by_node),
                "basis_sha256_by_node": dict(self.basis_sha256_by_node),
                "mean_sha256_by_node": dict(self.mean_sha256_by_node),
                "encoder_sha256_by_node": dict(self.encoder_sha256_by_node),
                "decoder_sha256_by_node": dict(self.decoder_sha256_by_node),
                "basis_artifacts_unchanged_after_decode": True,
                "mean_tensors_byte_identical_after_decode": True,
                "encoder_tensors_byte_identical_after_decode": True,
                "decoder_tensors_byte_identical_after_decode": True,
            },
            "joint_roundtrip_audit": {
                "definition": (
                    "sum_node_decode(coordinate_slice)_equals_"
                    "sum_node_mean_plus_joint_coordinate_times_"
                    "concatenated_frozen_decoder"
                ),
                "joint_roundtrip_sha256": self.joint_roundtrip_sha256,
                "summed_decoded_contribution_sha256": (
                    self.summed_decoded_contribution_sha256
                ),
                "max_abs_difference": self.roundtrip_max_abs_difference,
                "rms_difference": self.roundtrip_rms_difference,
                "relative_tolerance": _ROUNDTRIP_RTOL,
                "absolute_tolerance": _ROUNDTRIP_ATOL,
                "passed": True,
            },
            "inner_split": {
                "method": (
                    "domain_separated_hash_rank_per_training_family_"
                    "then_preserve_source_row_order"
                ),
                "inner_split_binding_sha256": self.inner_split_binding_sha256,
                "inner_audit_examples_per_family": (
                    self.inner_audit_examples_per_family
                ),
                "fit_example_count": len(fit_examples),
                "audit_example_count": len(audit_examples),
                "fit_example_membership_sha256": _membership_sha256(
                    fit_examples,
                    self.family_alias_by_example,
                    role="fit",
                    split_binding_sha256=self.inner_split_binding_sha256,
                ),
                "audit_example_membership_sha256": _membership_sha256(
                    audit_examples,
                    self.family_alias_by_example,
                    role="audit",
                    split_binding_sha256=self.inner_split_binding_sha256,
                ),
                "row_overlap_count": len(
                    set(self.fit_rows.row_keys) & set(self.audit_rows.row_keys)
                ),
                "example_overlap_count": len(fit_examples & audit_examples),
                "rows_exactly_partitioned": True,
                "examples_exactly_partitioned": True,
            },
            "fisher_normalization": {
                "all_rows_preserve_raw_authenticated_fisher_weights": True,
                "fit_and_audit_normalized_independently": True,
                "audit_weights_influence_fit_normalization": False,
                "policy": (
                    "equal_total_mass_per_outer_training_family_per_role_"
                    "and_node"
                ),
                "training_family_count": len(self.training_family_aliases),
                "target_total_mass_per_role_and_node": 1.0,
                "target_mass_per_family": 1.0
                / len(self.training_family_aliases),
                "fit_by_node": _fisher_role_receipt(
                    self.fit_rows,
                    node_order=self.node_order,
                    fragment_id_by_node=self.fragment_id_by_node,
                    family_alias_by_example=self.family_alias_by_example,
                    training_family_aliases=self.training_family_aliases,
                ),
                "audit_by_node": _fisher_role_receipt(
                    self.audit_rows,
                    node_order=self.node_order,
                    fragment_id_by_node=self.fragment_id_by_node,
                    family_alias_by_example=self.family_alias_by_example,
                    training_family_aliases=self.training_family_aliases,
                ),
            },
            "row_accounting": {
                "all": _row_tensor_hashes(self.all_rows),
                "fit": _row_tensor_hashes(self.fit_rows),
                "audit": _row_tensor_hashes(self.audit_rows),
                "all_observations_equal_fit_plus_audit": (
                    self.all_rows.observations
                    == self.fit_rows.observations + self.audit_rows.observations
                ),
                "all_examples_equal_fit_plus_audit": (
                    self.all_rows.sequences
                    == self.fit_rows.sequences + self.audit_rows.sequences
                ),
            },
            "consumer_contract": {
                "compatible_with": "fit_frozen_basis_coordinate_generators",
                "contribution_target": (
                    "frozen_basis_decode_of_downstream_selected_coordinate_slice"
                ),
                "generator_fit_performed": False,
            },
        }

    def receipt(self) -> dict[str, object]:
        payload = self._receipt_payload()
        if _contains_tensor(payload):
            raise RuntimeError("A5b target receipt contains a Tensor")
        computed = _domain_sha256(_RECEIPT_DOMAIN, payload)
        if computed != self.receipt_sha256:
            raise RuntimeError("A5b target rows or receipt lineage drifted")
        return {**payload, "receipt_sha256": self.receipt_sha256}

    def metadata(self) -> dict[str, object]:
        return self.receipt()


def _coordinate_starts(ranks: Sequence[int]) -> tuple[int, ...]:
    starts: list[int] = []
    offset = 0
    for rank in ranks:
        starts.append(offset)
        offset += rank
    return tuple(starts)


def build_a5b_downstream_coordinate_target_bridge(
    *,
    compiled_inputs: Tensor,
    authenticated_compiled_inputs_sha256: str,
    selected_joint_coordinates: Tensor,
    authenticated_joint_coordinates_sha256: str,
    bases_by_node: Mapping[str, ComputationalModeBasis],
    node_order: Sequence[str],
    fisher_weights_by_node: Mapping[str, Tensor],
    fragment_id_by_node: Mapping[str, str],
    row_keys: tuple[tuple[str, int], ...],
    family_alias_by_example: Mapping[str, str],
    training_family_aliases: Sequence[str],
    held_family_alias: str,
    inner_split_binding_sha256: str,
    inner_audit_examples_per_family: int,
) -> A5bDownstreamCoordinateTargetBridge:
    """Decode authenticated joint coordinates and partition generator rows."""

    inputs = _canonical_matrix(compiled_inputs, label="compiled inputs")
    coordinates = _canonical_matrix(
        selected_joint_coordinates,
        label="selected joint coordinates",
    )
    expected_input_sha256 = _require_sha256(
        authenticated_compiled_inputs_sha256,
        label="authenticated compiled inputs",
    )
    expected_coordinate_sha256 = _require_sha256(
        authenticated_joint_coordinates_sha256,
        label="authenticated joint coordinates",
    )
    if a5b_tensor_sha256(inputs) != expected_input_sha256:
        raise ValueError("compiled inputs differ from their authenticated hash")
    # The batched capacity solver emits ``selected_coefficient_sha256`` with
    # A5a's tensor domain.  Verify that exact upstream identity rather than a
    # new bridge-local digest.
    if _a5_capacity_tensor_sha256(coordinates) != expected_coordinate_sha256:
        raise ValueError(
            "selected joint coordinates differ from their authenticated hash"
        )
    if (
        coordinates.shape != (inputs.shape[0], A5B_JOINT_COORDINATE_WIDTH)
    ):
        raise ValueError("selected joint coordinates must have shape [rows, 182]")

    names = _ordered_node_catalog(bases_by_node, node_order)
    if not isinstance(fragment_id_by_node, Mapping) or set(
        fragment_id_by_node
    ) != set(names):
        raise ValueError("fragment map must exactly cover the frozen nodes")
    fragments = {name: fragment_id_by_node[name] for name in names}
    if len(set(fragments.values())) != _NODE_COUNT or any(
        not isinstance(value, str) or not value for value in fragments.values()
    ):
        raise ValueError("fragment map must be one-to-one")
    if not isinstance(fisher_weights_by_node, Mapping) or set(
        fisher_weights_by_node
    ) != set(names):
        raise ValueError("Fisher-weight catalog must exactly cover the nodes")
    fisher = {
        name: _canonical_weights(
            fisher_weights_by_node[name],
            rows=inputs.shape[0],
            label=f"{name} Fisher weights",
        )
        for name in names
    }
    training, ownership = _validate_ownership(
        row_keys,
        family_alias_by_example,
        training_family_aliases=training_family_aliases,
        held_family_alias=held_family_alias,
        rows=inputs.shape[0],
    )
    fit_indices, audit_indices, _, _ = _example_split(
        row_keys,
        ownership,
        training_family_aliases=training,
        inner_split_binding_sha256=inner_split_binding_sha256,
        inner_audit_examples_per_family=inner_audit_examples_per_family,
    )

    image = build_frozen_affine_image(bases_by_node, node_order=names)
    ranks = image.rank_by_node
    if sum(ranks) != A5B_JOINT_COORDINATE_WIDTH:
        raise RuntimeError("frozen affine image coordinate width drifted")
    basis_snapshots = {
        name: {
            "artifact_sha256": bases_by_node[name].artifact_sha256,
            "mean_sha256": bases_by_node[name].mean_bias_sha256,
            "encoder_sha256": bases_by_node[name].encoder_basis_sha256,
            "decoder_sha256": bases_by_node[name].decoder_basis_sha256,
            "mean": bases_by_node[name].mean_bias.clone(),
            "encoder": bases_by_node[name].encoder_basis.clone(),
            "decoder": bases_by_node[name].decoder_basis.clone(),
        }
        for name in names
    }
    contributions: dict[str, Tensor] = {}
    starts = _coordinate_starts(ranks)
    for name, rank, start in zip(names, ranks, starts, strict=True):
        basis = bases_by_node[name]
        current = coordinates[:, start : start + rank].contiguous()
        contributions[name] = basis.decode(current).to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
    for name in names:
        basis = bases_by_node[name]
        snapshot = basis_snapshots[name]
        basis.validate_integrity()
        if (
            basis.artifact_sha256 != snapshot["artifact_sha256"]
            or basis.mean_bias_sha256 != snapshot["mean_sha256"]
            or basis.encoder_basis_sha256 != snapshot["encoder_sha256"]
            or basis.decoder_basis_sha256 != snapshot["decoder_sha256"]
            or not torch.equal(basis.mean_bias, snapshot["mean"])
            or not torch.equal(basis.encoder_basis, snapshot["encoder"])
            or not torch.equal(basis.decoder_basis, snapshot["decoder"])
        ):
            raise RuntimeError("A5b decoding changed a frozen basis tensor")

    summed = torch.stack(
        tuple(contributions[name] for name in names),
        dim=0,
    ).sum(dim=0)
    joint_roundtrip = (image.mean_sum + coordinates @ image.decoder).contiguous()
    difference = summed - joint_roundtrip
    if not torch.allclose(
        summed,
        joint_roundtrip,
        rtol=_ROUNDTRIP_RTOL,
        atol=_ROUNDTRIP_ATOL,
    ):
        raise RuntimeError("joint coordinate roundtrip differs from node decodes")
    roundtrip_max_abs = float(difference.abs().max().item())
    roundtrip_rms = float(torch.sqrt(difference.square().mean()).item())

    all_examples = {example_id for example_id, _ in row_keys}
    all_rows = AlignedFragmentRows(
        rows_by_fragment={
            fragments[name]: LayerFragmentRows(
                inputs=inputs,
                contributions=contributions[name],
                fisher_weights=fisher[name],
                sequences=len(all_examples),
            )
            for name in names
        },
        row_keys=row_keys,
    )
    fit_rows = _index_and_normalize_aligned_rows(
        all_rows,
        fit_indices,
        family_alias_by_example=ownership,
        training_family_aliases=training,
    )
    audit_rows = _index_and_normalize_aligned_rows(
        all_rows,
        audit_indices,
        family_alias_by_example=ownership,
        training_family_aliases=training,
    )
    return A5bDownstreamCoordinateTargetBridge(
        all_rows=all_rows,
        fit_rows=fit_rows,
        audit_rows=audit_rows,
        node_order=names,
        rank_by_node=ranks,
        fragment_id_by_node=fragments,
        family_alias_by_example=ownership,
        training_family_aliases=training,
        held_family_alias=held_family_alias,
        inner_split_binding_sha256=inner_split_binding_sha256,
        inner_audit_examples_per_family=inner_audit_examples_per_family,
        authenticated_compiled_inputs_sha256=expected_input_sha256,
        authenticated_joint_coordinates_sha256=expected_coordinate_sha256,
        basis_sha256_by_node={
            name: str(basis_snapshots[name]["artifact_sha256"])
            for name in names
        },
        mean_sha256_by_node={
            name: str(basis_snapshots[name]["mean_sha256"])
            for name in names
        },
        encoder_sha256_by_node={
            name: str(basis_snapshots[name]["encoder_sha256"])
            for name in names
        },
        decoder_sha256_by_node={
            name: str(basis_snapshots[name]["decoder_sha256"])
            for name in names
        },
        joint_roundtrip_sha256=a5b_tensor_sha256(joint_roundtrip),
        summed_decoded_contribution_sha256=a5b_tensor_sha256(summed),
        roundtrip_max_abs_difference=roundtrip_max_abs,
        roundtrip_rms_difference=roundtrip_rms,
    )
