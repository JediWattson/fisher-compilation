"""Pure leakage-safe breadth split for A5c generator rows.

The A5b coordinate bridge produces one aligned row bank from the seven outer
training families.  A wider A5c fit needs more examples, but widening the bank
also makes a subtle leakage path more likely: different examples can contain
byte-identical compiled Layer-17 inputs.  An example-disjoint split alone does
not prevent those exact input rows from appearing on both sides.

This module is deliberately independent of model loading, target solving, and
generator fitting.  It:

* requires at least four captured examples per training family;
* chooses audit examples by a domain-separated deterministic hash rank;
* keeps the audit rows intact and quarantines every fit row whose exact
  compiled-input byte signature occurs in audit;
* requires zero post-quarantine signature overlap; and
* independently gives every family equal Fisher mass in fit and audit.

Raw tensors and row identities remain ephemeral.  The public receipt contains
only counts, booleans, family aliases, and domain-separated hash commitments.
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

from .gemma3_l10_l17_a5b_downstream_coordinate_targets import (
    A5bDownstreamCoordinateTargetBridge,
    a5b_tensor_sha256,
)
from .gemma3_modal_generator_dev_experiment import LayerFragmentRows
from .gemma3_modal_generator_terminal_fanin import AlignedFragmentRows


__all__ = [
    "A5C_MINIMUM_EXAMPLES_PER_FAMILY",
    "GEMMA3_L10_L17_A5C_BREADTH_SPLIT_FORMAT_VERSION",
    "GEMMA3_L10_L17_A5C_BREADTH_SPLIT_SCHEMA",
    "A5cBreadthSplit",
    "A5cCrossSplitInputPurge",
    "a5c_compiled_input_row_signatures",
    "build_a5c_breadth_split",
    "build_a5c_breadth_split_from_bridge",
    "purge_a5c_cross_split_input_signatures",
    "validate_a5c_breadth_split_receipt",
]


GEMMA3_L10_L17_A5C_BREADTH_SPLIT_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5c_breadth_split"
)
GEMMA3_L10_L17_A5C_BREADTH_SPLIT_FORMAT_VERSION = 1
A5C_MINIMUM_EXAMPLES_PER_FAMILY = 4

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-breadth-receipt:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-breadth-tensor:v1\0"
_ROW_SIGNATURE_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5c-compiled-input-row:v1\0"
)
_SIGNATURE_SET_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5c-signature-multiset:v1\0"
)
_EXAMPLE_ORDER_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5c-audit-example-order:v1\0"
)
_MEMBERSHIP_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-membership:v1\0"
_ROW_KEYS_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-row-keys:v1\0"
_INDICES_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-source-indices:v1\0"

_SPLIT_METHOD = (
    "domain_separated_hash_rank_per_family_then_preserve_source_row_order"
)
_SIGNATURE_DEFINITION = (
    "sha256_of_domain_plus_exact_row_dtype_shape_and_contiguous_cpu_bytes"
)
_COLLISION_POLICY = (
    "protect_audit_and_remove_every_initial_fit_row_whose_exact_compiled_"
    "input_signature_occurs_in_audit"
)
_FISHER_POLICY = (
    "normalize_fit_and_audit_independently_to_equal_total_mass_per_"
    "training_family_and_fragment"
)
_SAFETY = {
    "contains_tensors": False,
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "outer_held_family_rows_accepted": False,
    "outer_held_family_accessed": False,
    "heldout_confirmation": False,
    "generator_fit_performed": False,
    "model_loaded": False,
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
        value, (str, bytes, bytearray)
    ):
        return any(_contains_tensor(child) for child in value)
    return False


def _canonical_compiled_inputs(value: Tensor, *, rows: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.layout is not torch.strided
        or value.ndim != 2
        or value.shape[0] != rows
        or value.shape[1] <= 0
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            "compiled_inputs must be a finite contiguous CPU float64 "
            "matrix aligned to the rows"
        )
    return value.detach()


def _tensor_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.layout is not torch.strided
        or not value.is_floating_point()
        or value.numel() <= 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("A5c tensor hash input must be finite and floating")
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


def a5c_compiled_input_row_signatures(value: Tensor) -> tuple[str, ...]:
    """Hash each row using its exact dtype, width, and byte representation."""

    if (
        not isinstance(value, Tensor)
        or value.layout is not torch.strided
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("compiled inputs must be a finite floating matrix")
    canonical = value.detach().to(device="cpu").contiguous()
    result: list[str] = []
    for row in canonical:
        row = row.contiguous()
        digest = hashlib.sha256()
        digest.update(_ROW_SIGNATURE_DOMAIN)
        digest.update(
            _canonical_json_bytes(
                {
                    "dtype": str(row.dtype),
                    "shape": tuple(int(size) for size in row.shape),
                }
            )
        )
        digest.update(b"\0")
        digest.update(row.view(torch.uint8).numpy().tobytes(order="C"))
        result.append(digest.hexdigest())
    return tuple(result)


@dataclass(frozen=True, slots=True)
class A5cCrossSplitInputPurge:
    """Indices after conservatively removing exact audit inputs from fit."""

    initial_fit_indices: tuple[int, ...]
    audit_indices: tuple[int, ...]
    retained_fit_indices: tuple[int, ...]
    removed_fit_indices: tuple[int, ...]
    colliding_signature_count: int

    def __post_init__(self) -> None:
        for values, label in (
            (self.initial_fit_indices, "initial fit"),
            (self.audit_indices, "audit"),
            (self.retained_fit_indices, "retained fit"),
            (self.removed_fit_indices, "removed fit"),
        ):
            if (
                type(values) is not tuple
                or len(values) != len(set(values))
                or tuple(sorted(values)) != values
                or any(type(index) is not int or index < 0 for index in values)
            ):
                raise ValueError(f"{label} indices must be ordered and unique")
        if (
            set(self.initial_fit_indices) & set(self.audit_indices)
            or set(self.retained_fit_indices) & set(self.removed_fit_indices)
            or set(self.retained_fit_indices) | set(self.removed_fit_indices)
            != set(self.initial_fit_indices)
            or set(self.retained_fit_indices) & set(self.audit_indices)
            or type(self.colliding_signature_count) is not int
            or self.colliding_signature_count < 0
        ):
            raise ValueError("cross-split purge index partition is invalid")


def purge_a5c_cross_split_input_signatures(
    compiled_inputs: Tensor,
    fit_indices: Sequence[int],
    audit_indices: Sequence[int],
) -> A5cCrossSplitInputPurge:
    """Protect audit and purge byte-identical input signatures from fit.

    The inputs may represent a full example split or one family-disjoint CV
    fold.  Indices need not cover every row, but their two roles must be
    disjoint.  Source order is canonicalized so caller container order cannot
    influence the result.
    """

    signatures = a5c_compiled_input_row_signatures(compiled_inputs)
    raw_fit = tuple(fit_indices)
    raw_audit = tuple(audit_indices)
    if (
        not raw_fit
        or not raw_audit
        or any(
            type(index) is not int or not 0 <= index < len(signatures)
            for index in (*raw_fit, *raw_audit)
        )
    ):
        raise ValueError("fit and audit indices must be nonempty and disjoint")
    fit = tuple(sorted(raw_fit))
    audit = tuple(sorted(raw_audit))
    if (
        len(fit) != len(set(fit))
        or len(audit) != len(set(audit))
        or set(fit) & set(audit)
    ):
        raise ValueError("fit and audit indices must be nonempty and disjoint")
    audit_signatures = {signatures[index] for index in audit}
    colliding = {signatures[index] for index in fit} & audit_signatures
    removed = tuple(
        index for index in fit if signatures[index] in audit_signatures
    )
    retained = tuple(
        index for index in fit if signatures[index] not in audit_signatures
    )
    if {signatures[index] for index in retained} & audit_signatures:
        raise RuntimeError("A5c signature purge failed to remove overlap")
    return A5cCrossSplitInputPurge(
        initial_fit_indices=fit,
        audit_indices=audit,
        retained_fit_indices=retained,
        removed_fit_indices=removed,
        colliding_signature_count=len(colliding),
    )


def _signature_commitment(signatures: Sequence[str], *, role: str) -> str:
    values = tuple(signatures)
    if any(_SHA256.fullmatch(value) is None for value in values):
        raise ValueError("compiled-input signatures are invalid")
    return _domain_sha256(
        _SIGNATURE_SET_DOMAIN,
        {"role": role, "ordered_signature_multiset": tuple(sorted(values))},
    )


def _indices_sha256(indices: Sequence[int], *, role: str) -> str:
    values = tuple(indices)
    if any(type(index) is not int or index < 0 for index in values):
        raise ValueError("source indices are invalid")
    return _domain_sha256(
        _INDICES_DOMAIN,
        {"role": role, "ordered_source_indices": values},
    )


def _row_keys_sha256(
    row_keys: Sequence[tuple[str, int]], *, role: str
) -> str:
    return _domain_sha256(
        _ROW_KEYS_DOMAIN,
        {"role": role, "ordered_row_keys": tuple(row_keys)},
    )


def _membership_sha256(
    examples: set[str],
    ownership: Mapping[str, str],
    *,
    role: str,
    split_binding_sha256: str,
) -> str:
    return _domain_sha256(
        _MEMBERSHIP_DOMAIN,
        {
            "role": role,
            "split_binding_sha256": split_binding_sha256,
            "ordered_members": tuple(
                sorted((example_id, ownership[example_id]) for example_id in examples)
            ),
        },
    )


def _validate_row_keys_and_ownership(
    rows: AlignedFragmentRows,
    row_keys: tuple[tuple[str, int], ...],
    family_alias_by_example: Mapping[str, str],
    training_family_aliases: Sequence[str],
    outer_held_family_alias: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    if not isinstance(rows, AlignedFragmentRows):
        raise TypeError("rows must be AlignedFragmentRows")
    if type(row_keys) is not tuple or row_keys != rows.row_keys:
        raise ValueError("row_keys must exactly equal the aligned-row axis")
    aliases = tuple(training_family_aliases)
    if (
        not aliases
        or len(aliases) != len(set(aliases))
        or any(not isinstance(alias, str) or not alias for alias in aliases)
    ):
        raise ValueError("training_family_aliases must be unique and nonempty")
    if (
        not isinstance(outer_held_family_alias, str)
        or not outer_held_family_alias
        or outer_held_family_alias in aliases
    ):
        raise ValueError("outer held family must be nonempty and not training")
    if not isinstance(family_alias_by_example, Mapping):
        raise TypeError("family_alias_by_example must be a mapping")
    ownership = dict(family_alias_by_example)
    examples = {example_id for example_id, _ in row_keys}
    if (
        set(ownership) != examples
        or any(
            not isinstance(example_id, str)
            or not example_id
            or not isinstance(alias, str)
            or not alias
            for example_id, alias in ownership.items()
        )
        or set(ownership.values()) != set(aliases)
    ):
        raise ValueError(
            "ownership must exactly cover the row examples and training families"
        )
    if outer_held_family_alias in ownership.values():
        raise ValueError("outer held-family ownership is forbidden")
    return aliases, ownership


def _examples_by_family(
    ownership: Mapping[str, str], aliases: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    return {
        alias: tuple(
            sorted(
                example_id
                for example_id, owner in ownership.items()
                if owner == alias
            )
        )
        for alias in aliases
    }


def _initial_example_split(
    ownership: Mapping[str, str],
    aliases: tuple[str, ...],
    *,
    split_binding_sha256: str,
    audit_examples_per_family: int,
) -> tuple[set[str], set[str]]:
    binding = _require_sha256(split_binding_sha256, label="split binding")
    if type(audit_examples_per_family) is not int or audit_examples_per_family <= 0:
        raise ValueError("audit_examples_per_family must be positive")
    by_family = _examples_by_family(ownership, aliases)
    audit: set[str] = set()
    for alias in aliases:
        examples = by_family[alias]
        if len(examples) < A5C_MINIMUM_EXAMPLES_PER_FAMILY:
            raise ValueError(
                "A5c breadth requires at least four examples per training family"
            )
        if audit_examples_per_family >= len(examples):
            raise ValueError("the audit split must leave fit examples per family")
        ranked = sorted(
            examples,
            key=lambda example_id: (
                _domain_sha256(
                    _EXAMPLE_ORDER_DOMAIN,
                    {
                        "split_binding_sha256": binding,
                        "family_alias": alias,
                        "example_id": example_id,
                    },
                ),
                example_id,
            ),
        )
        audit.update(ranked[:audit_examples_per_family])
    fit = set(ownership) - audit
    if not fit or not audit or fit & audit or fit | audit != set(ownership):
        raise RuntimeError("initial A5c example partition is not exact")
    return fit, audit


def _indices_for_examples(
    row_keys: Sequence[tuple[str, int]], examples: set[str]
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (example_id, _) in enumerate(row_keys)
        if example_id in examples
    )


def _subset_equal_family_rows(
    rows: AlignedFragmentRows,
    indices: tuple[int, ...],
    *,
    ownership: Mapping[str, str],
    aliases: tuple[str, ...],
) -> AlignedFragmentRows:
    if (
        not indices
        or len(indices) != len(set(indices))
        or tuple(sorted(indices)) != indices
        or any(index < 0 or index >= rows.observations for index in indices)
    ):
        raise ValueError("row subset indices must be ordered, unique, and nonempty")
    index = torch.tensor(indices, dtype=torch.long)
    row_keys = tuple(rows.row_keys[offset] for offset in indices)
    present_examples = {example_id for example_id, _ in row_keys}
    if {ownership[example_id] for example_id in present_examples} != set(aliases):
        raise ValueError("post-purge rows must retain every training family")
    normalized: dict[str, LayerFragmentRows] = {}
    for fragment_id, fragment in rows.rows_by_fragment.items():
        raw = fragment.fisher_weights.index_select(0, index).contiguous()
        fisher = torch.zeros_like(raw)
        for alias in aliases:
            family_index = torch.tensor(
                [
                    offset
                    for offset, (example_id, _) in enumerate(row_keys)
                    if ownership[example_id] == alias
                ],
                dtype=torch.long,
            )
            if family_index.numel() <= 0:
                raise ValueError("row subset is missing a training family")
            current = raw.index_select(0, family_index)
            mass = current.sum()
            if not bool(torch.isfinite(mass)) or float(mass.item()) <= 0.0:
                raise ValueError("role-local family Fisher mass must be positive")
            fisher.index_copy_(
                0,
                family_index,
                current / mass / len(aliases),
            )
        if not torch.allclose(
            fisher.sum(),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=0.0,
            atol=2.0e-12,
        ):
            raise RuntimeError("equal-family Fisher normalization drifted")
        normalized[fragment_id] = LayerFragmentRows(
            inputs=fragment.inputs.index_select(0, index),
            contributions=fragment.contributions.index_select(0, index),
            fisher_weights=fisher,
            sequences=len(present_examples),
        )
    return AlignedFragmentRows(rows_by_fragment=normalized, row_keys=row_keys)


def _role_counts(
    row_keys: Sequence[tuple[str, int]],
    ownership: Mapping[str, str],
    aliases: Sequence[str],
) -> dict[str, object]:
    examples = {example_id for example_id, _ in row_keys}
    return {
        "observations": len(row_keys),
        "examples": len(examples),
        "observations_by_family": {
            alias: sum(
                ownership[example_id] == alias for example_id, _ in row_keys
            )
            for alias in aliases
        },
        "examples_by_family": {
            alias: sum(ownership[example_id] == alias for example_id in examples)
            for alias in aliases
        },
    }


def _fisher_receipt(
    rows: AlignedFragmentRows,
    ownership: Mapping[str, str],
    aliases: tuple[str, ...],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for fragment_id, fragment in rows.rows_by_fragment.items():
        family_mass: dict[str, float] = {}
        for alias in aliases:
            mask = torch.tensor(
                [
                    ownership[example_id] == alias
                    for example_id, _ in rows.row_keys
                ],
                dtype=torch.bool,
            )
            family_mass[alias] = float(fragment.fisher_weights[mask].sum().item())
        total = float(fragment.fisher_weights.sum().item())
        result[fragment_id] = {
            "fisher_weights_sha256": _tensor_sha256(fragment.fisher_weights),
            "total_mass": total,
            "family_mass_by_alias": family_mass,
            "unit_total_mass": math.isclose(
                total, 1.0, rel_tol=0.0, abs_tol=2.0e-12
            ),
            "equal_family_mass": all(
                math.isclose(
                    value,
                    1.0 / len(aliases),
                    rel_tol=0.0,
                    abs_tol=2.0e-12,
                )
                for value in family_mass.values()
            ),
        }
    return result


def _fragment_tensor_receipt(rows: AlignedFragmentRows) -> dict[str, object]:
    return {
        fragment_id: {
            "inputs_sha256": _tensor_sha256(fragment.inputs),
            "contributions_sha256": _tensor_sha256(fragment.contributions),
            "fisher_weights_sha256": _tensor_sha256(fragment.fisher_weights),
        }
        for fragment_id, fragment in rows.rows_by_fragment.items()
    }


@dataclass(frozen=True, slots=True)
class A5cBreadthSplit:
    """Executable in-memory split plus a tensor-free authenticated receipt."""

    all_rows: AlignedFragmentRows = field(repr=False)
    fit_rows: AlignedFragmentRows = field(repr=False)
    audit_rows: AlignedFragmentRows = field(repr=False)
    compiled_inputs: Tensor = field(repr=False)
    initial_fit_source_indices: tuple[int, ...]
    initial_audit_source_indices: tuple[int, ...]
    fit_source_indices: tuple[int, ...]
    audit_source_indices: tuple[int, ...]
    removed_fit_source_indices: tuple[int, ...]
    node_order: tuple[str, ...]
    fragment_id_by_node: Mapping[str, str]
    family_alias_by_example: Mapping[str, str] = field(repr=False)
    training_family_aliases: tuple[str, ...]
    outer_held_family_alias: str
    split_binding_sha256: str
    audit_examples_per_family: int
    bridge_compiled_inputs_sha256: str
    source_bridge_receipt_sha256: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        aliases, ownership = _validate_row_keys_and_ownership(
            self.all_rows,
            self.all_rows.row_keys,
            self.family_alias_by_example,
            self.training_family_aliases,
            self.outer_held_family_alias,
        )
        inputs = _canonical_compiled_inputs(
            self.compiled_inputs, rows=self.all_rows.observations
        )
        if any(
            not torch.equal(fragment.inputs, inputs)
            for fragment in self.all_rows.rows_by_fragment.values()
        ):
            raise ValueError("all fragment inputs must equal compiled_inputs exactly")
        fragments = dict(self.fragment_id_by_node)
        if (
            type(self.node_order) is not tuple
            or not self.node_order
            or len(self.node_order) != len(set(self.node_order))
            or any(not isinstance(name, str) or not name for name in self.node_order)
            or set(fragments) != set(self.node_order)
            or len(set(fragments.values())) != len(self.node_order)
            or tuple(fragments[name] for name in self.node_order)
            != tuple(self.all_rows.rows_by_fragment)
        ):
            raise ValueError("A5c node/fragment catalog differs from the row bank")
        if self.source_bridge_receipt_sha256 is not None:
            _require_sha256(
                self.source_bridge_receipt_sha256,
                label="source bridge receipt",
            )
        bridge_input_sha256 = _require_sha256(
            self.bridge_compiled_inputs_sha256,
            label="bridge-domain compiled inputs",
        )
        if bridge_input_sha256 != a5b_tensor_sha256(inputs):
            raise ValueError(
                "bridge-domain compiled-input identity differs from source rows"
            )
        _require_sha256(self.split_binding_sha256, label="split binding")
        initial_fit_examples, initial_audit_examples = _initial_example_split(
            ownership,
            aliases,
            split_binding_sha256=self.split_binding_sha256,
            audit_examples_per_family=self.audit_examples_per_family,
        )
        expected_initial_fit = _indices_for_examples(
            self.all_rows.row_keys, initial_fit_examples
        )
        expected_initial_audit = _indices_for_examples(
            self.all_rows.row_keys, initial_audit_examples
        )
        signatures = a5c_compiled_input_row_signatures(inputs)
        audit_signatures = {
            signatures[index] for index in expected_initial_audit
        }
        expected_removed = tuple(
            index
            for index in expected_initial_fit
            if signatures[index] in audit_signatures
        )
        expected_fit = tuple(
            index
            for index in expected_initial_fit
            if signatures[index] not in audit_signatures
        )
        if (
            self.initial_fit_source_indices != expected_initial_fit
            or self.initial_audit_source_indices != expected_initial_audit
            or self.removed_fit_source_indices != expected_removed
            or self.fit_source_indices != expected_fit
            or self.audit_source_indices != expected_initial_audit
        ):
            raise ValueError("A5c split source indices contradict the policy")
        expected_fit_keys = tuple(
            self.all_rows.row_keys[index] for index in expected_fit
        )
        expected_audit_keys = tuple(
            self.all_rows.row_keys[index] for index in expected_initial_audit
        )
        if (
            self.fit_rows.row_keys != expected_fit_keys
            or self.audit_rows.row_keys != expected_audit_keys
            or set(expected_fit_keys) & set(expected_audit_keys)
        ):
            raise ValueError("A5c retained rows contradict source alignment")
        fit_examples = {example_id for example_id, _ in expected_fit_keys}
        audit_examples = {example_id for example_id, _ in expected_audit_keys}
        if fit_examples & audit_examples:
            raise ValueError("A5c fit and audit examples overlap")
        if set(ownership[example_id] for example_id in fit_examples) != set(aliases):
            raise ValueError("signature quarantine removed a fit family")
        fit_signatures = {signatures[index] for index in expected_fit}
        retained_audit_signatures = {
            signatures[index] for index in expected_initial_audit
        }
        if fit_signatures & retained_audit_signatures:
            raise ValueError("A5c retained compiled-input signatures overlap")
        for rows, indices, label in (
            (self.fit_rows, expected_fit, "fit"),
            (self.audit_rows, expected_initial_audit, "audit"),
        ):
            index = torch.tensor(indices, dtype=torch.long)
            for fragment_id, fragment in rows.rows_by_fragment.items():
                source = self.all_rows.rows_by_fragment[fragment_id]
                if (
                    not torch.equal(
                        fragment.inputs, source.inputs.index_select(0, index)
                    )
                    or not torch.equal(
                        fragment.contributions,
                        source.contributions.index_select(0, index),
                    )
                ):
                    raise ValueError(f"A5c {label} tensors lost source alignment")
            fisher = _fisher_receipt(rows, ownership, aliases)
            if any(
                value["unit_total_mass"] is not True
                or value["equal_family_mass"] is not True
                for value in fisher.values()
            ):
                raise ValueError(f"A5c {label} Fisher normalization is invalid")
        object.__setattr__(self, "compiled_inputs", inputs)
        object.__setattr__(
            self, "family_alias_by_example", MappingProxyType(ownership)
        )
        object.__setattr__(self, "fragment_id_by_node", MappingProxyType(fragments))
        payload = self._receipt_payload()
        computed = _domain_sha256(_RECEIPT_DOMAIN, payload)
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(self.receipt_sha256, label="A5c receipt") != computed:
            raise ValueError("A5c breadth receipt hash mismatch")

    @property
    def held_family_alias(self) -> str:
        """Protocol alias used by A5c nested family CV."""

        return self.outer_held_family_alias

    def _receipt_payload(self) -> dict[str, object]:
        ownership = self.family_alias_by_example
        aliases = self.training_family_aliases
        signatures = a5c_compiled_input_row_signatures(self.compiled_inputs)
        initial_fit = self.initial_fit_source_indices
        initial_audit = self.initial_audit_source_indices
        final_fit = self.fit_source_indices
        final_audit = self.audit_source_indices
        removed = self.removed_fit_source_indices

        def keys(indices: tuple[int, ...]) -> tuple[tuple[str, int], ...]:
            return tuple(self.all_rows.row_keys[index] for index in indices)

        def examples(indices: tuple[int, ...]) -> set[str]:
            return {example_id for example_id, _ in keys(indices)}

        initial_fit_signatures = tuple(signatures[index] for index in initial_fit)
        initial_audit_signatures = tuple(
            signatures[index] for index in initial_audit
        )
        final_fit_signatures = tuple(signatures[index] for index in final_fit)
        final_audit_signatures = tuple(signatures[index] for index in final_audit)
        colliding = set(initial_fit_signatures) & set(initial_audit_signatures)
        source_counts = _role_counts(self.all_rows.row_keys, ownership, aliases)
        initial_fit_counts = _role_counts(keys(initial_fit), ownership, aliases)
        initial_audit_counts = _role_counts(keys(initial_audit), ownership, aliases)
        final_fit_counts = _role_counts(self.fit_rows.row_keys, ownership, aliases)
        final_audit_counts = _role_counts(
            self.audit_rows.row_keys, ownership, aliases
        )
        removed_keys = keys(removed)
        removed_examples = examples(initial_fit) - examples(final_fit)
        return {
            "schema": GEMMA3_L10_L17_A5C_BREADTH_SPLIT_SCHEMA,
            "format_version": GEMMA3_L10_L17_A5C_BREADTH_SPLIT_FORMAT_VERSION,
            "scientific_role": (
                "outer_training_only_example_disjoint_breadth_split_with_"
                "exact_compiled_input_decontamination"
            ),
            "configuration": {
                "split_method": _SPLIT_METHOD,
                "split_binding_sha256": self.split_binding_sha256,
                "audit_examples_per_family": self.audit_examples_per_family,
                "minimum_examples_per_family": (
                    A5C_MINIMUM_EXAMPLES_PER_FAMILY
                ),
                "compiled_input_signature_definition": _SIGNATURE_DEFINITION,
                "cross_role_signature_collision_policy": _COLLISION_POLICY,
                "fisher_normalization_policy": _FISHER_POLICY,
            },
            "ownership": {
                "training_family_aliases": aliases,
                "training_family_count": len(aliases),
                "outer_held_family_alias": self.outer_held_family_alias,
                "outer_held_family_present": False,
                "ownership_membership_sha256": _membership_sha256(
                    set(ownership),
                    ownership,
                    role="all",
                    split_binding_sha256=self.split_binding_sha256,
                ),
            },
            "source": {
                **source_counts,
                "source_bridge_receipt_sha256": (
                    self.source_bridge_receipt_sha256
                ),
                "node_order": self.node_order,
                "fragment_id_by_node": dict(self.fragment_id_by_node),
                "fragment_ids": tuple(self.all_rows.rows_by_fragment),
                "row_key_sha256": self.all_rows.row_key_sha256,
                "row_key_commitment_sha256": _row_keys_sha256(
                    self.all_rows.row_keys, role="all"
                ),
                "bridge_compiled_inputs_sha256": (
                    self.bridge_compiled_inputs_sha256
                ),
                "compiled_inputs_sha256": _tensor_sha256(self.compiled_inputs),
                "compiled_input_signature_multiset_sha256": (
                    _signature_commitment(signatures, role="all")
                ),
                "unique_compiled_input_signature_count": len(set(signatures)),
                "fragment_tensor_sha256s": _fragment_tensor_receipt(
                    self.all_rows
                ),
            },
            "initial_split": {
                "fit": {
                    **initial_fit_counts,
                    "source_indices_sha256": _indices_sha256(
                        initial_fit, role="initial_fit"
                    ),
                    "row_key_commitment_sha256": _row_keys_sha256(
                        keys(initial_fit), role="initial_fit"
                    ),
                    "example_membership_sha256": _membership_sha256(
                        examples(initial_fit),
                        ownership,
                        role="initial_fit",
                        split_binding_sha256=self.split_binding_sha256,
                    ),
                    "compiled_input_signature_multiset_sha256": (
                        _signature_commitment(
                            initial_fit_signatures, role="initial_fit"
                        )
                    ),
                },
                "audit": {
                    **initial_audit_counts,
                    "source_indices_sha256": _indices_sha256(
                        initial_audit, role="initial_audit"
                    ),
                    "row_key_commitment_sha256": _row_keys_sha256(
                        keys(initial_audit), role="initial_audit"
                    ),
                    "example_membership_sha256": _membership_sha256(
                        examples(initial_audit),
                        ownership,
                        role="initial_audit",
                        split_binding_sha256=self.split_binding_sha256,
                    ),
                    "compiled_input_signature_multiset_sha256": (
                        _signature_commitment(
                            initial_audit_signatures, role="initial_audit"
                        )
                    ),
                },
                "example_overlap_count": len(
                    examples(initial_fit) & examples(initial_audit)
                ),
                "examples_exactly_partition_source": (
                    examples(initial_fit) | examples(initial_audit)
                    == set(ownership)
                ),
                "observations_exactly_partition_source": (
                    len(initial_fit) + len(initial_audit)
                    == self.all_rows.observations
                ),
                "cross_role_signature_count": len(colliding),
            },
            "collision_quarantine": {
                "policy": _COLLISION_POLICY,
                "colliding_signature_count": len(colliding),
                "colliding_signature_set_sha256": _signature_commitment(
                    tuple(colliding), role="colliding"
                ),
                "fit_rows_removed": len(removed),
                "audit_rows_removed": 0,
                "fit_examples_fully_removed": len(removed_examples),
                "fit_rows_removed_by_family": {
                    alias: sum(
                        ownership[example_id] == alias
                        for example_id, _ in removed_keys
                    )
                    for alias in aliases
                },
                "fit_examples_fully_removed_by_family": {
                    alias: sum(
                        ownership[example_id] == alias
                        for example_id in removed_examples
                    )
                    for alias in aliases
                },
                "removed_fit_source_indices_sha256": _indices_sha256(
                    removed, role="removed_fit"
                ),
                "removed_fit_row_key_commitment_sha256": _row_keys_sha256(
                    removed_keys, role="removed_fit"
                ),
                "every_removed_fit_row_had_an_audit_signature": all(
                    signatures[index] in set(initial_audit_signatures)
                    for index in removed
                ),
                "every_retained_fit_row_has_no_audit_signature": all(
                    signatures[index] not in set(initial_audit_signatures)
                    for index in final_fit
                ),
            },
            "final_split": {
                "fit": {
                    **final_fit_counts,
                    "source_indices_sha256": _indices_sha256(
                        final_fit, role="final_fit"
                    ),
                    "row_key_sha256": self.fit_rows.row_key_sha256,
                    "row_key_commitment_sha256": _row_keys_sha256(
                        self.fit_rows.row_keys, role="final_fit"
                    ),
                    "example_membership_sha256": _membership_sha256(
                        examples(final_fit),
                        ownership,
                        role="final_fit",
                        split_binding_sha256=self.split_binding_sha256,
                    ),
                    "compiled_input_signature_multiset_sha256": (
                        _signature_commitment(
                            final_fit_signatures, role="final_fit"
                        )
                    ),
                    "fragment_tensor_sha256s": _fragment_tensor_receipt(
                        self.fit_rows
                    ),
                },
                "audit": {
                    **final_audit_counts,
                    "source_indices_sha256": _indices_sha256(
                        final_audit, role="final_audit"
                    ),
                    "row_key_sha256": self.audit_rows.row_key_sha256,
                    "row_key_commitment_sha256": _row_keys_sha256(
                        self.audit_rows.row_keys, role="final_audit"
                    ),
                    "example_membership_sha256": _membership_sha256(
                        examples(final_audit),
                        ownership,
                        role="final_audit",
                        split_binding_sha256=self.split_binding_sha256,
                    ),
                    "compiled_input_signature_multiset_sha256": (
                        _signature_commitment(
                            final_audit_signatures, role="final_audit"
                        )
                    ),
                    "fragment_tensor_sha256s": _fragment_tensor_receipt(
                        self.audit_rows
                    ),
                },
                "row_key_overlap_count": len(
                    set(self.fit_rows.row_keys) & set(self.audit_rows.row_keys)
                ),
                "example_overlap_count": len(
                    examples(final_fit) & examples(final_audit)
                ),
                "compiled_input_signature_overlap_count": len(
                    set(final_fit_signatures) & set(final_audit_signatures)
                ),
                "compiled_input_signatures_disjoint": not bool(
                    set(final_fit_signatures) & set(final_audit_signatures)
                ),
                "retained_plus_removed_observations_equal_source": (
                    len(final_fit) + len(final_audit) + len(removed)
                    == self.all_rows.observations
                ),
                "audit_unchanged_after_quarantine": (
                    final_audit == initial_audit
                ),
            },
            "fisher_normalization": {
                "policy": _FISHER_POLICY,
                "roles_normalized_independently": True,
                "audit_weights_influence_fit_normalization": False,
                "target_total_mass_per_role_and_fragment": 1.0,
                "target_mass_per_family": 1.0 / len(aliases),
                "fit_by_fragment": _fisher_receipt(
                    self.fit_rows, ownership, aliases
                ),
                "audit_by_fragment": _fisher_receipt(
                    self.audit_rows, ownership, aliases
                ),
            },
            "safety": dict(_SAFETY),
        }

    def receipt(self) -> dict[str, object]:
        payload = self._receipt_payload()
        if _contains_tensor(payload):
            raise RuntimeError("A5c breadth receipt contains a Tensor")
        computed = _domain_sha256(_RECEIPT_DOMAIN, payload)
        if computed != self.receipt_sha256:
            raise RuntimeError("A5c breadth rows or receipt lineage drifted")
        return validate_a5c_breadth_split_receipt(
            {**payload, "receipt_sha256": self.receipt_sha256}
        )

    def metadata(self) -> dict[str, object]:
        return self.receipt()


def _strict_fields(
    value: object, fields: set[str], *, label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _strict_role_counts(
    value: object,
    *,
    aliases: tuple[str, ...],
    label: str,
    extra_fields: set[str],
) -> Mapping[str, object]:
    role = _strict_fields(
        value,
        {
            "observations",
            "examples",
            "observations_by_family",
            "examples_by_family",
            *extra_fields,
        },
        label=label,
    )
    if (
        type(role["observations"]) is not int
        or role["observations"] <= 0
        or type(role["examples"]) is not int
        or role["examples"] <= 0
    ):
        raise ValueError(f"{label} counts must be positive integers")
    for name in ("observations_by_family", "examples_by_family"):
        catalog = role[name]
        if not isinstance(catalog, Mapping) or set(catalog) != set(aliases):
            raise ValueError(f"{label} {name} does not cover the families")
        if any(type(count) is not int or count <= 0 for count in catalog.values()):
            raise ValueError(f"{label} {name} counts must be positive")
        if sum(catalog.values()) != role[
            "observations" if name == "observations_by_family" else "examples"
        ]:
            raise ValueError(f"{label} {name} does not sum to its total")
    return role


def _validate_fisher_catalog(
    value: object,
    *,
    aliases: tuple[str, ...],
    fragment_ids: tuple[str, ...],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(fragment_ids):
        raise ValueError(f"{label} Fisher catalog differs from source fragments")
    target = 1.0 / len(aliases)
    for fragment_id, raw in value.items():
        item = _strict_fields(
            raw,
            {
                "fisher_weights_sha256",
                "total_mass",
                "family_mass_by_alias",
                "unit_total_mass",
                "equal_family_mass",
            },
            label=f"{label} {fragment_id}",
        )
        _require_sha256(
            item["fisher_weights_sha256"], label=f"{label} Fisher weights"
        )
        masses = item["family_mass_by_alias"]
        if not isinstance(masses, Mapping) or set(masses) != set(aliases):
            raise ValueError(f"{label} family masses differ")
        if (
            not math.isclose(
                _finite_number(
                    item["total_mass"], label=f"{label} total mass"
                ),
                1.0,
                rel_tol=0.0,
                abs_tol=2.0e-12,
            )
            or any(
                not math.isclose(
                    _finite_number(mass, label=f"{label} family mass"),
                    target,
                    rel_tol=0.0,
                    abs_tol=2.0e-12,
                )
                for mass in masses.values()
            )
            or item["unit_total_mass"] is not True
            or item["equal_family_mass"] is not True
        ):
            raise ValueError(f"{label} equal-family Fisher contract drifted")


def _validate_fragment_hashes(
    value: object, *, fragment_ids: tuple[str, ...], label: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(fragment_ids):
        raise ValueError(f"{label} fragment tensor catalog differs")
    for raw in value.values():
        item = _strict_fields(
            raw,
            {"inputs_sha256", "contributions_sha256", "fisher_weights_sha256"},
            label=label,
        )
        for digest in item.values():
            _require_sha256(digest, label=f"{label} tensor")


def validate_a5c_breadth_split_receipt(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Strictly validate a tensor-free A5c breadth-split receipt."""

    if not isinstance(value, Mapping):
        raise TypeError("A5c breadth receipt must be a mapping")
    raw = dict(value)
    _strict_fields(
        raw,
        {
            "schema",
            "format_version",
            "scientific_role",
            "configuration",
            "ownership",
            "source",
            "initial_split",
            "collision_quarantine",
            "final_split",
            "fisher_normalization",
            "safety",
            "receipt_sha256",
        },
        label="A5c breadth receipt",
    )
    if (
        raw["schema"] != GEMMA3_L10_L17_A5C_BREADTH_SPLIT_SCHEMA
        or raw["format_version"]
        != GEMMA3_L10_L17_A5C_BREADTH_SPLIT_FORMAT_VERSION
        or raw["scientific_role"]
        != (
            "outer_training_only_example_disjoint_breadth_split_with_"
            "exact_compiled_input_decontamination"
        )
        or raw["safety"] != _SAFETY
        or _contains_tensor(raw)
    ):
        raise ValueError("A5c breadth receipt header or safety contract drifted")
    config = _strict_fields(
        raw["configuration"],
        {
            "split_method",
            "split_binding_sha256",
            "audit_examples_per_family",
            "minimum_examples_per_family",
            "compiled_input_signature_definition",
            "cross_role_signature_collision_policy",
            "fisher_normalization_policy",
        },
        label="A5c breadth configuration",
    )
    if (
        config["split_method"] != _SPLIT_METHOD
        or config["minimum_examples_per_family"]
        != A5C_MINIMUM_EXAMPLES_PER_FAMILY
        or type(config["audit_examples_per_family"]) is not int
        or config["audit_examples_per_family"] <= 0
        or config["compiled_input_signature_definition"]
        != _SIGNATURE_DEFINITION
        or config["cross_role_signature_collision_policy"]
        != _COLLISION_POLICY
        or config["fisher_normalization_policy"] != _FISHER_POLICY
    ):
        raise ValueError("A5c breadth configuration drifted")
    _require_sha256(config["split_binding_sha256"], label="split binding")

    ownership = _strict_fields(
        raw["ownership"],
        {
            "training_family_aliases",
            "training_family_count",
            "outer_held_family_alias",
            "outer_held_family_present",
            "ownership_membership_sha256",
        },
        label="A5c breadth ownership",
    )
    aliases = tuple(ownership["training_family_aliases"])  # type: ignore[arg-type]
    if (
        not aliases
        or len(aliases) != len(set(aliases))
        or any(not isinstance(alias, str) or not alias for alias in aliases)
        or ownership["training_family_count"] != len(aliases)
        or not isinstance(ownership["outer_held_family_alias"], str)
        or ownership["outer_held_family_alias"] in aliases
        or ownership["outer_held_family_present"] is not False
    ):
        raise ValueError("A5c breadth ownership contract drifted")
    _require_sha256(
        ownership["ownership_membership_sha256"], label="ownership membership"
    )

    source = _strict_role_counts(
        raw["source"],
        aliases=aliases,
        label="A5c breadth source",
        extra_fields={
            "source_bridge_receipt_sha256",
            "node_order",
            "fragment_id_by_node",
            "fragment_ids",
            "row_key_sha256",
            "row_key_commitment_sha256",
            "bridge_compiled_inputs_sha256",
            "compiled_inputs_sha256",
            "compiled_input_signature_multiset_sha256",
            "unique_compiled_input_signature_count",
            "fragment_tensor_sha256s",
        },
    )
    if any(
        count < A5C_MINIMUM_EXAMPLES_PER_FAMILY
        for count in source["examples_by_family"].values()  # type: ignore[union-attr]
    ):
        raise ValueError("A5c breadth source is below the family breadth floor")
    fragment_ids = tuple(source["fragment_ids"])  # type: ignore[arg-type]
    node_order = tuple(source["node_order"])  # type: ignore[arg-type]
    fragment_id_by_node = source["fragment_id_by_node"]
    if (
        not fragment_ids
        or len(fragment_ids) != len(set(fragment_ids))
        or any(not isinstance(name, str) or not name for name in fragment_ids)
        or not node_order
        or len(node_order) != len(set(node_order))
        or any(not isinstance(name, str) or not name for name in node_order)
        or not isinstance(fragment_id_by_node, Mapping)
        or set(fragment_id_by_node) != set(node_order)
        or tuple(fragment_id_by_node[name] for name in node_order)
        != fragment_ids
        or type(source["unique_compiled_input_signature_count"]) is not int
        or not 1 <= source["unique_compiled_input_signature_count"] <= source[
            "observations"
        ]
    ):
        raise ValueError("A5c breadth source catalog or signature count drifted")
    if source["source_bridge_receipt_sha256"] is not None:
        _require_sha256(
            source["source_bridge_receipt_sha256"],
            label="A5c source bridge receipt",
        )
    for name in (
        "row_key_sha256",
        "row_key_commitment_sha256",
        "bridge_compiled_inputs_sha256",
        "compiled_inputs_sha256",
        "compiled_input_signature_multiset_sha256",
    ):
        _require_sha256(source[name], label=f"A5c source {name}")
    _validate_fragment_hashes(
        source["fragment_tensor_sha256s"],
        fragment_ids=fragment_ids,
        label="A5c source",
    )

    initial = _strict_fields(
        raw["initial_split"],
        {
            "fit",
            "audit",
            "example_overlap_count",
            "examples_exactly_partition_source",
            "observations_exactly_partition_source",
            "cross_role_signature_count",
        },
        label="A5c initial split",
    )
    initial_extra = {
        "source_indices_sha256",
        "row_key_commitment_sha256",
        "example_membership_sha256",
        "compiled_input_signature_multiset_sha256",
    }
    initial_fit = _strict_role_counts(
        initial["fit"], aliases=aliases, label="A5c initial fit", extra_fields=initial_extra
    )
    initial_audit = _strict_role_counts(
        initial["audit"],
        aliases=aliases,
        label="A5c initial audit",
        extra_fields=initial_extra,
    )
    for role in (initial_fit, initial_audit):
        for name in initial_extra:
            _require_sha256(role[name], label=f"A5c initial {name}")
    if (
        any(
            count != config["audit_examples_per_family"]
            for count in initial_audit["examples_by_family"].values()  # type: ignore[union-attr]
        )
        or any(
            initial_fit["examples_by_family"][alias]  # type: ignore[index]
            + initial_audit["examples_by_family"][alias]  # type: ignore[index]
            != source["examples_by_family"][alias]  # type: ignore[index]
            for alias in aliases
        )
        or initial_fit["observations"] + initial_audit["observations"]
        != source["observations"]
        or initial_fit["examples"] + initial_audit["examples"]
        != source["examples"]
        or initial["example_overlap_count"] != 0
        or initial["examples_exactly_partition_source"] is not True
        or initial["observations_exactly_partition_source"] is not True
        or type(initial["cross_role_signature_count"]) is not int
        or initial["cross_role_signature_count"] < 0
    ):
        raise ValueError("A5c initial split arithmetic drifted")

    quarantine = _strict_fields(
        raw["collision_quarantine"],
        {
            "policy",
            "colliding_signature_count",
            "colliding_signature_set_sha256",
            "fit_rows_removed",
            "audit_rows_removed",
            "fit_examples_fully_removed",
            "fit_rows_removed_by_family",
            "fit_examples_fully_removed_by_family",
            "removed_fit_source_indices_sha256",
            "removed_fit_row_key_commitment_sha256",
            "every_removed_fit_row_had_an_audit_signature",
            "every_retained_fit_row_has_no_audit_signature",
        },
        label="A5c collision quarantine",
    )
    if (
        quarantine["policy"] != _COLLISION_POLICY
        or quarantine["colliding_signature_count"]
        != initial["cross_role_signature_count"]
        or type(quarantine["fit_rows_removed"]) is not int
        or quarantine["fit_rows_removed"] < 0
        or quarantine["audit_rows_removed"] != 0
        or type(quarantine["fit_examples_fully_removed"]) is not int
        or not 0
        <= quarantine["fit_examples_fully_removed"]
        <= initial_fit["examples"]
        or quarantine["every_removed_fit_row_had_an_audit_signature"] is not True
        or quarantine["every_retained_fit_row_has_no_audit_signature"] is not True
    ):
        raise ValueError("A5c collision quarantine contract drifted")
    removed_rows_by_family = quarantine["fit_rows_removed_by_family"]
    removed_examples_by_family = quarantine[
        "fit_examples_fully_removed_by_family"
    ]
    if (
        not isinstance(removed_rows_by_family, Mapping)
        or set(removed_rows_by_family) != set(aliases)
        or not isinstance(removed_examples_by_family, Mapping)
        or set(removed_examples_by_family) != set(aliases)
        or any(
            type(count) is not int or count < 0
            for count in (
                *removed_rows_by_family.values(),
                *removed_examples_by_family.values(),
            )
        )
        or sum(removed_rows_by_family.values())
        != quarantine["fit_rows_removed"]
        or sum(removed_examples_by_family.values())
        != quarantine["fit_examples_fully_removed"]
    ):
        raise ValueError("A5c collision quarantine family counts drifted")
    for name in (
        "colliding_signature_set_sha256",
        "removed_fit_source_indices_sha256",
        "removed_fit_row_key_commitment_sha256",
    ):
        _require_sha256(quarantine[name], label=f"A5c quarantine {name}")

    final = _strict_fields(
        raw["final_split"],
        {
            "fit",
            "audit",
            "row_key_overlap_count",
            "example_overlap_count",
            "compiled_input_signature_overlap_count",
            "compiled_input_signatures_disjoint",
            "retained_plus_removed_observations_equal_source",
            "audit_unchanged_after_quarantine",
        },
        label="A5c final split",
    )
    final_extra = {
        "source_indices_sha256",
        "row_key_sha256",
        "row_key_commitment_sha256",
        "example_membership_sha256",
        "compiled_input_signature_multiset_sha256",
        "fragment_tensor_sha256s",
    }
    final_fit = _strict_role_counts(
        final["fit"], aliases=aliases, label="A5c final fit", extra_fields=final_extra
    )
    final_audit = _strict_role_counts(
        final["audit"],
        aliases=aliases,
        label="A5c final audit",
        extra_fields=final_extra,
    )
    for role, label in ((final_fit, "fit"), (final_audit, "audit")):
        for name in final_extra - {"fragment_tensor_sha256s"}:
            _require_sha256(role[name], label=f"A5c final {label} {name}")
        _validate_fragment_hashes(
            role["fragment_tensor_sha256s"],
            fragment_ids=fragment_ids,
            label=f"A5c final {label}",
        )
    if (
        final_fit["observations"] + quarantine["fit_rows_removed"]
        != initial_fit["observations"]
        or final_audit["observations"] != initial_audit["observations"]
        or final_audit["examples"] != initial_audit["examples"]
        or final_fit["examples"] + quarantine["fit_examples_fully_removed"]
        != initial_fit["examples"]
        or any(
            final_fit["observations_by_family"][alias]  # type: ignore[index]
            + removed_rows_by_family[alias]
            != initial_fit["observations_by_family"][alias]  # type: ignore[index]
            or final_fit["examples_by_family"][alias]  # type: ignore[index]
            + removed_examples_by_family[alias]
            != initial_fit["examples_by_family"][alias]  # type: ignore[index]
            for alias in aliases
        )
        or any(
            final_audit["observations_by_family"][alias]  # type: ignore[index]
            != initial_audit["observations_by_family"][alias]  # type: ignore[index]
            or final_audit["examples_by_family"][alias]  # type: ignore[index]
            != initial_audit["examples_by_family"][alias]  # type: ignore[index]
            for alias in aliases
        )
        or final["row_key_overlap_count"] != 0
        or final["example_overlap_count"] != 0
        or final["compiled_input_signature_overlap_count"] != 0
        or final["compiled_input_signatures_disjoint"] is not True
        or final["retained_plus_removed_observations_equal_source"] is not True
        or final["audit_unchanged_after_quarantine"] is not True
    ):
        raise ValueError("A5c final split arithmetic or leakage guard drifted")

    fisher = _strict_fields(
        raw["fisher_normalization"],
        {
            "policy",
            "roles_normalized_independently",
            "audit_weights_influence_fit_normalization",
            "target_total_mass_per_role_and_fragment",
            "target_mass_per_family",
            "fit_by_fragment",
            "audit_by_fragment",
        },
        label="A5c Fisher normalization",
    )
    if (
        fisher["policy"] != _FISHER_POLICY
        or fisher["roles_normalized_independently"] is not True
        or fisher["audit_weights_influence_fit_normalization"] is not False
        or not math.isclose(
            _finite_number(
                fisher["target_total_mass_per_role_and_fragment"],
                label="A5c target total Fisher mass",
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            _finite_number(
                fisher["target_mass_per_family"],
                label="A5c target family Fisher mass",
            ),
            1.0 / len(aliases),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("A5c Fisher normalization policy drifted")
    _validate_fisher_catalog(
        fisher["fit_by_fragment"],
        aliases=aliases,
        fragment_ids=fragment_ids,
        label="A5c fit",
    )
    _validate_fisher_catalog(
        fisher["audit_by_fragment"],
        aliases=aliases,
        fragment_ids=fragment_ids,
        label="A5c audit",
    )

    supplied = _require_sha256(raw["receipt_sha256"], label="A5c receipt")
    payload = dict(raw)
    payload.pop("receipt_sha256")
    if supplied != _domain_sha256(_RECEIPT_DOMAIN, payload):
        raise ValueError("A5c breadth receipt hash mismatch")
    return json.loads(json.dumps(raw, allow_nan=False))


def build_a5c_breadth_split(
    *,
    rows: AlignedFragmentRows,
    compiled_inputs: Tensor,
    row_keys: tuple[tuple[str, int], ...],
    family_alias_by_example: Mapping[str, str],
    training_family_aliases: Sequence[str],
    outer_held_family_alias: str,
    split_binding_sha256: str,
    audit_examples_per_family: int = 1,
    node_order: Sequence[str] | None = None,
    fragment_id_by_node: Mapping[str, str] | None = None,
    source_bridge_receipt_sha256: str | None = None,
    bridge_compiled_inputs_sha256: str | None = None,
) -> A5cBreadthSplit:
    """Build an example-disjoint and exact-input-disjoint A5c row split."""

    aliases, ownership = _validate_row_keys_and_ownership(
        rows,
        row_keys,
        family_alias_by_example,
        training_family_aliases,
        outer_held_family_alias,
    )
    inputs = _canonical_compiled_inputs(compiled_inputs, rows=rows.observations)
    if any(
        not torch.equal(fragment.inputs, inputs)
        for fragment in rows.rows_by_fragment.values()
    ):
        raise ValueError("compiled_inputs must exactly equal every fragment input bank")
    row_fragments = tuple(rows.rows_by_fragment)
    if node_order is None and fragment_id_by_node is None:
        names = row_fragments
        fragments = {name: name for name in names}
    elif node_order is None or fragment_id_by_node is None:
        raise ValueError("node_order and fragment_id_by_node must be supplied together")
    else:
        names = tuple(node_order)
        fragments = dict(fragment_id_by_node)
    if (
        not names
        or len(names) != len(set(names))
        or set(fragments) != set(names)
        or tuple(fragments[name] for name in names) != row_fragments
    ):
        raise ValueError("node/fragment catalog must match aligned-row order")
    if source_bridge_receipt_sha256 is not None:
        _require_sha256(
            source_bridge_receipt_sha256, label="source bridge receipt"
        )
    expected_bridge_input_sha256 = a5b_tensor_sha256(inputs)
    if bridge_compiled_inputs_sha256 is None:
        bridge_input_sha256 = expected_bridge_input_sha256
    else:
        bridge_input_sha256 = _require_sha256(
            bridge_compiled_inputs_sha256,
            label="bridge-domain compiled inputs",
        )
        if bridge_input_sha256 != expected_bridge_input_sha256:
            raise ValueError(
                "bridge-domain compiled-input identity differs from inputs"
            )
    fit_examples, audit_examples = _initial_example_split(
        ownership,
        aliases,
        split_binding_sha256=split_binding_sha256,
        audit_examples_per_family=audit_examples_per_family,
    )
    initial_fit = _indices_for_examples(row_keys, fit_examples)
    initial_audit = _indices_for_examples(row_keys, audit_examples)
    purge = purge_a5c_cross_split_input_signatures(
        inputs,
        initial_fit,
        initial_audit,
    )
    removed_fit = purge.removed_fit_indices
    fit = purge.retained_fit_indices
    if not fit:
        raise ValueError("signature quarantine removed every fit row")
    fit_rows = _subset_equal_family_rows(
        rows, fit, ownership=ownership, aliases=aliases
    )
    audit_rows = _subset_equal_family_rows(
        rows, initial_audit, ownership=ownership, aliases=aliases
    )
    return A5cBreadthSplit(
        all_rows=rows,
        fit_rows=fit_rows,
        audit_rows=audit_rows,
        compiled_inputs=inputs,
        initial_fit_source_indices=initial_fit,
        initial_audit_source_indices=initial_audit,
        fit_source_indices=fit,
        audit_source_indices=initial_audit,
        removed_fit_source_indices=removed_fit,
        node_order=names,
        fragment_id_by_node=fragments,
        family_alias_by_example=ownership,
        training_family_aliases=aliases,
        outer_held_family_alias=outer_held_family_alias,
        split_binding_sha256=split_binding_sha256,
        audit_examples_per_family=audit_examples_per_family,
        bridge_compiled_inputs_sha256=bridge_input_sha256,
        source_bridge_receipt_sha256=source_bridge_receipt_sha256,
    )


def build_a5c_breadth_split_from_bridge(
    *,
    bridge: A5bDownstreamCoordinateTargetBridge,
    split_binding_sha256: str,
    audit_examples_per_family: int = 1,
) -> A5cBreadthSplit:
    """Derive a breadth split from one authenticated A5b coordinate bridge."""

    if not isinstance(bridge, A5bDownstreamCoordinateTargetBridge):
        raise TypeError("bridge must be an A5b downstream-coordinate bridge")
    bridge_receipt = bridge.receipt()
    if bridge_receipt.get("receipt_sha256") != bridge.receipt_sha256:
        raise ValueError("A5b bridge receipt identity drifted")
    fragments = tuple(bridge.all_rows.rows_by_fragment.values())
    authentication = bridge_receipt.get("authentication")
    if not isinstance(authentication, Mapping):
        raise TypeError("A5b bridge authentication is unavailable")
    bridge_input_sha256 = _require_sha256(
        authentication.get("compiled_inputs_sha256"),
        label="A5b bridge compiled inputs",
    )
    if not fragments or any(
        not torch.equal(fragment.inputs, fragments[0].inputs)
        for fragment in fragments[1:]
    ):
        raise ValueError("A5b bridge does not contain one shared compiled input bank")
    return build_a5c_breadth_split(
        rows=bridge.all_rows,
        compiled_inputs=fragments[0].inputs,
        row_keys=bridge.all_rows.row_keys,
        family_alias_by_example=bridge.family_alias_by_example,
        training_family_aliases=bridge.training_family_aliases,
        outer_held_family_alias=bridge.held_family_alias,
        split_binding_sha256=split_binding_sha256,
        audit_examples_per_family=audit_examples_per_family,
        node_order=bridge.node_order,
        fragment_id_by_node=bridge.fragment_id_by_node,
        source_bridge_receipt_sha256=bridge.receipt_sha256,
        bridge_compiled_inputs_sha256=bridge_input_sha256,
    )
