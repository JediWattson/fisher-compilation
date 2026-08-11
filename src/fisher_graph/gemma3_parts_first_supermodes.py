"""Parts-first Gemma-3 atom catalog and ordered supermode paths.

This module is deliberately model-free and dependency-light.  It refines the
existing exact heterogeneous Gemma lift into physical pieces without claiming
that a transformer layer is itself a generator:

* the tied lexical matrix is 640 hidden-column atoms, each occurring once at
  ingress and again at readout while owning one physical column;
* each attention parent is four Q/O-head atoms plus explicit shared K/V,
  routing, RoPE, softmax, and norm substrate;
* each MLP parent is 2,048 channel atoms (gate row, up row, down column) plus
  explicit shared norm/activation substrate; and
* the terminal norm remains shared substrate.

Logical groups may overlap and receive no storage credit.  Exact partitions
are a separate, disjoint/exhaustive group semantics.  A supermode is an
ordered path of same-boundary atom groups: atoms can add inside one execution
boundary, but boundaries compose in execution order and may never be summed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
import struct


__all__ = [
    "ATTENTION_HEAD_COUNT",
    "ATTENTION_HEAD_DIMENSION",
    "ATOM_OWNED_SCALAR_COUNT",
    "GEMMA3_LAYER_COUNT",
    "HIDDEN_WIDTH",
    "MLP_CHANNEL_COUNT",
    "SHARED_SUBSTRATE_SCALAR_COUNT",
    "SOURCE_UNIQUE_PARAMETER_COUNT",
    "TOTAL_PHYSICAL_ATOM_COUNT",
    "VOCABULARY_SIZE",
    "AtomGroupDraft",
    "AtomGroupMembershipRef",
    "ExecutionBoundary",
    "FunctionalAtomOccurrence",
    "FunctionalRole",
    "Gemma3PartsCatalog",
    "LexicalRowViewSpec",
    "ParentRefinementReceipt",
    "PhysicalAtomFamily",
    "PhysicalAtomFamilySpec",
    "PhysicalAtomId",
    "PhysicalAtomSpec",
    "SharedSubstrateSpec",
    "SubstrateComponentSpec",
    "SupermodePathSpec",
    "SupermodeStepSpec",
    "build_atom_group_sidecar",
    "build_gemma3_parts_catalog",
    "validate_atom_group_membership_ref",
    "validate_exact_atom_partition",
    "validate_gemma3_parts_catalog",
    "validate_supermode_path",
]


GEMMA3_LAYER_COUNT = 18
HIDDEN_WIDTH = 640
VOCABULARY_SIZE = 262_144
ATTENTION_HEAD_COUNT = 4
ATTENTION_KEY_VALUE_HEAD_COUNT = 1
ATTENTION_HEAD_DIMENSION = 256
MLP_CHANNEL_COUNT = 2_048

LEXICAL_ATOM_COUNT = HIDDEN_WIDTH
ATTENTION_ATOM_COUNT = GEMMA3_LAYER_COUNT * ATTENTION_HEAD_COUNT
MLP_ATOM_COUNT = GEMMA3_LAYER_COUNT * MLP_CHANNEL_COUNT
TOTAL_PHYSICAL_ATOM_COUNT = (
    LEXICAL_ATOM_COUNT + ATTENTION_ATOM_COUNT + MLP_ATOM_COUNT
)

LEXICAL_SCALARS_PER_ATOM = VOCABULARY_SIZE
ATTENTION_SCALARS_PER_ATOM = 2 * HIDDEN_WIDTH * ATTENTION_HEAD_DIMENSION
MLP_SCALARS_PER_ATOM = 3 * HIDDEN_WIDTH

LEXICAL_ATOM_SCALARS = LEXICAL_ATOM_COUNT * LEXICAL_SCALARS_PER_ATOM
ATTENTION_ATOM_SCALARS = ATTENTION_ATOM_COUNT * ATTENTION_SCALARS_PER_ATOM
MLP_ATOM_SCALARS = MLP_ATOM_COUNT * MLP_SCALARS_PER_ATOM
ATOM_OWNED_SCALAR_COUNT = (
    LEXICAL_ATOM_SCALARS + ATTENTION_ATOM_SCALARS + MLP_ATOM_SCALARS
)

ATTENTION_SHARED_SCALARS_PER_LAYER = (
    2 * HIDDEN_WIDTH * ATTENTION_KEY_VALUE_HEAD_COUNT * ATTENTION_HEAD_DIMENSION
    + 2 * ATTENTION_HEAD_DIMENSION
    + 2 * HIDDEN_WIDTH
)
MLP_SHARED_SCALARS_PER_LAYER = 2 * HIDDEN_WIDTH
TERMINAL_SHARED_SCALARS = HIDDEN_WIDTH
SHARED_SUBSTRATE_SCALAR_COUNT = (
    GEMMA3_LAYER_COUNT * ATTENTION_SHARED_SCALARS_PER_LAYER
    + GEMMA3_LAYER_COUNT * MLP_SHARED_SCALARS_PER_LAYER
    + TERMINAL_SHARED_SCALARS
)
SOURCE_UNIQUE_PARAMETER_COUNT = (
    ATOM_OWNED_SCALAR_COUNT + SHARED_SUBSTRATE_SCALAR_COUNT
)

_CATALOG_SCHEMA = "fisher_graph.gemma3_parts_first_catalog.v1"
_CATALOG_DOMAIN = b"fisher_graph:gemma3-parts-first-catalog:v1\0"
_GROUP_ENCODING = "sorted_unique_uint32_le.v1"
_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GLOBAL_ATTENTION_LAYERS = frozenset((5, 11, 17))
_RESOLUTION_CONTRACT = {
    "tied_lexical_hidden_column": {
        "global_ordinal_formula": "g",
        "owned_parameter_slices": (("model.embed_tokens.weight", 1, "g:g+1"),),
        "functional_occurrences": ("lexical_ingress", "lexical_readout"),
        "storage_counted_once_across_occurrences": True,
    },
    "attention_qo_head": {
        "global_ordinal_formula": "640+4*layer+head",
        "owned_parameter_slices": (
            ("model.layers.L.self_attn.q_proj.weight", 0, "256h:256(h+1)"),
            ("model.layers.L.self_attn.o_proj.weight", 1, "256h:256(h+1)"),
        ),
        "functional_occurrences": ("layer_L_attention",),
        "shared_substrate_excluded": True,
    },
    "mlp_channel": {
        "global_ordinal_formula": "712+2048*layer+channel",
        "owned_parameter_slices": (
            ("model.layers.L.mlp.gate_proj.weight", 0, "j:j+1"),
            ("model.layers.L.mlp.up_proj.weight", 0, "j:j+1"),
            ("model.layers.L.mlp.down_proj.weight", 1, "j:j+1"),
        ),
        "functional_occurrences": ("layer_L_mlp",),
        "shared_substrate_excluded": True,
    },
}
_STRICTLY_VALIDATED_CATALOG_SHA256S: set[str] = set()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_sha256(domain: bytes, value: object) -> str:
    return _sha256(domain + _canonical_json_bytes(value))


_RESOLUTION_CONTRACT_SHA256 = _domain_sha256(
    b"fisher_graph:gemma3-parts-first-resolution-contract:v1\0",
    _RESOLUTION_CONTRACT,
)


def _require_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


class PhysicalAtomFamily(str, Enum):
    """Only addressable physical parts; a native layer is not a member."""

    TIED_LEXICAL_HIDDEN_COLUMN = "tied_lexical_hidden_column"
    ATTENTION_QO_HEAD = "attention_qo_head"
    MLP_CHANNEL = "mlp_channel"


class FunctionalRole(str, Enum):
    LEXICAL_INGRESS = "lexical_ingress"
    ATTENTION_QO_CONTRIBUTION = "attention_qo_contribution"
    MLP_CHANNEL_CONTRIBUTION = "mlp_channel_contribution"
    LEXICAL_READOUT = "lexical_readout"


class BoundaryKind(str, Enum):
    LEXICAL_INGRESS = "lexical_ingress"
    ATTENTION = "attention"
    MLP = "mlp"
    TERMINAL_NORM = "terminal_norm"
    LEXICAL_READOUT = "lexical_readout"


@dataclass(frozen=True, slots=True, order=True)
class ExecutionBoundary:
    """One native execution location; layers are containers, not generators."""

    execution_ordinal: int
    boundary_kind: BoundaryKind = field(compare=False)
    layer_ordinal: int | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_kind, BoundaryKind):
            raise TypeError("boundary_kind must be a BoundaryKind")
        _require_int(
            self.execution_ordinal,
            label="execution_ordinal",
        )
        expected: int
        if self.boundary_kind is BoundaryKind.LEXICAL_INGRESS:
            if self.layer_ordinal is not None:
                raise ValueError("lexical ingress has no layer container")
            expected = 0
        elif self.boundary_kind is BoundaryKind.ATTENTION:
            _require_int(self.layer_ordinal, label="layer_ordinal")
            if self.layer_ordinal >= GEMMA3_LAYER_COUNT:  # type: ignore[operator]
                raise ValueError("attention layer container is out of range")
            expected = 1 + 2 * self.layer_ordinal  # type: ignore[operator]
        elif self.boundary_kind is BoundaryKind.MLP:
            _require_int(self.layer_ordinal, label="layer_ordinal")
            if self.layer_ordinal >= GEMMA3_LAYER_COUNT:  # type: ignore[operator]
                raise ValueError("MLP layer container is out of range")
            expected = 2 + 2 * self.layer_ordinal  # type: ignore[operator]
        elif self.boundary_kind is BoundaryKind.TERMINAL_NORM:
            if self.layer_ordinal is not None:
                raise ValueError("terminal norm has no layer container")
            expected = 37
        else:
            if self.layer_ordinal is not None:
                raise ValueError("lexical readout has no layer container")
            expected = 38
        if self.execution_ordinal != expected:
            raise ValueError("boundary execution ordinal is not canonical")

    @property
    def boundary_id(self) -> str:
        if self.layer_ordinal is None:
            return self.boundary_kind.value
        return f"layer.{self.layer_ordinal}.{self.boundary_kind.value}"

    @classmethod
    def lexical_ingress(cls) -> ExecutionBoundary:
        return cls(0, BoundaryKind.LEXICAL_INGRESS)

    @classmethod
    def attention(cls, layer_ordinal: int) -> ExecutionBoundary:
        return cls(1 + 2 * layer_ordinal, BoundaryKind.ATTENTION, layer_ordinal)

    @classmethod
    def mlp(cls, layer_ordinal: int) -> ExecutionBoundary:
        return cls(2 + 2 * layer_ordinal, BoundaryKind.MLP, layer_ordinal)

    @classmethod
    def terminal_norm(cls) -> ExecutionBoundary:
        return cls(37, BoundaryKind.TERMINAL_NORM)

    @classmethod
    def lexical_readout(cls) -> ExecutionBoundary:
        return cls(38, BoundaryKind.LEXICAL_READOUT)

    def to_dict(self) -> dict[str, object]:
        return {
            "boundary_id": self.boundary_id,
            "boundary_kind": self.boundary_kind.value,
            "execution_ordinal": self.execution_ordinal,
            "layer_ordinal": self.layer_ordinal,
        }


@dataclass(frozen=True, slots=True)
class PhysicalAtomId:
    """Typed physical identity independent from its functional occurrences."""

    family: PhysicalAtomFamily
    part_ordinal: int
    layer_ordinal: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family, PhysicalAtomFamily):
            if str(self.family) in {"layer", "whole_layer", "generator.layer"}:
                raise ValueError(
                    "a whole layer is a container, not a physical generator atom"
                )
            raise TypeError("family must be a PhysicalAtomFamily")
        _require_int(self.part_ordinal, label="part_ordinal")
        if self.family is PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN:
            if self.layer_ordinal is not None:
                raise ValueError("tied lexical storage has no layer owner")
            if self.part_ordinal >= HIDDEN_WIDTH:
                raise ValueError("lexical hidden-column ordinal is out of range")
        elif self.family is PhysicalAtomFamily.ATTENTION_QO_HEAD:
            _require_int(self.layer_ordinal, label="layer_ordinal")
            if self.layer_ordinal >= GEMMA3_LAYER_COUNT:  # type: ignore[operator]
                raise ValueError("attention layer container is out of range")
            if self.part_ordinal >= ATTENTION_HEAD_COUNT:
                raise ValueError("attention head ordinal is out of range")
        else:
            _require_int(self.layer_ordinal, label="layer_ordinal")
            if self.layer_ordinal >= GEMMA3_LAYER_COUNT:  # type: ignore[operator]
                raise ValueError("MLP layer container is out of range")
            if self.part_ordinal >= MLP_CHANNEL_COUNT:
                raise ValueError("MLP channel ordinal is out of range")

    @property
    def canonical_id(self) -> str:
        if self.family is PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN:
            return f"atom.lexical.hidden_column.{self.part_ordinal}"
        if self.family is PhysicalAtomFamily.ATTENTION_QO_HEAD:
            return (
                f"atom.attention.layer.{self.layer_ordinal}."
                f"qo_head.{self.part_ordinal}"
            )
        return f"atom.mlp.layer.{self.layer_ordinal}.channel.{self.part_ordinal}"


@dataclass(frozen=True, slots=True)
class ParameterSliceSpec:
    parameter_name: str
    axis: int
    start: int
    stop: int
    scalar_count: int

    def __post_init__(self) -> None:
        _require_id(self.parameter_name, label="parameter_name")
        if self.axis not in (0, 1):
            raise ValueError("parameter slice axis must be zero or one")
        _require_int(self.start, label="slice start")
        _require_int(self.stop, label="slice stop", minimum=1)
        _require_int(self.scalar_count, label="slice scalar_count", minimum=1)
        if self.stop <= self.start:
            raise ValueError("parameter slice must be nonempty")

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_name": self.parameter_name,
            "axis": self.axis,
            "start": self.start,
            "stop": self.stop,
            "scalar_count": self.scalar_count,
        }


@dataclass(frozen=True, slots=True)
class FunctionalAtomOccurrence:
    """One use of a physical atom at one execution boundary."""

    occurrence_id: str
    atom_id: PhysicalAtomId
    role: FunctionalRole
    boundary: ExecutionBoundary

    def __post_init__(self) -> None:
        _require_id(self.occurrence_id, label="occurrence_id")
        if not isinstance(self.atom_id, PhysicalAtomId):
            raise TypeError("occurrence atom_id must be physical")
        if not isinstance(self.role, FunctionalRole):
            raise TypeError("occurrence role must be a FunctionalRole")
        if not isinstance(self.boundary, ExecutionBoundary):
            raise TypeError("occurrence boundary is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "occurrence_id": self.occurrence_id,
            "physical_atom_id": self.atom_id.canonical_id,
            "role": self.role.value,
            "boundary": self.boundary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PhysicalAtomSpec:
    atom_id: PhysicalAtomId
    global_ordinal: int
    scalar_count: int
    predecessor_parent_atom_id: str
    predecessor_parent_family_id: str
    parameter_slices: tuple[ParameterSliceSpec, ...]
    occurrences: tuple[FunctionalAtomOccurrence, ...]

    def __post_init__(self) -> None:
        _require_int(self.global_ordinal, label="global_ordinal")
        _require_int(self.scalar_count, label="scalar_count", minimum=1)
        _require_id(
            self.predecessor_parent_atom_id,
            label="predecessor_parent_atom_id",
        )
        _require_id(
            self.predecessor_parent_family_id,
            label="predecessor_parent_family_id",
        )
        if not self.parameter_slices or sum(
            item.scalar_count for item in self.parameter_slices
        ) != self.scalar_count:
            raise ValueError("atom parameter slices do not close scalar_count")
        if not self.occurrences:
            raise ValueError("physical atom must have a functional occurrence")
        if any(item.atom_id != self.atom_id for item in self.occurrences):
            raise ValueError("functional occurrence points at another atom")


@dataclass(frozen=True, slots=True)
class PhysicalAtomFamilySpec:
    family: PhysicalAtomFamily
    global_ordinal_start: int
    atom_count: int
    scalars_per_atom: int
    physical_scalar_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.family, PhysicalAtomFamily):
            raise TypeError("physical atom family is invalid")
        _require_int(self.global_ordinal_start, label="global_ordinal_start")
        _require_int(self.atom_count, label="atom_count", minimum=1)
        _require_int(self.scalars_per_atom, label="scalars_per_atom", minimum=1)
        if self.physical_scalar_count != self.atom_count * self.scalars_per_atom:
            raise ValueError("physical atom family scalar accounting does not close")

    @property
    def global_ordinal_stop(self) -> int:
        return self.global_ordinal_start + self.atom_count

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "global_ordinal_start": self.global_ordinal_start,
            "global_ordinal_stop": self.global_ordinal_stop,
            "atom_count": self.atom_count,
            "scalars_per_atom": self.scalars_per_atom,
            "physical_scalar_count": self.physical_scalar_count,
        }


@dataclass(frozen=True, slots=True)
class SubstrateComponentSpec:
    component_id: str
    physical_scalar_count: int
    parameter_free: bool

    def __post_init__(self) -> None:
        _require_id(self.component_id, label="component_id")
        _require_int(
            self.physical_scalar_count,
            label="component physical_scalar_count",
        )
        if type(self.parameter_free) is not bool:
            raise TypeError("parameter_free must be bool")
        if self.parameter_free != (self.physical_scalar_count == 0):
            raise ValueError("parameter_free flag and scalar count disagree")

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "physical_scalar_count": self.physical_scalar_count,
            "parameter_free": self.parameter_free,
        }


@dataclass(frozen=True, slots=True)
class SharedSubstrateSpec:
    substrate_id: str
    boundary: ExecutionBoundary
    predecessor_parent_atom_id: str
    components: tuple[SubstrateComponentSpec, ...]
    physical_scalar_count: int
    owned_by_physical_atoms: bool = False

    def __post_init__(self) -> None:
        _require_id(self.substrate_id, label="substrate_id")
        if not isinstance(self.boundary, ExecutionBoundary):
            raise TypeError("shared substrate boundary is invalid")
        _require_id(
            self.predecessor_parent_atom_id,
            label="predecessor_parent_atom_id",
        )
        if not self.components:
            raise ValueError("shared substrate must name its components")
        if self.physical_scalar_count != sum(
            item.physical_scalar_count for item in self.components
        ):
            raise ValueError("shared substrate scalar accounting does not close")
        if self.owned_by_physical_atoms is not False:
            raise ValueError("shared substrate cannot be assigned to an atom")

    def to_dict(self) -> dict[str, object]:
        return {
            "substrate_id": self.substrate_id,
            "boundary": self.boundary.to_dict(),
            "predecessor_parent_atom_id": self.predecessor_parent_atom_id,
            "components": tuple(item.to_dict() for item in self.components),
            "physical_scalar_count": self.physical_scalar_count,
            "owned_by_physical_atoms": self.owned_by_physical_atoms,
        }


@dataclass(frozen=True, slots=True)
class ParentRefinementReceipt:
    """Exact scalar closure from one predecessor atom into parts + substrate."""

    predecessor_parent_atom_id: str
    predecessor_parent_family_id: str
    child_family: PhysicalAtomFamily | None
    child_atom_count: int
    child_scalars_per_atom: int
    atom_owned_scalar_count: int
    shared_substrate_scalar_count: int
    predecessor_physical_scalar_count: int

    def __post_init__(self) -> None:
        _require_id(
            self.predecessor_parent_atom_id,
            label="predecessor_parent_atom_id",
        )
        _require_id(
            self.predecessor_parent_family_id,
            label="predecessor_parent_family_id",
        )
        if self.child_family is not None and not isinstance(
            self.child_family,
            PhysicalAtomFamily,
        ):
            raise TypeError("child_family is invalid")
        for name in (
            "child_atom_count",
            "child_scalars_per_atom",
            "atom_owned_scalar_count",
            "shared_substrate_scalar_count",
            "predecessor_physical_scalar_count",
        ):
            _require_int(getattr(self, name), label=name)
        if self.atom_owned_scalar_count != (
            self.child_atom_count * self.child_scalars_per_atom
        ):
            raise ValueError("parent child-atom accounting does not close")
        if self.predecessor_physical_scalar_count != (
            self.atom_owned_scalar_count + self.shared_substrate_scalar_count
        ):
            raise ValueError("parent refinement scalar accounting does not close")

    def to_dict(self) -> dict[str, object]:
        return {
            "predecessor_parent_atom_id": self.predecessor_parent_atom_id,
            "predecessor_parent_family_id": self.predecessor_parent_family_id,
            "child_family": (
                None if self.child_family is None else self.child_family.value
            ),
            "child_atom_count": self.child_atom_count,
            "child_scalars_per_atom": self.child_scalars_per_atom,
            "atom_owned_scalar_count": self.atom_owned_scalar_count,
            "shared_substrate_scalar_count": self.shared_substrate_scalar_count,
            "predecessor_physical_scalar_count": (
                self.predecessor_physical_scalar_count
            ),
        }


@dataclass(frozen=True, slots=True)
class LexicalRowViewSpec:
    """A logical vocabulary-row grouping with no independent weight ownership."""

    view_id: str
    row_start: int
    row_stop: int
    functional_role: FunctionalRole
    storage_credit_scalar_count: int = 0
    may_enter_physical_supermode_stage: bool = False

    def __post_init__(self) -> None:
        _require_id(self.view_id, label="view_id")
        _require_int(self.row_start, label="row_start")
        _require_int(self.row_stop, label="row_stop", minimum=1)
        if not 0 <= self.row_start < self.row_stop <= VOCABULARY_SIZE:
            raise ValueError("lexical row view is outside the vocabulary")
        if self.functional_role not in (
            FunctionalRole.LEXICAL_INGRESS,
            FunctionalRole.LEXICAL_READOUT,
        ):
            raise ValueError("lexical row views require a lexical role")
        if self.storage_credit_scalar_count != 0:
            raise ValueError("logical lexical row views receive no storage credit")
        if self.may_enter_physical_supermode_stage is not False:
            raise ValueError("logical lexical row views are not physical atoms")


@dataclass(frozen=True, slots=True)
class Gemma3PartsCatalog:
    """Compact indexed atom families with exact predecessor/source closure."""

    atom_families: tuple[PhysicalAtomFamilySpec, ...]
    shared_substrates: tuple[SharedSubstrateSpec, ...]
    refinement_receipts: tuple[ParentRefinementReceipt, ...]
    atom_owned_scalar_count: int
    shared_substrate_scalar_count: int
    source_unique_parameter_count: int
    total_physical_atom_count: int
    layer_semantics: str = "container_only_not_generator"
    resolution_contract_sha256: str = _RESOLUTION_CONTRACT_SHA256
    schema: str = _CATALOG_SCHEMA
    format_version: int = 1
    catalog_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != _CATALOG_SCHEMA or self.format_version != 1:
            raise ValueError("parts-first catalog header is invalid")
        if self.layer_semantics != "container_only_not_generator":
            raise ValueError("a whole layer may not be declared a generator")
        if self.resolution_contract_sha256 != _RESOLUTION_CONTRACT_SHA256:
            raise ValueError("physical atom resolution contract drifted")
        if tuple(item.family for item in self.atom_families) != (
            PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
            PhysicalAtomFamily.ATTENTION_QO_HEAD,
            PhysicalAtomFamily.MLP_CHANNEL,
        ):
            raise ValueError("physical atom families are not canonical")
        cursor = 0
        for family in self.atom_families:
            if family.global_ordinal_start != cursor:
                raise ValueError("physical atom global ordinals are not contiguous")
            cursor = family.global_ordinal_stop
        if cursor != self.total_physical_atom_count:
            raise ValueError("physical atom universe does not close")
        if self.atom_owned_scalar_count != sum(
            item.physical_scalar_count for item in self.atom_families
        ):
            raise ValueError("catalog atom-owned scalar accounting does not close")
        if self.shared_substrate_scalar_count != sum(
            item.physical_scalar_count for item in self.shared_substrates
        ):
            raise ValueError("catalog shared scalar accounting does not close")
        if self.source_unique_parameter_count != (
            self.atom_owned_scalar_count + self.shared_substrate_scalar_count
        ):
            raise ValueError("parts-first source scalar accounting does not close")
        if sum(
            item.predecessor_physical_scalar_count
            for item in self.refinement_receipts
        ) != self.source_unique_parameter_count:
            raise ValueError("predecessor refinement receipts do not close source")
        computed = _domain_sha256(_CATALOG_DOMAIN, self._payload())
        if self.catalog_sha256 == "":
            object.__setattr__(self, "catalog_sha256", computed)
        elif _require_sha256(
            self.catalog_sha256,
            label="catalog_sha256",
        ) != computed:
            raise ValueError("parts-first catalog hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "architecture": {
                "layer_count": GEMMA3_LAYER_COUNT,
                "hidden_width": HIDDEN_WIDTH,
                "vocabulary_size": VOCABULARY_SIZE,
                "attention_query_head_count": ATTENTION_HEAD_COUNT,
                "attention_key_value_head_count": ATTENTION_KEY_VALUE_HEAD_COUNT,
                "attention_head_dimension": ATTENTION_HEAD_DIMENSION,
                "mlp_channel_count_per_layer": MLP_CHANNEL_COUNT,
            },
            "layer_semantics": self.layer_semantics,
            "resolution_contract_sha256": self.resolution_contract_sha256,
            "atom_families": tuple(item.to_dict() for item in self.atom_families),
            "shared_substrates": tuple(
                item.to_dict() for item in self.shared_substrates
            ),
            "refinement_receipts": tuple(
                item.to_dict() for item in self.refinement_receipts
            ),
            "atom_owned_scalar_count": self.atom_owned_scalar_count,
            "shared_substrate_scalar_count": self.shared_substrate_scalar_count,
            "source_unique_parameter_count": self.source_unique_parameter_count,
            "total_physical_atom_count": self.total_physical_atom_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "catalog_sha256": self.catalog_sha256}

    def family_spec(self, family: PhysicalAtomFamily) -> PhysicalAtomFamilySpec:
        for item in self.atom_families:
            if item.family is family:
                return item
        raise KeyError(f"unknown physical atom family: {family!r}")

    def global_ordinal(self, atom_id: PhysicalAtomId) -> int:
        if not isinstance(atom_id, PhysicalAtomId):
            raise TypeError("global ordinal requires a PhysicalAtomId")
        family = self.family_spec(atom_id.family)
        if atom_id.family is PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN:
            local = atom_id.part_ordinal
        elif atom_id.family is PhysicalAtomFamily.ATTENTION_QO_HEAD:
            assert atom_id.layer_ordinal is not None
            local = atom_id.layer_ordinal * ATTENTION_HEAD_COUNT + atom_id.part_ordinal
        else:
            assert atom_id.layer_ordinal is not None
            local = atom_id.layer_ordinal * MLP_CHANNEL_COUNT + atom_id.part_ordinal
        return family.global_ordinal_start + local

    def atom_id_from_global_ordinal(self, ordinal: int) -> PhysicalAtomId:
        _require_int(ordinal, label="global atom ordinal")
        if ordinal >= self.total_physical_atom_count:
            raise ValueError("global atom ordinal is outside the catalog")
        lexical_stop = LEXICAL_ATOM_COUNT
        attention_stop = lexical_stop + ATTENTION_ATOM_COUNT
        if ordinal < lexical_stop:
            return PhysicalAtomId(
                PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
                ordinal,
            )
        if ordinal < attention_stop:
            local = ordinal - lexical_stop
            return PhysicalAtomId(
                PhysicalAtomFamily.ATTENTION_QO_HEAD,
                local % ATTENTION_HEAD_COUNT,
                local // ATTENTION_HEAD_COUNT,
            )
        local = ordinal - attention_stop
        return PhysicalAtomId(
            PhysicalAtomFamily.MLP_CHANNEL,
            local % MLP_CHANNEL_COUNT,
            local // MLP_CHANNEL_COUNT,
        )

    def resolve(self, atom_id: PhysicalAtomId) -> PhysicalAtomSpec:
        """Resolve one atom without expanding the 37,576-member universe."""

        global_ordinal = self.global_ordinal(atom_id)
        if atom_id.family is PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN:
            column = atom_id.part_ordinal
            parent = "atom.lexical.tied"
            parent_family = "atom_bank.lexical.tied.full_rank"
            slices = (
                ParameterSliceSpec(
                    "model.embed_tokens.weight",
                    1,
                    column,
                    column + 1,
                    VOCABULARY_SIZE,
                ),
            )
            occurrences = (
                FunctionalAtomOccurrence(
                    f"occurrence.lexical.ingress.hidden_column.{column}",
                    atom_id,
                    FunctionalRole.LEXICAL_INGRESS,
                    ExecutionBoundary.lexical_ingress(),
                ),
                FunctionalAtomOccurrence(
                    f"occurrence.lexical.readout.hidden_column.{column}",
                    atom_id,
                    FunctionalRole.LEXICAL_READOUT,
                    ExecutionBoundary.lexical_readout(),
                ),
            )
            scalar_count = LEXICAL_SCALARS_PER_ATOM
        elif atom_id.family is PhysicalAtomFamily.ATTENTION_QO_HEAD:
            assert atom_id.layer_ordinal is not None
            layer = atom_id.layer_ordinal
            head = atom_id.part_ordinal
            start = head * ATTENTION_HEAD_DIMENSION
            stop = start + ATTENTION_HEAD_DIMENSION
            parent = f"atom.attention.layer.{layer}"
            parent_family = _attention_parent_family(layer)
            prefix = f"model.layers.{layer}.self_attn"
            slices = (
                ParameterSliceSpec(
                    f"{prefix}.q_proj.weight",
                    0,
                    start,
                    stop,
                    HIDDEN_WIDTH * ATTENTION_HEAD_DIMENSION,
                ),
                ParameterSliceSpec(
                    f"{prefix}.o_proj.weight",
                    1,
                    start,
                    stop,
                    HIDDEN_WIDTH * ATTENTION_HEAD_DIMENSION,
                ),
            )
            occurrences = (
                FunctionalAtomOccurrence(
                    f"occurrence.layer.{layer}.attention.qo_head.{head}",
                    atom_id,
                    FunctionalRole.ATTENTION_QO_CONTRIBUTION,
                    ExecutionBoundary.attention(layer),
                ),
            )
            scalar_count = ATTENTION_SCALARS_PER_ATOM
        else:
            assert atom_id.layer_ordinal is not None
            layer = atom_id.layer_ordinal
            channel = atom_id.part_ordinal
            parent = f"atom.mlp.layer.{layer}"
            parent_family = "atom_bank.mlp.full_rank"
            prefix = f"model.layers.{layer}.mlp"
            slices = (
                ParameterSliceSpec(
                    f"{prefix}.gate_proj.weight",
                    0,
                    channel,
                    channel + 1,
                    HIDDEN_WIDTH,
                ),
                ParameterSliceSpec(
                    f"{prefix}.up_proj.weight",
                    0,
                    channel,
                    channel + 1,
                    HIDDEN_WIDTH,
                ),
                ParameterSliceSpec(
                    f"{prefix}.down_proj.weight",
                    1,
                    channel,
                    channel + 1,
                    HIDDEN_WIDTH,
                ),
            )
            occurrences = (
                FunctionalAtomOccurrence(
                    f"occurrence.layer.{layer}.mlp.channel.{channel}",
                    atom_id,
                    FunctionalRole.MLP_CHANNEL_CONTRIBUTION,
                    ExecutionBoundary.mlp(layer),
                ),
            )
            scalar_count = MLP_SCALARS_PER_ATOM
        return PhysicalAtomSpec(
            atom_id=atom_id,
            global_ordinal=global_ordinal,
            scalar_count=scalar_count,
            predecessor_parent_atom_id=parent,
            predecessor_parent_family_id=parent_family,
            parameter_slices=slices,
            occurrences=occurrences,
        )

    def occurrence_at(
        self,
        atom_id: PhysicalAtomId,
        boundary: ExecutionBoundary,
    ) -> FunctionalAtomOccurrence:
        matches = tuple(
            item
            for item in self.resolve(atom_id).occurrences
            if item.boundary == boundary
        )
        if len(matches) != 1:
            raise ValueError("physical atom has no unique occurrence at boundary")
        return matches[0]


def _attention_parent_family(layer: int) -> str:
    if layer in _GLOBAL_ATTENTION_LAYERS:
        return "atom_bank.attention.global.full_rank"
    return "atom_bank.attention.sliding.full_rank"


def _component(
    component_id: str,
    physical_scalar_count: int,
) -> SubstrateComponentSpec:
    return SubstrateComponentSpec(
        component_id,
        physical_scalar_count,
        parameter_free=physical_scalar_count == 0,
    )


def build_gemma3_parts_catalog() -> Gemma3PartsCatalog:
    """Build the canonical compact parts-first catalog for Gemma-3 270M."""

    families = (
        PhysicalAtomFamilySpec(
            PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
            0,
            LEXICAL_ATOM_COUNT,
            LEXICAL_SCALARS_PER_ATOM,
            LEXICAL_ATOM_SCALARS,
        ),
        PhysicalAtomFamilySpec(
            PhysicalAtomFamily.ATTENTION_QO_HEAD,
            LEXICAL_ATOM_COUNT,
            ATTENTION_ATOM_COUNT,
            ATTENTION_SCALARS_PER_ATOM,
            ATTENTION_ATOM_SCALARS,
        ),
        PhysicalAtomFamilySpec(
            PhysicalAtomFamily.MLP_CHANNEL,
            LEXICAL_ATOM_COUNT + ATTENTION_ATOM_COUNT,
            MLP_ATOM_COUNT,
            MLP_SCALARS_PER_ATOM,
            MLP_ATOM_SCALARS,
        ),
    )
    substrates: list[SharedSubstrateSpec] = []
    receipts: list[ParentRefinementReceipt] = [
        ParentRefinementReceipt(
            "atom.lexical.tied",
            "atom_bank.lexical.tied.full_rank",
            PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
            HIDDEN_WIDTH,
            VOCABULARY_SIZE,
            LEXICAL_ATOM_SCALARS,
            0,
            LEXICAL_ATOM_SCALARS,
        )
    ]
    for layer in range(GEMMA3_LAYER_COUNT):
        attention_parent = f"atom.attention.layer.{layer}"
        attention_substrate = SharedSubstrateSpec(
            f"substrate.layer.{layer}.attention.shared",
            ExecutionBoundary.attention(layer),
            attention_parent,
            (
                _component("k_projection", HIDDEN_WIDTH * ATTENTION_HEAD_DIMENSION),
                _component("v_projection", HIDDEN_WIDTH * ATTENTION_HEAD_DIMENSION),
                _component("query_rmsnorm", ATTENTION_HEAD_DIMENSION),
                _component("key_rmsnorm", ATTENTION_HEAD_DIMENSION),
                _component("attention_input_rmsnorm", HIDDEN_WIDTH),
                _component("attention_output_rmsnorm", HIDDEN_WIDTH),
                _component("grouped_query_broadcast", 0),
                _component("causal_mask", 0),
                _component("rope", 0),
                _component("softmax", 0),
            ),
            ATTENTION_SHARED_SCALARS_PER_LAYER,
        )
        substrates.append(attention_substrate)
        receipts.append(
            ParentRefinementReceipt(
                attention_parent,
                _attention_parent_family(layer),
                PhysicalAtomFamily.ATTENTION_QO_HEAD,
                ATTENTION_HEAD_COUNT,
                ATTENTION_SCALARS_PER_ATOM,
                ATTENTION_HEAD_COUNT * ATTENTION_SCALARS_PER_ATOM,
                ATTENTION_SHARED_SCALARS_PER_LAYER,
                ATTENTION_HEAD_COUNT * ATTENTION_SCALARS_PER_ATOM
                + ATTENTION_SHARED_SCALARS_PER_LAYER,
            )
        )
        mlp_parent = f"atom.mlp.layer.{layer}"
        mlp_substrate = SharedSubstrateSpec(
            f"substrate.layer.{layer}.mlp.shared",
            ExecutionBoundary.mlp(layer),
            mlp_parent,
            (
                _component("mlp_input_rmsnorm", HIDDEN_WIDTH),
                _component("mlp_output_rmsnorm", HIDDEN_WIDTH),
                _component("gated_gelu", 0),
            ),
            MLP_SHARED_SCALARS_PER_LAYER,
        )
        substrates.append(mlp_substrate)
        receipts.append(
            ParentRefinementReceipt(
                mlp_parent,
                "atom_bank.mlp.full_rank",
                PhysicalAtomFamily.MLP_CHANNEL,
                MLP_CHANNEL_COUNT,
                MLP_SCALARS_PER_ATOM,
                MLP_CHANNEL_COUNT * MLP_SCALARS_PER_ATOM,
                MLP_SHARED_SCALARS_PER_LAYER,
                MLP_CHANNEL_COUNT * MLP_SCALARS_PER_ATOM
                + MLP_SHARED_SCALARS_PER_LAYER,
            )
        )
    substrates.append(
        SharedSubstrateSpec(
            "substrate.terminal.rmsnorm",
            ExecutionBoundary.terminal_norm(),
            "terminal.rmsnorm",
            (_component("terminal_rmsnorm", HIDDEN_WIDTH),),
            HIDDEN_WIDTH,
        )
    )
    receipts.append(
        ParentRefinementReceipt(
            "terminal.rmsnorm",
            "terminal.exact",
            None,
            0,
            0,
            0,
            HIDDEN_WIDTH,
            HIDDEN_WIDTH,
        )
    )
    catalog = Gemma3PartsCatalog(
        atom_families=families,
        shared_substrates=tuple(substrates),
        refinement_receipts=tuple(receipts),
        atom_owned_scalar_count=ATOM_OWNED_SCALAR_COUNT,
        shared_substrate_scalar_count=SHARED_SUBSTRATE_SCALAR_COUNT,
        source_unique_parameter_count=SOURCE_UNIQUE_PARAMETER_COUNT,
        total_physical_atom_count=TOTAL_PHYSICAL_ATOM_COUNT,
    )
    return validate_gemma3_parts_catalog(catalog)


def _validate_all_resolved_atoms(catalog: Gemma3PartsCatalog) -> None:
    """Exhaustively bind every compact ordinal to its slices and occurrences."""

    for ordinal in range(TOTAL_PHYSICAL_ATOM_COUNT):
        atom_id = catalog.atom_id_from_global_ordinal(ordinal)
        resolved = catalog.resolve(atom_id)
        if resolved.atom_id != atom_id or resolved.global_ordinal != ordinal:
            raise ValueError("physical atom ordinal resolver drifted")
        slices = tuple(
            (
                item.parameter_name,
                item.axis,
                item.start,
                item.stop,
                item.scalar_count,
            )
            for item in resolved.parameter_slices
        )
        occurrences = tuple(
            (
                item.occurrence_id,
                item.atom_id,
                item.role,
                item.boundary,
            )
            for item in resolved.occurrences
        )
        if ordinal < HIDDEN_WIDTH:
            column = ordinal
            expected_atom = PhysicalAtomId(
                PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
                column,
            )
            expected_scalar_count = VOCABULARY_SIZE
            expected_parent = "atom.lexical.tied"
            expected_parent_family = "atom_bank.lexical.tied.full_rank"
            expected_slices = (
                (
                    "model.embed_tokens.weight",
                    1,
                    column,
                    column + 1,
                    VOCABULARY_SIZE,
                ),
            )
            expected_occurrences = (
                (
                    f"occurrence.lexical.ingress.hidden_column.{column}",
                    expected_atom,
                    FunctionalRole.LEXICAL_INGRESS,
                    ExecutionBoundary.lexical_ingress(),
                ),
                (
                    f"occurrence.lexical.readout.hidden_column.{column}",
                    expected_atom,
                    FunctionalRole.LEXICAL_READOUT,
                    ExecutionBoundary.lexical_readout(),
                ),
            )
        elif ordinal < HIDDEN_WIDTH + ATTENTION_ATOM_COUNT:
            local = ordinal - HIDDEN_WIDTH
            layer = local // ATTENTION_HEAD_COUNT
            head = local % ATTENTION_HEAD_COUNT
            expected_atom = PhysicalAtomId(
                PhysicalAtomFamily.ATTENTION_QO_HEAD,
                head,
                layer,
            )
            expected_scalar_count = ATTENTION_SCALARS_PER_ATOM
            expected_parent = f"atom.attention.layer.{layer}"
            expected_parent_family = _attention_parent_family(layer)
            start = head * ATTENTION_HEAD_DIMENSION
            stop = start + ATTENTION_HEAD_DIMENSION
            prefix = f"model.layers.{layer}.self_attn"
            expected_slices = (
                (
                    f"{prefix}.q_proj.weight",
                    0,
                    start,
                    stop,
                    HIDDEN_WIDTH * ATTENTION_HEAD_DIMENSION,
                ),
                (
                    f"{prefix}.o_proj.weight",
                    1,
                    start,
                    stop,
                    HIDDEN_WIDTH * ATTENTION_HEAD_DIMENSION,
                ),
            )
            expected_occurrences = (
                (
                    f"occurrence.layer.{layer}.attention.qo_head.{head}",
                    expected_atom,
                    FunctionalRole.ATTENTION_QO_CONTRIBUTION,
                    ExecutionBoundary.attention(layer),
                ),
            )
        else:
            local = ordinal - HIDDEN_WIDTH - ATTENTION_ATOM_COUNT
            layer = local // MLP_CHANNEL_COUNT
            channel = local % MLP_CHANNEL_COUNT
            expected_atom = PhysicalAtomId(
                PhysicalAtomFamily.MLP_CHANNEL,
                channel,
                layer,
            )
            expected_scalar_count = MLP_SCALARS_PER_ATOM
            expected_parent = f"atom.mlp.layer.{layer}"
            expected_parent_family = "atom_bank.mlp.full_rank"
            prefix = f"model.layers.{layer}.mlp"
            expected_slices = (
                (
                    f"{prefix}.gate_proj.weight",
                    0,
                    channel,
                    channel + 1,
                    HIDDEN_WIDTH,
                ),
                (
                    f"{prefix}.up_proj.weight",
                    0,
                    channel,
                    channel + 1,
                    HIDDEN_WIDTH,
                ),
                (
                    f"{prefix}.down_proj.weight",
                    1,
                    channel,
                    channel + 1,
                    HIDDEN_WIDTH,
                ),
            )
            expected_occurrences = (
                (
                    f"occurrence.layer.{layer}.mlp.channel.{channel}",
                    expected_atom,
                    FunctionalRole.MLP_CHANNEL_CONTRIBUTION,
                    ExecutionBoundary.mlp(layer),
                ),
            )
        if (
            atom_id != expected_atom
            or catalog.global_ordinal(atom_id) != ordinal
            or resolved.scalar_count != expected_scalar_count
            or resolved.predecessor_parent_atom_id != expected_parent
            or resolved.predecessor_parent_family_id != expected_parent_family
            or slices != expected_slices
            or occurrences != expected_occurrences
        ):
            raise ValueError("physical atom resolution inventory drifted")


def validate_gemma3_parts_catalog(catalog: Gemma3PartsCatalog) -> Gemma3PartsCatalog:
    """Strictly validate architecture, ordinals, substrate, and parent closure."""

    if not isinstance(catalog, Gemma3PartsCatalog):
        raise TypeError("catalog must be a Gemma3PartsCatalog")
    expected_family_rows = (
        (
            PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
            0,
            HIDDEN_WIDTH,
            VOCABULARY_SIZE,
            LEXICAL_ATOM_SCALARS,
        ),
        (
            PhysicalAtomFamily.ATTENTION_QO_HEAD,
            HIDDEN_WIDTH,
            ATTENTION_ATOM_COUNT,
            ATTENTION_SCALARS_PER_ATOM,
            ATTENTION_ATOM_SCALARS,
        ),
        (
            PhysicalAtomFamily.MLP_CHANNEL,
            HIDDEN_WIDTH + ATTENTION_ATOM_COUNT,
            MLP_ATOM_COUNT,
            MLP_SCALARS_PER_ATOM,
            MLP_ATOM_SCALARS,
        ),
    )
    observed_family_rows = tuple(
        (
            item.family,
            item.global_ordinal_start,
            item.atom_count,
            item.scalars_per_atom,
            item.physical_scalar_count,
        )
        for item in catalog.atom_families
    )
    if observed_family_rows != expected_family_rows:
        raise ValueError("parts-first physical family inventory drifted")
    if (
        catalog.atom_owned_scalar_count != 262_144_000
        or catalog.shared_substrate_scalar_count != 5_954_176
        or catalog.source_unique_parameter_count != 268_098_176
        or catalog.total_physical_atom_count != 37_576
    ):
        raise ValueError("pinned Gemma parts accounting drifted")
    if len(catalog.shared_substrates) != 37:
        raise ValueError("shared substrate inventory must contain 18+18+1 rows")
    if len(catalog.refinement_receipts) != 38:
        raise ValueError("predecessor refinement inventory must contain 1+18+18+1 rows")
    expected_parents = {"atom.lexical.tied", "terminal.rmsnorm"}
    expected_parents.update(
        f"atom.attention.layer.{layer}" for layer in range(GEMMA3_LAYER_COUNT)
    )
    expected_parents.update(
        f"atom.mlp.layer.{layer}" for layer in range(GEMMA3_LAYER_COUNT)
    )
    if {
        item.predecessor_parent_atom_id for item in catalog.refinement_receipts
    } != expected_parents:
        raise ValueError("predecessor refinement parent coverage drifted")
    substrate_by_parent = {
        item.predecessor_parent_atom_id: item for item in catalog.shared_substrates
    }
    expected_substrate_rows: list[tuple[object, ...]] = []
    for layer in range(GEMMA3_LAYER_COUNT):
        expected_substrate_rows.extend(
            (
                (
                    f"substrate.layer.{layer}.attention.shared",
                    f"atom.attention.layer.{layer}",
                    ExecutionBoundary.attention(layer),
                    (
                        ("k_projection", 163_840),
                        ("v_projection", 163_840),
                        ("query_rmsnorm", 256),
                        ("key_rmsnorm", 256),
                        ("attention_input_rmsnorm", 640),
                        ("attention_output_rmsnorm", 640),
                        ("grouped_query_broadcast", 0),
                        ("causal_mask", 0),
                        ("rope", 0),
                        ("softmax", 0),
                    ),
                    ATTENTION_SHARED_SCALARS_PER_LAYER,
                    False,
                ),
                (
                    f"substrate.layer.{layer}.mlp.shared",
                    f"atom.mlp.layer.{layer}",
                    ExecutionBoundary.mlp(layer),
                    (
                        ("mlp_input_rmsnorm", 640),
                        ("mlp_output_rmsnorm", 640),
                        ("gated_gelu", 0),
                    ),
                    MLP_SHARED_SCALARS_PER_LAYER,
                    False,
                ),
            )
        )
    expected_substrate_rows.append(
        (
            "substrate.terminal.rmsnorm",
            "terminal.rmsnorm",
            ExecutionBoundary.terminal_norm(),
            (("terminal_rmsnorm", 640),),
            TERMINAL_SHARED_SCALARS,
            False,
        )
    )
    observed_substrate_rows = tuple(
        (
            item.substrate_id,
            item.predecessor_parent_atom_id,
            item.boundary,
            tuple(
                (component.component_id, component.physical_scalar_count)
                for component in item.components
            ),
            item.physical_scalar_count,
            item.owned_by_physical_atoms,
        )
        for item in catalog.shared_substrates
    )
    if observed_substrate_rows != tuple(expected_substrate_rows):
        raise ValueError("shared substrate component inventory drifted")
    expected_receipt_rows: list[tuple[object, ...]] = [
        (
            "atom.lexical.tied",
            "atom_bank.lexical.tied.full_rank",
            PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
            HIDDEN_WIDTH,
            VOCABULARY_SIZE,
            LEXICAL_ATOM_SCALARS,
            0,
            LEXICAL_ATOM_SCALARS,
        )
    ]
    for layer in range(GEMMA3_LAYER_COUNT):
        expected_receipt_rows.extend(
            (
                (
                    f"atom.attention.layer.{layer}",
                    _attention_parent_family(layer),
                    PhysicalAtomFamily.ATTENTION_QO_HEAD,
                    ATTENTION_HEAD_COUNT,
                    ATTENTION_SCALARS_PER_ATOM,
                    ATTENTION_HEAD_COUNT * ATTENTION_SCALARS_PER_ATOM,
                    ATTENTION_SHARED_SCALARS_PER_LAYER,
                    ATTENTION_HEAD_COUNT * ATTENTION_SCALARS_PER_ATOM
                    + ATTENTION_SHARED_SCALARS_PER_LAYER,
                ),
                (
                    f"atom.mlp.layer.{layer}",
                    "atom_bank.mlp.full_rank",
                    PhysicalAtomFamily.MLP_CHANNEL,
                    MLP_CHANNEL_COUNT,
                    MLP_SCALARS_PER_ATOM,
                    MLP_CHANNEL_COUNT * MLP_SCALARS_PER_ATOM,
                    MLP_SHARED_SCALARS_PER_LAYER,
                    MLP_CHANNEL_COUNT * MLP_SCALARS_PER_ATOM
                    + MLP_SHARED_SCALARS_PER_LAYER,
                ),
            )
        )
    expected_receipt_rows.append(
        (
            "terminal.rmsnorm",
            "terminal.exact",
            None,
            0,
            0,
            0,
            TERMINAL_SHARED_SCALARS,
            TERMINAL_SHARED_SCALARS,
        )
    )
    observed_receipt_rows = tuple(
        (
            item.predecessor_parent_atom_id,
            item.predecessor_parent_family_id,
            item.child_family,
            item.child_atom_count,
            item.child_scalars_per_atom,
            item.atom_owned_scalar_count,
            item.shared_substrate_scalar_count,
            item.predecessor_physical_scalar_count,
        )
        for item in catalog.refinement_receipts
    )
    if observed_receipt_rows != tuple(expected_receipt_rows):
        raise ValueError("predecessor refinement receipt inventory drifted")
    for receipt in catalog.refinement_receipts:
        substrate = substrate_by_parent.get(receipt.predecessor_parent_atom_id)
        observed_shared = 0 if substrate is None else substrate.physical_scalar_count
        if observed_shared != receipt.shared_substrate_scalar_count:
            raise ValueError("refinement receipt/substrate binding drifted")
    if catalog.catalog_sha256 not in _STRICTLY_VALIDATED_CATALOG_SHA256S:
        _validate_all_resolved_atoms(catalog)
        _STRICTLY_VALIDATED_CATALOG_SHA256S.add(catalog.catalog_sha256)
    return catalog


@dataclass(frozen=True, slots=True)
class AtomGroupDraft:
    """One logical group request before compact sidecar encoding."""

    group_id: str
    boundary: ExecutionBoundary
    atom_ids: tuple[PhysicalAtomId, ...]
    group_semantics: str = "logical_overlap"

    def __post_init__(self) -> None:
        _require_id(self.group_id, label="group_id")
        if not isinstance(self.boundary, ExecutionBoundary):
            raise TypeError("group boundary is invalid")
        if not self.atom_ids:
            raise ValueError("atom group must contain at least one atom")
        if self.group_semantics not in {"logical_overlap", "exact_partition"}:
            raise ValueError("group semantics are invalid")
        if any(not isinstance(item, PhysicalAtomId) for item in self.atom_ids):
            raise TypeError(
                "atom group members must be physical atoms; lexical row views "
                "cannot enter a physical supermode stage"
            )


@dataclass(frozen=True, slots=True)
class AtomGroupMembershipRef:
    """Authenticated slice of sorted unique uint32 global atom ordinals."""

    group_id: str
    atom_universe_sha256: str
    encoding: str
    blob_sha256: str
    slice_sha256: str
    byte_offset: int
    byte_length: int
    member_count: int
    minimum_global_ordinal: int
    maximum_global_ordinal: int
    atom_family: PhysicalAtomFamily
    boundary: ExecutionBoundary
    group_semantics: str
    storage_credit_scalar_count: int = 0

    def __post_init__(self) -> None:
        _require_id(self.group_id, label="group_id")
        _require_sha256(
            self.atom_universe_sha256,
            label="atom_universe_sha256",
        )
        if self.encoding != _GROUP_ENCODING:
            raise ValueError("atom group membership encoding is unsupported")
        _require_sha256(self.blob_sha256, label="blob_sha256")
        _require_sha256(self.slice_sha256, label="slice_sha256")
        for name in (
            "byte_offset",
            "byte_length",
            "member_count",
            "minimum_global_ordinal",
            "maximum_global_ordinal",
        ):
            _require_int(getattr(self, name), label=name)
        if self.member_count == 0 or self.byte_length != 4 * self.member_count:
            raise ValueError("atom group sidecar length/count is invalid")
        if self.minimum_global_ordinal > self.maximum_global_ordinal:
            raise ValueError("atom group ordinal range is invalid")
        if not isinstance(self.atom_family, PhysicalAtomFamily):
            raise TypeError("atom group family is invalid")
        if self.group_semantics not in {"logical_overlap", "exact_partition"}:
            raise ValueError("atom group semantics are invalid")
        if self.storage_credit_scalar_count != 0:
            raise ValueError("logical atom groups receive no storage credit")


def _canonical_group_ordinals(
    catalog: Gemma3PartsCatalog,
    draft: AtomGroupDraft,
) -> tuple[int, ...]:
    ordinals = tuple(catalog.global_ordinal(item) for item in draft.atom_ids)
    canonical = tuple(sorted(set(ordinals)))
    if ordinals != canonical:
        raise ValueError("atom group members must be unique and canonically ordered")
    family = draft.atom_ids[0].family
    if any(item.family is not family for item in draft.atom_ids):
        raise ValueError("one atom group must be homogeneous")
    for atom_id in draft.atom_ids:
        catalog.occurrence_at(atom_id, draft.boundary)
    return ordinals


def build_atom_group_sidecar(
    catalog: Gemma3PartsCatalog,
    drafts: Sequence[AtomGroupDraft],
) -> tuple[bytes, tuple[AtomGroupMembershipRef, ...]]:
    """Encode same-boundary groups; overlapping logical groups are permitted."""

    validate_gemma3_parts_catalog(catalog)
    if not isinstance(drafts, Sequence) or not drafts:
        raise ValueError("at least one atom group draft is required")
    if any(not isinstance(item, AtomGroupDraft) for item in drafts):
        raise TypeError("drafts must contain AtomGroupDraft values")
    group_ids = tuple(item.group_id for item in drafts)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("atom group IDs must be unique")
    encoded_slices: list[bytes] = []
    ordinals_by_group: list[tuple[int, ...]] = []
    for draft in drafts:
        ordinals = _canonical_group_ordinals(catalog, draft)
        ordinals_by_group.append(ordinals)
        encoded_slices.append(b"".join(struct.pack("<I", value) for value in ordinals))
    blob = b"".join(encoded_slices)
    blob_sha256 = _sha256(blob)
    refs: list[AtomGroupMembershipRef] = []
    offset = 0
    for draft, ordinals, encoded in zip(
        drafts,
        ordinals_by_group,
        encoded_slices,
        strict=True,
    ):
        refs.append(
            AtomGroupMembershipRef(
                group_id=draft.group_id,
                atom_universe_sha256=catalog.catalog_sha256,
                encoding=_GROUP_ENCODING,
                blob_sha256=blob_sha256,
                slice_sha256=_sha256(encoded),
                byte_offset=offset,
                byte_length=len(encoded),
                member_count=len(ordinals),
                minimum_global_ordinal=ordinals[0],
                maximum_global_ordinal=ordinals[-1],
                atom_family=draft.atom_ids[0].family,
                boundary=draft.boundary,
                group_semantics=draft.group_semantics,
            )
        )
        offset += len(encoded)
    return blob, tuple(refs)


def validate_atom_group_membership_ref(
    catalog: Gemma3PartsCatalog,
    group: AtomGroupMembershipRef,
    blob: bytes,
) -> tuple[PhysicalAtomId, ...]:
    """Authenticate and resolve one compact group membership slice."""

    validate_gemma3_parts_catalog(catalog)
    if not isinstance(group, AtomGroupMembershipRef):
        raise TypeError("group must be an AtomGroupMembershipRef")
    if not isinstance(blob, bytes):
        raise TypeError("atom group blob must be bytes")
    if group.atom_universe_sha256 != catalog.catalog_sha256:
        raise ValueError("atom group belongs to another atom universe")
    if _sha256(blob) != group.blob_sha256:
        raise ValueError("atom group blob hash mismatch")
    stop = group.byte_offset + group.byte_length
    if stop > len(blob):
        raise ValueError("atom group slice exceeds its blob")
    encoded = blob[group.byte_offset:stop]
    if _sha256(encoded) != group.slice_sha256:
        raise ValueError("atom group slice hash mismatch")
    ordinals = tuple(
        value[0] for value in struct.iter_unpack("<I", encoded)
    )
    if (
        len(ordinals) != group.member_count
        or ordinals != tuple(sorted(set(ordinals)))
        or ordinals[0] != group.minimum_global_ordinal
        or ordinals[-1] != group.maximum_global_ordinal
    ):
        raise ValueError("atom group sidecar members are not canonical")
    atoms = tuple(catalog.atom_id_from_global_ordinal(value) for value in ordinals)
    if any(item.family is not group.atom_family for item in atoms):
        raise ValueError("atom group sidecar is not homogeneous")
    for atom in atoms:
        catalog.occurrence_at(atom, group.boundary)
    return atoms


def validate_exact_atom_partition(
    catalog: Gemma3PartsCatalog,
    groups: Sequence[AtomGroupMembershipRef],
    blob: bytes,
    *,
    atom_family: PhysicalAtomFamily,
    boundary: ExecutionBoundary,
) -> tuple[AtomGroupMembershipRef, ...]:
    """Require disjoint/exhaustive groups for one family's boundary occurrence."""

    if not groups:
        raise ValueError("an exact partition requires at least one group")
    seen: set[int] = set()
    for group in groups:
        if (
            group.group_semantics != "exact_partition"
            or group.atom_family is not atom_family
            or group.boundary != boundary
        ):
            raise ValueError("exact partition groups must share family and boundary")
        atoms = validate_atom_group_membership_ref(catalog, group, blob)
        ordinals = {catalog.global_ordinal(item) for item in atoms}
        if seen.intersection(ordinals):
            raise ValueError("exact atom partition groups overlap")
        seen.update(ordinals)
    family = catalog.family_spec(atom_family)
    if atom_family is PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN:
        expected = set(range(family.global_ordinal_start, family.global_ordinal_stop))
    elif atom_family is PhysicalAtomFamily.ATTENTION_QO_HEAD:
        if boundary.boundary_kind is not BoundaryKind.ATTENTION:
            raise ValueError("attention partition requires an attention boundary")
        assert boundary.layer_ordinal is not None
        start = family.global_ordinal_start + (
            boundary.layer_ordinal * ATTENTION_HEAD_COUNT
        )
        expected = set(range(start, start + ATTENTION_HEAD_COUNT))
    else:
        if boundary.boundary_kind is not BoundaryKind.MLP:
            raise ValueError("MLP partition requires an MLP boundary")
        assert boundary.layer_ordinal is not None
        start = family.global_ordinal_start + boundary.layer_ordinal * MLP_CHANNEL_COUNT
        expected = set(range(start, start + MLP_CHANNEL_COUNT))
    if seen != expected:
        raise ValueError("exact atom partition is not exhaustive")
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class SupermodeStepSpec:
    """One homogeneous atom coalition at one execution boundary."""

    step_id: str
    atom_group: AtomGroupMembershipRef
    chart_id: str
    input_carrier_abi: str
    additive_output_abi: str
    post_boundary_abi: str
    within_boundary_operator: str = "additive_coalition"

    def __post_init__(self) -> None:
        _require_id(self.step_id, label="step_id")
        if not isinstance(self.atom_group, AtomGroupMembershipRef):
            if isinstance(self.atom_group, LexicalRowViewSpec):
                raise TypeError(
                    "lexical row views are logical and cannot enter a physical "
                    "supermode step"
                )
            raise TypeError("supermode step requires a physical atom group")
        for name in (
            "chart_id",
            "input_carrier_abi",
            "additive_output_abi",
            "post_boundary_abi",
        ):
            _require_id(getattr(self, name), label=name)
        if self.within_boundary_operator != "additive_coalition":
            raise ValueError("atoms may only add inside one execution boundary")

    @property
    def boundary(self) -> ExecutionBoundary:
        return self.atom_group.boundary


@dataclass(frozen=True, slots=True)
class SupermodePathSpec:
    """An ordered program of local atom coalitions across native boundaries."""

    path_id: str
    atom_universe_sha256: str
    steps: tuple[SupermodeStepSpec, ...]
    cross_boundary_operator: str = "ordered_composition"
    generator_subject: str = "physical_atom_coalitions"
    layer_semantics: str = "container_only_not_generator"
    storage_credit_scalar_count: int = 0
    cross_layer_claimed: bool = False

    def __post_init__(self) -> None:
        _require_id(self.path_id, label="path_id")
        _require_sha256(
            self.atom_universe_sha256,
            label="atom_universe_sha256",
        )
        if len(self.steps) < 2:
            raise ValueError("supermode path must contain at least two steps")
        if any(not isinstance(item, SupermodeStepSpec) for item in self.steps):
            raise TypeError("supermode path steps are invalid")
        if self.cross_boundary_operator != "ordered_composition":
            raise ValueError(
                "cross-boundary atom effects must compose in execution order; "
                "they cannot be summed"
            )
        if self.generator_subject != "physical_atom_coalitions":
            raise ValueError("a whole layer is a container, not a generator")
        if self.layer_semantics != "container_only_not_generator":
            raise ValueError("a whole layer is a container, not a generator")
        if self.storage_credit_scalar_count != 0:
            raise ValueError("a logical supermode path receives no storage credit")
        if type(self.cross_layer_claimed) is not bool:
            raise TypeError("cross_layer_claimed must be bool")
        ordinals = tuple(item.boundary.execution_ordinal for item in self.steps)
        if ordinals != tuple(sorted(set(ordinals))):
            raise ValueError(
                "supermode steps must occupy strictly increasing execution boundaries"
            )
        step_ids = tuple(item.step_id for item in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("supermode step IDs must be unique")
        layer_ordinals = {
            item.boundary.layer_ordinal
            for item in self.steps
            if item.boundary.layer_ordinal is not None
        }
        if self.cross_layer_claimed and len(layer_ordinals) < 2:
            raise ValueError(
                "a cross-layer claim requires at least two distinct layer containers"
            )


def validate_supermode_path(
    path: SupermodePathSpec,
    catalog: Gemma3PartsCatalog,
    group_blob: bytes,
) -> SupermodePathSpec:
    """Bind every step to the catalog and authenticate group membership."""

    if not isinstance(path, SupermodePathSpec):
        raise TypeError("path must be a SupermodePathSpec")
    validate_gemma3_parts_catalog(catalog)
    if path.atom_universe_sha256 != catalog.catalog_sha256:
        raise ValueError("supermode path belongs to another atom universe")
    for step in path.steps:
        validate_atom_group_membership_ref(catalog, step.atom_group, group_blob)
    return path


# Import-time arithmetic guards are intentionally scalar-only.
assert LEXICAL_ATOM_SCALARS == 167_772_160
assert ATTENTION_ATOM_SCALARS == 23_592_960
assert MLP_ATOM_SCALARS == 70_778_880
assert ATOM_OWNED_SCALAR_COUNT == 262_144_000
assert ATTENTION_SHARED_SCALARS_PER_LAYER == 329_472
assert SHARED_SUBSTRATE_SCALAR_COUNT == 5_954_176
assert SOURCE_UNIQUE_PARAMETER_COUNT == 268_098_176
assert TOTAL_PHYSICAL_ATOM_COUNT == 37_576
