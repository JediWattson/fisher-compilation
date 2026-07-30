"""Prompt-free commitment for one fresh H4 damping selection panel.

This module creates a deliberately narrow boundary between the completed
expanded-fit work and one new finite-NLL selection panel:

* the source side is a prompt-free expanded-fit lineage receipt;
* the only raw input accepted is one caller-supplied *new* selection file;
* the selection family schedule is fixed at eight families and two prompts
  per family;
* exact prompt and family identities are rejected if they overlap any
  prompt-free development metadata carried by the expanded corpus;
* the frozen Calibration-B manifest is consulted only through its prompt-
  blind hash-to-family map; and
* the published artifact contains hashes and family IDs, never prompt text.

There is intentionally no selection-search loop, guard path, assessment path,
model loader, tokenizer, or candidate executable here.  The resulting
artifact is an immutable membership commitment that a later one-shot
finite-NLL runner can authenticate before it opens the new selection input.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
    frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest,
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveACorpusArtifact,
    _load_progressive_a_artifact,
)


__all__ = [
    "FRESH_DAMPING_SELECTION_FAMILIES",
    "FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE",
    "GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_FORMAT_VERSION",
    "GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID",
    "GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_SCHEMA",
    "GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE",
    "GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE_FORMAT_VERSION",
    "GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE_SCHEMA",
    "Gemma3L3L4H4DampingExpandedFitLineage",
    "Gemma3L3L4H4DampingSelectionPanelArtifact",
    "Gemma3L3L4H4DampingSelectionPanelClosedError",
    "Gemma3L3L4H4DampingSelectionPanelError",
    "Gemma3L3L4H4DampingSelectionPanelIntegrityError",
    "Gemma3L3L4H4DampingSelectionPanelSource",
    "Gemma3L3L4H4DampingSelectionRoleInput",
    "expanded_fit_lineage_from_corpus_artifact",
    "freeze_gemma3_l3_l4_h4_damping_selection_panel",
    "load_gemma3_l3_l4_h4_damping_expanded_fit_lineage",
    "load_gemma3_l3_l4_h4_damping_selection_panel_artifact",
    "load_gemma3_l3_l4_h4_damping_selection_role_input",
    "write_gemma3_l3_l4_h4_damping_selection_panel_artifact",
    "write_gemma3_l3_l4_h4_damping_selection_role_input",
]


GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_h4_damping_selection_panel"
)
GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_FORMAT_VERSION = 1
GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_h4_damping_selection_role"
)
GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE_FORMAT_VERSION = 1
GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID = (
    "gemma3-l3-l4-h4-damping-finite-nll-selection-v1"
)
# Keep the canonical compiler role for downstream panel materialization.  The
# new identity comes from the panel-specific schema, manifest, and membership.
GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE = "calibration_a_selection"

FRESH_DAMPING_SELECTION_FAMILIES = (
    "progressive-damping-selection-v1-algorithm-execution",
    "progressive-damping-selection-v1-evidence-attribution",
    "progressive-damping-selection-v1-formal-validity",
    "progressive-damping-selection-v1-rule-exceptions",
    "progressive-damping-selection-v1-structured-extraction",
    "progressive-damping-selection-v1-symbolic-equivalence",
    "progressive-damping-selection-v1-unit-consistency",
    "progressive-damping-selection-v1-verbal-entailment",
)
FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE = (
    FRESH_DAMPING_SELECTION_FAMILIES
    + FRESH_DAMPING_SELECTION_FAMILIES
)

_LINEAGE_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_h4_damping_expanded_fit_lineage"
)
_LINEAGE_FORMAT_VERSION = 1
_MANIFEST_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_h4_damping_selection_manifest"
)
_ROLE_INPUT_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "panel_id",
        "role",
        "prompts",
        "family_ids",
    }
)
_LINEAGE_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "expanded_corpus_artifact_sha256",
        "tokenizer_contract_sha256",
        "fit_manifest_sha256",
        "fit_role_input_file_sha256",
        "fit_binding_sha256",
        "fit_example_count",
        "fit_family_ids",
        "ordered_fit_prompt_sha256s",
        "ordered_fit_family_ids",
        "occupied_development_manifest_sha256s",
        "occupied_development_prompt_sha256s",
        "occupied_development_family_ids",
        "forbidden_assessment_manifest_sha256s",
        "receipt_sha256",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "panel_id",
        "role",
        "expanded_fit_lineage",
        "selection",
        "policy",
        "safety",
        "manifest_sha256",
        "membership_receipt_sha256",
        "artifact_sha256",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "role_input_file_sha256",
        "example_count",
        "family_count",
        "examples_per_family",
        "family_ids",
        "ordered_members",
    }
)
_MEMBER_FIELDS = frozenset({"prompt_sha256", "family_id"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_LINEAGE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-fit-lineage:v1\0"
)
_MANIFEST_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-selection-manifest:v1\0"
)
_MEMBERSHIP_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-selection-membership:v1\0"
)
_ARTIFACT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-selection-artifact:v1\0"
)


class Gemma3L3L4H4DampingSelectionPanelError(RuntimeError):
    """Base class for fresh damping-selection commitment failures."""


class Gemma3L3L4H4DampingSelectionPanelIntegrityError(
    Gemma3L3L4H4DampingSelectionPanelError
):
    """A private role input or prompt-free artifact failed authentication."""


class Gemma3L3L4H4DampingSelectionPanelClosedError(
    Gemma3L3L4H4DampingSelectionPanelError
):
    """The local one-shot selection source has already been consumed."""


def _selection_policy() -> dict[str, object]:
    """Return a fresh copy so callers cannot mutate the frozen protocol."""

    return {
        "opening": "one_shot_selection",
        "maximum_panel_open_count": 1,
        "authorized_alpha_pair": (0.0, 0.5),
        "authorized_candidate_count": 2,
        "adaptive_candidate_changes_authorized": False,
        "guard_authorized": False,
        "assessment_authorized": False,
    }


def _artifact_safety() -> dict[str, bool]:
    """Return prompt-free capability metadata without shared mutable state."""

    return {
        "prompt_text_in_artifact": False,
        "token_ids_in_artifact": False,
        "activation_rows_in_artifact": False,
        "gradient_rows_in_artifact": False,
        "old_selection_input_capability_present": False,
        "guard_input_capability_present": False,
        "calibration_b_input_capability_present": False,
        "forbidden_assessment_payload_opened": False,
        "forbidden_prompt_blind_manifest_consulted": True,
    }


def _canonical_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {
            key: _canonical_json_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError("panel commitments must contain only JSON values")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable nonempty identifier")
    return value


def _canonical_sha256s(
    values: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(values) is not tuple or (not values and not allow_empty):
        qualifier = "possibly empty " if allow_empty else "nonempty "
        raise ValueError(f"{label} must be a {qualifier}tuple")
    parsed = tuple(
        _require_sha256(value, label=f"{label}[]") for value in values
    )
    if parsed != tuple(sorted(set(parsed))):
        raise ValueError(f"{label} must be sorted and unique")
    return parsed


def _canonical_identifiers(
    values: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise ValueError(f"{label} must be a nonempty tuple")
    parsed = tuple(
        _require_identifier(value, label=f"{label}[]") for value in values
    )
    if parsed != tuple(sorted(set(parsed))):
        raise ValueError(f"{label} must be sorted and unique")
    return parsed


def _read_json_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file():
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            f"{label} must be a regular file"
        )
    encoded = path.read_bytes()
    if not encoded or len(encoded) > _MAX_JSON_BYTES:
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            f"{label} must be nonempty and no larger than "
            f"{_MAX_JSON_BYTES} bytes"
        )
    return encoded


def _decode_canonical_mapping(
    encoded: bytes,
    *,
    label: str,
) -> dict[str, object]:
    try:
        raw = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            f"{label} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(raw, dict):
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            f"{label} must contain one JSON object"
        )
    if _canonical_json_bytes(raw) != encoded:
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            f"{label} must use deterministic canonical JSON encoding"
        )
    return raw


def _prompt_blind_forbidden_binding(
) -> tuple[str, frozenset[str], frozenset[str]]:
    """Return only frozen assessment hashes; no prompt loader is reachable."""

    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    protocol = (
        default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    )
    protocol.validate_integrity()
    metadata = protocol.metadata()
    try:
        manifest_sha256 = metadata["corpus"]["calibration_b_manifest"][
            "artifact_sha256"
        ]
    except (KeyError, TypeError) as error:
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            "frozen protocol lacks its prompt-blind forbidden manifest"
        ) from error
    return (
        _require_sha256(
            manifest_sha256,
            label="forbidden assessment manifest",
        ),
        frozenset(manifest),
        frozenset(manifest.values()),
    )


@dataclass(frozen=True, slots=True)
class Gemma3L3L4H4DampingSelectionRoleInput:
    """One caller-supplied fresh selection input before prompt removal."""

    prompts: tuple[str, ...]
    family_ids: tuple[str, ...]
    source_file_sha256: str
    ordered_prompt_sha256s: tuple[str, ...] = field(init=False)
    panel_id: str = field(
        default=GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID,
        init=False,
    )
    role: str = field(
        default=GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE,
        init=False,
    )

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_file_sha256,
            label="fresh selection input file",
        )
        if (
            type(self.prompts) is not tuple
            or len(self.prompts)
            != len(FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE)
            or any(
                not isinstance(prompt, str)
                or not prompt
                or prompt != prompt.strip()
                for prompt in self.prompts
            )
        ):
            raise ValueError(
                "fresh selection prompts must contain exactly 16 canonical "
                "nonempty strings"
            )
        if self.family_ids != FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE:
            raise ValueError(
                "fresh selection family schedule must equal the frozen "
                "eight-family, two-round schedule"
            )
        prompt_sha256s = tuple(
            gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
            for prompt in self.prompts
        )
        if len(set(prompt_sha256s)) != len(prompt_sha256s):
            raise ValueError(
                "fresh selection prompts must have unique exact identities"
            )
        object.__setattr__(
            self,
            "ordered_prompt_sha256s",
            prompt_sha256s,
        )

    def private_payload(self) -> dict[str, object]:
        """Return the canonical private source payload, including prompt text."""

        return {
            "schema": GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE_SCHEMA,
            "format_version": (
                GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE_FORMAT_VERSION
            ),
            "panel_id": self.panel_id,
            "role": self.role,
            "prompts": self.prompts,
            "family_ids": self.family_ids,
        }


def _parse_selection_role_input(
    path: Path,
) -> Gemma3L3L4H4DampingSelectionRoleInput:
    encoded = _read_json_bytes(path, label="fresh selection input")
    raw = _decode_canonical_mapping(
        encoded,
        label="fresh selection input",
    )
    if set(raw) != _ROLE_INPUT_FIELDS:
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            "fresh selection input fields differ from the private schema"
        )
    if (
        raw["schema"]
        != GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE_SCHEMA
        or raw["format_version"]
        != GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE_FORMAT_VERSION
        or raw["panel_id"]
        != GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID
        or raw["role"] != GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE
    ):
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            "fresh selection input metadata differs from the frozen protocol"
        )
    prompts = raw["prompts"]
    family_ids = raw["family_ids"]
    if not isinstance(prompts, list) or not isinstance(family_ids, list):
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            "fresh selection prompts and family IDs must be JSON lists"
        )
    try:
        return Gemma3L3L4H4DampingSelectionRoleInput(
            prompts=tuple(prompts),
            family_ids=tuple(family_ids),
            source_file_sha256=_file_sha256(encoded),
        )
    except (TypeError, ValueError) as error:
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            "fresh selection input is invalid"
        ) from error


def write_gemma3_l3_l4_h4_damping_selection_role_input(
    path: Path | str,
    *,
    prompts: Sequence[str],
) -> str:
    """Write a private fresh selection input with the frozen family schedule."""

    if isinstance(prompts, (str, bytes)) or not isinstance(
        prompts,
        Sequence,
    ):
        raise TypeError("prompts must be a sequence")
    provisional = {
        "schema": GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE_SCHEMA,
        "format_version": (
            GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE_FORMAT_VERSION
        ),
        "panel_id": GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID,
        "role": GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE,
        "prompts": tuple(prompts),
        "family_ids": FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE,
    }
    encoded = _canonical_json_bytes(provisional)
    # Validate all fixed-count, schedule, and uniqueness invariants before
    # publishing a private input file.
    role_input = Gemma3L3L4H4DampingSelectionRoleInput(
        prompts=tuple(prompts),
        family_ids=FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE,
        source_file_sha256=_file_sha256(encoded),
    )
    if _canonical_json_bytes(role_input.private_payload()) != encoded:
        raise RuntimeError("fresh selection input canonicalization drifted")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(encoded)
    return _file_sha256(encoded)


def load_gemma3_l3_l4_h4_damping_selection_role_input(
    path: Path | str,
) -> Gemma3L3L4H4DampingSelectionRoleInput:
    """Open only one caller-supplied fresh selection input."""

    return _parse_selection_role_input(Path(path))


@dataclass(frozen=True, slots=True)
class Gemma3L3L4H4DampingExpandedFitLineage:
    """Prompt-free expanded-fit identity supplied to panel commitment."""

    expanded_corpus_artifact_sha256: str
    tokenizer_contract_sha256: str
    fit_manifest_sha256: str
    fit_role_input_file_sha256: str
    fit_binding_sha256: str
    fit_example_count: int
    fit_family_ids: tuple[str, ...]
    ordered_fit_prompt_sha256s: tuple[str, ...]
    ordered_fit_family_ids: tuple[str, ...]
    occupied_development_manifest_sha256s: tuple[str, ...]
    occupied_development_prompt_sha256s: tuple[str, ...]
    occupied_development_family_ids: tuple[str, ...]
    forbidden_assessment_manifest_sha256s: tuple[str, ...]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "expanded_corpus_artifact_sha256",
            "tokenizer_contract_sha256",
            "fit_manifest_sha256",
            "fit_role_input_file_sha256",
            "fit_binding_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.fit_example_count != 16:
            raise ValueError(
                "expanded-fit lineage must bind exactly 16 fit examples"
            )
        _canonical_identifiers(
            self.fit_family_ids,
            label="fit_family_ids",
        )
        if len(self.fit_family_ids) != 8:
            raise ValueError(
                "expanded-fit lineage must bind exactly eight fit families"
            )
        if (
            type(self.ordered_fit_prompt_sha256s) is not tuple
            or len(self.ordered_fit_prompt_sha256s)
            != self.fit_example_count
            or len(set(self.ordered_fit_prompt_sha256s))
            != self.fit_example_count
        ):
            raise ValueError(
                "expanded-fit prompt hashes must contain 16 unique members"
            )
        for prompt_sha256 in self.ordered_fit_prompt_sha256s:
            _require_sha256(
                prompt_sha256,
                label="ordered_fit_prompt_sha256s[]",
            )
        if (
            type(self.ordered_fit_family_ids) is not tuple
            or len(self.ordered_fit_family_ids) != self.fit_example_count
            or tuple(sorted(set(self.ordered_fit_family_ids)))
            != self.fit_family_ids
            or Counter(self.ordered_fit_family_ids)
            != Counter({family: 2 for family in self.fit_family_ids})
        ):
            raise ValueError(
                "expanded-fit lineage must bind two examples per fit family"
            )
        occupied_manifests = _canonical_sha256s(
            self.occupied_development_manifest_sha256s,
            label="occupied_development_manifest_sha256s",
        )
        occupied_prompts = _canonical_sha256s(
            self.occupied_development_prompt_sha256s,
            label="occupied_development_prompt_sha256s",
        )
        occupied_families = _canonical_identifiers(
            self.occupied_development_family_ids,
            label="occupied_development_family_ids",
        )
        forbidden = _canonical_sha256s(
            self.forbidden_assessment_manifest_sha256s,
            label="forbidden_assessment_manifest_sha256s",
        )
        if (
            self.fit_manifest_sha256 not in occupied_manifests
            or not set(self.ordered_fit_prompt_sha256s).issubset(
                occupied_prompts
            )
            or not set(self.fit_family_ids).issubset(occupied_families)
        ):
            raise ValueError(
                "expanded-fit members must be contained in occupied "
                "prompt-free corpus metadata"
            )
        if set(occupied_manifests) & set(forbidden):
            raise ValueError(
                "development and forbidden assessment manifests must differ"
            )
        object.__setattr__(
            self,
            "receipt_sha256",
            _domain_sha256(_LINEAGE_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _LINEAGE_SCHEMA,
            "format_version": _LINEAGE_FORMAT_VERSION,
            "expanded_corpus_artifact_sha256": (
                self.expanded_corpus_artifact_sha256
            ),
            "tokenizer_contract_sha256": (
                self.tokenizer_contract_sha256
            ),
            "fit_manifest_sha256": self.fit_manifest_sha256,
            "fit_role_input_file_sha256": (
                self.fit_role_input_file_sha256
            ),
            "fit_binding_sha256": self.fit_binding_sha256,
            "fit_example_count": self.fit_example_count,
            "fit_family_ids": self.fit_family_ids,
            "ordered_fit_prompt_sha256s": (
                self.ordered_fit_prompt_sha256s
            ),
            "ordered_fit_family_ids": self.ordered_fit_family_ids,
            "occupied_development_manifest_sha256s": (
                self.occupied_development_manifest_sha256s
            ),
            "occupied_development_prompt_sha256s": (
                self.occupied_development_prompt_sha256s
            ),
            "occupied_development_family_ids": (
                self.occupied_development_family_ids
            ),
            "forbidden_assessment_manifest_sha256s": (
                self.forbidden_assessment_manifest_sha256s
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "receipt_sha256": self.receipt_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> Gemma3L3L4H4DampingExpandedFitLineage:
        if not isinstance(raw, Mapping) or set(raw) != _LINEAGE_FIELDS:
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "expanded-fit lineage fields differ"
            )
        if (
            raw["schema"] != _LINEAGE_SCHEMA
            or raw["format_version"] != _LINEAGE_FORMAT_VERSION
        ):
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "unsupported expanded-fit lineage"
            )
        try:
            lineage = cls(
                expanded_corpus_artifact_sha256=str(
                    raw["expanded_corpus_artifact_sha256"]
                ),
                tokenizer_contract_sha256=str(
                    raw["tokenizer_contract_sha256"]
                ),
                fit_manifest_sha256=str(raw["fit_manifest_sha256"]),
                fit_role_input_file_sha256=str(
                    raw["fit_role_input_file_sha256"]
                ),
                fit_binding_sha256=str(raw["fit_binding_sha256"]),
                fit_example_count=raw["fit_example_count"],  # type: ignore[arg-type]
                fit_family_ids=tuple(raw["fit_family_ids"]),  # type: ignore[arg-type]
                ordered_fit_prompt_sha256s=tuple(
                    raw["ordered_fit_prompt_sha256s"]  # type: ignore[arg-type]
                ),
                ordered_fit_family_ids=tuple(
                    raw["ordered_fit_family_ids"]  # type: ignore[arg-type]
                ),
                occupied_development_manifest_sha256s=tuple(
                    raw[  # type: ignore[arg-type]
                        "occupied_development_manifest_sha256s"
                    ]
                ),
                occupied_development_prompt_sha256s=tuple(
                    raw[  # type: ignore[arg-type]
                        "occupied_development_prompt_sha256s"
                    ]
                ),
                occupied_development_family_ids=tuple(
                    raw["occupied_development_family_ids"]  # type: ignore[arg-type]
                ),
                forbidden_assessment_manifest_sha256s=tuple(
                    raw[  # type: ignore[arg-type]
                        "forbidden_assessment_manifest_sha256s"
                    ]
                ),
            )
        except (TypeError, ValueError) as error:
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "expanded-fit lineage is invalid"
            ) from error
        if (
            _require_sha256(
                raw["receipt_sha256"],
                label="expanded-fit lineage receipt",
            )
            != lineage.receipt_sha256
        ):
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "expanded-fit lineage receipt differs"
            )
        return lineage


def expanded_fit_lineage_from_corpus_artifact(
    artifact: Gemma3L3L4ProgressiveACorpusArtifact,
    *,
    fit_binding_sha256: str,
) -> Gemma3L3L4H4DampingExpandedFitLineage:
    """Reduce a prompt-free expanded corpus to the only accepted lineage."""

    if not isinstance(artifact, Gemma3L3L4ProgressiveACorpusArtifact):
        raise TypeError(
            "artifact must be a prompt-free progressive A corpus artifact"
        )
    fit = artifact.role_view("calibration_a_fit")
    views = artifact.role_views
    return Gemma3L3L4H4DampingExpandedFitLineage(
        expanded_corpus_artifact_sha256=artifact.artifact_sha256,
        tokenizer_contract_sha256=artifact.tokenizer_contract_sha256,
        fit_manifest_sha256=fit.manifest_sha256,
        fit_role_input_file_sha256=fit.role_input_file_sha256,
        fit_binding_sha256=_require_sha256(
            fit_binding_sha256,
            label="expanded fit binding",
        ),
        fit_example_count=fit.example_count,
        fit_family_ids=fit.family_ids,
        ordered_fit_prompt_sha256s=fit.ordered_prompt_sha256s,
        ordered_fit_family_ids=fit.ordered_family_ids,
        occupied_development_manifest_sha256s=tuple(
            sorted(view.manifest_sha256 for view in views)
        ),
        occupied_development_prompt_sha256s=tuple(
            sorted(
                {
                    prompt_sha256
                    for view in views
                    for prompt_sha256 in view.ordered_prompt_sha256s
                }
            )
        ),
        occupied_development_family_ids=tuple(
            sorted(
                {
                    family_id
                    for view in views
                    for family_id in view.family_ids
                }
            )
        ),
        forbidden_assessment_manifest_sha256s=(
            artifact.forbidden_assessment_manifest_sha256s
        ),
    )


def load_gemma3_l3_l4_h4_damping_expanded_fit_lineage(
    expanded_corpus_artifact_path: Path | str,
    *,
    expected_expanded_corpus_artifact_sha256: str,
    fit_binding_sha256: str,
) -> Gemma3L3L4H4DampingExpandedFitLineage:
    """Load one prompt-free corpus artifact; no role input path is accepted."""

    artifact = _load_progressive_a_artifact(
        expanded_corpus_artifact_path,
        expected_artifact_sha256=(
            expected_expanded_corpus_artifact_sha256
        ),
    )
    return expanded_fit_lineage_from_corpus_artifact(
        artifact,
        fit_binding_sha256=fit_binding_sha256,
    )


def _manifest_payload(
    *,
    lineage: Gemma3L3L4H4DampingExpandedFitLineage,
    role_input_file_sha256: str,
    ordered_prompt_sha256s: tuple[str, ...],
    ordered_family_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema": _MANIFEST_SCHEMA,
        "format_version": 1,
        "panel_id": GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID,
        "role": GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE,
        "expanded_fit_lineage_receipt_sha256": lineage.receipt_sha256,
        "tokenizer_contract_sha256": lineage.tokenizer_contract_sha256,
        "role_input_file_sha256": role_input_file_sha256,
        "ordered_members": tuple(
            {
                "prompt_sha256": prompt_sha256,
                "family_id": family_id,
            }
            for prompt_sha256, family_id in zip(
                ordered_prompt_sha256s,
                ordered_family_ids,
                strict=True,
            )
        ),
        "policy": _selection_policy(),
    }


@dataclass(frozen=True, slots=True)
class Gemma3L3L4H4DampingSelectionPanelArtifact:
    """Immutable prompt-free one-shot selection membership commitment."""

    expanded_fit_lineage: Gemma3L3L4H4DampingExpandedFitLineage
    selection_role_input_file_sha256: str
    ordered_prompt_sha256s: tuple[str, ...]
    ordered_family_ids: tuple[str, ...]
    manifest_sha256: str = field(init=False)
    membership_receipt_sha256: str = field(init=False)
    artifact_sha256: str = field(init=False)
    panel_id: str = field(
        default=GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID,
        init=False,
    )
    role: str = field(
        default=GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.expanded_fit_lineage,
            Gemma3L3L4H4DampingExpandedFitLineage,
        ):
            raise TypeError(
                "expanded_fit_lineage must be an authenticated lineage"
            )
        role_input_sha256 = _require_sha256(
            self.selection_role_input_file_sha256,
            label="fresh selection role input file",
        )
        if (
            type(self.ordered_prompt_sha256s) is not tuple
            or len(self.ordered_prompt_sha256s)
            != len(FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE)
            or len(set(self.ordered_prompt_sha256s))
            != len(self.ordered_prompt_sha256s)
        ):
            raise ValueError(
                "selection commitment must contain 16 unique prompt hashes"
            )
        for prompt_sha256 in self.ordered_prompt_sha256s:
            _require_sha256(
                prompt_sha256,
                label="ordered_prompt_sha256s[]",
            )
        if self.ordered_family_ids != (
            FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE
        ):
            raise ValueError(
                "selection commitment family schedule differs from the "
                "frozen eight-family, two-round schedule"
            )
        lineage = self.expanded_fit_lineage
        prompt_overlap = set(self.ordered_prompt_sha256s) & set(
            lineage.occupied_development_prompt_sha256s
        )
        family_overlap = set(self.ordered_family_ids) & set(
            lineage.occupied_development_family_ids
        )
        if prompt_overlap:
            raise ValueError(
                "fresh selection prompts overlap prompt-free expanded "
                "development metadata"
            )
        if family_overlap:
            raise ValueError(
                "fresh selection families overlap prompt-free expanded "
                "development metadata"
            )
        (
            forbidden_manifest,
            forbidden_prompts,
            forbidden_families,
        ) = _prompt_blind_forbidden_binding()
        if lineage.forbidden_assessment_manifest_sha256s != (
            forbidden_manifest,
        ):
            raise ValueError(
                "expanded-fit lineage forbidden manifest identity differs "
                "from the frozen prompt-blind assessment identity"
            )
        if set(self.ordered_prompt_sha256s) & set(forbidden_prompts):
            raise ValueError(
                "fresh selection prompts overlap the prompt-blind forbidden "
                "assessment manifest"
            )
        if set(self.ordered_family_ids) & set(forbidden_families):
            raise ValueError(
                "fresh selection families overlap the prompt-blind forbidden "
                "assessment manifest"
            )
        manifest = _domain_sha256(
            _MANIFEST_DOMAIN,
            _manifest_payload(
                lineage=lineage,
                role_input_file_sha256=role_input_sha256,
                ordered_prompt_sha256s=self.ordered_prompt_sha256s,
                ordered_family_ids=self.ordered_family_ids,
            ),
        )
        if (
            manifest
            in lineage.occupied_development_manifest_sha256s
            or manifest in lineage.forbidden_assessment_manifest_sha256s
        ):
            raise ValueError(
                "fresh selection manifest reuses an occupied or forbidden "
                "manifest identity"
            )
        membership = _domain_sha256(
            _MEMBERSHIP_DOMAIN,
            {
                "role": self.role,
                "manifest_sha256": manifest,
                "members": tuple(
                    sorted(
                        zip(
                            self.ordered_prompt_sha256s,
                            self.ordered_family_ids,
                            strict=True,
                        )
                    )
                ),
            },
        )
        object.__setattr__(self, "manifest_sha256", manifest)
        object.__setattr__(
            self,
            "membership_receipt_sha256",
            membership,
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _domain_sha256(_ARTIFACT_DOMAIN, self._payload()),
        )

    @property
    def family_ids(self) -> tuple[str, ...]:
        return FRESH_DAMPING_SELECTION_FAMILIES

    @property
    def family_by_example(self) -> dict[str, str]:
        return dict(
            zip(
                self.ordered_prompt_sha256s,
                self.ordered_family_ids,
                strict=True,
            )
        )

    def _selection_payload(self) -> dict[str, object]:
        return {
            "role_input_file_sha256": (
                self.selection_role_input_file_sha256
            ),
            "example_count": len(self.ordered_prompt_sha256s),
            "family_count": len(self.family_ids),
            "examples_per_family": 2,
            "family_ids": self.family_ids,
            "ordered_members": tuple(
                {
                    "prompt_sha256": prompt_sha256,
                    "family_id": family_id,
                }
                for prompt_sha256, family_id in zip(
                    self.ordered_prompt_sha256s,
                    self.ordered_family_ids,
                    strict=True,
                )
            ),
        }

    def _payload(self) -> dict[str, object]:
        return {
            "schema": GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_SCHEMA,
            "format_version": (
                GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_FORMAT_VERSION
            ),
            "panel_id": self.panel_id,
            "role": self.role,
            "expanded_fit_lineage": self.expanded_fit_lineage.to_dict(),
            "selection": self._selection_payload(),
            "policy": _selection_policy(),
            "safety": _artifact_safety(),
            "manifest_sha256": self.manifest_sha256,
            "membership_receipt_sha256": (
                self.membership_receipt_sha256
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> Gemma3L3L4H4DampingSelectionPanelArtifact:
        if not isinstance(raw, Mapping) or set(raw) != _ARTIFACT_FIELDS:
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "fresh selection artifact fields differ"
            )
        if (
            raw["schema"]
            != GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_SCHEMA
            or raw["format_version"]
            != GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_FORMAT_VERSION
            or raw["panel_id"]
            != GEMMA3_L3_L4_H4_DAMPING_SELECTION_PANEL_ID
            or raw["role"] != GEMMA3_L3_L4_H4_DAMPING_SELECTION_ROLE
            or raw["policy"]
            != _canonical_json_value(_selection_policy())
            or raw["safety"] != _canonical_json_value(_artifact_safety())
        ):
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "fresh selection artifact protocol metadata differs"
            )
        lineage_raw = raw["expanded_fit_lineage"]
        selection = raw["selection"]
        if not isinstance(lineage_raw, Mapping) or not isinstance(
            selection,
            Mapping,
        ):
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "fresh selection lineage or selection payload is malformed"
            )
        if set(selection) != _SELECTION_FIELDS:
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "fresh selection membership fields differ"
            )
        members = selection["ordered_members"]
        if not isinstance(members, list):
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "fresh selection ordered members must be a JSON list"
            )
        prompt_sha256s: list[str] = []
        family_ids: list[str] = []
        for member in members:
            if not isinstance(member, Mapping) or set(member) != _MEMBER_FIELDS:
                raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                    "fresh selection member fields differ"
                )
            prompt_sha256s.append(str(member["prompt_sha256"]))
            family_ids.append(str(member["family_id"]))
        try:
            lineage = Gemma3L3L4H4DampingExpandedFitLineage.from_dict(
                lineage_raw
            )
            artifact = cls(
                expanded_fit_lineage=lineage,
                selection_role_input_file_sha256=str(
                    selection["role_input_file_sha256"]
                ),
                ordered_prompt_sha256s=tuple(prompt_sha256s),
                ordered_family_ids=tuple(family_ids),
            )
        except (
            Gemma3L3L4H4DampingSelectionPanelIntegrityError,
            TypeError,
            ValueError,
        ) as error:
            if isinstance(
                error,
                Gemma3L3L4H4DampingSelectionPanelIntegrityError,
            ):
                raise
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "fresh selection artifact is invalid"
            ) from error
        expected_scalar_metadata = {
            "example_count": 16,
            "family_count": 8,
            "examples_per_family": 2,
            "family_ids": list(FRESH_DAMPING_SELECTION_FAMILIES),
        }
        if any(
            selection[name] != expected
            for name, expected in expected_scalar_metadata.items()
        ):
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "fresh selection fixed geometry differs"
            )
        for name, observed in (
            ("manifest_sha256", artifact.manifest_sha256),
            (
                "membership_receipt_sha256",
                artifact.membership_receipt_sha256,
            ),
            ("artifact_sha256", artifact.artifact_sha256),
        ):
            if _require_sha256(raw[name], label=name) != observed:
                raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                    f"fresh selection {name} differs"
                )
        return artifact


def freeze_gemma3_l3_l4_h4_damping_selection_panel(
    *,
    expanded_fit_lineage: Gemma3L3L4H4DampingExpandedFitLineage,
    selection_input_path: Path | str,
) -> Gemma3L3L4H4DampingSelectionPanelArtifact:
    """Freeze exactly one new role input against expanded-fit metadata."""

    if not isinstance(
        expanded_fit_lineage,
        Gemma3L3L4H4DampingExpandedFitLineage,
    ):
        raise TypeError(
            "expanded_fit_lineage must be a prompt-free expanded-fit lineage"
        )
    role_input = _parse_selection_role_input(Path(selection_input_path))
    return Gemma3L3L4H4DampingSelectionPanelArtifact(
        expanded_fit_lineage=expanded_fit_lineage,
        selection_role_input_file_sha256=role_input.source_file_sha256,
        ordered_prompt_sha256s=role_input.ordered_prompt_sha256s,
        ordered_family_ids=role_input.family_ids,
    )


def write_gemma3_l3_l4_h4_damping_selection_panel_artifact(
    path: Path | str,
    artifact: Gemma3L3L4H4DampingSelectionPanelArtifact,
) -> str:
    """Publish one canonical prompt-free artifact without overwriting."""

    if not isinstance(
        artifact,
        Gemma3L3L4H4DampingSelectionPanelArtifact,
    ):
        raise TypeError(
            "artifact must be a damping selection panel artifact"
        )
    destination = Path(path)
    encoded = _canonical_json_bytes(artifact.to_dict())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(encoded)
    return _file_sha256(encoded)


def load_gemma3_l3_l4_h4_damping_selection_panel_artifact(
    path: Path | str,
    *,
    expected_artifact_sha256: str | None = None,
) -> Gemma3L3L4H4DampingSelectionPanelArtifact:
    """Authenticate one prompt-free fresh selection panel artifact."""

    encoded = _read_json_bytes(
        Path(path),
        label="fresh selection panel artifact",
    )
    raw = _decode_canonical_mapping(
        encoded,
        label="fresh selection panel artifact",
    )
    artifact = Gemma3L3L4H4DampingSelectionPanelArtifact.from_dict(raw)
    if (
        expected_artifact_sha256 is not None
        and artifact.artifact_sha256
        != _require_sha256(
            expected_artifact_sha256,
            label="expected fresh selection artifact",
        )
    ):
        raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
            "fresh selection panel differs from expected identity"
        )
    return artifact


class Gemma3L3L4H4DampingSelectionPanelSource:
    """One-use opener holding only the new selection input capability."""

    __slots__ = (
        "_artifact",
        "_selection_input_path",
        "_consumed",
        "_opened",
    )

    def __init__(
        self,
        *,
        artifact: Gemma3L3L4H4DampingSelectionPanelArtifact,
        selection_input_path: Path | str,
    ) -> None:
        if not isinstance(
            artifact,
            Gemma3L3L4H4DampingSelectionPanelArtifact,
        ):
            raise TypeError(
                "artifact must be a damping selection panel artifact"
            )
        self._artifact = artifact
        self._selection_input_path = Path(selection_input_path)
        self._consumed = False
        self._opened = False

    @property
    def artifact(self) -> Gemma3L3L4H4DampingSelectionPanelArtifact:
        return self._artifact

    @property
    def consumed(self) -> bool:
        return self._consumed

    @property
    def opened(self) -> bool:
        return self._opened

    def open_once(self) -> Gemma3L3L4H4DampingSelectionRoleInput:
        """Consume before reading, then authenticate exact private membership."""

        if self._consumed:
            raise Gemma3L3L4H4DampingSelectionPanelClosedError(
                "fresh selection input has already been consumed"
            )
        # Fail closed: a malformed or changed source cannot create a retry
        # opportunity through this opener.
        self._consumed = True
        role_input = _parse_selection_role_input(
            self._selection_input_path
        )
        if (
            role_input.source_file_sha256
            != self._artifact.selection_role_input_file_sha256
            or role_input.ordered_prompt_sha256s
            != self._artifact.ordered_prompt_sha256s
            or role_input.family_ids
            != self._artifact.ordered_family_ids
        ):
            raise Gemma3L3L4H4DampingSelectionPanelIntegrityError(
                "fresh selection input differs from its frozen artifact"
            )
        self._opened = True
        return role_input
