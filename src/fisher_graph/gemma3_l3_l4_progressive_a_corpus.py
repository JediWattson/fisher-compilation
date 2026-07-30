"""Strict three-role Calibration-A corpus artifacts for progressive Gemma.

This module owns only development data.  It never accepts Calibration-B,
validation, or test payloads.  The frozen Calibration-B manifest is imported
only as a hash-to-family map so new A examples and families can be rejected on
identity overlap without opening any held-out prompt.

Corpus freezing and corpus use are deliberately separate:

* ``build_gemma3_l3_l4_progressive_a_corpus_artifact`` reads three deterministic
  role inputs before model or tokenizer access and emits a prompt-free artifact;
* ``load_gemma3_l3_l4_progressive_a_corpus`` authenticates that artifact and
  exposes hash-only preclaim views without reading any role input; and
* fit/selection text is opened explicitly, while guard text requires a durable
  manifest-bound claim receipt and can be opened only once.

The artifact binds the exact ordered prompt identities, their ordered family
assignments, the source JSON bytes, and the tokenizer-contract identity.  It is
an integrity/audit object, not hostile-process cryptographic attestation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
    frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest,
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from .gemma3_l3_l4_progressive_guard_ledger import (
    Gemma3L3L4ProgressiveGuardClaimReceipt,
)


DevelopmentRole = Literal[
    "calibration_a_fit",
    "calibration_a_selection",
    "calibration_a_guard",
]
CorpusProfile = Literal["pilot", "full"]

__all__ = [
    "GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_FORMAT_VERSION",
    "GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_SCHEMA",
    "GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_FORMAT_VERSION",
    "GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_SCHEMA",
    "Gemma3L3L4ProgressiveACorpus",
    "Gemma3L3L4ProgressiveACorpusArtifact",
    "Gemma3L3L4ProgressiveACorpusError",
    "Gemma3L3L4ProgressiveACorpusIntegrityError",
    "Gemma3L3L4ProgressiveAGuardClosedError",
    "Gemma3L3L4ProgressiveARolePreclaimView",
    "Gemma3L3L4ProgressiveARolePrompts",
    "build_gemma3_l3_l4_progressive_a_corpus_artifact",
    "gemma3_l3_l4_progressive_a_tokenizer_contract_sha256",
    "gemma3_l3_l4_progressive_a_fit_replacement_lineage",
    "load_gemma3_l3_l4_progressive_a_corpus",
    "load_gemma3_l3_l4_progressive_a_fit_role",
    "replace_gemma3_l3_l4_progressive_a_fit_role",
    "write_gemma3_l3_l4_progressive_a_corpus_artifact",
    "write_gemma3_l3_l4_progressive_a_role_input",
]


GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_progressive_a_corpus"
)
GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_FORMAT_VERSION = 1
GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_progressive_a_role_prompts"
)
GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_FORMAT_VERSION = 1

_ROLE_MANIFEST_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_progressive_a_role_manifest"
)
_ROLES: tuple[DevelopmentRole, ...] = (
    "calibration_a_fit",
    "calibration_a_selection",
    "calibration_a_guard",
)
_PROFILES = frozenset({"pilot", "full"})
_ROLE_INPUT_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "corpus_id",
        "profile",
        "role",
        "prompts",
        "family_ids",
    }
)
_ROLE_VIEW_FIELDS = frozenset(
    {
        "role",
        "manifest_sha256",
        "role_input_file_sha256",
        "example_count",
        "family_ids",
        "ordered_prompt_sha256s",
        "ordered_family_ids",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "corpus_id",
        "profile",
        "tokenizer_contract_sha256",
        "forbidden_assessment_manifest_sha256s",
        "roles",
        "artifact_sha256",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_TOKENIZER_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-progressive-a-tokenizer:v1\0"
)
_ROLE_MANIFEST_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-progressive-a-role-manifest:v1\0"
)
_CORPUS_ARTIFACT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-progressive-a-corpus:v1\0"
)
_MAX_JSON_BYTES = 64 * 1024 * 1024


class Gemma3L3L4ProgressiveACorpusError(RuntimeError):
    """Base class for fail-closed three-role A corpus errors."""


class Gemma3L3L4ProgressiveACorpusIntegrityError(
    Gemma3L3L4ProgressiveACorpusError
):
    """A corpus artifact or role input differs from its frozen identity."""


class Gemma3L3L4ProgressiveAGuardClosedError(
    Gemma3L3L4ProgressiveACorpusError
):
    """Guard prompt text was requested without a valid prior claim."""


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
    raise TypeError(
        "tokenizer contracts and corpus artifacts must contain only JSON "
        "values"
    )


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


def _require_role(value: object) -> DevelopmentRole:
    if value not in _ROLES:
        raise ValueError(
            "role must be calibration_a_fit, calibration_a_selection, or "
            "calibration_a_guard"
        )
    return value  # type: ignore[return-value]


def _require_profile(value: object) -> CorpusProfile:
    if value not in _PROFILES:
        raise ValueError("profile must be 'pilot' or 'full'")
    return value  # type: ignore[return-value]


def _read_json_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file():
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"{label} must be a regular file"
        )
    encoded = path.read_bytes()
    if not encoded or len(encoded) > _MAX_JSON_BYTES:
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"{label} must be nonempty and no larger than {_MAX_JSON_BYTES} "
            "bytes"
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
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"{label} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(raw, dict):
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"{label} must contain one JSON object"
        )
    if _canonical_json_bytes(raw) != encoded:
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"{label} must use the deterministic canonical JSON encoding"
        )
    return raw


def _frozen_b_binding() -> tuple[str, frozenset[str], frozenset[str]]:
    """Return only frozen B identities; no prompt loader is reachable here."""

    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    protocol = (
        default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    )
    protocol.validate_integrity()
    metadata = protocol.metadata()
    try:
        manifest_metadata = metadata["corpus"]["calibration_b_manifest"]
        manifest_sha256 = manifest_metadata["artifact_sha256"]
    except (KeyError, TypeError) as error:
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            "frozen protocol lacks its Calibration-B manifest identity"
        ) from error
    return (
        _require_sha256(
            manifest_sha256,
            label="frozen Calibration-B manifest",
        ),
        frozenset(manifest),
        frozenset(manifest.values()),
    )


def gemma3_l3_l4_progressive_a_tokenizer_contract_sha256(
    tokenizer_contract: Mapping[str, object],
) -> str:
    """Derive the identity committed by every role manifest."""

    if not isinstance(tokenizer_contract, Mapping) or not tokenizer_contract:
        raise ValueError("tokenizer_contract must be a nonempty mapping")
    return _domain_sha256(_TOKENIZER_DOMAIN, tokenizer_contract)


def _role_manifest_payload(
    *,
    corpus_id: str,
    profile: CorpusProfile,
    role: DevelopmentRole,
    tokenizer_contract_sha256: str,
    ordered_prompt_sha256s: tuple[str, ...],
    ordered_family_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema": _ROLE_MANIFEST_SCHEMA,
        "format_version": 1,
        "corpus_id": corpus_id,
        "profile": profile,
        "role": role,
        "tokenizer_contract_sha256": tokenizer_contract_sha256,
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
    }


def _role_manifest_sha256(
    *,
    corpus_id: str,
    profile: CorpusProfile,
    role: DevelopmentRole,
    tokenizer_contract_sha256: str,
    ordered_prompt_sha256s: tuple[str, ...],
    ordered_family_ids: tuple[str, ...],
) -> str:
    return _domain_sha256(
        _ROLE_MANIFEST_DOMAIN,
        _role_manifest_payload(
            corpus_id=corpus_id,
            profile=profile,
            role=role,
            tokenizer_contract_sha256=tokenizer_contract_sha256,
            ordered_prompt_sha256s=ordered_prompt_sha256s,
            ordered_family_ids=ordered_family_ids,
        ),
    )


@dataclass(frozen=True, slots=True)
class Gemma3L3L4ProgressiveARolePrompts:
    """One authenticated role input after its explicit opening boundary."""

    corpus_id: str
    profile: CorpusProfile
    role: DevelopmentRole
    prompts: tuple[str, ...]
    family_ids: tuple[str, ...]
    source_file_sha256: str
    ordered_prompt_sha256s: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        _require_identifier(self.corpus_id, label="corpus_id")
        _require_profile(self.profile)
        _require_role(self.role)
        _require_sha256(
            self.source_file_sha256,
            label="role input file",
        )
        if (
            type(self.prompts) is not tuple
            or not self.prompts
            or any(
                not isinstance(prompt, str)
                or not prompt
                or prompt != prompt.strip()
                for prompt in self.prompts
            )
        ):
            raise ValueError(
                "prompts must be a nonempty tuple of canonical nonempty text"
            )
        if (
            type(self.family_ids) is not tuple
            or len(self.family_ids) != len(self.prompts)
        ):
            raise ValueError(
                "family_ids must contain one family per prompt"
            )
        for family_id in self.family_ids:
            _require_identifier(family_id, label="family_id")
        prompt_sha256s = tuple(
            gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
            for prompt in self.prompts
        )
        if len(set(prompt_sha256s)) != len(prompt_sha256s):
            raise ValueError("role prompts must have unique exact identities")
        object.__setattr__(
            self,
            "ordered_prompt_sha256s",
            prompt_sha256s,
        )

    @property
    def family_by_example(self) -> dict[str, str]:
        return dict(
            zip(
                self.ordered_prompt_sha256s,
                self.family_ids,
                strict=True,
            )
        )


def _parse_role_input(
    *,
    path: Path,
    expected_role: DevelopmentRole,
    expected_corpus_id: str,
    expected_profile: CorpusProfile,
) -> Gemma3L3L4ProgressiveARolePrompts:
    encoded = _read_json_bytes(path, label=f"{expected_role} input")
    raw = _decode_canonical_mapping(
        encoded,
        label=f"{expected_role} input",
    )
    if set(raw) != _ROLE_INPUT_FIELDS:
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"{expected_role} input fields differ from the A-only schema"
        )
    if (
        raw["schema"] != GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_SCHEMA
        or raw["format_version"]
        != GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_FORMAT_VERSION
        or raw["corpus_id"] != expected_corpus_id
        or raw["profile"] != expected_profile
        or raw["role"] != expected_role
    ):
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"{expected_role} input metadata differs from expectation"
        )
    prompts = raw["prompts"]
    families = raw["family_ids"]
    if not isinstance(prompts, list) or not isinstance(families, list):
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"{expected_role} prompts and family_ids must be JSON lists"
        )
    try:
        return Gemma3L3L4ProgressiveARolePrompts(
            corpus_id=expected_corpus_id,
            profile=expected_profile,
            role=expected_role,
            prompts=tuple(prompts),
            family_ids=tuple(families),
            source_file_sha256=_file_sha256(encoded),
        )
    except (TypeError, ValueError) as error:
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"{expected_role} input is invalid"
        ) from error


@dataclass(frozen=True, slots=True)
class Gemma3L3L4ProgressiveARolePreclaimView:
    """Prompt-free role membership and tokenizer binding."""

    role: DevelopmentRole
    manifest_sha256: str
    role_input_file_sha256: str
    example_count: int
    family_ids: tuple[str, ...]
    ordered_prompt_sha256s: tuple[str, ...]
    ordered_family_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_role(self.role)
        _require_sha256(self.manifest_sha256, label="role manifest")
        _require_sha256(
            self.role_input_file_sha256,
            label="role input file",
        )
        if type(self.example_count) is not int or self.example_count <= 0:
            raise ValueError("example_count must be positive")
        if (
            type(self.ordered_prompt_sha256s) is not tuple
            or len(self.ordered_prompt_sha256s) != self.example_count
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.ordered_prompt_sha256s
            )
            or len(set(self.ordered_prompt_sha256s))
            != self.example_count
        ):
            raise ValueError(
                "ordered_prompt_sha256s must contain unique exact identities"
            )
        if (
            type(self.ordered_family_ids) is not tuple
            or len(self.ordered_family_ids) != self.example_count
        ):
            raise ValueError(
                "ordered_family_ids must align with ordered prompt hashes"
            )
        for family_id in self.ordered_family_ids:
            _require_identifier(family_id, label="ordered family")
        canonical_families = tuple(sorted(set(self.ordered_family_ids)))
        if (
            type(self.family_ids) is not tuple
            or self.family_ids != canonical_families
        ):
            raise ValueError(
                "family_ids must be the sorted unique ordered families"
            )

    @property
    def family_by_example(self) -> dict[str, str]:
        return dict(
            zip(
                self.ordered_prompt_sha256s,
                self.ordered_family_ids,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "manifest_sha256": self.manifest_sha256,
            "role_input_file_sha256": self.role_input_file_sha256,
            "example_count": self.example_count,
            "family_ids": self.family_ids,
            "ordered_prompt_sha256s": self.ordered_prompt_sha256s,
            "ordered_family_ids": self.ordered_family_ids,
        }


def _view_from_role_input(
    role_input: Gemma3L3L4ProgressiveARolePrompts,
    *,
    tokenizer_contract_sha256: str,
) -> Gemma3L3L4ProgressiveARolePreclaimView:
    manifest_sha256 = _role_manifest_sha256(
        corpus_id=role_input.corpus_id,
        profile=role_input.profile,
        role=role_input.role,
        tokenizer_contract_sha256=tokenizer_contract_sha256,
        ordered_prompt_sha256s=role_input.ordered_prompt_sha256s,
        ordered_family_ids=role_input.family_ids,
    )
    return Gemma3L3L4ProgressiveARolePreclaimView(
        role=role_input.role,
        manifest_sha256=manifest_sha256,
        role_input_file_sha256=role_input.source_file_sha256,
        example_count=len(role_input.prompts),
        family_ids=tuple(sorted(set(role_input.family_ids))),
        ordered_prompt_sha256s=role_input.ordered_prompt_sha256s,
        ordered_family_ids=role_input.family_ids,
    )


def _validate_role_views(
    views: Sequence[Gemma3L3L4ProgressiveARolePreclaimView],
) -> None:
    if (
        len(views) != len(_ROLES)
        or tuple(view.role for view in views) != _ROLES
    ):
        raise ValueError("role views must cover exact canonical A roles")
    if len({view.manifest_sha256 for view in views}) != len(_ROLES):
        raise ValueError("A role manifests must be distinct")
    if (
        len({view.role_input_file_sha256 for view in views})
        != len(_ROLES)
    ):
        raise ValueError("A roles must use distinct source JSON files")
    prompt_sets = tuple(set(view.ordered_prompt_sha256s) for view in views)
    family_sets = tuple(set(view.family_ids) for view in views)
    if any(
        left & right
        for index, left in enumerate(prompt_sets)
        for right in prompt_sets[index + 1 :]
    ):
        raise ValueError("A roles must be prompt-identity disjoint")
    if any(
        left & right
        for index, left in enumerate(family_sets)
        for right in family_sets[index + 1 :]
    ):
        raise ValueError("A roles must be family disjoint")
    b_manifest_sha256, b_examples, b_families = _frozen_b_binding()
    if any(
        view.manifest_sha256 == b_manifest_sha256
        or bool(set(view.ordered_prompt_sha256s) & b_examples)
        or bool(set(view.family_ids) & b_families)
        for view in views
    ):
        raise ValueError(
            "Calibration-A roles overlap the frozen Calibration-B hash-only "
            "manifest"
        )


def _artifact_payload(
    *,
    corpus_id: str,
    profile: CorpusProfile,
    tokenizer_contract_sha256: str,
    forbidden_assessment_manifest_sha256s: tuple[str, ...],
    views: tuple[Gemma3L3L4ProgressiveARolePreclaimView, ...],
) -> dict[str, object]:
    return {
        "schema": GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_SCHEMA,
        "format_version": (
            GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_FORMAT_VERSION
        ),
        "corpus_id": corpus_id,
        "profile": profile,
        "tokenizer_contract_sha256": tokenizer_contract_sha256,
        "forbidden_assessment_manifest_sha256s": (
            forbidden_assessment_manifest_sha256s
        ),
        "roles": {
            view.role: view.to_dict()
            for view in views
        },
    }


@dataclass(frozen=True, slots=True)
class Gemma3L3L4ProgressiveACorpusArtifact:
    """Prompt-free, exact three-role A corpus commitment."""

    corpus_id: str
    profile: CorpusProfile
    tokenizer_contract_sha256: str
    forbidden_assessment_manifest_sha256s: tuple[str, ...]
    role_views: tuple[
        Gemma3L3L4ProgressiveARolePreclaimView,
        ...,
    ]
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_identifier(self.corpus_id, label="corpus_id")
        _require_profile(self.profile)
        _require_sha256(
            self.tokenizer_contract_sha256,
            label="tokenizer contract",
        )
        b_manifest_sha256, _, _ = _frozen_b_binding()
        if self.forbidden_assessment_manifest_sha256s != (
            b_manifest_sha256,
        ):
            raise ValueError(
                "forbidden assessment identities must contain exactly the "
                "frozen Calibration-B manifest"
            )
        if type(self.role_views) is not tuple:
            raise TypeError("role_views must be a tuple")
        _validate_role_views(self.role_views)
        for view in self.role_views:
            expected_manifest = _role_manifest_sha256(
                corpus_id=self.corpus_id,
                profile=self.profile,
                role=view.role,
                tokenizer_contract_sha256=(
                    self.tokenizer_contract_sha256
                ),
                ordered_prompt_sha256s=view.ordered_prompt_sha256s,
                ordered_family_ids=view.ordered_family_ids,
            )
            if view.manifest_sha256 != expected_manifest:
                raise ValueError(
                    f"{view.role} manifest differs from exact ordered "
                    "membership and tokenizer identity"
                )
        payload = _artifact_payload(
            corpus_id=self.corpus_id,
            profile=self.profile,
            tokenizer_contract_sha256=self.tokenizer_contract_sha256,
            forbidden_assessment_manifest_sha256s=(
                self.forbidden_assessment_manifest_sha256s
            ),
            views=self.role_views,
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _domain_sha256(_CORPUS_ARTIFACT_DOMAIN, payload),
        )

    def role_view(
        self,
        role: DevelopmentRole,
    ) -> Gemma3L3L4ProgressiveARolePreclaimView:
        selected = _require_role(role)
        return self.role_views[_ROLES.index(selected)]

    def to_dict(self) -> dict[str, object]:
        payload = _artifact_payload(
            corpus_id=self.corpus_id,
            profile=self.profile,
            tokenizer_contract_sha256=self.tokenizer_contract_sha256,
            forbidden_assessment_manifest_sha256s=(
                self.forbidden_assessment_manifest_sha256s
            ),
            views=self.role_views,
        )
        payload["artifact_sha256"] = self.artifact_sha256
        return payload

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> Gemma3L3L4ProgressiveACorpusArtifact:
        if not isinstance(raw, Mapping) or set(raw) != _ARTIFACT_FIELDS:
            raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                "corpus artifact fields differ from the A-only schema"
            )
        if (
            raw["schema"] != GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_SCHEMA
            or raw["format_version"]
            != GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_FORMAT_VERSION
        ):
            raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                "unsupported progressive A corpus artifact"
            )
        corpus_id = _require_identifier(
            raw["corpus_id"],
            label="corpus_id",
        )
        profile = _require_profile(raw["profile"])
        tokenizer_contract_sha256 = _require_sha256(
            raw["tokenizer_contract_sha256"],
            label="tokenizer contract",
        )
        forbidden = raw["forbidden_assessment_manifest_sha256s"]
        roles = raw["roles"]
        if not isinstance(forbidden, list) or not isinstance(roles, Mapping):
            raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                "corpus forbidden identities and roles are malformed"
            )
        if set(roles) != set(_ROLES):
            raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                "corpus must contain exactly the three Calibration-A roles"
            )
        views: list[Gemma3L3L4ProgressiveARolePreclaimView] = []
        for role in _ROLES:
            value = roles[role]
            if not isinstance(value, Mapping) or set(value) != _ROLE_VIEW_FIELDS:
                raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                    f"{role} preclaim view fields are malformed"
                )
            if value["role"] != role:
                raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                    f"{role} preclaim view role differs"
                )
            try:
                views.append(
                    Gemma3L3L4ProgressiveARolePreclaimView(
                        role=role,
                        manifest_sha256=str(value["manifest_sha256"]),
                        role_input_file_sha256=str(
                            value["role_input_file_sha256"]
                        ),
                        example_count=value["example_count"],  # type: ignore[arg-type]
                        family_ids=tuple(value["family_ids"]),  # type: ignore[arg-type]
                        ordered_prompt_sha256s=tuple(
                            value["ordered_prompt_sha256s"]  # type: ignore[arg-type]
                        ),
                        ordered_family_ids=tuple(
                            value["ordered_family_ids"]  # type: ignore[arg-type]
                        ),
                    )
                )
            except (TypeError, ValueError) as error:
                raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                    f"{role} preclaim view is invalid"
                ) from error
        try:
            artifact = cls(
                corpus_id=corpus_id,
                profile=profile,
                tokenizer_contract_sha256=tokenizer_contract_sha256,
                forbidden_assessment_manifest_sha256s=tuple(forbidden),
                role_views=tuple(views),
            )
        except (TypeError, ValueError) as error:
            raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                "corpus artifact integrity validation failed"
            ) from error
        supplied_sha256 = _require_sha256(
            raw["artifact_sha256"],
            label="corpus artifact",
        )
        if supplied_sha256 != artifact.artifact_sha256:
            raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                "corpus artifact hash mismatch"
            )
        return artifact


def _canonical_role_paths(
    role_input_paths: Mapping[DevelopmentRole, Path | str],
) -> dict[DevelopmentRole, Path]:
    if not isinstance(role_input_paths, Mapping) or set(
        role_input_paths
    ) != set(_ROLES):
        raise ValueError(
            "role_input_paths must contain exactly fit, selection, and guard"
        )
    result = {
        role: Path(role_input_paths[role])
        for role in _ROLES
    }
    resolved = tuple(path.resolve() for path in result.values())
    if len(set(resolved)) != len(_ROLES):
        raise ValueError("A roles must use three distinct JSON input paths")
    return result


def build_gemma3_l3_l4_progressive_a_corpus_artifact(
    *,
    corpus_id: str,
    profile: CorpusProfile,
    tokenizer_contract: Mapping[str, object],
    role_input_paths: Mapping[DevelopmentRole, Path | str],
) -> Gemma3L3L4ProgressiveACorpusArtifact:
    """Freeze three separate A-only JSON inputs into one prompt-free artifact."""

    identity = _require_identifier(corpus_id, label="corpus_id")
    selected_profile = _require_profile(profile)
    tokenizer_sha256 = (
        gemma3_l3_l4_progressive_a_tokenizer_contract_sha256(
            tokenizer_contract
        )
    )
    paths = _canonical_role_paths(role_input_paths)
    role_inputs = tuple(
        _parse_role_input(
            path=paths[role],
            expected_role=role,
            expected_corpus_id=identity,
            expected_profile=selected_profile,
        )
        for role in _ROLES
    )
    views = tuple(
        _view_from_role_input(
            role_input,
            tokenizer_contract_sha256=tokenizer_sha256,
        )
        for role_input in role_inputs
    )
    b_manifest_sha256, _, _ = _frozen_b_binding()
    try:
        return Gemma3L3L4ProgressiveACorpusArtifact(
            corpus_id=identity,
            profile=selected_profile,
            tokenizer_contract_sha256=tokenizer_sha256,
            forbidden_assessment_manifest_sha256s=(
                b_manifest_sha256,
            ),
            role_views=views,
        )
    except (TypeError, ValueError) as error:
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"three-role Calibration-A corpus is invalid: {error}"
        ) from error


def write_gemma3_l3_l4_progressive_a_role_input(
    path: Path | str,
    *,
    corpus_id: str,
    profile: CorpusProfile,
    role: DevelopmentRole,
    prompts: Sequence[str],
    family_ids: Sequence[str],
) -> str:
    """Write one deterministic role JSON and return its exact file SHA-256."""

    destination = Path(path)
    identity = _require_identifier(corpus_id, label="corpus_id")
    selected_profile = _require_profile(profile)
    selected_role = _require_role(role)
    if isinstance(prompts, (str, bytes)) or not isinstance(
        prompts,
        Sequence,
    ):
        raise TypeError("prompts must be a sequence")
    if isinstance(family_ids, (str, bytes)) or not isinstance(
        family_ids,
        Sequence,
    ):
        raise TypeError("family_ids must be a sequence")
    payload = {
        "schema": GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_SCHEMA,
        "format_version": (
            GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_FORMAT_VERSION
        ),
        "corpus_id": identity,
        "profile": selected_profile,
        "role": selected_role,
        "prompts": list(prompts),
        "family_ids": list(family_ids),
    }
    encoded = _canonical_json_bytes(payload)
    # Validate before publishing bytes so malformed text cannot become a
    # seemingly frozen role input.
    _decode_canonical_mapping(encoded, label=f"{selected_role} input")
    parsed = Gemma3L3L4ProgressiveARolePrompts(
        corpus_id=identity,
        profile=selected_profile,
        role=selected_role,
        prompts=tuple(prompts),
        family_ids=tuple(family_ids),
        source_file_sha256=_file_sha256(encoded),
    )
    del parsed
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(encoded)
    return _file_sha256(encoded)


def write_gemma3_l3_l4_progressive_a_corpus_artifact(
    path: Path | str,
    artifact: Gemma3L3L4ProgressiveACorpusArtifact,
) -> str:
    """Publish one canonical prompt-free artifact without overwriting."""

    if not isinstance(
        artifact,
        Gemma3L3L4ProgressiveACorpusArtifact,
    ):
        raise TypeError(
            "artifact must be Gemma3L3L4ProgressiveACorpusArtifact"
        )
    destination = Path(path)
    encoded = _canonical_json_bytes(artifact.to_dict())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(encoded)
    return _file_sha256(encoded)


class Gemma3L3L4ProgressiveACorpus:
    """Prompt-gated runtime view over one authenticated A corpus artifact."""

    def __init__(
        self,
        *,
        artifact: Gemma3L3L4ProgressiveACorpusArtifact,
        role_input_paths: Mapping[DevelopmentRole, Path | str],
    ) -> None:
        if not isinstance(
            artifact,
            Gemma3L3L4ProgressiveACorpusArtifact,
        ):
            raise TypeError(
                "artifact must be Gemma3L3L4ProgressiveACorpusArtifact"
            )
        paths = _canonical_role_paths(role_input_paths)
        for role, path in paths.items():
            if not path.is_file():
                raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                    f"{role} input path is not a regular file"
                )
        self._artifact = artifact
        self._paths = paths
        self._opened_development: dict[
            DevelopmentRole,
            Gemma3L3L4ProgressiveARolePrompts,
        ] = {}
        self._guard_claim_sha256: str | None = None
        self._guard_consumed = False

    @property
    def artifact(self) -> Gemma3L3L4ProgressiveACorpusArtifact:
        return self._artifact

    @property
    def guard_opened(self) -> bool:
        return self._guard_claim_sha256 is not None

    @property
    def guard_consumed(self) -> bool:
        return self._guard_consumed

    def preclaim_view(
        self,
        role: DevelopmentRole,
    ) -> Gemma3L3L4ProgressiveARolePreclaimView:
        """Return exact role membership without opening its source JSON."""

        return self._artifact.role_view(role)

    def _open_and_authenticate(
        self,
        role: DevelopmentRole,
    ) -> Gemma3L3L4ProgressiveARolePrompts:
        view = self._artifact.role_view(role)
        role_input = _parse_role_input(
            path=self._paths[role],
            expected_role=role,
            expected_corpus_id=self._artifact.corpus_id,
            expected_profile=self._artifact.profile,
        )
        if (
            role_input.source_file_sha256
            != view.role_input_file_sha256
            or role_input.ordered_prompt_sha256s
            != view.ordered_prompt_sha256s
            or role_input.family_ids != view.ordered_family_ids
        ):
            raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                f"{role} input differs from its prompt-free preclaim view"
            )
        manifest_sha256 = _role_manifest_sha256(
            corpus_id=role_input.corpus_id,
            profile=role_input.profile,
            role=role_input.role,
            tokenizer_contract_sha256=(
                self._artifact.tokenizer_contract_sha256
            ),
            ordered_prompt_sha256s=role_input.ordered_prompt_sha256s,
            ordered_family_ids=role_input.family_ids,
        )
        if manifest_sha256 != view.manifest_sha256:
            raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                f"{role} manifest changed during opening"
            )
        return role_input

    def open_development_role(
        self,
        role: Literal[
            "calibration_a_fit",
            "calibration_a_selection",
        ],
    ) -> Gemma3L3L4ProgressiveARolePrompts:
        """Open reusable fit or selection text; never dispatch to the guard."""

        selected = _require_role(role)
        if selected == "calibration_a_guard":
            raise Gemma3L3L4ProgressiveAGuardClosedError(
                "guard text requires open_guard_after_claim"
            )
        cached = self._opened_development.get(selected)
        if cached is None:
            cached = self._open_and_authenticate(selected)
            self._opened_development[selected] = cached
        return cached

    def open_guard_after_claim(
        self,
        claim: Gemma3L3L4ProgressiveGuardClaimReceipt,
    ) -> Gemma3L3L4ProgressiveARolePrompts:
        """Consume and open guard text only after a durable matching claim."""

        if not isinstance(
            claim,
            Gemma3L3L4ProgressiveGuardClaimReceipt,
        ):
            raise TypeError(
                "claim must be a durable progressive guard claim receipt"
            )
        if self._guard_consumed:
            raise Gemma3L3L4ProgressiveAGuardClosedError(
                "guard text has already been consumed by this loader"
            )
        guard = self._artifact.role_view("calibration_a_guard")
        if claim.guard_manifest_sha256 != guard.manifest_sha256:
            raise Gemma3L3L4ProgressiveAGuardClosedError(
                "guard claim belongs to another manifest"
            )
        claim.validate_integrity()
        # A valid claim permanently consumes the local opening attempt.  A
        # corrupt role file after this point must not make the guard reusable.
        self._guard_consumed = True
        role_input = self._open_and_authenticate("calibration_a_guard")
        self._guard_claim_sha256 = claim.claim_sha256
        return role_input


def _load_progressive_a_artifact(
    artifact_path: Path | str,
    *,
    expected_artifact_sha256: str | None = None,
    tokenizer_contract: Mapping[str, object] | None = None,
) -> Gemma3L3L4ProgressiveACorpusArtifact:
    source = Path(artifact_path)
    encoded = _read_json_bytes(source, label="progressive A artifact")
    raw = _decode_canonical_mapping(
        encoded,
        label="progressive A artifact",
    )
    artifact = Gemma3L3L4ProgressiveACorpusArtifact.from_dict(raw)
    if (
        expected_artifact_sha256 is not None
        and artifact.artifact_sha256
        != _require_sha256(
            expected_artifact_sha256,
            label="expected corpus artifact",
        )
    ):
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            "progressive A artifact differs from expected logical SHA-256"
        )
    if tokenizer_contract is not None:
        supplied_tokenizer = (
            gemma3_l3_l4_progressive_a_tokenizer_contract_sha256(
                tokenizer_contract
            )
        )
        if supplied_tokenizer != artifact.tokenizer_contract_sha256:
            raise Gemma3L3L4ProgressiveACorpusIntegrityError(
                "live tokenizer contract identity differs from the corpus"
            )
    return artifact


def gemma3_l3_l4_progressive_a_fit_replacement_lineage(
    parent: Gemma3L3L4ProgressiveACorpusArtifact,
    replacement: Gemma3L3L4ProgressiveACorpusArtifact,
) -> dict[str, object]:
    """Validate and summarize an exact fit-role-only corpus replacement."""

    if not isinstance(
        parent,
        Gemma3L3L4ProgressiveACorpusArtifact,
    ) or not isinstance(
        replacement,
        Gemma3L3L4ProgressiveACorpusArtifact,
    ):
        raise TypeError("fit lineage requires two corpus artifacts")
    if (
        parent.corpus_id != replacement.corpus_id
        or parent.profile != replacement.profile
        or parent.tokenizer_contract_sha256
        != replacement.tokenizer_contract_sha256
        or parent.forbidden_assessment_manifest_sha256s
        != replacement.forbidden_assessment_manifest_sha256s
    ):
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            "fit replacement changed a corpus-wide identity"
        )
    preserved_roles = (
        "calibration_a_selection",
        "calibration_a_guard",
    )
    if any(
        parent.role_view(role) != replacement.role_view(role)
        for role in preserved_roles
    ):
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            "fit replacement changed a protected role view"
        )
    parent_fit = parent.role_view("calibration_a_fit")
    replacement_fit = replacement.role_view("calibration_a_fit")
    if parent_fit == replacement_fit:
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            "fit replacement must change the fit role"
        )
    if (
        set(parent_fit.ordered_prompt_sha256s)
        & set(replacement_fit.ordered_prompt_sha256s)
        or set(parent_fit.family_ids) & set(replacement_fit.family_ids)
    ):
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            "replacement fit must be family and prompt disjoint from parent"
        )
    return {
        "kind": "fit_role_only_replacement",
        "parent_corpus_artifact_sha256": parent.artifact_sha256,
        "replacement_corpus_artifact_sha256": (
            replacement.artifact_sha256
        ),
        "parent_fit_manifest_sha256": parent_fit.manifest_sha256,
        "replacement_fit_manifest_sha256": (
            replacement_fit.manifest_sha256
        ),
        "preserved_role_manifest_sha256s": {
            role: replacement.role_view(role).manifest_sha256
            for role in preserved_roles
        },
        "preserved_role_input_file_sha256s": {
            role: replacement.role_view(role).role_input_file_sha256
            for role in preserved_roles
        },
    }


def replace_gemma3_l3_l4_progressive_a_fit_role(
    parent_artifact_path: Path | str,
    *,
    fit_input_path: Path | str,
    expected_parent_artifact_sha256: str,
    tokenizer_contract: Mapping[str, object] | None = None,
) -> Gemma3L3L4ProgressiveACorpusArtifact:
    """Replace only A-fit while preserving unopened selection/guard views."""

    parent = _load_progressive_a_artifact(
        parent_artifact_path,
        expected_artifact_sha256=expected_parent_artifact_sha256,
        tokenizer_contract=tokenizer_contract,
    )
    fit = _parse_role_input(
        path=Path(fit_input_path),
        expected_role="calibration_a_fit",
        expected_corpus_id=parent.corpus_id,
        expected_profile=parent.profile,
    )
    replacement_fit = _view_from_role_input(
        fit,
        tokenizer_contract_sha256=parent.tokenizer_contract_sha256,
    )
    try:
        replacement = Gemma3L3L4ProgressiveACorpusArtifact(
            corpus_id=parent.corpus_id,
            profile=parent.profile,
            tokenizer_contract_sha256=(
                parent.tokenizer_contract_sha256
            ),
            forbidden_assessment_manifest_sha256s=(
                parent.forbidden_assessment_manifest_sha256s
            ),
            role_views=(
                replacement_fit,
                parent.role_view("calibration_a_selection"),
                parent.role_view("calibration_a_guard"),
            ),
        )
        gemma3_l3_l4_progressive_a_fit_replacement_lineage(
            parent,
            replacement,
        )
    except (TypeError, ValueError) as error:
        if isinstance(
            error,
            Gemma3L3L4ProgressiveACorpusIntegrityError,
        ):
            raise
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            f"fit-role-only corpus replacement is invalid: {error}"
        ) from error
    return replacement


def load_gemma3_l3_l4_progressive_a_fit_role(
    artifact_path: Path | str,
    *,
    fit_input_path: Path | str,
    expected_artifact_sha256: str | None = None,
    tokenizer_contract: Mapping[str, object] | None = None,
) -> tuple[
    Gemma3L3L4ProgressiveACorpusArtifact,
    Gemma3L3L4ProgressiveARolePrompts,
]:
    """Open only authenticated A-fit text; selection and guard stay absent."""

    artifact = _load_progressive_a_artifact(
        artifact_path,
        expected_artifact_sha256=expected_artifact_sha256,
        tokenizer_contract=tokenizer_contract,
    )
    fit = _parse_role_input(
        path=Path(fit_input_path),
        expected_role="calibration_a_fit",
        expected_corpus_id=artifact.corpus_id,
        expected_profile=artifact.profile,
    )
    observed = _view_from_role_input(
        fit,
        tokenizer_contract_sha256=artifact.tokenizer_contract_sha256,
    )
    if observed != artifact.role_view("calibration_a_fit"):
        raise Gemma3L3L4ProgressiveACorpusIntegrityError(
            "calibration_a_fit input differs from its prompt-free view"
        )
    return artifact, fit


def load_gemma3_l3_l4_progressive_a_corpus(
    artifact_path: Path | str,
    *,
    role_input_paths: Mapping[DevelopmentRole, Path | str],
    expected_artifact_sha256: str | None = None,
    tokenizer_contract: Mapping[str, object] | None = None,
) -> Gemma3L3L4ProgressiveACorpus:
    """Load only the prompt-free artifact; role JSONs remain unopened."""

    artifact = _load_progressive_a_artifact(
        artifact_path,
        expected_artifact_sha256=expected_artifact_sha256,
        tokenizer_contract=tokenizer_contract,
    )
    return Gemma3L3L4ProgressiveACorpus(
        artifact=artifact,
        role_input_paths=role_input_paths,
    )
