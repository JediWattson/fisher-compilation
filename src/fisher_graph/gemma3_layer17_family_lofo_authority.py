"""Fit-only v8 authority and opaque family-block materialization.

This module is deliberately narrower than the general progressive-A corpus
loader.  Its public authority function accepts only the prompt-free v8
receipt, the prompt-free corpus artifact, and the Calibration-A fit input.
There is no path or role selector through which selection, guard,
Calibration-B, validation, or test text can be opened.

The authenticated fit rows are held only in private runtime slices.  Public
metadata contains aggregate hashes and counts, with opaque ``family_XX``
labels; prompt text, prompt identities, source family identifiers, token ids,
and tokenized-content identities are rejected at the metadata boundary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

import torch

from .compiler.calibration import CalibrationBatch
from .gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveARolePreclaimView,
    Gemma3L3L4ProgressiveARolePrompts,
    load_gemma3_l3_l4_progressive_a_fit_role,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FIT_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
    GEMMA3_LAYER10_V8_CORPUS_ID,
    load_gemma3_layer10_v8_corpus_receipt,
)
from .gemma3_state_conditioned_shape_flow_experiment import (
    _materialize_role,
    _tokenizer_contract as _default_tokenizer_contract,
)
from .gemma3_layer17_family_lofo_protocol import (
    FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
    build_authenticated_v8_layer17_family_lofo_protocol,
    build_default_v8_layer17_family_lofo_protocol,
    validate_v8_layer17_family_lofo_protocol,
)


__all__ = [
    "GEMMA3_LAYER17_FAMILY_LOFO_AUTHORITY_SCHEMA",
    "GEMMA3_LAYER17_FAMILY_LOFO_MATERIALIZATION_SCHEMA",
    "Gemma3Layer17FamilyLOFOAuthority",
    "Gemma3Layer17FamilyLOFOAuthorityError",
    "load_gemma3_layer17_family_lofo_authority",
    "materialize_gemma3_layer17_family_lofo",
    "validate_gemma3_layer17_family_lofo_authority_metadata",
    "validate_gemma3_layer17_family_lofo_materialization_metadata",
]


GEMMA3_LAYER17_FAMILY_LOFO_AUTHORITY_SCHEMA = (
    "fisher_graph.gemma3_layer17_family_lofo_authority"
)
GEMMA3_LAYER17_FAMILY_LOFO_MATERIALIZATION_SCHEMA = (
    "fisher_graph.gemma3_layer17_family_lofo_materialization"
)

_FORMAT_VERSION = 1
_SCIENTIFIC_ROLE = "open_development_calibration_a_fit_family_lofo"
_EXPECTED_EXAMPLES = 256
_EXPECTED_FAMILIES = 8
_EXPECTED_EXAMPLES_PER_FAMILY = 32
_OPAQUE_LABELS = tuple(
    f"family_{index:02d}" for index in range(_EXPECTED_FAMILIES)
)
_AUTHORITY_DOMAIN = b"fisher-graph:layer17-family-lofo-authority:v1\0"
_STREAM_CATALOG_DOMAIN = b"fisher-graph:layer17-family-lofo-streams:v1\0"
_MATERIALIZATION_DOMAIN = (
    b"fisher-graph:layer17-family-lofo-materialization:v1\0"
)
_PROTOCOL_MEMBERSHIP_DOMAIN = (
    b"fisher-graph:gemma3-layer17-family-lofo-membership:v1\0"
)
_PROTOCOL_ALIAS_MAPPING_DOMAIN = (
    b"fisher-graph:gemma3-layer17-family-lofo-alias-map:v1\0"
)
_PROTOCOL_MEMBERSHIP_SCHEMA = (
    "fisher_graph.gemma3_layer17_family_lofo_membership"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_HELDOUT_RECEIPT = {
    "role_ids": ["calibration_b", "validation", "test"],
    "roles_materialized": False,
    "roles_exported": False,
    "roles_tokenized": False,
    "roles_model_evaluated": False,
}
_RECEIPT_SAFETY = {
    "contains_prompt_text": False,
    "contains_family_ids": False,
    "contains_token_ids": False,
    "contains_model_outputs": False,
    "source_safe": True,
}
_PUBLIC_SAFETY = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_family_ids": False,
    "contains_token_ids": False,
    "contains_tokenized_content_identities": False,
    "contains_logits": False,
    "contains_model_or_candidate_weights": False,
    "source_safe": True,
}
_AUTHORITY_ACCESS = {
    "fit_opened": True,
    "fit_tokenized": False,
    "selection_opened": False,
    "guard_opened": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "model_loaded": False,
    "model_evaluated": False,
}
_MATERIALIZATION_ACCESS = {
    **_AUTHORITY_ACCESS,
    "fit_tokenized": True,
}


class Gemma3Layer17FamilyLOFOAuthorityError(ValueError):
    """Raised when the fit-only authority or block boundary drifts."""


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


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            f"{label} must be a mapping"
        )
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            f"{label} must be a positive integer"
        )
    return value


def _sequence_of_strings(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            f"{label} must be a sequence"
        )
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            f"{label} must contain nonempty strings"
        )
    return result


def _resolved_regular_path(path: Path | str, *, label: str) -> Path:
    source = Path(path)
    if not source.is_file():
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            f"{label} must be a regular file"
        )
    return source.resolve(strict=True)


def _forbidden_output_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if normalized.startswith("contains_"):
        return False
    return normalized in {
        "prompt",
        "prompts",
        "prompt_text",
        "prompt_texts",
        "prompt_sha256",
        "prompt_sha256s",
        "ordered_prompt_sha256s",
        "example_id",
        "example_ids",
        "family_id",
        "family_ids",
        "ordered_family_ids",
        "input_ids",
        "token_id",
        "token_ids",
        "tokens",
        "content_sha256",
        "content_sha256s",
        "logits",
        "weights",
        "state_dict",
    }


def _reject_forbidden_output_fields(
    value: object,
    *,
    path: str = "metadata",
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise Gemma3Layer17FamilyLOFOAuthorityError(
                    f"{path} contains a non-string key"
                )
            if _forbidden_output_key(key):
                raise Gemma3Layer17FamilyLOFOAuthorityError(
                    f"{path}.{key} is a forbidden source field"
                )
            _reject_forbidden_output_fields(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_forbidden_output_fields(child, path=f"{path}[{index}]")


def _reject_exact_source_values(
    value: object,
    *,
    forbidden: frozenset[str],
    path: str = "metadata",
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_exact_source_values(
                child,
                forbidden=forbidden,
                path=f"{path}.{key}",
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_exact_source_values(
                child,
                forbidden=forbidden,
                path=f"{path}[{index}]",
            )
    elif isinstance(value, str) and value in forbidden:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            f"{path} contains an exact source identity"
        )


@dataclass(frozen=True, slots=True)
class _FamilySlice:
    opaque_label: str
    source_family: str = field(repr=False)
    source_indices: tuple[int, ...] = field(repr=False)
    prompts: tuple[str, ...] = field(repr=False)
    ordered_prompt_sha256s: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if self.opaque_label not in _OPAQUE_LABELS:
            raise ValueError("family slice label is not canonical")
        if not isinstance(self.source_family, str) or not self.source_family:
            raise ValueError("family slice source family must be nonempty")
        if (
            type(self.source_indices) is not tuple
            or self.source_indices
            != tuple(sorted(set(self.source_indices)))
            or any(
                type(index) is not int or index < 0
                for index in self.source_indices
            )
            or type(self.prompts) is not tuple
            or type(self.ordered_prompt_sha256s) is not tuple
            or len(self.source_indices) != _EXPECTED_EXAMPLES_PER_FAMILY
            or len(self.prompts) != _EXPECTED_EXAMPLES_PER_FAMILY
            or len(self.ordered_prompt_sha256s)
            != _EXPECTED_EXAMPLES_PER_FAMILY
        ):
            raise ValueError("family slice must contain exactly 32 examples")
        if any(
            not isinstance(prompt, str)
            or not prompt
            or prompt != prompt.strip()
            for prompt in self.prompts
        ):
            raise ValueError("family slice prompts must be canonical text")
        if any(
            _SHA256.fullmatch(identity) is None
            for identity in self.ordered_prompt_sha256s
        ) or len(set(self.ordered_prompt_sha256s)) != len(
            self.ordered_prompt_sha256s
        ):
            raise ValueError("family slice prompt identities are invalid")
        if tuple(
            gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
            for prompt in self.prompts
        ) != self.ordered_prompt_sha256s:
            raise ValueError("family slice prompt text differs from its identity")


def _protocol_membership_sha256(
    *,
    fit_manifest_sha256: str,
    partition_kind: str,
    held_family_id: str | None,
    ordered_members: tuple[tuple[str, str], ...],
) -> str:
    return _domain_sha256(
        _PROTOCOL_MEMBERSHIP_DOMAIN,
        {
            "schema": _PROTOCOL_MEMBERSHIP_SCHEMA,
            "format_version": 1,
            "fit_role_manifest_sha256": fit_manifest_sha256,
            "partition_kind": partition_kind,
            "held_family_id": held_family_id,
            "ordered_members": [
                {"identity_sha256": identity, "family_id": family}
                for identity, family in ordered_members
            ],
        },
    )


def _protocol_alias_mapping_sha256(
    *,
    fit_manifest_sha256: str,
    slices: tuple[_FamilySlice, ...],
) -> str:
    return _domain_sha256(
        _PROTOCOL_ALIAS_MAPPING_DOMAIN,
        {
            "schema": "fisher_graph.gemma3_layer17_family_lofo_alias_map",
            "format_version": 1,
            "fit_role_manifest_sha256": fit_manifest_sha256,
            "ordered_alias_mapping": [
                {
                    "family_alias": block.opaque_label,
                    "authenticated_family_id": block.source_family,
                }
                for block in slices
            ],
        },
    )


def _derived_private_protocol_binding(
    *,
    fit_manifest_sha256: str,
    slices: tuple[_FamilySlice, ...],
) -> dict[str, object]:
    """Commit private slices using the frozen protocol's public hash domains."""

    _require_sha256(fit_manifest_sha256, label="fit manifest")
    if (
        type(slices) is not tuple
        or len(slices) != _EXPECTED_FAMILIES
        or tuple(block.opaque_label for block in slices) != _OPAQUE_LABELS
        or len({block.source_family for block in slices})
        != _EXPECTED_FAMILIES
        or any(block.opaque_label == block.source_family for block in slices)
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "fit authority private family catalog is invalid"
        )
    indexed_members = tuple(
        (index, identity, block.source_family)
        for block in slices
        for index, identity in zip(
            block.source_indices,
            block.ordered_prompt_sha256s,
            strict=True,
        )
    )
    ordered_indexed_members = tuple(sorted(indexed_members))
    if (
        len(ordered_indexed_members) != _EXPECTED_EXAMPLES
        or tuple(row[0] for row in ordered_indexed_members)
        != tuple(range(_EXPECTED_EXAMPLES))
        or len({row[1] for row in ordered_indexed_members})
        != _EXPECTED_EXAMPLES
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "fit authority private membership is incomplete or overlapping"
        )
    ordered_members = tuple(
        (identity, family)
        for _, identity, family in ordered_indexed_members
    )
    folds: list[dict[str, object]] = []
    for block in slices:
        held_members = tuple(
            member
            for member in ordered_members
            if member[1] == block.source_family
        )
        training_members = tuple(
            member
            for member in ordered_members
            if member[1] != block.source_family
        )
        if (
            len(held_members) != _EXPECTED_EXAMPLES_PER_FAMILY
            or len(training_members)
            != _EXPECTED_EXAMPLES - _EXPECTED_EXAMPLES_PER_FAMILY
        ):
            raise Gemma3Layer17FamilyLOFOAuthorityError(
                "fit authority private fold cardinality drifted"
            )
        folds.append(
            {
                "held_family_alias": block.opaque_label,
                "training_family_aliases": tuple(
                    candidate.opaque_label
                    for candidate in slices
                    if candidate.opaque_label != block.opaque_label
                ),
                "held_example_count": len(held_members),
                "training_example_count": len(training_members),
                "held_membership_sha256": _protocol_membership_sha256(
                    fit_manifest_sha256=fit_manifest_sha256,
                    partition_kind="held_family",
                    held_family_id=block.source_family,
                    ordered_members=held_members,
                ),
                "training_membership_sha256": _protocol_membership_sha256(
                    fit_manifest_sha256=fit_manifest_sha256,
                    partition_kind="training_complement",
                    held_family_id=block.source_family,
                    ordered_members=training_members,
                ),
            }
        )
    return {
        "fit_membership_sha256": _protocol_membership_sha256(
            fit_manifest_sha256=fit_manifest_sha256,
            partition_kind="fit_all",
            held_family_id=None,
            ordered_members=ordered_members,
        ),
        "family_alias_mapping_sha256": _protocol_alias_mapping_sha256(
            fit_manifest_sha256=fit_manifest_sha256,
            slices=slices,
        ),
        "fold_count": len(folds),
        "folds": tuple(folds),
    }


def _public_protocol_binding(raw: Mapping[str, object]) -> dict[str, object]:
    protocol = validate_v8_layer17_family_lofo_protocol(raw)
    protocol_sha256 = _require_sha256(
        protocol.get("artifact_sha256"),
        label="frozen family-LOFO protocol",
    )
    if protocol_sha256 != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "family-LOFO protocol identity differs from frozen v8"
        )
    corpus = _mapping(
        protocol.get("corpus_authority"),
        label="family-LOFO protocol corpus authority",
    )
    role_bindings = _mapping(
        protocol.get("role_bindings"),
        label="family-LOFO protocol role bindings",
    )
    fit = _mapping(
        role_bindings.get("fit"),
        label="family-LOFO protocol fit binding",
    )
    folds_raw = protocol.get("folds")
    if isinstance(folds_raw, (str, bytes)) or not isinstance(
        folds_raw, Sequence
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "family-LOFO protocol folds must be a sequence"
        )
    folds: list[dict[str, object]] = []
    for index, raw_fold in enumerate(folds_raw):
        fold = _mapping(raw_fold, label=f"family-LOFO protocol fold {index}")
        training_aliases = _sequence_of_strings(
            fold.get("training_family_aliases"),
            label=f"family-LOFO protocol fold {index} training aliases",
        )
        row = {
            "held_family_alias": fold.get("held_family_alias"),
            "training_family_aliases": training_aliases,
            "held_example_count": fold.get("held_example_count"),
            "training_example_count": fold.get("training_example_count"),
            "held_membership_sha256": _require_sha256(
                fold.get("held_membership_sha256"),
                label=f"family-LOFO protocol fold {index} held membership",
            ),
            "training_membership_sha256": _require_sha256(
                fold.get("training_membership_sha256"),
                label=f"family-LOFO protocol fold {index} training membership",
            ),
        }
        if (
            row["held_family_alias"] != _OPAQUE_LABELS[index]
            or training_aliases
            != tuple(
                alias
                for alias in _OPAQUE_LABELS
                if alias != _OPAQUE_LABELS[index]
            )
            or row["held_example_count"] != _EXPECTED_EXAMPLES_PER_FAMILY
            or row["training_example_count"]
            != _EXPECTED_EXAMPLES - _EXPECTED_EXAMPLES_PER_FAMILY
        ):
            raise Gemma3Layer17FamilyLOFOAuthorityError(
                "family-LOFO protocol fold ownership drifted"
            )
        folds.append(row)
    if len(folds) != _EXPECTED_FAMILIES:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "family-LOFO protocol must contain exactly eight folds"
        )
    return {
        "protocol_artifact_sha256": protocol_sha256,
        "corpus_artifact_sha256": _require_sha256(
            corpus.get("artifact_sha256"),
            label="protocol corpus artifact",
        ),
        "tokenizer_contract_sha256": _require_sha256(
            corpus.get("tokenizer_contract_sha256"),
            label="protocol tokenizer contract",
        ),
        "fit_manifest_sha256": _require_sha256(
            fit.get("manifest_sha256"),
            label="protocol fit manifest",
        ),
        "fit_role_file_sha256": _require_sha256(
            fit.get("source_file_sha256"),
            label="protocol fit role file",
        ),
        "fit_membership_sha256": _require_sha256(
            corpus.get("fit_membership_sha256"),
            label="protocol fit membership",
        ),
        "family_alias_mapping_sha256": _require_sha256(
            corpus.get("family_alias_mapping_sha256"),
            label="protocol family alias mapping",
        ),
        "fold_count": len(folds),
        "folds": tuple(folds),
    }


@dataclass(frozen=True, slots=True)
class Gemma3Layer17FamilyLOFOAuthority:
    """Authenticated fit-only authority with private source-bearing slices."""

    receipt_sha256: str
    receipt_file_sha256: str
    corpus_artifact_sha256: str
    corpus_artifact_file_sha256: str
    tokenizer_contract_sha256: str
    fit_manifest_sha256: str
    fit_role_file_sha256: str
    protocol_artifact_sha256: str
    _max_length: int = field(repr=False)
    _tokenization_batch_size: int = field(repr=False)
    _device_name: str = field(repr=False)
    _slices: tuple[_FamilySlice, ...] = field(repr=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("receipt", self.receipt_sha256),
            ("receipt file", self.receipt_file_sha256),
            ("corpus artifact", self.corpus_artifact_sha256),
            ("corpus artifact file", self.corpus_artifact_file_sha256),
            ("tokenizer contract", self.tokenizer_contract_sha256),
            ("fit manifest", self.fit_manifest_sha256),
            ("fit role file", self.fit_role_file_sha256),
            ("family-LOFO protocol", self.protocol_artifact_sha256),
        ):
            _require_sha256(value, label=label)
        _positive_int(self._max_length, label="max_length")
        _positive_int(
            self._tokenization_batch_size,
            label="tokenization_batch_size",
        )
        if not isinstance(self._device_name, str) or not self._device_name:
            raise ValueError("device name must be nonempty")
        if (
            type(self._slices) is not tuple
            or len(self._slices) != _EXPECTED_FAMILIES
            or tuple(item.opaque_label for item in self._slices)
            != _OPAQUE_LABELS
        ):
            raise ValueError("fit authority must contain eight opaque slices")
        self._validated_protocol_binding()

    def _validated_protocol_binding(self) -> dict[str, object]:
        expected = _public_protocol_binding(
            build_default_v8_layer17_family_lofo_protocol()
        )
        derived = _derived_private_protocol_binding(
            fit_manifest_sha256=self.fit_manifest_sha256,
            slices=self._slices,
        )
        expected_private = {
            key: expected[key]
            for key in (
                "fit_membership_sha256",
                "family_alias_mapping_sha256",
                "fold_count",
                "folds",
            )
        }
        if (
            self.protocol_artifact_sha256
            != expected["protocol_artifact_sha256"]
            or self.corpus_artifact_sha256
            != expected["corpus_artifact_sha256"]
            or self.tokenizer_contract_sha256
            != expected["tokenizer_contract_sha256"]
            or self.fit_manifest_sha256
            != expected["fit_manifest_sha256"]
            or self.fit_role_file_sha256
            != expected["fit_role_file_sha256"]
            or derived != expected_private
        ):
            raise Gemma3Layer17FamilyLOFOAuthorityError(
                "fit authority differs from the frozen family-LOFO protocol"
            )
        return {
            "protocol_artifact_sha256": self.protocol_artifact_sha256,
            **derived,
        }

    @property
    def authority_sha256(self) -> str:
        return _domain_sha256(_AUTHORITY_DOMAIN, self._metadata_payload())

    def _metadata_payload(self) -> dict[str, object]:
        protocol_binding = self._validated_protocol_binding()
        return {
            "schema": GEMMA3_LAYER17_FAMILY_LOFO_AUTHORITY_SCHEMA,
            "format_version": _FORMAT_VERSION,
            "scientific_role": _SCIENTIFIC_ROLE,
            "heldout_confirmation": False,
            "receipt": {
                "receipt_sha256": self.receipt_sha256,
                "receipt_file_sha256": self.receipt_file_sha256,
            },
            "protocol": protocol_binding,
            "corpus": {
                "corpus_artifact_sha256": self.corpus_artifact_sha256,
                "corpus_artifact_file_sha256": (
                    self.corpus_artifact_file_sha256
                ),
                "tokenizer_contract_sha256": (
                    self.tokenizer_contract_sha256
                ),
                "fit_manifest_sha256": self.fit_manifest_sha256,
                "fit_role_file_sha256": self.fit_role_file_sha256,
                "example_count": _EXPECTED_EXAMPLES,
                "block_count": _EXPECTED_FAMILIES,
                "examples_per_block": _EXPECTED_EXAMPLES_PER_FAMILY,
                "block_labels": _OPAQUE_LABELS,
            },
            "access": dict(_AUTHORITY_ACCESS),
            "safety": dict(_PUBLIC_SAFETY),
        }

    def _source_values(self) -> frozenset[str]:
        return frozenset(
            value
            for block in self._slices
            for value in (
                block.source_family,
                *block.prompts,
                *block.ordered_prompt_sha256s,
            )
        )

    def metadata(self) -> dict[str, object]:
        payload = self._metadata_payload()
        result = {**payload, "authority_sha256": self.authority_sha256}
        validate_gemma3_layer17_family_lofo_authority_metadata(result)
        _reject_exact_source_values(
            result,
            forbidden=self._source_values(),
        )
        return result


def _build_family_slices(
    role: Gemma3L3L4ProgressiveARolePrompts,
    view: Gemma3L3L4ProgressiveARolePreclaimView,
) -> tuple[_FamilySlice, ...]:
    family_order = view.family_ids
    if (
        len(family_order) != _EXPECTED_FAMILIES
        or family_order != tuple(sorted(set(family_order)))
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "fit view must contain eight canonical families"
        )
    counts = Counter(role.family_ids)
    if set(counts) != set(family_order) or any(
        count != _EXPECTED_EXAMPLES_PER_FAMILY
        for count in counts.values()
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "fit role must contain exactly 32 examples per family"
        )
    slices: list[_FamilySlice] = []
    for opaque_label, source_family in zip(
        _OPAQUE_LABELS,
        family_order,
        strict=True,
    ):
        selected = tuple(
            (index, prompt, prompt_sha256)
            for index, (
                prompt,
                prompt_sha256,
                observed_family,
            ) in enumerate(
                zip(
                    role.prompts,
                    role.ordered_prompt_sha256s,
                    role.family_ids,
                    strict=True,
                )
            )
            if observed_family == source_family
        )
        slices.append(
            _FamilySlice(
                opaque_label=opaque_label,
                source_family=source_family,
                source_indices=tuple(row[0] for row in selected),
                prompts=tuple(row[1] for row in selected),
                ordered_prompt_sha256s=tuple(row[2] for row in selected),
            )
        )
    if any(block.opaque_label in role.family_ids for block in slices):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "opaque labels collide with source family identities"
        )
    return tuple(slices)


def _tokenizer_runtime(
    tokenizer_contract: Mapping[str, object],
) -> tuple[int, int, str]:
    max_length = _positive_int(
        tokenizer_contract.get("max_length"),
        label="tokenizer max_length",
    )
    batch_size = _positive_int(
        tokenizer_contract.get("tokenization_batch_size"),
        label="tokenization_batch_size",
    )
    device = tokenizer_contract.get("device")
    if not isinstance(device, str) or not device:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "tokenizer device must be a nonempty string"
        )
    return max_length, batch_size, device


def load_gemma3_layer17_family_lofo_authority(
    *,
    corpus_receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    fit_input_path: Path | str = DEFAULT_FIT_OUTPUT,
    tokenizer_contract: Mapping[str, object] | None = None,
) -> Gemma3Layer17FamilyLOFOAuthority:
    """Authenticate receipt -> corpus -> A-fit without protected role paths."""

    receipt_path = _resolved_regular_path(
        corpus_receipt_path,
        label="v8 receipt",
    )
    artifact_path = _resolved_regular_path(
        corpus_artifact_path,
        label="v8 corpus artifact",
    )
    fit_path = _resolved_regular_path(
        fit_input_path,
        label="v8 A-fit input",
    )
    if len({receipt_path, artifact_path, fit_path}) != 3:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "receipt, corpus artifact, and fit input must be distinct files"
        )

    receipt = load_gemma3_layer10_v8_corpus_receipt(receipt_path)
    receipt_corpus = _mapping(
        receipt.get("corpus"),
        label="v8 receipt corpus binding",
    )
    artifact_file_sha256 = _file_sha256(artifact_path)
    if (
        receipt_corpus.get("artifact_file_sha256")
        != artifact_file_sha256
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "v8 corpus artifact file differs from the receipt"
        )
    receipt_roles = _mapping(
        receipt.get("roles"),
        label="v8 receipt role catalog",
    )
    receipt_fit = _mapping(
        receipt_roles.get("calibration_a_fit"),
        label="v8 receipt fit binding",
    )
    frozen_protocol_binding = _public_protocol_binding(
        build_default_v8_layer17_family_lofo_protocol()
    )
    if (
        receipt_corpus.get("artifact_sha256")
        != frozen_protocol_binding["corpus_artifact_sha256"]
        or receipt_corpus.get("tokenizer_contract_sha256")
        != frozen_protocol_binding["tokenizer_contract_sha256"]
        or receipt_fit.get("manifest_sha256")
        != frozen_protocol_binding["fit_manifest_sha256"]
        or receipt_fit.get("role_input_file_sha256")
        != frozen_protocol_binding["fit_role_file_sha256"]
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "v8 receipt differs from the frozen family-LOFO protocol"
        )

    contract = dict(
        _default_tokenizer_contract()
        if tokenizer_contract is None
        else tokenizer_contract
    )
    max_length, batch_size, device_name = _tokenizer_runtime(contract)
    artifact, fit_role = load_gemma3_l3_l4_progressive_a_fit_role(
        artifact_path,
        fit_input_path=fit_path,
        expected_artifact_sha256=_require_sha256(
            receipt_corpus.get("artifact_sha256"),
            label="receipt corpus artifact",
        ),
        tokenizer_contract=contract,
    )
    fit_view = artifact.role_view("calibration_a_fit")
    if (
        receipt.get("corpus_id") != GEMMA3_LAYER10_V8_CORPUS_ID
        or receipt.get("corpus_id") != artifact.corpus_id
        or receipt.get("profile") != "full"
        or receipt.get("profile") != artifact.profile
        or receipt.get("heldout") != _HELDOUT_RECEIPT
        or receipt.get("safety") != _RECEIPT_SAFETY
        or receipt_corpus.get("tokenizer_contract_sha256")
        != artifact.tokenizer_contract_sha256
        or receipt_fit.get("role_id") != "calibration_a_fit"
        or receipt_fit.get("manifest_sha256") != fit_view.manifest_sha256
        or receipt_fit.get("role_input_file_sha256")
        != fit_role.source_file_sha256
        or receipt_fit.get("example_count") != _EXPECTED_EXAMPLES
        or receipt_fit.get("family_count") != _EXPECTED_FAMILIES
        or fit_view.example_count != _EXPECTED_EXAMPLES
        or len(fit_role.prompts) != _EXPECTED_EXAMPLES
        or fit_role.source_file_sha256 != fit_view.role_input_file_sha256
        or fit_role.ordered_prompt_sha256s
        != fit_view.ordered_prompt_sha256s
        or fit_role.family_ids != fit_view.ordered_family_ids
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "v8 receipt, corpus artifact, and A-fit role disagree"
        )
    authenticated_protocol_binding = _public_protocol_binding(
        build_authenticated_v8_layer17_family_lofo_protocol(
            artifact.to_dict()
        )
    )
    if authenticated_protocol_binding != frozen_protocol_binding:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "authenticated family-LOFO protocol binding drifted"
        )

    authority = Gemma3Layer17FamilyLOFOAuthority(
        receipt_sha256=_require_sha256(
            receipt.get("receipt_sha256"),
            label="v8 receipt",
        ),
        receipt_file_sha256=_file_sha256(receipt_path),
        corpus_artifact_sha256=artifact.artifact_sha256,
        corpus_artifact_file_sha256=artifact_file_sha256,
        tokenizer_contract_sha256=artifact.tokenizer_contract_sha256,
        fit_manifest_sha256=fit_view.manifest_sha256,
        fit_role_file_sha256=fit_role.source_file_sha256,
        protocol_artifact_sha256=_require_sha256(
            authenticated_protocol_binding["protocol_artifact_sha256"],
            label="authenticated family-LOFO protocol",
        ),
        _max_length=max_length,
        _tokenization_batch_size=batch_size,
        _device_name=device_name,
        _slices=_build_family_slices(fit_role, fit_view),
    )
    authority.metadata()
    return authority


def _batch_example_ids(
    batches: Sequence[CalibrationBatch],
) -> tuple[str, ...]:
    if not batches or any(
        not isinstance(batch, CalibrationBatch)
        or batch.example_ids is None
        for batch in batches
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "family materialization omitted bound example identities"
        )
    return tuple(
        example_id
        for batch in batches
        for example_id in batch.example_ids or ()
    )


def materialize_gemma3_layer17_family_lofo(
    authority: Gemma3Layer17FamilyLOFOAuthority,
    tokenizer: object,
) -> tuple[
    tuple[tuple[str, tuple[CalibrationBatch, ...]], ...],
    dict[str, object],
]:
    """Tokenize eight private family slices and return source-safe metadata.

    The returned batches are ephemeral runtime data and retain their internal
    example bindings for integrity checks.  Only the second return value is a
    serialization-safe public artifact.
    """

    if not isinstance(authority, Gemma3Layer17FamilyLOFOAuthority):
        raise TypeError(
            "authority must be Gemma3Layer17FamilyLOFOAuthority"
        )
    # Recompute the private membership commitments at the last point before
    # tokenization so a replaced or drifted authority cannot reuse an old hash.
    authority.metadata()
    device = torch.device(authority._device_name)
    materialized: list[tuple[str, tuple[CalibrationBatch, ...]]] = []
    expected_ids: list[str] = []
    observed_ids: list[str] = []
    stream_sha256s: list[str] = []
    blocks_metadata: dict[str, dict[str, int]] = {}
    total_batches = 0
    total_valid = 0
    total_supervised = 0

    for block in authority._slices:
        batches, stream = _materialize_role(
            tokenizer,
            block,  # type: ignore[arg-type]
            split_name=f"layer17_fit_lofo_{block.opaque_label}",
            max_length=authority._max_length,
            tokenization_batch_size=authority._tokenization_batch_size,
            device=device,
        )
        if type(batches) is not tuple:
            batches = tuple(batches)
        block_ids = _batch_example_ids(batches)
        if block_ids != block.ordered_prompt_sha256s:
            raise Gemma3Layer17FamilyLOFOAuthorityError(
                "family materialization membership drifted"
            )
        stream_mapping = _mapping(
            stream,
            label="family tokenized stream",
        )
        if (
            stream_mapping.get("split")
            != f"layer17_fit_lofo_{block.opaque_label}"
            or stream_mapping.get("batches") != len(batches)
            or stream_mapping.get("sequences")
            != _EXPECTED_EXAMPLES_PER_FAMILY
        ):
            raise Gemma3Layer17FamilyLOFOAuthorityError(
                "family tokenized stream metadata drifted"
            )
        stream_sha256s.append(
            _require_sha256(
                stream_mapping.get("serialized_sha256"),
                label="family tokenized stream",
            )
        )
        valid = sum(
            int(batch.valid_positions.sum().item()) for batch in batches
        )
        supervised = sum(
            int((batch.targets != -100).sum().item()) for batch in batches
        )
        if valid <= 0 or supervised <= 0:
            raise Gemma3Layer17FamilyLOFOAuthorityError(
                "family materialization contains no usable tokens"
            )
        blocks_metadata[block.opaque_label] = {
            "example_count": len(block_ids),
            "batch_count": len(batches),
            "logical_valid_tokens": valid,
            "supervised_tokens": supervised,
        }
        total_batches += len(batches)
        total_valid += valid
        total_supervised += supervised
        expected_ids.extend(block.ordered_prompt_sha256s)
        observed_ids.extend(block_ids)
        materialized.append((block.opaque_label, batches))

    if (
        tuple(observed_ids) != tuple(expected_ids)
        or len(observed_ids) != _EXPECTED_EXAMPLES
        or len(set(observed_ids)) != _EXPECTED_EXAMPLES
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "family materialization is incomplete or overlapping"
        )

    tokenization = {
        "block_count": _EXPECTED_FAMILIES,
        "block_labels": _OPAQUE_LABELS,
        "example_count": _EXPECTED_EXAMPLES,
        "examples_per_block": _EXPECTED_EXAMPLES_PER_FAMILY,
        "batch_count": total_batches,
        "logical_valid_tokens": total_valid,
        "supervised_tokens": total_supervised,
        "max_length": authority._max_length,
        "tokenization_batch_size": authority._tokenization_batch_size,
        "device": authority._device_name,
        "stream_catalog_sha256": _domain_sha256(
            _STREAM_CATALOG_DOMAIN,
            tuple(stream_sha256s),
        ),
        "blocks": blocks_metadata,
    }
    payload: dict[str, object] = {
        "schema": GEMMA3_LAYER17_FAMILY_LOFO_MATERIALIZATION_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "scientific_role": _SCIENTIFIC_ROLE,
        "heldout_confirmation": False,
        "authority_sha256": authority.authority_sha256,
        "tokenization": tokenization,
        "access": dict(_MATERIALIZATION_ACCESS),
        "safety": dict(_PUBLIC_SAFETY),
    }
    metadata = {
        **payload,
        "materialization_sha256": _domain_sha256(
            _MATERIALIZATION_DOMAIN,
            payload,
        ),
    }
    validate_gemma3_layer17_family_lofo_materialization_metadata(metadata)
    _reject_exact_source_values(
        metadata,
        forbidden=authority._source_values(),
    )
    return tuple(materialized), metadata


def validate_gemma3_layer17_family_lofo_authority_metadata(
    value: Mapping[str, object],
) -> None:
    """Validate the exact source-safe authority metadata schema."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "format_version",
        "scientific_role",
        "heldout_confirmation",
        "receipt",
        "protocol",
        "corpus",
        "access",
        "safety",
        "authority_sha256",
    }:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "authority metadata fields differ"
        )
    receipt = _mapping(value.get("receipt"), label="authority receipt")
    protocol = _mapping(value.get("protocol"), label="authority protocol")
    corpus = _mapping(value.get("corpus"), label="authority corpus")
    if set(receipt) != {"receipt_sha256", "receipt_file_sha256"} or any(
        _SHA256.fullmatch(item) is None
        for item in receipt.values()
        if isinstance(item, str)
    ) or any(not isinstance(item, str) for item in receipt.values()):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "authority receipt binding is invalid"
        )
    if set(corpus) != {
        "corpus_artifact_sha256",
        "corpus_artifact_file_sha256",
        "tokenizer_contract_sha256",
        "fit_manifest_sha256",
        "fit_role_file_sha256",
        "example_count",
        "block_count",
        "examples_per_block",
        "block_labels",
    }:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "authority corpus binding fields differ"
        )
    for field_name in (
        "corpus_artifact_sha256",
        "corpus_artifact_file_sha256",
        "tokenizer_contract_sha256",
        "fit_manifest_sha256",
        "fit_role_file_sha256",
    ):
        _require_sha256(corpus.get(field_name), label=field_name)
    expected_binding = _public_protocol_binding(
        build_default_v8_layer17_family_lofo_protocol()
    )
    expected_public_protocol = {
        key: expected_binding[key]
        for key in (
            "protocol_artifact_sha256",
            "fit_membership_sha256",
            "family_alias_mapping_sha256",
            "fold_count",
            "folds",
        )
    }
    labels = _sequence_of_strings(
        corpus.get("block_labels"),
        label="authority block labels",
    )
    payload = {
        key: child for key, child in value.items() if key != "authority_sha256"
    }
    if (
        value.get("schema")
        != GEMMA3_LAYER17_FAMILY_LOFO_AUTHORITY_SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or value.get("scientific_role") != _SCIENTIFIC_ROLE
        or value.get("heldout_confirmation") is not False
        or corpus.get("example_count") != _EXPECTED_EXAMPLES
        or corpus.get("block_count") != _EXPECTED_FAMILIES
        or corpus.get("examples_per_block")
        != _EXPECTED_EXAMPLES_PER_FAMILY
        or labels != _OPAQUE_LABELS
        or _canonical_json_bytes(protocol)
        != _canonical_json_bytes(expected_public_protocol)
        or corpus.get("corpus_artifact_sha256")
        != expected_binding["corpus_artifact_sha256"]
        or corpus.get("tokenizer_contract_sha256")
        != expected_binding["tokenizer_contract_sha256"]
        or corpus.get("fit_manifest_sha256")
        != expected_binding["fit_manifest_sha256"]
        or corpus.get("fit_role_file_sha256")
        != expected_binding["fit_role_file_sha256"]
        or value.get("access") != _AUTHORITY_ACCESS
        or value.get("safety") != _PUBLIC_SAFETY
        or value.get("authority_sha256")
        != _domain_sha256(_AUTHORITY_DOMAIN, payload)
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "authority metadata integrity check failed"
        )
    _reject_forbidden_output_fields(value)


def validate_gemma3_layer17_family_lofo_materialization_metadata(
    value: Mapping[str, object],
) -> None:
    """Validate exact public metadata without accepting source identities."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "format_version",
        "scientific_role",
        "heldout_confirmation",
        "authority_sha256",
        "tokenization",
        "access",
        "safety",
        "materialization_sha256",
    }:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "materialization metadata fields differ"
        )
    tokenization = _mapping(
        value.get("tokenization"),
        label="materialization tokenization",
    )
    if set(tokenization) != {
        "block_count",
        "block_labels",
        "example_count",
        "examples_per_block",
        "batch_count",
        "logical_valid_tokens",
        "supervised_tokens",
        "max_length",
        "tokenization_batch_size",
        "device",
        "stream_catalog_sha256",
        "blocks",
    }:
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "materialization tokenization fields differ"
        )
    labels = _sequence_of_strings(
        tokenization.get("block_labels"),
        label="materialization block labels",
    )
    blocks = _mapping(
        tokenization.get("blocks"),
        label="materialization blocks",
    )
    if set(blocks) != set(_OPAQUE_LABELS):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "materialization blocks differ from opaque catalog"
        )
    batch_total = 0
    valid_total = 0
    supervised_total = 0
    for label in _OPAQUE_LABELS:
        block = _mapping(blocks[label], label=f"materialization {label}")
        if set(block) != {
            "example_count",
            "batch_count",
            "logical_valid_tokens",
            "supervised_tokens",
        } or block.get("example_count") != _EXPECTED_EXAMPLES_PER_FAMILY:
            raise Gemma3Layer17FamilyLOFOAuthorityError(
                "materialization block fields or count differ"
            )
        batch_total += _positive_int(
            block.get("batch_count"), label="block batch_count"
        )
        valid_total += _positive_int(
            block.get("logical_valid_tokens"),
            label="block logical_valid_tokens",
        )
        supervised_total += _positive_int(
            block.get("supervised_tokens"),
            label="block supervised_tokens",
        )
    payload = {
        key: child
        for key, child in value.items()
        if key != "materialization_sha256"
    }
    _require_sha256(
        value.get("authority_sha256"),
        label="materialization authority",
    )
    _require_sha256(
        tokenization.get("stream_catalog_sha256"),
        label="materialization stream catalog",
    )
    device = tokenization.get("device")
    if (
        value.get("schema")
        != GEMMA3_LAYER17_FAMILY_LOFO_MATERIALIZATION_SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or value.get("scientific_role") != _SCIENTIFIC_ROLE
        or value.get("heldout_confirmation") is not False
        or tokenization.get("block_count") != _EXPECTED_FAMILIES
        or labels != _OPAQUE_LABELS
        or tokenization.get("example_count") != _EXPECTED_EXAMPLES
        or tokenization.get("examples_per_block")
        != _EXPECTED_EXAMPLES_PER_FAMILY
        or tokenization.get("batch_count") != batch_total
        or tokenization.get("logical_valid_tokens") != valid_total
        or tokenization.get("supervised_tokens") != supervised_total
        or type(tokenization.get("max_length")) is not int
        or int(tokenization["max_length"]) <= 0
        or type(tokenization.get("tokenization_batch_size")) is not int
        or int(tokenization["tokenization_batch_size"]) <= 0
        or not isinstance(device, str)
        or not device
        or value.get("access") != _MATERIALIZATION_ACCESS
        or value.get("safety") != _PUBLIC_SAFETY
        or value.get("materialization_sha256")
        != _domain_sha256(_MATERIALIZATION_DOMAIN, payload)
    ):
        raise Gemma3Layer17FamilyLOFOAuthorityError(
            "materialization metadata integrity check failed"
        )
    _reject_forbidden_output_fields(value)
