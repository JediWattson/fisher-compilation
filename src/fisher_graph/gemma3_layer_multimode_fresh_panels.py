"""Author and authenticate the prospective layer-multimode fresh panels.

The module is standard-library-only and import inert.  It can deterministically
author candidate B3, C3, D3, and reserve corpora with a caller-supplied pinned
tokenizer, but it never writes them.  A future publication step may freeze the
returned canonical bytes and pin their hashes in the v3 setup.

Runtime access is deliberately separate from authoring.  ``opaque_preflight``
decodes only the prompt-free manifest and byte-hashes every raw corpus.  An
original preflight receipt plus an already-frozen 18-layer A selection is then
required to create ``SequentialPanelSource``.  The source opens B3 -> C3 -> D3
once, requires a terminal receipt between roles, and has no reserve-open path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence
import unicodedata
import weakref


PANEL_ID = "gemma3-layer-local-flow-multimode-fresh-panels-v1"
MANIFEST_SCHEMA = "fisher_graph.gemma3_layer_multimode.fresh_manifest.v1"
RECEIPT_SCHEMA = (
    "fisher_graph.gemma3_layer_multimode.fresh_panel_receipt.v1"
)
ROLE_CORPUS_SCHEMA = (
    "fisher_graph.gemma3_layer_multimode.fresh_role_corpus.v1"
)
FORMAT_VERSION = 1

B3_ROLE = "B3_teacher_forced_layerwise"
C3_ROLE = "C3_full_stack_source_free"
D3_ROLE = "D3_final_heldout_validation"
RESERVE_ROLE = "reserve_v3_unopened"
EXECUTABLE_ROLES = (B3_ROLE, C3_ROLE, D3_ROLE)
ORDERED_ROLES = EXECUTABLE_ROLES + (RESERVE_ROLE,)

DEFAULT_ROLE_CORPORA = {
    role: Path("examples")
    / f"gemma3_layer_multimode_{role}_corpus_v1.json"
    for role in ORDERED_ROLES
}
DEFAULT_MANIFEST = Path(
    "examples/gemma3_layer_multimode_fresh_manifest_v1.json"
)
DEFAULT_RECEIPT = Path(
    "artifacts/research/gemma3-layer-multimode-fresh-panels-v1.json"
)

FAMILY_IDS = (
    "fresh-v3/symbol-permutation-chain",
    "fresh-v3/multi-bin-transfer-ledger",
    "fresh-v3/binary-neighborhood-update",
    "fresh-v3/deque-operation-trace",
    "fresh-v3/fractional-recipe-scaling",
    "fresh-v3/base-conversion-checksum",
)
FAMILY_SEMANTICS = (
    "composition of ordered swaps over a finite symbol permutation",
    "conservation-aware transfers among independently named integer bins",
    "synchronous binary updates under a local neighborhood stencil",
    "ordered push pop and rotation operations over a bounded deque",
    "exact rational scaling of a synthetic multi-part recipe",
    "digit conversion followed by a weighted modular checksum",
)
TARGET_TOKEN_COUNTS = (96, 511, 512, 513, 514, 576, 640, 768)
RESERVE_TARGET_TOKEN_COUNTS = (576, 768)
HARD_LONG_MINIMUM_TOKEN_COUNT = 576
LAYER_COUNT = 18
PRIOR_GENERATIONS = ("v1", "v2")
PINNED_PRIOR_MANIFEST_FILE_SHA256S = {
    # v2 deliberately reused the v1 fresh-panel manifest and left its unopened
    # roles unopened; the generation label still gets a distinct catalog root.
    "v1": "5031f44454873a369ba7968c5bbfd6cf7e4a7340feaf0eddf6aee51398c948d7",
    "v2": "5031f44454873a369ba7968c5bbfd6cf7e4a7340feaf0eddf6aee51398c948d7",
}
PINNED_PRIOR_LINEAGE_CATALOG_SHA256S = {
    "v1": "55dcf145026323622008cddf6226da817e84a03840b104ced48f214042168f6c",
    "v2": "98a7d0671d3fb30988444be4d3e5f2f54c9770ab4e0561db2e5964f72d303057",
}

REQUIRED_FREEZE_STAGE_KIND_BY_ROLE = {
    B3_ROLE: "B3_teacher_forced_evaluation",
    C3_ROLE: "C3_full_stack_source_free_evaluation",
    D3_ROLE: "D3_final_heldout_evaluation",
}

TOKENIZER_CONTRACT = {
    "model_id": "google/gemma-3-270m",
    "revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
    "tokenizer_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
    "tokenizer_json_sha256": (
        "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c"
    ),
    "tokenizer_config_sha256": (
        "94b03056ec5831e9021c3e3fe9db682778c4f2d081c4188f1fcb8b90541cf1cf"
    ),
    "special_tokens_map_sha256": (
        "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397"
    ),
    "add_special_tokens": True,
    "add_bos_token": True,
    "add_eos_token": False,
    "bos_token_id": 2,
    "sliding_window": 512,
}

_IDENTITY_KINDS = (
    "prompt",
    "canonical_prompt",
    "template",
    "semantic_lineage",
)
_PAD_ATOMS = (" lumen", " x", " z", " q", " filler")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MANIFEST_DOMAIN = b"fisher-graph:layer-multimode:fresh-manifest:v1\0"
_RECEIPT_DOMAIN = b"fisher-graph:layer-multimode:fresh-receipt:v1\0"
_BUNDLE_DOMAIN = b"fisher-graph:layer-multimode:fresh-bundle:v1\0"
_CATALOG_DOMAIN = b"fisher-graph:layer-multimode:prior-lineages:v1\0"
_PROMPT_DOMAIN = b"fisher-graph:layer-multimode:prompt:v1\0"
_CANONICAL_DOMAIN = b"fisher-graph:layer-multimode:canonical:v1\0"
_TEMPLATE_DOMAIN = b"fisher-graph:layer-multimode:template:v1\0"
_LINEAGE_DOMAIN = b"fisher-graph:layer-multimode:lineage:v1\0"
_TOKEN_IDS_DOMAIN = b"fisher-graph:layer-multimode:token-ids:v1\0"


class FreshPanelIntegrityError(RuntimeError):
    """Candidate panel bytes or semantic receipts failed authentication."""


class FreshPanelFirewallError(FreshPanelIntegrityError):
    """A role was opened early, reopened, or accessed after terminal failure."""


def _canonical_payload(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_bytes(value: object) -> bytes:
    """Return the only accepted on-disk JSON encoding for panel artifacts."""

    return _canonical_payload(value) + b"\n"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_payload(value)).hexdigest()


def _text_hash(domain: bytes, value: str) -> str:
    return hashlib.sha256(domain + value.encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FreshPanelIntegrityError(f"{label} must be lowercase SHA-256")
    return value


def _strict_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FreshPanelIntegrityError(f"{label} must be a mapping")

    def plain(item: object) -> object:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise FreshPanelIntegrityError(
                    f"{label} mapping keys must be strings"
                )
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        return item

    try:
        copied = json.loads(json.dumps(plain(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise FreshPanelIntegrityError(f"{label} must be strict JSON") from exc
    if not isinstance(copied, dict):
        raise FreshPanelIntegrityError(f"{label} must be a JSON object")
    return copied


def _strict_json_loads(encoded: bytes, *, label: str) -> object:
    if len(encoded) > _MAX_JSON_BYTES:
        raise FreshPanelIntegrityError(f"{label} exceeds the JSON byte cap")

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise FreshPanelIntegrityError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FreshPanelIntegrityError(
                    f"{label} contains non-finite constant {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreshPanelIntegrityError(f"{label} is not strict JSON") from exc


def _decode_canonical_mapping(encoded: bytes, *, label: str) -> dict[str, object]:
    decoded = _strict_json_loads(encoded, label=label)
    if not isinstance(decoded, dict):
        raise FreshPanelIntegrityError(f"{label} must decode to an object")
    if canonical_json_bytes(decoded) != encoded:
        raise FreshPanelIntegrityError(f"{label} is not canonical JSON")
    return decoded


def _canonical_prompt(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _contains_mapping_key(value: object, forbidden: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden in value or any(
            _contains_mapping_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_mapping_key(item, forbidden) for item in value)
    return False


def _token_ids(tokenizer: object, text: str) -> tuple[int, ...]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must expose encode(text, add_special_tokens=True)")
    try:
        encoded = encode(text, add_special_tokens=True)
    except TypeError:
        encoded = encode(text)
    ids = getattr(encoded, "ids", encoded)
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise TypeError("tokenizer encode result must expose an integer id sequence")
    result = tuple(ids)
    if not result or any(type(item) is not int or item < 0 for item in result):
        raise TypeError("token ids must be nonempty nonnegative integers")
    return result


def _token_ids_sha256(ids: Sequence[int]) -> str:
    return _domain_hash(_TOKEN_IDS_DOMAIN, list(ids))


def _family_core(
    family_index: int,
    *,
    nonce: str,
    role_index: int,
    example_index: int,
) -> str:
    a = 2 + (role_index + example_index) % 7
    b = 3 + (family_index * 2 + example_index) % 9
    c = 5 + (role_index * 3 + family_index + example_index) % 11
    if family_index == 0:
        return (
            f"Sealed case {nonce}. Symbols P Q R S begin in that order. "
            f"Swap positions one and {a % 4 + 1}, then two and {b % 4 + 1}, "
            "then reverse the first three. Determine the final symbol order."
        )
    if family_index == 1:
        return (
            f"Sealed case {nonce}. Bins amber cobalt and ivory start with "
            f"{a + 8}, {b + 9}, and {c + 10} units. Move {a} amber to cobalt, "
            f"then {b} cobalt to ivory. Determine all final balances."
        )
    if family_index == 2:
        bits = "".join(str((role_index + example_index + i) % 2) for i in range(7))
        return (
            f"Sealed case {nonce}. A circular row begins as {bits}. During each "
            f"of {a % 3 + 2} synchronous rounds, a cell becomes the xor of its "
            "two previous neighbors. Determine the resulting circular row."
        )
    if family_index == 3:
        return (
            f"Sealed case {nonce}. A deque starts red blue gold teal. Push silver "
            f"left, rotate right {a % 3 + 1} places, pop right, then push violet "
            f"right and rotate left {b % 3 + 1}. Determine its final order."
        )
    if family_index == 4:
        return (
            f"Sealed case {nonce}. A recipe uses {a} parts mica, {b} parts flax, "
            f"and {c} parts salt. Scale it so mica becomes {a * 3} parts, then "
            "halve the complete batch. Determine each exact final quantity."
        )
    return (
        f"Sealed case {nonce}. Interpret the digits {a % 7 + 1}{b % 7 + 1}"
        f"{c % 7 + 1} in base eight. Convert it to base ten for the record. "
        "Separately multiply the original digits by weights one two and three, "
        "then report that checksum modulo eleven."
    )


def _pad_to_exact_token_count(
    tokenizer: object,
    semantic_core: str,
    target_token_count: int,
) -> tuple[str, tuple[int, ...], int]:
    core_ids = _token_ids(tokenizer, semantic_core)
    core_count = len(core_ids)
    if core_count > target_token_count:
        raise FreshPanelIntegrityError(
            f"semantic core has {core_count} tokens, above target "
            f"{target_token_count}"
        )
    remaining = target_token_count - core_count
    if remaining == 0:
        return semantic_core, core_ids, core_count

    for atom in _PAD_ATOMS:
        candidate = semantic_core + atom * remaining
        ids = _token_ids(tokenizer, candidate)
        if len(ids) == target_token_count:
            return candidate, ids, core_count

    text = semantic_core
    ids = core_ids
    for _ in range(target_token_count * 2):
        if len(ids) == target_token_count:
            return text, ids, core_count
        best: tuple[str, tuple[int, ...]] | None = None
        for atom in _PAD_ATOMS:
            candidate = text + atom
            candidate_ids = _token_ids(tokenizer, candidate)
            if len(ids) < len(candidate_ids) <= target_token_count:
                if best is None or len(candidate_ids) > len(best[1]):
                    best = (candidate, candidate_ids)
        if best is None:
            break
        text, ids = best
    raise FreshPanelIntegrityError(
        f"neutral padding could not reach exact target {target_token_count}"
    )


@dataclass(frozen=True, slots=True)
class PriorSemanticLineageCatalog:
    """One frozen prior-generation semantic-lineage denylist."""

    generation: str
    lineage_sha256s: tuple[str, ...]
    catalog_sha256: str

    def __post_init__(self) -> None:
        if self.generation not in PRIOR_GENERATIONS:
            raise FreshPanelIntegrityError("prior generation must be v1 or v2")
        if not self.lineage_sha256s:
            raise FreshPanelIntegrityError("prior lineage catalog may not be empty")
        if self.lineage_sha256s != tuple(sorted(set(self.lineage_sha256s))):
            raise FreshPanelIntegrityError(
                "prior lineage hashes must be sorted and unique"
            )
        for value in self.lineage_sha256s:
            _require_sha256(value, label="prior semantic lineage")
        expected = _domain_hash(
            _CATALOG_DOMAIN,
            {
                "generation": self.generation,
                "lineage_sha256s": list(self.lineage_sha256s),
            },
        )
        if self.catalog_sha256 != expected:
            raise FreshPanelIntegrityError("prior lineage catalog digest drifted")

    def state_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "lineage_sha256s": list(self.lineage_sha256s),
            "catalog_sha256": self.catalog_sha256,
        }


def make_prior_lineage_catalog(
    generation: str,
    lineage_sha256s: Sequence[str],
) -> PriorSemanticLineageCatalog:
    values = tuple(sorted(set(lineage_sha256s)))
    digest = _domain_hash(
        _CATALOG_DOMAIN,
        {"generation": generation, "lineage_sha256s": list(values)},
    )
    return PriorSemanticLineageCatalog(generation, values, digest)


def prior_lineage_catalog_from_manifest(
    generation: str,
    manifest: Mapping[str, object],
) -> PriorSemanticLineageCatalog:
    """Extract old semantic-instance hashes without reading prompt text."""

    state = _strict_mapping(manifest, label=f"{generation} prior manifest")
    roles = state.get("roles")
    if not isinstance(roles, Mapping):
        raise FreshPanelIntegrityError("prior manifest roles are missing")
    values: set[str] = set()
    for role in roles.values():
        if not isinstance(role, Mapping):
            continue
        members = role.get("ordered_members")
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, Mapping):
                    continue
                value = member.get("semantic_lineage_sha256")
                if value is None:
                    value = member.get("semantic_instance_sha256")
                if value is not None:
                    values.add(_require_sha256(value, label="prior lineage"))
        catalog = role.get("identity_catalog")
        if isinstance(catalog, Mapping):
            for field in ("semantic_lineage_sha256s", "semantic_sha256s"):
                field_values = catalog.get(field)
                if isinstance(field_values, list):
                    for value in field_values:
                        values.add(
                            _require_sha256(value, label="prior lineage catalog")
                        )
    return make_prior_lineage_catalog(generation, tuple(values))


@dataclass(frozen=True, slots=True)
class AuthoredFreshPanelBundle:
    """Immutable canonical candidate bytes; construction performs no writes."""

    manifest_bytes: bytes
    role_corpus_items: tuple[tuple[str, bytes], ...]
    receipt_bytes: bytes
    bundle_sha256: str

    def __post_init__(self) -> None:
        if tuple(role for role, _ in self.role_corpus_items) != ORDERED_ROLES:
            raise FreshPanelIntegrityError("authored role byte order drifted")
        if any(not isinstance(value, bytes) for _, value in self.role_corpus_items):
            raise FreshPanelIntegrityError("authored corpus payloads must be bytes")
        expected = _domain_hash(
            _BUNDLE_DOMAIN,
            {
                "manifest_file_sha256": _sha_bytes(self.manifest_bytes),
                "role_file_sha256s": {
                    role: _sha_bytes(value)
                    for role, value in self.role_corpus_items
                },
            },
        )
        if self.bundle_sha256 != expected:
            raise FreshPanelIntegrityError("authored bundle digest drifted")
        manifest = validate_manifest(
            _decode_canonical_mapping(
                self.manifest_bytes,
                label="authored prompt-free manifest",
            )
        )
        receipt = validate_receipt(
            _decode_canonical_mapping(
                self.receipt_bytes,
                label="authored prompt-free receipt",
            )
        )
        _validate_receipt_bindings(
            receipt,
            manifest=manifest,
            manifest_file_sha256=self.manifest_file_sha256,
            role_file_sha256s=self.role_file_sha256s,
            bundle_sha256=self.bundle_sha256,
        )

    @property
    def manifest_file_sha256(self) -> str:
        return _sha_bytes(self.manifest_bytes)

    @property
    def role_file_sha256s(self) -> Mapping[str, str]:
        return MappingProxyType(
            {role: _sha_bytes(value) for role, value in self.role_corpus_items}
        )

    @property
    def role_corpus_bytes(self) -> Mapping[str, bytes]:
        return MappingProxyType(dict(self.role_corpus_items))

    @property
    def receipt_file_sha256(self) -> str:
        return _sha_bytes(self.receipt_bytes)


def _role_schedule(role: str) -> tuple[int, ...]:
    if role == RESERVE_ROLE:
        return RESERVE_TARGET_TOKEN_COUNTS
    return TARGET_TOKEN_COUNTS


def _example_lineage_id(
    role: str,
    family_id: str,
    example_index: int,
    target: int,
) -> str:
    seed = _domain_hash(
        _LINEAGE_DOMAIN,
        {
            "panel_id": PANEL_ID,
            "role": role,
            "family_id": family_id,
            "example_index": example_index,
            "target_token_count": target,
        },
    )
    return f"{PANEL_ID}/{role}/{family_id}/{example_index}/{target}/{seed[:16]}"


def _author_role(
    tokenizer: object,
    *,
    role: str,
    role_index: int,
) -> dict[str, object]:
    schedule = _role_schedule(role)
    examples: list[dict[str, object]] = []
    for family_index, family_id in enumerate(FAMILY_IDS):
        for example_index, target in enumerate(schedule):
            lineage_id = _example_lineage_id(
                role,
                family_id,
                example_index,
                target,
            )
            lineage_sha = _text_hash(_LINEAGE_DOMAIN, lineage_id)
            nonce = lineage_sha[:12]
            semantic_core = _family_core(
                family_index,
                nonce=nonce,
                role_index=role_index,
                example_index=example_index,
            )
            prompt, ids, semantic_prefix_count = _pad_to_exact_token_count(
                tokenizer,
                semantic_core,
                target,
            )
            hard_long = target >= HARD_LONG_MINIMUM_TOKEN_COUNT
            if hard_long and semantic_prefix_count > target - 512:
                raise FreshPanelIntegrityError(
                    f"{role} {family_id} target {target} places semantic "
                    "content inside the final 512-token neutral suffix"
                )
            example_id = (
                f"v3-r{role_index + 1}-f{family_index + 1}-"
                f"e{example_index + 1}-l{target}-{nonce[:8]}"
            )
            template_id = f"{PANEL_ID}/{role}/{family_id}/template-v1"
            examples.append(
                {
                    "example_id": example_id,
                    "family_id": family_id,
                    "target_token_count": target,
                    "observed_token_count": len(ids),
                    "predicted_token_count": len(ids) - 1,
                    "hard_long": hard_long,
                    "semantic_prefix_token_count": semantic_prefix_count,
                    "hard_long_final512_semantic_prefix_excluded": hard_long,
                    "prompt": prompt,
                    "prompt_sha256": _text_hash(_PROMPT_DOMAIN, prompt),
                    "canonical_prompt_sha256": _text_hash(
                        _CANONICAL_DOMAIN,
                        _canonical_prompt(prompt),
                    ),
                    "template_id": template_id,
                    "template_sha256": _text_hash(_TEMPLATE_DOMAIN, template_id),
                    "semantic_lineage_id": lineage_id,
                    "semantic_lineage_sha256": lineage_sha,
                    "lineage_parent_sha256s": [],
                    "token_ids_sha256": _token_ids_sha256(ids),
                }
            )
    return {
        "schema": ROLE_CORPUS_SCHEMA,
        "format_version": FORMAT_VERSION,
        "panel_id": PANEL_ID,
        "role": role,
        "family_ids": list(FAMILY_IDS),
        "family_semantics": list(FAMILY_SEMANTICS),
        "target_token_counts_per_family": list(schedule),
        "example_count": len(examples),
        "examples": examples,
    }


def _member_from_example(example: Mapping[str, object]) -> dict[str, object]:
    return {
        key: example[key]
        for key in (
            "example_id",
            "family_id",
            "target_token_count",
            "observed_token_count",
            "predicted_token_count",
            "hard_long",
            "semantic_prefix_token_count",
            "hard_long_final512_semantic_prefix_excluded",
            "prompt_sha256",
            "canonical_prompt_sha256",
            "template_sha256",
            "semantic_lineage_sha256",
            "lineage_parent_sha256s",
            "token_ids_sha256",
        )
    }


def _identity_catalog(members: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "prompt_sha256s": sorted({str(value["prompt_sha256"]) for value in members}),
        "canonical_prompt_sha256s": sorted(
            {str(value["canonical_prompt_sha256"]) for value in members}
        ),
        "template_sha256s": sorted(
            {str(value["template_sha256"]) for value in members}
        ),
        "semantic_lineage_sha256s": sorted(
            {str(value["semantic_lineage_sha256"]) for value in members}
        ),
    }


def _catalog_sets(role: Mapping[str, object]) -> dict[str, set[str]]:
    catalog = role["identity_catalog"]
    assert isinstance(catalog, Mapping)
    fields = {
        "prompt": "prompt_sha256s",
        "canonical_prompt": "canonical_prompt_sha256s",
        "template": "template_sha256s",
        "semantic_lineage": "semantic_lineage_sha256s",
    }
    return {
        kind: set(catalog[field])  # type: ignore[arg-type]
        for kind, field in fields.items()
    }


def _overlap_audit(
    roles: Mapping[str, Mapping[str, object]],
    prior_catalogs: Sequence[PriorSemanticLineageCatalog],
) -> dict[str, object]:
    identity = {role: _catalog_sets(roles[role]) for role in ORDERED_ROLES}
    role_pair: dict[str, int] = {}
    for left_index, left in enumerate(ORDERED_ROLES):
        for right in ORDERED_ROLES[left_index + 1 :]:
            for kind in _IDENTITY_KINDS:
                role_pair[f"{left}|{right}|{kind}"] = len(
                    identity[left][kind] & identity[right][kind]
                )
    prior: dict[str, int] = {}
    for role in ORDERED_ROLES:
        for catalog in prior_catalogs:
            prior[f"{role}|{catalog.generation}|semantic_lineage"] = len(
                identity[role]["semantic_lineage"]
                & set(catalog.lineage_sha256s)
            )
    return {
        "identity_kinds": list(_IDENTITY_KINDS),
        "role_pair_intersection_counts": role_pair,
        "prior_generation_semantic_lineage_intersection_counts": prior,
        "all_intersection_counts_zero": not any(role_pair.values())
        and not any(prior.values()),
        "declared_lineage_parent_lists_all_empty": True,
        "semantic_nonreuse_author_review_required": True,
    }


def _role_manifest(
    role: str,
    corpus: Mapping[str, object],
    encoded: bytes,
) -> dict[str, object]:
    raw_examples = corpus["examples"]
    assert isinstance(raw_examples, list)
    members = [_member_from_example(value) for value in raw_examples]
    family_counts = Counter(str(value["family_id"]) for value in members)
    hard_counts = Counter(bool(value["hard_long"]) for value in members)
    descriptor = {
        "role": role,
        "path": f"examples/gemma3_layer_multimode_{role}_corpus_v1.json",
        "access_class": (
            "opaque_bytes_only_never_decode_tokenize_or_model"
            if role == RESERVE_ROLE
            else "sequential_semantic_open_then_tokenizer_or_model"
        ),
        "file_sha256": _sha_bytes(encoded),
        "example_count": len(members),
        "family_count": len(FAMILY_IDS),
        "examples_per_family": len(_role_schedule(role)),
        "target_token_counts_per_family": list(_role_schedule(role)),
        "predicted_token_count": sum(
            int(value["predicted_token_count"]) for value in members
        ),
        "hard_long_example_count": hard_counts[True],
        "non_hard_long_example_count": hard_counts[False],
        "family_example_counts": dict(sorted(family_counts.items())),
        "ordered_members": members,
        "identity_catalog": _identity_catalog(members),
    }
    return descriptor


def _build_receipt(
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    role_file_sha256s: Mapping[str, str],
    bundle_sha256: str,
) -> dict[str, object]:
    roles = manifest["roles"]
    assert isinstance(roles, Mapping)
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "format_version": FORMAT_VERSION,
        "panel_id": PANEL_ID,
        "artifact_role": "prompt_free_publication_commit_marker",
        "manifest": {
            "path": str(DEFAULT_MANIFEST),
            "file_sha256": manifest_file_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "role_files": {
            role: {
                "path": str(DEFAULT_ROLE_CORPORA[role]),
                "file_sha256": role_file_sha256s[role],
                "example_count": roles[role]["example_count"],
                "predicted_token_count": roles[role]["predicted_token_count"],
            }
            for role in ORDERED_ROLES
        },
        "bundle_sha256": bundle_sha256,
        "overlap_audit": {
            "pairwise_and_v1_v2_semantic_lineage_intersections_zero": True,
            "manifest_overlap_audit_sha256": _domain_hash(
                _RECEIPT_DOMAIN,
                manifest["overlap_audit"],
            ),
        },
        "D3_bootstrap_strata": manifest["D3_bootstrap_strata"],
        "D3_all_12_family_x_hard_long_strata_nonempty": True,
        "publication_protocol": {
            "no_clobber": True,
            "regular_files_only": True,
            "canonical_parent_directories_must_preexist": True,
            "file_fsync_before_publish": True,
            "directory_fsync_after_publish": True,
            "role_files_published_before_manifest": True,
            "receipt_published_last_as_commit_marker": True,
            "rollback_removes_only_files_created_by_failed_invocation": True,
            "rollback_attempts_every_registered_target_and_touched_parent": (
                True
            ),
            "rollback_failures_are_notes_on_the_primary_exception": True,
        },
        "claim_boundary": {
            "tokenizer_only_authoring": True,
            "model_weights_loaded": False,
            "model_execution_performed": False,
            "scientific_result_present": False,
            "canonical_write_performed_by_author_function": False,
            "reserve_semantically_opened": False,
        },
    }
    receipt["receipt_sha256"] = _domain_hash(_RECEIPT_DOMAIN, receipt)
    return receipt


def author_fresh_panel_bundle(
    tokenizer: object,
    *,
    prior_lineage_catalogs: Sequence[PriorSemanticLineageCatalog],
) -> AuthoredFreshPanelBundle:
    """Author deterministic canonical candidate bytes without writing files."""

    catalogs = tuple(prior_lineage_catalogs)
    if tuple(value.generation for value in catalogs) != PRIOR_GENERATIONS:
        raise FreshPanelIntegrityError(
            "exactly ordered v1 and v2 prior lineage catalogs are required"
        )
    raw_roles: dict[str, dict[str, object]] = {}
    encoded_roles: dict[str, bytes] = {}
    role_manifests: dict[str, dict[str, object]] = {}
    for role_index, role in enumerate(ORDERED_ROLES):
        corpus = _author_role(tokenizer, role=role, role_index=role_index)
        encoded = canonical_json_bytes(corpus)
        raw_roles[role] = corpus
        encoded_roles[role] = encoded
        role_manifests[role] = _role_manifest(role, corpus, encoded)

    overlap = _overlap_audit(role_manifests, catalogs)
    if overlap["all_intersection_counts_zero"] is not True:
        raise FreshPanelIntegrityError(
            "fresh roles overlap each other or a v1/v2 semantic lineage"
        )
    d3_members = role_manifests[D3_ROLE]["ordered_members"]
    assert isinstance(d3_members, list)
    d3_strata = Counter(
        f"{value['family_id']}|hard_long={str(value['hard_long']).lower()}"
        for value in d3_members
    )
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "format_version": FORMAT_VERSION,
        "panel_id": PANEL_ID,
        "ordered_roles": list(ORDERED_ROLES),
        "executable_roles": list(EXECUTABLE_ROLES),
        "reserve_role": RESERVE_ROLE,
        "family_ids": list(FAMILY_IDS),
        "family_semantics": list(FAMILY_SEMANTICS),
        "target_token_counts": list(TARGET_TOKEN_COUNTS),
        "reserve_target_token_counts": list(RESERVE_TARGET_TOKEN_COUNTS),
        "hard_long_minimum_token_count": HARD_LONG_MINIMUM_TOKEN_COUNT,
        "tokenizer_contract": TOKENIZER_CONTRACT,
        "required_freeze_stage_kind_by_role": (
            REQUIRED_FREEZE_STAGE_KIND_BY_ROLE
        ),
        "roles": role_manifests,
        "prior_lineage_catalogs": [value.state_dict() for value in catalogs],
        "overlap_audit": overlap,
        "D3_bootstrap_strata": dict(sorted(d3_strata.items())),
        "D3_all_12_family_x_hard_long_strata_nonempty": (
            len(d3_strata) == 12 and all(d3_strata.values())
        ),
        "sequential_firewall": {
            "A_selection_layer_count_required_before_B3": LAYER_COUNT,
            "ordered_fresh_roles": list(EXECUTABLE_ROLES),
            "B3_requires_frozen_A_selection_receipt": True,
            "C3_requires_terminal_passing_B3_receipt": True,
            "D3_requires_terminal_passing_C3_receipt": True,
            "role_reopen_authorized": False,
            "reserve_semantic_open_authorized": False,
            "later_role_early_access_invalidates_run": True,
        },
        "authoring_receipt": {
            "deterministic_source_only": True,
            "tokenizer_execution_performed": True,
            "model_weights_loaded": False,
            "model_execution_performed": False,
            "canonical_manifest_or_artifact_written": False,
            "raw_prompt_text_allowed_in_future_reports": False,
            "raw_token_ids_allowed_in_future_reports": False,
        },
    }
    manifest["manifest_sha256"] = _domain_hash(_MANIFEST_DOMAIN, manifest)
    validated = validate_manifest(manifest)
    manifest_bytes = canonical_json_bytes(validated)
    role_items = tuple((role, encoded_roles[role]) for role in ORDERED_ROLES)
    bundle_digest = _domain_hash(
        _BUNDLE_DOMAIN,
        {
            "manifest_file_sha256": _sha_bytes(manifest_bytes),
            "role_file_sha256s": {
                role: _sha_bytes(encoded_roles[role]) for role in ORDERED_ROLES
            },
        },
    )
    receipt = _build_receipt(
        manifest=validated,
        manifest_file_sha256=_sha_bytes(manifest_bytes),
        role_file_sha256s={
            role: _sha_bytes(encoded_roles[role]) for role in ORDERED_ROLES
        },
        bundle_sha256=bundle_digest,
    )
    validated_receipt = validate_receipt(receipt)
    return AuthoredFreshPanelBundle(
        manifest_bytes,
        role_items,
        canonical_json_bytes(validated_receipt),
        bundle_digest,
    )


def _validate_prior_catalogs(value: object) -> tuple[PriorSemanticLineageCatalog, ...]:
    if not isinstance(value, list) or len(value) != 2:
        raise FreshPanelIntegrityError("manifest needs v1 and v2 lineage catalogs")
    result: list[PriorSemanticLineageCatalog] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise FreshPanelIntegrityError("prior lineage catalog is invalid")
        result.append(
            PriorSemanticLineageCatalog(
                generation=str(item.get("generation")),
                lineage_sha256s=tuple(item.get("lineage_sha256s", ())),
                catalog_sha256=str(item.get("catalog_sha256")),
            )
        )
    catalogs = tuple(result)
    if tuple(value.generation for value in catalogs) != PRIOR_GENERATIONS:
        raise FreshPanelIntegrityError("prior lineage catalog order drifted")
    return catalogs


def validate_manifest(value: Mapping[str, object]) -> dict[str, object]:
    """Validate a prompt-free candidate manifest and all declared inventories."""

    manifest = _strict_mapping(value, label="fresh multimode manifest")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("panel_id") != PANEL_ID
        or tuple(manifest.get("ordered_roles", ())) != ORDERED_ROLES
        or tuple(manifest.get("executable_roles", ())) != EXECUTABLE_ROLES
        or manifest.get("reserve_role") != RESERVE_ROLE
        or tuple(manifest.get("family_ids", ())) != FAMILY_IDS
        or tuple(manifest.get("family_semantics", ())) != FAMILY_SEMANTICS
        or tuple(manifest.get("target_token_counts", ()))
        != TARGET_TOKEN_COUNTS
        or tuple(manifest.get("reserve_target_token_counts", ()))
        != RESERVE_TARGET_TOKEN_COUNTS
        or manifest.get("hard_long_minimum_token_count")
        != HARD_LONG_MINIMUM_TOKEN_COUNT
        or manifest.get("tokenizer_contract") != TOKENIZER_CONTRACT
        or manifest.get("required_freeze_stage_kind_by_role")
        != REQUIRED_FREEZE_STAGE_KIND_BY_ROLE
    ):
        raise FreshPanelIntegrityError("fresh multimode manifest header drifted")
    observed_digest = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    if observed_digest != _domain_hash(_MANIFEST_DOMAIN, payload):
        raise FreshPanelIntegrityError("fresh multimode manifest digest drifted")

    catalogs = _validate_prior_catalogs(manifest.get("prior_lineage_catalogs"))
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or tuple(roles) != ORDERED_ROLES:
        raise FreshPanelIntegrityError("ordered role inventory drifted")
    role_states: dict[str, dict[str, object]] = {}
    for role in ORDERED_ROLES:
        descriptor = roles.get(role)
        if not isinstance(descriptor, Mapping):
            raise FreshPanelIntegrityError(f"{role} descriptor is missing")
        state = dict(descriptor)
        schedule = _role_schedule(role)
        expected_count = len(FAMILY_IDS) * len(schedule)
        expected_rows = len(FAMILY_IDS) * sum(value - 1 for value in schedule)
        expected_access = (
            "opaque_bytes_only_never_decode_tokenize_or_model"
            if role == RESERVE_ROLE
            else "sequential_semantic_open_then_tokenizer_or_model"
        )
        if (
            state.get("role") != role
            or state.get("access_class") != expected_access
            or state.get("example_count") != expected_count
            or state.get("family_count") != len(FAMILY_IDS)
            or state.get("examples_per_family") != len(schedule)
            or tuple(state.get("target_token_counts_per_family", ())) != schedule
            or state.get("predicted_token_count") != expected_rows
            or state.get("family_example_counts")
            != {family: len(schedule) for family in FAMILY_IDS}
        ):
            raise FreshPanelIntegrityError(f"{role} geometry drifted")
        _require_sha256(state.get("file_sha256"), label=f"{role} file")
        members = state.get("ordered_members")
        if not isinstance(members, list) or len(members) != expected_count:
            raise FreshPanelIntegrityError(f"{role} member inventory drifted")
        for index, member in enumerate(members):
            if not isinstance(member, Mapping):
                raise FreshPanelIntegrityError(f"{role} member is invalid")
            family_index, within_family = divmod(index, len(schedule))
            target = schedule[within_family]
            hard_long = target >= HARD_LONG_MINIMUM_TOKEN_COUNT
            if (
                member.get("family_id") != FAMILY_IDS[family_index]
                or member.get("target_token_count") != target
                or member.get("observed_token_count") != target
                or member.get("predicted_token_count") != target - 1
                or member.get("hard_long") is not hard_long
                or member.get(
                    "hard_long_final512_semantic_prefix_excluded"
                )
                is not hard_long
                or member.get("lineage_parent_sha256s") != []
            ):
                raise FreshPanelIntegrityError(
                    f"{role} member schedule drifted at {index}"
                )
            prefix_count = member.get("semantic_prefix_token_count")
            if type(prefix_count) is not int or not 1 <= prefix_count <= target:
                raise FreshPanelIntegrityError(f"{role} semantic prefix drifted")
            if hard_long and prefix_count > target - 512:
                raise FreshPanelIntegrityError(
                    f"{role} hard-long semantic prefix enters final 512"
                )
            for field in (
                "prompt_sha256",
                "canonical_prompt_sha256",
                "template_sha256",
                "semantic_lineage_sha256",
                "token_ids_sha256",
            ):
                _require_sha256(member.get(field), label=f"{role} {field}")
        expected_catalog = _identity_catalog(members)
        if state.get("identity_catalog") != expected_catalog:
            raise FreshPanelIntegrityError(f"{role} identity catalog drifted")
        hard_count = sum(bool(value["hard_long"]) for value in members)
        if (
            state.get("hard_long_example_count") != hard_count
            or state.get("non_hard_long_example_count")
            != expected_count - hard_count
        ):
            raise FreshPanelIntegrityError(f"{role} hard-long counts drifted")
        role_states[role] = state

    computed_overlap = _overlap_audit(role_states, catalogs)
    if manifest.get("overlap_audit") != computed_overlap:
        raise FreshPanelIntegrityError("semantic-lineage overlap audit drifted")
    if computed_overlap["all_intersection_counts_zero"] is not True:
        raise FreshPanelIntegrityError("semantic-lineage firewall is not zero")

    d3_members = role_states[D3_ROLE]["ordered_members"]
    assert isinstance(d3_members, list)
    strata = Counter(
        f"{value['family_id']}|hard_long={str(value['hard_long']).lower()}"
        for value in d3_members
    )
    if (
        manifest.get("D3_bootstrap_strata") != dict(sorted(strata.items()))
        or len(strata) != 12
        or not all(strata.values())
        or manifest.get("D3_all_12_family_x_hard_long_strata_nonempty")
        is not True
    ):
        raise FreshPanelIntegrityError("D3 bootstrap stratum closure drifted")
    expected_firewall = {
        "A_selection_layer_count_required_before_B3": LAYER_COUNT,
        "ordered_fresh_roles": list(EXECUTABLE_ROLES),
        "B3_requires_frozen_A_selection_receipt": True,
        "C3_requires_terminal_passing_B3_receipt": True,
        "D3_requires_terminal_passing_C3_receipt": True,
        "role_reopen_authorized": False,
        "reserve_semantic_open_authorized": False,
        "later_role_early_access_invalidates_run": True,
    }
    if manifest.get("sequential_firewall") != expected_firewall:
        raise FreshPanelIntegrityError("sequential firewall contract drifted")
    expected_authoring = {
        "deterministic_source_only": True,
        "tokenizer_execution_performed": True,
        "model_weights_loaded": False,
        "model_execution_performed": False,
        "canonical_manifest_or_artifact_written": False,
        "raw_prompt_text_allowed_in_future_reports": False,
        "raw_token_ids_allowed_in_future_reports": False,
    }
    if manifest.get("authoring_receipt") != expected_authoring:
        raise FreshPanelIntegrityError("authoring claim boundary drifted")
    if _contains_mapping_key(manifest, "prompt"):
        raise FreshPanelIntegrityError("prompt text field leaked into manifest")
    return manifest


def validate_receipt(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the prompt-free publication commit marker."""

    receipt = _strict_mapping(value, label="fresh multimode panel receipt")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("format_version") != FORMAT_VERSION
        or receipt.get("panel_id") != PANEL_ID
        or receipt.get("artifact_role")
        != "prompt_free_publication_commit_marker"
    ):
        raise FreshPanelIntegrityError("fresh panel receipt header drifted")
    observed = receipt.get("receipt_sha256")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if observed != _domain_hash(_RECEIPT_DOMAIN, payload):
        raise FreshPanelIntegrityError("fresh panel receipt digest drifted")
    manifest = receipt.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("path") != str(
        DEFAULT_MANIFEST
    ):
        raise FreshPanelIntegrityError("receipt manifest path drifted")
    _require_sha256(manifest.get("file_sha256"), label="receipt manifest file")
    _require_sha256(
        manifest.get("manifest_sha256"),
        label="receipt manifest content",
    )
    role_files = receipt.get("role_files")
    if not isinstance(role_files, Mapping) or tuple(role_files) != ORDERED_ROLES:
        raise FreshPanelIntegrityError("receipt role file inventory drifted")
    for role in ORDERED_ROLES:
        value = role_files[role]
        if not isinstance(value, Mapping):
            raise FreshPanelIntegrityError(f"receipt {role} file is invalid")
        schedule = _role_schedule(role)
        if (
            value.get("path") != str(DEFAULT_ROLE_CORPORA[role])
            or value.get("example_count") != len(FAMILY_IDS) * len(schedule)
            or value.get("predicted_token_count")
            != len(FAMILY_IDS) * sum(target - 1 for target in schedule)
        ):
            raise FreshPanelIntegrityError(f"receipt {role} geometry drifted")
        _require_sha256(value.get("file_sha256"), label=f"receipt {role} file")
    _require_sha256(receipt.get("bundle_sha256"), label="receipt bundle")
    expected_overlap = {
        "pairwise_and_v1_v2_semantic_lineage_intersections_zero": True,
        "manifest_overlap_audit_sha256": receipt.get("overlap_audit", {}).get(
            "manifest_overlap_audit_sha256"
        )
        if isinstance(receipt.get("overlap_audit"), Mapping)
        else None,
    }
    overlap = receipt.get("overlap_audit")
    if overlap != expected_overlap:
        raise FreshPanelIntegrityError("receipt overlap claim drifted")
    _require_sha256(
        expected_overlap["manifest_overlap_audit_sha256"],
        label="receipt overlap audit",
    )
    strata = receipt.get("D3_bootstrap_strata")
    if (
        not isinstance(strata, Mapping)
        or len(strata) != 12
        or any(type(value) is not int or value < 1 for value in strata.values())
        or receipt.get("D3_all_12_family_x_hard_long_strata_nonempty")
        is not True
    ):
        raise FreshPanelIntegrityError("receipt D3 stratum closure drifted")
    expected_publication = {
        "no_clobber": True,
        "regular_files_only": True,
        "canonical_parent_directories_must_preexist": True,
        "file_fsync_before_publish": True,
        "directory_fsync_after_publish": True,
        "role_files_published_before_manifest": True,
        "receipt_published_last_as_commit_marker": True,
        "rollback_removes_only_files_created_by_failed_invocation": True,
        "rollback_attempts_every_registered_target_and_touched_parent": True,
        "rollback_failures_are_notes_on_the_primary_exception": True,
    }
    if receipt.get("publication_protocol") != expected_publication:
        raise FreshPanelIntegrityError("receipt publication protocol drifted")
    expected_boundary = {
        "tokenizer_only_authoring": True,
        "model_weights_loaded": False,
        "model_execution_performed": False,
        "scientific_result_present": False,
        "canonical_write_performed_by_author_function": False,
        "reserve_semantically_opened": False,
    }
    if receipt.get("claim_boundary") != expected_boundary:
        raise FreshPanelIntegrityError("receipt claim boundary drifted")
    if _contains_mapping_key(receipt, "prompt"):
        raise FreshPanelIntegrityError("prompt text field leaked into receipt")
    return receipt


def _validate_receipt_bindings(
    receipt: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    role_file_sha256s: Mapping[str, str],
    bundle_sha256: str,
) -> None:
    manifest_binding = receipt["manifest"]
    role_bindings = receipt["role_files"]
    assert isinstance(manifest_binding, Mapping)
    assert isinstance(role_bindings, Mapping)
    if (
        manifest_binding["file_sha256"] != manifest_file_sha256
        or manifest_binding["manifest_sha256"] != manifest["manifest_sha256"]
        or receipt["bundle_sha256"] != bundle_sha256
        or receipt["D3_bootstrap_strata"] != manifest["D3_bootstrap_strata"]
        or receipt["overlap_audit"]["manifest_overlap_audit_sha256"]
        != _domain_hash(_RECEIPT_DOMAIN, manifest["overlap_audit"])
    ):
        raise FreshPanelIntegrityError("receipt/manifest binding drifted")
    roles = manifest["roles"]
    assert isinstance(roles, Mapping)
    for role in ORDERED_ROLES:
        role_binding = role_bindings[role]
        assert isinstance(role_binding, Mapping)
        descriptor = roles[role]
        assert isinstance(descriptor, Mapping)
        if (
            role_binding["file_sha256"] != role_file_sha256s[role]
            or descriptor["file_sha256"] != role_file_sha256s[role]
            or role_binding["example_count"] != descriptor["example_count"]
            or role_binding["predicted_token_count"]
            != descriptor["predicted_token_count"]
        ):
            raise FreshPanelIntegrityError(f"receipt/{role} binding drifted")


def validate_open_role_corpus(
    value: Mapping[str, object],
    *,
    role: str,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate one authorized executable role against the frozen manifest."""

    if role == RESERVE_ROLE:
        raise FreshPanelFirewallError("reserve_v3_unopened may never be decoded")
    if role not in EXECUTABLE_ROLES:
        raise FreshPanelFirewallError(f"unknown executable role {role!r}")
    frozen = validate_manifest(manifest)
    corpus = _strict_mapping(value, label=f"opened {role} corpus")
    schedule = _role_schedule(role)
    expected_count = len(FAMILY_IDS) * len(schedule)
    if (
        corpus.get("schema") != ROLE_CORPUS_SCHEMA
        or corpus.get("format_version") != FORMAT_VERSION
        or corpus.get("panel_id") != PANEL_ID
        or corpus.get("role") != role
        or tuple(corpus.get("family_ids", ())) != FAMILY_IDS
        or tuple(corpus.get("family_semantics", ())) != FAMILY_SEMANTICS
        or tuple(corpus.get("target_token_counts_per_family", ())) != schedule
        or corpus.get("example_count") != expected_count
    ):
        raise FreshPanelIntegrityError(f"{role} raw corpus header drifted")
    examples = corpus.get("examples")
    if not isinstance(examples, list) or len(examples) != expected_count:
        raise FreshPanelIntegrityError(f"{role} raw corpus size drifted")
    descriptor = frozen["roles"][role]  # type: ignore[index]
    members = descriptor["ordered_members"]
    for index, (example, member) in enumerate(zip(examples, members)):
        if not isinstance(example, Mapping) or not isinstance(member, Mapping):
            raise FreshPanelIntegrityError(f"{role} raw member is invalid")
        prompt = example.get("prompt")
        template_id = example.get("template_id")
        lineage_id = example.get("semantic_lineage_id")
        if not all(
            isinstance(item, str) and item
            for item in (prompt, template_id, lineage_id)
        ):
            raise FreshPanelIntegrityError(f"{role} raw identity is invalid")
        expected_member = _member_from_example(example)
        if expected_member != member:
            raise FreshPanelIntegrityError(
                f"{role} raw member differs from manifest at {index}"
            )
        expected_hashes = {
            "prompt_sha256": _text_hash(_PROMPT_DOMAIN, prompt),
            "canonical_prompt_sha256": _text_hash(
                _CANONICAL_DOMAIN, _canonical_prompt(prompt)
            ),
            "template_sha256": _text_hash(_TEMPLATE_DOMAIN, template_id),
            "semantic_lineage_sha256": _text_hash(_LINEAGE_DOMAIN, lineage_id),
        }
        if any(example.get(key) != value for key, value in expected_hashes.items()):
            raise FreshPanelIntegrityError(f"{role} raw identity hash drifted")
        if example.get("lineage_parent_sha256s") != []:
            raise FreshPanelIntegrityError(f"{role} lineage has a parent")
    return corpus


@dataclass(frozen=True, slots=True)
class RoleTokenizationReceipt:
    role: str
    example_count: int
    predicted_token_count: int
    ordered_example_tokenization_sha256: str


def authenticate_open_role_tokenization(
    corpus: Mapping[str, object],
    *,
    role: str,
    manifest: Mapping[str, object],
    tokenizer: object,
) -> RoleTokenizationReceipt:
    """Replay exact token counts and hashes after an authorized semantic open."""

    opened = validate_open_role_corpus(corpus, role=role, manifest=manifest)
    records: list[dict[str, object]] = []
    total_rows = 0
    for example in opened["examples"]:  # type: ignore[index]
        assert isinstance(example, Mapping)
        prompt = example["prompt"]
        assert isinstance(prompt, str)
        ids = _token_ids(tokenizer, prompt)
        if (
            len(ids) != example["target_token_count"]
            or _token_ids_sha256(ids) != example["token_ids_sha256"]
        ):
            raise FreshPanelIntegrityError(
                f"{role} tokenizer replay drifted for {example['example_id']}"
            )
        rows = len(ids) - 1
        total_rows += rows
        records.append(
            {
                "example_id": example["example_id"],
                "token_ids_sha256": example["token_ids_sha256"],
                "predicted_token_count": rows,
            }
        )
    return RoleTokenizationReceipt(
        role=role,
        example_count=len(records),
        predicted_token_count=total_rows,
        ordered_example_tokenization_sha256=_domain_hash(
            _TOKEN_IDS_DOMAIN, records
        ),
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class OpaquePanelPreflightReceipt:
    """Original-live bytes-only receipt; raw role JSON was not decoded."""

    manifest: Mapping[str, object]
    manifest_file_sha256: str
    receipt_file_sha256: str
    receipt_sha256: str
    bundle_sha256: str
    role_file_sha256_items: tuple[tuple[str, str], ...]
    reserve_semantically_decoded: bool = False

    @property
    def role_file_sha256s(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.role_file_sha256_items))


_LIVE_PREFLIGHTS: dict[int, weakref.ReferenceType[OpaquePanelPreflightReceipt]] = {}


def _grant_live_preflight(value: OpaquePanelPreflightReceipt) -> None:
    identity = id(value)

    def cleanup(_reference: object) -> None:
        _LIVE_PREFLIGHTS.pop(identity, None)

    _LIVE_PREFLIGHTS[identity] = weakref.ref(value, cleanup)


def _is_live_preflight(value: object) -> bool:
    reference = _LIVE_PREFLIGHTS.get(id(value))
    return reference is not None and reference() is value


def opaque_preflight(
    *,
    manifest_bytes: bytes,
    receipt_bytes: bytes,
    role_corpus_bytes: Mapping[str, bytes],
    expected_manifest_file_sha256: str,
    expected_receipt_file_sha256: str,
    expected_role_file_sha256s: Mapping[str, str],
) -> OpaquePanelPreflightReceipt:
    """Authenticate explicit future pins without decoding any raw corpus."""

    expected_receipt = _require_sha256(
        expected_receipt_file_sha256,
        label="expected receipt file",
    )
    if _sha_bytes(receipt_bytes) != expected_receipt:
        raise FreshPanelIntegrityError("receipt bytes differ from future pin")
    publication_receipt = validate_receipt(
        _decode_canonical_mapping(receipt_bytes, label="prompt-free receipt")
    )
    expected_manifest = _require_sha256(
        expected_manifest_file_sha256,
        label="expected manifest file",
    )
    if _sha_bytes(manifest_bytes) != expected_manifest:
        raise FreshPanelIntegrityError("manifest bytes differ from future pin")
    manifest = validate_manifest(
        _decode_canonical_mapping(manifest_bytes, label="prompt-free manifest")
    )
    if tuple(role_corpus_bytes) != ORDERED_ROLES:
        raise FreshPanelIntegrityError("raw role byte inventory or order drifted")
    if tuple(expected_role_file_sha256s) != ORDERED_ROLES:
        raise FreshPanelIntegrityError("expected role pin inventory drifted")
    observed: list[tuple[str, str]] = []
    for role in ORDERED_ROLES:
        expected = _require_sha256(
            expected_role_file_sha256s[role],
            label=f"expected {role} file",
        )
        encoded = role_corpus_bytes[role]
        if not isinstance(encoded, bytes):
            raise FreshPanelIntegrityError(f"{role} corpus must be raw bytes")
        digest = _sha_bytes(encoded)
        descriptor = manifest["roles"][role]  # type: ignore[index]
        if digest != expected or digest != descriptor["file_sha256"]:
            raise FreshPanelIntegrityError(f"{role} corpus bytes drifted")
        observed.append((role, digest))
    bundle_digest = _domain_hash(
        _BUNDLE_DOMAIN,
        {
            "manifest_file_sha256": expected_manifest,
            "role_file_sha256s": dict(observed),
        },
    )
    _validate_receipt_bindings(
        publication_receipt,
        manifest=manifest,
        manifest_file_sha256=expected_manifest,
        role_file_sha256s=dict(observed),
        bundle_sha256=bundle_digest,
    )
    receipt = OpaquePanelPreflightReceipt(
        manifest=MappingProxyType(manifest),
        manifest_file_sha256=expected_manifest,
        receipt_file_sha256=expected_receipt,
        receipt_sha256=str(publication_receipt["receipt_sha256"]),
        bundle_sha256=bundle_digest,
        role_file_sha256_items=tuple(observed),
        reserve_semantically_decoded=False,
    )
    _grant_live_preflight(receipt)
    return receipt


_SOURCE_CONSTRUCTION_TOKEN = object()


class SequentialPanelSource:
    """One-shot B3 -> C3 -> D3 semantic-open state machine."""

    def __init__(
        self,
        *,
        preflight: OpaquePanelPreflightReceipt,
        role_corpus_bytes: Mapping[str, bytes],
        selection_receipt_sha256: str,
        selected_layer_count: int,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _SOURCE_CONSTRUCTION_TOKEN:
            raise FreshPanelFirewallError(
                "use from_frozen_a_selection after opaque preflight"
            )
        if not _is_live_preflight(preflight):
            raise FreshPanelFirewallError(
                "an original live opaque-preflight receipt is required"
            )
        if selected_layer_count != LAYER_COUNT:
            raise FreshPanelFirewallError(
                "B3 requires exactly 18 frozen A layer selections"
            )
        self._preflight = preflight
        self._role_corpus_items = tuple(role_corpus_bytes.items())
        if tuple(role for role, _ in self._role_corpus_items) != ORDERED_ROLES:
            raise FreshPanelFirewallError("role byte order drifted")
        for role, encoded in self._role_corpus_items:
            if _sha_bytes(encoded) != preflight.role_file_sha256s[role]:
                raise FreshPanelFirewallError(f"{role} bytes differ from preflight")
        self._selection_receipt_sha256 = _require_sha256(
            selection_receipt_sha256,
            label="frozen A selection receipt",
        )
        self._next_index = 0
        self._current_role: str | None = None
        self._opened_roles: list[str] = []
        self._freeze_receipts: dict[str, dict[str, object]] = {}
        self._terminal_state = "ready_B3"

    @classmethod
    def from_frozen_a_selection(
        cls,
        preflight: OpaquePanelPreflightReceipt,
        role_corpus_bytes: Mapping[str, bytes],
        *,
        selection_receipt_sha256: str,
        selected_layer_count: int,
    ) -> "SequentialPanelSource":
        return cls(
            preflight=preflight,
            role_corpus_bytes=role_corpus_bytes,
            selection_receipt_sha256=selection_receipt_sha256,
            selected_layer_count=selected_layer_count,
            _construction_token=_SOURCE_CONSTRUCTION_TOKEN,
        )

    def __copy__(self) -> object:
        raise TypeError("SequentialPanelSource is one-shot and non-copyable")

    def __deepcopy__(self, _memo: object) -> object:
        raise TypeError("SequentialPanelSource is one-shot and non-copyable")

    @property
    def manifest(self) -> Mapping[str, object]:
        return self._preflight.manifest

    @property
    def panel_manifest_sha256(self) -> str:
        return str(self._preflight.manifest["manifest_sha256"])

    @property
    def selection_receipt_sha256(self) -> str:
        return self._selection_receipt_sha256

    @property
    def next_role(self) -> str | None:
        if self._terminal_state.startswith("terminal_") or self._next_index >= 3:
            return None
        return EXECUTABLE_ROLES[self._next_index]

    @property
    def current_role(self) -> str | None:
        return self._current_role

    @property
    def opened_roles(self) -> tuple[str, ...]:
        return tuple(self._opened_roles)

    @property
    def reserve_open_count(self) -> int:
        return 0

    @property
    def terminal_state(self) -> str:
        return self._terminal_state

    @property
    def freeze_receipts(self) -> Mapping[str, Mapping[str, object]]:
        return MappingProxyType(
            {
                role: MappingProxyType(dict(receipt))
                for role, receipt in self._freeze_receipts.items()
            }
        )

    @property
    def required_freeze_stage_kind(self) -> str | None:
        if self._current_role is None:
            return None
        return REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[self._current_role]

    def open_role(self, role: str) -> dict[str, object]:
        if role == RESERVE_ROLE:
            raise FreshPanelFirewallError(
                "reserve_v3_unopened has no semantic-open capability"
            )
        if role not in EXECUTABLE_ROLES:
            raise FreshPanelFirewallError(f"unknown executable role {role!r}")
        if self._terminal_state.startswith("terminal_"):
            raise FreshPanelFirewallError(
                f"panel source is terminal: {self._terminal_state}"
            )
        if self._current_role is not None:
            raise FreshPanelFirewallError(
                f"{self._current_role} must be frozen before another role opens"
            )
        expected = self.next_role
        if role != expected:
            raise FreshPanelFirewallError(
                f"expected semantic role {expected}, not {role}"
            )
        if role in self._opened_roles:
            raise FreshPanelFirewallError(f"{role} may not be reopened")
        encoded = dict(self._role_corpus_items)[role]
        if _sha_bytes(encoded) != self._preflight.role_file_sha256s[role]:
            raise FreshPanelIntegrityError(f"{role} corpus bytes drifted")
        raw = _decode_canonical_mapping(encoded, label=f"authorized {role} corpus")
        opened = validate_open_role_corpus(
            raw,
            role=role,
            manifest=self._preflight.manifest,
        )
        self._opened_roles.append(role)
        self._current_role = role
        self._terminal_state = f"open_{role.split('_', 1)[0]}"
        return opened

    def freeze_current(
        self,
        stage_kind: str,
        stage_receipt_sha256: str,
        *,
        terminal_passed: bool,
    ) -> None:
        if self._current_role is None:
            raise FreshPanelFirewallError("no semantic role is open")
        if type(terminal_passed) is not bool:
            raise FreshPanelFirewallError("terminal_passed must be a bool")
        role = self._current_role
        expected = REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[role]
        if stage_kind != expected:
            raise FreshPanelFirewallError(
                f"{role} requires freeze stage kind {expected!r}, not "
                f"{stage_kind!r}"
            )
        digest = _require_sha256(
            stage_receipt_sha256,
            label=f"{role} terminal receipt",
        )
        self._freeze_receipts[role] = {
            "stage_kind": stage_kind,
            "stage_receipt_sha256": digest,
            "terminal_passed": terminal_passed,
        }
        self._current_role = None
        self._next_index += 1
        short = role.split("_", 1)[0]
        if role in (B3_ROLE, C3_ROLE) and not terminal_passed:
            self._terminal_state = f"terminal_{short}_failed"
        elif role == D3_ROLE:
            outcome = "passed" if terminal_passed else "failed"
            self._terminal_state = f"terminal_D3_{outcome}"
        else:
            next_short = EXECUTABLE_ROLES[self._next_index].split("_", 1)[0]
            self._terminal_state = f"ready_{next_short}"

    def open_reserve(self) -> None:
        raise FreshPanelFirewallError(
            "reserve_v3_unopened is byte-pinned but may never be decoded"
        )


@dataclass(frozen=True, slots=True)
class PublishedFreshPanelFiles:
    repository_root: Path
    manifest_file_sha256: str
    receipt_file_sha256: str
    role_file_sha256_items: tuple[tuple[str, str], ...]
    bundle_sha256: str


def _safe_relative_path(value: Path, *, label: str) -> Path:
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise FreshPanelIntegrityError(f"{label} must be a safe relative path")
    return value


def _require_regular_directory(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    anchor = Path(absolute.anchor)
    current = anchor
    for component in absolute.parts[1:]:
        current = current / component
        try:
            status = os.lstat(current)
        except FileNotFoundError as exc:
            raise FreshPanelIntegrityError(
                f"publication parent must already exist: {current}"
            ) from exc
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise FreshPanelIntegrityError(
                f"publication parent is not a regular directory: {current}"
            )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short write while publishing fresh panels")
        written += count


def _publish_one_no_clobber(
    target: Path,
    payload: bytes,
    *,
    created: list[tuple[Path, tuple[int, int]]],
) -> None:
    _require_regular_directory(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    primary_error: BaseException | None = None
    temporary_owned = True
    identity: tuple[int, int] | None = None
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        temporary_status = os.fstat(descriptor)
        identity = (temporary_status.st_dev, temporary_status.st_ino)
        # Register before the atomic link.  A failed link is harmless because
        # rollback identity-checks and observes no owned target; any failure
        # after a successful link is already inside the rollback inventory.
        created.append((target, identity))
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, target, follow_symlinks=False)
        target_status = os.lstat(target)
        if (
            not stat.S_ISREG(target_status.st_mode)
            or (target_status.st_dev, target_status.st_ino) != identity
        ):
            raise FreshPanelIntegrityError(
                f"published target identity drifted: {target}"
            )
        temporary_status = os.lstat(temporary)
        if (
            not stat.S_ISREG(temporary_status.st_mode)
            or (temporary_status.st_dev, temporary_status.st_ino) != identity
        ):
            raise FreshPanelIntegrityError(
                f"publication temporary identity drifted: {temporary}"
            )
        temporary.unlink()
        temporary_owned = False
        _fsync_directory(target.parent)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors: list[tuple[str, BaseException]] = []
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as error:
                cleanup_errors.append(("temporary descriptor close", error))
        temporary_removed = False
        if temporary_owned and identity is None:
            cleanup_errors.append(
                (
                    "temporary file identity recovery",
                    FreshPanelIntegrityError(
                        "publication temporary identity was not established"
                    ),
                )
            )
        elif temporary_owned:
            try:
                temporary_status = os.lstat(temporary)
            except FileNotFoundError:
                temporary_owned = False
            except BaseException as error:
                cleanup_errors.append(("temporary file lstat", error))
            else:
                if (
                    not stat.S_ISREG(temporary_status.st_mode)
                    or (
                        temporary_status.st_dev,
                        temporary_status.st_ino,
                    )
                    != identity
                ):
                    cleanup_errors.append(
                        (
                            "temporary file identity check",
                            FreshPanelIntegrityError(
                                "refusing to unlink a replacement publication "
                                f"temporary: {temporary}"
                            ),
                        )
                    )
                else:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        temporary_owned = False
                    except BaseException as error:
                        cleanup_errors.append(("temporary file unlink", error))
                    else:
                        temporary_owned = False
                        temporary_removed = True
        if temporary_removed:
            try:
                _fsync_directory(target.parent)
            except BaseException as error:
                cleanup_errors.append(("temporary parent fsync", error))
        if primary_error is not None:
            for label, error in cleanup_errors:
                primary_error.add_note(
                    f"fresh-panel {label} also failed: "
                    f"{type(error).__name__}: {error}"
                )
        elif cleanup_errors:
            cleanup_primary = cleanup_errors[0][1]
            for label, error in cleanup_errors[1:]:
                cleanup_primary.add_note(
                    f"fresh-panel {label} also failed: "
                    f"{type(error).__name__}: {error}"
                )
            raise cleanup_primary


def _rollback_created(
    created: Sequence[tuple[Path, tuple[int, int]]],
) -> tuple[str, ...]:
    touched_directories: set[Path] = set()
    cleanup_failures: list[str] = []
    for target, identity in reversed(created):
        try:
            status = os.lstat(target)
        except FileNotFoundError:
            continue
        except BaseException as exc:
            cleanup_failures.append(
                f"lstat {target}: {type(exc).__name__}: {exc}"
            )
            continue
        if (
            stat.S_ISREG(status.st_mode)
            and (status.st_dev, status.st_ino) == identity
        ):
            try:
                target.unlink()
            except BaseException as exc:
                cleanup_failures.append(
                    f"unlink {target}: {type(exc).__name__}: {exc}"
                )
            else:
                touched_directories.add(target.parent)
        else:
            cleanup_failures.append(
                f"identity mismatch prevented rollback of {target}"
            )
    for directory in sorted(touched_directories, key=str):
        try:
            _fsync_directory(directory)
        except BaseException as exc:
            cleanup_failures.append(
                f"fsync {directory}: {type(exc).__name__}: {exc}"
            )
    return tuple(cleanup_failures)


def publish_fresh_panel_bundle(
    bundle: AuthoredFreshPanelBundle,
    *,
    repository_root: Path | str = Path("."),
    _fault_after_publish_count: int | None = None,
) -> PublishedFreshPanelFiles:
    """Durably publish all files no-clobber, with the receipt committed last."""

    if type(bundle) is not AuthoredFreshPanelBundle:
        raise TypeError("bundle must be an exact AuthoredFreshPanelBundle")
    if _fault_after_publish_count is not None and (
        type(_fault_after_publish_count) is not int
        or _fault_after_publish_count < 1
    ):
        raise ValueError("fault count must be a positive integer")
    root = Path(repository_root)
    _require_regular_directory(root)
    ordered_payloads = [
        (
            root / _safe_relative_path(DEFAULT_ROLE_CORPORA[role], label=role),
            bundle.role_corpus_bytes[role],
        )
        for role in ORDERED_ROLES
    ]
    ordered_payloads.extend(
        (
            (
                root
                / _safe_relative_path(DEFAULT_MANIFEST, label="manifest"),
                bundle.manifest_bytes,
            ),
            (
                root / _safe_relative_path(DEFAULT_RECEIPT, label="receipt"),
                bundle.receipt_bytes,
            ),
        )
    )
    for parent in sorted(
        {target.parent for target, _payload in ordered_payloads},
        key=str,
    ):
        _require_regular_directory(parent)
    for target, _payload in ordered_payloads:
        try:
            os.lstat(target)
        except FileNotFoundError:
            continue
        raise FileExistsError(f"refusing to clobber existing panel file: {target}")

    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        for publish_count, (target, payload) in enumerate(
            ordered_payloads,
            start=1,
        ):
            _publish_one_no_clobber(target, payload, created=created)
            if _fault_after_publish_count == publish_count:
                raise FreshPanelIntegrityError("injected publication fault")
    except BaseException as primary:
        for cleanup_failure in _rollback_created(created):
            primary.add_note(f"fresh-panel rollback failure: {cleanup_failure}")
        raise
    return PublishedFreshPanelFiles(
        repository_root=root,
        manifest_file_sha256=bundle.manifest_file_sha256,
        receipt_file_sha256=bundle.receipt_file_sha256,
        role_file_sha256_items=tuple(bundle.role_file_sha256s.items()),
        bundle_sha256=bundle.bundle_sha256,
    )


def _read_regular(path: Path, *, byte_cap: int = _MAX_JSON_BYTES) -> bytes:
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise FreshPanelIntegrityError(
            f"required panel file is missing: {path}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FreshPanelIntegrityError(f"panel path is not a regular file: {path}")
    if before.st_size > byte_cap:
        raise FreshPanelIntegrityError(f"panel file exceeds byte cap: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise FreshPanelIntegrityError(
                f"panel file identity changed during open: {path}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, byte_cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > byte_cap:
                raise FreshPanelIntegrityError(
                    f"panel file exceeds byte cap while reading: {path}"
                )
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or total != opened.st_size
        ):
            raise FreshPanelIntegrityError(
                f"panel file changed during read: {path}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_frozen_preflight(
    *,
    repository_root: Path | str = Path("."),
    expected_receipt_file_sha256: str,
) -> OpaquePanelPreflightReceipt:
    """Strictly load a published bundle from one externally frozen root pin."""

    root = Path(repository_root)
    receipt_bytes = _read_regular(root / DEFAULT_RECEIPT)
    expected_receipt = _require_sha256(
        expected_receipt_file_sha256,
        label="externally frozen receipt file",
    )
    if _sha_bytes(receipt_bytes) != expected_receipt:
        raise FreshPanelIntegrityError("published receipt differs from root pin")
    receipt = validate_receipt(
        _decode_canonical_mapping(receipt_bytes, label="published panel receipt")
    )
    manifest_binding = receipt["manifest"]
    role_bindings = receipt["role_files"]
    assert isinstance(manifest_binding, Mapping)
    assert isinstance(role_bindings, Mapping)
    manifest_bytes = _read_regular(root / DEFAULT_MANIFEST)
    role_bytes = {
        role: _read_regular(root / DEFAULT_ROLE_CORPORA[role])
        for role in ORDERED_ROLES
    }
    return opaque_preflight(
        manifest_bytes=manifest_bytes,
        receipt_bytes=receipt_bytes,
        role_corpus_bytes=role_bytes,
        expected_manifest_file_sha256=str(manifest_binding["file_sha256"]),
        expected_receipt_file_sha256=expected_receipt,
        expected_role_file_sha256s={
            role: str(role_bindings[role]["file_sha256"])
            for role in ORDERED_ROLES
        },
    )


def load_frozen_panel_source(
    *,
    repository_root: Path | str = Path("."),
    expected_receipt_file_sha256: str,
    selection_receipt_sha256: str,
    selected_layer_count: int,
) -> SequentialPanelSource:
    """Strictly load and bind a source without semantically opening any role."""

    root = Path(repository_root)
    preflight = load_frozen_preflight(
        repository_root=root,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
    )
    role_bytes = {
        role: _read_regular(root / DEFAULT_ROLE_CORPORA[role])
        for role in ORDERED_ROLES
    }
    return SequentialPanelSource.from_frozen_a_selection(
        preflight,
        role_bytes,
        selection_receipt_sha256=selection_receipt_sha256,
        selected_layer_count=selected_layer_count,
    )


def check_frozen_panels(
    *,
    repository_root: Path | str = Path("."),
    expected_receipt_file_sha256: str,
) -> OpaquePanelPreflightReceipt:
    """Read-only bytes-only verification; reserve remains semantically unopened."""

    return load_frozen_preflight(
        repository_root=repository_root,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
    )


def _load_prior_manifest(path: Path, *, generation: str) -> dict[str, object]:
    encoded = _read_regular(path)
    expected_file = PINNED_PRIOR_MANIFEST_FILE_SHA256S[generation]
    if _sha_bytes(encoded) != expected_file:
        raise FreshPanelIntegrityError(
            f"{generation} prior manifest differs from its frozen file pin"
        )
    value = _strict_json_loads(encoded, label=f"prior manifest {path}")
    if not isinstance(value, dict):
        raise FreshPanelIntegrityError("prior manifest must be a JSON object")
    return value


def _read_pinned_asset(path: Path, *, expected: str, label: str) -> bytes:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FreshPanelIntegrityError(f"{label} is missing: {path}") from exc
    encoded = _read_regular(resolved)
    if _sha_bytes(encoded) != expected:
        raise FreshPanelIntegrityError(f"{label} differs from its frozen pin")
    return encoded


def _build_from_cli(args: argparse.Namespace) -> AuthoredFreshPanelBundle:
    if not all(
        (
            args.tokenizer_json,
            args.tokenizer_config,
            args.special_tokens_map,
            args.v1_manifest,
            args.v2_manifest,
        )
    ):
        raise FreshPanelIntegrityError(
            "tokenizer JSON/config/special-token assets plus v1/v2 manifests "
            "are required"
        )
    tokenizer_json = _read_pinned_asset(
        Path(args.tokenizer_json),
        expected=str(TOKENIZER_CONTRACT["tokenizer_json_sha256"]),
        label="tokenizer JSON",
    )
    _read_pinned_asset(
        Path(args.tokenizer_config),
        expected=str(TOKENIZER_CONTRACT["tokenizer_config_sha256"]),
        label="tokenizer config",
    )
    _read_pinned_asset(
        Path(args.special_tokens_map),
        expected=str(TOKENIZER_CONTRACT["special_tokens_map_sha256"]),
        label="special-token map",
    )
    tokenizers = importlib.import_module("tokenizers")
    tokenizer_class = getattr(tokenizers, "Tokenizer", None)
    if tokenizer_class is None:
        raise FreshPanelIntegrityError("tokenizers.Tokenizer is unavailable")
    tokenizer = tokenizer_class.from_str(tokenizer_json.decode("utf-8"))
    catalogs = (
        prior_lineage_catalog_from_manifest(
            "v1",
            _load_prior_manifest(Path(args.v1_manifest), generation="v1"),
        ),
        prior_lineage_catalog_from_manifest(
            "v2",
            _load_prior_manifest(Path(args.v2_manifest), generation="v2"),
        ),
    )
    for catalog in catalogs:
        if (
            catalog.catalog_sha256
            != PINNED_PRIOR_LINEAGE_CATALOG_SHA256S[catalog.generation]
        ):
            raise FreshPanelIntegrityError(
                f"{catalog.generation} prior lineage catalog root drifted"
            )
    return author_fresh_panel_bundle(
        tokenizer,
        prior_lineage_catalogs=catalogs,
    )


def _summary(bundle: AuthoredFreshPanelBundle) -> dict[str, object]:
    return {
        "panel_id": PANEL_ID,
        "manifest_file_sha256": bundle.manifest_file_sha256,
        "receipt_file_sha256": bundle.receipt_file_sha256,
        "role_file_sha256s": dict(bundle.role_file_sha256s),
        "bundle_sha256": bundle.bundle_sha256,
        "canonical_files_written": False,
        "model_executed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--build", action="store_true")
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--check", action="store_true")
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--tokenizer-json")
    parser.add_argument("--tokenizer-config")
    parser.add_argument("--special-tokens-map")
    parser.add_argument("--v1-manifest")
    parser.add_argument("--v2-manifest")
    parser.add_argument("--expected-receipt-file-sha256")
    args = parser.parse_args(argv)

    if args.check:
        if not args.expected_receipt_file_sha256:
            parser.error("--check requires --expected-receipt-file-sha256")
        preflight = check_frozen_panels(
            repository_root=args.repository_root,
            expected_receipt_file_sha256=args.expected_receipt_file_sha256,
        )
        print(
            json.dumps(
                {
                    "panel_id": PANEL_ID,
                    "receipt_file_sha256": preflight.receipt_file_sha256,
                    "manifest_file_sha256": preflight.manifest_file_sha256,
                    "bundle_sha256": preflight.bundle_sha256,
                    "reserve_semantically_decoded": False,
                },
                sort_keys=True,
            )
        )
        return 0

    try:
        bundle = _build_from_cli(args)
    except FreshPanelIntegrityError as exc:
        parser.error(str(exc))
    summary = _summary(bundle)
    if args.write:
        published = publish_fresh_panel_bundle(
            bundle,
            repository_root=args.repository_root,
        )
        summary["canonical_files_written"] = True
        summary["receipt_file_sha256"] = published.receipt_file_sha256
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
