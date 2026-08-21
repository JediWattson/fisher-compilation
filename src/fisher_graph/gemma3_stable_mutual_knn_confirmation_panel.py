"""Authenticated fresh panel for one-pass stable mutual-kNN graph confirmation.

This module is deliberately model-free.  It validates a frozen prompt corpus,
keeps the only model-facing role separate from a never-touched reserve, audits
freshness against checked-in Gemma prompt sources, and authenticates exact
token lengths when a caller explicitly supplies the pinned tokenizer.

No function in this module imports Transformers or loads a model.  In
particular, :func:`confirmation_prompts` is the sole public text iterator for
the experiment role; reserve text is available only through the audit loader.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


PANEL_SCHEMA = (
    "fisher_graph.gemma3_stable_mutual_knn_confirmation_panel"
)
PANEL_FORMAT_VERSION = 1
CORPUS_SCHEMA = (
    "fisher_graph.gemma3_stable_mutual_knn_confirmation_corpus"
)
CORPUS_FORMAT_VERSION = 1
RECEIPT_SCHEMA = (
    "fisher_graph.gemma3_stable_mutual_knn_confirmation_prompt_audit"
)
RECEIPT_FORMAT_VERSION = 1

PANEL_ID = "gemma3_stable_mutual_knn_confirmation_v1"
DEFAULT_CORPUS = Path(
    "examples/gemma3_stable_mutual_knn_confirmation_corpus.json"
)
DEFAULT_PANEL = Path(
    "examples/gemma3_stable_mutual_knn_confirmation_panel.json"
)
DEFAULT_RECEIPT = Path(
    "artifacts/research/"
    "gemma3-stable-mutual-knn-confirmation-panel-audit-v1.json"
)
MODEL_ID = "google/gemma-3-270m"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
MODEL_CONFIG_SHA256 = (
    "8983343e5bfbf8e336d222a522a1d781443491e9de2d081b8856f8fd26ab6046"
)
TOKENIZER_CONFIG_SHA256 = (
    "94b03056ec5831e9021c3e3fe9db682778c4f2d081c4188f1fcb8b90541cf1cf"
)
TOKENIZER_JSON_SHA256 = (
    "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c"
)
SLIDING_WINDOW = 512
MAX_POSITION_EMBEDDINGS = 32768

CONFIRMATION_ROLE = "C_confirmation"
RESERVE_ROLE = "reserve"
ROLE_NAMES = (CONFIRMATION_ROLE, RESERVE_ROLE)
SEMANTIC_FAMILIES = (
    "observational_diagnosis",
    "quantitative_reasoning",
    "technical_explanation",
    "procedural_planning",
    "formal_logic",
    "causal_evidence_synthesis",
)
FORMAL_LOGIC_CHOICES = (
    " entailed",
    " contradicted",
    " both",
    " neither",
)
FORMAL_LOGIC_QUERY_SUFFIX = (
    "Map YN entailed; NY contradicted; YY both; NN neither. The claim is"
)
EXPECTED_ROLE_COUNTS = {
    CONFIRMATION_ROLE: 60,
    RESERVE_ROLE: 12,
}
BOUNDARY_LENGTHS = (511, 512)
SHORT_CONTROL_LENGTHS = (32, 64, 96)
TRANSITION_LENGTHS = (513, 514)
BEYOND_WINDOW_LENGTHS = (576, 640, 768)
EXACT_TARGET_LENGTHS = (
    *SHORT_CONTROL_LENGTHS,
    *BOUNDARY_LENGTHS,
    *TRANSITION_LENGTHS,
    *BEYOND_WINDOW_LENGTHS,
)
EXPECTED_PROBE_SCHEDULE = {
    family: EXACT_TARGET_LENGTHS
    for family in SEMANTIC_FAMILIES
}
EXPECTED_CONFIRMATION_CHOICE_COORDINATES = {
    "observational_diagnosis": (0, 1, 2, 0, 1, 3, 0, 1, 2, 3),
    "quantitative_reasoning": (0, 2, 3, 1, 2, 3, 1, 2, 3, 0),
    "technical_explanation": (1, 0, 2, 0, 3, 2, 2, 3, 0, 1),
    "procedural_planning": (1, 0, 3, 1, 3, 2, 3, 0, 1, 2),
    "formal_logic": (3, 1, 0, 3, 2, 0, 0, 1, 2, 3),
    "causal_evidence_synthesis": (2, 3, 1, 2, 0, 1, 1, 2, 3, 0),
}
EXPECTED_CHOICE_COORDINATE_CONTEXT_COUNTS = {
    "short": {0: 5, 1: 5, 2: 4, 3: 4},
    "boundary": {0: 6, 1: 6, 2: 6, 3: 6},
    "long": {0: 4, 1: 4, 2: 5, 3: 5},
}
EXPECTED_CONFIRMATION_BUCKETS = (
    ("short_control", "short_32", 32, 0, 6),
    ("short_control", "short_64", 64, 6, 6),
    ("short_control", "short_96", 96, 12, 6),
    ("boundary_probe", "boundary_511", 511, 18, 6),
    ("boundary_probe", "boundary_512", 512, 24, 6),
    ("transition_probe", "transition_513", 513, 30, 6),
    ("transition_probe", "transition_514", 514, 36, 6),
    ("beyond_window_probe", "outside_window_576", 576, 42, 6),
    ("beyond_window_probe", "outside_window_640", 640, 48, 6),
    ("beyond_window_probe", "outside_window_768", 768, 54, 6),
)
EXPECTED_RESERVE_BUCKETS = (("short_control", None, None, 0, 12),)
HARD_LONG_VISIBLE_SUFFIX_PREFIX = (
    " Neutral suffix block 1 records alpha, beta, gamma, delta, and "
    "epsilon as inert placeholders; it adds no task evidence."
)

BLIND_AUTHORING_RECEIPT = {
    "atlas_result_inputs": "absent",
    "candidate_edge_inputs": "absent",
    "distance_metric_inputs": "absent",
    "model_execution": "none",
    "tokenizer_access": (
        "C_confirmation_exact_prompt_choice_and_evidence_authentication_only"
    ),
    "reserve_tokenizer_access": "none",
    "authorship_timing": (
        "completed_before_any_C_confirmation_model_result"
    ),
}

_SCIENTIFIC_STATUS = (
    "frozen_candidate_single_pass_C_confirmation_"
    "reserve_parse_hash_audit_only"
)
_RECEIPT_DOMAIN = (
    b"fisher_graph.gemma3_stable_mutual_knn_confirmation_prompt_audit.v1\0"
)
_TOKEN_STREAM_DOMAIN = (
    b"fisher_graph.gemma3_stable_mutual_knn_confirmation_token_stream.v1\0"
)
_TEMPLATE_SIGNATURE_DOMAIN = (
    b"fisher_graph.gemma3_stable_mutual_knn_template_signature.v1\0"
)
_CHOICE_TOKEN_IDS_DOMAIN = (
    b"fisher_graph.gemma3_stable_mutual_knn_choice_token_ids.v1\0"
)
_VISIBLE_SUFFIX_TOKEN_IDS_DOMAIN = (
    b"fisher_graph.gemma3_stable_mutual_knn_visible_suffix_token_ids.v1\0"
)
_MAX_PANEL_FILE_BYTES = 4 * 1024 * 1024
_MAX_REFERENCE_FILE_BYTES = 16 * 1024 * 1024
_PINNED_DISCOVERY_PANEL_SHA256 = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "again",
        "against",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "case",
        "checking",
        "clause",
        "compares",
        "condition",
        "conclusion",
        "context",
        "did",
        "do",
        "does",
        "each",
        "entry",
        "evidence",
        "explains",
        "field",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "ledger",
        "marker",
        "may",
        "more",
        "most",
        "no",
        "not",
        "note",
        "observation",
        "observations",
        "of",
        "on",
        "one",
        "only",
        "or",
        "opening",
        "opens",
        "other",
        "record",
        "recorded",
        "records",
        "return",
        "review",
        "reviewing",
        "same",
        "section",
        "separates",
        "so",
        "some",
        "than",
        "that",
        "the",
        "timing",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "under",
        "until",
        "use",
        "used",
        "using",
        "vocabulary",
        "was",
        "were",
        "when",
        "where",
        "which",
        "while",
        "why",
        "will",
        "with",
        "without",
    }
)

PARAPHRASE_POLICY = {
    "content_token_minimum": 6,
    "content_set_containment_minimum": 0.80,
    "content_set_jaccard_minimum": 0.65,
    "word_bigram_jaccard_minimum": 0.40,
    "character_5gram_jaccard_minimum": 0.60,
}


class TokenizerLike(Protocol):
    """Narrow tokenizer surface needed for construction-time authentication."""

    name_or_path: str
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None

    def __call__(self, text: str, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class PromptRecord:
    role: str
    corpus_index: int
    semantic_family: str
    prompt_id: str
    template_id: str
    instance_id: str
    domain_template: str
    kind: str
    stratum_id: str | None
    text: str
    text_sha256: str
    canonical_sha256: str
    semantic_lineage_id: str
    paraphrase_lineage_id: str
    template_signature_sha256: str
    target_valid_tokens: int | None
    input_ids_sha256: str | None
    choices: tuple[str, str, str, str]
    correct_choice: int
    supports_correct_choice_index: int
    choices_sha256: str
    choice_token_ids: tuple[int, int, int, int] | None
    answer_rationale: str
    answer_rationale_sha256: str
    answer_evidence: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class LoadedPanel:
    corpus_path: Path
    manifest_path: Path
    corpus: Mapping[str, object]
    manifest: Mapping[str, object]
    records: tuple[PromptRecord, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _prompt_sha256(text: str) -> str:
    """Match the existing prompt-fixture one-item JSON authentication."""

    encoded = json.dumps(
        [text],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _ordered_prompt_sha256(values: Sequence[str]) -> str:
    return _json_sha256([_prompt_sha256(value) for value in values])


def _canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = _WORD_RE.findall(normalized)
    return " ".join(tokens)


def _canonical_text_sha256(value: str) -> str:
    return _sha256_bytes(_canonical_text(value).encode("utf-8"))


def _answer_specific_text(semantic_family: str, text: str) -> str:
    """Remove the symmetric logic taxonomy before answer-leak checks."""

    if (
        semantic_family == "formal_logic"
        and text.casefold().endswith(FORMAL_LOGIC_QUERY_SUFFIX.casefold())
    ):
        return text[: -len(FORMAL_LOGIC_QUERY_SUFFIX)]
    return text


def _paraphrase_audit_projection(text: str) -> str:
    """Remove authenticated neutral long padding before semantic overlap."""

    if HARD_LONG_VISIBLE_SUFFIX_PREFIX not in text:
        return text
    first_clause_end = text.find(".") + 1
    if first_clause_end <= 0:
        raise ValueError("hard-long prompt lacks a decisive first clause")
    return text[:first_clause_end]


def _template_signature_sha256(
    *,
    role: str,
    semantic_family: str,
    template_id: str,
    instance_id: str,
    domain_template: str,
) -> str:
    payload = json.dumps(
        [
            role,
            semantic_family,
            template_id,
            instance_id,
            domain_template,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(_TEMPLATE_SIGNATURE_DOMAIN + payload)


def _content_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _WORD_RE.findall(
            unicodedata.normalize("NFKC", value).casefold()
        )
        if token not in _STOPWORDS and len(token) >= 3
    )


def _ngrams(values: Sequence[str], width: int) -> frozenset[tuple[str, ...]]:
    if len(values) < width:
        return frozenset()
    return frozenset(
        tuple(values[index : index + width])
        for index in range(len(values) - width + 1)
    )


def _character_ngrams(value: str, width: int) -> frozenset[str]:
    canonical = _canonical_text(value)
    if len(canonical) < width:
        return frozenset()
    return frozenset(
        canonical[index : index + width]
        for index in range(len(canonical) - width + 1)
    )


def _jaccard(left: set[object] | frozenset[object], right: set[object] | frozenset[object]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _content_containment(left: set[str], right: set[str]) -> float:
    denominator = min(len(left), len(right))
    if denominator == 0:
        return 0.0
    return len(left & right) / denominator


def _looks_like_paraphrase(left: str, right: str) -> bool:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    minimum = int(PARAPHRASE_POLICY["content_token_minimum"])
    if min(len(left_tokens), len(right_tokens)) < minimum:
        return False
    left_content = set(left_tokens)
    right_content = set(right_tokens)
    containment = _content_containment(left_content, right_content)
    content_jaccard = _jaccard(left_content, right_content)
    if (
        containment
        >= float(PARAPHRASE_POLICY["content_set_containment_minimum"])
        and content_jaccard
        >= float(PARAPHRASE_POLICY["content_set_jaccard_minimum"])
    ):
        return True
    if _jaccard(
        _ngrams(left_tokens, 2),
        _ngrams(right_tokens, 2),
    ) >= float(PARAPHRASE_POLICY["word_bigram_jaccard_minimum"]):
        return True
    return _jaccard(
        _character_ngrams(left, 5),
        _character_ngrams(right, 5),
    ) >= float(PARAPHRASE_POLICY["character_5gram_jaccard_minimum"])


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _strict_json(
    path: Path,
    *,
    maximum_bytes: int = _MAX_PANEL_FILE_BYTES,
    require_canonical: bool = False,
) -> Mapping[str, object]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{path.name} must be a non-symlink regular file")
    if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
        raise ValueError(
            f"{path.name} size must be between 1 and {maximum_bytes} bytes"
        )
    encoded = path.read_bytes()
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} must be strict UTF-8") from error
    raw = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path.name} must contain a JSON object")
    if require_canonical and encoded != _canonical_json_bytes(raw):
        raise ValueError(f"{path.name} must use canonical JSON bytes")
    return raw


def _strict_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _strict_string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    values = tuple(
        _strict_string(item, label=f"{label} item")
        for item in value
    )
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicate text")
    return values


def _manifest_prompt_records(
    manifest: Mapping[str, object],
    role: str,
) -> tuple[Mapping[str, object], ...]:
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise TypeError("panel roles must be a mapping")
    role_manifest = roles.get(role)
    if not isinstance(role_manifest, Mapping):
        raise TypeError(f"panel role {role!r} must be a mapping")
    families = role_manifest.get("families")
    if not isinstance(families, list):
        raise TypeError(f"panel role {role!r} families must be a list")
    records: list[Mapping[str, object]] = []
    for family in families:
        if not isinstance(family, Mapping):
            raise TypeError("panel family entries must be mappings")
        prompts = family.get("prompts")
        if not isinstance(prompts, list):
            raise TypeError("panel family prompts must be a list")
        for prompt in prompts:
            if not isinstance(prompt, Mapping):
                raise TypeError("panel prompt entries must be mappings")
            records.append(prompt)
    return tuple(records)


def load_panel(
    corpus_path: Path | str,
    manifest_path: Path | str,
) -> LoadedPanel:
    """Strict-load and cross-authenticate the fresh two-role panel."""

    corpus_path = Path(corpus_path)
    manifest_path = Path(manifest_path)
    if (
        corpus_path.parent.resolve(strict=True)
        != manifest_path.parent.resolve(strict=True)
    ):
        raise ValueError("panel corpus and manifest must be sibling files")
    corpus = _strict_json(corpus_path, require_canonical=True)
    manifest = _strict_json(manifest_path, require_canonical=True)
    if set(corpus) != {
        "schema",
        "format_version",
        "scientific_status",
        *ROLE_NAMES,
    }:
        raise ValueError("confirmation corpus fields do not match format v1")
    if corpus["schema"] != CORPUS_SCHEMA:
        raise ValueError("unsupported confirmation corpus schema")
    if corpus["format_version"] != CORPUS_FORMAT_VERSION:
        raise ValueError("unsupported confirmation corpus format")
    if corpus["scientific_status"] != _SCIENTIFIC_STATUS:
        raise ValueError("confirmation corpus scientific status is invalid")

    expected_manifest_fields = {
        "schema",
        "format_version",
        "panel_id",
        "scientific_status",
        "model",
        "tokenizer",
        "prompt_corpus",
        "prompt_corpus_sha256",
        "role_policy",
        "semantic_family_policy",
        "freshness_policy",
        "probe_policy",
        "blind_authoring",
        "roles",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("confirmation panel fields do not match format v1")
    if manifest["schema"] != PANEL_SCHEMA:
        raise ValueError("unsupported confirmation panel schema")
    if manifest["format_version"] != PANEL_FORMAT_VERSION:
        raise ValueError("unsupported confirmation panel format")
    if manifest["panel_id"] != PANEL_ID:
        raise ValueError("confirmation panel identifier is invalid")
    if manifest["scientific_status"] != _SCIENTIFIC_STATUS:
        raise ValueError("confirmation panel scientific status is invalid")
    if manifest["prompt_corpus"] != corpus_path.name:
        raise ValueError("confirmation panel points to the wrong corpus")
    if (
        Path(str(manifest["prompt_corpus"])).name
        != manifest["prompt_corpus"]
    ):
        raise ValueError("confirmation panel corpus path traversal is forbidden")
    if manifest["prompt_corpus_sha256"] != _file_sha256(corpus_path):
        raise ValueError("confirmation corpus hash does not authenticate")

    model = manifest["model"]
    if model != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "config_sha256": MODEL_CONFIG_SHA256,
        "model_type": "gemma3_text",
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "sliding_window": SLIDING_WINDOW,
    }:
        raise ValueError("pinned Gemma model configuration is invalid")
    tokenizer = manifest["tokenizer"]
    if tokenizer != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "class": "GemmaTokenizer",
        "config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "add_special_tokens": True,
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "length_measure": "sum_attention_mask_after_add_special_tokens",
    }:
        raise ValueError("pinned Gemma tokenizer configuration is invalid")
    if manifest["role_policy"] != {
        CONFIRMATION_ROLE: (
            "one_frozen_candidate_graph_pass_no_fit_no_selection_no_tuning"
        ),
        RESERVE_ROLE: (
            "parse_hash_overlap_audit_only_no_tokenizer_no_model_access"
        ),
    }:
        raise ValueError("confirmation role policy is invalid")
    if manifest["semantic_family_policy"] != (
        "six_matched_broad_families_equal_counts_role_disjoint_domains_"
        "globally_unique_templates_and_instances"
    ):
        raise ValueError("semantic family policy is invalid")
    if manifest["freshness_policy"] != {
        "canonicalization": "unicode_nfkc_casefold_ascii_alnum_token_join",
        "exact_text_authentication": "one_item_json_sha256",
        "paraphrase_audit": (
            "model_free_task_projection_content_word_bigram_and_"
            "character_5gram_thresholds"
        ),
        "semantic_lineage": "globally_unique_authored_lineage_per_row",
        "template_signature": (
            "domain_separated_sha256_over_role_family_template_instance_domain"
        ),
        "reference_scope": (
            "all_checked_in_gemma3_prompt_sources_including_pinned_"
            "downstream_panels_and_family_manifests"
        ),
    }:
        raise ValueError("freshness policy is invalid")
    if manifest["blind_authoring"] != BLIND_AUTHORING_RECEIPT:
        raise ValueError("blind-authoring receipt is invalid")

    probe_policy = manifest["probe_policy"]
    if not isinstance(probe_policy, Mapping):
        raise TypeError("probe policy must be a mapping")
    if set(probe_policy) != {
        "sliding_window",
        "short_control_lengths",
        "boundary_lengths",
        "transition_lengths",
        "beyond_window_lengths",
        "schedule",
        "length_bucket_batching",
        "answer_evidence_policy",
        "truncation_policy",
    }:
        raise ValueError("probe policy fields do not match format v1")
    if probe_policy["sliding_window"] != SLIDING_WINDOW:
        raise ValueError("probe sliding window is invalid")
    if tuple(probe_policy["short_control_lengths"]) != SHORT_CONTROL_LENGTHS:
        raise ValueError("short-control lengths are invalid")
    if tuple(probe_policy["boundary_lengths"]) != BOUNDARY_LENGTHS:
        raise ValueError("boundary probe lengths are invalid")
    if tuple(probe_policy["transition_lengths"]) != TRANSITION_LENGTHS:
        raise ValueError("transition probe lengths are invalid")
    if tuple(probe_policy["beyond_window_lengths"]) != BEYOND_WINDOW_LENGTHS:
        raise ValueError("beyond-window probe lengths are invalid")
    if probe_policy["truncation_policy"] != (
        "forbidden_exact_authenticated_lengths_must_be_preserved"
    ):
        raise ValueError("probe truncation policy is invalid")
    if probe_policy["length_bucket_batching"] != (
        "restart_batching_at_each_exact_length_bucket"
    ):
        raise ValueError("length-bucket batching policy is invalid")
    if probe_policy["answer_evidence_policy"] != (
        "hard_long_rows_bind_full_minimal_sufficient_decisive_clause_"
        "ending_before_the_final_query_sliding_window"
    ):
        raise ValueError("answer-evidence policy is invalid")
    schedule = probe_policy["schedule"]
    if not isinstance(schedule, Mapping) or set(schedule) != set(
        SEMANTIC_FAMILIES
    ):
        raise ValueError("probe schedule families are invalid")
    for family, expected_lengths in EXPECTED_PROBE_SCHEDULE.items():
        if tuple(schedule[family]) != expected_lengths:
            raise ValueError(f"probe schedule for {family!r} is invalid")

    corpus_values = {
        role: _strict_string_list(corpus[role], label=role)
        for role in ROLE_NAMES
    }
    all_text = tuple(
        text for role in ROLE_NAMES for text in corpus_values[role]
    )
    if len(all_text) != len(set(all_text)):
        raise ValueError("confirmation corpus roles must be text-disjoint")
    if len({_canonical_text(text) for text in all_text}) != len(all_text):
        raise ValueError("confirmation corpus roles must be canonical-disjoint")

    roles = manifest["roles"]
    if not isinstance(roles, Mapping) or set(roles) != set(ROLE_NAMES):
        raise ValueError("confirmation panel roles are invalid")
    records: list[PromptRecord] = []
    seen_prompt_ids: set[str] = set()
    seen_template_ids: set[str] = set()
    seen_instance_ids: set[str] = set()
    seen_domains: set[str] = set()
    seen_semantic_lineages: set[str] = set()
    seen_paraphrase_lineages: set[str] = set()
    seen_template_signatures: set[str] = set()
    for role in ROLE_NAMES:
        role_manifest = roles[role]
        if not isinstance(role_manifest, Mapping) or set(role_manifest) != {
            "count",
            "normalized_sha256",
            "canonical_order",
            "length_buckets",
            "families",
        }:
            raise ValueError(f"panel role {role!r} fields are invalid")
        values = corpus_values[role]
        if role_manifest["count"] != EXPECTED_ROLE_COUNTS[role]:
            raise ValueError(f"panel role {role!r} count is invalid")
        if len(values) != EXPECTED_ROLE_COUNTS[role]:
            raise ValueError(f"corpus role {role!r} count is invalid")
        if role_manifest["normalized_sha256"] != _ordered_prompt_sha256(
            values
        ):
            raise ValueError(f"panel role {role!r} hash is invalid")
        if role_manifest["canonical_order"] != (
            "length_bucket_then_short_round_or_semantic_family"
        ):
            raise ValueError(f"panel role {role!r} order policy is invalid")
        length_buckets = role_manifest["length_buckets"]
        if not isinstance(length_buckets, list):
            raise TypeError("panel length buckets must be a list")
        expected_buckets = (
            EXPECTED_CONFIRMATION_BUCKETS
            if role == CONFIRMATION_ROLE
            else EXPECTED_RESERVE_BUCKETS
        )
        observed_buckets: list[
            tuple[object, object, object, object, object]
        ] = []
        for bucket in length_buckets:
            if not isinstance(bucket, Mapping) or set(bucket) != {
                "kind",
                "stratum_id",
                "target_valid_tokens",
                "start",
                "count",
            }:
                raise ValueError("panel length bucket fields are invalid")
            observed_buckets.append(
                (
                    bucket["kind"],
                    bucket["stratum_id"],
                    bucket["target_valid_tokens"],
                    bucket["start"],
                    bucket["count"],
                )
            )
        if tuple(observed_buckets) != expected_buckets:
            raise ValueError(f"panel role {role!r} length buckets are invalid")
        families = role_manifest["families"]
        if not isinstance(families, list) or len(families) != len(
            SEMANTIC_FAMILIES
        ):
            raise ValueError(f"panel role {role!r} families are invalid")
        if tuple(
            family.get("semantic_family")
            if isinstance(family, Mapping)
            else None
            for family in families
        ) != SEMANTIC_FAMILIES:
            raise ValueError(f"panel role {role!r} family order is invalid")
        expected_family_count = 10 if role == CONFIRMATION_ROLE else 2
        role_records: list[PromptRecord] = []
        for family in families:
            if not isinstance(family, Mapping) or set(family) != {
                "id",
                "semantic_family",
                "description",
                "count",
                "prompts",
            }:
                raise ValueError("panel family fields are invalid")
            semantic_family = _strict_string(
                family["semantic_family"],
                label="semantic family",
            )
            if family["id"] != f"{role}.{semantic_family}":
                raise ValueError("panel family identifier is invalid")
            _strict_string(family["description"], label="family description")
            if family["count"] != expected_family_count:
                raise ValueError("panel family count is invalid")
            prompt_entries = family["prompts"]
            if not isinstance(prompt_entries, list) or len(
                prompt_entries
            ) != expected_family_count:
                raise ValueError("panel family prompt entries are invalid")
            expected_kinds = (
                Counter(
                    {
                        "short_control": 3,
                        "boundary_probe": 2,
                        "transition_probe": 2,
                        "beyond_window_probe": 3,
                    }
                )
                if role == CONFIRMATION_ROLE
                else Counter({"short_control": 2})
            )
            kinds = Counter(
                entry.get("kind")
                if isinstance(entry, Mapping)
                else None
                for entry in prompt_entries
            )
            if kinds != expected_kinds:
                raise ValueError("panel family prompt kinds are invalid")
            for offset, entry in enumerate(prompt_entries):
                if not isinstance(entry, Mapping) or set(entry) != {
                    "prompt_id",
                    "corpus_index",
                    "template_id",
                    "instance_id",
                    "domain_template",
                    "kind",
                    "stratum_id",
                    "text_sha256",
                    "canonical_sha256",
                    "semantic_lineage_id",
                    "paraphrase_lineage_id",
                    "template_signature_sha256",
                    "target_valid_tokens",
                    "input_ids_sha256",
                    "choices",
                    "correct_choice",
                    "supports_correct_choice_index",
                    "choices_sha256",
                    "choice_token_ids",
                    "answer_rationale",
                    "answer_rationale_sha256",
                    "answer_evidence",
                }:
                    raise ValueError("panel prompt fields are invalid")
                corpus_index = entry["corpus_index"]
                if type(corpus_index) is not int or not (
                    0 <= corpus_index < len(values)
                ):
                    raise ValueError("prompt corpus index is invalid")
                text = values[corpus_index]
                prompt_id = _strict_string(
                    entry["prompt_id"], label="prompt id"
                )
                template_id = _strict_string(
                    entry["template_id"], label="template id"
                )
                instance_id = _strict_string(
                    entry["instance_id"], label="instance id"
                )
                domain_template = _strict_string(
                    entry["domain_template"], label="domain template"
                )
                kind = _strict_string(entry["kind"], label="prompt kind")
                stratum_id_value = entry["stratum_id"]
                if role == CONFIRMATION_ROLE:
                    stratum_id = _strict_string(
                        stratum_id_value,
                        label="stratum id",
                    )
                elif stratum_id_value is None:
                    stratum_id = None
                else:
                    raise ValueError("reserve stratum id must be null")
                if entry["text_sha256"] != _prompt_sha256(text):
                    raise ValueError(f"prompt hash for {prompt_id!r} is invalid")
                canonical_sha256 = entry["canonical_sha256"]
                if canonical_sha256 != _canonical_text_sha256(text):
                    raise ValueError(
                        f"canonical prompt hash for {prompt_id!r} is invalid"
                    )
                semantic_lineage_id = _strict_string(
                    entry["semantic_lineage_id"],
                    label="semantic lineage id",
                )
                paraphrase_lineage_id = _strict_string(
                    entry["paraphrase_lineage_id"],
                    label="paraphrase lineage id",
                )
                template_signature_sha256 = entry[
                    "template_signature_sha256"
                ]
                if template_signature_sha256 != _template_signature_sha256(
                    role=role,
                    semantic_family=semantic_family,
                    template_id=template_id,
                    instance_id=instance_id,
                    domain_template=domain_template,
                ):
                    raise ValueError("template signature hash is invalid")
                for value, seen, label in (
                    (prompt_id, seen_prompt_ids, "prompt"),
                    (template_id, seen_template_ids, "template"),
                    (instance_id, seen_instance_ids, "instance"),
                    (domain_template, seen_domains, "domain template"),
                    (
                        semantic_lineage_id,
                        seen_semantic_lineages,
                        "semantic lineage",
                    ),
                    (
                        paraphrase_lineage_id,
                        seen_paraphrase_lineages,
                        "paraphrase lineage",
                    ),
                    (
                        template_signature_sha256,
                        seen_template_signatures,
                        "template signature",
                    ),
                ):
                    if value in seen:
                        raise ValueError(f"panel {label} identifiers must be unique")
                    seen.add(value)
                target = entry["target_valid_tokens"]
                token_sha = entry["input_ids_sha256"]
                choices_value = entry["choices"]
                if (
                    not isinstance(choices_value, list)
                    or len(choices_value) != 4
                    or any(not isinstance(choice, str) for choice in choices_value)
                ):
                    raise ValueError("prompt choices must contain four strings")
                choices = tuple(choices_value)
                if any(
                    not choice.startswith(" ")
                    or choice.startswith("  ")
                    or not choice.strip()
                    or choice != " " + choice.strip()
                    for choice in choices
                ):
                    raise ValueError(
                        "each choice must have exactly one leading space"
                    )
                if len(set(choices)) != 4:
                    raise ValueError("prompt choices must be distinct")
                correct_choice = entry["correct_choice"]
                if type(correct_choice) is not int or not (
                    0 <= correct_choice < 4
                ):
                    raise ValueError("correct choice must be an index from 0 to 3")
                supports_correct_choice_index = entry[
                    "supports_correct_choice_index"
                ]
                if supports_correct_choice_index != correct_choice:
                    raise ValueError(
                        "supported choice index must equal the gold coordinate"
                    )
                choices_sha256 = entry["choices_sha256"]
                if choices_sha256 != _json_sha256(list(choices)):
                    raise ValueError("ordered choice text hash is invalid")
                choice_token_ids_value = entry["choice_token_ids"]
                if role == RESERVE_ROLE:
                    if (
                        target is not None
                        or token_sha is not None
                        or choice_token_ids_value is not None
                    ):
                        raise ValueError(
                            "reserve cannot carry tokenizer-derived fields"
                        )
                    choice_token_ids = None
                else:
                    if type(target) is not int or not (
                        2 <= target <= MAX_POSITION_EMBEDDINGS
                    ):
                        raise ValueError("probe target length is invalid")
                    if not isinstance(token_sha, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", token_sha
                    ):
                        raise ValueError("probe token-stream hash is invalid")
                    if (
                        not isinstance(choice_token_ids_value, list)
                        or len(choice_token_ids_value) != 4
                        or any(
                            type(token_id) is not int or token_id < 0
                            for token_id in choice_token_ids_value
                        )
                        or len(set(choice_token_ids_value)) != 4
                    ):
                        raise ValueError(
                            "C choice token IDs must be four distinct integers"
                        )
                    choice_token_ids = tuple(choice_token_ids_value)
                answer_rationale = _strict_string(
                    entry["answer_rationale"],
                    label="answer rationale",
                )
                answer_rationale_sha256 = entry[
                    "answer_rationale_sha256"
                ]
                if answer_rationale_sha256 != _sha256_bytes(
                    answer_rationale.encode("utf-8")
                ):
                    raise ValueError("answer rationale hash is invalid")
                answer_evidence_value = entry["answer_evidence"]
                if role == CONFIRMATION_ROLE and kind == "beyond_window_probe":
                    if not isinstance(answer_evidence_value, Mapping) or set(
                        answer_evidence_value
                    ) != {
                        "text",
                        "text_sha256",
                        "char_start",
                        "char_end_exclusive",
                        "token_start",
                        "token_end_inclusive",
                        "final_query_token_index",
                        "outside_final_query_sliding_window",
                        "supports_correct_choice_index",
                        "rationale_sha256",
                    }:
                        raise ValueError("outside-window evidence is invalid")
                    evidence_text = _strict_string(
                        answer_evidence_value["text"],
                        label="answer evidence text",
                    )
                    char_start = answer_evidence_value["char_start"]
                    char_end = answer_evidence_value["char_end_exclusive"]
                    token_start = answer_evidence_value["token_start"]
                    token_end = answer_evidence_value["token_end_inclusive"]
                    final_query_index = answer_evidence_value[
                        "final_query_token_index"
                    ]
                    if (
                        type(char_start) is not int
                        or type(char_end) is not int
                        or not 0 <= char_start < char_end <= len(text)
                        or char_start != 0
                        or char_end != text.find(".") + 1
                        or text[char_start:char_end] != evidence_text
                        or not evidence_text.endswith(".")
                        or text.casefold().count(evidence_text.casefold()) != 1
                        or answer_evidence_value["text_sha256"]
                        != _sha256_bytes(evidence_text.encode("utf-8"))
                        or type(token_start) is not int
                        or type(token_end) is not int
                        or token_start != 1
                        or token_end <= token_start
                        or type(final_query_index) is not int
                        or final_query_index != target - 1
                        or token_end >= final_query_index - SLIDING_WINDOW + 1
                        or answer_evidence_value[
                            "outside_final_query_sliding_window"
                        ]
                        is not True
                        or answer_evidence_value[
                            "supports_correct_choice_index"
                        ]
                        != correct_choice
                        or answer_evidence_value["rationale_sha256"]
                        != answer_rationale_sha256
                        or choices[correct_choice].strip().casefold()
                        in _WORD_RE.findall(
                            _answer_specific_text(
                                semantic_family,
                                text[char_end:],
                            ).casefold()
                        )
                    ):
                        raise ValueError(
                            "outside-window answer evidence does not bind gold"
                        )
                    answer_evidence = dict(answer_evidence_value)
                elif answer_evidence_value is None:
                    answer_evidence = None
                else:
                    raise ValueError(
                        "only C outside-window rows may carry answer evidence"
                    )
                records.append(
                    PromptRecord(
                        role=role,
                        corpus_index=corpus_index,
                        semantic_family=semantic_family,
                        prompt_id=prompt_id,
                        template_id=template_id,
                        instance_id=instance_id,
                        domain_template=domain_template,
                        kind=kind,
                        stratum_id=stratum_id,
                        text=text,
                        text_sha256=str(entry["text_sha256"]),
                        canonical_sha256=canonical_sha256,
                        semantic_lineage_id=semantic_lineage_id,
                        paraphrase_lineage_id=paraphrase_lineage_id,
                        template_signature_sha256=(
                            template_signature_sha256
                        ),
                        target_valid_tokens=target,
                        input_ids_sha256=token_sha,
                        choices=choices,
                        correct_choice=correct_choice,
                        supports_correct_choice_index=(
                            supports_correct_choice_index
                        ),
                        choices_sha256=choices_sha256,
                        choice_token_ids=choice_token_ids,
                        answer_rationale=answer_rationale,
                        answer_rationale_sha256=answer_rationale_sha256,
                        answer_evidence=answer_evidence,
                    )
                )
                role_records.append(records[-1])
        if len(role_records) != len(values) or {
            record.corpus_index for record in role_records
        } != set(range(len(values))):
            raise ValueError(f"panel role {role!r} catalog does not cover text")
        by_index = {
            record.corpus_index: record for record in role_records
        }
        for kind, stratum_id, target, start, count in expected_buckets:
            bucket_records = [
                by_index[index] for index in range(start, start + count)
            ]
            if any(
                record.kind != kind
                or record.stratum_id != stratum_id
                or record.target_valid_tokens != target
                for record in bucket_records
            ):
                raise ValueError("corpus order does not match length buckets")
            if role == CONFIRMATION_ROLE and tuple(
                record.semantic_family for record in bucket_records
            ) != SEMANTIC_FAMILIES:
                raise ValueError("C buckets must use canonical family order")
        short_records = [
            by_index[index]
            for kind, stratum_id, target, start, count in expected_buckets
            if kind == "short_control"
            for index in range(start, start + count)
        ]
        expected_short_rounds = 3 if role == CONFIRMATION_ROLE else 2
        if tuple(record.semantic_family for record in short_records) != (
            SEMANTIC_FAMILIES * expected_short_rounds
        ):
            raise ValueError("short controls must use canonical family rounds")

    for family, lengths in EXPECTED_PROBE_SCHEDULE.items():
        observed = tuple(
            record.target_valid_tokens
            for record in records
            if record.role == CONFIRMATION_ROLE
            and record.semantic_family == family
        )
        if observed != lengths:
            raise ValueError(f"authenticated lengths for {family!r} are invalid")
    confirmation_records = tuple(
        record
        for record in records
        if record.role == CONFIRMATION_ROLE
    )
    for family, expected_coordinates in (
        EXPECTED_CONFIRMATION_CHOICE_COORDINATES.items()
    ):
        family_records = tuple(
            record
            for record in confirmation_records
            if record.semantic_family == family
        )
        observed_coordinates = tuple(
            record.correct_choice for record in family_records
        )
        if observed_coordinates != expected_coordinates:
            raise ValueError(
                f"choice-coordinate schedule for {family!r} is invalid"
            )
        if sorted(Counter(observed_coordinates).values()) != [2, 2, 3, 3]:
            raise ValueError(
                f"choice coordinates for {family!r} are not 3/3/2/2"
            )
        if len({record.correct_choice for record in family_records[:3]}) != 3:
            raise ValueError(
                f"short choice coordinates for {family!r} are confounded"
            )
        if len({record.correct_choice for record in family_records[3:6]}) != 3:
            raise ValueError(
                f"boundary choice coordinates for {family!r} are confounded"
            )
        if len(
            {record.correct_choice for record in family_records[7:]}
        ) != 3:
            raise ValueError(
                f"long choice coordinates for {family!r} are confounded"
            )
    if Counter(
        record.correct_choice for record in confirmation_records
    ) != Counter({0: 15, 1: 15, 2: 15, 3: 15}):
        raise ValueError("global choice coordinates are not 15/15/15/15")
    context_records = {
        "short": tuple(
            record
            for record in confirmation_records
            if record.target_valid_tokens in SHORT_CONTROL_LENGTHS
        ),
        "boundary": tuple(
            record
            for record in confirmation_records
            if record.target_valid_tokens
            in (*BOUNDARY_LENGTHS, *TRANSITION_LENGTHS)
        ),
        "long": tuple(
            record
            for record in confirmation_records
            if record.target_valid_tokens in BEYOND_WINDOW_LENGTHS
        ),
    }
    for context, context_counts in (
        EXPECTED_CHOICE_COORDINATE_CONTEXT_COUNTS.items()
    ):
        if Counter(
            record.correct_choice for record in context_records[context]
        ) != Counter(context_counts):
            raise ValueError(
                f"{context} choice-coordinate counts are invalid"
            )
    for family in SEMANTIC_FAMILIES:
        hard_long_records = tuple(
            record
            for record in confirmation_records
            if record.semantic_family == family
            and record.target_valid_tokens in BEYOND_WINDOW_LENGTHS
        )
        if len(hard_long_records) != 3:
            raise ValueError(
                f"hard-long row count for {family!r} is invalid"
            )
        visible_suffixes: list[str] = []
        for record in hard_long_records:
            if record.text.count(HARD_LONG_VISIBLE_SUFFIX_PREFIX) != 1:
                raise ValueError(
                    f"hard-long suffix marker for {family!r} is invalid"
                )
            suffix_start = record.text.index(
                HARD_LONG_VISIBLE_SUFFIX_PREFIX
            )
            if (
                record.answer_evidence is None
                or suffix_start
                <= int(record.answer_evidence["char_end_exclusive"])
            ):
                raise ValueError(
                    f"hard-long evidence/suffix order for {family!r} is invalid"
                )
            visible_suffixes.append(record.text[suffix_start:])
        if len(set(visible_suffixes)) != 1:
            raise ValueError(
                f"hard-long visible suffix for {family!r} encodes row identity"
            )
    for record in records:
        if record.semantic_family != "formal_logic":
            continue
        if set(record.choices) != set(FORMAL_LOGIC_CHOICES):
            raise ValueError(
                "formal-logic choices must use the exclusive four-way taxonomy"
            )
        if not record.text.endswith(FORMAL_LOGIC_QUERY_SUFFIX):
            raise ValueError(
                "formal-logic prompt omits the operational status definition"
            )
    if any(
        record.role == RESERVE_ROLE and record.kind != "short_control"
        for record in records
    ):
        raise ValueError("reserve must contain short parse/hash controls only")
    return LoadedPanel(
        corpus_path=corpus_path,
        manifest_path=manifest_path,
        corpus=corpus,
        manifest=manifest,
        records=tuple(records),
    )


def confirmation_prompts(panel: LoadedPanel) -> tuple[str, ...]:
    """Return the only prompts authorized for the frozen graph experiment."""

    values = panel.corpus[CONFIRMATION_ROLE]
    if not isinstance(values, list):
        raise TypeError("loaded confirmation corpus role is invalid")
    return tuple(str(value) for value in values)


def confirmation_prompt_records(panel: LoadedPanel) -> tuple[PromptRecord, ...]:
    """Return C records in canonical corpus order with explicit family IDs."""

    return tuple(
        sorted(
            (
                record
                for record in panel.records
                if record.role == CONFIRMATION_ROLE
            ),
            key=lambda record: record.corpus_index,
        )
    )


def load_default_confirmation_records(
    repository_root: Path | str,
) -> tuple[PromptRecord, ...]:
    """Load and return only the executable C role from the frozen panel."""

    root = Path(repository_root).resolve()
    panel = load_panel(root / DEFAULT_CORPUS, root / DEFAULT_PANEL)
    return confirmation_prompt_records(panel)


def iter_confirmation_length_batches(
    records: Sequence[PromptRecord],
    *,
    batch_size: int = 4,
) -> Iterator[tuple[PromptRecord, ...]]:
    """Yield C rows without ever mixing exact-length strata or reserve rows."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    values = tuple(records)
    if len(values) != EXPECTED_ROLE_COUNTS[CONFIRMATION_ROLE]:
        raise ValueError("confirmation execution requires exactly 60 C rows")
    expected = tuple(
        (stratum_id, target)
        for _, stratum_id, target, _, count in EXPECTED_CONFIRMATION_BUCKETS
        for _ in range(count)
    )
    observed = tuple(
        (record.stratum_id, record.target_valid_tokens) for record in values
    )
    if observed != expected or any(
        record.role != CONFIRMATION_ROLE for record in values
    ):
        raise ValueError(
            "confirmation rows must be canonical, C-only, and length-bucketed"
        )
    for _, _, _, start, count in EXPECTED_CONFIRMATION_BUCKETS:
        bucket = values[start : start + count]
        for offset in range(0, len(bucket), batch_size):
            yield bucket[offset : offset + batch_size]


def _tokenizer_vector_and_offsets(
    tokenizer: TokenizerLike,
    text: str,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer output must be a mapping")
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    offsets = encoded.get("offset_mapping")
    if (
        not isinstance(input_ids, list)
        or not isinstance(attention_mask, list)
        or len(input_ids) != len(attention_mask)
        or any(type(token) is not int for token in input_ids)
        or any(type(value) is not int for value in attention_mask)
        or any(value != 1 for value in attention_mask)
        or not isinstance(offsets, list)
        or len(offsets) != len(input_ids)
        or any(
            not isinstance(offset, (tuple, list))
            or len(offset) != 2
            or any(type(value) is not int for value in offset)
            or not 0 <= offset[0] <= offset[1] <= len(text)
            for offset in offsets
        )
    ):
        raise ValueError("single-prompt tokenizer output is invalid")
    return tuple(input_ids), tuple((offset[0], offset[1]) for offset in offsets)


def _choice_token_ids(
    tokenizer: TokenizerLike,
    choices: Sequence[str],
) -> tuple[int, int, int, int]:
    token_ids: list[int] = []
    for choice in choices:
        encoded = tokenizer(
            choice,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("choice tokenizer output must be a mapping")
        values = encoded.get("input_ids")
        if (
            not isinstance(values, list)
            or len(values) != 1
            or type(values[0]) is not int
        ):
            raise ValueError(
                "each leading-space choice must tokenize to one token"
            )
        token_ids.append(values[0])
    if len(set(token_ids)) != 4:
        raise ValueError("choice token IDs must be distinct within a row")
    return tuple(token_ids)  # type: ignore[return-value]


def _token_stream_sha256(input_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(_TOKEN_STREAM_DOMAIN)
    digest.update(
        json.dumps(
            list(input_ids),
            separators=(",", ":"),
        ).encode("ascii")
    )
    return digest.hexdigest()


def _choice_token_ids_sha256(input_ids: Sequence[int]) -> str:
    return _sha256_bytes(
        _CHOICE_TOKEN_IDS_DOMAIN
        + json.dumps(list(input_ids), separators=(",", ":")).encode("ascii")
    )


def _visible_suffix_token_ids_sha256(input_ids: Sequence[int]) -> str:
    return _sha256_bytes(
        _VISIBLE_SUFFIX_TOKEN_IDS_DOMAIN
        + json.dumps(list(input_ids), separators=(",", ":")).encode("ascii")
    )


def authenticate_confirmation_tokenization(
    panel: LoadedPanel,
    tokenizer: TokenizerLike,
) -> dict[str, object]:
    """Verify every C prompt, choice, and long evidence authentication.

    Reserve strings are never passed to ``tokenizer``.  The return value is a
    compact receipt and contains no token IDs or prompt text.
    """

    if getattr(tokenizer, "name_or_path", None) != MODEL_ID:
        raise ValueError("tokenizer name_or_path does not match the panel pin")
    if (
        getattr(tokenizer, "bos_token_id", None) != 2
        or getattr(tokenizer, "eos_token_id", None) != 1
        or getattr(tokenizer, "pad_token_id", None) != 0
    ):
        raise ValueError("tokenizer special-token identifiers do not match")
    observed: list[dict[str, object]] = []
    hard_long_suffixes: dict[str, list[str]] = {
        family: [] for family in SEMANTIC_FAMILIES
    }
    for record in confirmation_prompt_records(panel):
        input_ids, offsets = _tokenizer_vector_and_offsets(
            tokenizer,
            record.text,
        )
        digest = _token_stream_sha256(input_ids)
        if len(input_ids) != record.target_valid_tokens:
            raise ValueError(
                f"probe {record.prompt_id!r} has {len(input_ids)} valid "
                f"tokens, expected {record.target_valid_tokens}"
            )
        if digest != record.input_ids_sha256:
            raise ValueError(
                f"probe {record.prompt_id!r} token-stream hash changed"
            )
        choice_token_ids = _choice_token_ids(tokenizer, record.choices)
        if choice_token_ids != record.choice_token_ids:
            raise ValueError(
                f"probe {record.prompt_id!r} choice token IDs changed"
            )
        evidence_verified = False
        evidence_token_count = 0
        visible_suffix_token_ids_sha256: str | None = None
        if record.answer_evidence is not None:
            evidence = record.answer_evidence
            char_start = int(evidence["char_start"])
            char_end = int(evidence["char_end_exclusive"])
            evidence_token_indices = tuple(
                index
                for index, (start, end) in enumerate(offsets)
                if end > char_start and start < char_end
            )
            expected_indices = tuple(
                range(
                    int(evidence["token_start"]),
                    int(evidence["token_end_inclusive"]) + 1,
                )
            )
            if evidence_token_indices != expected_indices:
                raise ValueError(
                    f"probe {record.prompt_id!r} evidence token span changed"
                )
            if len(expected_indices) <= 1:
                raise ValueError(
                    f"probe {record.prompt_id!r} evidence is not a full clause"
                )
            final_query_index = len(input_ids) - 1
            if (
                final_query_index
                != int(evidence["final_query_token_index"])
                or expected_indices[-1]
                >= final_query_index - SLIDING_WINDOW + 1
            ):
                raise ValueError(
                    f"probe {record.prompt_id!r} evidence is not outside window"
                )
            visible_start = final_query_index - SLIDING_WINDOW + 1
            visible_suffix_ids = input_ids[visible_start:]
            if len(visible_suffix_ids) != SLIDING_WINDOW:
                raise ValueError(
                    f"probe {record.prompt_id!r} visible suffix is not exact"
                )
            visible_suffix_token_ids_sha256 = (
                _visible_suffix_token_ids_sha256(visible_suffix_ids)
            )
            hard_long_suffixes[record.semantic_family].append(
                visible_suffix_token_ids_sha256
            )
            visible_char_start = offsets[visible_start][0]
            visible_suffix = record.text[visible_char_start:].casefold()
            answer_specific_suffix = _answer_specific_text(
                record.semantic_family,
                visible_suffix,
            )
            evidence_text = str(evidence["text"]).casefold()
            correct_text = record.choices[record.correct_choice].strip().casefold()
            if (
                evidence_text in visible_suffix
                or correct_text in _WORD_RE.findall(answer_specific_suffix)
            ):
                raise ValueError(
                    f"probe {record.prompt_id!r} leaks clue or answer in window"
                )
            evidence_verified = True
            evidence_token_count = len(expected_indices)
        observed.append(
            {
                "prompt_id": record.prompt_id,
                "stratum_id": record.stratum_id,
                "valid_tokens": len(input_ids),
                "input_ids_sha256": digest,
                "choice_token_ids_sha256": _choice_token_ids_sha256(
                    choice_token_ids
                ),
                "choices_sha256": record.choices_sha256,
                "answer_evidence_verified": evidence_verified,
                "answer_evidence_token_count": evidence_token_count,
                "visible_suffix_token_ids_sha256": (
                    visible_suffix_token_ids_sha256
                ),
                "visible_suffix_leak_free": True,
            }
        )
    hard_long_visible_suffixes: list[dict[str, object]] = []
    for family in SEMANTIC_FAMILIES:
        suffix_digests = hard_long_suffixes[family]
        if len(suffix_digests) != 3 or len(set(suffix_digests)) != 1:
            raise ValueError(
                f"hard-long suffix token stream for {family!r} is not shared"
            )
        hard_long_visible_suffixes.append(
            {
                "semantic_family": family,
                "row_count": 3,
                "visible_suffix_token_ids_sha256": suffix_digests[0],
            }
        )
    return {
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "row_count": len(observed),
        "reserve_tokenized": False,
        "visible_suffix_digest_domain": (
            _VISIBLE_SUFFIX_TOKEN_IDS_DOMAIN.rstrip(b"\0").decode("ascii")
        ),
        "hard_long_visible_suffixes": hard_long_visible_suffixes,
        "rows": observed,
    }


authenticate_confirmation_probe_lengths = authenticate_confirmation_tokenization


def _reference_prompt_values(path: Path) -> tuple[str, ...]:
    if path.suffix == ".txt":
        return tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    raw = _strict_json(path)
    values: list[str] = []
    for role in ("calibration_a", "calibration_b", "validation", "test"):
        role_values = raw.get(role)
        if not isinstance(role_values, list) or any(
            not isinstance(value, str) for value in role_values
        ):
            raise ValueError(
                f"reference prompt source {path.name!r} has invalid splits"
            )
        values.extend(value.strip() for value in role_values if value.strip())
    return tuple(values)


def _downstream_prompt_values(path: Path) -> tuple[str, ...]:
    raw = _strict_json(path, maximum_bytes=_MAX_REFERENCE_FILE_BYTES)
    schema = raw.get("schema")
    if schema == "fisher_graph.gemma3_downstream_retention_panel":
        if path.name == "gemma3_downstream_retention_v1.json" and (
            _file_sha256(path) != _PINNED_DISCOVERY_PANEL_SHA256
        ):
            raise ValueError("spent discovery panel file hash changed")
        examples = raw.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("downstream retention examples are invalid")
        values: list[str] = []
        for example in examples:
            if not isinstance(example, Mapping):
                raise TypeError("downstream retention example must be an object")
            values.append(
                _strict_string(example.get("prompt"), label="discovery prompt")
            )
        return tuple(values)
    if schema == "fisher_graph.gemma3_downstream_qualification_bank":
        families = raw.get("families")
        if not isinstance(families, list) or not families:
            raise ValueError("downstream qualification families are invalid")
        values = []
        for family in families:
            if not isinstance(family, Mapping):
                raise TypeError("qualification family must be an object")
            template = _strict_string(
                family.get("prompt_template"),
                label="qualification prompt template",
            )
            if template.count("{stem}") != 1:
                raise ValueError("qualification template must contain one stem")
            for role in ("qualification", "evaluation"):
                rows = family.get(role)
                if not isinstance(rows, list):
                    raise TypeError("qualification rows must be a list")
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise TypeError("qualification row must be an object")
                    stem = _strict_string(
                        row.get("stem"),
                        label="qualification stem",
                    )
                    values.append(template.format(stem=stem))
        return tuple(values)
    raise ValueError(f"unsupported downstream prompt schema in {path.name!r}")


def _reference_domain_templates(path: Path) -> tuple[str, ...]:
    raw = _strict_json(path)
    roles = raw.get("roles")
    if not isinstance(roles, Mapping):
        return ()
    values: list[str] = []
    for role in roles.values():
        if not isinstance(role, Mapping):
            continue
        families = role.get("families")
        if not isinstance(families, list):
            continue
        for family in families:
            if not isinstance(family, Mapping):
                continue
            value = family.get("domain_template")
            if isinstance(value, str) and value:
                values.append(value)
    return tuple(values)


def _inventory_digest(entries: Sequence[Mapping[str, object]]) -> str:
    return _json_sha256(list(entries))


def _paraphrase_violation_count(
    left_values: Sequence[str],
    right_values: Sequence[str],
    *,
    same_collection: bool,
) -> int:
    violations = 0
    projected_left = tuple(
        _paraphrase_audit_projection(value) for value in left_values
    )
    projected_right = tuple(
        _paraphrase_audit_projection(value) for value in right_values
    )
    for left_index, left in enumerate(projected_left):
        start = left_index + 1 if same_collection else 0
        for right in projected_right[start:]:
            if _looks_like_paraphrase(left, right):
                violations += 1
    return violations


def build_overlap_audit_receipt(
    repository_root: Path | str,
    panel: LoadedPanel,
) -> dict[str, object]:
    """Build a deterministic, pair-free freshness and authentication receipt."""

    root = Path(repository_root).resolve()
    for path in (panel.corpus_path, panel.manifest_path, Path(__file__)):
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("panel audit paths must remain inside repository root")
    examples = root / "examples"
    prompt_sources = sorted(examples.glob("gemma3_*_prompts.json"))
    legacy = examples / "gemma3_prompts.txt"
    if legacy.is_file():
        prompt_sources.append(legacy)
    prompt_sources = sorted(set(prompt_sources))
    if not prompt_sources:
        raise ValueError("no checked-in Gemma reference prompt sources found")
    family_sources = sorted(
        examples.glob("gemma3_*_prompt_families.json")
    )
    downstream_sources = tuple(
        examples / name
        for name in (
            "gemma3_downstream_retention_v1.json",
            "gemma3_downstream_qualification_v2.json",
        )
    )
    if any(not path.is_file() for path in downstream_sources):
        raise ValueError("pinned downstream prompt source is missing")

    source_entries: list[dict[str, object]] = []
    reference_prompts: list[str] = []
    for path in prompt_sources:
        values = _reference_prompt_values(path)
        reference_prompts.extend(values)
        source_entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
                "prompt_count": len(values),
                "ordered_raw_prompt_sha256": _ordered_prompt_sha256(values),
                "ordered_canonical_prompt_sha256": _json_sha256(
                    [_canonical_text_sha256(value) for value in values]
                ),
            }
        )
    for path in downstream_sources:
        values = _downstream_prompt_values(path)
        reference_prompts.extend(values)
        source_entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
                "prompt_count": len(values),
                "ordered_raw_prompt_sha256": _ordered_prompt_sha256(values),
                "ordered_canonical_prompt_sha256": _json_sha256(
                    [_canonical_text_sha256(value) for value in values]
                ),
            }
        )
    source_entries.sort(key=lambda entry: str(entry["path"]))
    family_entries: list[dict[str, object]] = []
    reference_domains: set[str] = set()
    for path in family_sources:
        domains = _reference_domain_templates(path)
        reference_domains.update(domains)
        family_entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
                "domain_template_count": len(domains),
            }
        )

    panel_values = [record.text for record in panel.records]
    panel_exact = [_prompt_sha256(value) for value in panel_values]
    panel_canonical = [_canonical_text(value) for value in panel_values]
    reference_exact = {
        _prompt_sha256(value) for value in reference_prompts
    }
    reference_canonical = {
        _canonical_text(value) for value in reference_prompts
    }
    panel_domains = {record.domain_template for record in panel.records}
    outside_window_leaks = 0
    hard_long_evidence_clause_invalid = 0
    hard_long_suffixes: dict[str, list[str]] = {
        family: [] for family in SEMANTIC_FAMILIES
    }
    for record in panel.records:
        if record.answer_evidence is None:
            continue
        evidence = record.answer_evidence
        char_end = int(evidence["char_end_exclusive"])
        later = record.text[char_end:].casefold()
        answer_specific_later = _answer_specific_text(
            record.semantic_family,
            later,
        )
        if (
            str(evidence["text"]).casefold() in later
            or record.choices[record.correct_choice].strip().casefold()
            in _WORD_RE.findall(answer_specific_later)
        ):
            outside_window_leaks += 1
        evidence_text = str(evidence["text"])
        if (
            int(evidence["char_start"]) != 0
            or int(evidence["token_start"]) != 1
            or int(evidence["token_end_inclusive"]) <= 1
            or not evidence_text.endswith(".")
            or record.text[:char_end] != evidence_text
        ):
            hard_long_evidence_clause_invalid += 1
        suffix_start = record.text.index(HARD_LONG_VISIBLE_SUFFIX_PREFIX)
        hard_long_suffixes[record.semantic_family].append(
            record.text[suffix_start:]
        )
    hard_long_visible_suffix_mismatches = sum(
        max(0, len(set(suffixes)) - 1)
        for suffixes in hard_long_suffixes.values()
    )
    checks = {
        "panel_prompt_count": len(panel_values),
        "reference_prompt_count": len(reference_prompts),
        "panel_exact_duplicate_count": (
            len(panel_exact) - len(set(panel_exact))
        ),
        "panel_canonical_duplicate_count": (
            len(panel_canonical) - len(set(panel_canonical))
        ),
        "panel_paraphrase_overlap_count": _paraphrase_violation_count(
            panel_values,
            panel_values,
            same_collection=True,
        ),
        "reference_exact_overlap_count": len(
            set(panel_exact) & reference_exact
        ),
        "reference_canonical_overlap_count": len(
            set(panel_canonical) & reference_canonical
        ),
        "reference_paraphrase_overlap_count": _paraphrase_violation_count(
            panel_values,
            reference_prompts,
            same_collection=False,
        ),
        "reference_domain_template_overlap_count": len(
            panel_domains & reference_domains
        ),
        "prompt_id_duplicate_count": (
            len(panel.records)
            - len({record.prompt_id for record in panel.records})
        ),
        "template_id_duplicate_count": (
            len(panel.records)
            - len({record.template_id for record in panel.records})
        ),
        "instance_id_duplicate_count": (
            len(panel.records)
            - len({record.instance_id for record in panel.records})
        ),
        "domain_template_duplicate_count": (
            len(panel.records)
            - len(panel_domains)
        ),
        "semantic_lineage_duplicate_count": (
            len(panel.records)
            - len({record.semantic_lineage_id for record in panel.records})
        ),
        "paraphrase_lineage_duplicate_count": (
            len(panel.records)
            - len({record.paraphrase_lineage_id for record in panel.records})
        ),
        "template_signature_duplicate_count": (
            len(panel.records)
            - len(
                {
                    record.template_signature_sha256
                    for record in panel.records
                }
            )
        ),
        "reserve_model_access_count": 0,
        "reserve_tokenizer_access_count": 0,
        "outside_window_clue_or_answer_leak_count": outside_window_leaks,
        "hard_long_evidence_clause_invalid_count": (
            hard_long_evidence_clause_invalid
        ),
        "hard_long_visible_suffix_mismatch_count": (
            hard_long_visible_suffix_mismatches
        ),
    }
    count_checks = {
        key: value
        for key, value in checks.items()
        if key.endswith("_count") and key not in {
            "panel_prompt_count",
            "reference_prompt_count",
        }
    }
    passed = all(value == 0 for value in count_checks.values())
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "format_version": RECEIPT_FORMAT_VERSION,
        "panel_id": PANEL_ID,
        "prompt_corpus": panel.corpus_path.relative_to(root).as_posix(),
        "prompt_corpus_sha256": _file_sha256(panel.corpus_path),
        "panel_manifest": panel.manifest_path.relative_to(root).as_posix(),
        "panel_manifest_sha256": _file_sha256(panel.manifest_path),
        "audit_code": Path(__file__).resolve().relative_to(root).as_posix(),
        "audit_code_sha256": _file_sha256(Path(__file__).resolve()),
        "source_prompt_inventory": source_entries,
        "source_prompt_inventory_sha256": _inventory_digest(source_entries),
        "source_family_inventory": family_entries,
        "source_family_inventory_sha256": _inventory_digest(family_entries),
        "paraphrase_policy": dict(PARAPHRASE_POLICY),
        "blind_authoring": dict(BLIND_AUTHORING_RECEIPT),
        "checks": checks,
        "passed": passed,
    }
    receipt["payload_sha256"] = _sha256_bytes(
        _RECEIPT_DOMAIN
        + json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return receipt


def canonical_receipt_bytes(receipt: Mapping[str, object]) -> bytes:
    return _canonical_json_bytes(receipt)


def validate_frozen_receipt(
    repository_root: Path | str,
    panel: LoadedPanel,
    receipt_path: Path | str,
) -> dict[str, object]:
    expected = build_overlap_audit_receipt(repository_root, panel)
    actual_path = Path(receipt_path)
    root = Path(repository_root).resolve()
    if not actual_path.resolve(strict=True).is_relative_to(root):
        raise ValueError("receipt path must remain inside repository root")
    actual = _strict_json(actual_path, require_canonical=True)
    if actual != expected:
        raise ValueError("frozen overlap-audit receipt is stale")
    if actual_path.read_bytes() != canonical_receipt_bytes(actual):
        raise ValueError("frozen overlap-audit receipt is not canonical JSON")
    if actual.get("passed") is not True:
        raise ValueError("frozen overlap-audit receipt did not pass")
    return expected


__all__ = [
    "BEYOND_WINDOW_LENGTHS",
    "BLIND_AUTHORING_RECEIPT",
    "BOUNDARY_LENGTHS",
    "CONFIRMATION_ROLE",
    "CORPUS_FORMAT_VERSION",
    "CORPUS_SCHEMA",
    "DEFAULT_CORPUS",
    "DEFAULT_PANEL",
    "DEFAULT_RECEIPT",
    "EXACT_TARGET_LENGTHS",
    "EXPECTED_CHOICE_COORDINATE_CONTEXT_COUNTS",
    "EXPECTED_CONFIRMATION_CHOICE_COORDINATES",
    "EXPECTED_PROBE_SCHEDULE",
    "EXPECTED_CONFIRMATION_BUCKETS",
    "EXPECTED_RESERVE_BUCKETS",
    "EXPECTED_ROLE_COUNTS",
    "FORMAL_LOGIC_CHOICES",
    "FORMAL_LOGIC_QUERY_SUFFIX",
    "HARD_LONG_VISIBLE_SUFFIX_PREFIX",
    "LoadedPanel",
    "MAX_POSITION_EMBEDDINGS",
    "MODEL_ID",
    "MODEL_REVISION",
    "PANEL_FORMAT_VERSION",
    "PANEL_ID",
    "PANEL_SCHEMA",
    "PromptRecord",
    "RECEIPT_FORMAT_VERSION",
    "RECEIPT_SCHEMA",
    "RESERVE_ROLE",
    "ROLE_NAMES",
    "SEMANTIC_FAMILIES",
    "SHORT_CONTROL_LENGTHS",
    "SLIDING_WINDOW",
    "TRANSITION_LENGTHS",
    "authenticate_confirmation_probe_lengths",
    "authenticate_confirmation_tokenization",
    "build_overlap_audit_receipt",
    "canonical_receipt_bytes",
    "confirmation_prompts",
    "confirmation_prompt_records",
    "iter_confirmation_length_batches",
    "load_default_confirmation_records",
    "load_panel",
    "validate_frozen_receipt",
]
