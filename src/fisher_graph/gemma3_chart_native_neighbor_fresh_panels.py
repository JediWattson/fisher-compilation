"""Authenticate the blind, role-separated chart-native neighbor panels.

This module is standard-library-only and import inert.  The five raw role
corpora are independently byte-pinned.  Static preflight reads their bytes
only; it does not decode any role corpus.  Semantic access is then one-shot
and ordered A -> B -> C -> D, with a caller-supplied frozen-stage SHA-256
between openings.  ``reserve_unopened`` has no semantic-open path.

Tokenizer replay is also caller supplied.  The adapter must expose the small
``tokenizers.Tokenizer.encode`` surface (``ids`` and ``offsets``); this module
does not import a tokenizer package, tensor library, or model package.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence
import unicodedata


PANEL_ID = "gemma3-chart-native-neighbor-fresh-panels-v1"
ROLE_CORPUS_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.fresh_role_corpus.v1"
)
MANIFEST_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.fresh_manifest.v1"
)
RECEIPT_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.fresh_panel_receipt.v1"
)
FORMAT_VERSION = 1

EXECUTABLE_ROLES = (
    "A_chart_transport_fit",
    "B_neighbor_discovery",
    "C_neighbor_confirmation",
    "D_response_validation",
)
RESERVE_ROLE = "reserve_unopened"
ORDERED_ROLES = EXECUTABLE_ROLES + (RESERVE_ROLE,)
FAMILY_IDS = (
    "fresh-v1/nonce-lexicon-binding",
    "fresh-v1/cyclic-beacon-state",
    "fresh-v1/route-parity-resolution",
    "fresh-v1/witness-reliability-ranking",
    "fresh-v1/temporal-slot-placement",
    "fresh-v1/counterfactual-token-exchange",
)
FAMILY_SEMANTICS = (
    "lexical binding of a fresh nonce term to one referent",
    "quantitative state update on a finite cyclic beacon",
    "routing and Boolean logic over independently named relays",
    "evidence selection by authenticated independence and reliability",
    "temporal planning under slot and precedence constraints",
    "counterfactual application of a synthetic token-exchange rule",
)
TARGET_TOKEN_COUNTS = (96, 511, 512, 513, 514, 576, 640, 768)
STRATA = (
    "short_96",
    "boundary_511",
    "boundary_512",
    "boundary_513",
    "boundary_transition_514",
    "hard_long_576",
    "hard_long_640",
    "hard_long_768",
)
CHOICE_LABELS = ("A", "B", "C", "D")
REQUIRED_FREEZE_STAGE_KIND_BY_ROLE = {
    "A_chart_transport_fit": "chart_atlas_transport_fit",
    "B_neighbor_discovery": "neighbor_candidate_catalog",
    "C_neighbor_confirmation": "chart_neighbor_graph_and_gates",
    "D_response_validation": "response_validation",
}
SEMANTIC_CUES_BY_FAMILY = {
    FAMILY_IDS[0]: (
        "bind",
        "denote",
        "lexicon",
        "meaning",
        "nonce",
        "referent",
        "term",
    ),
    FAMILY_IDS[1]: ("beacon", "modulo", "phase", "pulse", "ring", "step"),
    FAMILY_IDS[2]: (
        "and",
        "dark",
        "east",
        "lit",
        "north",
        "open",
        "relay",
        "route",
        "xor",
    ),
    FAMILY_IDS[3]: (
        "accept",
        "copied",
        "evidence",
        "independent",
        "note",
        "reading",
        "seal",
        "verified",
        "witness",
    ),
    FAMILY_IDS[4]: (
        "after",
        "before",
        "follow",
        "plan",
        "slot",
        "task",
        "temporal",
    ),
    FAMILY_IDS[5]: (
        "counterfactual",
        "exchange",
        "first",
        "flag",
        "frost",
        "input",
        "rotate",
        "swap",
        "third",
        "token",
    ),
}
ROW_AXIS_CONTRACT = {
    "ordered_row_key_fields": [
        "role",
        "family_id",
        "example_id",
        "prediction_query_position",
        "target_label_position",
    ],
    "token_count_symbol": "L",
    "row_count_formula": "L_minus_1",
    "prediction_query_position_field": "prediction_query_position",
    "prediction_query_position_range": "0_through_L_minus_2",
    "target_label_position_field": "target_label_position",
    "target_label_position_range": "1_through_L_minus_1",
    "position_relation": (
        "target_label_position_equals_prediction_query_position_plus_1"
    ),
    "carrier_capture_position_field": "prediction_query_position",
    "contribution_capture_position_field": "prediction_query_position",
    "carrier_chart_signature_key_fields": [
        "example_id",
        "prediction_query_position",
    ],
    "label_position_field": "target_label_position",
    "next_token_NLL_position_field": "target_label_position",
    "score_objective_receipt_binds_full_q_t_catalog": True,
    "padding_rows_forbidden": True,
}


def _stratum_reporting_class(target: int) -> str:
    if target == 96:
        return "short_diagnostic"
    if target in (511, 512, 513):
        return "boundary_diagnostic"
    if target == 514:
        return "boundary_transition_diagnostic"
    return "hard_long_confirmation_veto"


def _expected_role_stratum_eligibility(
    role: str,
    target: int,
) -> dict[str, object]:
    pooled = role in {
        "B_neighbor_discovery",
        "C_neighbor_confirmation",
    } and target != 514
    hard_veto = role in {
        "B_neighbor_discovery",
        "C_neighbor_confirmation",
    } and target >= 576
    return {
        "stratum_reporting_class": _stratum_reporting_class(target),
        "diagnostic_reporting_required": target in (96, 511, 512, 513, 514),
        "boundary_transition_diagnostic_only": target == 514,
        "eligible_for_A_fit": role == "A_chart_transport_fit",
        "eligible_for_pooled_point_topology": pooled,
        "eligible_for_bootstrap": pooled,
        "eligible_for_LOFO": pooled,
        "eligible_for_hard_long_veto": hard_veto,
        "eligible_for_D_response_reporting": role == "D_response_validation",
        "may_change_chart_neighbor_graph": pooled,
    }


STRATUM_CONTRACT = {
    "ordered_strata": list(STRATA),
    "semantics": {
        stratum: {
            "token_count": target,
            "stratum_reporting_class": _stratum_reporting_class(target),
            "diagnostic_reporting_required": target
            in (96, 511, 512, 513, 514),
            "boundary_transition_diagnostic_only": target == 514,
            "hard_long_confirmation_veto_stratum": target >= 576,
        }
        for stratum, target in zip(STRATA, TARGET_TOKEN_COUNTS)
    },
    "short_diagnostic_strata": ["short_96"],
    "boundary_diagnostic_strata": [
        "boundary_511",
        "boundary_512",
        "boundary_513",
        "boundary_transition_514",
    ],
    "hard_long_confirmation_strata": [
        "hard_long_576",
        "hard_long_640",
        "hard_long_768",
    ],
    "boundary_transition_514_is_diagnostic_only": True,
    "boundary_transition_514_excluded_from_point_topology": True,
    "boundary_transition_514_excluded_from_bootstrap": True,
    "boundary_transition_514_excluded_from_LOFO": True,
    "boundary_transition_514_excluded_from_hard_confirmation": True,
    "B_C_pooled_eligible_strata": [
        "short_96",
        "boundary_511",
        "boundary_512",
        "boundary_513",
        "hard_long_576",
        "hard_long_640",
        "hard_long_768",
    ],
    "role_eligibility_counts": {
        "A_chart_transport_fit": {
            "fit_examples": 48,
            "fit_prediction_rows": 24_732,
            "pooled_examples": 0,
            "response_reporting_examples": 0,
        },
        "B_neighbor_discovery": {
            "pooled_examples": 42,
            "pooled_prediction_rows": 21_654,
            "bootstrap_examples": 42,
            "LOFO_examples": 42,
            "transition_diagnostic_examples": 6,
            "transition_diagnostic_prediction_rows": 3_078,
            "short_boundary_diagnostic_reporting_examples": 24,
            "short_boundary_diagnostic_reporting_rows": 9_768,
            "hard_long_veto_examples": 18,
            "hard_long_veto_prediction_rows": 11_886,
        },
        "C_neighbor_confirmation": {
            "pooled_examples": 42,
            "pooled_prediction_rows": 21_654,
            "bootstrap_examples": 42,
            "LOFO_examples": 42,
            "transition_diagnostic_examples": 6,
            "transition_diagnostic_prediction_rows": 3_078,
            "short_boundary_diagnostic_reporting_examples": 24,
            "short_boundary_diagnostic_reporting_rows": 9_768,
            "hard_long_veto_examples": 18,
            "hard_long_veto_prediction_rows": 11_886,
        },
        "D_response_validation": {
            "response_reporting_examples": 48,
            "response_reporting_prediction_rows": 24_732,
            "graph_changing_examples": 0,
            "transition_diagnostic_examples": 6,
            "transition_diagnostic_prediction_rows": 3_078,
        },
    },
}

DEFAULT_ROLE_CORPORA = {
    role: Path("examples")
    / f"gemma3_chart_native_neighbor_{role}_corpus_v1.json"
    for role in ORDERED_ROLES
}
DEFAULT_MANIFEST = Path(
    "examples/gemma3_chart_native_neighbor_fresh_manifest_v1.json"
)
DEFAULT_RECEIPT = Path(
    "artifacts/research/gemma3-chart-native-neighbor-fresh-panels-v1.json"
)

PINNED_ROLE_FILE_SHA256S = {
    "A_chart_transport_fit": (
        "112023d40793f1ff1c9952cc8b44e7d02b1f24a3473a72c941c4a494c2d6d0bf"
    ),
    "B_neighbor_discovery": (
        "f0c0b60e2e42b69703a4c81fc9a35c906a49d7a3e604b186c8d3fbc9c7755cc1"
    ),
    "C_neighbor_confirmation": (
        "ea8e533f6a4fe6c6461577211bd26d25ea0211a43b2847ab5afc75d90550ebc4"
    ),
    "D_response_validation": (
        "f16f4a3c04d04c403db8e50f9c2ed6db8c951b78065b5bb18f68956fb729a51a"
    ),
    "reserve_unopened": (
        "687824c48689a7a54698e3d1aa6d6d685e1bdfcddc703675b03ee5b68d429004"
    ),
}
PINNED_MANIFEST_FILE_SHA256 = (
    "5031f44454873a369ba7968c5bbfd6cf7e4a7340feaf0eddf6aee51398c948d7"
)
PINNED_MANIFEST_SHA256 = (
    "fe045222c8d4f7ec21b14d158da63878c3462a1c38c6b9f9dcecf2b4d1d5e57f"
)
PINNED_RECEIPT_FILE_SHA256 = (
    "65c5f6ea2e1241ad9bdd389da63304962d966fd988a2323ef6baf5dee56410d5"
)
PINNED_RECEIPT_SHA256 = (
    "a1586d6601323390573e750e9732bc7db4e415eefc0f15bf0a2fa8ef820cf1f0"
)
PINNED_NEUTRALITY_AUDIT_SHA256 = (
    "bbfcf08ea07e7d079f53d2a4aa7c905d33e2962a7b3798202fb5867c889f737c"
)
PINNED_TOKENIZER_JSON_SHA256 = (
    "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c"
)
PINNED_TOKENIZER_CONFIG_SHA256 = (
    "94b03056ec5831e9021c3e3fe9db682778c4f2d081c4188f1fcb8b90541cf1cf"
)
PINNED_SPECIAL_TOKENS_MAP_SHA256 = (
    "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397"
)
PINNED_MODEL_CONFIG_SHA256 = (
    "8983343e5bfbf8e336d222a522a1d781443491e9de2d081b8856f8fd26ab6046"
)

_DOMAINS = {
    "prompt": b"fisher-graph:chart-neighbor-fresh:prompt:v1\0",
    "canonical": b"fisher-graph:chart-neighbor-fresh:canonical:v1\0",
    "paraphrase": b"fisher-graph:chart-neighbor-fresh:paraphrase:v1\0",
    "template": b"fisher-graph:chart-neighbor-fresh:template:v1\0",
    "semantic": b"fisher-graph:chart-neighbor-fresh:semantic-instance:v1\0",
    "token_ids": b"fisher-graph:chart-neighbor-fresh:token-ids:v1\0",
    "row_stream": b"fisher-graph:chart-neighbor-fresh:row-stream:v1\0",
    "score_objective": (
        b"fisher-graph:chart-neighbor-fresh:score-objective-receipt:v1\0"
    ),
    "evidence": b"fisher-graph:chart-neighbor-fresh:evidence:v1\0",
    "suffix": b"fisher-graph:chart-neighbor-fresh:neutral-suffix:v1\0",
    "choice_ids": b"fisher-graph:chart-neighbor-fresh:choice-token-ids:v1\0",
    "choice_text": b"fisher-graph:chart-neighbor-fresh:choice-text:v1\0",
    "identity_catalog": b"fisher-graph:chart-neighbor-fresh:identity-catalog:v1\0",
    "neutrality_audit": (
        b"fisher-graph:chart-neighbor-fresh:hard-neutrality-audit:v1\0"
    ),
    "prior_catalog": b"fisher-graph:chart-neighbor-fresh:prior-catalog:v1\0",
    "manifest": b"fisher-graph:chart-neighbor-fresh:manifest:v1\0",
    "receipt": b"fisher-graph:chart-neighbor-fresh:receipt:v1\0",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has in into is it of on or that the "
    "their then to under was which with".split()
)
_IDENTITY_KINDS = (
    "prompt",
    "canonical",
    "paraphrase",
    "template",
    "semantic",
)
_MAX_JSON_BYTES = 64 * 1024 * 1024


class FreshPanelIntegrityError(RuntimeError):
    """A frozen prompt-free or opened role artifact failed validation."""


class FreshPanelFirewallError(FreshPanelIntegrityError):
    """A role was opened out of order, reopened, or was the reserve."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_sha(kind: str, value: object) -> str:
    return _sha_bytes(_DOMAINS[kind] + canonical_json_bytes(value))


def _text_sha(kind: str, value: str) -> str:
    return _sha_bytes(_DOMAINS[kind] + value.encode("utf-8"))


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FreshPanelIntegrityError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_prompt(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def _paraphrase_signature(value: str) -> object:
    words = [
        word
        for word in _canonical_prompt(value).split()
        if word not in _STOPWORDS
    ]
    return sorted(Counter(words).items())


def _token_ids_sha(ids: Sequence[int]) -> str:
    encoded = b"".join(int(value).to_bytes(4, "big") for value in ids)
    return _sha_bytes(_DOMAINS["token_ids"] + encoded)


def _read_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file():
        raise FreshPanelIntegrityError(f"{label} must be a regular file")
    encoded = path.read_bytes()
    if not encoded or len(encoded) > _MAX_JSON_BYTES:
        raise FreshPanelIntegrityError(f"{label} has an invalid byte size")
    return encoded


def _decode_canonical_mapping(encoded: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshPanelIntegrityError(f"{label} is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise FreshPanelIntegrityError(f"{label} must contain one JSON object")
    if canonical_json_bytes(value) != encoded:
        raise FreshPanelIntegrityError(
            f"{label} must use exact canonical JSON bytes"
        )
    return value


def _copy_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(canonical_json_bytes(value))


def _without_digest(value: Mapping[str, object], field: str) -> dict[str, object]:
    payload = _copy_mapping(value)
    payload.pop(field, None)
    return payload


def _identity_sets(role: Mapping[str, object]) -> dict[str, set[str]]:
    try:
        catalog = role["identity_catalog"]
    except KeyError as error:
        raise FreshPanelIntegrityError("role lacks its identity catalog") from error
    if not isinstance(catalog, Mapping):
        raise FreshPanelIntegrityError("role identity catalog must be a mapping")
    output: dict[str, set[str]] = {}
    for kind in _IDENTITY_KINDS:
        field_name = f"{kind}_sha256s"
        values = catalog.get(field_name)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and _SHA256.fullmatch(value)
            for value in values
        ):
            raise FreshPanelIntegrityError(
                f"role identity catalog {field_name} is invalid"
            )
        if values != sorted(set(values)):
            raise FreshPanelIntegrityError(
                f"role identity catalog {field_name} is not sorted unique"
            )
        output[kind] = set(values)
    if role.get("identity_catalog_sha256") != _domain_sha(
        "identity_catalog", catalog
    ):
        raise FreshPanelIntegrityError("role identity catalog digest drifted")
    return output


def _expected_role_eligibility_counts(
    role: str,
) -> tuple[dict[str, int], dict[str, int]]:
    examples: Counter[str] = Counter()
    rows: Counter[str] = Counter()
    for _family in FAMILY_IDS:
        for target in TARGET_TOKEN_COUNTS:
            eligibility = _expected_role_stratum_eligibility(role, target)
            for field_name, enabled in eligibility.items():
                if isinstance(enabled, bool) and enabled:
                    examples[field_name] += 1
                    rows[field_name] += target - 1
    return dict(examples), dict(rows)


def _validate_neutrality_audit(
    manifest: Mapping[str, object],
    *,
    group_suffixes: Mapping[tuple[str, str], set[str]],
) -> None:
    audit = manifest.get("hard_long_neutrality_audit")
    if not isinstance(audit, Mapping):
        raise FreshPanelIntegrityError("hard-long neutrality audit is missing")
    audit_digest = audit.get("audit_sha256")
    if audit_digest != _domain_sha(
        "neutrality_audit", _without_digest(audit, "audit_sha256")
    ):
        raise FreshPanelIntegrityError("hard-long neutrality audit digest drifted")
    if audit_digest != PINNED_NEUTRALITY_AUDIT_SHA256:
        raise FreshPanelIntegrityError("hard-long neutrality audit pin drifted")
    expected_header = {
        "group_count": 24,
        "choice_bearing_role": "D_response_validation",
        "all_role_family_hard_suffixes_audited": True,
        "raw_choice_text_and_token_sequence_audited": True,
        "response_choice_label_token_ids_audited": True,
        "semantic_cue_text_and_token_sequence_audited": True,
        "all_overlap_counts_zero": True,
    }
    if any(audit.get(key) != value for key, value in expected_header.items()):
        raise FreshPanelIntegrityError("hard-long neutrality audit header drifted")
    groups = audit.get("groups")
    expected_keys = {
        f"{role}|{family}"
        for role in EXECUTABLE_ROLES
        for family in FAMILY_IDS
    }
    if not isinstance(groups, Mapping) or set(groups) != expected_keys:
        raise FreshPanelIntegrityError("hard-long neutrality groups drifted")
    zero_fields = (
        "raw_choice_text_overlap_count",
        "choice_token_subsequence_overlap_count",
        "response_choice_label_token_overlap_count",
        "raw_semantic_cue_overlap_count",
        "semantic_cue_token_subsequence_overlap_count",
    )
    sha_list_fields = (
        "corresponding_choice_text_sha256s",
        "corresponding_choice_token_sequence_sha256s",
        "response_choice_label_token_sequence_sha256s",
        "semantic_cue_sha256s",
    )
    for role in EXECUTABLE_ROLES:
        for family in FAMILY_IDS:
            group = groups[f"{role}|{family}"]
            if not isinstance(group, Mapping):
                raise FreshPanelIntegrityError("hard neutrality group is invalid")
            suffix_values = group_suffixes.get((role, family))
            expected_suffix = (
                next(iter(suffix_values))
                if suffix_values is not None and len(suffix_values) == 1
                else None
            )
            expected_choice_count = 12 if role == "D_response_validation" else 0
            if (
                group.get("role") != role
                or group.get("family_id") != family
                or group.get("hard_example_count") != 3
                or group.get("shared_suffix_sha256") != expected_suffix
                or group.get("corresponding_choice_text_count")
                != expected_choice_count
                or group.get("all_overlap_counts_zero") is not True
                or any(group.get(field_name) != 0 for field_name in zero_fields)
            ):
                raise FreshPanelIntegrityError(
                    f"hard neutrality group drifted for {role}/{family}"
                )
            for field_name in sha_list_fields:
                values = group.get(field_name)
                if (
                    not isinstance(values, list)
                    or values != sorted(set(values))
                    or not all(
                        isinstance(value, str) and _SHA256.fullmatch(value)
                        for value in values
                    )
                ):
                    raise FreshPanelIntegrityError(
                        f"hard neutrality {field_name} drifted"
                    )
            expected_cue_hashes = sorted(
                _text_sha("choice_text", cue)
                for cue in SEMANTIC_CUES_BY_FAMILY[family]
            )
            if group.get("semantic_cue_sha256s") != expected_cue_hashes:
                raise FreshPanelIntegrityError(
                    f"hard neutrality cue catalog drifted for {family}"
                )
            choice_hashes = group["corresponding_choice_text_sha256s"]
            choice_token_hashes = group[
                "corresponding_choice_token_sequence_sha256s"
            ]
            label_token_hashes = group[
                "response_choice_label_token_sequence_sha256s"
            ]
            if role == "D_response_validation":
                if (
                    not choice_hashes
                    or not choice_token_hashes
                    or not label_token_hashes
                ):
                    raise FreshPanelIntegrityError(
                        "D hard neutrality choice audit is empty"
                    )
            elif choice_hashes or choice_token_hashes or label_token_hashes:
                raise FreshPanelIntegrityError(
                    f"{role} unexpectedly has hard neutrality choices"
                )


def validate_manifest(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the prompt-free frozen manifest without opening raw corpora."""

    if not isinstance(value, Mapping):
        raise FreshPanelIntegrityError("manifest must be a mapping")
    manifest = _copy_mapping(value)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise FreshPanelIntegrityError("fresh manifest schema drifted")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise FreshPanelIntegrityError("fresh manifest format version drifted")
    if manifest.get("panel_id") != PANEL_ID:
        raise FreshPanelIntegrityError("fresh manifest panel ID drifted")
    if tuple(manifest.get("family_ids", ())) != FAMILY_IDS:
        raise FreshPanelIntegrityError("six-family catalog drifted")
    if tuple(manifest.get("family_semantics", ())) != FAMILY_SEMANTICS:
        raise FreshPanelIntegrityError("six-family semantics drifted")
    if manifest.get("row_axis_contract") != ROW_AXIS_CONTRACT:
        raise FreshPanelIntegrityError("q/t row-axis contract drifted")
    if manifest.get("stratum_contract") != STRATUM_CONTRACT:
        raise FreshPanelIntegrityError("stratum semantics and eligibility drifted")
    if manifest.get("required_freeze_stage_kind_by_role") != (
        REQUIRED_FREEZE_STAGE_KIND_BY_ROLE
    ):
        raise FreshPanelIntegrityError("required freeze-stage kinds drifted")
    observed_digest = manifest.get("manifest_sha256")
    if observed_digest != _domain_sha(
        "manifest", _without_digest(manifest, "manifest_sha256")
    ):
        raise FreshPanelIntegrityError("fresh manifest digest drifted")
    if observed_digest != PINNED_MANIFEST_SHA256:
        raise FreshPanelIntegrityError("fresh manifest public pin drifted")

    tokenizer = manifest.get("tokenizer_contract")
    if not isinstance(tokenizer, Mapping):
        raise FreshPanelIntegrityError("tokenizer contract is missing")
    expected_tokenizer = {
        "model_id": "google/gemma-3-270m",
        "revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "tokenizer_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "tokenizer_json_sha256": PINNED_TOKENIZER_JSON_SHA256,
        "tokenizer_config_sha256": PINNED_TOKENIZER_CONFIG_SHA256,
        "special_tokens_map_sha256": PINNED_SPECIAL_TOKENS_MAP_SHA256,
        "model_config_sha256": PINNED_MODEL_CONFIG_SHA256,
        "add_special_tokens": True,
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token_id": 2,
        "sliding_window": 512,
        "row_key": ROW_AXIS_CONTRACT["ordered_row_key_fields"],
        "prediction_query_positions": "0_through_token_count_minus_2",
        "target_label_positions": "1_through_token_count_minus_1",
    }
    if tokenizer != expected_tokenizer:
        raise FreshPanelIntegrityError("pinned tokenizer contract drifted")

    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or tuple(roles) != ORDERED_ROLES:
        raise FreshPanelIntegrityError("ordered role manifest drifted")
    identity_by_role: dict[str, dict[str, set[str]]] = {}
    group_suffixes: dict[tuple[str, str], set[str]] = {}
    group_suffix_templates: dict[tuple[str, str], set[str]] = {}
    expected_family_rows = sum(target - 1 for target in TARGET_TOKEN_COUNTS)
    expected_stratum_rows = {
        stratum: (target - 1) * len(FAMILY_IDS)
        for target, stratum in zip(TARGET_TOKEN_COUNTS, STRATA)
    }
    for role in EXECUTABLE_ROLES:
        role_value = roles.get(role)
        if not isinstance(role_value, Mapping):
            raise FreshPanelIntegrityError(f"{role} manifest is missing")
        if role_value.get("role") != role:
            raise FreshPanelIntegrityError(f"{role} role binding drifted")
        if role_value.get("access_class") != (
            "sequential_semantic_open_then_tokenizer_or_model"
        ):
            raise FreshPanelIntegrityError(f"{role} access class drifted")
        if (
            role_value.get("example_count") != 48
            or role_value.get("family_count") != 6
            or role_value.get("examples_per_family") != 8
        ):
            raise FreshPanelIntegrityError(f"{role} balanced geometry drifted")
        if tuple(role_value.get("target_token_counts_per_family", ())) != (
            TARGET_TOKEN_COUNTS
        ):
            raise FreshPanelIntegrityError(f"{role} token schedule drifted")
        if (
            role_value.get("prompt_token_count") != 24_780
            or role_value.get("prediction_row_count") != 24_732
        ):
            raise FreshPanelIntegrityError(f"{role} exact cost surface drifted")
        if role_value.get("family_prediction_row_counts") != {
            family: expected_family_rows for family in FAMILY_IDS
        }:
            raise FreshPanelIntegrityError(f"{role} family row counts drifted")
        if role_value.get("stratum_prediction_row_counts") != (
            expected_stratum_rows
        ):
            raise FreshPanelIntegrityError(f"{role} stratum row counts drifted")
        members = role_value.get("ordered_members")
        if not isinstance(members, list) or len(members) != 48:
            raise FreshPanelIntegrityError(f"{role} must have 48 members")
        correct_indices: Counter[int] = Counter()
        for member_index, member in enumerate(members):
            if not isinstance(member, Mapping):
                raise FreshPanelIntegrityError(f"{role} member is invalid")
            family_index, within_family = divmod(member_index, 8)
            family = FAMILY_IDS[family_index]
            target = TARGET_TOKEN_COUNTS[within_family]
            stratum = STRATA[within_family]
            if (
                member.get("family_id") != family
                or member.get("target_token_count") != target
                or member.get("observed_token_count") != target
                or member.get("prediction_row_count") != target - 1
                or member.get("stratum") != stratum
            ):
                raise FreshPanelIntegrityError(
                    f"{role} member schedule drifted at {member_index}"
                )
            expected_eligibility = _expected_role_stratum_eligibility(
                role,
                target,
            )
            if any(
                member.get(field_name) != expected_value
                for field_name, expected_value in expected_eligibility.items()
            ):
                raise FreshPanelIntegrityError(
                    f"{role} eligibility drifted at {member_index}"
                )
            if (
                member.get("prediction_query_position_start") != 0
                or member.get(
                    "prediction_query_position_stop_exclusive"
                )
                != target - 1
                or member.get("target_label_position_start") != 1
                or member.get("target_label_position_stop_exclusive")
                != target
            ):
                raise FreshPanelIntegrityError(
                    f"{role} q/t row ranges drifted at {member_index}"
                )
            for field_name in (
                "prompt_sha256",
                "canonical_prompt_sha256",
                "paraphrase_fingerprint_sha256",
                "template_id_sha256",
                "semantic_instance_sha256",
                "token_ids_sha256",
                "row_stream_sha256",
                "objective_row_stream_sha256",
                "evidence_text_sha256",
                "neutral_suffix_sha256",
                "neutral_suffix_template_sha256",
            ):
                _require_sha256(
                    member.get(field_name),
                    label=f"{role} member {field_name}",
                )
            if member.get("objective_row_stream_sha256") != member.get(
                "row_stream_sha256"
            ):
                raise FreshPanelIntegrityError(
                    f"{role} objective row stream alias drifted"
                )
            evidence_span = member.get("evidence_token_span")
            suffix_span = member.get("neutral_suffix_token_span")
            if not (
                isinstance(evidence_span, list)
                and len(evidence_span) == 2
                and all(isinstance(item, int) for item in evidence_span)
                and 0 <= evidence_span[0] < evidence_span[1] <= target
            ):
                raise FreshPanelIntegrityError(f"{role} evidence span drifted")
            if not (
                isinstance(suffix_span, list)
                and len(suffix_span) == 2
                and all(isinstance(item, int) for item in suffix_span)
                and 0 <= suffix_span[0] < suffix_span[1] <= target
            ):
                raise FreshPanelIntegrityError(f"{role} suffix span drifted")
            hard = target >= 576
            if member.get("hard_long_final512_evidence_excluded") is not hard:
                raise FreshPanelIntegrityError(f"{role} evidence gate drifted")
            if member.get("hard_long_final512_answer_neutral") is not hard:
                raise FreshPanelIntegrityError(f"{role} suffix gate drifted")
            if hard:
                if evidence_span[1] > target - 512:
                    raise FreshPanelIntegrityError(
                        f"{role} decisive evidence enters final512"
                    )
                if suffix_span != [target - 512, target]:
                    raise FreshPanelIntegrityError(
                        f"{role} answer-neutral suffix is not exact final512"
                    )
                key = (role, family)
                group_suffixes.setdefault(key, set()).add(
                    str(member["neutral_suffix_sha256"])
                )
                group_suffix_templates.setdefault(key, set()).add(
                    str(member["neutral_suffix_template_sha256"])
                )
            choice_fields = {
                "choice_count",
                "bare_choice_token_counts",
                "append_choice_token_counts",
                "distinct_bare_choice_token_id_count",
                "distinct_append_choice_token_id_count",
                "bare_choice_token_ids_sha256",
                "append_choice_token_ids_sha256",
                "correct_choice_index",
                "choice_text_count",
                "choice_text_sha256s",
                "choice_text_bare_token_ids_sha256s",
                "choice_text_append_token_ids_sha256s",
            }
            if role == "D_response_validation":
                if not choice_fields.issubset(member):
                    raise FreshPanelIntegrityError("D choice receipt is incomplete")
                if (
                    member.get("choice_count") != 4
                    or member.get("bare_choice_token_counts") != [1, 1, 1, 1]
                    or member.get("append_choice_token_counts") != [1, 1, 1, 1]
                    or member.get("distinct_bare_choice_token_id_count") != 4
                    or member.get("distinct_append_choice_token_id_count") != 4
                    or member.get("choice_text_count") != 4
                ):
                    raise FreshPanelIntegrityError("D single-token choice ABI drifted")
                _require_sha256(
                    member.get("bare_choice_token_ids_sha256"),
                    label="D bare choice token digest",
                )
                _require_sha256(
                    member.get("append_choice_token_ids_sha256"),
                    label="D append choice token digest",
                )
                for field_name in (
                    "choice_text_sha256s",
                    "choice_text_bare_token_ids_sha256s",
                    "choice_text_append_token_ids_sha256s",
                ):
                    values = member.get(field_name)
                    if not (
                        isinstance(values, list)
                        and len(values) == 4
                        and all(
                            isinstance(value, str)
                            and _SHA256.fullmatch(value)
                            for value in values
                        )
                    ):
                        raise FreshPanelIntegrityError(
                            f"D {field_name} drifted"
                        )
                correct = member.get("correct_choice_index")
                if not isinstance(correct, int) or correct not in range(4):
                    raise FreshPanelIntegrityError("D correct choice index drifted")
                correct_indices[correct] += 1
            elif choice_fields.intersection(member):
                raise FreshPanelIntegrityError(f"{role} unexpectedly has D choices")
        expected_examples, expected_rows = _expected_role_eligibility_counts(role)
        if role_value.get("eligibility_example_counts") != expected_examples:
            raise FreshPanelIntegrityError(
                f"{role} eligibility example counts drifted"
            )
        if role_value.get("eligibility_prediction_row_counts") != expected_rows:
            raise FreshPanelIntegrityError(
                f"{role} eligibility row counts drifted"
            )
        if role == "D_response_validation" and correct_indices != Counter(
            {0: 12, 1: 12, 2: 12, 3: 12}
        ):
            raise FreshPanelIntegrityError("D correct indices are not balanced")
        objective_catalog_sha256 = role_value.get(
            "ordered_objective_row_catalog_sha256"
        )
        _require_sha256(
            objective_catalog_sha256,
            label=f"{role} ordered objective row catalog",
        )
        if objective_catalog_sha256 != role_value.get(
            "ordered_row_catalog_sha256"
        ):
            raise FreshPanelIntegrityError(
                f"{role} objective row catalog alias drifted"
            )
        expected_score_objective_receipt = {
            "role": role,
            "objective_row_key_fields": ROW_AXIS_CONTRACT[
                "ordered_row_key_fields"
            ],
            "carrier_chart_signature_key_fields": ROW_AXIS_CONTRACT[
                "carrier_chart_signature_key_fields"
            ],
            "objective_row_count": 24_732,
            "ordered_objective_row_catalog_sha256": (
                objective_catalog_sha256
            ),
            "position_relation": ROW_AXIS_CONTRACT["position_relation"],
            "carrier_capture_position_field": ROW_AXIS_CONTRACT[
                "carrier_capture_position_field"
            ],
            "contribution_capture_position_field": ROW_AXIS_CONTRACT[
                "contribution_capture_position_field"
            ],
            "label_position_field": ROW_AXIS_CONTRACT[
                "label_position_field"
            ],
            "next_token_NLL_position_field": ROW_AXIS_CONTRACT[
                "next_token_NLL_position_field"
            ],
        }
        if role_value.get("score_objective_receipt") != (
            expected_score_objective_receipt
        ):
            raise FreshPanelIntegrityError(
                f"{role} score-objective receipt drifted"
            )
        if role_value.get("score_objective_receipt_sha256") != _domain_sha(
            "score_objective", expected_score_objective_receipt
        ):
            raise FreshPanelIntegrityError(
                f"{role} score-objective receipt digest drifted"
            )
        identity_by_role[role] = _identity_sets(role_value)
        expected_identity_counts = {
            "prompt": 48,
            "canonical": 48,
            "paraphrase": 48,
            "template": 6,
            "semantic": 48,
        }
        if {
            kind: len(values) for kind, values in identity_by_role[role].items()
        } != expected_identity_counts:
            raise FreshPanelIntegrityError(f"{role} identity counts drifted")

    reserve = roles.get(RESERVE_ROLE)
    if not isinstance(reserve, Mapping):
        raise FreshPanelIntegrityError("reserve commitment is missing")
    if reserve.get("access_class") != (
        "opaque_bytes_only_never_decode_tokenize_or_model"
    ):
        raise FreshPanelIntegrityError("reserve access class drifted")
    if (
        reserve.get("declared_example_count") != 12
        or reserve.get("declared_family_count") != 6
        or reserve.get("declared_examples_per_family") != 2
        or reserve.get("tokenization_performed") is not False
        or reserve.get("model_access_performed") is not False
        or reserve.get("token_cost_claimed") is not False
        or "ordered_members" in reserve
        or "prompt_token_count" in reserve
        or "prediction_row_count" in reserve
    ):
        raise FreshPanelIntegrityError("reserve opaque-only commitment drifted")
    identity_by_role[RESERVE_ROLE] = _identity_sets(reserve)
    if {
        kind: len(values)
        for kind, values in identity_by_role[RESERVE_ROLE].items()
    } != {
        "prompt": 12,
        "canonical": 12,
        "paraphrase": 12,
        "template": 6,
        "semantic": 12,
    }:
        raise FreshPanelIntegrityError("reserve identity counts drifted")

    suffix_contract = manifest.get("hard_long_suffix_contract")
    if not isinstance(suffix_contract, Mapping):
        raise FreshPanelIntegrityError("hard-long suffix contract is missing")
    if (
        suffix_contract.get("exact_final_token_count") != 512
        or suffix_contract.get(
            "same_raw_suffix_required_only_within_role_family"
        )
        is not True
        or suffix_contract.get(
            "cross_role_or_family_suffix_sha256_overlap_forbidden"
        )
        is not True
    ):
        raise FreshPanelIntegrityError("hard-long suffix policy drifted")
    if any(len(values) != 1 for values in group_suffixes.values()):
        raise FreshPanelIntegrityError("hard suffix differs within role/family")
    if any(len(values) != 1 for values in group_suffix_templates.values()):
        raise FreshPanelIntegrityError(
            "hard suffix template differs within role/family"
        )
    all_suffixes = [next(iter(values)) for values in group_suffixes.values()]
    all_templates = [
        next(iter(values)) for values in group_suffix_templates.values()
    ]
    if len(group_suffixes) != 24 or len(set(all_suffixes)) != 24:
        raise FreshPanelIntegrityError("hard suffix overlaps across groups")
    if len(set(all_templates)) != 24:
        raise FreshPanelIntegrityError(
            "hard suffix template overlaps across groups"
        )
    frozen_suffixes = suffix_contract.get("role_family_suffix_sha256s")
    expected_suffixes = {
        role: {
            family: next(iter(group_suffixes[(role, family)]))
            for family in FAMILY_IDS
        }
        for role in EXECUTABLE_ROLES
    }
    if frozen_suffixes != expected_suffixes:
        raise FreshPanelIntegrityError("hard suffix catalog drifted")
    _validate_neutrality_audit(
        manifest,
        group_suffixes=group_suffixes,
    )

    prior = manifest.get("prior_identity_catalog")
    if not isinstance(prior, Mapping):
        raise FreshPanelIntegrityError("prior identity catalog is missing")
    prior_payload = _copy_mapping(prior)
    prior_digest = prior_payload.pop("catalog_sha256", None)
    if prior_digest != _domain_sha("prior_catalog", prior_payload):
        raise FreshPanelIntegrityError("prior identity catalog digest drifted")
    prior_sets: dict[str, set[str]] = {}
    prior_field_by_kind = {
        "prompt": "prompt_sha256s",
        "canonical": "canonical_sha256s",
        "paraphrase": "paraphrase_sha256s",
        "template": "template_sha256s",
        "semantic": "semantic_sha256s",
    }
    for kind, field_name in prior_field_by_kind.items():
        values = prior.get(field_name)
        if not isinstance(values, list) or values != sorted(set(values)):
            raise FreshPanelIntegrityError(
                f"prior identity catalog {field_name} drifted"
            )
        if not all(
            isinstance(item, str) and _SHA256.fullmatch(item)
            for item in values
        ):
            raise FreshPanelIntegrityError(
                f"prior identity catalog {field_name} is invalid"
            )
        prior_sets[kind] = set(values)

    computed_role_pair: dict[str, int] = {}
    computed_prior: dict[str, int] = {}
    for left_index, left in enumerate(ORDERED_ROLES):
        for right in ORDERED_ROLES[left_index + 1 :]:
            for kind in _IDENTITY_KINDS:
                key = f"{left}|{right}|{kind}"
                computed_role_pair[key] = len(
                    identity_by_role[left][kind]
                    & identity_by_role[right][kind]
                )
    for role in ORDERED_ROLES:
        for kind in _IDENTITY_KINDS:
            key = f"{role}|prior|{kind}"
            computed_prior[key] = len(
                identity_by_role[role][kind] & prior_sets[kind]
            )
    if any(computed_role_pair.values()) or any(computed_prior.values()):
        raise FreshPanelIntegrityError("role/prior identity firewall is not zero")
    audit = manifest.get("overlap_audit")
    if not isinstance(audit, Mapping) or (
        audit.get("identity_kinds") != list(_IDENTITY_KINDS)
        or audit.get("role_pair_intersection_counts") != computed_role_pair
        or audit.get("prior_intersection_counts") != computed_prior
        or audit.get("all_intersection_counts_zero") is not True
        or audit.get("semantic_nonreuse_author_review_required") is not True
    ):
        raise FreshPanelIntegrityError("frozen overlap audit drifted")
    if manifest.get("future_report_boundary") != {
        "raw_prompt_text_allowed": False,
        "raw_choice_text_allowed": False,
        "raw_token_ids_allowed": False,
        "hashes_counts_spans_and_receipts_only": True,
    }:
        raise FreshPanelIntegrityError("future report boundary drifted")
    return manifest


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, object]:
    source = Path(path)
    encoded = _read_bytes(source, label="fresh prompt-free manifest")
    if _sha_bytes(encoded) != PINNED_MANIFEST_FILE_SHA256:
        raise FreshPanelIntegrityError("fresh manifest file bytes drifted")
    return validate_manifest(
        _decode_canonical_mapping(encoded, label="fresh prompt-free manifest")
    )


def score_objective_receipt_for_role(
    role: str,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Return the worker-facing full q/t objective receipt and its digest."""

    if role == RESERVE_ROLE:
        raise FreshPanelFirewallError(
            "reserve_unopened has no score-objective receipt"
        )
    if role not in EXECUTABLE_ROLES:
        raise FreshPanelFirewallError(f"unknown executable role {role!r}")
    frozen = validate_manifest(manifest)
    role_manifest = frozen["roles"][role]
    return {
        "role": role,
        "ordered_objective_row_catalog_sha256": role_manifest[
            "ordered_objective_row_catalog_sha256"
        ],
        "score_objective_receipt": _copy_mapping(
            role_manifest["score_objective_receipt"]
        ),
        "score_objective_receipt_sha256": role_manifest[
            "score_objective_receipt_sha256"
        ],
    }


def validate_receipt(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the blind-author, source, cost, and sequential firewall receipt."""

    if not isinstance(value, Mapping):
        raise FreshPanelIntegrityError("receipt must be a mapping")
    receipt = _copy_mapping(value)
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("format_version") != FORMAT_VERSION
        or receipt.get("panel_id") != PANEL_ID
        or receipt.get("artifact_role")
        != "frozen_prompt_free_panel_receipt"
    ):
        raise FreshPanelIntegrityError("fresh panel receipt identity drifted")
    observed_digest = receipt.get("receipt_sha256")
    if observed_digest != _domain_sha(
        "receipt", _without_digest(receipt, "receipt_sha256")
    ):
        raise FreshPanelIntegrityError("fresh panel receipt digest drifted")
    if observed_digest != PINNED_RECEIPT_SHA256:
        raise FreshPanelIntegrityError("fresh panel receipt public pin drifted")
    if receipt.get("manifest") != {
        "path": str(DEFAULT_MANIFEST),
        "file_sha256": PINNED_MANIFEST_FILE_SHA256,
        "manifest_sha256": PINNED_MANIFEST_SHA256,
    }:
        raise FreshPanelIntegrityError("receipt manifest binding drifted")
    if receipt.get("row_axis_contract") != ROW_AXIS_CONTRACT:
        raise FreshPanelIntegrityError("receipt q/t row-axis contract drifted")
    if receipt.get("stratum_contract") != STRATUM_CONTRACT:
        raise FreshPanelIntegrityError("receipt stratum contract drifted")
    if receipt.get("required_freeze_stage_kind_by_role") != (
        REQUIRED_FREEZE_STAGE_KIND_BY_ROLE
    ):
        raise FreshPanelIntegrityError("receipt freeze-stage kinds drifted")
    preflight = receipt.get("opaque_static_preflight")
    if not isinstance(preflight, Mapping):
        raise FreshPanelIntegrityError("opaque preflight receipt is missing")
    expected_pins = {
        role: {
            "path": str(DEFAULT_ROLE_CORPORA[role]),
            "file_sha256": PINNED_ROLE_FILE_SHA256S[role],
        }
        for role in ORDERED_ROLES
    }
    if (
        preflight.get("access_mode") != "byte_hash_only_no_json_decode"
        or preflight.get("all_files_must_match_before_any_semantic_open")
        is not True
        or preflight.get("opaque_hash_access_is_not_semantic_parse_access")
        is not True
        or preflight.get("role_file_pins") != expected_pins
    ):
        raise FreshPanelIntegrityError("opaque preflight contract drifted")
    firewall = receipt.get("sequential_semantic_firewall")
    expected_firewall = {
        "ordered_roles": list(ORDERED_ROLES),
        "A_open_requires": "successful_opaque_static_preflight",
        "B_open_requires": (
            "frozen_A_chart_atlas_and_transport_fit_receipt_sha256"
        ),
        "C_open_requires": "frozen_B_neighbor_candidate_catalog_sha256",
        "D_open_requires": (
            "frozen_C_chart_neighbor_graph_and_gate_receipt_sha256"
        ),
        "required_freeze_stage_kind_by_role": (
            REQUIRED_FREEZE_STAGE_KIND_BY_ROLE
        ),
        "reserve_open_authorized": False,
        "reserve_json_decode_authorized": False,
        "reserve_tokenizer_authorized": False,
        "reserve_model_authorized": False,
        "role_reopen_authorized": False,
    }
    if firewall != expected_firewall:
        raise FreshPanelIntegrityError("sequential semantic firewall drifted")
    authoring_access = receipt.get("authoring_access_receipt")
    expected_authoring_access = {
        "opaque_byte_hash_access": {
            role: True for role in ORDERED_ROLES
        },
        "semantic_parse_access": {
            role: role in EXECUTABLE_ROLES for role in ORDERED_ROLES
        },
        "tokenizer_access": {
            role: role in EXECUTABLE_ROLES for role in ORDERED_ROLES
        },
        "model_access": {role: False for role in ORDERED_ROLES},
        "opaque_hash_access_recorded_separately_from_semantic_parse_access": (
            True
        ),
    }
    if authoring_access != expected_authoring_access:
        raise FreshPanelIntegrityError("authoring access receipt drifted")
    if receipt.get("hard_long_neutrality_audit") != {
        "manifest_audit_sha256": PINNED_NEUTRALITY_AUDIT_SHA256,
        "group_count": 24,
        "all_overlap_counts_zero": True,
        "D_family_6_hard_suffix_remediated": True,
    }:
        raise FreshPanelIntegrityError("receipt hard neutrality binding drifted")
    cost = receipt.get("cost_surface")
    if not isinstance(cost, Mapping) or (
        cost.get("executable_role_example_count") != 48
        or cost.get("executable_role_family_count") != 6
        or cost.get("executable_role_examples_per_family") != 8
        or cost.get("per_executable_role_prompt_token_count") != 24_780
        or cost.get("per_executable_role_prediction_row_count") != 24_732
        or cost.get("all_executable_role_prompt_token_count") != 99_120
        or cost.get("all_executable_role_prediction_row_count") != 98_928
        or cost.get("A_transport_source_feature_width") != 136
        or cost.get("A_fit_only") is not True
        or cost.get("reserve_example_count") != 12
        or cost.get("reserve_prompt_token_count") is not None
        or cost.get("reserve_prediction_row_count") is not None
        or cost.get("reserve_token_cost_claimed") is not False
        or cost.get("D_choice_count_per_example") != 4
        or cost.get("D_choice_candidate_count") != 192
        or cost.get("D_all_choices_single_token_and_distinct") is not True
        or cost.get("A_fit_eligible_example_count") != 48
        or cost.get("A_fit_eligible_prediction_row_count") != 24_732
        or cost.get("per_B_C_pooled_eligible_example_count") != 42
        or cost.get("per_B_C_pooled_eligible_prediction_row_count") != 21_654
        or cost.get("per_B_C_bootstrap_draws_per_family") != 7
        or cost.get("per_B_C_LOFO_eligible_examples_per_family") != 7
        or cost.get(
            "per_B_C_short_boundary_diagnostic_reporting_example_count"
        )
        != 24
        or cost.get(
            "per_B_C_short_boundary_diagnostic_reporting_prediction_row_count"
        )
        != 9_768
        or cost.get("per_B_C_hard_long_veto_example_count") != 18
        or cost.get("per_B_C_hard_long_veto_prediction_row_count") != 11_886
        or cost.get("per_B_C_boundary_transition_514_example_count") != 6
        or cost.get(
            "per_B_C_boundary_transition_514_prediction_row_count"
        )
        != 3_078
        or cost.get("boundary_transition_514_eligible_for_confirmation")
        is not False
        or cost.get("D_response_reporting_example_count") != 48
        or cost.get("D_response_reporting_prediction_row_count") != 24_732
        or cost.get("D_graph_changing_example_count") != 0
    ):
        raise FreshPanelIntegrityError("exact cost surface drifted")
    ratio = cost.get("A_fit_row_to_feature_ratio")
    if not isinstance(ratio, (int, float)) or ratio != 24_732 / 136:
        raise FreshPanelIntegrityError("A row-to-feature ratio drifted")
    inventory = receipt.get("source_inventory")
    if not isinstance(inventory, Mapping):
        raise FreshPanelIntegrityError("source inventory is missing")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        raise FreshPanelIntegrityError("source inventory entries are missing")
    if inventory.get("entry_count") != len(entries):
        raise FreshPanelIntegrityError("source inventory count drifted")
    if inventory.get("inventory_sha256") != _domain_sha(
        "identity_catalog", entries
    ):
        raise FreshPanelIntegrityError("source inventory digest drifted")
    if any(
        not isinstance(entry, Mapping)
        or not isinstance(entry.get("path"), str)
        or _SHA256.fullmatch(str(entry.get("file_sha256"))) is None
        or not isinstance(entry.get("access_mode"), str)
        for entry in entries
    ):
        raise FreshPanelIntegrityError("source inventory entry is invalid")
    if any(".local-runs" in str(entry["path"]) for entry in entries):
        raise FreshPanelIntegrityError("source inventory contains local-run data")
    declaration = receipt.get("blind_author_declaration")
    if not isinstance(declaration, Mapping):
        raise FreshPanelIntegrityError("blind-author declaration is missing")
    false_fields = (
        "local_runs_opened",
        "result_report_opened",
        "chart_report_opened",
        "atlas_report_opened",
        "neighbor_report_opened",
        "candidate_report_opened",
        "distance_report_opened",
        "prior_scientific_outcome_opened_outside_frozen_setup",
        "model_weights_opened_or_loaded",
        "model_execution_performed",
        "tensor_library_imported",
    )
    true_fields = (
        "tokenizer_only_execution_performed",
        "model_config_only_read",
        "repository_dependency_config_only_read",
        "prior_prompt_panel_sources_used_only_for_nonoverlap_review",
        "all_five_raw_role_corpora_authored_before_any_experiment_outcome",
        "semantic_nonreuse_review_completed",
    )
    if any(declaration.get(field_name) is not False for field_name in false_fields):
        raise FreshPanelIntegrityError("blind-author negative declaration drifted")
    if any(declaration.get(field_name) is not True for field_name in true_fields):
        raise FreshPanelIntegrityError("blind-author positive declaration drifted")
    if not isinstance(declaration.get("declaration"), str):
        raise FreshPanelIntegrityError("blind-author prose declaration is missing")
    if receipt.get("raw_data_boundary") != {
        "raw_text_locations": [
            str(DEFAULT_ROLE_CORPORA[role]) for role in ORDERED_ROLES
        ],
        "manifest_contains_raw_text": False,
        "receipt_contains_raw_text": False,
        "future_reports_may_embed_raw_text": False,
        "future_reports_may_embed_raw_token_ids": False,
    }:
        raise FreshPanelIntegrityError("raw-data boundary drifted")
    claim_boundary = receipt.get("claim_boundary")
    if claim_boundary != {
        "panel_membership_and_tokenization_frozen": True,
        "model_executed": False,
        "chart_fit": False,
        "neighbor_discovered": False,
        "neighbor_confirmed": False,
        "response_validated": False,
        "scientific_result_present": False,
    }:
        raise FreshPanelIntegrityError("claim boundary drifted")
    return receipt


def load_receipt(path: Path | str = DEFAULT_RECEIPT) -> dict[str, object]:
    source = Path(path)
    encoded = _read_bytes(source, label="fresh blind panel receipt")
    if _sha_bytes(encoded) != PINNED_RECEIPT_FILE_SHA256:
        raise FreshPanelIntegrityError("fresh receipt file bytes drifted")
    return validate_receipt(
        _decode_canonical_mapping(encoded, label="fresh blind panel receipt")
    )


@dataclass(frozen=True, slots=True)
class OpaquePreflightReceipt:
    """Byte-hash evidence; construction never decodes a raw role corpus."""

    manifest_file_sha256: str
    role_file_sha256s: Mapping[str, str]
    reserve_semantically_decoded: bool = False

    def __post_init__(self) -> None:
        if self.manifest_file_sha256 != PINNED_MANIFEST_FILE_SHA256:
            raise FreshPanelIntegrityError("preflight manifest digest drifted")
        if dict(self.role_file_sha256s) != PINNED_ROLE_FILE_SHA256S:
            raise FreshPanelIntegrityError("preflight role digests drifted")
        if self.reserve_semantically_decoded:
            raise FreshPanelIntegrityError("reserve may not be decoded")


def opaque_preflight(
    *,
    repository_root: Path | str = Path("."),
    manifest: Mapping[str, object] | None = None,
    receipt: Mapping[str, object] | None = None,
) -> OpaquePreflightReceipt:
    """Validate prompt-free artifacts, then byte-hash raw files without JSON."""

    root = Path(repository_root)
    frozen_manifest = (
        load_manifest(root / DEFAULT_MANIFEST)
        if manifest is None
        else validate_manifest(manifest)
    )
    frozen_receipt = (
        load_receipt(root / DEFAULT_RECEIPT)
        if receipt is None
        else validate_receipt(receipt)
    )
    if frozen_receipt["manifest"]["manifest_sha256"] != frozen_manifest[
        "manifest_sha256"
    ]:
        raise FreshPanelIntegrityError("manifest/receipt binding drifted")
    observed: dict[str, str] = {}
    for role in ORDERED_ROLES:
        # This is deliberately a bytes-only read.  Do not route it through a
        # JSON helper, especially for reserve_unopened.
        encoded = _read_bytes(
            root / DEFAULT_ROLE_CORPORA[role],
            label=f"opaque {role} corpus bytes",
        )
        digest = _sha_bytes(encoded)
        if digest != PINNED_ROLE_FILE_SHA256S[role]:
            raise FreshPanelIntegrityError(f"{role} corpus bytes drifted")
        observed[role] = digest
    return OpaquePreflightReceipt(
        manifest_file_sha256=PINNED_MANIFEST_FILE_SHA256,
        role_file_sha256s=observed,
        reserve_semantically_decoded=False,
    )


def _validate_char_span(
    value: object,
    *,
    text_length: int,
    label: str,
) -> tuple[int, int]:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
        and 0 <= value[0] < value[1] <= text_length
    ):
        raise FreshPanelIntegrityError(f"{label} character span is invalid")
    return value[0], value[1]


def validate_open_role_corpus(
    value: Mapping[str, object],
    *,
    role: str,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate one already-authorized executable role; reserve is refused."""

    if role == RESERVE_ROLE:
        raise FreshPanelFirewallError("reserve_unopened may never be decoded")
    if role not in EXECUTABLE_ROLES:
        raise FreshPanelFirewallError(f"unknown executable role {role!r}")
    frozen_manifest = validate_manifest(manifest)
    corpus = _copy_mapping(value)
    if (
        corpus.get("schema") != ROLE_CORPUS_SCHEMA
        or corpus.get("format_version") != FORMAT_VERSION
        or corpus.get("panel_id") != PANEL_ID
        or corpus.get("role") != role
        or tuple(corpus.get("family_ids", ())) != FAMILY_IDS
        or tuple(corpus.get("family_semantics", ())) != FAMILY_SEMANTICS
        or corpus.get("examples_per_family") != 8
        or corpus.get("example_count") != 48
        or corpus.get("stratum_contract") != STRATUM_CONTRACT
    ):
        raise FreshPanelIntegrityError(f"{role} raw corpus header drifted")
    examples = corpus.get("examples")
    if not isinstance(examples, list) or len(examples) != 48:
        raise FreshPanelIntegrityError(f"{role} raw corpus needs 48 examples")
    role_manifest = frozen_manifest["roles"][role]
    members = role_manifest["ordered_members"]
    for index, (example, member) in enumerate(zip(examples, members)):
        if not isinstance(example, Mapping):
            raise FreshPanelIntegrityError(f"{role} raw example is invalid")
        family_index, within_family = divmod(index, 8)
        if (
            example.get("example_id") != member.get("example_id")
            or example.get("family_id") != FAMILY_IDS[family_index]
            or example.get("stratum") != STRATA[within_family]
            or example.get("target_token_count")
            != TARGET_TOKEN_COUNTS[within_family]
        ):
            raise FreshPanelIntegrityError(f"{role} raw member schedule drifted")
        target = TARGET_TOKEN_COUNTS[within_family]
        expected_eligibility = _expected_role_stratum_eligibility(role, target)
        if example.get("stratum_eligibility") != expected_eligibility:
            raise FreshPanelIntegrityError(f"{role} raw eligibility drifted")
        prompt = example.get("prompt")
        template_id = example.get("template_id")
        semantic_id = example.get("semantic_instance_id")
        paraphrase_key = example.get("paraphrase_key")
        if not all(
            isinstance(item, str) and item
            for item in (prompt, template_id, semantic_id, paraphrase_key)
        ):
            raise FreshPanelIntegrityError(f"{role} raw identity is invalid")
        evidence_chars = _validate_char_span(
            example.get("evidence_char_span"),
            text_length=len(prompt),
            label=f"{role} evidence",
        )
        suffix_chars = _validate_char_span(
            example.get("neutral_suffix_char_span"),
            text_length=len(prompt),
            label=f"{role} neutral suffix",
        )
        if evidence_chars[1] > suffix_chars[0]:
            raise FreshPanelIntegrityError(
                f"{role} evidence overlaps its neutral suffix"
            )
        expected = {
            "prompt_sha256": _text_sha("prompt", prompt),
            "canonical_prompt_sha256": _text_sha(
                "canonical", _canonical_prompt(prompt)
            ),
            "paraphrase_fingerprint_sha256": _domain_sha(
                "paraphrase", _paraphrase_signature(prompt)
            ),
            "template_id_sha256": _text_sha("template", template_id),
            "semantic_instance_sha256": _text_sha(
                "semantic", semantic_id + "\0" + paraphrase_key
            ),
            "evidence_text_sha256": _text_sha(
                "evidence", prompt[evidence_chars[0] : evidence_chars[1]]
            ),
            "neutral_suffix_sha256": _text_sha(
                "suffix", prompt[suffix_chars[0] : suffix_chars[1]]
            ),
        }
        for field_name, expected_value in expected.items():
            if member.get(field_name) != expected_value:
                raise FreshPanelIntegrityError(
                    f"{role} raw {field_name} differs from manifest"
                )
        if role == "D_response_validation":
            choice_texts = example.get("choice_texts")
            if (
                tuple(example.get("choices", ())) != CHOICE_LABELS
                or example.get("correct_choice_index")
                != member.get("correct_choice_index")
                or not isinstance(choice_texts, list)
                or len(choice_texts) != 4
                or not all(
                    isinstance(choice_text, str) and choice_text
                    for choice_text in choice_texts
                )
                or member.get("choice_text_sha256s")
                != [
                    _text_sha("choice_text", choice_text)
                    for choice_text in choice_texts
                ]
            ):
                raise FreshPanelIntegrityError("D raw choice ABI drifted")
        elif (
            "choices" in example
            or "correct_choice_index" in example
            or "choice_texts" in example
        ):
            raise FreshPanelIntegrityError(f"{role} unexpectedly has choices")
    return corpus


@dataclass(slots=True)
class SequentialPanelSource:
    """One-shot semantic-open state machine over the four executable roles."""

    repository_root: Path
    manifest: Mapping[str, object]
    receipt: Mapping[str, object]
    preflight: OpaquePreflightReceipt
    _next_index: int = 0
    _current_role: str | None = None
    _opened_roles: set[str] = field(default_factory=set)
    _freeze_receipts: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_defaults(
        cls,
        repository_root: Path | str = Path("."),
    ) -> "SequentialPanelSource":
        root = Path(repository_root)
        manifest = load_manifest(root / DEFAULT_MANIFEST)
        receipt = load_receipt(root / DEFAULT_RECEIPT)
        preflight = opaque_preflight(
            repository_root=root,
            manifest=manifest,
            receipt=receipt,
        )
        return cls(
            repository_root=root,
            manifest=manifest,
            receipt=receipt,
            preflight=preflight,
        )

    @property
    def next_role(self) -> str | None:
        if self._next_index >= len(EXECUTABLE_ROLES):
            return None
        return EXECUTABLE_ROLES[self._next_index]

    @property
    def opened_roles(self) -> tuple[str, ...]:
        """Return the prompt-free semantic-open role inventory in ABI order."""

        return tuple(role for role in EXECUTABLE_ROLES if role in self._opened_roles)

    @property
    def current_role(self) -> str | None:
        """Return the currently open semantic role without exposing its data."""

        return self._current_role

    @property
    def freeze_receipts(self) -> dict[str, dict[str, str]]:
        return {
            role: dict(stage_receipt)
            for role, stage_receipt in self._freeze_receipts.items()
        }

    @property
    def required_freeze_stage_kind(self) -> str | None:
        if self._current_role is None:
            return None
        return REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[self._current_role]

    def open_role(self, role: str) -> dict[str, object]:
        if role == RESERVE_ROLE:
            raise FreshPanelFirewallError(
                "reserve_unopened has no semantic-open capability"
            )
        if role not in EXECUTABLE_ROLES:
            raise FreshPanelFirewallError(f"unknown executable role {role!r}")
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
        encoded = _read_bytes(
            self.repository_root / DEFAULT_ROLE_CORPORA[role],
            label=f"authorized {role} corpus",
        )
        if _sha_bytes(encoded) != PINNED_ROLE_FILE_SHA256S[role]:
            raise FreshPanelIntegrityError(f"{role} corpus bytes drifted")
        raw = _decode_canonical_mapping(
            encoded,
            label=f"authorized {role} corpus",
        )
        opened = validate_open_role_corpus(
            raw,
            role=role,
            manifest=self.manifest,
        )
        self._opened_roles.add(role)
        self._current_role = role
        return opened

    def freeze_current(
        self,
        stage_kind: str,
        stage_receipt_sha256: str,
    ) -> None:
        if self._current_role is None:
            raise FreshPanelFirewallError("no semantic role is open")
        role = self._current_role
        expected_stage_kind = REQUIRED_FREEZE_STAGE_KIND_BY_ROLE[role]
        if stage_kind != expected_stage_kind:
            raise FreshPanelFirewallError(
                f"{role} requires freeze stage kind {expected_stage_kind!r}, "
                f"not {stage_kind!r}"
            )
        digest = _require_sha256(
            stage_receipt_sha256,
            label=f"{role} frozen stage receipt",
        )
        self._freeze_receipts[role] = {
            "stage_kind": stage_kind,
            "stage_receipt_sha256": digest,
        }
        self._current_role = None
        self._next_index += 1

    def open_reserve(self) -> None:
        raise FreshPanelFirewallError(
            "reserve_unopened is byte-pinned but may never be decoded"
        )


def _encoding_parts(
    tokenizer: object,
    text: str,
    *,
    add_special_tokens: bool,
) -> tuple[list[int], list[tuple[int, int]]]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError(
            "tokenizer adapter must expose encode(text, add_special_tokens=)"
        )
    encoding = encode(text, add_special_tokens=add_special_tokens)
    ids = getattr(encoding, "ids", None)
    offsets = getattr(encoding, "offsets", None)
    if not isinstance(ids, list) or not all(
        isinstance(item, int) and item >= 0 for item in ids
    ):
        raise TypeError("tokenizer encoding.ids must be a list of integers")
    if not isinstance(offsets, list) or not all(
        isinstance(item, tuple)
        and len(item) == 2
        and all(isinstance(part, int) for part in item)
        for item in offsets
    ):
        raise TypeError("tokenizer encoding.offsets must be character pairs")
    if len(ids) != len(offsets):
        raise TypeError("tokenizer IDs and offsets must have equal length")
    return ids, offsets


def _token_span(
    offsets: Sequence[tuple[int, int]],
    chars: tuple[int, int],
) -> list[int]:
    start, end = chars
    hits = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]
    if not hits:
        raise FreshPanelIntegrityError("character span has no tokenizer rows")
    return [hits[0], hits[-1] + 1]


def _contains_subsequence(
    haystack: Sequence[int],
    needle: Sequence[int],
) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        list(haystack[index : index + len(needle)]) == list(needle)
        for index in range(len(haystack) - len(needle) + 1)
    )


def _whole_phrase_present(text: str, phrase: str) -> bool:
    text_words = _canonical_prompt(text).split()
    phrase_words = _canonical_prompt(phrase).split()
    if not phrase_words:
        return False
    return any(
        text_words[index : index + len(phrase_words)] == phrase_words
        for index in range(len(text_words) - len(phrase_words) + 1)
    )


def authenticate_executable_tokenization(
    *,
    role: str,
    corpus: Mapping[str, object],
    manifest: Mapping[str, object],
    tokenizer: object,
) -> dict[str, object]:
    """Replay exact token lengths, rows, evidence, suffixes, and D choices."""

    if role == RESERVE_ROLE:
        raise FreshPanelFirewallError(
            "reserve_unopened may never reach a tokenizer"
        )
    opened = validate_open_role_corpus(
        corpus,
        role=role,
        manifest=manifest,
    )
    frozen_manifest = validate_manifest(manifest)
    role_manifest = frozen_manifest["roles"][role]
    role_rows: list[dict[str, object]] = []
    token_count = 0
    prediction_rows = 0
    hard_audit_inputs: dict[str, list[dict[str, object]]] = {
        family: [] for family in FAMILY_IDS
    }
    for example, member in zip(
        opened["examples"], role_manifest["ordered_members"]
    ):
        prompt = example["prompt"]
        ids, offsets = _encoding_parts(
            tokenizer,
            prompt,
            add_special_tokens=True,
        )
        target = member["target_token_count"]
        if len(ids) != target or member["observed_token_count"] != target:
            raise FreshPanelIntegrityError(
                f"{role} exact tokenizer length drifted for {member['example_id']}"
            )
        if not ids or ids[0] != 2:
            raise FreshPanelIntegrityError("pinned BOS token is missing")
        if _token_ids_sha(ids) != member["token_ids_sha256"]:
            raise FreshPanelIntegrityError("token ID stream digest drifted")
        evidence_chars = tuple(example["evidence_char_span"])
        suffix_chars = tuple(example["neutral_suffix_char_span"])
        if _token_span(offsets, evidence_chars) != member[
            "evidence_token_span"
        ]:
            raise FreshPanelIntegrityError("evidence token span drifted")
        if _token_span(offsets, suffix_chars) != member[
            "neutral_suffix_token_span"
        ]:
            raise FreshPanelIntegrityError("neutral suffix token span drifted")
        if target >= 576:
            suffix_text = prompt[suffix_chars[0] : suffix_chars[1]]
            suffix_ids, _suffix_offsets = _encoding_parts(
                tokenizer,
                suffix_text,
                add_special_tokens=False,
            )
            if len(suffix_ids) != 512:
                raise FreshPanelIntegrityError(
                    "hard answer-neutral suffix ceased to be final512"
                )
            hard_audit_inputs[member["family_id"]].append(
                {
                    "suffix_text": suffix_text,
                    "suffix_ids": suffix_ids,
                    "choice_texts": list(example.get("choice_texts", ())),
                }
            )
        rows = [
            {
                "role": role,
                "family_id": member["family_id"],
                "example_id": member["example_id"],
                "prediction_query_position": query_position,
                "target_label_position": query_position + 1,
            }
            for query_position in range(len(ids) - 1)
        ]
        row_stream_sha256 = _domain_sha("row_stream", rows)
        if row_stream_sha256 != member["row_stream_sha256"]:
            raise FreshPanelIntegrityError("example row stream digest drifted")
        if row_stream_sha256 != member["objective_row_stream_sha256"]:
            raise FreshPanelIntegrityError(
                "example objective row stream digest drifted"
            )
        role_rows.extend(rows)
        token_count += len(ids)
        prediction_rows += len(ids) - 1
        if role == "D_response_validation":
            choices = example["choices"]
            bare_ids = [
                _encoding_parts(
                    tokenizer,
                    choice,
                    add_special_tokens=False,
                )[0]
                for choice in choices
            ]
            append_ids = [
                _encoding_parts(
                    tokenizer,
                    " " + choice,
                    add_special_tokens=False,
                )[0]
                for choice in choices
            ]
            if any(len(values) != 1 for values in bare_ids + append_ids):
                raise FreshPanelIntegrityError("D choice ceased to be one token")
            bare_values = [values[0] for values in bare_ids]
            append_values = [values[0] for values in append_ids]
            if len(set(bare_values)) != 4 or len(set(append_values)) != 4:
                raise FreshPanelIntegrityError("D choice token IDs are not distinct")
            if _domain_sha("choice_ids", bare_values) != member[
                "bare_choice_token_ids_sha256"
            ]:
                raise FreshPanelIntegrityError("D bare choice digest drifted")
            if _domain_sha("choice_ids", append_values) != member[
                "append_choice_token_ids_sha256"
            ]:
                raise FreshPanelIntegrityError("D append choice digest drifted")
            choice_texts = example["choice_texts"]
            bare_choice_text_ids = [
                _encoding_parts(
                    tokenizer,
                    choice_text,
                    add_special_tokens=False,
                )[0]
                for choice_text in choice_texts
            ]
            append_choice_text_ids = [
                _encoding_parts(
                    tokenizer,
                    " " + choice_text,
                    add_special_tokens=False,
                )[0]
                for choice_text in choice_texts
            ]
            if member["choice_text_bare_token_ids_sha256s"] != [
                _token_ids_sha(values) for values in bare_choice_text_ids
            ]:
                raise FreshPanelIntegrityError(
                    "D bare choice-text token stream drifted"
                )
            if member["choice_text_append_token_ids_sha256s"] != [
                _token_ids_sha(values) for values in append_choice_text_ids
            ]:
                raise FreshPanelIntegrityError(
                    "D append choice-text token stream drifted"
                )
    frozen_groups = frozen_manifest["hard_long_neutrality_audit"]["groups"]
    for family, entries in hard_audit_inputs.items():
        if len(entries) != 3:
            raise FreshPanelIntegrityError(
                f"{role}/{family} hard neutrality group is incomplete"
            )
        suffix_texts = {str(entry["suffix_text"]) for entry in entries}
        suffix_id_streams = {
            tuple(entry["suffix_ids"]) for entry in entries
        }
        if len(suffix_texts) != 1 or len(suffix_id_streams) != 1:
            raise FreshPanelIntegrityError(
                f"{role}/{family} hard suffix differs within group"
            )
        suffix_text = next(iter(suffix_texts))
        suffix_ids = list(next(iter(suffix_id_streams)))
        choice_texts = [
            choice_text
            for entry in entries
            for choice_text in entry["choice_texts"]
        ]
        choice_sequences: list[list[int]] = []
        for choice_text in choice_texts:
            choice_sequences.extend(
                (
                    _encoding_parts(
                        tokenizer,
                        choice_text,
                        add_special_tokens=False,
                    )[0],
                    _encoding_parts(
                        tokenizer,
                        " " + choice_text,
                        add_special_tokens=False,
                    )[0],
                )
            )
        label_sequences: list[list[int]] = []
        if role == "D_response_validation":
            for label in CHOICE_LABELS:
                label_sequences.extend(
                    (
                        _encoding_parts(
                            tokenizer,
                            label,
                            add_special_tokens=False,
                        )[0],
                        _encoding_parts(
                            tokenizer,
                            " " + label,
                            add_special_tokens=False,
                        )[0],
                    )
                )
        cue_sequences: list[list[int]] = []
        for cue in SEMANTIC_CUES_BY_FAMILY[family]:
            cue_sequences.extend(
                (
                    _encoding_parts(
                        tokenizer,
                        cue,
                        add_special_tokens=False,
                    )[0],
                    _encoding_parts(
                        tokenizer,
                        " " + cue,
                        add_special_tokens=False,
                    )[0],
                )
            )
        raw_choice_overlap = sum(
            _whole_phrase_present(suffix_text, choice_text)
            for choice_text in choice_texts
        )
        choice_token_overlap = sum(
            _contains_subsequence(suffix_ids, values)
            for values in choice_sequences
        )
        label_token_overlap = sum(
            _contains_subsequence(suffix_ids, values)
            for values in label_sequences
        )
        raw_cue_overlap = sum(
            _whole_phrase_present(suffix_text, cue)
            for cue in SEMANTIC_CUES_BY_FAMILY[family]
        )
        cue_token_overlap = sum(
            _contains_subsequence(suffix_ids, values)
            for values in cue_sequences
        )
        expected_group = {
            "role": role,
            "family_id": family,
            "hard_example_count": 3,
            "shared_suffix_sha256": _text_sha("suffix", suffix_text),
            "corresponding_choice_text_count": len(choice_texts),
            "corresponding_choice_text_sha256s": sorted(
                {_text_sha("choice_text", value) for value in choice_texts}
            ),
            "corresponding_choice_token_sequence_sha256s": sorted(
                {_token_ids_sha(values) for values in choice_sequences}
            ),
            "response_choice_label_token_sequence_sha256s": sorted(
                {_token_ids_sha(values) for values in label_sequences}
            ),
            "semantic_cue_sha256s": sorted(
                _text_sha("choice_text", cue)
                for cue in SEMANTIC_CUES_BY_FAMILY[family]
            ),
            "raw_choice_text_overlap_count": raw_choice_overlap,
            "choice_token_subsequence_overlap_count": choice_token_overlap,
            "response_choice_label_token_overlap_count": label_token_overlap,
            "raw_semantic_cue_overlap_count": raw_cue_overlap,
            "semantic_cue_token_subsequence_overlap_count": cue_token_overlap,
            "all_overlap_counts_zero": not any(
                (
                    raw_choice_overlap,
                    choice_token_overlap,
                    label_token_overlap,
                    raw_cue_overlap,
                    cue_token_overlap,
                )
            ),
        }
        if expected_group != frozen_groups[f"{role}|{family}"]:
            raise FreshPanelIntegrityError(
                f"{role}/{family} hard neutrality tokenizer audit drifted"
            )
    objective_catalog_sha256 = _domain_sha("row_stream", role_rows)
    if (
        token_count != 24_780
        or prediction_rows != 24_732
        or objective_catalog_sha256
        != role_manifest["ordered_row_catalog_sha256"]
        or objective_catalog_sha256
        != role_manifest["ordered_objective_row_catalog_sha256"]
    ):
        raise FreshPanelIntegrityError(f"{role} aggregate row receipt drifted")
    return {
        "role": role,
        "example_count": 48,
        "prompt_token_count": token_count,
        "prediction_row_count": prediction_rows,
        "ordered_objective_row_catalog_sha256": (
            objective_catalog_sha256
        ),
        "score_objective_receipt_sha256": role_manifest[
            "score_objective_receipt_sha256"
        ],
        "carrier_chart_signature_key_fields": ROW_AXIS_CONTRACT[
            "carrier_chart_signature_key_fields"
        ],
        "hard_long_neutrality_groups_authenticated": len(FAMILY_IDS),
        "hard_long_neutrality_audit_sha256": (
            PINNED_NEUTRALITY_AUDIT_SHA256
        ),
        "raw_prompt_text_present": False,
        "raw_token_ids_present": False,
    }


def authenticate_tokenizer_assets(
    tokenizer_root: Path | str,
) -> dict[str, str]:
    """Byte-authenticate tokenizer/config assets; never opens model weights."""

    root = Path(tokenizer_root)
    expected = {
        "tokenizer.json": PINNED_TOKENIZER_JSON_SHA256,
        "tokenizer_config.json": PINNED_TOKENIZER_CONFIG_SHA256,
        "special_tokens_map.json": PINNED_SPECIAL_TOKENS_MAP_SHA256,
        "config.json": PINNED_MODEL_CONFIG_SHA256,
    }
    observed: dict[str, str] = {}
    for name, digest in expected.items():
        encoded = _read_bytes(root / name, label=f"pinned {name}")
        observed[name] = _sha_bytes(encoded)
        if observed[name] != digest:
            raise FreshPanelIntegrityError(f"pinned {name} bytes drifted")
    return observed


def check_frozen_panels(
    repository_root: Path | str = Path("."),
) -> OpaquePreflightReceipt:
    """Run the model-free, raw-corpus-opaque default authentication path."""

    return opaque_preflight(repository_root=repository_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    args = parser.parse_args(argv)
    check_frozen_panels(args.repository_root)
    return 0


__all__ = [
    "CHOICE_LABELS",
    "DEFAULT_MANIFEST",
    "DEFAULT_RECEIPT",
    "DEFAULT_ROLE_CORPORA",
    "EXECUTABLE_ROLES",
    "FAMILY_IDS",
    "FAMILY_SEMANTICS",
    "FORMAT_VERSION",
    "FreshPanelFirewallError",
    "FreshPanelIntegrityError",
    "MANIFEST_SCHEMA",
    "OpaquePreflightReceipt",
    "ORDERED_ROLES",
    "PANEL_ID",
    "PINNED_MANIFEST_FILE_SHA256",
    "PINNED_MANIFEST_SHA256",
    "PINNED_NEUTRALITY_AUDIT_SHA256",
    "PINNED_RECEIPT_FILE_SHA256",
    "PINNED_RECEIPT_SHA256",
    "PINNED_ROLE_FILE_SHA256S",
    "RECEIPT_SCHEMA",
    "REQUIRED_FREEZE_STAGE_KIND_BY_ROLE",
    "RESERVE_ROLE",
    "ROW_AXIS_CONTRACT",
    "ROLE_CORPUS_SCHEMA",
    "SEMANTIC_CUES_BY_FAMILY",
    "STRATA",
    "STRATUM_CONTRACT",
    "SequentialPanelSource",
    "TARGET_TOKEN_COUNTS",
    "authenticate_executable_tokenization",
    "authenticate_tokenizer_assets",
    "canonical_json_bytes",
    "check_frozen_panels",
    "load_manifest",
    "load_receipt",
    "main",
    "opaque_preflight",
    "score_objective_receipt_for_role",
    "validate_manifest",
    "validate_open_role_corpus",
    "validate_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
