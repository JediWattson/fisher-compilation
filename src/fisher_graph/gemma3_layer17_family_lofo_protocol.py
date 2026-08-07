"""Source-safe family-LOFO protocol for the frozen Gemma layer-17 arm.

This module is deliberately a pure planner.  It performs no filesystem,
prompt, tokenizer, model, tensor, or metric access.  The default artifact
freezes one first arm (cap 48, generator rank 16, edgeless) and eight
leave-one-family-out folds over the authenticated v8 Calibration-A fit role.

An optional authentication entry point accepts the already prompt-free v8
corpus artifact as an in-memory mapping.  It verifies the corpus and role
manifest commitments, derives the fold membership commitments, and returns
the same frozen public protocol.  Per-example identities are consumed only
inside that verification call and are never emitted by the protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re


__all__ = [
    "FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256",
    "GEMMA3_LAYER17_FAMILY_LOFO_PROTOCOL_FORMAT_VERSION",
    "GEMMA3_LAYER17_FAMILY_LOFO_PROTOCOL_SCHEMA",
    "V8_FAMILY_LOFO_FAMILY_ALIASES",
    "V8_FAMILY_LOFO_ROLES",
    "build_authenticated_v8_layer17_family_lofo_protocol",
    "build_default_v8_layer17_family_lofo_protocol",
    "validate_v8_layer17_family_lofo_protocol",
]


GEMMA3_LAYER17_FAMILY_LOFO_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_layer17_family_lofo_protocol"
)
GEMMA3_LAYER17_FAMILY_LOFO_PROTOCOL_FORMAT_VERSION = 1

_PROTOCOL_DOMAIN = (
    b"fisher-graph:gemma3-layer17-family-lofo-protocol:v1\0"
)
_FOLD_DOMAIN = b"fisher-graph:gemma3-layer17-family-lofo-fold:v1\0"
_MEMBERSHIP_DOMAIN = (
    b"fisher-graph:gemma3-layer17-family-lofo-membership:v1\0"
)
_FAMILY_ALIAS_MAPPING_DOMAIN = (
    b"fisher-graph:gemma3-layer17-family-lofo-alias-map:v1\0"
)
_SOURCE_ROLE_MANIFEST_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-progressive-a-role-manifest:v1\0"
)
_SOURCE_CORPUS_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-progressive-a-corpus:v1\0"
)
_SOURCE_CORPUS_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_progressive_a_corpus"
)
_SOURCE_ROLE_MANIFEST_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_progressive_a_role_manifest"
)
_SOURCE_CORPUS_FORMAT_VERSION = 1
_MEMBERSHIP_SCHEMA = (
    "fisher_graph.gemma3_layer17_family_lofo_membership"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")

V8_FAMILY_LOFO_ROLES = (
    "calibration_a_fit",
    "calibration_a_selection",
    "calibration_a_guard",
)
V8_FAMILY_LOFO_FAMILY_ALIASES = tuple(
    f"family_{index:02d}" for index in range(8)
)
_V8_FIT_FAMILY_IDS = (
    "structured-strong-v8-calibration_a-coral-density-band-chronicle",
    "structured-strong-v8-calibration_a-handloom-warp-tension-map",
    "structured-strong-v8-calibration_a-magnetotelluric-impedance-sounding",
    "structured-strong-v8-calibration_a-meteorite-widmanstatten-trace",
    "structured-strong-v8-calibration_a-papyrus-fiber-registration",
    "structured-strong-v8-calibration_a-pollen-exine-acetolysis",
    "structured-strong-v8-calibration_a-qanat-seepage-gradient",
    "structured-strong-v8-calibration_a-wax-cylinder-groove-eccentricity",
)

_V8_CORPUS_ID = "gemma3-layer10-shape-flow-v8-v1"
_V8_PROFILE = "full"
_V8_CORPUS_ARTIFACT_SHA256 = (
    "2f92a087512966f0af80247ab4d54370266295eff2228c65ab82695c08f8c6c7"
)
_V8_TOKENIZER_CONTRACT_SHA256 = (
    "ab3f2156c74c0ce2c97e83d4d987b4796051bf6774a10b4327972bc01f92dcab"
)
_V8_FORBIDDEN_CALIBRATION_B_MANIFEST_SHA256 = (
    "986ee9da505fb056853f4fc7ed4f5eee6e9313f0419f2ca9ebc54e0df8607bdd"
)
_V8_ROLE_BINDINGS = {
    "calibration_a_fit": {
        "manifest_sha256": (
            "ff1f80944b2d166641bdb14e88690927873e77f4dd1d84c457cf7304b5096564"
        ),
        "source_file_sha256": (
            "f79df1f6d295cdce5f6951a3b289c1cf05e00db3fe1b32ba3b244812c6048029"
        ),
        "example_count": 256,
        "family_count": 8,
    },
    "calibration_a_selection": {
        "manifest_sha256": (
            "c785589a591dca5c4d39b802afaf47271150c2452ad7f4e12622871943f2d0c5"
        ),
        "source_file_sha256": (
            "2a67eae43dc168468ded1cc260fca682f3dd3b9b151b833dddeaa96860b52282"
        ),
        "example_count": 128,
        "family_count": 4,
    },
    "calibration_a_guard": {
        "manifest_sha256": (
            "63740aff1f134e52a754a3f641466225b5d8a1bd9e10203dccfd3c1d06c3dfd6"
        ),
        "source_file_sha256": (
            "8509d975c4bf7885befe32223a6af08a43d399afd6b4c36927b39c99d24de0a2"
        ),
        "example_count": 128,
        "family_count": 4,
    },
}

_FIT_MEMBERSHIP_SHA256 = (
    "b14a58b8c5cd0f8dac7596f7765cbe4a140c4a8a22122d10feb362a84121011f"
)
_FROZEN_FOLD_MEMBERSHIP_BINDINGS = (
    (
        _V8_FIT_FAMILY_IDS[0],
        "30f0be1b53fe73b659012bf571ddecf689c005e54db49bef6047514ccb4e9b0a",
        "73c5ed4efd5d801e6a4f3f72799990f2bacdda87b46b35629796d4824d11a465",
    ),
    (
        _V8_FIT_FAMILY_IDS[1],
        "ba27ff12d07faa1169eb4febb295fc980c1b417f36cb75ede55bc43169d23d53",
        "c77e37f170f8047ae68f6c14121ffb52d4421d125a77b67764995dd433755c5a",
    ),
    (
        _V8_FIT_FAMILY_IDS[2],
        "ebde5254f719e0be8756479705042a4bdf5e6c0a6dc492953395cb6a61d042e4",
        "6eda21320c65cdf941f574938c1ae48bfe4fe1b10336c7b0187dec8b4ae5d3b1",
    ),
    (
        _V8_FIT_FAMILY_IDS[3],
        "c2f6f2e73a4644d21d864a0cfb4042d0aaa41f2ca7e0ffe5f3f9d8f774d83f4e",
        "bbe50067671001da80f9b768b1fe354f2d1c75f3e694594cfd4d2dc21f7b6936",
    ),
    (
        _V8_FIT_FAMILY_IDS[4],
        "a74e4e0db1363c09af2fea7a76f7a25d82952bdf78c6242806038f73251fec00",
        "91a17ca4d7b95244b35ea688798cdb65ee304ff4412d7013a81b01e8d70a7a80",
    ),
    (
        _V8_FIT_FAMILY_IDS[5],
        "cabf6a2e5adea35b6db9acd3aaedc057b9c7dccb5480b132ab03c7cf81f73e92",
        "0db08a1dcc5c9fb98591d221764f6ba84873b34efeb014d78dcb32f450bc9b49",
    ),
    (
        _V8_FIT_FAMILY_IDS[6],
        "7b004500ebfd301f33fd6a59fcbbbbb5505790576af53d647555debdc71b8776",
        "a4b116a5ef743350eda2151163c54eb99604dd2be1b247f9a2a2844de783f8c5",
    ),
    (
        _V8_FIT_FAMILY_IDS[7],
        "f93fa938691cc88ab096bf2a36a977e9e1f6d5fbeb647e241e82a456c5541a72",
        "4a73cba030ae1015ee83a94aa975367ccb9a9c711e769e082b814b35924198ef",
    ),
)

_LAYER17_FRAGMENT_IDS = (
    "cluster.0/layer.17",
    "cluster.28/layer.17",
    "cluster.34/layer.17",
    "cluster.54/layer.17",
)
_LAYER17_NATIVE_MODE_COUNTS = (54, 38, 85, 53)
_LAYER17_TOPOLOGY_SHA256 = (
    "f5bacb5bef861ac2dbc95e4ccd9baf8f749ed04b347106c7a9815364f847bad8"
)

# Validation requires this exact identity, so protocol fields cannot be
# retuned after outcomes.
FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256 = (
    "e619bfe4a0aa32625f5acc89e4aba3d2d55cc6849b974cb06745609ca5cf38d9"
)

_CORPUS_FIELDS = {
    "schema",
    "format_version",
    "corpus_id",
    "profile",
    "tokenizer_contract_sha256",
    "forbidden_assessment_manifest_sha256s",
    "roles",
    "artifact_sha256",
}
_ROLE_VIEW_FIELDS = {
    "role",
    "manifest_sha256",
    "role_input_file_sha256",
    "example_count",
    "family_ids",
    "ordered_prompt_sha256s",
    "ordered_family_ids",
}
_PROTOCOL_FIELDS = {
    "schema",
    "format_version",
    "corpus_authority",
    "role_bindings",
    "first_arm",
    "folds",
    "gates",
    "evaluation_contract",
    "claim_boundary",
    "safety",
    "artifact_sha256",
}


def _canonical_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("canonical JSON does not permit non-finite values")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {
            key: _canonical_json_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_canonical_json_value(item) for item in value]
    raise TypeError("protocols must contain only strict JSON values")


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


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable nonempty identifier")
    return value


def _strict_sequence(value: object, *, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _strict_mapping(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _role_manifest_sha256(
    *,
    role: str,
    ordered_identity_sha256s: tuple[str, ...],
    ordered_family_ids: tuple[str, ...],
) -> str:
    payload = {
        "schema": _SOURCE_ROLE_MANIFEST_SCHEMA,
        "format_version": 1,
        "corpus_id": _V8_CORPUS_ID,
        "profile": _V8_PROFILE,
        "role": role,
        "tokenizer_contract_sha256": _V8_TOKENIZER_CONTRACT_SHA256,
        "ordered_members": [
            {
                "prompt_sha256": identity_sha256,
                "family_id": family_id,
            }
            for identity_sha256, family_id in zip(
                ordered_identity_sha256s,
                ordered_family_ids,
                strict=True,
            )
        ],
    }
    return _domain_sha256(_SOURCE_ROLE_MANIFEST_DOMAIN, payload)


def _validate_role_view(
    raw: object,
    *,
    expected_role: str,
) -> dict[str, object]:
    view = _strict_mapping(
        raw,
        fields=_ROLE_VIEW_FIELDS,
        label=f"{expected_role} role view",
    )
    if view["role"] != expected_role:
        raise ValueError(f"{expected_role} role name drifted")
    binding = _V8_ROLE_BINDINGS[expected_role]
    manifest_sha256 = _require_sha256(
        view["manifest_sha256"], label=f"{expected_role} manifest"
    )
    source_file_sha256 = _require_sha256(
        view["role_input_file_sha256"],
        label=f"{expected_role} source file",
    )
    example_count = view["example_count"]
    if type(example_count) is not int or example_count <= 0:
        raise ValueError(f"{expected_role} example_count must be positive")
    if (
        manifest_sha256 != binding["manifest_sha256"]
        or source_file_sha256 != binding["source_file_sha256"]
        or example_count != binding["example_count"]
    ):
        raise ValueError(f"{expected_role} frozen authority binding drifted")

    family_ids_raw = _strict_sequence(
        view["family_ids"], label=f"{expected_role} family_ids"
    )
    family_ids = tuple(
        _require_identifier(value, label=f"{expected_role} family")
        for value in family_ids_raw
    )
    if (
        family_ids != tuple(sorted(set(family_ids)))
        or len(family_ids) != binding["family_count"]
    ):
        raise ValueError(
            f"{expected_role} family_ids must be exact sorted unique families"
        )

    identities_raw = _strict_sequence(
        view["ordered_prompt_sha256s"],
        label=f"{expected_role} ordered identities",
    )
    ordered_identity_sha256s = tuple(
        _require_sha256(value, label=f"{expected_role} example identity")
        for value in identities_raw
    )
    ordered_families_raw = _strict_sequence(
        view["ordered_family_ids"],
        label=f"{expected_role} ordered families",
    )
    ordered_family_ids = tuple(
        _require_identifier(value, label=f"{expected_role} ordered family")
        for value in ordered_families_raw
    )
    if (
        len(ordered_identity_sha256s) != example_count
        or len(set(ordered_identity_sha256s)) != example_count
        or len(ordered_family_ids) != example_count
        or tuple(sorted(set(ordered_family_ids))) != family_ids
    ):
        raise ValueError(f"{expected_role} ordered membership is invalid")
    expected_manifest_sha256 = _role_manifest_sha256(
        role=expected_role,
        ordered_identity_sha256s=ordered_identity_sha256s,
        ordered_family_ids=ordered_family_ids,
    )
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"{expected_role} manifest does not authenticate ordered membership"
        )
    return {
        "role": expected_role,
        "manifest_sha256": manifest_sha256,
        "source_file_sha256": source_file_sha256,
        "example_count": example_count,
        "family_ids": family_ids,
        "ordered_identity_sha256s": ordered_identity_sha256s,
        "ordered_family_ids": ordered_family_ids,
    }


def _authenticate_v8_corpus(
    corpus_artifact: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    corpus = _strict_mapping(
        corpus_artifact,
        fields=_CORPUS_FIELDS,
        label="v8 prompt-free corpus artifact",
    )
    if (
        corpus["schema"] != _SOURCE_CORPUS_SCHEMA
        or type(corpus["format_version"]) is not int
        or corpus["format_version"] != _SOURCE_CORPUS_FORMAT_VERSION
        or corpus["corpus_id"] != _V8_CORPUS_ID
        or corpus["profile"] != _V8_PROFILE
        or corpus["tokenizer_contract_sha256"]
        != _V8_TOKENIZER_CONTRACT_SHA256
    ):
        raise ValueError("v8 prompt-free corpus header drifted")
    forbidden = _strict_sequence(
        corpus["forbidden_assessment_manifest_sha256s"],
        label="forbidden assessment manifests",
    )
    if forbidden != (_V8_FORBIDDEN_CALIBRATION_B_MANIFEST_SHA256,):
        raise ValueError("frozen Calibration-B exclusion drifted")
    supplied_artifact_sha256 = _require_sha256(
        corpus["artifact_sha256"], label="v8 corpus artifact"
    )
    corpus_payload = {
        key: corpus[key] for key in _CORPUS_FIELDS if key != "artifact_sha256"
    }
    recomputed_artifact_sha256 = _domain_sha256(
        _SOURCE_CORPUS_DOMAIN, corpus_payload
    )
    if supplied_artifact_sha256 != recomputed_artifact_sha256:
        raise ValueError("v8 prompt-free corpus artifact hash mismatch")
    if supplied_artifact_sha256 != _V8_CORPUS_ARTIFACT_SHA256:
        raise ValueError("v8 prompt-free corpus differs from frozen authority")

    roles_raw = corpus["roles"]
    if not isinstance(roles_raw, Mapping) or set(roles_raw) != set(
        V8_FAMILY_LOFO_ROLES
    ):
        raise ValueError("v8 corpus role names drifted")
    roles = {
        role: _validate_role_view(roles_raw[role], expected_role=role)
        for role in V8_FAMILY_LOFO_ROLES
    }
    identity_sets = [
        set(role["ordered_identity_sha256s"]) for role in roles.values()
    ]
    family_sets = [set(role["family_ids"]) for role in roles.values()]
    if any(
        left & right
        for index, left in enumerate(identity_sets)
        for right in identity_sets[index + 1 :]
    ):
        raise ValueError("v8 role example identities overlap")
    if any(
        left & right
        for index, left in enumerate(family_sets)
        for right in family_sets[index + 1 :]
    ):
        raise ValueError("v8 role families overlap")
    if roles["calibration_a_fit"]["family_ids"] != _V8_FIT_FAMILY_IDS:
        raise ValueError("v8 fit families differ from frozen LOFO families")
    return roles


def _membership_sha256(
    *,
    partition_kind: str,
    held_family_id: str | None,
    ordered_members: tuple[tuple[str, str], ...],
) -> str:
    payload = {
        "schema": _MEMBERSHIP_SCHEMA,
        "format_version": 1,
        "fit_role_manifest_sha256": _V8_ROLE_BINDINGS[
            "calibration_a_fit"
        ]["manifest_sha256"],
        "partition_kind": partition_kind,
        "held_family_id": held_family_id,
        "ordered_members": [
            {"identity_sha256": identity, "family_id": family}
            for identity, family in ordered_members
        ],
    }
    return _domain_sha256(_MEMBERSHIP_DOMAIN, payload)


def _family_alias_mapping_sha256() -> str:
    payload = {
        "schema": "fisher_graph.gemma3_layer17_family_lofo_alias_map",
        "format_version": 1,
        "fit_role_manifest_sha256": _V8_ROLE_BINDINGS[
            "calibration_a_fit"
        ]["manifest_sha256"],
        "ordered_alias_mapping": [
            {"family_alias": alias, "authenticated_family_id": family_id}
            for alias, family_id in zip(
                V8_FAMILY_LOFO_FAMILY_ALIASES,
                _V8_FIT_FAMILY_IDS,
                strict=True,
            )
        ],
    }
    return _domain_sha256(_FAMILY_ALIAS_MAPPING_DOMAIN, payload)


def _verify_frozen_fold_memberships(
    fit_role: Mapping[str, object],
) -> None:
    identities = fit_role["ordered_identity_sha256s"]
    families = fit_role["ordered_family_ids"]
    if not isinstance(identities, tuple) or not isinstance(families, tuple):
        raise TypeError("authenticated fit membership is malformed")
    members = tuple(zip(identities, families, strict=True))
    if (
        _membership_sha256(
            partition_kind="fit_all",
            held_family_id=None,
            ordered_members=members,
        )
        != _FIT_MEMBERSHIP_SHA256
    ):
        raise ValueError("v8 fit membership commitment drifted")
    for family_id, held_sha256, training_sha256 in (
        _FROZEN_FOLD_MEMBERSHIP_BINDINGS
    ):
        held_members = tuple(
            member for member in members if member[1] == family_id
        )
        training_members = tuple(
            member for member in members if member[1] != family_id
        )
        if len(held_members) != 32 or len(training_members) != 224:
            raise ValueError("v8 family fold cardinality drifted")
        if (
            _membership_sha256(
                partition_kind="held_family",
                held_family_id=family_id,
                ordered_members=held_members,
            )
            != held_sha256
            or _membership_sha256(
                partition_kind="training_complement",
                held_family_id=family_id,
                ordered_members=training_members,
            )
            != training_sha256
        ):
            raise ValueError("v8 family fold membership commitment drifted")


def _fold_payload(
    *,
    fold_index: int,
    held_family_alias: str,
    held_membership_sha256: str,
    training_membership_sha256: str,
) -> dict[str, object]:
    training_family_aliases = [
        alias
        for alias in V8_FAMILY_LOFO_FAMILY_ALIASES
        if alias != held_family_alias
    ]
    return {
        "fold_index": fold_index,
        "fold_id": f"family-{fold_index + 1:02d}-of-08",
        "held_family_alias": held_family_alias,
        "training_family_aliases": training_family_aliases,
        "held_example_count": 32,
        "training_example_count": 224,
        "held_membership_sha256": held_membership_sha256,
        "training_membership_sha256": training_membership_sha256,
        "fit_role_manifest_sha256": _V8_ROLE_BINDINGS[
            "calibration_a_fit"
        ]["manifest_sha256"],
        "fit_policy": "refit_from_frozen_source_arm_on_training_complement",
        "score_policy": "score_held_family_once_after_fold_fit",
    }


def _folds() -> list[dict[str, object]]:
    result = []
    for fold_index, (
        family_alias,
        (_, held_membership_sha256, training_membership_sha256),
    ) in enumerate(
        zip(
            V8_FAMILY_LOFO_FAMILY_ALIASES,
            _FROZEN_FOLD_MEMBERSHIP_BINDINGS,
            strict=True,
        )
    ):
        payload = _fold_payload(
            fold_index=fold_index,
            held_family_alias=family_alias,
            held_membership_sha256=held_membership_sha256,
            training_membership_sha256=training_membership_sha256,
        )
        result.append(
            {
                **payload,
                "artifact_sha256": _domain_sha256(_FOLD_DOMAIN, payload),
            }
        )
    return result


def _protocol_payload() -> dict[str, object]:
    return {
        "schema": GEMMA3_LAYER17_FAMILY_LOFO_PROTOCOL_SCHEMA,
        "format_version": (
            GEMMA3_LAYER17_FAMILY_LOFO_PROTOCOL_FORMAT_VERSION
        ),
        "corpus_authority": {
            "schema": _SOURCE_CORPUS_SCHEMA,
            "format_version": _SOURCE_CORPUS_FORMAT_VERSION,
            "corpus_id": _V8_CORPUS_ID,
            "profile": _V8_PROFILE,
            "artifact_sha256": _V8_CORPUS_ARTIFACT_SHA256,
            "tokenizer_contract_sha256": _V8_TOKENIZER_CONTRACT_SHA256,
            "forbidden_assessment_manifest_sha256s": [
                _V8_FORBIDDEN_CALIBRATION_B_MANIFEST_SHA256
            ],
            "fit_membership_sha256": _FIT_MEMBERSHIP_SHA256,
            "family_alias_mapping_sha256": (
                _family_alias_mapping_sha256()
            ),
        },
        "role_bindings": {
            "fit": {
                "role": "calibration_a_fit",
                **_V8_ROLE_BINDINGS["calibration_a_fit"],
                "family_aliases": list(V8_FAMILY_LOFO_FAMILY_ALIASES),
                "family_alias_mapping_sha256": (
                    _family_alias_mapping_sha256()
                ),
                "used_for_fold_fitting": True,
            },
            "open_development_assessment": {
                "role": "calibration_a_selection",
                **_V8_ROLE_BINDINGS["calibration_a_selection"],
                "used_for_fold_fitting": False,
                "historical_status": "open_development",
                "eligible_only_after_lofo_gate_decision": True,
            },
            "sealed_guard": {
                "role": "calibration_a_guard",
                **_V8_ROLE_BINDINGS["calibration_a_guard"],
                "used_for_fold_fitting": False,
                "must_remain_sealed": True,
            },
            "forbidden_external_roles": [
                "calibration_b",
                "validation",
                "test",
            ],
        },
        "first_arm": {
            "arm_id": "cap48-r16-edgeless-v1",
            "layer_ordinal": 17,
            "topology_sha256": _LAYER17_TOPOLOGY_SHA256,
            "fragment_ids_in_execution_order": list(_LAYER17_FRAGMENT_IDS),
            "native_mode_counts_in_execution_order": list(
                _LAYER17_NATIVE_MODE_COUNTS
            ),
            "mode_rank_cap": 48,
            "resolved_node_ranks_in_execution_order": [48, 38, 48, 48],
            "generator_rank": 16,
            "edge_policy": "edgeless",
            "interaction_count": 0,
            "rank_policy": "fixed_predeclared_no_foldwise_selection",
            "fit_target": "teacher_layer_residual_flow",
            "ridge": 0.0,
        },
        "folds": _folds(),
        "gates": {
            "decision_policy": "all_required_gates_must_pass",
            "required_completed_fold_count": 8,
            "maximum_failed_fold_count": 0,
            "maximum_family_macro_delta_nll_per_token": 0.075,
            "maximum_worst_family_delta_nll_per_token": 0.1,
            "maximum_family_macro_native_to_candidate_kl_per_token": 0.06,
            "minimum_family_macro_top1_agreement_to_native": 0.875,
            "minimum_family_macro_deletion_nll_recovery_fraction": 0.6,
            "minimum_worst_family_deletion_nll_recovery_fraction": 0.4,
            "require_positive_exact_parameter_savings": True,
            "require_positive_logical_mac_savings": True,
            "permit_latency_or_kernel_speed_claim": False,
        },
        "evaluation_contract": {
            "outer_split_unit": "authenticated_family_alias",
            "outer_fold_count": 8,
            "held_family_count_per_fold": 1,
            "training_family_count_per_fold": 7,
            "fold_local_refit_required": True,
            "reuse_fitted_parameters_across_folds": False,
            "arm_selection_inside_lofo": False,
            "held_family_used_for_fitting_or_early_stopping": False,
            "aggregation": "unweighted_macro_mean_over_held_families",
            "worst_family_metrics_required": True,
            "deletion_control_required": True,
            "deletion_nll_recovery": {
                "candidate_condition": "lofo_refit",
                "control_condition": "matched_deletion",
                "formula": (
                    "(matched_deletion_delta_nll_per_token-"
                    "lofo_refit_delta_nll_per_token)/"
                    "matched_deletion_delta_nll_per_token"
                ),
                "denominator": (
                    "matched_deletion_delta_nll_per_token"
                ),
                "denominator_requirement": "strictly_greater_than_zero",
                "invalid_denominator_policy": "fail_closed",
                "macro_aggregation": "equal_family_arithmetic_mean",
                "worst_aggregation": "minimum_over_held_families",
            },
            "native_teacher_reference_required": True,
            "randomness_policy": "seed_and_recipe_committed_before_execution",
        },
        "claim_boundary": {
            "scientific_role": "open_development_family_generalization_test",
            "supports_family_blocked_internal_estimate": True,
            "supports_heldout_confirmation": False,
            "supports_lossless_claim": False,
            "supports_compression_claim_before_gates": False,
            "selection_role_is_not_an_outer_lofo_fold": True,
            "guard_and_external_assessment_roles_remain_untouched": True,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_per_example_identity": False,
            "contains_activation_or_gradient_tensors": False,
            "contains_model_or_candidate_weights": False,
            "role_input_file_opened": False,
            "model_or_tokenizer_accessed": False,
            "model_executed": False,
            "metrics_read": False,
            "calibration_a_selection_opened_by_planner": False,
            "calibration_a_guard_opened": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
    }


def build_default_v8_layer17_family_lofo_protocol() -> dict[str, object]:
    """Return the frozen prompt-free eight-fold layer-17 protocol."""

    payload = _protocol_payload()
    artifact_sha256 = _domain_sha256(_PROTOCOL_DOMAIN, payload)
    if artifact_sha256 != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256:
        raise RuntimeError("frozen family-LOFO protocol identity drifted")
    return {
        **payload,
        "artifact_sha256": artifact_sha256,
    }


def build_authenticated_v8_layer17_family_lofo_protocol(
    corpus_artifact: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate the prompt-free v8 corpus before returning the protocol.

    The argument must already be the in-memory, prompt-free corpus artifact.
    Paths are intentionally unsupported so this module cannot open any role
    input file.  Example identity hashes are used transiently to verify the
    frozen fold commitments and do not appear in the returned protocol.
    """

    roles = _authenticate_v8_corpus(corpus_artifact)
    _verify_frozen_fold_memberships(roles["calibration_a_fit"])
    return build_default_v8_layer17_family_lofo_protocol()


def validate_v8_layer17_family_lofo_protocol(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed unless *raw* is the exact frozen source-safe protocol."""

    protocol = _strict_mapping(
        raw,
        fields=_PROTOCOL_FIELDS,
        label="family-LOFO protocol",
    )
    supplied_sha256 = _require_sha256(
        protocol["artifact_sha256"], label="family-LOFO protocol"
    )
    payload = {
        key: protocol[key]
        for key in _PROTOCOL_FIELDS
        if key != "artifact_sha256"
    }
    if _canonical_json_value(payload) != _canonical_json_value(
        _protocol_payload()
    ):
        raise ValueError("family-LOFO protocol differs from frozen plan")
    recomputed_sha256 = _domain_sha256(_PROTOCOL_DOMAIN, payload)
    if supplied_sha256 != recomputed_sha256:
        raise ValueError("family-LOFO protocol hash mismatch")
    if supplied_sha256 != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256:
        raise ValueError("family-LOFO protocol identity is not frozen v8")
    return json.loads(_canonical_json_bytes(protocol).decode("utf-8"))
