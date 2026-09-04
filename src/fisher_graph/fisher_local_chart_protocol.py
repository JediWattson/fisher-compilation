"""Freeze and one-shot guard protocol for the Fisher local-chart probe.

This module is deliberately model-free.  It wraps the repository's existing
three-role Calibration-A corpus and manifest-global guard ledger, while adding
an independently reloadable pre-guard candidate bundle.  The guard role is
never parsed before the bundle has been authenticated and the irreversible
claim has been persisted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Literal, Protocol, cast

import torch
from torch import Tensor

from .gemma3_l3_l4_progressive_a_corpus import (
    DevelopmentRole,
    Gemma3L3L4ProgressiveACorpus,
    Gemma3L3L4ProgressiveACorpusArtifact,
    Gemma3L3L4ProgressiveARolePrompts,
)
from .gemma3_l3_l4_progressive_guard_ledger import (
    Gemma3L3L4ProgressiveGuardClaimReceipt,
    claim_gemma3_l3_l4_progressive_guard,
)


CalibrationARole = Literal["A_fit", "A_selection", "A_guard"]

__all__ = [
    "CalibrationARole",
    "FisherLocalChartFrozenCandidate",
    "FisherLocalChartSequenceSplitManifest",
    "claim_and_open_fisher_local_chart_guard",
    "freeze_fisher_local_chart_candidate",
    "load_fisher_local_chart_frozen_candidate",
    "sequence_split_manifest_from_corpus",
    "validate_calibration_a_role",
    "write_fisher_local_chart_guard_result",
]


_CANDIDATE_SCHEMA = "fisher_graph.fisher_local_chart.frozen_candidate.v2"
_CANDIDATE_FORMAT_VERSION = 2
_MANIFEST_SCHEMA = "fisher_graph.fisher_local_chart.split_manifest.v1"
_GUARD_RESULT_SCHEMA = "fisher_graph.fisher_local_chart.guard_result.v1"
_CANDIDATE_DOMAIN = b"fisher-graph:fisher-local-chart:candidate:v2\0"
_MANIFEST_DOMAIN = b"fisher-graph:fisher-local-chart:split-manifest:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:fisher-local-chart:tensor:v1\0"
_GUARD_RESULT_DOMAIN = b"fisher-graph:fisher-local-chart:guard-result:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CALIBRATION_B_MARKER = re.compile(
    r"(?<![a-z0-9])(?:calibration[^a-z0-9]*b|b[^a-z0-9]+[a-z])",
    re.IGNORECASE,
)
_GUARD_DATA_MARKER = re.compile(
    r"(?<![a-z0-9])(?:(?:calibration[^a-z0-9]*a|a)[^a-z0-9]*)?guard"
    r"(?:[^a-z0-9]*(?:payload|text|prompts?|data|rows?|examples?|"
    r"metrics?|results?))?(?![a-z0-9])",
    re.IGNORECASE,
)
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_BASELINE_IDS = ("B0", "B1", "B2", "B3", "B4")
_PARTITION_SCHEMES = ("contiguous", "interleaved")
_BASELINE_ENTRY_FIELDS = frozenset(
    {"assignment_sha256", "capacity_signature", "schemes"}
)
_CANDIDATE_BUNDLE_FIELDS = frozenset(
    {
        "baselines",
        "selected_hyperparameters",
        "generators",
        "routers",
        "support",
        "selection_transcript",
        "bindings",
        "report_schema",
    }
)
_CANDIDATE_BINDING_FIELDS = frozenset(
    {
        "protocol_sha256",
        "split_manifest_sha256",
        "carrier_sha256",
        "corpus_artifact_sha256",
        "guard_manifest_sha256",
        "model_config_sha256",
        "source_model_sha256",
        "source_execution_sha256",
    }
)
_PROMPT_FREE_GUARD_KEYS = frozenset({"guard_manifest_sha256"})
_ROLE_MAP: dict[CalibrationARole, DevelopmentRole] = {
    "A_fit": "calibration_a_fit",
    "A_selection": "calibration_a_selection",
    "A_guard": "calibration_a_guard",
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
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _domain_digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise TypeError("tensor digest requires a Tensor")
    if (
        value.layout != torch.strided
        or value.is_sparse
        or value.is_quantized
        or value.device.type == "meta"
    ):
        raise ValueError("candidate tensors must use dense strided storage")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(list(tensor.shape)))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _normalize_weights_only_tree(
    value: object,
    *,
    active_container_ids: set[int] | None = None,
) -> object:
    """Clone a value into built-in containers accepted by weights-only load."""

    active = set() if active_container_ids is None else active_container_ids
    if isinstance(value, Tensor):
        if value.requires_grad:
            raise ValueError("candidate tensors must be detached and finite")
        if (
            value.layout != torch.strided
            or value.is_sparse
            or value.is_quantized
            or value.device.type == "meta"
        ):
            raise ValueError("candidate tensors must use dense strided storage")
        tensor = value.detach().to(device="cpu").clone().contiguous()
        try:
            finite = bool(torch.isfinite(tensor).all())
        except RuntimeError as error:
            raise ValueError("candidate tensor dtype is unsupported") from error
        if not finite:
            raise ValueError("candidate tensors must be detached and finite")
        return tensor
    if value is None or isinstance(value, (str, bool, int, float)):
        _canonical_json_bytes(value)
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("candidate bundle containers must be acyclic")
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("candidate mapping keys must be nonempty strings")
        active.add(identity)
        try:
            return {
                key: _normalize_weights_only_tree(
                    item,
                    active_container_ids=active,
                )
                for key, item in sorted(value.items())
            }
        finally:
            active.remove(identity)
    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active:
            raise ValueError("candidate bundle containers must be acyclic")
        active.add(identity)
        try:
            # One canonical sequence type prevents list/tuple candidates from
            # sharing a logical digest while reloading to different trees.
            return [
                _normalize_weights_only_tree(
                    item,
                    active_container_ids=active,
                )
                for item in value
            ]
        finally:
            active.remove(identity)
    raise TypeError(
        "candidate bundle must contain only tensors and weights-only-safe "
        f"containers, not {type(value).__name__}"
    )


def _tree_receipt(value: object) -> object:
    """Return a deterministic, tensor-free receipt for a weights-only tree."""

    if isinstance(value, Tensor):
        if value.requires_grad:
            raise ValueError("candidate tensors must be detached and finite")
        try:
            finite = bool(torch.isfinite(value).all())
        except RuntimeError as error:
            raise ValueError("candidate tensor dtype is unsupported") from error
        if not finite:
            raise ValueError("candidate tensors must be detached and finite")
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _tensor_sha256(value),
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        # Re-encode scalars to reject NaN and infinity.
        _canonical_json_bytes(value)
        return {"kind": "scalar", "value": value}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("candidate mapping keys must be nonempty strings")
        return {
            "kind": "mapping",
            "items": [
                {"key": key, "value": _tree_receipt(item)}
                for key, item in sorted(value.items())
            ],
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": "sequence",
            "items": [_tree_receipt(item) for item in value],
        }
    raise TypeError(
        "candidate bundle must contain only tensors and weights-only-safe "
        f"containers, not {type(value).__name__}"
    )


def _assert_no_forbidden_serialized_data(
    value: object,
    *,
    forbid_guard_data: bool,
) -> None:
    """Reject held-out names and, pre-claim, any serialized guard payload."""

    def check_text(text: str, *, is_key: bool) -> None:
        if _CALIBRATION_B_MARKER.search(text) is not None:
            raise ValueError(
                "Calibration B is prohibited from serialized local-chart data"
            )
        if (
            forbid_guard_data
            and (not is_key or text not in _PROMPT_FREE_GUARD_KEYS)
            and _GUARD_DATA_MARKER.search(text) is not None
        ):
            raise ValueError(
                "pre-guard candidate bundle cannot contain guard payload, "
                "text, prompts, or results"
            )

    if isinstance(value, Tensor) or value is None or isinstance(
        value, (bool, int, float)
    ):
        return
    if isinstance(value, str):
        check_text(value, is_key=False)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("serialized mappings require string keys")
            check_text(key, is_key=True)
            _assert_no_forbidden_serialized_data(
                item,
                forbid_guard_data=forbid_guard_data,
            )
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_forbidden_serialized_data(
                item,
                forbid_guard_data=forbid_guard_data,
            )
        return
    raise TypeError(f"unsupported serialized value: {type(value).__name__}")


def _require_nonempty_mapping(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"candidate {label} must be a nonempty mapping")
    return cast(Mapping[str, object], value)


def _require_partition_baseline_grid(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    """Validate one complete partition-by-baseline candidate section."""

    root = _require_nonempty_mapping(value, label=label)
    if set(root) != set(_PARTITION_SCHEMES):
        raise ValueError(
            f"candidate {label} must contain exactly contiguous and "
            "interleaved partition schemes"
        )
    for scheme in _PARTITION_SCHEMES:
        arms = _require_nonempty_mapping(
            root[scheme],
            label=f"{label} {scheme}",
        )
        if set(arms) != set(_BASELINE_IDS):
            raise ValueError(
                f"candidate {label} {scheme} must contain exactly B0-B4"
            )
        for baseline_id in _BASELINE_IDS:
            _require_nonempty_mapping(
                arms[baseline_id],
                label=f"{label} {scheme} {baseline_id}",
            )
    return root


def _mapping_has_key_fragment(value: Mapping[str, object], fragment: str) -> bool:
    for key, item in value.items():
        if fragment in key.lower():
            return True
        if isinstance(item, Mapping) and _mapping_has_key_fragment(
            cast(Mapping[str, object], item), fragment
        ):
            return True
    return False


def _validate_candidate_bundle(
    bundle: Mapping[str, object],
    *,
    protocol_sha256: str,
    split_manifest_sha256: str,
    carrier_sha256: str,
) -> dict[str, object]:
    normalized = _normalize_weights_only_tree(bundle)
    _assert_no_forbidden_serialized_data(
        normalized,
        forbid_guard_data=True,
    )
    if not isinstance(normalized, dict) or set(normalized) != (
        _CANDIDATE_BUNDLE_FIELDS
    ):
        raise ValueError(
            "candidate bundle must contain exactly baselines, "
            "selected_hyperparameters, generators, routers, support, "
            "selection_transcript, bindings, and report_schema"
        )

    baselines = _require_nonempty_mapping(
        normalized["baselines"],
        label="baselines",
    )
    if set(baselines) != set(_BASELINE_IDS):
        raise ValueError("candidate baselines must contain exactly B0-B4")
    assignment_sha256s: set[str] = set()
    capacity_signatures: set[bytes] = set()
    for baseline_id in _BASELINE_IDS:
        baseline = _require_nonempty_mapping(
            baselines[baseline_id],
            label=f"baseline {baseline_id}",
        )
        if set(baseline) != set(_BASELINE_ENTRY_FIELDS):
            raise ValueError(
                f"candidate baseline {baseline_id} must contain exactly "
                "assignment_sha256, capacity_signature, and schemes"
            )
        assignment_sha256s.add(
            _require_sha256(
                baseline.get("assignment_sha256"),
                label=f"{baseline_id} assignment",
            )
        )
        capacity_signature = baseline.get("capacity_signature")
        if capacity_signature is None:
            raise ValueError(
                f"candidate baseline {baseline_id} lacks capacity_signature"
            )
        capacity_signatures.add(
            _canonical_json_bytes(_tree_receipt(capacity_signature))
        )
        schemes = _require_nonempty_mapping(
            baseline["schemes"],
            label=f"baseline {baseline_id} schemes",
        )
        if set(schemes) != set(_PARTITION_SCHEMES):
            raise ValueError(
                f"candidate baseline {baseline_id} must contain exactly "
                "contiguous and interleaved partition schemes"
            )
        for scheme in _PARTITION_SCHEMES:
            _require_nonempty_mapping(
                schemes[scheme],
                label=f"baseline {baseline_id} {scheme} atlas",
            )
    if len(assignment_sha256s) != 1:
        raise ValueError("all B0-B4 baselines must share one assignment hash")
    if len(capacity_signatures) != 1:
        raise ValueError("all B0-B4 baselines must have equal capacity")

    selected = _require_nonempty_mapping(
        normalized["selected_hyperparameters"],
        label="selected_hyperparameters",
    )
    if selected.get("selected_baseline") not in _BASELINE_IDS:
        raise ValueError("selected_hyperparameters must select one B0-B4 arm")
    if selected.get("selected_partition_scheme") not in _PARTITION_SCHEMES:
        raise ValueError(
            "selected_hyperparameters must select contiguous or interleaved"
        )
    for key in ("attention_affine_control_arm", "mlp_affine_control_arm"):
        if selected.get(key) not in _BASELINE_IDS[:-1]:
            raise ValueError(
                f"selected_hyperparameters {key} must freeze one Euclidean B0-B3 comparator"
            )
    if not _mapping_has_key_fragment(selected, "ridge"):
        raise ValueError("selected_hyperparameters must freeze ridge settings")
    for section in ("generators", "routers", "support"):
        _require_partition_baseline_grid(normalized[section], label=section)
    transcript = normalized["selection_transcript"]
    if not isinstance(transcript, (Mapping, list)) or not transcript:
        raise ValueError("candidate selection_transcript must be nonempty")
    report_schema = normalized["report_schema"]
    if not (
        (isinstance(report_schema, str) and bool(report_schema))
        or (isinstance(report_schema, Mapping) and bool(report_schema))
    ):
        raise ValueError("candidate report_schema must be nonempty")

    bindings = _require_nonempty_mapping(
        normalized["bindings"],
        label="bindings",
    )
    if not _CANDIDATE_BINDING_FIELDS.issubset(bindings):
        raise ValueError(
            "candidate bindings must include protocol, split, carrier, "
            "corpus, prompt-free guard manifest, model config, and source "
            "model/execution hashes"
        )
    for key, value in bindings.items():
        if not key.endswith("_sha256"):
            raise ValueError("candidate binding keys must name SHA-256 digests")
        _require_sha256(value, label=f"candidate binding {key}")
    expected_bindings = {
        "protocol_sha256": protocol_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "carrier_sha256": carrier_sha256,
    }
    if any(bindings[key] != value for key, value in expected_bindings.items()):
        raise ValueError("candidate envelope and bundle bindings differ")
    _require_sha256(
        bindings["corpus_artifact_sha256"],
        label="candidate corpus artifact binding",
    )
    _require_sha256(
        bindings["guard_manifest_sha256"],
        label="candidate guard manifest binding",
    )
    return normalized


def _candidate_digest(
    *,
    protocol_sha256: str,
    split_manifest_sha256: str,
    carrier_sha256: str,
    bundle_receipt: object,
) -> str:
    return _domain_digest(
        _CANDIDATE_DOMAIN,
        {
            "schema": _CANDIDATE_SCHEMA,
            "format_version": _CANDIDATE_FORMAT_VERSION,
            "state": "frozen_before_A_guard_claim",
            "protocol_sha256": protocol_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "carrier_sha256": carrier_sha256,
            "bundle_receipt": bundle_receipt,
        },
    )


def validate_calibration_a_role(value: object) -> CalibrationARole:
    if value not in _ROLE_MAP:
        if (
            isinstance(value, str)
            and _CALIBRATION_B_MARKER.search(value) is not None
        ):
            raise ValueError("Calibration B is prohibited by this experiment")
        raise ValueError("role must be A_fit, A_selection, or A_guard")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class FisherLocalChartSequenceSplitManifest:
    """Prompt-free sequence identities fixed before token-row expansion."""

    corpus_artifact_sha256: str
    tokenizer_contract_sha256: str
    role_manifest_sha256s: Mapping[CalibrationARole, str]
    role_input_file_sha256s: Mapping[CalibrationARole, str]
    sequence_ids: Mapping[CalibrationARole, tuple[str, ...]]
    family_ids: Mapping[CalibrationARole, tuple[str, ...]]
    forbidden_assessment_manifest_sha256s: tuple[str, ...]
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(
            self.corpus_artifact_sha256,
            label="corpus artifact",
        )
        _require_sha256(
            self.tokenizer_contract_sha256,
            label="tokenizer contract",
        )
        roles = tuple(_ROLE_MAP)
        for name, attribute in (
            ("role manifests", "role_manifest_sha256s"),
            ("role input files", "role_input_file_sha256s"),
            ("sequence IDs", "sequence_ids"),
            ("family IDs", "family_ids"),
        ):
            mapping = getattr(self, attribute)
            if not isinstance(mapping, Mapping) or set(mapping) != set(roles):
                raise ValueError(f"{name} must contain exactly the A roles")
            ordered = {role: mapping[role] for role in roles}
            object.__setattr__(self, attribute, MappingProxyType(ordered))
        for role in roles:
            _require_sha256(
                self.role_manifest_sha256s[role],
                label=f"{role} manifest",
            )
            _require_sha256(
                self.role_input_file_sha256s[role],
                label=f"{role} input file",
            )
            ids = self.sequence_ids[role]
            families = self.family_ids[role]
            if (
                type(ids) is not tuple
                or not ids
                or len(ids) != len(set(ids))
                or any(not isinstance(item, str) or not item for item in ids)
                or type(families) is not tuple
                or len(families) != len(ids)
                or any(
                    not isinstance(item, str) or not item for item in families
                )
            ):
                raise ValueError(f"{role} sequence/family rows are invalid")
        if len(set(self.role_manifest_sha256s.values())) != len(roles):
            raise ValueError("A roles must use distinct role manifests")
        if len(set(self.role_input_file_sha256s.values())) != len(roles):
            raise ValueError("A roles must use distinct role input files")
        sequence_sets = tuple(set(self.sequence_ids[role]) for role in roles)
        if any(
            left & right
            for index, left in enumerate(sequence_sets)
            for right in sequence_sets[index + 1 :]
        ):
            raise ValueError("A roles must be sequence/document disjoint")
        family_sets = tuple(set(self.family_ids[role]) for role in roles)
        if any(
            left & right
            for index, left in enumerate(family_sets)
            for right in family_sets[index + 1 :]
        ):
            raise ValueError("A roles must be family disjoint")
        forbidden = self.forbidden_assessment_manifest_sha256s
        if (
            type(forbidden) is not tuple
            or not forbidden
            or tuple(sorted(set(forbidden))) != forbidden
        ):
            raise ValueError(
                "forbidden manifests must be nonempty, sorted, and unique"
            )
        for value in forbidden:
            _require_sha256(value, label="forbidden assessment manifest")
        payload = self.to_dict(include_digest=False)
        expected = _domain_digest(_MANIFEST_DOMAIN, payload)
        if self.manifest_sha256 and self.manifest_sha256 != expected:
            raise ValueError("sequence split manifest SHA-256 mismatch")
        object.__setattr__(self, "manifest_sha256", expected)

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": _MANIFEST_SCHEMA,
            "format_version": 1,
            "split_unit": "sequence_or_document_before_token_expansion",
            "bootstrap_unit": "sequence_or_document",
            "calibration_b_prohibited": True,
            "corpus_artifact_sha256": self.corpus_artifact_sha256,
            "tokenizer_contract_sha256": self.tokenizer_contract_sha256,
            "role_manifest_sha256s": dict(self.role_manifest_sha256s),
            "role_input_file_sha256s": dict(
                self.role_input_file_sha256s
            ),
            "sequence_ids": {
                role: list(values) for role, values in self.sequence_ids.items()
            },
            "family_ids": {
                role: list(values) for role, values in self.family_ids.items()
            },
            "forbidden_assessment_manifest_sha256s": list(
                self.forbidden_assessment_manifest_sha256s
            ),
        }
        if include_digest:
            result["manifest_sha256"] = self.manifest_sha256
        return result

    def assert_row_provenance(
        self,
        role: CalibrationARole,
        row_sequence_ids: Sequence[str],
    ) -> None:
        selected = validate_calibration_a_role(role)
        allowed = set(self.sequence_ids[selected])
        if not row_sequence_ids or any(
            not isinstance(item, str) or item not in allowed
            for item in row_sequence_ids
        ):
            raise ValueError("expanded token rows escaped their sequence split")


def sequence_split_manifest_from_corpus(
    artifact: Gemma3L3L4ProgressiveACorpusArtifact,
) -> FisherLocalChartSequenceSplitManifest:
    if not isinstance(artifact, Gemma3L3L4ProgressiveACorpusArtifact):
        raise TypeError("artifact must be a progressive Calibration-A corpus")
    role_manifest_sha256s: dict[CalibrationARole, str] = {}
    role_input_file_sha256s: dict[CalibrationARole, str] = {}
    sequence_ids: dict[CalibrationARole, tuple[str, ...]] = {}
    family_ids: dict[CalibrationARole, tuple[str, ...]] = {}
    for role, source_role in _ROLE_MAP.items():
        view = artifact.role_view(source_role)
        role_manifest_sha256s[role] = view.manifest_sha256
        role_input_file_sha256s[role] = view.role_input_file_sha256
        sequence_ids[role] = view.ordered_prompt_sha256s
        family_ids[role] = view.ordered_family_ids
    return FisherLocalChartSequenceSplitManifest(
        corpus_artifact_sha256=artifact.artifact_sha256,
        tokenizer_contract_sha256=artifact.tokenizer_contract_sha256,
        role_manifest_sha256s=role_manifest_sha256s,
        role_input_file_sha256s=role_input_file_sha256s,
        sequence_ids=sequence_ids,
        family_ids=family_ids,
        forbidden_assessment_manifest_sha256s=(
            artifact.forbidden_assessment_manifest_sha256s
        ),
    )


@dataclass(frozen=True, slots=True)
class FisherLocalChartFrozenCandidate:
    """Strictly reloaded content-addressed pre-guard candidate."""

    path: Path
    manifest_path: Path
    protocol_sha256: str
    split_manifest_sha256: str
    carrier_sha256: str
    candidate_sha256: str
    candidate_file_sha256: str
    bundle: Mapping[str, object]
    bundle_receipt: object

    @property
    def challenger_receipt_sha256(self) -> str:
        return self.candidate_sha256


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
            linked = True
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite frozen artifact: {destination}"
            ) from None
        _fsync_directory(destination.parent)
    except BaseException:
        if linked:
            destination.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_bytes_write(destination: Path, encoded: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        destination,
        flags,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("exclusive artifact write made no progress")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        destination.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        _fsync_directory(destination.parent)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _load_weights_only_payload(path: Path) -> tuple[object, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError("frozen candidate cannot be opened safely") from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size <= 0:
            raise ValueError("frozen candidate must be a nonempty regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            before = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                before.update(chunk)
            handle.seek(0)
            try:
                raw = torch.load(
                    handle,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception as error:
                raise ValueError(
                    "frozen candidate is not a readable weights-only artifact"
                ) from error
            handle.seek(0)
            after = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                after.update(chunk)
        if before.digest() != after.digest():
            raise ValueError("frozen candidate changed while it was loaded")
        return raw, after.hexdigest()
    finally:
        os.close(descriptor)


def _load_canonical_manifest(path: Path) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("frozen candidate manifest is unreadable") from error
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_size <= 0
            or details.st_size > _MAX_MANIFEST_BYTES
        ):
            raise ValueError(
                "frozen candidate manifest must be a bounded regular file"
            )
        chunks: list[bytes] = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise ValueError("frozen candidate manifest ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("frozen candidate manifest changed while read")
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)
    try:
        raw = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen candidate manifest is unreadable") from error
    if not isinstance(raw, dict) or _canonical_json_bytes(raw) != encoded:
        raise ValueError("frozen candidate manifest is not canonical JSON")
    return raw


def freeze_fisher_local_chart_candidate(
    *,
    bundle: Mapping[str, object],
    protocol_sha256: str,
    split_manifest_sha256: str,
    carrier_sha256: str,
    path: Path | str,
    manifest_path: Path | str | None = None,
) -> FisherLocalChartFrozenCandidate:
    """Persist one immutable candidate before any A-guard authority exists."""

    protocol = _require_sha256(protocol_sha256, label="protocol")
    split = _require_sha256(split_manifest_sha256, label="split manifest")
    carrier = _require_sha256(carrier_sha256, label="Fisher carrier")
    if not isinstance(bundle, Mapping) or not bundle:
        raise TypeError("candidate bundle must be a nonempty mapping")
    normalized_bundle = _validate_candidate_bundle(
        cast(Mapping[str, object], bundle),
        protocol_sha256=protocol,
        split_manifest_sha256=split,
        carrier_sha256=carrier,
    )
    bindings = cast(Mapping[str, object], normalized_bundle["bindings"])
    receipt = _tree_receipt(normalized_bundle)
    digest = _candidate_digest(
        protocol_sha256=protocol,
        split_manifest_sha256=split,
        carrier_sha256=carrier,
        bundle_receipt=receipt,
    )
    payload: dict[str, object] = {
        "schema": _CANDIDATE_SCHEMA,
        "format_version": _CANDIDATE_FORMAT_VERSION,
        "state": "frozen_before_A_guard_claim",
        "protocol_sha256": protocol,
        "split_manifest_sha256": split,
        "carrier_sha256": carrier,
        "candidate_sha256": digest,
        "bundle": normalized_bundle,
    }
    destination = Path(path)
    manifest_destination = (
        destination.with_suffix(destination.suffix + ".manifest.json")
        if manifest_path is None
        else Path(manifest_path)
    )
    if destination.suffix != ".pt":
        raise ValueError("frozen candidate path must use .pt")
    if destination.resolve(strict=False) == manifest_destination.resolve(
        strict=False
    ):
        raise ValueError("candidate and manifest paths must be distinct")
    _exclusive_torch_save(payload, destination)
    manifest_written = False
    try:
        file_digest = _file_sha256(destination)
        manifest = {
            "schema": _CANDIDATE_SCHEMA,
            "format_version": _CANDIDATE_FORMAT_VERSION,
            "state": "frozen_before_A_guard_claim",
            "candidate_path": str(destination),
            "candidate_file_sha256": file_digest,
            "candidate_sha256": digest,
            "protocol_sha256": protocol,
            "split_manifest_sha256": split,
            "carrier_sha256": carrier,
            "bindings": dict(bindings),
            "bundle_receipt": receipt,
        }
        _exclusive_bytes_write(
            manifest_destination,
            _canonical_json_bytes(manifest),
        )
        manifest_written = True
        restored = load_fisher_local_chart_frozen_candidate(
            destination,
            manifest_path=manifest_destination,
        )
    except BaseException:
        if manifest_written:
            manifest_destination.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    return restored


def load_fisher_local_chart_frozen_candidate(
    path: Path | str,
    *,
    manifest_path: Path | str | None = None,
) -> FisherLocalChartFrozenCandidate:
    source = Path(path)
    manifest_source = (
        source.with_suffix(source.suffix + ".manifest.json")
        if manifest_path is None
        else Path(manifest_path)
    )
    raw, candidate_file_sha256 = _load_weights_only_payload(source)
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema",
        "format_version",
        "state",
        "protocol_sha256",
        "split_manifest_sha256",
        "carrier_sha256",
        "candidate_sha256",
        "bundle",
    }:
        raise ValueError("frozen candidate fields are invalid")
    if (
        raw["schema"] != _CANDIDATE_SCHEMA
        or raw["format_version"] != _CANDIDATE_FORMAT_VERSION
        or raw["state"] != "frozen_before_A_guard_claim"
        or not isinstance(raw["bundle"], Mapping)
    ):
        raise ValueError("frozen candidate identity is invalid")
    protocol = _require_sha256(raw["protocol_sha256"], label="protocol")
    split = _require_sha256(
        raw["split_manifest_sha256"], label="split manifest"
    )
    carrier = _require_sha256(raw["carrier_sha256"], label="Fisher carrier")
    bundle = _validate_candidate_bundle(
        cast(Mapping[str, object], raw["bundle"]),
        protocol_sha256=protocol,
        split_manifest_sha256=split,
        carrier_sha256=carrier,
    )
    bindings = cast(Mapping[str, object], bundle["bindings"])
    receipt = _tree_receipt(bundle)
    candidate = _candidate_digest(
        protocol_sha256=protocol,
        split_manifest_sha256=split,
        carrier_sha256=carrier,
        bundle_receipt=receipt,
    )
    if raw["candidate_sha256"] != candidate:
        raise ValueError("frozen candidate logical SHA-256 mismatch")
    manifest = _load_canonical_manifest(manifest_source)
    expected_manifest = {
        "schema": _CANDIDATE_SCHEMA,
        "format_version": _CANDIDATE_FORMAT_VERSION,
        "state": "frozen_before_A_guard_claim",
        "candidate_path": str(source),
        "candidate_file_sha256": candidate_file_sha256,
        "candidate_sha256": candidate,
        "protocol_sha256": protocol,
        "split_manifest_sha256": split,
        "carrier_sha256": carrier,
        "bindings": dict(bindings),
        "bundle_receipt": receipt,
    }
    if manifest != expected_manifest:
        raise ValueError("frozen candidate manifest or file hash mismatch")
    return FisherLocalChartFrozenCandidate(
        path=source,
        manifest_path=manifest_source,
        protocol_sha256=protocol,
        split_manifest_sha256=split,
        carrier_sha256=carrier,
        candidate_sha256=candidate,
        candidate_file_sha256=candidate_file_sha256,
        bundle=bundle,
        bundle_receipt=receipt,
    )


class ClaimFunction(Protocol):
    def __call__(
        self,
        *,
        protocol_sha256: str,
        guard_manifest_sha256: str,
        challenger_receipt_sha256: str,
    ) -> Gemma3L3L4ProgressiveGuardClaimReceipt: ...


class DryRunFunction(Protocol):
    def __call__(self, candidate: FisherLocalChartFrozenCandidate) -> None: ...


def _candidate_bindings(
    candidate: FisherLocalChartFrozenCandidate,
) -> Mapping[str, object]:
    return cast(Mapping[str, object], candidate.bundle["bindings"])


def _validate_claim_binding(
    *,
    claim: Gemma3L3L4ProgressiveGuardClaimReceipt,
    candidate: FisherLocalChartFrozenCandidate,
    guard_manifest_sha256: str,
) -> None:
    if not isinstance(claim, Gemma3L3L4ProgressiveGuardClaimReceipt):
        raise TypeError("claim function must return a durable A-guard receipt")
    claim.validate_integrity()
    if (
        claim.protocol_sha256 != candidate.protocol_sha256
        or claim.guard_manifest_sha256 != guard_manifest_sha256
        or claim.challenger_receipt_sha256
        != candidate.challenger_receipt_sha256
    ):
        raise ValueError("durable guard claim does not bind the frozen candidate")


def _reload_matching_candidate(
    candidate: FisherLocalChartFrozenCandidate,
) -> FisherLocalChartFrozenCandidate:
    if not isinstance(candidate.path, Path) or not isinstance(
        candidate.manifest_path, Path
    ):
        raise TypeError("frozen candidate paths must be Path values")
    persisted = load_fisher_local_chart_frozen_candidate(
        candidate.path,
        manifest_path=candidate.manifest_path,
    )
    try:
        supplied_receipt = _canonical_json_bytes(candidate.bundle_receipt)
        current_receipt = _canonical_json_bytes(
            _tree_receipt(candidate.bundle)
        )
        persisted_receipt = _canonical_json_bytes(
            persisted.bundle_receipt
        )
    except (TypeError, ValueError) as error:
        raise ValueError("frozen candidate object was mutated") from error
    if (
        candidate.protocol_sha256 != persisted.protocol_sha256
        or candidate.split_manifest_sha256
        != persisted.split_manifest_sha256
        or candidate.carrier_sha256 != persisted.carrier_sha256
        or candidate.candidate_sha256 != persisted.candidate_sha256
        or candidate.candidate_file_sha256
        != persisted.candidate_file_sha256
        or supplied_receipt != persisted_receipt
        or current_receipt != persisted_receipt
    ):
        raise ValueError("frozen candidate object differs from persisted bytes")
    return persisted


def claim_and_open_fisher_local_chart_guard(
    *,
    corpus: Gemma3L3L4ProgressiveACorpus,
    candidate_path: Path | str,
    candidate_manifest_path: Path | str | None = None,
    expected_protocol_sha256: str,
    expected_split_manifest_sha256: str,
    expected_carrier_sha256: str,
    dry_run: DryRunFunction,
    claim_function: ClaimFunction = claim_gemma3_l3_l4_progressive_guard,
) -> tuple[
    FisherLocalChartFrozenCandidate,
    Gemma3L3L4ProgressiveGuardClaimReceipt,
    Gemma3L3L4ProgressiveARolePrompts,
]:
    """Authenticate, dry-run, claim, then materialize A-guard exactly once."""

    if not isinstance(corpus, Gemma3L3L4ProgressiveACorpus):
        raise TypeError("corpus must be a guarded Calibration-A corpus")
    if not callable(dry_run) or not callable(claim_function):
        raise TypeError("dry_run and claim_function must be callable")
    corpus_manifest = sequence_split_manifest_from_corpus(corpus.artifact)
    guard_view = corpus.preclaim_view("calibration_a_guard")
    frozen = load_fisher_local_chart_frozen_candidate(
        candidate_path,
        manifest_path=candidate_manifest_path,
    )
    expected_protocol = _require_sha256(
        expected_protocol_sha256, label="expected protocol"
    )
    expected_split = _require_sha256(
        expected_split_manifest_sha256, label="expected split manifest"
    )
    expected_carrier = _require_sha256(
        expected_carrier_sha256, label="expected carrier"
    )
    if (
        frozen.protocol_sha256 != expected_protocol
        or frozen.split_manifest_sha256 != expected_split
        or frozen.carrier_sha256 != expected_carrier
        or expected_split != corpus_manifest.manifest_sha256
    ):
        raise ValueError("frozen candidate preflight binding mismatch")
    bindings = _candidate_bindings(frozen)
    if (
        bindings["corpus_artifact_sha256"]
        != corpus.artifact.artifact_sha256
        or bindings["guard_manifest_sha256"]
        != guard_view.manifest_sha256
    ):
        raise ValueError("frozen candidate does not bind the supplied A corpus")
    # All fallible model/code/dry-evaluation checks must happen before the
    # irreversible manifest-global claim.
    pre_dry_receipt = _canonical_json_bytes(frozen.bundle_receipt)
    pre_dry_identity = (
        frozen.protocol_sha256,
        frozen.split_manifest_sha256,
        frozen.carrier_sha256,
        frozen.candidate_sha256,
        frozen.candidate_file_sha256,
    )
    dry_run(frozen)
    try:
        post_dry_receipt = _canonical_json_bytes(
            _tree_receipt(frozen.bundle)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("dry run mutated the frozen candidate") from error
    if post_dry_receipt != pre_dry_receipt:
        raise ValueError("dry run mutated the frozen candidate")
    reloaded = load_fisher_local_chart_frozen_candidate(
        candidate_path,
        manifest_path=candidate_manifest_path,
    )
    post_dry_identity = (
        reloaded.protocol_sha256,
        reloaded.split_manifest_sha256,
        reloaded.carrier_sha256,
        reloaded.candidate_sha256,
        reloaded.candidate_file_sha256,
    )
    if (
        post_dry_identity != pre_dry_identity
        or _canonical_json_bytes(reloaded.bundle_receipt)
        != pre_dry_receipt
    ):
        raise ValueError("frozen candidate changed during dry run")
    frozen = reloaded
    claim = claim_function(
        protocol_sha256=frozen.protocol_sha256,
        guard_manifest_sha256=guard_view.manifest_sha256,
        challenger_receipt_sha256=frozen.challenger_receipt_sha256,
    )
    # A malformed or foreign result from an injected claim adapter is still a
    # burned ledger attempt; fail closed without materializing guard text.
    _validate_claim_binding(
        claim=claim,
        candidate=frozen,
        guard_manifest_sha256=guard_view.manifest_sha256,
    )
    # The corpus marks the role consumed before parsing its JSON, so a corrupt
    # file or later exception cannot turn a claimed guard back into a reusable
    # panel.
    guard = corpus.open_guard_after_claim(claim)
    return frozen, claim, guard


def write_fisher_local_chart_guard_result(
    *,
    path: Path | str,
    candidate: FisherLocalChartFrozenCandidate,
    claim: Gemma3L3L4ProgressiveGuardClaimReceipt,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(candidate, FisherLocalChartFrozenCandidate):
        raise TypeError("candidate must be a frozen local-chart bundle")
    if not isinstance(claim, Gemma3L3L4ProgressiveGuardClaimReceipt):
        raise TypeError("claim must be a durable A-guard claim")
    candidate = _reload_matching_candidate(candidate)
    bindings = _candidate_bindings(candidate)
    _validate_claim_binding(
        claim=claim,
        candidate=candidate,
        guard_manifest_sha256=cast(
            str,
            bindings["guard_manifest_sha256"],
        ),
    )
    if not isinstance(metrics, Mapping) or not metrics:
        raise TypeError("guard metrics must be a nonempty mapping")
    safe_metrics = _canonical_json_value(metrics)
    _assert_no_forbidden_serialized_data(
        safe_metrics,
        forbid_guard_data=False,
    )
    payload: dict[str, object] = {
        "schema": _GUARD_RESULT_SCHEMA,
        "format_version": 1,
        "role": "A_guard",
        "evaluation_count": 1,
        "candidate_sha256": candidate.candidate_sha256,
        "candidate_file_sha256": candidate.candidate_file_sha256,
        "protocol_sha256": candidate.protocol_sha256,
        "split_manifest_sha256": candidate.split_manifest_sha256,
        "corpus_artifact_sha256": bindings["corpus_artifact_sha256"],
        "carrier_sha256": candidate.carrier_sha256,
        "guard_manifest_sha256": claim.guard_manifest_sha256,
        "guard_claim_sha256": claim.claim_sha256,
        "guard_claim_file_sha256": claim.claim_file_sha256,
        "metrics": safe_metrics,
    }
    payload["result_sha256"] = _domain_digest(_GUARD_RESULT_DOMAIN, payload)
    _exclusive_bytes_write(Path(path), _canonical_json_bytes(payload))
    return payload
