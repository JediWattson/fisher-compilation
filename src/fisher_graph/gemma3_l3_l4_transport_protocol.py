"""Leakage-resistant corpus protocol for the Gemma L3-to-L4 transport rung.

The structured-strong-v9 corpus was frozen with a family-disjoint split inside
calibration A and unopened calibration-B/validation/test roles.  This module
turns that external policy into a small fail-closed object used by the
transport builder and the one-shot shadow evaluator.

The builder may read and tokenize only the declared calibration-A fit and
guard members.  The shadow evaluator may claim calibration B exactly once,
after a candidate artifact has been frozen and authenticated.  Validation and
test are never exposed by this API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

from .gemma3_full_width_single_layer_experiment import (
    PromptFamilyManifest,
    load_prompt_family_manifest,
)
from .gemma3_stability_experiment import (
    Gemma3PromptSplits,
    load_gemma3_prompt_splits,
)


__all__ = [
    "CalibrationBClaim",
    "TransportCorpusPreclaimView",
    "TransportCorpusPartition",
    "TransportCorpusPlan",
    "claim_transport_calibration_b",
    "load_transport_corpus_preclaim_view",
    "load_transport_corpus_plan",
]


_AUDIT_SCHEMA = "fisher_graph.structured_strong_corpus_audit"
_CORPUS_ID = "structured-strong-v9"
_AUDIT_FORMAT_VERSION = 4
_CLAIM_SCHEMA = "fisher_graph.gemma3_l3_l4_transport_calibration_b_claim"
_CLAIM_FORMAT_VERSION = 2
_CLAIM_DOMAIN = b"fisher-graph:l3-l4-transport-calibration-b-claim:v2\0"
_PRECLAIM_DOMAIN = b"fisher-graph:l3-l4-transport-preclaim:v1\0"
_PARTITION_DOMAIN = b"fisher-graph:l3-l4-transport-partition:v1\0"
_FULL_CALIBRATION_B_PROMPT_COUNT = 96
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(
        json.dumps(
            [prompt],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _prompt_index_sha256(indices: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(indices),
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _string_sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(values),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_index_tuple(
    value: object,
    *,
    label: str,
    upper_bound: int,
) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(index) is not int for index in value)
    ):
        raise ValueError(f"{label} prompt_indices must be nonempty integers")
    result = tuple(value)
    if (
        len(result) != len(set(result))
        or any(index < 0 or index >= upper_bound for index in result)
    ):
        raise ValueError(f"{label} prompt_indices are invalid")
    return result


@dataclass(frozen=True, slots=True)
class TransportCorpusPartition:
    """One exact, prompt- and family-bound transport protocol role."""

    name: str
    prompts: tuple[str, ...]
    family_ids: tuple[str, ...]
    prompt_sha256s: tuple[str, ...]
    source_indices: tuple[int, ...]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        count = len(self.prompts)
        if (
            not isinstance(self.name, str)
            or not self.name
            or count <= 0
            or len(self.family_ids) != count
            or len(self.prompt_sha256s) != count
            or len(self.source_indices) != count
            or len(set(self.prompts)) != count
            or len(set(self.prompt_sha256s)) != count
            or len(set(self.source_indices)) != count
        ):
            raise ValueError("transport partition fields do not align")
        if any(
            not isinstance(prompt, str) or not prompt
            for prompt in self.prompts
        ):
            raise ValueError("transport prompts must be nonempty strings")
        if any(
            not isinstance(family, str) or not family
            for family in self.family_ids
        ):
            raise ValueError("transport families must be nonempty strings")
        if any(type(index) is not int or index < 0 for index in self.source_indices):
            raise ValueError("transport source indices must be nonnegative")
        expected_prompt_sha256s = tuple(
            _prompt_sha256(prompt) for prompt in self.prompts
        )
        if self.prompt_sha256s != expected_prompt_sha256s:
            raise ValueError("transport prompt hashes do not match prompts")
        computed = _json_sha256(
            {
                "name": self.name,
                "prompt_sha256s": self.prompt_sha256s,
                "family_ids": self.family_ids,
                "source_indices": self.source_indices,
            },
            domain=_PARTITION_DOMAIN,
        )
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="partition artifact_sha256",
            ) != computed:
                raise ValueError("transport partition hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def family_count(self) -> int:
        return len(set(self.family_ids))

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "prompt_count": len(self.prompts),
            "family_count": self.family_count,
            "family_ids": tuple(sorted(set(self.family_ids))),
            "prompt_sha256s": self.prompt_sha256s,
            "source_indices": self.source_indices,
            "artifact_sha256": self.artifact_sha256,
            "contains_prompt_text": False,
        }


@dataclass(frozen=True, slots=True)
class TransportCorpusPreclaimView:
    """Hash-only identity of the fixed, unopened calibration-B role.

    This is the object callers may inspect while authenticating a frozen
    candidate.  It deliberately contains no prompt or family strings.  The
    ordered prompt hashes remain visible so a later full-role opening can be
    checked exactly without having disclosed prompt text before the claim.
    """

    corpus_id: str
    role_name: str
    prompt_count: int
    family_count: int
    prompt_sha256s: tuple[str, ...]
    ordered_family_sha256: str
    source_index_sha256: str
    partition_artifact_sha256: str
    prompt_file_sha256: str
    family_file_sha256: str
    audit_file_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            self.corpus_id != _CORPUS_ID
            or self.role_name != "calibration_b_one_shot"
        ):
            raise ValueError("preclaim view does not identify fixed calibration B")
        if self.prompt_count != _FULL_CALIBRATION_B_PROMPT_COUNT:
            raise ValueError(
                "preclaim view must cover the full 96-prompt calibration-B role"
            )
        if type(self.family_count) is not int or self.family_count <= 0:
            raise ValueError("preclaim family_count must be positive")
        if (
            type(self.prompt_sha256s) is not tuple
            or len(self.prompt_sha256s) != self.prompt_count
            or len(set(self.prompt_sha256s)) != self.prompt_count
        ):
            raise ValueError("preclaim prompt hashes must cover full calibration B")
        for index, digest in enumerate(self.prompt_sha256s):
            _require_sha256(digest, label=f"preclaim prompt_sha256s[{index}]")
        for name in (
            "ordered_family_sha256",
            "source_index_sha256",
            "partition_artifact_sha256",
            "prompt_file_sha256",
            "family_file_sha256",
            "audit_file_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"preclaim {name}")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_PRECLAIM_DOMAIN,
        )
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="preclaim artifact_sha256",
            ) != computed:
                raise ValueError("preclaim view artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.gemma3_l3_l4_transport_preclaim",
            "format_version": 1,
            "corpus_id": self.corpus_id,
            "role_name": self.role_name,
            "prompt_count": self.prompt_count,
            "family_count": self.family_count,
            "prompt_sha256s": self.prompt_sha256s,
            "ordered_family_sha256": self.ordered_family_sha256,
            "source_index_sha256": self.source_index_sha256,
            "partition_artifact_sha256": self.partition_artifact_sha256,
            "prompt_file_sha256": self.prompt_file_sha256,
            "family_file_sha256": self.family_file_sha256,
            "audit_file_sha256": self.audit_file_sha256,
            "contains_prompt_text": False,
            "contains_family_strings": False,
            "full_calibration_b_role": True,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class TransportCorpusPlan:
    """Authenticated A-fit/A-guard/B-only view of structured-strong-v9."""

    fit: TransportCorpusPartition
    guard: TransportCorpusPartition
    calibration_b: TransportCorpusPartition
    prompt_file_sha256: str
    family_file_sha256: str
    audit_file_sha256: str
    audit_prompt_index_sha256s: tuple[str, str]
    calibration_b_preclaim: TransportCorpusPreclaimView
    corpus_id: str = _CORPUS_ID

    def __post_init__(self) -> None:
        if self.corpus_id != _CORPUS_ID:
            raise ValueError("transport corpus id is not structured-strong-v9")
        for name in (
            "prompt_file_sha256",
            "family_file_sha256",
            "audit_file_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            type(self.audit_prompt_index_sha256s) is not tuple
            or len(self.audit_prompt_index_sha256s) != 2
        ):
            raise ValueError("audit prompt-index digests must contain fit/guard")
        for value in self.audit_prompt_index_sha256s:
            _require_sha256(value, label="audit prompt-index digest")
        partitions = (self.fit, self.guard, self.calibration_b)
        if any(
            not isinstance(value, TransportCorpusPartition)
            for value in partitions
        ):
            raise TypeError("transport corpus partitions are invalid")
        if not isinstance(
            self.calibration_b_preclaim,
            TransportCorpusPreclaimView,
        ):
            raise TypeError("calibration-B preclaim view is invalid")
        if (
            self.calibration_b_preclaim.corpus_id != self.corpus_id
            or self.calibration_b_preclaim.prompt_file_sha256
            != self.prompt_file_sha256
            or self.calibration_b_preclaim.family_file_sha256
            != self.family_file_sha256
            or self.calibration_b_preclaim.audit_file_sha256
            != self.audit_file_sha256
            or self.calibration_b.prompt_sha256s
            != self.calibration_b_preclaim.prompt_sha256s[
                : len(self.calibration_b.prompt_sha256s)
            ]
        ):
            raise ValueError("calibration-B partition differs from preclaim view")
        for left_index, left in enumerate(partitions):
            for right in partitions[left_index + 1 :]:
                if (
                    set(left.prompt_sha256s) & set(right.prompt_sha256s)
                    or set(left.family_ids) & set(right.family_ids)
                ):
                    raise ValueError(
                        "transport roles must be prompt- and family-disjoint"
                    )

    def preclaim_view(self) -> TransportCorpusPreclaimView:
        """Return the immutable hash-only calibration-B identity."""

        return self.calibration_b_preclaim

    def metadata(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "prompt_file_sha256": self.prompt_file_sha256,
            "family_file_sha256": self.family_file_sha256,
            "audit_file_sha256": self.audit_file_sha256,
            "audit_prompt_index_sha256s": self.audit_prompt_index_sha256s,
            "fit": self.fit.metadata(),
            "guard": self.guard.metadata(),
            "calibration_b": self.calibration_b.metadata(),
            "calibration_b_preclaim": self.calibration_b_preclaim.metadata(),
            "cross_role_prompt_overlap_count": 0,
            "cross_role_family_overlap_count": 0,
            "validation_or_test_exposed": False,
        }


def _partition(
    *,
    name: str,
    prompts: Sequence[str],
    families: Sequence[str],
    indices: Sequence[int],
    limit: int,
) -> TransportCorpusPartition:
    _exact_positive_int(limit, label=f"{name}_limit")
    chosen = tuple(indices[: min(limit, len(indices))])
    if not chosen:
        raise ValueError(f"{name} has no selected prompts")
    return TransportCorpusPartition(
        name=name,
        prompts=tuple(prompts[index] for index in chosen),
        family_ids=tuple(families[index] for index in chosen),
        prompt_sha256s=tuple(_prompt_sha256(prompts[index]) for index in chosen),
        source_indices=chosen,
    )


def _audit_partition(
    raw: Mapping[str, object],
    *,
    name: str,
    prompt_count: int,
) -> tuple[tuple[int, ...], tuple[str, ...], str]:
    partitions = raw.get("calibration_a_family_partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("corpus audit lacks calibration-A partitions")
    value = partitions.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"corpus audit lacks {name} partition")
    indices = _strict_index_tuple(
        value.get("prompt_indices"),
        label=name,
        upper_bound=prompt_count,
    )
    families = value.get("family_ids")
    if (
        not isinstance(families, list)
        or not families
        or any(not isinstance(item, str) or not item for item in families)
        or len(families) != len(set(families))
    ):
        raise ValueError(f"corpus audit {name} family ids are invalid")
    digest = _require_sha256(
        value.get("prompt_index_sha256"),
        label=f"{name} prompt_index_sha256",
    )
    if (
        value.get("name") != name
        or value.get("prompt_count") != len(indices)
        or digest != _prompt_index_sha256(indices)
    ):
        raise ValueError(f"corpus audit {name} counts drifted")
    return indices, tuple(families), digest


def load_transport_corpus_plan(
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    corpus_audit_path: Path | str,
    *,
    fit_limit: int = 8,
    guard_limit: int = 8,
    calibration_b_limit: int = _FULL_CALIBRATION_B_PROMPT_COUNT,
) -> TransportCorpusPlan:
    """Load only the roles admitted by the frozen v9 transport protocol."""

    prompt_path = Path(prompt_splits_path)
    family_path = Path(family_manifest_path)
    audit_path = Path(corpus_audit_path)
    prompt_digest = _file_sha256(prompt_path)
    family_digest = _file_sha256(family_path)
    audit_digest = _file_sha256(audit_path)
    raw = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("transport corpus audit must contain an object")
    prompts: Gemma3PromptSplits = load_gemma3_prompt_splits(prompt_path)
    families: PromptFamilyManifest = load_prompt_family_manifest(
        family_path,
        prompts=prompts,
    )
    if (
        raw.get("schema") != _AUDIT_SCHEMA
        or raw.get("format_version") != _AUDIT_FORMAT_VERSION
        or raw.get("corpus_id") != _CORPUS_ID
        or raw.get("prompt_file_sha256") != prompt_digest
        or raw.get("family_file_sha256") != family_digest
        or raw.get("calibration_a_policy")
        != "family_disjoint_fit_guard_development_only"
        or raw.get("calibration_a_fit_may_train_candidate") is not True
        or raw.get("calibration_a_guard_may_change_candidate") is not False
        or raw.get("calibration_b_policy")
        != "one_shot_frozen_candidate_selection"
        or raw.get("calibration_b_model_evaluated") is not False
        or raw.get("calibration_b_reuse_allowed") is not False
        or raw.get("validation_model_evaluated") is not False
        or raw.get("test_model_evaluated") is not False
        or raw.get("heldout_splits_evaluated") is not False
        or raw.get("heldout_splits_tokenized") is not False
        or raw.get("heldout_splits_unevaluated") is not True
        or raw.get("heldout_splits_untokenized") is not True
        or raw.get("cross_role_family_overlap_count") != 0
        or raw.get("corpus_frozen_before_model_load") is not True
        or raw.get("generator_adjacent_to_outputs") is not True
        or raw.get("generator_bound_by_sha256") is not True
    ):
        raise ValueError("structured-strong-v9 audit is not transport-safe")
    if len(prompts.calibration_b) != _FULL_CALIBRATION_B_PROMPT_COUNT:
        raise ValueError(
            "structured-strong-v9 calibration B must contain exactly "
            "96 prompts"
        )

    fit_indices, fit_families, fit_index_digest = _audit_partition(
        raw,
        name="fit",
        prompt_count=len(prompts.calibration_a),
    )
    guard_indices, guard_families, guard_index_digest = _audit_partition(
        raw,
        name="guard",
        prompt_count=len(prompts.calibration_a),
    )
    if (
        set(fit_indices) & set(guard_indices)
        or set(fit_indices) | set(guard_indices)
        != set(range(len(prompts.calibration_a)))
        or set(fit_families) & set(guard_families)
        or set(fit_families)
        != {
            families.calibration_a[index] for index in fit_indices
        }
        or set(guard_families)
        != {
            families.calibration_a[index] for index in guard_indices
        }
    ):
        raise ValueError("calibration-A fit/guard audit binding drifted")

    fit = _partition(
        name="calibration_a_fit",
        prompts=prompts.calibration_a,
        families=families.calibration_a,
        indices=fit_indices,
        limit=fit_limit,
    )
    guard = _partition(
        name="calibration_a_guard",
        prompts=prompts.calibration_a,
        families=families.calibration_a,
        indices=guard_indices,
        limit=guard_limit,
    )
    full_calibration_b = _partition(
        name="calibration_b_one_shot",
        prompts=prompts.calibration_b,
        families=families.calibration_b,
        indices=tuple(range(len(prompts.calibration_b))),
        limit=len(prompts.calibration_b),
    )
    calibration_b_preclaim = TransportCorpusPreclaimView(
        corpus_id=_CORPUS_ID,
        role_name=full_calibration_b.name,
        prompt_count=len(full_calibration_b.prompts),
        family_count=full_calibration_b.family_count,
        prompt_sha256s=full_calibration_b.prompt_sha256s,
        ordered_family_sha256=_string_sequence_sha256(
            full_calibration_b.family_ids
        ),
        source_index_sha256=_prompt_index_sha256(
            full_calibration_b.source_indices
        ),
        partition_artifact_sha256=full_calibration_b.artifact_sha256,
        prompt_file_sha256=prompt_digest,
        family_file_sha256=family_digest,
        audit_file_sha256=audit_digest,
    )
    calibration_b = _partition(
        name="calibration_b_one_shot",
        prompts=prompts.calibration_b,
        families=families.calibration_b,
        indices=tuple(range(len(prompts.calibration_b))),
        limit=calibration_b_limit,
    )
    return TransportCorpusPlan(
        fit=fit,
        guard=guard,
        calibration_b=calibration_b,
        prompt_file_sha256=prompt_digest,
        family_file_sha256=family_digest,
        audit_file_sha256=audit_digest,
        audit_prompt_index_sha256s=(
            fit_index_digest,
            guard_index_digest,
        ),
        calibration_b_preclaim=calibration_b_preclaim,
    )


def load_transport_corpus_preclaim_view(
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    corpus_audit_path: Path | str,
) -> TransportCorpusPreclaimView:
    """Load only the hash-only identity needed before opening calibration B."""

    corpus = load_transport_corpus_plan(
        prompt_splits_path,
        family_manifest_path,
        corpus_audit_path,
        fit_limit=1,
        guard_limit=1,
        calibration_b_limit=1,
    )
    return corpus.preclaim_view()


@dataclass(frozen=True, slots=True)
class CalibrationBClaim:
    """Immutable receipt for one exclusive calibration-B candidate claim."""

    path: Path
    claim_sha256: str
    candidate_artifact_sha256: str
    partition_artifact_sha256: str
    preclaim_artifact_sha256: str


def _full_calibration_b_preclaim(
    corpus: TransportCorpusPlan | TransportCorpusPreclaimView,
) -> TransportCorpusPreclaimView:
    if isinstance(corpus, TransportCorpusPreclaimView):
        return corpus
    if not isinstance(corpus, TransportCorpusPlan):
        raise TypeError(
            "corpus must be a TransportCorpusPlan or "
            "TransportCorpusPreclaimView"
        )
    preclaim = corpus.preclaim_view()
    partition = corpus.calibration_b
    if (
        len(partition.prompts) != _FULL_CALIBRATION_B_PROMPT_COUNT
        or partition.source_indices
        != tuple(range(_FULL_CALIBRATION_B_PROMPT_COUNT))
        or partition.prompt_sha256s != preclaim.prompt_sha256s
        or partition.artifact_sha256 != preclaim.partition_artifact_sha256
    ):
        raise ValueError(
            "calibration-B claim requires the full fixed 96-prompt role"
        )
    return preclaim


def claim_transport_calibration_b(
    ledger_dir: Path | str,
    *,
    candidate_artifact_sha256: str,
    corpus: TransportCorpusPlan | TransportCorpusPreclaimView,
) -> CalibrationBClaim:
    """Claim the full fixed calibration-B role once using ``O_EXCL``.

    The caller must strict-authenticate the candidate before invoking this
    function.  Lock identity depends only on the fixed full corpus role, never
    on the candidate hash or a caller-selected subset.  A failed run consumes
    the global role claim; callers should therefore complete all dependency,
    artifact, and calibration-A guard checks first.
    """

    candidate_digest = _require_sha256(
        candidate_artifact_sha256,
        label="candidate_artifact_sha256",
    )
    preclaim = _full_calibration_b_preclaim(corpus)
    identity = _json_sha256(
        {
            "corpus_id": preclaim.corpus_id,
            "role_name": preclaim.role_name,
            "preclaim_artifact_sha256": preclaim.artifact_sha256,
            "calibration_b_partition_sha256": (
                preclaim.partition_artifact_sha256
            ),
        },
        domain=_CLAIM_DOMAIN,
    )
    directory = Path(ledger_dir) / "l3-l4-transport-calibration-b"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identity}.json"
    payload: dict[str, object] = {
        "schema": _CLAIM_SCHEMA,
        "format_version": _CLAIM_FORMAT_VERSION,
        "candidate_artifact_sha256": candidate_digest,
        "calibration_b_partition_sha256": (
            preclaim.partition_artifact_sha256
        ),
        "calibration_b_prompt_count": preclaim.prompt_count,
        "calibration_b_prompt_sha256s": preclaim.prompt_sha256s,
        "calibration_b_ordered_family_sha256": (
            preclaim.ordered_family_sha256
        ),
        "calibration_b_source_index_sha256": preclaim.source_index_sha256,
        "preclaim_artifact_sha256": preclaim.artifact_sha256,
        "prompt_file_sha256": preclaim.prompt_file_sha256,
        "family_file_sha256": preclaim.family_file_sha256,
        "audit_file_sha256": preclaim.audit_file_sha256,
        "candidate_authentication_precondition": (
            "strictly_authenticated_before_claim_by_caller"
        ),
        "global_lock_identity_excludes_candidate": True,
        "global_lock_identity_excludes_subset": True,
        "source_authoritative_shadow": True,
        "validation_or_test_accessed": False,
        "claim_identity_sha256": identity,
    }
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            "calibration B is already claimed globally for this fixed "
            "corpus role"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partially written exclusive claim remains evidence that the role
        # was opened.  Do not delete it and accidentally authorize a retry.
        raise
    return CalibrationBClaim(
        path=path,
        claim_sha256=_file_sha256(path),
        candidate_artifact_sha256=candidate_digest,
        partition_artifact_sha256=preclaim.partition_artifact_sha256,
        preclaim_artifact_sha256=preclaim.artifact_sha256,
    )
