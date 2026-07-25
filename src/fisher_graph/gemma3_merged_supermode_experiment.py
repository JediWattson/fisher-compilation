"""Test Fisher-aware merging of low-ranked Gemma block modes.

This is an oracle representation experiment.  It reads the true native
layers-4--6 block delta, preserves the validated rank-639 rotated span, and
compresses only its surviving 31-dimensional low-ranked tail through a
generalized Fisher encoder/decoder learned on calibration A.  Calibration B
selects the smallest passing preregistered tail rank; validation is touched
only for that locked candidate, and test remains parse-and-hash-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .adapters import (
    Gemma3CausalLMAdapter,
    LayerBlockBoundaryPlan,
    ModelAdapter,
)
from .compiler.calibration import CalibrationBatch
from .gemma3_ablation_experiment import (
    _FrozenModelTensorGuard,
    _is_sha256,
    _update_payload_digest,
    _validate_model_metadata,
)
from .gemma3_codimension_rotation_experiment import (
    _codec_prefix_normal,
    _file_sha256,
    _fit_tail_sensitivity,
    _matrix_sha256,
    _native_top1_stream_sha256,
    _semantic_numeric_equal,
    load_gemma3_codimension_rotation_artifact,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    make_causal_lm_calibration_batches,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_gated_executor_experiment import (
    _BoundaryBatch,
    _aggregate_direct_examples,
    _materialize_split,
    _safe_cosine,
)
from .gemma3_rotated_span_executor_experiment import (
    _aggregate_behavior_with_kl,
    _behavior_examples_with_kl,
    _behavior_gates,
    _direct_rows,
    _run_native_stack,
    _run_suffix_from_boundary,
)
from .gemma3_stability_experiment import (
    _CalibrationStreamProvenance,
    _library_versions,
    _ordered_prompt_hash_digest,
    _tokenizer_provenance,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .merged_supermodes import (
    AnchoredTailSupermodeMerge,
    build_anchored_tail_supermode_merge,
)
from .modal_ablation import _example_ids
from .linear_codec import LinearActivationCodec


DEFAULT_PROMPT_SPLITS = Path(
    "examples/gemma3_merged_supermode_oracle_prompts.json"
)
DEFAULT_FAMILY_MANIFEST = Path(
    "examples/gemma3_merged_supermode_oracle_prompt_families.json"
)
DEFAULT_SUPERMODE_RANKS = (0, 1, 2, 4, 8, 16, 24, 28, 30, 31)
DEFAULT_NLL_ATOL = 0.05
DEFAULT_TOP1_MIN = 0.95
DEFAULT_TEACHER_KL_MAX = 0.05
DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX = 0.10
DEFAULT_PER_PROMPT_P10_TOP1_MIN = 0.90
DEFAULT_IDENTITY_NLL_ATOL = 1e-6
DEFAULT_MINIMUM_SUBSPACE_STABILITY = 0.90

_PROMPT_STATUS = (
    "merged_supermode_oracle_fresh_calibration_a_train_b_selection_"
    "validation_locked_test_hash_only"
)
_ARTIFACT_SCHEMA = "fisher_graph.gemma3_merged_supermode_oracle"
_ARTIFACT_FORMAT_VERSION = 1
_PAYLOAD_DOMAIN = b"fisher_graph.gemma3_merged_supermode_payload.v1\0"
_REPORT_DOMAIN = b"fisher_graph.gemma3_merged_supermode_report.v1\0"
_SOURCE_ROTATION_SCHEMA = "fisher_graph.gemma3_codimension_rotation"
_FIT_OBJECTIVE = (
    "unregularized_generalized_fisher_of_block_delta_second_"
    "moment_and_downstream_pseudo_top1_score_fisher"
)
_PROJECTION_POLICY = (
    "target_informed_native_delta_anchored_tail_supermode_oracle"
)
_SELECTION_POLICY = (
    "smallest_stable_q_below_full_tail_passing_all_b_gates_"
    "after_rank639_and_rank640_controls"
)
_VALIDATION_POLICY = (
    "one_locked_merged_candidate_only_if_calibration_b_passes"
)
_TEST_POLICY = "parse_validate_hash_only"


@dataclass(frozen=True, slots=True)
class SupermodeCandidate:
    candidate_id: str
    kind: str
    supermode_rank: int | None
    total_rank: int
    residual_width: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or self.kind
            not in {
                "merged_tail_supermodes",
                "locked_rotated_span_control",
                "native_identity_control",
            }
            or type(self.total_rank) is not int
            or type(self.residual_width) is not int
            or self.residual_width < 3
            or not 0 < self.total_rank <= self.residual_width
        ):
            raise ValueError("merged-supermode candidate is invalid")
        if self.kind == "native_identity_control":
            if (
                self.supermode_rank is not None
                or self.total_rank != self.residual_width
            ):
                raise ValueError("identity candidate geometry is invalid")
        elif (
            type(self.supermode_rank) is not int
            or self.supermode_rank < 0
            or self.total_rank >= self.residual_width
        ):
            raise ValueError("tail candidate geometry is invalid")

    @property
    def removed_dimensions(self) -> int:
        return self.residual_width - self.total_rank

    @property
    def retained_fraction(self) -> float:
        return self.total_rank / self.residual_width

    def metadata(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "supermode_rank": self.supermode_rank,
            "total_rank": self.total_rank,
            "residual_width": self.residual_width,
            "removed_dimensions": self.removed_dimensions,
            "retained_fraction": self.retained_fraction,
            "projection": (
                "native_block_delta_identity_replay"
                if self.kind == "native_identity_control"
                else (
                    "validated_rotated_rank_639_span"
                    if self.kind == "locked_rotated_span_control"
                    else (
                        "anchored_generalized_fisher_tail_"
                        "supermode_merge"
                    )
                )
            ),
        }


def default_gemma3_merged_supermode_output(
    model_id: str,
    start_layer: int,
    end_layer: int,
) -> Path:
    slug = model_id.replace("/", "--")
    return (
        Path(".local-runs")
        / slug
        / (
            f"layers-{start_layer}-{end_layer}-"
            "merged-supermode-oracle.pt"
        )
    )


def _scientific_payload_sha256(
    payload: Mapping[str, object],
) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, payload)
    return digest.hexdigest()


def _report_sha256(report: Mapping[str, object]) -> str:
    encoded = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_REPORT_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()


def _semantic_tensor_equal(
    left: object,
    right: object,
    *,
    rel_tol: float = 1e-11,
    abs_tol: float = 1e-13,
) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.device.type != right.device.type
        ):
            return False
        if left.is_floating_point():
            return bool(
                torch.allclose(
                    left,
                    right,
                    rtol=rel_tol,
                    atol=abs_tol,
                )
            )
        return bool(torch.equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _semantic_tensor_equal(
                left[key],
                right[key],
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(
        right,
        type(left),
    ):
        return len(left) == len(right) and all(
            _semantic_tensor_equal(
                left_item,
                right_item,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            for left_item, right_item in zip(
                left,
                right,
                strict=True,
            )
        )
    return _semantic_numeric_equal(
        left,
        right,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def _finite(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return result


def _candidate_schedule(
    merge: AnchoredTailSupermodeMerge,
    supermode_ranks: Sequence[int],
) -> tuple[SupermodeCandidate, ...]:
    ranks = tuple(supermode_ranks)
    if (
        not ranks
        or any(type(rank) is not int for rank in ranks)
        or tuple(sorted(set(ranks))) != ranks
        or ranks[-1] != merge.maximum_supermodes
        or ranks[0] < 0
    ):
        raise ValueError(
            "supermode ranks must be unique ascending values ending at "
            "the full surviving-tail rank"
        )
    candidates = []
    for rank in ranks:
        total = merge.total_rank(rank)
        kind = (
            "locked_rotated_span_control"
            if rank == merge.maximum_supermodes
            else "merged_tail_supermodes"
        )
        candidates.append(
            SupermodeCandidate(
                candidate_id=f"tail_q_{rank}.rank_{total}",
                kind=kind,
                supermode_rank=rank,
                total_rank=total,
                residual_width=merge.width,
            )
        )
    candidates.append(
        SupermodeCandidate(
            candidate_id=f"native_identity.rank_{merge.width}",
            kind="native_identity_control",
            supermode_rank=None,
            total_rank=merge.width,
            residual_width=merge.width,
        )
    )
    return tuple(candidates)


def _prompt_hashes(metadata: Mapping[str, object]) -> set[str]:
    per_prompt = metadata.get("per_prompt_sha256")
    if not isinstance(per_prompt, Mapping):
        raise ValueError("prompt metadata is missing per-prompt hashes")
    result: set[str] = set()
    for values in per_prompt.values():
        if not isinstance(values, (list, tuple)):
            raise ValueError("per-prompt hashes must be sequences")
        for value in values:
            if not _is_sha256(value) or value in result:
                raise ValueError("prompt hashes are invalid or duplicated")
            result.add(value)
    return result


def _fixture_prompt_hashes(path: Path) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema") != "fisher_graph.gemma3_prompt_splits"
    ):
        raise ValueError(f"invalid predecessor prompt fixture: {path}")
    result = set()
    for split in (
        "calibration_a",
        "calibration_b",
        "validation",
        "test",
    ):
        values = raw.get(split)
        if not isinstance(values, list):
            raise ValueError(f"invalid predecessor prompt split: {path}")
        for prompt in values:
            if not isinstance(prompt, str):
                raise ValueError(f"invalid predecessor prompt: {path}")
            encoded = json.dumps(
                [prompt],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            result.add(hashlib.sha256(encoded).hexdigest())
    return result


def _assert_repository_prompt_disjointness(
    *,
    prompt_path: Path,
    prompt_metadata: Mapping[str, object],
) -> dict[str, object]:
    fresh = _prompt_hashes(prompt_metadata)
    examples = Path(__file__).resolve().parents[2] / "examples"
    predecessors: dict[str, tuple[str, ...]] = {}
    for path in sorted(examples.glob("gemma3_*_prompts.json")):
        if path.resolve() == prompt_path.resolve():
            continue
        hashes = _fixture_prompt_hashes(path)
        overlap = fresh & hashes
        if overlap:
            raise ValueError(
                f"merged-supermode prompts overlap {path.name}"
            )
        predecessors[path.name] = tuple(sorted(hashes))
    legacy = examples / "gemma3_prompts.txt"
    if legacy.is_file():
        hashes = {
            hashlib.sha256(
                json.dumps(
                    [line.strip()],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for line in legacy.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if fresh & hashes:
            raise ValueError(
                "merged-supermode prompts overlap gemma3_prompts.txt"
            )
        predecessors[legacy.name] = tuple(sorted(hashes))
    if not predecessors:
        raise ValueError("no predecessor prompt fixtures were found")
    return {
        "fresh_prompt_sha256": tuple(sorted(fresh)),
        "fresh_count": len(fresh),
        "predecessor_prompt_sha256": predecessors,
        "predecessor_counts": {
            name: len(values)
            for name, values in predecessors.items()
        },
        "overlap_counts": {
            name: 0 for name in predecessors
        },
        "verified_before_model_load_or_tokenization": True,
    }


def _load_family_manifest(
    path: Path,
    *,
    prompt_path: Path,
    prompt_metadata: Mapping[str, object],
) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema",
        "format_version",
        "prompt_fixture",
        "prompt_fixture_sha256",
        "scientific_policy",
        "roles",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != fields
        or raw["schema"]
        != "fisher_graph.gemma3_prompt_family_manifest"
        or raw["format_version"] != 1
        or raw["prompt_fixture"] != prompt_path.name
        or raw["prompt_fixture_sha256"] != _file_sha256(prompt_path)
        or raw["scientific_policy"]
        != "one_template_family_one_role_no_cross_split_reuse"
    ):
        raise ValueError("merged-supermode family manifest is invalid")
    roles = raw["roles"]
    counts = prompt_metadata.get("counts")
    normalized = prompt_metadata.get("normalized_sha256")
    split_names = (
        "calibration_a",
        "calibration_b",
        "validation",
        "test",
    )
    if (
        not isinstance(roles, Mapping)
        or tuple(roles) != split_names
        or not isinstance(counts, Mapping)
        or not isinstance(normalized, Mapping)
    ):
        raise ValueError("family manifest roles are invalid")
    family_ids: set[str] = set()
    family_suffixes: set[str] = set()
    for role in split_names:
        entry = roles[role]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"normalized_sha256", "families"}
            or entry["normalized_sha256"] != normalized[role]
            or not isinstance(entry["families"], list)
        ):
            raise ValueError("family manifest role entry is invalid")
        cursor = 0
        for family in entry["families"]:
            if (
                not isinstance(family, Mapping)
                or set(family)
                != {"id", "description", "start", "count"}
                or not isinstance(family["id"], str)
                or not family["id"].startswith(f"{role}.")
                or family["id"] in family_ids
                or not isinstance(family["description"], str)
                or not family["description"]
                or family["start"] != cursor
                or type(family["count"]) is not int
                or family["count"] <= 0
            ):
                raise ValueError("family manifest range is invalid")
            suffix = family["id"].split(".", 1)[1]
            if suffix in family_suffixes:
                raise ValueError(
                    "template family suffix crosses role boundaries"
                )
            family_ids.add(family["id"])
            family_suffixes.add(suffix)
            cursor += family["count"]
        if cursor != counts[role]:
            raise ValueError("family manifest does not cover its role")
    return copy.deepcopy(dict(raw))


def _subspace_stability(
    pooled: AnchoredTailSupermodeMerge,
    split_zero: AnchoredTailSupermodeMerge,
    split_one: AnchoredTailSupermodeMerge,
    *,
    supermode_ranks: Sequence[int],
    minimum_alignment: float,
) -> tuple[dict[str, object], ...]:
    rows = []
    maximum = pooled.maximum_supermodes
    for rank in supermode_ranks:
        if rank in {0, maximum}:
            correlations = torch.ones(
                max(rank, 1),
                dtype=torch.float64,
            )
        else:
            left = torch.linalg.qr(
                split_zero.codec.decoder[:, :rank],
                mode="reduced",
            ).Q
            right = torch.linalg.qr(
                split_one.codec.decoder[:, :rank],
                mode="reduced",
            ).Q
            correlations = torch.linalg.svdvals(left.T @ right)
        mean_squared = float(correlations.square().mean().item())
        minimum = float(correlations.min().item())
        rows.append(
            {
                "supermode_rank": rank,
                "total_rank": pooled.total_rank(rank),
                "retained_factorized_weighted_fraction": (
                    pooled.retained_weighted_fraction(rank)
                ),
                "split_half_mean_squared_canonical_correlation": (
                    mean_squared
                ),
                "split_half_minimum_canonical_correlation": minimum,
                "minimum_required_mean_squared_correlation": (
                    minimum_alignment
                ),
                "stable": mean_squared >= minimum_alignment,
            }
        )
    return tuple(rows)


def _evaluate_candidates(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    plan: LayerBlockBoundaryPlan,
    merge: AnchoredTailSupermodeMerge,
    candidates: Sequence[SupermodeCandidate],
    source_rotated_projector: object,
) -> dict[str, object]:
    behavior_rows = {
        candidate.candidate_id: [] for candidate in candidates
    }
    direct_rows = {
        candidate.candidate_id: [] for candidate in candidates
    }
    sequence_offset = 0
    identity_logit_error = 0.0
    span_boundary_error = 0.0
    native_batches = 0
    with torch.no_grad():
        for batch in batches:
            ids = _example_ids(batch, sequence_offset=sequence_offset)
            sequence_offset += batch.batch_size
            native = _run_native_stack(
                adapter,
                batch,
                plan=plan,
                full_logits=True,
            )
            if native.logits is None:
                raise RuntimeError("native oracle logits are missing")
            native_batches += 1
            boundary = _BoundaryBatch(
                input_hidden=native.block_input.detach(),
                output_hidden=native.block_output.detach(),
                valid_positions=batch.valid_positions,
                logical_positions=native.sequence.logical_positions,
                example_ids=ids,
            )
            source_rotated = source_rotated_projector.project_output(
                native.block_input,
                native.block_output,
                valid_positions=batch.valid_positions,
            )
            for candidate in candidates:
                if candidate.kind == "native_identity_control":
                    projected = native.block_output
                else:
                    assert candidate.supermode_rank is not None
                    projected = merge.project_output(
                        native.block_input,
                        native.block_output,
                        valid_positions=batch.valid_positions,
                        supermode_rank=candidate.supermode_rank,
                    )
                    if (
                        candidate.kind
                        == "locked_rotated_span_control"
                    ):
                        span_boundary_error = max(
                            span_boundary_error,
                            float(
                                (
                                    projected.to(torch.float64)
                                    - source_rotated.to(torch.float64)
                                )
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                _, predicted_logits, _ = _run_suffix_from_boundary(
                    adapter,
                    batch,
                    plan=plan,
                    sequence=native.sequence,
                    boundary_output=projected,
                    full_logits=True,
                )
                if predicted_logits is None:
                    raise RuntimeError(
                        "candidate suffix logits are missing"
                    )
                if candidate.kind == "native_identity_control":
                    identity_logit_error = max(
                        identity_logit_error,
                        float(
                            (
                                predicted_logits.to(torch.float64)
                                - native.logits.to(torch.float64)
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                behavior_rows[candidate.candidate_id].extend(
                    _behavior_examples_with_kl(
                        batch=batch,
                        example_ids=ids,
                        baseline_logits=native.logits,
                        predicted_logits=predicted_logits,
                    )
                )
                direct_rows[candidate.candidate_id].extend(
                    _direct_rows(boundary, projected)
                )
                del predicted_logits, projected
            del native
    width = merge.width
    return {
        "behavior": {
            candidate_id: _aggregate_behavior_with_kl(rows)
            for candidate_id, rows in behavior_rows.items()
        },
        "direct": {
            candidate_id: _aggregate_direct_examples(
                rows,
                width=width,
            )
            for candidate_id, rows in direct_rows.items()
        },
        "execution_audit": {
            "native_batches": native_batches,
            "native_block_executed_once_per_batch": True,
            "candidate_suffix_replays_per_batch": len(candidates),
            "native_identity_maximum_logit_error": (
                identity_logit_error
            ),
            "rank_639_vs_source_rotated_maximum_boundary_error": (
                span_boundary_error
            ),
        },
    }


def _candidate_gates(
    behavior: Mapping[str, object],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    return _behavior_gates(
        behavior,
        nll_atol=thresholds["nll_atol"],
        top1_min=thresholds["top1_min"],
        teacher_kl_max=thresholds["teacher_kl_max"],
        p90_abs_nll_max=thresholds["p90_abs_nll_max"],
        p10_top1_min=thresholds["p10_top1_min"],
    )


def _build_ledger(
    *,
    candidates: Sequence[SupermodeCandidate],
    behavior: Mapping[str, Mapping[str, object]],
    direct: Mapping[str, Mapping[str, object]],
    stability: Mapping[int, Mapping[str, object]],
    thresholds: Mapping[str, float],
) -> list[dict[str, object]]:
    rows = []
    for candidate in candidates:
        gates = _candidate_gates(
            behavior[candidate.candidate_id],
            thresholds=thresholds,
        )
        stable = (
            True
            if candidate.supermode_rank is None
            else stability[candidate.supermode_rank]["stable"] is True
        )
        eligible = (
            candidate.kind == "merged_tail_supermodes" and stable
        )
        rows.append(
            {
                "candidate": candidate.metadata(),
                "behavior": copy.deepcopy(
                    dict(behavior[candidate.candidate_id])
                ),
                "direct_diagnostic": copy.deepcopy(
                    dict(direct[candidate.candidate_id])
                ),
                "behavior_gates": gates,
                "behavior_fidelity_passed": all(gates.values()),
                "calibration_a_subspace_stable": stable,
                "eligible_for_lock": eligible,
                "direct_metrics_influence_lock": False,
            }
        )
    return rows


def _lock_candidate(
    *,
    candidates: Sequence[SupermodeCandidate],
    ledger: Sequence[Mapping[str, object]],
    controls_passed: bool,
) -> tuple[SupermodeCandidate, dict[str, object]]:
    by_id = {
        row["candidate"]["candidate_id"]: row  # type: ignore[index]
        for row in ledger
    }
    if controls_passed:
        for candidate in candidates:
            row = by_id[candidate.candidate_id]
            if (
                row["eligible_for_lock"] is True
                and row["behavior_fidelity_passed"] is True
            ):
                return candidate, {
                    "ordering": (
                        "ascending_supermode_rank_then_rotated_span_"
                        "control_then_native_identity"
                    ),
                    "locked_candidate_id": candidate.candidate_id,
                    "selection_failed": False,
                    "merged_candidate_found": True,
                    "reason": (
                        "smallest_stable_merged_rank_passing_all_"
                        "calibration_b_behavior_gates"
                    ),
                    "calibration_b_only": True,
                }
    span = next(
        candidate
        for candidate in candidates
        if candidate.kind == "locked_rotated_span_control"
    )
    return span, {
        "ordering": (
            "ascending_supermode_rank_then_rotated_span_control_then_"
            "native_identity"
        ),
        "locked_candidate_id": span.candidate_id,
        "selection_failed": True,
        "merged_candidate_found": False,
        "reason": (
            "controls_failed"
            if not controls_passed
            else "no_stable_merged_rank_passed_calibration_b"
        ),
        "calibration_b_only": True,
    }


def _build_report(
    payload: Mapping[str, object],
    *,
    output: Path,
    scientific_digest: str,
) -> dict[str, object]:
    calibration_a = payload["calibration_a"]
    assert isinstance(calibration_a, Mapping)
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "scientific_status": copy.deepcopy(
            dict(payload["scientific_status"])  # type: ignore[arg-type]
        ),
        "model": copy.deepcopy(
            dict(payload["model"])  # type: ignore[arg-type]
        ),
        "source_rotation": copy.deepcopy(
            dict(payload["source_rotation"])  # type: ignore[arg-type]
        ),
        "protocol": copy.deepcopy(
            dict(payload["protocol"])  # type: ignore[arg-type]
        ),
        "analysis": {
            "calibration_a": {
                "source_fit_summary": copy.deepcopy(
                    dict(calibration_a["source_fit_summary"])  # type: ignore[arg-type]
                ),
                "supermode_spectrum": copy.deepcopy(
                    calibration_a["supermode_spectrum"]
                ),
                "split_half_subspace_stability": copy.deepcopy(
                    calibration_a[
                        "split_half_subspace_stability"
                    ]
                ),
                "tokenized_stream": copy.deepcopy(
                    dict(calibration_a["tokenized_stream"])  # type: ignore[arg-type]
                ),
            },
            "selection": copy.deepcopy(
                dict(payload["selection"])  # type: ignore[arg-type]
            ),
            "validation": copy.deepcopy(
                dict(payload["validation"])  # type: ignore[arg-type]
            ),
        },
        "artifact": {
            "tensor_output": output.name,
            "contains_source_model_weights": False,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "scientific_payload_sha256": scientific_digest,
        },
    }


def run_gemma3_merged_supermode_oracle(
    *,
    rotation_artifact_path: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    prompt_splits_path: Path | str = DEFAULT_PROMPT_SPLITS,
    family_manifest_path: Path | str = DEFAULT_FAMILY_MANIFEST,
    max_length: int = 128,
    tokenization_batch_size: int = 2,
    supermode_ranks: Sequence[int] = DEFAULT_SUPERMODE_RANKS,
    selection_nll_atol: float = DEFAULT_NLL_ATOL,
    selection_top1_min: float = DEFAULT_TOP1_MIN,
    selection_teacher_kl_max: float = DEFAULT_TEACHER_KL_MAX,
    selection_p90_abs_nll_max: float = (
        DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX
    ),
    selection_p10_top1_min: float = (
        DEFAULT_PER_PROMPT_P10_TOP1_MIN
    ),
    identity_nll_atol: float = DEFAULT_IDENTITY_NLL_ATOL,
    minimum_subspace_stability: float = (
        DEFAULT_MINIMUM_SUBSPACE_STABILITY
    ),
    device_name: str = "cpu",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Fit on A, select a merged oracle on B, and validate one lock."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    thresholds = {
        "nll_atol": _finite(
            selection_nll_atol,
            label="selection_nll_atol",
            minimum=0.0,
        ),
        "top1_min": _finite(
            selection_top1_min,
            label="selection_top1_min",
            minimum=0.0,
            maximum=1.0,
        ),
        "teacher_kl_max": _finite(
            selection_teacher_kl_max,
            label="selection_teacher_kl_max",
            minimum=0.0,
        ),
        "p90_abs_nll_max": _finite(
            selection_p90_abs_nll_max,
            label="selection_p90_abs_nll_max",
            minimum=0.0,
        ),
        "p10_top1_min": _finite(
            selection_p10_top1_min,
            label="selection_p10_top1_min",
            minimum=0.0,
            maximum=1.0,
        ),
    }
    identity_tolerance = _finite(
        identity_nll_atol,
        label="identity_nll_atol",
        minimum=0.0,
    )
    stability_minimum = _finite(
        minimum_subspace_stability,
        label="minimum_subspace_stability",
        minimum=0.0,
        maximum=1.0,
    )

    source_path = Path(rotation_artifact_path)
    rotation = load_gemma3_codimension_rotation_artifact(source_path)
    source_file_sha = _file_sha256(source_path)
    source_model = rotation["model"]
    source_metadata = rotation["metadata"]
    source_protocol = source_metadata["protocol"]  # type: ignore[index]
    source_status = rotation["report"]["scientific_status"]  # type: ignore[index]
    source_codec = rotation["output_codec"]
    source_sensitivity = rotation["calibration_a_sensitivity"]
    source_projector = rotation["rotated_projector"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            source_model,
            source_metadata,
            source_protocol,
            source_status,
            source_sensitivity,
        )
    ) or not isinstance(source_codec, LinearActivationCodec):
        raise ValueError("rotation source payload is invalid")
    width = source_protocol.get("residual_width")
    start_layer = source_protocol.get("start_layer")
    end_layer = source_protocol.get("end_layer_inclusive")
    layer_ids = source_protocol.get("layer_ids")
    boundaries = source_protocol.get("canonical_boundaries")
    tail_basis = source_sensitivity.get("tail_basis")
    if (
        source_status.get("rank_639_fidelity_viable") is not True
        or source_status.get("basis_ordering_supported") is not True
        or source_status.get("selection_failed") is not False
        or source_status.get("test_evaluated") is not False
        or type(width) is not int
        or width < 3
        or type(start_layer) is not int
        or type(end_layer) is not int
        or not isinstance(layer_ids, tuple)
        or not isinstance(boundaries, tuple)
        or not isinstance(tail_basis, Tensor)
        or tail_basis.ndim != 2
        or tail_basis.shape[0] != width
        or source_codec.width != width
        or source_projector.width != width
    ):
        raise ValueError(
            "merged-supermode oracle requires the validated rotated "
            "rank-639 predecessor"
        )
    if source_model.get("model_id") != model_id:
        raise ValueError("requested model_id does not match rotation source")
    if revision is not None and revision not in {
        source_model.get("requested_revision"),
        source_model.get("resolved_commit"),
    }:
        raise ValueError("explicit revision does not match rotation source")
    ranks = tuple(supermode_ranks)
    if width == 640 and (
        ranks != DEFAULT_SUPERMODE_RANKS
        or tail_basis.shape[1] != 32
        or thresholds
        != {
            "nll_atol": DEFAULT_NLL_ATOL,
            "top1_min": DEFAULT_TOP1_MIN,
            "teacher_kl_max": DEFAULT_TEACHER_KL_MAX,
            "p90_abs_nll_max": (
                DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX
            ),
            "p10_top1_min": (
                DEFAULT_PER_PROMPT_P10_TOP1_MIN
            ),
        }
        or stability_minimum != DEFAULT_MINIMUM_SUBSPACE_STABILITY
        or identity_tolerance != DEFAULT_IDENTITY_NLL_ATOL
        or max_length != 128
        or tokenization_batch_size != 2
    ):
        raise ValueError(
            "width-640 merged-supermode experiment must use the "
            "preregistered ranks and gates"
        )

    prompt_path = Path(prompt_splits_path)
    manifest_path = Path(family_manifest_path)
    prompts = load_gemma3_prompt_splits(prompt_path)
    prompt_metadata = prompts.metadata()
    if (
        prompts.scientific_status != _PROMPT_STATUS
        or prompt_metadata["counts"]
        != {
            "calibration_a": 64,
            "calibration_b": 16,
            "validation": 16,
            "test": 16,
        }
    ):
        raise ValueError(
            "merged-supermode prompt fixture is noncanonical"
        )
    family_manifest = _load_family_manifest(
        manifest_path,
        prompt_path=prompt_path,
        prompt_metadata=prompt_metadata,
    )
    prompt_disjointness = _assert_repository_prompt_disjointness(
        prompt_path=prompt_path,
        prompt_metadata=prompt_metadata,
    )

    resolved_output = (
        default_gemma3_merged_supermode_output(
            model_id,
            start_layer,
            end_layer,
        )
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    if (
        resolved_output.parent.resolve()
        != source_path.parent.resolve()
    ):
        raise ValueError(
            "output must share a directory with its authenticated "
            "rotation predecessor"
        )
    if resolved_output.exists() or resolved_output.with_suffix(
        ".json"
    ).exists():
        raise FileExistsError(
            "refusing to overwrite an existing merged-supermode "
            "artifact; use a new held-out fixture for a new run"
        )

    device = resolve_torch_device(device_name)
    if device.type == "mps":
        raise ValueError(
            "merged-supermode fitting requires CPU or CUDA because its "
            "matrix audits use float64"
        )
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    requested_revision = (
        revision
        if revision is not None
        else (
            source_model.get("resolved_commit")
            or source_model.get("requested_revision")
        )
    )
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=requested_revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    model.eval()
    model.requires_grad_(False)
    guard = _FrozenModelTensorGuard(model)
    adapter = Gemma3CausalLMAdapter(model)
    plan = adapter.plan_layer_block(start_layer, end_layer)
    if (
        plan.activation_sites != boundaries
        or plan.layer_ids != layer_ids
        or plan.widths != (width,) * len(boundaries)
    ):
        raise ValueError("live adapter block does not match rotation source")
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=requested_revision,
    )
    for field in (
        "model_id",
        "config_sha256",
        "hidden_size",
        "num_hidden_layers",
    ):
        if source_model.get(field) != model_metadata.get(field):
            raise ValueError(f"live model {field} does not match source")
    if (
        source_model.get("resolved_commit") is not None
        and model_metadata.get("resolved_commit") is not None
        and source_model["resolved_commit"]
        != model_metadata["resolved_commit"]
    ):
        raise ValueError("live model commit does not match source")

    calibration_a_provenance = _CalibrationStreamProvenance(
        "calibration_a",
        prompts.calibration_a,
    )
    calibration_a_batches = make_causal_lm_calibration_batches(
        tokenizer,
        prompts.calibration_a,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    sensitivity = _fit_tail_sensitivity(
        adapter,
        calibration_a_provenance.wrap(calibration_a_batches),
        plan=plan,
        tail_basis=tail_basis,
        codec_normal=_codec_prefix_normal(source_codec.decoder),
    )
    calibration_a_stream = calibration_a_provenance.metadata()
    score_fisher = sensitivity["score_fisher"]
    delta_second_moment = sensitivity["delta_second_moment"]
    split_score = sensitivity["split_half_score_fisher"]
    split_delta = sensitivity["split_half_delta_second_moment"]
    if not all(
        isinstance(value, Tensor)
        for value in (
            score_fisher,
            delta_second_moment,
            split_score,
            split_delta,
        )
    ):
        raise RuntimeError("calibration-A tail moments are invalid")
    merge = build_anchored_tail_supermode_merge(
        tail_basis=tail_basis,
        locked_normal=source_projector.normal,
        score_fisher=score_fisher,
        delta_second_moment=delta_second_moment,
    )
    split_merges = tuple(
        build_anchored_tail_supermode_merge(
            tail_basis=tail_basis,
            locked_normal=source_projector.normal,
            score_fisher=split_score[index],
            delta_second_moment=split_delta[index],
        )
        for index in range(2)
    )
    candidates = _candidate_schedule(merge, ranks)
    stability_rows = _subspace_stability(
        merge,
        split_merges[0],
        split_merges[1],
        supermode_ranks=ranks,
        minimum_alignment=stability_minimum,
    )
    stability_by_rank = {
        int(row["supermode_rank"]): row for row in stability_rows
    }
    spectrum = [
        {
            "supermode_rank": rank,
            "total_rank": merge.total_rank(rank),
            "retained_factorized_weighted_fraction": (
                merge.retained_weighted_fraction(rank)
            ),
            "discarded_factorized_weighted_fraction": (
                1.0 - merge.retained_weighted_fraction(rank)
            ),
        }
        for rank in ranks
    ]
    guard.assert_unchanged()

    selection_batches, selection_stream = _materialize_split(
        tokenizer,
        prompts.calibration_b,
        split_name="calibration_b",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    selection_result = _evaluate_candidates(
        adapter,
        selection_batches,
        plan=plan,
        merge=merge,
        candidates=candidates,
        source_rotated_projector=source_projector,
    )
    selection_behavior = selection_result["behavior"]
    selection_direct = selection_result["direct"]
    execution_audit = selection_result["execution_audit"]
    assert isinstance(selection_behavior, Mapping)
    assert isinstance(selection_direct, Mapping)
    assert isinstance(execution_audit, Mapping)
    ledger = _build_ledger(
        candidates=candidates,
        behavior=selection_behavior,  # type: ignore[arg-type]
        direct=selection_direct,  # type: ignore[arg-type]
        stability=stability_by_rank,
        thresholds=thresholds,
    )
    identity = candidates[-1]
    span = candidates[-2]
    identity_row = ledger[-1]
    span_row = ledger[-2]
    identity_behavior_gates = _behavior_gates(
        identity_row["behavior"],  # type: ignore[arg-type]
        nll_atol=identity_tolerance,
        top1_min=1.0,
        teacher_kl_max=identity_tolerance,
        p90_abs_nll_max=identity_tolerance,
        p10_top1_min=1.0,
    )
    identity_direct = identity_row["direct_diagnostic"]
    assert isinstance(identity_direct, Mapping)
    identity_passed = (
        all(identity_behavior_gates.values())
        and float(identity_direct["block_delta_nrmse"]) <= 1e-12
        and float(identity_direct["block_delta_cosine"])
        >= 1.0 - 1e-12
        and float(
            execution_audit["native_identity_maximum_logit_error"]
        )
        <= identity_tolerance
    )
    span_equivalence_passed = (
        float(
            execution_audit[
                "rank_639_vs_source_rotated_maximum_boundary_error"
            ]
        )
        <= 1e-5
    )
    span_passed = (
        span_row["behavior_fidelity_passed"] is True
        and span_equivalence_passed
    )
    controls_passed = identity_passed and span_passed
    locked, lock = _lock_candidate(
        candidates=candidates,
        ledger=ledger,
        controls_passed=controls_passed,
    )
    guard.assert_unchanged()

    validation_evaluated = (
        lock["selection_failed"] is False
        and locked.kind == "merged_tail_supermodes"
    )
    validation_stream: dict[str, object] | None = None
    validation_behavior: dict[str, object] | None = None
    validation_direct: dict[str, object] | None = None
    validation_gates: dict[str, bool] | None = None
    validation_execution: dict[str, object] | None = None
    validation_passed = False
    if validation_evaluated:
        validation_batches, validation_stream = _materialize_split(
            tokenizer,
            prompts.validation,
            split_name="validation",
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        validation_result = _evaluate_candidates(
            adapter,
            validation_batches,
            plan=plan,
            merge=merge,
            candidates=(locked,),
            source_rotated_projector=source_projector,
        )
        validation_behavior = validation_result["behavior"][
            locked.candidate_id
        ]
        validation_direct = validation_result["direct"][
            locked.candidate_id
        ]
        validation_execution = validation_result["execution_audit"]
        validation_gates = _candidate_gates(
            validation_behavior,
            thresholds=thresholds,
        )
        validation_passed = all(validation_gates.values())
    guard.assert_unchanged()

    merged_representation_viable = (
        validation_evaluated and validation_passed
    )
    source_binding = {
        "schema": rotation["report"]["schema"],  # type: ignore[index]
        "format_version": rotation["report"]["format_version"],  # type: ignore[index]
        "scientific_payload_sha256": source_metadata[
            "scientific_payload_sha256"
        ],
        "report_sha256": source_metadata["report_sha256"],
        "tensor_file": source_path.name,
        "tensor_file_sha256": source_file_sha,
        "tail_basis_sha256": _matrix_sha256(
            tail_basis,
            domain=(
                b"fisher_graph.merged_supermode.source_tail_basis.v1\0"
            ),
        ),
        "locked_normal_sha256": _matrix_sha256(
            source_projector.normal,
            domain=(
                b"fisher_graph.merged_supermode.source_locked_normal.v1\0"
            ),
        ),
        "model_binding": {
            field: source_model.get(field)
            for field in (
                "model_id",
                "config_sha256",
                "resolved_commit",
                "hidden_size",
                "num_hidden_layers",
            )
        },
        "block_geometry": {
            "start_layer": start_layer,
            "end_layer_inclusive": end_layer,
            "layer_ids": layer_ids,
            "canonical_boundaries": boundaries,
            "residual_width": width,
        },
        "locked_candidate": copy.deepcopy(
            dict(rotation["locked_candidate"])  # type: ignore[arg-type]
        ),
        "rank_639_fidelity_viable": True,
        "basis_ordering_supported": True,
        "prompt_disjointness": prompt_disjointness,
    }
    tokenized_splits = {
        "calibration_a": calibration_a_stream,
        "calibration_b": selection_stream,
    }
    if validation_evaluated:
        assert validation_stream is not None
        tokenized_splits["validation"] = validation_stream
    protocol = {
        "start_layer": start_layer,
        "end_layer_inclusive": end_layer,
        "layer_ids": layer_ids,
        "canonical_boundaries": boundaries,
        "residual_width": width,
        "source_tail_width": merge.tail_width,
        "preserved_prefix_rank": merge.preserved_prefix_rank,
        "surviving_tail_rank": merge.maximum_supermodes,
        "supermode_rank_schedule": ranks,
        "candidate_schedule": tuple(
            candidate.metadata() for candidate in candidates
        ),
        "fit_split": "calibration_a_only",
        "fit_objective": _FIT_OBJECTIVE,
        "projection": _PROJECTION_POLICY,
        "selection_policy": _SELECTION_POLICY,
        "validation_policy": _VALIDATION_POLICY,
        "test_policy": _TEST_POLICY,
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "thresholds": thresholds,
        "identity_nll_atol": identity_tolerance,
        "minimum_subspace_stability": stability_minimum,
        "prompt_fixture_file_sha256": _file_sha256(prompt_path),
        "family_manifest_file_sha256": _file_sha256(manifest_path),
        "family_manifest": family_manifest,
        "prompt_splits": prompt_metadata,
        "tokenized_splits": tokenized_splits,
        "model_state_guard": guard.metadata(),
        "library_versions": _library_versions(),
        "tokenizer": _tokenizer_provenance(tokenizer),
        "native_block_executed": True,
        "inference_executor": False,
        "compression_claim": False,
        "parameter_mac_latency_claim": False,
    }
    payload = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "scientific_status": {
            "scope": (
                "target_informed_fisher_aware_lower_tail_mode_merge_"
                "oracle"
            ),
            "calibration_a_merge_fitted": True,
            "calibration_b_rank_sweep_evaluated": True,
            "calibration_b_controls_passed": controls_passed,
            "selection_failed": lock["selection_failed"],
            "validation_locked_before_evaluation": True,
            "validation_evaluated": validation_evaluated,
            "validation_passed": validation_passed,
            "test_evaluated": False,
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "prompt_text_in_artifact": False,
            "native_block_executed": True,
            "target_informed_oracle": True,
            "inference_executor": False,
            "merged_representation_viable": (
                merged_representation_viable
            ),
            "merged_representation_rank_reduction_supported": (
                merged_representation_viable
                and locked.total_rank < width - 1
            ),
            "compression_claim": False,
            "parameter_reduction_claim": False,
            "analytic_mac_reduction_claim": False,
            "latency_or_kernel_speed_claim": False,
        },
        "model": model_metadata,
        "source_rotation": source_binding,
        "protocol": protocol,
        "merge": merge.state_dict(),
        "calibration_a": {
            "score_fisher": score_fisher,
            "ground_truth_nll_score_fisher": sensitivity[
                "ground_truth_nll_score_fisher"
            ],
            "delta_second_moment": delta_second_moment,
            "split_half_score_fisher": split_score,
            "split_half_delta_second_moment": split_delta,
            "split_half_observations": sensitivity[
                "split_half_observations"
            ],
            "examples": sensitivity["summary"]["examples"],  # type: ignore[index]
            "source_fit_summary": sensitivity["summary"],
            "supermode_spectrum": spectrum,
            "split_half_subspace_stability": stability_rows,
            "tokenized_stream": calibration_a_stream,
        },
        "selection": {
            "candidate_behavior": selection_behavior,
            "candidate_direct_diagnostics": selection_direct,
            "ledger": ledger,
            "controls": {
                "native_identity_candidate": identity.metadata(),
                "native_identity_behavior_gates": (
                    identity_behavior_gates
                ),
                "native_identity_passed": identity_passed,
                "locked_span_candidate": span.metadata(),
                "locked_span_behavior_passed": (
                    span_row["behavior_fidelity_passed"]
                ),
                "locked_span_equivalence_passed": (
                    span_equivalence_passed
                ),
                "passed": controls_passed,
            },
            "execution_audit": execution_audit,
            "lock": lock,
            "tokenized_stream": selection_stream,
        },
        "validation": {
            "evaluated": validation_evaluated,
            "reason": (
                "one_calibration_b_locked_merged_candidate_evaluated"
                if validation_evaluated
                else "no_merged_candidate_passed_validation_not_tokenized"
            ),
            "locked_candidate": locked.metadata(),
            "behavior": validation_behavior,
            "direct_diagnostic": validation_direct,
            "behavior_gates": validation_gates,
            "behavior_fidelity_passed": validation_passed,
            "execution_audit": validation_execution,
            "tokenized_stream": validation_stream,
        },
    }
    digest = _scientific_payload_sha256(payload)
    report = _build_report(
        payload,
        output=resolved_output,
        scientific_digest=digest,
    )
    report_digest = _report_sha256(report)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **payload,
            "scientific_payload_sha256": digest,
            "report_sha256": report_digest,
        },
        resolved_output,
    )
    resolved_output.with_suffix(".json").write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _validated_prompt_and_stream_provenance(
    protocol: Mapping[str, object],
    *,
    validation_evaluated: bool,
    calibration_a: Mapping[str, object],
    selection: Mapping[str, object],
    validation: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    prompt_metadata = protocol.get("prompt_splits")
    stream_values = protocol.get("tokenized_splits")
    split_names = (
        "calibration_a",
        "calibration_b",
        "validation",
        "test",
    )
    if (
        not isinstance(prompt_metadata, Mapping)
        or set(prompt_metadata)
        != {
            "scientific_status",
            "counts",
            "normalized_sha256",
            "per_prompt_sha256",
        }
        or prompt_metadata.get("scientific_status") != _PROMPT_STATUS
        or not isinstance(stream_values, Mapping)
    ):
        raise ValueError("merged-supermode prompt provenance is invalid")
    counts = prompt_metadata["counts"]
    normalized = prompt_metadata["normalized_sha256"]
    per_prompt = prompt_metadata["per_prompt_sha256"]
    if (
        not isinstance(counts, Mapping)
        or tuple(counts) != split_names
        or dict(counts)
        != {
            "calibration_a": 64,
            "calibration_b": 16,
            "validation": 16,
            "test": 16,
        }
        or not isinstance(normalized, Mapping)
        or tuple(normalized) != split_names
        or not isinstance(per_prompt, Mapping)
        or tuple(per_prompt) != split_names
    ):
        raise ValueError("merged-supermode prompt mappings are invalid")
    all_hashes = []
    for split_name in split_names:
        hashes = per_prompt[split_name]
        if (
            not isinstance(hashes, list)
            or len(hashes) != counts[split_name]
            or any(not _is_sha256(value) for value in hashes)
            or normalized[split_name]
            != _ordered_prompt_hash_digest(hashes)
        ):
            raise ValueError("merged-supermode prompt hashes are invalid")
        all_hashes.extend(hashes)
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError(
            "merged-supermode prompt hashes must be globally disjoint"
        )
    expected_names = (
        ("calibration_a", "calibration_b", "validation")
        if validation_evaluated
        else ("calibration_a", "calibration_b")
    )
    if tuple(stream_values) != expected_names:
        raise ValueError(
            "merged-supermode tokenized split set is invalid"
        )
    streams: dict[str, Mapping[str, object]] = {}
    for split_name in expected_names:
        stream, _ = _validated_tokenized_stream(
            stream_values[split_name],
            split_name=split_name,
        )
        if (
            stream["sequences"] != counts[split_name]
            or stream["source_prompt_sha256"]
            != per_prompt[split_name]
        ):
            raise ValueError(
                "merged-supermode stream does not bind prompt hashes"
            )
        streams[split_name] = stream
    if (
        calibration_a.get("tokenized_stream")
        != streams["calibration_a"]
        or selection.get("tokenized_stream")
        != streams["calibration_b"]
        or (
            validation_evaluated
            and validation.get("tokenized_stream")
            != streams["validation"]
        )
        or (
            not validation_evaluated
            and validation.get("tokenized_stream") is not None
        )
    ):
        raise ValueError(
            "merged-supermode duplicated stream provenance differs"
        )
    streamed = {
        digest
        for stream in streams.values()
        for digest in stream["source_prompt_sha256"]  # type: ignore[union-attr]
    }
    hash_only = set(per_prompt["test"])
    if not validation_evaluated:
        hash_only.update(per_prompt["validation"])
    if streamed & hash_only:
        raise ValueError(
            "merged-supermode hash-only prompt entered model stream"
        )

    manifest = protocol.get("family_manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema")
        != "fisher_graph.gemma3_prompt_family_manifest"
        or manifest.get("format_version") != 1
        or manifest.get("prompt_fixture_sha256")
        != protocol.get("prompt_fixture_file_sha256")
        or manifest.get("scientific_policy")
        != "one_template_family_one_role_no_cross_split_reuse"
        or not isinstance(manifest.get("roles"), Mapping)
    ):
        raise ValueError(
            "merged-supermode embedded family manifest is invalid"
        )
    roles = manifest["roles"]
    assert isinstance(roles, Mapping)
    if tuple(roles) != split_names:
        raise ValueError("family manifest role order is invalid")
    suffixes = set()
    identifiers = set()
    for role in split_names:
        entry = roles[role]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"normalized_sha256", "families"}
            or entry["normalized_sha256"] != normalized[role]
            or not isinstance(entry["families"], list)
        ):
            raise ValueError("family manifest role is invalid")
        cursor = 0
        for family in entry["families"]:
            if (
                not isinstance(family, Mapping)
                or set(family)
                != {"id", "description", "start", "count"}
                or not isinstance(family["id"], str)
                or not family["id"].startswith(f"{role}.")
                or family["id"] in identifiers
                or not isinstance(family["description"], str)
                or not family["description"]
                or family["start"] != cursor
                or type(family["count"]) is not int
                or family["count"] <= 0
            ):
                raise ValueError("family manifest range is invalid")
            suffix = family["id"].split(".", 1)[1]
            if suffix in suffixes:
                raise ValueError("family suffix crosses prompt roles")
            suffixes.add(suffix)
            identifiers.add(family["id"])
            cursor += family["count"]
        if cursor != counts[role]:
            raise ValueError("family manifest role coverage is invalid")
    return streams


def load_gemma3_merged_supermode_oracle_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strictly authenticate a merged-supermode oracle artifact."""

    artifact_path = Path(path)
    raw = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    required = {
        "schema",
        "format_version",
        "contains_model_weights",
        "contains_prompt_text",
        "contains_tokenizer_state",
        "scientific_status",
        "model",
        "source_rotation",
        "protocol",
        "merge",
        "calibration_a",
        "selection",
        "validation",
        "scientific_payload_sha256",
        "report_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("merged-supermode artifact fields are invalid")
    if (
        raw["schema"] != _ARTIFACT_SCHEMA
        or raw["format_version"] != _ARTIFACT_FORMAT_VERSION
        or raw["contains_model_weights"] is not False
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
        or not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError("unsupported or unsafe merged-supermode artifact")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    digest = _scientific_payload_sha256(payload)
    if digest != raw["scientific_payload_sha256"]:
        raise ValueError("merged-supermode scientific digest mismatch")
    model = _validate_model_metadata(raw["model"])
    source = raw["source_rotation"]
    protocol = raw["protocol"]
    calibration_a = raw["calibration_a"]
    selection = raw["selection"]
    validation = raw["validation"]
    status = raw["scientific_status"]
    merge_state = raw["merge"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            source,
            protocol,
            calibration_a,
            selection,
            validation,
            status,
            merge_state,
        )
    ):
        raise ValueError("merged-supermode payload mappings are invalid")
    for field in (
        "scientific_payload_sha256",
        "report_sha256",
        "tensor_file_sha256",
        "tail_basis_sha256",
        "locked_normal_sha256",
    ):
        if not _is_sha256(source.get(field)):
            raise ValueError("merged-supermode source digest is invalid")
    source_fields = {
        "schema",
        "format_version",
        "scientific_payload_sha256",
        "report_sha256",
        "tensor_file",
        "tensor_file_sha256",
        "tail_basis_sha256",
        "locked_normal_sha256",
        "model_binding",
        "block_geometry",
        "locked_candidate",
        "rank_639_fidelity_viable",
        "basis_ordering_supported",
        "prompt_disjointness",
    }
    protocol_fields = {
        "start_layer",
        "end_layer_inclusive",
        "layer_ids",
        "canonical_boundaries",
        "residual_width",
        "source_tail_width",
        "preserved_prefix_rank",
        "surviving_tail_rank",
        "supermode_rank_schedule",
        "candidate_schedule",
        "fit_split",
        "fit_objective",
        "projection",
        "selection_policy",
        "validation_policy",
        "test_policy",
        "maximum_tokenized_length",
        "tokenization_batch_size",
        "thresholds",
        "identity_nll_atol",
        "minimum_subspace_stability",
        "prompt_fixture_file_sha256",
        "family_manifest_file_sha256",
        "family_manifest",
        "prompt_splits",
        "tokenized_splits",
        "model_state_guard",
        "library_versions",
        "tokenizer",
        "native_block_executed",
        "inference_executor",
        "compression_claim",
        "parameter_mac_latency_claim",
    }
    calibration_a_fields = {
        "score_fisher",
        "ground_truth_nll_score_fisher",
        "delta_second_moment",
        "split_half_score_fisher",
        "split_half_delta_second_moment",
        "split_half_observations",
        "examples",
        "source_fit_summary",
        "supermode_spectrum",
        "split_half_subspace_stability",
        "tokenized_stream",
    }
    selection_fields = {
        "candidate_behavior",
        "candidate_direct_diagnostics",
        "ledger",
        "controls",
        "execution_audit",
        "lock",
        "tokenized_stream",
    }
    validation_fields = {
        "evaluated",
        "reason",
        "locked_candidate",
        "behavior",
        "direct_diagnostic",
        "behavior_gates",
        "behavior_fidelity_passed",
        "execution_audit",
        "tokenized_stream",
    }
    if (
        set(source) != source_fields
        or set(protocol) != protocol_fields
        or set(calibration_a) != calibration_a_fields
        or set(selection) != selection_fields
        or set(validation) != validation_fields
        or source.get("schema") != _SOURCE_ROTATION_SCHEMA
        or source.get("format_version") != 1
    ):
        raise ValueError(
            "merged-supermode source or protocol field schema is invalid"
        )
    width = protocol.get("residual_width")
    start = protocol.get("start_layer")
    end = protocol.get("end_layer_inclusive")
    layer_ids = protocol.get("layer_ids")
    boundaries = protocol.get("canonical_boundaries")
    ranks = protocol.get("supermode_rank_schedule")
    thresholds = protocol.get("thresholds")
    maximum_tokenized_length = protocol.get(
        "maximum_tokenized_length"
    )
    tokenization_batch_size = protocol.get(
        "tokenization_batch_size"
    )
    if (
        type(width) is not int
        or width < 3
        or type(start) is not int
        or type(end) is not int
        or end < start
        or not isinstance(layer_ids, tuple)
        or len(layer_ids) != end - start + 1
        or layer_ids
        != tuple(f"layer.{index}" for index in range(start, end + 1))
        or not isinstance(boundaries, tuple)
        or len(boundaries) != len(layer_ids) + 1
        or any(
            not isinstance(boundary, str) or not boundary
            for boundary in boundaries
        )
        or len(set(boundaries)) != len(boundaries)
        or not isinstance(ranks, tuple)
        or not isinstance(thresholds, Mapping)
        or set(thresholds)
        != {
            "nll_atol",
            "top1_min",
            "teacher_kl_max",
            "p90_abs_nll_max",
            "p10_top1_min",
        }
        or protocol.get("fit_split") != "calibration_a_only"
        or protocol.get("fit_objective") != _FIT_OBJECTIVE
        or protocol.get("projection") != _PROJECTION_POLICY
        or protocol.get("selection_policy") != _SELECTION_POLICY
        or protocol.get("validation_policy") != _VALIDATION_POLICY
        or protocol.get("test_policy") != _TEST_POLICY
        or type(maximum_tokenized_length) is not int
        or maximum_tokenized_length < 2
        or type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
        or not _is_sha256(
            protocol.get("prompt_fixture_file_sha256")
        )
        or not _is_sha256(
            protocol.get("family_manifest_file_sha256")
        )
        or protocol.get("native_block_executed") is not True
        or protocol.get("inference_executor") is not False
        or protocol.get("compression_claim") is not False
        or protocol.get("parameter_mac_latency_claim") is not False
    ):
        raise ValueError("merged-supermode protocol is invalid")
    threshold_values = {
        "nll_atol": _finite(
            thresholds["nll_atol"],
            label="nll_atol",
            minimum=0.0,
        ),
        "top1_min": _finite(
            thresholds["top1_min"],
            label="top1_min",
            minimum=0.0,
            maximum=1.0,
        ),
        "teacher_kl_max": _finite(
            thresholds["teacher_kl_max"],
            label="teacher_kl_max",
            minimum=0.0,
        ),
        "p90_abs_nll_max": _finite(
            thresholds["p90_abs_nll_max"],
            label="p90_abs_nll_max",
            minimum=0.0,
        ),
        "p10_top1_min": _finite(
            thresholds["p10_top1_min"],
            label="p10_top1_min",
            minimum=0.0,
            maximum=1.0,
        ),
    }
    identity_tolerance = _finite(
        protocol.get("identity_nll_atol"),
        label="identity_nll_atol",
        minimum=0.0,
    )
    stability_minimum = _finite(
        protocol.get("minimum_subspace_stability"),
        label="minimum_subspace_stability",
        minimum=0.0,
        maximum=1.0,
    )
    if width == 640 and (
        ranks != DEFAULT_SUPERMODE_RANKS
        or threshold_values
        != {
            "nll_atol": DEFAULT_NLL_ATOL,
            "top1_min": DEFAULT_TOP1_MIN,
            "teacher_kl_max": DEFAULT_TEACHER_KL_MAX,
            "p90_abs_nll_max": (
                DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX
            ),
            "p10_top1_min": DEFAULT_PER_PROMPT_P10_TOP1_MIN,
        }
        or identity_tolerance != DEFAULT_IDENTITY_NLL_ATOL
        or stability_minimum != DEFAULT_MINIMUM_SUBSPACE_STABILITY
        or maximum_tokenized_length != 128
        or tokenization_batch_size != 2
    ):
        raise ValueError(
            "width-640 merged-supermode protocol is not preregistered"
        )
    model_binding = source.get("model_binding")
    block_geometry = source.get("block_geometry")
    source_locked = source.get("locked_candidate")
    expected_model_binding_fields = {
        "model_id",
        "config_sha256",
        "resolved_commit",
        "hidden_size",
        "num_hidden_layers",
    }
    expected_source_locked = {
        "candidate_id": (
            f"rank_{width - 1}."
            "calibration_a_balanced_tail_rotation"
        ),
        "normal_source": "calibration_a_balanced_tail_rotation",
        "retained_rank": width - 1,
        "residual_width": width,
        "retained_fraction": (width - 1) / width,
        "removed_dimensions": 1,
        "projection": (
            "target_informed_shared_euclidean_codimension_one_"
            "block_delta_projection"
        ),
    }
    if (
        not isinstance(model_binding, Mapping)
        or set(model_binding) != expected_model_binding_fields
        or any(
            model_binding.get(field) != model.get(field)
            for field in expected_model_binding_fields
        )
        or not isinstance(block_geometry, Mapping)
        or dict(block_geometry)
        != {
            "start_layer": start,
            "end_layer_inclusive": end,
            "layer_ids": layer_ids,
            "canonical_boundaries": boundaries,
            "residual_width": width,
        }
        or (
            model.get("hidden_size") is not None
            and model["hidden_size"] != width
        )
        or source.get("rank_639_fidelity_viable") is not True
        or source.get("basis_ordering_supported") is not True
        or source_locked != expected_source_locked
    ):
        raise ValueError(
            "merged-supermode model/source binding is invalid"
        )
    source_tensor_file = source.get("tensor_file")
    if (
        not isinstance(source_tensor_file, str)
        or not source_tensor_file
        or Path(source_tensor_file).name != source_tensor_file
        or Path(source_tensor_file).suffix != ".pt"
    ):
        raise ValueError(
            "merged-supermode source tensor filename is invalid"
        )
    source_artifact_path = artifact_path.parent / source_tensor_file
    if (
        not source_artifact_path.is_file()
        or source_artifact_path.resolve() == artifact_path.resolve()
        or _file_sha256(source_artifact_path)
        != source["tensor_file_sha256"]
    ):
        raise ValueError(
            "merged-supermode predecessor artifact is unavailable or "
            "has changed"
        )
    authenticated_source = load_gemma3_codimension_rotation_artifact(
        source_artifact_path
    )
    authenticated_metadata = authenticated_source.get("metadata")
    authenticated_report = authenticated_source.get("report")
    authenticated_model = authenticated_source.get("model")
    authenticated_locked = authenticated_source.get("locked_candidate")
    authenticated_sensitivity = authenticated_source.get(
        "calibration_a_sensitivity"
    )
    authenticated_projector = authenticated_source.get(
        "rotated_projector"
    )
    if not all(
        isinstance(value, Mapping)
        for value in (
            authenticated_metadata,
            authenticated_report,
            authenticated_model,
            authenticated_locked,
            authenticated_sensitivity,
        )
    ):
        raise ValueError(
            "merged-supermode authenticated predecessor is invalid"
        )
    authenticated_protocol = authenticated_metadata.get("protocol")
    authenticated_status = authenticated_report.get(
        "scientific_status"
    )
    authenticated_tail_basis = authenticated_sensitivity.get(
        "tail_basis"
    )
    authenticated_locked_normal = getattr(
        authenticated_projector,
        "normal",
        None,
    )
    if (
        authenticated_metadata.get("scientific_payload_sha256")
        != source["scientific_payload_sha256"]
        or authenticated_metadata.get("report_sha256")
        != source["report_sha256"]
        or authenticated_report.get("schema") != source["schema"]
        or authenticated_report.get("format_version")
        != source["format_version"]
        or not isinstance(authenticated_protocol, Mapping)
        or not isinstance(authenticated_status, Mapping)
        or authenticated_locked != source_locked
        or any(
            authenticated_model.get(field)
            != model_binding.get(field)
            for field in expected_model_binding_fields
        )
        or {
            "start_layer": authenticated_protocol.get("start_layer"),
            "end_layer_inclusive": authenticated_protocol.get(
                "end_layer_inclusive"
            ),
            "layer_ids": authenticated_protocol.get("layer_ids"),
            "canonical_boundaries": authenticated_protocol.get(
                "canonical_boundaries"
            ),
            "residual_width": authenticated_protocol.get(
                "residual_width"
            ),
        }
        != dict(block_geometry)
        or authenticated_status.get("rank_639_fidelity_viable")
        is not True
        or authenticated_status.get("basis_ordering_supported")
        is not True
        or authenticated_status.get("selection_failed") is not False
        or authenticated_status.get("test_evaluated") is not False
        or not isinstance(authenticated_tail_basis, Tensor)
        or not isinstance(authenticated_locked_normal, Tensor)
        or source.get("tail_basis_sha256")
        != _matrix_sha256(
            authenticated_tail_basis,
            domain=(
                b"fisher_graph.merged_supermode.source_tail_basis.v1\0"
            ),
        )
        or source.get("locked_normal_sha256")
        != _matrix_sha256(
            authenticated_locked_normal,
            domain=(
                b"fisher_graph.merged_supermode."
                b"source_locked_normal.v1\0"
            ),
        )
    ):
        raise ValueError(
            "merged-supermode predecessor trust anchor does not match"
        )
    guard = protocol.get("model_state_guard")
    libraries = protocol.get("library_versions")
    tokenizer = protocol.get("tokenizer")
    if (
        not isinstance(guard, Mapping)
        or set(guard)
        != {
            "verified",
            "training",
            "parameters_frozen",
            "parameter_tensors",
            "buffer_tensors",
            "checks",
        }
        or guard.get("verified") is not True
        or guard.get("training") is not False
        or guard.get("parameters_frozen") is not True
        or type(guard.get("parameter_tensors")) is not int
        or guard["parameter_tensors"] < 0
        or type(guard.get("buffer_tensors")) is not int
        or guard["buffer_tensors"] < 0
        or guard.get("checks")
        != (
            "tensor_object_identity",
            "tensor_version_counter",
            "tensor_storage_identity",
        )
        or not isinstance(libraries, Mapping)
        or set(libraries)
        != {
            "python",
            "torch",
            "transformers",
            "tokenizers",
            "sentencepiece",
        }
        or any(
            value is not None
            and (not isinstance(value, str) or not value)
            for value in libraries.values()
        )
        or not isinstance(libraries.get("python"), str)
        or not isinstance(libraries.get("torch"), str)
        or not isinstance(tokenizer, Mapping)
        or set(tokenizer)
        != {
            "tokenizer_class",
            "name_or_path",
            "configuration_sha256",
        }
        or not isinstance(tokenizer.get("tokenizer_class"), str)
        or not tokenizer["tokenizer_class"]
        or (
            tokenizer.get("name_or_path") is not None
            and not isinstance(tokenizer["name_or_path"], str)
        )
        or not _is_sha256(tokenizer.get("configuration_sha256"))
    ):
        raise ValueError(
            "merged-supermode runtime provenance is invalid"
        )

    validation_evaluated = validation.get("evaluated")
    if type(validation_evaluated) is not bool:
        raise ValueError("merged-supermode validation status is invalid")
    streams = _validated_prompt_and_stream_provenance(
        protocol,
        validation_evaluated=validation_evaluated,
        calibration_a=calibration_a,
        selection=selection,
        validation=validation,
    )
    prompt_metadata = protocol["prompt_splits"]
    assert isinstance(prompt_metadata, Mapping)
    per_prompt = prompt_metadata["per_prompt_sha256"]
    assert isinstance(per_prompt, Mapping)
    fresh_prompt_hashes = tuple(
        sorted(
            digest
            for split_hashes in per_prompt.values()
            for digest in split_hashes
        )
    )
    prompt_disjointness = source.get("prompt_disjointness")
    if (
        not isinstance(prompt_disjointness, Mapping)
        or set(prompt_disjointness)
        != {
            "fresh_prompt_sha256",
            "fresh_count",
            "predecessor_prompt_sha256",
            "predecessor_counts",
            "overlap_counts",
            "verified_before_model_load_or_tokenization",
        }
        or prompt_disjointness.get("fresh_prompt_sha256")
        != fresh_prompt_hashes
        or prompt_disjointness.get("fresh_count")
        != len(fresh_prompt_hashes)
        or prompt_disjointness.get(
            "verified_before_model_load_or_tokenization"
        )
        is not True
    ):
        raise ValueError(
            "merged-supermode repository prompt audit is invalid"
        )
    predecessors = prompt_disjointness["predecessor_prompt_sha256"]
    predecessor_counts = prompt_disjointness["predecessor_counts"]
    overlap_counts = prompt_disjointness["overlap_counts"]
    if (
        not isinstance(predecessors, Mapping)
        or not predecessors
        or not isinstance(predecessor_counts, Mapping)
        or set(predecessor_counts) != set(predecessors)
        or not isinstance(overlap_counts, Mapping)
        or set(overlap_counts) != set(predecessors)
    ):
        raise ValueError(
            "merged-supermode predecessor prompt audit is invalid"
        )
    fresh_prompt_set = set(fresh_prompt_hashes)
    for name, hashes in predecessors.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(hashes, tuple)
            or not hashes
            or tuple(sorted(set(hashes))) != hashes
            or any(not _is_sha256(value) for value in hashes)
            or predecessor_counts[name] != len(hashes)
            or overlap_counts[name] != 0
            or fresh_prompt_set.intersection(hashes)
        ):
            raise ValueError(
                "merged-supermode predecessor prompt hashes are invalid"
            )

    merge = AnchoredTailSupermodeMerge.from_state_dict(merge_state)
    if (
        merge.width != width
        or protocol.get("source_tail_width") != merge.tail_width
        or protocol.get("preserved_prefix_rank")
        != merge.preserved_prefix_rank
        or protocol.get("surviving_tail_rank")
        != merge.maximum_supermodes
        or not torch.equal(
            merge.tail_basis,
            authenticated_tail_basis.detach().to(
                device="cpu",
                dtype=torch.float64,
            ),
        )
        or not torch.equal(
            merge.locked_normal,
            authenticated_locked_normal.detach().to(
                device="cpu",
                dtype=torch.float64,
            ),
        )
    ):
        raise ValueError(
            "merged-supermode merge geometry or predecessor endpoint is "
            "invalid"
        )
    score_fisher = calibration_a.get("score_fisher")
    nll_score_fisher = calibration_a.get(
        "ground_truth_nll_score_fisher"
    )
    delta_second_moment = calibration_a.get("delta_second_moment")
    split_score = calibration_a.get("split_half_score_fisher")
    split_delta = calibration_a.get("split_half_delta_second_moment")
    if not all(
        isinstance(value, Tensor)
        for value in (
            score_fisher,
            nll_score_fisher,
            delta_second_moment,
            split_score,
            split_delta,
        )
    ):
        raise ValueError("merged-supermode A moments are invalid")
    tail_width = merge.tail_width
    if (
        split_score.shape != (2, tail_width, tail_width)
        or split_delta.shape != (2, tail_width, tail_width)
    ):
        raise ValueError("merged-supermode split moments are invalid")

    def validate_psd_matrix(
        value: Tensor,
        *,
        label: str,
    ) -> Tensor:
        if (
            value.dtype is not torch.float64
            or value.device.type != "cpu"
            or value.shape != (tail_width, tail_width)
            or not torch.isfinite(value).all()
            or not torch.allclose(
                value,
                value.T,
                rtol=1e-10,
                atol=1e-12,
            )
            or float(
                torch.linalg.eigvalsh(
                    (value + value.T) * 0.5
                ).min().item()
            )
            < -1e-12
        ):
            raise ValueError(f"{label} is not a valid PSD matrix")
        return value

    score_fisher = validate_psd_matrix(
        score_fisher,
        label="calibration-A score Fisher",
    )
    nll_score_fisher = validate_psd_matrix(
        nll_score_fisher,
        label="calibration-A NLL score Fisher",
    )
    delta_second_moment = validate_psd_matrix(
        delta_second_moment,
        label="calibration-A delta second moment",
    )
    for index in range(2):
        validate_psd_matrix(
            split_score[index],
            label=f"calibration-A split {index} score Fisher",
        )
        validate_psd_matrix(
            split_delta[index],
            label=f"calibration-A split {index} delta moment",
        )
    examples = calibration_a.get("examples")
    split_observations = calibration_a.get(
        "split_half_observations"
    )
    calibration_stream = calibration_a.get("tokenized_stream")
    if (
        not isinstance(examples, list)
        or not examples
        or not isinstance(split_observations, tuple)
        or len(split_observations) != 2
        or not isinstance(calibration_stream, Mapping)
        or not isinstance(calibration_stream.get("examples"), list)
        or len(examples) != len(calibration_stream["examples"])
    ):
        raise ValueError(
            "merged-supermode calibration-A ledger is invalid"
        )
    for row, stream_row in zip(
        examples,
        calibration_stream["examples"],
        strict=True,
    ):
        if (
            not isinstance(row, Mapping)
            or not isinstance(stream_row, Mapping)
            or row.get("example_id") != stream_row.get("example_id")
            or row.get("valid_tokens") != stream_row.get("valid_tokens")
            or row.get("supervised_tokens")
            != stream_row.get("supervised_positions")
        ):
            raise ValueError(
                "merged-supermode A ledger does not bind its stream"
            )
    observations = sum(int(row["valid_tokens"]) for row in examples)
    expected_split_observations = tuple(
        sum(
            int(row["valid_tokens"])
            for row in examples[index::2]
        )
        for index in range(2)
    )
    if (
        observations <= 0
        or any(
            type(value) is not int or value <= 0
            for value in split_observations
        )
        or split_observations != expected_split_observations
    ):
        raise ValueError(
            "merged-supermode A observations do not recompute"
        )
    weighted_score_fisher = sum(
        split_score[index] * split_observations[index]
        for index in range(2)
    ) / observations
    weighted_delta_second_moment = sum(
        split_delta[index] * split_observations[index]
        for index in range(2)
    ) / observations
    if (
        not torch.allclose(
            weighted_score_fisher,
            score_fisher,
            rtol=1e-12,
            atol=1e-13,
        )
        or not torch.allclose(
            weighted_delta_second_moment,
            delta_second_moment,
            rtol=1e-12,
            atol=1e-13,
        )
    ):
        raise ValueError(
            "merged-supermode A split halves do not reconstruct moments"
        )
    for index, (matrix, field) in enumerate(
        (
            (split_score, "tail_score_gradient_squared_norm"),
            (split_delta, "tail_block_delta_squared_norm"),
        )
    ):
        for split_index in range(2):
            expected_trace = (
                sum(
                    float(row[field])
                    for row in examples[split_index::2]
                )
                / split_observations[split_index]
            )
            if not math.isclose(
                float(torch.trace(matrix[split_index]).item()),
                expected_trace,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "merged-supermode A split-half trace does not bind "
                    f"the example ledger for moment {index}"
                )
    trace_controls = (
        (
            score_fisher,
            "tail_score_gradient_squared_norm",
        ),
        (
            nll_score_fisher,
            "tail_ground_truth_nll_score_gradient_squared_norm",
        ),
        (
            delta_second_moment,
            "tail_block_delta_squared_norm",
        ),
    )
    for matrix, field in trace_controls:
        expected_trace = (
            sum(float(row[field]) for row in examples)
            / observations
        )
        if not math.isclose(
            float(torch.trace(matrix).item()),
            expected_trace,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "merged-supermode A moment trace does not bind ledger"
            )
    source_fit_summary = calibration_a.get("source_fit_summary")
    summary_fields = {
        "fit_split",
        "score_objective",
        "score_fisher",
        "ground_truth_nll_score_fisher_control",
        "delta_moment",
        "tail_constraint",
        "tail_width",
        "preserved_codec_prefix_rank",
        "observations",
        "supervised_tokens",
        "sequences",
        "native_top1_teacher_stream_sha256",
        "native_top1_teacher_provenance",
        "native_top1_teacher_tokens",
        "native_top1_exact_ties",
        "native_top1_margin_min",
        "native_top1_margin_mean",
        "score_fisher_trace",
        "ground_truth_nll_score_fisher_trace",
        "delta_second_moment_trace",
        "combined_operator",
        "combined_minimum_eigenvalue",
        "combined_minimum_eigengap",
        "combined_minimum_relative_eigengap",
        "split_half_policy",
        "stability_policy",
        "split_half_absolute_normal_alignment",
        "minimum_split_half_alignment",
        "minimum_relative_eigengap",
        "minimum_split_half_operator_frobenius_cosine",
        "maximum_split_half_relative_regret",
        "split_half_objective_diagnostics",
        "sensitivity_fit_stable",
        "balanced_candidate_pareto_dominates_codec",
        "rotated",
        "codec_prefix",
        "absolute_normal_alignment",
        "tail_basis_sha256",
        "score_fisher_sha256",
        "ground_truth_nll_score_fisher_sha256",
        "delta_second_moment_sha256",
        "examples",
    }
    supervised_tokens = sum(
        int(row["supervised_tokens"]) for row in examples
    )
    native_top1_exact_ties = sum(
        int(row["native_top1_exact_ties"]) for row in examples
    )
    native_top1_margin_min = min(
        float(row["native_top1_margin_min"]) for row in examples
    )
    native_top1_margin_mean = (
        sum(float(row["native_top1_margin_sum"]) for row in examples)
        / supervised_tokens
    )
    if (
        not isinstance(source_fit_summary, Mapping)
        or set(source_fit_summary) != summary_fields
        or source_fit_summary.get("fit_split") != "calibration_a_only"
        or source_fit_summary.get("score_objective")
        != "summed_cross_entropy_to_native_detached_top1_tokens"
        or source_fit_summary.get("score_fisher")
        != (
            "width_pooled_uncentered_pseudo_top1_score_gradient_"
            "second_moment"
        )
        or source_fit_summary.get(
            "ground_truth_nll_score_fisher_control"
        )
        != (
            "width_pooled_uncentered_ground_truth_nll_score_gradient_"
            "second_moment_does_not_influence_fit"
        )
        or source_fit_summary.get("delta_moment")
        != (
            "width_pooled_uncentered_native_block_delta_second_moment"
        )
        or source_fit_summary.get("tail_constraint")
        != "euclidean_complement_of_source_codec_prefix"
        or source_fit_summary.get("tail_width") != tail_width
        or source_fit_summary.get("preserved_codec_prefix_rank")
        != width - tail_width
        or source_fit_summary.get("observations") != observations
        or source_fit_summary.get("supervised_tokens")
        != supervised_tokens
        or source_fit_summary.get("sequences") != len(examples)
        or source_fit_summary.get("native_top1_teacher_tokens")
        != supervised_tokens
        or source_fit_summary.get("native_top1_exact_ties")
        != native_top1_exact_ties
        or source_fit_summary.get("split_half_policy")
        != "alternating_sequence_index_parity"
        or source_fit_summary.get("stability_policy")
        != "split_half_direction_alignment"
        or source_fit_summary.get("minimum_split_half_alignment")
        != 0.8
        or source_fit_summary.get("minimum_relative_eigengap")
        != 1e-3
        or source_fit_summary.get(
            "minimum_split_half_operator_frobenius_cosine"
        )
        != 0.99
        or source_fit_summary.get(
            "maximum_split_half_relative_regret"
        )
        != 0.10
        or source_fit_summary.get("native_top1_teacher_provenance")
        != (
            "hash_bound_not_offline_derivation_replay_without_"
            "model_logits"
        )
        or source_fit_summary.get("combined_operator")
        != (
            "0.5_score_fisher_over_trace_plus_"
            "0.5_delta_second_moment_over_trace"
        )
        or not _semantic_numeric_equal(
            source_fit_summary.get("examples"),
            examples,
        )
        or source_fit_summary.get("native_top1_teacher_stream_sha256")
        != _native_top1_stream_sha256(examples)
        or source_fit_summary.get("tail_basis_sha256")
        != _matrix_sha256(
            merge.tail_basis,
            domain=(
                b"fisher_graph.codimension_rotation.tail_basis.v1\0"
            ),
        )
        or source_fit_summary.get("score_fisher_sha256")
        != _matrix_sha256(
            score_fisher,
            domain=(
                b"fisher_graph.codimension_rotation.score_fisher.v1\0"
            ),
        )
        or source_fit_summary.get(
            "ground_truth_nll_score_fisher_sha256"
        )
        != _matrix_sha256(
            nll_score_fisher,
            domain=(
                b"fisher_graph.codimension_rotation."
                b"ground_truth_nll_score_fisher.v1\0"
            ),
        )
        or source_fit_summary.get("delta_second_moment_sha256")
        != _matrix_sha256(
            delta_second_moment,
            domain=(
                b"fisher_graph.codimension_rotation.delta_moment.v1\0"
            ),
        )
        or not math.isclose(
            float(source_fit_summary.get("native_top1_margin_min")),
            native_top1_margin_min,
            rel_tol=1e-12,
            abs_tol=1e-13,
        )
        or not math.isclose(
            float(source_fit_summary.get("native_top1_margin_mean")),
            native_top1_margin_mean,
            rel_tol=1e-12,
            abs_tol=1e-13,
        )
        or not math.isclose(
            float(source_fit_summary.get("score_fisher_trace")),
            float(torch.trace(score_fisher).item()),
            rel_tol=1e-12,
            abs_tol=1e-13,
        )
        or not math.isclose(
            float(
                source_fit_summary.get(
                    "ground_truth_nll_score_fisher_trace"
                )
            ),
            float(torch.trace(nll_score_fisher).item()),
            rel_tol=1e-12,
            abs_tol=1e-13,
        )
        or not math.isclose(
            float(source_fit_summary.get("delta_second_moment_trace")),
            float(torch.trace(delta_second_moment).item()),
            rel_tol=1e-12,
            abs_tol=1e-13,
        )
    ):
        raise ValueError(
            "merged-supermode calibration-A source summary is invalid"
        )
    recomputed_merge = build_anchored_tail_supermode_merge(
        tail_basis=merge.tail_basis,
        locked_normal=merge.locked_normal,
        score_fisher=score_fisher,
        delta_second_moment=delta_second_moment,
        maximum_rank_projection=merge.maximum_rank_projection,
    )
    if not _semantic_tensor_equal(
        merge.state_dict(),
        recomputed_merge.state_dict(),
    ):
        raise ValueError("merged-supermode fit does not recompute")
    if (
        split_score.shape
        != (2, merge.tail_width, merge.tail_width)
        or split_delta.shape
        != (2, merge.tail_width, merge.tail_width)
    ):
        raise ValueError("merged-supermode split moments are invalid")
    split_merges = tuple(
        build_anchored_tail_supermode_merge(
            tail_basis=merge.tail_basis,
            locked_normal=merge.locked_normal,
            score_fisher=split_score[index],
            delta_second_moment=split_delta[index],
            maximum_rank_projection=(
                merge.maximum_rank_projection
            ),
        )
        for index in range(2)
    )
    expected_stability = _subspace_stability(
        merge,
        split_merges[0],
        split_merges[1],
        supermode_ranks=ranks,
        minimum_alignment=stability_minimum,
    )
    if not _semantic_numeric_equal(
        calibration_a.get("split_half_subspace_stability"),
        expected_stability,
    ):
        raise ValueError(
            "merged-supermode split stability does not recompute"
        )
    expected_spectrum = [
        {
            "supermode_rank": rank,
            "total_rank": merge.total_rank(rank),
            "retained_factorized_weighted_fraction": (
                merge.retained_weighted_fraction(rank)
            ),
            "discarded_factorized_weighted_fraction": (
                1.0 - merge.retained_weighted_fraction(rank)
            ),
        }
        for rank in ranks
    ]
    if not _semantic_numeric_equal(
        calibration_a.get("supermode_spectrum"),
        expected_spectrum,
    ):
        raise ValueError("merged-supermode spectrum does not recompute")
    candidates = _candidate_schedule(merge, ranks)
    expected_schedule = tuple(
        candidate.metadata() for candidate in candidates
    )
    if protocol.get("candidate_schedule") != expected_schedule:
        raise ValueError(
            "merged-supermode candidate schedule is invalid"
        )
    stability_by_rank = {
        int(row["supermode_rank"]): row
        for row in expected_stability
    }

    def validate_behavior(
        value: object,
        *,
        label: str,
        stream: Mapping[str, object],
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} behavior is invalid")
        examples = value.get("examples")
        stream_examples = stream.get("examples")
        fields = {
            "example_id",
            "supervised_tokens",
            "baseline_summed_nll",
            "predicted_summed_nll",
            "delta_summed_nll",
            "delta_nll_per_token",
            "top1_matches",
            "top1_agreement",
            "teacher_kl_summed",
            "teacher_kl_per_token",
        }
        if (
            not isinstance(examples, list)
            or not examples
            or not isinstance(stream_examples, list)
            or len(examples) != len(stream_examples)
        ):
            raise ValueError(f"{label} behavior examples are invalid")
        for row, stream_row in zip(
            examples,
            stream_examples,
            strict=True,
        ):
            if (
                not isinstance(row, Mapping)
                or set(row) != fields
                or not isinstance(stream_row, Mapping)
                or row.get("example_id")
                != stream_row.get("example_id")
                or row.get("supervised_tokens")
                != stream_row.get("supervised_positions")
                or type(row.get("supervised_tokens")) is not int
                or row["supervised_tokens"] <= 0
                or type(row.get("top1_matches")) is not int
                or not 0
                <= row["top1_matches"]
                <= row["supervised_tokens"]
                or any(
                    not isinstance(row[field], (int, float))
                    or isinstance(row[field], bool)
                    or not math.isfinite(float(row[field]))
                    for field in fields
                    - {
                        "example_id",
                        "supervised_tokens",
                        "top1_matches",
                    }
                )
                or float(row["baseline_summed_nll"]) < 0.0
                or float(row["predicted_summed_nll"]) < 0.0
                or not math.isclose(
                    float(row["delta_summed_nll"]),
                    float(row["predicted_summed_nll"])
                    - float(row["baseline_summed_nll"]),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["delta_nll_per_token"]),
                    float(row["delta_summed_nll"])
                    / row["supervised_tokens"],
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["top1_agreement"]),
                    row["top1_matches"] / row["supervised_tokens"],
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["teacher_kl_per_token"]),
                    float(row["teacher_kl_summed"])
                    / row["supervised_tokens"],
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"{label} behavior row provenance is invalid"
                )
        expected = _aggregate_behavior_with_kl(examples)
        if not _semantic_numeric_equal(value, expected):
            raise ValueError(f"{label} behavior does not recompute")
        return value

    def validate_direct(
        value: object,
        *,
        label: str,
        stream: Mapping[str, object],
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} direct diagnostic is invalid")
        examples = value.get("examples")
        stream_examples = stream.get("examples")
        fields = {
            "example_id",
            "valid_tokens",
            "squared_error",
            "block_delta_energy",
            "full_output_energy",
            "prediction_energy",
            "predicted_block_delta_energy",
            "block_delta_dot",
            "full_output_dot",
            "block_delta_nrmse",
            "full_output_nrmse",
            "block_delta_cosine",
            "full_output_cosine",
        }
        if (
            not isinstance(examples, list)
            or not examples
            or not isinstance(stream_examples, list)
            or len(examples) != len(stream_examples)
        ):
            raise ValueError(f"{label} direct examples are invalid")
        for row, stream_row in zip(
            examples,
            stream_examples,
            strict=True,
        ):
            if (
                not isinstance(row, Mapping)
                or set(row) != fields
                or not isinstance(stream_row, Mapping)
                or row.get("example_id")
                != stream_row.get("example_id")
                or row.get("valid_tokens")
                != stream_row.get("valid_tokens")
                or type(row.get("valid_tokens")) is not int
                or row["valid_tokens"] <= 0
                or any(
                    not isinstance(row[field], (int, float))
                    or isinstance(row[field], bool)
                    or not math.isfinite(float(row[field]))
                    for field in fields
                    - {"example_id", "valid_tokens"}
                )
                or any(
                    float(row[field]) < 0.0
                    for field in (
                        "squared_error",
                        "block_delta_energy",
                        "full_output_energy",
                        "prediction_energy",
                        "predicted_block_delta_energy",
                        "block_delta_nrmse",
                        "full_output_nrmse",
                    )
                )
                or not math.isclose(
                    float(row["block_delta_nrmse"]),
                    math.sqrt(
                        float(row["squared_error"])
                        / max(
                            float(row["block_delta_energy"]),
                            torch.finfo(torch.float64).tiny,
                        )
                    ),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["full_output_nrmse"]),
                    math.sqrt(
                        float(row["squared_error"])
                        / max(
                            float(row["full_output_energy"]),
                            torch.finfo(torch.float64).tiny,
                        )
                    ),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["block_delta_cosine"]),
                    _safe_cosine(
                        float(row["block_delta_dot"]),
                        float(row["block_delta_energy"]),
                        float(row["predicted_block_delta_energy"]),
                    ),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["full_output_cosine"]),
                    _safe_cosine(
                        float(row["full_output_dot"]),
                        float(row["full_output_energy"]),
                        float(row["prediction_energy"]),
                    ),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"{label} direct row provenance is invalid"
                )
        expected = _aggregate_direct_examples(examples, width=width)
        if not _semantic_numeric_equal(value, expected):
            raise ValueError(f"{label} direct aggregate does not recompute")
        return value

    behavior = selection.get("candidate_behavior")
    direct = selection.get("candidate_direct_diagnostics")
    if (
        not isinstance(behavior, Mapping)
        or tuple(behavior)
        != tuple(candidate.candidate_id for candidate in candidates)
        or not isinstance(direct, Mapping)
        or tuple(direct) != tuple(behavior)
    ):
        raise ValueError(
            "merged-supermode selection candidates are invalid"
        )
    for candidate in candidates:
        validate_behavior(
            behavior[candidate.candidate_id],
            label=f"selection {candidate.candidate_id}",
            stream=streams["calibration_b"],
        )
        validate_direct(
            direct[candidate.candidate_id],
            label=f"selection {candidate.candidate_id}",
            stream=streams["calibration_b"],
        )
    expected_ledger = _build_ledger(
        candidates=candidates,
        behavior=behavior,  # type: ignore[arg-type]
        direct=direct,  # type: ignore[arg-type]
        stability=stability_by_rank,
        thresholds=threshold_values,
    )
    if not _semantic_numeric_equal(
        selection.get("ledger"),
        expected_ledger,
    ):
        raise ValueError("merged-supermode ledger does not recompute")

    def validate_execution_audit(
        value: object,
        *,
        label: str,
        expected_batches: int,
        expected_suffix_replays: int,
        controls_evaluated: bool,
    ) -> Mapping[str, object]:
        fields = {
            "native_batches",
            "native_block_executed_once_per_batch",
            "candidate_suffix_replays_per_batch",
            "native_identity_maximum_logit_error",
            "rank_639_vs_source_rotated_maximum_boundary_error",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or type(value.get("native_batches")) is not int
            or value["native_batches"] != expected_batches
            or value.get(
                "native_block_executed_once_per_batch"
            )
            is not True
            or type(
                value.get("candidate_suffix_replays_per_batch")
            )
            is not int
            or value["candidate_suffix_replays_per_batch"]
            != expected_suffix_replays
        ):
            raise ValueError(
                f"merged-supermode {label} execution audit is invalid"
            )
        identity_error = _finite(
            value["native_identity_maximum_logit_error"],
            label=f"{label} native identity logit error",
            minimum=0.0,
        )
        span_error = _finite(
            value[
                "rank_639_vs_source_rotated_maximum_boundary_error"
            ],
            label=f"{label} rotated-span boundary error",
            minimum=0.0,
        )
        if not controls_evaluated and (
            identity_error != 0.0 or span_error != 0.0
        ):
            raise ValueError(
                f"merged-supermode {label} records unevaluated controls"
            )
        return value

    execution = selection.get("execution_audit")
    controls = selection.get("controls")
    selection_batches = streams["calibration_b"].get("batches")
    assert type(selection_batches) is int
    execution = validate_execution_audit(
        execution,
        label="selection",
        expected_batches=selection_batches,
        expected_suffix_replays=len(candidates),
        controls_evaluated=True,
    )
    if not isinstance(controls, Mapping):
        raise ValueError("merged-supermode controls are invalid")
    identity_row = expected_ledger[-1]
    span_row = expected_ledger[-2]
    identity_direct = identity_row["direct_diagnostic"]
    assert isinstance(identity_direct, Mapping)
    identity_gates = _behavior_gates(
        identity_row["behavior"],  # type: ignore[arg-type]
        nll_atol=identity_tolerance,
        top1_min=1.0,
        teacher_kl_max=identity_tolerance,
        p90_abs_nll_max=identity_tolerance,
        p10_top1_min=1.0,
    )
    identity_passed = (
        all(identity_gates.values())
        and float(identity_direct["block_delta_nrmse"]) <= 1e-12
        and float(identity_direct["block_delta_cosine"])
        >= 1.0 - 1e-12
        and float(
            execution.get(
                "native_identity_maximum_logit_error",
                math.inf,
            )
        )
        <= identity_tolerance
    )
    span_equivalence = (
        float(
            execution.get(
                "rank_639_vs_source_rotated_maximum_boundary_error",
                math.inf,
            )
        )
        <= 1e-5
    )
    span_passed = (
        span_row["behavior_fidelity_passed"] is True
        and span_equivalence
    )
    controls_passed = identity_passed and span_passed
    expected_controls = {
        "native_identity_candidate": candidates[-1].metadata(),
        "native_identity_behavior_gates": identity_gates,
        "native_identity_passed": identity_passed,
        "locked_span_candidate": candidates[-2].metadata(),
        "locked_span_behavior_passed": (
            span_row["behavior_fidelity_passed"]
        ),
        "locked_span_equivalence_passed": span_equivalence,
        "passed": controls_passed,
    }
    if controls != expected_controls:
        raise ValueError(
            "merged-supermode controls do not recompute"
        )
    expected_locked, expected_lock = _lock_candidate(
        candidates=candidates,
        ledger=expected_ledger,
        controls_passed=controls_passed,
    )
    if selection.get("lock") != expected_lock:
        raise ValueError("merged-supermode lock does not recompute")

    if validation_evaluated:
        validation_behavior = validate_behavior(
            validation.get("behavior"),
            label="validation",
            stream=streams["validation"],
        )
        validate_direct(
            validation.get("direct_diagnostic"),
            label="validation",
            stream=streams["validation"],
        )
        expected_validation_gates = _candidate_gates(
            validation_behavior,
            thresholds=threshold_values,
        )
        expected_validation_passed = all(
            expected_validation_gates.values()
        )
        validation_batches = streams["validation"].get("batches")
        assert type(validation_batches) is int
        validate_execution_audit(
            validation.get("execution_audit"),
            label="validation",
            expected_batches=validation_batches,
            expected_suffix_replays=1,
            controls_evaluated=False,
        )
        if (
            validation.get("reason")
            != "one_calibration_b_locked_merged_candidate_evaluated"
            or
            validation.get("locked_candidate")
            != expected_locked.metadata()
            or validation.get("behavior_gates")
            != expected_validation_gates
            or validation.get("behavior_fidelity_passed")
            is not expected_validation_passed
            or expected_lock["selection_failed"] is not False
            or expected_locked.kind != "merged_tail_supermodes"
        ):
            raise ValueError(
                "merged-supermode validation does not recompute"
            )
    else:
        expected_validation_passed = False
        if (
            expected_lock["selection_failed"] is not True
            or validation.get("reason")
            != "no_merged_candidate_passed_validation_not_tokenized"
            or any(
                validation.get(field) is not None
                for field in (
                    "behavior",
                    "direct_diagnostic",
                    "behavior_gates",
                    "execution_audit",
                    "tokenized_stream",
                )
            )
            or validation.get("behavior_fidelity_passed") is not False
            or validation.get("locked_candidate")
            != expected_locked.metadata()
        ):
            raise ValueError(
                "unevaluated merged-supermode validation is invalid"
            )
    viable = validation_evaluated and expected_validation_passed
    expected_status = {
        "scope": (
            "target_informed_fisher_aware_lower_tail_mode_merge_oracle"
        ),
        "calibration_a_merge_fitted": True,
        "calibration_b_rank_sweep_evaluated": True,
        "calibration_b_controls_passed": controls_passed,
        "selection_failed": expected_lock["selection_failed"],
        "validation_locked_before_evaluation": True,
        "validation_evaluated": validation_evaluated,
        "validation_passed": expected_validation_passed,
        "test_evaluated": False,
        "model_weights_changed": False,
        "model_weights_in_artifact": False,
        "prompt_text_in_artifact": False,
        "native_block_executed": True,
        "target_informed_oracle": True,
        "inference_executor": False,
        "merged_representation_viable": viable,
        "merged_representation_rank_reduction_supported": (
            viable and expected_locked.total_rank < width - 1
        ),
        "compression_claim": False,
        "parameter_reduction_claim": False,
        "analytic_mac_reduction_claim": False,
        "latency_or_kernel_speed_claim": False,
    }
    if status != expected_status:
        raise ValueError(
            "merged-supermode scientific status is invalid"
        )

    expected_report = _build_report(
        payload,
        output=artifact_path,
        scientific_digest=digest,
    )
    report = json.loads(
        artifact_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(report, Mapping)
        or _report_sha256(report) != raw["report_sha256"]
        or report
        != json.loads(
            json.dumps(
                expected_report,
                sort_keys=True,
                allow_nan=False,
            )
        )
    ):
        raise ValueError(
            "merged-supermode JSON report does not match payload"
        )
    return {
        "model": model,
        "merge": merge,
        "calibration_a": copy.deepcopy(dict(calibration_a)),
        "selection": copy.deepcopy(dict(selection)),
        "validation": copy.deepcopy(dict(validation)),
        "scientific_status": copy.deepcopy(dict(status)),
        "metadata": {
            "scientific_payload_sha256": digest,
            "report_sha256": raw["report_sha256"],
            "source_rotation": copy.deepcopy(dict(source)),
            "protocol": copy.deepcopy(dict(protocol)),
        },
        "report": copy.deepcopy(dict(report)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and test Fisher-aware merged tail supermodes in the "
            "validated Gemma rank-639 span."
        )
    )
    parser.add_argument(
        "--rotation-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        default=DEFAULT_PROMPT_SPLITS,
    )
    parser.add_argument(
        "--family-manifest",
        type=Path,
        default=DEFAULT_FAMILY_MANIFEST,
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--supermode-ranks",
        default=",".join(str(value) for value in DEFAULT_SUPERMODE_RANKS),
        help="comma-separated surviving-tail q ranks",
    )
    parser.add_argument(
        "--selection-nll-atol",
        type=float,
        default=DEFAULT_NLL_ATOL,
    )
    parser.add_argument(
        "--selection-top1-min",
        type=float,
        default=DEFAULT_TOP1_MIN,
    )
    parser.add_argument(
        "--selection-teacher-kl-max",
        type=float,
        default=DEFAULT_TEACHER_KL_MAX,
    )
    parser.add_argument(
        "--selection-p90-abs-nll-max",
        type=float,
        default=DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX,
    )
    parser.add_argument(
        "--selection-p10-top1-min",
        type=float,
        default=DEFAULT_PER_PROMPT_P10_TOP1_MIN,
    )
    parser.add_argument(
        "--identity-nll-atol",
        type=float,
        default=DEFAULT_IDENTITY_NLL_ATOL,
    )
    parser.add_argument(
        "--minimum-subspace-stability",
        type=float,
        default=DEFAULT_MINIMUM_SUBSPACE_STABILITY,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _parse_supermode_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError(
            "supermode ranks must be comma-separated integers"
        ) from error
    if not ranks:
        raise ValueError("supermode ranks cannot be empty")
    return ranks


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_merged_supermode_oracle(
        rotation_artifact_path=arguments.rotation_artifact,
        model_id=arguments.model_id,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        prompt_splits_path=arguments.prompt_splits,
        family_manifest_path=arguments.family_manifest,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        supermode_ranks=_parse_supermode_ranks(
            arguments.supermode_ranks
        ),
        selection_nll_atol=arguments.selection_nll_atol,
        selection_top1_min=arguments.selection_top1_min,
        selection_teacher_kl_max=(
            arguments.selection_teacher_kl_max
        ),
        selection_p90_abs_nll_max=(
            arguments.selection_p90_abs_nll_max
        ),
        selection_p10_top1_min=(
            arguments.selection_p10_top1_min
        ),
        identity_nll_atol=arguments.identity_nll_atol,
        minimum_subspace_stability=(
            arguments.minimum_subspace_stability
        ),
        device_name=arguments.device,
        dtype=arguments.dtype,
        local_files_only=arguments.local_files_only,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FAMILY_MANIFEST",
    "DEFAULT_PROMPT_SPLITS",
    "DEFAULT_SUPERMODE_RANKS",
    "default_gemma3_merged_supermode_output",
    "load_gemma3_merged_supermode_oracle_artifact",
    "run_gemma3_merged_supermode_oracle",
]
