"""Fresh codimension-one span diagnostic for a frozen Gemma block.

Calibration A fits one omitted direction inside the 32-dimensional Euclidean
complement of the existing rank-608 output-decoder prefix.  The fit minimizes
an equal-weight normalized sum of:

* a downstream score-gradient Fisher formed from self-distillation
  cross-entropy to the native model's top-1 tokens; and
* the native block-delta second moment.

Calibration B compares that rotated rank-639 hyperplane with the original
rank-639 codec-prefix hyperplane and mandatory rank-640 identity.  The broader
primary estimand is whether either preregistered rank-639 span is viable; the
narrow secondary discriminator asks whether rotation succeeds where the codec
prefix fails.  Only aggregate NLL and top-1 gates select a candidate.
Validation sees exactly one locked reduced intervention, or remains untouched
if neither reduced candidate passes.  Reserved test remains hash-only.

Every reduced intervention consumes the true native block delta.  This is a
representation diagnostic, not an inference executor or compression claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter, LayerBlockBoundaryPlan, ModelAdapter
from .codimension_projection import (
    CodimensionOneDeltaProjector,
    canonical_unit_direction,
)
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .gemma3_ablation_experiment import (
    _FrozenModelTensorGuard,
    _is_sha256,
    _update_payload_digest,
    _validate_model_metadata,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    make_causal_lm_calibration_batches,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_gated_executor_experiment import (
    _BoundaryBatch,
    _aggregate_direct_examples,
    _behavior_aggregate,
    _behavior_examples,
    _collect_boundaries,
    _direct_example,
    _finite,
    _identity_passed,
    _materialize_split,
    _safe_cosine,
)
from .gemma3_projection_ladder_experiment import (
    _behavior_gate,
    load_gemma3_projection_ladder_artifact,
)
from .gemma3_stability_experiment import (
    _CalibrationStreamProvenance,
    _library_versions,
    _ordered_prompt_hash_digest,
    _tokenizer_provenance,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .gemma3_weighted_jacobian_experiment import _codec_state_sha256
from .linear_codec import LinearActivationCodec
from .modal_ablation import _causal_lm_batch_scores, _example_ids


DEFAULT_PROMPT_SPLITS = Path(
    "examples/gemma3_codimension_rotation_prompts.json"
)
DEFAULT_TAIL_WIDTH = 32
DEFAULT_NLL_ATOL = 0.05
DEFAULT_TOP1_MIN = 0.95
DEFAULT_IDENTITY_NLL_ATOL = 1e-5
DEFAULT_MAX_MEANINGFUL_RETAINED_FRACTION = 0.75
DEFAULT_MIN_SPLIT_HALF_ALIGNMENT = 0.8
DEFAULT_MIN_RELATIVE_EIGENGAP = 1e-3
DEFAULT_MIN_SPLIT_HALF_OPERATOR_COSINE = 0.99
DEFAULT_MAX_SPLIT_HALF_RELATIVE_REGRET = 0.10
DEFAULT_STABILITY_POLICY = "split_half_direction_alignment"
EXPANDED_STABILITY_POLICY = "split_half_objective_regret"
_STABILITY_POLICIES = {
    DEFAULT_STABILITY_POLICY,
    EXPANDED_STABILITY_POLICY,
}

_ARTIFACT_SCHEMA = "fisher_graph.gemma3_codimension_rotation"
_ARTIFACT_FORMAT_VERSION = 1
_PAYLOAD_DOMAIN = b"fisher_graph.gemma3_codimension_rotation_payload.v1\0"
_REPORT_DOMAIN = b"fisher_graph.gemma3_codimension_rotation_report.v1\0"
_PROMPT_STATUS = (
    "fresh_codimension_rotation_diagnostic_calibration_a_sensitivity_"
    "b_selection_validation_locked_test_hash_only"
)
_EXPANDED_PROMPT_STATUS = (
    "expanded_codimension_rotation_calibration_a_after_"
    "identifiability_failure_b_validation_test_untouched"
)
_PROMPT_STATUSES = {_PROMPT_STATUS, _EXPANDED_PROMPT_STATUS}


def default_gemma3_codimension_rotation_output(
    model_id: str = DEFAULT_MODEL_ID,
    start_layer: int = 4,
    end_layer: int = 6,
) -> Path:
    """Return the ignored model/block-specific rotation artifact path."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if (
        type(start_layer) is not int
        or type(end_layer) is not int
        or start_layer < 0
        or end_layer < start_layer
    ):
        raise ValueError("layer range must be nonnegative and ascending")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip("._-")
    return (
        Path(".local-runs")
        / (slug or "gemma3-model")
        / f"layers-{start_layer}-{end_layer}-codimension-rotation.pt"
    )


def _scientific_payload_sha256(payload: Mapping[str, object]) -> str:
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


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _native_top1_ids_sha256(
    *,
    example_id: str,
    teacher_ids: Tensor,
) -> str:
    if (
        not isinstance(example_id, str)
        or not example_id
        or not isinstance(teacher_ids, Tensor)
        or teacher_ids.ndim != 1
        or teacher_ids.numel() <= 0
    ):
        raise ValueError("native top-1 teacher stream is invalid")
    canonical = teacher_ids.detach().to(
        device="cpu",
        dtype=torch.int64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(
        b"fisher_graph.codimension_rotation.native_top1_ids.v1\0"
    )
    digest.update(
        json.dumps(
            {
                "example_id": example_id,
                "teacher_tokens": canonical.numel(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(canonical.numpy().tobytes())
    return digest.hexdigest()


def _native_top1_stream_sha256(
    examples: Sequence[Mapping[str, object]],
) -> str:
    rows = []
    for row in examples:
        example_id = row.get("example_id")
        teacher_tokens = row.get("supervised_tokens")
        teacher_sha = row.get("native_top1_teacher_sha256")
        if (
            not isinstance(example_id, str)
            or not example_id
            or type(teacher_tokens) is not int
            or teacher_tokens <= 0
            or not _is_sha256(teacher_sha)
        ):
            raise ValueError(
                "native top-1 teacher provenance is invalid"
            )
        rows.append(
            {
                "example_id": example_id,
                "teacher_tokens": teacher_tokens,
                "native_top1_teacher_sha256": teacher_sha,
            }
        )
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(
        b"fisher_graph.codimension_rotation.native_top1_stream.v1\0"
    )
    digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RotationCandidate:
    """One preregistered codimension-one or identity candidate."""

    candidate_id: str
    normal_source: str
    retained_rank: int
    residual_width: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or self.normal_source
            not in {
                "calibration_a_balanced_tail_rotation",
                "source_codec_prefix",
                "identity",
            }
            or type(self.retained_rank) is not int
            or type(self.residual_width) is not int
            or self.residual_width < 2
            or self.retained_rank
            not in {self.residual_width - 1, self.residual_width}
        ):
            raise ValueError("rotation candidate is invalid")
        if (
            self.normal_source == "identity"
        ) != (self.retained_rank == self.residual_width):
            raise ValueError("identity candidate rank is invalid")

    @property
    def retained_fraction(self) -> float:
        return self.retained_rank / self.residual_width

    @property
    def removed_dimensions(self) -> int:
        return self.residual_width - self.retained_rank

    def metadata(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "normal_source": self.normal_source,
            "retained_rank": self.retained_rank,
            "residual_width": self.residual_width,
            "retained_fraction": self.retained_fraction,
            "removed_dimensions": self.removed_dimensions,
            "projection": (
                "target_informed_shared_euclidean_codimension_one_"
                "block_delta_projection"
                if self.removed_dimensions
                else "target_informed_full_width_identity"
            ),
        }


def _candidate_schedule(width: int) -> tuple[RotationCandidate, ...]:
    if type(width) is not int or width < 2:
        raise ValueError("residual width must be at least two")
    reduced = width - 1
    return (
        RotationCandidate(
            (
                f"rank_{reduced}."
                "calibration_a_balanced_tail_rotation"
            ),
            "calibration_a_balanced_tail_rotation",
            reduced,
            width,
        ),
        RotationCandidate(
            f"rank_{reduced}.source_codec_prefix",
            "source_codec_prefix",
            reduced,
            width,
        ),
        RotationCandidate(
            f"rank_{width}.identity",
            "identity",
            width,
            width,
        ),
    )


def _prompt_hash_set(metadata: Mapping[str, object]) -> set[str]:
    per_prompt = metadata.get("per_prompt_sha256")
    if not isinstance(per_prompt, Mapping):
        raise ValueError("prompt provenance lacks per-prompt hashes")
    result = {
        digest
        for values in per_prompt.values()
        if isinstance(values, (list, tuple))
        for digest in values
    }
    if not result or any(not _is_sha256(value) for value in result):
        raise ValueError("prompt provenance contains invalid hashes")
    return result


def _assert_prompt_disjointness(
    *,
    fresh: Mapping[str, object],
    source_protocol: Mapping[str, object],
    source_predecessors: Mapping[str, object],
) -> dict[str, object]:
    fresh_hashes = _prompt_hash_set(fresh)
    source_prompts = source_protocol.get("prompt_splits")
    source_disjointness = source_predecessors.get("prompt_disjointness")
    if not isinstance(source_prompts, Mapping) or not isinstance(
        source_disjointness,
        Mapping,
    ):
        raise ValueError("source projection prompt provenance is missing")
    projection_hashes = _prompt_hash_set(source_prompts)
    weighted_hashes = set(
        source_disjointness.get("weighted_prompt_sha256", ())
    )
    gated_hashes = set(source_disjointness.get("gated_prompt_sha256", ()))
    if any(
        not values or any(not _is_sha256(value) for value in values)
        for values in (
            projection_hashes,
            weighted_hashes,
            gated_hashes,
        )
    ):
        raise ValueError("source projection prompt hashes are invalid")
    for label, values in (
        ("projection-ladder", projection_hashes),
        ("weighted", weighted_hashes),
        ("gated", gated_hashes),
    ):
        if fresh_hashes & values:
            raise ValueError(
                f"rotation prompts overlap {label} source prompts"
            )
    return {
        "fresh_prompt_sha256": tuple(sorted(fresh_hashes)),
        "projection_prompt_sha256": tuple(sorted(projection_hashes)),
        "weighted_prompt_sha256": tuple(sorted(weighted_hashes)),
        "gated_prompt_sha256": tuple(sorted(gated_hashes)),
        "fresh_count": len(fresh_hashes),
        "projection_count": len(projection_hashes),
        "weighted_count": len(weighted_hashes),
        "gated_count": len(gated_hashes),
        "projection_overlap_count": 0,
        "weighted_overlap_count": 0,
        "gated_overlap_count": 0,
        "verified_before_model_load_or_tokenization": True,
    }


def _tail_complement_basis(
    decoder: Tensor,
    *,
    tail_width: int,
) -> Tensor:
    if (
        not isinstance(decoder, Tensor)
        or not decoder.is_floating_point()
        or decoder.ndim != 2
        or decoder.shape[0] != decoder.shape[1]
        or not torch.isfinite(decoder).all()
    ):
        raise ValueError("decoder must be a finite square floating Tensor")
    width = decoder.shape[0]
    if (
        type(tail_width) is not int
        or tail_width <= 0
        or tail_width >= width
    ):
        raise ValueError("tail_width must be between one and width minus one")
    prefix = decoder.detach().to(
        device="cpu",
        dtype=torch.float64,
    )[:, : width - tail_width]
    complete, _ = torch.linalg.qr(prefix, mode="complete")
    tail = complete[:, width - tail_width :].contiguous()
    for index in range(tail.shape[1]):
        column = tail[:, index]
        pivot = int(column.abs().argmax().item())
        if float(column[pivot].item()) < 0.0:
            tail[:, index].neg_()
    identity = torch.eye(tail_width, dtype=torch.float64)
    if (
        not torch.allclose(
            tail.T @ tail,
            identity,
            rtol=1e-9,
            atol=1e-10,
        )
        or float((prefix.T @ tail).abs().max().item()) > 1e-8
    ):
        raise RuntimeError("failed to construct codec-tail complement")
    return tail


def _codec_prefix_normal(decoder: Tensor) -> Tensor:
    if (
        not isinstance(decoder, Tensor)
        or decoder.ndim != 2
        or decoder.shape[0] != decoder.shape[1]
    ):
        raise ValueError("decoder must be square")
    width = decoder.shape[0]
    matrix = decoder.detach().to(device="cpu", dtype=torch.float64)
    target = torch.zeros(width, dtype=torch.float64)
    target[-1] = 1.0
    normal = canonical_unit_direction(
        torch.linalg.solve(matrix.T, target),
        label="codec-prefix normal",
    )
    if float((matrix[:, :-1].T @ normal).abs().max().item()) > 1e-8:
        raise RuntimeError("codec-prefix normal is not orthogonal to prefix")
    return normal


def _codec_prefix_projector_equivalence(
    decoder: Tensor,
    normal: Tensor,
) -> dict[str, object]:
    matrix = decoder.detach().to(device="cpu", dtype=torch.float64)
    prefix = matrix[:, :-1]
    least_squares_projector = prefix @ torch.linalg.pinv(prefix)
    normal_projector = (
        torch.eye(matrix.shape[0], dtype=torch.float64)
        - torch.outer(normal, normal)
    )
    error = least_squares_projector - normal_projector
    max_absolute_error = float(error.abs().max().item())
    frobenius_error = float(torch.linalg.matrix_norm(error).item())
    passed = max_absolute_error <= 1e-9 and frobenius_error <= 1e-8
    return {
        "passed": passed,
        "comparison": (
            "decoder_prefix_times_pseudoinverse_vs_identity_minus_"
            "normal_outer_product"
        ),
        "retained_rank": matrix.shape[0] - 1,
        "max_absolute_error": max_absolute_error,
        "frobenius_error": frobenius_error,
        "max_absolute_error_atol": 1e-9,
        "frobenius_error_atol": 1e-8,
    }


def _matrix_sha256(value: Tensor, *, domain: bytes) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _rayleigh(matrix: Tensor, vector: Tensor) -> float:
    return float((vector @ matrix @ vector).item())


def _split_half_objective_diagnostics(
    *,
    tail_basis: Tensor,
    split_half_score_fisher: Tensor,
    split_half_delta_second_moment: Tensor,
    rotated_normal: Tensor,
) -> dict[str, object]:
    tail_width = tail_basis.shape[1]
    coefficients = tail_basis.T @ rotated_normal
    operators = []
    eigensystems = []
    halves = []
    for index in range(2):
        score = split_half_score_fisher[index]
        delta = split_half_delta_second_moment[index]
        operator = (
            0.5 * score / torch.trace(score)
            + 0.5 * delta / torch.trace(delta)
        )
        operator = ((operator + operator.T) * 0.5).contiguous()
        eigenvalues, eigenvectors = torch.linalg.eigh(operator)
        average = float(torch.trace(operator).item()) / tail_width
        candidate_rayleigh = _rayleigh(operator, coefficients)
        minimum = float(eigenvalues[0].item())
        relative_excess = max(
            0.0,
            (candidate_rayleigh - minimum)
            / max(average, torch.finfo(torch.float64).tiny),
        )
        operators.append(operator)
        eigensystems.append((eigenvalues, eigenvectors))
        halves.append(
            {
                "half": index,
                "minimum_eigenvalue": minimum,
                "minimum_relative_eigengap": (
                    float((eigenvalues[1] - eigenvalues[0]).item())
                    / average
                ),
                "pooled_candidate_rayleigh": candidate_rayleigh,
                "pooled_candidate_relative_excess_over_minimum": (
                    relative_excess
                ),
            }
        )
    operator_cosine = float(
        (operators[0] * operators[1]).sum().item()
        / (
            torch.linalg.matrix_norm(operators[0]).item()
            * torch.linalg.matrix_norm(operators[1]).item()
        )
    )
    subspaces = []
    widths = tuple(
        width
        for width in (1, 2, 4, 8, 16)
        if width <= tail_width
    )
    for width in widths:
        left = eigensystems[0][1][:, :width]
        right = eigensystems[1][1][:, :width]
        singular_values = torch.linalg.svdvals(left.T @ right)
        subspaces.append(
            {
                "bottom_width": width,
                "mean_squared_canonical_correlation": float(
                    singular_values.square().mean().item()
                ),
                "minimum_canonical_correlation": float(
                    singular_values.min().item()
                ),
                "pooled_candidate_projection_fraction_half_0": float(
                    (left.T @ coefficients).square().sum().item()
                ),
                "pooled_candidate_projection_fraction_half_1": float(
                    (right.T @ coefficients).square().sum().item()
                ),
            }
        )
    return {
        "normalized_operator_frobenius_cosine": operator_cosine,
        "maximum_pooled_candidate_relative_excess_over_minimum": max(
            float(
                row[
                    "pooled_candidate_relative_excess_over_minimum"
                ]
            )
            for row in halves
        ),
        "halves": halves,
        "bottom_subspaces": subspaces,
    }


def _sensitivity_summary(
    *,
    tail_basis: Tensor,
    score_fisher: Tensor,
    ground_truth_nll_score_fisher: Tensor,
    delta_second_moment: Tensor,
    combined_operator: Tensor,
    combined_eigenvalues: Tensor,
    rotated_normal: Tensor,
    codec_normal: Tensor,
    split_half_alignment: float,
    split_half_score_fisher: Tensor,
    split_half_delta_second_moment: Tensor,
    stability_policy: str,
    minimum_split_half_alignment: float,
    minimum_relative_eigengap: float,
    minimum_split_half_operator_cosine: float,
    maximum_split_half_relative_regret: float,
    examples: Sequence[Mapping[str, object]],
    observations: int,
    supervised_tokens: int,
    tail_width: int,
) -> dict[str, object]:
    split_half_objective = _split_half_objective_diagnostics(
        tail_basis=tail_basis,
        split_half_score_fisher=split_half_score_fisher,
        split_half_delta_second_moment=(
            split_half_delta_second_moment
        ),
        rotated_normal=rotated_normal,
    )
    teacher_margin_sum = sum(
        float(row["native_top1_margin_sum"]) for row in examples
    )
    teacher_exact_ties = sum(
        int(row["native_top1_exact_ties"]) for row in examples
    )
    teacher_margin_min = min(
        float(row["native_top1_margin_min"]) for row in examples
    )
    rotated_coefficients = tail_basis.T @ rotated_normal
    codec_coefficients = tail_basis.T @ codec_normal
    score_trace = float(torch.trace(score_fisher).item())
    delta_trace = float(torch.trace(delta_second_moment).item())
    rotated_score = _rayleigh(score_fisher, rotated_coefficients)
    codec_score = _rayleigh(score_fisher, codec_coefficients)
    nll_score_trace = float(
        torch.trace(ground_truth_nll_score_fisher).item()
    )
    rotated_nll_score = _rayleigh(
        ground_truth_nll_score_fisher,
        rotated_coefficients,
    )
    codec_nll_score = _rayleigh(
        ground_truth_nll_score_fisher,
        codec_coefficients,
    )
    rotated_delta = _rayleigh(
        delta_second_moment,
        rotated_coefficients,
    )
    codec_delta = _rayleigh(delta_second_moment, codec_coefficients)
    rotated_combined = _rayleigh(
        combined_operator,
        rotated_coefficients,
    )
    codec_combined = _rayleigh(
        combined_operator,
        codec_coefficients,
    )
    eigengap = (
        0.0
        if combined_eigenvalues.numel() < 2
        else float(
            (
                combined_eigenvalues[1]
                - combined_eigenvalues[0]
            ).item()
        )
    )
    average_eigenvalue = float(
        torch.trace(combined_operator).item()
    ) / tail_width
    relative_eigengap = eigengap / max(
        average_eigenvalue,
        torch.finfo(torch.float64).tiny,
    )
    if stability_policy == DEFAULT_STABILITY_POLICY:
        split_half_gate_passed = (
            split_half_alignment >= minimum_split_half_alignment
        )
    elif stability_policy == EXPANDED_STABILITY_POLICY:
        split_half_gate_passed = (
            split_half_objective[
                "normalized_operator_frobenius_cosine"
            ]
            >= minimum_split_half_operator_cosine
            and split_half_objective[
                "maximum_pooled_candidate_relative_excess_over_minimum"
            ]
            <= maximum_split_half_relative_regret
        )
    else:
        raise ValueError("unknown sensitivity stability policy")
    fit_stable = (
        split_half_gate_passed
        and relative_eigengap >= minimum_relative_eigengap
    )
    pareto_dominates_codec = (
        rotated_score
        <= codec_score + max(1e-12, abs(codec_score) * 1e-8)
        and rotated_delta
        <= codec_delta + max(1e-12, abs(codec_delta) * 1e-8)
    )
    return {
        "fit_split": "calibration_a_only",
        "score_objective": (
            "summed_cross_entropy_to_native_detached_top1_tokens"
        ),
        "score_fisher": (
            "width_pooled_uncentered_pseudo_top1_score_gradient_"
            "second_moment"
        ),
        "ground_truth_nll_score_fisher_control": (
            "width_pooled_uncentered_ground_truth_nll_score_gradient_"
            "second_moment_does_not_influence_fit"
        ),
        "delta_moment": (
            "width_pooled_uncentered_native_block_delta_second_moment"
        ),
        "tail_constraint": (
            "euclidean_complement_of_source_codec_prefix"
        ),
        "tail_width": tail_width,
        "preserved_codec_prefix_rank": tail_basis.shape[0] - tail_width,
        "observations": observations,
        "supervised_tokens": supervised_tokens,
        "sequences": len(examples),
        "native_top1_teacher_stream_sha256": (
            _native_top1_stream_sha256(examples)
        ),
        "native_top1_teacher_provenance": (
            "hash_bound_not_offline_derivation_replay_without_"
            "model_logits"
        ),
        "native_top1_teacher_tokens": supervised_tokens,
        "native_top1_exact_ties": teacher_exact_ties,
        "native_top1_margin_min": teacher_margin_min,
        "native_top1_margin_mean": (
            teacher_margin_sum / supervised_tokens
        ),
        "score_fisher_trace": score_trace,
        "ground_truth_nll_score_fisher_trace": nll_score_trace,
        "delta_second_moment_trace": delta_trace,
        "combined_operator": (
            "0.5_score_fisher_over_trace_plus_"
            "0.5_delta_second_moment_over_trace"
        ),
        "combined_minimum_eigenvalue": float(
            combined_eigenvalues[0].item()
        ),
        "combined_minimum_eigengap": eigengap,
        "combined_minimum_relative_eigengap": relative_eigengap,
        "split_half_policy": "alternating_sequence_index_parity",
        "stability_policy": stability_policy,
        "split_half_absolute_normal_alignment": split_half_alignment,
        "minimum_split_half_alignment": minimum_split_half_alignment,
        "minimum_relative_eigengap": minimum_relative_eigengap,
        "minimum_split_half_operator_frobenius_cosine": (
            minimum_split_half_operator_cosine
        ),
        "maximum_split_half_relative_regret": (
            maximum_split_half_relative_regret
        ),
        "split_half_objective_diagnostics": split_half_objective,
        "sensitivity_fit_stable": fit_stable,
        "balanced_candidate_pareto_dominates_codec": (
            pareto_dominates_codec
        ),
        "rotated": {
            "score_fisher_rayleigh": rotated_score,
            "score_fisher_trace_fraction": rotated_score / score_trace,
            "delta_moment_rayleigh": rotated_delta,
            "delta_moment_trace_fraction": rotated_delta / delta_trace,
            "combined_objective": rotated_combined,
            "ground_truth_nll_score_fisher_rayleigh": (
                rotated_nll_score
            ),
            "ground_truth_nll_score_fisher_trace_fraction": (
                rotated_nll_score / nll_score_trace
            ),
        },
        "codec_prefix": {
            "score_fisher_rayleigh": codec_score,
            "score_fisher_trace_fraction": codec_score / score_trace,
            "delta_moment_rayleigh": codec_delta,
            "delta_moment_trace_fraction": codec_delta / delta_trace,
            "combined_objective": codec_combined,
            "ground_truth_nll_score_fisher_rayleigh": codec_nll_score,
            "ground_truth_nll_score_fisher_trace_fraction": (
                codec_nll_score / nll_score_trace
            ),
        },
        "absolute_normal_alignment": float(
            torch.dot(rotated_normal, codec_normal).abs().item()
        ),
        "tail_basis_sha256": _matrix_sha256(
            tail_basis,
            domain=b"fisher_graph.codimension_rotation.tail_basis.v1\0",
        ),
        "score_fisher_sha256": _matrix_sha256(
            score_fisher,
            domain=b"fisher_graph.codimension_rotation.score_fisher.v1\0",
        ),
        "ground_truth_nll_score_fisher_sha256": _matrix_sha256(
            ground_truth_nll_score_fisher,
            domain=(
                b"fisher_graph.codimension_rotation."
                b"ground_truth_nll_score_fisher.v1\0"
            ),
        ),
        "delta_second_moment_sha256": _matrix_sha256(
            delta_second_moment,
            domain=b"fisher_graph.codimension_rotation.delta_moment.v1\0",
        ),
        "examples": [copy.deepcopy(dict(row)) for row in examples],
    }


def _fit_tail_sensitivity(
    adapter: ModelAdapter,
    batches: Iterable[CalibrationBatch],
    *,
    plan: LayerBlockBoundaryPlan,
    tail_basis: Tensor,
    codec_normal: Tensor,
    stability_policy: str = DEFAULT_STABILITY_POLICY,
    minimum_split_half_alignment: float = (
        DEFAULT_MIN_SPLIT_HALF_ALIGNMENT
    ),
    minimum_relative_eigengap: float = (
        DEFAULT_MIN_RELATIVE_EIGENGAP
    ),
    minimum_split_half_operator_cosine: float = (
        DEFAULT_MIN_SPLIT_HALF_OPERATOR_COSINE
    ),
    maximum_split_half_relative_regret: float = (
        DEFAULT_MAX_SPLIT_HALF_RELATIVE_REGRET
    ),
) -> dict[str, object]:
    """Fit the deterministic tail-constrained omission direction."""

    width, tail_width = tail_basis.shape
    if (
        plan.widths[0] != width
        or plan.widths[-1] != width
        or tail_width <= 0
    ):
        raise ValueError("tail sensitivity geometry does not match block")
    module = adapter.module
    if module.training or any(
        parameter.requires_grad for parameter in module.parameters()
    ):
        raise ValueError("tail sensitivity requires a frozen eval model")
    score_sum = torch.zeros(
        tail_width,
        tail_width,
        dtype=torch.float64,
    )
    nll_score_sum = torch.zeros_like(score_sum)
    delta_sum = torch.zeros_like(score_sum)
    split_score_sums = torch.zeros(
        2,
        tail_width,
        tail_width,
        dtype=torch.float64,
    )
    split_nll_score_sums = torch.zeros_like(split_score_sums)
    split_delta_sums = torch.zeros_like(split_score_sums)
    split_observations = [0, 0]
    observations = 0
    supervised_tokens = 0
    examples: list[dict[str, object]] = []
    input_site = plan.activation_sites[0]
    output_site = plan.activation_sites[-1]

    def detached_leaf(value: Tensor) -> Tensor:
        return value.detach().requires_grad_(True)

    for batch in batches:
        if not isinstance(batch, CalibrationBatch):
            raise TypeError("calibration A must contain CalibrationBatch")
        for index in range(batch.batch_size):
            sample = batch.sample(index)
            with torch.enable_grad():
                run = adapter.forward(
                    sample.model_inputs,
                    capture_sites=(input_site, output_site),
                    interventions={output_site: detached_leaf},
                )
                source = run.activations[input_site]
                output = run.activations[output_site]
                supervised = sample.targets != -100
                if not supervised.any():
                    raise ValueError(
                        "calibration A sample has no supervised tokens"
                    )
                pseudo_targets = run.logits.detach().argmax(dim=-1)
                supervised_logits = (
                    run.logits.detach()[supervised].to(torch.float32)
                )
                if supervised_logits.shape[-1] < 2:
                    raise ValueError(
                        "native top-1 teacher requires two logits"
                    )
                top_two = torch.topk(
                    supervised_logits,
                    k=2,
                    dim=-1,
                ).values
                top1_margins = (
                    top_two[:, 0] - top_two[:, 1]
                ).to(device="cpu", dtype=torch.float64)
                pseudo_loss = F.cross_entropy(
                    run.logits[supervised].to(torch.float32),
                    pseudo_targets[supervised],
                    reduction="sum",
                )
                true_nll = F.cross_entropy(
                    run.logits[supervised].to(torch.float32),
                    sample.targets[supervised],
                    reduction="sum",
                )
                gradient = torch.autograd.grad(
                    pseudo_loss,
                    output,
                    allow_unused=False,
                    retain_graph=True,
                )[0]
                nll_gradient = torch.autograd.grad(
                    true_nll,
                    output,
                    allow_unused=False,
                )[0]
            valid = sample.valid_positions[0].to(device=output.device)
            gradient_rows = (
                gradient.detach()[0, valid]
                .to(device="cpu", dtype=torch.float64)
            )
            delta_rows = (
                (output.detach() - source.detach())[0, valid]
                .to(device="cpu", dtype=torch.float64)
            )
            nll_gradient_rows = (
                nll_gradient.detach()[0, valid]
                .to(device="cpu", dtype=torch.float64)
            )
            if (
                gradient_rows.shape != delta_rows.shape
                or nll_gradient_rows.shape != delta_rows.shape
                or gradient_rows.shape[1] != width
                or not torch.isfinite(gradient_rows).all()
                or not torch.isfinite(nll_gradient_rows).all()
                or not torch.isfinite(delta_rows).all()
            ):
                raise ValueError(
                    "calibration A gradient/delta rows are invalid"
                )
            tail_gradients = gradient_rows @ tail_basis
            tail_nll_gradients = nll_gradient_rows @ tail_basis
            tail_deltas = delta_rows @ tail_basis
            score_sum.add_(tail_gradients.T @ tail_gradients)
            nll_score_sum.add_(
                tail_nll_gradients.T @ tail_nll_gradients
            )
            delta_sum.add_(tail_deltas.T @ tail_deltas)
            valid_tokens = int(valid.sum().item())
            supervised_count = int(supervised.sum().item())
            observations += valid_tokens
            supervised_tokens += supervised_count
            split_index = len(examples) % 2
            split_score_sums[split_index].add_(
                tail_gradients.T @ tail_gradients
            )
            split_nll_score_sums[split_index].add_(
                tail_nll_gradients.T @ tail_nll_gradients
            )
            split_delta_sums[split_index].add_(
                tail_deltas.T @ tail_deltas
            )
            split_observations[split_index] += valid_tokens
            example_id = (
                f"prompt.{len(examples):06d}"
                if sample.example_ids is None
                else sample.example_ids[0]
            )
            if not isinstance(example_id, str) or not example_id:
                raise ValueError(
                    "calibration A example IDs must be nonempty strings"
                )
            teacher_ids = (
                pseudo_targets[supervised]
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .contiguous()
            )
            examples.append(
                {
                    "example_id": example_id,
                    "valid_tokens": valid_tokens,
                    "supervised_tokens": supervised_count,
                    "native_top1_teacher_sha256": (
                        _native_top1_ids_sha256(
                            example_id=example_id,
                            teacher_ids=teacher_ids,
                        )
                    ),
                    "native_top1_margin_min": float(
                        top1_margins.min().item()
                    ),
                    "native_top1_margin_sum": float(
                        top1_margins.sum().item()
                    ),
                    "native_top1_exact_ties": int(
                        (top1_margins == 0.0).sum().item()
                    ),
                    "pseudo_label_summed_cross_entropy": float(
                        pseudo_loss.detach().item()
                    ),
                    "ground_truth_summed_nll": float(
                        true_nll.detach().item()
                    ),
                    "score_gradient_squared_norm": float(
                        gradient_rows.square().sum().item()
                    ),
                    "block_delta_squared_norm": float(
                        delta_rows.square().sum().item()
                    ),
                    "ground_truth_nll_score_gradient_squared_norm": (
                        float(nll_gradient_rows.square().sum().item())
                    ),
                    "tail_score_gradient_squared_norm": float(
                        tail_gradients.square().sum().item()
                    ),
                    "tail_block_delta_squared_norm": float(
                        tail_deltas.square().sum().item()
                    ),
                    "tail_ground_truth_nll_score_gradient_squared_norm": (
                        float(tail_nll_gradients.square().sum().item())
                    ),
                }
            )
    if observations <= 0 or not examples:
        raise ValueError("calibration A cannot be empty")
    if any(value <= 0 for value in split_observations):
        raise ValueError("both calibration-A split halves must be nonempty")
    score_fisher = (score_sum / observations).contiguous()
    ground_truth_nll_score_fisher = (
        nll_score_sum / observations
    ).contiguous()
    delta_second_moment = (delta_sum / observations).contiguous()
    score_trace = float(torch.trace(score_fisher).item())
    delta_trace = float(torch.trace(delta_second_moment).item())
    nll_score_trace = float(
        torch.trace(ground_truth_nll_score_fisher).item()
    )
    if (
        not math.isfinite(score_trace)
        or not math.isfinite(delta_trace)
        or not math.isfinite(nll_score_trace)
        or score_trace <= 0.0
        or delta_trace <= 0.0
        or nll_score_trace <= 0.0
    ):
        raise RuntimeError("tail sensitivity moments have zero trace")
    combined = (
        0.5 * score_fisher / score_trace
        + 0.5 * delta_second_moment / delta_trace
    )
    combined = ((combined + combined.T) * 0.5).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(combined)
    coefficients = eigenvectors[:, 0]
    rotated_normal = canonical_unit_direction(
        tail_basis @ coefficients,
        label="rotated omitted normal",
    )
    split_score_fisher = torch.stack(
        [
            split_score_sums[index] / split_observations[index]
            for index in range(2)
        ]
    ).contiguous()
    split_delta_second_moment = torch.stack(
        [
            split_delta_sums[index] / split_observations[index]
            for index in range(2)
        ]
    ).contiguous()
    split_ground_truth_nll_score_fisher = torch.stack(
        [
            split_nll_score_sums[index] / split_observations[index]
            for index in range(2)
        ]
    ).contiguous()
    split_normals = []
    for index in range(2):
        split_score_trace = torch.trace(split_score_fisher[index])
        split_delta_trace = torch.trace(
            split_delta_second_moment[index]
        )
        if (
            float(split_score_trace.item()) <= 0.0
            or float(split_delta_trace.item()) <= 0.0
        ):
            raise RuntimeError(
                "split-half sensitivity moments have zero trace"
            )
        split_operator = (
            0.5
            * split_score_fisher[index]
            / split_score_trace
            + 0.5
            * split_delta_second_moment[index]
            / split_delta_trace
        )
        _, split_vectors = torch.linalg.eigh(
            (split_operator + split_operator.T) * 0.5
        )
        split_normals.append(
            canonical_unit_direction(
                tail_basis @ split_vectors[:, 0],
                label=f"split-half-{index} omitted normal",
            )
        )
    split_half_alignment = float(
        torch.dot(split_normals[0], split_normals[1]).abs().item()
    )
    summary = _sensitivity_summary(
        tail_basis=tail_basis,
        score_fisher=score_fisher,
        ground_truth_nll_score_fisher=(
            ground_truth_nll_score_fisher
        ),
        delta_second_moment=delta_second_moment,
        combined_operator=combined,
        combined_eigenvalues=eigenvalues,
        rotated_normal=rotated_normal,
        codec_normal=codec_normal,
        split_half_alignment=split_half_alignment,
        split_half_score_fisher=split_score_fisher,
        split_half_delta_second_moment=(
            split_delta_second_moment
        ),
        stability_policy=stability_policy,
        minimum_split_half_alignment=minimum_split_half_alignment,
        minimum_relative_eigengap=minimum_relative_eigengap,
        minimum_split_half_operator_cosine=(
            minimum_split_half_operator_cosine
        ),
        maximum_split_half_relative_regret=(
            maximum_split_half_relative_regret
        ),
        examples=examples,
        observations=observations,
        supervised_tokens=supervised_tokens,
        tail_width=tail_width,
    )
    return {
        "tail_basis": tail_basis.clone(),
        "score_fisher": score_fisher,
        "ground_truth_nll_score_fisher": (
            ground_truth_nll_score_fisher
        ),
        "delta_second_moment": delta_second_moment,
        "split_half_score_fisher": split_score_fisher,
        "split_half_ground_truth_nll_score_fisher": (
            split_ground_truth_nll_score_fisher
        ),
        "split_half_delta_second_moment": (
            split_delta_second_moment
        ),
        "split_half_observations": tuple(split_observations),
        "split_half_normals": torch.stack(split_normals),
        "combined_operator": combined,
        "combined_eigenvalues": eigenvalues.contiguous(),
        "rotated_projector": CodimensionOneDeltaProjector(
            rotated_normal
        ).state_dict(),
        "codec_prefix_projector": CodimensionOneDeltaProjector(
            codec_normal
        ).state_dict(),
        "summary": summary,
    }


def _project_boundary(
    boundary: _BoundaryBatch,
    projector: CodimensionOneDeltaProjector | None,
) -> Tensor:
    if projector is None:
        return boundary.output_hidden
    return projector.project_output(
        boundary.input_hidden,
        boundary.output_hidden,
        valid_positions=boundary.valid_positions,
    )


def _evaluate_direct(
    boundaries: Sequence[_BoundaryBatch],
    *,
    candidates: Sequence[RotationCandidate],
    projectors: Mapping[
        str,
        CodimensionOneDeltaProjector | None,
    ],
) -> dict[str, dict[str, object]]:
    result = {}
    width = boundaries[0].input_hidden.shape[-1]
    for candidate in candidates:
        prediction_batches = [
            _project_boundary(
                boundary,
                projectors[candidate.candidate_id],
            )
            for boundary in boundaries
        ]
        examples = []
        for boundary, prediction in zip(
            boundaries,
            prediction_batches,
            strict=True,
        ):
            for index, example_id in enumerate(boundary.example_ids):
                valid = boundary.valid_positions[index]
                source = boundary.input_hidden[index, valid].to(
                    torch.float64
                )
                target = boundary.output_hidden[index, valid].to(
                    torch.float64
                )
                selected_prediction = prediction[index, valid].to(
                    torch.float64
                )
                examples.append(
                    _direct_example(
                        example_id=example_id,
                        valid_tokens=int(valid.sum().item()),
                        source=source,
                        target=target,
                        prediction=selected_prediction,
                    )
                )
        aggregate = _aggregate_direct_examples(examples, width=width)
        aggregate["projection"] = candidate.metadata()["projection"]
        aggregate["normal_source"] = candidate.normal_source
        result[candidate.candidate_id] = aggregate
    return result


def _evaluate_behavior(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    plan: LayerBlockBoundaryPlan,
    candidates: Sequence[RotationCandidate],
    projectors: Mapping[
        str,
        CodimensionOneDeltaProjector | None,
    ],
) -> dict[str, dict[str, object]]:
    objective = CausalLanguageModelNLL()
    examples: dict[str, list[dict[str, object]]] = {
        candidate.candidate_id: [] for candidate in candidates
    }
    sequence_offset = 0
    module = adapter.module
    if module.training or any(
        parameter.requires_grad for parameter in module.parameters()
    ):
        raise ValueError("rotation behavior requires a frozen eval model")
    with torch.inference_mode():
        for batch in batches:
            ids = _example_ids(batch, sequence_offset=sequence_offset)
            sequence_offset += batch.batch_size
            native = adapter.forward(batch.model_inputs)
            baseline = _causal_lm_batch_scores(
                native.logits,
                batch,
                objective=objective,
            )
            for candidate in candidates:
                projector = projectors[candidate.candidate_id]
                captured: dict[str, Tensor] = {}

                def capture_input(value: Tensor) -> Tensor:
                    captured["input"] = value
                    return value

                def replace_output(value: Tensor) -> Tensor:
                    source = captured.get("input")
                    if source is None:
                        raise RuntimeError(
                            "block output ran before input capture"
                        )
                    if projector is None:
                        return value
                    return projector.project_output(
                        source,
                        value,
                        valid_positions=batch.valid_positions,
                    )

                projected = adapter.forward(
                    batch.model_inputs,
                    interventions={
                        plan.activation_sites[0]: capture_input,
                        plan.activation_sites[-1]: replace_output,
                    },
                )
                scores = _causal_lm_batch_scores(
                    projected.logits,
                    batch,
                    objective=objective,
                )
                examples[candidate.candidate_id].extend(
                    _behavior_examples(
                        batch=batch,
                        example_ids=ids,
                        baseline=baseline,
                        predicted=scores,
                    )
                )
    return {
        candidate_id: _behavior_aggregate(rows)
        for candidate_id, rows in examples.items()
    }


def _candidate_ledger(
    candidates: Sequence[RotationCandidate],
    *,
    direct: Mapping[str, Mapping[str, object]],
    behavior: Mapping[str, Mapping[str, object]],
    nll_atol: float,
    top1_min: float,
) -> list[dict[str, object]]:
    rows = []
    for candidate in candidates:
        gates = _behavior_gate(
            behavior[candidate.candidate_id],
            nll_atol=nll_atol,
            top1_min=top1_min,
        )
        rows.append(
            {
                "candidate": candidate.metadata(),
                "direct_diagnostic": copy.deepcopy(
                    dict(direct[candidate.candidate_id])
                ),
                "behavior": copy.deepcopy(
                    dict(behavior[candidate.candidate_id])
                ),
                "behavior_gates": gates,
                "behavior_fidelity_passed": all(gates.values()),
                "direct_metrics_influence_lock": False,
            }
        )
    return rows


def _lock_candidate(
    candidates: Sequence[RotationCandidate],
    ledger: Sequence[Mapping[str, object]],
) -> tuple[RotationCandidate, dict[str, object]]:
    if len(candidates) != 3 or len(ledger) != 3:
        raise ValueError("rotation schedule must contain three candidates")
    for candidate, row in zip(candidates[:2], ledger[:2], strict=True):
        if row["behavior_fidelity_passed"] is True:
            return candidate, {
                "ordering": (
                    "rotated_then_codec_control_then_identity"
                ),
                "direct_metrics_influence_lock": False,
                "locked_candidate_id": candidate.candidate_id,
                "locked_normal_source": candidate.normal_source,
                "selection_failed": False,
                "reduced_candidate_found": True,
                "reason": (
                    "first_preferred_rank_639_candidate_passing_"
                    "behavioral_gates"
                ),
                "calibration_b_only": True,
                "ledger": [
                    copy.deepcopy(dict(value)) for value in ledger
                ],
            }
    identity = candidates[-1]
    return identity, {
        "ordering": "rotated_then_codec_control_then_identity",
        "direct_metrics_influence_lock": False,
        "locked_candidate_id": identity.candidate_id,
        "locked_normal_source": identity.normal_source,
        "selection_failed": True,
        "reduced_candidate_found": False,
        "reason": "no_rank_639_candidate_passed_identity_fallback",
        "calibration_b_only": True,
        "ledger": [copy.deepcopy(dict(value)) for value in ledger],
    }


def _full_width_identity_passed(
    *,
    direct: Mapping[str, object],
    behavior: Mapping[str, object],
    identity_nll_atol: float,
) -> bool:
    return (
        float(direct["block_delta_nrmse"]) <= 1e-12
        and float(direct["block_delta_cosine"]) >= 1.0 - 1e-12
        and _identity_passed(
            behavior,
            nll_atol=identity_nll_atol,
        )
    )


def _build_report(
    payload: Mapping[str, object],
    *,
    output: Path,
    scientific_digest: str,
) -> dict[str, object]:
    sensitivity = payload["calibration_a_sensitivity"]
    assert isinstance(sensitivity, Mapping)
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "scientific_status": copy.deepcopy(
            dict(payload["scientific_status"])  # type: ignore[arg-type]
        ),
        "model": copy.deepcopy(
            dict(payload["model"])  # type: ignore[arg-type]
        ),
        "source_projection": copy.deepcopy(
            dict(payload["source_projection"])  # type: ignore[arg-type]
        ),
        "protocol": copy.deepcopy(
            dict(payload["protocol"])  # type: ignore[arg-type]
        ),
        "analysis": {
            "calibration_a_sensitivity": copy.deepcopy(
                dict(sensitivity["summary"])  # type: ignore[arg-type]
            ),
            "selection": copy.deepcopy(
                dict(payload["selection"])  # type: ignore[arg-type]
            ),
            "validation": copy.deepcopy(
                dict(payload["validation"])  # type: ignore[arg-type]
            ),
        },
        "artifact": {
            "tensor_output": output.name,
            "contains_model_state_dict": False,
            "contains_codec_state": True,
            "contains_sensitivity_moments": True,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "scientific_payload_sha256": scientific_digest,
        },
    }


def run_gemma3_codimension_rotation(
    *,
    projection_artifact_path: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    prompt_splits_path: Path | str = DEFAULT_PROMPT_SPLITS,
    max_length: int = 128,
    tokenization_batch_size: int = 4,
    tail_width: int = DEFAULT_TAIL_WIDTH,
    selection_nll_atol: float = DEFAULT_NLL_ATOL,
    selection_top1_min: float = DEFAULT_TOP1_MIN,
    identity_nll_atol: float = DEFAULT_IDENTITY_NLL_ATOL,
    max_meaningful_retained_fraction: float = (
        DEFAULT_MAX_MEANINGFUL_RETAINED_FRACTION
    ),
    minimum_split_half_alignment: float = (
        DEFAULT_MIN_SPLIT_HALF_ALIGNMENT
    ),
    minimum_relative_eigengap: float = (
        DEFAULT_MIN_RELATIVE_EIGENGAP
    ),
    stability_policy: str = DEFAULT_STABILITY_POLICY,
    minimum_split_half_operator_cosine: float = (
        DEFAULT_MIN_SPLIT_HALF_OPERATOR_COSINE
    ),
    maximum_split_half_relative_regret: float = (
        DEFAULT_MAX_SPLIT_HALF_RELATIVE_REGRET
    ),
    device_name: str = "cpu",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Test whether either preregistered rank-639 span is viable."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    nll_atol = _finite(
        selection_nll_atol,
        label="selection_nll_atol",
        minimum=0.0,
    )
    top1_min = _finite(
        selection_top1_min,
        label="selection_top1_min",
        minimum=0.0,
        maximum=1.0,
    )
    identity_tolerance = _finite(
        identity_nll_atol,
        label="identity_nll_atol",
        minimum=0.0,
    )
    meaningful_fraction = _finite(
        max_meaningful_retained_fraction,
        label="max_meaningful_retained_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    minimum_alignment = _finite(
        minimum_split_half_alignment,
        label="minimum_split_half_alignment",
        minimum=0.0,
        maximum=1.0,
    )
    minimum_eigengap = _finite(
        minimum_relative_eigengap,
        label="minimum_relative_eigengap",
        minimum=0.0,
    )
    if stability_policy not in _STABILITY_POLICIES:
        raise ValueError("unknown sensitivity stability policy")
    minimum_operator_cosine = _finite(
        minimum_split_half_operator_cosine,
        label="minimum_split_half_operator_cosine",
        minimum=0.0,
        maximum=1.0,
    )
    maximum_relative_regret = _finite(
        maximum_split_half_relative_regret,
        label="maximum_split_half_relative_regret",
        minimum=0.0,
    )

    source_artifact_path = Path(projection_artifact_path)
    source = load_gemma3_projection_ladder_artifact(
        source_artifact_path
    )
    source_artifact_sha256 = _file_sha256(source_artifact_path)
    source_model = source["model"]
    source_metadata = source["metadata"]
    source_protocol = source_metadata["protocol"]  # type: ignore[index]
    source_predecessors = source_metadata["predecessors"]  # type: ignore[index]
    source_selection = source["selection"]
    source_validation = source["validation"]
    output_codec = source["output_codec"]
    assert isinstance(source_model, Mapping)
    assert isinstance(source_metadata, Mapping)
    assert isinstance(source_protocol, Mapping)
    assert isinstance(source_predecessors, Mapping)
    assert isinstance(source_selection, Mapping)
    assert isinstance(source_validation, Mapping)
    assert isinstance(output_codec, LinearActivationCodec)
    source_lock = source_selection["lock"]
    if (
        not isinstance(source_lock, Mapping)
        or source_lock.get("selection_failed") is not True
        or source_lock.get("reduced_candidate_found") is not False
        or source_validation.get("fidelity_viable_reduced_rank")
        is not False
    ):
        raise ValueError(
            "codimension rotation requires the negative projection ladder"
        )
    width = source_protocol.get("residual_width")
    start_layer = source_protocol.get("start_layer")
    end_layer = source_protocol.get("end_layer_inclusive")
    boundaries = source_protocol.get("canonical_boundaries")
    layer_ids = source_protocol.get("layer_ids")
    if (
        type(width) is not int
        or width < 2
        or type(start_layer) is not int
        or type(end_layer) is not int
        or not isinstance(boundaries, tuple)
        or not isinstance(layer_ids, tuple)
        or len(boundaries) != end_layer - start_layer + 2
        or output_codec.width != width
        or output_codec.activation_name != boundaries[-1]
        or source["locked_candidate"]["retained_rank"] != width  # type: ignore[index]
    ):
        raise ValueError("source projection geometry is invalid")
    if width == 640 and tail_width != DEFAULT_TAIL_WIDTH:
        raise ValueError(
            "width-640 rotation must use the preregistered tail width 32"
        )
    if width == 640 and (
        minimum_alignment != DEFAULT_MIN_SPLIT_HALF_ALIGNMENT
        or minimum_eigengap != DEFAULT_MIN_RELATIVE_EIGENGAP
        or minimum_operator_cosine
        != DEFAULT_MIN_SPLIT_HALF_OPERATOR_COSINE
        or maximum_relative_regret
        != DEFAULT_MAX_SPLIT_HALF_RELATIVE_REGRET
    ):
        raise ValueError(
            "width-640 rotation must use preregistered stability gates"
        )
    if (
        type(tail_width) is not int
        or tail_width <= 1
        or tail_width >= width
    ):
        raise ValueError("tail_width must be between two and width minus one")
    if source_model.get("model_id") != model_id:
        raise ValueError("requested model_id does not match source")
    if revision is not None and revision not in {
        source_model.get("requested_revision"),
        source_model.get("resolved_commit"),
    }:
        raise ValueError("explicit revision does not match source")

    prompts = load_gemma3_prompt_splits(prompt_splits_path)
    if prompts.scientific_status not in _PROMPT_STATUSES:
        raise ValueError("rotation prompt scientific status is invalid")
    prompt_metadata = prompts.metadata()
    if width == 640:
        expected_stability_policy = (
            DEFAULT_STABILITY_POLICY
            if prompts.scientific_status == _PROMPT_STATUS
            else EXPANDED_STABILITY_POLICY
        )
        if stability_policy != expected_stability_policy:
            raise ValueError(
                "width-640 rotation stability policy is noncanonical"
            )
        expected_counts = (
            {
                "calibration_a": 16,
                "calibration_b": 16,
                "validation": 16,
                "test": 16,
            }
            if prompts.scientific_status == _PROMPT_STATUS
            else {
                "calibration_a": 64,
                "calibration_b": 16,
                "validation": 16,
                "test": 16,
            }
        )
        if prompt_metadata["counts"] != expected_counts:
            raise ValueError(
                "width-640 rotation prompt counts are noncanonical"
            )
    disjointness = _assert_prompt_disjointness(
        fresh=prompt_metadata,
        source_protocol=source_protocol,
        source_predecessors=source_predecessors,
    )
    candidates = _candidate_schedule(width)
    candidate_ids = tuple(
        candidate.candidate_id for candidate in candidates
    )
    tail_basis = _tail_complement_basis(
        output_codec.decoder,
        tail_width=tail_width,
    )
    codec_normal = _codec_prefix_normal(output_codec.decoder)
    codec_projector_equivalence = (
        _codec_prefix_projector_equivalence(
            output_codec.decoder,
            codec_normal,
        )
    )
    if codec_projector_equivalence["passed"] is not True:
        raise ValueError(
            "codec-prefix normal does not reproduce rank-639 "
            "least-squares projection"
        )
    codec_tail_residual = float(
        (
            codec_normal
            - tail_basis @ (tail_basis.T @ codec_normal)
        )
        .abs()
        .max()
        .item()
    )
    if codec_tail_residual > 1e-8:
        raise ValueError(
            "codec-prefix normal is outside preregistered tail complement"
        )
    resolved_output = (
        default_gemma3_codimension_rotation_output(
            model_id,
            start_layer,
            end_layer,
        )
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")

    device = resolve_torch_device(device_name)
    if device.type == "mps":
        raise ValueError(
            "codimension rotation requires CPU or CUDA because its "
            "sensitivity fit uses float64 matrix controls"
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
        raise ValueError("live adapter block does not match source")
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=requested_revision,
    )
    for field in ("model_id", "config_sha256", "hidden_size"):
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
        codec_normal=codec_normal,
        stability_policy=stability_policy,
        minimum_split_half_alignment=minimum_alignment,
        minimum_relative_eigengap=minimum_eigengap,
        minimum_split_half_operator_cosine=(
            minimum_operator_cosine
        ),
        maximum_split_half_relative_regret=(
            maximum_relative_regret
        ),
    )
    calibration_a_stream = calibration_a_provenance.metadata()
    rotated_projector = CodimensionOneDeltaProjector.from_state_dict(
        sensitivity["rotated_projector"]  # type: ignore[arg-type]
    )
    codec_projector = CodimensionOneDeltaProjector.from_state_dict(
        sensitivity["codec_prefix_projector"]  # type: ignore[arg-type]
    )
    sensitivity_summary = sensitivity["summary"]
    assert isinstance(sensitivity_summary, Mapping)
    if sensitivity_summary["sensitivity_fit_stable"] is not True:
        split_objective = sensitivity_summary[
            "split_half_objective_diagnostics"
        ]
        assert isinstance(split_objective, Mapping)
        split_relative_regret = split_objective[
            "maximum_pooled_candidate_relative_excess_over_minimum"
        ]
        raise RuntimeError(
            "calibration-A balanced tail direction is not identifiable: "
            "split_half_absolute_normal_alignment="
            f"{sensitivity_summary['split_half_absolute_normal_alignment']}"
            f" (minimum {minimum_alignment}); "
            "combined_minimum_relative_eigengap="
            f"{sensitivity_summary['combined_minimum_relative_eigengap']}"
            f" (minimum {minimum_eigengap}); "
            "split_half_operator_frobenius_cosine="
            f"{split_objective['normalized_operator_frobenius_cosine']}"
            f" (minimum {minimum_operator_cosine}); "
            "maximum_split_half_relative_regret="
            f"{split_relative_regret}"
            f" (maximum {maximum_relative_regret}); "
            f"stability_policy={stability_policy}"
        )
    projectors = {
        candidates[0].candidate_id: rotated_projector,
        candidates[1].candidate_id: codec_projector,
        candidates[2].candidate_id: None,
    }
    guard.assert_unchanged()

    selection_batches, selection_stream = _materialize_split(
        tokenizer,
        prompts.calibration_b,
        split_name="calibration_b",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    selection_boundaries = _collect_boundaries(
        adapter,
        selection_batches,
        plan=plan,
    )
    selection_direct = _evaluate_direct(
        selection_boundaries,
        candidates=candidates,
        projectors=projectors,
    )
    selection_behavior = _evaluate_behavior(
        adapter,
        selection_batches,
        plan=plan,
        candidates=candidates,
        projectors=projectors,
    )
    identity_candidate = candidates[-1]
    identity_passed = _full_width_identity_passed(
        direct=selection_direct[identity_candidate.candidate_id],
        behavior=selection_behavior[identity_candidate.candidate_id],
        identity_nll_atol=identity_tolerance,
    )
    if not identity_passed:
        raise RuntimeError(
            "calibration-B full-width rotation identity failed"
        )
    ledger = _candidate_ledger(
        candidates,
        direct=selection_direct,
        behavior=selection_behavior,
        nll_atol=nll_atol,
        top1_min=top1_min,
    )
    locked, lock = _lock_candidate(candidates, ledger)
    locked_b_row = next(
        row
        for row in ledger
        if row["candidate"]["candidate_id"]  # type: ignore[index]
        == locked.candidate_id
    )
    locked_b_passed = (
        locked_b_row["behavior_fidelity_passed"] is True
    )
    guard.assert_unchanged()

    reduced = locked.retained_rank < width
    validation_evaluated = reduced and not bool(
        lock["selection_failed"]
    )
    validation_stream: dict[str, object] | None
    validation_direct: dict[str, object] | None
    validation_behavior: dict[str, object] | None
    validation_gates: dict[str, bool] | None
    if validation_evaluated:
        validation_batches, validation_stream = _materialize_split(
            tokenizer,
            prompts.validation,
            split_name="validation",
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        validation_boundaries = _collect_boundaries(
            adapter,
            validation_batches,
            plan=plan,
        )
        validation_direct = _evaluate_direct(
            validation_boundaries,
            candidates=(locked,),
            projectors={
                locked.candidate_id: projectors[locked.candidate_id]
            },
        )[locked.candidate_id]
        validation_behavior = _evaluate_behavior(
            adapter,
            validation_batches,
            plan=plan,
            candidates=(locked,),
            projectors={
                locked.candidate_id: projectors[locked.candidate_id]
            },
        )[locked.candidate_id]
        validation_gates = _behavior_gate(
            validation_behavior,
            nll_atol=nll_atol,
            top1_min=top1_min,
        )
        validation_passed = all(validation_gates.values())
        validation_reason = (
            "one_locked_reduced_candidate_evaluated"
        )
    else:
        validation_stream = None
        validation_direct = None
        validation_behavior = None
        validation_gates = None
        validation_passed = False
        validation_reason = (
            "no_reduced_candidate_passed_calibration_b_"
            "validation_not_tokenized"
        )
    rank_639_viable = (
        validation_evaluated
        and reduced
        and locked_b_passed
        and validation_passed
    )
    meaningful = (
        rank_639_viable
        and locked.retained_fraction <= meaningful_fraction
    )
    rotated_b_passed = (
        ledger[0]["behavior_fidelity_passed"] is True
    )
    codec_b_passed = ledger[1]["behavior_fidelity_passed"] is True
    sensitivity_fit_stable = (
        sensitivity_summary["sensitivity_fit_stable"] is True
    )
    pareto_dominates_codec = (
        sensitivity_summary[
            "balanced_candidate_pareto_dominates_codec"
        ]
        is True
    )
    basis_ordering_supported = (
        locked.normal_source
        == "calibration_a_balanced_tail_rotation"
        and rotated_b_passed
        and not codec_b_passed
        and validation_passed
        and sensitivity_fit_stable
        and pareto_dominates_codec
    )
    guard.assert_unchanged()

    source_binding = {
        "schema": source["report"]["schema"],  # type: ignore[index]
        "format_version": source["report"]["format_version"],  # type: ignore[index]
        "scientific_payload_sha256": source_metadata[
            "scientific_payload_sha256"
        ],
        "report_sha256": source_metadata["report_sha256"],
        "tensor_file_sha256": source_artifact_sha256,
        "selection_failed": source_lock["selection_failed"],
        "locked_candidate": copy.deepcopy(
            dict(source["locked_candidate"])  # type: ignore[arg-type]
        ),
        "output_codec_sha256": _codec_state_sha256(output_codec),
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
        "prompt_disjointness": disjointness,
        "codec_prefix_projector_equivalence": (
            codec_projector_equivalence
        ),
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
        "tail_width": tail_width,
        "preserved_codec_prefix_rank": width - tail_width,
        "candidate_schedule": tuple(
            candidate.metadata() for candidate in candidates
        ),
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "sensitivity_fit_split": "calibration_a_only",
        "sensitivity_score": (
            "summed_cross_entropy_to_native_detached_top1_tokens"
        ),
        "sensitivity_operator": (
            "equal_weight_trace_normalized_tail_score_fisher_and_"
            "block_delta_second_moment"
        ),
        "minimum_split_half_alignment": (
            minimum_alignment
        ),
        "minimum_relative_eigengap": (
            minimum_eigengap
        ),
        "stability_policy": stability_policy,
        "minimum_split_half_operator_cosine": (
            minimum_operator_cosine
        ),
        "maximum_split_half_relative_regret": (
            maximum_relative_regret
        ),
        "projection": (
            "target_informed_shared_euclidean_codimension_one_"
            "block_delta_projection"
        ),
        "inference_executor": False,
        "behavioral_upper_bound_claim": False,
        "direct_metrics_influence_lock": False,
        "selection_policy": (
            "calibration_b_rotated_then_codec_control_passing_behavior_"
            "else_full_width_identity"
        ),
        "validation_policy": (
            "evaluate_one_locked_reduced_candidate_per_batch_else_"
            "do_not_tokenize_validation"
        ),
        "test_policy": "parse_validate_hash_only",
        "prompt_fixture_file_sha256": _file_sha256(
            prompt_splits_path
        ),
        "selection_nll_atol": nll_atol,
        "selection_top1_min": top1_min,
        "identity_nll_atol": identity_tolerance,
        "max_meaningful_retained_fraction": meaningful_fraction,
        "compression_claim": False,
        "parameter_mac_speed_claim": False,
        "model_state_guard": guard.metadata(),
        "library_versions": _library_versions(),
        "tokenizer": _tokenizer_provenance(tokenizer),
        "tokenized_splits": tokenized_splits,
        "prompt_splits": prompt_metadata,
    }
    payload = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "scientific_status": {
            "scope": (
                "target_informed_codimension_one_tail_span_search_"
                "with_rotation_discriminator"
            ),
            "calibration_a_sensitivity_fitted": True,
            "calibration_b_candidates_evaluated": True,
            "validation_locked_before_evaluation": True,
            "validation_evaluated": validation_evaluated,
            "locked_validation_interventions_per_batch": (
                1 if validation_evaluated else 0
            ),
            "test_evaluated": False,
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "prompt_text_in_artifact": False,
            "inference_executor": False,
            "behavioral_upper_bound_claim": False,
            "compression_claim": False,
            "parameter_mac_speed_claim": False,
            "selection_failed": lock["selection_failed"],
            "sensitivity_fit_stable": sensitivity_fit_stable,
            "balanced_candidate_pareto_dominates_codec": (
                pareto_dominates_codec
            ),
            "rank_639_fidelity_viable": rank_639_viable,
            "basis_ordering_supported": basis_ordering_supported,
            "meaningful_rank_compression": meaningful,
        },
        "model": model_metadata,
        "source_projection": source_binding,
        "protocol": protocol,
        "output_codec": output_codec.state_dict(),
        "calibration_a_sensitivity": sensitivity,
        "selection": {
            "candidate_direct_diagnostics": selection_direct,
            "candidate_behavior": selection_behavior,
            "full_width_identity": {
                "candidate": identity_candidate.metadata(),
                "direct": selection_direct[
                    identity_candidate.candidate_id
                ],
                "behavior": selection_behavior[
                    identity_candidate.candidate_id
                ],
                "passed": True,
            },
            "lock": lock,
            "tokenized_stream": selection_stream,
        },
        "validation": {
            "evaluated": validation_evaluated,
            "reason": validation_reason,
            "locked_candidate": locked.metadata(),
            "direct_diagnostic": validation_direct,
            "behavior": validation_behavior,
            "behavior_gates": validation_gates,
            "behavior_fidelity_passed": validation_passed,
            "rank_639_fidelity_viable": rank_639_viable,
            "sensitivity_fit_stable": sensitivity_fit_stable,
            "balanced_candidate_pareto_dominates_codec": (
                pareto_dominates_codec
            ),
            "basis_ordering_supported": basis_ordering_supported,
            "meaningful_rank_compression": meaningful,
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
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _validated_symmetric_psd(
    value: object,
    *,
    size: int,
    label: str,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype is not torch.float64
        or value.device.type != "cpu"
        or value.shape != (size, size)
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{label} matrix is invalid")
    if not torch.allclose(value, value.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{label} matrix is not symmetric")
    eigenvalues = torch.linalg.eigvalsh((value + value.T) * 0.5)
    tolerance = max(
        1e-12,
        float(value.abs().max().item()) * 1e-10,
    )
    if float(eigenvalues[0].item()) < -tolerance:
        raise ValueError(f"{label} matrix is not positive semidefinite")
    return value


def _direction_is_minimum_eigenvector(
    matrix: Tensor,
    direction: Tensor,
    *,
    label: str,
) -> None:
    if (
        direction.dtype is not torch.float64
        or direction.device.type != "cpu"
        or direction.ndim != 1
        or direction.shape[0] != matrix.shape[0]
        or not torch.isfinite(direction).all()
        or not math.isclose(
            float(torch.linalg.vector_norm(direction).item()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise ValueError(f"{label} direction is invalid")
    eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) * 0.5)
    rayleigh = _rayleigh(matrix, direction)
    scale = max(float(matrix.abs().max().item()), 1.0)
    tolerance = max(1e-11, scale * 1e-9)
    residual = torch.linalg.vector_norm(
        matrix @ direction - rayleigh * direction
    )
    if (
        rayleigh > float(eigenvalues[0].item()) + tolerance
        or float(residual.item()) > tolerance * math.sqrt(matrix.shape[0])
    ):
        raise ValueError(f"{label} is not a minimum eigenvector")


def _close_scalar(
    left: object,
    right: float,
    *,
    label: str,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-12,
) -> None:
    if (
        type(left) is not float
        or not math.isfinite(left)
        or not math.isclose(
            left,
            right,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
    ):
        raise ValueError(f"{label} does not recompute")


def _semantic_numeric_equal(
    left: object,
    right: object,
    *,
    rel_tol: float = 1e-11,
    abs_tol: float = 1e-13,
) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _semantic_numeric_equal(
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
            _semantic_numeric_equal(
                left_value,
                right_value,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            for left_value, right_value in zip(
                left,
                right,
                strict=True,
            )
        )
    if type(left) is float and type(right) is float:
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(
                left,
                right,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        )
    return type(left) is type(right) and left == right


def _validate_direct_example_rows(
    examples: object,
    *,
    stream: Mapping[str, object],
) -> list[Mapping[str, object]]:
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
    stream_examples = stream.get("examples")
    if not isinstance(examples, list) or not isinstance(
        stream_examples,
        list,
    ):
        raise ValueError("rotation direct example rows are invalid")
    if len(examples) != len(stream_examples):
        raise ValueError("rotation direct example count is invalid")
    validated: list[Mapping[str, object]] = []
    tiny = torch.finfo(torch.float64).tiny
    for row, stream_row in zip(examples, stream_examples, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != fields
            or not isinstance(stream_row, Mapping)
            or row["example_id"] != stream_row.get("example_id")
            or row["valid_tokens"] != stream_row.get("valid_tokens")
            or type(row["valid_tokens"]) is not int
            or row["valid_tokens"] <= 0
        ):
            raise ValueError("rotation direct example identity is invalid")
        for field in fields - {"example_id", "valid_tokens"}:
            value = row[field]
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(
                    "rotation direct example scalar is invalid"
                )
        for field in (
            "squared_error",
            "block_delta_energy",
            "full_output_energy",
            "prediction_energy",
            "predicted_block_delta_energy",
            "block_delta_nrmse",
            "full_output_nrmse",
        ):
            if row[field] < 0.0:
                raise ValueError(
                    "rotation direct example energy is negative"
                )
        for field in ("block_delta_cosine", "full_output_cosine"):
            if not -1.0 <= row[field] <= 1.0:
                raise ValueError(
                    "rotation direct example cosine is outside [-1, 1]"
                )
        _close_scalar(
            row["block_delta_nrmse"],
            math.sqrt(
                row["squared_error"]
                / max(row["block_delta_energy"], tiny)
            ),
            label="per-example block-delta NRMSE",
        )
        _close_scalar(
            row["full_output_nrmse"],
            math.sqrt(
                row["squared_error"]
                / max(row["full_output_energy"], tiny)
            ),
            label="per-example full-output NRMSE",
        )
        _close_scalar(
            row["block_delta_cosine"],
            _safe_cosine(
                row["block_delta_dot"],
                row["block_delta_energy"],
                row["predicted_block_delta_energy"],
            ),
            label="per-example block-delta cosine",
        )
        _close_scalar(
            row["full_output_cosine"],
            _safe_cosine(
                row["full_output_dot"],
                row["full_output_energy"],
                row["prediction_energy"],
            ),
            label="per-example full-output cosine",
        )
        validated.append(row)
    return validated


def _validate_behavior_example_rows(
    examples: object,
    *,
    stream: Mapping[str, object],
) -> list[Mapping[str, object]]:
    fields = {
        "example_id",
        "supervised_tokens",
        "baseline_summed_nll",
        "predicted_summed_nll",
        "delta_summed_nll",
        "delta_nll_per_token",
        "top1_matches",
        "top1_agreement",
    }
    stream_examples = stream.get("examples")
    if not isinstance(examples, list) or not isinstance(
        stream_examples,
        list,
    ):
        raise ValueError("rotation behavior example rows are invalid")
    if len(examples) != len(stream_examples):
        raise ValueError("rotation behavior example count is invalid")
    validated: list[Mapping[str, object]] = []
    for row, stream_row in zip(examples, stream_examples, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != fields
            or not isinstance(stream_row, Mapping)
            or row["example_id"] != stream_row.get("example_id")
            or row["supervised_tokens"]
            != stream_row.get("supervised_positions")
            or type(row["supervised_tokens"]) is not int
            or row["supervised_tokens"] <= 0
            or type(row["top1_matches"]) is not int
            or not 0
            <= row["top1_matches"]
            <= row["supervised_tokens"]
        ):
            raise ValueError("rotation behavior example identity is invalid")
        for field in fields - {
            "example_id",
            "supervised_tokens",
            "top1_matches",
        }:
            value = row[field]
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(
                    "rotation behavior example scalar is invalid"
                )
        if (
            row["baseline_summed_nll"] < 0.0
            or row["predicted_summed_nll"] < 0.0
            or not 0.0 <= row["top1_agreement"] <= 1.0
        ):
            raise ValueError(
                "rotation behavior example range is invalid"
            )
        expected_delta = (
            row["predicted_summed_nll"]
            - row["baseline_summed_nll"]
        )
        _close_scalar(
            row["delta_summed_nll"],
            expected_delta,
            label="per-example summed NLL delta",
        )
        _close_scalar(
            row["delta_nll_per_token"],
            expected_delta / row["supervised_tokens"],
            label="per-example NLL delta per token",
        )
        _close_scalar(
            row["top1_agreement"],
            row["top1_matches"] / row["supervised_tokens"],
            label="per-example top-1 agreement",
        )
        validated.append(row)
    return validated


def _recompute_rotation_direct(
    value: object,
    *,
    stream: Mapping[str, object],
    candidate: RotationCandidate,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or not isinstance(
        value.get("examples"),
        list,
    ):
        raise ValueError("rotation direct aggregate examples are invalid")
    examples = _validate_direct_example_rows(
        value["examples"],
        stream=stream,
    )
    recomputed = _aggregate_direct_examples(
        examples,  # type: ignore[arg-type]
        width=candidate.residual_width,
    )
    recomputed["projection"] = candidate.metadata()["projection"]
    recomputed["normal_source"] = candidate.normal_source
    if recomputed != value:
        raise ValueError(
            "rotation direct aggregate does not recompute from examples"
        )
    return recomputed


def _recompute_rotation_behavior(
    value: object,
    *,
    stream: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or "examples" not in value:
        raise ValueError("rotation behavior aggregate is invalid")
    examples = _validate_behavior_example_rows(
        value["examples"],
        stream=stream,
    )
    recomputed = _behavior_aggregate(examples)
    if recomputed != value:
        raise ValueError(
            "rotation behavior aggregate does not recompute from examples"
        )
    return recomputed


def _validate_shared_candidate_native_values(
    candidate_ids: Sequence[str],
    *,
    direct: Mapping[str, Mapping[str, object]],
    behavior: Mapping[str, Mapping[str, object]],
) -> None:
    if not candidate_ids:
        raise ValueError("rotation candidate IDs cannot be empty")
    reference_direct = direct[candidate_ids[0]]["examples"]
    reference_behavior = behavior[candidate_ids[0]]["examples"]
    assert isinstance(reference_direct, list)
    assert isinstance(reference_behavior, list)
    for candidate_id in candidate_ids[1:]:
        candidate_direct = direct[candidate_id]["examples"]
        candidate_behavior = behavior[candidate_id]["examples"]
        if not isinstance(candidate_direct, list) or not isinstance(
            candidate_behavior,
            list,
        ):
            raise ValueError("rotation candidate example ledgers are invalid")
        if tuple(
            (
                row["example_id"],
                row["block_delta_energy"],
                row["full_output_energy"],
            )
            for row in candidate_direct
        ) != tuple(
            (
                row["example_id"],
                row["block_delta_energy"],
                row["full_output_energy"],
            )
            for row in reference_direct
        ):
            raise ValueError(
                "rotation direct candidates do not share native targets"
            )
        if tuple(
            (
                row["example_id"],
                row["baseline_summed_nll"],
            )
            for row in candidate_behavior
        ) != tuple(
            (
                row["example_id"],
                row["baseline_summed_nll"],
            )
            for row in reference_behavior
        ):
            raise ValueError(
                "rotation behavior candidates do not share native baselines"
            )


def _validate_sensitivity_payload(
    value: object,
    *,
    codec: LinearActivationCodec,
    tail_width: int,
    stream: Mapping[str, object],
    stability_policy: str,
    minimum_split_half_alignment: float,
    minimum_relative_eigengap: float,
    minimum_split_half_operator_cosine: float,
    maximum_split_half_relative_regret: float,
) -> tuple[
    CodimensionOneDeltaProjector,
    CodimensionOneDeltaProjector,
    Mapping[str, object],
]:
    fields = {
        "tail_basis",
        "score_fisher",
        "ground_truth_nll_score_fisher",
        "delta_second_moment",
        "split_half_score_fisher",
        "split_half_ground_truth_nll_score_fisher",
        "split_half_delta_second_moment",
        "split_half_observations",
        "split_half_normals",
        "combined_operator",
        "combined_eigenvalues",
        "rotated_projector",
        "codec_prefix_projector",
        "summary",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("rotation sensitivity payload fields are invalid")
    width = codec.width
    tail_basis = value["tail_basis"]
    if (
        not isinstance(tail_basis, Tensor)
        or tail_basis.dtype is not torch.float64
        or tail_basis.device.type != "cpu"
        or tail_basis.shape != (width, tail_width)
        or not torch.isfinite(tail_basis).all()
    ):
        raise ValueError("rotation tail basis is invalid")
    identity = torch.eye(tail_width, dtype=torch.float64)
    prefix = codec.decoder.detach().to(torch.float64)[
        :, : width - tail_width
    ]
    if (
        not torch.allclose(
            tail_basis.T @ tail_basis,
            identity,
            rtol=1e-9,
            atol=1e-10,
        )
        or float((prefix.T @ tail_basis).abs().max().item()) > 1e-8
    ):
        raise ValueError("rotation tail basis does not span the complement")
    score_fisher = _validated_symmetric_psd(
        value["score_fisher"],
        size=tail_width,
        label="pseudo-top1 score Fisher",
    )
    nll_fisher = _validated_symmetric_psd(
        value["ground_truth_nll_score_fisher"],
        size=tail_width,
        label="ground-truth NLL score Fisher",
    )
    delta_moment = _validated_symmetric_psd(
        value["delta_second_moment"],
        size=tail_width,
        label="block-delta second moment",
    )
    score_trace = float(torch.trace(score_fisher).item())
    nll_trace = float(torch.trace(nll_fisher).item())
    delta_trace = float(torch.trace(delta_moment).item())
    if min(score_trace, nll_trace, delta_trace) <= 0.0:
        raise ValueError("rotation sensitivity traces must be positive")
    expected_combined = (
        0.5 * score_fisher / score_trace
        + 0.5 * delta_moment / delta_trace
    )
    expected_combined = (
        (expected_combined + expected_combined.T) * 0.5
    )
    combined = _validated_symmetric_psd(
        value["combined_operator"],
        size=tail_width,
        label="combined sensitivity-energy",
    )
    if not torch.allclose(
        combined,
        expected_combined,
        rtol=1e-12,
        atol=1e-13,
    ):
        raise ValueError("combined sensitivity operator does not recompute")
    eigenvalues = value["combined_eigenvalues"]
    expected_eigenvalues = torch.linalg.eigvalsh(combined)
    if (
        not isinstance(eigenvalues, Tensor)
        or eigenvalues.dtype is not torch.float64
        or eigenvalues.device.type != "cpu"
        or eigenvalues.shape != (tail_width,)
        or not torch.allclose(
            eigenvalues,
            expected_eigenvalues,
            rtol=1e-12,
            atol=1e-13,
        )
    ):
        raise ValueError("combined sensitivity eigenvalues do not recompute")
    rotated = CodimensionOneDeltaProjector.from_state_dict(
        value["rotated_projector"]  # type: ignore[arg-type]
    )
    codec_projector = CodimensionOneDeltaProjector.from_state_dict(
        value["codec_prefix_projector"]  # type: ignore[arg-type]
    )
    if rotated.width != width or codec_projector.width != width:
        raise ValueError("rotation projector widths are invalid")
    rotated_coefficients = tail_basis.T @ rotated.normal
    rotated_tail_residual = (
        rotated.normal - tail_basis @ rotated_coefficients
    )
    if float(rotated_tail_residual.abs().max().item()) > 1e-8:
        raise ValueError("rotated normal is outside the tail complement")
    _direction_is_minimum_eigenvector(
        combined,
        rotated_coefficients,
        label="rotated tail direction",
    )
    expected_codec_normal = _codec_prefix_normal(codec.decoder)
    if not torch.allclose(
        codec_projector.normal,
        expected_codec_normal,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("codec-prefix normal does not recompute")

    split_score = value["split_half_score_fisher"]
    split_nll_score = value[
        "split_half_ground_truth_nll_score_fisher"
    ]
    split_delta = value["split_half_delta_second_moment"]
    split_observations = value["split_half_observations"]
    split_normals = value["split_half_normals"]
    if (
        not isinstance(split_score, Tensor)
        or split_score.dtype is not torch.float64
        or split_score.shape != (2, tail_width, tail_width)
        or not isinstance(split_nll_score, Tensor)
        or split_nll_score.dtype is not torch.float64
        or split_nll_score.shape != (2, tail_width, tail_width)
        or not isinstance(split_delta, Tensor)
        or split_delta.dtype is not torch.float64
        or split_delta.shape != (2, tail_width, tail_width)
        or not isinstance(split_observations, tuple)
        or len(split_observations) != 2
        or any(type(item) is not int or item <= 0 for item in split_observations)
        or not isinstance(split_normals, Tensor)
        or split_normals.dtype is not torch.float64
        or split_normals.shape != (2, width)
        or not torch.isfinite(split_normals).all()
    ):
        raise ValueError("rotation split-half payload is invalid")
    for index in range(2):
        _validated_symmetric_psd(
            split_score[index],
            size=tail_width,
            label=f"split-half-{index} score Fisher",
        )
        _validated_symmetric_psd(
            split_nll_score[index],
            size=tail_width,
            label=f"split-half-{index} ground-truth NLL Fisher",
        )
        _validated_symmetric_psd(
            split_delta[index],
            size=tail_width,
            label=f"split-half-{index} delta moment",
        )
    total_observations = sum(split_observations)
    weighted_score = sum(
        split_score[index] * split_observations[index]
        for index in range(2)
    ) / total_observations
    weighted_nll_score = sum(
        split_nll_score[index] * split_observations[index]
        for index in range(2)
    ) / total_observations
    weighted_delta = sum(
        split_delta[index] * split_observations[index]
        for index in range(2)
    ) / total_observations
    if (
        not torch.allclose(
            weighted_score,
            score_fisher,
            rtol=1e-12,
            atol=1e-13,
        )
        or not torch.allclose(
            weighted_nll_score,
            nll_fisher,
            rtol=1e-12,
            atol=1e-13,
        )
        or not torch.allclose(
            weighted_delta,
            delta_moment,
            rtol=1e-12,
            atol=1e-13,
        )
    ):
        raise ValueError("rotation split halves do not reconstruct moments")
    expected_split_normals = []
    for index in range(2):
        half_score_trace = torch.trace(split_score[index])
        half_delta_trace = torch.trace(split_delta[index])
        half_operator = (
            0.5 * split_score[index] / half_score_trace
            + 0.5 * split_delta[index] / half_delta_trace
        )
        _, half_vectors = torch.linalg.eigh(
            (half_operator + half_operator.T) * 0.5
        )
        expected_split_normals.append(
            canonical_unit_direction(
                tail_basis @ half_vectors[:, 0],
                label=f"split-half-{index} omitted normal",
            )
        )
    expected_split_normals_tensor = torch.stack(
        expected_split_normals
    )
    if not torch.allclose(
        split_normals,
        expected_split_normals_tensor,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("rotation split-half normals do not recompute")

    summary = value["summary"]
    if not isinstance(summary, Mapping):
        raise ValueError("rotation sensitivity summary is invalid")
    examples = summary.get("examples")
    expected_stream_examples = stream["examples"]
    if not isinstance(examples, list) or not isinstance(
        expected_stream_examples,
        list,
    ):
        raise ValueError("rotation sensitivity examples are invalid")
    example_fields = {
        "example_id",
        "valid_tokens",
        "supervised_tokens",
        "native_top1_teacher_sha256",
        "native_top1_margin_min",
        "native_top1_margin_sum",
        "native_top1_exact_ties",
        "pseudo_label_summed_cross_entropy",
        "ground_truth_summed_nll",
        "score_gradient_squared_norm",
        "block_delta_squared_norm",
        "ground_truth_nll_score_gradient_squared_norm",
        "tail_score_gradient_squared_norm",
        "tail_block_delta_squared_norm",
        "tail_ground_truth_nll_score_gradient_squared_norm",
    }
    scalar_fields = example_fields - {
        "example_id",
        "valid_tokens",
        "supervised_tokens",
        "native_top1_teacher_sha256",
        "native_top1_exact_ties",
    }
    seen_ids: set[str] = set()
    for row in examples:
        if (
            not isinstance(row, Mapping)
            or set(row) != example_fields
            or not isinstance(row["example_id"], str)
            or not row["example_id"]
            or row["example_id"] in seen_ids
            or type(row["valid_tokens"]) is not int
            or row["valid_tokens"] <= 0
            or type(row["supervised_tokens"]) is not int
            or row["supervised_tokens"] <= 0
            or not _is_sha256(row["native_top1_teacher_sha256"])
            or type(row["native_top1_exact_ties"]) is not int
            or not 0
            <= row["native_top1_exact_ties"]
            <= row["supervised_tokens"]
            or any(
                not isinstance(row[field], (int, float))
                or isinstance(row[field], bool)
                or not math.isfinite(float(row[field]))
                or float(row[field]) < 0.0
                for field in scalar_fields
            )
            or float(row["native_top1_margin_sum"])
            + 1e-12
            < (
                row["supervised_tokens"]
                * float(row["native_top1_margin_min"])
            )
            or float(row["tail_score_gradient_squared_norm"])
            > float(row["score_gradient_squared_norm"])
            + max(
                1e-12,
                float(row["score_gradient_squared_norm"]) * 1e-9,
            )
            or float(
                row[
                    "tail_ground_truth_nll_score_gradient_squared_norm"
                ]
            )
            > float(
                row["ground_truth_nll_score_gradient_squared_norm"]
            )
            + max(
                1e-12,
                float(
                    row[
                        "ground_truth_nll_score_gradient_squared_norm"
                    ]
                )
                * 1e-9,
            )
            or float(row["tail_block_delta_squared_norm"])
            > float(row["block_delta_squared_norm"])
            + max(
                1e-12,
                float(row["block_delta_squared_norm"]) * 1e-9,
            )
        ):
            raise ValueError(
                "rotation sensitivity example ledger is invalid"
            )
        seen_ids.add(row["example_id"])
    if tuple(
        (
            row.get("example_id"),
            row.get("valid_tokens"),
            row.get("supervised_tokens"),
        )
        for row in examples
        if isinstance(row, Mapping)
    ) != tuple(
        (
            row.get("example_id"),
            row.get("valid_tokens"),
            row.get("supervised_positions"),
        )
        for row in expected_stream_examples
        if isinstance(row, Mapping)
    ):
        raise ValueError(
            "rotation sensitivity examples do not bind calibration A"
        )
    expected_split_observations = tuple(
        sum(
            int(row["valid_tokens"])
            for row in examples[index::2]
        )
        for index in range(2)
    )
    if split_observations != expected_split_observations:
        raise ValueError(
            "rotation split-half observations do not bind examples"
        )
    ledger_trace_controls = (
        (
            score_fisher,
            "tail_score_gradient_squared_norm",
            "pseudo-top1 score Fisher",
        ),
        (
            nll_fisher,
            "tail_ground_truth_nll_score_gradient_squared_norm",
            "ground-truth NLL score Fisher",
        ),
        (
            delta_moment,
            "tail_block_delta_squared_norm",
            "block-delta second moment",
        ),
    )
    for matrix, field, label in ledger_trace_controls:
        ledger_trace = (
            sum(float(row[field]) for row in examples)
            / total_observations
        )
        if not math.isclose(
            ledger_trace,
            float(torch.trace(matrix).item()),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{label} trace does not bind example ledger"
            )
    expected_summary = _sensitivity_summary(
        tail_basis=tail_basis,
        score_fisher=score_fisher,
        ground_truth_nll_score_fisher=nll_fisher,
        delta_second_moment=delta_moment,
        combined_operator=combined,
        combined_eigenvalues=eigenvalues,
        rotated_normal=rotated.normal,
        codec_normal=codec_projector.normal,
        split_half_alignment=float(
            torch.dot(split_normals[0], split_normals[1]).abs().item()
        ),
        split_half_score_fisher=split_score,
        split_half_delta_second_moment=split_delta,
        stability_policy=stability_policy,
        minimum_split_half_alignment=minimum_split_half_alignment,
        minimum_relative_eigengap=minimum_relative_eigengap,
        minimum_split_half_operator_cosine=(
            minimum_split_half_operator_cosine
        ),
        maximum_split_half_relative_regret=(
            maximum_split_half_relative_regret
        ),
        examples=examples,  # type: ignore[arg-type]
        observations=total_observations,
        supervised_tokens=sum(
            int(row["supervised_tokens"]) for row in examples
        ),
        tail_width=tail_width,
    )
    if not _semantic_numeric_equal(expected_summary, summary):
        raise ValueError("rotation sensitivity summary does not recompute")
    return rotated, codec_projector, summary


def _validated_rotation_prompt_metadata(
    value: object,
    *,
    streams: Mapping[str, Mapping[str, object]],
    validation_evaluated: bool,
) -> set[str]:
    if not isinstance(value, Mapping) or set(value) != {
        "scientific_status",
        "counts",
        "normalized_sha256",
        "per_prompt_sha256",
    }:
        raise ValueError("rotation prompt provenance fields are invalid")
    if value["scientific_status"] not in _PROMPT_STATUSES:
        raise ValueError("rotation prompt scientific status is invalid")
    split_names = (
        "calibration_a",
        "calibration_b",
        "validation",
        "test",
    )
    counts = value["counts"]
    normalized = value["normalized_sha256"]
    per_prompt = value["per_prompt_sha256"]
    if (
        not isinstance(counts, Mapping)
        or tuple(counts) != split_names
        or not isinstance(normalized, Mapping)
        or tuple(normalized) != split_names
        or not isinstance(per_prompt, Mapping)
        or tuple(per_prompt) != split_names
    ):
        raise ValueError("rotation prompt provenance mappings are invalid")
    all_hashes: list[str] = []
    for split_name in split_names:
        count = counts[split_name]
        hashes = per_prompt[split_name]
        if (
            type(count) is not int
            or count <= 0
            or not isinstance(hashes, list)
            or len(hashes) != count
            or any(not _is_sha256(item) for item in hashes)
            or normalized[split_name]
            != _ordered_prompt_hash_digest(hashes)
        ):
            raise ValueError("rotation prompt counts or hashes are invalid")
        all_hashes.extend(hashes)
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("rotation prompt hashes must be pairwise disjoint")
    expected_stream_names = (
        ("calibration_a", "calibration_b", "validation")
        if validation_evaluated
        else ("calibration_a", "calibration_b")
    )
    if tuple(streams) != expected_stream_names:
        raise ValueError("rotation tokenized split set is invalid")
    for split_name in expected_stream_names:
        stream = streams[split_name]
        if (
            stream["sequences"] != counts[split_name]
            or stream["source_prompt_sha256"]
            != per_prompt[split_name]
        ):
            raise ValueError(
                "rotation tokenized stream does not bind prompt hashes"
            )
    streamed_hashes = {
        digest
        for split_name in expected_stream_names
        for digest in per_prompt[split_name]
    }
    hash_only_splits = {"test"}
    if not validation_evaluated:
        hash_only_splits.add("validation")
    if streamed_hashes & {
        digest
        for split_name in hash_only_splits
        for digest in per_prompt[split_name]
    }:
        raise ValueError("hash-only rotation prompt entered model stream")
    return set(all_hashes)


def load_gemma3_codimension_rotation_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strictly load and semantically replay a rotation artifact."""

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
        "source_projection",
        "protocol",
        "output_codec",
        "calibration_a_sensitivity",
        "selection",
        "validation",
        "scientific_payload_sha256",
        "report_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("codimension-rotation artifact fields are invalid")
    if (
        raw["schema"] != _ARTIFACT_SCHEMA
        or type(raw["format_version"]) is not int
        or raw["format_version"] != _ARTIFACT_FORMAT_VERSION
        or raw["contains_model_weights"] is not False
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
        or not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError(
            "unsupported or unsafe codimension-rotation artifact"
        )
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    digest = _scientific_payload_sha256(payload)
    if digest != raw["scientific_payload_sha256"]:
        raise ValueError("codimension-rotation scientific digest mismatch")

    model = _validate_model_metadata(raw["model"])
    source = raw["source_projection"]
    protocol = raw["protocol"]
    codec_state = raw["output_codec"]
    sensitivity = raw["calibration_a_sensitivity"]
    selection = raw["selection"]
    validation = raw["validation"]
    status = raw["scientific_status"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            source,
            protocol,
            codec_state,
            sensitivity,
            selection,
            validation,
            status,
        )
    ):
        raise ValueError("codimension-rotation payload mappings are invalid")
    output_codec = LinearActivationCodec.from_state_dict(codec_state)

    protocol_fields = {
        "start_layer",
        "end_layer_inclusive",
        "layer_ids",
        "canonical_boundaries",
        "residual_width",
        "tail_width",
        "preserved_codec_prefix_rank",
        "candidate_schedule",
        "maximum_tokenized_length",
        "tokenization_batch_size",
        "sensitivity_fit_split",
        "sensitivity_score",
        "sensitivity_operator",
        "minimum_split_half_alignment",
        "minimum_relative_eigengap",
        "stability_policy",
        "minimum_split_half_operator_cosine",
        "maximum_split_half_relative_regret",
        "projection",
        "inference_executor",
        "behavioral_upper_bound_claim",
        "direct_metrics_influence_lock",
        "selection_policy",
        "validation_policy",
        "test_policy",
        "prompt_fixture_file_sha256",
        "selection_nll_atol",
        "selection_top1_min",
        "identity_nll_atol",
        "max_meaningful_retained_fraction",
        "compression_claim",
        "parameter_mac_speed_claim",
        "model_state_guard",
        "library_versions",
        "tokenizer",
        "tokenized_splits",
        "prompt_splits",
    }
    if set(protocol) != protocol_fields:
        raise ValueError("codimension-rotation protocol fields are invalid")
    width = protocol["residual_width"]
    tail_width = protocol["tail_width"]
    start = protocol["start_layer"]
    end = protocol["end_layer_inclusive"]
    layer_ids = protocol["layer_ids"]
    boundaries = protocol["canonical_boundaries"]
    schedule = protocol["candidate_schedule"]
    if (
        type(width) is not int
        or width < 2
        or type(tail_width) is not int
        or not 1 < tail_width < width
        or protocol["preserved_codec_prefix_rank"] != width - tail_width
        or type(start) is not int
        or type(end) is not int
        or start < 0
        or end < start
        or not isinstance(layer_ids, tuple)
        or layer_ids
        != tuple(f"layer.{index}" for index in range(start, end + 1))
        or not isinstance(boundaries, tuple)
        or len(boundaries) != end - start + 2
        or not isinstance(schedule, tuple)
        or output_codec.width != width
        or output_codec.activation_name != boundaries[-1]
    ):
        raise ValueError("codimension-rotation geometry is invalid")
    candidates = _candidate_schedule(width)
    if schedule != tuple(candidate.metadata() for candidate in candidates):
        raise ValueError(
            "codimension-rotation candidate schedule is noncanonical"
        )
    minimum_alignment = _finite(
        protocol["minimum_split_half_alignment"],
        label="minimum_split_half_alignment",
        minimum=0.0,
        maximum=1.0,
    )
    minimum_eigengap = _finite(
        protocol["minimum_relative_eigengap"],
        label="minimum_relative_eigengap",
        minimum=0.0,
    )
    stability_policy = protocol["stability_policy"]
    if stability_policy not in _STABILITY_POLICIES:
        raise ValueError("unknown stored sensitivity stability policy")
    minimum_operator_cosine = _finite(
        protocol["minimum_split_half_operator_cosine"],
        label="minimum_split_half_operator_cosine",
        minimum=0.0,
        maximum=1.0,
    )
    maximum_relative_regret = _finite(
        protocol["maximum_split_half_relative_regret"],
        label="maximum_split_half_relative_regret",
        minimum=0.0,
    )
    nll_atol = _finite(
        protocol["selection_nll_atol"],
        label="selection_nll_atol",
        minimum=0.0,
    )
    top1_min = _finite(
        protocol["selection_top1_min"],
        label="selection_top1_min",
        minimum=0.0,
        maximum=1.0,
    )
    identity_tolerance = _finite(
        protocol["identity_nll_atol"],
        label="identity_nll_atol",
        minimum=0.0,
    )
    meaningful_fraction = _finite(
        protocol["max_meaningful_retained_fraction"],
        label="max_meaningful_retained_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    model_guard = protocol["model_state_guard"]
    library_versions = protocol["library_versions"]
    tokenizer_metadata = protocol["tokenizer"]
    if (
        protocol["sensitivity_fit_split"] != "calibration_a_only"
        or protocol["sensitivity_score"]
        != "summed_cross_entropy_to_native_detached_top1_tokens"
        or protocol["sensitivity_operator"]
        != "equal_weight_trace_normalized_tail_score_fisher_and_"
        "block_delta_second_moment"
        or protocol["projection"]
        != "target_informed_shared_euclidean_codimension_one_"
        "block_delta_projection"
        or protocol["inference_executor"] is not False
        or protocol["behavioral_upper_bound_claim"] is not False
        or protocol["direct_metrics_influence_lock"] is not False
        or protocol["selection_policy"]
        != "calibration_b_rotated_then_codec_control_passing_behavior_"
        "else_full_width_identity"
        or protocol["validation_policy"]
        != "evaluate_one_locked_reduced_candidate_per_batch_else_"
        "do_not_tokenize_validation"
        or protocol["test_policy"] != "parse_validate_hash_only"
        or not _is_sha256(protocol["prompt_fixture_file_sha256"])
        or protocol["compression_claim"] is not False
        or protocol["parameter_mac_speed_claim"] is not False
        or type(protocol["maximum_tokenized_length"]) is not int
        or protocol["maximum_tokenized_length"] < 2
        or type(protocol["tokenization_batch_size"]) is not int
        or protocol["tokenization_batch_size"] <= 0
        or not isinstance(model_guard, Mapping)
        or set(model_guard)
        != {
            "verified",
            "training",
            "parameters_frozen",
            "parameter_tensors",
            "buffer_tensors",
            "checks",
        }
        or model_guard["verified"] is not True
        or model_guard["training"] is not False
        or model_guard["parameters_frozen"] is not True
        or type(model_guard["parameter_tensors"]) is not int
        or model_guard["parameter_tensors"] <= 0
        or type(model_guard["buffer_tensors"]) is not int
        or model_guard["buffer_tensors"] < 0
        or model_guard["checks"]
        != (
            "tensor_object_identity",
            "tensor_version_counter",
            "tensor_storage_identity",
        )
        or not isinstance(library_versions, Mapping)
        or set(library_versions)
        != {
            "python",
            "torch",
            "transformers",
            "tokenizers",
            "sentencepiece",
        }
        or any(
            value is not None and not isinstance(value, str)
            for value in library_versions.values()
        )
        or not isinstance(tokenizer_metadata, Mapping)
        or set(tokenizer_metadata)
        != {
            "tokenizer_class",
            "name_or_path",
            "configuration_sha256",
        }
        or not isinstance(
            tokenizer_metadata["tokenizer_class"],
            str,
        )
        or not tokenizer_metadata["tokenizer_class"]
        or (
            tokenizer_metadata["name_or_path"] is not None
            and not isinstance(
                tokenizer_metadata["name_or_path"],
                str,
            )
        )
        or not _is_sha256(
            tokenizer_metadata["configuration_sha256"]
        )
    ):
        raise ValueError(
            "codimension-rotation scientific semantics are invalid"
        )
    if width == 640 and (
        tail_width != DEFAULT_TAIL_WIDTH
        or minimum_alignment != DEFAULT_MIN_SPLIT_HALF_ALIGNMENT
        or minimum_eigengap != DEFAULT_MIN_RELATIVE_EIGENGAP
        or minimum_operator_cosine
        != DEFAULT_MIN_SPLIT_HALF_OPERATOR_COSINE
        or maximum_relative_regret
        != DEFAULT_MAX_SPLIT_HALF_RELATIVE_REGRET
    ):
        raise ValueError("width-640 rotation protocol is not preregistered")
    if (
        (
            model.get("hidden_size") is not None
            and model["hidden_size"] != width
        )
        or (
            model.get("num_hidden_layers") is not None
            and end >= model["num_hidden_layers"]
        )
    ):
        raise ValueError(
            "codimension-rotation model dimensions do not bind geometry"
        )

    status_validation = status.get("validation_evaluated")
    if type(status_validation) is not bool:
        raise ValueError("rotation validation status is invalid")
    stream_values = protocol["tokenized_splits"]
    if not isinstance(stream_values, Mapping):
        raise ValueError("rotation tokenized streams are invalid")
    streams: dict[str, Mapping[str, object]] = {}
    expected_stream_names = (
        ("calibration_a", "calibration_b", "validation")
        if status_validation
        else ("calibration_a", "calibration_b")
    )
    if tuple(stream_values) != expected_stream_names:
        raise ValueError("rotation tokenized stream names are invalid")
    for split_name in expected_stream_names:
        validated_stream, _ = _validated_tokenized_stream(
            stream_values[split_name],
            split_name=split_name,
        )
        streams[split_name] = validated_stream
    fresh_hashes = _validated_rotation_prompt_metadata(
        protocol["prompt_splits"],
        streams=streams,
        validation_evaluated=status_validation,
    )
    prompt_metadata = protocol["prompt_splits"]
    assert isinstance(prompt_metadata, Mapping)
    if width == 640:
        expected_stability_policy = (
            DEFAULT_STABILITY_POLICY
            if prompt_metadata["scientific_status"] == _PROMPT_STATUS
            else EXPANDED_STABILITY_POLICY
        )
        if stability_policy != expected_stability_policy:
            raise ValueError(
                "width-640 rotation stability policy is noncanonical"
            )
        expected_counts = (
            {
                "calibration_a": 16,
                "calibration_b": 16,
                "validation": 16,
                "test": 16,
            }
            if prompt_metadata["scientific_status"] == _PROMPT_STATUS
            else {
                "calibration_a": 64,
                "calibration_b": 16,
                "validation": 16,
                "test": 16,
            }
        )
        if prompt_metadata["counts"] != expected_counts:
            raise ValueError(
                "width-640 rotation prompt counts are noncanonical"
            )

    source_fields = {
        "schema",
        "format_version",
        "scientific_payload_sha256",
        "report_sha256",
        "tensor_file_sha256",
        "selection_failed",
        "locked_candidate",
        "output_codec_sha256",
        "model_binding",
        "block_geometry",
        "prompt_disjointness",
        "codec_prefix_projector_equivalence",
    }
    if set(source) != source_fields:
        raise ValueError("rotation source-projection fields are invalid")
    source_locked = source["locked_candidate"]
    model_binding = source["model_binding"]
    block_geometry = source["block_geometry"]
    disjointness = source["prompt_disjointness"]
    if (
        source["schema"] != "fisher_graph.gemma3_projection_ladder"
        or type(source["format_version"]) is not int
        or source["format_version"] != 1
        or any(
            not _is_sha256(source[field])
            for field in (
                "scientific_payload_sha256",
                "report_sha256",
                "tensor_file_sha256",
                "output_codec_sha256",
            )
        )
        or source["selection_failed"] is not True
        or not isinstance(source_locked, Mapping)
        or set(source_locked)
        != {
            "candidate_id",
            "retained_rank",
            "residual_width",
            "retained_fraction",
            "removed_dimensions",
            "projection",
        }
        or source_locked.get("retained_rank") != width
        or source_locked.get("residual_width") != width
        or source_locked.get("candidate_id")
        != f"rank_{width}.target_ls_projection"
        or source_locked.get("retained_fraction") != 1.0
        or source_locked.get("removed_dimensions") != 0
        or source_locked.get("projection")
        != "target_informed_per_token_least_squares_in_output_"
        "decoder_span"
        or not isinstance(model_binding, Mapping)
        or set(model_binding)
        != {
            "model_id",
            "config_sha256",
            "resolved_commit",
            "hidden_size",
            "num_hidden_layers",
        }
        or any(
            model_binding[field] != model.get(field)
            for field in model_binding
        )
        or not isinstance(block_geometry, Mapping)
        or block_geometry
        != {
            "start_layer": start,
            "end_layer_inclusive": end,
            "layer_ids": layer_ids,
            "canonical_boundaries": boundaries,
            "residual_width": width,
        }
        or source["output_codec_sha256"]
        != _codec_state_sha256(output_codec)
    ):
        raise ValueError("rotation source-projection binding is invalid")
    expected_equivalence = _codec_prefix_projector_equivalence(
        output_codec.decoder,
        _codec_prefix_normal(output_codec.decoder),
    )
    if (
        not _semantic_numeric_equal(
            source["codec_prefix_projector_equivalence"],
            expected_equivalence,
        )
        or expected_equivalence["passed"] is not True
    ):
        raise ValueError(
            "rotation codec-prefix projector control does not recompute"
        )
    disjointness_fields = {
        "fresh_prompt_sha256",
        "projection_prompt_sha256",
        "weighted_prompt_sha256",
        "gated_prompt_sha256",
        "fresh_count",
        "projection_count",
        "weighted_count",
        "gated_count",
        "projection_overlap_count",
        "weighted_overlap_count",
        "gated_overlap_count",
        "verified_before_model_load_or_tokenization",
    }
    if (
        not isinstance(disjointness, Mapping)
        or set(disjointness) != disjointness_fields
    ):
        raise ValueError("rotation prompt-disjointness fields are invalid")
    source_hash_sets: dict[str, set[str]] = {}
    for label in ("fresh", "projection", "weighted", "gated"):
        values = disjointness[f"{label}_prompt_sha256"]
        if (
            not isinstance(values, tuple)
            or not values
            or any(not _is_sha256(item) for item in values)
            or len(set(values)) != len(values)
            or disjointness[f"{label}_count"] != len(values)
        ):
            raise ValueError("rotation prompt-disjointness hashes are invalid")
        source_hash_sets[label] = set(values)
    if (
        source_hash_sets["fresh"] != fresh_hashes
        or any(
            source_hash_sets["fresh"] & source_hash_sets[label]
            for label in ("projection", "weighted", "gated")
        )
        or any(
            disjointness[f"{label}_overlap_count"] != 0
            for label in ("projection", "weighted", "gated")
        )
        or disjointness[
            "verified_before_model_load_or_tokenization"
        ]
        is not True
    ):
        raise ValueError("rotation prompt disjointness is invalid")

    rotated_projector, codec_projector, sensitivity_summary = (
        _validate_sensitivity_payload(
            sensitivity,
            codec=output_codec,
            tail_width=tail_width,
            stream=streams["calibration_a"],
            stability_policy=stability_policy,
            minimum_split_half_alignment=minimum_alignment,
            minimum_relative_eigengap=minimum_eigengap,
            minimum_split_half_operator_cosine=(
                minimum_operator_cosine
            ),
            maximum_split_half_relative_regret=(
                maximum_relative_regret
            ),
        )
    )
    if sensitivity_summary["sensitivity_fit_stable"] is not True:
        raise ValueError("stored rotation sensitivity fit is unstable")

    selection_fields = {
        "candidate_direct_diagnostics",
        "candidate_behavior",
        "full_width_identity",
        "lock",
        "tokenized_stream",
    }
    if set(selection) != selection_fields:
        raise ValueError("codimension-rotation selection fields are invalid")
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    direct = selection["candidate_direct_diagnostics"]
    behavior = selection["candidate_behavior"]
    lock = selection["lock"]
    full_width = selection["full_width_identity"]
    if (
        not isinstance(direct, Mapping)
        or tuple(direct) != candidate_ids
        or not isinstance(behavior, Mapping)
        or tuple(behavior) != candidate_ids
        or not isinstance(lock, Mapping)
        or not isinstance(full_width, Mapping)
        or selection["tokenized_stream"] != streams["calibration_b"]
    ):
        raise ValueError("codimension-rotation selection mappings are invalid")
    for candidate in candidates:
        _recompute_rotation_direct(
            direct[candidate.candidate_id],
            stream=streams["calibration_b"],
            candidate=candidate,
        )
        _recompute_rotation_behavior(
            behavior[candidate.candidate_id],
            stream=streams["calibration_b"],
        )
    _validate_shared_candidate_native_values(
        candidate_ids,
        direct=direct,  # type: ignore[arg-type]
        behavior=behavior,  # type: ignore[arg-type]
    )
    expected_ledger = _candidate_ledger(
        candidates,
        direct=direct,
        behavior=behavior,
        nll_atol=nll_atol,
        top1_min=top1_min,
    )
    expected_locked, expected_lock = _lock_candidate(
        candidates,
        expected_ledger,
    )
    if lock != expected_lock:
        raise ValueError("codimension-rotation selection lock does not recompute")
    identity_candidate = candidates[-1]
    if (
        set(full_width) != {"candidate", "direct", "behavior", "passed"}
        or full_width["candidate"] != identity_candidate.metadata()
        or full_width["direct"] != direct[identity_candidate.candidate_id]
        or full_width["behavior"]
        != behavior[identity_candidate.candidate_id]
        or full_width["passed"] is not True
        or not _full_width_identity_passed(
            direct=full_width["direct"],  # type: ignore[arg-type]
            behavior=full_width["behavior"],  # type: ignore[arg-type]
            identity_nll_atol=identity_tolerance,
        )
    ):
        raise ValueError(
            "codimension-rotation full-width identity does not recompute"
        )

    validation_fields = {
        "evaluated",
        "reason",
        "locked_candidate",
        "direct_diagnostic",
        "behavior",
        "behavior_gates",
        "behavior_fidelity_passed",
        "rank_639_fidelity_viable",
        "sensitivity_fit_stable",
        "balanced_candidate_pareto_dominates_codec",
        "basis_ordering_supported",
        "meaningful_rank_compression",
        "tokenized_stream",
    }
    if (
        set(validation) != validation_fields
        or validation["evaluated"] is not status_validation
        or validation["locked_candidate"] != expected_locked.metadata()
        or validation["sensitivity_fit_stable"] is not True
        or validation["balanced_candidate_pareto_dominates_codec"]
        is not sensitivity_summary[
            "balanced_candidate_pareto_dominates_codec"
        ]
    ):
        raise ValueError("codimension-rotation validation aliases are invalid")
    locked_b_row = next(
        row
        for row in expected_ledger
        if row["candidate"]["candidate_id"]  # type: ignore[index]
        == expected_locked.candidate_id
    )
    locked_b_passed = locked_b_row["behavior_fidelity_passed"] is True
    reduced = expected_locked.retained_rank < width
    if status_validation:
        validation_stream = streams["validation"]
        validation_behavior = validation["behavior"]
        if (
            expected_lock["selection_failed"] is not False
            or not reduced
            or validation["reason"]
            != "one_locked_reduced_candidate_evaluated"
            or validation["tokenized_stream"] != validation_stream
            or not isinstance(validation_behavior, Mapping)
        ):
            raise ValueError(
                "codimension-rotation validation lock is invalid"
            )
        _recompute_rotation_direct(
            validation["direct_diagnostic"],
            stream=validation_stream,
            candidate=expected_locked,
        )
        _recompute_rotation_behavior(
            validation_behavior,
            stream=validation_stream,
        )
        expected_validation_gates = _behavior_gate(
            validation_behavior,
            nll_atol=nll_atol,
            top1_min=top1_min,
        )
        validation_passed = all(expected_validation_gates.values())
        if (
            validation["behavior_gates"] != expected_validation_gates
            or validation["behavior_fidelity_passed"]
            is not validation_passed
        ):
            raise ValueError(
                "codimension-rotation validation gates do not recompute"
            )
    else:
        validation_passed = False
        if (
            expected_lock["selection_failed"] is not True
            or reduced
            or validation["reason"]
            != "no_reduced_candidate_passed_calibration_b_"
            "validation_not_tokenized"
            or validation["direct_diagnostic"] is not None
            or validation["behavior"] is not None
            or validation["behavior_gates"] is not None
            or validation["behavior_fidelity_passed"] is not False
            or validation["tokenized_stream"] is not None
        ):
            raise ValueError(
                "unevaluated rotation validation payload is invalid"
            )
    rank_viable = (
        status_validation
        and reduced
        and locked_b_passed
        and validation_passed
    )
    meaningful = (
        rank_viable
        and expected_locked.retained_fraction <= meaningful_fraction
    )
    rotated_b_passed = (
        expected_ledger[0]["behavior_fidelity_passed"] is True
    )
    codec_b_passed = (
        expected_ledger[1]["behavior_fidelity_passed"] is True
    )
    pareto = (
        sensitivity_summary[
            "balanced_candidate_pareto_dominates_codec"
        ]
        is True
    )
    basis_ordering_supported = (
        expected_locked.normal_source
        == "calibration_a_balanced_tail_rotation"
        and rotated_b_passed
        and not codec_b_passed
        and validation_passed
        and pareto
    )
    if (
        validation["rank_639_fidelity_viable"] is not rank_viable
        or validation["basis_ordering_supported"]
        is not basis_ordering_supported
        or validation["meaningful_rank_compression"] is not meaningful
    ):
        raise ValueError(
            "codimension-rotation validation conclusions do not recompute"
        )

    expected_status = {
        "scope": (
            "target_informed_codimension_one_tail_span_search_"
            "with_rotation_discriminator"
        ),
        "calibration_a_sensitivity_fitted": True,
        "calibration_b_candidates_evaluated": True,
        "validation_locked_before_evaluation": True,
        "validation_evaluated": status_validation,
        "locked_validation_interventions_per_batch": (
            1 if status_validation else 0
        ),
        "test_evaluated": False,
        "model_weights_changed": False,
        "model_weights_in_artifact": False,
        "prompt_text_in_artifact": False,
        "inference_executor": False,
        "behavioral_upper_bound_claim": False,
        "compression_claim": False,
        "parameter_mac_speed_claim": False,
        "selection_failed": expected_lock["selection_failed"],
        "sensitivity_fit_stable": True,
        "balanced_candidate_pareto_dominates_codec": pareto,
        "rank_639_fidelity_viable": rank_viable,
        "basis_ordering_supported": basis_ordering_supported,
        "meaningful_rank_compression": meaningful,
    }
    if status != expected_status:
        raise ValueError("codimension-rotation scientific status is invalid")

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
            "codimension-rotation JSON report does not match payload"
        )
    return {
        "model": model,
        "output_codec": output_codec,
        "rotated_projector": rotated_projector,
        "codec_prefix_projector": codec_projector,
        "locked_candidate": expected_locked.metadata(),
        "selection": copy.deepcopy(dict(selection)),
        "validation": copy.deepcopy(dict(validation)),
        "calibration_a_sensitivity": copy.deepcopy(dict(sensitivity)),
        "metadata": {
            "scientific_payload_sha256": digest,
            "report_sha256": raw["report_sha256"],
            "source_projection": copy.deepcopy(dict(source)),
            "protocol": copy.deepcopy(dict(protocol)),
        },
        "report": copy.deepcopy(dict(report)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test two preregistered codimension-one spans in a frozen "
            "Gemma block, including a fresh sensitivity rotation."
        )
    )
    parser.add_argument(
        "--projection-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        default=DEFAULT_PROMPT_SPLITS,
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tokenization-batch-size", type=int, default=4)
    parser.add_argument(
        "--tail-width",
        type=int,
        default=DEFAULT_TAIL_WIDTH,
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
        "--identity-nll-atol",
        type=float,
        default=DEFAULT_IDENTITY_NLL_ATOL,
    )
    parser.add_argument(
        "--max-meaningful-retained-fraction",
        type=float,
        default=DEFAULT_MAX_MEANINGFUL_RETAINED_FRACTION,
    )
    parser.add_argument(
        "--minimum-split-half-alignment",
        type=float,
        default=DEFAULT_MIN_SPLIT_HALF_ALIGNMENT,
    )
    parser.add_argument(
        "--minimum-relative-eigengap",
        type=float,
        default=DEFAULT_MIN_RELATIVE_EIGENGAP,
    )
    parser.add_argument(
        "--stability-policy",
        choices=sorted(_STABILITY_POLICIES),
        default=DEFAULT_STABILITY_POLICY,
    )
    parser.add_argument(
        "--minimum-split-half-operator-cosine",
        type=float,
        default=DEFAULT_MIN_SPLIT_HALF_OPERATOR_COSINE,
    )
    parser.add_argument(
        "--maximum-split-half-relative-regret",
        type=float,
        default=DEFAULT_MAX_SPLIT_HALF_RELATIVE_REGRET,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_codimension_rotation(
        projection_artifact_path=arguments.projection_artifact,
        model_id=arguments.model,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        prompt_splits_path=arguments.prompt_splits,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        tail_width=arguments.tail_width,
        selection_nll_atol=arguments.selection_nll_atol,
        selection_top1_min=arguments.selection_top1_min,
        identity_nll_atol=arguments.identity_nll_atol,
        max_meaningful_retained_fraction=(
            arguments.max_meaningful_retained_fraction
        ),
        minimum_split_half_alignment=(
            arguments.minimum_split_half_alignment
        ),
        minimum_relative_eigengap=(
            arguments.minimum_relative_eigengap
        ),
        stability_policy=arguments.stability_policy,
        minimum_split_half_operator_cosine=(
            arguments.minimum_split_half_operator_cosine
        ),
        maximum_split_half_relative_regret=(
            arguments.maximum_split_half_relative_regret
        ),
        device_name=arguments.device,
        dtype=arguments.dtype,
        local_files_only=arguments.local_files_only,
        output=arguments.output,
    )
    status = report["scientific_status"]
    selection = report["analysis"]["selection"]  # type: ignore[index]
    validation = report["analysis"]["validation"]  # type: ignore[index]
    assert isinstance(status, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(validation, Mapping)
    behavior = validation["behavior"]
    print(
        json.dumps(
            {
                "output": report["artifact"]["tensor_output"],  # type: ignore[index]
                "selection_failed": status["selection_failed"],
                "locked_candidate": validation["locked_candidate"],
                "validation_evaluated": validation["evaluated"],
                "delta_nll_per_token": (
                    None
                    if not isinstance(behavior, Mapping)
                    else behavior["delta_nll_per_token"]
                ),
                "top1_agreement": (
                    None
                    if not isinstance(behavior, Mapping)
                    else behavior["top1_agreement_to_baseline"]
                ),
                "rank_639_fidelity_viable": status[
                    "rank_639_fidelity_viable"
                ],
                "basis_ordering_supported": status[
                    "basis_ordering_supported"
                ],
                "meaningful_rank_compression": status[
                    "meaningful_rank_compression"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
