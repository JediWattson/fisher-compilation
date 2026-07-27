"""Source-safe causal fingerprints for modal-generator interventions.

This module compares a baseline execution with executions in which exactly one
modal generator is muted.  Every condition is evaluated in the same output
vocabulary, so the resulting effects have a common downstream coordinate
system even when the generators originate in different transformer layers.

For each supervised token, the bounded shared frame contains the target
coordinate followed by the baseline's highest-logit non-target coordinates.
Muted-minus-baseline effects are centered within that frame.  Collection keeps
only prompt-level scalar signatures and their aggregate cross-generator Gram
matrix; prompt text, raw example identifiers, token ids, targets, and logits
are never retained in the authenticated artifact.

The pair labels produced here are observational hypotheses.  They are useful
for deciding which generator relationships deserve a later intervention, but
they never authorize merging, pruning, routing, compilation, or mutation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor


__all__ = [
    "GeneratorCausalFingerprintAccumulator",
    "GeneratorCausalFingerprintAnalysis",
    "GeneratorCausalFingerprintProvenance",
    "GeneratorCausalPromptSignature",
    "GeneratorInterventionLogitBatch",
    "GeneratorPairSimilarity",
    "ObservationalFamilyPolicy",
    "collect_generator_causal_fingerprints",
    "generator_fingerprint_example_id_sha256",
]


_ARTIFACT_KIND = "fisher_graph.modal_generator_causal_fingerprints"
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = b"fisher_graph.generator_causal_fingerprint.artifact.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.generator_causal_fingerprint.tensor.v1\0"
_EXAMPLE_ID_DOMAIN = (
    b"fisher_graph.generator_causal_fingerprint.example_id.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SHARED_FRAME = (
    "per_supervised_token_target_then_stable_baseline_top_non_target_logits"
)
_EFFECT_CENTERING = "per_supervised_token_anchor_mean"
_GRAM_WEIGHTING = "equal_prompt_mean_over_supervised_anchor_coordinates"
_INTERVENTION = "exactly_one_generator_muted_against_shared_baseline"
_VOCABULARY_CHUNK = 4096

_SAFETY_METADATA: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_generator_weights": False,
    "contains_prompt_text": False,
    "contains_raw_example_ids": False,
    "contains_token_ids": False,
    "contains_targets": False,
    "contains_raw_logits": False,
    "contains_token_level_effect_rows": False,
    "analysis_only": True,
    "observational_hypotheses_only": True,
    "authorizes_intervention": False,
    "authorizes_merge": False,
    "authorizes_pruning": False,
    "authorizes_routing": False,
    "authorizes_compilation": False,
    "authorizes_execution": False,
    "authorizes_mutation": False,
}

_TENSOR_FIELDS = (
    "supervised_token_counts",
    "prompt_nll_effects",
    "prompt_baseline_to_muted_kls",
    "prompt_top1_agreements",
    "prompt_centered_anchor_effect_rms",
    "centered_shared_effect_gram",
)

_HYPOTHESIS_LABELS = frozenset(
    {
        "aligned_observational_family_hypothesis",
        "mixed_observational_family_evidence",
        "distinct_observational_effect_hypothesis",
        "insufficient_causal_variation",
    }
)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    digest = hashlib.sha256()
    digest.update(_ARTIFACT_DOMAIN)
    digest.update(_json_bytes(value))
    return digest.hexdigest()


def _tensor_sha256(value: Tensor, *, label: str) -> str:
    if not isinstance(value, Tensor) or value.device.type != "cpu":
        raise ValueError(f"{label} must be a CPU Tensor")
    if value.dtype not in (torch.float64, torch.int64):
        raise ValueError(f"{label} must use float64 or int64")
    if value.dtype == torch.float64 and not torch.isfinite(value).all():
        raise ValueError(f"{label} must be finite")
    canonical = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        f"{tuple(canonical.shape)}\0{canonical.dtype}\0".encode("utf-8")
    )
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def generator_fingerprint_example_id_sha256(example_id: str) -> str:
    """Hash an ephemeral example id for a source-safe artifact."""

    value = _require_nonempty(example_id, label="example_id")
    digest = hashlib.sha256()
    digest.update(_EXAMPLE_ID_DOMAIN)
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _canonical_generator_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("generator_ids must be a sequence")
    result = tuple(values)
    if len(result) < 2:
        raise ValueError("at least two generator ids are required")
    for index, value in enumerate(result):
        _require_nonempty(value, label=f"generator_ids[{index}]")
    if len(set(result)) != len(result):
        raise ValueError("generator ids must be unique")
    return result


def _require_probability_threshold(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be a finite float")
    if value < -1.0 or value > 1.0:
        raise ValueError(f"{label} must be between -1 and 1")
    return value


@dataclass(frozen=True, slots=True)
class ObservationalFamilyPolicy:
    """Thresholds used only to label observational pair evidence."""

    minimum_centered_effect_cosine: float = 0.90
    minimum_prompt_nll_spearman: float = 0.80
    minimum_top_importance_overlap: float = 0.60
    minimum_top_importance_sign_agreement: float = 0.80
    minimum_prompt_count: int = 3

    def __post_init__(self) -> None:
        _require_probability_threshold(
            self.minimum_centered_effect_cosine,
            label="minimum_centered_effect_cosine",
        )
        _require_probability_threshold(
            self.minimum_prompt_nll_spearman,
            label="minimum_prompt_nll_spearman",
        )
        for name in (
            "minimum_top_importance_overlap",
            "minimum_top_importance_sign_agreement",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite float")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if (
            type(self.minimum_prompt_count) is not int
            or self.minimum_prompt_count < 2
        ):
            raise ValueError("minimum_prompt_count must be an integer >= 2")

    def metadata(self) -> dict[str, object]:
        return {
            "minimum_centered_effect_cosine": (
                self.minimum_centered_effect_cosine
            ),
            "minimum_prompt_nll_spearman": (
                self.minimum_prompt_nll_spearman
            ),
            "minimum_top_importance_overlap": (
                self.minimum_top_importance_overlap
            ),
            "minimum_top_importance_sign_agreement": (
                self.minimum_top_importance_sign_agreement
            ),
            "minimum_prompt_count": self.minimum_prompt_count,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ObservationalFamilyPolicy:
        fields = {
            "minimum_centered_effect_cosine",
            "minimum_prompt_nll_spearman",
            "minimum_top_importance_overlap",
            "minimum_top_importance_sign_agreement",
            "minimum_prompt_count",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError("observational family policy fields are invalid")
        return cls(
            minimum_centered_effect_cosine=state[  # type: ignore[arg-type]
                "minimum_centered_effect_cosine"
            ],
            minimum_prompt_nll_spearman=state[  # type: ignore[arg-type]
                "minimum_prompt_nll_spearman"
            ],
            minimum_top_importance_overlap=state[  # type: ignore[arg-type]
                "minimum_top_importance_overlap"
            ],
            minimum_top_importance_sign_agreement=state[  # type: ignore[arg-type]
                "minimum_top_importance_sign_agreement"
            ],
            minimum_prompt_count=state[  # type: ignore[arg-type]
                "minimum_prompt_count"
            ],
        )


@dataclass(frozen=True, slots=True)
class GeneratorCausalFingerprintProvenance:
    """Authenticated sources for one non-destructive fingerprint run."""

    source_model_sha256: str
    generator_catalog_sha256: str
    evaluation_split_sha256: str
    objective_sha256: str
    intervention: str = _INTERVENTION

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_model_sha256,
            label="source_model_sha256",
        )
        _require_sha256(
            self.generator_catalog_sha256,
            label="generator_catalog_sha256",
        )
        _require_sha256(
            self.evaluation_split_sha256,
            label="evaluation_split_sha256",
        )
        _require_sha256(self.objective_sha256, label="objective_sha256")
        if self.intervention != _INTERVENTION:
            raise ValueError("fingerprint intervention semantics drifted")

    def metadata(self) -> dict[str, object]:
        return {
            "source_model_sha256": self.source_model_sha256,
            "generator_catalog_sha256": self.generator_catalog_sha256,
            "evaluation_split_sha256": self.evaluation_split_sha256,
            "objective_sha256": self.objective_sha256,
            "intervention": self.intervention,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> GeneratorCausalFingerprintProvenance:
        fields = {
            "source_model_sha256",
            "generator_catalog_sha256",
            "evaluation_split_sha256",
            "objective_sha256",
            "intervention",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError("causal fingerprint provenance fields are invalid")
        return cls(
            source_model_sha256=state[  # type: ignore[arg-type]
                "source_model_sha256"
            ],
            generator_catalog_sha256=state[  # type: ignore[arg-type]
                "generator_catalog_sha256"
            ],
            evaluation_split_sha256=state[  # type: ignore[arg-type]
                "evaluation_split_sha256"
            ],
            objective_sha256=state["objective_sha256"],  # type: ignore[arg-type]
            intervention=state["intervention"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class GeneratorInterventionLogitBatch:
    """Ephemeral aligned logits for one batch of causal interventions."""

    example_ids: tuple[str, ...]
    baseline_logits: Tensor
    muted_logits_by_generator: Mapping[str, Tensor]
    targets: Tensor
    supervised_mask: Tensor

    def __post_init__(self) -> None:
        if type(self.example_ids) is not tuple:
            raise TypeError("example_ids must be a tuple")
        if not self.example_ids:
            raise ValueError("an intervention batch cannot be empty")
        for index, value in enumerate(self.example_ids):
            _require_nonempty(value, label=f"example_ids[{index}]")
        if len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError("example ids must be unique within a batch")

        logits = self.baseline_logits
        if (
            not isinstance(logits, Tensor)
            or logits.ndim != 3
            or not logits.dtype.is_floating_point
            or logits.shape[0] != len(self.example_ids)
            or logits.shape[1] <= 0
            or logits.shape[2] <= 1
            or not torch.isfinite(logits).all()
        ):
            raise ValueError(
                "baseline_logits must be a finite floating Tensor with "
                "shape [examples, positions, vocabulary]"
            )
        if (
            not isinstance(self.targets, Tensor)
            or self.targets.dtype != torch.int64
            or self.targets.shape != logits.shape[:2]
        ):
            raise ValueError(
                "targets must be an int64 Tensor matching the first two "
                "logit dimensions"
            )
        if (
            not isinstance(self.supervised_mask, Tensor)
            or self.supervised_mask.dtype != torch.bool
            or self.supervised_mask.shape != logits.shape[:2]
        ):
            raise ValueError(
                "supervised_mask must be a bool Tensor matching the first "
                "two logit dimensions"
            )
        if not isinstance(self.muted_logits_by_generator, Mapping):
            raise TypeError("muted_logits_by_generator must be a mapping")
        if not self.muted_logits_by_generator:
            raise ValueError("muted_logits_by_generator cannot be empty")
        for generator_id, value in self.muted_logits_by_generator.items():
            _require_nonempty(generator_id, label="muted generator id")
            if (
                not isinstance(value, Tensor)
                or value.shape != logits.shape
                or value.dtype != logits.dtype
                or value.device != logits.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"muted logits for {generator_id!r} must be finite "
                    "logits with the baseline shape, dtype, and device"
                )


@dataclass(frozen=True, slots=True)
class GeneratorCausalPromptSignature:
    """Finite prompt-level effects for one muted generator."""

    generator_id: str
    muted_minus_baseline_nll: Tensor
    baseline_to_muted_kl: Tensor
    top1_agreement: Tensor
    centered_anchor_logit_effect_rms: Tensor

    def __post_init__(self) -> None:
        _require_nonempty(self.generator_id, label="generator_id")
        prompt_count: int | None = None
        for name in (
            "muted_minus_baseline_nll",
            "baseline_to_muted_kl",
            "top1_agreement",
            "centered_anchor_logit_effect_rms",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or value.ndim != 1
                or value.numel() == 0
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"{name} must be a nonempty finite CPU float64 vector"
                )
            if prompt_count is None:
                prompt_count = value.numel()
            elif value.numel() != prompt_count:
                raise ValueError("prompt signature vector lengths differ")
            object.__setattr__(
                self,
                name,
                value.detach().clone().contiguous(),
            )
        if (self.baseline_to_muted_kl < 0).any():
            raise ValueError("baseline_to_muted_kl must be nonnegative")
        if (
            (self.top1_agreement < 0).any()
            or (self.top1_agreement > 1).any()
        ):
            raise ValueError("top1_agreement must be between zero and one")
        if (self.centered_anchor_logit_effect_rms < 0).any():
            raise ValueError(
                "centered_anchor_logit_effect_rms must be nonnegative"
            )

    @property
    def prompt_count(self) -> int:
        return self.muted_minus_baseline_nll.numel()

    def metadata(self) -> dict[str, object]:
        return {
            "generator_id": self.generator_id,
            "prompt_count": self.prompt_count,
            "mean_muted_minus_baseline_nll": float(
                self.muted_minus_baseline_nll.mean().item()
            ),
            "mean_baseline_to_muted_kl": float(
                self.baseline_to_muted_kl.mean().item()
            ),
            "mean_top1_agreement": float(
                self.top1_agreement.mean().item()
            ),
            "root_mean_square_centered_anchor_logit_effect": float(
                self.centered_anchor_logit_effect_rms.square()
                .mean()
                .sqrt()
                .item()
            ),
        }


@dataclass(frozen=True, slots=True)
class GeneratorPairSimilarity:
    """Observational similarity between two causal generator effects."""

    generator_a: str
    generator_b: str
    centered_shared_logit_effect_cosine: float
    prompt_nll_effect_spearman: float
    top_importance_overlap: float
    top_importance_sign_agreement: float
    top_importance_intersection_count: int
    sufficient_causal_variation: bool
    observational_hypothesis: str
    observational_only: bool = True
    authorizes_merge: bool = False
    authorizes_pruning: bool = False
    authorizes_routing: bool = False
    authorizes_mutation: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.generator_a, label="generator_a")
        _require_nonempty(self.generator_b, label="generator_b")
        if self.generator_a == self.generator_b:
            raise ValueError("a pair must contain distinct generators")
        for name in (
            "centered_shared_logit_effect_cosine",
            "prompt_nll_effect_spearman",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite float")
            if value < -1.0 or value > 1.0:
                raise ValueError(f"{name} must be between -1 and 1")
        for name in (
            "top_importance_overlap",
            "top_importance_sign_agreement",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite float")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if (
            type(self.top_importance_intersection_count) is not int
            or self.top_importance_intersection_count < 0
        ):
            raise ValueError(
                "top_importance_intersection_count must be nonnegative"
            )
        if type(self.sufficient_causal_variation) is not bool:
            raise TypeError("sufficient_causal_variation must be a bool")
        if self.observational_hypothesis not in _HYPOTHESIS_LABELS:
            raise ValueError("unknown observational hypothesis")
        if (
            self.observational_only is not True
            or self.authorizes_merge is not False
            or self.authorizes_pruning is not False
            or self.authorizes_routing is not False
            or self.authorizes_mutation is not False
        ):
            raise ValueError("pair authority metadata is invalid")

    def metadata(self) -> dict[str, object]:
        return {
            "generator_a": self.generator_a,
            "generator_b": self.generator_b,
            "centered_shared_logit_effect_cosine": (
                self.centered_shared_logit_effect_cosine
            ),
            "prompt_nll_effect_spearman": self.prompt_nll_effect_spearman,
            "top_importance_overlap": self.top_importance_overlap,
            "top_importance_sign_agreement": (
                self.top_importance_sign_agreement
            ),
            "top_importance_intersection_count": (
                self.top_importance_intersection_count
            ),
            "sufficient_causal_variation": self.sufficient_causal_variation,
            "observational_hypothesis": self.observational_hypothesis,
            "observational_only": self.observational_only,
            "authorizes_merge": self.authorizes_merge,
            "authorizes_pruning": self.authorizes_pruning,
            "authorizes_routing": self.authorizes_routing,
            "authorizes_mutation": self.authorizes_mutation,
        }


def _average_ranks(values: Tensor) -> Tensor:
    """Return deterministic average ranks with exact-value tie groups."""

    if values.ndim != 1 or values.dtype != torch.float64:
        raise ValueError("rank inputs must be float64 vectors")
    order = sorted(
        range(values.numel()),
        key=lambda index: (float(values[index].item()), index),
    )
    ranks = torch.empty_like(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        first = float(values[order[cursor]].item())
        while (
            end < len(order)
            and float(values[order[end]].item()) == first
        ):
            end += 1
        average = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def _centered_cosine(first: Tensor, second: Tensor) -> float:
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = float(
        (
            first_centered.square().sum()
            * second_centered.square().sum()
        )
        .sqrt()
        .item()
    )
    if denominator == 0.0:
        return 0.0
    value = float(
        (first_centered * second_centered).sum().item() / denominator
    )
    return max(-1.0, min(1.0, value))


def _spearman(first: Tensor, second: Tensor) -> float:
    return _centered_cosine(_average_ranks(first), _average_ranks(second))


def _top_importance_indices(values: Tensor, count: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(values.numel()),
            key=lambda index: (-abs(float(values[index].item())), index),
        )[:count]
    )


def _analysis_payload(
    *,
    provenance: GeneratorCausalFingerprintProvenance,
    generator_ids: tuple[str, ...],
    example_id_sha256s: tuple[str, ...],
    anchor_count: int,
    top_importance_count: int,
    policy: ObservationalFamilyPolicy,
    tensors: Mapping[str, Tensor],
) -> dict[str, object]:
    return {
        "artifact_kind": _ARTIFACT_KIND,
        "format_version": _FORMAT_VERSION,
        **_SAFETY_METADATA,
        "provenance": provenance.metadata(),
        "generator_ids": generator_ids,
        "example_id_sha256s": example_id_sha256s,
        "generator_count": len(generator_ids),
        "prompt_count": len(example_id_sha256s),
        "anchor_count": anchor_count,
        "anchor_frame_width": anchor_count + 1,
        "shared_frame": _SHARED_FRAME,
        "effect_centering": _EFFECT_CENTERING,
        "gram_weighting": _GRAM_WEIGHTING,
        "top_importance_count": top_importance_count,
        "observational_family_policy": policy.metadata(),
        "tensor_sha256s": {
            name: _tensor_sha256(value, label=name)
            for name, value in tensors.items()
        },
    }


@dataclass(frozen=True, slots=True)
class GeneratorCausalFingerprintAnalysis:
    """Authenticated prompt signatures and observational pair hypotheses."""

    provenance: GeneratorCausalFingerprintProvenance
    generator_ids: tuple[str, ...]
    example_id_sha256s: tuple[str, ...]
    anchor_count: int
    top_importance_count: int
    policy: ObservationalFamilyPolicy
    supervised_token_counts: Tensor
    prompt_nll_effects: Tensor
    prompt_baseline_to_muted_kls: Tensor
    prompt_top1_agreements: Tensor
    prompt_centered_anchor_effect_rms: Tensor
    centered_shared_effect_gram: Tensor
    artifact_sha256: str
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION
    contains_source_model_weights: bool = False
    contains_generator_weights: bool = False
    contains_prompt_text: bool = False
    contains_raw_example_ids: bool = False
    contains_token_ids: bool = False
    contains_targets: bool = False
    contains_raw_logits: bool = False
    contains_token_level_effect_rows: bool = False
    analysis_only: bool = True
    observational_hypotheses_only: bool = True
    authorizes_intervention: bool = False
    authorizes_merge: bool = False
    authorizes_pruning: bool = False
    authorizes_routing: bool = False
    authorizes_compilation: bool = False
    authorizes_execution: bool = False
    authorizes_mutation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(
            self.provenance,
            GeneratorCausalFingerprintProvenance,
        ):
            raise TypeError(
                "provenance must be GeneratorCausalFingerprintProvenance"
            )
        generator_ids = _canonical_generator_ids(self.generator_ids)
        if (
            type(self.generator_ids) is not tuple
            or generator_ids != self.generator_ids
        ):
            raise ValueError("generator_ids must be a canonical tuple")
        if type(self.example_id_sha256s) is not tuple:
            raise TypeError("example_id_sha256s must be a tuple")
        if not self.example_id_sha256s:
            raise ValueError("a causal fingerprint analysis cannot be empty")
        for index, value in enumerate(self.example_id_sha256s):
            _require_sha256(value, label=f"example_id_sha256s[{index}]")
        if len(set(self.example_id_sha256s)) != len(
            self.example_id_sha256s
        ):
            raise ValueError("example id hashes must be unique")
        if type(self.anchor_count) is not int or self.anchor_count <= 0:
            raise ValueError("anchor_count must be a positive integer")
        prompt_count = len(self.example_id_sha256s)
        if (
            type(self.top_importance_count) is not int
            or self.top_importance_count <= 0
            or self.top_importance_count > prompt_count
        ):
            raise ValueError(
                "top_importance_count must be in [1, prompt_count]"
            )
        if not isinstance(self.policy, ObservationalFamilyPolicy):
            raise TypeError("policy must be ObservationalFamilyPolicy")

        generator_count = len(generator_ids)
        if (
            not isinstance(self.supervised_token_counts, Tensor)
            or self.supervised_token_counts.device.type != "cpu"
            or self.supervised_token_counts.dtype != torch.int64
            or self.supervised_token_counts.shape != (prompt_count,)
            or (self.supervised_token_counts <= 0).any()
        ):
            raise ValueError(
                "supervised_token_counts must be a positive CPU int64 "
                "vector with one entry per prompt"
            )
        object.__setattr__(
            self,
            "supervised_token_counts",
            self.supervised_token_counts.detach().clone().contiguous(),
        )

        prompt_shape = (generator_count, prompt_count)
        for name in (
            "prompt_nll_effects",
            "prompt_baseline_to_muted_kls",
            "prompt_top1_agreements",
            "prompt_centered_anchor_effect_rms",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or value.shape != prompt_shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"{name} must be a finite CPU float64 Tensor with "
                    f"shape {prompt_shape}"
                )
            object.__setattr__(
                self,
                name,
                value.detach().clone().contiguous(),
            )
        if (self.prompt_baseline_to_muted_kls < 0).any():
            raise ValueError(
                "prompt_baseline_to_muted_kls must be nonnegative"
            )
        if (
            (self.prompt_top1_agreements < 0).any()
            or (self.prompt_top1_agreements > 1).any()
        ):
            raise ValueError(
                "prompt_top1_agreements must be between zero and one"
            )
        if (self.prompt_centered_anchor_effect_rms < 0).any():
            raise ValueError(
                "prompt_centered_anchor_effect_rms must be nonnegative"
            )

        gram = self.centered_shared_effect_gram
        gram_shape = (generator_count, generator_count)
        if (
            not isinstance(gram, Tensor)
            or gram.device.type != "cpu"
            or gram.dtype != torch.float64
            or gram.shape != gram_shape
            or not torch.isfinite(gram).all()
            or not torch.allclose(gram, gram.T, atol=1e-12, rtol=1e-12)
            or (torch.diagonal(gram) < 0).any()
        ):
            raise ValueError(
                "centered_shared_effect_gram must be a finite symmetric "
                f"CPU float64 Gram matrix with shape {gram_shape}"
            )
        object.__setattr__(
            self,
            "centered_shared_effect_gram",
            gram.detach().clone().contiguous(),
        )

        if (
            type(self.artifact_kind) is not str
            or self.artifact_kind != _ARTIFACT_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
            or any(
                getattr(self, name) is not expected
                for name, expected in _SAFETY_METADATA.items()
            )
        ):
            raise ValueError("causal fingerprint safety metadata is invalid")
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("causal fingerprint artifact hash mismatch")

    @property
    def generator_count(self) -> int:
        return len(self.generator_ids)

    @property
    def prompt_count(self) -> int:
        return len(self.example_id_sha256s)

    @property
    def anchor_frame_width(self) -> int:
        return self.anchor_count + 1

    def generator_signature(
        self,
        generator_id: str,
    ) -> GeneratorCausalPromptSignature:
        """Return a defensive prompt-signature view for one generator."""

        _require_nonempty(generator_id, label="generator_id")
        try:
            index = self.generator_ids.index(generator_id)
        except ValueError as error:
            raise KeyError(f"unknown generator id {generator_id!r}") from error
        return GeneratorCausalPromptSignature(
            generator_id=generator_id,
            muted_minus_baseline_nll=self.prompt_nll_effects[index],
            baseline_to_muted_kl=self.prompt_baseline_to_muted_kls[index],
            top1_agreement=self.prompt_top1_agreements[index],
            centered_anchor_logit_effect_rms=(
                self.prompt_centered_anchor_effect_rms[index]
            ),
        )

    def _pair_similarity(
        self,
        first: int,
        second: int,
    ) -> GeneratorPairSimilarity:
        gram = self.centered_shared_effect_gram
        denominator = math.sqrt(
            float(gram[first, first].item())
            * float(gram[second, second].item())
        )
        cosine = (
            0.0
            if denominator == 0.0
            else float(gram[first, second].item()) / denominator
        )
        cosine = max(-1.0, min(1.0, cosine))

        first_nll = self.prompt_nll_effects[first]
        second_nll = self.prompt_nll_effects[second]
        spearman = _spearman(first_nll, second_nll)
        first_top = set(
            _top_importance_indices(first_nll, self.top_importance_count)
        )
        second_top = set(
            _top_importance_indices(second_nll, self.top_importance_count)
        )
        intersection = sorted(first_top & second_top)
        overlap = len(intersection) / self.top_importance_count
        sign_agreement = (
            0.0
            if not intersection
            else sum(
                float(
                    torch.sign(first_nll[index]).item()
                    == torch.sign(second_nll[index]).item()
                )
                for index in intersection
            )
            / len(intersection)
        )
        has_nll_variation = bool(
            (first_nll != first_nll[0]).any()
            and (second_nll != second_nll[0]).any()
        )
        sufficient = bool(
            self.prompt_count >= self.policy.minimum_prompt_count
            and float(gram[first, first].item()) > 0.0
            and float(gram[second, second].item()) > 0.0
            and has_nll_variation
        )
        if not sufficient:
            hypothesis = "insufficient_causal_variation"
        else:
            passed = (
                cosine >= self.policy.minimum_centered_effect_cosine,
                spearman >= self.policy.minimum_prompt_nll_spearman,
                overlap >= self.policy.minimum_top_importance_overlap,
                sign_agreement
                >= self.policy.minimum_top_importance_sign_agreement,
            )
            if all(passed):
                hypothesis = "aligned_observational_family_hypothesis"
            elif sum(passed) >= 2:
                hypothesis = "mixed_observational_family_evidence"
            else:
                hypothesis = "distinct_observational_effect_hypothesis"
        return GeneratorPairSimilarity(
            generator_a=self.generator_ids[first],
            generator_b=self.generator_ids[second],
            centered_shared_logit_effect_cosine=cosine,
            prompt_nll_effect_spearman=spearman,
            top_importance_overlap=float(overlap),
            top_importance_sign_agreement=float(sign_agreement),
            top_importance_intersection_count=len(intersection),
            sufficient_causal_variation=sufficient,
            observational_hypothesis=hypothesis,
        )

    @property
    def pair_similarities(self) -> tuple[GeneratorPairSimilarity, ...]:
        return tuple(
            self._pair_similarity(first, second)
            for first in range(self.generator_count)
            for second in range(first + 1, self.generator_count)
        )

    def _tensors(self) -> dict[str, Tensor]:
        return {
            name: getattr(self, name)
            for name in _TENSOR_FIELDS
        }

    def _payload(self) -> dict[str, object]:
        return _analysis_payload(
            provenance=self.provenance,
            generator_ids=self.generator_ids,
            example_id_sha256s=self.example_id_sha256s,
            anchor_count=self.anchor_count,
            top_importance_count=self.top_importance_count,
            policy=self.policy,
            tensors=self._tensors(),
        )

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload())

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "generator_signatures": tuple(
                self.generator_signature(generator_id).metadata()
                for generator_id in self.generator_ids
            ),
            "pair_similarities": tuple(
                pair.metadata() for pair in self.pair_similarities
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            **{
                name: getattr(self, name)
                for name in _SAFETY_METADATA
            },
            "provenance": self.provenance.state_dict(),
            "generator_ids": self.generator_ids,
            "example_id_sha256s": self.example_id_sha256s,
            "anchor_count": self.anchor_count,
            "top_importance_count": self.top_importance_count,
            "policy": self.policy.state_dict(),
            **{
                name: value.detach().clone()
                for name, value in self._tensors().items()
            },
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> GeneratorCausalFingerprintAnalysis:
        fields = {
            "artifact_kind",
            "format_version",
            *_SAFETY_METADATA,
            "provenance",
            "generator_ids",
            "example_id_sha256s",
            "anchor_count",
            "top_importance_count",
            "policy",
            *_TENSOR_FIELDS,
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError("causal fingerprint artifact fields are invalid")
        raw_provenance = state["provenance"]
        raw_policy = state["policy"]
        raw_generator_ids = state["generator_ids"]
        raw_example_ids = state["example_id_sha256s"]
        if not isinstance(raw_provenance, Mapping):
            raise TypeError("causal fingerprint provenance must be a mapping")
        if not isinstance(raw_policy, Mapping):
            raise TypeError("causal fingerprint policy must be a mapping")
        if type(raw_generator_ids) is not tuple:
            raise TypeError("causal fingerprint generator_ids must be a tuple")
        if type(raw_example_ids) is not tuple:
            raise TypeError(
                "causal fingerprint example_id_sha256s must be a tuple"
            )
        tensors: dict[str, Tensor] = {}
        for name in _TENSOR_FIELDS:
            value = state[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            tensors[name] = value
        return cls(
            provenance=GeneratorCausalFingerprintProvenance.from_state_dict(
                raw_provenance
            ),
            generator_ids=raw_generator_ids,  # type: ignore[arg-type]
            example_id_sha256s=raw_example_ids,  # type: ignore[arg-type]
            anchor_count=state["anchor_count"],  # type: ignore[arg-type]
            top_importance_count=state[  # type: ignore[arg-type]
                "top_importance_count"
            ],
            policy=ObservationalFamilyPolicy.from_state_dict(raw_policy),
            **tensors,
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                name: state[name]
                for name in _SAFETY_METADATA
            },  # type: ignore[arg-type]
        )


def _create_analysis(
    *,
    provenance: GeneratorCausalFingerprintProvenance,
    generator_ids: tuple[str, ...],
    example_id_sha256s: tuple[str, ...],
    anchor_count: int,
    top_importance_count: int,
    policy: ObservationalFamilyPolicy,
    supervised_token_counts: Tensor,
    prompt_nll_effects: Tensor,
    prompt_baseline_to_muted_kls: Tensor,
    prompt_top1_agreements: Tensor,
    prompt_centered_anchor_effect_rms: Tensor,
    centered_shared_effect_gram: Tensor,
) -> GeneratorCausalFingerprintAnalysis:
    tensors = {
        "supervised_token_counts": supervised_token_counts,
        "prompt_nll_effects": prompt_nll_effects,
        "prompt_baseline_to_muted_kls": prompt_baseline_to_muted_kls,
        "prompt_top1_agreements": prompt_top1_agreements,
        "prompt_centered_anchor_effect_rms": (
            prompt_centered_anchor_effect_rms
        ),
        "centered_shared_effect_gram": centered_shared_effect_gram,
    }
    payload = _analysis_payload(
        provenance=provenance,
        generator_ids=generator_ids,
        example_id_sha256s=example_id_sha256s,
        anchor_count=anchor_count,
        top_importance_count=top_importance_count,
        policy=policy,
        tensors=tensors,
    )
    return GeneratorCausalFingerprintAnalysis(
        provenance=provenance,
        generator_ids=generator_ids,
        example_id_sha256s=example_id_sha256s,
        anchor_count=anchor_count,
        top_importance_count=top_importance_count,
        policy=policy,
        **tensors,
        artifact_sha256=_json_sha256(payload),
    )


@dataclass(slots=True)
class _ActiveFingerprintBatch:
    """The one bounded, non-serialized batch retained by an accumulator."""

    example_id_sha256s: tuple[str, ...]
    source_shape: tuple[int, int, int]
    source_device: torch.device
    source_dtype: torch.dtype
    supervised_row_indices: Tensor
    prompt_token_counts: Tensor
    prompt_offsets: tuple[int, ...]
    baseline_log_probabilities: Tensor
    anchors: Tensor
    baseline_anchor_logits: Tensor
    baseline_top1: Tensor
    baseline_prompt_nll: Tensor
    prompt_nll_effects: dict[str, Tensor]
    prompt_kls: dict[str, Tensor]
    prompt_agreements: dict[str, Tensor]
    prompt_rms: dict[str, Tensor]
    centered_anchor_effects: dict[str, Tensor]


class GeneratorCausalFingerprintAccumulator:
    """Memory-bounded state machine for causal fingerprint collection.

    ``begin_batch`` retains one supervised-row baseline log-probability
    tensor.  ``add_muted_generator`` consumes exactly one full-vocabulary
    muted condition and stores only prompt scalars plus centered bounded
    anchor effects.  The muted tensor is never attached to accumulator state.
    After every declared generator has been consumed, ``finish_batch`` drops
    the baseline and bounded row effects, retaining only final prompt scalars
    and one small Gram contribution per prompt.

    Any error while an active batch is being consumed discards that batch so
    a large baseline cannot remain pinned accidentally.  Previously completed
    batches remain intact and collection may continue with a fresh batch.
    """

    def __init__(
        self,
        *,
        generator_ids: Sequence[str],
        provenance: GeneratorCausalFingerprintProvenance,
        anchor_count: int = 8,
        top_importance_count: int = 5,
        policy: ObservationalFamilyPolicy | None = None,
    ) -> None:
        self.generator_ids = _canonical_generator_ids(generator_ids)
        if not isinstance(provenance, GeneratorCausalFingerprintProvenance):
            raise TypeError(
                "provenance must be GeneratorCausalFingerprintProvenance"
            )
        if type(anchor_count) is not int or anchor_count <= 0:
            raise ValueError("anchor_count must be a positive integer")
        if (
            type(top_importance_count) is not int
            or top_importance_count <= 0
        ):
            raise ValueError(
                "top_importance_count must be a positive integer"
            )
        if policy is None:
            policy = ObservationalFamilyPolicy()
        if not isinstance(policy, ObservationalFamilyPolicy):
            raise TypeError("policy must be ObservationalFamilyPolicy")
        self.provenance = provenance
        self.anchor_count = anchor_count
        self.top_importance_count = top_importance_count
        self.policy = policy
        self._active: _ActiveFingerprintBatch | None = None
        self._vocabulary_size: int | None = None
        self._example_hashes: list[str] = []
        self._seen_example_hashes: set[str] = set()
        self._token_counts: list[int] = []
        self._nll_rows: list[Tensor] = []
        self._kl_rows: list[Tensor] = []
        self._agreement_rows: list[Tensor] = []
        self._rms_rows: list[Tensor] = []
        self._gram_rows: list[Tensor] = []
        self._finalized_analysis: (
            GeneratorCausalFingerprintAnalysis | None
        ) = None

    def __enter__(self) -> GeneratorCausalFingerprintAccumulator:
        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        self.close()

    @property
    def has_active_batch(self) -> bool:
        return self._active is not None

    @property
    def active_received_generator_ids(self) -> tuple[str, ...]:
        if self._active is None:
            return ()
        return tuple(
            generator_id
            for generator_id in self.generator_ids
            if generator_id in self._active.prompt_nll_effects
        )

    @property
    def active_baseline_full_vocabulary_tensor_count(self) -> int:
        """Return one exactly while a baseline batch is active."""

        return int(self._active is not None)

    @property
    def active_baseline_device(self) -> torch.device | None:
        if self._active is None:
            return None
        return self._active.source_device

    @property
    def active_baseline_dtype(self) -> torch.dtype | None:
        if self._active is None:
            return None
        return self._active.source_dtype

    @property
    def active_muted_full_vocabulary_tensor_count(self) -> int:
        """Muted full-vocabulary tensors are never retained."""

        return 0

    @property
    def completed_prompt_count(self) -> int:
        return len(self._nll_rows)

    def _ensure_not_finalized(self) -> None:
        if self._finalized_analysis is not None:
            raise RuntimeError("causal fingerprint accumulator is finalized")

    def abort_batch(self) -> None:
        """Discard the active baseline and all unfinished bounded effects."""

        self._active = None

    def close(self) -> None:
        """Release an unfinished active batch without changing completed rows."""

        self.abort_batch()

    def begin_batch(
        self,
        *,
        example_ids: Sequence[str],
        baseline_logits: Tensor,
        targets: Tensor,
        supervised_mask: Tensor,
    ) -> None:
        """Freeze one baseline shared frame without retaining raw logits."""

        self._ensure_not_finalized()
        if self._active is not None:
            self.abort_batch()
            raise RuntimeError(
                "a fingerprint batch was already active and was discarded"
            )
        if isinstance(example_ids, (str, bytes)) or not isinstance(
            example_ids,
            Sequence,
        ):
            raise TypeError("example_ids must be a sequence")
        raw_example_ids = tuple(example_ids)
        if not raw_example_ids:
            raise ValueError("an intervention batch cannot be empty")
        for index, value in enumerate(raw_example_ids):
            _require_nonempty(value, label=f"example_ids[{index}]")
        example_hashes = tuple(
            generator_fingerprint_example_id_sha256(value)
            for value in raw_example_ids
        )
        if len(set(example_hashes)) != len(example_hashes):
            raise ValueError("example ids must be unique within a batch")
        if any(
            value in self._seen_example_hashes for value in example_hashes
        ):
            raise ValueError(
                "duplicate example_id hash from a completed batch"
            )

        if (
            not isinstance(baseline_logits, Tensor)
            or baseline_logits.ndim != 3
            or not baseline_logits.dtype.is_floating_point
            or baseline_logits.shape[0] != len(raw_example_ids)
            or baseline_logits.shape[1] <= 0
            or baseline_logits.shape[2] <= 1
            or not torch.isfinite(baseline_logits).all()
        ):
            raise ValueError(
                "baseline_logits must be a finite floating Tensor with "
                "shape [examples, positions, vocabulary]"
            )
        shape = tuple(baseline_logits.shape)
        if len(shape) != 3:
            raise AssertionError("validated baseline shape drifted")
        batch_size, position_count, vocabulary_size = shape
        if self.anchor_count > vocabulary_size - 1:
            raise ValueError(
                "anchor_count cannot exceed vocabulary_size - 1"
            )
        if (
            self._vocabulary_size is not None
            and self._vocabulary_size != vocabulary_size
        ):
            raise ValueError(
                "all batches must use the same output vocabulary"
            )
        if (
            not isinstance(targets, Tensor)
            or targets.dtype != torch.int64
            or targets.shape != baseline_logits.shape[:2]
        ):
            raise ValueError(
                "targets must be an int64 Tensor matching the first two "
                "logit dimensions"
            )
        if (
            not isinstance(supervised_mask, Tensor)
            or supervised_mask.dtype != torch.bool
            or supervised_mask.shape != baseline_logits.shape[:2]
        ):
            raise ValueError(
                "supervised_mask must be a bool Tensor matching the first "
                "two logit dimensions"
            )

        mask = supervised_mask.detach().to(device="cpu")
        prompt_counts = mask.sum(dim=1, dtype=torch.int64)
        if (prompt_counts <= 0).any():
            first = int((prompt_counts <= 0).nonzero()[0, 0].item())
            raise ValueError(f"prompt {first} has no supervised positions")
        row_indices = mask.reshape(-1).nonzero().reshape(-1)
        target_rows = (
            targets.detach()
            .to(device="cpu")
            .reshape(-1)
            .index_select(0, row_indices)
        )
        if (
            (target_rows < 0).any()
            or (target_rows >= vocabulary_size).any()
        ):
            raise ValueError("an out-of-range supervised target was found")

        source_row_indices = row_indices.to(
            device=baseline_logits.device,
        )
        target_rows_device = target_rows.to(device=baseline_logits.device)
        baseline_log_probabilities = (
            baseline_logits.detach()
            .reshape(batch_size * position_count, vocabulary_size)
            .index_select(0, source_row_indices)
        )
        baseline_order = torch.argsort(
            baseline_log_probabilities,
            dim=-1,
            descending=True,
            stable=True,
        )
        non_target_order = baseline_order[
            baseline_order != target_rows_device[:, None]
        ].reshape(target_rows_device.numel(), vocabulary_size - 1)
        anchors = torch.cat(
            (
                target_rows_device[:, None],
                non_target_order[:, : self.anchor_count],
            ),
            dim=1,
        )
        baseline_anchor_logits = (
            baseline_log_probabilities.gather(1, anchors)
        )
        baseline_log_probabilities.sub_(
            torch.logsumexp(
                baseline_log_probabilities,
                dim=-1,
                keepdim=True,
            )
        )
        baseline_token_nll = -baseline_log_probabilities.gather(
            1,
            anchors[:, :1],
        )[:, 0]
        offsets = [0]
        for value in prompt_counts:
            offsets.append(offsets[-1] + int(value.item()))
        baseline_prompt_nll = torch.stack(
            tuple(
                baseline_token_nll[offsets[index] : offsets[index + 1]]
                .mean()
                for index in range(batch_size)
            )
        ).to(device="cpu", dtype=torch.float64)

        self._vocabulary_size = vocabulary_size
        self._active = _ActiveFingerprintBatch(
            example_id_sha256s=example_hashes,
            source_shape=shape,
            source_device=baseline_logits.device,
            source_dtype=baseline_logits.dtype,
            supervised_row_indices=source_row_indices,
            prompt_token_counts=prompt_counts,
            prompt_offsets=tuple(offsets),
            baseline_log_probabilities=baseline_log_probabilities,
            anchors=anchors,
            baseline_anchor_logits=baseline_anchor_logits,
            baseline_top1=baseline_log_probabilities.argmax(dim=-1),
            baseline_prompt_nll=baseline_prompt_nll,
            prompt_nll_effects={},
            prompt_kls={},
            prompt_agreements={},
            prompt_rms={},
            centered_anchor_effects={},
        )

    def add_muted_generator(
        self,
        generator_id: str,
        muted_logits: Tensor,
    ) -> None:
        """Consume one muted full-vocabulary result and retain no reference."""

        self._ensure_not_finalized()
        active = self._active
        if active is None:
            raise RuntimeError("no fingerprint batch is active")
        try:
            _require_nonempty(generator_id, label="generator_id")
            if generator_id not in self.generator_ids:
                raise ValueError(f"unknown generator id {generator_id!r}")
            if generator_id in active.prompt_nll_effects:
                raise ValueError(
                    f"generator {generator_id!r} was already consumed"
                )
            if (
                not isinstance(muted_logits, Tensor)
                or tuple(muted_logits.shape) != active.source_shape
                or muted_logits.dtype != active.source_dtype
                or muted_logits.device != active.source_device
                or not torch.isfinite(muted_logits).all()
            ):
                raise ValueError(
                    "muted_logits must be finite logits with the active "
                    "baseline shape, dtype, and device"
                )

            batch_size, position_count, vocabulary_size = active.source_shape
            muted_log_probabilities = (
                muted_logits.detach()
                .reshape(batch_size * position_count, vocabulary_size)
                .index_select(0, active.supervised_row_indices)
            )
            muted_anchor_logits = muted_log_probabilities.gather(
                1,
                active.anchors,
            )
            muted_log_probabilities.sub_(
                torch.logsumexp(
                    muted_log_probabilities,
                    dim=-1,
                    keepdim=True,
                )
            )

            muted_token_nll = -muted_log_probabilities.gather(
                1,
                active.anchors[:, :1],
            )[:, 0]
            kl_per_row = torch.zeros(
                muted_token_nll.shape,
                dtype=muted_log_probabilities.dtype,
                device=muted_log_probabilities.device,
            )
            for start in range(0, vocabulary_size, _VOCABULARY_CHUNK):
                stop = min(start + _VOCABULARY_CHUNK, vocabulary_size)
                baseline_chunk = active.baseline_log_probabilities[
                    :, start:stop
                ]
                muted_chunk = muted_log_probabilities[:, start:stop]
                kl_per_row.add_(
                    (
                        baseline_chunk.exp()
                        * (baseline_chunk - muted_chunk)
                    ).sum(dim=-1)
                )
            agreements_per_row = (
                muted_log_probabilities.argmax(dim=-1)
                == active.baseline_top1
            ).to(dtype=muted_log_probabilities.dtype)
            anchor_effect = (
                muted_anchor_logits - active.baseline_anchor_logits
            )
            centered_effect = anchor_effect - anchor_effect.mean(
                dim=-1,
                keepdim=True,
            )
            centered_effect_cpu = centered_effect.to(
                device="cpu",
                dtype=torch.float64,
            )

            prompt_nll: list[Tensor] = []
            prompt_kl: list[Tensor] = []
            prompt_agreement: list[Tensor] = []
            prompt_rms: list[Tensor] = []
            for prompt_index in range(batch_size):
                start = active.prompt_offsets[prompt_index]
                stop = active.prompt_offsets[prompt_index + 1]
                prompt_nll.append(
                    muted_token_nll[start:stop]
                    .mean()
                    .to(device="cpu", dtype=torch.float64)
                    - active.baseline_prompt_nll[prompt_index]
                )
                prompt_kl.append(
                    kl_per_row[start:stop]
                    .mean()
                    .clamp_min(0.0)
                    .to(device="cpu", dtype=torch.float64)
                )
                prompt_agreement.append(
                    agreements_per_row[start:stop]
                    .mean()
                    .to(device="cpu", dtype=torch.float64)
                )
                prompt_rms.append(
                    centered_effect_cpu[start:stop]
                    .square()
                    .mean()
                    .sqrt()
                )
            active.prompt_nll_effects[generator_id] = torch.stack(
                prompt_nll
            )
            active.prompt_kls[generator_id] = torch.stack(prompt_kl)
            active.prompt_agreements[generator_id] = torch.stack(
                prompt_agreement
            )
            active.prompt_rms[generator_id] = torch.stack(prompt_rms)
            active.centered_anchor_effects[
                generator_id
            ] = centered_effect_cpu
        except Exception:
            self.abort_batch()
            raise

    def finish_batch(self) -> None:
        """Commit one complete active batch and release its baseline."""

        self._ensure_not_finalized()
        active = self._active
        if active is None:
            raise RuntimeError("no fingerprint batch is active")
        try:
            missing = tuple(
                generator_id
                for generator_id in self.generator_ids
                if generator_id not in active.prompt_nll_effects
            )
            if missing:
                raise RuntimeError(
                    "cannot finish an incomplete fingerprint batch; "
                    f"missing={missing}"
                )

            local_nll_rows: list[Tensor] = []
            local_kl_rows: list[Tensor] = []
            local_agreement_rows: list[Tensor] = []
            local_rms_rows: list[Tensor] = []
            local_gram_rows: list[Tensor] = []
            for prompt_index in range(len(active.example_id_sha256s)):
                local_nll_rows.append(
                    torch.stack(
                        tuple(
                            active.prompt_nll_effects[generator_id][
                                prompt_index
                            ]
                            for generator_id in self.generator_ids
                        )
                    )
                )
                local_kl_rows.append(
                    torch.stack(
                        tuple(
                            active.prompt_kls[generator_id][prompt_index]
                            for generator_id in self.generator_ids
                        )
                    )
                )
                local_agreement_rows.append(
                    torch.stack(
                        tuple(
                            active.prompt_agreements[generator_id][
                                prompt_index
                            ]
                            for generator_id in self.generator_ids
                        )
                    )
                )
                local_rms_rows.append(
                    torch.stack(
                        tuple(
                            active.prompt_rms[generator_id][prompt_index]
                            for generator_id in self.generator_ids
                        )
                    )
                )
                start = active.prompt_offsets[prompt_index]
                stop = active.prompt_offsets[prompt_index + 1]
                effects = torch.stack(
                    tuple(
                        active.centered_anchor_effects[generator_id][
                            start:stop
                        ]
                        for generator_id in self.generator_ids
                    )
                )
                flattened = effects.reshape(len(self.generator_ids), -1)
                local_gram_rows.append(
                    flattened @ flattened.T / flattened.shape[1]
                )

            self._example_hashes.extend(active.example_id_sha256s)
            self._seen_example_hashes.update(active.example_id_sha256s)
            self._token_counts.extend(
                int(value.item()) for value in active.prompt_token_counts
            )
            self._nll_rows.extend(local_nll_rows)
            self._kl_rows.extend(local_kl_rows)
            self._agreement_rows.extend(local_agreement_rows)
            self._rms_rows.extend(local_rms_rows)
            self._gram_rows.extend(local_gram_rows)
        finally:
            self.abort_batch()

    def finalize(self) -> GeneratorCausalFingerprintAnalysis:
        """Freeze completed prompt summaries into an authenticated artifact."""

        if self._finalized_analysis is not None:
            return self._finalized_analysis
        if self._active is not None:
            self.abort_batch()
            raise RuntimeError(
                "cannot finalize with an incomplete active batch; it was "
                "discarded"
            )
        if not self._nll_rows:
            raise ValueError("cannot fingerprint an empty intervention stream")
        if self.top_importance_count > len(self._nll_rows):
            raise ValueError(
                "top_importance_count cannot exceed the collected prompt "
                "count"
            )
        self._finalized_analysis = _create_analysis(
            provenance=self.provenance,
            generator_ids=self.generator_ids,
            example_id_sha256s=tuple(self._example_hashes),
            anchor_count=self.anchor_count,
            top_importance_count=self.top_importance_count,
            policy=self.policy,
            supervised_token_counts=torch.tensor(
                self._token_counts,
                dtype=torch.int64,
            ),
            prompt_nll_effects=torch.stack(self._nll_rows, dim=1),
            prompt_baseline_to_muted_kls=torch.stack(
                self._kl_rows,
                dim=1,
            ),
            prompt_top1_agreements=torch.stack(
                self._agreement_rows,
                dim=1,
            ),
            prompt_centered_anchor_effect_rms=torch.stack(
                self._rms_rows,
                dim=1,
            ),
            centered_shared_effect_gram=torch.stack(
                self._gram_rows
            ).sum(dim=0),
        )
        return self._finalized_analysis


def collect_generator_causal_fingerprints(
    batches: Iterable[GeneratorInterventionLogitBatch],
    *,
    generator_ids: Sequence[str],
    provenance: GeneratorCausalFingerprintProvenance,
    anchor_count: int = 8,
    top_importance_count: int = 5,
    policy: ObservationalFamilyPolicy | None = None,
) -> GeneratorCausalFingerprintAnalysis:
    """Compatibility collector implemented through the streaming state machine.

    Memory-sensitive runners should call the accumulator directly so they can
    execute and submit one muted generator condition at a time.  This bulk
    wrapper preserves the original aligned-batch API and its exact output.
    """

    accumulator = GeneratorCausalFingerprintAccumulator(
        generator_ids=generator_ids,
        provenance=provenance,
        anchor_count=anchor_count,
        top_importance_count=top_importance_count,
        policy=policy,
    )
    iterator = iter(batches)
    try:
        for batch in iterator:
            if not isinstance(batch, GeneratorInterventionLogitBatch):
                raise TypeError(
                    "batches must contain GeneratorInterventionLogitBatch"
                )
            if set(batch.muted_logits_by_generator) != set(
                accumulator.generator_ids
            ):
                missing = sorted(
                    set(accumulator.generator_ids)
                    - set(batch.muted_logits_by_generator)
                )
                unexpected = sorted(
                    set(batch.muted_logits_by_generator)
                    - set(accumulator.generator_ids)
                )
                raise ValueError(
                    "muted generator catalog does not match generator_ids; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            accumulator.begin_batch(
                example_ids=batch.example_ids,
                baseline_logits=batch.baseline_logits,
                targets=batch.targets,
                supervised_mask=batch.supervised_mask,
            )
            for generator_id in accumulator.generator_ids:
                accumulator.add_muted_generator(
                    generator_id,
                    batch.muted_logits_by_generator[generator_id],
                )
            accumulator.finish_batch()
        return accumulator.finalize()
    finally:
        accumulator.close()
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
