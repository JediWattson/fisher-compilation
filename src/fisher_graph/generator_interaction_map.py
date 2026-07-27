"""Streaming exact-intervention maps for modal-generator interactions.

This module consumes one frozen baseline, every singleton generator
suppression, and every canonical two-generator suppression.  It retains only
prompt scalars, bounded baseline-anchor effects, and directed local response
summaries.  Joint full-vocabulary logits and singleton local-output tensors are
consumed synchronously and are never attached to authenticated analysis state.

Joint suppression produces an undirected finite second difference.  Directed
edges are separate: suppressing upstream generator ``i`` may change the local
residual output of downstream generator ``j`` only when ``i < j``.  Exact
invariance of every earlier generator output is a required causal negative
control.

All results are observational.  They do not authorize a merge, prune, route,
compile, execute, or mutate operation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import re

import torch
from torch import Tensor


__all__ = [
    "GeneratorInteractionMapAccumulator",
    "GeneratorInteractionMapAnalysis",
    "GeneratorInteractionMapProvenance",
    "interaction_map_example_id_sha256",
]


_ARTIFACT_KIND = "fisher_graph.modal_generator_interaction_map"
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = b"fisher_graph.generator_interaction_map.artifact.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.generator_interaction_map.tensor.v1\0"
_EXAMPLE_ID_DOMAIN = (
    b"fisher_graph.generator_interaction_map.example_id.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VOCABULARY_CHUNK = 4096

_INTERVENTION = (
    "frozen_baseline_all_singletons_and_all_canonical_joint_pairs"
)
_LOCAL_RESPONSE = (
    "singleton_upstream_suppression_to_downstream_generator_output"
)
_SHARED_FRAME = (
    "per_supervised_token_target_then_stable_baseline_top_non_target_logits"
)
_EFFECT_CENTERING = "per_supervised_token_anchor_mean"
_INTERACTION_NORMALIZATION = (
    "residual_rms_over_root_sum_singleton_anchor_mean_square"
)

_SAFETY_METADATA: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_generator_weights": False,
    "contains_prompt_text": False,
    "contains_raw_example_ids": False,
    "contains_token_ids": False,
    "contains_targets": False,
    "contains_raw_logits": False,
    "contains_local_activation_rows": False,
    "contains_local_generator_output_rows": False,
    "analysis_only": True,
    "observational_hypotheses_only": True,
    "strict_upstream_invariance": True,
    "mediation_measured": False,
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
    "valid_token_counts",
    "prompt_nll_second_differences",
    "prompt_joint_baseline_to_condition_kls",
    "prompt_joint_top1_agreements",
    "prompt_centered_anchor_interaction_residual_rms",
    "prompt_relative_interaction_denominator_rms",
    "prompt_relative_interaction_ratios",
    "prompt_relative_interaction_defined",
    "prompt_directed_response_rms",
    "prompt_directed_baseline_output_rms",
    "prompt_directed_response_cosines",
    "prompt_directed_response_cosine_defined",
    "prompt_directed_response_ratios",
    "prompt_directed_response_ratio_defined",
)

_FLOAT_PAIR_FIELDS = (
    "prompt_nll_second_differences",
    "prompt_joint_baseline_to_condition_kls",
    "prompt_joint_top1_agreements",
    "prompt_centered_anchor_interaction_residual_rms",
    "prompt_relative_interaction_denominator_rms",
    "prompt_relative_interaction_ratios",
)
_BOOL_PAIR_FIELDS = ("prompt_relative_interaction_defined",)
_FLOAT_DIRECTED_FIELDS = (
    "prompt_directed_response_rms",
    "prompt_directed_baseline_output_rms",
    "prompt_directed_response_cosines",
    "prompt_directed_response_ratios",
)
_BOOL_DIRECTED_FIELDS = (
    "prompt_directed_response_cosine_defined",
    "prompt_directed_response_ratio_defined",
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
    if value.dtype not in (torch.float64, torch.int64, torch.bool):
        raise ValueError(f"{label} must use float64, int64, or bool")
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


def interaction_map_example_id_sha256(example_id: str) -> str:
    """Return the opaque identifier retained for one prompt."""

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


def _pair_catalog(
    generator_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(itertools.combinations(generator_ids, 2))


def _validate_catalog(
    value: object,
    *,
    expected: tuple[tuple[str, str], ...],
    label: str,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    result = value
    if any(
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(item) is not str for item in pair)
        for pair in result
    ):
        raise TypeError(f"{label} entries must be string pairs")
    if result != expected:
        raise ValueError(f"{label} is not the canonical complete catalog")
    return result


@dataclass(frozen=True, slots=True)
class GeneratorInteractionMapProvenance:
    """Authenticated frozen sources for one interaction-map run."""

    source_model_sha256: str
    generator_catalog_sha256: str
    evaluation_split_sha256: str
    objective_sha256: str
    intervention: str = _INTERVENTION
    local_response: str = _LOCAL_RESPONSE

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
        if (
            self.intervention != _INTERVENTION
            or self.local_response != _LOCAL_RESPONSE
        ):
            raise ValueError("interaction-map intervention semantics drifted")

    def metadata(self) -> dict[str, object]:
        return {
            "source_model_sha256": self.source_model_sha256,
            "generator_catalog_sha256": self.generator_catalog_sha256,
            "evaluation_split_sha256": self.evaluation_split_sha256,
            "objective_sha256": self.objective_sha256,
            "intervention": self.intervention,
            "local_response": self.local_response,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> GeneratorInteractionMapProvenance:
        fields = {
            "source_model_sha256",
            "generator_catalog_sha256",
            "evaluation_split_sha256",
            "objective_sha256",
            "intervention",
            "local_response",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError("interaction-map provenance fields are invalid")
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
            local_response=state["local_response"],  # type: ignore[arg-type]
        )


def _analysis_payload(
    *,
    provenance: GeneratorInteractionMapProvenance,
    generator_ids: tuple[str, ...],
    pair_catalog: tuple[tuple[str, str], ...],
    directed_edge_catalog: tuple[tuple[str, str], ...],
    example_id_sha256s: tuple[str, ...],
    anchor_count: int,
    tensors: Mapping[str, Tensor],
) -> dict[str, object]:
    pair_count = len(pair_catalog)
    prompt_count = len(example_id_sha256s)
    return {
        "artifact_kind": _ARTIFACT_KIND,
        "format_version": _FORMAT_VERSION,
        **_SAFETY_METADATA,
        "provenance": provenance.metadata(),
        "generator_ids": generator_ids,
        "pair_catalog": pair_catalog,
        "directed_edge_catalog": directed_edge_catalog,
        "example_id_sha256s": example_id_sha256s,
        "generator_count": len(generator_ids),
        "pair_count": pair_count,
        "directed_edge_count": len(directed_edge_catalog),
        "prompt_count": prompt_count,
        "upstream_invariance_prompt_checks": pair_count * prompt_count,
        "anchor_count": anchor_count,
        "anchor_frame_width": anchor_count + 1,
        "shared_frame": _SHARED_FRAME,
        "effect_centering": _EFFECT_CENTERING,
        "interaction_normalization": _INTERACTION_NORMALIZATION,
        "relative_interaction_numerator_field": (
            "prompt_centered_anchor_interaction_residual_rms"
        ),
        "tensor_sha256s": {
            name: _tensor_sha256(value, label=name)
            for name, value in tensors.items()
        },
    }


@dataclass(frozen=True, slots=True)
class GeneratorInteractionMapAnalysis:
    """Strict interaction tensors with no executable mutation authority."""

    provenance: GeneratorInteractionMapProvenance
    generator_ids: tuple[str, ...]
    pair_catalog: tuple[tuple[str, str], ...]
    directed_edge_catalog: tuple[tuple[str, str], ...]
    example_id_sha256s: tuple[str, ...]
    anchor_count: int
    supervised_token_counts: Tensor
    valid_token_counts: Tensor
    prompt_nll_second_differences: Tensor
    prompt_joint_baseline_to_condition_kls: Tensor
    prompt_joint_top1_agreements: Tensor
    prompt_centered_anchor_interaction_residual_rms: Tensor
    prompt_relative_interaction_denominator_rms: Tensor
    prompt_relative_interaction_ratios: Tensor
    prompt_relative_interaction_defined: Tensor
    prompt_directed_response_rms: Tensor
    prompt_directed_baseline_output_rms: Tensor
    prompt_directed_response_cosines: Tensor
    prompt_directed_response_cosine_defined: Tensor
    prompt_directed_response_ratios: Tensor
    prompt_directed_response_ratio_defined: Tensor
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
    contains_local_activation_rows: bool = False
    contains_local_generator_output_rows: bool = False
    analysis_only: bool = True
    observational_hypotheses_only: bool = True
    strict_upstream_invariance: bool = True
    mediation_measured: bool = False
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
            GeneratorInteractionMapProvenance,
        ):
            raise TypeError(
                "provenance must be GeneratorInteractionMapProvenance"
            )
        ids = _canonical_generator_ids(self.generator_ids)
        if type(self.generator_ids) is not tuple or ids != self.generator_ids:
            raise ValueError("generator_ids must be a canonical tuple")
        expected_catalog = _pair_catalog(ids)
        _validate_catalog(
            self.pair_catalog,
            expected=expected_catalog,
            label="pair_catalog",
        )
        _validate_catalog(
            self.directed_edge_catalog,
            expected=expected_catalog,
            label="directed_edge_catalog",
        )
        if type(self.example_id_sha256s) is not tuple:
            raise TypeError("example_id_sha256s must be a tuple")
        if not self.example_id_sha256s:
            raise ValueError("an interaction map cannot be empty")
        for index, value in enumerate(self.example_id_sha256s):
            _require_sha256(value, label=f"example_id_sha256s[{index}]")
        if len(set(self.example_id_sha256s)) != len(
            self.example_id_sha256s
        ):
            raise ValueError("example id hashes must be unique")
        if type(self.anchor_count) is not int or self.anchor_count <= 0:
            raise ValueError("anchor_count must be a positive integer")

        prompt_count = len(self.example_id_sha256s)
        pair_count = len(expected_catalog)
        for name in ("supervised_token_counts", "valid_token_counts"):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.int64
                or value.shape != (prompt_count,)
                or (value <= 0).any()
            ):
                raise ValueError(
                    f"{name} must be a positive CPU int64 prompt vector"
                )
            object.__setattr__(
                self,
                name,
                value.detach().clone().contiguous(),
            )
        if (
            self.supervised_token_counts > self.valid_token_counts
        ).any():
            raise ValueError(
                "supervised token counts cannot exceed valid token counts"
            )

        matrix_shape = (pair_count, prompt_count)
        for name in (*_FLOAT_PAIR_FIELDS, *_FLOAT_DIRECTED_FIELDS):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or value.shape != matrix_shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"{name} must be a finite CPU float64 Tensor with "
                    f"shape {matrix_shape}"
                )
            object.__setattr__(
                self,
                name,
                value.detach().clone().contiguous(),
            )
        for name in (*_BOOL_PAIR_FIELDS, *_BOOL_DIRECTED_FIELDS):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.bool
                or value.shape != matrix_shape
            ):
                raise ValueError(
                    f"{name} must be a CPU bool Tensor with shape "
                    f"{matrix_shape}"
                )
            object.__setattr__(
                self,
                name,
                value.detach().clone().contiguous(),
            )

        if (self.prompt_joint_baseline_to_condition_kls < 0).any():
            raise ValueError("joint KL values must be nonnegative")
        if (
            (self.prompt_joint_top1_agreements < 0).any()
            or (self.prompt_joint_top1_agreements > 1).any()
        ):
            raise ValueError("joint top1 agreements must be in [0, 1]")
        for name in (
            "prompt_centered_anchor_interaction_residual_rms",
            "prompt_relative_interaction_denominator_rms",
            "prompt_relative_interaction_ratios",
            "prompt_directed_response_rms",
            "prompt_directed_baseline_output_rms",
            "prompt_directed_response_ratios",
        ):
            if (getattr(self, name) < 0).any():
                raise ValueError(f"{name} must be nonnegative")
        if (
            (self.prompt_directed_response_cosines < -1).any()
            or (self.prompt_directed_response_cosines > 1).any()
        ):
            raise ValueError("directed response cosines must be in [-1, 1]")

        numerator = (
            self.prompt_centered_anchor_interaction_residual_rms
        )
        interaction_defined = self.prompt_relative_interaction_defined
        denominator = self.prompt_relative_interaction_denominator_rms
        if not torch.equal(interaction_defined, denominator > 0):
            raise ValueError(
                "relative interaction defined flags disagree with denominator"
            )
        expected_ratio = torch.where(
            interaction_defined,
            numerator / torch.where(
                interaction_defined,
                denominator,
                torch.ones_like(denominator),
            ),
            torch.zeros_like(numerator),
        )
        if not torch.allclose(
            self.prompt_relative_interaction_ratios,
            expected_ratio,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("relative interaction ratios are inconsistent")

        response = self.prompt_directed_response_rms
        baseline = self.prompt_directed_baseline_output_rms
        ratio_defined = self.prompt_directed_response_ratio_defined
        cosine_defined = self.prompt_directed_response_cosine_defined
        if not torch.equal(ratio_defined, baseline > 0):
            raise ValueError("response ratio defined flags are inconsistent")
        if not torch.equal(cosine_defined, (baseline > 0) & (response > 0)):
            raise ValueError("response cosine defined flags are inconsistent")
        expected_response_ratio = torch.where(
            ratio_defined,
            response
            / torch.where(
                ratio_defined,
                baseline,
                torch.ones_like(baseline),
            ),
            torch.zeros_like(response),
        )
        if not torch.allclose(
            self.prompt_directed_response_ratios,
            expected_response_ratio,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("directed response ratios are inconsistent")
        if (
            self.prompt_directed_response_cosines[~cosine_defined] != 0
        ).any():
            raise ValueError("undefined directed cosines must be zero")

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
            raise ValueError("interaction-map safety metadata is invalid")
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("interaction-map artifact hash mismatch")

    @property
    def generator_count(self) -> int:
        return len(self.generator_ids)

    @property
    def prompt_count(self) -> int:
        return len(self.example_id_sha256s)

    @property
    def pair_count(self) -> int:
        return len(self.pair_catalog)

    @property
    def directed_edge_count(self) -> int:
        return len(self.directed_edge_catalog)

    @property
    def anchor_frame_width(self) -> int:
        return self.anchor_count + 1

    def pair_index(self, left: str, right: str) -> int:
        pair = (left, right)
        try:
            return self.pair_catalog.index(pair)
        except ValueError as error:
            raise KeyError(f"unknown canonical pair {pair!r}") from error

    def directed_edge_index(self, upstream: str, downstream: str) -> int:
        edge = (upstream, downstream)
        try:
            return self.directed_edge_catalog.index(edge)
        except ValueError as error:
            raise KeyError(f"unknown directed edge {edge!r}") from error

    def _tensors(self) -> dict[str, Tensor]:
        return {name: getattr(self, name) for name in _TENSOR_FIELDS}

    def _payload(self) -> dict[str, object]:
        return _analysis_payload(
            provenance=self.provenance,
            generator_ids=self.generator_ids,
            pair_catalog=self.pair_catalog,
            directed_edge_catalog=self.directed_edge_catalog,
            example_id_sha256s=self.example_id_sha256s,
            anchor_count=self.anchor_count,
            tensors=self._tensors(),
        )

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload())

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

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
            "pair_catalog": self.pair_catalog,
            "directed_edge_catalog": self.directed_edge_catalog,
            "example_id_sha256s": self.example_id_sha256s,
            "anchor_count": self.anchor_count,
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
    ) -> GeneratorInteractionMapAnalysis:
        fields = {
            "artifact_kind",
            "format_version",
            *_SAFETY_METADATA,
            "provenance",
            "generator_ids",
            "pair_catalog",
            "directed_edge_catalog",
            "example_id_sha256s",
            "anchor_count",
            *_TENSOR_FIELDS,
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError("interaction-map artifact fields are invalid")
        raw_provenance = state["provenance"]
        if not isinstance(raw_provenance, Mapping):
            raise TypeError("interaction-map provenance must be a mapping")
        for name in (
            "generator_ids",
            "pair_catalog",
            "directed_edge_catalog",
            "example_id_sha256s",
        ):
            if type(state[name]) is not tuple:
                raise TypeError(f"{name} must be a tuple")
        tensors: dict[str, Tensor] = {}
        for name in _TENSOR_FIELDS:
            value = state[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            tensors[name] = value
        return cls(
            provenance=GeneratorInteractionMapProvenance.from_state_dict(
                raw_provenance
            ),
            generator_ids=state["generator_ids"],  # type: ignore[arg-type]
            pair_catalog=state["pair_catalog"],  # type: ignore[arg-type]
            directed_edge_catalog=state[  # type: ignore[arg-type]
                "directed_edge_catalog"
            ],
            example_id_sha256s=state[  # type: ignore[arg-type]
                "example_id_sha256s"
            ],
            anchor_count=state["anchor_count"],  # type: ignore[arg-type]
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
    provenance: GeneratorInteractionMapProvenance,
    generator_ids: tuple[str, ...],
    pair_catalog: tuple[tuple[str, str], ...],
    directed_edge_catalog: tuple[tuple[str, str], ...],
    example_id_sha256s: tuple[str, ...],
    anchor_count: int,
    tensors: Mapping[str, Tensor],
) -> GeneratorInteractionMapAnalysis:
    payload = _analysis_payload(
        provenance=provenance,
        generator_ids=generator_ids,
        pair_catalog=pair_catalog,
        directed_edge_catalog=directed_edge_catalog,
        example_id_sha256s=example_id_sha256s,
        anchor_count=anchor_count,
        tensors=tensors,
    )
    return GeneratorInteractionMapAnalysis(
        provenance=provenance,
        generator_ids=generator_ids,
        pair_catalog=pair_catalog,
        directed_edge_catalog=directed_edge_catalog,
        example_id_sha256s=example_id_sha256s,
        anchor_count=anchor_count,
        **tensors,
        artifact_sha256=_json_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class _LogitConditionSummary:
    prompt_nll: Tensor
    centered_anchor_effect: Tensor
    prompt_kl: Tensor | None = None
    prompt_top1_agreement: Tensor | None = None


@dataclass(frozen=True, slots=True)
class _DirectedResponseSummary:
    response_rms: Tensor
    baseline_output_rms: Tensor
    response_cosine: Tensor
    response_cosine_defined: Tensor
    response_ratio: Tensor
    response_ratio_defined: Tensor


@dataclass(frozen=True, slots=True)
class _JointInteractionSummary:
    nll_second_difference: Tensor
    baseline_to_joint_kl: Tensor
    top1_agreement: Tensor
    interaction_residual_rms: Tensor
    relative_denominator_rms: Tensor
    relative_ratio: Tensor
    relative_defined: Tensor


@dataclass(slots=True)
class _ActiveInteractionBatch:
    example_id_sha256s: tuple[str, ...]
    source_shape: tuple[int, int, int]
    source_device: torch.device
    source_dtype: torch.dtype
    supervised_row_indices: Tensor
    supervised_token_counts: Tensor
    supervised_offsets: tuple[int, ...]
    valid_mask: Tensor
    valid_token_counts: Tensor
    baseline_log_probabilities: Tensor
    anchors: Tensor
    baseline_anchor_logits: Tensor
    baseline_top1: Tensor
    baseline_prompt_nll: Tensor
    baseline_generator_outputs: dict[str, Tensor]
    baseline_generator_output_rms: dict[str, Tensor]
    singletons: dict[str, _LogitConditionSummary]
    directed_responses: dict[
        tuple[str, str],
        _DirectedResponseSummary,
    ]
    joints: dict[tuple[str, str], _JointInteractionSummary]


class GeneratorInteractionMapAccumulator:
    """Bounded baseline/singleton/joint interaction collector.

    Full-vocabulary state consists of one baseline log-probability tensor.
    Singleton and joint logits are consumed one condition at a time.  Only
    singleton bounded anchor effects survive until their joint pairs arrive;
    joint logits and centered joint effects are immediately discarded.
    """

    def __init__(
        self,
        *,
        generator_ids: Sequence[str],
        provenance: GeneratorInteractionMapProvenance,
        anchor_count: int = 8,
    ) -> None:
        self.generator_ids = _canonical_generator_ids(generator_ids)
        if not isinstance(provenance, GeneratorInteractionMapProvenance):
            raise TypeError(
                "provenance must be GeneratorInteractionMapProvenance"
            )
        if type(anchor_count) is not int or anchor_count <= 0:
            raise ValueError("anchor_count must be a positive integer")
        self.provenance = provenance
        self.anchor_count = anchor_count
        self.pair_catalog = _pair_catalog(self.generator_ids)
        self.directed_edge_catalog = self.pair_catalog
        self._generator_index = {
            generator_id: index
            for index, generator_id in enumerate(self.generator_ids)
        }
        self._active: _ActiveInteractionBatch | None = None
        self._vocabulary_size: int | None = None
        self._seen_example_hashes: set[str] = set()
        self._example_hashes: list[str] = []
        self._supervised_counts: list[int] = []
        self._valid_counts: list[int] = []
        self._rows: dict[str, list[Tensor]] = {
            name: [] for name in _TENSOR_FIELDS if "token_counts" not in name
        }
        self._finalized_analysis: GeneratorInteractionMapAnalysis | None = None

    def __enter__(self) -> GeneratorInteractionMapAccumulator:
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
    def completed_prompt_count(self) -> int:
        return len(self._example_hashes)

    @property
    def active_baseline_full_vocabulary_tensor_count(self) -> int:
        return int(self._active is not None)

    @property
    def active_retained_joint_full_vocabulary_tensor_count(self) -> int:
        """Joint full-vocabulary tensors are never retained."""

        return 0

    @property
    def active_singleton_count(self) -> int:
        return 0 if self._active is None else len(self._active.singletons)

    @property
    def active_joint_count(self) -> int:
        return 0 if self._active is None else len(self._active.joints)

    def _ensure_not_finalized(self) -> None:
        if self._finalized_analysis is not None:
            raise RuntimeError("generator interaction accumulator is finalized")

    def abort_batch(self) -> None:
        """Release all unfinished baseline and bounded active-batch state."""

        self._active = None

    def close(self) -> None:
        self.abort_batch()

    def _validate_logits(
        self,
        value: object,
        *,
        expected_shape: tuple[int, int, int] | None,
        expected_device: torch.device | None,
        expected_dtype: torch.dtype | None,
        label: str,
    ) -> Tensor:
        if (
            not isinstance(value, Tensor)
            or value.ndim != 3
            or not value.dtype.is_floating_point
            or value.shape[0] <= 0
            or value.shape[1] <= 0
            or value.shape[2] <= 1
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"{label} must be finite floating "
                "[examples, positions, vocabulary] logits"
            )
        if (
            expected_shape is not None
            and (
                tuple(value.shape) != expected_shape
                or value.device != expected_device
                or value.dtype != expected_dtype
            )
        ):
            raise ValueError(
                f"{label} must match the active baseline shape, dtype, "
                "and device"
            )
        return value

    def _validate_generator_outputs(
        self,
        values: Mapping[str, Tensor],
        *,
        expected_generator_ids: Sequence[str],
        batch_size: int,
        position_count: int,
        baseline: Mapping[str, Tensor] | None,
        clone: bool,
    ) -> dict[str, Tensor]:
        expected_ids = tuple(expected_generator_ids)
        if not isinstance(values, Mapping) or set(values) != set(expected_ids):
            raise ValueError(
                "generator outputs must match the exact active generator "
                "catalog"
            )
        result: dict[str, Tensor] = {}
        for generator_id in expected_ids:
            value = values[generator_id]
            reference = None if baseline is None else baseline[generator_id]
            if (
                not isinstance(value, Tensor)
                or value.ndim != 3
                or not value.dtype.is_floating_point
                or value.shape[0] != batch_size
                or value.shape[1] != position_count
                or value.shape[2] <= 0
                or not torch.isfinite(value).all()
                or (
                    reference is not None
                    and (
                        value.shape != reference.shape
                        or value.device != reference.device
                        or value.dtype != reference.dtype
                    )
                )
            ):
                raise ValueError(
                    f"generator output {generator_id!r} is not finite or "
                    "does not match its frozen baseline shape/dtype/device"
                )
            result[generator_id] = (
                value.detach().clone() if clone else value.detach()
            )
        return result

    @staticmethod
    def _prompt_offsets(counts: Tensor) -> tuple[int, ...]:
        offsets = [0]
        for value in counts:
            offsets.append(offsets[-1] + int(value.item()))
        return tuple(offsets)

    @staticmethod
    def _prompt_means(
        rows: Tensor,
        offsets: tuple[int, ...],
    ) -> Tensor:
        return torch.stack(
            tuple(
                rows[offsets[index] : offsets[index + 1]].mean()
                for index in range(len(offsets) - 1)
            )
        ).to(device="cpu", dtype=torch.float64)

    def begin_batch(
        self,
        *,
        example_ids: Sequence[str],
        baseline_logits: Tensor,
        targets: Tensor,
        supervised_mask: Tensor,
        valid_mask: Tensor,
        baseline_generator_outputs: Mapping[str, Tensor],
    ) -> None:
        """Begin one batch and retain only its frozen bounded baseline state."""

        self._ensure_not_finalized()
        if self._active is not None:
            self.abort_batch()
            raise RuntimeError(
                "an interaction batch was already active and was discarded"
            )
        if isinstance(example_ids, (str, bytes)) or not isinstance(
            example_ids,
            Sequence,
        ):
            raise TypeError("example_ids must be a sequence")
        raw_ids = tuple(example_ids)
        if not raw_ids:
            raise ValueError("an interaction batch cannot be empty")
        for index, value in enumerate(raw_ids):
            _require_nonempty(value, label=f"example_ids[{index}]")
        example_hashes = tuple(
            interaction_map_example_id_sha256(value) for value in raw_ids
        )
        if len(set(example_hashes)) != len(example_hashes):
            raise ValueError("example ids must be unique within a batch")
        if any(value in self._seen_example_hashes for value in example_hashes):
            raise ValueError(
                "duplicate example_id hash from a completed batch"
            )

        baseline = self._validate_logits(
            baseline_logits,
            expected_shape=None,
            expected_device=None,
            expected_dtype=None,
            label="baseline_logits",
        )
        shape = tuple(baseline.shape)
        if len(shape) != 3:
            raise AssertionError("validated baseline shape drifted")
        batch_size, position_count, vocabulary_size = shape
        if batch_size != len(raw_ids):
            raise ValueError(
                "baseline batch size must match the example id count"
            )
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
            or targets.shape != baseline.shape[:2]
        ):
            raise ValueError(
                "targets must be int64 and match the first two logit "
                "dimensions"
            )
        for name, mask in (
            ("supervised_mask", supervised_mask),
            ("valid_mask", valid_mask),
        ):
            if (
                not isinstance(mask, Tensor)
                or mask.dtype != torch.bool
                or mask.shape != baseline.shape[:2]
            ):
                raise ValueError(
                    f"{name} must be bool and match the first two logit "
                    "dimensions"
                )
        supervised_cpu = supervised_mask.detach().to(device="cpu")
        # This mask remains live for all directed-response reductions in the
        # batch.  ``Tensor.to("cpu")`` may alias a CPU caller, so retain an
        # owned snapshot just as we do for baseline generator outputs.
        valid_cpu = valid_mask.detach().to(device="cpu").clone()
        if (supervised_cpu & ~valid_cpu).any():
            raise ValueError("supervised positions must also be valid")
        supervised_counts = supervised_cpu.sum(dim=1, dtype=torch.int64)
        valid_counts = valid_cpu.sum(dim=1, dtype=torch.int64)
        if (supervised_counts <= 0).any() or (valid_counts <= 0).any():
            raise ValueError(
                "every prompt needs positive supervised and valid positions"
            )
        supervised_row_indices_cpu = (
            supervised_cpu.reshape(-1).nonzero().reshape(-1)
        )
        targets_cpu = (
            targets.detach()
            .to(device="cpu")
            .reshape(-1)
            .index_select(0, supervised_row_indices_cpu)
        )
        if (targets_cpu < 0).any() or (targets_cpu >= vocabulary_size).any():
            raise ValueError("an out-of-range supervised target was found")
        supervised_row_indices = supervised_row_indices_cpu.to(
            device=baseline.device
        )
        target_rows = targets_cpu.to(device=baseline.device)

        baseline_log_probabilities = (
            baseline.detach()
            .reshape(batch_size * position_count, vocabulary_size)
            .index_select(0, supervised_row_indices)
        )
        baseline_order = torch.argsort(
            baseline_log_probabilities,
            dim=-1,
            descending=True,
            stable=True,
        )
        non_target_order = baseline_order[
            baseline_order != target_rows[:, None]
        ].reshape(target_rows.numel(), vocabulary_size - 1)
        anchors = torch.cat(
            (
                target_rows[:, None],
                non_target_order[:, : self.anchor_count],
            ),
            dim=1,
        )
        baseline_anchor_logits = baseline_log_probabilities.gather(1, anchors)
        baseline_log_probabilities.sub_(
            torch.logsumexp(
                baseline_log_probabilities,
                dim=-1,
                keepdim=True,
            )
        )
        offsets = self._prompt_offsets(supervised_counts)
        baseline_token_nll = -baseline_log_probabilities.gather(
            1,
            anchors[:, :1],
        )[:, 0]
        baseline_prompt_nll = self._prompt_means(
            baseline_token_nll,
            offsets,
        )

        outputs = self._validate_generator_outputs(
            baseline_generator_outputs,
            expected_generator_ids=self.generator_ids,
            batch_size=batch_size,
            position_count=position_count,
            baseline=None,
            clone=True,
        )
        baseline_output_rms: dict[str, Tensor] = {}
        for generator_id in self.generator_ids:
            output = outputs[generator_id]
            prompt_values: list[Tensor] = []
            for prompt_index in range(batch_size):
                mask = valid_cpu[prompt_index].to(device=output.device)
                prompt_values.append(
                    output[prompt_index][mask].square().mean().sqrt()
                )
            baseline_output_rms[generator_id] = torch.stack(
                prompt_values
            ).to(device="cpu", dtype=torch.float64)

        self._vocabulary_size = vocabulary_size
        self._active = _ActiveInteractionBatch(
            example_id_sha256s=example_hashes,
            source_shape=shape,
            source_device=baseline.device,
            source_dtype=baseline.dtype,
            supervised_row_indices=supervised_row_indices,
            supervised_token_counts=supervised_counts,
            supervised_offsets=offsets,
            valid_mask=valid_cpu,
            valid_token_counts=valid_counts,
            baseline_log_probabilities=baseline_log_probabilities,
            anchors=anchors,
            baseline_anchor_logits=baseline_anchor_logits,
            baseline_top1=baseline_log_probabilities.argmax(dim=-1),
            baseline_prompt_nll=baseline_prompt_nll,
            baseline_generator_outputs=outputs,
            baseline_generator_output_rms=baseline_output_rms,
            singletons={},
            directed_responses={},
            joints={},
        )

    def _consume_logits(
        self,
        logits: Tensor,
        *,
        include_distribution_metrics: bool,
        label: str,
    ) -> _LogitConditionSummary:
        active = self._active
        if active is None:
            raise RuntimeError("no interaction batch is active")
        value = self._validate_logits(
            logits,
            expected_shape=active.source_shape,
            expected_device=active.source_device,
            expected_dtype=active.source_dtype,
            label=label,
        )
        batch_size, position_count, vocabulary_size = active.source_shape
        log_probabilities = (
            value.detach()
            .reshape(batch_size * position_count, vocabulary_size)
            .index_select(0, active.supervised_row_indices)
        )
        anchor_logits = log_probabilities.gather(1, active.anchors)
        log_probabilities.sub_(
            torch.logsumexp(
                log_probabilities,
                dim=-1,
                keepdim=True,
            )
        )
        token_nll = -log_probabilities.gather(
            1,
            active.anchors[:, :1],
        )[:, 0]
        prompt_nll = self._prompt_means(
            token_nll,
            active.supervised_offsets,
        )
        anchor_effect = anchor_logits - active.baseline_anchor_logits
        centered_effect = anchor_effect - anchor_effect.mean(
            dim=-1,
            keepdim=True,
        )
        centered_effect_cpu = centered_effect.to(
            device="cpu",
            dtype=torch.float64,
        )
        if not include_distribution_metrics:
            return _LogitConditionSummary(
                prompt_nll=prompt_nll,
                centered_anchor_effect=centered_effect_cpu,
            )

        kl_per_row = torch.zeros(
            token_nll.shape,
            dtype=log_probabilities.dtype,
            device=log_probabilities.device,
        )
        for start in range(0, vocabulary_size, _VOCABULARY_CHUNK):
            stop = min(start + _VOCABULARY_CHUNK, vocabulary_size)
            baseline_chunk = active.baseline_log_probabilities[:, start:stop]
            condition_chunk = log_probabilities[:, start:stop]
            kl_per_row.add_(
                (
                    baseline_chunk.exp()
                    * (baseline_chunk - condition_chunk)
                ).sum(dim=-1)
            )
        prompt_kl = self._prompt_means(
            kl_per_row,
            active.supervised_offsets,
        ).clamp_min(0.0)
        top1_per_row = (
            log_probabilities.argmax(dim=-1) == active.baseline_top1
        ).to(dtype=log_probabilities.dtype)
        prompt_top1 = self._prompt_means(
            top1_per_row,
            active.supervised_offsets,
        )
        return _LogitConditionSummary(
            prompt_nll=prompt_nll,
            centered_anchor_effect=centered_effect_cpu,
            prompt_kl=prompt_kl,
            prompt_top1_agreement=prompt_top1,
        )

    def _directed_response(
        self,
        *,
        upstream_id: str,
        downstream_id: str,
        current_output: Tensor,
    ) -> _DirectedResponseSummary:
        active = self._active
        if active is None:
            raise RuntimeError("no interaction batch is active")
        baseline_output = active.baseline_generator_outputs[downstream_id]
        baseline_rms = active.baseline_generator_output_rms[downstream_id]
        response_rms_values: list[Tensor] = []
        cosine_values: list[float] = []
        cosine_defined_values: list[bool] = []
        ratio_values: list[float] = []
        ratio_defined_values: list[bool] = []
        for prompt_index in range(len(active.example_id_sha256s)):
            mask = active.valid_mask[prompt_index].to(
                device=current_output.device
            )
            baseline_rows = baseline_output[prompt_index][mask]
            response_rows = (
                current_output[prompt_index][mask] - baseline_rows
            )
            response_square_sum = response_rows.square().sum()
            baseline_square_sum = baseline_rows.square().sum()
            response_rms = response_rows.square().mean().sqrt()
            response_rms_values.append(response_rms)
            response_norm = math.sqrt(
                max(0.0, float(response_square_sum.item()))
            )
            baseline_norm = math.sqrt(
                max(0.0, float(baseline_square_sum.item()))
            )
            cosine_defined = response_norm > 0.0 and baseline_norm > 0.0
            cosine = (
                float(
                    (response_rows * baseline_rows).sum().item()
                    / (response_norm * baseline_norm)
                )
                if cosine_defined
                else 0.0
            )
            cosine_values.append(max(-1.0, min(1.0, cosine)))
            cosine_defined_values.append(cosine_defined)
            base_rms_value = float(baseline_rms[prompt_index].item())
            response_rms_value = float(response_rms.item())
            ratio_defined = base_rms_value > 0.0
            ratio_values.append(
                response_rms_value / base_rms_value
                if ratio_defined
                else 0.0
            )
            ratio_defined_values.append(ratio_defined)
        return _DirectedResponseSummary(
            response_rms=torch.stack(response_rms_values).to(
                device="cpu",
                dtype=torch.float64,
            ),
            baseline_output_rms=baseline_rms.detach().clone(),
            response_cosine=torch.tensor(
                cosine_values,
                dtype=torch.float64,
            ),
            response_cosine_defined=torch.tensor(
                cosine_defined_values,
                dtype=torch.bool,
            ),
            response_ratio=torch.tensor(
                ratio_values,
                dtype=torch.float64,
            ),
            response_ratio_defined=torch.tensor(
                ratio_defined_values,
                dtype=torch.bool,
            ),
        )

    def add_singleton(
        self,
        generator_id: str,
        logits: Tensor,
        generator_outputs: Mapping[str, Tensor],
    ) -> None:
        """Consume one exact singleton and its ephemeral local outputs."""

        self._ensure_not_finalized()
        active = self._active
        if active is None:
            raise RuntimeError("no interaction batch is active")
        try:
            _require_nonempty(generator_id, label="generator_id")
            if generator_id not in self._generator_index:
                raise ValueError(f"unknown generator id {generator_id!r}")
            if generator_id in active.singletons:
                raise ValueError(
                    f"singleton {generator_id!r} was already consumed"
                )
            if active.joints:
                raise RuntimeError(
                    "singletons cannot be added after joint conditions"
                )
            batch_size, position_count, _ = active.source_shape
            muted_index = self._generator_index[generator_id]
            active_generator_ids = (
                self.generator_ids[:muted_index]
                + self.generator_ids[muted_index + 1 :]
            )
            outputs = self._validate_generator_outputs(
                generator_outputs,
                expected_generator_ids=active_generator_ids,
                batch_size=batch_size,
                position_count=position_count,
                baseline=active.baseline_generator_outputs,
                clone=False,
            )
            for other_id in self.generator_ids[:muted_index]:
                if not torch.equal(
                    outputs[other_id],
                    active.baseline_generator_outputs[other_id],
                ):
                    raise ValueError(
                        "strict upstream invariance failed: suppressing "
                        f"{generator_id!r} changed earlier output {other_id!r}"
                    )
            summary = self._consume_logits(
                logits,
                include_distribution_metrics=False,
                label=f"singleton logits for {generator_id!r}",
            )
            directed: dict[
                tuple[str, str],
                _DirectedResponseSummary,
            ] = {}
            for downstream_id in self.generator_ids[muted_index + 1 :]:
                edge = (generator_id, downstream_id)
                directed[edge] = self._directed_response(
                    upstream_id=generator_id,
                    downstream_id=downstream_id,
                    current_output=outputs[downstream_id],
                )
            active.singletons[generator_id] = summary
            active.directed_responses.update(directed)
        except Exception:
            self.abort_batch()
            raise

    def _canonical_pair(
        self,
        left_generator_id: str,
        right_generator_id: str,
    ) -> tuple[str, str]:
        _require_nonempty(
            left_generator_id,
            label="left_generator_id",
        )
        _require_nonempty(
            right_generator_id,
            label="right_generator_id",
        )
        if (
            left_generator_id not in self._generator_index
            or right_generator_id not in self._generator_index
        ):
            raise ValueError("joint pair contains an unknown generator id")
        left_index = self._generator_index[left_generator_id]
        right_index = self._generator_index[right_generator_id]
        if not left_index < right_index:
            raise ValueError(
                "joint pair must be a distinct canonical upstream/downstream "
                "pair"
            )
        return left_generator_id, right_generator_id

    def add_joint(
        self,
        left_generator_id: str,
        right_generator_id: str,
        logits: Tensor,
    ) -> None:
        """Consume one exact canonical two-generator suppression."""

        self._ensure_not_finalized()
        active = self._active
        if active is None:
            raise RuntimeError("no interaction batch is active")
        try:
            pair = self._canonical_pair(
                left_generator_id,
                right_generator_id,
            )
            if len(active.singletons) != len(self.generator_ids):
                raise RuntimeError(
                    "all singleton conditions must precede joint conditions"
                )
            if pair in active.joints:
                raise ValueError(f"joint pair {pair!r} was already consumed")
            joint = self._consume_logits(
                logits,
                include_distribution_metrics=True,
                label=f"joint logits for {pair!r}",
            )
            if (
                joint.prompt_kl is None
                or joint.prompt_top1_agreement is None
            ):
                raise AssertionError("joint distribution metrics are absent")
            left = active.singletons[pair[0]]
            right = active.singletons[pair[1]]
            nll_second_difference = (
                joint.prompt_nll
                - left.prompt_nll
                - right.prompt_nll
                + active.baseline_prompt_nll
            )
            residual = (
                joint.centered_anchor_effect
                - left.centered_anchor_effect
                - right.centered_anchor_effect
            )
            prompt_residual_rms: list[Tensor] = []
            prompt_denominator: list[Tensor] = []
            for prompt_index in range(len(active.example_id_sha256s)):
                start = active.supervised_offsets[prompt_index]
                stop = active.supervised_offsets[prompt_index + 1]
                residual_rows = residual[start:stop]
                left_rows = left.centered_anchor_effect[start:stop]
                right_rows = right.centered_anchor_effect[start:stop]
                prompt_residual_rms.append(
                    residual_rows.square().mean().sqrt()
                )
                prompt_denominator.append(
                    (
                        left_rows.square().mean()
                        + right_rows.square().mean()
                    ).sqrt()
                )
            residual_rms = torch.stack(prompt_residual_rms)
            denominator = torch.stack(prompt_denominator)
            defined = denominator > 0
            ratio = torch.where(
                defined,
                residual_rms
                / torch.where(
                    defined,
                    denominator,
                    torch.ones_like(denominator),
                ),
                torch.zeros_like(residual_rms),
            )
            active.joints[pair] = _JointInteractionSummary(
                nll_second_difference=nll_second_difference,
                baseline_to_joint_kl=joint.prompt_kl,
                top1_agreement=joint.prompt_top1_agreement,
                interaction_residual_rms=residual_rms,
                relative_denominator_rms=denominator,
                relative_ratio=ratio,
                relative_defined=defined,
            )
        except Exception:
            self.abort_batch()
            raise

    def finish_batch(self) -> None:
        """Commit one complete batch and release all active large tensors."""

        self._ensure_not_finalized()
        active = self._active
        if active is None:
            raise RuntimeError("no interaction batch is active")
        try:
            missing_singletons = tuple(
                generator_id
                for generator_id in self.generator_ids
                if generator_id not in active.singletons
            )
            missing_joints = tuple(
                pair for pair in self.pair_catalog if pair not in active.joints
            )
            missing_directed = tuple(
                edge
                for edge in self.directed_edge_catalog
                if edge not in active.directed_responses
            )
            if missing_singletons or missing_joints or missing_directed:
                raise RuntimeError(
                    "cannot finish an incomplete interaction batch; "
                    f"missing_singletons={missing_singletons}, "
                    f"missing_joints={missing_joints}, "
                    f"missing_directed={missing_directed}"
                )

            local_rows: dict[str, list[Tensor]] = {
                name: [] for name in self._rows
            }
            for prompt_index in range(len(active.example_id_sha256s)):
                local_rows["prompt_nll_second_differences"].append(
                    torch.stack(
                        tuple(
                            active.joints[pair].nll_second_difference[
                                prompt_index
                            ]
                            for pair in self.pair_catalog
                        )
                    )
                )
                local_rows[
                    "prompt_joint_baseline_to_condition_kls"
                ].append(
                    torch.stack(
                        tuple(
                            active.joints[pair].baseline_to_joint_kl[
                                prompt_index
                            ]
                            for pair in self.pair_catalog
                        )
                    )
                )
                local_rows["prompt_joint_top1_agreements"].append(
                    torch.stack(
                        tuple(
                            active.joints[pair].top1_agreement[prompt_index]
                            for pair in self.pair_catalog
                        )
                    )
                )
                local_rows[
                    "prompt_centered_anchor_interaction_residual_rms"
                ].append(
                    torch.stack(
                        tuple(
                            active.joints[pair].interaction_residual_rms[
                                prompt_index
                            ]
                            for pair in self.pair_catalog
                        )
                    )
                )
                local_rows[
                    "prompt_relative_interaction_denominator_rms"
                ].append(
                    torch.stack(
                        tuple(
                            active.joints[pair].relative_denominator_rms[
                                prompt_index
                            ]
                            for pair in self.pair_catalog
                        )
                    )
                )
                local_rows["prompt_relative_interaction_ratios"].append(
                    torch.stack(
                        tuple(
                            active.joints[pair].relative_ratio[prompt_index]
                            for pair in self.pair_catalog
                        )
                    )
                )
                local_rows["prompt_relative_interaction_defined"].append(
                    torch.stack(
                        tuple(
                            active.joints[pair].relative_defined[prompt_index]
                            for pair in self.pair_catalog
                        )
                    )
                )
                local_rows["prompt_directed_response_rms"].append(
                    torch.stack(
                        tuple(
                            active.directed_responses[
                                edge
                            ].response_rms[prompt_index]
                            for edge in self.directed_edge_catalog
                        )
                    )
                )
                local_rows[
                    "prompt_directed_baseline_output_rms"
                ].append(
                    torch.stack(
                        tuple(
                            active.directed_responses[
                                edge
                            ].baseline_output_rms[prompt_index]
                            for edge in self.directed_edge_catalog
                        )
                    )
                )
                local_rows["prompt_directed_response_cosines"].append(
                    torch.stack(
                        tuple(
                            active.directed_responses[
                                edge
                            ].response_cosine[prompt_index]
                            for edge in self.directed_edge_catalog
                        )
                    )
                )
                local_rows[
                    "prompt_directed_response_cosine_defined"
                ].append(
                    torch.stack(
                        tuple(
                            active.directed_responses[
                                edge
                            ].response_cosine_defined[prompt_index]
                            for edge in self.directed_edge_catalog
                        )
                    )
                )
                local_rows["prompt_directed_response_ratios"].append(
                    torch.stack(
                        tuple(
                            active.directed_responses[
                                edge
                            ].response_ratio[prompt_index]
                            for edge in self.directed_edge_catalog
                        )
                    )
                )
                local_rows[
                    "prompt_directed_response_ratio_defined"
                ].append(
                    torch.stack(
                        tuple(
                            active.directed_responses[
                                edge
                            ].response_ratio_defined[prompt_index]
                            for edge in self.directed_edge_catalog
                        )
                    )
                )

            self._example_hashes.extend(active.example_id_sha256s)
            self._seen_example_hashes.update(active.example_id_sha256s)
            self._supervised_counts.extend(
                int(value.item())
                for value in active.supervised_token_counts
            )
            self._valid_counts.extend(
                int(value.item()) for value in active.valid_token_counts
            )
            for name, values in local_rows.items():
                self._rows[name].extend(values)
        finally:
            self.abort_batch()

    def finalize(self) -> GeneratorInteractionMapAnalysis:
        """Freeze completed prompt summaries into authenticated CPU state."""

        if self._finalized_analysis is not None:
            return self._finalized_analysis
        if self._active is not None:
            self.abort_batch()
            raise RuntimeError(
                "cannot finalize with an incomplete active batch; it was "
                "discarded"
            )
        if not self._example_hashes:
            raise ValueError("cannot finalize an empty interaction map")
        tensors: dict[str, Tensor] = {
            "supervised_token_counts": torch.tensor(
                self._supervised_counts,
                dtype=torch.int64,
            ),
            "valid_token_counts": torch.tensor(
                self._valid_counts,
                dtype=torch.int64,
            ),
        }
        for name, rows in self._rows.items():
            if not rows:
                raise AssertionError(f"completed interaction rows lack {name}")
            tensors[name] = torch.stack(rows, dim=1)
        self._finalized_analysis = _create_analysis(
            provenance=self.provenance,
            generator_ids=self.generator_ids,
            pair_catalog=self.pair_catalog,
            directed_edge_catalog=self.directed_edge_catalog,
            example_id_sha256s=tuple(self._example_hashes),
            anchor_count=self.anchor_count,
            tensors=tensors,
        )
        return self._finalized_analysis
