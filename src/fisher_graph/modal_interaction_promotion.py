"""Authenticated promotion receipts for state-conditioned modal graphs.

The legacy :class:`ModalInteractionSelection` artifact owns the coefficients
and step-by-step held-out evidence for affine interactions.  Its v1 state and
hash domain must remain stable.  A state-conditioned graph needs a different
authorization boundary because its executable coefficients already live in
the authenticated graph and its routes are selected per token at runtime.

``ModalInteractionGraphPromotion`` is therefore a compact receipt.  It binds
one exact graph hash and ordered interaction-hash tuple to the compiler
lineage and to a simple, recomputable open-development improvement gate.  It
contains no graph weights, fit rows, teacher outputs, prompts, or token ids.
Unknown authorization kinds continue to fail closed at deserialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .modal_generator_graph import (
    ModalGeneratorGraphPlan,
    StateConditionedModalGeneratorInteraction,
)
from .modal_interaction_fitting import ModalInteractionSelection


__all__ = [
    "ModalInteractionAuthorization",
    "ModalInteractionGraphPromotion",
    "authenticate_modal_interaction_authorization",
    "build_modal_interaction_graph_promotion",
    "modal_interaction_authorization_from_state_dict",
]


_FORMAT_VERSION = 1
_PROMOTION_KIND = "fisher_graph.modal_interaction_graph_promotion"
_LEGACY_SELECTION_KIND = "fisher_graph.modal_interaction_selection"
_PROMOTION_DOMAIN = b"fisher_graph.modal_interaction_graph_promotion.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROMOTION_METRICS = frozenset(
    {
        "nll_per_token",
        "native_to_candidate_kl_per_token",
        "weighted_nrmse",
    }
)
_SAFETY_METADATA: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_raw_latent_rows": False,
    "contains_target_residual_rows": False,
    "contains_teacher_outputs": False,
    "contains_generator_weights": False,
    "contains_interaction_weights": False,
    "executable": False,
    "tuned_on_closed_split": False,
}


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_fields(
    state: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


@dataclass(frozen=True, slots=True)
class ModalInteractionGraphPromotion:
    """Source-safe authorization for one exact conditional interaction graph.

    The promotion metric is measured on open development after the graph has
    been frozen.  All supported metrics are lower-is-better.  Promotion
    requires a strict improvement over the edgeless baseline, matching the
    legacy selector's refusal to admit a zero-improvement affine edge.
    """

    source_model_sha256: str
    parameter_cluster_plan_sha256: str
    fit_split_sha256: str
    eval_split_sha256: str
    graph_plan_sha256: str
    interaction_artifact_sha256s: tuple[str, ...]
    conditional_interaction_count: int
    selection_metric: str
    baseline_metric_value: float
    candidate_metric_value: float
    minimum_heldout_improvement: float
    heldout_observations: int
    artifact_sha256: str = ""
    artifact_kind: str = _PROMOTION_KIND
    format_version: int = _FORMAT_VERSION
    contains_source_model_weights: bool = False
    contains_prompt_text: bool = False
    contains_token_ids: bool = False
    contains_raw_latent_rows: bool = False
    contains_target_residual_rows: bool = False
    contains_teacher_outputs: bool = False
    contains_generator_weights: bool = False
    contains_interaction_weights: bool = False
    executable: bool = False
    tuned_on_closed_split: bool = False

    def __post_init__(self) -> None:
        for field in (
            "source_model_sha256",
            "parameter_cluster_plan_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
            "graph_plan_sha256",
        ):
            _require_sha256(getattr(self, field), label=field)
        if self.fit_split_sha256 == self.eval_split_sha256:
            raise ValueError("promotion fit and evaluation splits must differ")
        if (
            type(self.interaction_artifact_sha256s) is not tuple
            or not self.interaction_artifact_sha256s
        ):
            raise ValueError(
                "interaction_artifact_sha256s must be a nonempty tuple"
            )
        for index, digest in enumerate(self.interaction_artifact_sha256s):
            _require_sha256(
                digest,
                label=f"interaction_artifact_sha256s[{index}]",
            )
        if len(set(self.interaction_artifact_sha256s)) != len(
            self.interaction_artifact_sha256s
        ):
            raise ValueError("promoted interaction hashes must be unique")
        _positive_int(
            self.conditional_interaction_count,
            label="conditional_interaction_count",
        )
        if self.conditional_interaction_count > self.interaction_count:
            raise ValueError(
                "conditional interaction count exceeds all interactions"
            )
        if self.selection_metric not in _PROMOTION_METRICS:
            raise ValueError("promotion selection_metric is invalid")
        baseline = _finite_nonnegative(
            self.baseline_metric_value,
            label="baseline_metric_value",
        )
        candidate = _finite_nonnegative(
            self.candidate_metric_value,
            label="candidate_metric_value",
        )
        minimum = _finite_nonnegative(
            self.minimum_heldout_improvement,
            label="minimum_heldout_improvement",
        )
        object.__setattr__(self, "baseline_metric_value", baseline)
        object.__setattr__(self, "candidate_metric_value", candidate)
        object.__setattr__(self, "minimum_heldout_improvement", minimum)
        _positive_int(self.heldout_observations, label="heldout_observations")
        if self.heldout_improvement <= max(0.0, minimum):
            raise ValueError(
                "conditional graph does not satisfy held-out promotion policy"
            )
        for field, expected in _SAFETY_METADATA.items():
            if getattr(self, field) is not expected:
                raise ValueError("interaction promotion safety metadata is invalid")
        if (
            self.artifact_kind != _PROMOTION_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("interaction graph promotion header is invalid")
        computed = _json_sha256(self._payload(), domain=_PROMOTION_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("interaction graph promotion hash mismatch")

    @property
    def interaction_count(self) -> int:
        return len(self.interaction_artifact_sha256s)

    @property
    def heldout_improvement(self) -> float:
        return self.baseline_metric_value - self.candidate_metric_value

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            **_SAFETY_METADATA,
            "source_model_sha256": self.source_model_sha256,
            "parameter_cluster_plan_sha256": (
                self.parameter_cluster_plan_sha256
            ),
            "fit_split_sha256": self.fit_split_sha256,
            "eval_split_sha256": self.eval_split_sha256,
            "eval_split_role": "open_development",
            "graph_plan_sha256": self.graph_plan_sha256,
            "interaction_artifact_sha256s": (
                self.interaction_artifact_sha256s
            ),
            "interaction_count": self.interaction_count,
            "conditional_interaction_count": (
                self.conditional_interaction_count
            ),
            "selection_metric": self.selection_metric,
            "baseline_condition": "edgeless_graph",
            "baseline_metric_value": self.baseline_metric_value,
            "candidate_metric_value": self.candidate_metric_value,
            "heldout_improvement": self.heldout_improvement,
            "minimum_heldout_improvement": (
                self.minimum_heldout_improvement
            ),
            "heldout_observations": self.heldout_observations,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    def validate_integrity(self) -> None:
        if (
            _json_sha256(self._payload(), domain=_PROMOTION_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("interaction graph promotion hash mismatch")

    def validate_against_graph(self, graph: ModalGeneratorGraphPlan) -> None:
        if not isinstance(graph, ModalGeneratorGraphPlan):
            raise TypeError("promotion graph must be ModalGeneratorGraphPlan")
        graph.validate_integrity()
        conditional_count = sum(
            isinstance(edge, StateConditionedModalGeneratorInteraction)
            for edge in graph.interactions
        )
        if (
            graph.artifact_sha256 != self.graph_plan_sha256
            or graph.model_fingerprint != self.source_model_sha256
            or graph.parameter_cluster_plan_sha256
            != self.parameter_cluster_plan_sha256
            or tuple(
                edge.artifact_sha256 for edge in graph.interactions
            )
            != self.interaction_artifact_sha256s
            or len(graph.interactions) != self.interaction_count
            or conditional_count != self.conditional_interaction_count
            or conditional_count <= 0
        ):
            raise ValueError(
                "interaction promotion does not authorize the exact graph"
            )

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalInteractionGraphPromotion:
        fields = {
            "artifact_kind",
            "format_version",
            *_SAFETY_METADATA,
            "source_model_sha256",
            "parameter_cluster_plan_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
            "eval_split_role",
            "graph_plan_sha256",
            "interaction_artifact_sha256s",
            "interaction_count",
            "conditional_interaction_count",
            "selection_metric",
            "baseline_condition",
            "baseline_metric_value",
            "candidate_metric_value",
            "heldout_improvement",
            "minimum_heldout_improvement",
            "heldout_observations",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="interaction graph promotion")
        if state["eval_split_role"] != "open_development":
            raise ValueError("promotion did not use open development")
        if state["baseline_condition"] != "edgeless_graph":
            raise ValueError("promotion baseline condition is invalid")
        for field, expected in _SAFETY_METADATA.items():
            if state[field] is not expected:
                raise ValueError(
                    "interaction promotion safety metadata is invalid"
                )
        result = cls(
            source_model_sha256=state[
                "source_model_sha256"
            ],  # type: ignore[arg-type]
            parameter_cluster_plan_sha256=state[
                "parameter_cluster_plan_sha256"
            ],  # type: ignore[arg-type]
            fit_split_sha256=state[
                "fit_split_sha256"
            ],  # type: ignore[arg-type]
            eval_split_sha256=state[
                "eval_split_sha256"
            ],  # type: ignore[arg-type]
            graph_plan_sha256=state[
                "graph_plan_sha256"
            ],  # type: ignore[arg-type]
            interaction_artifact_sha256s=state[
                "interaction_artifact_sha256s"
            ],  # type: ignore[arg-type]
            conditional_interaction_count=state[
                "conditional_interaction_count"
            ],  # type: ignore[arg-type]
            selection_metric=state[
                "selection_metric"
            ],  # type: ignore[arg-type]
            baseline_metric_value=state[
                "baseline_metric_value"
            ],  # type: ignore[arg-type]
            candidate_metric_value=state[
                "candidate_metric_value"
            ],  # type: ignore[arg-type]
            minimum_heldout_improvement=state[
                "minimum_heldout_improvement"
            ],  # type: ignore[arg-type]
            heldout_observations=state[
                "heldout_observations"
            ],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                field: state[field] for field in _SAFETY_METADATA
            },  # type: ignore[arg-type]
        )
        if (
            state["interaction_count"] != result.interaction_count
            or state["heldout_improvement"] != result.heldout_improvement
        ):
            raise ValueError("serialized promotion accounting is inconsistent")
        return result


ModalInteractionAuthorization = (
    ModalInteractionSelection | ModalInteractionGraphPromotion
)


def build_modal_interaction_graph_promotion(
    graph_plan: ModalGeneratorGraphPlan,
    *,
    fit_split_sha256: str,
    eval_split_sha256: str,
    selection_metric: str,
    baseline_metric_value: float,
    candidate_metric_value: float,
    minimum_heldout_improvement: float = 0.0,
    heldout_observations: int,
) -> ModalInteractionGraphPromotion:
    """Authenticate a frozen conditional or mixed graph promotion receipt."""

    if not isinstance(graph_plan, ModalGeneratorGraphPlan):
        raise TypeError("graph_plan must be ModalGeneratorGraphPlan")
    graph = ModalGeneratorGraphPlan.from_state_dict(graph_plan.state_dict())
    conditional_count = sum(
        isinstance(edge, StateConditionedModalGeneratorInteraction)
        for edge in graph.interactions
    )
    if conditional_count <= 0:
        raise ValueError(
            "graph promotion requires at least one state-conditioned interaction"
        )
    result = ModalInteractionGraphPromotion(
        source_model_sha256=graph.model_fingerprint,
        parameter_cluster_plan_sha256=(
            graph.parameter_cluster_plan_sha256
        ),
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
        graph_plan_sha256=graph.artifact_sha256,
        interaction_artifact_sha256s=tuple(
            edge.artifact_sha256 for edge in graph.interactions
        ),
        conditional_interaction_count=conditional_count,
        selection_metric=selection_metric,
        baseline_metric_value=baseline_metric_value,
        candidate_metric_value=candidate_metric_value,
        minimum_heldout_improvement=minimum_heldout_improvement,
        heldout_observations=heldout_observations,
    )
    result.validate_against_graph(graph)
    return result


def modal_interaction_authorization_from_state_dict(
    state: Mapping[str, object],
) -> ModalInteractionAuthorization:
    """Strictly restore one recognized interaction authorization kind."""

    if not isinstance(state, Mapping):
        raise TypeError("interaction authorization state must be a mapping")
    kind = state.get("artifact_kind")
    if kind == _LEGACY_SELECTION_KIND:
        return ModalInteractionSelection.from_state_dict(state)
    if kind == _PROMOTION_KIND:
        return ModalInteractionGraphPromotion.from_state_dict(state)
    raise ValueError(f"unsupported interaction authorization kind: {kind!r}")


def authenticate_modal_interaction_authorization(
    value: object,
) -> ModalInteractionAuthorization:
    """Validate, isolate, and strict-roundtrip a recognized authorization."""

    if not isinstance(
        value,
        (ModalInteractionSelection, ModalInteractionGraphPromotion),
    ):
        raise TypeError(
            "interaction authorization must be ModalInteractionSelection or "
            "ModalInteractionGraphPromotion"
        )
    value.validate_integrity()
    restored = modal_interaction_authorization_from_state_dict(
        value.state_dict()
    )
    restored.validate_integrity()
    if (
        type(restored) is not type(value)
        or restored.artifact_sha256 != value.artifact_sha256
    ):
        raise ValueError("interaction authorization roundtrip changed its hash")
    return restored
