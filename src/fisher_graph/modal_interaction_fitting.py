"""Fit authenticated interactions between frozen modal-generator states.

This module implements the fitted edge between two nodes in a modal-generator
graph.  Given a frozen source latent state ``X`` and a target latent residual
``R`` it fits the affine message

``message = X @ message_matrix + message_bias``.

The residual may be the desired compact computational mode minus its
standalone generator state, or the residual left after already-selected
incoming messages.  Fits use calibration rows only.  A disjoint development
split measures held-out improvement, and the greedy selector refuses to use a
closed guard/test split for model selection.

Artifacts retain only learned edge coefficients, aggregate metrics, provenance
hashes, and exact resource accounting.  Raw latent and residual rows are never
retained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .modal_generator_graph import ModalGeneratorInteraction


__all__ = [
    "ModalInteractionBinding",
    "ModalInteractionCandidate",
    "ModalInteractionFactors",
    "ModalInteractionFitConfig",
    "ModalInteractionMetrics",
    "ModalInteractionRateCurve",
    "ModalInteractionSelection",
    "ModalInteractionSelectionPolicy",
    "ModalInteractionSelectionStep",
    "fit_modal_interaction_rate_curve",
    "select_modal_interactions_greedily",
]


_FORMAT_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_BINDING_KIND = "fisher_graph.modal_interaction_binding"
_CONFIG_KIND = "fisher_graph.modal_interaction_fit_config"
_FACTORS_KIND = "fisher_graph.modal_interaction_factors"
_CANDIDATE_KIND = "fisher_graph.modal_interaction_candidate"
_CURVE_KIND = "fisher_graph.modal_interaction_rate_curve"
_POLICY_KIND = "fisher_graph.modal_interaction_selection_policy"
_STEP_KIND = "fisher_graph.modal_interaction_selection_step"
_SELECTION_KIND = "fisher_graph.modal_interaction_selection"

_BINDING_DOMAIN = b"fisher_graph.modal_interaction.binding.v1\0"
_CONFIG_DOMAIN = b"fisher_graph.modal_interaction.config.v1\0"
_FACTORS_DOMAIN = b"fisher_graph.modal_interaction.factors.v1\0"
_CANDIDATE_DOMAIN = b"fisher_graph.modal_interaction.candidate.v1\0"
_CURVE_DOMAIN = b"fisher_graph.modal_interaction.curve.v1\0"
_POLICY_DOMAIN = b"fisher_graph.modal_interaction.policy.v1\0"
_STEP_DOMAIN = b"fisher_graph.modal_interaction.step.v1\0"
_SELECTION_DOMAIN = b"fisher_graph.modal_interaction.selection.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.modal_interaction.tensor.v1\0"

_EVAL_SPLIT_ROLES = frozenset(
    {
        "open_development",
        "closed_guard",
        "closed_test",
        "descriptive_holdout",
    }
)
_SELECTION_METRICS = frozenset({"nrmse", "weighted_nrmse"})
_SAFETY_METADATA: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_prompt_text": False,
    "contains_raw_latent_rows": False,
    "contains_target_residual_rows": False,
    "contains_generator_weights": False,
    "contains_interaction_weights": True,
    "executable": True,
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


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical source-safe name")
    return value


def _require_int(value: object, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _strict_fields(
    state: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


def _as_matrix(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(f"{label} must be a nonempty floating matrix")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result.clone()


def _as_vector(value: Tensor, *, label: str, length: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.shape[0] != length
        or not value.is_floating_point()
    ):
        raise ValueError(
            f"{label} must be a floating vector with shape ({length},)"
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result.clone()


def _as_weights(
    value: Tensor | None,
    *,
    observations: int,
    label: str,
) -> Tensor:
    if value is None:
        return torch.ones(observations, dtype=torch.float64)
    result = _as_vector(value, label=label, length=observations)
    if bool((result < 0).any()) or float(result.sum().item()) <= 0.0:
        raise ValueError(f"{label} must have finite nonnegative positive mass")
    return result


def _tensor_sha256(value: Tensor, *, label: str) -> str:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"{label} must be a finite contiguous CPU float64 Tensor"
        )
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        f"{tuple(value.shape)}\0float64\0".encode("utf-8")
    )
    digest.update(value.detach().numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModalInteractionBinding:
    """Authenticated identity and causal provenance for one candidate edge."""

    source_node: str
    target_node: str
    source_causal_order: int
    target_causal_order: int
    source_model_sha256: str
    parameter_cluster_plan_sha256: str
    source_generator_sha256: str
    target_generator_sha256: str
    fit_split_sha256: str
    eval_split_sha256: str
    eval_split_role: str = "open_development"
    artifact_sha256: str = ""
    artifact_kind: str = _BINDING_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        source = _require_name(self.source_node, label="source_node")
        target = _require_name(self.target_node, label="target_node")
        if source == target:
            raise ValueError("modal interaction cannot be a self-edge")
        source_order = _require_int(
            self.source_causal_order,
            label="source_causal_order",
            minimum=0,
        )
        target_order = _require_int(
            self.target_causal_order,
            label="target_causal_order",
            minimum=0,
        )
        if source_order >= target_order:
            raise ValueError(
                "modal interactions must point strictly forward in causal order"
            )
        for field in (
            "source_model_sha256",
            "parameter_cluster_plan_sha256",
            "source_generator_sha256",
            "target_generator_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
        ):
            _require_sha256(getattr(self, field), label=field)
        if self.fit_split_sha256 == self.eval_split_sha256:
            raise ValueError(
                "fit and evaluation split hashes must differ; hashes alone "
                "do not prove membership disjointness"
            )
        if self.eval_split_role not in _EVAL_SPLIT_ROLES:
            raise ValueError("eval_split_role is invalid")
        if (
            self.artifact_kind != _BINDING_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal interaction binding header is invalid")
        computed = _json_sha256(self._payload(), domain=_BINDING_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("modal interaction binding hash mismatch")

    @property
    def key(self) -> str:
        return f"{self.source_node}->{self.target_node}"

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "source_causal_order": self.source_causal_order,
            "target_causal_order": self.target_causal_order,
            "source_model_sha256": self.source_model_sha256,
            "parameter_cluster_plan_sha256": (
                self.parameter_cluster_plan_sha256
            ),
            "source_generator_sha256": self.source_generator_sha256,
            "target_generator_sha256": self.target_generator_sha256,
            "fit_split_sha256": self.fit_split_sha256,
            "eval_split_sha256": self.eval_split_sha256,
            "eval_split_role": self.eval_split_role,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    metadata = state_dict

    def validate_integrity(self) -> None:
        if (
            _json_sha256(self._payload(), domain=_BINDING_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("modal interaction binding hash mismatch")

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalInteractionBinding:
        fields = {
            "artifact_kind",
            "format_version",
            "source_node",
            "target_node",
            "source_causal_order",
            "target_causal_order",
            "source_model_sha256",
            "parameter_cluster_plan_sha256",
            "source_generator_sha256",
            "target_generator_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
            "eval_split_role",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="modal interaction binding")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModalInteractionFitConfig:
    """Predeclared affine-ridge ladder for an interaction candidate."""

    ridges: tuple[float, ...]
    fit_intercept: bool = True
    artifact_sha256: str = ""
    artifact_kind: str = _CONFIG_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if type(self.ridges) is not tuple or not self.ridges:
            raise ValueError("ridges must be a nonempty tuple")
        ridges = tuple(
            _require_float(value, label="ridge", minimum=0.0)
            for value in self.ridges
        )
        if ridges != tuple(sorted(set(ridges))):
            raise ValueError("ridges must be unique and increasing")
        object.__setattr__(self, "ridges", ridges)
        if type(self.fit_intercept) is not bool:
            raise TypeError("fit_intercept must be bool")
        if (
            self.artifact_kind != _CONFIG_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal interaction fit config header is invalid")
        computed = _json_sha256(self._payload(), domain=_CONFIG_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("modal interaction fit config hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "ridges": self.ridges,
            "fit_intercept": self.fit_intercept,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    metadata = state_dict

    def validate_integrity(self) -> None:
        if (
            _json_sha256(self._payload(), domain=_CONFIG_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("modal interaction fit config hash mismatch")

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalInteractionFitConfig:
        fields = {
            "artifact_kind",
            "format_version",
            "ridges",
            "fit_intercept",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="modal interaction fit config")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModalInteractionFactors:
    """Copied affine message coefficients for one graph edge."""

    message_matrix: Tensor
    message_bias: Tensor
    message_matrix_sha256: str = ""
    message_bias_sha256: str = ""
    artifact_sha256: str = ""
    artifact_kind: str = _FACTORS_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        matrix = _as_matrix(self.message_matrix, label="message_matrix")
        bias = _as_vector(
            self.message_bias,
            label="message_bias",
            length=matrix.shape[1],
        )
        object.__setattr__(self, "message_matrix", matrix)
        object.__setattr__(self, "message_bias", bias)
        for field, value in (
            ("message_matrix_sha256", matrix),
            ("message_bias_sha256", bias),
        ):
            computed = _tensor_sha256(value, label=field)
            supplied = getattr(self, field)
            if supplied == "":
                object.__setattr__(self, field, computed)
            elif _require_sha256(supplied, label=field) != computed:
                raise ValueError(f"{field.removesuffix('_sha256')} hash mismatch")
        if (
            self.artifact_kind != _FACTORS_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal interaction factors header is invalid")
        computed = _json_sha256(self._payload(), domain=_FACTORS_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("modal interaction factors hash mismatch")

    @property
    def source_width(self) -> int:
        return int(self.message_matrix.shape[0])

    @property
    def target_width(self) -> int:
        return int(self.message_matrix.shape[1])

    @property
    def parameter_count(self) -> int:
        # ModalGeneratorInteraction always owns a target-width bias vector.
        return self.source_width * self.target_width + self.target_width

    @property
    def macs_per_token(self) -> int:
        return self.source_width * self.target_width

    @property
    def bias_additions_per_token(self) -> int:
        return self.target_width

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "message_matrix_sha256": self.message_matrix_sha256,
            "message_bias_sha256": self.message_bias_sha256,
            "parameter_count": self.parameter_count,
            "macs_per_token": self.macs_per_token,
            "bias_additions_per_token": self.bias_additions_per_token,
        }

    def validate_integrity(self) -> None:
        if (
            _tensor_sha256(
                self.message_matrix,
                label="message_matrix",
            )
            != self.message_matrix_sha256
        ):
            raise ValueError("message_matrix hash mismatch")
        if (
            _tensor_sha256(self.message_bias, label="message_bias")
            != self.message_bias_sha256
        ):
            raise ValueError("message_bias hash mismatch")
        if (
            _json_sha256(self._payload(), domain=_FACTORS_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("modal interaction factors hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "message_matrix": self.message_matrix.clone(),
            "message_bias": self.message_bias.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalInteractionFactors:
        fields = {
            "artifact_kind",
            "format_version",
            "source_width",
            "target_width",
            "message_matrix_sha256",
            "message_bias_sha256",
            "parameter_count",
            "macs_per_token",
            "bias_additions_per_token",
            "message_matrix",
            "message_bias",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="modal interaction factors")
        result = cls(
            message_matrix=state["message_matrix"],  # type: ignore[arg-type]
            message_bias=state["message_bias"],  # type: ignore[arg-type]
            message_matrix_sha256=state[
                "message_matrix_sha256"
            ],  # type: ignore[arg-type]
            message_bias_sha256=state[
                "message_bias_sha256"
            ],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        for field, actual in (
            ("source_width", result.source_width),
            ("target_width", result.target_width),
            ("parameter_count", result.parameter_count),
            ("macs_per_token", result.macs_per_token),
            ("bias_additions_per_token", result.bias_additions_per_token),
        ):
            if state[field] != actual:
                raise ValueError(f"serialized {field} is inconsistent")
        return result


@dataclass(frozen=True, slots=True)
class ModalInteractionMetrics:
    """Unweighted and Fisher-weighted reconstruction metrics."""

    observations: int
    mse: float
    nrmse: float
    weighted_mse: float
    weighted_nrmse: float
    cosine_similarity: float
    weighted_cosine_similarity: float
    target_rms: float
    weighted_target_rms: float
    max_abs_error: float

    def __post_init__(self) -> None:
        _require_int(self.observations, label="observations", minimum=1)
        for field in (
            "mse",
            "nrmse",
            "weighted_mse",
            "weighted_nrmse",
            "target_rms",
            "weighted_target_rms",
            "max_abs_error",
        ):
            object.__setattr__(
                self,
                field,
                _require_float(
                    getattr(self, field),
                    label=field,
                    minimum=0.0,
                ),
            )
        for field in ("cosine_similarity", "weighted_cosine_similarity"):
            value = _require_float(getattr(self, field), label=field)
            if value < -1.000000000001 or value > 1.000000000001:
                raise ValueError(f"{field} must be in [-1, 1]")
            object.__setattr__(self, field, max(-1.0, min(1.0, value)))

    def metadata(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "mse": self.mse,
            "nrmse": self.nrmse,
            "weighted_mse": self.weighted_mse,
            "weighted_nrmse": self.weighted_nrmse,
            "cosine_similarity": self.cosine_similarity,
            "weighted_cosine_similarity": (
                self.weighted_cosine_similarity
            ),
            "target_rms": self.target_rms,
            "weighted_target_rms": self.weighted_target_rms,
            "max_abs_error": self.max_abs_error,
        }

    state_dict = metadata

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalInteractionMetrics:
        fields = {
            "observations",
            "mse",
            "nrmse",
            "weighted_mse",
            "weighted_nrmse",
            "cosine_similarity",
            "weighted_cosine_similarity",
            "target_rms",
            "weighted_target_rms",
            "max_abs_error",
        }
        _strict_fields(state, fields, label="modal interaction metrics")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModalInteractionCandidate:
    """One ridge candidate with authenticated factors and split metrics."""

    binding: ModalInteractionBinding
    config: ModalInteractionFitConfig
    ridge: float
    factors: ModalInteractionFactors
    fit_metrics: ModalInteractionMetrics
    eval_metrics: ModalInteractionMetrics
    artifact_sha256: str = ""
    artifact_kind: str = _CANDIDATE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ModalInteractionBinding):
            raise TypeError("binding must be ModalInteractionBinding")
        if not isinstance(self.config, ModalInteractionFitConfig):
            raise TypeError("config must be ModalInteractionFitConfig")
        if not isinstance(self.factors, ModalInteractionFactors):
            raise TypeError("factors must be ModalInteractionFactors")
        if not isinstance(self.fit_metrics, ModalInteractionMetrics):
            raise TypeError("fit_metrics must be ModalInteractionMetrics")
        if not isinstance(self.eval_metrics, ModalInteractionMetrics):
            raise TypeError("eval_metrics must be ModalInteractionMetrics")
        self.binding.validate_integrity()
        self.config.validate_integrity()
        ridge = _require_float(self.ridge, label="ridge", minimum=0.0)
        object.__setattr__(self, "ridge", ridge)
        if ridge not in self.config.ridges:
            raise ValueError("candidate ridge is not in configured ladder")
        self.factors.validate_integrity()
        if (
            self.artifact_kind != _CANDIDATE_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal interaction candidate header is invalid")
        computed = _json_sha256(self._payload(), domain=_CANDIDATE_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("modal interaction candidate hash mismatch")

    @property
    def parameter_count(self) -> int:
        return self.factors.parameter_count

    @property
    def macs_per_token(self) -> int:
        return self.factors.macs_per_token

    @property
    def bias_additions_per_token(self) -> int:
        return self.factors.bias_additions_per_token

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "binding_sha256": self.binding.artifact_sha256,
            "config_sha256": self.config.artifact_sha256,
            "ridge": self.ridge,
            "factors_sha256": self.factors.artifact_sha256,
            "fit_metrics": self.fit_metrics.metadata(),
            "eval_metrics": self.eval_metrics.metadata(),
            "parameter_count": self.parameter_count,
            "macs_per_token": self.macs_per_token,
            "bias_additions_per_token": self.bias_additions_per_token,
        }

    def validate_integrity(self) -> None:
        self.binding.validate_integrity()
        self.config.validate_integrity()
        self.factors.validate_integrity()
        if (
            _json_sha256(self._payload(), domain=_CANDIDATE_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("modal interaction candidate hash mismatch")

    def to_graph_interaction(self) -> ModalGeneratorInteraction:
        """Copy this fitted candidate into the graph executor edge type."""

        self.validate_integrity()
        return ModalGeneratorInteraction(
            source_node=self.binding.source_node,
            target_node=self.binding.target_node,
            message_matrix=self.factors.message_matrix,
            message_bias=self.factors.message_bias,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "binding": self.binding.state_dict(),
            "config": self.config.state_dict(),
            "factors": self.factors.state_dict(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalInteractionCandidate:
        fields = {
            "artifact_kind",
            "format_version",
            "binding_sha256",
            "config_sha256",
            "ridge",
            "factors_sha256",
            "fit_metrics",
            "eval_metrics",
            "parameter_count",
            "macs_per_token",
            "bias_additions_per_token",
            "binding",
            "config",
            "factors",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="modal interaction candidate")
        binding = ModalInteractionBinding.from_state_dict(
            state["binding"]  # type: ignore[arg-type]
        )
        config = ModalInteractionFitConfig.from_state_dict(
            state["config"]  # type: ignore[arg-type]
        )
        factors = ModalInteractionFactors.from_state_dict(
            state["factors"]  # type: ignore[arg-type]
        )
        for field, actual in (
            ("binding_sha256", binding.artifact_sha256),
            ("config_sha256", config.artifact_sha256),
            ("factors_sha256", factors.artifact_sha256),
        ):
            if state[field] != actual:
                raise ValueError(f"{field} does not match nested artifact")
        result = cls(
            binding=binding,
            config=config,
            ridge=state["ridge"],  # type: ignore[arg-type]
            factors=factors,
            fit_metrics=ModalInteractionMetrics.from_state_dict(
                state["fit_metrics"]  # type: ignore[arg-type]
            ),
            eval_metrics=ModalInteractionMetrics.from_state_dict(
                state["eval_metrics"]  # type: ignore[arg-type]
            ),
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        for field, actual in (
            ("parameter_count", result.parameter_count),
            ("macs_per_token", result.macs_per_token),
            ("bias_additions_per_token", result.bias_additions_per_token),
        ):
            if state[field] != actual:
                raise ValueError(f"serialized {field} is inconsistent")
        return result


@dataclass(frozen=True, slots=True)
class ModalInteractionRateCurve:
    """Authenticated ridge ladder with a literal no-edge baseline."""

    binding: ModalInteractionBinding
    config: ModalInteractionFitConfig
    source_width: int
    target_width: int
    zero_fit_metrics: ModalInteractionMetrics
    zero_eval_metrics: ModalInteractionMetrics
    candidates: tuple[ModalInteractionCandidate, ...]
    artifact_sha256: str = ""
    artifact_kind: str = _CURVE_KIND
    format_version: int = _FORMAT_VERSION
    contains_source_model_weights: bool = False
    contains_prompt_text: bool = False
    contains_raw_latent_rows: bool = False
    contains_target_residual_rows: bool = False
    contains_generator_weights: bool = False
    contains_interaction_weights: bool = True
    executable: bool = True
    tuned_on_closed_split: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ModalInteractionBinding):
            raise TypeError("binding must be ModalInteractionBinding")
        if not isinstance(self.config, ModalInteractionFitConfig):
            raise TypeError("config must be ModalInteractionFitConfig")
        source_width = _require_int(
            self.source_width,
            label="source_width",
            minimum=1,
        )
        target_width = _require_int(
            self.target_width,
            label="target_width",
            minimum=1,
        )
        if not isinstance(self.zero_fit_metrics, ModalInteractionMetrics):
            raise TypeError("zero_fit_metrics must be ModalInteractionMetrics")
        if not isinstance(self.zero_eval_metrics, ModalInteractionMetrics):
            raise TypeError("zero_eval_metrics must be ModalInteractionMetrics")
        if (
            type(self.candidates) is not tuple
            or tuple(candidate.ridge for candidate in self.candidates)
            != self.config.ridges
        ):
            raise ValueError("candidates must exactly follow configured ridges")
        for candidate in self.candidates:
            if (
                not isinstance(candidate, ModalInteractionCandidate)
                or candidate.binding.artifact_sha256
                != self.binding.artifact_sha256
                or candidate.config.artifact_sha256
                != self.config.artifact_sha256
                or candidate.factors.source_width != source_width
                or candidate.factors.target_width != target_width
            ):
                raise ValueError("candidate does not match its rate curve")
        for field, expected in _SAFETY_METADATA.items():
            if getattr(self, field) is not expected:
                raise ValueError("modal interaction safety metadata is invalid")
        if (
            self.artifact_kind != _CURVE_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal interaction curve header is invalid")
        computed = _json_sha256(self._payload(), domain=_CURVE_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("modal interaction curve hash mismatch")

    @property
    def parameter_count(self) -> int:
        return self.candidates[0].parameter_count

    @property
    def macs_per_token(self) -> int:
        return self.candidates[0].macs_per_token

    def candidate_for_ridge(self, ridge: float) -> ModalInteractionCandidate:
        value = float(ridge)
        for candidate in self.candidates:
            if candidate.ridge == value:
                return candidate
        raise KeyError(f"ridge {value} is not in the interaction curve")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            **_SAFETY_METADATA,
            "binding_sha256": self.binding.artifact_sha256,
            "config_sha256": self.config.artifact_sha256,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "zero_fit_metrics": self.zero_fit_metrics.metadata(),
            "zero_eval_metrics": self.zero_eval_metrics.metadata(),
            "candidate_sha256s": tuple(
                candidate.artifact_sha256 for candidate in self.candidates
            ),
            "parameter_count": self.parameter_count,
            "macs_per_token": self.macs_per_token,
        }

    def validate_integrity(self) -> None:
        for candidate in self.candidates:
            candidate.validate_integrity()
        if (
            _json_sha256(self._payload(), domain=_CURVE_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("modal interaction curve hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "binding": self.binding.state_dict(),
            "config": self.config.state_dict(),
            "candidates": tuple(
                candidate.state_dict() for candidate in self.candidates
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    metadata = _payload

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalInteractionRateCurve:
        fields = {
            "artifact_kind",
            "format_version",
            *_SAFETY_METADATA,
            "binding_sha256",
            "config_sha256",
            "source_width",
            "target_width",
            "zero_fit_metrics",
            "zero_eval_metrics",
            "candidate_sha256s",
            "parameter_count",
            "macs_per_token",
            "binding",
            "config",
            "candidates",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="modal interaction curve")
        for field, expected in _SAFETY_METADATA.items():
            if state[field] is not expected:
                raise ValueError("modal interaction safety metadata is invalid")
        binding = ModalInteractionBinding.from_state_dict(
            state["binding"]  # type: ignore[arg-type]
        )
        config = ModalInteractionFitConfig.from_state_dict(
            state["config"]  # type: ignore[arg-type]
        )
        raw_candidates = state["candidates"]
        if type(raw_candidates) is not tuple:
            raise TypeError("serialized candidates must be a tuple")
        candidates = tuple(
            ModalInteractionCandidate.from_state_dict(value)
            for value in raw_candidates  # type: ignore[arg-type]
        )
        if state["binding_sha256"] != binding.artifact_sha256:
            raise ValueError("binding_sha256 does not match nested binding")
        if state["config_sha256"] != config.artifact_sha256:
            raise ValueError("config_sha256 does not match nested config")
        if state["candidate_sha256s"] != tuple(
            candidate.artifact_sha256 for candidate in candidates
        ):
            raise ValueError(
                "candidate_sha256s do not match nested candidates"
            )
        result = cls(
            binding=binding,
            config=config,
            source_width=state["source_width"],  # type: ignore[arg-type]
            target_width=state["target_width"],  # type: ignore[arg-type]
            zero_fit_metrics=ModalInteractionMetrics.from_state_dict(
                state["zero_fit_metrics"]  # type: ignore[arg-type]
            ),
            zero_eval_metrics=ModalInteractionMetrics.from_state_dict(
                state["zero_eval_metrics"]  # type: ignore[arg-type]
            ),
            candidates=candidates,
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                field: state[field] for field in _SAFETY_METADATA
            },  # type: ignore[arg-type]
        )
        if (
            state["parameter_count"] != result.parameter_count
            or state["macs_per_token"] != result.macs_per_token
        ):
            raise ValueError("serialized curve accounting is inconsistent")
        return result


def _cosine(
    target: Tensor,
    prediction: Tensor,
    weights: Tensor | None,
) -> float:
    if weights is None:
        left = target.reshape(-1)
        right = prediction.reshape(-1)
    else:
        root = weights.sqrt().unsqueeze(1)
        left = (target * root).reshape(-1)
        right = (prediction * root).reshape(-1)
    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(torch.dot(left, right).item()) / (left_norm * right_norm)


def _metrics(
    target: Tensor,
    prediction: Tensor,
    weights: Tensor,
) -> ModalInteractionMetrics:
    error = prediction - target
    mse = float(error.square().mean().item())
    target_power = float(target.square().mean().item())
    normalized_weights = weights / weights.sum()
    weighted_mse = float(
        torch.dot(
            normalized_weights,
            error.square().mean(dim=1),
        ).item()
    )
    weighted_target_power = float(
        torch.dot(
            normalized_weights,
            target.square().mean(dim=1),
        ).item()
    )
    def normalized_rmse(error_power: float, reference_power: float) -> float:
        if reference_power == 0.0:
            if error_power == 0.0:
                return 0.0
            reference_power = torch.finfo(torch.float64).tiny
        return math.sqrt(error_power / reference_power)

    return ModalInteractionMetrics(
        observations=target.shape[0],
        mse=mse,
        nrmse=normalized_rmse(mse, target_power),
        weighted_mse=weighted_mse,
        weighted_nrmse=normalized_rmse(
            weighted_mse,
            weighted_target_power,
        ),
        cosine_similarity=_cosine(target, prediction, None),
        weighted_cosine_similarity=_cosine(
            target,
            prediction,
            normalized_weights,
        ),
        target_rms=math.sqrt(target_power),
        weighted_target_rms=math.sqrt(weighted_target_power),
        max_abs_error=float(error.abs().max().item()),
    )


def _fit_affine_ridge(
    source: Tensor,
    target: Tensor,
    weights: Tensor,
    *,
    ridge: float,
    fit_intercept: bool,
) -> ModalInteractionFactors:
    normalized_weights = weights / weights.sum()
    if fit_intercept:
        x_mean = torch.sum(
            source * normalized_weights.unsqueeze(1),
            dim=0,
        )
        y_mean = torch.sum(
            target * normalized_weights.unsqueeze(1),
            dim=0,
        )
        centered_source = source - x_mean
        centered_target = target - y_mean
    else:
        x_mean = torch.zeros(source.shape[1], dtype=torch.float64)
        y_mean = torch.zeros(target.shape[1], dtype=torch.float64)
        centered_source = source
        centered_target = target
    root_weight = normalized_weights.sqrt().unsqueeze(1)
    design = centered_source * root_weight
    response = centered_target * root_weight
    if ridge == 0.0:
        matrix = torch.linalg.lstsq(
            design,
            response,
            driver="gelsd",
        ).solution
    else:
        gram = design.T @ design
        regularized = gram + ridge * torch.eye(
            gram.shape[0],
            dtype=torch.float64,
        )
        matrix = torch.linalg.solve(regularized, design.T @ response)
    bias = y_mean - x_mean @ matrix if fit_intercept else y_mean
    return ModalInteractionFactors(
        message_matrix=matrix,
        message_bias=bias,
    )


def _prepare_pair(
    source_fit: Tensor,
    target_residual_fit: Tensor,
    source_eval: Tensor,
    target_residual_eval: Tensor,
    fisher_weights_fit: Tensor | None,
    fisher_weights_eval: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    X_fit = _as_matrix(source_fit, label="source_fit")
    R_fit = _as_matrix(target_residual_fit, label="target_residual_fit")
    X_eval = _as_matrix(source_eval, label="source_eval")
    R_eval = _as_matrix(target_residual_eval, label="target_residual_eval")
    if X_fit.shape[0] != R_fit.shape[0]:
        raise ValueError("fit source and target row counts must match")
    if X_eval.shape[0] != R_eval.shape[0]:
        raise ValueError("eval source and target row counts must match")
    if X_fit.shape[1] != X_eval.shape[1]:
        raise ValueError("fit and eval source widths must match")
    if R_fit.shape[1] != R_eval.shape[1]:
        raise ValueError("fit and eval target widths must match")
    fit_weights = _as_weights(
        fisher_weights_fit,
        observations=X_fit.shape[0],
        label="fisher_weights_fit",
    )
    eval_weights = _as_weights(
        fisher_weights_eval,
        observations=X_eval.shape[0],
        label="fisher_weights_eval",
    )
    return X_fit, R_fit, X_eval, R_eval, fit_weights, eval_weights


def fit_modal_interaction_rate_curve(
    source_fit: Tensor,
    target_residual_fit: Tensor,
    source_eval: Tensor,
    target_residual_eval: Tensor,
    *,
    binding: ModalInteractionBinding,
    ridges: Sequence[float] | float = (0.0,),
    fisher_weights_fit: Tensor | None = None,
    fisher_weights_eval: Tensor | None = None,
    fit_intercept: bool = True,
) -> ModalInteractionRateCurve:
    """Fit a fixed ridge or predeclared ridge ladder on calibration rows only."""

    if not isinstance(binding, ModalInteractionBinding):
        raise TypeError("binding must be ModalInteractionBinding")
    if isinstance(ridges, (int, float)) and not isinstance(ridges, bool):
        ridge_values = (float(ridges),)
    else:
        ridge_values = tuple(ridges)  # type: ignore[arg-type]
    config = ModalInteractionFitConfig(
        ridges=ridge_values,
        fit_intercept=fit_intercept,
    )
    (
        X_fit,
        R_fit,
        X_eval,
        R_eval,
        fit_weights,
        eval_weights,
    ) = _prepare_pair(
        source_fit,
        target_residual_fit,
        source_eval,
        target_residual_eval,
        fisher_weights_fit,
        fisher_weights_eval,
    )
    zero_fit = torch.zeros_like(R_fit)
    zero_eval = torch.zeros_like(R_eval)
    candidates: list[ModalInteractionCandidate] = []
    for ridge in config.ridges:
        factors = _fit_affine_ridge(
            X_fit,
            R_fit,
            fit_weights,
            ridge=ridge,
            fit_intercept=config.fit_intercept,
        )
        fit_prediction = (
            X_fit @ factors.message_matrix + factors.message_bias
        )
        eval_prediction = (
            X_eval @ factors.message_matrix + factors.message_bias
        )
        candidates.append(
            ModalInteractionCandidate(
                binding=binding,
                config=config,
                ridge=ridge,
                factors=factors,
                fit_metrics=_metrics(R_fit, fit_prediction, fit_weights),
                eval_metrics=_metrics(R_eval, eval_prediction, eval_weights),
            )
        )
    return ModalInteractionRateCurve(
        binding=binding,
        config=config,
        source_width=X_fit.shape[1],
        target_width=R_fit.shape[1],
        zero_fit_metrics=_metrics(R_fit, zero_fit, fit_weights),
        zero_eval_metrics=_metrics(R_eval, zero_eval, eval_weights),
        candidates=tuple(candidates),
    )


@dataclass(frozen=True, slots=True)
class ModalInteractionSelectionPolicy:
    """Authenticated held-out greedy-selection policy."""

    fit_config: ModalInteractionFitConfig
    selection_metric: str
    minimum_heldout_improvement: float
    max_incoming_edges: int
    artifact_sha256: str = ""
    artifact_kind: str = _POLICY_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.fit_config, ModalInteractionFitConfig):
            raise TypeError("fit_config must be ModalInteractionFitConfig")
        self.fit_config.validate_integrity()
        if self.selection_metric not in _SELECTION_METRICS:
            raise ValueError("selection_metric is invalid")
        improvement = _require_float(
            self.minimum_heldout_improvement,
            label="minimum_heldout_improvement",
            minimum=0.0,
        )
        object.__setattr__(
            self,
            "minimum_heldout_improvement",
            improvement,
        )
        _require_int(
            self.max_incoming_edges,
            label="max_incoming_edges",
            minimum=1,
        )
        if (
            self.artifact_kind != _POLICY_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal interaction selection policy is invalid")
        computed = _json_sha256(self._payload(), domain=_POLICY_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("modal interaction selection policy hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "fit_config_sha256": self.fit_config.artifact_sha256,
            "selection_metric": self.selection_metric,
            "minimum_heldout_improvement": (
                self.minimum_heldout_improvement
            ),
            "max_incoming_edges": self.max_incoming_edges,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "fit_config": self.fit_config.state_dict(),
            "artifact_sha256": self.artifact_sha256,
        }

    def validate_integrity(self) -> None:
        self.fit_config.validate_integrity()
        if (
            _json_sha256(self._payload(), domain=_POLICY_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("modal interaction selection policy hash mismatch")

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalInteractionSelectionPolicy:
        fields = {
            "artifact_kind",
            "format_version",
            "fit_config_sha256",
            "selection_metric",
            "minimum_heldout_improvement",
            "max_incoming_edges",
            "fit_config",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="interaction selection policy")
        fit_config = ModalInteractionFitConfig.from_state_dict(
            state["fit_config"]  # type: ignore[arg-type]
        )
        if state["fit_config_sha256"] != fit_config.artifact_sha256:
            raise ValueError(
                "fit_config_sha256 does not match nested fit config"
            )
        return cls(
            fit_config=fit_config,
            selection_metric=state["selection_metric"],  # type: ignore[arg-type]
            minimum_heldout_improvement=state[
                "minimum_heldout_improvement"
            ],  # type: ignore[arg-type]
            max_incoming_edges=state[
                "max_incoming_edges"
            ],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ModalInteractionSelectionStep:
    """One accepted edge and its cumulative held-out improvement."""

    step_index: int
    candidate: ModalInteractionCandidate
    cumulative_fit_metrics_before: ModalInteractionMetrics
    cumulative_fit_metrics_after: ModalInteractionMetrics
    cumulative_eval_metrics_before: ModalInteractionMetrics
    cumulative_eval_metrics_after: ModalInteractionMetrics
    selection_metric: str
    heldout_improvement: float
    artifact_sha256: str = ""
    artifact_kind: str = _STEP_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_int(self.step_index, label="step_index", minimum=0)
        if not isinstance(self.candidate, ModalInteractionCandidate):
            raise TypeError("candidate must be ModalInteractionCandidate")
        self.candidate.validate_integrity()
        for field in (
            "cumulative_fit_metrics_before",
            "cumulative_fit_metrics_after",
            "cumulative_eval_metrics_before",
            "cumulative_eval_metrics_after",
        ):
            if not isinstance(getattr(self, field), ModalInteractionMetrics):
                raise TypeError(f"{field} must be ModalInteractionMetrics")
        if self.selection_metric not in _SELECTION_METRICS:
            raise ValueError("selection_metric is invalid")
        improvement = _require_float(
            self.heldout_improvement,
            label="heldout_improvement",
            minimum=0.0,
        )
        expected = (
            getattr(
                self.cumulative_eval_metrics_before,
                self.selection_metric,
            )
            - getattr(
                self.cumulative_eval_metrics_after,
                self.selection_metric,
            )
        )
        if not math.isclose(improvement, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("heldout_improvement does not match metrics")
        object.__setattr__(self, "heldout_improvement", improvement)
        if (
            self.artifact_kind != _STEP_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal interaction selection step is invalid")
        computed = _json_sha256(self._payload(), domain=_STEP_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("modal interaction selection step hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "step_index": self.step_index,
            "candidate_sha256": self.candidate.artifact_sha256,
            "cumulative_fit_metrics_before": (
                self.cumulative_fit_metrics_before.metadata()
            ),
            "cumulative_fit_metrics_after": (
                self.cumulative_fit_metrics_after.metadata()
            ),
            "cumulative_eval_metrics_before": (
                self.cumulative_eval_metrics_before.metadata()
            ),
            "cumulative_eval_metrics_after": (
                self.cumulative_eval_metrics_after.metadata()
            ),
            "selection_metric": self.selection_metric,
            "heldout_improvement": self.heldout_improvement,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "candidate": self.candidate.state_dict(),
            "artifact_sha256": self.artifact_sha256,
        }

    def validate_integrity(self) -> None:
        self.candidate.validate_integrity()
        if (
            _json_sha256(self._payload(), domain=_STEP_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("modal interaction selection step hash mismatch")

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalInteractionSelectionStep:
        fields = {
            "artifact_kind",
            "format_version",
            "step_index",
            "candidate_sha256",
            "cumulative_fit_metrics_before",
            "cumulative_fit_metrics_after",
            "cumulative_eval_metrics_before",
            "cumulative_eval_metrics_after",
            "selection_metric",
            "heldout_improvement",
            "candidate",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="interaction selection step")
        candidate = ModalInteractionCandidate.from_state_dict(
            state["candidate"]  # type: ignore[arg-type]
        )
        if state["candidate_sha256"] != candidate.artifact_sha256:
            raise ValueError(
                "candidate_sha256 does not match nested candidate"
            )
        return cls(
            step_index=state["step_index"],  # type: ignore[arg-type]
            candidate=candidate,
            cumulative_fit_metrics_before=(
                ModalInteractionMetrics.from_state_dict(
                    state["cumulative_fit_metrics_before"]  # type: ignore[arg-type]
                )
            ),
            cumulative_fit_metrics_after=(
                ModalInteractionMetrics.from_state_dict(
                    state["cumulative_fit_metrics_after"]  # type: ignore[arg-type]
                )
            ),
            cumulative_eval_metrics_before=(
                ModalInteractionMetrics.from_state_dict(
                    state["cumulative_eval_metrics_before"]  # type: ignore[arg-type]
                )
            ),
            cumulative_eval_metrics_after=(
                ModalInteractionMetrics.from_state_dict(
                    state["cumulative_eval_metrics_after"]  # type: ignore[arg-type]
                )
            ),
            selection_metric=state["selection_metric"],  # type: ignore[arg-type]
            heldout_improvement=state[
                "heldout_improvement"
            ],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ModalInteractionSelection:
    """Authenticated sparse interaction graph selected on open development."""

    source_model_sha256: str
    parameter_cluster_plan_sha256: str
    fit_split_sha256: str
    eval_split_sha256: str
    node_catalog: tuple[tuple[str, int, int, str], ...]
    candidate_edges: tuple[tuple[str, str], ...]
    policy: ModalInteractionSelectionPolicy
    steps: tuple[ModalInteractionSelectionStep, ...]
    artifact_sha256: str = ""
    artifact_kind: str = _SELECTION_KIND
    format_version: int = _FORMAT_VERSION
    contains_source_model_weights: bool = False
    contains_prompt_text: bool = False
    contains_raw_latent_rows: bool = False
    contains_target_residual_rows: bool = False
    contains_generator_weights: bool = False
    contains_interaction_weights: bool = True
    executable: bool = True
    tuned_on_closed_split: bool = False

    def __post_init__(self) -> None:
        for field in (
            "source_model_sha256",
            "parameter_cluster_plan_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
        ):
            _require_sha256(getattr(self, field), label=field)
        if self.fit_split_sha256 == self.eval_split_sha256:
            raise ValueError(
                "fit and evaluation split hashes must differ; hashes alone "
                "do not prove membership disjointness"
            )
        if not isinstance(self.policy, ModalInteractionSelectionPolicy):
            raise TypeError("policy must be ModalInteractionSelectionPolicy")
        self.policy.validate_integrity()
        if type(self.node_catalog) is not tuple or not self.node_catalog:
            raise ValueError("node_catalog must be a nonempty tuple")
        names: list[str] = []
        orders: dict[str, int] = {}
        widths: dict[str, int] = {}
        generator_hashes: dict[str, str] = {}
        for item in self.node_catalog:
            if type(item) is not tuple or len(item) != 4:
                raise ValueError("node_catalog entry is invalid")
            name, order, width, generator_hash = item
            canonical_name = _require_name(name, label="node name")
            names.append(canonical_name)
            orders[canonical_name] = _require_int(
                order,
                label="node causal order",
                minimum=0,
            )
            widths[canonical_name] = _require_int(
                width,
                label="node latent width",
                minimum=1,
            )
            generator_hashes[canonical_name] = _require_sha256(
                generator_hash,
                label="node generator hash",
            )
        if len(names) != len(set(names)):
            raise ValueError("node catalog names must be unique")
        if self.node_catalog != tuple(
            sorted(self.node_catalog, key=lambda value: (value[1], value[0]))
        ):
            raise ValueError("node_catalog must be in canonical causal order")
        if type(self.candidate_edges) is not tuple:
            raise TypeError("candidate_edges must be a tuple")
        if self.candidate_edges != tuple(sorted(set(self.candidate_edges))):
            raise ValueError("candidate_edges must be sorted and unique")
        for source, target in self.candidate_edges:
            if source not in orders or target not in orders:
                raise ValueError("candidate edge references an unknown node")
            if orders[source] >= orders[target]:
                raise ValueError("candidate edge is not strictly causal")
        if type(self.steps) is not tuple:
            raise TypeError("steps must be a tuple")
        incoming: dict[str, int] = {}
        selected_pairs: set[tuple[str, str]] = set()
        for index, step in enumerate(self.steps):
            if (
                not isinstance(step, ModalInteractionSelectionStep)
                or step.step_index != index
            ):
                raise ValueError("selection steps are not canonical")
            step.validate_integrity()
            binding = step.candidate.binding
            pair = (binding.source_node, binding.target_node)
            if pair not in self.candidate_edges or pair in selected_pairs:
                raise ValueError("selected edge is invalid or duplicated")
            selected_pairs.add(pair)
            incoming[binding.target_node] = (
                incoming.get(binding.target_node, 0) + 1
            )
            if (
                incoming[binding.target_node]
                > self.policy.max_incoming_edges
            ):
                raise ValueError("selection exceeds max incoming edges")
            if (
                binding.source_model_sha256 != self.source_model_sha256
                or binding.parameter_cluster_plan_sha256
                != self.parameter_cluster_plan_sha256
                or binding.fit_split_sha256 != self.fit_split_sha256
                or binding.eval_split_sha256 != self.eval_split_sha256
                or binding.eval_split_role != "open_development"
                or binding.source_causal_order
                != orders[binding.source_node]
                or binding.target_causal_order
                != orders[binding.target_node]
                or binding.source_generator_sha256
                != generator_hashes[binding.source_node]
                or binding.target_generator_sha256
                != generator_hashes[binding.target_node]
                or step.candidate.config.artifact_sha256
                != self.policy.fit_config.artifact_sha256
                or step.candidate.factors.source_width
                != widths[binding.source_node]
                or step.candidate.factors.target_width
                != widths[binding.target_node]
            ):
                raise ValueError("selected edge provenance does not match graph")
            if (
                step.selection_metric != self.policy.selection_metric
                or step.heldout_improvement
                <= max(0.0, self.policy.minimum_heldout_improvement)
            ):
                raise ValueError(
                    "selected edge does not satisfy held-out policy"
                )
        for field, expected in _SAFETY_METADATA.items():
            if getattr(self, field) is not expected:
                raise ValueError("modal interaction safety metadata is invalid")
        if (
            self.artifact_kind != _SELECTION_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal interaction selection header is invalid")
        computed = _json_sha256(self._payload(), domain=_SELECTION_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("modal interaction selection hash mismatch")

    @property
    def interactions(self) -> tuple[ModalGeneratorInteraction, ...]:
        return tuple(
            sorted(
                (
                    step.candidate.to_graph_interaction()
                    for step in self.steps
                ),
                key=lambda edge: (edge.source_node, edge.target_node),
            )
        )

    @property
    def parameter_count(self) -> int:
        return sum(step.candidate.parameter_count for step in self.steps)

    @property
    def macs_per_token(self) -> int:
        return sum(step.candidate.macs_per_token for step in self.steps)

    @property
    def bias_additions_per_token(self) -> int:
        return sum(
            step.candidate.bias_additions_per_token for step in self.steps
        )

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
            "node_catalog": self.node_catalog,
            "candidate_edges": self.candidate_edges,
            "policy_sha256": self.policy.artifact_sha256,
            "step_sha256s": tuple(
                step.artifact_sha256 for step in self.steps
            ),
            "parameter_count": self.parameter_count,
            "macs_per_token": self.macs_per_token,
            "bias_additions_per_token": self.bias_additions_per_token,
        }

    def validate_integrity(self) -> None:
        self.policy.validate_integrity()
        for step in self.steps:
            step.validate_integrity()
        if (
            _json_sha256(self._payload(), domain=_SELECTION_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("modal interaction selection hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "policy": self.policy.state_dict(),
            "steps": tuple(step.state_dict() for step in self.steps),
            "artifact_sha256": self.artifact_sha256,
        }

    metadata = _payload

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalInteractionSelection:
        fields = {
            "artifact_kind",
            "format_version",
            *_SAFETY_METADATA,
            "source_model_sha256",
            "parameter_cluster_plan_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
            "eval_split_role",
            "node_catalog",
            "candidate_edges",
            "policy_sha256",
            "step_sha256s",
            "parameter_count",
            "macs_per_token",
            "bias_additions_per_token",
            "policy",
            "steps",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="modal interaction selection")
        for field, expected in _SAFETY_METADATA.items():
            if state[field] is not expected:
                raise ValueError("modal interaction safety metadata is invalid")
        if state["eval_split_role"] != "open_development":
            raise ValueError("selection did not use open development")
        policy = ModalInteractionSelectionPolicy.from_state_dict(
            state["policy"]  # type: ignore[arg-type]
        )
        raw_steps = state["steps"]
        if type(raw_steps) is not tuple:
            raise TypeError("serialized steps must be a tuple")
        steps = tuple(
            ModalInteractionSelectionStep.from_state_dict(value)
            for value in raw_steps  # type: ignore[arg-type]
        )
        if state["policy_sha256"] != policy.artifact_sha256:
            raise ValueError("policy_sha256 does not match nested policy")
        if state["step_sha256s"] != tuple(
            step.artifact_sha256 for step in steps
        ):
            raise ValueError("step_sha256s do not match nested steps")
        result = cls(
            source_model_sha256=state[
                "source_model_sha256"
            ],  # type: ignore[arg-type]
            parameter_cluster_plan_sha256=state[
                "parameter_cluster_plan_sha256"
            ],  # type: ignore[arg-type]
            fit_split_sha256=state["fit_split_sha256"],  # type: ignore[arg-type]
            eval_split_sha256=state[
                "eval_split_sha256"
            ],  # type: ignore[arg-type]
            node_catalog=state["node_catalog"],  # type: ignore[arg-type]
            candidate_edges=state["candidate_edges"],  # type: ignore[arg-type]
            policy=policy,
            steps=steps,
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                field: state[field] for field in _SAFETY_METADATA
            },  # type: ignore[arg-type]
        )
        if (
            state["parameter_count"] != result.parameter_count
            or state["macs_per_token"] != result.macs_per_token
            or state["bias_additions_per_token"]
            != result.bias_additions_per_token
        ):
            raise ValueError("serialized selection accounting is inconsistent")
        return result


def _canonical_node_inputs(
    node_states_fit: Mapping[str, Tensor],
    node_states_eval: Mapping[str, Tensor],
    target_residuals_fit: Mapping[str, Tensor],
    target_residuals_eval: Mapping[str, Tensor],
    node_causal_orders: Mapping[str, int],
    generator_artifact_sha256s: Mapping[str, str],
) -> tuple[
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Tensor],
    tuple[tuple[str, int, int, str], ...],
]:
    mappings = (
        node_states_fit,
        node_states_eval,
        target_residuals_fit,
        target_residuals_eval,
        node_causal_orders,
        generator_artifact_sha256s,
    )
    if any(not isinstance(value, Mapping) for value in mappings):
        raise TypeError("modal interaction graph inputs must be mappings")
    state_names = set(node_states_fit)
    if (
        not state_names
        or set(node_states_eval) != state_names
        or set(node_causal_orders) != state_names
        or set(generator_artifact_sha256s) != state_names
    ):
        raise ValueError("node state, order, and generator catalogs differ")
    target_names = set(target_residuals_fit)
    if (
        not target_names
        or set(target_residuals_eval) != target_names
        or not target_names.issubset(state_names)
    ):
        raise ValueError("target residual catalogs are invalid")
    fit_states = {
        _require_name(name, label="node name"): _as_matrix(
            value,
            label=f"node_states_fit[{name}]",
        )
        for name, value in node_states_fit.items()
    }
    eval_states = {
        name: _as_matrix(value, label=f"node_states_eval[{name}]")
        for name, value in node_states_eval.items()
    }
    fit_targets = {
        name: _as_matrix(value, label=f"target_residuals_fit[{name}]")
        for name, value in target_residuals_fit.items()
    }
    eval_targets = {
        name: _as_matrix(value, label=f"target_residuals_eval[{name}]")
        for name, value in target_residuals_eval.items()
    }
    fit_rows = {value.shape[0] for value in fit_states.values()}
    eval_rows = {value.shape[0] for value in eval_states.values()}
    if len(fit_rows) != 1 or len(eval_rows) != 1:
        raise ValueError("all frozen node states must share split row counts")
    for name in state_names:
        if fit_states[name].shape[1] != eval_states[name].shape[1]:
            raise ValueError(f"node {name!r} latent width changes by split")
    for name in target_names:
        if (
            fit_targets[name].shape[0] not in fit_rows
            or eval_targets[name].shape[0] not in eval_rows
            or fit_targets[name].shape[1] != fit_states[name].shape[1]
            or eval_targets[name].shape[1] != eval_states[name].shape[1]
        ):
            raise ValueError(
                f"target residual {name!r} does not match node latent shape"
            )
    orders: dict[str, int] = {}
    generator_hashes: dict[str, str] = {}
    for name in state_names:
        orders[name] = _require_int(
            node_causal_orders[name],
            label=f"node_causal_orders[{name}]",
            minimum=0,
        )
        generator_hashes[name] = _require_sha256(
            generator_artifact_sha256s[name],
            label=f"generator_artifact_sha256s[{name}]",
        )
    catalog = tuple(
        sorted(
            (
                (
                    name,
                    orders[name],
                    fit_states[name].shape[1],
                    generator_hashes[name],
                )
                for name in state_names
            ),
            key=lambda value: (value[1], value[0]),
        )
    )
    return fit_states, eval_states, fit_targets, eval_targets, catalog


def _candidate_edge_catalog(
    candidate_edges: Sequence[tuple[str, str]] | None,
    *,
    node_catalog: tuple[tuple[str, int, int, str], ...],
    targets: set[str],
) -> tuple[tuple[str, str], ...]:
    orders = {name: order for name, order, _, _ in node_catalog}
    if candidate_edges is None:
        raw_edges = tuple(
            (source, target)
            for target in sorted(targets)
            for source in sorted(orders)
            if orders[source] < orders[target]
        )
    else:
        raw_edges = tuple(candidate_edges)
    if len(raw_edges) != len(set(raw_edges)):
        raise ValueError("candidate_edges must be unique")
    result: list[tuple[str, str]] = []
    for edge in raw_edges:
        if (
            type(edge) is not tuple
            or len(edge) != 2
            or edge[0] not in orders
            or edge[1] not in targets
        ):
            raise ValueError("candidate edge references an unknown node")
        source, target = edge
        if orders[source] >= orders[target]:
            raise ValueError(
                "candidate edges must point strictly forward in causal order"
            )
        result.append((source, target))
    return tuple(sorted(result))


def select_modal_interactions_greedily(
    node_states_fit: Mapping[str, Tensor],
    node_states_eval: Mapping[str, Tensor],
    target_residuals_fit: Mapping[str, Tensor],
    target_residuals_eval: Mapping[str, Tensor],
    *,
    node_causal_orders: Mapping[str, int],
    generator_artifact_sha256s: Mapping[str, str],
    source_model_sha256: str,
    parameter_cluster_plan_sha256: str,
    fit_split_sha256: str,
    eval_split_sha256: str,
    candidate_edges: Sequence[tuple[str, str]] | None = None,
    ridges: Sequence[float] | float = (0.0,),
    fisher_weights_fit: Tensor | None = None,
    fisher_weights_eval: Tensor | None = None,
    fit_intercept: bool = True,
    selection_metric: str = "weighted_nrmse",
    minimum_heldout_improvement: float = 0.0,
    max_incoming_edges: int = 1,
    eval_split_role: str = "open_development",
) -> ModalInteractionSelection:
    """Greedily select causal edges using only an open held-out split.

    Every trial fit uses the current calibration residual.  Candidate
    acceptance is based on cumulative reconstruction of the original
    development residual, so redundant fan-in cannot appear beneficial merely
    because each edge was fitted independently. Accepted messages update the
    target's runtime modal state before it can act as a downstream source.
    Once a node has sourced an accepted edge, its incoming set is frozen so a
    later selection cannot invalidate an already-fitted downstream message.
    Ties are resolved by metric, resource count, ridge, source name, then
    target name.
    """

    if eval_split_role != "open_development":
        raise ValueError(
            "greedy interaction selection requires open_development; "
            "closed guard/test splits cannot tune graph edges"
        )
    for value, label in (
        (source_model_sha256, "source_model_sha256"),
        (parameter_cluster_plan_sha256, "parameter_cluster_plan_sha256"),
        (fit_split_sha256, "fit_split_sha256"),
        (eval_split_sha256, "eval_split_sha256"),
    ):
        _require_sha256(value, label=label)
    if fit_split_sha256 == eval_split_sha256:
        raise ValueError("fit and evaluation splits must be disjoint")
    if isinstance(ridges, (int, float)) and not isinstance(ridges, bool):
        ridge_values = (float(ridges),)
    else:
        ridge_values = tuple(ridges)  # type: ignore[arg-type]
    fit_config = ModalInteractionFitConfig(
        ridges=ridge_values,
        fit_intercept=fit_intercept,
    )
    policy = ModalInteractionSelectionPolicy(
        fit_config=fit_config,
        selection_metric=selection_metric,
        minimum_heldout_improvement=minimum_heldout_improvement,
        max_incoming_edges=max_incoming_edges,
    )
    (
        fit_states,
        eval_states,
        original_fit,
        original_eval,
        node_catalog,
    ) = _canonical_node_inputs(
        node_states_fit,
        node_states_eval,
        target_residuals_fit,
        target_residuals_eval,
        node_causal_orders,
        generator_artifact_sha256s,
    )
    edges = _candidate_edge_catalog(
        candidate_edges,
        node_catalog=node_catalog,
        targets=set(original_fit),
    )
    orders = {name: order for name, order, _, _ in node_catalog}
    generator_hashes = {
        name: digest for name, _, _, digest in node_catalog
    }
    fit_rows = next(iter(fit_states.values())).shape[0]
    eval_rows = next(iter(eval_states.values())).shape[0]
    fit_weights = _as_weights(
        fisher_weights_fit,
        observations=fit_rows,
        label="fisher_weights_fit",
    )
    eval_weights = _as_weights(
        fisher_weights_eval,
        observations=eval_rows,
        label="fisher_weights_eval",
    )
    fit_prediction = {
        target: torch.zeros_like(value) for target, value in original_fit.items()
    }
    eval_prediction = {
        target: torch.zeros_like(value) for target, value in original_eval.items()
    }
    runtime_fit_states = {
        name: value.clone() for name, value in fit_states.items()
    }
    runtime_eval_states = {
        name: value.clone() for name, value in eval_states.items()
    }
    selected: set[tuple[str, str]] = set()
    locked_sources: set[str] = set()
    incoming = {target: 0 for target in original_fit}
    steps: list[ModalInteractionSelectionStep] = []

    while True:
        trials: list[
            tuple[
                float,
                float,
                int,
                float,
                str,
                str,
                ModalInteractionCandidate,
                Tensor,
                Tensor,
                ModalInteractionMetrics,
                ModalInteractionMetrics,
                ModalInteractionMetrics,
                ModalInteractionMetrics,
            ]
        ] = []
        for source, target in edges:
            pair = (source, target)
            if (
                pair in selected
                or incoming[target] >= policy.max_incoming_edges
                or target in locked_sources
            ):
                continue
            residual_fit = original_fit[target] - fit_prediction[target]
            residual_eval = original_eval[target] - eval_prediction[target]
            binding = ModalInteractionBinding(
                source_node=source,
                target_node=target,
                source_causal_order=orders[source],
                target_causal_order=orders[target],
                source_model_sha256=source_model_sha256,
                parameter_cluster_plan_sha256=(
                    parameter_cluster_plan_sha256
                ),
                source_generator_sha256=generator_hashes[source],
                target_generator_sha256=generator_hashes[target],
                fit_split_sha256=fit_split_sha256,
                eval_split_sha256=eval_split_sha256,
                eval_split_role=eval_split_role,
            )
            curve = fit_modal_interaction_rate_curve(
                runtime_fit_states[source],
                residual_fit,
                runtime_eval_states[source],
                residual_eval,
                binding=binding,
                ridges=fit_config.ridges,
                fisher_weights_fit=fit_weights,
                fisher_weights_eval=eval_weights,
                fit_intercept=fit_config.fit_intercept,
            )
            before_fit = _metrics(
                original_fit[target],
                fit_prediction[target],
                fit_weights,
            )
            before_eval = _metrics(
                original_eval[target],
                eval_prediction[target],
                eval_weights,
            )
            for candidate in curve.candidates:
                factors = candidate.factors
                edge_fit = (
                    runtime_fit_states[source] @ factors.message_matrix
                    + factors.message_bias
                )
                edge_eval = (
                    runtime_eval_states[source] @ factors.message_matrix
                    + factors.message_bias
                )
                after_fit = _metrics(
                    original_fit[target],
                    fit_prediction[target] + edge_fit,
                    fit_weights,
                )
                after_eval = _metrics(
                    original_eval[target],
                    eval_prediction[target] + edge_eval,
                    eval_weights,
                )
                improvement = (
                    getattr(before_eval, policy.selection_metric)
                    - getattr(after_eval, policy.selection_metric)
                )
                trials.append(
                    (
                        -improvement,
                        getattr(after_eval, policy.selection_metric),
                        candidate.parameter_count,
                        candidate.ridge,
                        source,
                        target,
                        candidate,
                        edge_fit,
                        edge_eval,
                        before_fit,
                        after_fit,
                        before_eval,
                        after_eval,
                    )
                )
        if not trials:
            break
        best = min(trials, key=lambda value: value[:6])
        improvement = -best[0]
        if improvement <= max(
            0.0,
            policy.minimum_heldout_improvement,
        ):
            break
        (
            _,
            _,
            _,
            _,
            source,
            target,
            candidate,
            edge_fit,
            edge_eval,
            before_fit,
            after_fit,
            before_eval,
            after_eval,
        ) = best
        selected.add((source, target))
        locked_sources.add(source)
        incoming[target] += 1
        fit_prediction[target] += edge_fit
        eval_prediction[target] += edge_eval
        runtime_fit_states[target] = (
            fit_states[target] + fit_prediction[target]
        )
        runtime_eval_states[target] = (
            eval_states[target] + eval_prediction[target]
        )
        steps.append(
            ModalInteractionSelectionStep(
                step_index=len(steps),
                candidate=candidate,
                cumulative_fit_metrics_before=before_fit,
                cumulative_fit_metrics_after=after_fit,
                cumulative_eval_metrics_before=before_eval,
                cumulative_eval_metrics_after=after_eval,
                selection_metric=policy.selection_metric,
                heldout_improvement=improvement,
            )
        )
    return ModalInteractionSelection(
        source_model_sha256=source_model_sha256,
        parameter_cluster_plan_sha256=parameter_cluster_plan_sha256,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
        node_catalog=node_catalog,
        candidate_edges=edges,
        policy=policy,
        steps=tuple(steps),
    )
