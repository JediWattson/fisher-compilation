"""Authenticated graph execution for fitted modal generators.

The objects in this module are the source-independent runtime half of the
Fisher compilation pipeline:

``parameter clusters -> computational modes -> modal generators -> graph``.

Each node owns a copied linear modal generator.  Legacy dense-residual nodes
may expose the generator's private reduced-rank latent.  Lowered coordinate
nodes instead expose the exact frozen computational-mode coordinates, receive
learned messages from strictly earlier coordinate states, and decode through
the authenticated mode-basis transpose plus mean.  Contributions sharing an
output boundary are summed.

No source-model module or callback is retained.  The graph therefore cannot
silently fall back to the transformer computation that it is intended to
replace.  Optional instrumentation is lazy: modal states and edge messages are
only cloned when explicitly requested.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re

import torch
from torch import Tensor


__all__ = [
    "LinearModalGeneratorNodeWeights",
    "ModalGeneratorGraphAccounting",
    "ModalGeneratorGraphExecution",
    "ModalGeneratorGraphExecutor",
    "ModalGeneratorGraphPlan",
    "ModalGeneratorInteraction",
    "ModalGeneratorNode",
]


_WEIGHTS_KIND = "fisher_graph.linear_modal_generator_node_weights"
_NODE_KIND = "fisher_graph.modal_generator_node"
_EDGE_KIND = "fisher_graph.modal_generator_interaction"
_GRAPH_KIND = "fisher_graph.modal_generator_graph_plan"
_FORMAT_VERSION = 1
_WEIGHTS_HASH_DOMAIN = b"fisher_graph.linear_modal_generator_node_weights.v1\0"
_NODE_HASH_DOMAIN = b"fisher_graph.modal_generator_node.v1\0"
_EDGE_HASH_DOMAIN = b"fisher_graph.modal_generator_interaction.v1\0"
_GRAPH_HASH_DOMAIN = b"fisher_graph.modal_generator_graph_plan.v1\0"
_TENSOR_HASH_DOMAIN = b"fisher_graph.modal_generator_graph_tensor.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
_STATE_KINDS = frozenset(
    {
        "generator_internal_latent",
        "computational_mode_coordinates",
    }
)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a nonempty canonical name")
    return value


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


def _as_float64_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
    ):
        raise ValueError(f"{label} must be a floating {ndim}D Tensor")
    if any(size <= 0 for size in value.shape):
        raise ValueError(f"{label} dimensions must be nonzero")
    canonical = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(canonical).all()):
        raise ValueError(f"{label} must contain only finite values")
    return canonical.clone()


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
    digest.update(_TENSOR_HASH_DOMAIN)
    digest.update(
        f"{tuple(value.shape)}\0float64\0".encode("utf-8")
    )
    digest.update(value.detach().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _strict_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch: expected {sorted(expected)}, "
            f"got {sorted(actual)}"
        )


@dataclass(frozen=True, slots=True)
class LinearModalGeneratorNodeWeights:
    """Copied factors for one fitted linear modal generator.

    ``generator_artifact_sha256`` binds these runtime factors to the fitted
    generator artifact from which they were compiled.  The factors are copied
    rather than retaining that artifact, so traversal cannot invoke source
    behavior.
    """

    generator_artifact_sha256: str
    source_model_sha256: str
    parameter_cluster_plan_sha256: str
    input_factor: Tensor
    output_factor: Tensor
    output_bias: Tensor | None
    state_kind: str = "generator_internal_latent"
    computational_mode_basis_sha256: str | None = None
    parameter_cluster_fragment_sha256: str | None = None
    state_factor: Tensor | None = None
    latent_bias: Tensor | None = None
    input_factor_sha256: str = ""
    output_factor_sha256: str = ""
    state_factor_sha256: str | None = ""
    output_bias_sha256: str | None = ""
    latent_bias_sha256: str | None = ""
    artifact_sha256: str = ""
    artifact_kind: str = _WEIGHTS_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.generator_artifact_sha256,
            label="generator_artifact_sha256",
        )
        _require_sha256(
            self.source_model_sha256,
            label="source_model_sha256",
        )
        _require_sha256(
            self.parameter_cluster_plan_sha256,
            label="parameter_cluster_plan_sha256",
        )
        if self.state_kind not in _STATE_KINDS:
            raise ValueError("modal generator state_kind is invalid")
        if self.computational_mode_basis_sha256 is not None:
            _require_sha256(
                self.computational_mode_basis_sha256,
                label="computational_mode_basis_sha256",
            )
        if self.parameter_cluster_fragment_sha256 is not None:
            _require_sha256(
                self.parameter_cluster_fragment_sha256,
                label="parameter_cluster_fragment_sha256",
            )
        if (
            self.state_kind == "computational_mode_coordinates"
            and (
                self.computational_mode_basis_sha256 is None
                or self.parameter_cluster_fragment_sha256 is None
            )
        ):
            raise ValueError(
                "computational-mode coordinate state must bind its basis "
                "and parameter-cluster fragment"
            )
        input_factor = _as_float64_tensor(
            self.input_factor,
            label="input_factor",
            ndim=2,
        )
        output_factor = _as_float64_tensor(
            self.output_factor,
            label="output_factor",
            ndim=2,
        )
        state_factor = None
        state_width = int(input_factor.shape[1])
        if self.state_factor is not None:
            state_factor = _as_float64_tensor(
                self.state_factor,
                label="state_factor",
                ndim=2,
            )
            if state_factor.shape[0] != input_factor.shape[1]:
                raise ValueError(
                    "state_factor input must match the private generator rank"
                )
            state_width = int(state_factor.shape[1])
        if state_width != output_factor.shape[0]:
            raise ValueError(
                "generated state and output_factor dimensions differ"
            )
        latent_bias = None
        if self.latent_bias is not None:
            latent_bias = _as_float64_tensor(
                self.latent_bias,
                label="latent_bias",
                ndim=1,
            )
            if latent_bias.shape != (state_width,):
                raise ValueError(
                    "latent_bias must match the generated state width"
                )
        output_bias = None
        if self.output_bias is not None:
            output_bias = _as_float64_tensor(
                self.output_bias,
                label="output_bias",
                ndim=1,
            )
            if output_factor.shape[1] != output_bias.shape[0]:
                raise ValueError(
                    "output_factor and output_bias output dimensions differ"
                )
        object.__setattr__(self, "input_factor", input_factor)
        object.__setattr__(self, "state_factor", state_factor)
        object.__setattr__(self, "output_factor", output_factor)
        object.__setattr__(self, "latent_bias", latent_bias)
        object.__setattr__(self, "output_bias", output_bias)
        for field, tensor in (
            ("input_factor_sha256", input_factor),
            ("output_factor_sha256", output_factor),
        ):
            computed = _tensor_sha256(tensor, label=field)
            supplied = getattr(self, field)
            if supplied == "":
                object.__setattr__(self, field, computed)
            elif _require_sha256(supplied, label=field) != computed:
                raise ValueError(f"{field.removesuffix('_sha256')} hash mismatch")
        for field, tensor in (
            ("state_factor_sha256", state_factor),
            ("latent_bias_sha256", latent_bias),
            ("output_bias_sha256", output_bias),
        ):
            supplied = getattr(self, field)
            if tensor is None:
                if supplied not in ("", None):
                    raise ValueError(
                        f"{field} must be absent when its tensor is absent"
                    )
                object.__setattr__(self, field, None)
            else:
                computed = _tensor_sha256(tensor, label=field)
                if supplied == "":
                    object.__setattr__(self, field, computed)
                elif _require_sha256(supplied, label=field) != computed:
                    raise ValueError(
                        f"{field.removesuffix('_sha256')} hash mismatch"
                    )
        if (
            self.artifact_kind != _WEIGHTS_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal generator weight artifact header is invalid")
        computed_artifact = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(
                self,
                "artifact_sha256",
                computed_artifact,
            )
        elif (
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            != computed_artifact
        ):
            raise ValueError("modal generator weight artifact hash mismatch")

    @property
    def input_width(self) -> int:
        return int(self.input_factor.shape[0])

    @property
    def latent_width(self) -> int:
        if self.state_factor is None:
            return self.private_width
        return int(self.state_factor.shape[1])

    @property
    def private_width(self) -> int:
        return int(self.input_factor.shape[1])

    @property
    def output_width(self) -> int:
        return int(self.output_factor.shape[1])

    @property
    def parameter_count(self) -> int:
        return (
            self.input_width * self.private_width
            + (
                self.private_width * self.latent_width
                if self.state_factor is not None
                else 0
            )
            + self.latent_width * self.output_width
            + (self.latent_width if self.latent_bias is not None else 0)
            + (self.output_width if self.output_bias is not None else 0)
        )

    @property
    def macs_per_token(self) -> int:
        return (
            self.input_width * self.private_width
            + (
                self.private_width * self.latent_width
                if self.state_factor is not None
                else 0
            )
            + self.latent_width * self.output_width
        )

    @property
    def bias_additions_per_token(self) -> int:
        return (
            (self.latent_width if self.latent_bias is not None else 0)
            + (self.output_width if self.output_bias is not None else 0)
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "generator_artifact_sha256": self.generator_artifact_sha256,
            "source_model_sha256": self.source_model_sha256,
            "parameter_cluster_plan_sha256": (
                self.parameter_cluster_plan_sha256
            ),
            "state_kind": self.state_kind,
            "computational_mode_basis_sha256": (
                self.computational_mode_basis_sha256
            ),
            "parameter_cluster_fragment_sha256": (
                self.parameter_cluster_fragment_sha256
            ),
            "input_width": self.input_width,
            "private_width": self.private_width,
            "latent_width": self.latent_width,
            "output_width": self.output_width,
            "input_factor_sha256": self.input_factor_sha256,
            "output_factor_sha256": self.output_factor_sha256,
            "state_factor_sha256": self.state_factor_sha256,
            "latent_bias_sha256": self.latent_bias_sha256,
            "output_bias_sha256": self.output_bias_sha256,
            "has_latent_bias": self.latent_bias is not None,
            "has_output_bias": self.output_bias is not None,
            "has_state_factor": self.state_factor is not None,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._hash_payload(),
            domain=_WEIGHTS_HASH_DOMAIN,
        )

    def validate_integrity(self) -> None:
        if _tensor_sha256(
            self.input_factor,
            label="input_factor",
        ) != self.input_factor_sha256:
            raise ValueError("input_factor hash mismatch")
        if _tensor_sha256(
            self.output_factor,
            label="output_factor",
        ) != self.output_factor_sha256:
            raise ValueError("output_factor hash mismatch")
        for value, expected, label in (
            (self.state_factor, self.state_factor_sha256, "state_factor"),
            (self.latent_bias, self.latent_bias_sha256, "latent_bias"),
            (self.output_bias, self.output_bias_sha256, "output_bias"),
        ):
            if value is None:
                if expected is not None:
                    raise ValueError(f"{label} hash presence mismatch")
            elif _tensor_sha256(value, label=label) != expected:
                raise ValueError(f"{label} hash mismatch")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("modal generator weight artifact hash mismatch")

    @classmethod
    def from_modal_generator_plan(
        cls,
        plan: object,
    ) -> LinearModalGeneratorNodeWeights:
        """Copy the narrow public factor protocol of ``ModalGeneratorPlan``."""

        validator = getattr(plan, "validate_integrity", None)
        if callable(validator):
            validator()
            authenticated_plan = plan
        else:
            state_builder = getattr(plan, "state_dict", None)
            state_loader = getattr(type(plan), "from_state_dict", None)
            if not callable(state_builder) or not callable(state_loader):
                raise TypeError(
                    "modal generator plan must expose validate_integrity() "
                    "or an authenticated state_dict()/from_state_dict() "
                    "roundtrip"
                )
            # ModalGeneratorPlan currently authenticates by reconstructing its
            # complete nested artifact.  Use that copy so mutable tensors
            # changed after fitting cannot enter the graph under a stale hash.
            authenticated_plan = state_loader(state_builder())
        artifact_sha256 = getattr(
            authenticated_plan,
            "artifact_sha256",
            None,
        )
        binding = getattr(authenticated_plan, "binding", None)
        if binding is None:
            raise TypeError("modal generator plan must expose binding")
        factors = getattr(authenticated_plan, "factors", None)
        if factors is None:
            raise TypeError("modal generator plan must expose factors")
        if (
            getattr(binding, "target_kind", None)
            == "computational_mode_coordinates"
        ):
            raise ValueError(
                "coordinate-target generators require authenticated "
                "computational-mode lowering"
            )
        try:
            input_factor = factors.input_factor
            output_factor = factors.output_factor
            output_bias = factors.bias
        except AttributeError as error:
            raise TypeError(
                "modal generator factors must expose input_factor, "
                "output_factor, and bias"
            ) from error
        return cls(
            generator_artifact_sha256=_require_sha256(
                artifact_sha256,
                label="plan.artifact_sha256",
            ),
            source_model_sha256=_require_sha256(
                getattr(binding, "source_model_sha256", None),
                label="plan.binding.source_model_sha256",
            ),
            parameter_cluster_plan_sha256=_require_sha256(
                getattr(binding, "cluster_plan_sha256", None),
                label="plan.binding.cluster_plan_sha256",
            ),
            input_factor=input_factor,
            output_factor=output_factor,
            output_bias=output_bias,
            computational_mode_basis_sha256=getattr(
                binding,
                "computational_mode_basis_sha256",
                None,
            ),
            parameter_cluster_fragment_sha256=getattr(
                binding,
                "parameter_cluster_fragment_sha256",
                None,
            ),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "input_factor": self.input_factor.clone(),
            "output_factor": self.output_factor.clone(),
            "state_factor": (
                None
                if self.state_factor is None
                else self.state_factor.clone()
            ),
            "latent_bias": (
                None if self.latent_bias is None else self.latent_bias.clone()
            ),
            "output_bias": (
                None if self.output_bias is None else self.output_bias.clone()
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LinearModalGeneratorNodeWeights:
        expected = {
            "artifact_kind",
            "format_version",
            "generator_artifact_sha256",
            "source_model_sha256",
            "parameter_cluster_plan_sha256",
            "state_kind",
            "computational_mode_basis_sha256",
            "parameter_cluster_fragment_sha256",
            "input_width",
            "private_width",
            "latent_width",
            "output_width",
            "input_factor_sha256",
            "output_factor_sha256",
            "state_factor_sha256",
            "latent_bias_sha256",
            "output_bias_sha256",
            "has_latent_bias",
            "has_output_bias",
            "has_state_factor",
            "input_factor",
            "output_factor",
            "state_factor",
            "latent_bias",
            "output_bias",
            "artifact_sha256",
        }
        _strict_keys(state, expected=expected, label="generator weights")
        result = cls(
            generator_artifact_sha256=state[
                "generator_artifact_sha256"
            ],
            source_model_sha256=state["source_model_sha256"],
            parameter_cluster_plan_sha256=state[
                "parameter_cluster_plan_sha256"
            ],
            state_kind=state["state_kind"],
            computational_mode_basis_sha256=state[
                "computational_mode_basis_sha256"
            ],
            parameter_cluster_fragment_sha256=state[
                "parameter_cluster_fragment_sha256"
            ],
            input_factor=state["input_factor"],
            output_factor=state["output_factor"],
            state_factor=state["state_factor"],
            latent_bias=state["latent_bias"],
            output_bias=state["output_bias"],
            input_factor_sha256=state["input_factor_sha256"],
            output_factor_sha256=state["output_factor_sha256"],
            state_factor_sha256=state["state_factor_sha256"],
            latent_bias_sha256=state["latent_bias_sha256"],
            output_bias_sha256=state["output_bias_sha256"],
            artifact_sha256=state["artifact_sha256"],
            artifact_kind=state["artifact_kind"],
            format_version=state["format_version"],
        )
        if (
            state["input_width"] != result.input_width
            or state["private_width"] != result.private_width
            or state["latent_width"] != result.latent_width
            or state["output_width"] != result.output_width
            or state["has_latent_bias"]
            is not (result.latent_bias is not None)
            or state["has_output_bias"]
            is not (result.output_bias is not None)
            or state["has_state_factor"]
            is not (result.state_factor is not None)
        ):
            raise ValueError("serialized generator dimensions drifted")
        return result


@dataclass(frozen=True, slots=True)
class ModalGeneratorNode:
    """One causally placed modal generator in the traversal graph."""

    name: str
    causal_order: int
    input_boundary: str
    output_boundary: str
    weights: LinearModalGeneratorNodeWeights
    artifact_sha256: str = ""
    artifact_kind: str = _NODE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.name, label="node name")
        _require_name(self.input_boundary, label="input_boundary")
        _require_name(self.output_boundary, label="output_boundary")
        if type(self.causal_order) is not int or self.causal_order < 0:
            raise ValueError("causal_order must be a nonnegative integer")
        if not isinstance(self.weights, LinearModalGeneratorNodeWeights):
            raise TypeError("weights must be LinearModalGeneratorNodeWeights")
        self.weights.validate_integrity()
        if (
            self.artifact_kind != _NODE_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal generator node artifact header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            != computed
        ):
            raise ValueError("modal generator node artifact hash mismatch")

    @property
    def input_width(self) -> int:
        return self.weights.input_width

    @property
    def latent_width(self) -> int:
        return self.weights.latent_width

    @property
    def output_width(self) -> int:
        return self.weights.output_width

    @property
    def state_kind(self) -> str:
        return self.weights.state_kind

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "name": self.name,
            "causal_order": self.causal_order,
            "input_boundary": self.input_boundary,
            "output_boundary": self.output_boundary,
            "weights_artifact_sha256": self.weights.artifact_sha256,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_NODE_HASH_DOMAIN)

    def validate_integrity(self) -> None:
        self.weights.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("modal generator node artifact hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "name": self.name,
            "causal_order": self.causal_order,
            "input_boundary": self.input_boundary,
            "output_boundary": self.output_boundary,
            "weights": self.weights.to_state_dict(),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorNode:
        expected = {
            "artifact_kind",
            "format_version",
            "name",
            "causal_order",
            "input_boundary",
            "output_boundary",
            "weights",
            "artifact_sha256",
        }
        _strict_keys(state, expected=expected, label="generator node")
        return cls(
            name=state["name"],
            causal_order=state["causal_order"],
            input_boundary=state["input_boundary"],
            output_boundary=state["output_boundary"],
            weights=LinearModalGeneratorNodeWeights.from_state_dict(
                state["weights"]
            ),
            artifact_sha256=state["artifact_sha256"],
            artifact_kind=state["artifact_kind"],
            format_version=state["format_version"],
        )


@dataclass(frozen=True, slots=True)
class ModalGeneratorInteraction:
    """Learned affine message from one generated latent state to another."""

    source_node: str
    target_node: str
    message_matrix: Tensor
    message_bias: Tensor
    message_matrix_sha256: str = ""
    message_bias_sha256: str = ""
    artifact_sha256: str = ""
    artifact_kind: str = _EDGE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.source_node, label="source_node")
        _require_name(self.target_node, label="target_node")
        if self.source_node == self.target_node:
            raise ValueError("a generator interaction cannot be a self-edge")
        matrix = _as_float64_tensor(
            self.message_matrix,
            label="message_matrix",
            ndim=2,
        )
        bias = _as_float64_tensor(
            self.message_bias,
            label="message_bias",
            ndim=1,
        )
        if matrix.shape[1] != bias.shape[0]:
            raise ValueError(
                "message matrix output and message bias dimensions differ"
            )
        object.__setattr__(self, "message_matrix", matrix)
        object.__setattr__(self, "message_bias", bias)
        for field, tensor in (
            ("message_matrix_sha256", matrix),
            ("message_bias_sha256", bias),
        ):
            computed = _tensor_sha256(tensor, label=field)
            supplied = getattr(self, field)
            if supplied == "":
                object.__setattr__(self, field, computed)
            elif _require_sha256(supplied, label=field) != computed:
                raise ValueError(f"{field.removesuffix('_sha256')} hash mismatch")
        if (
            self.artifact_kind != _EDGE_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("generator interaction artifact header is invalid")
        computed_artifact = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(
                self,
                "artifact_sha256",
                computed_artifact,
            )
        elif (
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            != computed_artifact
        ):
            raise ValueError("generator interaction artifact hash mismatch")

    @property
    def source_width(self) -> int:
        return int(self.message_matrix.shape[0])

    @property
    def target_width(self) -> int:
        return int(self.message_matrix.shape[1])

    @property
    def parameter_count(self) -> int:
        return self.source_width * self.target_width + self.target_width

    @property
    def macs_per_token(self) -> int:
        return self.source_width * self.target_width

    @property
    def bias_additions_per_token(self) -> int:
        return self.target_width

    @property
    def key(self) -> str:
        return f"{self.source_node}->{self.target_node}"

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "message_matrix_sha256": self.message_matrix_sha256,
            "message_bias_sha256": self.message_bias_sha256,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_EDGE_HASH_DOMAIN)

    def validate_integrity(self) -> None:
        if _tensor_sha256(
            self.message_matrix,
            label="message_matrix",
        ) != self.message_matrix_sha256:
            raise ValueError("message_matrix hash mismatch")
        if _tensor_sha256(
            self.message_bias,
            label="message_bias",
        ) != self.message_bias_sha256:
            raise ValueError("message_bias hash mismatch")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("generator interaction artifact hash mismatch")

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "message_matrix": self.message_matrix.clone(),
            "message_bias": self.message_bias.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorInteraction:
        expected = {
            "artifact_kind",
            "format_version",
            "source_node",
            "target_node",
            "source_width",
            "target_width",
            "message_matrix_sha256",
            "message_bias_sha256",
            "message_matrix",
            "message_bias",
            "artifact_sha256",
        }
        _strict_keys(state, expected=expected, label="generator interaction")
        result = cls(
            source_node=state["source_node"],
            target_node=state["target_node"],
            message_matrix=state["message_matrix"],
            message_bias=state["message_bias"],
            message_matrix_sha256=state["message_matrix_sha256"],
            message_bias_sha256=state["message_bias_sha256"],
            artifact_sha256=state["artifact_sha256"],
            artifact_kind=state["artifact_kind"],
            format_version=state["format_version"],
        )
        if (
            state["source_width"] != result.source_width
            or state["target_width"] != result.target_width
        ):
            raise ValueError("serialized interaction dimensions drifted")
        return result


@dataclass(frozen=True, slots=True)
class ModalGeneratorGraphAccounting:
    """Exact one-token storage and arithmetic accounting."""

    node_parameter_count: int
    interaction_parameter_count: int
    node_macs_per_token: int
    interaction_macs_per_token: int
    bias_additions_per_token: int
    message_accumulation_additions_per_token: int
    output_accumulation_additions_per_token: int

    @property
    def parameter_count(self) -> int:
        return self.node_parameter_count + self.interaction_parameter_count

    @property
    def macs_per_token(self) -> int:
        return self.node_macs_per_token + self.interaction_macs_per_token

    @property
    def elementwise_additions_per_token(self) -> int:
        return (
            self.bias_additions_per_token
            + self.message_accumulation_additions_per_token
            + self.output_accumulation_additions_per_token
        )


@dataclass(frozen=True, slots=True)
class ModalGeneratorGraphPlan:
    """Strict authenticated DAG of source-independent modal generators."""

    model_fingerprint: str
    parameter_cluster_plan_sha256: str
    nodes: tuple[ModalGeneratorNode, ...]
    interactions: tuple[ModalGeneratorInteraction, ...]
    artifact_sha256: str = ""
    artifact_kind: str = _GRAPH_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.model_fingerprint, label="model_fingerprint")
        _require_sha256(
            self.parameter_cluster_plan_sha256,
            label="parameter_cluster_plan_sha256",
        )
        if (
            type(self.nodes) is not tuple
            or not self.nodes
            or any(not isinstance(node, ModalGeneratorNode) for node in self.nodes)
        ):
            raise ValueError(
                "nodes must be a nonempty tuple of ModalGeneratorNode"
            )
        if (
            type(self.interactions) is not tuple
            or any(
                not isinstance(edge, ModalGeneratorInteraction)
                for edge in self.interactions
            )
        ):
            raise ValueError(
                "interactions must be a tuple of ModalGeneratorInteraction"
            )
        expected_nodes = tuple(
            sorted(self.nodes, key=lambda node: (node.causal_order, node.name))
        )
        if self.nodes != expected_nodes:
            raise ValueError("nodes must be in canonical causal order")
        names = tuple(node.name for node in self.nodes)
        if len(names) != len(set(names)):
            raise ValueError("generator node names must be unique")
        expected_edges = tuple(
            sorted(
                self.interactions,
                key=lambda edge: (edge.source_node, edge.target_node),
            )
        )
        if self.interactions != expected_edges:
            raise ValueError("interactions must be in canonical order")
        edge_pairs = tuple(
            (edge.source_node, edge.target_node)
            for edge in self.interactions
        )
        if len(edge_pairs) != len(set(edge_pairs)):
            raise ValueError("generator interactions must be unique")
        self._validate_boundaries_and_edges()
        for node in self.nodes:
            node.validate_integrity()
        for edge in self.interactions:
            edge.validate_integrity()
        if (
            self.artifact_kind != _GRAPH_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal generator graph artifact header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            != computed
        ):
            raise ValueError("modal generator graph artifact hash mismatch")

    def _validate_boundaries_and_edges(self) -> None:
        input_widths: dict[str, int] = {}
        output_widths: dict[str, int] = {}
        by_name = {node.name: node for node in self.nodes}
        for node in self.nodes:
            if node.weights.source_model_sha256 != self.model_fingerprint:
                raise ValueError(
                    "generator source model does not match graph model"
                )
            if (
                node.weights.parameter_cluster_plan_sha256
                != self.parameter_cluster_plan_sha256
            ):
                raise ValueError(
                    "generator parameter cluster plan does not match graph"
                )
            prior_input = input_widths.setdefault(
                node.input_boundary,
                node.input_width,
            )
            if prior_input != node.input_width:
                raise ValueError(
                    "nodes sharing an input boundary must share its width"
                )
            prior_output = output_widths.setdefault(
                node.output_boundary,
                node.output_width,
            )
            if prior_output != node.output_width:
                raise ValueError(
                    "nodes sharing an output boundary must share its width"
                )
        for edge in self.interactions:
            if (
                edge.source_node not in by_name
                or edge.target_node not in by_name
            ):
                raise ValueError(
                    "generator interaction references an unknown node"
                )
            source = by_name[edge.source_node]
            target = by_name[edge.target_node]
            if source.causal_order >= target.causal_order:
                raise ValueError(
                    "generator interactions must point strictly forward "
                    "in causal order"
                )
            if source.state_kind != target.state_kind:
                raise ValueError(
                    "generator interactions cannot mix internal latents "
                    "with computational-mode coordinates"
                )
            if edge.source_width != source.latent_width:
                raise ValueError(
                    "interaction source dimension does not match source latent"
                )
            if edge.target_width != target.latent_width:
                raise ValueError(
                    "interaction target dimension does not match target latent"
                )
        # Strictly forward edges already imply acyclicity.  Keep an explicit
        # topological audit so this invariant remains true if ordering evolves.
        indegree = {name: 0 for name in by_name}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in self.interactions:
            indegree[edge.target_node] += 1
            outgoing[edge.source_node].append(edge.target_node)
        frontier = sorted(name for name, degree in indegree.items() if degree == 0)
        visited = 0
        while frontier:
            current = frontier.pop(0)
            visited += 1
            for target in sorted(outgoing[current]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    frontier.append(target)
                    frontier.sort()
        if visited != len(by_name):
            raise ValueError("generator interaction graph contains a cycle")

    @property
    def input_boundary_widths(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for node in self.nodes:
            result[node.input_boundary] = node.input_width
        return dict(sorted(result.items()))

    @property
    def output_boundary_widths(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for node in self.nodes:
            result[node.output_boundary] = node.output_width
        return dict(sorted(result.items()))

    @property
    def traversal_order(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    @property
    def accounting(self) -> ModalGeneratorGraphAccounting:
        output_counts: dict[str, int] = defaultdict(int)
        for node in self.nodes:
            output_counts[node.output_boundary] += 1
        output_adds = sum(
            (count - 1) * self.output_boundary_widths[boundary]
            for boundary, count in output_counts.items()
        )
        return ModalGeneratorGraphAccounting(
            node_parameter_count=sum(
                node.weights.parameter_count for node in self.nodes
            ),
            interaction_parameter_count=sum(
                edge.parameter_count for edge in self.interactions
            ),
            node_macs_per_token=sum(
                node.weights.macs_per_token for node in self.nodes
            ),
            interaction_macs_per_token=sum(
                edge.macs_per_token for edge in self.interactions
            ),
            bias_additions_per_token=(
                sum(
                    node.weights.bias_additions_per_token
                    for node in self.nodes
                )
                + sum(
                    edge.bias_additions_per_token
                    for edge in self.interactions
                )
            ),
            message_accumulation_additions_per_token=sum(
                edge.target_width for edge in self.interactions
            ),
            output_accumulation_additions_per_token=output_adds,
        )

    @property
    def parameter_count(self) -> int:
        return self.accounting.parameter_count

    @property
    def macs_per_token(self) -> int:
        return self.accounting.macs_per_token

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "model_fingerprint": self.model_fingerprint,
            "parameter_cluster_plan_sha256": (
                self.parameter_cluster_plan_sha256
            ),
            "node_artifact_sha256s": tuple(
                node.artifact_sha256 for node in self.nodes
            ),
            "interaction_artifact_sha256s": tuple(
                edge.artifact_sha256 for edge in self.interactions
            ),
            "input_boundary_widths": self.input_boundary_widths,
            "output_boundary_widths": self.output_boundary_widths,
            "parameter_count": self.parameter_count,
            "macs_per_token": self.macs_per_token,
            "bias_additions_per_token": (
                self.accounting.bias_additions_per_token
            ),
            "message_accumulation_additions_per_token": (
                self.accounting.message_accumulation_additions_per_token
            ),
            "output_accumulation_additions_per_token": (
                self.accounting.output_accumulation_additions_per_token
            ),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_GRAPH_HASH_DOMAIN)

    def validate_integrity(self) -> None:
        for node in self.nodes:
            node.validate_integrity()
        for edge in self.interactions:
            edge.validate_integrity()
        self._validate_boundaries_and_edges()
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("modal generator graph artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "traversal_order": self.traversal_order,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "model_fingerprint": self.model_fingerprint,
            "parameter_cluster_plan_sha256": (
                self.parameter_cluster_plan_sha256
            ),
            "nodes": [
                node.to_state_dict() for node in self.nodes
            ],
            "interactions": [
                edge.to_state_dict() for edge in self.interactions
            ],
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorGraphPlan:
        expected = {
            "artifact_kind",
            "format_version",
            "model_fingerprint",
            "parameter_cluster_plan_sha256",
            "nodes",
            "interactions",
            "artifact_sha256",
        }
        _strict_keys(state, expected=expected, label="generator graph")
        raw_nodes = state["nodes"]
        raw_edges = state["interactions"]
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise TypeError(
                "serialized graph nodes and interactions must be lists"
            )
        return cls(
            model_fingerprint=state["model_fingerprint"],
            parameter_cluster_plan_sha256=state[
                "parameter_cluster_plan_sha256"
            ],
            nodes=tuple(
                ModalGeneratorNode.from_state_dict(value)
                for value in raw_nodes
            ),
            interactions=tuple(
                ModalGeneratorInteraction.from_state_dict(value)
                for value in raw_edges
            ),
            artifact_sha256=state["artifact_sha256"],
            artifact_kind=state["artifact_kind"],
            format_version=state["format_version"],
        )


@dataclass(frozen=True, slots=True)
class ModalGeneratorGraphExecution:
    """Outputs and optional lazy instrumentation from one traversal."""

    outputs: dict[str, Tensor]
    traversal_order: tuple[str, ...]
    modal_states: dict[str, Tensor] | None = None
    edge_messages: dict[str, Tensor] | None = None


class ModalGeneratorGraphExecutor:
    """Execute an authenticated modal-generator DAG without source calls."""

    def __init__(self, plan: ModalGeneratorGraphPlan) -> None:
        if not isinstance(plan, ModalGeneratorGraphPlan):
            raise TypeError("plan must be a ModalGeneratorGraphPlan")
        plan.validate_integrity()
        # Reconstruct from cloned serialized tensors.  This isolates the
        # runtime from later mutation of the caller's artifact object.
        self.plan = ModalGeneratorGraphPlan.from_state_dict(
            plan.to_state_dict()
        )
        self._nodes = {node.name: node for node in self.plan.nodes}
        incoming: dict[str, list[ModalGeneratorInteraction]] = defaultdict(list)
        for edge in self.plan.interactions:
            incoming[edge.target_node].append(edge)
        self._incoming = {
            name: tuple(edges) for name, edges in incoming.items()
        }

    @staticmethod
    def _runtime_weight(weight: Tensor, like: Tensor) -> Tensor:
        result = weight.to(device=like.device, dtype=like.dtype)
        if not bool(torch.isfinite(result).all()):
            raise ValueError(
                "modal graph weight is not finite in the runtime dtype"
            )
        return result

    def _validate_inputs(
        self,
        boundary_inputs: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        # The cloned plan is intentionally public for instrumentation, so
        # fail closed if a caller mutates any coefficient after construction.
        self.plan.validate_integrity()
        if not isinstance(boundary_inputs, Mapping):
            raise TypeError("boundary_inputs must be a mapping")
        expected = set(self.plan.input_boundary_widths)
        actual = set(boundary_inputs)
        if actual != expected:
            raise ValueError(
                "external input boundaries mismatch: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )
        result: dict[str, Tensor] = {}
        for boundary, width in self.plan.input_boundary_widths.items():
            value = boundary_inputs[boundary]
            if (
                not isinstance(value, Tensor)
                or value.ndim < 1
                or value.shape[-1] != width
                or not value.is_floating_point()
                or value.numel() == 0
            ):
                raise ValueError(
                    f"boundary {boundary!r} must be a nonempty floating "
                    f"Tensor with trailing width {width}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"boundary {boundary!r} must contain only finite values"
                )
            result[boundary] = value
        return result

    def execute(
        self,
        boundary_inputs: Mapping[str, Tensor],
        *,
        capture_modal_states: bool = False,
        capture_edge_messages: bool = False,
    ) -> ModalGeneratorGraphExecution:
        inputs = self._validate_inputs(boundary_inputs)
        states: dict[str, Tensor] = {}
        outputs: dict[str, Tensor] = {}
        captured_states = {} if capture_modal_states else None
        captured_messages = {} if capture_edge_messages else None

        for node in self.plan.nodes:
            own_input = inputs[node.input_boundary]
            weights = node.weights
            latent = own_input @ self._runtime_weight(
                weights.input_factor,
                own_input,
            )
            if weights.state_factor is not None:
                latent = latent @ self._runtime_weight(
                    weights.state_factor,
                    latent,
                )
            if weights.latent_bias is not None:
                latent = latent + self._runtime_weight(
                    weights.latent_bias,
                    latent,
                )
            for edge in self._incoming.get(node.name, ()):
                source_state = states[edge.source_node]
                if (
                    source_state.shape[:-1] != latent.shape[:-1]
                    or source_state.device != latent.device
                    or source_state.dtype != latent.dtype
                ):
                    raise ValueError(
                        f"interaction {edge.key!r} runtime batch, device, "
                        "or dtype dimensions drifted"
                    )
                message = source_state @ self._runtime_weight(
                    edge.message_matrix,
                    source_state,
                )
                message = message + self._runtime_weight(
                    edge.message_bias,
                    message,
                )
                if captured_messages is not None:
                    captured_messages[edge.key] = message.detach().clone()
                latent = latent + message
            if not bool(torch.isfinite(latent).all()):
                raise ValueError(
                    f"modal state for node {node.name!r} became non-finite"
                )
            states[node.name] = latent
            if captured_states is not None:
                captured_states[node.name] = latent.detach().clone()
            contribution = latent @ self._runtime_weight(
                weights.output_factor,
                latent,
            )
            if weights.output_bias is not None:
                contribution = contribution + self._runtime_weight(
                    weights.output_bias,
                    contribution,
                )
            if not bool(torch.isfinite(contribution).all()):
                raise ValueError(
                    f"residual contribution for node {node.name!r} "
                    "became non-finite"
                )
            prior = outputs.get(node.output_boundary)
            if prior is None:
                outputs[node.output_boundary] = contribution
            else:
                if (
                    prior.shape != contribution.shape
                    or prior.device != contribution.device
                    or prior.dtype != contribution.dtype
                ):
                    raise ValueError(
                        f"output boundary {node.output_boundary!r} "
                        "runtime dimensions drifted"
                    )
                outputs[node.output_boundary] = prior + contribution

        return ModalGeneratorGraphExecution(
            outputs=outputs,
            traversal_order=self.plan.traversal_order,
            modal_states=captured_states,
            edge_messages=captured_messages,
        )

    __call__ = execute
