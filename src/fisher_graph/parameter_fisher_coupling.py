"""Natural MLP parameter groups and their implicit virtual-gate Fisher.

This module implements the first two boundaries of the modal compiler:

``weights -> Fisher coupling -> parameter clusters``.

The weight boundary is metadata-only.  One natural scalar MLP channel owns
three parameter slices: row ``j`` of ``gate_proj``, row ``j`` of ``up_proj``,
and column ``j`` of ``down_proj``.  The catalog records the paths, shapes,
slices, and exact parameter counts, but never copies or serializes a model
weight.

The Fisher boundary is deliberately precise about what it represents.  If
``R[x, j]`` is the prompt-level derivative with respect to a virtual scalar
gate on natural channel ``j``, then the grouped pullback empirical Fisher is

``F = R.T @ R`` (sum convention), or ``F = R.T @ R / prompts`` (mean
convention).

This is *not* the raw full parameter Fisher.  It is the Fisher pulled back to
one virtual-gate coordinate per natural parameter group.  The artifact stores
the low-rank score factor ``R`` and the diagonal Fisher mass, never a dense
``groups x groups`` matrix.  Couplings are evaluated in bounded chunks.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor

from .structured_mlp_cross_block_bundling import (
    CrossBlockLayerSpec,
    ModeKey,
)


_CATALOG_KIND = "fisher_graph.natural_mlp_parameter_group_catalog"
_FISHER_KIND = "fisher_graph.grouped_virtual_gate_empirical_fisher"
_FORMAT_VERSION = 1
_CATALOG_HASH_DOMAIN = b"fisher_graph.natural_mlp_parameter_catalog.v1\0"
_FISHER_HASH_DOMAIN = b"fisher_graph.grouped_virtual_gate_fisher.v1\0"
_TENSOR_HASH_DOMAIN = b"fisher_graph.grouped_virtual_gate_tensor.v1\0"
_INPUT_CATALOG_HASH_DOMAIN = b"fisher_graph.natural_mlp_input_catalog.v1\0"
_GROUPING_DEFINITION = (
    "gate_proj_row_j+up_proj_row_j+down_proj_column_j"
)
_SCORE_DEFINITION = "prompt_sum_z_times_d_nll_dz"
_PULLBACK_DEFINITION = (
    "empirical_fisher_pulled_back_to_natural_mlp_virtual_gates"
)
_BIAS_POLICY = "weight_matrices_only_no_bias"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

FisherNormalization = Literal["sum_over_prompts", "mean_over_prompts"]
LayerPairPolicy = Literal["any", "same_layer", "cross_layer"]


def natural_mlp_input_catalog_sha256(
    *,
    source_model_sha256: str,
    input_site: str,
    input_width: int,
) -> str:
    """Authenticate one layer-input contract used by modal generators."""

    _require_sha256(source_model_sha256, label="source_model_sha256")
    _require_nonempty(input_site, label="input_site")
    _require_positive_int(input_width, label="input_width")
    return _json_sha256(
        {
            "source_model_sha256": source_model_sha256,
            "input_site": input_site,
            "input_width": input_width,
        },
        domain=_INPUT_CATALOG_HASH_DOMAIN,
    )


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _json_sha256(value: object, *, domain: bytes) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(serialized)
    return digest.hexdigest()


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


def _as_float64_matrix(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(
            f"{label} must be a nonempty floating [rows, columns] matrix"
        )
    result = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result.clone()


@dataclass(frozen=True, order=True, slots=True)
class ParameterGroupKey:
    """Stable identity of one natural MLP scalar channel."""

    layer_ordinal: int
    layer_id: str
    activation_site: str
    channel_index: int

    def __post_init__(self) -> None:
        if type(self.layer_ordinal) is not int or self.layer_ordinal < 0:
            raise ValueError("layer_ordinal must be nonnegative")
        _require_nonempty(self.layer_id, label="layer_id")
        _require_nonempty(self.activation_site, label="activation_site")
        if type(self.channel_index) is not int or self.channel_index < 0:
            raise ValueError("channel_index must be nonnegative")

    def as_mode_key(self, *, fisher_rank: int) -> ModeKey:
        """Return the corresponding existing mode identity after ranking."""

        return ModeKey(
            layer_ordinal=self.layer_ordinal,
            layer_id=self.layer_id,
            activation_site=self.activation_site,
            mode_index=self.channel_index,
            fisher_rank=fisher_rank,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "layer_ordinal": self.layer_ordinal,
            "layer_id": self.layer_id,
            "activation_site": self.activation_site,
            "channel_index": self.channel_index,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ParameterGroupKey:
        expected = {
            "layer_ordinal",
            "layer_id",
            "activation_site",
            "channel_index",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("parameter-group key fields are invalid")
        return cls(
            layer_ordinal=int(state["layer_ordinal"]),
            layer_id=str(state["layer_id"]),
            activation_site=str(state["activation_site"]),
            channel_index=int(state["channel_index"]),
        )


@dataclass(frozen=True, slots=True)
class NaturalMLPParameterSlice:
    """One row or column owned by a natural scalar MLP channel."""

    role: str
    parameter_path: str
    axis: int
    index: int
    matrix_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.role not in {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError("parameter-slice role is invalid")
        _require_nonempty(self.parameter_path, label="parameter_path")
        expected_axis = 1 if self.role == "down_proj" else 0
        if self.axis != expected_axis:
            raise ValueError(
                f"{self.role} must be sliced on axis {expected_axis}"
            )
        if (
            type(self.matrix_shape) is not tuple
            or len(self.matrix_shape) != 2
            or any(type(value) is not int or value <= 0 for value in self.matrix_shape)
        ):
            raise ValueError("matrix_shape must contain two positive integers")
        if (
            type(self.index) is not int
            or not 0 <= self.index < self.matrix_shape[self.axis]
        ):
            raise ValueError("slice index is outside its matrix shape")

    @property
    def parameter_count(self) -> int:
        return self.matrix_shape[1 - self.axis]

    @property
    def slice_kind(self) -> str:
        return "column" if self.axis == 1 else "row"

    def metadata(self) -> dict[str, object]:
        return {
            "role": self.role,
            "parameter_path": self.parameter_path,
            "axis": self.axis,
            "index": self.index,
            "matrix_shape": self.matrix_shape,
            "slice_kind": self.slice_kind,
            "parameter_count": self.parameter_count,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "parameter_path": self.parameter_path,
            "axis": self.axis,
            "index": self.index,
            "matrix_shape": self.matrix_shape,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> NaturalMLPParameterSlice:
        expected = {
            "role",
            "parameter_path",
            "axis",
            "index",
            "matrix_shape",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("natural MLP parameter-slice fields are invalid")
        shape = state["matrix_shape"]
        if type(shape) is not tuple:
            raise TypeError("parameter-slice matrix_shape must be a tuple")
        return cls(
            role=str(state["role"]),
            parameter_path=str(state["parameter_path"]),
            axis=int(state["axis"]),
            index=int(state["index"]),
            matrix_shape=tuple(int(value) for value in shape),
        )


@dataclass(frozen=True, slots=True)
class NaturalMLPLayerParameterSpec:
    """Shape/path metadata for one native gated MLP site."""

    layer_id: str
    layer_ordinal: int
    activation_site: str
    input_site: str
    output_site: str
    intermediate_width: int
    input_width: int
    output_width: int
    gate_proj_path: str
    up_proj_path: str
    down_proj_path: str

    def __post_init__(self) -> None:
        _require_nonempty(self.layer_id, label="layer_id")
        _require_nonempty(self.activation_site, label="activation_site")
        _require_nonempty(self.input_site, label="input_site")
        _require_nonempty(self.output_site, label="output_site")
        if len(
            {
                self.activation_site,
                self.input_site,
                self.output_site,
            }
        ) != 3:
            raise ValueError("MLP activation/input/output sites must be distinct")
        if type(self.layer_ordinal) is not int or self.layer_ordinal < 0:
            raise ValueError("layer_ordinal must be nonnegative")
        _require_positive_int(
            self.intermediate_width,
            label="intermediate_width",
        )
        _require_positive_int(self.input_width, label="input_width")
        _require_positive_int(self.output_width, label="output_width")
        paths = (
            self.gate_proj_path,
            self.up_proj_path,
            self.down_proj_path,
        )
        for label, path in zip(
            ("gate_proj_path", "up_proj_path", "down_proj_path"),
            paths,
            strict=True,
        ):
            _require_nonempty(path, label=label)
        if len(set(paths)) != 3:
            raise ValueError("MLP projection parameter paths must be distinct")

    @classmethod
    def from_cross_block_layer_spec(
        cls,
        spec: CrossBlockLayerSpec,
        *,
        input_width: int,
        output_width: int | None = None,
        parameter_prefix: str | None = None,
        input_site: str | None = None,
        output_site: str | None = None,
    ) -> NaturalMLPLayerParameterSpec:
        """Lift an activation-site spec into weight-slice metadata."""

        if not isinstance(spec, CrossBlockLayerSpec):
            raise TypeError("spec must be CrossBlockLayerSpec")
        output = input_width if output_width is None else output_width
        prefix = (
            f"{spec.layer_id}.mlp"
            if parameter_prefix is None
            else parameter_prefix
        )
        _require_nonempty(prefix, label="parameter_prefix")
        return cls(
            layer_id=spec.layer_id,
            layer_ordinal=spec.layer_ordinal,
            activation_site=spec.activation_site,
            input_site=(
                f"{spec.layer_id}.mlp.input"
                if input_site is None
                else input_site
            ),
            output_site=(
                f"{spec.layer_id}.mlp.residual_delta"
                if output_site is None
                else output_site
            ),
            intermediate_width=spec.width,
            input_width=input_width,
            output_width=output,
            gate_proj_path=f"{prefix}.gate_proj.weight",
            up_proj_path=f"{prefix}.up_proj.weight",
            down_proj_path=f"{prefix}.down_proj.weight",
        )

    @property
    def cross_block_layer_spec(self) -> CrossBlockLayerSpec:
        return CrossBlockLayerSpec(
            layer_id=self.layer_id,
            layer_ordinal=self.layer_ordinal,
            activation_site=self.activation_site,
            width=self.intermediate_width,
        )

    @property
    def parameter_count_per_group(self) -> int:
        return 2 * self.input_width + self.output_width

    @property
    def grouped_parameter_count(self) -> int:
        return self.intermediate_width * self.parameter_count_per_group

    def metadata(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "layer_ordinal": self.layer_ordinal,
            "activation_site": self.activation_site,
            "input_site": self.input_site,
            "output_site": self.output_site,
            "intermediate_width": self.intermediate_width,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "gate_proj_path": self.gate_proj_path,
            "up_proj_path": self.up_proj_path,
            "down_proj_path": self.down_proj_path,
            "parameter_count_per_group": self.parameter_count_per_group,
            "grouped_parameter_count": self.grouped_parameter_count,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "layer_ordinal": self.layer_ordinal,
            "activation_site": self.activation_site,
            "input_site": self.input_site,
            "output_site": self.output_site,
            "intermediate_width": self.intermediate_width,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "gate_proj_path": self.gate_proj_path,
            "up_proj_path": self.up_proj_path,
            "down_proj_path": self.down_proj_path,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> NaturalMLPLayerParameterSpec:
        expected = {
            "layer_id",
            "layer_ordinal",
            "activation_site",
            "input_site",
            "output_site",
            "intermediate_width",
            "input_width",
            "output_width",
            "gate_proj_path",
            "up_proj_path",
            "down_proj_path",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("natural MLP layer-spec fields are invalid")
        return cls(
            layer_id=str(state["layer_id"]),
            layer_ordinal=int(state["layer_ordinal"]),
            activation_site=str(state["activation_site"]),
            input_site=str(state["input_site"]),
            output_site=str(state["output_site"]),
            intermediate_width=int(state["intermediate_width"]),
            input_width=int(state["input_width"]),
            output_width=int(state["output_width"]),
            gate_proj_path=str(state["gate_proj_path"]),
            up_proj_path=str(state["up_proj_path"]),
            down_proj_path=str(state["down_proj_path"]),
        )


@dataclass(frozen=True, slots=True)
class NaturalMLPParameterGroup:
    """The three weight slices controlled by one virtual scalar gate."""

    group_index: int
    key: ParameterGroupKey
    gate_proj: NaturalMLPParameterSlice
    up_proj: NaturalMLPParameterSlice
    down_proj: NaturalMLPParameterSlice

    def __post_init__(self) -> None:
        if type(self.group_index) is not int or self.group_index < 0:
            raise ValueError("group_index must be nonnegative")
        if not isinstance(self.key, ParameterGroupKey):
            raise TypeError("key must be ParameterGroupKey")
        slices = (self.gate_proj, self.up_proj, self.down_proj)
        if any(
            not isinstance(value, NaturalMLPParameterSlice)
            for value in slices
        ):
            raise TypeError("parameter-group slices have invalid types")
        if tuple(value.role for value in slices) != (
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            raise ValueError("parameter-group slice roles are invalid")
        if any(value.index != self.key.channel_index for value in slices):
            raise ValueError("parameter slices must select the group channel")

    @property
    def parameter_count(self) -> int:
        return (
            self.gate_proj.parameter_count
            + self.up_proj.parameter_count
            + self.down_proj.parameter_count
        )

    def metadata(self) -> dict[str, object]:
        return {
            "group_index": self.group_index,
            "key": self.key.metadata(),
            "gate_proj": self.gate_proj.metadata(),
            "up_proj": self.up_proj.metadata(),
            "down_proj": self.down_proj.metadata(),
            "parameter_count": self.parameter_count,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "group_index": self.group_index,
            "key": self.key.metadata(),
            "gate_proj": self.gate_proj.state_dict(),
            "up_proj": self.up_proj.state_dict(),
            "down_proj": self.down_proj.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> NaturalMLPParameterGroup:
        expected = {
            "group_index",
            "key",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("natural MLP parameter-group fields are invalid")
        for name in ("key", "gate_proj", "up_proj", "down_proj"):
            if not isinstance(state[name], Mapping):
                raise TypeError(f"parameter-group {name} must be a mapping")
        return cls(
            group_index=int(state["group_index"]),
            key=ParameterGroupKey.from_state_dict(state["key"]),
            gate_proj=NaturalMLPParameterSlice.from_state_dict(
                state["gate_proj"]
            ),
            up_proj=NaturalMLPParameterSlice.from_state_dict(
                state["up_proj"]
            ),
            down_proj=NaturalMLPParameterSlice.from_state_dict(
                state["down_proj"]
            ),
        )


def _groups_for_specs(
    specs: tuple[NaturalMLPLayerParameterSpec, ...],
) -> tuple[NaturalMLPParameterGroup, ...]:
    groups: list[NaturalMLPParameterGroup] = []
    for spec in specs:
        for channel_index in range(spec.intermediate_width):
            key = ParameterGroupKey(
                layer_ordinal=spec.layer_ordinal,
                layer_id=spec.layer_id,
                activation_site=spec.activation_site,
                channel_index=channel_index,
            )
            groups.append(
                NaturalMLPParameterGroup(
                    group_index=len(groups),
                    key=key,
                    gate_proj=NaturalMLPParameterSlice(
                        role="gate_proj",
                        parameter_path=spec.gate_proj_path,
                        axis=0,
                        index=channel_index,
                        matrix_shape=(
                            spec.intermediate_width,
                            spec.input_width,
                        ),
                    ),
                    up_proj=NaturalMLPParameterSlice(
                        role="up_proj",
                        parameter_path=spec.up_proj_path,
                        axis=0,
                        index=channel_index,
                        matrix_shape=(
                            spec.intermediate_width,
                            spec.input_width,
                        ),
                    ),
                    down_proj=NaturalMLPParameterSlice(
                        role="down_proj",
                        parameter_path=spec.down_proj_path,
                        axis=1,
                        index=channel_index,
                        matrix_shape=(
                            spec.output_width,
                            spec.intermediate_width,
                        ),
                    ),
                )
            )
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class NaturalMLPParameterGroupCatalog:
    """Authenticated metadata-only catalog of natural MLP channels."""

    model_fingerprint: str
    layer_specs: tuple[NaturalMLPLayerParameterSpec, ...]
    groups: tuple[NaturalMLPParameterGroup, ...]
    total_parameter_count: int
    artifact_sha256: str = ""
    grouping_definition: str = _GROUPING_DEFINITION
    bias_policy: str = _BIAS_POLICY
    artifact_kind: str = _CATALOG_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.model_fingerprint, label="model_fingerprint")
        if (
            type(self.layer_specs) is not tuple
            or not self.layer_specs
            or any(
                not isinstance(value, NaturalMLPLayerParameterSpec)
                for value in self.layer_specs
            )
        ):
            raise ValueError(
                "layer_specs must be a nonempty tuple of natural MLP specs"
            )
        canonical_specs = tuple(
            sorted(
                self.layer_specs,
                key=lambda value: (
                    value.layer_ordinal,
                    value.layer_id,
                    value.activation_site,
                ),
            )
        )
        if self.layer_specs != canonical_specs:
            raise ValueError("layer_specs must be in canonical layer order")
        layer_coordinates = tuple(
            (value.layer_ordinal, value.layer_id, value.activation_site)
            for value in self.layer_specs
        )
        if len(layer_coordinates) != len(set(layer_coordinates)):
            raise ValueError("natural MLP layer specs must be unique")
        if len({value.layer_ordinal for value in self.layer_specs}) != len(
            self.layer_specs
        ):
            raise ValueError("layer ordinals must be unique")
        all_paths = tuple(
            path
            for spec in self.layer_specs
            for path in (
                spec.gate_proj_path,
                spec.up_proj_path,
                spec.down_proj_path,
            )
        )
        if len(all_paths) != len(set(all_paths)):
            raise ValueError("projection parameter paths must be globally unique")
        expected_groups = _groups_for_specs(self.layer_specs)
        if type(self.groups) is not tuple or self.groups != expected_groups:
            raise ValueError(
                "groups do not match the canonical natural-channel catalog"
            )
        expected_count = sum(
            value.grouped_parameter_count for value in self.layer_specs
        )
        if self.total_parameter_count != expected_count:
            raise ValueError("total_parameter_count is inconsistent")
        if (
            self.grouping_definition != _GROUPING_DEFINITION
            or self.bias_policy != _BIAS_POLICY
            or self.artifact_kind != _CATALOG_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("parameter-group catalog semantics are invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        else:
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            if self.artifact_sha256 != computed:
                raise ValueError("parameter-group catalog hash mismatch")

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def contains_model_weights(self) -> bool:
        return False

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "model_fingerprint": self.model_fingerprint,
            "layer_specs": tuple(
                value.metadata() for value in self.layer_specs
            ),
            "groups": tuple(value.metadata() for value in self.groups),
            "group_count": self.group_count,
            "total_parameter_count": self.total_parameter_count,
            "grouping_definition": self.grouping_definition,
            "bias_policy": self.bias_policy,
            "contains_model_weights": False,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._payload(),
            domain=_CATALOG_HASH_DOMAIN,
        )

    def validate_integrity(self) -> None:
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("parameter-group catalog hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "model_fingerprint": self.model_fingerprint,
            "layer_specs": tuple(
                value.state_dict() for value in self.layer_specs
            ),
            "groups": tuple(value.state_dict() for value in self.groups),
            "total_parameter_count": self.total_parameter_count,
            "grouping_definition": self.grouping_definition,
            "bias_policy": self.bias_policy,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> NaturalMLPParameterGroupCatalog:
        expected = {
            "artifact_kind",
            "format_version",
            "model_fingerprint",
            "layer_specs",
            "groups",
            "total_parameter_count",
            "grouping_definition",
            "bias_policy",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("parameter-group catalog state fields are invalid")
        if type(state["layer_specs"]) is not tuple or type(
            state["groups"]
        ) is not tuple:
            raise TypeError("parameter-group catalog collections must be tuples")
        return cls(
            model_fingerprint=str(state["model_fingerprint"]),
            layer_specs=tuple(
                NaturalMLPLayerParameterSpec.from_state_dict(value)
                for value in state["layer_specs"]
            ),
            groups=tuple(
                NaturalMLPParameterGroup.from_state_dict(value)
                for value in state["groups"]
            ),
            total_parameter_count=int(state["total_parameter_count"]),
            artifact_sha256=str(state["artifact_sha256"]),
            grouping_definition=str(state["grouping_definition"]),
            bias_policy=str(state["bias_policy"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
        )


def build_natural_mlp_parameter_group_catalog(
    *,
    model_fingerprint: str,
    layer_specs: Sequence[NaturalMLPLayerParameterSpec],
) -> NaturalMLPParameterGroupCatalog:
    """Build the canonical metadata catalog without reading model weights."""

    if isinstance(layer_specs, (str, bytes)) or not isinstance(
        layer_specs,
        Sequence,
    ):
        raise TypeError("layer_specs must be a sequence")
    specs = tuple(
        sorted(
            layer_specs,
            key=lambda value: (
                value.layer_ordinal,
                value.layer_id,
                value.activation_site,
            ),
        )
    )
    if not specs or any(
        not isinstance(value, NaturalMLPLayerParameterSpec)
        for value in specs
    ):
        raise ValueError("layer_specs must contain natural MLP layer specs")
    groups = _groups_for_specs(specs)
    return NaturalMLPParameterGroupCatalog(
        model_fingerprint=model_fingerprint,
        layer_specs=specs,
        groups=groups,
        total_parameter_count=sum(
            value.grouped_parameter_count for value in specs
        ),
    )


@dataclass(frozen=True, slots=True)
class FisherCouplingEdge:
    """One deterministic undirected edge from the implicit Fisher."""

    first_group_index: int
    second_group_index: int
    first: ParameterGroupKey
    second: ParameterGroupKey
    signed_coupling: float
    ranking_strength: float
    ranking: str

    def __post_init__(self) -> None:
        if (
            type(self.first_group_index) is not int
            or type(self.second_group_index) is not int
            or not 0 <= self.first_group_index < self.second_group_index
        ):
            raise ValueError("Fisher edge indices must be canonical and distinct")
        if not isinstance(self.first, ParameterGroupKey) or not isinstance(
            self.second,
            ParameterGroupKey,
        ):
            raise TypeError("Fisher edge endpoints must be parameter-group keys")
        if self.ranking not in {"absolute", "signed"}:
            raise ValueError("Fisher edge ranking is invalid")
        if (
            isinstance(self.signed_coupling, bool)
            or not isinstance(self.signed_coupling, (int, float))
            or not math.isfinite(float(self.signed_coupling))
            or isinstance(self.ranking_strength, bool)
            or not isinstance(self.ranking_strength, (int, float))
            or not math.isfinite(float(self.ranking_strength))
        ):
            raise ValueError("Fisher edge values must be finite")
        object.__setattr__(self, "signed_coupling", float(self.signed_coupling))
        object.__setattr__(self, "ranking_strength", float(self.ranking_strength))
        expected = (
            abs(self.signed_coupling)
            if self.ranking == "absolute"
            else self.signed_coupling
        )
        if self.ranking_strength != expected:
            raise ValueError("Fisher edge ranking strength is inconsistent")


@dataclass(frozen=True, slots=True)
class GroupedVirtualGateFisher:
    """Authenticated low-rank factorization of grouped empirical Fisher."""

    catalog: NaturalMLPParameterGroupCatalog
    calibration_split_sha256: str
    objective_sha256: str
    score_factor: Tensor
    fisher_mass: Tensor
    normalization: FisherNormalization
    score_factor_sha256: str
    fisher_mass_sha256: str
    artifact_sha256: str
    source_prompt_trace_sha256: str | None = None
    score_definition: str = _SCORE_DEFINITION
    pullback_definition: str = _PULLBACK_DEFINITION
    artifact_kind: str = _FISHER_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, NaturalMLPParameterGroupCatalog):
            raise TypeError("catalog must be NaturalMLPParameterGroupCatalog")
        self.catalog.validate_integrity()
        _require_sha256(
            self.calibration_split_sha256,
            label="calibration_split_sha256",
        )
        _require_sha256(self.objective_sha256, label="objective_sha256")
        if self.source_prompt_trace_sha256 is not None:
            _require_sha256(
                self.source_prompt_trace_sha256,
                label="source_prompt_trace_sha256",
            )
        if self.normalization not in {
            "sum_over_prompts",
            "mean_over_prompts",
        }:
            raise ValueError("Fisher normalization is invalid")
        factor = _as_float64_matrix(
            self.score_factor,
            label="score_factor",
        )
        if factor.shape[1] != self.catalog.group_count:
            raise ValueError(
                "score_factor columns must match the parameter-group catalog"
            )
        mass = self.fisher_mass
        if (
            not isinstance(mass, Tensor)
            or mass.ndim != 1
            or mass.shape[0] != self.catalog.group_count
            or not mass.is_floating_point()
        ):
            raise ValueError(
                "fisher_mass must be floating with one value per group"
            )
        mass = mass.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        if (
            not bool(torch.isfinite(mass).all())
            or bool((mass < 0).any())
        ):
            raise ValueError("fisher_mass must be finite and nonnegative")
        object.__setattr__(self, "score_factor", factor)
        object.__setattr__(self, "fisher_mass", mass.clone())
        if (
            self.score_definition != _SCORE_DEFINITION
            or self.pullback_definition != _PULLBACK_DEFINITION
            or self.artifact_kind != _FISHER_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("grouped Fisher artifact semantics are invalid")
        for name in (
            "score_factor_sha256",
            "fisher_mass_sha256",
            "artifact_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        self.validate_integrity()

    @property
    def prompt_count(self) -> int:
        return int(self.score_factor.shape[0])

    @property
    def group_count(self) -> int:
        return self.catalog.group_count

    @property
    def normalization_divisor(self) -> float:
        if self.normalization == "mean_over_prompts":
            return float(self.prompt_count)
        return 1.0

    @property
    def rank_upper_bound(self) -> int:
        return min(self.prompt_count, self.group_count)

    @property
    def contains_dense_group_fisher(self) -> bool:
        return False

    @property
    def contains_model_weights(self) -> bool:
        return False

    @property
    def contains_raw_prompts(self) -> bool:
        return False

    def fisher_ranked_mode_catalog(self) -> tuple[ModeKey, ...]:
        """Return mode identities in score-factor column order.

        Fisher ranks are assigned independently within each layer by
        descending diagonal mass, using the native channel index as the
        stable tie-breaker. Keeping catalog order makes the result align
        directly with the columns of ``score_factor``.
        """

        rank_by_group: dict[int, int] = {}
        groups_by_layer: dict[int, list[int]] = {}
        for group in self.catalog.groups:
            groups_by_layer.setdefault(
                group.key.layer_ordinal,
                [],
            ).append(group.group_index)
        for indices in groups_by_layer.values():
            ordered = sorted(
                indices,
                key=lambda index: (
                    -float(self.fisher_mass[index].item()),
                    self.catalog.groups[index].key.channel_index,
                ),
            )
            rank_by_group.update(
                (group_index, rank)
                for rank, group_index in enumerate(ordered)
            )
        return tuple(
            group.key.as_mode_key(
                fisher_rank=rank_by_group[group.group_index]
            )
            for group in self.catalog.groups
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "catalog_artifact_sha256": self.catalog.artifact_sha256,
            "calibration_split_sha256": self.calibration_split_sha256,
            "objective_sha256": self.objective_sha256,
            "source_prompt_trace_sha256": self.source_prompt_trace_sha256,
            "source_trace_authenticated": (
                self.source_prompt_trace_sha256 is not None
            ),
            "prompt_count": self.prompt_count,
            "group_count": self.group_count,
            "normalization": self.normalization,
            "normalization_divisor": self.normalization_divisor,
            "score_definition": self.score_definition,
            "pullback_definition": self.pullback_definition,
            "score_factor_sha256": self.score_factor_sha256,
            "fisher_mass_sha256": self.fisher_mass_sha256,
            "contains_dense_group_fisher": False,
            "contains_model_weights": False,
            "contains_raw_prompts": False,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_FISHER_HASH_DOMAIN)

    def _computed_mass(self) -> Tensor:
        return (
            self.score_factor.square().sum(dim=0)
            / self.normalization_divisor
        ).contiguous()

    def validate_integrity(self) -> None:
        self.catalog.validate_integrity()
        if (
            _tensor_sha256(
                self.score_factor,
                label="score_factor",
            )
            != self.score_factor_sha256
        ):
            raise ValueError("grouped Fisher score-factor hash mismatch")
        if (
            _tensor_sha256(
                self.fisher_mass,
                label="fisher_mass",
            )
            != self.fisher_mass_sha256
        ):
            raise ValueError("grouped Fisher mass hash mismatch")
        if not torch.equal(self.fisher_mass, self._computed_mass()):
            raise ValueError("grouped Fisher mass is inconsistent")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("grouped Fisher artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "rank_upper_bound": self.rank_upper_bound,
            "zero_mass_group_count": int(
                (self.fisher_mass == 0).sum().item()
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "catalog": self.catalog.state_dict(),
            "calibration_split_sha256": self.calibration_split_sha256,
            "objective_sha256": self.objective_sha256,
            "source_prompt_trace_sha256": self.source_prompt_trace_sha256,
            "score_factor": self.score_factor.detach().clone(),
            "fisher_mass": self.fisher_mass.detach().clone(),
            "normalization": self.normalization,
            "score_definition": self.score_definition,
            "pullback_definition": self.pullback_definition,
            "score_factor_sha256": self.score_factor_sha256,
            "fisher_mass_sha256": self.fisher_mass_sha256,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> GroupedVirtualGateFisher:
        expected = {
            "artifact_kind",
            "format_version",
            "catalog",
            "calibration_split_sha256",
            "objective_sha256",
            "source_prompt_trace_sha256",
            "score_factor",
            "fisher_mass",
            "normalization",
            "score_definition",
            "pullback_definition",
            "score_factor_sha256",
            "fisher_mass_sha256",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("grouped Fisher state fields are invalid")
        if not isinstance(state["catalog"], Mapping):
            raise TypeError("grouped Fisher catalog must be a mapping")
        if not isinstance(state["score_factor"], Tensor) or not isinstance(
            state["fisher_mass"],
            Tensor,
        ):
            raise TypeError("grouped Fisher factors must be Tensors")
        return cls(
            catalog=NaturalMLPParameterGroupCatalog.from_state_dict(
                state["catalog"]
            ),
            calibration_split_sha256=str(
                state["calibration_split_sha256"]
            ),
            objective_sha256=str(state["objective_sha256"]),
            source_prompt_trace_sha256=(
                None
                if state["source_prompt_trace_sha256"] is None
                else str(state["source_prompt_trace_sha256"])
            ),
            score_factor=state["score_factor"],
            fisher_mass=state["fisher_mass"],
            normalization=str(state["normalization"]),  # type: ignore[arg-type]
            score_definition=str(state["score_definition"]),
            pullback_definition=str(state["pullback_definition"]),
            score_factor_sha256=str(state["score_factor_sha256"]),
            fisher_mass_sha256=str(state["fisher_mass_sha256"]),
            artifact_sha256=str(state["artifact_sha256"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
        )

    def _indices(
        self,
        values: Sequence[int],
        *,
        label: str,
    ) -> tuple[int, ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{label} must be a sequence")
        result = tuple(values)
        if any(
            type(value) is not int or not 0 <= value < self.group_count
            for value in result
        ):
            raise ValueError(f"{label} contains an invalid group index")
        return result

    def coupling(
        self,
        first_group_index: int,
        second_group_index: int,
        *,
        absolute: bool = False,
    ) -> float:
        """Evaluate one signed or absolute Fisher coupling."""

        indices = self._indices(
            (first_group_index, second_group_index),
            label="group_indices",
        )
        value = float(
            torch.dot(
                self.score_factor[:, indices[0]],
                self.score_factor[:, indices[1]],
            ).item()
            / self.normalization_divisor
        )
        return abs(value) if absolute else value

    def coupling_pairs(
        self,
        first_group_indices: Sequence[int],
        second_group_indices: Sequence[int],
        *,
        absolute: bool = False,
        chunk_size: int = 4096,
    ) -> Tensor:
        """Evaluate aligned pairs without forming a dense Fisher matrix."""

        _require_positive_int(chunk_size, label="chunk_size")
        first = self._indices(
            first_group_indices,
            label="first_group_indices",
        )
        second = self._indices(
            second_group_indices,
            label="second_group_indices",
        )
        if len(first) != len(second):
            raise ValueError("coupling-pair index lists must have equal length")
        result = torch.empty(len(first), dtype=torch.float64)
        for start in range(0, len(first), chunk_size):
            stop = min(start + chunk_size, len(first))
            left = self.score_factor[:, first[start:stop]]
            right = self.score_factor[:, second[start:stop]]
            values = (left * right).sum(dim=0) / self.normalization_divisor
            result[start:stop] = values.abs() if absolute else values
        return result.contiguous()

    def coupling_block(
        self,
        row_group_indices: Sequence[int],
        column_group_indices: Sequence[int],
        *,
        absolute: bool = False,
        chunk_size: int = 1024,
    ) -> Tensor:
        """Evaluate a requested submatrix in bounded group chunks."""

        _require_positive_int(chunk_size, label="chunk_size")
        rows = self._indices(row_group_indices, label="row_group_indices")
        columns = self._indices(
            column_group_indices,
            label="column_group_indices",
        )
        result = torch.empty(
            (len(rows), len(columns)),
            dtype=torch.float64,
        )
        for row_start in range(0, len(rows), chunk_size):
            row_stop = min(row_start + chunk_size, len(rows))
            left = self.score_factor[:, rows[row_start:row_stop]]
            for column_start in range(0, len(columns), chunk_size):
                column_stop = min(
                    column_start + chunk_size,
                    len(columns),
                )
                right = self.score_factor[
                    :,
                    columns[column_start:column_stop],
                ]
                values = (
                    left.transpose(0, 1) @ right
                ) / self.normalization_divisor
                result[
                    row_start:row_stop,
                    column_start:column_stop,
                ] = values.abs() if absolute else values
        return result.contiguous()

    def iter_coupling_blocks(
        self,
        *,
        group_indices: Sequence[int] | None = None,
        absolute: bool = False,
        block_size: int = 1024,
        upper_triangle_only: bool = False,
    ) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], Tensor]]:
        """Yield coupling blocks while retaining only one block at a time."""

        _require_positive_int(block_size, label="block_size")
        indices = (
            tuple(range(self.group_count))
            if group_indices is None
            else self._indices(group_indices, label="group_indices")
        )
        for row_start in range(0, len(indices), block_size):
            rows = indices[row_start : row_start + block_size]
            first_column_start = row_start if upper_triangle_only else 0
            for column_start in range(
                first_column_start,
                len(indices),
                block_size,
            ):
                columns = indices[
                    column_start : column_start + block_size
                ]
                yield (
                    rows,
                    columns,
                    self.coupling_block(
                        rows,
                        columns,
                        absolute=absolute,
                        chunk_size=block_size,
                    ),
                )

    def top_k_edges(
        self,
        k: int,
        *,
        absolute: bool = True,
        layer_policy: LayerPairPolicy = "any",
        block_size: int = 1024,
        min_strength: float = 0.0,
    ) -> tuple[FisherCouplingEdge, ...]:
        """Return bounded-memory deterministic top-k undirected couplings.

        Ties are resolved by ascending canonical catalog indices.  Self edges
        are excluded.  With the default positive threshold, zero-mass groups
        cannot create meaningless zero-strength edges.
        """

        _require_positive_int(k, label="k")
        _require_positive_int(block_size, label="block_size")
        if layer_policy not in {"any", "same_layer", "cross_layer"}:
            raise ValueError("layer_policy is invalid")
        if (
            isinstance(min_strength, bool)
            or not isinstance(min_strength, (int, float))
            or not math.isfinite(float(min_strength))
        ):
            raise ValueError("min_strength must be finite")
        threshold = float(min_strength)
        # Heap key is ordered best-to-worst by strength, then gives earlier
        # catalog indices priority.  The root is therefore the current worst.
        heap: list[tuple[float, int, int, float, int, int]] = []
        for row_start in range(0, self.group_count, block_size):
            row_stop = min(row_start + block_size, self.group_count)
            rows = tuple(range(row_start, row_stop))
            for column_start in range(
                row_start,
                self.group_count,
                block_size,
            ):
                column_stop = min(
                    column_start + block_size,
                    self.group_count,
                )
                columns = tuple(range(column_start, column_stop))
                values = self.coupling_block(
                    rows,
                    columns,
                    chunk_size=block_size,
                )
                for row_offset, first_index in enumerate(rows):
                    first_key = self.catalog.groups[first_index].key
                    for column_offset, second_index in enumerate(columns):
                        if first_index >= second_index:
                            continue
                        second_key = self.catalog.groups[second_index].key
                        same_layer = (
                            first_key.layer_ordinal
                            == second_key.layer_ordinal
                        )
                        if (
                            layer_policy == "same_layer"
                            and not same_layer
                        ) or (
                            layer_policy == "cross_layer"
                            and same_layer
                        ):
                            continue
                        coupling = float(
                            values[row_offset, column_offset].item()
                        )
                        strength = abs(coupling) if absolute else coupling
                        if strength <= threshold:
                            continue
                        item = (
                            strength,
                            -first_index,
                            -second_index,
                            coupling,
                            first_index,
                            second_index,
                        )
                        if len(heap) < k:
                            heapq.heappush(heap, item)
                        elif item[:3] > heap[0][:3]:
                            heapq.heapreplace(heap, item)
        ranked = sorted(
            heap,
            key=lambda value: (
                -value[0],
                value[4],
                value[5],
            ),
        )
        ranking = "absolute" if absolute else "signed"
        return tuple(
            FisherCouplingEdge(
                first_group_index=first_index,
                second_group_index=second_index,
                first=self.catalog.groups[first_index].key,
                second=self.catalog.groups[second_index].key,
                signed_coupling=coupling,
                ranking_strength=strength,
                ranking=ranking,
            )
            for (
                strength,
                _negative_first,
                _negative_second,
                coupling,
                first_index,
                second_index,
            ) in ranked
        )


def build_grouped_virtual_gate_fisher(
    virtual_gate_scores: Tensor,
    *,
    catalog: NaturalMLPParameterGroupCatalog,
    calibration_split_sha256: str,
    objective_sha256: str,
    normalization: FisherNormalization = "mean_over_prompts",
    source_prompt_trace_sha256: str | None = None,
) -> GroupedVirtualGateFisher:
    """Build ``F = R.T R / divisor`` as an authenticated implicit factor."""

    if not isinstance(catalog, NaturalMLPParameterGroupCatalog):
        raise TypeError("catalog must be NaturalMLPParameterGroupCatalog")
    catalog.validate_integrity()
    factor = _as_float64_matrix(
        virtual_gate_scores,
        label="virtual_gate_scores",
    )
    if factor.shape[1] != catalog.group_count:
        raise ValueError(
            "virtual-gate score columns must match the group catalog"
        )
    if normalization not in {
        "sum_over_prompts",
        "mean_over_prompts",
    }:
        raise ValueError("Fisher normalization is invalid")
    if source_prompt_trace_sha256 is not None:
        _require_sha256(
            source_prompt_trace_sha256,
            label="source_prompt_trace_sha256",
        )
    divisor = (
        float(factor.shape[0])
        if normalization == "mean_over_prompts"
        else 1.0
    )
    mass = (factor.square().sum(dim=0) / divisor).contiguous()
    factor_sha256 = _tensor_sha256(factor, label="score_factor")
    mass_sha256 = _tensor_sha256(mass, label="fisher_mass")
    temporary = {
        "artifact_kind": _FISHER_KIND,
        "format_version": _FORMAT_VERSION,
        "catalog_artifact_sha256": catalog.artifact_sha256,
        "calibration_split_sha256": calibration_split_sha256,
        "objective_sha256": objective_sha256,
        "source_prompt_trace_sha256": source_prompt_trace_sha256,
        "source_trace_authenticated": source_prompt_trace_sha256 is not None,
        "prompt_count": int(factor.shape[0]),
        "group_count": catalog.group_count,
        "normalization": normalization,
        "normalization_divisor": divisor,
        "score_definition": _SCORE_DEFINITION,
        "pullback_definition": _PULLBACK_DEFINITION,
        "score_factor_sha256": factor_sha256,
        "fisher_mass_sha256": mass_sha256,
        "contains_dense_group_fisher": False,
        "contains_model_weights": False,
        "contains_raw_prompts": False,
    }
    return GroupedVirtualGateFisher(
        catalog=catalog,
        calibration_split_sha256=calibration_split_sha256,
        objective_sha256=objective_sha256,
        source_prompt_trace_sha256=source_prompt_trace_sha256,
        score_factor=factor,
        fisher_mass=mass,
        normalization=normalization,
        score_factor_sha256=factor_sha256,
        fisher_mass_sha256=mass_sha256,
        artifact_sha256=_json_sha256(
            temporary,
            domain=_FISHER_HASH_DOMAIN,
        ),
    )


def build_grouped_virtual_gate_fisher_from_trace(
    trace: object,
    *,
    catalog: NaturalMLPParameterGroupCatalog,
    normalization: FisherNormalization = "mean_over_prompts",
) -> GroupedVirtualGateFisher:
    """Build grouped Fisher from one authenticated prompt-mode trace."""

    from .prompt_mode_tracing import PromptModeTrace

    if not isinstance(trace, PromptModeTrace):
        raise TypeError("trace must be a PromptModeTrace")
    if not isinstance(catalog, NaturalMLPParameterGroupCatalog):
        raise TypeError("catalog must be a NaturalMLPParameterGroupCatalog")
    trace = PromptModeTrace.from_state_dict(trace.state_dict())
    catalog.validate_integrity()
    if (
        trace.provenance.source_model_fingerprint
        != catalog.model_fingerprint
        or trace.mode_count != catalog.group_count
        or trace.layer_specs
        != tuple(
            layer.cross_block_layer_spec for layer in catalog.layer_specs
        )
    ):
        raise ValueError(
            "prompt trace and natural parameter catalog do not align"
        )
    return build_grouped_virtual_gate_fisher(
        trace.prompt_effects,
        catalog=catalog,
        calibration_split_sha256=(
            trace.provenance.calibration_split_sha256
        ),
        objective_sha256=trace.provenance.objective_sha256,
        normalization=normalization,
        source_prompt_trace_sha256=trace.artifact_sha256,
    )


__all__ = [
    "FisherCouplingEdge",
    "FisherNormalization",
    "GroupedVirtualGateFisher",
    "LayerPairPolicy",
    "NaturalMLPLayerParameterSpec",
    "NaturalMLPParameterGroup",
    "NaturalMLPParameterGroupCatalog",
    "NaturalMLPParameterSlice",
    "ParameterGroupKey",
    "build_grouped_virtual_gate_fisher",
    "build_grouped_virtual_gate_fisher_from_trace",
    "build_natural_mlp_parameter_group_catalog",
    "natural_mlp_input_catalog_sha256",
]
