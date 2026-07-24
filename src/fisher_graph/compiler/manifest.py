"""Strict, model-independent manifests for compiled runtime segments.

The manifest is deliberately JSON rather than a PyTorch artifact.  A caller
can inspect and validate an execution plan without importing a backend or
opening any tensor bundle.  Tensor artifacts remain opaque, hash-addressed
resources which are handed to a backend only after their exact bytes have
been authenticated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO


_SCHEMA = "fisher_graph.runtime_manifest"
_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENCODINGS = {"torch_weights_only", "json", "text", "opaque"}
_SEQUENCE_POLICIES = {"fixed", "bounded_dynamic", "dynamic"}
_ATTENTION_MASK_POLICIES = {
    "unsupported",
    "optional_all_true",
    "optional",
    "required",
}
_PADDING_POLICIES = {"none", "left", "right", "either"}
_POSITION_ID_POLICIES = {"unsupported", "optional", "required"}
_CACHE_POLICIES = {"none", "prefill", "prefill_decode"}
_INSTRUMENTATION_POLICIES = {
    "none",
    "resident",
    "lazy_fail_fast_only",
}
_FALLBACK_POLICIES = {"source_model", "disabled"}
_VALIDATION_STATUSES = {"passed", "failed", "not_run"}
_MASK_INPUT_FORMS = {
    "unsupported": frozenset({"omitted"}),
    "optional_all_true": frozenset({"omitted", "all_true"}),
    "optional": frozenset({"omitted", "all_true", "arbitrary"}),
    "required": frozenset({"all_true", "arbitrary"}),
}
_PADDING_INPUT_FORMS = {
    "none": frozenset({"none"}),
    "left": frozenset({"none", "left"}),
    "right": frozenset({"none", "right"}),
    "either": frozenset({"none", "left", "right"}),
}
_POSITION_INPUT_FORMS = {
    "unsupported": frozenset({"omitted"}),
    "optional": frozenset({"omitted", "provided"}),
    "required": frozenset({"provided"}),
}
_CACHE_EXECUTION_FORMS = {
    "none": frozenset({"prefill"}),
    "prefill": frozenset({"prefill", "chunked_prefill"}),
    "prefill_decode": frozenset(
        {"prefill", "chunked_prefill", "decode"}
    ),
}


def _compiler_version() -> str:
    try:
        return version("fisher-graph")
    except PackageNotFoundError:
        return "unknown"


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(map(str, extra))))
        raise ValueError(f"{label} has invalid keys: {'; '.join(details)}")
    return value  # type: ignore[return-value]


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    text = _require_string(value, label=label)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValueError(f"{label} is not a portable identifier")
    return text


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _require_resource_path(value: object, *, label: str) -> str:
    path = _require_string(value, label=label)
    pure = PurePosixPath(path)
    if (
        "\\" in path
        or pure.is_absolute()
        or str(pure) != path
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return path


def _require_tuple(value: object, *, label: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{label} must be a tuple")
    return value


def _portable_json(value: object, *, label: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} keys must be strings")
            result[key] = _portable_json(item, label=f"{label}.{key}")
        return result
    if type(value) in (list, tuple):
        return [
            _portable_json(item, label=f"{label}[]")
            for item in value
        ]
    raise ValueError(f"{label} contains a non-JSON value")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _portable_json_sha256(value: object, *, label: str) -> str:
    portable = _portable_json(value, label=label)
    payload = json.dumps(
        portable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tuple_of_identifiers(value: object, *, label: str) -> tuple[str, ...]:
    values = _require_tuple(value, label=label)
    parsed = tuple(
        _require_identifier(item, label=f"{label}[]") for item in values
    )
    if not parsed:
        raise ValueError(f"{label} cannot be empty")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} cannot contain duplicates")
    return parsed


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    id: str
    path: str
    sha256: str
    size_bytes: int
    encoding: str
    artifact_kind: str
    format_version: int

    def __post_init__(self) -> None:
        _require_identifier(self.id, label="resource id")
        _require_sha256(self.sha256, label=f"resource {self.id} sha256")
        _require_int(
            self.size_bytes,
            label=f"resource {self.id} size_bytes",
            minimum=1,
        )
        if self.encoding not in _ENCODINGS:
            raise ValueError(f"resource {self.id} has unsupported encoding")
        _require_identifier(
            self.artifact_kind,
            label=f"resource {self.id} artifact_kind",
        )
        _require_int(
            self.format_version,
            label=f"resource {self.id} format_version",
            minimum=1,
        )
        _require_resource_path(
            self.path,
            label=f"resource {self.id} path",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "encoding": self.encoding,
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> ArtifactDescriptor:
        raw = _require_exact_keys(
            value,
            {
                "id",
                "path",
                "sha256",
                "size_bytes",
                "encoding",
                "artifact_kind",
                "format_version",
            },
            label="resource",
        )
        return cls(**dict(raw))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    adapter_id: str
    adapter_version: int
    architecture: str
    source_model_id: str | None
    source_revision: str | None
    source_state_sha256: str
    source_config_sha256: str
    source_resource: str | None
    layer_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.adapter_id, label="model adapter_id")
        _require_int(
            self.adapter_version,
            label="model adapter_version",
            minimum=1,
        )
        _require_identifier(self.architecture, label="model architecture")
        for name, value in (
            ("source_model_id", self.source_model_id),
            ("source_revision", self.source_revision),
        ):
            if value is not None:
                _require_string(value, label=f"model {name}")
        if (self.source_model_id is None) != (self.source_revision is None):
            raise ValueError(
                "external source identity requires both "
                "source_model_id and immutable source_revision"
            )
        _require_sha256(
            self.source_state_sha256,
            label="model source_state_sha256",
        )
        _require_sha256(
            self.source_config_sha256,
            label="model source_config_sha256",
        )
        if self.source_resource is not None:
            _require_identifier(
                self.source_resource,
                label="model source_resource",
            )
        _tuple_of_identifiers(self.layer_ids, label="model layer_ids")

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "architecture": self.architecture,
            "source_model_id": self.source_model_id,
            "source_revision": self.source_revision,
            "source_state_sha256": self.source_state_sha256,
            "source_config_sha256": self.source_config_sha256,
            "source_resource": self.source_resource,
            "layer_ids": list(self.layer_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> ModelIdentity:
        raw = _require_exact_keys(
            value,
            {
                "adapter_id",
                "adapter_version",
                "architecture",
                "source_model_id",
                "source_revision",
                "source_state_sha256",
                "source_config_sha256",
                "source_resource",
                "layer_ids",
            },
            label="model",
        )
        layer_ids = raw["layer_ids"]
        if type(layer_ids) is not list:
            raise ValueError("model layer_ids must be an array")
        return cls(
            adapter_id=raw["adapter_id"],  # type: ignore[arg-type]
            adapter_version=raw["adapter_version"],  # type: ignore[arg-type]
            architecture=raw["architecture"],  # type: ignore[arg-type]
            source_model_id=raw["source_model_id"],  # type: ignore[arg-type]
            source_revision=raw["source_revision"],  # type: ignore[arg-type]
            source_state_sha256=raw["source_state_sha256"],  # type: ignore[arg-type]
            source_config_sha256=raw[  # type: ignore[arg-type]
                "source_config_sha256"
            ],
            source_resource=raw["source_resource"],  # type: ignore[arg-type]
            layer_ids=tuple(layer_ids),
        )


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    policy: str
    minimum_length: int
    maximum_length: int | None
    causal: bool
    attention_mask: str
    padding: str
    position_ids: str
    cache: str

    def __post_init__(self) -> None:
        if self.policy not in _SEQUENCE_POLICIES:
            raise ValueError("unsupported sequence policy")
        minimum = _require_int(
            self.minimum_length,
            label="sequence minimum_length",
            minimum=1,
        )
        if self.policy == "dynamic":
            if self.maximum_length is not None:
                raise ValueError("dynamic sequence maximum_length must be null")
        else:
            maximum = _require_int(
                self.maximum_length,
                label="sequence maximum_length",
                minimum=minimum,
            )
            if self.policy == "fixed" and maximum != minimum:
                raise ValueError(
                    "fixed sequence minimum_length and maximum_length must match"
                )
        _require_bool(self.causal, label="sequence causal")
        if self.attention_mask not in _ATTENTION_MASK_POLICIES:
            raise ValueError("unsupported attention_mask policy")
        if self.padding not in _PADDING_POLICIES:
            raise ValueError("unsupported padding policy")
        if self.position_ids not in _POSITION_ID_POLICIES:
            raise ValueError("unsupported position_ids policy")
        if self.cache not in _CACHE_POLICIES:
            raise ValueError("unsupported cache policy")
        if self.padding != "none" and self.attention_mask in (
            "unsupported",
            "optional_all_true",
        ):
            raise ValueError("padding requires a real attention mask")
        if (
            self.attention_mask == "optional_all_true"
            and self.padding != "none"
        ):
            raise ValueError("optional_all_true does not support padding")

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "minimum_length": self.minimum_length,
            "maximum_length": self.maximum_length,
            "causal": self.causal,
            "attention_mask": self.attention_mask,
            "padding": self.padding,
            "position_ids": self.position_ids,
            "cache": self.cache,
        }

    @classmethod
    def from_dict(cls, value: object) -> SequenceSpec:
        raw = _require_exact_keys(
            value,
            {
                "policy",
                "minimum_length",
                "maximum_length",
                "causal",
                "attention_mask",
                "padding",
                "position_ids",
                "cache",
            },
            label="sequence",
        )
        return cls(**dict(raw))  # type: ignore[arg-type]

    def is_subset_of(self, other: SequenceSpec) -> bool:
        if not isinstance(other, SequenceSpec):
            return False
        if (
            self.causal != other.causal
            or self.minimum_length < other.minimum_length
            or not _MASK_INPUT_FORMS[self.attention_mask].issubset(
                _MASK_INPUT_FORMS[other.attention_mask]
            )
            or not _PADDING_INPUT_FORMS[self.padding].issubset(
                _PADDING_INPUT_FORMS[other.padding]
            )
            or not _POSITION_INPUT_FORMS[self.position_ids].issubset(
                _POSITION_INPUT_FORMS[other.position_ids]
            )
            or not _CACHE_EXECUTION_FORMS[self.cache].issubset(
                _CACHE_EXECUTION_FORMS[other.cache]
            )
        ):
            return False
        self_max = math.inf if self.maximum_length is None else self.maximum_length
        other_max = (
            math.inf if other.maximum_length is None else other.maximum_length
        )
        return self_max <= other_max


@dataclass(frozen=True, slots=True)
class BackendSpec:
    id: str
    abi_version: int

    def __post_init__(self) -> None:
        _require_identifier(self.id, label="backend id")
        _require_int(
            self.abi_version,
            label="backend abi_version",
            minimum=1,
        )

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "abi_version": self.abi_version}

    @classmethod
    def from_dict(cls, value: object) -> BackendSpec:
        raw = _require_exact_keys(
            value,
            {"id", "abi_version"},
            label="backend",
        )
        return cls(**dict(raw))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class InstrumentationResource:
    role: str
    resource: str

    def __post_init__(self) -> None:
        _require_identifier(self.role, label="instrumentation role")
        _require_identifier(self.resource, label="instrumentation resource")

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "resource": self.resource}

    @classmethod
    def from_dict(cls, value: object) -> InstrumentationResource:
        raw = _require_exact_keys(
            value,
            {"role", "resource"},
            label="instrumentation resource",
        )
        return cls(**dict(raw))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SegmentProvenance:
    source_model_state_sha256: str
    source_model_config_sha256: str
    dependency_resources: tuple[str, ...]
    compile_config_sha256: str | None

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_model_state_sha256,
            label="segment provenance source_model_state_sha256",
        )
        _require_sha256(
            self.source_model_config_sha256,
            label="segment provenance source_model_config_sha256",
        )
        _tuple_of_identifiers(
            self.dependency_resources,
            label="segment provenance dependency_resources",
        )
        if self.compile_config_sha256 is not None:
            _require_sha256(
                self.compile_config_sha256,
                label="segment provenance compile_config_sha256",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_model_state_sha256": self.source_model_state_sha256,
            "source_model_config_sha256": self.source_model_config_sha256,
            "dependency_resources": list(self.dependency_resources),
            "compile_config_sha256": self.compile_config_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> SegmentProvenance:
        raw = _require_exact_keys(
            value,
            {
                "source_model_state_sha256",
                "source_model_config_sha256",
                "dependency_resources",
                "compile_config_sha256",
            },
            label="segment provenance",
        )
        resources = raw["dependency_resources"]
        if type(resources) is not list:
            raise ValueError("dependency_resources must be an array")
        return cls(
            source_model_state_sha256=raw[  # type: ignore[arg-type]
                "source_model_state_sha256"
            ],
            source_model_config_sha256=raw[  # type: ignore[arg-type]
                "source_model_config_sha256"
            ],
            dependency_resources=tuple(resources),
            compile_config_sha256=raw[  # type: ignore[arg-type]
                "compile_config_sha256"
            ],
        )


@dataclass(frozen=True, slots=True)
class SegmentValidation:
    status: str
    validator_id: str
    validator_version: int
    report_resource: str | None

    def __post_init__(self) -> None:
        if self.status not in _VALIDATION_STATUSES:
            raise ValueError("unsupported segment validation status")
        _require_identifier(self.validator_id, label="validator_id")
        _require_int(
            self.validator_version,
            label="validator_version",
            minimum=1,
        )
        if self.report_resource is not None:
            _require_identifier(
                self.report_resource,
                label="validation report_resource",
            )
        if self.status in ("passed", "failed") and self.report_resource is None:
            raise ValueError(
                f"{self.status} validation requires a report_resource"
            )
        if self.status == "not_run" and self.report_resource is not None:
            raise ValueError(
                "not_run validation cannot have a report_resource"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "validator_id": self.validator_id,
            "validator_version": self.validator_version,
            "report_resource": self.report_resource,
        }

    @classmethod
    def from_dict(cls, value: object) -> SegmentValidation:
        raw = _require_exact_keys(
            value,
            {
                "status",
                "validator_id",
                "validator_version",
                "report_resource",
            },
            label="segment validation",
        )
        return cls(**dict(raw))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CompiledSegment:
    id: str
    order: int
    source_layers: tuple[str, ...]
    input_activation: str
    output_activation: str
    backend: BackendSpec
    sequence: SequenceSpec
    fast_resources: tuple[str, ...]
    instrumentation_resources: tuple[InstrumentationResource, ...]
    instrumentation_policy: str
    fallback_policy: str
    provenance: SegmentProvenance
    validation: SegmentValidation

    def __post_init__(self) -> None:
        _require_identifier(self.id, label="segment id")
        _require_int(self.order, label="segment order", minimum=0)
        _tuple_of_identifiers(
            self.source_layers,
            label=f"segment {self.id} source_layers",
        )
        _require_identifier(
            self.input_activation,
            label=f"segment {self.id} input_activation",
        )
        _require_identifier(
            self.output_activation,
            label=f"segment {self.id} output_activation",
        )
        if not isinstance(self.backend, BackendSpec):
            raise ValueError("segment backend must be a BackendSpec")
        if not isinstance(self.sequence, SequenceSpec):
            raise ValueError("segment sequence must be a SequenceSpec")
        _tuple_of_identifiers(
            self.fast_resources,
            label=f"segment {self.id} fast_resources",
        )
        instrumentation = _require_tuple(
            self.instrumentation_resources,
            label=f"segment {self.id} instrumentation_resources",
        )
        if any(
            not isinstance(item, InstrumentationResource)
            for item in instrumentation
        ):
            raise ValueError(
                "instrumentation_resources must contain "
                "InstrumentationResource values"
            )
        roles = [item.role for item in instrumentation]
        if len(set(roles)) != len(roles):
            raise ValueError("instrumentation roles must be unique")
        if self.instrumentation_policy not in _INSTRUMENTATION_POLICIES:
            raise ValueError("unsupported instrumentation policy")
        if self.instrumentation_policy == "none" and instrumentation:
            raise ValueError(
                "instrumentation policy none cannot reference resources"
            )
        if self.instrumentation_policy != "none" and not instrumentation:
            raise ValueError(
                "instrumentation policy requires instrumentation resources"
            )
        if self.fallback_policy not in _FALLBACK_POLICIES:
            raise ValueError("unsupported fallback policy")
        if not isinstance(self.provenance, SegmentProvenance):
            raise ValueError("segment provenance must be SegmentProvenance")
        if not isinstance(self.validation, SegmentValidation):
            raise ValueError("segment validation must be SegmentValidation")
        if (
            self.validation.status != "passed"
            and self.fallback_policy == "disabled"
        ):
            raise ValueError(
                "an unvalidated segment requires source-model fallback"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "order": self.order,
            "source_layers": list(self.source_layers),
            "input_activation": self.input_activation,
            "output_activation": self.output_activation,
            "backend": self.backend.to_dict(),
            "sequence": self.sequence.to_dict(),
            "fast_resources": list(self.fast_resources),
            "instrumentation_resources": [
                item.to_dict() for item in self.instrumentation_resources
            ],
            "instrumentation_policy": self.instrumentation_policy,
            "fallback_policy": self.fallback_policy,
            "provenance": self.provenance.to_dict(),
            "validation": self.validation.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> CompiledSegment:
        raw = _require_exact_keys(
            value,
            {
                "id",
                "order",
                "source_layers",
                "input_activation",
                "output_activation",
                "backend",
                "sequence",
                "fast_resources",
                "instrumentation_resources",
                "instrumentation_policy",
                "fallback_policy",
                "provenance",
                "validation",
            },
            label="compiled segment",
        )
        for key in (
            "source_layers",
            "fast_resources",
            "instrumentation_resources",
        ):
            if type(raw[key]) is not list:
                raise ValueError(f"segment {key} must be an array")
        return cls(
            id=raw["id"],  # type: ignore[arg-type]
            order=raw["order"],  # type: ignore[arg-type]
            source_layers=tuple(raw["source_layers"]),  # type: ignore[arg-type]
            input_activation=raw["input_activation"],  # type: ignore[arg-type]
            output_activation=raw["output_activation"],  # type: ignore[arg-type]
            backend=BackendSpec.from_dict(raw["backend"]),
            sequence=SequenceSpec.from_dict(raw["sequence"]),
            fast_resources=tuple(raw["fast_resources"]),  # type: ignore[arg-type]
            instrumentation_resources=tuple(
                InstrumentationResource.from_dict(item)
                for item in raw["instrumentation_resources"]  # type: ignore[union-attr]
            ),
            instrumentation_policy=raw[  # type: ignore[arg-type]
                "instrumentation_policy"
            ],
            fallback_policy=raw["fallback_policy"],  # type: ignore[arg-type]
            provenance=SegmentProvenance.from_dict(raw["provenance"]),
            validation=SegmentValidation.from_dict(raw["validation"]),
        )


@dataclass(frozen=True, slots=True)
class BuildIdentity:
    compiler_id: str
    compiler_version: str
    test_used_for_build_or_selection: bool

    def __post_init__(self) -> None:
        _require_identifier(self.compiler_id, label="compiler_id")
        _require_string(self.compiler_version, label="compiler_version")
        _require_bool(
            self.test_used_for_build_or_selection,
            label="test_used_for_build_or_selection",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "compiler_id": self.compiler_id,
            "compiler_version": self.compiler_version,
            "test_used_for_build_or_selection": (
                self.test_used_for_build_or_selection
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> BuildIdentity:
        raw = _require_exact_keys(
            value,
            {
                "compiler_id",
                "compiler_version",
                "test_used_for_build_or_selection",
            },
            label="build",
        )
        return cls(**dict(raw))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    schema: str
    schema_version: int
    model: ModelIdentity
    sequence: SequenceSpec
    resources: tuple[ArtifactDescriptor, ...]
    segments: tuple[CompiledSegment, ...]
    build: BuildIdentity
    annotations: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported runtime manifest schema")
        if type(self.schema_version) is not int or (
            self.schema_version != _SCHEMA_VERSION
        ):
            raise ValueError("unsupported runtime manifest schema version")
        if not isinstance(self.model, ModelIdentity):
            raise ValueError("manifest model must be ModelIdentity")
        if not isinstance(self.sequence, SequenceSpec):
            raise ValueError("manifest sequence must be SequenceSpec")
        resources = _require_tuple(self.resources, label="manifest resources")
        if not resources or any(
            not isinstance(item, ArtifactDescriptor) for item in resources
        ):
            raise ValueError(
                "manifest resources must contain ArtifactDescriptor values"
            )
        segments = _require_tuple(self.segments, label="manifest segments")
        if not segments or any(
            not isinstance(item, CompiledSegment) for item in segments
        ):
            raise ValueError(
                "manifest segments must contain CompiledSegment values"
            )
        if not isinstance(self.build, BuildIdentity):
            raise ValueError("manifest build must be BuildIdentity")
        portable_annotations = _portable_json(
            self.annotations,
            label="annotations",
        )
        if not isinstance(portable_annotations, dict):
            raise ValueError("annotations must be an object")
        object.__setattr__(
            self,
            "annotations",
            _freeze_json(portable_annotations),
        )

        resource_ids = [item.id for item in resources]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("manifest resource ids must be unique")
        resource_set = set(resource_ids)
        if (
            self.model.source_resource is not None
            and self.model.source_resource not in resource_set
        ):
            raise ValueError("model source_resource is not declared")

        segment_ids = [item.id for item in segments]
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("manifest segment ids must be unique")
        orders = [item.order for item in segments]
        if orders != list(range(len(segments))):
            raise ValueError(
                "segments must be stored in contiguous execution order"
            )
        known_layers = set(self.model.layer_ids)
        layer_positions = {
            layer_id: index
            for index, layer_id in enumerate(self.model.layer_ids)
        }
        claimed_layers: set[str] = set()
        previous_position = -1
        for segment in segments:
            unknown_layers = set(segment.source_layers) - known_layers
            if unknown_layers:
                raise ValueError(
                    f"segment {segment.id} references unknown source layers"
                )
            positions = [
                layer_positions[layer_id]
                for layer_id in segment.source_layers
            ]
            if positions != list(
                range(positions[0], positions[0] + len(positions))
            ):
                raise ValueError(
                    f"segment {segment.id} source layers are not contiguous"
                )
            if positions[0] <= previous_position:
                raise ValueError(
                    "segment source layers must follow execution order"
                )
            previous_position = positions[-1]
            overlap = claimed_layers.intersection(segment.source_layers)
            if overlap:
                raise ValueError("compiled segments cannot overlap source layers")
            claimed_layers.update(segment.source_layers)
            if segment.fallback_policy == "source_model":
                if not segment.sequence.is_subset_of(self.sequence):
                    raise ValueError(
                        f"segment {segment.id} sequence guard is outside the "
                        "runtime sequence contract"
                    )
            elif not self.sequence.is_subset_of(segment.sequence):
                raise ValueError(
                    f"segment {segment.id} does not cover the runtime "
                    "sequence contract and has no fallback"
                )
            referenced = (
                set(segment.fast_resources)
                | {
                    item.resource
                    for item in segment.instrumentation_resources
                }
                | set(segment.provenance.dependency_resources)
            )
            if segment.validation.report_resource is not None:
                referenced.add(segment.validation.report_resource)
            if referenced - resource_set:
                raise ValueError(
                    f"segment {segment.id} has undeclared resource references"
                )
            if (
                segment.provenance.source_model_state_sha256
                != self.model.source_state_sha256
            ):
                raise ValueError(
                    f"segment {segment.id} source model hash mismatch"
                )
            if (
                segment.provenance.source_model_config_sha256
                != self.model.source_config_sha256
            ):
                raise ValueError(
                    f"segment {segment.id} source config hash mismatch"
                )
            if (
                segment.fallback_policy == "source_model"
                and self.model.source_resource is None
                and self.model.source_model_id is None
            ):
                raise ValueError(
                    "source-model fallback requires a local or pinned "
                    "external source locator"
                )
        if (
            claimed_layers != known_layers
            and self.model.source_resource is None
            and self.model.source_model_id is None
        ):
            raise ValueError(
                "uncompiled source layers require a local or pinned "
                "external source locator"
            )

    def resource(self, resource_id: str) -> ArtifactDescriptor:
        for descriptor in self.resources:
            if descriptor.id == resource_id:
                return descriptor
        raise KeyError(resource_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "sequence": self.sequence.to_dict(),
            "resources": [item.to_dict() for item in self.resources],
            "segments": [item.to_dict() for item in self.segments],
            "build": self.build.to_dict(),
            "annotations": _portable_json(
                self.annotations,
                label="annotations",
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimeManifest:
        raw = _require_exact_keys(
            value,
            {
                "schema",
                "schema_version",
                "model",
                "sequence",
                "resources",
                "segments",
                "build",
                "annotations",
            },
            label="runtime manifest",
        )
        for key in ("resources", "segments"):
            if type(raw[key]) is not list:
                raise ValueError(f"manifest {key} must be an array")
        return cls(
            schema=raw["schema"],  # type: ignore[arg-type]
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            model=ModelIdentity.from_dict(raw["model"]),
            sequence=SequenceSpec.from_dict(raw["sequence"]),
            resources=tuple(
                ArtifactDescriptor.from_dict(item)
                for item in raw["resources"]  # type: ignore[union-attr]
            ),
            segments=tuple(
                CompiledSegment.from_dict(item)
                for item in raw["segments"]  # type: ignore[union-attr]
            ),
            build=BuildIdentity.from_dict(raw["build"]),
            annotations=raw["annotations"],  # type: ignore[arg-type]
        )


def runtime_manifest_bytes(manifest: RuntimeManifest) -> bytes:
    if not isinstance(manifest, RuntimeManifest):
        raise TypeError("manifest must be a RuntimeManifest")
    return (
        json.dumps(
            manifest.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def save_runtime_manifest(
    path: str | Path,
    manifest: RuntimeManifest,
) -> None:
    """Atomically save a manifest in its canonical JSON representation."""

    destination = Path(path)
    payload = runtime_manifest_bytes(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON contains non-finite value {value}")


def _decode_json(data: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def load_runtime_manifest(path: str | Path) -> RuntimeManifest:
    """Parse a manifest without opening any referenced resource."""

    return RuntimeManifest.from_dict(
        _decode_json(Path(path).read_bytes(), label="runtime manifest")
    )


def _open_resource_fd(
    root: str | Path,
    descriptor: ArtifactDescriptor,
) -> int:
    if not isinstance(descriptor, ArtifactDescriptor):
        raise TypeError("descriptor must be an ArtifactDescriptor")
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError("resource root cannot be a symlink")
    try:
        resolved_root = root_path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"resource root is missing: {root_path}") from None
    if not resolved_root.is_dir():
        raise ValueError("resource root must be a directory")

    candidate = resolved_root
    for component in PurePosixPath(descriptor.path).parts:
        candidate = candidate / component
        if candidate.is_symlink():
            raise ValueError(
                f"resource {descriptor.id} cannot traverse a symlink"
            )
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"resource {descriptor.id} is missing: {candidate}"
        ) from None
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"resource {descriptor.id} escapes the resource root"
        ) from None

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor_fd = os.open(resolved_candidate, flags)
    except OSError as error:
        raise ValueError(
            f"resource {descriptor.id} could not be opened safely"
        ) from error
    try:
        file_status = os.fstat(descriptor_fd)
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError(f"resource {descriptor.id} must be a regular file")
        if file_status.st_size != descriptor.size_bytes:
            raise ValueError(f"resource {descriptor.id} size mismatch")
    except BaseException:
        os.close(descriptor_fd)
        raise
    return descriptor_fd


@contextmanager
def open_verified_resource(
    root: str | Path,
    descriptor: ArtifactDescriptor,
) -> Iterator[BinaryIO]:
    """Yield a read-only, seekable snapshot of the authenticated bytes.

    The snapshot is disk-backed, so large resources are not materialized in
    memory.  Copying while hashing also closes the source-file TOCTOU gap: a
    later in-place mutation of the artifact cannot change what the backend
    reads from the yielded handle.
    """

    descriptor_fd = _open_resource_fd(root, descriptor)
    with (
        os.fdopen(descriptor_fd, "rb") as source,
        tempfile.TemporaryFile(mode="w+b") as snapshot,
    ):
        digest = hashlib.sha256()
        byte_count = 0
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
            snapshot.write(chunk)
        if byte_count != descriptor.size_bytes:
            raise ValueError(
                f"resource {descriptor.id} changed while hashing"
            )
        if digest.hexdigest() != descriptor.sha256:
            raise ValueError(f"resource {descriptor.id} SHA-256 mismatch")
        snapshot.flush()
        snapshot.seek(0)
        with os.fdopen(os.dup(snapshot.fileno()), "rb") as authenticated:
            yield authenticated


def resolve_resource_bytes(
    root: str | Path,
    descriptor: ArtifactDescriptor,
) -> bytes:
    """Authenticate a small resource and return its exact bytes."""

    with open_verified_resource(root, descriptor) as handle:
        return handle.read()


def resolve_manifest_resource(
    root: str | Path,
    manifest: RuntimeManifest,
    resource_id: str,
) -> bytes:
    return resolve_resource_bytes(root, manifest.resource(resource_id))


@contextmanager
def open_verified_manifest_resource(
    root: str | Path,
    manifest: RuntimeManifest,
    resource_id: str,
) -> Iterator[BinaryIO]:
    with open_verified_resource(
        root,
        manifest.resource(resource_id),
    ) as handle:
        yield handle


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_descriptor(
    root: Path,
    *,
    resource_id: str,
    filename: object,
    expected_sha256: object | None,
    expected_size: object | None,
    encoding: str,
    artifact_kind: str,
    format_version: int,
) -> ArtifactDescriptor:
    path = _require_resource_path(
        filename,
        label=f"legacy {resource_id} filename",
    )
    candidate = root / PurePosixPath(path)
    if expected_sha256 is None:
        expected_sha256 = _file_sha256(candidate)
    if expected_size is None:
        try:
            expected_size = candidate.stat().st_size
        except FileNotFoundError:
            raise FileNotFoundError(
                f"legacy resource {resource_id} is missing: {candidate}"
            ) from None
    descriptor = ArtifactDescriptor(
        id=resource_id,
        path=path,
        sha256=expected_sha256,  # type: ignore[arg-type]
        size_bytes=expected_size,  # type: ignore[arg-type]
        encoding=encoding,
        artifact_kind=artifact_kind,
        format_version=format_version,
    )
    resolve_resource_bytes(root, descriptor)
    return descriptor


def manifest_from_legacy_runtime(
    artifact_directory: str | Path,
) -> RuntimeManifest:
    """Losslessly wrap the checked two-layer runtime without rewriting it.

    The existing fused and Fisher reports duplicate the legacy PyTorch
    envelope's execution config, hashes, and sidecar descriptors. Reading
    those JSON files lets migration remain stdlib-only and ensures this
    function never deserializes a tensor artifact. Every referenced file is
    nevertheless authenticated before the manifest is returned.
    """

    root = Path(artifact_directory)
    report_path = root / "fused_executor_report.json"
    report = _decode_json(
        report_path.read_bytes(),
        label="legacy fused executor report",
    )
    analysis_report = _decode_json(
        (root / "fisher_report.json").read_bytes(),
        label="legacy Fisher report",
    )
    try:
        if (
            type(report["format_version"]) is not int
            or report["format_version"] != 2
        ):
            raise ValueError("unsupported legacy fused report format")
        lazy = report["lazy_fused_artifact"]
        sources = report["source_artifacts"]
        protocol = report["protocol"]
        if not all(
            isinstance(value, Mapping)
            for value in (lazy, sources, protocol)
        ):
            raise ValueError("legacy fused report sections must be objects")
        lazy = lazy  # type: ignore[assignment]
        sources = sources  # type: ignore[assignment]
        protocol = protocol  # type: ignore[assignment]
        if (
            type(lazy["format_version"]) is not int
            or lazy["format_version"] != 2
            or lazy["artifact_kind"]
            != "lazy_fused_two_layer_modal_stack"
        ):
            raise ValueError("unsupported legacy lazy runtime artifact")
        config = lazy["config"]
        metadata = lazy["metadata"]
        sidecars = lazy["sidecar_descriptors"]
        if not all(
            isinstance(value, Mapping)
            for value in (config, metadata, sidecars)
        ):
            raise ValueError("legacy lazy runtime sections must be objects")
        first = config["first"]  # type: ignore[index]
        second = config["second"]  # type: ignore[index]
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise ValueError("legacy layer configs must be objects")
        if config["cross_layer_bypass"] is not True:  # type: ignore[index]
            raise ValueError("legacy runtime lacks the exact layer bypass")
        sequence_length = _require_int(
            first["sequence_length"],
            label="legacy sequence length",
            minimum=1,
        )
        if (
            type(second["sequence_length"]) is not int
            or second["sequence_length"] != sequence_length
        ):
            raise ValueError("legacy layer sequence lengths differ")
        teacher_before = _require_sha256(
            report["teacher_state_sha256_before"],
            label="legacy teacher state hash",
        )
        if report["teacher_state_sha256_after"] != teacher_before:
            raise ValueError("legacy teacher state changed during compilation")
        if metadata["teacher_state_sha256"] != teacher_before:  # type: ignore[index]
            raise ValueError("legacy runtime teacher hash mismatch")
        validation_passed = _require_bool(
            protocol["validation_gate_passed"],  # type: ignore[index]
            label="legacy validation gate",
        )
        test_used = _require_bool(
            protocol["test_used_for_build_or_selection"],  # type: ignore[index]
            label="legacy test-selection flag",
        )
        model_config = analysis_report["model"]
        analysis_artifacts = analysis_report["artifacts"]
        if not isinstance(model_config, Mapping) or not isinstance(
            analysis_artifacts,
            Mapping,
        ):
            raise ValueError(
                "legacy Fisher model and artifacts must be objects"
            )
        source_config_sha256 = _portable_json_sha256(
            model_config,
            label="legacy source model config",
        )

        source_specs = (
            ("checkpoint", "checkpoint", "toy_transformer_checkpoint"),
            ("fisher", "fisher_modes", "fisher_build"),
            ("split_manifest", "split_manifest", "split_manifest"),
        )
        resources: list[ArtifactDescriptor] = []
        for resource_id, source_key, artifact_kind in source_specs:
            source = sources[source_key]  # type: ignore[index]
            if not isinstance(source, Mapping):
                raise ValueError(f"legacy source {source_key} must be an object")
            encoding = "json" if source_key == "split_manifest" else (
                "torch_weights_only"
            )
            descriptor = _legacy_descriptor(
                root,
                resource_id=resource_id,
                filename=source["filename"],
                expected_sha256=source["sha256"],
                expected_size=None,
                encoding=encoding,
                artifact_kind=artifact_kind,
                format_version=1,
            )
            resources.append(descriptor)
        expected_top_hashes = {
            "checkpoint": report["checkpoint_sha256"],
            "fisher": report["fisher_sha256"],
            "split_manifest": report["split_manifest_sha256"],
        }
        for descriptor in resources:
            if descriptor.sha256 != expected_top_hashes[descriptor.id]:
                raise ValueError(
                    f"legacy {descriptor.id} provenance hash mismatch"
                )
        if (
            analysis_artifacts["checkpoint_sha256"]
            != expected_top_hashes["checkpoint"]
        ):
            raise ValueError(
                "legacy Fisher report checkpoint provenance mismatch"
            )

        report_descriptor = _legacy_descriptor(
            root,
            resource_id="fused.report",
            filename=report_path.name,
            expected_sha256=None,
            expected_size=None,
            encoding="json",
            artifact_kind="fused_executor_report",
            format_version=2,
        )
        resources.append(report_descriptor)
        resources.append(
            _legacy_descriptor(
                root,
                resource_id="runtime.fast",
                filename=lazy["filename"],  # type: ignore[index]
                expected_sha256=lazy["sha256"],  # type: ignore[index]
                expected_size=None,
                encoding="torch_weights_only",
                artifact_kind="lazy_fused_two_layer_modal_stack",
                format_version=2,
            )
        )

        sidecar_bindings = (
            (
                "layer_0_executor",
                "layer.0.executor",
                "causal_position_modal_mlp",
            ),
            (
                "layer_0_output_completion",
                "layer.0.output_completion",
                "position_conditioned_modal_completion",
            ),
            (
                "layer_1_executor",
                "layer.1.executor",
                "causal_position_modal_mlp",
            ),
            (
                "layer_1_output_completion",
                "layer.1.output_completion",
                "position_conditioned_modal_completion",
            ),
        )
        instrumentation: list[InstrumentationResource] = []
        for role, resource_id, artifact_kind in sidecar_bindings:
            sidecar = sidecars[role]  # type: ignore[index]
            source = sources[role]  # type: ignore[index]
            if not isinstance(sidecar, Mapping) or not isinstance(
                source,
                Mapping,
            ):
                raise ValueError(f"legacy sidecar {role} must be an object")
            if (
                sidecar["filename"] != source["filename"]
                or sidecar["sha256"] != source["sha256"]
            ):
                raise ValueError(f"legacy sidecar {role} descriptor mismatch")
            resources.append(
                _legacy_descriptor(
                    root,
                    resource_id=resource_id,
                    filename=sidecar["filename"],
                    expected_sha256=sidecar["sha256"],
                    expected_size=sidecar["size_bytes"],
                    encoding="torch_weights_only",
                    artifact_kind=artifact_kind,
                    format_version=1,
                )
            )
            instrumentation.append(
                InstrumentationResource(role=role, resource=resource_id)
            )
    except KeyError as error:
        raise ValueError(
            f"legacy fused report is missing {error.args[0]!r}"
        ) from error

    sequence = SequenceSpec(
        policy="fixed",
        minimum_length=sequence_length,
        maximum_length=sequence_length,
        causal=True,
        attention_mask="optional_all_true",
        padding="none",
        position_ids="unsupported",
        cache="none",
    )
    dependencies = (
        "checkpoint",
        "fisher",
        "split_manifest",
        "layer.0.executor",
        "layer.0.output_completion",
        "layer.1.executor",
        "layer.1.output_completion",
    )
    manifest = RuntimeManifest(
        schema=_SCHEMA,
        schema_version=_SCHEMA_VERSION,
        model=ModelIdentity(
            adapter_id="toy_transformer",
            adapter_version=1,
            architecture="fisher_graph.toy_transformer",
            source_model_id=None,
            source_revision=None,
            source_state_sha256=teacher_before,
            source_config_sha256=source_config_sha256,
            source_resource="checkpoint",
            layer_ids=("layer.0", "layer.1"),
        ),
        sequence=sequence,
        resources=tuple(sorted(resources, key=lambda item: item.id)),
        segments=(
            CompiledSegment(
                id="segment.0",
                order=0,
                source_layers=("layer.0", "layer.1"),
                input_activation=_require_identifier(
                    first["input_activation"],
                    label="legacy first input activation",
                ),
                output_activation=_require_identifier(
                    second["output_activation"],
                    label="legacy second output activation",
                ),
                backend=BackendSpec(
                    id="fisher_graph.lazy_fused_two_layer",
                    abi_version=1,
                ),
                sequence=sequence,
                fast_resources=("runtime.fast",),
                instrumentation_resources=tuple(instrumentation),
                instrumentation_policy="lazy_fail_fast_only",
                fallback_policy="source_model",
                provenance=SegmentProvenance(
                    source_model_state_sha256=teacher_before,
                    source_model_config_sha256=source_config_sha256,
                    dependency_resources=dependencies,
                    compile_config_sha256=None,
                ),
                validation=SegmentValidation(
                    status="passed" if validation_passed else "failed",
                    validator_id="fisher_graph.fused_experiment_gate",
                    validator_version=1,
                    report_resource="fused.report",
                ),
            ),
        ),
        build=BuildIdentity(
            compiler_id="fisher_graph",
            compiler_version=_compiler_version(),
            test_used_for_build_or_selection=test_used,
        ),
        annotations={
            "migration": {
                "source_artifact": "fused_modal_runtime.pt",
                "source_format_version": 2,
                "legacy_files_rewritten": False,
            }
        },
    )
    return manifest


__all__ = [
    "ArtifactDescriptor",
    "BackendSpec",
    "BuildIdentity",
    "CompiledSegment",
    "InstrumentationResource",
    "ModelIdentity",
    "RuntimeManifest",
    "SegmentProvenance",
    "SegmentValidation",
    "SequenceSpec",
    "load_runtime_manifest",
    "manifest_from_legacy_runtime",
    "open_verified_manifest_resource",
    "open_verified_resource",
    "resolve_manifest_resource",
    "resolve_resource_bytes",
    "runtime_manifest_bytes",
    "save_runtime_manifest",
]
