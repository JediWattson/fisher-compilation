"""Prompt-blind state-conditioned reference-provider compilation.

This module is deliberately model agnostic.  A caller supplies synthetic
modal coordinates, row RMS values, target modes, masks, and logical positions.
No text, token, tokenizer, or model-specific type crosses the boundary.

The runtime feature ABI is fixed and inspectable::

    [1, whitened modal coordinates..., standardized null coordinates...,
     standardized log(row RMS)]

Those features drive :class:`ResidualGatedCausalModalExecutor`.  Targets are
fit in standardized coordinates and de-standardized only at the provider
boundary.  All persistent tensors are canonical CPU float64 values and are
bound into domain-separated SHA-256 artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor, nn

from fisher_graph.gated_executor import (
    GatedCausalExecutionAccounting,
    GatedCausalModalExecutorConfig,
    ResidualGatedCausalModalExecutor,
)


SyntheticSplit = Literal["fit", "selection", "assessment"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FEATURE_KIND = "fisher_graph.reference_provider_feature_codec"
_BATCH_KIND = "fisher_graph.synthetic_reference_batch"
_PLAN_KIND = "fisher_graph.state_conditioned_reference_provider"
_FORMAT_VERSION = 1
_FEATURE_DOMAIN = b"fisher_graph.reference_provider_feature_codec.v1\0"
_BATCH_CONTENT_DOMAIN = b"fisher_graph.synthetic_reference_batch.content.v1\0"
_BATCH_DOMAIN = b"fisher_graph.synthetic_reference_batch.v1\0"
_PLAN_DOMAIN = b"fisher_graph.state_conditioned_reference_provider.v1\0"
_EVALUATION_DOMAIN = (
    b"fisher_graph.state_conditioned_reference_provider.evaluation.v1\0"
)
_TENSOR_DOMAIN = b"fisher_graph.state_conditioned_reference_provider.tensor.v1\0"
_EXECUTOR_DOMAIN = (
    b"fisher_graph.state_conditioned_reference_provider.executor.v1\0"
)

_FEATURE_STATE_FIELDS = {
    "artifact_kind",
    "format_version",
    "modal_center",
    "modal_whitener",
    "null_center",
    "null_scale",
    "log_rms_center",
    "log_rms_scale",
    "source_binding_sha256",
    "artifact_sha256",
}
_BATCH_STATE_FIELDS = {
    "artifact_kind",
    "format_version",
    "split",
    "modal_coordinates",
    "null_coordinates",
    "row_rms",
    "target_modes",
    "logical_positions",
    "valid_mask",
    "synthetic_binding_sha256",
    "content_sha256",
    "artifact_sha256",
}
_PLAN_STATE_FIELDS = {
    "artifact_kind",
    "format_version",
    "feature_codec_state",
    "target_center",
    "target_scale",
    "executor_artifact",
    "synthetic_binding_sha256",
    "fit_batch_sha256s",
    "fit_batch_content_sha256s",
    "training_steps",
    "learning_rate",
    "seed",
    "initial_standardized_mse",
    "final_standardized_mse",
    "artifact_sha256",
}
_EXECUTOR_ARTIFACT_FIELDS = {
    "artifact_kind",
    "format_version",
    "config",
    "dtype",
    "model_state_dict",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_keys(
    state: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = set(state)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch: expected {sorted(expected)}, "
            f"got {sorted(actual)}"
        )


def _positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _canonical_float_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must be floating point")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if result.ndim != ndim or any(int(width) <= 0 for width in result.shape):
        raise ValueError(f"{label} must be nonempty and rank {ndim}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _canonical_positions(value: Tensor, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{label} must use int32 or int64")
    result = value.detach().to(device="cpu", dtype=torch.int64).contiguous().clone()
    if result.ndim != 2 or any(int(width) <= 0 for width in result.shape):
        raise ValueError(f"{label} must be nonempty and rank 2")
    return result


def _canonical_mask(value: Tensor, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.dtype is not torch.bool:
        raise TypeError(f"{label} must be boolean")
    result = value.detach().to(device="cpu").contiguous().clone()
    if result.ndim != 2 or any(int(width) <= 0 for width in result.shape):
        raise ValueError(f"{label} must be nonempty and rank 2")
    return result


def _require_canonical_tensor(
    value: object,
    *,
    label: str,
    dtype: torch.dtype,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{label} must be on CPU")
    if value.dtype != dtype:
        raise ValueError(f"{label} must use {dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{label} must be contiguous")
    if value.ndim != ndim or any(int(width) <= 0 for width in value.shape):
        raise ValueError(f"{label} must be nonempty and rank {ndim}")
    if dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must contain only finite values")
    return value.detach().clone()


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(int(width) for width in canonical.shape)).encode())
    digest.update(b"\0")
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sha256_tuple(
    values: Sequence[str],
    *,
    label: str,
    nonempty: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(
        _require_sha256(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    )
    if nonempty and not result:
        raise ValueError(f"{label} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _clone_executor_artifact(
    artifact: Mapping[str, object],
    *,
    strict_canonical: bool,
) -> dict[str, object]:
    _strict_keys(
        artifact,
        expected=_EXECUTOR_ARTIFACT_FIELDS,
        label="executor artifact",
    )
    if artifact["dtype"] != "float64":
        raise ValueError("provider executor artifact must use float64")
    raw_state = artifact["model_state_dict"]
    if not isinstance(raw_state, Mapping):
        raise TypeError("executor model_state_dict must be a mapping")
    if strict_canonical:
        for name, value in raw_state.items():
            _require_canonical_tensor(
                value,
                label=f"executor model_state_dict[{name!r}]",
                dtype=torch.float64,
                ndim=value.ndim if isinstance(value, Tensor) else 0,
            )
    restored = _restore_executor_artifact(artifact)
    canonical = restored.artifact_state_dict()
    canonical_state = canonical["model_state_dict"]
    assert isinstance(canonical_state, Mapping)
    return {
        "artifact_kind": canonical["artifact_kind"],
        "format_version": canonical["format_version"],
        "config": dict(canonical["config"]),
        "dtype": canonical["dtype"],
        "model_state_dict": {
            name: value.detach().contiguous().clone()
            for name, value in sorted(canonical_state.items())
        },
    }


def _restore_executor_artifact(
    artifact: Mapping[str, object],
) -> ResidualGatedCausalModalExecutor:
    """Restore without consuming the caller's CPU random stream."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return ResidualGatedCausalModalExecutor.from_artifact_state_dict(
            artifact
        )


def _executor_artifact_sha256(artifact: Mapping[str, object]) -> str:
    raw_state = artifact["model_state_dict"]
    assert isinstance(raw_state, Mapping)
    return _json_sha256(
        {
            "artifact_kind": artifact["artifact_kind"],
            "format_version": artifact["format_version"],
            "config": dict(artifact["config"]),
            "dtype": artifact["dtype"],
            "model_state_dict_sha256": {
                name: _tensor_sha256(value)
                for name, value in sorted(raw_state.items())
            },
        },
        domain=_EXECUTOR_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class ReferenceProviderFeatureCodec:
    """Authenticated normalization for the provider's runtime feature ABI."""

    modal_center: Tensor
    modal_whitener: Tensor
    null_center: Tensor
    null_scale: Tensor
    log_rms_center: float
    log_rms_scale: float
    source_binding_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        center = _canonical_float_tensor(
            self.modal_center,
            label="modal_center",
            ndim=1,
        )
        whitener = _canonical_float_tensor(
            self.modal_whitener,
            label="modal_whitener",
            ndim=2,
        )
        if whitener.shape != (center.numel(), center.numel()):
            raise ValueError(
                "modal_whitener must be square with modal_center width"
            )
        null_center = _canonical_float_tensor(
            self.null_center,
            label="null_center",
            ndim=1,
        )
        null_scale = _canonical_float_tensor(
            self.null_scale,
            label="null_scale",
            ndim=1,
        )
        if null_scale.shape != null_center.shape:
            raise ValueError("null_scale must match null_center")
        if bool((null_scale <= 0).any()):
            raise ValueError("null_scale must be strictly positive")
        log_center = _finite_float(
            self.log_rms_center,
            label="log_rms_center",
        )
        log_scale = _positive_float(
            self.log_rms_scale,
            label="log_rms_scale",
        )
        source = _require_sha256(
            self.source_binding_sha256,
            label="source_binding_sha256",
        )
        object.__setattr__(self, "modal_center", center)
        object.__setattr__(self, "modal_whitener", whitener)
        object.__setattr__(self, "null_center", null_center)
        object.__setattr__(self, "null_scale", null_scale)
        object.__setattr__(self, "log_rms_center", log_center)
        object.__setattr__(self, "log_rms_scale", log_scale)
        object.__setattr__(self, "source_binding_sha256", source)
        expected = self._computed_sha256()
        if self.artifact_sha256:
            supplied = _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if supplied != expected:
                raise ValueError("feature codec artifact hash mismatch")
        object.__setattr__(self, "artifact_sha256", expected)

    @property
    def modal_modes(self) -> int:
        return int(self.modal_center.numel())

    @property
    def feature_modes(self) -> int:
        return self.modal_modes + self.null_modes + 2

    @property
    def null_modes(self) -> int:
        return int(self.null_center.numel())

    @property
    def stored_scalar_count(self) -> int:
        return (
            self.modal_modes
            + self.modal_modes**2
            + 2 * self.null_modes
            + 2
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": _FEATURE_KIND,
            "format_version": _FORMAT_VERSION,
            "modal_center_sha256": _tensor_sha256(self.modal_center),
            "modal_whitener_sha256": _tensor_sha256(self.modal_whitener),
            "null_center_sha256": _tensor_sha256(self.null_center),
            "null_scale_sha256": _tensor_sha256(self.null_scale),
            "log_rms_center": self.log_rms_center,
            "log_rms_scale": self.log_rms_scale,
            "source_binding_sha256": self.source_binding_sha256,
            "feature_order": (
                "constant",
                "fisher_whitened_modal_coordinates",
                "standardized_gain_null_coordinates",
                "standardized_log_row_rms",
            ),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_FEATURE_DOMAIN)

    def validate_integrity(self) -> None:
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("feature codec artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "modal_modes": self.modal_modes,
            "null_modes": self.null_modes,
            "feature_modes": self.feature_modes,
            "stored_scalar_count": self.stored_scalar_count,
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "artifact_kind": _FEATURE_KIND,
            "format_version": _FORMAT_VERSION,
            "modal_center": self.modal_center.clone(),
            "modal_whitener": self.modal_whitener.clone(),
            "null_center": self.null_center.clone(),
            "null_scale": self.null_scale.clone(),
            "log_rms_center": self.log_rms_center,
            "log_rms_scale": self.log_rms_scale,
            "source_binding_sha256": self.source_binding_sha256,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ReferenceProviderFeatureCodec:
        _strict_keys(
            state,
            expected=_FEATURE_STATE_FIELDS,
            label="feature codec state",
        )
        if state["artifact_kind"] != _FEATURE_KIND:
            raise ValueError("unsupported feature codec artifact kind")
        if state["format_version"] != _FORMAT_VERSION:
            raise ValueError("unsupported feature codec format version")
        center = _require_canonical_tensor(
            state["modal_center"],
            label="modal_center",
            dtype=torch.float64,
            ndim=1,
        )
        whitener = _require_canonical_tensor(
            state["modal_whitener"],
            label="modal_whitener",
            dtype=torch.float64,
            ndim=2,
        )
        null_center = _require_canonical_tensor(
            state["null_center"],
            label="null_center",
            dtype=torch.float64,
            ndim=1,
        )
        null_scale = _require_canonical_tensor(
            state["null_scale"],
            label="null_scale",
            dtype=torch.float64,
            ndim=1,
        )
        return cls(
            modal_center=center,
            modal_whitener=whitener,
            null_center=null_center,
            null_scale=null_scale,
            log_rms_center=state["log_rms_center"],
            log_rms_scale=state["log_rms_scale"],
            source_binding_sha256=state["source_binding_sha256"],
            artifact_sha256=state["artifact_sha256"],
        )

    def prepare(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> PreparedReferenceProviderFeatureCodec:
        self.validate_integrity()
        return PreparedReferenceProviderFeatureCodec(
            self,
            dtype=dtype,
            device=device,
        )


class PreparedReferenceProviderFeatureCodec:
    """Device/dtype-specific feature encoder prepared from a strict codec."""

    def __init__(
        self,
        codec: ReferenceProviderFeatureCodec,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("prepared codec dtype must be floating point")
        target_device = torch.device("cpu" if device is None else device)
        self.modal_center = codec.modal_center.to(
            device=target_device,
            dtype=dtype,
        )
        self.modal_whitener = codec.modal_whitener.to(
            device=target_device,
            dtype=dtype,
        )
        self.null_center = codec.null_center.to(
            device=target_device,
            dtype=dtype,
        )
        self.null_scale = codec.null_scale.to(
            device=target_device,
            dtype=dtype,
        )
        self.log_rms_center = codec.log_rms_center
        self.log_rms_scale = codec.log_rms_scale
        self.source_binding_sha256 = codec.source_binding_sha256
        self.artifact_sha256 = codec.artifact_sha256

    @property
    def modal_modes(self) -> int:
        return int(self.modal_center.numel())

    @property
    def feature_modes(self) -> int:
        return self.modal_modes + self.null_modes + 2

    @property
    def null_modes(self) -> int:
        return int(self.null_center.numel())

    @property
    def dtype(self) -> torch.dtype:
        return self.modal_center.dtype

    @property
    def device(self) -> torch.device:
        return self.modal_center.device

    def __call__(
        self,
        modal_coordinates: Tensor,
        null_coordinates: Tensor,
        row_rms: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if not isinstance(modal_coordinates, Tensor):
            raise TypeError("modal_coordinates must be a Tensor")
        if (
            modal_coordinates.ndim != 3
            or modal_coordinates.shape[2] != self.modal_modes
            or modal_coordinates.shape[0] == 0
            or modal_coordinates.shape[1] == 0
        ):
            raise ValueError(
                "modal_coordinates must have nonempty shape "
                "[batch, sequence, modal_modes]"
            )
        if (
            not modal_coordinates.is_floating_point()
            or modal_coordinates.dtype != self.dtype
            or modal_coordinates.device != self.device
        ):
            raise ValueError(
                "modal_coordinates must match prepared codec dtype and device"
            )
        shape = modal_coordinates.shape[:2]
        if (
            not isinstance(null_coordinates, Tensor)
            or null_coordinates.shape != (*shape, self.null_modes)
            or not null_coordinates.is_floating_point()
            or null_coordinates.dtype != self.dtype
            or null_coordinates.device != self.device
        ):
            raise ValueError(
                "null_coordinates must have shape [batch, sequence, "
                "null_modes] and match codec dtype/device"
            )
        if (
            not isinstance(row_rms, Tensor)
            or row_rms.shape != shape
            or not row_rms.is_floating_point()
            or row_rms.dtype != self.dtype
            or row_rms.device != self.device
        ):
            raise ValueError(
                "row_rms must match modal coordinate rows, dtype, and device"
            )
        if (
            not isinstance(valid_mask, Tensor)
            or valid_mask.shape != shape
            or valid_mask.dtype is not torch.bool
            or valid_mask.device != self.device
        ):
            raise ValueError(
                "valid_mask must be boolean and match coordinate rows/device"
            )
        if not bool(torch.isfinite(modal_coordinates).all()):
            raise ValueError("modal_coordinates must be finite")
        if not bool(torch.isfinite(null_coordinates).all()):
            raise ValueError("null_coordinates must be finite")
        if not bool(torch.isfinite(row_rms).all()) or bool((row_rms < 0).any()):
            raise ValueError("row_rms must be finite and nonnegative")
        if bool((row_rms[valid_mask] <= 0).any()):
            raise ValueError("row_rms must be positive on valid rows")

        centered = modal_coordinates - self.modal_center
        whitened = centered @ self.modal_whitener.transpose(0, 1)
        standardized_null = (
            null_coordinates - self.null_center
        ) / self.null_scale
        safe_rms = torch.where(valid_mask, row_rms, torch.ones_like(row_rms))
        standardized_log_rms = (
            torch.log(safe_rms) - self.log_rms_center
        ) / self.log_rms_scale
        features = torch.cat(
            (
                torch.ones(
                    (*shape, 1),
                    dtype=self.dtype,
                    device=self.device,
                ),
                whitened,
                standardized_null,
                standardized_log_rms.unsqueeze(-1),
            ),
            dim=-1,
        )
        return torch.where(
            valid_mask.unsqueeze(-1),
            features,
            torch.zeros_like(features),
        )


@dataclass(frozen=True, slots=True)
class SyntheticReferenceBatch:
    """Canonical synthetic features and targets with no text-facing types."""

    split: SyntheticSplit
    modal_coordinates: Tensor
    null_coordinates: Tensor
    row_rms: Tensor
    target_modes: Tensor
    logical_positions: Tensor
    valid_mask: Tensor
    synthetic_binding_sha256: str
    content_sha256: str = ""
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.split not in ("fit", "selection", "assessment"):
            raise ValueError(
                "split must be 'fit', 'selection', or 'assessment'"
            )
        coordinates = _canonical_float_tensor(
            self.modal_coordinates,
            label="modal_coordinates",
            ndim=3,
        )
        null_coordinates = _canonical_float_tensor(
            self.null_coordinates,
            label="null_coordinates",
            ndim=3,
        )
        row_rms = _canonical_float_tensor(
            self.row_rms,
            label="row_rms",
            ndim=2,
        )
        targets = _canonical_float_tensor(
            self.target_modes,
            label="target_modes",
            ndim=3,
        )
        positions = _canonical_positions(
            self.logical_positions,
            label="logical_positions",
        )
        mask = _canonical_mask(self.valid_mask, label="valid_mask")
        shape = coordinates.shape[:2]
        if (
            null_coordinates.shape[:2] != shape
            or
            row_rms.shape != shape
            or targets.shape[:2] != shape
            or positions.shape != shape
            or mask.shape != shape
        ):
            raise ValueError("all synthetic batch rows must share [batch, sequence]")
        if not bool(mask.any()):
            raise ValueError("synthetic batch must contain at least one valid row")
        if bool((row_rms < 0).any()):
            raise ValueError("row_rms must be nonnegative")
        if bool((row_rms[mask] <= 0).any()):
            raise ValueError("row_rms must be positive on valid rows")
        if bool((positions[mask] < 0).any()):
            raise ValueError("valid logical positions cannot be negative")
        for row in range(int(mask.shape[0])):
            selected = positions[row][mask[row]]
            if selected.numel() > 1 and bool((selected[1:] <= selected[:-1]).any()):
                raise ValueError(
                    "valid logical positions must be strictly increasing"
                )
        binding = _require_sha256(
            self.synthetic_binding_sha256,
            label="synthetic_binding_sha256",
        )
        object.__setattr__(self, "modal_coordinates", coordinates)
        object.__setattr__(self, "null_coordinates", null_coordinates)
        object.__setattr__(self, "row_rms", row_rms)
        object.__setattr__(self, "target_modes", targets)
        object.__setattr__(self, "logical_positions", positions)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "synthetic_binding_sha256", binding)

        expected_content = self._computed_content_sha256()
        if self.content_sha256:
            supplied_content = _require_sha256(
                self.content_sha256,
                label="content_sha256",
            )
            if supplied_content != expected_content:
                raise ValueError("synthetic batch content hash mismatch")
        object.__setattr__(self, "content_sha256", expected_content)

        expected_artifact = self._computed_artifact_sha256()
        if self.artifact_sha256:
            supplied_artifact = _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if supplied_artifact != expected_artifact:
                raise ValueError("synthetic batch artifact hash mismatch")
        object.__setattr__(self, "artifact_sha256", expected_artifact)

    @property
    def batch_size(self) -> int:
        return int(self.modal_coordinates.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.modal_coordinates.shape[1])

    @property
    def modal_modes(self) -> int:
        return int(self.modal_coordinates.shape[2])

    @property
    def target_mode_count(self) -> int:
        return int(self.target_modes.shape[2])

    @property
    def null_modes(self) -> int:
        return int(self.null_coordinates.shape[2])

    @property
    def valid_row_count(self) -> int:
        return int(self.valid_mask.sum().item())

    def _content_payload(self) -> dict[str, object]:
        return {
            "modal_coordinates_sha256": _tensor_sha256(
                self.modal_coordinates
            ),
            "null_coordinates_sha256": _tensor_sha256(
                self.null_coordinates
            ),
            "row_rms_sha256": _tensor_sha256(self.row_rms),
            "target_modes_sha256": _tensor_sha256(self.target_modes),
            "logical_positions_sha256": _tensor_sha256(
                self.logical_positions
            ),
            "valid_mask_sha256": _tensor_sha256(self.valid_mask),
            "synthetic_binding_sha256": self.synthetic_binding_sha256,
        }

    def _computed_content_sha256(self) -> str:
        return _json_sha256(
            self._content_payload(),
            domain=_BATCH_CONTENT_DOMAIN,
        )

    def _computed_artifact_sha256(self) -> str:
        return _json_sha256(
            {
                "artifact_kind": _BATCH_KIND,
                "format_version": _FORMAT_VERSION,
                "split": self.split,
                "content_sha256": self.content_sha256,
            },
            domain=_BATCH_DOMAIN,
        )

    def validate_integrity(self) -> None:
        if self._computed_content_sha256() != self.content_sha256:
            raise ValueError("synthetic batch content hash mismatch")
        if self._computed_artifact_sha256() != self.artifact_sha256:
            raise ValueError("synthetic batch artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "artifact_kind": _BATCH_KIND,
            "format_version": _FORMAT_VERSION,
            "split": self.split,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "modal_modes": self.modal_modes,
            "null_modes": self.null_modes,
            "target_mode_count": self.target_mode_count,
            "valid_row_count": self.valid_row_count,
            "synthetic_binding_sha256": self.synthetic_binding_sha256,
            "content_sha256": self.content_sha256,
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "artifact_kind": _BATCH_KIND,
            "format_version": _FORMAT_VERSION,
            "split": self.split,
            "modal_coordinates": self.modal_coordinates.clone(),
            "null_coordinates": self.null_coordinates.clone(),
            "row_rms": self.row_rms.clone(),
            "target_modes": self.target_modes.clone(),
            "logical_positions": self.logical_positions.clone(),
            "valid_mask": self.valid_mask.clone(),
            "synthetic_binding_sha256": self.synthetic_binding_sha256,
            "content_sha256": self.content_sha256,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> SyntheticReferenceBatch:
        _strict_keys(
            state,
            expected=_BATCH_STATE_FIELDS,
            label="synthetic batch state",
        )
        if state["artifact_kind"] != _BATCH_KIND:
            raise ValueError("unsupported synthetic batch artifact kind")
        if state["format_version"] != _FORMAT_VERSION:
            raise ValueError("unsupported synthetic batch format version")
        return cls(
            split=state["split"],
            modal_coordinates=_require_canonical_tensor(
                state["modal_coordinates"],
                label="modal_coordinates",
                dtype=torch.float64,
                ndim=3,
            ),
            null_coordinates=_require_canonical_tensor(
                state["null_coordinates"],
                label="null_coordinates",
                dtype=torch.float64,
                ndim=3,
            ),
            row_rms=_require_canonical_tensor(
                state["row_rms"],
                label="row_rms",
                dtype=torch.float64,
                ndim=2,
            ),
            target_modes=_require_canonical_tensor(
                state["target_modes"],
                label="target_modes",
                dtype=torch.float64,
                ndim=3,
            ),
            logical_positions=_require_canonical_tensor(
                state["logical_positions"],
                label="logical_positions",
                dtype=torch.int64,
                ndim=2,
            ),
            valid_mask=_require_canonical_tensor(
                state["valid_mask"],
                label="valid_mask",
                dtype=torch.bool,
                ndim=2,
            ),
            synthetic_binding_sha256=state["synthetic_binding_sha256"],
            content_sha256=state["content_sha256"],
            artifact_sha256=state["artifact_sha256"],
        )


@dataclass(frozen=True, slots=True)
class StateConditionedReferenceProviderAccounting:
    """Persistent scalar accounting; this is not a latency measurement."""

    modal_modes: int
    null_modes: int
    feature_modes: int
    target_modes: int
    feature_codec_scalar_count: int
    target_standardization_scalar_count: int
    executor_parameter_count: int
    total_stored_scalar_count: int


@dataclass(frozen=True, slots=True)
class StateConditionedReferenceProviderExecutionAccounting:
    """Logical MAC accounting for one prepared-provider invocation.

    Feature whitening and target scale multiplication are added to the core
    executor accounting.  Logs, concatenation, additions, masking, activation,
    and softmax are excluded.  Counts describe an ideal mathematical
    implementation, not measured wall-clock performance.
    """

    core: GatedCausalExecutionAccounting
    valid_rows: int
    modal_whitening_mac_count: int
    null_standardization_mac_count: int
    target_destandardization_mac_count: int
    total_mac_count: int


@dataclass(frozen=True, slots=True)
class StateConditionedReferenceProviderPlan:
    """Strict, hash-authenticated provider normalization and executor state."""

    feature_codec: ReferenceProviderFeatureCodec
    target_center: Tensor
    target_scale: Tensor
    executor_artifact: Mapping[str, object]
    synthetic_binding_sha256: str
    fit_batch_sha256s: tuple[str, ...]
    fit_batch_content_sha256s: tuple[str, ...]
    training_steps: int
    learning_rate: float
    seed: int
    initial_standardized_mse: float
    final_standardized_mse: float
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.feature_codec,
            ReferenceProviderFeatureCodec,
        ):
            raise TypeError(
                "feature_codec must be a ReferenceProviderFeatureCodec"
            )
        self.feature_codec.validate_integrity()
        target_center = _canonical_float_tensor(
            self.target_center,
            label="target_center",
            ndim=1,
        )
        target_scale = _canonical_float_tensor(
            self.target_scale,
            label="target_scale",
            ndim=1,
        )
        if target_scale.shape != target_center.shape:
            raise ValueError("target_scale must match target_center")
        if bool((target_scale <= 0).any()):
            raise ValueError("target_scale must be strictly positive")
        executor_artifact = _clone_executor_artifact(
            self.executor_artifact,
            strict_canonical=True,
        )
        restored = _restore_executor_artifact(executor_artifact)
        if restored.input_modes != self.feature_codec.feature_modes:
            raise ValueError(
                "executor input modes must match feature codec width"
            )
        if restored.output_modes != int(target_center.numel()):
            raise ValueError(
                "executor output modes must match target standardization"
            )
        if restored.config.same_position_skip:
            raise ValueError(
                "reference provider executor must disable same_position_skip"
            )
        binding = _require_sha256(
            self.synthetic_binding_sha256,
            label="synthetic_binding_sha256",
        )
        fit_artifacts = _sha256_tuple(
            self.fit_batch_sha256s,
            label="fit_batch_sha256s",
            nonempty=True,
        )
        fit_contents = _sha256_tuple(
            self.fit_batch_content_sha256s,
            label="fit_batch_content_sha256s",
            nonempty=True,
        )
        if len(fit_artifacts) != len(fit_contents):
            raise ValueError(
                "fit artifact and content hash lists must have equal length"
            )
        steps = _positive_integer(
            self.training_steps,
            label="training_steps",
        )
        learning_rate = _positive_float(
            self.learning_rate,
            label="learning_rate",
        )
        seed = _nonnegative_integer(self.seed, label="seed")
        initial_mse = _finite_float(
            self.initial_standardized_mse,
            label="initial_standardized_mse",
        )
        final_mse = _finite_float(
            self.final_standardized_mse,
            label="final_standardized_mse",
        )
        if initial_mse < 0.0 or final_mse < 0.0:
            raise ValueError("standardized MSE values must be nonnegative")

        object.__setattr__(self, "target_center", target_center)
        object.__setattr__(self, "target_scale", target_scale)
        object.__setattr__(self, "executor_artifact", executor_artifact)
        object.__setattr__(self, "synthetic_binding_sha256", binding)
        object.__setattr__(self, "fit_batch_sha256s", fit_artifacts)
        object.__setattr__(
            self,
            "fit_batch_content_sha256s",
            fit_contents,
        )
        object.__setattr__(self, "training_steps", steps)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(
            self,
            "initial_standardized_mse",
            initial_mse,
        )
        object.__setattr__(self, "final_standardized_mse", final_mse)
        expected = self._computed_sha256()
        if self.artifact_sha256:
            supplied = _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if supplied != expected:
                raise ValueError("reference provider plan hash mismatch")
        object.__setattr__(self, "artifact_sha256", expected)

    @property
    def target_modes(self) -> int:
        return int(self.target_center.numel())

    @property
    def executor_config(self) -> GatedCausalModalExecutorConfig:
        executor = _restore_executor_artifact(self.executor_artifact)
        return executor.config

    @property
    def executor_sha256(self) -> str:
        return _executor_artifact_sha256(self.executor_artifact)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": _PLAN_KIND,
            "format_version": _FORMAT_VERSION,
            "feature_codec_sha256": self.feature_codec.artifact_sha256,
            "target_center_sha256": _tensor_sha256(self.target_center),
            "target_scale_sha256": _tensor_sha256(self.target_scale),
            "executor_sha256": self.executor_sha256,
            "synthetic_binding_sha256": self.synthetic_binding_sha256,
            "fit_batch_sha256s": self.fit_batch_sha256s,
            "fit_batch_content_sha256s": self.fit_batch_content_sha256s,
            "training_steps": self.training_steps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "initial_standardized_mse": self.initial_standardized_mse,
            "final_standardized_mse": self.final_standardized_mse,
            "target_semantics": "standardize_fit_then_destandardize_output",
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_PLAN_DOMAIN)

    def validate_integrity(self) -> None:
        self.feature_codec.validate_integrity()
        _clone_executor_artifact(
            self.executor_artifact,
            strict_canonical=True,
        )
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("reference provider plan hash mismatch")

    def accounting(self) -> StateConditionedReferenceProviderAccounting:
        self.validate_integrity()
        executor = _restore_executor_artifact(self.executor_artifact)
        target_normalization = 2 * self.target_modes
        total = (
            self.feature_codec.stored_scalar_count
            + target_normalization
            + executor.learned_parameter_count
        )
        return StateConditionedReferenceProviderAccounting(
            modal_modes=self.feature_codec.modal_modes,
            null_modes=self.feature_codec.null_modes,
            feature_modes=self.feature_codec.feature_modes,
            target_modes=self.target_modes,
            feature_codec_scalar_count=(
                self.feature_codec.stored_scalar_count
            ),
            target_standardization_scalar_count=target_normalization,
            executor_parameter_count=executor.learned_parameter_count,
            total_stored_scalar_count=total,
        )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        accounting = self.accounting()
        return {
            **self._hash_payload(),
            "executor_config": asdict(self.executor_config),
            "accounting": asdict(accounting),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        raw_state = self.executor_artifact["model_state_dict"]
        assert isinstance(raw_state, Mapping)
        return {
            "artifact_kind": _PLAN_KIND,
            "format_version": _FORMAT_VERSION,
            "feature_codec_state": self.feature_codec.state_dict(),
            "target_center": self.target_center.clone(),
            "target_scale": self.target_scale.clone(),
            "executor_artifact": {
                "artifact_kind": self.executor_artifact["artifact_kind"],
                "format_version": self.executor_artifact["format_version"],
                "config": dict(self.executor_artifact["config"]),
                "dtype": self.executor_artifact["dtype"],
                "model_state_dict": {
                    name: value.clone()
                    for name, value in sorted(raw_state.items())
                },
            },
            "synthetic_binding_sha256": self.synthetic_binding_sha256,
            "fit_batch_sha256s": self.fit_batch_sha256s,
            "fit_batch_content_sha256s": self.fit_batch_content_sha256s,
            "training_steps": self.training_steps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "initial_standardized_mse": self.initial_standardized_mse,
            "final_standardized_mse": self.final_standardized_mse,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> StateConditionedReferenceProviderPlan:
        _strict_keys(
            state,
            expected=_PLAN_STATE_FIELDS,
            label="reference provider plan state",
        )
        if state["artifact_kind"] != _PLAN_KIND:
            raise ValueError("unsupported reference provider plan kind")
        if state["format_version"] != _FORMAT_VERSION:
            raise ValueError("unsupported reference provider format version")
        codec_state = state["feature_codec_state"]
        if not isinstance(codec_state, Mapping):
            raise TypeError("feature_codec_state must be a mapping")
        executor_artifact = state["executor_artifact"]
        if not isinstance(executor_artifact, Mapping):
            raise TypeError("executor_artifact must be a mapping")
        return cls(
            feature_codec=ReferenceProviderFeatureCodec.from_state_dict(
                codec_state
            ),
            target_center=_require_canonical_tensor(
                state["target_center"],
                label="target_center",
                dtype=torch.float64,
                ndim=1,
            ),
            target_scale=_require_canonical_tensor(
                state["target_scale"],
                label="target_scale",
                dtype=torch.float64,
                ndim=1,
            ),
            executor_artifact=_clone_executor_artifact(
                executor_artifact,
                strict_canonical=True,
            ),
            synthetic_binding_sha256=state["synthetic_binding_sha256"],
            fit_batch_sha256s=state["fit_batch_sha256s"],
            fit_batch_content_sha256s=state[
                "fit_batch_content_sha256s"
            ],
            training_steps=state["training_steps"],
            learning_rate=state["learning_rate"],
            seed=state["seed"],
            initial_standardized_mse=state[
                "initial_standardized_mse"
            ],
            final_standardized_mse=state["final_standardized_mse"],
            artifact_sha256=state["artifact_sha256"],
        )

    def prepare(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> PreparedStateConditionedReferenceProvider:
        self.validate_integrity()
        return PreparedStateConditionedReferenceProvider(
            self,
            dtype=dtype,
            device=device,
        )


class PreparedStateConditionedReferenceProvider(nn.Module):
    """Prepared runtime mapping raw state features to target modal coordinates."""

    def __init__(
        self,
        plan: StateConditionedReferenceProviderPlan,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("prepared provider dtype must be floating point")
        plan.validate_integrity()
        target_device = torch.device("cpu" if device is None else device)
        canonical_executor = _restore_executor_artifact(
            plan.executor_artifact
        )
        # Construction happens on CPU under a forked RNG.  Loading then moving
        # avoids consuming caller CPU/GPU randomness for an already-fit plan.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            executor = ResidualGatedCausalModalExecutor(
                canonical_executor.config,
                dtype=torch.float64,
                device="cpu",
            )
        raw_state = plan.executor_artifact["model_state_dict"]
        assert isinstance(raw_state, Mapping)
        executor.load_state_dict(
            {name: value.clone() for name, value in raw_state.items()},
            strict=True,
        )
        executor.to(device=target_device, dtype=dtype)
        executor.eval()
        executor.requires_grad_(False)
        self.executor = executor
        self.feature_codec = plan.feature_codec.prepare(
            dtype=dtype,
            device=target_device,
        )
        self.register_buffer(
            "target_center",
            plan.target_center.to(device=target_device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "target_scale",
            plan.target_scale.to(device=target_device, dtype=dtype),
            persistent=False,
        )
        self.plan_sha256 = plan.artifact_sha256
        self._expected_runtime_sha256 = self._runtime_sha256()

    @property
    def dtype(self) -> torch.dtype:
        return self.target_center.dtype

    @property
    def device(self) -> torch.device:
        return self.target_center.device

    @property
    def modal_modes(self) -> int:
        return self.feature_codec.modal_modes

    @property
    def feature_modes(self) -> int:
        return self.feature_codec.feature_modes

    @property
    def null_modes(self) -> int:
        return self.feature_codec.null_modes

    @property
    def target_modes(self) -> int:
        return int(self.target_center.numel())

    def _runtime_sha256(self) -> str:
        state_hashes = {
            name: _tensor_sha256(value)
            for name, value in sorted(self.executor.state_dict().items())
        }
        return _json_sha256(
            {
                "plan_sha256": self.plan_sha256,
                "dtype": str(self.dtype),
                "device_type": self.device.type,
                "feature_modal_center": _tensor_sha256(
                    self.feature_codec.modal_center
                ),
                "feature_modal_whitener": _tensor_sha256(
                    self.feature_codec.modal_whitener
                ),
                "feature_null_center": _tensor_sha256(
                    self.feature_codec.null_center
                ),
                "feature_null_scale": _tensor_sha256(
                    self.feature_codec.null_scale
                ),
                "target_center": _tensor_sha256(self.target_center),
                "target_scale": _tensor_sha256(self.target_scale),
                "executor_state": state_hashes,
            },
            domain=_EXECUTOR_DOMAIN,
        )

    def validate_integrity(self) -> None:
        if self._runtime_sha256() != self._expected_runtime_sha256:
            raise ValueError("prepared provider runtime integrity mismatch")

    def _validate_positions_and_mask(
        self,
        modal_coordinates: Tensor,
        valid_mask: Tensor,
        logical_positions: Tensor,
    ) -> None:
        shape = modal_coordinates.shape[:2]
        if (
            not isinstance(valid_mask, Tensor)
            or valid_mask.dtype is not torch.bool
            or valid_mask.shape != shape
            or valid_mask.device != self.device
        ):
            raise ValueError(
                "valid_mask must be boolean and match provider rows/device"
            )
        if (
            not isinstance(logical_positions, Tensor)
            or logical_positions.dtype not in (torch.int32, torch.int64)
            or logical_positions.shape != shape
            or logical_positions.device != self.device
        ):
            raise ValueError(
                "logical_positions must be integer and match rows/device"
            )

    def forward(
        self,
        modal_coordinates: Tensor,
        null_coordinates: Tensor,
        row_rms: Tensor,
        *,
        valid_mask: Tensor,
        logical_positions: Tensor,
    ) -> Tensor:
        self._validate_positions_and_mask(
            modal_coordinates,
            valid_mask,
            logical_positions,
        )
        features = self.feature_codec(
            modal_coordinates,
            null_coordinates,
            row_rms,
            valid_mask,
        )
        standardized = self.executor(
            features,
            query_valid_mask=valid_mask,
            key_valid_mask=valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=logical_positions,
        )
        output = standardized * self.target_scale + self.target_center
        return torch.where(
            valid_mask.unsqueeze(-1),
            output,
            torch.zeros_like(output),
        )

    def execution_accounting(
        self,
        *,
        valid_mask: Tensor,
        logical_positions: Tensor,
    ) -> StateConditionedReferenceProviderExecutionAccounting:
        if (
            not isinstance(valid_mask, Tensor)
            or valid_mask.ndim != 2
            or valid_mask.dtype is not torch.bool
            or valid_mask.device != self.device
        ):
            raise ValueError(
                "valid_mask must be a rank-2 boolean Tensor on provider device"
            )
        if (
            not isinstance(logical_positions, Tensor)
            or logical_positions.shape != valid_mask.shape
            or logical_positions.dtype not in (torch.int32, torch.int64)
            or logical_positions.device != self.device
        ):
            raise ValueError(
                "logical_positions must be integer and match valid_mask"
            )
        batch_size, sequence_length = valid_mask.shape
        core = self.executor.execution_accounting(
            int(sequence_length),
            batch_size=int(batch_size),
            query_valid_mask=valid_mask,
            key_valid_mask=valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=logical_positions,
        )
        valid_rows = int(valid_mask.sum().item())
        modal_macs = valid_rows * self.modal_modes**2
        null_macs = valid_rows * self.null_modes
        target_macs = valid_rows * self.target_modes
        return StateConditionedReferenceProviderExecutionAccounting(
            core=core,
            valid_rows=valid_rows,
            modal_whitening_mac_count=modal_macs,
            null_standardization_mac_count=null_macs,
            target_destandardization_mac_count=target_macs,
            total_mac_count=(
                core.total_mac_count
                + modal_macs
                + null_macs
                + target_macs
            ),
        )


def _validated_batch_tuple(
    batches: Sequence[SyntheticReferenceBatch],
    *,
    label: str,
    required_split: SyntheticSplit | None,
    modal_modes: int,
    null_modes: int,
    target_modes: int,
    synthetic_binding_sha256: str | None,
) -> tuple[SyntheticReferenceBatch, ...]:
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(batches)
    if not result:
        raise ValueError(f"{label} must be nonempty")
    seen_content: set[str] = set()
    for index, batch in enumerate(result):
        if not isinstance(batch, SyntheticReferenceBatch):
            raise TypeError(
                f"{label}[{index}] must be a SyntheticReferenceBatch"
            )
        batch.validate_integrity()
        if required_split is not None and batch.split != required_split:
            raise ValueError(
                f"{label}[{index}] must use split {required_split!r}"
            )
        if batch.modal_modes != modal_modes:
            raise ValueError(f"{label}[{index}] modal width is incompatible")
        if batch.null_modes != null_modes:
            raise ValueError(f"{label}[{index}] null width is incompatible")
        if batch.target_mode_count != target_modes:
            raise ValueError(f"{label}[{index}] target width is incompatible")
        if (
            synthetic_binding_sha256 is not None
            and batch.synthetic_binding_sha256
            != synthetic_binding_sha256
        ):
            raise ValueError(
                f"{label}[{index}] synthetic binding is incompatible"
            )
        if batch.content_sha256 in seen_content:
            raise ValueError(f"{label} contains duplicate batch content")
        seen_content.add(batch.content_sha256)
    return result


def _validate_executor_config(
    config: GatedCausalModalExecutorConfig,
    *,
    feature_modes: int,
    target_modes: int,
) -> None:
    if not isinstance(config, GatedCausalModalExecutorConfig):
        raise TypeError(
            "executor_config must be a GatedCausalModalExecutorConfig"
        )
    if config.input_modes != feature_modes:
        raise ValueError(
            "executor_config input_modes must match encoded feature width"
        )
    if config.output_modes != target_modes:
        raise ValueError(
            "executor_config output_modes must match target width"
        )
    if config.same_position_skip:
        raise ValueError(
            "executor_config must disable same_position_skip for provider use"
        )


def _full_batch_standardized_mse(
    executor: ResidualGatedCausalModalExecutor,
    encoded_batches: Sequence[tuple[Tensor, Tensor, Tensor, Tensor]],
) -> Tensor:
    squared_error = executor.same_position_weight.new_zeros(())
    scalar_count = 0
    for features, targets, positions, mask in encoded_batches:
        predicted = executor(
            features,
            query_valid_mask=mask,
            key_valid_mask=mask,
            logical_positions=positions,
            key_logical_positions=positions,
        )
        error = predicted - targets
        squared_error = squared_error + (
            error.square() * mask.unsqueeze(-1)
        ).sum()
        scalar_count += int(mask.sum().item()) * int(targets.shape[-1])
    if scalar_count <= 0:
        raise ValueError("fit batches contain no valid target scalars")
    return squared_error / scalar_count


def fit_state_conditioned_reference_provider(
    *,
    feature_codec: ReferenceProviderFeatureCodec,
    target_center: Tensor,
    target_scale: Tensor,
    fit_batches: Sequence[SyntheticReferenceBatch],
    executor_config: GatedCausalModalExecutorConfig,
    steps: int,
    learning_rate: float,
    seed: int,
) -> StateConditionedReferenceProviderPlan:
    """Fit one provider with a deterministic, fixed-step, full-batch schedule.

    Only batches explicitly labeled ``fit`` are accepted.  The routine has no
    validation callback, no early stopping, and no candidate-selection input.
    This keeps held-out selection data outside the fitting API.
    """

    if not isinstance(feature_codec, ReferenceProviderFeatureCodec):
        raise TypeError(
            "feature_codec must be a ReferenceProviderFeatureCodec"
        )
    feature_codec.validate_integrity()
    canonical_target_center = _canonical_float_tensor(
        target_center,
        label="target_center",
        ndim=1,
    )
    canonical_target_scale = _canonical_float_tensor(
        target_scale,
        label="target_scale",
        ndim=1,
    )
    if canonical_target_scale.shape != canonical_target_center.shape:
        raise ValueError("target_scale must match target_center")
    if bool((canonical_target_scale <= 0).any()):
        raise ValueError("target_scale must be strictly positive")
    target_modes = int(canonical_target_center.numel())
    _validate_executor_config(
        executor_config,
        feature_modes=feature_codec.feature_modes,
        target_modes=target_modes,
    )
    training_steps = _positive_integer(steps, label="steps")
    rate = _positive_float(learning_rate, label="learning_rate")
    fit_seed = _nonnegative_integer(seed, label="seed")

    if isinstance(fit_batches, (str, bytes)) or not isinstance(
        fit_batches,
        Sequence,
    ):
        raise TypeError("fit_batches must be a sequence")
    raw_fit_batches = tuple(fit_batches)
    if not raw_fit_batches:
        raise ValueError("fit_batches must be nonempty")
    first = raw_fit_batches[0]
    if not isinstance(first, SyntheticReferenceBatch):
        raise TypeError("fit_batches entries must be SyntheticReferenceBatch")
    fit = _validated_batch_tuple(
        raw_fit_batches,
        label="fit_batches",
        required_split="fit",
        modal_modes=feature_codec.modal_modes,
        null_modes=feature_codec.null_modes,
        target_modes=target_modes,
        synthetic_binding_sha256=first.synthetic_binding_sha256,
    )
    prepared_codec = feature_codec.prepare(
        dtype=torch.float64,
        device="cpu",
    )
    encoded_batches: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for batch in fit:
        features = prepared_codec(
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
        )
        standardized_targets = (
            batch.target_modes - canonical_target_center
        ) / canonical_target_scale
        standardized_targets = torch.where(
            batch.valid_mask.unsqueeze(-1),
            standardized_targets,
            torch.zeros_like(standardized_targets),
        )
        encoded_batches.append(
            (
                features,
                standardized_targets,
                batch.logical_positions,
                batch.valid_mask,
            )
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fit_seed)
        executor = ResidualGatedCausalModalExecutor(
            executor_config,
            dtype=torch.float64,
            device="cpu",
        )
    optimizer = torch.optim.Adam(executor.parameters(), lr=rate)
    executor.train()
    with torch.no_grad():
        initial_mse = float(
            _full_batch_standardized_mse(executor, encoded_batches).item()
        )
    for _ in range(training_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _full_batch_standardized_mse(executor, encoded_batches)
        if not bool(torch.isfinite(loss)):
            raise ValueError("provider fit produced a nonfinite loss")
        loss.backward()
        optimizer.step()
    executor.eval()
    executor.requires_grad_(False)
    with torch.no_grad():
        final_mse = float(
            _full_batch_standardized_mse(executor, encoded_batches).item()
        )
    if not math.isfinite(final_mse):
        raise ValueError("provider fit produced a nonfinite final loss")

    return StateConditionedReferenceProviderPlan(
        feature_codec=feature_codec,
        target_center=canonical_target_center,
        target_scale=canonical_target_scale,
        executor_artifact=executor.artifact_state_dict(),
        synthetic_binding_sha256=fit[0].synthetic_binding_sha256,
        fit_batch_sha256s=tuple(
            batch.artifact_sha256 for batch in fit
        ),
        fit_batch_content_sha256s=tuple(
            batch.content_sha256 for batch in fit
        ),
        training_steps=training_steps,
        learning_rate=rate,
        seed=fit_seed,
        initial_standardized_mse=initial_mse,
        final_standardized_mse=final_mse,
    )


@dataclass(frozen=True, slots=True)
class StateConditionedReferenceProviderBatchMetric:
    """One held-out batch's standardized-space distortion."""

    batch_sha256: str
    content_sha256: str
    valid_scalar_count: int
    standardized_mse: float
    standardized_relative_error: float
    standardized_cosine: float


@dataclass(frozen=True, slots=True)
class StateConditionedReferenceProviderEvaluation:
    """Held-out distortion plus exact structural runtime checks."""

    plan_sha256: str
    split: SyntheticSplit
    batch_metrics: tuple[StateConditionedReferenceProviderBatchMetric, ...]
    pooled_standardized_mse: float
    pooled_standardized_relative_error: float
    pooled_standardized_cosine: float
    worst_batch_standardized_relative_error: float
    worst_batch_standardized_cosine: float
    repeat_exact: bool
    causal_prefix_exact: bool
    padding_exact: bool
    invalid_outputs_zero: bool
    integrity_verified: bool
    evaluation_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.plan_sha256, label="plan_sha256")
        if self.split not in ("fit", "selection", "assessment"):
            raise ValueError("evaluation split is invalid")
        if not self.batch_metrics:
            raise ValueError("batch_metrics must be nonempty")
        for label in (
            "pooled_standardized_mse",
            "pooled_standardized_relative_error",
            "pooled_standardized_cosine",
            "worst_batch_standardized_relative_error",
            "worst_batch_standardized_cosine",
        ):
            _finite_float(getattr(self, label), label=label)
        for label in (
            "repeat_exact",
            "causal_prefix_exact",
            "padding_exact",
            "invalid_outputs_zero",
            "integrity_verified",
        ):
            if type(getattr(self, label)) is not bool:
                raise TypeError(f"{label} must be boolean")
        expected = self._computed_sha256()
        if self.evaluation_sha256:
            supplied = _require_sha256(
                self.evaluation_sha256,
                label="evaluation_sha256",
            )
            if supplied != expected:
                raise ValueError("provider evaluation hash mismatch")
        object.__setattr__(self, "evaluation_sha256", expected)

    @property
    def structural_checks_passed(self) -> bool:
        return (
            self.repeat_exact
            and self.causal_prefix_exact
            and self.padding_exact
            and self.invalid_outputs_zero
            and self.integrity_verified
        )

    def _computed_sha256(self) -> str:
        return _json_sha256(
            {
                "plan_sha256": self.plan_sha256,
                "split": self.split,
                "batch_metrics": [
                    asdict(metric) for metric in self.batch_metrics
                ],
                "pooled_standardized_mse": (
                    self.pooled_standardized_mse
                ),
                "pooled_standardized_relative_error": (
                    self.pooled_standardized_relative_error
                ),
                "pooled_standardized_cosine": (
                    self.pooled_standardized_cosine
                ),
                "worst_batch_standardized_relative_error": (
                    self.worst_batch_standardized_relative_error
                ),
                "worst_batch_standardized_cosine": (
                    self.worst_batch_standardized_cosine
                ),
                "repeat_exact": self.repeat_exact,
                "causal_prefix_exact": self.causal_prefix_exact,
                "padding_exact": self.padding_exact,
                "invalid_outputs_zero": self.invalid_outputs_zero,
                "integrity_verified": self.integrity_verified,
            },
            domain=_EVALUATION_DOMAIN,
        )


def _distortion(
    predicted: Tensor,
    target: Tensor,
    *,
    label: str,
) -> tuple[float, float, float, float, float]:
    predicted_flat = (
        predicted.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    )
    target_flat = (
        target.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    )
    if predicted_flat.numel() == 0 or predicted_flat.shape != target_flat.shape:
        raise ValueError(f"{label} must contain aligned nonempty values")
    if (
        not bool(torch.isfinite(predicted_flat).all())
        or not bool(torch.isfinite(target_flat).all())
    ):
        raise ValueError(f"{label} must be finite")
    error_squared = float((predicted_flat - target_flat).square().sum())
    target_squared = float(target_flat.square().sum())
    if target_squared <= torch.finfo(torch.float64).eps:
        raise ValueError(
            f"{label} standardized target energy is too small for "
            "relative-error validation"
        )
    predicted_squared = float(predicted_flat.square().sum())
    mse = error_squared / int(predicted_flat.numel())
    relative_error = math.sqrt(error_squared / target_squared)
    if predicted_squared <= torch.finfo(torch.float64).eps:
        cosine = 0.0
    else:
        cosine = float(torch.dot(predicted_flat, target_flat)) / math.sqrt(
            predicted_squared * target_squared
        )
        cosine = max(-1.0, min(1.0, cosine))
    return mse, relative_error, cosine, error_squared, target_squared


def _batch_on_runtime(
    batch: SyntheticReferenceBatch,
    runtime: PreparedStateConditionedReferenceProvider,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        batch.modal_coordinates.to(
            device=runtime.device,
            dtype=runtime.dtype,
        ),
        batch.null_coordinates.to(
            device=runtime.device,
            dtype=runtime.dtype,
        ),
        batch.row_rms.to(device=runtime.device, dtype=runtime.dtype),
        batch.target_modes.to(device=runtime.device, dtype=runtime.dtype),
        batch.logical_positions.to(device=runtime.device),
        batch.valid_mask.to(device=runtime.device),
    )


def evaluate_state_conditioned_reference_provider(
    provider: (
        StateConditionedReferenceProviderPlan
        | PreparedStateConditionedReferenceProvider
    ),
    batches: Sequence[SyntheticReferenceBatch],
    *,
    required_split: SyntheticSplit | None = None,
) -> StateConditionedReferenceProviderEvaluation:
    """Evaluate distortion and exact causality/padding/repeat invariants."""

    if isinstance(provider, StateConditionedReferenceProviderPlan):
        provider.validate_integrity()
        runtime = provider.prepare(dtype=torch.float64, device="cpu")
        fit_contents = set(provider.fit_batch_content_sha256s)
        binding = provider.synthetic_binding_sha256
    elif isinstance(provider, PreparedStateConditionedReferenceProvider):
        runtime = provider
        runtime.validate_integrity()
        fit_contents = set()
        binding = None
    else:
        raise TypeError("provider must be a plan or prepared provider")

    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise TypeError("batches must be a sequence")
    raw_batches = tuple(batches)
    if not raw_batches:
        raise ValueError("batches must be nonempty")
    first = raw_batches[0]
    if not isinstance(first, SyntheticReferenceBatch):
        raise TypeError("batches entries must be SyntheticReferenceBatch")
    expected_split = first.split if required_split is None else required_split
    checked = _validated_batch_tuple(
        raw_batches,
        label="evaluation batches",
        required_split=expected_split,
        modal_modes=runtime.modal_modes,
        null_modes=runtime.null_modes,
        target_modes=runtime.target_modes,
        synthetic_binding_sha256=binding,
    )
    if any(batch.split != expected_split for batch in checked):
        raise ValueError("evaluation batches must share one split")
    if expected_split != "fit":
        overlap = fit_contents.intersection(
            batch.content_sha256 for batch in checked
        )
        if overlap:
            raise ValueError(
                "held-out evaluation content overlaps provider fit content"
            )

    metrics: list[StateConditionedReferenceProviderBatchMetric] = []
    pooled_predicted: list[Tensor] = []
    pooled_target: list[Tensor] = []
    repeat_exact = True
    causal_exact = True
    padding_exact = True
    invalid_zero = True
    runtime.validate_integrity()

    with torch.no_grad():
        for batch in checked:
            (
                modal,
                null_coordinates,
                row_rms,
                target,
                positions,
                mask,
            ) = _batch_on_runtime(batch, runtime)
            predicted = runtime(
                modal,
                null_coordinates,
                row_rms,
                valid_mask=mask,
                logical_positions=positions,
            )
            repeated = runtime(
                modal,
                null_coordinates,
                row_rms,
                valid_mask=mask,
                logical_positions=positions,
            )
            repeat_exact = repeat_exact and torch.equal(predicted, repeated)
            invalid_zero = invalid_zero and torch.equal(
                predicted[~mask],
                torch.zeros_like(predicted[~mask]),
            )

            standardized_predicted = (
                predicted - runtime.target_center
            ) / runtime.target_scale
            standardized_target = (
                target - runtime.target_center
            ) / runtime.target_scale
            selected_predicted = standardized_predicted[mask]
            selected_target = standardized_target[mask]
            (
                mse,
                relative_error,
                cosine,
                _,
                _,
            ) = _distortion(
                selected_predicted,
                selected_target,
                label=f"batch {batch.artifact_sha256}",
            )
            pooled_predicted.append(selected_predicted.detach().cpu())
            pooled_target.append(selected_target.detach().cpu())
            metrics.append(
                StateConditionedReferenceProviderBatchMetric(
                    batch_sha256=batch.artifact_sha256,
                    content_sha256=batch.content_sha256,
                    valid_scalar_count=int(selected_target.numel()),
                    standardized_mse=mse,
                    standardized_relative_error=relative_error,
                    standardized_cosine=cosine,
                )
            )

            if batch.sequence_length > 1:
                cut = max(1, batch.sequence_length // 2)
                changed_modal = modal.clone()
                changed_null = null_coordinates.clone()
                changed_rms = row_rms.clone()
                changed_modal[:, cut:] = (
                    -1.25 * changed_modal[:, cut:] + 7.0
                )
                changed_rms[:, cut:] = 1.5 * changed_rms[:, cut:] + 0.75
                changed_null[:, cut:] = (
                    1.75 * changed_null[:, cut:] - 5.0
                )
                changed_output = runtime(
                    changed_modal,
                    changed_null,
                    changed_rms,
                    valid_mask=mask,
                    logical_positions=positions,
                )
                causal_exact = causal_exact and torch.equal(
                    predicted[:, :cut],
                    changed_output[:, :cut],
                )
                invalid_zero = invalid_zero and torch.equal(
                    changed_output[~mask],
                    torch.zeros_like(changed_output[~mask]),
                )

            changed_padding_modal = torch.where(
                mask.unsqueeze(-1),
                modal,
                torch.full_like(modal, 9_973.0),
            )
            changed_padding_null = torch.where(
                mask.unsqueeze(-1),
                null_coordinates,
                torch.full_like(null_coordinates, -4_113.0),
            )
            changed_padding_rms = torch.where(
                mask,
                row_rms,
                torch.full_like(row_rms, 31.0),
            )
            changed_padding_positions = torch.where(
                mask,
                positions,
                torch.full_like(positions, -777),
            )
            changed_padding_output = runtime(
                changed_padding_modal,
                changed_padding_null,
                changed_padding_rms,
                valid_mask=mask,
                logical_positions=changed_padding_positions,
            )
            padding_exact = padding_exact and torch.equal(
                predicted[mask],
                changed_padding_output[mask],
            )
            invalid_zero = invalid_zero and torch.equal(
                changed_padding_output[~mask],
                torch.zeros_like(changed_padding_output[~mask]),
            )

    pooled_prediction = torch.cat(pooled_predicted, dim=0)
    pooled_truth = torch.cat(pooled_target, dim=0)
    pooled_mse, pooled_relative, pooled_cosine, _, _ = _distortion(
        pooled_prediction,
        pooled_truth,
        label="pooled evaluation",
    )
    runtime.validate_integrity()
    return StateConditionedReferenceProviderEvaluation(
        plan_sha256=runtime.plan_sha256,
        split=expected_split,
        batch_metrics=tuple(metrics),
        pooled_standardized_mse=pooled_mse,
        pooled_standardized_relative_error=pooled_relative,
        pooled_standardized_cosine=pooled_cosine,
        worst_batch_standardized_relative_error=max(
            metric.standardized_relative_error for metric in metrics
        ),
        worst_batch_standardized_cosine=min(
            metric.standardized_cosine for metric in metrics
        ),
        repeat_exact=repeat_exact,
        causal_prefix_exact=causal_exact,
        padding_exact=padding_exact,
        invalid_outputs_zero=invalid_zero,
        integrity_verified=True,
    )


@dataclass(frozen=True, slots=True)
class StateConditionedReferenceProviderSelectionGates:
    """Optional held-out gates; structural exactness is enabled by default."""

    max_pooled_standardized_relative_error: float | None = None
    min_pooled_standardized_cosine: float | None = None
    max_worst_batch_standardized_relative_error: float | None = None
    min_worst_batch_standardized_cosine: float | None = None
    require_structural_exactness: bool = True

    def __post_init__(self) -> None:
        for label in (
            "max_pooled_standardized_relative_error",
            "max_worst_batch_standardized_relative_error",
        ):
            value = getattr(self, label)
            if value is not None:
                canonical = _finite_float(value, label=label)
                if canonical < 0.0:
                    raise ValueError(f"{label} must be nonnegative")
                object.__setattr__(self, label, canonical)
        for label in (
            "min_pooled_standardized_cosine",
            "min_worst_batch_standardized_cosine",
        ):
            value = getattr(self, label)
            if value is not None:
                canonical = _finite_float(value, label=label)
                if canonical < -1.0 or canonical > 1.0:
                    raise ValueError(f"{label} must be in [-1, 1]")
                object.__setattr__(self, label, canonical)
        if type(self.require_structural_exactness) is not bool:
            raise TypeError("require_structural_exactness must be boolean")

    def accepts(
        self,
        evaluation: StateConditionedReferenceProviderEvaluation,
    ) -> bool:
        if (
            self.require_structural_exactness
            and not evaluation.structural_checks_passed
        ):
            return False
        if (
            self.max_pooled_standardized_relative_error is not None
            and evaluation.pooled_standardized_relative_error
            > self.max_pooled_standardized_relative_error
        ):
            return False
        if (
            self.min_pooled_standardized_cosine is not None
            and evaluation.pooled_standardized_cosine
            < self.min_pooled_standardized_cosine
        ):
            return False
        if (
            self.max_worst_batch_standardized_relative_error is not None
            and evaluation.worst_batch_standardized_relative_error
            > self.max_worst_batch_standardized_relative_error
        ):
            return False
        if (
            self.min_worst_batch_standardized_cosine is not None
            and evaluation.worst_batch_standardized_cosine
            < self.min_worst_batch_standardized_cosine
        ):
            return False
        return True


@dataclass(frozen=True, slots=True)
class StateConditionedReferenceProviderCandidate:
    """One fully fit ladder candidate and its held-out evaluation."""

    plan: StateConditionedReferenceProviderPlan
    evaluation: StateConditionedReferenceProviderEvaluation
    passes_selection_gates: bool

    def __post_init__(self) -> None:
        if self.evaluation.plan_sha256 != self.plan.artifact_sha256:
            raise ValueError("candidate evaluation does not bind its plan")
        if type(self.passes_selection_gates) is not bool:
            raise TypeError("passes_selection_gates must be boolean")

    @property
    def learned_parameter_count(self) -> int:
        return self.plan.accounting().executor_parameter_count

    @property
    def total_stored_scalar_count(self) -> int:
        return self.plan.accounting().total_stored_scalar_count


@dataclass(frozen=True, slots=True)
class StateConditionedReferenceProviderRatePoint:
    """Compact rate/distortion row for one ladder candidate."""

    plan_sha256: str
    executor_config: GatedCausalModalExecutorConfig
    learned_parameter_count: int
    total_stored_scalar_count: int
    pooled_standardized_relative_error: float
    pooled_standardized_cosine: float
    worst_batch_standardized_relative_error: float
    worst_batch_standardized_cosine: float
    structural_checks_passed: bool
    passes_selection_gates: bool


@dataclass(frozen=True, slots=True)
class StateConditionedReferenceProviderCompilation:
    """All fitted candidates and the smallest held-out-gate survivor."""

    candidates: tuple[StateConditionedReferenceProviderCandidate, ...]
    selected_index: int | None
    fit_batch_sha256s: tuple[str, ...]
    selection_batch_sha256s: tuple[str, ...]
    selection_batch_content_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("compilation candidates must be nonempty")
        if self.selected_index is not None:
            if (
                type(self.selected_index) is not int
                or self.selected_index < 0
                or self.selected_index >= len(self.candidates)
            ):
                raise ValueError("selected_index is out of range")
            if not self.candidates[
                self.selected_index
            ].passes_selection_gates:
                raise ValueError("selected candidate must pass selection gates")
        _sha256_tuple(
            self.fit_batch_sha256s,
            label="fit_batch_sha256s",
            nonempty=True,
        )
        _sha256_tuple(
            self.selection_batch_sha256s,
            label="selection_batch_sha256s",
            nonempty=True,
        )
        _sha256_tuple(
            self.selection_batch_content_sha256s,
            label="selection_batch_content_sha256s",
            nonempty=True,
        )

    @property
    def selected_plan(
        self,
    ) -> StateConditionedReferenceProviderPlan | None:
        if self.selected_index is None:
            return None
        return self.candidates[self.selected_index].plan

    @property
    def rate_curve(
        self,
    ) -> tuple[StateConditionedReferenceProviderRatePoint, ...]:
        return tuple(
            StateConditionedReferenceProviderRatePoint(
                plan_sha256=candidate.plan.artifact_sha256,
                executor_config=candidate.plan.executor_config,
                learned_parameter_count=candidate.learned_parameter_count,
                total_stored_scalar_count=(
                    candidate.total_stored_scalar_count
                ),
                pooled_standardized_relative_error=(
                    candidate.evaluation
                    .pooled_standardized_relative_error
                ),
                pooled_standardized_cosine=(
                    candidate.evaluation.pooled_standardized_cosine
                ),
                worst_batch_standardized_relative_error=(
                    candidate.evaluation
                    .worst_batch_standardized_relative_error
                ),
                worst_batch_standardized_cosine=(
                    candidate.evaluation.worst_batch_standardized_cosine
                ),
                structural_checks_passed=(
                    candidate.evaluation.structural_checks_passed
                ),
                passes_selection_gates=(
                    candidate.passes_selection_gates
                ),
            )
            for candidate in self.candidates
        )


def compile_state_conditioned_reference_provider_ladder(
    *,
    feature_codec: ReferenceProviderFeatureCodec,
    target_center: Tensor,
    target_scale: Tensor,
    fit_batches: Sequence[SyntheticReferenceBatch],
    selection_batch_factory: Callable[
        [], Sequence[SyntheticReferenceBatch]
    ],
    executor_configs: Sequence[GatedCausalModalExecutorConfig],
    steps: int,
    learning_rate: float,
    base_seed: int,
    selection_gates: (
        StateConditionedReferenceProviderSelectionGates | None
    ) = None,
) -> StateConditionedReferenceProviderCompilation:
    """Fit the whole ladder before materializing held-out selection batches.

    The factory boundary makes the selection firewall executable: every
    candidate plan is finished before the callback can reveal held-out data.
    The winner is the smallest learned executor passing the declared gates.
    """

    if not callable(selection_batch_factory):
        raise TypeError("selection_batch_factory must be callable")
    if isinstance(executor_configs, (str, bytes)) or not isinstance(
        executor_configs,
        Sequence,
    ):
        raise TypeError("executor_configs must be a sequence")
    configs = tuple(executor_configs)
    if not configs:
        raise ValueError("executor_configs must be nonempty")
    canonical_target_center = _canonical_float_tensor(
        target_center,
        label="target_center",
        ndim=1,
    )
    canonical_target_scale = _canonical_float_tensor(
        target_scale,
        label="target_scale",
        ndim=1,
    )
    if canonical_target_center.shape != canonical_target_scale.shape:
        raise ValueError("target_scale must match target_center")
    if bool((canonical_target_scale <= 0).any()):
        raise ValueError("target_scale must be strictly positive")
    config_keys: set[bytes] = set()
    for config in configs:
        _validate_executor_config(
            config,
            feature_modes=feature_codec.feature_modes,
            target_modes=int(canonical_target_center.numel()),
        )
        key = _canonical_json_bytes(asdict(config))
        if key in config_keys:
            raise ValueError("executor_configs must be unique")
        config_keys.add(key)
    gates = (
        StateConditionedReferenceProviderSelectionGates()
        if selection_gates is None
        else selection_gates
    )
    if not isinstance(
        gates,
        StateConditionedReferenceProviderSelectionGates,
    ):
        raise TypeError(
            "selection_gates must be provider selection gates"
        )
    seed = _nonnegative_integer(base_seed, label="base_seed")

    # Deliberately complete every fit before invoking the held-out factory.
    plans = tuple(
        fit_state_conditioned_reference_provider(
            feature_codec=feature_codec,
            target_center=canonical_target_center,
            target_scale=canonical_target_scale,
            fit_batches=fit_batches,
            executor_config=config,
            steps=steps,
            learning_rate=learning_rate,
            seed=seed + index,
        )
        for index, config in enumerate(configs)
    )
    selection_raw = selection_batch_factory()
    if isinstance(selection_raw, (str, bytes)) or not isinstance(
        selection_raw,
        Sequence,
    ):
        raise TypeError("selection_batch_factory must return a sequence")
    selection = _validated_batch_tuple(
        tuple(selection_raw),
        label="selection batches",
        required_split="selection",
        modal_modes=feature_codec.modal_modes,
        null_modes=feature_codec.null_modes,
        target_modes=int(canonical_target_center.numel()),
        synthetic_binding_sha256=plans[0].synthetic_binding_sha256,
    )
    fit_contents = set(plans[0].fit_batch_content_sha256s)
    if fit_contents.intersection(
        batch.content_sha256 for batch in selection
    ):
        raise ValueError("selection content overlaps fit content")

    candidates = tuple(
        StateConditionedReferenceProviderCandidate(
            plan=plan,
            evaluation=(
                evaluation := evaluate_state_conditioned_reference_provider(
                    plan,
                    selection,
                    required_split="selection",
                )
            ),
            passes_selection_gates=gates.accepts(evaluation),
        )
        for plan in plans
    )
    passing_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.passes_selection_gates
    ]
    selected_index = (
        min(
            passing_indices,
            key=lambda index: (
                candidates[index].learned_parameter_count,
                candidates[index].total_stored_scalar_count,
                _canonical_json_bytes(
                    asdict(candidates[index].plan.executor_config)
                ),
                candidates[index].plan.artifact_sha256,
            ),
        )
        if passing_indices
        else None
    )
    return StateConditionedReferenceProviderCompilation(
        candidates=candidates,
        selected_index=selected_index,
        fit_batch_sha256s=plans[0].fit_batch_sha256s,
        selection_batch_sha256s=tuple(
            batch.artifact_sha256 for batch in selection
        ),
        selection_batch_content_sha256s=tuple(
            batch.content_sha256 for batch in selection
        ),
    )


__all__ = [
    "PreparedReferenceProviderFeatureCodec",
    "PreparedStateConditionedReferenceProvider",
    "ReferenceProviderFeatureCodec",
    "StateConditionedReferenceProviderAccounting",
    "StateConditionedReferenceProviderBatchMetric",
    "StateConditionedReferenceProviderCandidate",
    "StateConditionedReferenceProviderCompilation",
    "StateConditionedReferenceProviderEvaluation",
    "StateConditionedReferenceProviderExecutionAccounting",
    "StateConditionedReferenceProviderPlan",
    "StateConditionedReferenceProviderRatePoint",
    "StateConditionedReferenceProviderSelectionGates",
    "SyntheticReferenceBatch",
    "SyntheticSplit",
    "compile_state_conditioned_reference_provider_ladder",
    "evaluate_state_conditioned_reference_provider",
    "fit_state_conditioned_reference_provider",
]
