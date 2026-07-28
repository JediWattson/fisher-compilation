"""Full-width selection for state-conditioned reference providers.

The provider itself may predict only a retained prefix of a frozen modal
gauge.  Selection must not score that prefix in isolation: doing so makes all
discarded target energy invisible.  This module therefore reconstructs every
candidate in one authenticated, standardized 64-mode gauge, filling omitted
modes with the target center measured on the fit split.

The module is runner independent.  It consumes aligned tensors and opaque
SHA-256 bindings; it does not import a model, tokenizer, prompt fixture, or
provider implementation.  Constant and normalized-position controls are fit
only from ``fit`` probes and are then frozen before ``selection`` probes are
opened.  Selection can defer its collision-identifiability gate to a sealed
assessment.  The assessment-only API scores exactly one already-frozen
candidate on the complete assessment panel, including collision-tagged probes
in both fidelity and target-identifiability metrics, without relabeling rows or
performing selection.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor

from fisher_graph.gemma3_l3_l4_synthetic_reference_protocol import (
    SyntheticReferenceGates,
)


__all__ = [
    "FULL_REFERENCE_WIDTH",
    "NORMALIZED_POSITION_BIN_SEMANTICS",
    "FullWidthCandidatePrediction",
    "FullWidthCandidateScore",
    "FullWidthCollisionMetric",
    "FullWidthFamilyMetric",
    "FullWidthGateFlags",
    "FullWidthProbeMetric",
    "FullWidthReferenceCandidate",
    "FullWidthReferenceControls",
    "FullWidthReferenceProbe",
    "FullWidthReferenceSelection",
    "FullWidthStructuralMetrics",
    "fit_full_width_reference_controls",
    "full_width_reference_gates_sha256",
    "reconstruct_full_width_prediction",
    "score_full_width_reference_assessment",
    "score_full_width_reference_candidate",
    "select_smallest_passing_full_width_reference_candidate",
]


ReferenceSplit = Literal["fit", "selection", "assessment"]

FULL_REFERENCE_WIDTH = 64
RELATIVE_ERROR_EPSILON = 1e-12
NORMALIZED_POSITION_BIN_SEMANTICS = (
    "per_sequence_valid_token_ordinal_divided_by_"
    "max(valid_token_count_minus_one,1);"
    "bin=min(floor(normalized_position*bin_count),bin_count_minus_one);"
    "empty_fit_bins_use_fit_target_center"
)
SELECTION_RULE = (
    "minimum_stored_scalars_then_source_rank_then_target_rank_then_candidate_id"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORMAT_VERSION = 1
_PROBE_KIND = "fisher_graph.full_width_reference_probe"
_CONTROLS_KIND = "fisher_graph.full_width_reference_controls"
_PREDICTION_KIND = "fisher_graph.full_width_candidate_prediction"
_CANDIDATE_KIND = "fisher_graph.full_width_reference_candidate"
_SCORE_KIND = "fisher_graph.full_width_candidate_score"
_SELECTION_KIND = "fisher_graph.full_width_reference_selection"
_TENSOR_DOMAIN = b"fisher_graph.full_width_reference.tensor.v1\0"
_PROBE_DOMAIN = b"fisher_graph.full_width_reference.probe.v1\0"
_CONTROLS_DOMAIN = b"fisher_graph.full_width_reference.controls.v1\0"
_PREDICTION_DOMAIN = b"fisher_graph.full_width_reference.prediction.v1\0"
_CANDIDATE_DOMAIN = b"fisher_graph.full_width_reference.candidate.v1\0"
_SCORE_DOMAIN = b"fisher_graph.full_width_reference.score.v1\0"
_SELECTION_DOMAIN = b"fisher_graph.full_width_reference.selection.v1\0"
_GATES_DOMAIN = b"fisher_graph.full_width_reference.gates.v1\0"


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


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


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


def _nonnegative_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _state_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{label} keys must be strings")
    return value


def _strict_state_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} fields do not match frozen format; "
            f"missing={missing}, unknown={unknown}"
        )


def _state_list(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a list")
    return value


def _state_string(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return _require_nonempty_string(value, label=label)


def _state_sha256(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return _require_sha256(value, label=label)


def _state_float(value: object, *, label: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{label} must be a float")
    return _finite_float(value, label=label)


def _state_nonnegative_float(value: object, *, label: str) -> float:
    result = _state_float(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _state_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    return value


def _validate_state_header(
    state: Mapping[str, object],
    *,
    artifact_kind: str,
    label: str,
) -> None:
    kind = state["artifact_kind"]
    version = state["format_version"]
    if type(kind) is not str or kind != artifact_kind:
        raise ValueError(f"{label} artifact kind is invalid")
    if type(version) is not int or version != _FORMAT_VERSION:
        raise ValueError(f"{label} format version is invalid")


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


def _digest_or_set(
    instance: object,
    *,
    field_name: str,
    payload: object,
    domain: bytes,
    label: str,
) -> None:
    computed = _json_sha256(payload, domain=domain)
    supplied = getattr(instance, field_name)
    if supplied:
        if supplied != computed:
            raise ValueError(f"{label} hash mismatch")
    else:
        object.__setattr__(instance, field_name, computed)


def _validate_probe_collection(
    probes: Sequence["FullWidthReferenceProbe"],
    *,
    split: ReferenceSplit,
    label: str,
    nonempty: bool,
) -> tuple["FullWidthReferenceProbe", ...]:
    if isinstance(probes, (str, bytes)) or not isinstance(probes, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(probes)
    if nonempty and not result:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(probe, FullWidthReferenceProbe) for probe in result):
        raise TypeError(f"{label} must contain FullWidthReferenceProbe values")
    if any(probe.split != split for probe in result):
        raise ValueError(f"{label} must contain only {split!r} probes")
    ids = tuple(probe.probe_id for probe in result)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate probe ids")
    return result


@dataclass(frozen=True, slots=True)
class FullWidthReferenceProbe:
    """One full-target probe in an authenticated standardized gauge."""

    probe_id: str
    split: ReferenceSplit
    family: str
    standardized_target: Tensor
    logical_positions: Tensor
    valid_mask: Tensor
    standardized_gauge_sha256: str
    collision_group: str | None = None
    collision_variant: str | None = None
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_string(self.probe_id, label="probe id")
        if self.split not in ("fit", "selection", "assessment"):
            raise ValueError("probe split is invalid")
        _require_nonempty_string(self.family, label="probe family")
        target = _canonical_float_tensor(
            self.standardized_target,
            label="standardized target",
            ndim=3,
        )
        positions = _canonical_positions(
            self.logical_positions,
            label="logical positions",
        )
        mask = _canonical_mask(self.valid_mask, label="valid mask")
        if target.shape[-1] != FULL_REFERENCE_WIDTH:
            raise ValueError(
                f"standardized target width must be {FULL_REFERENCE_WIDTH}"
            )
        if target.shape[:2] != positions.shape or positions.shape != mask.shape:
            raise ValueError("probe target, positions, and mask are not aligned")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every probe sequence must contain a valid row")
        for batch_index in range(mask.shape[0]):
            valid_positions = positions[batch_index][mask[batch_index]]
            if bool((valid_positions < 0).any()):
                raise ValueError("valid logical positions must be nonnegative")
            if valid_positions.numel() > 1 and not bool(
                (valid_positions[1:] > valid_positions[:-1]).all()
            ):
                raise ValueError(
                    "valid logical positions must be strictly increasing"
                )
        _require_sha256(
            self.standardized_gauge_sha256,
            label="standardized gauge SHA-256",
        )
        if (self.collision_group is None) != (self.collision_variant is None):
            raise ValueError(
                "collision group and collision variant must be supplied together"
            )
        if self.collision_group is not None:
            if self.split != "assessment":
                raise ValueError("collision probes must use the assessment split")
            _require_nonempty_string(
                self.collision_group,
                label="collision group",
            )
            _require_nonempty_string(
                self.collision_variant,
                label="collision variant",
            )
        object.__setattr__(self, "standardized_target", target)
        object.__setattr__(self, "logical_positions", positions)
        object.__setattr__(self, "valid_mask", mask)
        _digest_or_set(
            self,
            field_name="artifact_sha256",
            payload=self._payload(),
            domain=_PROBE_DOMAIN,
            label="full-width reference probe",
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": _PROBE_KIND,
            "format_version": _FORMAT_VERSION,
            "probe_id": self.probe_id,
            "split": self.split,
            "family": self.family,
            "standardized_target_sha256": _tensor_sha256(
                self.standardized_target
            ),
            "logical_positions_sha256": _tensor_sha256(
                self.logical_positions
            ),
            "valid_mask_sha256": _tensor_sha256(self.valid_mask),
            "standardized_gauge_sha256": self.standardized_gauge_sha256,
            "collision_group": self.collision_group,
            "collision_variant": self.collision_variant,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "standardized_target": self.standardized_target.detach().clone(),
            "logical_positions": self.logical_positions.detach().clone(),
            "valid_mask": self.valid_mask.detach().clone(),
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class FullWidthReferenceControls:
    """Fit-only constant and normalized-position controls."""

    fit_target_center: Tensor
    normalized_position_bin_centers: Tensor
    normalized_position_bin_counts: tuple[int, ...]
    fit_probe_ids: tuple[str, ...]
    fit_probe_sha256s: tuple[str, ...]
    standardized_gauge_sha256: str
    position_semantics: str = NORMALIZED_POSITION_BIN_SEMANTICS
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        center = _canonical_float_tensor(
            self.fit_target_center,
            label="fit target center",
            ndim=1,
        )
        bins = _canonical_float_tensor(
            self.normalized_position_bin_centers,
            label="normalized-position bin centers",
            ndim=2,
        )
        if center.shape != (FULL_REFERENCE_WIDTH,):
            raise ValueError(
                f"fit target center width must be {FULL_REFERENCE_WIDTH}"
            )
        if bins.shape[1] != FULL_REFERENCE_WIDTH:
            raise ValueError(
                f"position-bin target width must be {FULL_REFERENCE_WIDTH}"
            )
        if (
            not isinstance(self.normalized_position_bin_counts, tuple)
            or len(self.normalized_position_bin_counts) != bins.shape[0]
        ):
            raise ValueError("position-bin counts do not match bin centers")
        for count in self.normalized_position_bin_counts:
            _nonnegative_integer(count, label="normalized-position bin count")
        if (
            not isinstance(self.fit_probe_ids, tuple)
            or not self.fit_probe_ids
            or any(
                not isinstance(value, str) or not value
                for value in self.fit_probe_ids
            )
            or tuple(sorted(self.fit_probe_ids)) != self.fit_probe_ids
            or len(set(self.fit_probe_ids)) != len(self.fit_probe_ids)
        ):
            raise ValueError("fit probe ids must be a sorted unique tuple")
        if (
            not isinstance(self.fit_probe_sha256s, tuple)
            or len(self.fit_probe_sha256s) != len(self.fit_probe_ids)
            or tuple(sorted(self.fit_probe_sha256s))
            != self.fit_probe_sha256s
            or len(set(self.fit_probe_sha256s))
            != len(self.fit_probe_sha256s)
        ):
            raise ValueError(
                "fit probe SHA-256 values must be a sorted unique tuple"
            )
        for value in self.fit_probe_sha256s:
            _require_sha256(value, label="fit probe SHA-256")
        _require_sha256(
            self.standardized_gauge_sha256,
            label="standardized gauge SHA-256",
        )
        if self.position_semantics != NORMALIZED_POSITION_BIN_SEMANTICS:
            raise ValueError("normalized-position semantics drifted")
        object.__setattr__(self, "fit_target_center", center)
        object.__setattr__(
            self,
            "normalized_position_bin_centers",
            bins,
        )
        _digest_or_set(
            self,
            field_name="artifact_sha256",
            payload=self._payload(),
            domain=_CONTROLS_DOMAIN,
            label="full-width reference controls",
        )

    @property
    def position_bin_count(self) -> int:
        return int(self.normalized_position_bin_centers.shape[0])

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": _CONTROLS_KIND,
            "format_version": _FORMAT_VERSION,
            "fit_target_center_sha256": _tensor_sha256(
                self.fit_target_center
            ),
            "normalized_position_bin_centers_sha256": _tensor_sha256(
                self.normalized_position_bin_centers
            ),
            "normalized_position_bin_counts": list(
                self.normalized_position_bin_counts
            ),
            "fit_probe_ids": list(self.fit_probe_ids),
            "fit_probe_sha256s": list(self.fit_probe_sha256s),
            "standardized_gauge_sha256": self.standardized_gauge_sha256,
            "position_semantics": self.position_semantics,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "fit_target_center": self.fit_target_center.detach().clone(),
            "normalized_position_bin_centers": (
                self.normalized_position_bin_centers.detach().clone()
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    def constant_prediction_for(self, probe: FullWidthReferenceProbe) -> Tensor:
        self._validate_probe_gauge(probe)
        prediction = self.fit_target_center.view(1, 1, -1).expand(
            *probe.valid_mask.shape,
            FULL_REFERENCE_WIDTH,
        )
        return prediction * probe.valid_mask.unsqueeze(-1)

    def position_prediction_for(self, probe: FullWidthReferenceProbe) -> Tensor:
        self._validate_probe_gauge(probe)
        result = torch.zeros(
            (*probe.valid_mask.shape, FULL_REFERENCE_WIDTH),
            dtype=torch.float64,
        )
        for batch_index in range(probe.valid_mask.shape[0]):
            valid_indices = torch.nonzero(
                probe.valid_mask[batch_index],
                as_tuple=False,
            ).flatten()
            bins = _normalized_position_bins(
                int(valid_indices.numel()),
                bin_count=self.position_bin_count,
            )
            result[batch_index, valid_indices] = (
                self.normalized_position_bin_centers[bins]
            )
        return result

    def _validate_probe_gauge(self, probe: FullWidthReferenceProbe) -> None:
        if not isinstance(probe, FullWidthReferenceProbe):
            raise TypeError("probe must be a FullWidthReferenceProbe")
        if (
            probe.standardized_gauge_sha256
            != self.standardized_gauge_sha256
        ):
            raise ValueError("probe and controls use different standardized gauges")


@dataclass(frozen=True, slots=True)
class FullWidthCandidatePrediction:
    """One retained-prefix prediction aligned to a target probe."""

    probe_id: str
    retained_standardized_prediction: Tensor
    standardized_gauge_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_string(self.probe_id, label="prediction probe id")
        prediction = _canonical_float_tensor(
            self.retained_standardized_prediction,
            label="retained standardized prediction",
            ndim=3,
        )
        if prediction.shape[-1] > FULL_REFERENCE_WIDTH:
            raise ValueError(
                f"retained prediction width exceeds {FULL_REFERENCE_WIDTH}"
            )
        _require_sha256(
            self.standardized_gauge_sha256,
            label="standardized gauge SHA-256",
        )
        object.__setattr__(
            self,
            "retained_standardized_prediction",
            prediction,
        )
        _digest_or_set(
            self,
            field_name="artifact_sha256",
            payload=self._payload(),
            domain=_PREDICTION_DOMAIN,
            label="full-width candidate prediction",
        )

    @property
    def target_rank(self) -> int:
        return int(self.retained_standardized_prediction.shape[-1])

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": _PREDICTION_KIND,
            "format_version": _FORMAT_VERSION,
            "probe_id": self.probe_id,
            "retained_standardized_prediction_sha256": _tensor_sha256(
                self.retained_standardized_prediction
            ),
            "standardized_gauge_sha256": self.standardized_gauge_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "retained_standardized_prediction": (
                self.retained_standardized_prediction.detach().clone()
            ),
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class FullWidthStructuralMetrics:
    """Runner-supplied structural checks applied by the frozen gates."""

    prepared_vs_analytic_relative_error: float
    causality_violation: float
    padding_violation: float
    repeat_relative_error: float
    in_support_fraction: float

    def __post_init__(self) -> None:
        for field_name in (
            "prepared_vs_analytic_relative_error",
            "causality_violation",
            "padding_violation",
            "repeat_relative_error",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_float(
                    getattr(self, field_name),
                    label=field_name.replace("_", " "),
                ),
            )
        support = _finite_float(
            self.in_support_fraction,
            label="in-support fraction",
        )
        if not 0.0 <= support <= 1.0:
            raise ValueError("in-support fraction must lie in [0, 1]")
        object.__setattr__(self, "in_support_fraction", support)

    def state_dict(self) -> dict[str, float]:
        return {
            "prepared_vs_analytic_relative_error": (
                self.prepared_vs_analytic_relative_error
            ),
            "causality_violation": self.causality_violation,
            "padding_violation": self.padding_violation,
            "repeat_relative_error": self.repeat_relative_error,
            "in_support_fraction": self.in_support_fraction,
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "FullWidthStructuralMetrics":
        state = _state_mapping(raw, label="full-width structural metrics")
        expected = {
            "prepared_vs_analytic_relative_error",
            "causality_violation",
            "padding_violation",
            "repeat_relative_error",
            "in_support_fraction",
        }
        _strict_state_keys(
            state,
            expected=expected,
            label="full-width structural metrics",
        )
        return cls(
            prepared_vs_analytic_relative_error=_state_nonnegative_float(
                state["prepared_vs_analytic_relative_error"],
                label="prepared-vs-analytic relative error",
            ),
            causality_violation=_state_nonnegative_float(
                state["causality_violation"],
                label="causality violation",
            ),
            padding_violation=_state_nonnegative_float(
                state["padding_violation"],
                label="padding violation",
            ),
            repeat_relative_error=_state_nonnegative_float(
                state["repeat_relative_error"],
                label="repeat relative error",
            ),
            in_support_fraction=_state_float(
                state["in_support_fraction"],
                label="in-support fraction",
            ),
        )


@dataclass(frozen=True, slots=True)
class FullWidthReferenceCandidate:
    """A sealed provider candidate and its selection-set predictions."""

    candidate_id: str
    source_rank: int
    target_rank: int
    stored_scalar_count: int
    predictions: tuple[FullWidthCandidatePrediction, ...]
    structural_metrics: FullWidthStructuralMetrics
    candidate_binding_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_string(self.candidate_id, label="candidate id")
        _positive_integer(self.source_rank, label="source rank")
        _positive_integer(self.target_rank, label="target rank")
        if self.source_rank > FULL_REFERENCE_WIDTH:
            raise ValueError(
                f"source rank exceeds {FULL_REFERENCE_WIDTH}"
            )
        if self.target_rank > FULL_REFERENCE_WIDTH:
            raise ValueError(
                f"target rank exceeds {FULL_REFERENCE_WIDTH}"
            )
        _nonnegative_integer(
            self.stored_scalar_count,
            label="stored scalar count",
        )
        if (
            not isinstance(self.predictions, tuple)
            or not self.predictions
            or any(
                not isinstance(value, FullWidthCandidatePrediction)
                for value in self.predictions
            )
        ):
            raise TypeError(
                "predictions must be a nonempty tuple of candidate predictions"
            )
        prediction_ids = tuple(value.probe_id for value in self.predictions)
        if len(set(prediction_ids)) != len(prediction_ids):
            raise ValueError("candidate contains duplicate prediction probe ids")
        if any(value.target_rank != self.target_rank for value in self.predictions):
            raise ValueError("candidate prediction width does not match target rank")
        gauges = {
            value.standardized_gauge_sha256 for value in self.predictions
        }
        if len(gauges) != 1:
            raise ValueError("candidate predictions use multiple gauges")
        if not isinstance(self.structural_metrics, FullWidthStructuralMetrics):
            raise TypeError(
                "structural metrics must be FullWidthStructuralMetrics"
            )
        _require_sha256(
            self.candidate_binding_sha256,
            label="candidate binding SHA-256",
        )
        _digest_or_set(
            self,
            field_name="artifact_sha256",
            payload=self._payload(),
            domain=_CANDIDATE_DOMAIN,
            label="full-width reference candidate",
        )

    @property
    def standardized_gauge_sha256(self) -> str:
        return self.predictions[0].standardized_gauge_sha256

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": _CANDIDATE_KIND,
            "format_version": _FORMAT_VERSION,
            "candidate_id": self.candidate_id,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "stored_scalar_count": self.stored_scalar_count,
            "prediction_sha256s": sorted(
                value.artifact_sha256 for value in self.predictions
            ),
            "structural_metrics": self.structural_metrics.state_dict(),
            "candidate_binding_sha256": self.candidate_binding_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "predictions": [
                value.state_dict() for value in self.predictions
            ],
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class FullWidthProbeMetric:
    probe_id: str
    family: str
    relative_error: float
    reference_cosine: float
    p90_row_relative_error: float

    def __post_init__(self) -> None:
        _require_nonempty_string(self.probe_id, label="probe metric id")
        _require_nonempty_string(self.family, label="probe metric family")
        object.__setattr__(
            self,
            "relative_error",
            _nonnegative_float(
                self.relative_error,
                label="probe relative error",
            ),
        )
        cosine = _finite_float(
            self.reference_cosine,
            label="probe reference cosine",
        )
        object.__setattr__(self, "reference_cosine", cosine)
        object.__setattr__(
            self,
            "p90_row_relative_error",
            _nonnegative_float(
                self.p90_row_relative_error,
                label="probe p90 row relative error",
            ),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "family": self.family,
            "relative_error": self.relative_error,
            "reference_cosine": self.reference_cosine,
            "p90_row_relative_error": self.p90_row_relative_error,
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "FullWidthProbeMetric":
        state = _state_mapping(raw, label="full-width probe metric")
        _strict_state_keys(
            state,
            expected={
                "probe_id",
                "family",
                "relative_error",
                "reference_cosine",
                "p90_row_relative_error",
            },
            label="full-width probe metric",
        )
        return cls(
            probe_id=_state_string(state["probe_id"], label="probe metric id"),
            family=_state_string(
                state["family"],
                label="probe metric family",
            ),
            relative_error=_state_nonnegative_float(
                state["relative_error"],
                label="probe relative error",
            ),
            reference_cosine=_state_float(
                state["reference_cosine"],
                label="probe reference cosine",
            ),
            p90_row_relative_error=_state_nonnegative_float(
                state["p90_row_relative_error"],
                label="probe p90 row relative error",
            ),
        )


@dataclass(frozen=True, slots=True)
class FullWidthFamilyMetric:
    family: str
    probe_count: int
    pooled_relative_error: float

    def __post_init__(self) -> None:
        _require_nonempty_string(self.family, label="family metric family")
        _positive_integer(self.probe_count, label="family metric probe count")
        object.__setattr__(
            self,
            "pooled_relative_error",
            _nonnegative_float(
                self.pooled_relative_error,
                label="family pooled relative error",
            ),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "probe_count": self.probe_count,
            "pooled_relative_error": self.pooled_relative_error,
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "FullWidthFamilyMetric":
        state = _state_mapping(raw, label="full-width family metric")
        _strict_state_keys(
            state,
            expected={"family", "probe_count", "pooled_relative_error"},
            label="full-width family metric",
        )
        return cls(
            family=_state_string(
                state["family"],
                label="family metric family",
            ),
            probe_count=_positive_integer(
                state["probe_count"],
                label="family metric probe count",
            ),
            pooled_relative_error=_state_nonnegative_float(
                state["pooled_relative_error"],
                label="family pooled relative error",
            ),
        )


@dataclass(frozen=True, slots=True)
class FullWidthCollisionMetric:
    collision_group: str
    variant_count: int
    minimum_pairwise_target_relative_difference: float

    def __post_init__(self) -> None:
        _require_nonempty_string(
            self.collision_group,
            label="collision metric group",
        )
        count = _positive_integer(
            self.variant_count,
            label="collision metric variant count",
        )
        if count < 2:
            raise ValueError(
                "collision metric variant count must be at least two"
            )
        object.__setattr__(
            self,
            "minimum_pairwise_target_relative_difference",
            _nonnegative_float(
                self.minimum_pairwise_target_relative_difference,
                label="minimum pairwise target relative difference",
            ),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "collision_group": self.collision_group,
            "variant_count": self.variant_count,
            "minimum_pairwise_target_relative_difference": (
                self.minimum_pairwise_target_relative_difference
            ),
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "FullWidthCollisionMetric":
        state = _state_mapping(raw, label="full-width collision metric")
        _strict_state_keys(
            state,
            expected={
                "collision_group",
                "variant_count",
                "minimum_pairwise_target_relative_difference",
            },
            label="full-width collision metric",
        )
        return cls(
            collision_group=_state_string(
                state["collision_group"],
                label="collision metric group",
            ),
            variant_count=_positive_integer(
                state["variant_count"],
                label="collision metric variant count",
            ),
            minimum_pairwise_target_relative_difference=(
                _state_nonnegative_float(
                    state[
                        "minimum_pairwise_target_relative_difference"
                    ],
                    label=(
                        "minimum pairwise target relative difference"
                    ),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class FullWidthGateFlags:
    fisher_weighted_relative_error: bool
    reference_cosine: bool
    error_reduction_vs_constant: bool
    error_reduction_vs_position_only: bool
    per_probe_p90_relative_error: bool
    worst_family_relative_error: bool
    prepared_vs_analytic_relative_error: bool
    causality_violation: bool
    padding_violation: bool
    repeat_relative_error: bool
    collision_target_relative_difference: bool
    in_support_fraction: bool

    def __post_init__(self) -> None:
        for field_name in self.state_dict():
            _state_bool(
                getattr(self, field_name),
                label=field_name.replace("_", " "),
            )

    @property
    def all_passed(self) -> bool:
        return all(self.state_dict().values())

    def state_dict(self) -> dict[str, bool]:
        return {
            "fisher_weighted_relative_error": (
                self.fisher_weighted_relative_error
            ),
            "reference_cosine": self.reference_cosine,
            "error_reduction_vs_constant": self.error_reduction_vs_constant,
            "error_reduction_vs_position_only": (
                self.error_reduction_vs_position_only
            ),
            "per_probe_p90_relative_error": (
                self.per_probe_p90_relative_error
            ),
            "worst_family_relative_error": (
                self.worst_family_relative_error
            ),
            "prepared_vs_analytic_relative_error": (
                self.prepared_vs_analytic_relative_error
            ),
            "causality_violation": self.causality_violation,
            "padding_violation": self.padding_violation,
            "repeat_relative_error": self.repeat_relative_error,
            "collision_target_relative_difference": (
                self.collision_target_relative_difference
            ),
            "in_support_fraction": self.in_support_fraction,
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "FullWidthGateFlags":
        state = _state_mapping(raw, label="full-width gate flags")
        expected = set(cls.__dataclass_fields__)
        _strict_state_keys(
            state,
            expected=expected,
            label="full-width gate flags",
        )
        return cls(
            **{
                field_name: _state_bool(
                    state[field_name],
                    label=field_name.replace("_", " "),
                )
                for field_name in expected
            }
        )


@dataclass(frozen=True, slots=True)
class FullWidthCandidateScore:
    """Unrounded full-width metrics and gate decisions for one candidate."""

    candidate_id: str
    candidate_artifact_sha256: str
    source_rank: int
    target_rank: int
    stored_scalar_count: int
    fisher_weighted_relative_error: float
    reference_cosine: float
    constant_control_relative_error: float
    position_only_control_relative_error: float
    error_reduction_vs_constant: float
    error_reduction_vs_position_only: float
    maximum_per_probe_p90_relative_error: float
    worst_family_relative_error: float
    probe_metrics: tuple[FullWidthProbeMetric, ...]
    family_metrics: tuple[FullWidthFamilyMetric, ...]
    collision_metrics: tuple[FullWidthCollisionMetric, ...]
    minimum_collision_target_relative_difference: float
    structural_metrics: FullWidthStructuralMetrics
    gate_flags: FullWidthGateFlags
    passed: bool
    controls_artifact_sha256: str
    gates_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_string(self.candidate_id, label="candidate id")
        for field_name in (
            "candidate_artifact_sha256",
            "controls_artifact_sha256",
            "gates_sha256",
        ):
            _require_sha256(
                getattr(self, field_name),
                label=field_name.replace("_", " "),
            )
        _positive_integer(self.source_rank, label="source rank")
        _positive_integer(self.target_rank, label="target rank")
        if self.source_rank > FULL_REFERENCE_WIDTH:
            raise ValueError(
                f"source rank exceeds {FULL_REFERENCE_WIDTH}"
            )
        if self.target_rank > FULL_REFERENCE_WIDTH:
            raise ValueError(
                f"target rank exceeds {FULL_REFERENCE_WIDTH}"
            )
        _nonnegative_integer(
            self.stored_scalar_count,
            label="stored scalar count",
        )
        for field_name in (
            "fisher_weighted_relative_error",
            "reference_cosine",
            "constant_control_relative_error",
            "position_only_control_relative_error",
            "error_reduction_vs_constant",
            "error_reduction_vs_position_only",
            "maximum_per_probe_p90_relative_error",
            "worst_family_relative_error",
            "minimum_collision_target_relative_difference",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(
                    getattr(self, field_name),
                    label=field_name.replace("_", " "),
                ),
            )
        if (
            type(self.probe_metrics) is not tuple
            or not self.probe_metrics
            or any(
                not isinstance(value, FullWidthProbeMetric)
                for value in self.probe_metrics
            )
        ):
            raise TypeError(
                "probe metrics must be a nonempty tuple of "
                "FullWidthProbeMetric values"
            )
        if tuple(
            sorted(self.probe_metrics, key=lambda value: value.probe_id)
        ) != self.probe_metrics:
            raise ValueError("probe metrics must be sorted by probe id")
        if len({value.probe_id for value in self.probe_metrics}) != len(
            self.probe_metrics
        ):
            raise ValueError("probe metrics contain duplicate probe ids")
        if (
            type(self.family_metrics) is not tuple
            or not self.family_metrics
            or any(
                not isinstance(value, FullWidthFamilyMetric)
                for value in self.family_metrics
            )
        ):
            raise TypeError(
                "family metrics must be a nonempty tuple of "
                "FullWidthFamilyMetric values"
            )
        if tuple(
            sorted(self.family_metrics, key=lambda value: value.family)
        ) != self.family_metrics:
            raise ValueError("family metrics must be sorted by family")
        if len({value.family for value in self.family_metrics}) != len(
            self.family_metrics
        ):
            raise ValueError("family metrics contain duplicate families")
        if (
            type(self.collision_metrics) is not tuple
            or any(
                not isinstance(value, FullWidthCollisionMetric)
                for value in self.collision_metrics
            )
        ):
            raise TypeError(
                "collision metrics must be a tuple of "
                "FullWidthCollisionMetric values"
            )
        if tuple(
            sorted(
                self.collision_metrics,
                key=lambda value: value.collision_group,
            )
        ) != self.collision_metrics:
            raise ValueError(
                "collision metrics must be sorted by collision group"
            )
        if len(
            {value.collision_group for value in self.collision_metrics}
        ) != len(self.collision_metrics):
            raise ValueError("collision metrics contain duplicate groups")
        if not isinstance(
            self.structural_metrics,
            FullWidthStructuralMetrics,
        ):
            raise TypeError(
                "structural metrics must be FullWidthStructuralMetrics"
            )
        if not isinstance(self.gate_flags, FullWidthGateFlags):
            raise TypeError("gate flags must be FullWidthGateFlags")
        if type(self.passed) is not bool:
            raise TypeError("candidate pass flag must be a boolean")
        family_probe_counts: dict[str, int] = defaultdict(int)
        for metric in self.probe_metrics:
            family_probe_counts[metric.family] += 1
        declared_family_counts = {
            metric.family: metric.probe_count
            for metric in self.family_metrics
        }
        if dict(family_probe_counts) != declared_family_counts:
            raise ValueError(
                "family metric accounting does not match probe metrics"
            )
        if (
            self.maximum_per_probe_p90_relative_error
            != max(
                value.p90_row_relative_error
                for value in self.probe_metrics
            )
        ):
            raise ValueError("maximum probe p90 metric drifted")
        if self.worst_family_relative_error != max(
            value.pooled_relative_error for value in self.family_metrics
        ):
            raise ValueError("worst family metric drifted")
        expected_collision_minimum = (
            min(
                value.minimum_pairwise_target_relative_difference
                for value in self.collision_metrics
            )
            if self.collision_metrics
            else 0.0
        )
        if (
            self.minimum_collision_target_relative_difference
            != expected_collision_minimum
        ):
            raise ValueError("minimum collision metric drifted")
        if self.passed != self.gate_flags.all_passed:
            raise ValueError("candidate pass flag does not match gate flags")
        _digest_or_set(
            self,
            field_name="artifact_sha256",
            payload=self._payload(),
            domain=_SCORE_DOMAIN,
            label="full-width candidate score",
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": _SCORE_KIND,
            "format_version": _FORMAT_VERSION,
            "candidate_id": self.candidate_id,
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "stored_scalar_count": self.stored_scalar_count,
            "fisher_weighted_relative_error": (
                self.fisher_weighted_relative_error
            ),
            "reference_cosine": self.reference_cosine,
            "constant_control_relative_error": (
                self.constant_control_relative_error
            ),
            "position_only_control_relative_error": (
                self.position_only_control_relative_error
            ),
            "error_reduction_vs_constant": self.error_reduction_vs_constant,
            "error_reduction_vs_position_only": (
                self.error_reduction_vs_position_only
            ),
            "maximum_per_probe_p90_relative_error": (
                self.maximum_per_probe_p90_relative_error
            ),
            "worst_family_relative_error": self.worst_family_relative_error,
            "probe_metrics": [
                value.state_dict() for value in self.probe_metrics
            ],
            "family_metrics": [
                value.state_dict() for value in self.family_metrics
            ],
            "collision_metrics": [
                value.state_dict() for value in self.collision_metrics
            ],
            "minimum_collision_target_relative_difference": (
                self.minimum_collision_target_relative_difference
            ),
            "structural_metrics": self.structural_metrics.state_dict(),
            "gate_flags": self.gate_flags.state_dict(),
            "passed": self.passed,
            "controls_artifact_sha256": self.controls_artifact_sha256,
            "gates_sha256": self.gates_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "FullWidthCandidateScore":
        state = _state_mapping(raw, label="full-width candidate score")
        expected = {
            "artifact_kind",
            "format_version",
            "candidate_id",
            "candidate_artifact_sha256",
            "source_rank",
            "target_rank",
            "stored_scalar_count",
            "fisher_weighted_relative_error",
            "reference_cosine",
            "constant_control_relative_error",
            "position_only_control_relative_error",
            "error_reduction_vs_constant",
            "error_reduction_vs_position_only",
            "maximum_per_probe_p90_relative_error",
            "worst_family_relative_error",
            "probe_metrics",
            "family_metrics",
            "collision_metrics",
            "minimum_collision_target_relative_difference",
            "structural_metrics",
            "gate_flags",
            "passed",
            "controls_artifact_sha256",
            "gates_sha256",
            "artifact_sha256",
        }
        _strict_state_keys(
            state,
            expected=expected,
            label="full-width candidate score",
        )
        _validate_state_header(
            state,
            artifact_kind=_SCORE_KIND,
            label="full-width candidate score",
        )
        probe_states = _state_list(
            state["probe_metrics"],
            label="full-width probe metrics",
        )
        family_states = _state_list(
            state["family_metrics"],
            label="full-width family metrics",
        )
        collision_states = _state_list(
            state["collision_metrics"],
            label="full-width collision metrics",
        )
        return cls(
            candidate_id=_state_string(
                state["candidate_id"],
                label="candidate id",
            ),
            candidate_artifact_sha256=_state_sha256(
                state["candidate_artifact_sha256"],
                label="candidate artifact SHA-256",
            ),
            source_rank=_positive_integer(
                state["source_rank"],
                label="source rank",
            ),
            target_rank=_positive_integer(
                state["target_rank"],
                label="target rank",
            ),
            stored_scalar_count=_nonnegative_integer(
                state["stored_scalar_count"],
                label="stored scalar count",
            ),
            fisher_weighted_relative_error=_state_float(
                state["fisher_weighted_relative_error"],
                label="fisher-weighted relative error",
            ),
            reference_cosine=_state_float(
                state["reference_cosine"],
                label="reference cosine",
            ),
            constant_control_relative_error=_state_float(
                state["constant_control_relative_error"],
                label="constant control relative error",
            ),
            position_only_control_relative_error=_state_float(
                state["position_only_control_relative_error"],
                label="position-only control relative error",
            ),
            error_reduction_vs_constant=_state_float(
                state["error_reduction_vs_constant"],
                label="error reduction vs constant",
            ),
            error_reduction_vs_position_only=_state_float(
                state["error_reduction_vs_position_only"],
                label="error reduction vs position only",
            ),
            maximum_per_probe_p90_relative_error=_state_float(
                state["maximum_per_probe_p90_relative_error"],
                label="maximum per-probe p90 relative error",
            ),
            worst_family_relative_error=_state_float(
                state["worst_family_relative_error"],
                label="worst family relative error",
            ),
            probe_metrics=tuple(
                FullWidthProbeMetric.from_state_dict(value)
                for value in probe_states
            ),
            family_metrics=tuple(
                FullWidthFamilyMetric.from_state_dict(value)
                for value in family_states
            ),
            collision_metrics=tuple(
                FullWidthCollisionMetric.from_state_dict(value)
                for value in collision_states
            ),
            minimum_collision_target_relative_difference=_state_float(
                state["minimum_collision_target_relative_difference"],
                label="minimum collision target relative difference",
            ),
            structural_metrics=FullWidthStructuralMetrics.from_state_dict(
                state["structural_metrics"]
            ),
            gate_flags=FullWidthGateFlags.from_state_dict(
                state["gate_flags"]
            ),
            passed=_state_bool(
                state["passed"],
                label="candidate pass flag",
            ),
            controls_artifact_sha256=_state_sha256(
                state["controls_artifact_sha256"],
                label="controls artifact SHA-256",
            ),
            gates_sha256=_state_sha256(
                state["gates_sha256"],
                label="gates SHA-256",
            ),
            artifact_sha256=_state_sha256(
                state["artifact_sha256"],
                label="candidate score artifact SHA-256",
            ),
        )


@dataclass(frozen=True, slots=True)
class FullWidthReferenceSelection:
    """Deterministic smallest-passing selection result."""

    selected_candidate_id: str | None
    selected_candidate_artifact_sha256: str | None
    selected_stored_scalar_count: int | None
    selected_source_rank: int | None
    selected_target_rank: int | None
    candidate_scores: tuple[FullWidthCandidateScore, ...]
    controls_artifact_sha256: str
    selection_probe_sha256s: tuple[str, ...]
    collision_probe_sha256s: tuple[str, ...]
    gates_sha256: str
    selection_rule: str = SELECTION_RULE
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.candidate_scores) is not tuple
            or not self.candidate_scores
            or any(
                not isinstance(value, FullWidthCandidateScore)
                for value in self.candidate_scores
            )
        ):
            raise ValueError(
                "candidate scores must be a nonempty tuple of "
                "FullWidthCandidateScore values"
            )
        if tuple(
            sorted(self.candidate_scores, key=lambda value: value.candidate_id)
        ) != self.candidate_scores:
            raise ValueError("candidate scores must be sorted by candidate id")
        if len({value.candidate_id for value in self.candidate_scores}) != len(
            self.candidate_scores
        ):
            raise ValueError("candidate scores contain duplicate ids")
        if self.selection_rule != SELECTION_RULE:
            raise ValueError("selection rule drifted")
        _require_sha256(
            self.controls_artifact_sha256,
            label="selection controls artifact SHA-256",
        )
        _require_sha256(
            self.gates_sha256,
            label="selection gates SHA-256",
        )
        for field_name in (
            "selection_probe_sha256s",
            "collision_probe_sha256s",
        ):
            values = getattr(self, field_name)
            if type(values) is not tuple:
                raise TypeError(f"{field_name} must be a tuple")
            if tuple(sorted(values)) != values or len(set(values)) != len(
                values
            ):
                raise ValueError(
                    f"{field_name} must be a sorted unique tuple"
                )
            for value in values:
                _require_sha256(
                    value,
                    label="selection probe binding SHA-256",
                )
        if not self.selection_probe_sha256s:
            raise ValueError("selection probe SHA-256 values must not be empty")
        if any(
            score.controls_artifact_sha256
            != self.controls_artifact_sha256
            for score in self.candidate_scores
        ):
            raise ValueError(
                "candidate score controls binding differs from selection"
            )
        if any(
            score.gates_sha256 != self.gates_sha256
            for score in self.candidate_scores
        ):
            raise ValueError(
                "candidate score gates binding differs from selection"
            )
        selected_fields = (
            self.selected_candidate_id,
            self.selected_candidate_artifact_sha256,
            self.selected_stored_scalar_count,
            self.selected_source_rank,
            self.selected_target_rank,
        )
        if all(value is None for value in selected_fields):
            if any(value.passed for value in self.candidate_scores):
                raise ValueError("a passing candidate was not selected")
        elif any(value is None for value in selected_fields):
            raise ValueError("selected candidate fields must be all set or all null")
        else:
            selected = [
                value
                for value in self.candidate_scores
                if value.candidate_id == self.selected_candidate_id
            ]
            if len(selected) != 1 or not selected[0].passed:
                raise ValueError("selected candidate is not a unique passer")
            score = selected[0]
            if (
                score.candidate_artifact_sha256
                != self.selected_candidate_artifact_sha256
                or score.stored_scalar_count
                != self.selected_stored_scalar_count
                or score.source_rank != self.selected_source_rank
                or score.target_rank != self.selected_target_rank
            ):
                raise ValueError("selected candidate accounting drifted")
            expected = min(
                (value for value in self.candidate_scores if value.passed),
                key=_selection_key,
            )
            if score.candidate_id != expected.candidate_id:
                raise ValueError("selected candidate violates selection rule")
        _digest_or_set(
            self,
            field_name="artifact_sha256",
            payload=self._payload(),
            domain=_SELECTION_DOMAIN,
            label="full-width reference selection",
        )

    @property
    def passed_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            value.candidate_id
            for value in self.candidate_scores
            if value.passed
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": _SELECTION_KIND,
            "format_version": _FORMAT_VERSION,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_artifact_sha256": (
                self.selected_candidate_artifact_sha256
            ),
            "selected_stored_scalar_count": self.selected_stored_scalar_count,
            "selected_source_rank": self.selected_source_rank,
            "selected_target_rank": self.selected_target_rank,
            "candidate_score_sha256s": [
                value.artifact_sha256 for value in self.candidate_scores
            ],
            "controls_artifact_sha256": self.controls_artifact_sha256,
            "selection_probe_sha256s": list(self.selection_probe_sha256s),
            "collision_probe_sha256s": list(self.collision_probe_sha256s),
            "gates_sha256": self.gates_sha256,
            "selection_rule": self.selection_rule,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "candidate_scores": [
                value.state_dict() for value in self.candidate_scores
            ],
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "FullWidthReferenceSelection":
        state = _state_mapping(raw, label="full-width reference selection")
        expected = {
            "artifact_kind",
            "format_version",
            "selected_candidate_id",
            "selected_candidate_artifact_sha256",
            "selected_stored_scalar_count",
            "selected_source_rank",
            "selected_target_rank",
            "candidate_score_sha256s",
            "candidate_scores",
            "controls_artifact_sha256",
            "selection_probe_sha256s",
            "collision_probe_sha256s",
            "gates_sha256",
            "selection_rule",
            "artifact_sha256",
        }
        _strict_state_keys(
            state,
            expected=expected,
            label="full-width reference selection",
        )
        _validate_state_header(
            state,
            artifact_kind=_SELECTION_KIND,
            label="full-width reference selection",
        )
        score_states = _state_list(
            state["candidate_scores"],
            label="candidate scores",
        )
        if not score_states:
            raise ValueError("candidate scores must not be empty")
        scores = tuple(
            FullWidthCandidateScore.from_state_dict(value)
            for value in score_states
        )
        score_sha_states = _state_list(
            state["candidate_score_sha256s"],
            label="candidate score SHA-256 values",
        )
        score_sha256s = tuple(
            _state_sha256(
                value,
                label="candidate score SHA-256",
            )
            for value in score_sha_states
        )
        actual_score_sha256s = tuple(
            value.artifact_sha256 for value in scores
        )
        if score_sha256s != actual_score_sha256s:
            raise ValueError(
                "candidate score SHA-256 summary does not match "
                "restored scores"
            )
        selection_probe_states = _state_list(
            state["selection_probe_sha256s"],
            label="selection probe SHA-256 values",
        )
        collision_probe_states = _state_list(
            state["collision_probe_sha256s"],
            label="collision probe SHA-256 values",
        )

        selected_candidate_id = state["selected_candidate_id"]
        selected_candidate_sha256 = state[
            "selected_candidate_artifact_sha256"
        ]
        selected_stored_scalar_count = state[
            "selected_stored_scalar_count"
        ]
        selected_source_rank = state["selected_source_rank"]
        selected_target_rank = state["selected_target_rank"]
        if selected_candidate_id is not None:
            selected_candidate_id = _state_string(
                selected_candidate_id,
                label="selected candidate id",
            )
        if selected_candidate_sha256 is not None:
            selected_candidate_sha256 = _state_sha256(
                selected_candidate_sha256,
                label="selected candidate artifact SHA-256",
            )
        if selected_stored_scalar_count is not None:
            selected_stored_scalar_count = _nonnegative_integer(
                selected_stored_scalar_count,
                label="selected stored scalar count",
            )
        if selected_source_rank is not None:
            selected_source_rank = _positive_integer(
                selected_source_rank,
                label="selected source rank",
            )
        if selected_target_rank is not None:
            selected_target_rank = _positive_integer(
                selected_target_rank,
                label="selected target rank",
            )
        selection_rule = state["selection_rule"]
        if type(selection_rule) is not str:
            raise TypeError("selection rule must be a string")

        return cls(
            selected_candidate_id=selected_candidate_id,
            selected_candidate_artifact_sha256=selected_candidate_sha256,
            selected_stored_scalar_count=selected_stored_scalar_count,
            selected_source_rank=selected_source_rank,
            selected_target_rank=selected_target_rank,
            candidate_scores=scores,
            controls_artifact_sha256=_state_sha256(
                state["controls_artifact_sha256"],
                label="selection controls artifact SHA-256",
            ),
            selection_probe_sha256s=tuple(
                _state_sha256(
                    value,
                    label="selection probe SHA-256",
                )
                for value in selection_probe_states
            ),
            collision_probe_sha256s=tuple(
                _state_sha256(
                    value,
                    label="collision probe SHA-256",
                )
                for value in collision_probe_states
            ),
            gates_sha256=_state_sha256(
                state["gates_sha256"],
                label="selection gates SHA-256",
            ),
            selection_rule=selection_rule,
            artifact_sha256=_state_sha256(
                state["artifact_sha256"],
                label="selection artifact SHA-256",
            ),
        )


def _normalized_position_bins(length: int, *, bin_count: int) -> Tensor:
    _positive_integer(length, label="valid sequence length")
    _positive_integer(bin_count, label="position bin count")
    denominator = max(length - 1, 1)
    normalized = (
        torch.arange(length, dtype=torch.float64) / float(denominator)
    )
    return torch.clamp(
        torch.floor(normalized * bin_count).to(torch.int64),
        max=bin_count - 1,
    )


def fit_full_width_reference_controls(
    *,
    fit_probes: Sequence[FullWidthReferenceProbe],
    position_bin_count: int = 16,
) -> FullWidthReferenceControls:
    """Fit controls from fit probes only, then return a sealed artifact."""

    probes = _validate_probe_collection(
        fit_probes,
        split="fit",
        label="fit probes",
        nonempty=True,
    )
    _positive_integer(position_bin_count, label="position bin count")
    gauges = {probe.standardized_gauge_sha256 for probe in probes}
    if len(gauges) != 1:
        raise ValueError("fit probes use multiple standardized gauges")

    ordered = tuple(sorted(probes, key=lambda value: value.artifact_sha256))
    rows = torch.cat(
        [
            probe.standardized_target[probe.valid_mask]
            for probe in ordered
        ],
        dim=0,
    )
    center = rows.mean(dim=0)
    bin_sums = torch.zeros(
        position_bin_count,
        FULL_REFERENCE_WIDTH,
        dtype=torch.float64,
    )
    bin_counts = torch.zeros(position_bin_count, dtype=torch.int64)
    for probe in ordered:
        for batch_index in range(probe.valid_mask.shape[0]):
            valid_indices = torch.nonzero(
                probe.valid_mask[batch_index],
                as_tuple=False,
            ).flatten()
            bins = _normalized_position_bins(
                int(valid_indices.numel()),
                bin_count=position_bin_count,
            )
            values = probe.standardized_target[batch_index, valid_indices]
            bin_sums.index_add_(0, bins, values)
            bin_counts.index_add_(
                0,
                bins,
                torch.ones_like(bins, dtype=torch.int64),
            )
    bin_centers = center.unsqueeze(0).expand(position_bin_count, -1).clone()
    occupied = bin_counts > 0
    bin_centers[occupied] = (
        bin_sums[occupied] / bin_counts[occupied].to(torch.float64).unsqueeze(1)
    )
    return FullWidthReferenceControls(
        fit_target_center=center,
        normalized_position_bin_centers=bin_centers,
        normalized_position_bin_counts=tuple(
            int(value) for value in bin_counts.tolist()
        ),
        fit_probe_ids=tuple(sorted(probe.probe_id for probe in probes)),
        fit_probe_sha256s=tuple(
            sorted(probe.artifact_sha256 for probe in probes)
        ),
        standardized_gauge_sha256=next(iter(gauges)),
    )


def reconstruct_full_width_prediction(
    *,
    controls: FullWidthReferenceControls,
    probe: FullWidthReferenceProbe,
    prediction: FullWidthCandidatePrediction,
) -> Tensor:
    """Fill a retained prefix into the fit center in the full 64-mode gauge."""

    if not isinstance(controls, FullWidthReferenceControls):
        raise TypeError("controls must be FullWidthReferenceControls")
    if not isinstance(probe, FullWidthReferenceProbe):
        raise TypeError("probe must be FullWidthReferenceProbe")
    if not isinstance(prediction, FullWidthCandidatePrediction):
        raise TypeError("prediction must be FullWidthCandidatePrediction")
    if prediction.probe_id != probe.probe_id:
        raise ValueError("prediction and target probe ids do not match")
    if (
        controls.standardized_gauge_sha256
        != probe.standardized_gauge_sha256
        or prediction.standardized_gauge_sha256
        != probe.standardized_gauge_sha256
    ):
        raise ValueError("prediction, target, and controls use different gauges")
    if prediction.retained_standardized_prediction.shape[:2] != (
        probe.standardized_target.shape[:2]
    ):
        raise ValueError("prediction and target probe shapes are not aligned")
    result = controls.fit_target_center.view(1, 1, -1).expand(
        *probe.valid_mask.shape,
        FULL_REFERENCE_WIDTH,
    ).clone()
    result[..., : prediction.target_rank] = (
        prediction.retained_standardized_prediction
    )
    return result * probe.valid_mask.unsqueeze(-1)


def _relative_error(target: Tensor, prediction: Tensor) -> float:
    numerator = float(torch.linalg.vector_norm(target - prediction).item())
    denominator = max(
        float(torch.linalg.vector_norm(target).item()),
        RELATIVE_ERROR_EPSILON,
    )
    return numerator / denominator


def _reference_cosine(target: Tensor, prediction: Tensor) -> float:
    target_norm = float(torch.linalg.vector_norm(target).item())
    prediction_norm = float(torch.linalg.vector_norm(prediction).item())
    if target_norm <= RELATIVE_ERROR_EPSILON:
        return 1.0 if prediction_norm <= RELATIVE_ERROR_EPSILON else 0.0
    if prediction_norm <= RELATIVE_ERROR_EPSILON:
        return 0.0
    return float(
        torch.dot(target.flatten(), prediction.flatten()).item()
        / (target_norm * prediction_norm)
    )


def _p90_row_relative_error(target: Tensor, prediction: Tensor) -> float:
    numerator = torch.linalg.vector_norm(target - prediction, dim=-1)
    denominator = torch.clamp(
        torch.linalg.vector_norm(target, dim=-1),
        min=RELATIVE_ERROR_EPSILON,
    )
    values = numerator / denominator
    return float(torch.quantile(values, 0.9).item())


def _control_relative_error(
    probes: tuple[FullWidthReferenceProbe, ...],
    *,
    controls: FullWidthReferenceControls,
    position_only: bool,
) -> float:
    target_rows: list[Tensor] = []
    prediction_rows: list[Tensor] = []
    for probe in probes:
        prediction = (
            controls.position_prediction_for(probe)
            if position_only
            else controls.constant_prediction_for(probe)
        )
        target_rows.append(probe.standardized_target[probe.valid_mask])
        prediction_rows.append(prediction[probe.valid_mask])
    return _relative_error(
        torch.cat(target_rows, dim=0),
        torch.cat(prediction_rows, dim=0),
    )


def _collision_metrics(
    collision_probes: tuple[FullWidthReferenceProbe, ...],
) -> tuple[FullWidthCollisionMetric, ...]:
    by_group: dict[str, list[FullWidthReferenceProbe]] = defaultdict(list)
    for probe in collision_probes:
        if probe.collision_group is None:
            raise ValueError(
                "every collision probe must declare a collision group"
            )
        by_group[probe.collision_group].append(probe)
    result: list[FullWidthCollisionMetric] = []
    for group in sorted(by_group):
        variants = sorted(
            by_group[group],
            key=lambda value: str(value.collision_variant),
        )
        if len(variants) < 2:
            raise ValueError(
                f"collision group {group!r} must contain at least two variants"
            )
        variant_names = [value.collision_variant for value in variants]
        if len(set(variant_names)) != len(variant_names):
            raise ValueError(
                f"collision group {group!r} contains duplicate variants"
            )
        anchor = variants[0]
        for other in variants[1:]:
            if (
                other.standardized_target.shape
                != anchor.standardized_target.shape
                or not torch.equal(other.valid_mask, anchor.valid_mask)
                or not torch.equal(
                    other.logical_positions,
                    anchor.logical_positions,
                )
            ):
                raise ValueError(
                    f"collision group {group!r} variants are not aligned"
                )
        separations: list[float] = []
        for left_index, left in enumerate(variants):
            left_rows = left.standardized_target[left.valid_mask]
            left_norm = float(torch.linalg.vector_norm(left_rows).item())
            for right in variants[left_index + 1 :]:
                right_rows = right.standardized_target[right.valid_mask]
                right_norm = float(torch.linalg.vector_norm(right_rows).item())
                difference = float(
                    torch.linalg.vector_norm(left_rows - right_rows).item()
                )
                denominator = max(
                    0.5 * (left_norm + right_norm),
                    RELATIVE_ERROR_EPSILON,
                )
                separations.append(difference / denominator)
        result.append(
            FullWidthCollisionMetric(
                collision_group=group,
                variant_count=len(variants),
                minimum_pairwise_target_relative_difference=min(separations),
            )
        )
    return tuple(result)


def _score_candidate_rows(
    *,
    controls: FullWidthReferenceControls,
    probes: tuple[FullWidthReferenceProbe, ...],
    candidate: FullWidthReferenceCandidate,
) -> tuple[
    Tensor,
    Tensor,
    tuple[FullWidthProbeMetric, ...],
    tuple[FullWidthFamilyMetric, ...],
]:
    predictions_by_id = {
        value.probe_id: value for value in candidate.predictions
    }
    probe_ids = {probe.probe_id for probe in probes}
    if set(predictions_by_id) != probe_ids:
        missing = sorted(probe_ids - set(predictions_by_id))
        extra = sorted(set(predictions_by_id) - probe_ids)
        raise ValueError(
            f"candidate predictions are not probe-aligned; "
            f"missing={missing}, extra={extra}"
        )
    all_targets: list[Tensor] = []
    all_predictions: list[Tensor] = []
    probe_metrics: list[FullWidthProbeMetric] = []
    family_rows: dict[str, list[tuple[Tensor, Tensor]]] = defaultdict(list)
    for probe in sorted(probes, key=lambda value: value.probe_id):
        prediction = reconstruct_full_width_prediction(
            controls=controls,
            probe=probe,
            prediction=predictions_by_id[probe.probe_id],
        )
        target_rows = probe.standardized_target[probe.valid_mask]
        prediction_rows = prediction[probe.valid_mask]
        all_targets.append(target_rows)
        all_predictions.append(prediction_rows)
        family_rows[probe.family].append((target_rows, prediction_rows))
        probe_metrics.append(
            FullWidthProbeMetric(
                probe_id=probe.probe_id,
                family=probe.family,
                relative_error=_relative_error(
                    target_rows,
                    prediction_rows,
                ),
                reference_cosine=_reference_cosine(
                    target_rows,
                    prediction_rows,
                ),
                p90_row_relative_error=_p90_row_relative_error(
                    target_rows,
                    prediction_rows,
                ),
            )
        )
    family_metrics: list[FullWidthFamilyMetric] = []
    for family in sorted(family_rows):
        pairs = family_rows[family]
        family_metrics.append(
            FullWidthFamilyMetric(
                family=family,
                probe_count=len(pairs),
                pooled_relative_error=_relative_error(
                    torch.cat([target for target, _ in pairs], dim=0),
                    torch.cat([prediction for _, prediction in pairs], dim=0),
                ),
            )
        )
    return (
        torch.cat(all_targets, dim=0),
        torch.cat(all_predictions, dim=0),
        tuple(probe_metrics),
        tuple(family_metrics),
    )


def _reduction(candidate_error: float, control_error: float) -> float:
    return (control_error - candidate_error) / max(
        control_error,
        RELATIVE_ERROR_EPSILON,
    )


def _gates_sha256(gates: SyntheticReferenceGates) -> str:
    if not isinstance(gates, SyntheticReferenceGates):
        raise TypeError("gates must be SyntheticReferenceGates")
    return _json_sha256(gates.state_dict(), domain=_GATES_DOMAIN)


def full_width_reference_gates_sha256(
    gates: SyntheticReferenceGates,
) -> str:
    """Return the canonical gate binding used by scores and selections."""

    return _gates_sha256(gates)


def _score_validated_full_width_reference_candidate(
    *,
    controls: FullWidthReferenceControls,
    probes: tuple[FullWidthReferenceProbe, ...],
    collision_probes: tuple[FullWidthReferenceProbe, ...],
    collision_gate_deferred: bool,
    candidate: FullWidthReferenceCandidate,
    gates: SyntheticReferenceGates,
) -> FullWidthCandidateScore:
    """Apply the shared metrics and gates after split-specific validation."""

    (
        target_rows,
        prediction_rows,
        probe_metrics,
        family_metrics,
    ) = _score_candidate_rows(
        controls=controls,
        probes=probes,
        candidate=candidate,
    )
    relative_error = _relative_error(target_rows, prediction_rows)
    reference_cosine = _reference_cosine(target_rows, prediction_rows)
    constant_error = _control_relative_error(
        probes,
        controls=controls,
        position_only=False,
    )
    position_error = _control_relative_error(
        probes,
        controls=controls,
        position_only=True,
    )
    reduction_constant = _reduction(relative_error, constant_error)
    reduction_position = _reduction(relative_error, position_error)
    maximum_probe_p90 = max(
        value.p90_row_relative_error for value in probe_metrics
    )
    worst_family = max(
        value.pooled_relative_error for value in family_metrics
    )
    collision_metrics = (
        ()
        if collision_gate_deferred
        else _collision_metrics(collision_probes)
    )
    minimum_collision = (
        0.0
        if collision_gate_deferred
        else min(
            value.minimum_pairwise_target_relative_difference
            for value in collision_metrics
        )
    )
    structural = candidate.structural_metrics
    flags = FullWidthGateFlags(
        fisher_weighted_relative_error=(
            relative_error <= gates.maximum_fisher_weighted_relative_error
        ),
        reference_cosine=(
            reference_cosine >= gates.minimum_reference_cosine
        ),
        error_reduction_vs_constant=(
            reduction_constant
            >= gates.minimum_error_reduction_vs_constant
        ),
        error_reduction_vs_position_only=(
            reduction_position
            >= gates.minimum_error_reduction_vs_position_only
        ),
        per_probe_p90_relative_error=(
            maximum_probe_p90
            <= gates.maximum_per_probe_p90_relative_error
        ),
        worst_family_relative_error=(
            worst_family <= gates.maximum_worst_panel_relative_error
        ),
        prepared_vs_analytic_relative_error=(
            structural.prepared_vs_analytic_relative_error
            <= gates.maximum_prepared_vs_analytic_relative_error
        ),
        causality_violation=(
            structural.causality_violation
            <= gates.maximum_causality_violation
        ),
        padding_violation=(
            structural.padding_violation <= gates.maximum_padding_violation
        ),
        repeat_relative_error=(
            structural.repeat_relative_error
            <= gates.maximum_repeat_relative_error
        ),
        collision_target_relative_difference=(
            collision_gate_deferred
            or minimum_collision
            >= gates.minimum_collision_target_relative_difference
        ),
        in_support_fraction=(
            structural.in_support_fraction
            >= gates.minimum_in_support_fraction
        ),
    )
    gates_sha256 = _gates_sha256(gates)
    return FullWidthCandidateScore(
        candidate_id=candidate.candidate_id,
        candidate_artifact_sha256=candidate.artifact_sha256,
        source_rank=candidate.source_rank,
        target_rank=candidate.target_rank,
        stored_scalar_count=candidate.stored_scalar_count,
        fisher_weighted_relative_error=relative_error,
        reference_cosine=reference_cosine,
        constant_control_relative_error=constant_error,
        position_only_control_relative_error=position_error,
        error_reduction_vs_constant=reduction_constant,
        error_reduction_vs_position_only=reduction_position,
        maximum_per_probe_p90_relative_error=maximum_probe_p90,
        worst_family_relative_error=worst_family,
        probe_metrics=probe_metrics,
        family_metrics=family_metrics,
        collision_metrics=collision_metrics,
        minimum_collision_target_relative_difference=minimum_collision,
        structural_metrics=structural,
        gate_flags=flags,
        passed=flags.all_passed,
        controls_artifact_sha256=controls.artifact_sha256,
        gates_sha256=gates_sha256,
    )


def score_full_width_reference_candidate(
    *,
    controls: FullWidthReferenceControls,
    selection_probes: Sequence[FullWidthReferenceProbe],
    collision_probes: Sequence[FullWidthReferenceProbe],
    candidate: FullWidthReferenceCandidate,
    gates: SyntheticReferenceGates,
) -> FullWidthCandidateScore:
    """Score one retained-rank provider against full 64-mode targets.

    An empty collision collection is accepted only when the supplied gate has
    a zero collision threshold.  That is the explicit compile-time
    representation of an assessment-only collision gate, so selection never
    needs to open sealed assessment targets.
    """

    if not isinstance(controls, FullWidthReferenceControls):
        raise TypeError("controls must be FullWidthReferenceControls")
    if not isinstance(candidate, FullWidthReferenceCandidate):
        raise TypeError("candidate must be FullWidthReferenceCandidate")
    if not isinstance(gates, SyntheticReferenceGates):
        raise TypeError("gates must be SyntheticReferenceGates")
    probes = _validate_probe_collection(
        selection_probes,
        split="selection",
        label="selection probes",
        nonempty=True,
    )
    collision_gate_deferred = not collision_probes
    if (
        collision_gate_deferred
        and gates.minimum_collision_target_relative_difference != 0.0
    ):
        raise ValueError(
            "empty collision probes require a zero deferred collision gate"
        )
    collisions = _validate_probe_collection(
        collision_probes,
        split="assessment",
        label="collision probes",
        nonempty=not collision_gate_deferred,
    )
    if set(controls.fit_probe_ids) & {probe.probe_id for probe in probes}:
        raise ValueError("fit and selection probe ids overlap")
    if {probe.probe_id for probe in probes} & {
        probe.probe_id for probe in collisions
    }:
        raise ValueError("selection and collision probe ids overlap")
    all_gauges = {
        controls.standardized_gauge_sha256,
        candidate.standardized_gauge_sha256,
        *(probe.standardized_gauge_sha256 for probe in probes),
        *(probe.standardized_gauge_sha256 for probe in collisions),
    }
    if len(all_gauges) != 1:
        raise ValueError("controls, probes, and candidate use multiple gauges")

    return _score_validated_full_width_reference_candidate(
        controls=controls,
        probes=probes,
        collision_probes=collisions,
        collision_gate_deferred=collision_gate_deferred,
        candidate=candidate,
        gates=gates,
    )


def score_full_width_reference_assessment(
    *,
    controls: FullWidthReferenceControls,
    assessment_probes: Sequence[FullWidthReferenceProbe],
    candidate: FullWidthReferenceCandidate,
    gates: SyntheticReferenceGates,
) -> FullWidthCandidateScore:
    """Score one frozen candidate on one complete sealed assessment panel.

    Every probe remains on the ``assessment`` split and participates in the
    fidelity metrics.  Collision metrics are derived from collision-tagged
    members of that same collection, so callers cannot duplicate or relabel
    assessment targets as a separate collision panel.  This function performs
    neither fitting nor candidate selection.
    """

    if not isinstance(controls, FullWidthReferenceControls):
        raise TypeError("controls must be FullWidthReferenceControls")
    if not isinstance(candidate, FullWidthReferenceCandidate):
        raise TypeError("candidate must be FullWidthReferenceCandidate")
    if not isinstance(gates, SyntheticReferenceGates):
        raise TypeError("gates must be SyntheticReferenceGates")
    probes = _validate_probe_collection(
        assessment_probes,
        split="assessment",
        label="assessment probes",
        nonempty=True,
    )
    probe_ids = {probe.probe_id for probe in probes}
    if set(controls.fit_probe_ids) & probe_ids:
        raise ValueError("fit and assessment probe ids overlap")

    collisions = tuple(
        probe for probe in probes if probe.collision_group is not None
    )
    collision_gate_deferred = not collisions
    if (
        collision_gate_deferred
        and gates.minimum_collision_target_relative_difference != 0.0
    ):
        raise ValueError(
            "assessment probes contain no collision-tagged rows for the "
            "nonzero collision gate"
        )

    all_gauges = {
        controls.standardized_gauge_sha256,
        candidate.standardized_gauge_sha256,
        *(probe.standardized_gauge_sha256 for probe in probes),
    }
    if len(all_gauges) != 1:
        raise ValueError("controls, probes, and candidate use multiple gauges")

    return _score_validated_full_width_reference_candidate(
        controls=controls,
        probes=probes,
        collision_probes=collisions,
        collision_gate_deferred=collision_gate_deferred,
        candidate=candidate,
        gates=gates,
    )


def _selection_key(
    score: FullWidthCandidateScore,
) -> tuple[int, int, int, str]:
    return (
        score.stored_scalar_count,
        score.source_rank,
        score.target_rank,
        score.candidate_id,
    )


def select_smallest_passing_full_width_reference_candidate(
    *,
    controls: FullWidthReferenceControls,
    selection_probes: Sequence[FullWidthReferenceProbe],
    collision_probes: Sequence[FullWidthReferenceProbe],
    candidates: Sequence[FullWidthReferenceCandidate],
    gates: SyntheticReferenceGates,
) -> FullWidthReferenceSelection:
    """Apply unrounded gates and choose the smallest passing candidate.

    Pass no collision probes and a zero collision threshold to defer that
    candidate-independent gate to a separately sealed assessment.
    """

    if isinstance(candidates, (str, bytes)) or not isinstance(
        candidates,
        Sequence,
    ):
        raise TypeError("candidates must be a sequence")
    candidate_values = tuple(candidates)
    if not candidate_values:
        raise ValueError("candidates must not be empty")
    if any(
        not isinstance(value, FullWidthReferenceCandidate)
        for value in candidate_values
    ):
        raise TypeError(
            "candidates must contain FullWidthReferenceCandidate values"
        )
    if len({value.candidate_id for value in candidate_values}) != len(
        candidate_values
    ):
        raise ValueError("candidate ids must be unique")

    probes = _validate_probe_collection(
        selection_probes,
        split="selection",
        label="selection probes",
        nonempty=True,
    )
    collision_gate_deferred = not collision_probes
    if (
        collision_gate_deferred
        and gates.minimum_collision_target_relative_difference != 0.0
    ):
        raise ValueError(
            "empty collision probes require a zero deferred collision gate"
        )
    collisions = _validate_probe_collection(
        collision_probes,
        split="assessment",
        label="collision probes",
        nonempty=not collision_gate_deferred,
    )
    scores = tuple(
        sorted(
            (
                score_full_width_reference_candidate(
                    controls=controls,
                    selection_probes=probes,
                    collision_probes=collisions,
                    candidate=candidate,
                    gates=gates,
                )
                for candidate in candidate_values
            ),
            key=lambda value: value.candidate_id,
        )
    )
    passers = tuple(value for value in scores if value.passed)
    selected = min(passers, key=_selection_key) if passers else None
    return FullWidthReferenceSelection(
        selected_candidate_id=(
            selected.candidate_id if selected is not None else None
        ),
        selected_candidate_artifact_sha256=(
            selected.candidate_artifact_sha256
            if selected is not None
            else None
        ),
        selected_stored_scalar_count=(
            selected.stored_scalar_count if selected is not None else None
        ),
        selected_source_rank=(
            selected.source_rank if selected is not None else None
        ),
        selected_target_rank=(
            selected.target_rank if selected is not None else None
        ),
        candidate_scores=scores,
        controls_artifact_sha256=controls.artifact_sha256,
        selection_probe_sha256s=tuple(
            sorted(probe.artifact_sha256 for probe in probes)
        ),
        collision_probe_sha256s=tuple(
            sorted(probe.artifact_sha256 for probe in collisions)
        ),
        gates_sha256=_gates_sha256(gates),
    )
