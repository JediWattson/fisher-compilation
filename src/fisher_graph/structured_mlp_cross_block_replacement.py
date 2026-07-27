"""Fit-only scalar replacement evidence for cross-block MLP coordinates.

Discovery correlation is not enough to delete a native coordinate.  This
module supplies the next, deliberately narrower boundary:

* fit a no-intercept scalar map from an earlier native coordinate to a later
  native coordinate;
* report ordinary and consumer-Fisher-weighted candidates;
* replay each candidate with family-disjoint leave-one-fold-out fits; and
* aggregate a paired native/ablation/replacement/shuffled intervention oracle.

The implementation is model agnostic.  Callers provide detached CPU rows for
the scalar fit and already-computed condition metric sums for the oracle.  No
model parameter is accepted or updated.  Both resulting artifacts are
authenticated, omit corpus rows, and fail closed: fit-side evidence never
authorizes an intervention, compilation, execution, held-out guard, or
calibration-B evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import struct

import torch
from torch import Tensor

from .structured_mlp_cross_block_bundling import ModeKey


_FIT_KIND = "fisher_graph.cross_block_scalar_replacement_fit"
_ORACLE_KIND = "fisher_graph.cross_block_scalar_replacement_fit_oracle"
_FORMAT_VERSION = 1
_FIT_HASH_DOMAIN = b"fisher_graph.cross_block_replacement_fit.v1\0"
_ORACLE_HASH_DOMAIN = b"fisher_graph.cross_block_replacement_oracle.v1\0"
_ROW_HASH_DOMAIN = b"fisher_graph.cross_block_replacement_rows.v1\0"
_METRIC_HASH_DOMAIN = b"fisher_graph.cross_block_replacement_metrics.v1\0"
_TENSOR_HASH_DOMAIN = b"fisher_graph.cross_block_replacement_tensor.v1\0"
_SCALE_KINDS = (
    "unweighted_no_intercept",
    "consumer_fisher_weighted_no_intercept",
)
REPLACEMENT_CONDITIONS = (
    "native",
    "ablation",
    "replacement",
    "shuffled",
)


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value  # type: ignore[return-value]


def _finite_float(value: object, *, label: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
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
        or value.ndim != 1
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{label} must be a finite CPU float64 vector")
    canonical = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_HASH_DOMAIN)
    digest.update(struct.pack("<Q", canonical.numel()))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _float_close(first: float, second: float) -> bool:
    scale = max(abs(first), abs(second), 1.0)
    return math.isclose(first, second, rel_tol=2e-11, abs_tol=2e-12 * scale)


def _optional_float_close(
    first: float | None,
    second: float | None,
) -> bool:
    if first is None or second is None:
        return first is second
    return _float_close(first, second)


@dataclass(frozen=True, slots=True)
class CrossBlockReplacementProvenance:
    """Fit-only bindings supplied by a frozen-model caller."""

    model_fingerprint: str
    fit_split_sha256: str
    objective_sha256: str
    proposal_artifact_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("model_fingerprint", self.model_fingerprint),
            ("fit_split_sha256", self.fit_split_sha256),
            ("objective_sha256", self.objective_sha256),
            ("proposal_artifact_sha256", self.proposal_artifact_sha256),
        ):
            _require_sha256(value, label=label)

    def metadata(self) -> dict[str, object]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "fit_split_sha256": self.fit_split_sha256,
            "objective_sha256": self.objective_sha256,
            "proposal_artifact_sha256": self.proposal_artifact_sha256,
            "data_scope": "fit_only",
            "model_parameter_updates": 0,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockReplacementProvenance:
        expected = {
            "model_fingerprint",
            "fit_split_sha256",
            "objective_sha256",
            "proposal_artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("replacement provenance fields are invalid")
        return cls(
            model_fingerprint=str(state["model_fingerprint"]),
            fit_split_sha256=str(state["fit_split_sha256"]),
            objective_sha256=str(state["objective_sha256"]),
            proposal_artifact_sha256=str(
                state["proposal_artifact_sha256"]
            ),
        )


@dataclass(frozen=True, slots=True)
class CrossBlockReplacementFitRows:
    """Detached native scalar rows for one independent fit sequence.

    ``consumer_score_gradients.square()`` supplies the diagonal empirical
    Fisher weights.  Logical positions bind anchor and consumer values to the
    same token rows.
    """

    example_id: str
    family_id: str
    logical_positions: Tensor
    anchor_values: Tensor
    consumer_values: Tensor
    consumer_score_gradients: Tensor

    def __post_init__(self) -> None:
        _require_nonempty(self.example_id, label="example_id")
        _require_nonempty(self.family_id, label="family_id")
        for label, value in (
            ("anchor_values", self.anchor_values),
            ("consumer_values", self.consumer_values),
            ("consumer_score_gradients", self.consumer_score_gradients),
        ):
            _tensor_sha256(value, label=label)
        observations = self.anchor_values.numel()
        if observations <= 0 or any(
            value.shape != (observations,)
            for value in (
                self.consumer_values,
                self.consumer_score_gradients,
            )
        ):
            raise ValueError(
                "replacement scalar vectors must share a positive length"
            )
        if (
            not isinstance(self.logical_positions, Tensor)
            or self.logical_positions.device.type != "cpu"
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
            or self.logical_positions.shape != (observations,)
            or (self.logical_positions < 0).any()
            or (
                observations > 1
                and not torch.all(
                    self.logical_positions[1:]
                    > self.logical_positions[:-1]
                )
            )
        ):
            raise ValueError(
                "logical_positions must be a strictly increasing "
                "nonnegative CPU integer vector"
            )
        for label in (
            "logical_positions",
            "anchor_values",
            "consumer_values",
            "consumer_score_gradients",
        ):
            value = getattr(self, label)
            object.__setattr__(self, label, value.detach().clone())

    @property
    def observations(self) -> int:
        return self.anchor_values.numel()


@dataclass(frozen=True, slots=True)
class CrossBlockFoldScaleFit:
    """Leave-one-family-fold-out scale and its held-out residual."""

    fold_index: int
    train_sequences: int
    train_observations: int
    holdout_sequences: int
    holdout_observations: int
    scale: float
    holdout_residual_square_sum: float
    holdout_target_square_sum: float
    holdout_residual_nrmse: float | None

    def __post_init__(self) -> None:
        if type(self.fold_index) is not int or self.fold_index < 0:
            raise ValueError("fold_index must be nonnegative")
        for label, value in (
            ("train_sequences", self.train_sequences),
            ("train_observations", self.train_observations),
            ("holdout_sequences", self.holdout_sequences),
            ("holdout_observations", self.holdout_observations),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        for label, value in (
            ("scale", self.scale),
            (
                "holdout_residual_square_sum",
                self.holdout_residual_square_sum,
            ),
            ("holdout_target_square_sum", self.holdout_target_square_sum),
        ):
            _finite_float(value, label=label)
        if (
            self.holdout_residual_square_sum < 0.0
            or self.holdout_target_square_sum < 0.0
        ):
            raise ValueError("fold square sums must be nonnegative")
        expected_nrmse = _normalized_residual(
            self.holdout_residual_square_sum,
            self.holdout_target_square_sum,
        )
        if not _optional_float_close(
            self.holdout_residual_nrmse,
            expected_nrmse,
        ):
            raise ValueError("fold residual NRMSE is inconsistent")

    def metadata(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "train_sequences": self.train_sequences,
            "train_observations": self.train_observations,
            "holdout_sequences": self.holdout_sequences,
            "holdout_observations": self.holdout_observations,
            "scale": self.scale,
            "holdout_residual_square_sum": (
                self.holdout_residual_square_sum
            ),
            "holdout_target_square_sum": self.holdout_target_square_sum,
            "holdout_residual_nrmse": self.holdout_residual_nrmse,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockFoldScaleFit:
        expected = {
            "fold_index",
            "train_sequences",
            "train_observations",
            "holdout_sequences",
            "holdout_observations",
            "scale",
            "holdout_residual_square_sum",
            "holdout_target_square_sum",
            "holdout_residual_nrmse",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("fold scale state fields are invalid")
        nrmse = state["holdout_residual_nrmse"]
        return cls(
            fold_index=int(state["fold_index"]),
            train_sequences=int(state["train_sequences"]),
            train_observations=int(state["train_observations"]),
            holdout_sequences=int(state["holdout_sequences"]),
            holdout_observations=int(state["holdout_observations"]),
            scale=float(state["scale"]),
            holdout_residual_square_sum=float(
                state["holdout_residual_square_sum"]
            ),
            holdout_target_square_sum=float(
                state["holdout_target_square_sum"]
            ),
            holdout_residual_nrmse=(
                None if nrmse is None else float(nrmse)
            ),
        )


def _normalized_residual(
    residual_square_sum: float,
    target_square_sum: float,
) -> float | None:
    if target_square_sum == 0.0:
        return 0.0 if residual_square_sum == 0.0 else None
    return math.sqrt(max(residual_square_sum, 0.0) / target_square_sum)


@dataclass(frozen=True, slots=True)
class CrossBlockScaleCandidate:
    """One closed-form, no-intercept scalar replacement candidate."""

    kind: str
    scale: float
    numerator: float
    denominator: float
    target_square_sum: float
    residual_square_sum: float
    residual_nrmse: float | None
    leave_one_fold_out: tuple[CrossBlockFoldScaleFit, ...]
    maximum_relative_scale_deviation: float
    fold_sign_stable: bool

    def __post_init__(self) -> None:
        if self.kind not in _SCALE_KINDS:
            raise ValueError("unknown replacement scale kind")
        for label, value in (
            ("scale", self.scale),
            ("numerator", self.numerator),
            ("denominator", self.denominator),
            ("target_square_sum", self.target_square_sum),
            ("residual_square_sum", self.residual_square_sum),
            (
                "maximum_relative_scale_deviation",
                self.maximum_relative_scale_deviation,
            ),
        ):
            _finite_float(value, label=label)
        if (
            self.denominator <= 0.0
            or self.target_square_sum < 0.0
            or self.residual_square_sum < 0.0
            or self.maximum_relative_scale_deviation < 0.0
        ):
            raise ValueError("replacement scale sums are invalid")
        if not _float_close(self.scale, self.numerator / self.denominator):
            raise ValueError("replacement scale is not the closed-form fit")
        expected_residual = max(
            self.target_square_sum
            - 2.0 * self.scale * self.numerator
            + self.scale * self.scale * self.denominator,
            0.0,
        )
        if not _float_close(self.residual_square_sum, expected_residual):
            raise ValueError("replacement residual is inconsistent")
        if not _optional_float_close(
            self.residual_nrmse,
            _normalized_residual(
                self.residual_square_sum,
                self.target_square_sum,
            ),
        ):
            raise ValueError("replacement residual NRMSE is inconsistent")
        if (
            type(self.leave_one_fold_out) is not tuple
            or len(self.leave_one_fold_out) < 2
            or any(
                not isinstance(value, CrossBlockFoldScaleFit)
                for value in self.leave_one_fold_out
            )
            or tuple(
                value.fold_index for value in self.leave_one_fold_out
            )
            != tuple(range(len(self.leave_one_fold_out)))
        ):
            raise ValueError("leave-one-fold-out fits are invalid")
        expected_deviation = max(
            abs(value.scale - self.scale)
            / max(abs(self.scale), 1e-12)
            for value in self.leave_one_fold_out
        )
        if not _float_close(
            self.maximum_relative_scale_deviation,
            expected_deviation,
        ):
            raise ValueError("fold scale deviation is inconsistent")
        if type(self.fold_sign_stable) is not bool:
            raise TypeError("fold_sign_stable must be boolean")
        expected_sign_stable = all(
            value.scale * self.scale >= 0.0
            for value in self.leave_one_fold_out
        )
        if self.fold_sign_stable != expected_sign_stable:
            raise ValueError("fold sign stability is inconsistent")

    def metadata(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "fit": "closed_form_no_intercept",
            "weighting": (
                "none"
                if self.kind == "unweighted_no_intercept"
                else "consumer_score_gradient_squared"
            ),
            "scale": self.scale,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "target_square_sum": self.target_square_sum,
            "residual_square_sum": self.residual_square_sum,
            "residual_nrmse": self.residual_nrmse,
            "leave_one_fold_out": tuple(
                value.metadata() for value in self.leave_one_fold_out
            ),
            "maximum_relative_scale_deviation": (
                self.maximum_relative_scale_deviation
            ),
            "fold_sign_stable": self.fold_sign_stable,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "scale": self.scale,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "target_square_sum": self.target_square_sum,
            "residual_square_sum": self.residual_square_sum,
            "residual_nrmse": self.residual_nrmse,
            "leave_one_fold_out": tuple(
                value.metadata() for value in self.leave_one_fold_out
            ),
            "maximum_relative_scale_deviation": (
                self.maximum_relative_scale_deviation
            ),
            "fold_sign_stable": self.fold_sign_stable,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockScaleCandidate:
        expected = {
            "kind",
            "scale",
            "numerator",
            "denominator",
            "target_square_sum",
            "residual_square_sum",
            "residual_nrmse",
            "leave_one_fold_out",
            "maximum_relative_scale_deviation",
            "fold_sign_stable",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("replacement scale state fields are invalid")
        folds = state["leave_one_fold_out"]
        if not isinstance(folds, tuple):
            raise TypeError("leave_one_fold_out must be a tuple")
        residual_nrmse = state["residual_nrmse"]
        return cls(
            kind=str(state["kind"]),
            scale=float(state["scale"]),
            numerator=float(state["numerator"]),
            denominator=float(state["denominator"]),
            target_square_sum=float(state["target_square_sum"]),
            residual_square_sum=float(state["residual_square_sum"]),
            residual_nrmse=(
                None
                if residual_nrmse is None
                else float(residual_nrmse)
            ),
            leave_one_fold_out=tuple(
                CrossBlockFoldScaleFit.from_state_dict(value)
                for value in folds
            ),
            maximum_relative_scale_deviation=float(
                state["maximum_relative_scale_deviation"]
            ),
            fold_sign_stable=bool(state["fold_sign_stable"]),
        )


@dataclass(frozen=True, slots=True)
class _ScaleSums:
    numerator: float = 0.0
    denominator: float = 0.0
    target_square_sum: float = 0.0
    sequences: int = 0
    observations: int = 0

    def plus(self, other: _ScaleSums) -> _ScaleSums:
        return _ScaleSums(
            numerator=self.numerator + other.numerator,
            denominator=self.denominator + other.denominator,
            target_square_sum=(
                self.target_square_sum + other.target_square_sum
            ),
            sequences=self.sequences + other.sequences,
            observations=self.observations + other.observations,
        )

    def minus(self, other: _ScaleSums) -> _ScaleSums:
        return _ScaleSums(
            numerator=self.numerator - other.numerator,
            denominator=self.denominator - other.denominator,
            target_square_sum=(
                self.target_square_sum - other.target_square_sum
            ),
            sequences=self.sequences - other.sequences,
            observations=self.observations - other.observations,
        )


def _row_scale_sums(
    row: CrossBlockReplacementFitRows,
    *,
    fisher_weighted: bool,
) -> _ScaleSums:
    anchor = row.anchor_values
    consumer = row.consumer_values
    weight = (
        row.consumer_score_gradients.square()
        if fisher_weighted
        else torch.ones_like(anchor)
    )
    return _ScaleSums(
        numerator=float((weight * anchor * consumer).sum().item()),
        denominator=float((weight * anchor.square()).sum().item()),
        target_square_sum=float((weight * consumer.square()).sum().item()),
        sequences=1,
        observations=row.observations,
    )


def _residual_square_sum(sums: _ScaleSums, scale: float) -> float:
    value = (
        sums.target_square_sum
        - 2.0 * scale * sums.numerator
        + scale * scale * sums.denominator
    )
    tolerance = 2e-12 * max(
        sums.target_square_sum,
        abs(scale * sums.numerator),
        abs(scale * scale * sums.denominator),
        1.0,
    )
    if value < -tolerance:
        raise ValueError("replacement residual became materially negative")
    return max(value, 0.0)


def _scale_candidate(
    *,
    kind: str,
    total: _ScaleSums,
    folds: tuple[_ScaleSums, ...],
) -> CrossBlockScaleCandidate:
    if total.denominator <= 0.0:
        raise ValueError(f"{kind} scale denominator must be positive")
    scale = total.numerator / total.denominator
    residual = _residual_square_sum(total, scale)
    held_out: list[CrossBlockFoldScaleFit] = []
    for fold_index, holdout in enumerate(folds):
        train = total.minus(holdout)
        if (
            train.denominator <= 0.0
            or train.sequences <= 0
            or train.observations <= 0
            or holdout.sequences <= 0
            or holdout.observations <= 0
        ):
            raise ValueError(
                "every family fold must leave positive train and holdout "
                f"support for {kind}"
            )
        fold_scale = train.numerator / train.denominator
        holdout_residual = _residual_square_sum(holdout, fold_scale)
        held_out.append(
            CrossBlockFoldScaleFit(
                fold_index=fold_index,
                train_sequences=train.sequences,
                train_observations=train.observations,
                holdout_sequences=holdout.sequences,
                holdout_observations=holdout.observations,
                scale=float(fold_scale),
                holdout_residual_square_sum=float(holdout_residual),
                holdout_target_square_sum=float(
                    holdout.target_square_sum
                ),
                holdout_residual_nrmse=_normalized_residual(
                    holdout_residual,
                    holdout.target_square_sum,
                ),
            )
        )
    maximum_deviation = max(
        abs(value.scale - scale) / max(abs(scale), 1e-12)
        for value in held_out
    )
    return CrossBlockScaleCandidate(
        kind=kind,
        scale=float(scale),
        numerator=float(total.numerator),
        denominator=float(total.denominator),
        target_square_sum=float(total.target_square_sum),
        residual_square_sum=float(residual),
        residual_nrmse=_normalized_residual(
            residual,
            total.target_square_sum,
        ),
        leave_one_fold_out=tuple(held_out),
        maximum_relative_scale_deviation=float(maximum_deviation),
        fold_sign_stable=all(
            value.scale * scale >= 0.0 for value in held_out
        ),
    )


def _fit_row_digest(
    rows: tuple[CrossBlockReplacementFitRows, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(_ROW_HASH_DOMAIN)
    digest.update(struct.pack("<Q", len(rows)))
    for row in rows:
        for text in (row.example_id, row.family_id):
            encoded = text.encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
        positions = row.logical_positions.to(
            dtype=torch.int64,
        ).contiguous()
        digest.update(struct.pack("<Q", positions.numel()))
        digest.update(positions.numpy().tobytes(order="C"))
        for value in (
            row.anchor_values,
            row.consumer_values,
            row.consumer_score_gradients,
        ):
            digest.update(
                bytes.fromhex(_tensor_sha256(value, label="fit row"))
            )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CrossBlockScalarReplacementEvidence:
    """Authenticated fit-side scale evidence for one native coordinate pair."""

    provenance: CrossBlockReplacementProvenance
    anchor: ModeKey
    consumer: ModeKey
    sequences: int
    observations: int
    families: int
    fold_count: int
    fold_family_counts: tuple[int, ...]
    fold_sequence_counts: tuple[int, ...]
    family_fold_assignment_sha256: str
    fit_row_stream_sha256: str
    unweighted: CrossBlockScaleCandidate
    fisher_weighted: CrossBlockScaleCandidate
    proposed_scale_kind: str
    proposed_scale: float
    artifact_sha256: str
    artifact_kind: str = _FIT_KIND
    format_version: int = _FORMAT_VERSION
    contains_corpus_rows: bool = False
    fit_only: bool = True
    model_parameter_updates: int = 0
    authorizes_intervention: bool = False
    authorizes_compilation: bool = False
    authorizes_execution: bool = False
    authorizes_guard: bool = False
    authorizes_b: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, CrossBlockReplacementProvenance):
            raise TypeError("replacement provenance is invalid")
        if not isinstance(self.anchor, ModeKey) or not isinstance(
            self.consumer,
            ModeKey,
        ):
            raise TypeError("replacement endpoints must be ModeKey values")
        if self.anchor.layer_ordinal >= self.consumer.layer_ordinal:
            raise ValueError("replacement must point strictly forward")
        if (
            type(self.sequences) is not int
            or self.sequences < 2
            or type(self.observations) is not int
            or self.observations < self.sequences
            or type(self.families) is not int
            or self.families < 2
            or type(self.fold_count) is not int
            or self.fold_count < 2
        ):
            raise ValueError("replacement fit counts are invalid")
        if (
            type(self.fold_family_counts) is not tuple
            or type(self.fold_sequence_counts) is not tuple
            or len(self.fold_family_counts) != self.fold_count
            or len(self.fold_sequence_counts) != self.fold_count
            or any(value <= 0 for value in self.fold_family_counts)
            or any(value <= 0 for value in self.fold_sequence_counts)
            or sum(self.fold_family_counts) != self.families
            or sum(self.fold_sequence_counts) != self.sequences
        ):
            raise ValueError("replacement fold counts are invalid")
        _require_sha256(
            self.family_fold_assignment_sha256,
            label="family_fold_assignment_sha256",
        )
        _require_sha256(
            self.fit_row_stream_sha256,
            label="fit_row_stream_sha256",
        )
        if (
            not isinstance(self.unweighted, CrossBlockScaleCandidate)
            or not isinstance(
                self.fisher_weighted,
                CrossBlockScaleCandidate,
            )
            or self.unweighted.kind != _SCALE_KINDS[0]
            or self.fisher_weighted.kind != _SCALE_KINDS[1]
        ):
            raise ValueError("replacement scale candidates are invalid")
        for candidate in (self.unweighted, self.fisher_weighted):
            if len(candidate.leave_one_fold_out) != self.fold_count:
                raise ValueError(
                    "replacement candidate folds do not match fold_count"
                )
            for fold, expected_sequences in zip(
                candidate.leave_one_fold_out,
                self.fold_sequence_counts,
                strict=True,
            ):
                if (
                    fold.holdout_sequences != expected_sequences
                    or fold.train_sequences + fold.holdout_sequences
                    != self.sequences
                    or fold.train_observations
                    + fold.holdout_observations
                    != self.observations
                ):
                    raise ValueError(
                        "replacement candidate fold support is inconsistent"
                    )
        if self.proposed_scale_kind != self.fisher_weighted.kind:
            raise ValueError(
                "the proposed fit-side scale must be Fisher weighted"
            )
        _finite_float(self.proposed_scale, label="proposed_scale")
        if not _float_close(
            self.proposed_scale,
            self.fisher_weighted.scale,
        ):
            raise ValueError("proposed scale does not match its candidate")
        if (
            self.artifact_kind != _FIT_KIND
            or self.format_version != _FORMAT_VERSION
            or self.contains_corpus_rows
            or not self.fit_only
            or self.model_parameter_updates != 0
            or self.authorizes_intervention
            or self.authorizes_compilation
            or self.authorizes_execution
            or self.authorizes_guard
            or self.authorizes_b
        ):
            raise ValueError("replacement fit safety metadata is invalid")
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("replacement fit artifact hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "contains_corpus_rows": self.contains_corpus_rows,
            "fit_only": self.fit_only,
            "model_parameter_updates": self.model_parameter_updates,
            "authorizes_intervention": self.authorizes_intervention,
            "authorizes_compilation": self.authorizes_compilation,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_guard": self.authorizes_guard,
            "authorizes_b": self.authorizes_b,
            "provenance": self.provenance.metadata(),
            "anchor": self.anchor.metadata(),
            "consumer": self.consumer.metadata(),
            "sequences": self.sequences,
            "observations": self.observations,
            "families": self.families,
            "fold_count": self.fold_count,
            "fold_family_counts": self.fold_family_counts,
            "fold_sequence_counts": self.fold_sequence_counts,
            "family_fold_assignment_sha256": (
                self.family_fold_assignment_sha256
            ),
            "fit_row_stream_sha256": self.fit_row_stream_sha256,
            "unweighted": self.unweighted.metadata(),
            "fisher_weighted": self.fisher_weighted.metadata(),
            "proposed_scale_kind": self.proposed_scale_kind,
            "proposed_scale": self.proposed_scale,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_FIT_HASH_DOMAIN)

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "contains_corpus_rows": self.contains_corpus_rows,
            "fit_only": self.fit_only,
            "model_parameter_updates": self.model_parameter_updates,
            "authorizes_intervention": self.authorizes_intervention,
            "authorizes_compilation": self.authorizes_compilation,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_guard": self.authorizes_guard,
            "authorizes_b": self.authorizes_b,
            "provenance": {
                key: value
                for key, value in self.provenance.metadata().items()
                if key
                in {
                    "model_fingerprint",
                    "fit_split_sha256",
                    "objective_sha256",
                    "proposal_artifact_sha256",
                }
            },
            "anchor": self.anchor.metadata(),
            "consumer": self.consumer.metadata(),
            "sequences": self.sequences,
            "observations": self.observations,
            "families": self.families,
            "fold_count": self.fold_count,
            "fold_family_counts": self.fold_family_counts,
            "fold_sequence_counts": self.fold_sequence_counts,
            "family_fold_assignment_sha256": (
                self.family_fold_assignment_sha256
            ),
            "fit_row_stream_sha256": self.fit_row_stream_sha256,
            "unweighted": self.unweighted.state_dict(),
            "fisher_weighted": self.fisher_weighted.state_dict(),
            "proposed_scale_kind": self.proposed_scale_kind,
            "proposed_scale": self.proposed_scale,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockScalarReplacementEvidence:
        expected = {
            "artifact_kind",
            "format_version",
            "contains_corpus_rows",
            "fit_only",
            "model_parameter_updates",
            "authorizes_intervention",
            "authorizes_compilation",
            "authorizes_execution",
            "authorizes_guard",
            "authorizes_b",
            "provenance",
            "anchor",
            "consumer",
            "sequences",
            "observations",
            "families",
            "fold_count",
            "fold_family_counts",
            "fold_sequence_counts",
            "family_fold_assignment_sha256",
            "fit_row_stream_sha256",
            "unweighted",
            "fisher_weighted",
            "proposed_scale_kind",
            "proposed_scale",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("replacement fit state fields are invalid")
        for label in (
            "provenance",
            "anchor",
            "consumer",
            "unweighted",
            "fisher_weighted",
        ):
            if not isinstance(state[label], Mapping):
                raise TypeError(f"{label} state must be a mapping")
        if not isinstance(state["fold_family_counts"], tuple) or not isinstance(
            state["fold_sequence_counts"],
            tuple,
        ):
            raise TypeError("replacement fold counts must be tuples")
        return cls(
            provenance=CrossBlockReplacementProvenance.from_state_dict(
                state["provenance"]
            ),
            anchor=ModeKey.from_state_dict(state["anchor"]),
            consumer=ModeKey.from_state_dict(state["consumer"]),
            sequences=int(state["sequences"]),
            observations=int(state["observations"]),
            families=int(state["families"]),
            fold_count=int(state["fold_count"]),
            fold_family_counts=tuple(
                int(value) for value in state["fold_family_counts"]
            ),
            fold_sequence_counts=tuple(
                int(value) for value in state["fold_sequence_counts"]
            ),
            family_fold_assignment_sha256=str(
                state["family_fold_assignment_sha256"]
            ),
            fit_row_stream_sha256=str(state["fit_row_stream_sha256"]),
            unweighted=CrossBlockScaleCandidate.from_state_dict(
                state["unweighted"]
            ),
            fisher_weighted=CrossBlockScaleCandidate.from_state_dict(
                state["fisher_weighted"]
            ),
            proposed_scale_kind=str(state["proposed_scale_kind"]),
            proposed_scale=float(state["proposed_scale"]),
            artifact_sha256=str(state["artifact_sha256"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
            contains_corpus_rows=bool(state["contains_corpus_rows"]),
            fit_only=bool(state["fit_only"]),
            model_parameter_updates=int(state["model_parameter_updates"]),
            authorizes_intervention=bool(
                state["authorizes_intervention"]
            ),
            authorizes_compilation=bool(state["authorizes_compilation"]),
            authorizes_execution=bool(state["authorizes_execution"]),
            authorizes_guard=bool(state["authorizes_guard"]),
            authorizes_b=bool(state["authorizes_b"]),
        )


def fit_cross_block_scalar_replacement(
    rows: Iterable[CrossBlockReplacementFitRows],
    *,
    provenance: CrossBlockReplacementProvenance,
    anchor: ModeKey,
    consumer: ModeKey,
    family_fold_assignment: Mapping[str, int],
    fold_count: int,
) -> CrossBlockScalarReplacementEvidence:
    """Fit ordinary and consumer-Fisher-weighted scalar carry candidates.

    Fold assignment is keyed by family rather than example, making family
    leakage impossible inside this function.  Each reported fold scale is fit
    on every other fold and evaluated only on the excluded fold.
    """

    if not isinstance(provenance, CrossBlockReplacementProvenance):
        raise TypeError("provenance is invalid")
    if not isinstance(anchor, ModeKey) or not isinstance(consumer, ModeKey):
        raise TypeError("anchor and consumer must be ModeKey values")
    if anchor.layer_ordinal >= consumer.layer_ordinal:
        raise ValueError("replacement must point strictly forward")
    if type(fold_count) is not int or fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if not isinstance(family_fold_assignment, Mapping):
        raise TypeError("family_fold_assignment must be a mapping")

    materialized = tuple(rows)
    if (
        len(materialized) < 2
        or any(
            not isinstance(value, CrossBlockReplacementFitRows)
            for value in materialized
        )
    ):
        raise ValueError("at least two replacement fit rows are required")
    materialized = tuple(
        sorted(materialized, key=lambda value: value.example_id)
    )
    example_ids = tuple(value.example_id for value in materialized)
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("replacement fit example_id values must be unique")
    families = tuple(sorted({value.family_id for value in materialized}))
    if set(family_fold_assignment) != set(families):
        raise ValueError(
            "family_fold_assignment must cover exactly the fit families"
        )
    normalized_assignment: dict[str, int] = {}
    for family in families:
        fold = family_fold_assignment[family]
        if type(fold) is not int or not 0 <= fold < fold_count:
            raise ValueError("family fold indices are out of range")
        normalized_assignment[family] = fold
    if set(normalized_assignment.values()) != set(range(fold_count)):
        raise ValueError("every family fold must be represented")

    all_sums = {
        kind: _ScaleSums()
        for kind in _SCALE_KINDS
    }
    fold_sums = {
        kind: [_ScaleSums() for _ in range(fold_count)]
        for kind in _SCALE_KINDS
    }
    fold_sequence_counts = [0] * fold_count
    for row in materialized:
        fold = normalized_assignment[row.family_id]
        fold_sequence_counts[fold] += 1
        for kind, fisher_weighted in (
            (_SCALE_KINDS[0], False),
            (_SCALE_KINDS[1], True),
        ):
            sums = _row_scale_sums(
                row,
                fisher_weighted=fisher_weighted,
            )
            all_sums[kind] = all_sums[kind].plus(sums)
            fold_sums[kind][fold] = fold_sums[kind][fold].plus(sums)

    unweighted = _scale_candidate(
        kind=_SCALE_KINDS[0],
        total=all_sums[_SCALE_KINDS[0]],
        folds=tuple(fold_sums[_SCALE_KINDS[0]]),
    )
    fisher_weighted = _scale_candidate(
        kind=_SCALE_KINDS[1],
        total=all_sums[_SCALE_KINDS[1]],
        folds=tuple(fold_sums[_SCALE_KINDS[1]]),
    )
    fold_family_counts = tuple(
        sum(value == fold for value in normalized_assignment.values())
        for fold in range(fold_count)
    )
    assignment_hash = _json_sha256(
        tuple(sorted(normalized_assignment.items())),
        domain=_FIT_HASH_DOMAIN,
    )
    payload = {
        "artifact_kind": _FIT_KIND,
        "format_version": _FORMAT_VERSION,
        "contains_corpus_rows": False,
        "fit_only": True,
        "model_parameter_updates": 0,
        "authorizes_intervention": False,
        "authorizes_compilation": False,
        "authorizes_execution": False,
        "authorizes_guard": False,
        "authorizes_b": False,
        "provenance": provenance.metadata(),
        "anchor": anchor.metadata(),
        "consumer": consumer.metadata(),
        "sequences": len(materialized),
        "observations": sum(value.observations for value in materialized),
        "families": len(families),
        "fold_count": fold_count,
        "fold_family_counts": fold_family_counts,
        "fold_sequence_counts": tuple(fold_sequence_counts),
        "family_fold_assignment_sha256": assignment_hash,
        "fit_row_stream_sha256": _fit_row_digest(materialized),
        "unweighted": unweighted.metadata(),
        "fisher_weighted": fisher_weighted.metadata(),
        "proposed_scale_kind": fisher_weighted.kind,
        "proposed_scale": fisher_weighted.scale,
    }
    return CrossBlockScalarReplacementEvidence(
        provenance=provenance,
        anchor=anchor,
        consumer=consumer,
        sequences=len(materialized),
        observations=sum(value.observations for value in materialized),
        families=len(families),
        fold_count=fold_count,
        fold_family_counts=fold_family_counts,
        fold_sequence_counts=tuple(fold_sequence_counts),
        family_fold_assignment_sha256=assignment_hash,
        fit_row_stream_sha256=_fit_row_digest(materialized),
        unweighted=unweighted,
        fisher_weighted=fisher_weighted,
        proposed_scale_kind=fisher_weighted.kind,
        proposed_scale=fisher_weighted.scale,
        artifact_sha256=_json_sha256(
            payload,
            domain=_FIT_HASH_DOMAIN,
        ),
    )


@dataclass(frozen=True, slots=True)
class CrossBlockReplacementConditionMetric:
    """Already-evaluated metric sums for one paired oracle condition."""

    example_id: str
    family_id: str
    condition: str
    supervised_tokens: int
    summed_nll: float
    teacher_kl_sum_to_native: float
    top1_matches_to_native: int

    def __post_init__(self) -> None:
        _require_nonempty(self.example_id, label="example_id")
        _require_nonempty(self.family_id, label="family_id")
        if self.condition not in REPLACEMENT_CONDITIONS:
            raise ValueError("unknown replacement oracle condition")
        if (
            type(self.supervised_tokens) is not int
            or self.supervised_tokens <= 0
            or type(self.top1_matches_to_native) is not int
            or not 0
            <= self.top1_matches_to_native
            <= self.supervised_tokens
        ):
            raise ValueError("replacement condition token counts are invalid")
        for label, value in (
            ("summed_nll", self.summed_nll),
            ("teacher_kl_sum_to_native", self.teacher_kl_sum_to_native),
        ):
            _finite_float(value, label=label)
        if self.summed_nll < 0.0 or self.teacher_kl_sum_to_native < 0.0:
            raise ValueError("NLL and KL sums must be nonnegative")
        if self.condition == "native" and (
            self.teacher_kl_sum_to_native != 0.0
            or self.top1_matches_to_native != self.supervised_tokens
        ):
            raise ValueError(
                "native condition must be an exact self-comparison"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "condition": self.condition,
            "supervised_tokens": self.supervised_tokens,
            "summed_nll": self.summed_nll,
            "teacher_kl_sum_to_native": self.teacher_kl_sum_to_native,
            "top1_matches_to_native": self.top1_matches_to_native,
        }


@dataclass(frozen=True, slots=True)
class CrossBlockReplacementConditionAggregate:
    """Token-weighted aggregate for one intervention condition."""

    condition: str
    sequences: int
    supervised_tokens: int
    summed_nll: float
    nll_per_token: float
    delta_summed_nll_to_native: float
    delta_nll_per_token_to_native: float
    absolute_delta_summed_nll_to_native: float
    absolute_delta_nll_per_token_to_native: float
    teacher_kl_sum_to_native: float
    teacher_kl_per_token_to_native: float
    top1_matches_to_native: int
    top1_agreement_to_native: float

    def __post_init__(self) -> None:
        if self.condition not in REPLACEMENT_CONDITIONS:
            raise ValueError("unknown replacement aggregate condition")
        if (
            type(self.sequences) is not int
            or self.sequences <= 0
            or type(self.supervised_tokens) is not int
            or self.supervised_tokens < self.sequences
            or type(self.top1_matches_to_native) is not int
            or not 0
            <= self.top1_matches_to_native
            <= self.supervised_tokens
        ):
            raise ValueError("replacement aggregate counts are invalid")
        for label, value in (
            ("summed_nll", self.summed_nll),
            ("nll_per_token", self.nll_per_token),
            (
                "delta_summed_nll_to_native",
                self.delta_summed_nll_to_native,
            ),
            (
                "delta_nll_per_token_to_native",
                self.delta_nll_per_token_to_native,
            ),
            (
                "absolute_delta_summed_nll_to_native",
                self.absolute_delta_summed_nll_to_native,
            ),
            (
                "absolute_delta_nll_per_token_to_native",
                self.absolute_delta_nll_per_token_to_native,
            ),
            (
                "teacher_kl_sum_to_native",
                self.teacher_kl_sum_to_native,
            ),
            (
                "teacher_kl_per_token_to_native",
                self.teacher_kl_per_token_to_native,
            ),
            (
                "top1_agreement_to_native",
                self.top1_agreement_to_native,
            ),
        ):
            _finite_float(value, label=label)
        if (
            self.summed_nll < 0.0
            or self.nll_per_token < 0.0
            or self.absolute_delta_summed_nll_to_native < 0.0
            or self.absolute_delta_nll_per_token_to_native < 0.0
            or self.teacher_kl_sum_to_native < 0.0
            or self.teacher_kl_per_token_to_native < 0.0
            or not 0.0 <= self.top1_agreement_to_native <= 1.0
        ):
            raise ValueError("replacement aggregate metrics are invalid")
        if not _float_close(
            self.nll_per_token,
            self.summed_nll / self.supervised_tokens,
        ) or not _float_close(
            self.absolute_delta_nll_per_token_to_native,
            self.absolute_delta_summed_nll_to_native
            / self.supervised_tokens,
        ) or not _float_close(
            self.teacher_kl_per_token_to_native,
            self.teacher_kl_sum_to_native / self.supervised_tokens,
        ) or not _float_close(
            self.top1_agreement_to_native,
            self.top1_matches_to_native / self.supervised_tokens,
        ):
            raise ValueError("replacement aggregate reductions are invalid")
        if self.condition == "native" and (
            self.delta_summed_nll_to_native != 0.0
            or self.delta_nll_per_token_to_native != 0.0
            or self.absolute_delta_summed_nll_to_native != 0.0
            or self.absolute_delta_nll_per_token_to_native != 0.0
            or self.teacher_kl_sum_to_native != 0.0
            or self.teacher_kl_per_token_to_native != 0.0
            or self.top1_matches_to_native != self.supervised_tokens
            or self.top1_agreement_to_native != 1.0
        ):
            raise ValueError("native aggregate must be an exact self-control")

    def metadata(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "sequences": self.sequences,
            "supervised_tokens": self.supervised_tokens,
            "summed_nll": self.summed_nll,
            "nll_per_token": self.nll_per_token,
            "delta_summed_nll_to_native": (
                self.delta_summed_nll_to_native
            ),
            "delta_nll_per_token_to_native": (
                self.delta_nll_per_token_to_native
            ),
            "absolute_delta_summed_nll_to_native": (
                self.absolute_delta_summed_nll_to_native
            ),
            "absolute_delta_nll_per_token_to_native": (
                self.absolute_delta_nll_per_token_to_native
            ),
            "teacher_kl_sum_to_native": self.teacher_kl_sum_to_native,
            "teacher_kl_per_token_to_native": (
                self.teacher_kl_per_token_to_native
            ),
            "top1_matches_to_native": self.top1_matches_to_native,
            "top1_agreement_to_native": self.top1_agreement_to_native,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockReplacementConditionAggregate:
        expected = {
            "condition",
            "sequences",
            "supervised_tokens",
            "summed_nll",
            "nll_per_token",
            "delta_summed_nll_to_native",
            "delta_nll_per_token_to_native",
            "absolute_delta_summed_nll_to_native",
            "absolute_delta_nll_per_token_to_native",
            "teacher_kl_sum_to_native",
            "teacher_kl_per_token_to_native",
            "top1_matches_to_native",
            "top1_agreement_to_native",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("condition aggregate fields are invalid")
        return cls(
            condition=str(state["condition"]),
            sequences=int(state["sequences"]),
            supervised_tokens=int(state["supervised_tokens"]),
            summed_nll=float(state["summed_nll"]),
            nll_per_token=float(state["nll_per_token"]),
            delta_summed_nll_to_native=float(
                state["delta_summed_nll_to_native"]
            ),
            delta_nll_per_token_to_native=float(
                state["delta_nll_per_token_to_native"]
            ),
            absolute_delta_summed_nll_to_native=float(
                state["absolute_delta_summed_nll_to_native"]
            ),
            absolute_delta_nll_per_token_to_native=float(
                state["absolute_delta_nll_per_token_to_native"]
            ),
            teacher_kl_sum_to_native=float(
                state["teacher_kl_sum_to_native"]
            ),
            teacher_kl_per_token_to_native=float(
                state["teacher_kl_per_token_to_native"]
            ),
            top1_matches_to_native=int(state["top1_matches_to_native"]),
            top1_agreement_to_native=float(
                state["top1_agreement_to_native"]
            ),
        )


@dataclass(frozen=True, slots=True)
class CrossBlockReplacementFamilyAggregate:
    """Complete paired condition aggregates for one prompt family."""

    family_id: str
    conditions: tuple[CrossBlockReplacementConditionAggregate, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.family_id, label="family_id")
        if (
            type(self.conditions) is not tuple
            or tuple(value.condition for value in self.conditions)
            != REPLACEMENT_CONDITIONS
        ):
            raise ValueError(
                "family aggregate must contain the canonical condition quartet"
            )
        if len(
            {
                (value.sequences, value.supervised_tokens)
                for value in self.conditions
            }
        ) != 1:
            raise ValueError("family conditions must share paired counts")
        _validate_paired_condition_aggregates(self.conditions)

    def metadata(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "conditions": tuple(
                value.metadata() for value in self.conditions
            ),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockReplacementFamilyAggregate:
        if (
            not isinstance(state, Mapping)
            or set(state) != {"family_id", "conditions"}
            or not isinstance(state["conditions"], tuple)
        ):
            raise ValueError("family aggregate fields are invalid")
        return cls(
            family_id=str(state["family_id"]),
            conditions=tuple(
                CrossBlockReplacementConditionAggregate.from_state_dict(
                    value
                )
                for value in state["conditions"]
            ),
        )


@dataclass(frozen=True, slots=True)
class CrossBlockReplacementOracleProvenance:
    """Bindings for a disjoint fit-only native intervention evaluation.

    The replacement evidence digest binds the scale-fit stream.  The
    evaluation digest below separately binds the stream on which the frozen
    scale is intervened.
    """

    model_fingerprint: str
    evaluation_fit_split_sha256: str
    objective_sha256: str
    replacement_evidence_sha256: str
    shuffle_plan_sha256: str
    shuffle_policy: str

    def __post_init__(self) -> None:
        for label, value in (
            ("model_fingerprint", self.model_fingerprint),
            (
                "evaluation_fit_split_sha256",
                self.evaluation_fit_split_sha256,
            ),
            ("objective_sha256", self.objective_sha256),
            (
                "replacement_evidence_sha256",
                self.replacement_evidence_sha256,
            ),
            ("shuffle_plan_sha256", self.shuffle_plan_sha256),
        ):
            _require_sha256(value, label=label)
        _require_nonempty(self.shuffle_policy, label="shuffle_policy")

    def metadata(self) -> dict[str, object]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "evaluation_fit_split_sha256": (
                self.evaluation_fit_split_sha256
            ),
            "objective_sha256": self.objective_sha256,
            "replacement_evidence_sha256": (
                self.replacement_evidence_sha256
            ),
            "shuffle_plan_sha256": self.shuffle_plan_sha256,
            "shuffle_policy": self.shuffle_policy,
            "evaluation_scope": "fit_only_native_intervention_oracle",
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockReplacementOracleProvenance:
        expected = {
            "model_fingerprint",
            "evaluation_fit_split_sha256",
            "objective_sha256",
            "replacement_evidence_sha256",
            "shuffle_plan_sha256",
            "shuffle_policy",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("oracle provenance fields are invalid")
        return cls(
            model_fingerprint=str(state["model_fingerprint"]),
            evaluation_fit_split_sha256=str(
                state["evaluation_fit_split_sha256"]
            ),
            objective_sha256=str(state["objective_sha256"]),
            replacement_evidence_sha256=str(
                state["replacement_evidence_sha256"]
            ),
            shuffle_plan_sha256=str(state["shuffle_plan_sha256"]),
            shuffle_policy=str(state["shuffle_policy"]),
        )


def _condition_aggregate(
    rows: tuple[CrossBlockReplacementConditionMetric, ...],
    *,
    native_rows: tuple[CrossBlockReplacementConditionMetric, ...],
) -> CrossBlockReplacementConditionAggregate:
    condition = rows[0].condition
    sequences = len(rows)
    tokens = sum(value.supervised_tokens for value in rows)
    summed_nll = sum(value.summed_nll for value in rows)
    native_by_example = {
        value.example_id: value for value in native_rows
    }
    if set(native_by_example) != {
        value.example_id for value in rows
    }:
        raise ValueError(
            "condition rows must match native examples exactly"
        )
    native_summed_nll = sum(
        value.summed_nll for value in native_rows
    )
    absolute_delta_summed_nll = sum(
        abs(
            value.summed_nll
            - native_by_example[value.example_id].summed_nll
        )
        for value in rows
    )
    teacher_kl = sum(value.teacher_kl_sum_to_native for value in rows)
    matches = sum(value.top1_matches_to_native for value in rows)
    return CrossBlockReplacementConditionAggregate(
        condition=condition,
        sequences=sequences,
        supervised_tokens=tokens,
        summed_nll=float(summed_nll),
        nll_per_token=float(summed_nll / tokens),
        delta_summed_nll_to_native=float(
            summed_nll - native_summed_nll
        ),
        delta_nll_per_token_to_native=float(
            (summed_nll - native_summed_nll) / tokens
        ),
        absolute_delta_summed_nll_to_native=float(
            absolute_delta_summed_nll
        ),
        absolute_delta_nll_per_token_to_native=float(
            absolute_delta_summed_nll / tokens
        ),
        teacher_kl_sum_to_native=float(teacher_kl),
        teacher_kl_per_token_to_native=float(teacher_kl / tokens),
        top1_matches_to_native=matches,
        top1_agreement_to_native=float(matches / tokens),
    )


def _validate_paired_condition_aggregates(
    conditions: tuple[CrossBlockReplacementConditionAggregate, ...],
) -> None:
    by_condition = {value.condition: value for value in conditions}
    if tuple(by_condition) != REPLACEMENT_CONDITIONS:
        raise ValueError("condition aggregates are not in canonical order")
    native = by_condition["native"]
    for condition in conditions:
        if (
            condition.sequences != native.sequences
            or condition.supervised_tokens != native.supervised_tokens
            or not _float_close(
                condition.delta_summed_nll_to_native,
                condition.summed_nll - native.summed_nll,
            )
            or not _float_close(
                condition.delta_nll_per_token_to_native,
                (
                    condition.summed_nll - native.summed_nll
                )
                / native.supervised_tokens,
            )
            or (
                condition.absolute_delta_summed_nll_to_native < abs(
                    condition.delta_summed_nll_to_native
                )
            )
        ):
            raise ValueError(
                "paired condition deltas are inconsistent with native"
            )


def _recovery(
    replacement_distortion: float,
    ablation_distortion: float,
) -> float | None:
    if ablation_distortion == 0.0:
        return None
    return 1.0 - replacement_distortion / ablation_distortion


@dataclass(frozen=True, slots=True)
class CrossBlockReplacementOracleResult:
    """Authenticated aggregate of the paired fit-side intervention quartet."""

    provenance: CrossBlockReplacementOracleProvenance
    metric_stream_sha256: str
    conditions: tuple[CrossBlockReplacementConditionAggregate, ...]
    family_aggregates: tuple[CrossBlockReplacementFamilyAggregate, ...]
    replacement_kl_recovery_vs_ablation: float | None
    replacement_absolute_nll_recovery_vs_ablation: float | None
    replacement_kl_advantage_vs_shuffled: float
    replacement_top1_advantage_vs_shuffled: float
    artifact_sha256: str
    artifact_kind: str = _ORACLE_KIND
    format_version: int = _FORMAT_VERSION
    contains_corpus_rows: bool = False
    fit_only: bool = True
    model_parameter_updates: int = 0
    authorizes_intervention: bool = False
    authorizes_compilation: bool = False
    authorizes_execution: bool = False
    authorizes_guard: bool = False
    authorizes_b: bool = False

    def __post_init__(self) -> None:
        if not isinstance(
            self.provenance,
            CrossBlockReplacementOracleProvenance,
        ):
            raise TypeError("oracle provenance is invalid")
        _require_sha256(
            self.metric_stream_sha256,
            label="metric_stream_sha256",
        )
        if (
            type(self.conditions) is not tuple
            or tuple(value.condition for value in self.conditions)
            != REPLACEMENT_CONDITIONS
            or len(
                {
                    (value.sequences, value.supervised_tokens)
                    for value in self.conditions
                }
            )
            != 1
        ):
            raise ValueError("oracle conditions are not a paired quartet")
        _validate_paired_condition_aggregates(self.conditions)
        if (
            type(self.family_aggregates) is not tuple
            or not self.family_aggregates
            or tuple(
                value.family_id for value in self.family_aggregates
            )
            != tuple(
                sorted(
                    value.family_id for value in self.family_aggregates
                )
            )
            or len(
                {value.family_id for value in self.family_aggregates}
            )
            != len(self.family_aggregates)
        ):
            raise ValueError("oracle family aggregates are invalid")
        for condition_index, condition in enumerate(self.conditions):
            family_conditions = tuple(
                family.conditions[condition_index]
                for family in self.family_aggregates
            )
            if (
                sum(value.sequences for value in family_conditions)
                != condition.sequences
                or sum(
                    value.supervised_tokens
                    for value in family_conditions
                )
                != condition.supervised_tokens
                or sum(
                    value.top1_matches_to_native
                    for value in family_conditions
                )
                != condition.top1_matches_to_native
            ):
                raise ValueError(
                    "oracle family counts do not sum to the global aggregate"
                )
            for label in (
                "summed_nll",
                "delta_summed_nll_to_native",
                "absolute_delta_summed_nll_to_native",
                "teacher_kl_sum_to_native",
            ):
                if not _float_close(
                    sum(
                        getattr(value, label)
                        for value in family_conditions
                    ),
                    getattr(condition, label),
                ):
                    raise ValueError(
                        "oracle family metric sums do not match the "
                        "global aggregate"
                    )
        for label, value in (
            (
                "replacement_kl_recovery_vs_ablation",
                self.replacement_kl_recovery_vs_ablation,
            ),
            (
                "replacement_absolute_nll_recovery_vs_ablation",
                self.replacement_absolute_nll_recovery_vs_ablation,
            ),
        ):
            if value is not None:
                _finite_float(value, label=label)
        for label, value in (
            (
                "replacement_kl_advantage_vs_shuffled",
                self.replacement_kl_advantage_vs_shuffled,
            ),
            (
                "replacement_top1_advantage_vs_shuffled",
                self.replacement_top1_advantage_vs_shuffled,
            ),
        ):
            _finite_float(value, label=label)
        by_condition = {
            value.condition: value for value in self.conditions
        }
        expected_kl_recovery = _recovery(
            by_condition["replacement"].teacher_kl_per_token_to_native,
            by_condition["ablation"].teacher_kl_per_token_to_native,
        )
        expected_nll_recovery = _recovery(
            by_condition[
                "replacement"
            ].absolute_delta_nll_per_token_to_native,
            by_condition[
                "ablation"
            ].absolute_delta_nll_per_token_to_native,
        )
        if not _optional_float_close(
            self.replacement_kl_recovery_vs_ablation,
            expected_kl_recovery,
        ) or not _optional_float_close(
            self.replacement_absolute_nll_recovery_vs_ablation,
            expected_nll_recovery,
        ):
            raise ValueError("oracle recovery metrics are inconsistent")
        if not _float_close(
            self.replacement_kl_advantage_vs_shuffled,
            by_condition["shuffled"].teacher_kl_per_token_to_native
            - by_condition["replacement"].teacher_kl_per_token_to_native,
        ) or not _float_close(
            self.replacement_top1_advantage_vs_shuffled,
            by_condition["replacement"].top1_agreement_to_native
            - by_condition["shuffled"].top1_agreement_to_native,
        ):
            raise ValueError("oracle shuffled-control metrics are inconsistent")
        if (
            self.artifact_kind != _ORACLE_KIND
            or self.format_version != _FORMAT_VERSION
            or self.contains_corpus_rows
            or not self.fit_only
            or self.model_parameter_updates != 0
            or self.authorizes_intervention
            or self.authorizes_compilation
            or self.authorizes_execution
            or self.authorizes_guard
            or self.authorizes_b
        ):
            raise ValueError("replacement oracle safety metadata is invalid")
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("replacement oracle artifact hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "contains_corpus_rows": self.contains_corpus_rows,
            "fit_only": self.fit_only,
            "model_parameter_updates": self.model_parameter_updates,
            "authorizes_intervention": self.authorizes_intervention,
            "authorizes_compilation": self.authorizes_compilation,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_guard": self.authorizes_guard,
            "authorizes_b": self.authorizes_b,
            "provenance": self.provenance.metadata(),
            "metric_stream_sha256": self.metric_stream_sha256,
            "conditions": tuple(
                value.metadata() for value in self.conditions
            ),
            "family_aggregates": tuple(
                value.metadata() for value in self.family_aggregates
            ),
            "replacement_kl_recovery_vs_ablation": (
                self.replacement_kl_recovery_vs_ablation
            ),
            "replacement_absolute_nll_recovery_vs_ablation": (
                self.replacement_absolute_nll_recovery_vs_ablation
            ),
            "replacement_kl_advantage_vs_shuffled": (
                self.replacement_kl_advantage_vs_shuffled
            ),
            "replacement_top1_advantage_vs_shuffled": (
                self.replacement_top1_advantage_vs_shuffled
            ),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_ORACLE_HASH_DOMAIN)

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "contains_corpus_rows": self.contains_corpus_rows,
            "fit_only": self.fit_only,
            "model_parameter_updates": self.model_parameter_updates,
            "authorizes_intervention": self.authorizes_intervention,
            "authorizes_compilation": self.authorizes_compilation,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_guard": self.authorizes_guard,
            "authorizes_b": self.authorizes_b,
            "provenance": {
                key: value
                for key, value in self.provenance.metadata().items()
                if key != "evaluation_scope"
            },
            "metric_stream_sha256": self.metric_stream_sha256,
            "conditions": tuple(
                value.metadata() for value in self.conditions
            ),
            "family_aggregates": tuple(
                value.metadata() for value in self.family_aggregates
            ),
            "replacement_kl_recovery_vs_ablation": (
                self.replacement_kl_recovery_vs_ablation
            ),
            "replacement_absolute_nll_recovery_vs_ablation": (
                self.replacement_absolute_nll_recovery_vs_ablation
            ),
            "replacement_kl_advantage_vs_shuffled": (
                self.replacement_kl_advantage_vs_shuffled
            ),
            "replacement_top1_advantage_vs_shuffled": (
                self.replacement_top1_advantage_vs_shuffled
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockReplacementOracleResult:
        expected = {
            "artifact_kind",
            "format_version",
            "contains_corpus_rows",
            "fit_only",
            "model_parameter_updates",
            "authorizes_intervention",
            "authorizes_compilation",
            "authorizes_execution",
            "authorizes_guard",
            "authorizes_b",
            "provenance",
            "metric_stream_sha256",
            "conditions",
            "family_aggregates",
            "replacement_kl_recovery_vs_ablation",
            "replacement_absolute_nll_recovery_vs_ablation",
            "replacement_kl_advantage_vs_shuffled",
            "replacement_top1_advantage_vs_shuffled",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("replacement oracle state fields are invalid")
        if not isinstance(state["provenance"], Mapping):
            raise TypeError("oracle provenance state must be a mapping")
        if not isinstance(state["conditions"], tuple) or not isinstance(
            state["family_aggregates"],
            tuple,
        ):
            raise TypeError("oracle aggregate states must be tuples")
        kl_recovery = state["replacement_kl_recovery_vs_ablation"]
        nll_recovery = state[
            "replacement_absolute_nll_recovery_vs_ablation"
        ]
        return cls(
            provenance=CrossBlockReplacementOracleProvenance.from_state_dict(
                state["provenance"]
            ),
            metric_stream_sha256=str(state["metric_stream_sha256"]),
            conditions=tuple(
                CrossBlockReplacementConditionAggregate.from_state_dict(value)
                for value in state["conditions"]
            ),
            family_aggregates=tuple(
                CrossBlockReplacementFamilyAggregate.from_state_dict(value)
                for value in state["family_aggregates"]
            ),
            replacement_kl_recovery_vs_ablation=(
                None if kl_recovery is None else float(kl_recovery)
            ),
            replacement_absolute_nll_recovery_vs_ablation=(
                None if nll_recovery is None else float(nll_recovery)
            ),
            replacement_kl_advantage_vs_shuffled=float(
                state["replacement_kl_advantage_vs_shuffled"]
            ),
            replacement_top1_advantage_vs_shuffled=float(
                state["replacement_top1_advantage_vs_shuffled"]
            ),
            artifact_sha256=str(state["artifact_sha256"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
            contains_corpus_rows=bool(state["contains_corpus_rows"]),
            fit_only=bool(state["fit_only"]),
            model_parameter_updates=int(state["model_parameter_updates"]),
            authorizes_intervention=bool(
                state["authorizes_intervention"]
            ),
            authorizes_compilation=bool(state["authorizes_compilation"]),
            authorizes_execution=bool(state["authorizes_execution"]),
            authorizes_guard=bool(state["authorizes_guard"]),
            authorizes_b=bool(state["authorizes_b"]),
        )


def _metric_stream_sha256(
    rows: tuple[CrossBlockReplacementConditionMetric, ...],
) -> str:
    return _json_sha256(
        tuple(value.metadata() for value in rows),
        domain=_METRIC_HASH_DOMAIN,
    )


def aggregate_cross_block_replacement_conditions(
    metrics: Iterable[CrossBlockReplacementConditionMetric],
    *,
    provenance: CrossBlockReplacementOracleProvenance,
) -> CrossBlockReplacementOracleResult:
    """Aggregate a complete paired native/ablation/replacement/shuffle oracle."""

    if not isinstance(provenance, CrossBlockReplacementOracleProvenance):
        raise TypeError("oracle provenance is invalid")
    materialized = tuple(metrics)
    if not materialized or any(
        not isinstance(value, CrossBlockReplacementConditionMetric)
        for value in materialized
    ):
        raise ValueError("replacement oracle metrics cannot be empty")
    order = {name: index for index, name in enumerate(REPLACEMENT_CONDITIONS)}
    materialized = tuple(
        sorted(
            materialized,
            key=lambda value: (value.example_id, order[value.condition]),
        )
    )
    by_example: dict[
        str,
        dict[str, CrossBlockReplacementConditionMetric],
    ] = {}
    for value in materialized:
        conditions = by_example.setdefault(value.example_id, {})
        if value.condition in conditions:
            raise ValueError(
                "each example may contain each oracle condition only once"
            )
        conditions[value.condition] = value
    for example_id, conditions in by_example.items():
        if set(conditions) != set(REPLACEMENT_CONDITIONS):
            raise ValueError(
                f"{example_id!r} is missing an oracle condition"
            )
        rows = tuple(conditions[name] for name in REPLACEMENT_CONDITIONS)
        if len({value.family_id for value in rows}) != 1 or len(
            {value.supervised_tokens for value in rows}
        ) != 1:
            raise ValueError(
                "paired oracle conditions must share family and token count"
            )

    condition_rows = {
        condition: tuple(
            by_example[example_id][condition]
            for example_id in sorted(by_example)
        )
        for condition in REPLACEMENT_CONDITIONS
    }
    conditions = tuple(
        _condition_aggregate(
            condition_rows[condition],
            native_rows=condition_rows["native"],
        )
        for condition in REPLACEMENT_CONDITIONS
    )
    families = sorted(
        {
            value.family_id
            for value in condition_rows["native"]
        }
    )
    family_aggregates: list[CrossBlockReplacementFamilyAggregate] = []
    for family in families:
        family_rows = {
            condition: tuple(
                value
                for value in condition_rows[condition]
                if value.family_id == family
            )
            for condition in REPLACEMENT_CONDITIONS
        }
        family_aggregates.append(
            CrossBlockReplacementFamilyAggregate(
                family_id=family,
                conditions=tuple(
                    _condition_aggregate(
                        family_rows[condition],
                        native_rows=family_rows["native"],
                    )
                    for condition in REPLACEMENT_CONDITIONS
                ),
            )
        )
    by_condition = {value.condition: value for value in conditions}
    kl_recovery = _recovery(
        by_condition["replacement"].teacher_kl_per_token_to_native,
        by_condition["ablation"].teacher_kl_per_token_to_native,
    )
    nll_recovery = _recovery(
        by_condition[
            "replacement"
        ].absolute_delta_nll_per_token_to_native,
        by_condition[
            "ablation"
        ].absolute_delta_nll_per_token_to_native,
    )
    kl_advantage = (
        by_condition["shuffled"].teacher_kl_per_token_to_native
        - by_condition["replacement"].teacher_kl_per_token_to_native
    )
    top1_advantage = (
        by_condition["replacement"].top1_agreement_to_native
        - by_condition["shuffled"].top1_agreement_to_native
    )
    stream_hash = _metric_stream_sha256(materialized)
    payload = {
        "artifact_kind": _ORACLE_KIND,
        "format_version": _FORMAT_VERSION,
        "contains_corpus_rows": False,
        "fit_only": True,
        "model_parameter_updates": 0,
        "authorizes_intervention": False,
        "authorizes_compilation": False,
        "authorizes_execution": False,
        "authorizes_guard": False,
        "authorizes_b": False,
        "provenance": provenance.metadata(),
        "metric_stream_sha256": stream_hash,
        "conditions": tuple(value.metadata() for value in conditions),
        "family_aggregates": tuple(
            value.metadata() for value in family_aggregates
        ),
        "replacement_kl_recovery_vs_ablation": kl_recovery,
        "replacement_absolute_nll_recovery_vs_ablation": nll_recovery,
        "replacement_kl_advantage_vs_shuffled": kl_advantage,
        "replacement_top1_advantage_vs_shuffled": top1_advantage,
    }
    return CrossBlockReplacementOracleResult(
        provenance=provenance,
        metric_stream_sha256=stream_hash,
        conditions=conditions,
        family_aggregates=tuple(family_aggregates),
        replacement_kl_recovery_vs_ablation=kl_recovery,
        replacement_absolute_nll_recovery_vs_ablation=nll_recovery,
        replacement_kl_advantage_vs_shuffled=float(kl_advantage),
        replacement_top1_advantage_vs_shuffled=float(top1_advantage),
        artifact_sha256=_json_sha256(
            payload,
            domain=_ORACLE_HASH_DOMAIN,
        ),
    )
